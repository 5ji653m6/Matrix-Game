#!/usr/bin/env python
"""LTX-2.3 backend for Matrix-Game-3: streaming long-video generation.

Replaces the Wan2.2 backbone (`generate.py` -> `pipeline/inference_pipeline.py`)
with Lightricks' LTX-2.3 (22B audio-video DiT) while keeping Matrix-Game-3's
segmented autoregressive generation scheme:

    total_frames = 57 + (num_iterations - 1) * 40

  - Segment 0: image-to-video from `--image`, 57 frames (8k+1, LTX requirement).
  - Segment i: conditioned on the last decoded frame of segment i-1
    (LTX ImageConditioningInput at frame_idx=0), 41 frames generated, the
    duplicated first frame is dropped -> 40 new frames per iteration.

This mirrors MG3's `first_clip_frame=57` / 40-frames-per-iteration pacing and
its re-injection of previously generated frames as clean conditioning.

IMPORTANT — action/camera control: the stock LTX-2.3 weights have no
keyboard/mouse/camera conditioning. Those capabilities come from the trained
world-model checkpoints in the sibling project /data1/ltx-world-model, whose
own inference entry (`scripts/inference/infer_bidirectional_camera.py`) handles
PRoPE camera conditioning and action modules. This script covers the
high-quality text+image conditioned generation path (what MG3 calls the
non-interactive pipeline) and is the drop-in replacement for Wan2.2 quality
evaluation on game scenes.

Dependencies are NOT installed into this repo. They are imported at runtime
from the LTX-2 monorepo (env var LTX_ROOT, default /root/learning/LTX-2):
    packages/ltx-core/src, packages/ltx-pipelines/src
Run with an environment that has torch/diffusers/etc., e.g.:
    /data1/ltx-world-model/.venv/bin/python generate_ltx.py ...

Example:
    python generate_ltx.py \
        --image demo_images/001/image.png \
        --prompt "A colorful, animated cityscape with a gas station and various buildings." \
        --size 704*1280 --num_iterations 12 --seed 42 \
        --output_dir ./output_ltx --save_name test

Single GPU by default. `--mgpu` runs one generation across all visible GPUs via
ltx-pipelines' MGPU controller: sequence parallelism over the token sequence +
Accelerate-sharded Gemma + distributed VAE decode (numerically equivalent to
single-GPU; it is a latency tool, NOT a memory tool — each rank holds a full
transformer replica, fp8-cast by default). Requires the ltx-kernels CUDA
extension (built from $LTX_ROOT/packages/ltx-kernels). `--mgpu` writes one mp4
per segment and re-encodes the final combined video from them (one extra x264
generation; use --keep_segments to inspect per-segment output). `--one_stage`
is not supported under `--mgpu` (no one-stage MGPU runner ships upstream).
"""

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# LTX-2 monorepo on sys.path (nothing is pip-installed; see module docstring)
# ---------------------------------------------------------------------------
LTX_ROOT = Path(os.environ.get("LTX_ROOT", "/root/learning/LTX-2"))
for _pkg in ("ltx-core", "ltx-pipelines"):
    _src = LTX_ROOT / "packages" / _pkg / "src"
    if _src.is_dir():
        sys.path.insert(0, str(_src))

# Default local assets on this box
DEFAULT_CKPT_DIR = Path("/data/models/Lightricks--LTX-2.3/snapshots/master")
DEFAULT_DISTILLED_CKPT = DEFAULT_CKPT_DIR / "ltx-2.3-22b-distilled-1.1.safetensors"
DEFAULT_DEV_CKPT = DEFAULT_CKPT_DIR / "ltx-2.3-22b-dev.safetensors"
DEFAULT_UPSAMPLER = DEFAULT_CKPT_DIR / "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
DEFAULT_GEMMA = "/data1/models/google--gemma-3-12b-it/snapshots/master"

# MG3 pacing constants (pipeline/inference_pipeline.py): first clip 57 frames,
# 40 new frames per subsequent iteration. LTX requires num_frames % 8 == 1.
FIRST_SEGMENT_FRAMES = 57
SEGMENT_FRAMES = 41          # 40 new + 1 duplicated conditioning frame
NEW_FRAMES_PER_ITER = 40


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate long game-style video with the LTX-2.3 backend "
                    "(Matrix-Game-3 segmented autoregressive scheme).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # MG3-flavored core args
    parser.add_argument("--image", type=str, required=True, help="Path to the initial image")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--output_dir", type=str, default="./output_ltx")
    parser.add_argument("--save_name", type=str, default="ltx_test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_iterations", type=int, default=12,
                        help="Total frames = 57 + (num_iterations - 1) * 40")
    parser.add_argument("--size", type=str, default="704*1280",
                        help="height*width; H,W divisible by 32 (two-stage: by 64)")
    parser.add_argument("--frame_rate", type=float, default=24.0)

    # LTX model assets
    parser.add_argument("--ltx_checkpoint", type=str, default=str(DEFAULT_DISTILLED_CKPT),
                        help="Distilled (or dev, with --one_stage) checkpoint")
    parser.add_argument("--spatial_upsampler", type=str, default=str(DEFAULT_UPSAMPLER),
                        help="Spatial upsampler (two-stage distilled mode only)")
    parser.add_argument("--gemma_root", type=str, default=DEFAULT_GEMMA,
                        help="Gemma-3-12B-IT text encoder directory")

    # Mode selection
    parser.add_argument("--one_stage", action="store_true",
                        help="Single-stage generation with the dev checkpoint + CFG "
                             "(slower; ignores --spatial_upsampler)")
    parser.add_argument("--num_inference_steps", type=int, default=40,
                        help="One-stage mode only")
    parser.add_argument("--guidance_scale", type=float, default=3.0,
                        help="One-stage mode only (video and audio CFG)")
    parser.add_argument("--negative_prompt", type=str, default="",
                        help="One-stage mode only")

    # Output options
    parser.add_argument("--mgpu", action="store_true",
                        help="Run across all visible GPUs (sequence-parallel + "
                             "distributed VAE/Gemma; requires ltx-kernels; "
                             "distilled mode only, latency tool not memory)")
    parser.add_argument("--enhance_prompt", action="store_true",
                        help="Let Gemma enhance the prompt (uses the conditioning image)")
    parser.add_argument("--no_audio", action="store_true",
                        help="Drop the generated audio track from the output mp4")
    parser.add_argument("--keep_segments", action="store_true",
                        help="Also write one mp4 per segment next to the final video")
    parser.add_argument("--crf", type=int, default=19, help="x264 CRF for the output")
    return parser.parse_args()


def _check_assets(args):
    missing = [p for p in (args.ltx_checkpoint, args.gemma_root, args.image)
               if not Path(p).exists()]
    if not args.one_stage and not Path(args.spatial_upsampler).exists():
        missing.append(args.spatial_upsampler)
    if not (LTX_ROOT / "packages" / "ltx-core" / "src").is_dir():
        missing.append(str(LTX_ROOT) + " (set LTX_ROOT)")
    if missing:
        raise FileNotFoundError("Missing required paths:\n  " + "\n  ".join(missing))


def _build_pipeline(args):
    """Construct the LTX pipeline (loads transformer, VAEs, Gemma)."""
    if args.one_stage:
        from ltx_core.components.guiders import MultiModalGuiderParams
        from ltx_pipelines.ti2vid_one_stage import TI2VidOneStagePipeline

        pipeline = TI2VidOneStagePipeline(
            checkpoint_path=args.ltx_checkpoint,
            gemma_root=args.gemma_root,
            loras=[],
        )
        guider = MultiModalGuiderParams(cfg_scale=args.guidance_scale)

        def run_segment(prompt, seed, height, width, num_frames, images):
            return pipeline(
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                seed=seed,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=args.frame_rate,
                num_inference_steps=args.num_inference_steps,
                video_guider_params=guider,
                audio_guider_params=guider,
                images=images,
                enhance_prompt=args.enhance_prompt,
            )
    else:
        from ltx_pipelines.distilled import DistilledPipeline

        pipeline = DistilledPipeline(
            distilled_checkpoint_path=args.ltx_checkpoint,
            gemma_root=args.gemma_root,
            spatial_upsampler_path=args.spatial_upsampler,
            loras=[],
        )

        def run_segment(prompt, seed, height, width, num_frames, images):
            return pipeline(
                prompt=prompt,
                seed=seed,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=args.frame_rate,
                images=images,
                enhance_prompt=args.enhance_prompt,
            )
    return run_segment


def _save_last_frame(chunks, path):
    """Save the final decoded frame ((H,W,C) float [0,1]) as PNG."""
    from PIL import Image
    import numpy as np

    frame = chunks[-1][-1].clamp(0, 1).cpu().numpy()
    Image.fromarray((frame * 255.0).round().astype(np.uint8)).save(path)


def _concat_audios(audios, frame_rate):
    """Concatenate per-segment Audio objects, trimming the duplicated first
    frame's worth of samples from every segment after the first."""
    from ltx_core.types import Audio

    waveforms = []
    for i, audio in enumerate(audios):
        wf = audio.waveform
        if wf.ndim == 3 and wf.shape[0] == 1:
            wf = wf.squeeze(0)  # decode_audio_from_file returns (1, C, N)
        if wf.ndim != 2:
            raise ValueError(f"Unexpected audio waveform shape {tuple(wf.shape)}")
        # Normalize to (2, N)
        if wf.shape[0] != 2:
            wf = wf.transpose(0, 1)
        if i > 0:
            trim = int(round(audio.sampling_rate / frame_rate))
            wf = wf[:, trim:]
        waveforms.append(wf.cpu())
    return Audio(waveform=torch.cat(waveforms, dim=1),
                 sampling_rate=audios[0].sampling_rate)


# ---------------------------------------------------------------------------
# Multi-GPU path (--mgpu): ltx-pipelines MGPU controller, one job per segment
# ---------------------------------------------------------------------------

def _extract_last_frame_from_mp4(mp4_path, out_png):
    """Decode an mp4 and save its final frame as PNG (next segment's conditioning)."""
    import av
    from PIL import Image

    container = av.open(str(mp4_path))
    last = None
    for frame in container.decode(video=0):
        last = frame
    container.close()
    if last is None:
        raise RuntimeError(f"No frames decoded from {mp4_path}")
    Image.fromarray(last.to_ndarray(format="rgb24")).save(out_png)


def _read_mp4_frames(mp4_path):
    """Decode all frames of an mp4 to a list of (H,W,3) uint8 CPU tensors."""
    import av

    container = av.open(str(mp4_path))
    frames = [torch.from_numpy(f.to_ndarray(format="rgb24"))
              for f in container.decode(video=0)]
    container.close()
    return frames


def _generate_mgpu(args, height, width, output_dir):
    """Segmented generation driven through the MGPU worker fleet."""
    from ltx_pipelines.distilled_mgpu import DistilledRunner
    from ltx_pipelines.multigpu.controller import MGPUController
    from ltx_pipelines.utils.args import ImageConditioningInput
    from ltx_pipelines.utils.media_io import decode_audio_from_file, encode_video

    tmp_dir = Path(tempfile.mkdtemp(prefix="ltx_cond_", dir=output_dir))
    vae_queue = torch.multiprocessing.get_context("spawn").SimpleQueue()
    controller = MGPUController(DistilledRunner)
    controller.start(
        distilled_checkpoint_path=args.ltx_checkpoint,
        gemma_root=args.gemma_root,
        spatial_upsampler_path=args.spatial_upsampler,
        vae_queue=vae_queue,
    )

    seg_paths, cond_image_path = [], args.image
    try:
        for it in range(args.num_iterations):
            num_frames = FIRST_SEGMENT_FRAMES if it == 0 else SEGMENT_FRAMES
            seg_path = output_dir / f"{args.save_name}_seg{it:03d}.mp4"
            images = [ImageConditioningInput(path=str(cond_image_path),
                                             frame_idx=0, strength=1.0)]
            t0 = time.time()
            for _ in controller.stream(
                output_path=str(seg_path),
                prompt=args.prompt,
                seed=args.seed + it,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=int(args.frame_rate),
                images=images,
            ):
                pass  # drive the job; the runner writes seg_path on rank 0
            print(f"[LTX backend] segment {it}: {num_frames} frames in "
                  f"{time.time() - t0:.1f}s -> {seg_path}")
            seg_paths.append(seg_path)

            if it < args.num_iterations - 1:
                cond_image_path = tmp_dir / f"cond_{it:03d}.png"
                _extract_last_frame_from_mp4(seg_path, cond_image_path)
    finally:
        controller.shutdown()

    # Combine segments into the final video (one re-encode pass).
    all_frames = []
    for i, p in enumerate(seg_paths):
        frames = _read_mp4_frames(p)
        all_frames.extend(frames[1:] if i > 0 else frames)

    final_audio = None
    if not args.no_audio:
        audios = [a for p in seg_paths
                  if (a := decode_audio_from_file(str(p), device=torch.device("cpu")))
                  is not None]
        if audios:
            final_audio = _concat_audios(audios, args.frame_rate)

    output_path = output_dir / f"{args.save_name}.mp4"
    video = torch.stack(all_frames).float() / 255.0  # (F,H,W,C) in [0,1]
    encode_video(video=video, fps=int(args.frame_rate), audio=final_audio,
                 output_path=str(output_path), video_chunks_number=1, crf=args.crf)
    print(f"[LTX backend] saved {video.shape[0]} frames -> {output_path}")


@torch.inference_mode()
def generate(args):
    from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
    from ltx_pipelines.utils.args import ImageConditioningInput
    from ltx_pipelines.utils.media_io import encode_video

    _check_assets(args)
    height, width = (int(v) for v in args.size.split("*"))
    if not args.one_stage and (height % 64 or width % 64):
        raise ValueError(
            f"Resolution {height}x{width} is not divisible by 64. For two-stage "
            "pipelines, height and width must be multiples of 64.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_frames = FIRST_SEGMENT_FRAMES + (args.num_iterations - 1) * NEW_FRAMES_PER_ITER

    if args.mgpu:
        if args.one_stage:
            raise ValueError("--mgpu is only supported in distilled (two-stage) mode; "
                             "no one-stage MGPU runner ships upstream.")
        n_gpus = torch.cuda.device_count()
        if n_gpus < 2:
            raise RuntimeError(f"--mgpu needs >=2 visible GPUs, found {n_gpus}.")
        print(f"[LTX backend] mode=mgpu distilled on {n_gpus} GPUs, size={height}x{width}, "
              f"{args.num_iterations} iterations -> {total_frames} frames @ {args.frame_rate}fps")
        _generate_mgpu(args, height, width, output_dir)
        return

    mode = "one-stage dev+CFG" if args.one_stage else "two-stage distilled"
    print(f"[LTX backend] mode={mode}, size={height}x{width}, "
          f"{args.num_iterations} iterations -> {total_frames} frames @ {args.frame_rate}fps")

    run_segment = _build_pipeline(args)
    tiling = TilingConfig.default()

    all_chunks, all_audios = [], []
    cond_image_path = args.image
    tmp_dir = Path(tempfile.mkdtemp(prefix="ltx_cond_", dir=output_dir))

    for it in range(args.num_iterations):
        num_frames = FIRST_SEGMENT_FRAMES if it == 0 else SEGMENT_FRAMES
        seed = args.seed + it  # deterministic variation per segment
        images = [ImageConditioningInput(path=str(cond_image_path), frame_idx=0, strength=1.0)]

        t0 = time.time()
        video_iter, audio = run_segment(
            args.prompt, seed, height, width, num_frames, images)
        seg_chunks = [c.cpu() for c in video_iter]
        print(f"[LTX backend] segment {it}: {num_frames} frames in {time.time() - t0:.1f}s")

        if it > 0:
            # Drop the duplicated conditioning frame.
            seg_chunks[0] = seg_chunks[0][1:]
            if seg_chunks[0].shape[0] == 0:
                seg_chunks.pop(0)

        if args.keep_segments:
            seg_path = output_dir / f"{args.save_name}_seg{it:03d}.mp4"
            encode_video(video=iter(seg_chunks), fps=int(args.frame_rate),
                         audio=None if args.no_audio else audio,
                         output_path=str(seg_path),
                         video_chunks_number=len(seg_chunks), crf=args.crf)

        # Next segment is conditioned on this segment's last decoded frame.
        if it < args.num_iterations - 1:
            cond_image_path = tmp_dir / f"cond_{it:03d}.png"
            _save_last_frame(seg_chunks, cond_image_path)

        all_chunks.extend(seg_chunks)
        all_audios.append(audio)

    final_audio = None
    if not args.no_audio:
        final_audio = _concat_audios(all_audios, args.frame_rate)

    output_path = output_dir / f"{args.save_name}.mp4"
    encode_video(video=iter(all_chunks), fps=int(args.frame_rate),
                 audio=final_audio, output_path=str(output_path),
                 video_chunks_number=len(all_chunks), crf=args.crf)
    n_frames = sum(c.shape[0] for c in all_chunks)
    print(f"[LTX backend] saved {n_frames} frames -> {output_path}")


def main():
    args = parse_args()
    generate(args)


if __name__ == "__main__":
    main()
