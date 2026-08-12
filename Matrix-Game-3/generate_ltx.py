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

Zero-shot interactive mimicry (--interactive / --actions): between segments you
steer the next segment with a camera action (w/a/s/d/up/down) or free-form
motion text. Warp actions apply a geometric crop/pan/zoom to the last decoded
frame AND inject a matching motion phrase into the prompt; free text steers
semantically only (in the MG series the control signal is just conditioning, so
text is a first-class action channel via Gemma). This mimics MG3's action loop
WITHOUT trained action weights — control is per-segment (not per-frame) and
latency is turn-based (one segment's generation time per move), not real-time.
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

# Default local assets on this box (override the directory with LTX_CKPT_DIR)
DEFAULT_CKPT_DIR = Path(os.environ.get(
    "LTX_CKPT_DIR", "/data1/models/Lightricks--LTX-2.3/snapshots/master"))
DEFAULT_DISTILLED_CKPT = DEFAULT_CKPT_DIR / "ltx-2.3-22b-distilled-1.1.safetensors"
DEFAULT_DEV_CKPT = DEFAULT_CKPT_DIR / "ltx-2.3-22b-dev.safetensors"
DEFAULT_UPSAMPLER = DEFAULT_CKPT_DIR / "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
DEFAULT_DISTILLED_LORA = DEFAULT_CKPT_DIR / "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
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
    # Coherent-audio mode (a2vid): one continuous soundtrack for the whole video
    parser.add_argument("--audio_track", type=str, default="",
                        help="Path to a pre-made soundtrack (wav/mp3/...). Each segment's "
                             "video is generated conditioned on the matching slice of this "
                             "track (A2Vid pipeline, audio frozen) and the final mux uses "
                             "the track itself -> music/SFX stay coherent across segments")
    parser.add_argument("--coherent_audio", action="store_true",
                        help="Generate one full-length soundtrack with the text-to-audio "
                             "pipeline (dev checkpoint) first, then use it like --audio_track")
    parser.add_argument("--audio_prompt", type=str, default="",
                        help="Text-to-audio prompt for --coherent_audio "
                             "(default: derived from --prompt + continuity directives)")
    parser.add_argument("--ltx_dev_checkpoint", type=str, default=str(DEFAULT_DEV_CKPT),
                        help="Dev (non-distilled) checkpoint: T2A and A2Vid base model")
    parser.add_argument("--distilled_lora", type=str, default=str(DEFAULT_DISTILLED_LORA),
                        help="Distilled refinement LoRA for A2Vid stage 2")
    parser.add_argument("--audio_num_inference_steps", type=int, default=40,
                        help="T2A denoising steps (--coherent_audio)")
    parser.add_argument("--audio_cfg_scale", type=float, default=3.0,
                        help="T2A classifier-free guidance scale (--coherent_audio)")
    parser.add_argument("--crf", type=int, default=19, help="x264 CRF for the output")
    parser.add_argument("--interactive", action="store_true",
                        help="Zero-shot mimicry of MG3's interactive mode: prompt for a "
                             "camera action between segments; the next segment is "
                             "conditioned on a warped last frame (no trained action weights)")
    parser.add_argument("--actions", type=str, default="",
                        help="Semicolon-separated scripted steering, one entry per "
                             "iteration: a warp alias (left/right/forward/...) or "
                             "free-form motion text (e.g. 'left;the camera drifts "
                             "past the gas station'); non-interactive alternative")
    parser.add_argument("--segment_frames", type=int, default=SEGMENT_FRAMES,
                        help="Frames per iteration after the first (must be 1 mod 8; "
                             "shorter = faster action response in interactive mode)")
    return parser.parse_args()


def _check_assets(args):
    missing = [p for p in (args.ltx_checkpoint, args.gemma_root, args.image)
               if not Path(p).exists()]
    if not args.one_stage and not Path(args.spatial_upsampler).exists():
        missing.append(args.spatial_upsampler)
    if getattr(args, "coherent_audio", False) or getattr(args, "audio_track", ""):
        for p in (args.ltx_dev_checkpoint, args.distilled_lora):
            if not Path(p).exists():
                missing.append(p)
        if args.audio_track and not Path(args.audio_track).exists():
            missing.append(args.audio_track)
    if not (LTX_ROOT / "packages" / "ltx-core" / "src").is_dir():
        missing.append(str(LTX_ROOT) + " (set LTX_ROOT)")
    if missing:
        raise FileNotFoundError("Missing required paths:\n  " + "\n  ".join(missing))


def _build_pipeline(args):
    """Construct the LTX pipeline (loads transformer, VAEs, Gemma)."""
    if getattr(args, "audio_track_path", None):
        # Coherent-audio mode: A2Vid two-stage (dev ckpt + distilled LoRA).
        # The audio modality is FROZEN to the given soundtrack slice — the
        # video is generated to match it — so concatenated segments share one
        # continuous track instead of re-imagining the music every segment.
        from ltx_core.components.guiders import MultiModalGuiderParams
        from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
        from ltx_pipelines.a2vid_two_stage import A2VidPipelineTwoStage

        pipeline = A2VidPipelineTwoStage(
            checkpoint_path=args.ltx_dev_checkpoint,
            distilled_lora=[LoraPathStrengthAndSDOps(
                args.distilled_lora, 1.0, LTXV_LORA_COMFY_RENAMING_MAP)],
            spatial_upsampler_path=args.spatial_upsampler,
            gemma_root=args.gemma_root,
            loras=[],
        )
        guider = MultiModalGuiderParams(cfg_scale=args.guidance_scale)

        def run_segment(prompt, seed, height, width, num_frames, images,
                        audio_start_time=0.0):
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
                images=list(images),  # ImageConditioningInput namedtuples (.path access inside)
                audio_path=args.audio_track_path,
                audio_start_time=audio_start_time,
                audio_max_duration=num_frames / args.frame_rate,
                enhance_prompt=args.enhance_prompt,
            )
        return run_segment
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

    frame = chunks[-1][-1].clamp(0, 1).float().cpu().numpy()
    Image.fromarray((frame * 255.0).round().astype(np.uint8)).save(path)


def _generate_audio_track(args, total_frames, wav_path):
    """Generate one continuous soundtrack for the whole video via the
    text-to-audio pipeline (dev checkpoint), so every segment can be
    conditioned on a slice of the SAME music/SFX bed."""
    from ltx_core.components.guiders import MultiModalGuiderParams
    from ltx_pipelines.t2a_one_stage import T2AOneStagePipeline
    from ltx_pipelines.utils.media_io import encode_audio

    prompt = args.audio_prompt or (
        f"{args.prompt} Audio: one continuous unbroken musical track matching the "
        "scene's mood, with a consistent tempo, key and instrumentation from start "
        "to finish, plus subtle ambient sound effects that fit the environment. "
        "No vocals, no silence, no abrupt style or instrument changes."
    )
    duration = total_frames / args.frame_rate
    print(f"[LTX backend] generating coherent soundtrack: {duration:.1f}s via T2A "
          f"({args.audio_num_inference_steps} steps)")
    pipeline = T2AOneStagePipeline(
        checkpoint_path=args.ltx_dev_checkpoint,
        gemma_root=args.gemma_root,
        loras=[],
    )
    audio = pipeline(
        prompt=prompt,
        negative_prompt="",
        seed=args.seed,
        num_frames=total_frames,
        frame_rate=args.frame_rate,
        num_inference_steps=args.audio_num_inference_steps,
        audio_guider_params=MultiModalGuiderParams(cfg_scale=args.audio_cfg_scale),
    )
    encode_audio(audio, str(wav_path))
    print(f"[LTX backend] soundtrack saved -> {wav_path}")
    return wav_path


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
# Zero-shot action control: warp the conditioning frame to mimic camera motion
# (no trained action weights — the model is pinned to a first frame that looks
# like the camera already moved, and its learned motion priors continue it)
# ---------------------------------------------------------------------------
_ACTION_ALIASES = {
    "w": "forward", "forward": "forward",
    "s": "backward", "backward": "backward",
    "a": "left", "left": "left",
    "d": "right", "right": "right",
    "up": "up", "down": "down",
    "n": "none", "none": "none", "": "none",
}
# zoom = dolly factor (>1 forward); dx/dy = crop-window shift as a fraction of
# frame size, in the direction the camera turns.
_ACTION_WARPS = {
    "forward":  {"zoom": 1.15},
    "backward": {"zoom": 0.87},
    "left":     {"dx": -0.12},
    "right":    {"dx": 0.12},
    "up":       {"dy": -0.10},
    "down":     {"dy": 0.10},
}
# Each warp action also injects a matching motion phrase into the segment's
# prompt — Gemma steers generation semantically, the warp steers geometrically.
_MOTION_PHRASES = {
    "forward":  "the camera moves forward",
    "backward": "the camera pulls back",
    "left":     "the camera turns left",
    "right":    "the camera turns right",
    "up":       "the camera tilts up",
    "down":     "the camera tilts down",
}


def _warp_frame(src_path, action, dst_path):
    """Apply the camera-motion warp for `action` to the conditioning image."""
    from PIL import Image, ImageFilter

    img = Image.open(src_path).convert("RGB")
    w, h = img.size
    spec = _ACTION_WARPS[action]
    zoom = spec.get("zoom", 1.0)
    dx = spec.get("dx", 0.0) * w
    dy = spec.get("dy", 0.0) * h
    if zoom < 1.0:
        # Dolly out: paste the frame onto a blurred upscaled copy of itself so
        # the exposed borders carry scene color for the model to extend.
        bw, bh = round(w / zoom), round(h / zoom)
        canvas = img.resize((bw, bh), Image.LANCZOS).filter(
            ImageFilter.GaussianBlur(radius=max(2, round((1.0 - zoom) * 60))))
        canvas.paste(img, ((bw - w) // 2, (bh - h) // 2))
        warped = canvas.resize((w, h), Image.LANCZOS)
    else:
        cw, ch = w / zoom, h / zoom
        left = min(max((w - cw) / 2 + dx, 0.0), w - cw)
        top = min(max((h - ch) / 2 + dy, 0.0), h - ch)
        warped = img.crop((round(left), round(top),
                           round(left + cw), round(top + ch))).resize((w, h), Image.LANCZOS)
    warped.save(dst_path)


def _resolve_steering(args, it):
    """Steering for the segment after segment it.

    Returns (warp_action_or_None, motion_text_or_None, quit_flag). The action
    may be a warp alias (geometric warp + canned motion phrase) or free-form
    motion text (semantic steering only) — in the MG series the control signal
    is just conditioning, so text input is a first-class action channel.
    """
    if args.interactive:
        while True:
            raw = input(f"[LTX interactive] after segment {it}: action "
                        "w/a/s/d/up/down, none, quit, or motion text > ").strip()
            low = raw.lower()
            if low in ("q", "quit"):
                return None, None, True
            action = _ACTION_ALIASES.get(low)
            if action is not None:
                if action == "none":
                    return None, None, False
                return action, _MOTION_PHRASES[action], False
            if raw:
                return None, raw, False  # free-form motion text
            return None, None, False     # empty input = no steering
    if args.actions:
        seq = [s.strip() for s in args.actions.split(";")]
        if it < len(seq) and seq[it]:
            action = _ACTION_ALIASES.get(seq[it].lower())
            if action is not None:
                if action == "none":
                    return None, None, False
                return action, _MOTION_PHRASES[action], False
            return None, seq[it], False  # free-form motion text
    return None, None, False


def _apply_steering(args, it, last_frame_path, tmp_dir):
    """Apply the chosen steering. Returns (cond_image_path, motion_text) for
    the next segment, or None to stop early."""
    warp, motion, quit_ = _resolve_steering(args, it)
    if quit_:
        return None
    cond_path = last_frame_path
    if warp is not None:
        cond_path = Path(tmp_dir) / f"cond_{it:03d}_{warp}.png"
        _warp_frame(last_frame_path, warp, cond_path)
        print(f"[LTX backend] action '{warp}' -> warped conditioning frame {cond_path}")
    if motion:
        print(f"[LTX backend] motion text: '{motion}'")
    return cond_path, motion


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
    prompt_it = args.prompt
    try:
        for it in range(args.num_iterations):
            num_frames = FIRST_SEGMENT_FRAMES if it == 0 else args.segment_frames
            seg_path = output_dir / f"{args.save_name}_seg{it:03d}.mp4"
            images = [ImageConditioningInput(path=str(cond_image_path),
                                             frame_idx=0, strength=1.0)]
            t0 = time.time()
            for _ in controller.stream(
                output_path=str(seg_path),
                prompt=prompt_it,
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
                steering = _apply_steering(args, it, cond_image_path, tmp_dir)
                if steering is None:
                    break
                cond_image_path, motion = steering
                prompt_it = f"{args.prompt}, {motion}" if motion else args.prompt
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
    if args.segment_frames % 8 != 1:
        raise ValueError(f"--segment_frames must be 1 mod 8 (LTX constraint), "
                         f"got {args.segment_frames}.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_frames = FIRST_SEGMENT_FRAMES + (args.num_iterations - 1) * (args.segment_frames - 1)

    args.audio_track_path = None
    if args.audio_track or args.coherent_audio:
        if args.mgpu:
            raise ValueError("Coherent-audio mode uses the A2Vid pipeline, which has "
                             "no MGPU variant — run single-GPU (drop --mgpu).")
        if args.one_stage:
            raise ValueError("Coherent-audio mode is two-stage only (drop --one_stage).")
        if args.audio_track:
            args.audio_track_path = args.audio_track
        else:
            args.audio_track_path = str(_generate_audio_track(
                args, total_frames, output_dir / f"{args.save_name}_soundtrack.wav"))

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
    prompt_it = args.prompt
    tmp_dir = Path(tempfile.mkdtemp(prefix="ltx_cond_", dir=output_dir))

    for it in range(args.num_iterations):
        num_frames = FIRST_SEGMENT_FRAMES if it == 0 else args.segment_frames
        seed = args.seed + it  # deterministic variation per segment
        images = [ImageConditioningInput(path=str(cond_image_path), frame_idx=0, strength=1.0)]

        t0 = time.time()
        if args.audio_track_path:
            # Slice the soundtrack at this segment's window. Segment i>=1 spans
            # output frames [FIRST-1 + (i-1)*(seg-1), ...] (its first frame is
            # the duplicated conditioning frame), so its audio starts there.
            start_frame = 0 if it == 0 else FIRST_SEGMENT_FRAMES - 1 + (it - 1) * (args.segment_frames - 1)
            video_iter, audio = run_segment(
                prompt_it, seed, height, width, num_frames, images,
                audio_start_time=start_frame / args.frame_rate)
        else:
            video_iter, audio = run_segment(
                prompt_it, seed, height, width, num_frames, images)
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

        # Next segment is conditioned on this segment's last decoded frame,
        # steered (warp + motion text) by the chosen action in
        # interactive/scripted mode.
        if it < args.num_iterations - 1:
            cond_image_path = tmp_dir / f"cond_{it:03d}.png"
            _save_last_frame(seg_chunks, cond_image_path)
            steering = _apply_steering(args, it, cond_image_path, tmp_dir)
            if steering is None:
                break
            cond_image_path, motion = steering
            prompt_it = f"{args.prompt}, {motion}" if motion else args.prompt

        all_chunks.extend(seg_chunks)
        all_audios.append(audio)

    final_audio = None
    n_frames = sum(c.shape[0] for c in all_chunks)
    if not args.no_audio:
        if args.audio_track_path:
            # Mux the soundtrack itself (lossless, seamless by construction),
            # trimmed to the actual video duration.
            from ltx_pipelines.utils.media_io import decode_audio_from_file
            final_audio = decode_audio_from_file(
                args.audio_track_path, torch.device("cpu"),
                0.0, n_frames / args.frame_rate)
        else:
            final_audio = _concat_audios(all_audios, args.frame_rate)

    output_path = output_dir / f"{args.save_name}.mp4"
    encode_video(video=iter(all_chunks), fps=int(args.frame_rate),
                 audio=final_audio, output_path=str(output_path),
                 video_chunks_number=len(all_chunks), crf=args.crf)
    print(f"[LTX backend] saved {n_frames} frames -> {output_path}")


def main():
    args = parse_args()
    generate(args)


if __name__ == "__main__":
    main()
