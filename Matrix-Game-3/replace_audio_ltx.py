#!/usr/bin/env python
"""Retrofit coherent audio onto already-generated LTX videos.

The segmented generation loop gives each ~2s segment an independently imagined
soundtrack, so music snaps to a new style at every boundary. This script
generates ONE continuous text-to-audio soundtrack per video (same duration,
scene-matched prompt) and swaps it in, replacing the stitched per-segment audio.

Input: a TSV manifest, one row per video:
    path/to/video.mp4<TAB>scene description / audio prompt

The text-to-audio pipeline (dev checkpoint) is loaded ONCE for all rows.
Output: <video_stem>_coherent.mp4 next to each input. Requires ffmpeg on PATH.

Example:
    LTX_ROOT=/data1/LTX-2 /data1/ltx-world-model/.venv/bin/python \
        replace_audio_ltx.py --manifest retrofit_audio.tsv
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import av
import torch

import generate_ltx as G  # noqa: F401  (sys.path injection of ltx packages)

AUDIO_CONTINUITY_SUFFIX = (
    " Audio: one continuous unbroken musical track matching the scene's mood, "
    "with a consistent tempo, key and instrumentation from start to finish, "
    "plus subtle ambient sound effects that fit the environment. "
    "No vocals, no silence, no abrupt style or instrument changes."
)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=str, required=True,
                        help="TSV rows: video_path<TAB>prompt")
    parser.add_argument("--ltx_dev_checkpoint", type=str, default=str(G.DEFAULT_DEV_CKPT))
    parser.add_argument("--gemma_root", type=str, default=G.DEFAULT_GEMMA)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--cfg_scale", type=float, default=3.0)
    parser.add_argument("--frame_rate", type=float, default=24.0)
    args = parser.parse_args()

    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found on PATH (needed to remux audio).")

    rows = []
    for line in Path(args.manifest).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        video, prompt = line.split("\t", 1)
        if not Path(video).exists():
            print(f"[retrofit] SKIP (missing video): {video}")
            continue
        rows.append((Path(video), prompt.strip()))
    print(f"[retrofit] {len(rows)} videos")

    from ltx_core.components.guiders import MultiModalGuiderParams
    from ltx_pipelines.t2a_one_stage import T2AOneStagePipeline
    from ltx_pipelines.utils.media_io import encode_audio

    print("[retrofit] loading T2A pipeline (dev checkpoint, one-time load)...")
    pipeline = T2AOneStagePipeline(
        checkpoint_path=args.ltx_dev_checkpoint,
        gemma_root=args.gemma_root,
        loras=[],
    )

    for i, (video, prompt) in enumerate(rows):
        duration = av.open(str(video)).duration / 1_000_000.0
        num_frames = round(duration * args.frame_rate)
        wav_path = video.with_name(video.stem + "_soundtrack.wav")
        out_path = video.with_name(video.stem + "_coherent.mp4")
        print(f"[retrofit] ({i + 1}/{len(rows)}) {video.name}: {duration:.1f}s "
              f"({num_frames} frames @ {args.frame_rate}fps)")

        audio = pipeline(
            prompt=prompt + AUDIO_CONTINUITY_SUFFIX,
            negative_prompt="",
            seed=args.seed,
            num_frames=num_frames,
            frame_rate=args.frame_rate,
            num_inference_steps=args.num_inference_steps,
            audio_guider_params=MultiModalGuiderParams(cfg_scale=args.cfg_scale),
        )
        encode_audio(audio, str(wav_path))

        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-i", str(video), "-i", str(wav_path),
             "-map", "0:v:0", "-map", "1:a:0",
             "-c:v", "copy", "-c:a", "aac", "-shortest", str(out_path)],
            check=True)
        print(f"[retrofit] saved -> {out_path}")


if __name__ == "__main__":
    main()
