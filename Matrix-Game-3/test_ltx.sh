#!/usr/bin/env bash
# LTX-2.3 backend smoke run — single GPU, mirrors test.sh for the Wan2.2 backend.
# Requires: a free GPU (~70GB+ VRAM for the 22B DiT + Gemma-3-12B + VAEs),
# and the LTX-2 monorepo (LTX_ROOT, default /root/learning/LTX-2).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Use the LTX-capable venv (torch + ltx-core deps + av/openimageio).
PYTHON="${PYTHON:-/data1/ltx-world-model/.venv/bin/python}"

exec "$PYTHON" generate_ltx.py \
  --image demo_images/001/image.png \
  --prompt "A colorful, animated cityscape with a gas station and various buildings." \
  --size 704*1280 \
  --num_iterations 12 \
  --seed 42 \
  --frame_rate 24 \
  --output_dir ./output_ltx \
  --save_name test \
  "$@"
