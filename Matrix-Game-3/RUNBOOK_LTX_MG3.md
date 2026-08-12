# Runbook: Real-Time Gaming Pipeline with LTX-MG3

**Target machine**: 4× NVIDIA H20 96GB (fully dedicated — no co-tenant training).
**Source of truth**: all steps below were validated on an 8×H20 shared box on 2026-08-12.

---

## 0. Scope and honest expectations

The MG3 "realtime gaming experience" is four properties, not one:

| Property | Where it comes from | Status for LTX today |
|---|---|---|
| High-quality video (+audio) | Stock LTX-2.3 checkpoints | ✅ Working (`generate_ltx.py`) |
| Action/camera control | **Trained** world-model weights | ⏳ `ltx-world-model` stage-1 training underway; no checkpoint released yet |
| Streaming/causal generation (KV cache, frame-by-frame) | Stage-2 (teacher-forcing AR) + inference loop | ⏳ Needs stage-2 checkpoint |
| Interactive speed (few-step denoise) | Stage-3 (causal ODE) + stage-4 (DMD) distillation | ⏳ Needs stage-3/4 checkpoints |

`generate_ltx.py` on stock checkpoints gives property 1 only: offline, segment-by-segment, text+image conditioned. It is **not** interactive and cannot be made interactive by configuration — the action-conditioning weights and causal masking simply do not exist in the stock LTX-2.3 checkpoint.

Speed expectation: MG3 hits 40fps@720p with a **5B** DMD-distilled model + INT8 + LightVAE. LTX-2.3 is **22B** — even after stage-4 DMD distillation, plan for single-digit-to-low-teens fps at 720p on 4×H20, and target ~480–576p for true interactive rates first. Treat 40fps@720p as a research goal, not a config switch.

---

## 1. Provision the 4×H20 machine

### 1.1 System prerequisites

- Linux, CUDA 13.x toolkit with `nvcc` on PATH (`/usr/local/cuda`). Verify: `nvcc --version`.
- Python 3.12 via `uv` (the validated env is a uv venv).
- `ninja` (needed to build `ltx-kernels`): `uv pip install ninja` or system package.
- ~200GB free disk (LTX-2.3 checkpoints ≈ 90GB, Gemma ≈ 24GB, env ≈ 30GB).

### 1.2 Python environment

```bash
uv venv /data1/ltx-world-model/.venv --python 3.12   # or reuse an existing path
# torch 2.13.0+cu130 (validated), plus pipeline deps:
uv pip install --python /data1/ltx-world-model/.venv/bin/python \
    torch --index-url https://download.pytorch.org/whl/cu130
uv pip install --python /data1/ltx-world-model/.venv/bin/python \
    diffusers transformers accelerate safetensors einops omegaconf pyyaml \
    av openimageio cloudpickle pillow numpy sentencepiece ninja
```

> The venv has no `pip` — always install via `uv pip install --python <venv>/bin/python`.

### 1.3 LTX-2 monorepo

```bash
git clone https://github.com/Lightricks/LTX-2 /root/learning/LTX-2   # or rsync from the validation box
export LTX_ROOT=/root/learning/LTX-2
```

Nothing from the monorepo is pip-installed; `generate_ltx.py` sys.path-injects
`packages/ltx-core/src` and `packages/ltx-pipelines/src` at runtime.

### 1.4 Build `ltx-kernels` (required for `--mgpu`)

Two pitfalls, both validated on the validation box — follow exactly:

```bash
cd $LTX_ROOT/packages/ltx-kernels
# Pitfall 1: pinned CUTLASS git fetch can fail silently. Prefetch manually:
#   (setup.py honors CUTLASS_DIR / LTX_KERNELS_CACHE_DIR; the pinned commit is
#    afa1772203677c5118fcd82537a9c8fefbcc7008)
# Pitfall 2: pip's nvidia/cu13 headers (13.0) shadow the toolkit's (13.2) and
# nvcc aborts with "CUDA compiler and CUDA toolkit headers are incompatible".
# Fix: prepend the real toolkit headers.
CUDA_HOME=/usr/local/cuda NVCC_PREPEND_FLAGS="-I/usr/local/cuda/include" \
  uv pip install --python /data1/ltx-world-model/.venv/bin/python \
  . --no-build-isolation
# Verify:
/data1/ltx-world-model/.venv/bin/python -c "import ltx_kernels; print('ltx-kernels OK')"
```

Single-GPU runs do **not** need `ltx-kernels`; only `--mgpu` does.

### 1.5 FlashAttention (optional, MG3 Wan backend only)

The LTX path uses its own attention; skip FA unless you also run the Wan2.2 backend.

---

## 2. Model assets

| Asset | Size | Source path on the validation box | Notes |
|---|---|---|---|
| LTX-2.3 checkpoints | ~90GB | `/data/models/Lightricks--LTX-2.3/snapshots/master/` | `ltx-2.3-22b-distilled-1.1.safetensors` (default), `ltx-2.3-22b-dev.safetensors` (one-stage), `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` |
| Gemma-3-12B-IT (text encoder) | ~24GB | `/data1/models/google--gemma-3-12b-it/snapshots/master` | Required for every run (prompt encoding) |
| Trained world-model ckpts | TBD | `/data1/ltx-world-model/checkpoints/stage*_*/` | Not yet available — phase 4 |

`rsync -avP` these to matching paths on the new machine, or pass every path via CLI
flags (`--ltx_checkpoint`, `--spatial_upsampler`, `--gemma_root`). VAEs live **inside**
the monolithic checkpoints (`vae.*` keys) — no separate VAE download.

---

## 3. Validate: offline LTX generation (today's capability)

### 3.1 Single GPU (fits in 96GB easily)

```bash
cd Matrix-Game-3
bash test_ltx.sh            # 12 iterations -> 481 frames @ 704x1280
```

### 3.2 All 4 GPUs (latency-optimized, numerically equivalent)

```bash
cd Matrix-Game-3
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /data1/ltx-world-model/.venv/bin/python generate_ltx.py --mgpu \
  --image demo_images/001/image.png \
  --prompt "A colorful, animated cityscape with a gas station and various buildings." \
  --size 704*1280 --num_iterations 12 --seed 42 \
  --output_dir ./output_ltx --save_name full_run
```

With the whole machine free there is no OOM contention — 704×1280 works (validated
single-segment on the validation box at 61.6s/57 frames on 8 GPUs while sharing with training).

Hard requirements learned the hard way:

- **`--size` must be divisible by 64** in two-stage mode (`generate_ltx.py` now guards this).
- Frames per segment are fixed by the MG3 pacing: 57 first, then 41 (both ≡1 mod 8, an LTX constraint). `--num_iterations N` → `57 + (N-1)*40` frames.
- `--mgpu` is distilled-mode only; `--one_stage` is single-GPU only.

This is the deliverable for "same high quality game playing" video. It is offline:
a 481-frame video takes tens of minutes. Do not mistake this for the realtime pipeline.

---

## 4. The realtime gaming pipeline (requires trained checkpoints)

The realtime path is owned by the sibling project `/data1/ltx-world-model` (minWM 4-stage
recipe). Deploy each stage as its checkpoint lands:

### Stage 1 — Bidirectional SFT (training now, v4)

- Gives: camera-conditioned (PRoPE) **offline** inference.
- Entry: `/data1/ltx-world-model/scripts/inference/infer_bidirectional_camera.py`
  (`--checkpoint <stage1 ckpt> --config <yaml> --camera_npz <traj> --image <img>`,
  50 ODE steps default, CFG). Launcher: `run_infer_bidirectional_camera.sh`.
- This is camera-controllable but **not streaming and not interactive-speed**.

### Stage 2 — Teacher-forcing AR

- Gives: causal masking → block-wise autoregressive generation with KV cache.
- Wire into an MG3-style streaming loop (cf. `Matrix-Game-2/pipeline/causal_inference.py`
  for the KV-cache pattern and `Matrix-Game-3/pipeline/inference_interactive_pipeline.py`
  for action polling between blocks).

### Stage 3 — Causal ODE / consistency distillation

- Gives: 1–4 step denoising per block → interactive latency begins.

### Stage 4 — Asymmetric DMD

- Gives: self-rollout long-horizon stability; the MG3-comparable endpoint.

> ⚠️ `src/ltx_world_model/deploy_adapter.py` is a **broken skeleton** (wrong paths,
> zeroed positions, toy ODE loop). Do not build on it; use
> `scripts/inference/infer_bidirectional_camera.py` as the reference implementation
> and port its conditioning logic into the streaming loop.

---

## 5. Latency tuning ladder (apply in order once stage-3/4 ckpts exist)

1. **fp8 transformer** — default in the MGPU path (`_build_fp8_cast_policy`), ~23GB replica.
2. **Sequence parallelism across all 4 GPUs** (`--mgpu`-style; requires `ltx-kernels`).
3. **Distributed/async VAE decode** — dedicate GPU 3 to VAE (MG3's `--use_async_vae` pattern; ltx-pipelines has a distributed VAE decoder used by MGPU).
4. **Resolution ladder**: start 480×832 for interactivity, raise to 704×1280 for quality.
5. **Few-step schedule**: stage-3/4 checkpoints → 1–4 denoise steps per block.
6. **CPU offload is the opposite of realtime** — never enable it on the gaming path.

Realistic first target on 4×H20: **8–16fps at 480–576p** with audio, scaling toward
720p as distillation and kernels mature.

---

## 6. Troubleshooting appendix (all hit and solved on the validation box)

| Symptom | Cause | Fix |
|---|---|---|
| `CUDA compiler and CUDA toolkit headers are incompatible` (ltx-kernels build) | pip nvidia/cu13 include (13.0) shadows toolkit 13.2 | `NVCC_PREPEND_FLAGS="-I/usr/local/cuda/include"` with `CUDA_HOME=/usr/local/cuda` |
| CUTLASS fetch exit 128 during build | pinned commit fetch failed; empty cache repo left behind | `rm -rf` the cache dir under `~/.cache/ltx-kernels/`, re-run fetch manually, rebuild |
| `Resolution (HxW) is not divisible by 64` | two-stage pipeline constraint | use multiples of 64 (e.g. 512×832, 704×1280) |
| OOM only on **segment 1+** during Gemma encode | inter-segment allocator fragmentation | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; on shared machines also drop `--size` |
| `command not found: pip` inside venv | uv venv has no pip | `uv pip install --python <venv>/bin/python ...` |
| LTX rejects frame count | frames must be ≡1 mod 8 | keep MG3 pacing (57, then 41) — already ≡1 mod 8 |
