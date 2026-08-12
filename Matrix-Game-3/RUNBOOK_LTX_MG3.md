# Runbook: Real-Time Gaming Pipeline with LTX-MG3

**Target machine**: 8× NVIDIA H20 96GB with GPUs 0–3 free for inference (GPUs 4–7 run training).
**Source of truth**: steps first validated on an 8×H20 shared box on 2026-08-12; re-validated end-to-end on the target machine the same day (§3.2 full run: 497 frames in ~2.5 min of generation).

---

## 0. Scope and honest expectations

The MG3 "realtime gaming experience" is four properties, not one:

| Property | Where it comes from | Status for LTX today |
|---|---|---|
| High-quality video (+audio) | Stock LTX-2.3 checkpoints | ✅ Working (`generate_ltx.py`; per-segment audio snaps at boundaries by default — coherent-audio workflow in §3.4) |
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
> Every package in the list above is load-bearing: `generate_ltx.py --mgpu` failed on a
> venv that predated it with `ModuleNotFoundError: openimageio`, then `cloudpickle`
> (both imported by ltx-pipelines media I/O and the MGPU fleet respectively).

### 1.3 LTX-2 monorepo

```bash
git clone https://github.com/Lightricks/LTX-2 /data1/LTX-2   # already present on the target machine
export LTX_ROOT=/data1/LTX-2
```

Nothing from the monorepo is pip-installed; `generate_ltx.py` sys.path-injects
`packages/ltx-core/src` and `packages/ltx-pipelines/src` at runtime. Its compiled-in
default is `LTX_ROOT=/root/learning/LTX-2` — on the target machine always export
`LTX_ROOT=/data1/LTX-2` (or edit the default).

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

| Asset | Size | Path on the target machine | Notes |
|---|---|---|---|
| LTX-2.3 checkpoints | ~90GB | `/data1/models/Lightricks--LTX-2.3/snapshots/master/` | `ltx-2.3-22b-distilled-1.1.safetensors` (default), `ltx-2.3-22b-dev.safetensors` (one-stage; also T2A/A2Vid base, §3.4), `ltx-2.3-22b-distilled-lora-384-1.1.safetensors` (A2Vid stage-2, §3.4), `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` |
| Gemma-3-12B-IT (text encoder) | ~24GB | `/data1/models/google--gemma-3-12b-it/snapshots/master` | Required for every run (prompt encoding) |
| Stage-1 world-model ckpt | ~44GB | `/data1/models/LTX-WM/checkpoint_step300_merged_bf16.safetensors` | Bidirectional SFT step 300, merged bf16 — see §4 |

`generate_ltx.py`'s compiled-in checkpoint default points at
`/data1/models/Lightricks--LTX-2.3/snapshots/master` (override the directory with the
`LTX_CKPT_DIR` env var if the checkpoints live elsewhere). VAEs live **inside**
the monolithic checkpoints (`vae.*` keys) — no separate VAE download.

---

## 3. Validate: offline LTX generation (today's capability)

### 3.1 Single GPU (fits in 96GB easily)

```bash
cd Matrix-Game-3
LTX_ROOT=/data1/LTX-2 bash test_ltx.sh   # 12 iterations -> 497 frames @ 704x1280
```

> `test_ltx.sh` relies on `generate_ltx.py`'s defaults, which now match this machine
> (checkpoints under `/data1/models/...`); it still needs `LTX_ROOT=/data1/LTX-2`
> exported. To run single-GPU by hand, take the §3.2 command, drop `--mgpu`, and set
> `CUDA_VISIBLE_DEVICES=<one free GPU>`.

### 3.2 All 4 GPUs (latency-optimized, numerically equivalent)

```bash
cd Matrix-Game-3
CUDA_VISIBLE_DEVICES=0,1,2,3 LTX_ROOT=/data1/LTX-2 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /data1/ltx-world-model/.venv/bin/python generate_ltx.py --mgpu \
  --image demo_images/001/image.png \
  --prompt "A colorful, animated cityscape with a gas station and various buildings." \
  --ltx_checkpoint /data1/models/Lightricks--LTX-2.3/snapshots/master/ltx-2.3-22b-distilled-1.1.safetensors \
  --spatial_upsampler /data1/models/Lightricks--LTX-2.3/snapshots/master/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
  --gemma_root /data1/models/google--gemma-3-12b-it/snapshots/master \
  --size 704*1280 --num_iterations 12 --seed 42 \
  --output_dir ./output_ltx --save_name full_run
```

Validated on the target machine (GPUs 0–3 idle): first segment 51.3s (includes Gemma
encode + warmup), then **~8.1s per 41-frame segment** → a 12-iteration, 497-frame video
finishes in **~2.5 min of generation** plus ~4 min model load. Output verified:
h264 704×1280 @ 24fps + AAC 48kHz audio, visually coherent across all segments.
(Earlier 8-GPU shared-box measurement: 61.6s per 57-frame segment while co-tenant
with training — idle GPUs are ~7× faster per segment.)

Hard requirements learned the hard way:

- **`--size` must be divisible by 64** in two-stage mode (`generate_ltx.py` now guards this).
- Frames per segment are fixed by the MG3 pacing: 57 first, then 41 (both ≡1 mod 8, an LTX constraint). `--num_iterations N` → `57 + (N-1)*40` frames (12 iterations = 497).
- `--mgpu` is distilled-mode only; `--one_stage` is single-GPU only.

This is the deliverable for "same high quality game playing" video. It is offline but
fast on idle GPUs — still not the interactive pipeline (no action conditioning, no
streaming), just no longer a batch job either.

### 3.3 Interactive UX on stock weights (zero-shot steering)

`generate_ltx_interactive.py` ports `pipeline/inference_interactive_pipeline.py`'s
operator experience to the LTX backend: the same stdin two-channel prompts between
segments (mouse I/K/J/L/U + keyboard W/S/A/D/Q), the same 57-then-41-frame pacing,
and per-segment + concatenated mp4 output. Because stock LTX-2.3 has no action
conditioning, each action steers the next segment zero-shot — a geometric warp
(crop/pan/zoom) of the conditioning frame plus a matching motion phrase appended to
the prompt. MG3's exact action→pose math is vendored in and a per-segment c2w pose
history (`<save_name>_pose_history.npy`) is logged, ready to be fed into PRoPE
`CameraParams` once a stage-1+ checkpoint from `ltx-world-model` is usable.

```bash
cd Matrix-Game-3
CUDA_VISIBLE_DEVICES=0 LTX_ROOT=/data1/LTX-2 \
  /data1/ltx-world-model/.venv/bin/python generate_ltx_interactive.py \
  --image demo_images/001/image.png \
  --prompt "A colorful, animated cityscape with a gas station and various buildings." \
  --num_iterations 6 --size 704*1280 \
  --output_dir ./output_ltx_interactive --save_name city_walk
```

Scripted (non-interactive) mode for smoke tests: `--actions "w+u;w+l;d+j;none"`.
Do **not** point `--ltx_checkpoint` at the LTX-WM step-300 checkpoint — it generates
pure noise (foreign-VAE training data; see ltx-world-model/docs/ROOT_CAUSE_STEP300.md).

### 3.4 Coherent audio across segments (a2vid audio-first workflow)

**Problem**: the segmented loop generates audio jointly *per segment* — every ~2s
segment re-imagines the soundtrack from scratch (new seed, motion-phrase prompt
suffix), so music/SFX snap to a different style at every segment boundary.

**Fix**: flip the order — generate (or supply) ONE continuous soundtrack for the
whole video first, then generate each segment's video *conditioned on its slice* of
that track. LTX-2.3 ships both halves of this:

- `T2AOneStagePipeline` (text-to-audio, dev checkpoint) — one full-length track,
- `A2VidPipelineTwoStage` — per-segment video generation with the audio modality
  **frozen** to `audio_track[audio_start_time : +duration]`; the final mux uses the
  source track itself, so audio is seamless by construction (and motion syncs to it).

```bash
cd Matrix-Game-3
CUDA_VISIBLE_DEVICES=0 LTX_ROOT=/data1/LTX-2 \
  /data1/ltx-world-model/.venv/bin/python generate_ltx.py \
  --image demo_images/001/image.png --prompt "<scene prompt>" \
  --coherent_audio \
  --size 704*1280 --num_iterations 12 --seed 42 \
  --output_dir ./output_ltx --save_name city_coherent
```

- `--coherent_audio` — auto-generate the soundtrack first (writes
  `<save_name>_soundtrack.wav`; override the T2A text with `--audio_prompt`,
  tune with `--audio_num_inference_steps` / `--audio_cfg_scale`).
- `--audio_track <file.wav|mp3>` — skip T2A and use your own music/SFX bed.
- Constraints: **single-GPU only** (A2Vid has no MGPU variant), **two-stage only**,
  and stage-1 runs a full CFG denoise → slower per segment than the §3.2 distilled
  path. Uses `--ltx_dev_checkpoint` + `--distilled_lora` (defaults already point at
  the target machine's files).

**Retrofitting existing videos** (no regeneration): `replace_audio_ltx.py` takes a
TSV manifest (`video_path<TAB>scene prompt` per line), loads T2A once, generates a
duration-matched coherent track per video, and ffmpeg-muxes it in:

```bash
CUDA_VISIBLE_DEVICES=0 LTX_ROOT=/data1/LTX-2 \
  /data1/ltx-world-model/.venv/bin/python replace_audio_ltx.py --manifest retrofit_audio.tsv
# outputs: <name>_coherent.mp4 + <name>_soundtrack.wav next to each input
```

The repo's `retrofit_audio.tsv` covers all 25 batch videos (prompts extracted from
`batch_demo_ltx.sh` / `batch_aaa_ltx.sh`).

---

## 4. The realtime gaming pipeline (requires trained checkpoints)

The realtime path is owned by the sibling project `/data1/ltx-world-model` (minWM 4-stage
recipe). Deploy each stage as its checkpoint lands:

### Stage 1 — Bidirectional SFT (training now, v4)

- Gives: camera-conditioned (PRoPE) **offline** inference.
- A merged bf16 step-300 checkpoint is already on the target machine:
  `/data1/models/LTX-WM/checkpoint_step300_merged_bf16.safetensors` — usable for a
  first camera-conditioned inference test before training fully converges.
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
| `ModuleNotFoundError: openimageio` / `cloudpickle` at `generate_ltx.py` import time | venv missing ltx-pipelines runtime deps | install the full §1.2 package list, not just torch |
| LTX rejects frame count | frames must be ≡1 mod 8 | keep MG3 pacing (57, then 41) — already ≡1 mod 8 |
