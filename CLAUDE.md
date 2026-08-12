# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Layout

This is a monorepo of three **independent** research releases of Skywork AI's interactive world models. Each subdirectory is a self-contained project with its own conda environment, `requirements.txt`, model checkpoints, and inference entry points — there is no shared code, no cross-imports between versions, no test suite, and no build/lint tooling. Treat each as a separate codebase.

| Dir | Base model lineage | Scale | Domain |
|---|---|---|---|
| `Matrix-Game-1/` | HunyuanVideo DiT | 17B | Minecraft |
| `Matrix-Game-2/` | Wan (SkyReels-V2), Self-Forcing causal autoregression | — | universal / GTA driving / TempleRun |
| `Matrix-Game-3/` | Wan2.2 (5B, or 2×14B MoE), DMD-distilled | 5B / 28B | Unreal-engine + real-world scenes |

All three generate video conditioned on keyboard/mouse (camera) actions from an initial image. Model weights are **not in the repo** — they are downloaded separately from HuggingFace (`Skywork/Matrix-Game*`).

## Common Commands

There is nothing to build or test. "Running" means inference, which requires a large NVIDIA GPU (MG1: ≥80GB VRAM; MG2: ≥24GB; MG3: A/H-series, multi-GPU supported) and downloaded checkpoints.

**Matrix-Game-1** (env: `pip install -r Matrix-Game-1/requirements.txt`, plus NVIDIA apex and FlashAttention-3):
```bash
cd Matrix-Game-1
bash run_inference.sh      # single-GPU; edit MODEL_ROOT etc. env vars in the script first
bash run_2gpu.sh           # 2-GPU via parallel_infer.py
python inference_bench.py --dit_path ... --vae_path ... --textenc_path ... --bfloat16
```

**Matrix-Game-2** (env: conda python=3.10, `pip install -r requirements.txt` then `python setup.py develop` — the latter installs the `wan`/`pipeline`/`utils` packages; both steps required):
```bash
cd Matrix-Game-2
python inference.py --config_path configs/inference_yaml/inference_universal.yaml \
    --checkpoint_path <ckpt> --img_path <image> --pretrained_model_path <vae-folder> \
    --output_folder outputs --num_output_frames 150 --seed 42
python inference_streaming.py ...   # interactive variant: reads keyboard input live during generation
```
Scene-specific configs live in `configs/inference_yaml/` (`inference_universal.yaml`, `inference_gta_drive.yaml`, `inference_templerun.yaml`).

**Matrix-Game-3** (env: conda python=3.12, `pip install -r requirements.txt`, plus FlashAttention):
```bash
cd Matrix-Game-3
bash test.sh                        # canonical multi-GPU run (8 GPUs sync / 7 with async VAE)
torchrun --nproc_per_node=$N generate.py --size 704*1280 --dit_fsdp --t5_fsdp \
    --ckpt_dir Matrix-Game-3.0 --fa_version 3 --use_int8 --num_iterations 12 \
    --num_inference_steps 3 --image <img> --prompt "..." --vae_type mg_lightvae \
    --lightvae_pruning_rate 0.5 --output_dir ./output
```
Key `generate.py` flags: `--interactive` (custom action input), `--use_base_model --num_inference_steps 50` (base instead of distilled model), `--use_async_vae` (dedicate a GPU to VAE decode), `--vae_type mg_lightvae_v2 --lightvae_pruning_rate 0.75` (faster VAE). Total frames = `57 + (num_iterations - 1) * 40`. Single-GPU runs auto-disable FSDP when `ulysses_size <= 1`.

**LTX-2.3 backend (local addition)**: `generate_ltx.py` (+ `test_ltx.sh`) replaces the Wan2.2 backbone with LTX-2.3 (22B audio-video DiT) using the same segmented autoregressive scheme (57 + N×40 frames, last decoded frame re-injected as next segment's conditioning). It sys.path-injects `ltx-core`/`ltx-pipelines` from `LTX_ROOT` (default `/root/learning/LTX-2`), uses checkpoints from `/data/models/Lightricks--LTX-2.3/snapshots/master/` and Gemma-3-12B-IT from `/data1/models/google--gemma-3-12b-it/snapshots/master`, and runs under `/data1/ltx-world-model/.venv`. `--mgpu` runs one generation across all visible GPUs (ltx-pipelines MGPU controller: sequence parallelism + Accelerate Gemma + distributed VAE; latency tool, full replica per rank, fp8-cast; requires `ltx-kernels` built from `$LTX_ROOT/packages/ltx-kernels`). No action/camera conditioning (that requires the trained checkpoints from the sibling project `/data1/ltx-world-model`, which has its own inference entry `scripts/inference/infer_bidirectional_camera.py`).

**GameWorldScore benchmark** (MG1's evaluation suite, `Matrix-Game-1/GameWorldScore/`): `evaluate.py`, `evaluate_per_action.py`, `evaluate_per_scene.py` score generated Minecraft videos on visual quality, temporal consistency, action controllability, and physical-rule understanding.

## Architecture Notes

**Matrix-Game-1** — classic full-video diffusion transformer:
- `matrixgame/model_variants/matrixgame_dit_src/matrixgame_i2v.py` — the 17B image-to-video DiT.
- `matrixgame/encoder_variants/` — T5 text encoder; `matrixgame/vae_variants/` — causal 3D VAE.
- `matrixgame/sample/pipeline_matrixgame.py` — the monolithic (~50KB) diffusers-style sampling pipeline where action conditioning, denoising loop, and decoding all live; `flow_matching_scheduler_matrixgame.py` is its scheduler.
- `parallel_infer.py` — multi-GPU inference; `teacache_forward.py` — TeaCache timestep-caching acceleration for the DiT forward pass.
- `condtions.py` / `config.py` (repo root of MG1) — action/mouse-condition encoding and run configuration.

**Matrix-Game-2** — causal autoregressive (block-wise) streaming diffusion, adapted from Self-Forcing:
- `pipeline/causal_inference.py` — the core: `CausalInferencePipeline` generates video in blocks of frames, maintaining **KV caches** (`_initialize_kv_cache`, cross-attention cache, and separate mouse/keyboard condition caches) so each new block attends to previously generated context. `CausalInferenceStreamingPipeline` is the variant that polls live keyboard input between blocks.
- `wan/modules/causal_model.py` + `action_module.py` — causal DiT and the keyboard/mouse action-conditioning module; `wan/modules/model.py` is the non-causal base.
- `utils/wan_wrapper.py` (`WanDiffusionWrapper`, `WanVAEWrapper`) — thin wrappers bridging the `wan` package to the pipeline; `utils/scheduler.py` — the few-step diffusion scheduler used with distilled checkpoints.
- `demo_utils/` — alternative fast VAE decoders (`taehv.py`, `vae_block3.py`, `vae_torch2trt.py` for TensorRT) used to hit real-time 25fps.
- `wan/` is a vendored fork of the Wan video-model codebase — it looks like a library but is edited in place; changes belong there, not upstream.

**Matrix-Game-3** — memory-augmented streaming generation at 720p/40fps:
- `pipeline/inference_pipeline.py` (`MatrixGame3Pipeline`) — multi-segment autoregressive inference with prediction-residual frame re-injection for long-horizon consistency; `inference_interactive_pipeline.py` — the `--interactive` variant; `vae_worker.py` — runs VAE decoding on a separate GPU as an async worker (`--use_async_vae`).
- `wan/distributed/` — `fsdp.py` (DiT/T5 sharding), `ulysses.py` + `sequence_parallel.py` (context parallel; `--ulysses_size` must match GPU count — see `test.sh`).
- `wan/modules/vae2_2.py` — Wan2.2 VAE plus the distilled **LightVAE** variants selected by `--vae_type`/`--lightvae_pruning_rate`; `wan/triton_kernels.py` — Triton kernels backing the `--use_int8` quantized DiT (quantization framework adapted from LightX2V).
- `utils/cam_utils.py` — camera-pose math for the camera-aware memory mechanism.
- `wan/configs/config.py` — `WAN_CONFIGS["matrix_game3"]` holds model defaults (sample_shift, guidance scale); `MAX_AREA_CONFIGS` maps `--size` strings.

## Conventions and Gotchas

- Python versions differ per subproject (MG1/MG2: 3.10; MG3: 3.12) — don't share environments.
- FlashAttention must be installed manually per subproject README; MG2/MG3 select implementation via `--fa_version` (FA3 on Hopper).
- Inference scripts expect checkpoint paths passed as CLI args or env vars edited into the shell scripts; there are no checked-in weights and no default download step in the code.
- `Matrix-Game-2/inference.py` defaults `--pretrained_model_path` to a `Matrix-Game-2.0/` folder that only exists after the HuggingFace download.
- Known MG2 issue: upward camera movement can cause brief black frames (documented in its README).
