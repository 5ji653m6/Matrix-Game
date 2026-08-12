#!/usr/bin/env python
"""LTX-2.3 interactive backend: Matrix-Game-3's interactive pipeline ported to LTX.

This is the LTX-compatible counterpart of `pipeline/inference_interactive_pipeline.py`
(the Wan2.2 backend). It keeps MG3's interactive UX and segmented pacing verbatim:

  - Segment 0: 57 frames image-to-video from --image.
  - Between segments the operator is prompted on stdin for a mouse action
    (I/K/J/L/U = tilt up/tilt down/turn left/turn right/no move) and a keyboard
    action (W/S/A/D/Q = forward/back/left/right/no move) — the same two-channel
    input scheme as the Wan interactive pipeline.
  - Segment i: --segment_frames (default 41) frames conditioned on the steered
    last --reinject_frames (default 8) decoded frames of segment i-1
    (multi-frame re-injection via stock LTX keyframe conditioning — frame_idx 0
    is an exact latent replacement, frame_idx>0 are clean keyframe tokens,
    which is MG3's memory-bank mechanism). The re-injected frames are dropped
    from the output -> segment_frames - reinject_frames new frames per segment.
  - Per-segment mp4s plus one concatenated mp4 with audio are written.

CONTROL REALITY (stock checkpoints): LTX-2.3 stock weights have no trained
action/camera conditioning (PRoPE enters only via ltx-world-model stage-1+
checkpoints, and the step-300 SFT checkpoints are poisoned — see
ltx-world-model/docs/ROOT_CAUSE_STEP300.md). So actions steer through the
zero-shot channel, two modes (--steer_mode):

  - reproject (default): depth-based camera reprojection (steer3d.py) —
    estimate monocular depth of each tail frame, unproject to a point cloud,
    move a virtual pinhole camera by the action's rotation/translation, and
    re-render with true parallax. Closest zero-shot substitute for MG3's
    trained Plücker injector. Falls back to warp if the depth model
    cannot be loaded.
  - warp: the legacy 2D crop/pan/zoom warp of the conditioning frame.

Both modes append a matching motion phrase to the prompt. Camera pose state
IS tracked with MG3's exact action->pose math (vendored below from
utils/utils.py + utils/cam_utils.py) and logged per segment, so swapping in a
camera-conditioned checkpoint later only requires feeding `pose_history` into
PRoPE CameraParams instead of the warp.

Run with the LTX venv, e.g.:
    LTX_ROOT=/data1/LTX-2 /data1/ltx-world-model/.venv/bin/python \
        generate_ltx_interactive.py --image demo_images/001/image.png \
        --prompt "..." --num_iterations 6

Scripted (non-interactive) mode:
    --actions "w+u;w+j;d+l;none"   # one "keyboard+mouse" entry per iteration
"""

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

# Reuse the LTX backend: sys.path injection of ltx-core/ltx-pipelines happens
# at import time, and we reuse its pipeline builders and media helpers.
import generate_ltx as G

# MG3 pacing constants (identical to pipeline/inference_interactive_pipeline.py)
FIRST_SEGMENT_FRAMES = G.FIRST_SEGMENT_FRAMES   # 57
SEGMENT_FRAMES = G.SEGMENT_FRAMES               # 41 (40 new + 1 duplicated)

# ---------------------------------------------------------------------------
# MG3 interactive input scheme (verbatim from
# pipeline/inference_interactive_pipeline.py:get_current_action)
# ---------------------------------------------------------------------------
CAM_VALUE = 0.1
CAMERA_VALUE_MAP = {
    "i": [CAM_VALUE, 0], "k": [-CAM_VALUE, 0],
    "j": [0, -CAM_VALUE], "l": [0, CAM_VALUE], "u": [0, 0],
}
KEYBOARD_IDX = {
    "w": [1, 0, 0, 0, 0, 0], "s": [0, 1, 0, 0, 0, 0],
    "a": [0, 0, 1, 0, 0, 0], "d": [0, 0, 0, 1, 0, 0],
    "q": [0, 0, 0, 0, 0, 0],
}


def get_current_action():
    """MG3 two-channel stdin prompt. Returns (keyboard_onehot, mouse_vec)."""
    print()
    print("-" * 30)
    print("PRESS [I, K, J, L, U] FOR CAMERA TRANSFORM\n"
          " (I: up, K: down, J: left, L: right, U: no move)")
    print("PRESS [W, S, A, D, Q] FOR MOVEMENT\n"
          " (W: forward, S: back, A: left, D: right, Q: no move)")
    print("-" * 30)
    while True:
        try:
            idx_mouse = input("Please input the mouse action (e.g. `U`):\n").strip().lower()
            idx_keyboard = input("Please input the keyboard action (e.g. `W`):\n").strip().lower()
            if idx_mouse in CAMERA_VALUE_MAP and idx_keyboard in KEYBOARD_IDX:
                break
        except Exception:
            pass
    return KEYBOARD_IDX[idx_keyboard], CAMERA_VALUE_MAP[idx_mouse], idx_keyboard, idx_mouse


# ---------------------------------------------------------------------------
# MG3 action -> camera-pose math (vendored from utils/utils.py and
# utils/cam_utils.py:get_extrinsics so this script does not pull the Wan
# dependency chain — scipy/trimesh/pandas — into the LTX venv). Units and
# conventions are identical to the Wan backend: positions scaled by 0.01,
# R_init axis remap applied, poses are c2w.
# ---------------------------------------------------------------------------
WSAD_OFFSET = 12.35           # units per frame for single direction
DIAGONAL_OFFSET = 8.73        # 12.35 / sqrt(2)
MOUSE_PITCH_SENSITIVITY = 15.0
MOUSE_YAW_SENSITIVITY = 15.0
MOUSE_THRESHOLD = 0.02

_R_INIT = np.array([[0, 0, 1], [1, 0, 0], [0, -1, 0]], dtype=np.float64)


def compute_next_pose_from_action(current_pose, keyboard_action, mouse_action):
    x, y, z, pitch, yaw = current_pose
    w, s, a, d = keyboard_action[:4]
    mouse_x, mouse_y = mouse_action[:2]

    delta_pitch = MOUSE_PITCH_SENSITIVITY * mouse_x if abs(mouse_x) >= MOUSE_THRESHOLD else 0.0
    delta_yaw = MOUSE_YAW_SENSITIVITY * mouse_y if abs(mouse_y) >= MOUSE_THRESHOLD else 0.0
    new_pitch = pitch + delta_pitch
    new_yaw = yaw + delta_yaw
    while new_yaw > 180:
        new_yaw -= 360
    while new_yaw < -180:
        new_yaw += 360

    local_forward = WSAD_OFFSET if (w > 0.5 and s < 0.5) else (-WSAD_OFFSET if s > 0.5 and w < 0.5 else 0.0)
    local_right = WSAD_OFFSET if (d > 0.5 and a < 0.5) else (-WSAD_OFFSET if a > 0.5 and d < 0.5 else 0.0)
    if abs(local_forward) > 0.1 and abs(local_right) > 0.1:
        local_forward = np.sign(local_forward) * DIAGONAL_OFFSET
        local_right = np.sign(local_right) * DIAGONAL_OFFSET

    avg_yaw = np.deg2rad((yaw + new_yaw) / 2.0)
    new_x = x + np.cos(avg_yaw) * local_forward - np.sin(avg_yaw) * local_right
    new_y = y + np.sin(avg_yaw) * local_forward + np.cos(avg_yaw) * local_right
    return np.array([new_x, new_y, z, new_pitch, new_yaw])


def pose_to_c2w(pose):
    """MG3 get_extrinsics for one [x,y,z,pitch,yaw] pose (roll=0) -> 4x4 c2w."""
    x, y, z, pitch, yaw = pose
    roll = 0.0
    roll, pitch, yaw = np.radians([roll, pitch, yaw])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    c2w = np.eye(4)
    c2w[:3, :3] = Rz @ Ry @ Rx @ _R_INIT
    c2w[:3, 3] = np.array([x, y, z]) * 0.01
    return c2w


# ---------------------------------------------------------------------------
# Zero-shot steering: (keyboard, mouse) -> warp spec + motion phrase.
# Same idea as generate_ltx.py's _ACTION_WARPS but two-channel, matching the
# MG3 semantics (mouse = look, keyboard = move).
# ---------------------------------------------------------------------------
_MOUSE_WARPS = {
    "i": {"dy": -0.10, "phrase": "the camera tilts up"},
    "k": {"dy": 0.10, "phrase": "the camera tilts down"},
    "j": {"dx": -0.12, "phrase": "the camera pans left"},
    "l": {"dx": 0.12, "phrase": "the camera pans right"},
}
_KEYBOARD_WARPS = {
    "w": {"zoom": 1.15, "phrase": "the camera moves forward"},
    "s": {"zoom": 0.87, "phrase": "the camera pulls back"},
    "a": {"dx": -0.12, "phrase": "the camera moves left"},
    "d": {"dx": 0.12, "phrase": "the camera moves right"},
}


def _compose_warp(kb_key, mouse_key):
    """Combine the keyboard and mouse warps into one spec + combined phrase."""
    spec, phrases = {}, []
    for table, key in ((_KEYBOARD_WARPS, kb_key), (_MOUSE_WARPS, mouse_key)):
        entry = table.get(key)
        if entry is None:
            continue
        spec["zoom"] = spec.get("zoom", 1.0) * entry.get("zoom", 1.0)
        spec["dx"] = spec.get("dx", 0.0) + entry.get("dx", 0.0)
        spec["dy"] = spec.get("dy", 0.0) + entry.get("dy", 0.0)
        phrases.append(entry["phrase"])
    return spec, " and ".join(phrases)


def _warp_frame_spec(src_path, spec, dst_path):
    """Apply a composed {zoom, dx, dy} warp — same geometry as
    generate_ltx._warp_frame but taking an explicit spec."""
    from PIL import Image, ImageFilter

    img = Image.open(src_path).convert("RGB")
    w, h = img.size
    zoom = spec.get("zoom", 1.0)
    dx = spec.get("dx", 0.0) * w
    dy = spec.get("dy", 0.0) * h
    if zoom < 1.0:
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


# ---------------------------------------------------------------------------
# Multi-frame re-injection + steering helpers (shared by this CLI and
# server_ltx.py — do not duplicate the logic there).
# ---------------------------------------------------------------------------
def resolve_steering(args, log=print):
    """Resolve (mode, reprojector, cond_strength, keyframe_strength).

    reproject mode builds the Depth-Anything model; if it cannot load (no
    network, OOM, missing transformers) we fall back to the 2D warp so the
    session still runs. Strength defaults: reproject heals steered pixels
    (0.92/0.90), warp is exact on the first frame (1.0/0.95)."""
    mode = args.steer_mode
    reprojector = None
    if mode == "reproject":
        try:
            from steer3d import DepthReprojector
            reprojector = DepthReprojector(model_id=args.depth_model, device=0,
                                           focal_scale=args.reproj_focal,
                                           step=args.reproj_step)
            log(f"[steering] reproject mode: depth model {args.depth_model} loaded")
        except Exception as e:
            log(f"[steering] WARNING: depth reprojection unavailable "
                f"({type(e).__name__}: {e}); falling back to 2D warp")
            mode, reprojector = "warp", None
    cond = args.cond_strength if args.cond_strength is not None else (
        0.92 if mode == "reproject" else 1.0)
    keyf = args.keyframe_strength if args.keyframe_strength is not None else (
        0.90 if mode == "reproject" else 0.95)
    return mode, reprojector, cond, keyf


def save_tail_frames(chunks, tmp_dir, it, k):
    """Save the last k decoded frames (oldest -> newest) as PNGs; returns paths."""
    from PIL import Image

    flat = torch.cat(chunks, dim=0)  # (T, H, W, C) float [0,1]
    paths = []
    for j in range(k):
        frame = flat[-(k - j)].clamp(0, 1).float().cpu().numpy()
        p = Path(tmp_dir) / f"cond_{it:03d}_{j:02d}.png"
        Image.fromarray((frame * 255.0).round().astype(np.uint8)).save(p)
        paths.append(p)
    return paths


def steer_tail_frames(paths, kb_key, mouse_key, mode, reprojector, tmp_dir, it,
                      log=print):
    """Apply the action's camera delta to every tail frame (the virtual camera
    moved once between segments, so all re-injected frames share the delta).
    No-op actions return the paths unchanged. Reprojection is all-or-nothing:
    if any frame fails (e.g. transient OOM sharing a GPU with the MGPU fleet),
    the whole turn falls back to the 2D warp so the K frames stay geometrically
    consistent."""
    if kb_key == "q" and mouse_key == "u":
        return [Path(p) for p in paths]
    if mode == "reproject":
        from PIL import Image
        try:
            steered = []
            for j, p in enumerate(paths):
                out = Path(tmp_dir) / f"cond_{it:03d}_{j:02d}_steered.png"
                reprojector.reproject(Image.open(p).convert("RGB"),
                                      kb_key, mouse_key).save(out)
                steered.append(out)
            return steered
        except Exception as e:
            log(f"[steering] WARNING: reprojection failed ({type(e).__name__}: {e}); "
                "falling back to 2D warp for this turn")
    spec, _ = _compose_warp(kb_key, mouse_key)
    if not spec:
        return [Path(p) for p in paths]
    steered = []
    for j, p in enumerate(paths):
        out = Path(tmp_dir) / f"cond_{it:03d}_{j:02d}_steered.png"
        _warp_frame_spec(p, spec, out)
        steered.append(out)
    return steered


def build_image_conditionings(paths, cond_strength, keyframe_strength):
    """Oldest->newest tail frames -> LTX conditionings at frame_idx 0..K-1.

    frame_idx 0 is an exact latent replacement (strength < 1 lets the model
    heal the steered/inpainted pixels); frame_idx > 0 are clean keyframe
    tokens — MG3's memory mechanism via stock LTX conditioning."""
    from ltx_pipelines.utils.args import ImageConditioningInput
    return [ImageConditioningInput(
        path=str(p), frame_idx=j,
        strength=cond_strength if j == 0 else keyframe_strength)
        for j, p in enumerate(paths)]


def drop_frames(chunks, k):
    """Drop the first k frames of a decoded segment (the re-injected
    conditioning frames), returning a single (T-k, H, W, C) chunk."""
    return [torch.cat(chunks, dim=0)[k:].contiguous()]


def _resolve_action(args, it):
    """Action for the segment after segment it.

    Returns (kb_key, mouse_key) or None to quit. Interactive mode uses the MG3
    two-prompt stdin UX; --actions uses scripted "keyboard+mouse" entries.
    """
    if args.actions:
        seq = [s.strip().lower() for s in args.actions.split(";")]
        if it >= len(seq) or seq[it] in ("", "none", "q+u"):
            return ("q", "u")
        parts = seq[it].split("+")
        kb = parts[0] if parts[0] in _KEYBOARD_WARPS or parts[0] == "q" else "q"
        ms = parts[1] if len(parts) > 1 and parts[1] in _MOUSE_WARPS else "u"
        return (kb, ms)
    _, _, kb_key, mouse_key = get_current_action()
    return (kb_key, mouse_key)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive LTX-2.3 game-video generation "
                    "(Matrix-Game-3 interactive pipeline ported to the LTX backend).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./output_ltx_interactive")
    parser.add_argument("--save_name", type=str, default="ltx_interactive")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_iterations", type=int, default=6,
                        help="Total frames = 57 + (num_iterations - 1) * "
                             "(segment_frames - reinject_frames)")
    parser.add_argument("--segment_frames", type=int, default=SEGMENT_FRAMES,
                        help="Frames per continuation segment (must be ≡1 mod 8)")
    parser.add_argument("--reinject_frames", type=int, default=8,
                        help="Condition each continuation segment on the steered "
                             "last K frames of the previous one (K=1 = legacy "
                             "single-frame re-injection)")
    parser.add_argument("--steer_mode", choices=["reproject", "warp"],
                        default="reproject",
                        help="reproject: depth-based camera reprojection with true "
                             "parallax (steer3d.py; falls back to warp if the depth "
                             "model cannot load). warp: legacy 2D crop/pan/zoom.")
    parser.add_argument("--reproj_step", type=float, default=0.13,
                        help="Per-segment translation as a fraction of the central "
                             "subject's depth (0.13 ~ the 2D warp's zoom 1.15)")
    parser.add_argument("--reproj_focal", type=float, default=1.2,
                        help="Virtual camera focal length, in units of frame width")
    parser.add_argument("--depth_model", type=str,
                        default="depth-anything/Depth-Anything-V2-Small-hf")
    parser.add_argument("--cond_strength", type=float, default=None,
                        help="Conditioning strength for the newest tail frame "
                             "(default: 0.92 reproject / 1.0 warp)")
    parser.add_argument("--keyframe_strength", type=float, default=None,
                        help="Conditioning strength for the older tail keyframes "
                             "(default: 0.90 reproject / 0.95 warp)")
    parser.add_argument("--size", type=str, default="704*1280")
    parser.add_argument("--frame_rate", type=float, default=24.0)
    parser.add_argument("--ltx_checkpoint", type=str, default=str(G.DEFAULT_DISTILLED_CKPT))
    parser.add_argument("--spatial_upsampler", type=str, default=str(G.DEFAULT_UPSAMPLER))
    parser.add_argument("--gemma_root", type=str, default=G.DEFAULT_GEMMA)
    parser.add_argument("--actions", type=str, default="",
                        help='Scripted mode: one "keyboard+mouse" entry per iteration, '
                             'e.g. "w+u;w+j;d+l;none". Empty = interactive stdin prompts.')
    parser.add_argument("--enhance_prompt", action="store_true",
                        help="Let Gemma enhance the prompt (uses the conditioning image)")
    parser.add_argument("--no_audio", action="store_true")
    parser.add_argument("--crf", type=int, default=19)
    parser.add_argument("--use_base_model", action="store_true",
                        help="Use the one-stage base (dev) checkpoint with full CFG "
                             "denoise instead of the distilled model (MG3-style "
                             "--use_base_model; slower per segment, higher fidelity)")
    parser.add_argument("--mgpu", action="store_true",
                        help="Run every segment across all visible GPUs via the "
                             "ltx-pipelines MGPU fleet (sequence parallelism + "
                             "distributed VAE/Gemma; distilled mode only, requires "
                             "ltx-kernels). Cuts per-turn latency; each segment is "
                             "decoded back from the fleet's mp4, costing one extra "
                             "x264 round-trip vs single-GPU")
    parser.add_argument("--num_inference_steps", type=int, default=40,
                        help="Denoise steps per segment (base model only; distilled is fixed)")
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--negative_prompt", type=str, default="")
    args = parser.parse_args()
    args.one_stage = args.use_base_model
    if args.use_base_model and args.ltx_checkpoint == str(G.DEFAULT_DISTILLED_CKPT):
        args.ltx_checkpoint = str(G.DEFAULT_DEV_CKPT)
    return args


@torch.inference_mode()
def generate(args):
    from ltx_pipelines.utils.media_io import encode_video

    G._check_assets(args)
    height, width = (int(v) for v in args.size.split("*"))
    div = 32 if args.one_stage else 64  # two-stage upsampler needs /64
    if height % div or width % div:
        raise ValueError(f"Resolution {height}x{width} is not divisible by {div}.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="ltx_interactive_cond_", dir=output_dir))

    if (args.segment_frames - 1) % 8:
        raise ValueError(f"--segment_frames must be ≡1 mod 8 (got {args.segment_frames})")
    if not 1 <= args.reinject_frames < args.segment_frames:
        raise ValueError(f"--reinject_frames must be in [1, segment_frames) "
                         f"(got {args.reinject_frames})")
    k = args.reinject_frames
    total_frames = FIRST_SEGMENT_FRAMES + (args.num_iterations - 1) * (args.segment_frames - k)
    input_mode = "scripted actions" if args.actions else "interactive stdin"
    print(f"[LTX interactive] mode={input_mode}, size={height}x{width}, "
          f"{args.num_iterations} iterations -> up to {total_frames} frames @ {args.frame_rate}fps")

    if args.mgpu:
        if args.one_stage:
            raise ValueError("--mgpu is only supported in distilled (two-stage) mode; "
                             "no one-stage MGPU runner ships upstream.")
        n_gpus = torch.cuda.device_count()
        if n_gpus < 2:
            raise RuntimeError(f"--mgpu needs >=2 visible GPUs, found {n_gpus}.")
        print(f"[LTX interactive] MGPU fleet on {n_gpus} GPUs (sequence-parallel "
              "distilled + distributed VAE/Gemma)")
        run_segment = G.build_mgpu_run_segment(args, tmp_dir)
    else:
        run_segment = G._build_pipeline(args)
    steer_mode, reprojector, cond_strength, keyframe_strength = resolve_steering(args)
    print(f"[LTX interactive] steer_mode={steer_mode}, reinject K={k} "
          f"(cond={cond_strength}, keyframe={keyframe_strength}); "
          "pose state is tracked for future camera-conditioned checkpoints")

    all_chunks, all_audios = [], []
    cond_images = build_image_conditionings([Path(args.image)], 1.0, 1.0)
    prompt_it = args.prompt
    pose = np.zeros(5, dtype=np.float32)  # [x, y, z, pitch, yaw], MG3 convention
    pose_history = [pose_to_c2w(pose)]

    for it in range(args.num_iterations):
        num_frames = FIRST_SEGMENT_FRAMES if it == 0 else args.segment_frames

        t0 = time.time()
        video_iter, audio = run_segment(
            prompt_it, args.seed + it, height, width, num_frames, cond_images)
        seg_chunks = [c.cpu() for c in video_iter]
        print(f"[LTX interactive] segment {it}: {num_frames} frames in {time.time() - t0:.1f}s")

        if it > 0:
            seg_chunks = drop_frames(seg_chunks, k)  # drop re-injected conditioning frames

        seg_path = output_dir / f"{args.save_name}_seg{it:03d}.mp4"
        encode_video(video=iter(seg_chunks), fps=int(args.frame_rate),
                     audio=None if args.no_audio else audio,
                     output_path=str(seg_path),
                     video_chunks_number=len(seg_chunks), crf=args.crf)

        all_chunks.extend(seg_chunks)
        all_audios.append(audio)

        if it == args.num_iterations - 1:
            break

        # --- action step (the interactive part) ---
        kb_key, mouse_key = _resolve_action(args, it)
        kb_vec = KEYBOARD_IDX[kb_key]
        mouse_vec = CAMERA_VALUE_MAP[mouse_key]
        pose = compute_next_pose_from_action(pose, kb_vec, mouse_vec)
        pose_history.append(pose_to_c2w(pose))
        print(f"[LTX interactive] action kb={kb_key} mouse={mouse_key} -> "
              f"pose xyz=({pose[0]:.2f},{pose[1]:.2f},{pose[2]:.2f}) "
              f"pitch={pose[3]:.1f} yaw={pose[4]:.1f}")

        tail = save_tail_frames(seg_chunks, tmp_dir, it, k)
        steered = steer_tail_frames(tail, kb_key, mouse_key, steer_mode,
                                    reprojector, tmp_dir, it)
        cond_images = build_image_conditionings(steered, cond_strength,
                                                keyframe_strength)
        _, phrase = _compose_warp(kb_key, mouse_key)
        if kb_key != "q" or mouse_key != "u":
            print(f"[LTX interactive] steering: {steer_mode} on last {k} frames")
        if phrase:
            print(f"[LTX interactive] steering: '{phrase}'")
        prompt_it = f"{args.prompt}, {phrase}" if phrase else args.prompt

    if args.mgpu:
        run_segment.shutdown()  # release the fleet before the final encode
    np.save(output_dir / f"{args.save_name}_pose_history.npy",
            np.stack(pose_history).astype(np.float32))

    final_audio = None
    if not args.no_audio and all_audios:
        final_audio = G._concat_audios(all_audios, args.frame_rate, trim_frames=k)

    output_path = output_dir / f"{args.save_name}.mp4"
    encode_video(video=iter(all_chunks), fps=int(args.frame_rate),
                 audio=final_audio, output_path=str(output_path),
                 video_chunks_number=len(all_chunks), crf=args.crf)
    n_frames = sum(c.shape[0] for c in all_chunks)
    print(f"[LTX interactive] saved {n_frames} frames -> {output_path}")


def main():
    generate(parse_args())


if __name__ == "__main__":
    main()
