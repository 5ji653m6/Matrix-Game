#!/usr/bin/env python
"""LTX interactive console server: WebSocket-driven segmented generation.

Wraps the generate_ltx_interactive.py loop in a FastAPI server:
  - Models load ONCE at startup (background thread; /api/status reports readiness)
  - One active session: client sends start/image+prompt, then actions over WS
    whenever it likes — generation NEVER waits for input (continuous mode):
    the latest queued action applies at each segment boundary, and with no
    input the video keeps going with the camera unchanged (neutral q/u).
    Each generated segment is written as an mp4 chunk and sent back as a URL
    for play-while-generating playback
  - Steering is the stock-weights zero-shot channel (depth-reprojection or 2D
    warp of the last K re-injected frames + motion phrase); pose state tracked
    with MG3's math for future PRoPE checkpoints

Run:
    CUDA_VISIBLE_DEVICES=0 LTX_ROOT=/data1/LTX-2 \
      /data1/ltx-world-model/.venv/bin/python server_ltx.py --port 8600
"""

import argparse
import asyncio
import queue
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import generate_ltx as G
import generate_ltx_interactive as GI

ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"

PRESET_PROMPTS = {
    "cyberpunk2077": "Photorealistic cyberpunk garage workshop at night: a low wedge-shaped concept sports car in yellow, white and blue on an oil-stained concrete floor, tires and tool chests scattered around. Moody neon-noir atmosphere, wet reflective floor, cinematic depth of field.",
    "rdr2": "Two weathered cowboys in heavy winter ponchos lead their horses up a snowy mountain trail, snow-capped peaks fading into white haze. Soft overcast winter light, photorealistic western frontier, cinematic and quiet.",
    "elden_ring": "An armored knight on horseback on a grassy cliff overlooking misty ruins, a colossal glowing golden tree dominating the sky, luminous branches raining golden sparks. Epic dark fantasy, volumetric golden-green light.",
    "blackmyth_wukong": "A monkey warrior in fur-trimmed robes faces a massive smoldering stone guardian inside a ruined mountain temple, shafts of pale light cutting through darkness, embers drifting. Mythic Chinese dark fantasy, dramatic chiaroscuro.",
    "gow_ragnarok": "A muscular bald warrior with red tattoos seen from behind at a mossy cliff edge, looking across a misty Norse canyon toward a wooden cage-lift on chains. Cold overcast light, photorealistic mythic wilderness.",
    "horizon_fw": "A vast overgrown ruin: towering carved stone pillars wrapped in vines, brilliant shafts of sunlight piercing the broken ceiling. Lush post-apocalyptic jungle temple, vibrant greens against warm stone, photorealistic.",
    "forza5": "A silver supercar leads a night street race down a rain-slick desert highway, headlights carving through the storm, neon reflections smearing across wet asphalt, saguaro cacti against lightning. Photorealistic racing footage.",
    "witcher3": "A white-haired swordsman on horseback on a rocky hilltop trail overlooking an alpine valley, lakeside village with red rooftops below, snow-capped mountains beyond. Painterly medieval fantasy, crisp daylight.",
    "ghost_tsushima": "A lone samurai in a straw raincoat beneath a brilliant golden ginkgo tree on a rocky bluff, overlooking a mist-filled valley. Feudal Japan, golden-hour light, cinematic beauty.",
    "starfield": "An astronaut in a white and red spacesuit on the rocky ridge of an alien world, a colossal ringed planet in a pale rose sky above jagged dark mountains. Hard sci-fi realism, serene and vast.",
    "death_stranding": "A lone porter in a blue-grey expedition suit with a towering stack of orange cargo cases climbs a windswept mossy hillside toward a half-buried ring-shaped ruin. Melancholic photorealistic wilderness.",
    "gta5": "An orange muscle car with glowing blue underglow speeds through a rain-soaked downtown intersection at night, neon signs blurring into bokeh, reflections streaking across wet pavement. Photorealistic open-world footage.",
    "001": "A colorful, animated cityscape with a gas station and various buildings.",
}


# ---------------------------------------------------------------------------
# Session worker (blocking thread; talks to asyncio via queues)
# ---------------------------------------------------------------------------
class Session:
    def __init__(self, server, params):
        self.server = server
        self.params = params
        self.actions = queue.Queue()
        self.stop_flag = threading.Event()
        self.sid = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        self.thread = threading.Thread(target=self._run, daemon=True)
        # resume state: what a (re)connecting client needs to catch up
        self.segments = []       # segment event dicts, in order
        self.current = -1        # segment index currently generating
        self.pose = [0.0] * 5

    def emit(self, msg):
        self.server.broadcast(msg)

    def start(self):
        self.thread.start()

    def snapshot(self):
        """State a (re)connecting client needs to resume watching live."""
        return {"type": "session_resume", "sid": self.sid,
                "prompt": self.params.get("prompt", ""),
                "segments": list(self.segments), "current": self.current,
                "pose": [round(float(v), 2) for v in self.pose]}

    @torch.inference_mode()  # whole thread: run_segment returns a LAZY iterator,
    def _run(self):          # so everything downstream must stay in inference mode
        from ltx_pipelines.utils.media_io import encode_video

        args = self.server.pipeline_args
        sargs = self.server.args
        steer_mode, reprojector, cond_strength, keyframe_strength = self.server.steering
        k = sargs.reinject_frames
        out_dir = self.server.output_dir / self.sid
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir = out_dir / "cond"
        tmp_dir.mkdir()

        try:
            height, width = (int(v) for v in self.params.get(
                "size", self.server.args.size).split("*"))
            max_it = min(int(self.params.get("num_iterations", 12)),
                         self.server.max_iterations)
            seed = int(self.params.get("seed", args.seed))
            prompt0 = self.params["prompt"]

            pose = np.zeros(5, dtype=np.float32)
            pose_history = [GI.pose_to_c2w(pose)]
            all_chunks, all_audios = [], []
            cond_images = GI.build_image_conditionings(
                [self.params["image_path"]], 1.0, 1.0)
            prompt_it = prompt0

            for it in range(max_it):
                self.current = it
                num_frames = GI.FIRST_SEGMENT_FRAMES if it == 0 else sargs.segment_frames
                self.emit({"type": "status", "state": "generating", "segment": it})
                t0 = time.time()
                video_iter, audio = self.server.run_segment(
                    prompt_it, seed + it, height, width, num_frames, cond_images)
                seg_chunks = [c.cpu() for c in video_iter]
                elapsed = time.time() - t0

                if it > 0:
                    seg_chunks = GI.drop_frames(seg_chunks, k)  # re-injected frames

                seg_path = out_dir / f"seg_{it:03d}.mp4"
                encode_video(video=iter(seg_chunks), fps=int(args.frame_rate),
                             audio=audio, output_path=str(seg_path),
                             video_chunks_number=len(seg_chunks), crf=args.crf)
                all_chunks.extend(seg_chunks)
                all_audios.append(audio)
                seg_event = {
                    "type": "segment", "index": it,
                    "url": f"/chunks/{self.sid}/seg_{it:03d}.mp4",
                    "elapsed": round(elapsed, 1),
                    "frames": int(sum(c.shape[0] for c in seg_chunks)),
                    "pose": [round(float(v), 2) for v in pose],
                }
                self.segments.append(seg_event)  # resume backlog
                self.pose = seg_event["pose"]
                self.emit(seg_event)

                if it == max_it - 1 or self.stop_flag.is_set():
                    break

                # --- continuous generation: input is OPTIONAL ---
                # Never wait for the operator: drain whatever was typed while
                # this segment was generating (latest action wins); nothing
                # queued -> neutral q/u, i.e. keep going, camera unchanged.
                act = None
                while True:
                    try:
                        act = self.actions.get_nowait()
                    except queue.Empty:
                        break
                if self.stop_flag.is_set():
                    break
                if act is None:
                    kb_key, mouse_key = "q", "u"
                else:
                    kb_key, mouse_key = act["kb"], act["mouse"]
                pose = GI.compute_next_pose_from_action(
                    pose, GI.KEYBOARD_IDX[kb_key], GI.CAMERA_VALUE_MAP[mouse_key])
                pose_history.append(GI.pose_to_c2w(pose))

                tail = GI.save_tail_frames(seg_chunks, tmp_dir, it, k)
                steered = GI.steer_tail_frames(tail, kb_key, mouse_key,
                                               steer_mode, reprojector, tmp_dir, it)
                cond_images = GI.build_image_conditionings(
                    steered, cond_strength, keyframe_strength)
                _, phrase = GI._compose_warp(kb_key, mouse_key)
                prompt_it = f"{prompt0}, {phrase}" if phrase else prompt0
                if act is not None:
                    self.emit({"type": "action_ack", "kb": kb_key,
                               "mouse": mouse_key, "phrase": phrase,
                               "pose": [round(float(v), 2) for v in pose]})

            np.save(out_dir / "pose_history.npy",
                    np.stack(pose_history).astype(np.float32))
            final_audio = G._concat_audios(all_audios, args.frame_rate,
                                           trim_frames=k)
            final_path = out_dir / "final.mp4"
            encode_video(video=iter(all_chunks), fps=int(args.frame_rate),
                         audio=final_audio, output_path=str(final_path),
                         video_chunks_number=len(all_chunks), crf=args.crf)
            self.emit({"type": "final",
                       "url": f"/chunks/{self.sid}/final.mp4",
                       "frames": int(sum(c.shape[0] for c in all_chunks))})
            self.emit({"type": "session_end",
                       "reason": "stopped" if self.stop_flag.is_set()
                                 else "completed"})
        except Exception as e:
            self.emit({"type": "error", "message": f"{type(e).__name__}: {e}"})
            self.emit({"type": "session_end", "reason": "error"})
        finally:
            self.emit({"type": "status", "state": "idle"})


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
class ConsoleServer:
    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "uploads").mkdir(exist_ok=True)
        self.max_iterations = args.max_iterations
        self.run_segment = None
        self.steering = None  # (mode, reprojector, cond_strength, keyframe_strength)
        self.ready = False
        self.load_error = None  # set if the loader thread dies (else invisible)
        self.session = None
        self.lock = threading.Lock()
        self.loop = None      # the uvicorn asyncio loop, captured at startup
        self.subs = set()     # asyncio.Queue of every connected WS client
        self.pipeline_args = SimpleNamespace(
            one_stage=args.use_base_model,
            ltx_checkpoint=(str(G.DEFAULT_DEV_CKPT) if args.use_base_model
                            else str(G.DEFAULT_DISTILLED_CKPT)),
            ltx_dev_checkpoint=str(G.DEFAULT_DEV_CKPT),
            distilled_lora=str(G.DEFAULT_DISTILLED_LORA),
            spatial_upsampler=str(G.DEFAULT_UPSAMPLER),
            gemma_root=str(G.DEFAULT_GEMMA),
            negative_prompt="", guidance_scale=3.0,
            num_inference_steps=args.num_inference_steps,
            frame_rate=24.0, enhance_prompt=False,
            seed=args.seed, crf=19, no_audio=False,
            audio_track_path=None,
        )

    def broadcast(self, msg):
        """Fan an event out to every connected client (0 or more). Safe from
        the session thread: each put is marshalled onto the uvicorn loop."""
        if self.loop is None:
            return
        for q in list(self.subs):
            self.loop.call_soon_threadsafe(q.put_nowait, msg)

    def load_pipeline(self):
        try:
            if self.args.mgpu:
                self.run_segment = G.build_mgpu_run_segment(
                    self.pipeline_args, self.output_dir / "mgpu_fleet")
            else:
                self.run_segment = G._build_pipeline(self.pipeline_args)
            self.steering = GI.resolve_steering(self.args)
            self.ready = True
        except Exception as e:
            self.load_error = f"{type(e).__name__}: {e}"
            raise

    @property
    def busy(self):
        return self.session is not None and self.session.thread.is_alive()


server: ConsoleServer = None  # set in main()
app = FastAPI(title="LTX Interactive Console")


@app.on_event("startup")
def _startup():
    server.loop = asyncio.get_running_loop()
    threading.Thread(target=server.load_pipeline, daemon=True).start()


@app.on_event("shutdown")
def _shutdown():
    if server is not None and server.args.mgpu and server.run_segment is not None:
        server.run_segment.shutdown()  # release the worker fleet


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "console.html")


@app.get("/api/status")
def status():
    return {"ready": server.ready, "busy": server.busy,
            "load_error": server.load_error,
            "model": Path(server.pipeline_args.ltx_checkpoint).name}


@app.get("/api/presets")
def presets():
    out = []
    for base, pattern, mount in (
        ("demo_images", "*/image.png", "/presets/demo"),
        ("demo_images_aaa", "*/image.jpg", "/presets/aaa"),
    ):
        for p in sorted((ROOT / base).glob(pattern)):
            key = p.parent.name
            out.append({"key": key, "url": f"{mount}/{key}/{p.name}",
                        "prompt": PRESET_PROMPTS.get(key, "")})
    return out


@app.post("/api/upload")
async def upload(file: UploadFile):
    ext = Path(file.filename).suffix or ".png"
    dest = server.output_dir / "uploads" / f"{uuid.uuid4().hex[:12]}{ext}"
    dest.write_bytes(await file.read())
    return {"url": f"/chunks/uploads/{dest.name}"}


def _resolve_image(url: str) -> str:
    """Map a client-supplied image URL back to a local path (allowed roots only)."""
    for prefix, base in (("/presets/demo/", ROOT / "demo_images"),
                         ("/presets/aaa/", ROOT / "demo_images_aaa"),
                         ("/chunks/uploads/", server.output_dir / "uploads")):
        if url.startswith(prefix):
            rel = Path(url[len(prefix):])
            if rel.is_absolute() or ".." in rel.parts:
                break
            p = base / rel
            if p.exists():
                return str(p)
    raise ValueError(f"Unknown image: {url}")


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    event_q = asyncio.Queue()
    server.subs.add(event_q)

    async def send(msg):
        try:
            await ws.send_json(msg)
        except Exception:
            pass

    await send({"type": "hello", "ready": server.ready, "busy": server.busy,
                "model": Path(server.pipeline_args.ltx_checkpoint).name})
    s = server.session
    if s is not None and s.thread.is_alive():
        await send(s.snapshot())  # let a (re)connecting client catch up live

    async def receiver():
        async for raw in ws.iter_json():
            t = raw.get("type")
            if t == "start":
                if not server.ready:
                    await send({"type": "error", "message": "model still loading"})
                elif server.busy:
                    await send(server.session.snapshot())
                else:
                    try:
                        raw["image_path"] = _resolve_image(raw["image"])
                    except ValueError as e:
                        await send({"type": "error", "message": str(e)})
                        continue
                    with server.lock:
                        server.session = Session(server, raw)
                        server.session.start()
            elif t == "action":
                s = server.session
                if s and raw.get("kb") in GI.KEYBOARD_IDX \
                        and raw.get("mouse") in GI.CAMERA_VALUE_MAP:
                    s.actions.put({"kb": raw["kb"], "mouse": raw["mouse"]})
            elif t == "stop":
                if server.session:
                    server.session.stop_flag.set()

    async def sender():
        while True:
            await send(await event_q.get())

    recv_task = asyncio.create_task(receiver())
    send_task = asyncio.create_task(sender())
    try:
        await recv_task
    except WebSocketDisconnect:
        pass
    finally:
        send_task.cancel()
        server.subs.discard(event_q)
        # NOTE: the session is deliberately NOT stopped on disconnect — a
        # dropped link must not kill a run; the next client resumes live via
        # snapshot(). Only an explicit "stop" message (or the iteration cap)
        # ends a session.


def main():
    global server
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8600)
    parser.add_argument("--size", type=str, default="704*1280")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_iterations", type=int, default=24)
    parser.add_argument("--use_base_model", action="store_true")
    parser.add_argument("--mgpu", action="store_true",
                        help="Run every segment across all visible GPUs via the "
                             "ltx-pipelines MGPU fleet (sequence-parallel distilled; "
                             "requires ltx-kernels, >=2 GPUs, incompatible with "
                             "--use_base_model). Fleet stays resident between turns.")
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--output_dir", type=str, default="./output_ltx_server")
    parser.add_argument("--segment_frames", type=int, default=GI.SEGMENT_FRAMES,
                        help="Frames per continuation segment (must be ≡1 mod 8)")
    parser.add_argument("--reinject_frames", type=int, default=8,
                        help="Condition each continuation segment on the steered "
                             "last K frames of the previous one")
    parser.add_argument("--steer_mode", choices=["reproject", "warp"],
                        default="reproject",
                        help="reproject: depth-based camera reprojection (steer3d.py, "
                             "warp fallback); warp: legacy 2D crop/pan/zoom")
    parser.add_argument("--reproj_step", type=float, default=0.13)
    parser.add_argument("--reproj_focal", type=float, default=1.2)
    parser.add_argument("--depth_model", type=str,
                        default="depth-anything/Depth-Anything-V2-Small-hf")
    parser.add_argument("--cond_strength", type=float, default=None)
    parser.add_argument("--keyframe_strength", type=float, default=None)
    args = parser.parse_args()
    if (args.segment_frames - 1) % 8:
        parser.error(f"--segment_frames must be ≡1 mod 8 (got {args.segment_frames})")
    if not 1 <= args.reinject_frames < args.segment_frames:
        parser.error(f"--reinject_frames must be in [1, segment_frames) "
                     f"(got {args.reinject_frames})")
    if args.mgpu and args.use_base_model:
        parser.error("--mgpu is distilled-mode only (no one-stage MGPU runner "
                     "ships upstream); drop --use_base_model")
    if args.mgpu and torch.cuda.device_count() < 2:
        parser.error(f"--mgpu needs >=2 visible GPUs, found "
                     f"{torch.cuda.device_count()}")

    server = ConsoleServer(args)
    app.mount("/chunks", StaticFiles(directory=server.output_dir), name="chunks")
    app.mount("/presets/demo", StaticFiles(directory=ROOT / "demo_images"), name="pd")
    app.mount("/presets/aaa", StaticFiles(directory=ROOT / "demo_images_aaa"), name="pa")
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
