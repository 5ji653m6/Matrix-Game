#!/usr/bin/env python
"""Depth-based camera reprojection steering for stock LTX checkpoints.

Zero-shot substitute for MG3's trained Plücker camera injector. Instead of a
2D crop/pan/zoom warp of the conditioning frame, estimate monocular depth,
unproject the frame to a point cloud, move a virtual pinhole camera by the
action's rotation/translation, and re-render with a z-buffer splat. The model
then sees true parallax (near objects move more than far ones), which is a
much closer approximation of real camera motion than a 2D warp.

Pure inference-time geometry — no video-model training, no new heavyweight
deps (Depth-Anything-V2-Small via the transformers pipeline, torch-only hole
filling; cv2 is intentionally NOT required).

Camera convention: OpenCV, x right, y down, z forward. Action -> camera delta
matches the per-segment magnitudes used by the MG3 pose math vendored in
generate_ltx_interactive.py (15.0 * 0.1 = 1.5 degrees look per segment).
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
MOUSE_ANGLE_DEG = 1.5  # per-segment look delta, matches MG3 (15.0 * CAM_VALUE=0.1)


def action_to_rt(kb_key, mouse_key, step):
    """Map a (keyboard, mouse) action to camera rotation (3x3) and translation
    (3,) in the CURRENT camera frame (x right, y down, z forward).

    Returns (None, None) for a no-op action (q+u).
    Sign checks (verified visually):
      w: t=+z  -> points shrink in z -> zoom-in (approach)            ✓
      j: yaw=- -> world content shifts RIGHT in frame (look left)     ✓
      i: pitch=+ -> world content shifts DOWN in frame (look up)      ✓
    """
    yaw = {"j": -1.0, "l": +1.0}.get(mouse_key, 0.0) * math.radians(MOUSE_ANGLE_DEG)
    pitch = {"i": +1.0, "k": -1.0}.get(mouse_key, 0.0) * math.radians(MOUSE_ANGLE_DEG)
    t = np.zeros(3, dtype=np.float64)
    if kb_key == "w":
        t[2] += step
    elif kb_key == "s":
        t[2] -= step
    elif kb_key == "a":
        t[0] -= step
    elif kb_key == "d":
        t[0] += step
    if abs(yaw) < 1e-12 and abs(pitch) < 1e-12 and not np.any(t):
        return None, None
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    # Ry(+yaw): z-axis tilts toward +x (right). Rx(+pitch): z-axis tilts toward -y (up).
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    return ry @ rx, t


def _close_cracks(img, known, passes=2):
    """Fill 1-px splat cracks with the average of their known 3x3 neighbors."""
    dev = img.device
    k3 = torch.ones(3, 1, 3, 3, device=dev)
    k1 = torch.ones(1, 1, 3, 3, device=dev)
    for _ in range(passes):
        known_f = known.float()
        num = F.conv2d((img * known_f).unsqueeze(0), k3, padding=1, groups=3).squeeze(0)
        den = F.conv2d(known_f.unsqueeze(0), k1, padding=1).squeeze(0)
        reach = den > 0.5
        fill = num / den.clamp_min(1.0)
        target = reach & ~known
        img = torch.where(target.expand_as(img), fill, img)
        known = known | reach
    return img, known


def _hole_fill(img, known, src_weight=None, max_iters=400):
    """Fill remaining holes (out-of-frame borders, disocclusions) by normalized
    diffusion at reduced resolution, then composite the known pixels back.

    src_weight (1, H, W) biases fill sources: disocclusion gaps reveal
    BACKGROUND, so passing the depth map as weight makes fills grow from the
    far side instead of smearing foreground colors across the gap.
    Produces smooth inpaint-like fills; good enough because the conditioning
    strength is < 1 and the video model re-denoises these regions."""
    h, w = img.shape[-2:]
    dev = img.device
    m = known.float()
    if src_weight is None:
        src_weight = torch.ones_like(m)
    wm = m * src_weight
    factor = 1
    while min(h // (factor * 2), w // (factor * 2)) >= 96:
        factor *= 2
    if factor > 1:
        ms = F.avg_pool2d(m.unsqueeze(0), factor).squeeze(0)
        ws = F.avg_pool2d(wm.unsqueeze(0), factor).squeeze(0)
        xs = F.avg_pool2d((img * wm).unsqueeze(0), factor).squeeze(0)
        xs = torch.where(ms > 1e-6, xs / ws.clamp_min(1e-6), torch.zeros_like(xs))
    else:
        ms, ws, xs = m.clone(), wm.clone(), img.clone()
    kn = torch.ones(3, 1, 3, 3, device=dev)
    k1 = torch.ones(1, 1, 3, 3, device=dev)
    known_c = ms > 1e-6
    for _ in range(max_iters):
        if bool(known_c.all()):
            break
        # propagate color from known cells, weighted toward far (background) sources
        num = F.conv2d((xs * ws).unsqueeze(0), kn, padding=1, groups=3).squeeze(0)
        den = F.conv2d(ws.unsqueeze(0), k1, padding=1).squeeze(0)
        has = den > 1e-4
        fill = num / den.clamp_min(1e-4)
        upd = ~known_c & has
        xs = torch.where(upd.expand_as(xs), fill, xs)
        ws = torch.where(upd, torch.ones_like(ws), ws)
        known_c = known_c | has
    up = F.interpolate(xs.unsqueeze(0), size=(h, w), mode="bilinear",
                       align_corners=False).squeeze(0)
    return torch.where(known.expand_as(img), img, up)


class DepthReprojector:
    """Monocular depth + point-cloud reprojection of conditioning frames."""

    def __init__(self, model_id=DEFAULT_DEPTH_MODEL, device=0,
                 focal_scale=1.2, step=0.13, lateral_scale=0.5):
        """step: per-segment translation as a fraction of the central
        subject's depth (0.13 ~ the legacy 2D warp's zoom 1.15).
        lateral_scale: a/d strafes are scaled down by this — a lateral
        subject-anchored step moves the subject by step*fx PIXELS
        regardless of depth, so on close subjects the disocclusion band
        at the subject boundary is (subject - background) shift wide and
        unfillable; halving the strafe keeps it paintable (and matches
        games, where strafing is slower than walking)."""
        from transformers import pipeline as hf_pipeline

        self.pipe = hf_pipeline("depth-estimation", model=model_id, device=device)
        self.device = self.pipe.device
        self.focal_scale = float(focal_scale)
        self.step = float(step)
        self.lateral_scale = float(lateral_scale)

    def estimate_z(self, img):
        """PIL image -> (H, W) depth (larger = farther), median-normalized to 1.
        Depth-Anything-V2 emits affine-invariant disparity (larger = closer,
        can go slightly negative) — verified on forza5: road 6.2, sky ~0."""
        out = self.pipe(img)
        d = out["predicted_depth"]
        if not torch.is_tensor(d):
            d = torch.from_numpy(np.asarray(d))
        d = d.squeeze().float().to(self.device)
        d = d - d.min()
        dn = d / d.max().clamp_min(1e-6)
        z = 1.0 / (0.1 + 0.9 * dn)  # far(sky) ~10, near ~1
        return z / z.median()

    def reproject(self, img, kb_key, mouse_key):
        """PIL RGB -> PIL RGB: the same scene viewed after the action's camera
        motion. No-op actions return a copy unchanged."""
        from PIL import Image

        rot, t = action_to_rt(kb_key, mouse_key, 1.0)  # unit translation, scaled below
        if rot is None:
            return img.copy()
        w, h = img.size
        dev = self.device
        z = self.estimate_z(img)
        # Scale the translation to the central subject's depth: a fixed
        # fraction-of-scene step moves a close-up subject by half the frame
        # (giant smears). Anchoring to the subject reproduces the old 2D warp
        # magnitudes (zoom ~1.15, shift ~0.12 frame) on whatever is centered,
        # while far background still gets correct (smaller) parallax.
        ch, cw = h // 4, w // 4
        z_subj = float(z[ch:3 * ch, cw:3 * cw].median())
        t = t * (self.step * z_subj)
        t[0] *= self.lateral_scale  # strafes: see __init__ note
        fx = fy = self.focal_scale * w
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        ys, xs = torch.meshgrid(
            torch.arange(h, device=dev, dtype=torch.float32),
            torch.arange(w, device=dev, dtype=torch.float32), indexing="ij")
        # unproject in the old camera frame
        px = (xs - cx) * z / fx
        py = (ys - cy) * z / fy
        p = torch.stack([px, py, z], dim=0).reshape(3, -1)
        # camera moves by (rot, t) expressed in the old camera frame:
        # fixed world points become p2 = rot^T (p - t) in the new frame
        rot_t = torch.from_numpy(np.ascontiguousarray(rot.T)).float().to(dev)
        tt = torch.from_numpy(t).float().to(dev)
        p2 = rot_t @ (p - tt[:, None])
        z2 = p2[2]
        u = fx * p2[0] / z2.clamp_min(1e-3) + cx
        v = fy * p2[1] / z2.clamp_min(1e-3) + cy
        valid = (z2 > 0.05 * z_subj) & (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
        # Depth-discontinuity guard: pixels straddling a depth edge smear into
        # streaks when the camera moves (foreground/background parallax differs).
        # Invalidate a 5x5 band; the depth-weighted hole fill repaints it.
        z4 = z.unsqueeze(0)
        z_range = (F.max_pool2d(z4, 5, 1, 2)
                   + F.max_pool2d(-z4, 5, 1, 2)).squeeze(0)  # max - min over 5x5
        valid &= (z_range <= 0.10 * z).reshape(-1)
        cols = torch.from_numpy(
            np.asarray(img, dtype=np.float32) / 255.0).to(dev).reshape(-1, 3)[valid]
        # Splat into a 2x supersampled canvas (painter's z-buffer, far first):
        # forward motion magnifies near content ~2x and a 1-px splat then
        # leaves pepper-noise gaps; supersampling + normalized downsample
        # closes them.
        ss = 2
        lin = ((v[valid] * ss).round().long() * (w * ss)
               + (u[valid] * ss).round().long())
        order = torch.argsort(z2[valid], descending=True)
        canvas = torch.zeros(3, h * ss * w * ss, device=dev)
        canvas[:, lin[order]] = cols[order].T
        img2 = canvas.view(3, h * ss, w * ss)
        kn2 = torch.zeros(1, h * ss * w * ss, device=dev)
        kn2[0, lin] = 1.0
        kn2 = kn2.view(1, h * ss, w * ss)
        num = F.avg_pool2d((img2 * kn2).unsqueeze(0), ss).squeeze(0)
        den = F.avg_pool2d(kn2.unsqueeze(0), ss).squeeze(0)
        known = (den > 0.2).view(1, h, w)
        img_t = torch.where(known.expand_as(num), num / den.clamp_min(1e-6),
                            torch.zeros_like(num))
        img_t, known = _close_cracks(img_t, known)
        out = _hole_fill(img_t, known, src_weight=z.unsqueeze(0))
        arr = (out.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0
               ).round().astype(np.uint8)
        return Image.fromarray(arr)
