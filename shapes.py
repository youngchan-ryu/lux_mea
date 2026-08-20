"""Shape generation, in image coordinates.

Shapes are built in image space and converted point by point to galvo angles.
Building them in galvo space instead makes circles come out as ellipses on any
surface that is not perpendicular to the beam."""
from __future__ import annotations
import numpy as np

STEPS_EDGE = 24
DWELL_CORNER = 5


def circle(cx, cy, r, n=180):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], axis=1)


def rect(cx, cy, w, h, steps=STEPS_EDGE, dwell=DWELL_CORNER):
    c = [(cx - w / 2, cy - h / 2), (cx + w / 2, cy - h / 2),
         (cx + w / 2, cy + h / 2), (cx - w / 2, cy + h / 2)]
    pts = []
    for i in range(4):
        x0, y0 = c[i]; x1, y1 = c[(i + 1) % 4]
        pts += [(x0, y0)] * dwell
        t = np.linspace(0, 1, steps, endpoint=False)
        pts += list(zip(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
    return np.array(pts, float)


def crosshair(cx, cy, r, gap=0.35, steps=12):
    """Cross with an open centre, for pointing without covering."""
    segs = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        t = np.linspace(gap, 1.0, steps)
        segs.append(np.stack([cx + dx * r * t, cy + dy * r * t], axis=1))
    return np.concatenate(segs)


def ellipse(cx, cy, rx, ry, n=180):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([cx + rx * np.cos(t), cy + ry * np.sin(t)], axis=1)


def underline(cx, cy, w, dwell=DWELL_CORNER, steps=2):
    """A single rule through the anchor, for text."""
    x0, x1 = cx - w / 2, cx + w / 2
    t = np.linspace(0, 1, max(steps, 2))
    line = np.stack([x0 + (x1 - x0) * t, np.full_like(t, cy)], axis=1)
    return np.concatenate([[line[0]] * dwell, line, [line[-1]] * dwell])


def rounded_rect(cx, cy, w, h, radius=None, steps_edge=STEPS_EDGE,
                 steps_corner=10, dwell=DWELL_CORNER):
    """Rounded rectangle; radius defaults to 22% of the short side."""
    if radius is None:
        radius = min(w, h) * 0.22
    radius = max(1e-6, min(radius, w / 2, h / 2))
    x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    corners = [
        (x0 + radius, y0 + radius, np.pi, 1.5 * np.pi),
        (x1 - radius, y0 + radius, 1.5 * np.pi, 2 * np.pi),
        (x1 - radius, y1 - radius, 0.0, 0.5 * np.pi),
        (x0 + radius, y1 - radius, 0.5 * np.pi, np.pi),
    ]
    pts = []
    for i, (ccx, ccy, a0, a1) in enumerate(corners):
        t = np.linspace(a0, a1, steps_corner)
        arc = np.stack([ccx + radius * np.cos(t), ccy + radius * np.sin(t)], axis=1)
        pts.append(arc)
        pts.append(np.array([arc[-1]] * dwell))
        ncx, ncy, na0, _ = corners[(i + 1) % 4]
        p1 = np.array([ncx + radius * np.cos(na0), ncy + radius * np.sin(na0)])
        e = np.linspace(0, 1, steps_edge, endpoint=False)[:, None]
        pts.append(arc[-1] + (p1 - arc[-1]) * e)
    return np.concatenate(pts)


def make(spec: dict, cx: float, cy: float, scale: float = 1.0):
    """parts.yaml shape spec -> point array."""
    t = spec.get("type", "circle")
    if t == "rect":
        return rect(cx, cy, spec["w"] * scale, spec.get("h", spec["w"]) * scale)
    if t in ("roundrect", "rounded_rect"):
        radius = spec.get("radius")
        return rounded_rect(cx, cy, spec["w"] * scale, spec.get("h", spec["w"]) * scale,
                            radius=radius * scale if radius else None)
    if t == "ellipse":
        return ellipse(cx, cy, spec["rx"] * scale, spec.get("ry", spec["rx"]) * scale)
    if t == "underline":
        return underline(cx, cy, spec["w"] * scale)
    if t == "crosshair":
        return crosshair(cx, cy, spec["r"] * scale)
    return circle(cx, cy, spec.get("r", 40) * scale)


def glide(p_from, p_to, t):
    """Cosine-eased move between two shapes, t in [0,1]."""
    e = 0.5 - 0.5 * np.cos(np.pi * np.clip(t, 0, 1))
    c_from, c_to = np.mean(p_from, axis=0), np.mean(p_to, axis=0)
    shape = p_to if t > 0.5 else p_from
    center = np.mean(shape, axis=0)
    return shape - center + (c_from + (c_to - c_from) * e)


def interleave(*shapes):
    """Merge shapes into one frame; returns the transition spans to blank."""
    pts, blanks, i = [], [], 0
    for k, s in enumerate(shapes):
        if k:
            blanks.append((i, i + 2))
        pts.append(s); i += len(s)
    return np.concatenate(pts), blanks


def total_points(*shapes):
    n = sum(len(s) for s in shapes)
    return n, 20000 / max(n, 1)
