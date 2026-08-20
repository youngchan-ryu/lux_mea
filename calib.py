"""Galvo <-> image mapping.

A homography handles the plane-to-plane projection; a 2nd order polynomial
absorbs what is left (galvo pincushion, lens distortion, axis scale mismatch).
Accuracy is reported on points held out of the fit -- training error says
nothing useful here.

    python3 calib.py fit pairs.csv"""
from __future__ import annotations
import json
import sys
import numpy as np


def _poly(xy: np.ndarray) -> np.ndarray:
    """(N,2) -> (N,6) quadratic basis [1, x, y, x^2, xy, y^2]."""
    x, y = xy[:, 0], xy[:, 1]
    return np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], axis=1)


def _homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """3x3 homography by DLT. Hand-rolled so calib.py needs no cv2."""
    n = len(src)
    A = np.zeros((2 * n, 9))
    for i, ((x, y), (u, v)) in enumerate(zip(src, dst)):
        A[2 * i] = [-x, -y, -1, 0, 0, 0, u * x, u * y, u]
        A[2 * i + 1] = [0, 0, 0, -x, -y, -1, v * x, v * y, v]
    _, _, vt = np.linalg.svd(A)
    H = vt[-1].reshape(3, 3)
    return H / H[2, 2]


def _apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.concatenate([pts, np.ones((len(pts), 1))], axis=1) @ H.T
    return p[:, :2] / p[:, 2:3]


class Map:
    """One direction: src -> dst, homography plus polynomial residual."""

    def __init__(self, H, coef, s_mu, s_sd):
        self.H, self.coef = np.asarray(H), np.asarray(coef)
        self.s_mu, self.s_sd = np.asarray(s_mu), np.asarray(s_sd)

    @classmethod
    def fit(cls, src: np.ndarray, dst: np.ndarray):
        H = _homography(src, dst)
        s_mu, s_sd = src.mean(0), src.std(0) + 1e-9
        base = _apply_h(H, src)
        resid = dst - base
        F = _poly((src - s_mu) / s_sd)
        coef, *_ = np.linalg.lstsq(F, resid, rcond=None)
        return cls(H, coef, s_mu, s_sd)

    def __call__(self, src: np.ndarray) -> np.ndarray:
        src = np.atleast_2d(np.asarray(src, float))
        out = _apply_h(self.H, src)
        out = out + _poly((src - self.s_mu) / self.s_sd) @ self.coef
        return out

    def to_dict(self):
        return {"H": self.H.tolist(), "coef": self.coef.tolist(),
                "s_mu": self.s_mu.tolist(), "s_sd": self.s_sd.tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(d["H"], d["coef"], d["s_mu"], d["s_sd"])


def load_pairs(path: str):
    """Returns (galvo, img, img_size); size comes from the '# size,W,H' header."""
    rows, size = [], None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("# size"):
                _, w, h = line.split(","); size = [int(w), int(h)]
                continue
            if not line or line[0].isalpha() or line[0] == "#":
                continue
            parts = [p for p in line.replace(",", " ").split() if p]
            if len(parts) >= 4:
                rows.append([float(p) for p in parts[:4]])
    a = np.array(rows)
    return a[:, :2], a[:, 2:4], size


def fit_and_report(pairs_path: str, mm_per_px: float | None = None,
                   holdout: float = 0.3, seed: int = 0):
    galvo, img, img_size = load_pairs(pairs_path)
    n = len(galvo)
    if img_size is None:
        print("[!] pairs.csv에 '# size,W,H' 가 없음 — extract_dots.py를 다시 실행하거나")
        print("    calib.json의 img_size를 수동으로 채울 것 (app.py가 필요로 함)")
    if n < 10:
        print(f"[!] 점이 {n}개뿐. 최소 10개(권장 25개) 필요.")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    k = max(4, int(n * (1 - holdout)))
    tr, te = idx[:k], idx[k:]

    g2i = Map.fit(galvo[tr], img[tr])
    i2g = Map.fit(img[tr], galvo[tr])

    def rms(pred, true):
        return float(np.sqrt(((pred - true) ** 2).sum(1).mean()))

    r_tr = rms(g2i(galvo[tr]), img[tr])
    r_te = rms(g2i(galvo[te]), img[te]) if len(te) else float("nan")
    r_rt = rms(i2g(g2i(galvo)), galvo)

    print(f"학습 {len(tr)}점 / 검증 {len(te)}점")
    print(f"  갈보→이미지 학습 RMS : {r_tr:.2f} px")
    print(f"  갈보→이미지 검증 RMS : {r_te:.2f} px   ← 이 값이 실제 정확도")
    print(f"  왕복 일관성          : {r_rt:.2f} galvo units")
    if mm_per_px:
        print(f"  검증 RMS (mm)        : {r_te * mm_per_px:.2f} mm  (목표 ≤3mm)")
    if not np.isnan(r_te) and r_te > 3 * max(r_tr, 1e-6):
        print("  ⚠️ 검증 오차가 학습 오차보다 훨씬 큼 → 과적합/이상점 의심")

    lo = galvo.min(0).tolist(); hi = galvo.max(0).tolist()
    out = {"galvo_to_img": g2i.to_dict(), "img_to_galvo": i2g.to_dict(),
           "n_points": n, "rms_px": r_te, "mm_per_px": mm_per_px,
           "img_size": img_size,
           "galvo_bounds": [lo, hi]}
    with open("calib.json", "w") as f:
        json.dump(out, f, indent=1)
    print("  → calib.json 저장")
    return g2i, i2g


def load(path="calib.json"):
    d = json.load(open(path))
    return Map.from_dict(d["galvo_to_img"]), Map.from_dict(d["img_to_galvo"])


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "fit":
        mm = None
        if "--mm" in sys.argv:
            mm = float(sys.argv[sys.argv.index("--mm") + 1])
        fit_and_report(sys.argv[2], mm)
    else:
        print(__doc__)
