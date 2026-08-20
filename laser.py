"""Output layer. The app only ever deals in image coordinates; everything below
(galvo mapping, fencing, blanking, parking) happens here.

    out = make_output(sim=False, calib_json={"poster": "calib_poster.json"})
    out.show(points_img, "poster")
    out.show_many([(pts_a, "poster"), (pts_b, "mockup")])
    out.park(); out.close()

Every point is clamped to the safe box from device.yaml before it reaches the
DAC, so a coordinate bug cannot steer the beam outside the tested envelope."""
from __future__ import annotations
import ctypes
import os
import time
import numpy as np
from paths import data_path

DAC_MAX = 4095
PPS = 20000
PARK_XY = (200, 200)
BLANK_ON, BLANK_OFF = 255, 0

DEVICE_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "device.yaml")
_DEVICE = None

BLANK_PTS = 24


def join_blanked(chunks, n_blank=None):
    """Concatenate galvo-space chunks into one frame, dark between shapes.

    Must happen in galvo space: each surface has its own image->galvo map, so
    merging in image space would apply one surface's map to the other's points.
    """
    if n_blank is None:
        n_blank = int(device().get("blank_points", BLANK_PTS))
    chunks = [np.asarray(c, float) for c in chunks if c is not None and len(c)]
    if not chunks:
        return np.empty((0, 2)), np.empty(0, bool)
    pts, mask = [], []
    for k, c in enumerate(chunks):
        if k:
            bridge = np.linspace(pts[-1][-1], c[0], n_blank)
            pts.append(bridge); mask.append(np.zeros(len(bridge), bool))
        pts.append(c); mask.append(np.ones(len(c), bool))
    if len(chunks) > 1:
        back = np.linspace(pts[-1][-1], chunks[0][0], n_blank)
        pts.append(back); mask.append(np.zeros(len(back), bool))
    return np.concatenate(pts), np.concatenate(mask)


def device(path=DEVICE_YAML) -> dict:
    """Load device.yaml. Falls back to the full DAC range if missing."""
    global _DEVICE
    if _DEVICE is None:
        try:
            import yaml
            _DEVICE = yaml.safe_load(open(path)) or {}
        except Exception as e:
            print(f"[!] {path} 없음/오류({e}) — 장치 한계를 DAC 전범위로 가정")
            _DEVICE = {}
    return _DEVICE


class Fence:
    """Clamps outgoing points to a safe region.

    device.yaml picks the reference: the measured hardware envelope (device),
    the calibration grid plus margin (calib), or the intersection (both).
    """

    def __init__(self, lo=(0, 0), hi=(DAC_MAX, DAC_MAX), margin=0.0):
        lo, hi = np.array(lo, float), np.array(hi, float)
        pad = (hi - lo) * margin
        self.lo, self.hi = lo - pad, hi + pad
        self.lo = np.maximum(self.lo, 0)
        self.hi = np.minimum(self.hi, DAC_MAX)
        self.violations = 0

    @classmethod
    def from_device(cls):
        box = device().get("safe_box") or {}
        x, y = box.get("x"), box.get("y")
        if not (x and y):
            return cls()
        return cls((x[0], y[0]), (x[1], y[1]))

    @classmethod
    def from_calib(cls, calib_json="calib.json", margin=None):
        """Pick the fence reference according to fence_mode."""
        d = device()
        mode = d.get("fence_mode", "device")
        margin = d.get("fence_margin", 0.05) if margin is None else margin

        dev = cls.from_device()
        if mode == "device":
            return dev

        cal = None
        try:
            import json
            pts = np.array(json.load(open(data_path(calib_json))).get("galvo_bounds", []), float)
            if len(pts) == 2:
                cal = cls(pts[0], pts[1], margin)
        except Exception:
            pass
        if cal is None:
            return dev
        if mode == "both":
            out = cls()
            out.lo = np.maximum(dev.lo, cal.lo)
            out.hi = np.minimum(dev.hi, cal.hi)
            return out
        return cal

    def clamp(self, pts: np.ndarray) -> np.ndarray:
        out = np.clip(pts, self.lo, self.hi)
        n = int((out != pts).any(axis=1).sum())
        if n:
            self.violations += n
        return out

    def outside(self, pts: np.ndarray) -> int:
        """Count points outside the fence without clamping them."""
        p = np.atleast_2d(np.asarray(pts, float))
        return int(((p < self.lo) | (p > self.hi)).any(axis=1).sum())


def find_helios_lib():
    """Locate the Helios DAC shared library.

    Checks $HELIOS_LIB first, then the directory holding this file, so the
    bundled libHeliosLaserDAC.dylib works with no environment setup.
    """
    env = os.environ.get("HELIOS_LIB")
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    for name in ("libHeliosLaserDAC.dylib", "libHeliosDacAPI.dylib",
                 "libHeliosLaserDAC.so", "libHeliosDacAPI.so"):
        p = os.path.join(here, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "Helios library not found. Put libHeliosLaserDAC.dylib next to laser.py "
        "or set HELIOS_LIB=/path/to/lib")


class _Pt(ctypes.Structure):
    _fields_ = [("x", ctypes.c_uint16), ("y", ctypes.c_uint16),
                ("r", ctypes.c_uint8), ("g", ctypes.c_uint8),
                ("b", ctypes.c_uint8), ("i", ctypes.c_uint8)]


class _OutBase:
    """Holds one (i2g, fence) profile per surface.

    The fence has to be per surface -- galvo_bounds differ, so sharing one would
    clip valid coordinates on whichever surface sits further out.
    """

    def __init__(self, profiles: dict):
        self.profiles = profiles
        self.default = next(iter(profiles))
        self.cur_surface = None
        self.blank = True
        self._warned = set()

    def _profile(self, surface):
        if surface in self.profiles:
            return self.profiles[surface]
        if surface is not None and surface not in self._warned:
            self._warned.add(surface)
            print(f"[!] 표면 '{surface}' 의 캘리브가 없음 — "
                  f"'{self.default}' 프로파일로 대체 (조준이 어긋난다)")
        return self.profiles[self.default]


class HeliosOut(_OutBase):
    def __init__(self, profiles: dict):
        super().__init__(profiles)
        lib = find_helios_lib()
        self.dll = ctypes.cdll.LoadLibrary(lib)
        if self.dll.OpenDevices() < 1:
            raise RuntimeError("Helios 장치를 찾을 수 없음")
        self.dev = 0

    def _write(self, galvo: np.ndarray, on, fence: Fence = None):
        """on: a bool for the whole frame, or a per-point mask."""
        if fence is not None:
            galvo = fence.clamp(galvo)
        n = len(galvo)
        mask = (np.full(n, bool(on)) if np.ndim(on) == 0
                else np.asarray(on, bool))
        arr = (_Pt * n)(*[
            _Pt(int(x), int(y), *((BLANK_ON,) * 4 if m else (BLANK_OFF,) * 4))
            for (x, y), m in zip(galvo, mask)])
        for _ in range(200):
            if self.dll.GetStatus(self.dev) == 1:
                break
            time.sleep(0.002)
        self.dll.WriteFrame(self.dev, PPS, ctypes.c_uint8(1),
                            ctypes.pointer(arr), ctypes.c_int(n))

    def show(self, pts_img, surface=None):
        if pts_img is None or len(pts_img) == 0:
            return self.park()
        i2g, fence = self._profile(surface)
        if i2g is None:
            raise RuntimeError("실기 출력에는 캘리브가 반드시 필요")
        g = fence.clamp(np.asarray(i2g(np.asarray(pts_img, float)), float))
        if surface != self.cur_surface and not self.blank:
            self._write(g, on=False)
        self._write(g, on=True)
        self.cur_surface = surface
        self.blank = False

    def show_many(self, items):
        """items = [(points_img, surface), ...] drawn in one frame.

        Each surface is converted through its own map and fence, then merged in
        galvo space, so shapes on different surfaces can be lit at the same time.
        """
        chunks = []
        for pts_img, surface in items:
            if pts_img is None or len(pts_img) == 0:
                continue
            i2g, fence = self._profile(surface)
            if i2g is None:
                raise RuntimeError("실기 출력에는 캘리브가 반드시 필요")
            chunks.append(fence.clamp(np.asarray(i2g(np.asarray(pts_img, float)), float)))
        if not chunks:
            return self.park()
        if len(chunks) == 1:
            self._write(chunks[0], True)
        else:
            g, mask = join_blanked(chunks)
            self._write(g, mask)
        self.cur_surface = None
        self.blank = False

    def park(self):
        self._write(np.array([PARK_XY] * 64, float), False)
        self.blank = True
        self.cur_surface = None

    def close(self):
        self.park(); time.sleep(0.2); self.dll.CloseDevices()


class SimOut(_OutBase):
    """Draws to a window instead of the DAC, same interface."""

    def __init__(self, profiles: dict, bg=None):
        super().__init__(profiles)
        import cv2
        self.cv2 = cv2
        self.bg = bg if bg is not None else np.full((720, 1080, 3), 22, np.uint8)

    def show(self, pts_img, surface=None):
        if pts_img is None or len(pts_img) == 0:
            return self.park()
        i2g, fence = self._profile(surface)
        p = np.asarray(pts_img, float)
        if i2g is not None:
            fence.clamp(np.asarray(i2g(p), float))
        img = self.bg.copy()
        for a, b in zip(p[:-1], p[1:]):
            self.cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)),
                          (60, 255, 60), 2, self.cv2.LINE_AA)
        if surface is not None:
            self.cv2.putText(img, surface, (12, 28), self.cv2.FONT_HERSHEY_SIMPLEX,
                             0.7, (60, 255, 60), 2, self.cv2.LINE_AA)
        self.cv2.imshow("laser (sim)", img); self.cv2.waitKey(1)
        self.cur_surface = surface
        self.blank = False

    def show_many(self, items):
        """items = [(points_img, surface), ...] drawn in one frame."""
        items = [(p, s) for p, s in items if p is not None and len(p)]
        if not items:
            return self.park()
        img = self.bg.copy()
        for k, (pts_img, surface) in enumerate(items):
            i2g, fence = self._profile(surface)
            p = np.asarray(pts_img, float)
            if i2g is not None:
                fence.clamp(np.asarray(i2g(p), float))
            col = (60, 255, 60) if k == 0 else (60, 220, 255)
            for a, b in zip(p[:-1], p[1:]):
                self.cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)),
                              col, 2, self.cv2.LINE_AA)
            if surface is not None:
                self.cv2.putText(img, surface, (12, 28 + k * 26),
                                 self.cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2,
                                 self.cv2.LINE_AA)
        self.cv2.imshow("laser (sim)", img); self.cv2.waitKey(1)
        self.cur_surface = None
        self.blank = False

    def park(self):
        self.cv2.imshow("laser (sim)", self.bg); self.cv2.waitKey(1)
        self.blank = True
        self.cur_surface = None

    def close(self):
        self.cv2.destroyAllWindows()


def _load_profile(calib_json: str):
    """calib_<surface>.json → (i2g, fence)."""
    i2g = None
    try:
        import calib
        _, i2g = calib.load(calib_json)
    except Exception as e:
        print(f"[!] 캘리브 로드 실패({calib_json}: {e}) — 이미지 좌표를 그대로 사용")
    return i2g, Fence.from_calib(calib_json)


def make_output(sim=True, calib_json="calib.json", bg=None):
    """calib_json: one path, or {surface: path} for multiple surfaces."""
    if isinstance(calib_json, dict):
        if not calib_json:
            raise ValueError("calib_json dict 가 비어 있다")
        profiles = {s: _load_profile(data_path(p)) for s, p in calib_json.items()}
        print(f"[i] 출력 프로파일 {list(profiles)}")
    else:
        profiles = {None: _load_profile(data_path(calib_json))}

    if sim:
        return SimOut(profiles, bg)
    missing = [s for s, (i2g, _) in profiles.items() if i2g is None]
    if missing:
        raise RuntimeError(f"실기 출력에는 캘리브가 반드시 필요 — 실패한 표면: {missing}")
    return HeliosOut(profiles)
