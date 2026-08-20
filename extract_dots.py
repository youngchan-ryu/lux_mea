"""Find the laser dot in each still segment of a calibration recording.

Detection is background subtraction plus an automatic threshold: build a median
background, measure how much the laser's colour channel rises above it, then
split the resulting distribution with Otsu. Works without knowing the exposure.

Segments are found by looking for stretches where the dot stops moving, so the
result depends only on order, never on timing."""
from __future__ import annotations
import sys
import cv2
import numpy as np

OFFSET = 0.75


def build_background(cap, n=21):
    """Median of n frames spread across the video.

    The dot keeps moving, so it vanishes from the median and what remains is a
    clean background. Subtracting it finds a dim dot on a bright wall, which
    plain brightness thresholding does not.
    """
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    idxs = np.linspace(0, max(total - 1, 0), n).astype(int)
    frames = []
    for k in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(k))
        ok, fr = cap.read()
        if ok:
            frames.append(fr)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if not frames:
        return None
    return np.median(np.stack(frames), axis=0).astype(np.int16)


def response_map(bgr, bg, green=False):
    """How far the laser's colour channel rises above the background."""
    f = bgr.astype(np.int16)
    d = f - bg if bg is not None else f
    b, g, r = d[:, :, 0], d[:, :, 1], d[:, :, 2]
    resp = (g - np.maximum(r, b)) if green else (r - np.maximum(g, b))
    resp = np.clip(resp, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(resp, (5, 5), 0)


def peak_of(resp, win=21):
    """(strength, x, y) of the peak, sub-pixel via a weighted centroid."""
    _, mx, _, loc = cv2.minMaxLoc(resp)
    x0, y0 = loc
    h, w = resp.shape
    a, b = max(0, y0 - win), min(h, y0 + win + 1)
    c, e = max(0, x0 - win), min(w, x0 + win + 1)
    patch = resp[a:b, c:e].astype(float)
    if patch.sum() <= 0:
        return mx, float(x0), float(y0)
    ys, xs = np.mgrid[a:b, c:e]
    return mx, float((xs * patch).sum() / patch.sum()), float((ys * patch).sum() / patch.sum())


def detect_dot(bgr, green=False, bg=None, min_resp=12):
    """Single frame detection. Pass bg when you have it."""
    mx, x, y = peak_of(response_map(bgr, bg, green))
    return (x, y) if mx >= min_resp else None


def scan_segments(cap, green, sample_dt=0.06, min_samples=3, move_th=12.0,
                  debug=False):
    """Find the stretches where the dot sits still, in order.

    Sampling at fixed times drifts: a slightly wrong dwell accumulates until the
    last points land on their neighbours. Using order alone removes that failure.
    """
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(round(fps * sample_dt)))

    bg = build_background(cap)
    print("[i] 중앙값 배경 생성 " + ("완료" if bg is not None else "실패(차분 없이 진행)"))

    samples, idx = [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            mx, x, y = peak_of(response_map(fr, bg, green))
            samples.append((idx / fps, mx, x, y))
        idx += 1
    if not samples:
        return []

    peaks = np.array([s[1] for s in samples], dtype=np.uint8)
    th, _ = cv2.threshold(peaks.reshape(-1, 1), 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = max(float(th), 8.0)
    n_on = int((peaks >= th).sum())
    print(f"[i] 응답 세기 자동 임계 {th:.0f} "
          f"(범위 {peaks.min()}~{peaks.max()}, 점 있음 {n_on}/{len(peaks)} 샘플)")
    if peaks.max() < 15:
        print("  ⚠️ 최대 응답이 매우 약함 — 실내등을 낮추거나 노출을 줄여 재촬영 권장")

    if debug:
        np.savetxt("peaks.csv", np.array([[s[0], s[1]] for s in samples]),
                   delimiter=",", header="t_sec,peak", comments="")
        print("  [i] peaks.csv 저장 — 세기 시계열로 임계 확인 가능")

    segs, cur = [], []
    for t, mx, x, y in samples:
        if mx < th:
            if len(cur) >= min_samples:
                segs.append(cur)
            cur = []
            continue
        if cur and np.hypot(x - cur[-1][1], y - cur[-1][2]) > move_th:
            if len(cur) >= min_samples:
                segs.append(cur)
            cur = []
        cur.append((t, x, y))
    if len(cur) >= min_samples:
        segs.append(cur)

    out = []
    for sg in segs:
        a = np.array(sg)
        out.append((float(np.median(a[:, 0])), float(np.median(a[:, 1])),
                    float(np.median(a[:, 2])), len(sg)))
    return out


def main():
    if len(sys.argv) < 3:
        print(__doc__); return
    video, log = sys.argv[1], sys.argv[2]
    green = "--green" in sys.argv
    review = "--review" in sys.argv

    entries = []
    for line in open(log):
        line = line.strip()
        if not line or line[0].isalpha():
            continue
        t, gx, gy = line.split(",")[:3]
        entries.append((float(t), float(gx), float(gy)))
    print(f"[i] 격자점 {len(entries)}개")

    cap = cv2.VideoCapture(video)
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[i] 영상 해상도 {vw}x{vh}  ← 캘리브 이미지 좌표계의 기준")

    segs = scan_segments(cap, green, debug="--debug" in sys.argv)
    print(f"[i] 정지 구간 {len(segs)}개 검출 / 기대 {len(entries)}개")

    if len(segs) != len(entries):
        print("  ⚠️ 개수 불일치 — 아래 중 하나를 조정할 것:")
        print("     · 점이 어두워 검출 실패 → 실내등 낮추기, 재촬영")
        print("     · 인접 격자점이 너무 가까움 → scan_segments(move_th) 낮추기")
        print("     · 반사광이 점으로 오인 → 무광 배경 확인")
        for k, (t, x, y, n) in enumerate(segs):
            print(f"     seg{k:2d} t={t:6.2f}s ({x:7.1f},{y:7.1f}) {n}프레임")
        if len(segs) < len(entries):
            print("  [!] 구간이 부족해 대응 불가. 재촬영 권장."); cap.release(); return
        segs = segs[:len(entries)]
        print("  [i] 앞에서부터 잘라 진행 — --review 로 반드시 확인할 것")

    rows = [(gx, gy, sx, sy) for (_, gx, gy), (_, sx, sy, _) in zip(entries, segs)]
    with open("pairs.csv", "w") as f:
        f.write(f"# size,{vw},{vh}\n")
        f.write("galvo_x,galvo_y,img_x,img_y\n")
        for r in rows:
            f.write("%.1f,%.1f,%.3f,%.3f\n" % r)
    print(f"[i] pairs.csv 저장: {len(rows)}점")

    if review:
        cap.set(cv2.CAP_PROP_POS_MSEC, segs[0][0] * 1000.0)
        ok, fr = cap.read()
        if ok:
            v = fr.copy()
            for k, (_, x, y, _) in enumerate(segs):
                cv2.circle(v, (int(x), int(y)), 11, (0, 255, 0), 2)
                cv2.putText(v, str(k), (int(x) + 12, int(y) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imwrite("grid_review.png", v)
            print("[i] grid_review.png — 번호가 격자 순서(행 우선)와 맞는지 확인")
    cap.release()
    print("\n다음: python3 calib.py fit pairs.csv --mm <mm_per_px>")


if __name__ == "__main__":
    main()
