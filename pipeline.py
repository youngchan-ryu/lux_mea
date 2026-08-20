"""One recording -> plate, point pairs, calibration, quality report.

    python3 pipeline.py grid.mov grid_log.csv --pitch-mm poster=275,mockup=50

The log's `surface` column splits the segments, so a single video containing
two sweeps yields two independent calibrations. Surfaces at different depths
need separate fits: the map bends at the boundary and one homography cannot
represent that bend.

The registration plate is the per-pixel median of the video. The laser dot
moves, so it disappears from the median, and the viewpoint is identical to the
calibration by construction."""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys

import cv2
import numpy as np

import extract_dots as ED
import calib as CB


def make_plate(video: str, out_path: str) -> tuple[int, int]:
    """Write the laser-free plate and return its size."""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        sys.exit(f"[!] 영상을 못 엶: {video}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    bg = ED.build_background(cap, n=31)
    cap.release()
    if bg is None:
        sys.exit("[!] 배경판 생성 실패 — 영상이 비었을 수 있음")
    cv2.imwrite(out_path, np.clip(bg, 0, 255).astype(np.uint8))
    print(f"[i] 배경판 저장 {out_path}  ({w}x{h})")
    return w, h


def check_segments(segs, n_expect: int, dwell_s: float | None,
                   sample_dt: float = 0.06) -> bool:
    """Flag odd segment counts and lengths. True if clean."""
    ok = True
    print(f"\n[검사] 정지 구간 {len(segs)}개 / 기대 {n_expect}개")
    if len(segs) != n_expect:
        ok = False
        if len(segs) < n_expect:
            print("  ✗ 부족 — 두 점이 한 구간으로 붙었거나 검출 실패.")
            print("    → firstlight.py grid 의 gap_s 를 0.30 이상으로 늘려 재촬영")
        else:
            print("  ✗ 초과 — 반사광이 점으로 오인됐거나 한 점이 쪼개졌다.")
            print("    → 무광 배경 확인, 실내등 낮추기")

    counts = np.array([s[3] for s in segs], float)
    if len(counts) == 0:
        return False
    med = float(np.median(counts))
    print(f"  구간당 샘플 수: 중앙값 {med:.0f}  범위 {counts.min():.0f}~{counts.max():.0f}")
    if dwell_s:
        exp = dwell_s / sample_dt
        print(f"  기대 샘플 수 ≈ {exp:.0f} (체류 {dwell_s}s ÷ {sample_dt}s)")
        if med < exp * 0.6:
            print("  ⚠️ 실측이 기대의 60% 미만 — 점이 어두워 중간중간 놓치고 있다")
            ok = False

    long = np.where(counts > med * 1.7)[0]
    short = np.where(counts < max(med * 0.4, 3))[0]
    for i in long:
        print(f"  ⚠️ seg{i:2d} 가 비정상적으로 김 ({counts[i]:.0f}샘플) "
              f"— 인접 두 점이 붙었을 가능성. 이후 대응이 전부 밀린다")
        ok = False
    for i in short:
        print(f"  ⚠️ seg{i:2d} 가 비정상적으로 짧음 ({counts[i]:.0f}샘플) — 반사광 오검출 의심")
        ok = False
    if ok:
        print("  ✅ 구간 품질 정상")
    return ok


def mm_per_px_from_top_row(img: np.ndarray, n: int, grid_mm: float) -> float | None:
    """mm per pixel, from the measured width of the top grid row.

    grid_mm is the full row, first point to last -- not the spacing between
    adjacent points, which is (n-1) times smaller. Use --pitch-mm for that.
    """
    if grid_mm is None or len(img) < n:
        return None
    d = float(np.hypot(*(img[n - 1] - img[0])))
    if d < 1:
        return None
    return grid_mm / d


def read_log(path: str):
    """grid_log.csv -> [(galvo_x, galvo_y, surface|None)], in order."""
    rows = []
    for line in open(path):
        line = line.strip()
        if not line or line[0].isalpha() or line[0] == "#":
            continue
        p = [c.strip() for c in line.split(",")]
        surf = p[3] if len(p) > 3 and p[3] else None
        rows.append((float(p[1]), float(p[2]), surf))
    return rows


def group_by_surface(rows, default: str):
    """Group by surface, keeping order, and check each run is contiguous."""
    order, groups, seen_runs = [], {}, []
    for gx, gy, s in rows:
        s = s or default
        if s not in groups:
            groups[s] = []
            order.append(s)
        groups[s].append((gx, gy))
        if not seen_runs or seen_runs[-1] != s:
            seen_runs.append(s)
    if len(seen_runs) != len(order):
        print("[!] 로그에서 표면이 번갈아 나온다 — sweep을 표면별로 연속 실행하지 않았다.")
        print("    세그먼트를 순서대로 쪼갤 수 없으므로 결과를 신뢰할 수 없다.")
    return order, groups


def boundaries_by_time_gap(segs, k: int):
    """Guess surface boundaries from the largest gaps between segments."""
    if k < 2 or len(segs) < k:
        return None
    gaps = sorted(((segs[i + 1][0] - segs[i][0], i + 1)
                   for i in range(len(segs) - 1)), reverse=True)
    return sorted(i for _, i in gaps[:k - 1])


def parse_per_surface(spec: str | None, surfaces, what: str):
    """'1100' or 'poster=1100,mockup=200' -> {surface: float|None}."""
    out = {s: None for s in surfaces}
    if not spec:
        return out
    if "=" not in spec:
        try:
            v = float(spec)
        except ValueError:
            sys.exit(f"[!] {what} 값을 못 읽음: {spec!r}")
        return {s: v for s in surfaces}
    for part in spec.split(","):
        k, _, v = part.partition("=")
        k = k.strip()
        if k not in out:
            print(f"[!] {what}: 로그에 없는 표면 '{k}' — 무시")
            continue
        try:
            out[k] = float(v)
        except ValueError:
            sys.exit(f"[!] {what}: '{part}' 를 못 읽음")
    return out


def coverage_report(img: np.ndarray, W: int, H: int, mmpp: float | None,
                    multi: bool = False):
    """Report the image region the grid actually covered.

    Anchors outside it get clamped by the fence and extrapolated by the polynomial.
    With two surfaces the frame-area fraction is meaningless (a small surface is
    supposed to be small), so only the enclosing rectangle is checked.
    """
    x0, x1 = img[:, 0].min(), img[:, 0].max()
    y0, y1 = img[:, 1].min(), img[:, 1].max()
    frac = (x1 - x0) * (y1 - y0) / (W * H)
    print(f"\n[커버리지] 격자가 덮은 이미지 영역")
    print(f"  x {x0:7.1f}~{x1:7.1f} px   (정규 {x0/W:.3f}~{x1/W:.3f})")
    print(f"  y {y0:7.1f}~{y1:7.1f} px   (정규 {y0/H:.3f}~{y1/H:.3f})")
    print(f"  프레임 면적의 {frac*100:.1f}%")
    if mmpp:
        print(f"  실물 크기 ≈ {(x1-x0)*mmpp:.0f} x {(y1-y0)*mmpp:.0f} mm")
    if multi:
        print("  → review 이미지의 주황 사각형이 **이 표면의 등록 대상을 덮는지**만 확인할 것")
        print("    (이중표면에서는 프레임 면적 비율이 작은 게 정상이다)")
    elif frac < 0.15:
        print("  ✗ 너무 좁다. AMPLITUDE 를 키워 재촬영할 것.")
        print("    (등록 대상이 이 사각형 안에 완전히 들어와야 한다)")
    elif frac < 0.30:
        print("  ⚠️ 좁은 편. 대상이 이 사각형 안에 다 들어오는지 배경판에서 확인.")
    else:
        print("  ✅ 충분")
    print(f"  → 유효 anchor 범위: x {x0/W:.2f}~{x1/W:.2f}, y {y0/H:.2f}~{y1/H:.2f}")
    return (x0 / W, x1 / W, y0 / H, y1 / H)


def fit(pairs_path: str, out_path: str, mmpp: float | None,
        holdout: float = 0.3, seed: int = 0):
    """Same fit as calib.fit_and_report, with a chosen output path."""
    galvo, img, size = CB.load_pairs(pairs_path)
    n = len(galvo)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    k = max(4, int(n * (1 - holdout)))
    tr, te = idx[:k], idx[k:]

    g2i = CB.Map.fit(galvo[tr], img[tr])
    i2g = CB.Map.fit(img[tr], galvo[tr])

    def rms(p, t):
        return float(np.sqrt(((p - t) ** 2).sum(1).mean()))

    r_tr = rms(g2i(galvo[tr]), img[tr])
    r_te = rms(g2i(galvo[te]), img[te]) if len(te) else float("nan")
    r_rt = rms(i2g(g2i(galvo)), galvo)

    print(f"\n[적합] 학습 {len(tr)}점 / 검증 {len(te)}점")
    print(f"  학습 RMS : {r_tr:.2f} px")
    print(f"  검증 RMS : {r_te:.2f} px   ← 실제 조준 정확도")
    print(f"  왕복     : {r_rt:.2f} galvo unit")
    if mmpp:
        print(f"  검증 RMS : {r_te*mmpp:.2f} mm   (목표 ≤3mm)")
    if not np.isnan(r_te) and r_te > 3 * max(r_tr, 1e-6):
        print("  ⚠️ 검증 ≫ 학습 → 과적합/이상점 의심")

    out = {"galvo_to_img": g2i.to_dict(), "img_to_galvo": i2g.to_dict(),
           "n_points": n, "rms_px": r_te, "mm_per_px": mmpp,
           "img_size": size,
           "galvo_bounds": [galvo.min(0).tolist(), galvo.max(0).tolist()]}
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"  → {out_path} 저장")
    return r_te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("log", help="firstlight.py grid 이 만든 grid_log.csv")
    ap.add_argument("--surface", default="mockup",
                    help="로그에 surface 컬럼이 없을 때 쓸 표면 이름 (구버전 로그용). "
                         "컬럼이 있으면 무시되고 로그의 표면들이 각각 처리된다")
    ap.add_argument("--grid-mm", default=None,
                    help="최상단 행 **양 끝** 점 사이 실측 거리(mm). "
                         "'1100' 또는 표면별로 'poster=1100,mockup=200'")
    ap.add_argument("--pitch-mm", dest="pitch_mm", default=None,
                    help="**인접** 격자점 사이 실측 거리(mm). 내부에서 ×(n-1) 해 "
                         "행 전체 폭으로 환산한다. gridall 로 재기엔 이쪽이 편하다. "
                         "'275' 또는 'poster=275,mockup=50'")
    ap.add_argument("--n", type=int, default=5, help="격자 한 변의 점 수")
    ap.add_argument("--dwell", type=float, default=None,
                    help="점당 체류 시간(s). 주면 샘플 수 기대치를 검사한다")
    ap.add_argument("--green", action="store_true")
    ap.add_argument("--register", action="store_true",
                    help="끝나면 register.py 를 바로 실행")
    ap.add_argument("--force", action="store_true",
                    help="품질 검사 실패해도 계속 진행")
    a = ap.parse_args()

    rows = read_log(a.log)
    surfaces, groups = group_by_surface(rows, a.surface)
    multi = len(surfaces) > 1
    print(f"[i] 격자점 {len(rows)}개 / 표면 {len(surfaces)}개")
    for s in surfaces:
        print(f"     {s}: {len(groups[s])}점")

    grid_mm = parse_per_surface(a.grid_mm, surfaces, "--grid-mm")
    pitch_mm = parse_per_surface(a.pitch_mm, surfaces, "--pitch-mm")

    plate = "plate.jpg" if multi else f"plate_{a.surface}.jpg"
    W, H = make_plate(a.video, plate)

    cap = cv2.VideoCapture(a.video)
    segs = ED.scan_segments(cap, a.green)
    cap.release()

    ok = check_segments(segs, len(rows), a.dwell)
    if len(segs) != len(rows):
        print(f"\n[!] 세그먼트 {len(segs)}개 ≠ 격자점 {len(rows)}개.")
        if multi:
            print("    이중표면에서는 개수가 어긋나면 **어느 표면의 것이 빠졌는지**")
            print("      알 수 없어 표면 분할이 통째로 밀린다. 자동 절단하지 않는다.")
            if not a.force:
                print("    재촬영을 권한다. 그래도 진행하려면 --force")
                return
        else:
            if not ok and not a.force:
                print("    재촬영을 권한다. 그래도 진행하려면 --force")
                if len(segs) < len(rows):
                    return
            if len(segs) > len(rows):
                segs = segs[:len(rows)]
                print("  [i] 앞에서부터 잘라 진행")
            elif len(segs) < len(rows):
                rows = rows[:len(segs)]
                groups[a.surface] = groups[a.surface][:len(segs)]
                print("  [i] 갈보 로그를 잘라 진행 — 대응이 밀렸을 수 있다, 반드시 확인")

    counts = [len(groups[s]) for s in surfaces]
    if multi:
        expect = [int(e) for e in np.cumsum(counts[:-1])]
        guess = boundaries_by_time_gap(segs, len(surfaces))
        print(f"\n[검사] 표면 경계  로그기준={expect}  시간공백추정={guess}")
        if guess is not None and list(guess) != expect:
            print("  ⚠️ 두 추정이 다르다 — 검출 누락/과검출로 대응이 밀렸을 가능성이 높다.")
            print("     review_*.png 에서 각 표면의 점 번호가 격자 순서와 맞는지 반드시 확인할 것.")
        else:
            print("  ✅ 일치 — 표면 분할 신뢰 가능")

    idx0 = 0
    for s, cnt in zip(surfaces, counts):
        seg_s = segs[idx0:idx0 + cnt]
        idx0 += cnt
        if len(seg_s) < 4:
            print(f"\n[!] {s}: 점이 {len(seg_s)}개뿐 — 적합 불가, 건너뜀")
            continue
        print(f"\n{'='*62}\n[{s}] {len(seg_s)}점\n{'='*62}")

        imgpts = np.array([[q[1], q[2]] for q in seg_s])
        galvo = np.array(groups[s])
        pairs, cal = f"pairs_{s}.csv", f"calib_{s}.json"

        with open(pairs, "w") as f:
            f.write(f"# size,{W},{H}\n")
            f.write("galvo_x,galvo_y,img_x,img_y\n")
            for (gx, gy), (ix, iy) in zip(galvo, imgpts):
                f.write(f"{gx:.1f},{gy:.1f},{ix:.3f},{iy:.3f}\n")
        print(f"[i] {pairs} 저장 ({len(imgpts)}점)")

        v = cv2.imread(plate)
        for k, (ix, iy) in enumerate(imgpts):
            c = (int(ix), int(iy))
            cv2.circle(v, c, 12, (0, 255, 0), 2)
            cv2.putText(v, str(k), (c[0] + 12, c[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.rectangle(v, (int(imgpts[:, 0].min()), int(imgpts[:, 1].min())),
                      (int(imgpts[:, 0].max()), int(imgpts[:, 1].max())),
                      (0, 200, 255), 2)
        rev = f"review_{s}.png"
        cv2.imwrite(rev, v)
        print(f"[i] {rev} 저장 — 0번이 첫 격자점인지, 주황 사각형이 대상을 덮는지 확인")

        gm = grid_mm[s]
        if pitch_mm[s] is not None:
            gm = pitch_mm[s] * (a.n - 1)
            print(f"[i] --pitch-mm {pitch_mm[s]}mm × {a.n-1} = 행 전체 폭 {gm}mm")
        mmpp = mm_per_px_from_top_row(imgpts, a.n, gm)
        if mmpp:
            print(f"[i] mm_per_px = {mmpp:.4f}  (최상단 행 {gm}mm 기준)")
        else:
            print("[i] mm 실측 미지정 — px 단위로만 리포트 "
                  "(firstlight.py gridall --surface %s 로 재서 넣을 것)" % s)
        coverage_report(imgpts, W, H, mmpp, multi)

        fit(pairs, cal, mmpp)

    print(f"\n{'='*62}")
    if multi:
        print("[i] 등록 — 두 표면을 **한 parts.yaml** 에 모은다:")
        print(f"    python3 register.py {plate} --edit parts.yaml")
        print("    ⚠️ 목업은 이 배경판에 '평판'만 찍혀 있다. 목업을 놓고 다시 찍은")
        print("       영상에서 프레임을 뽑아 그 이미지로 등록할 것 (LAB-DAY5 §3.2)")
    else:
        parts = f"parts_{a.surface}.yaml"
        print(f"다음: python3 register.py {plate} --edit {parts}")
        if a.register:
            if not os.path.exists("register.py"):
                print("[!] register.py 없음")
                return
            subprocess.run([sys.executable, "register.py", plate, "--edit", parts])


if __name__ == "__main__":
    main()
