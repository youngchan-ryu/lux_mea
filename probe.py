"""Check what OpenCV actually reads out of a video file.

QuickTime tone-maps HDR footage on screen; OpenCV hands back the raw transfer
function. "Looks fine but nothing is detected" is usually this.

    python3 probe.py IMG_4250.mov"""
from __future__ import annotations
import argparse
import sys

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--t", type=float, nargs="*", default=None,
                    help="확인할 시각(초). 생략하면 영상 전체에서 8개 균등 샘플")
    ap.add_argument("--green", action="store_true")
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        sys.exit(f"[!] 영상을 못 엶: {a.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    cc = "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4))
    print(f"[i] {W}x{H}  {fps:.2f}fps  {nf}프레임 ({nf/fps:.1f}초)  codec={cc!r}")

    idxs = np.linspace(0, max(nf - 1, 0), 31).astype(int)
    frames = []
    for k in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(k))
        ok, fr = cap.read()
        if ok:
            frames.append(fr)
    if not frames:
        sys.exit("[!] 프레임을 하나도 못 읽음")
    bg = np.median(np.stack(frames), axis=0).astype(np.int16)
    gb = cv2.cvtColor(np.clip(bg, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    print(f"[i] 배경 회색값  평균 {gb.mean():.0f}  중앙 {np.median(gb):.0f}  "
          f"p99 {np.percentile(gb,99):.0f}  최대 {gb.max()}")
    if gb.mean() > 170:
        print("  ⚠️ 배경이 너무 밝다 — 노출을 더 낮출 것 (목표 평균 70~110)")
    if (gb >= 250).mean() > 0.01:
        print(f"  ⚠️ 포화 픽셀 {(gb>=250).mean()*100:.1f}% — 클리핑되면 색 우세가 소멸한다")
    cv2.imwrite("probe_bg.png", np.clip(bg, 0, 255).astype(np.uint8))

    ts = a.t if a.t else list(np.linspace(0, max(nf - 1, 0) / fps, 8))
    print(f"\n{'t(s)':>7} {'배경대비 최대응답':>16} {'위치':>16} {'코어 BGR':>18} "
          f"{'blob면적':>9}")
    print("-" * 74)
    for i, t in enumerate(ts):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, fr = cap.read()
        if not ok:
            print(f"{t:7.2f}  읽기 실패")
            continue
        d = fr.astype(np.int16) - bg
        b, g, r = d[:, :, 0], d[:, :, 1], d[:, :, 2]
        resp = (g - np.maximum(r, b)) if a.green else (r - np.maximum(g, b))
        resp = cv2.GaussianBlur(np.clip(resp, 0, 255).astype(np.uint8), (5, 5), 0)
        _, mx, _, loc = cv2.minMaxLoc(resp)
        x, y = loc
        core = fr[max(y-2,0):y+3, max(x-2,0):x+3].reshape(-1, 3).max(0)
        area = int((resp > max(mx * 0.4, 6)).sum())
        print(f"{t:7.2f} {mx:16.0f} {str(loc):>16} {str(core):>18} {area:9d}")

        cv2.imwrite(f"probe_{i:02d}_raw.png", fr)
        hm = cv2.applyColorMap(cv2.normalize(resp, None, 0, 255,
                                             cv2.NORM_MINMAX).astype(np.uint8),
                               cv2.COLORMAP_INFERNO)
        cv2.circle(hm, loc, 18, (255, 255, 255), 2)
        cv2.imwrite(f"probe_{i:02d}_resp.png", hm)

    cap.release()
    print("""
[판정 기준]
  · 최대응답이 40 이상인 프레임이 여럿 → 검출은 가능. 임계/세그먼트 설정 문제
  · 최대응답이 20 미만 → 신호 자체가 약함. 아래 순서로 해결
      1) probe_00_raw.png 를 열어 QuickTime 화면과 비교
         ▸ 색이 뿌옇거나 대비가 다르면 **HDR 디코딩 문제**
           ffmpeg -i 원본.MOV -vf format=yuv420p -c:v libx264 -crf 18 grid_sdr.mp4
           로 SDR 변환 후 다시 시도
         ▸ 같아 보이면 촬영 조건 문제 → 2)
      2) 카메라를 대상 쪽으로 **가깝게**. 점이 화면에서 커질수록 검출이 쉽다
      3) 실내등 소등, 노출 잠금 후 하향
  · blob 면적이 5 px 미만 → 점이 거의 1픽셀. 거리가 멀거나 화각이 넓다""")


if __name__ == "__main__":
    main()
