"""Hardware bring-up and calibration capture.

    python3 firstlight.py detect         is the DAC there
    python3 firstlight.py square         does it draw
    python3 firstlight.py park | blink   safety checks
    python3 firstlight.py aim            mark out each surface with the beam
    python3 firstlight.py grid           capture calibration points
    python3 firstlight.py gridall        show the grid at once, for measuring
    python3 firstlight.py sweep          find the amplitude/pps limits

For two surfaces, run `aim` once, then `grid` per surface into the same
recording -- the second and later runs need --append or they overwrite the log."""
import argparse
import csv
import ctypes
import os
import select
import sys
import termios
import time
import tty
from paths import data_path, ensure_data

DAC_MAX = 4095
CENTERX = 1800
CENTERY = 2688
AMPLITUDE = 700
PPS = 16000
LASER_ON = 255
LASER_OFF = 0
PARK = (100, 100)


class HeliosPoint(ctypes.Structure):
    _fields_ = [("x", ctypes.c_uint16), ("y", ctypes.c_uint16),
                ("r", ctypes.c_uint8), ("g", ctypes.c_uint8),
                ("b", ctypes.c_uint8), ("i", ctypes.c_uint8)]


FLAG_START_IMMEDIATELY = 1
FLAG_SINGLE_MODE = 2
FLAG_DONT_BLOCK = 4


class Helios:
    def __init__(self):
        from laser import find_helios_lib
        lib = find_helios_lib()
        self.dll = ctypes.cdll.LoadLibrary(lib)
        n = self.dll.OpenDevices()
        print(f"[i] 인식된 장치 수: {n}")
        if n < 1:
            sys.exit("[!] Helios를 못 찾음. USB 연결/권한 확인.")
        self.dev = 0

    def write(self, pts, pps=PPS, loop=True):
        n = len(pts)
        arr = (HeliosPoint * n)(*pts)
        flags = FLAG_START_IMMEDIATELY
        if not loop:
            flags |= FLAG_SINGLE_MODE
        for _ in range(200):
            if self.dll.GetStatus(self.dev) == 1:
                break
            time.sleep(0.002)
        self.dll.WriteFrame(self.dev, int(pps), ctypes.c_uint8(flags),
                            ctypes.pointer(arr), ctypes.c_int(n))

    def park(self):
        """Beam to the dump and laser off. Always finish with this."""
        self.write([pt(PARK[0], PARK[1], on=False)] * 64, pps=1000)
        try:
            self.dll.SetShutter(self.dev, ctypes.c_bool(False))
        except Exception:
            pass

    def close(self):
        self.park()
        time.sleep(0.2)
        self.dll.CloseDevices()


def pt(x, y, on=True):
    v = LASER_ON if on else LASER_OFF
    x = max(0, min(DAC_MAX, int(x)))
    y = max(0, min(DAC_MAX, int(y)))
    return HeliosPoint(x, y, v, v, v, v)


def square(cx=CENTERX, cy=CENTERY, a=AMPLITUDE, dwell=6):
    """Corner dwell keeps galvo inertia from rounding the corners off."""
    corners = [(cx - a, cy - a), (cx + a, cy - a), (cx + a, cy + a), (cx - a, cy + a)]
    pts, steps = [], 40
    for i in range(4):
        x0, y0 = corners[i]
        x1, y1 = corners[(i + 1) % 4]
        pts += [pt(x0, y0)] * dwell
        for s in range(steps):
            t = s / steps
            pts.append(pt(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
    return pts


def rect_frame(cx, cy, ax, ay, steps=30, dwell=5):
    """Rectangle outline with independent half-widths."""
    corners = [(cx - ax, cy - ay), (cx + ax, cy - ay),
               (cx + ax, cy + ay), (cx - ax, cy + ay)]
    pts = []
    for i in range(4):
        x0, y0 = corners[i]
        x1, y1 = corners[(i + 1) % 4]
        pts += [pt(x0, y0)] * dwell
        for s in range(steps):
            t = s / steps
            pts.append(pt(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
    return pts


def circle(cx=CENTERX, cy=CENTERY, r=AMPLITUDE, n=180):
    import math
    return [pt(cx + r * math.cos(2 * math.pi * i / n),
               cy + r * math.sin(2 * math.pi * i / n)) for i in range(n)]


PLAN_FIELDS = ["surface", "cx", "cy", "amp_x", "amp_y"]


def load_plan(path):
    """surfaces.csv -> {surface: [cx, cy, amp_x, amp_y]}."""
    plan = {}
    if not os.path.exists(path):
        return plan
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                plan[row["surface"]] = [int(float(row[k])) for k in PLAN_FIELDS[1:]]
            except (KeyError, ValueError):
                continue
    return plan


def save_plan(path, plan):
    ensure_data()
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(PLAN_FIELDS)
        for s, v in plan.items():
            w.writerow([s] + list(v))
    print(f"[i] {path} 저장 ({len(plan)}개 표면)")


def clamp_region(cx, cy, ax, ay):
    """Keep a rectangle inside the DAC range."""
    ax = max(20, min(ax, DAC_MAX // 2))
    ay = max(20, min(ay, DAC_MAX // 2))
    cx = max(ax, min(cx, DAC_MAX - ax))
    cy = max(ay, min(cy, DAC_MAX - ay))
    return cx, cy, ax, ay


class RawKeys:
    """Non-blocking terminal keys. cbreak, so Ctrl+C still works."""

    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def get(self, timeout=0.03):
        r, _, _ = select.select([sys.stdin], [], [], timeout)
        return sys.stdin.read(1) if r else None


def cmd_detect(h, a=None):
    print("[i] 장치 인식 OK. 상태:", h.dll.GetStatus(h.dev))

def cmd_on(h, a=None):
    """Park the beam at centre and leave it on."""
    print("[i] 레이저 center ON. Ctrl+C로 종료.")
    while True:
        h.write([pt(CENTERX, CENTERY, on=True)] * 64)
        time.sleep(0.05)

def cmd_square(h, a=None):
    print("[i] 사각형 출력 중. Ctrl+C로 종료.")
    while True:
        h.write(square())
        time.sleep(0.05)


def cmd_circle(h, a=None):
    print("[i] 원 출력 중. Ctrl+C로 종료.")
    while True:
        h.write(circle())
        time.sleep(0.05)


def cmd_park(h, a=None):
    h.park()
    print("[i] 파킹 완료 (레이저 OFF).")


def cmd_blink(h, a=None):
    """Blanking check: the beam should blink without the galvo moving."""
    print("[i] 1초 주기 블랭킹. MOSFET 배선 검증용. Ctrl+C로 종료.")
    while True:
        h.write([pt(CENTERX, CENTERY, on=True)] * 64, pps=1000)
        time.sleep(1)
        h.write([pt(CENTERX, CENTERY, on=False)] * 64, pps=1000)
        time.sleep(1)


AIM_HELP = """
  1~9   표면 선택            0     전체 표면 동시 표시 (사정권 확인용)
  w a s d   중심 이동         f     이동/크기 스텝 전환 (미세 20 ↔ 거친 100)
  [ ]   전체 축소/확대        , .   가로만 축소/확대      - =   세로만 축소/확대
  p     현재 값 출력          v     surfaces.csv 저장     q     저장 후 종료
"""


def cmd_aim(h, a):
    """Mark out each surface by steering the beam around its edges.

    Picking the region on a camera image would need the image->galvo map, which is
    exactly what has not been measured yet. Drawing the rectangle with the beam
    itself sidesteps that, and shows immediately whether both surfaces are in
    range at all.
    """
    surfaces = [s.strip() for s in a.surfaces.split(",") if s.strip()]
    if not surfaces:
        sys.exit("[!] --surfaces poster,mockup 형식으로 표면 이름을 줄 것")

    plan = load_plan(data_path(a.plan))
    for s in surfaces:
        plan.setdefault(s, [CENTERX, CENTERY, AMPLITUDE // 2, AMPLITUDE // 2])

    idx, step, show_all = 0, 20, False
    print(__doc__ if False else "")
    print(f"[i] 표면 {surfaces}  계획파일 {a.plan}")
    print(AIM_HELP)

    def status():
        s = surfaces[idx]
        cx, cy, ax, ay = plan[s]
        print(f"  [{s}] cx={cx:4d} cy={cy:4d} amp_x={ax:4d} amp_y={ay:4d}"
              f"  (x {cx-ax}~{cx+ax}, y {cy-ay}~{cy+ay})  step={step}"
              f"{'  [전체표시]' if show_all else ''}")

    status()
    with RawKeys() as keys:
        while True:
            if show_all:
                pts = []
                for s in surfaces:
                    pts += rect_frame(*plan[s])
            else:
                pts = rect_frame(*plan[surfaces[idx]])
            h.write(pts, pps=PPS)

            k = keys.get(0.03)
            if not k:
                continue
            s = surfaces[idx]
            cx, cy, ax, ay = plan[s]

            if k == "q":
                break
            elif k == "0":
                show_all = not show_all; status()
            elif k.isdigit() and 1 <= int(k) <= len(surfaces):
                idx = int(k) - 1; show_all = False; status()
            elif k == "f":
                step = 100 if step == 20 else 20; status()
            elif k in "wasd":
                if k == "w": cy += step
                elif k == "s": cy -= step
                elif k == "a": cx -= step
                elif k == "d": cx += step
                plan[s] = list(clamp_region(cx, cy, ax, ay)); status()
            elif k in "[],.-=":
                if k == "[": ax -= step; ay -= step
                elif k == "]": ax += step; ay += step
                elif k == ",": ax -= step
                elif k == ".": ax += step
                elif k == "-": ay -= step
                elif k == "=": ay += step
                plan[s] = list(clamp_region(cx, cy, ax, ay)); status()
            elif k == "p":
                status()
            elif k == "v":
                save_plan(data_path(a.plan), plan)

    save_plan(data_path(a.plan), plan)
    print("[i] 이제: python3 firstlight.py grid --surface <이름>  "
          "(두 번째부터 --append)")


def resolve_region(a):
    """Region for the grid: explicit flags, else surfaces.csv, else defaults."""
    plan = load_plan(data_path(a.plan))
    if a.surface and a.surface in plan:
        cx, cy, ax, ay = plan[a.surface]
        src = f"{a.plan}[{a.surface}]"
    else:
        cx, cy, ax, ay = CENTERX, CENTERY, AMPLITUDE, AMPLITUDE
        src = "기본값(CENTER/AMPLITUDE)"
        if a.surface:
            print(f"[!] {a.plan}에 '{a.surface}' 없음 — 먼저 "
                  f"'python3 firstlight.py aim --surfaces {a.surface}' 를 권한다")
    if a.cx is not None: cx, src = a.cx, "CLI"
    if a.cy is not None: cy, src = a.cy, "CLI"
    if a.amp is not None: ax = ay = a.amp; src = "CLI"
    if a.amp_y is not None: ay, src = a.amp_y, "CLI"
    cx, cy, ax, ay = clamp_region(cx, cy, ax, ay)
    print(f"[i] 영역 출처={src}  cx={cx} cy={cy} amp_x={ax} amp_y={ay}")
    return cx, cy, ax, ay


def cmd_grid(h, a):
    """Step through the grid, pausing on each point, and log the coordinates.

    Run once per surface inside a single recording. Anything after the first needs
    --append; without it the previous log is overwritten and the shoot is wasted.
    """
    n, dwell_s, gap_s = a.n, a.dwell, a.gap
    surface = a.surface or "default"
    cx, cy, ax, ay = resolve_region(a)
    lo_x, hi_x = cx - ax, cx + ax
    lo_y, hi_y = cy - ay, cy + ay
    step_x = (hi_x - lo_x) // (n - 1)
    step_y = (hi_y - lo_y) // (n - 1)
    pairs = []
    print(f"[i] 표면 '{surface}'  {n}x{n} 격자, 점당 {dwell_s}초 + 간격 {gap_s}초 "
          f"(총 {n*n*(dwell_s+gap_s):.0f}초).")
    print("[i] 한 점에 계속 머무르므로 duty 100% — 동시표시 대비 25배 밝다.")
    print("[i] 실내등을 낮추고, 카메라는 삼각대 고정 + AE/AF 잠금.")
    if a.append:
        print(f"[i] --append: {a.log} 에 이어 쓴다 (녹화를 멈추지 말 것)")
    else:
        print(f"[i] {a.log} 를 새로 만든다. 두 번째 표면부터는 --append 를 줄 것")
    print("[i] 녹화 시작 → Enter (지연은 자동 보정됨)")
    input()
    print("[i] 동기용 블랭킹 2초...")
    for _ in range(40):
        h.write([pt(PARK[0], PARK[1], on=False)] * 64, pps=1000)
        time.sleep(0.05)
    t0 = time.time()
    for iy in range(n):
        for ix in range(n):
            x, y = lo_x + ix * step_x, lo_y + iy * step_y
            ts = time.time() - t0
            pairs.append((ts, x, y))
            print(f"  t={ts:6.2f}s  galvo=({x},{y})")
            end = time.time() + dwell_s
            while time.time() < end:
                h.write([pt(x, y)] * 64, pps=1000)
                time.sleep(0.05)
            end = time.time() + gap_s
            while time.time() < end:
                h.write([pt(x, y, on=False)] * 64, pps=1000)
                time.sleep(0.03)

    ensure_data()
    exists = os.path.exists(data_path(a.log))
    with open(data_path(a.log), "a" if a.append else "w") as f:
        if not (a.append and exists):
            f.write("t_sec,galvo_x,galvo_y,surface\n")
        for ts, x, y in pairs:
            f.write(f"{ts:.3f},{x},{y},{surface}\n")
    total = sum(1 for ln in open(data_path(a.log)) if ln[:1].isdigit())
    print(f"[i] {a.log} 저장 — 이번 {len(pairs)}점, 누적 {total}점")
    print("[i] 다음 표면: 같은 녹화 상태에서 "
          f"'python3 firstlight.py grid --surface <이름> --append'")


def cmd_gridall(h, a):
    """Show the whole grid at once so it can be measured with a ruler.

    What you measure decides the flag: point spacing goes to --pitch-mm, the full
    grid width goes to --grid-mm. They differ by a factor of (n-1).
    """
    n, dwell = a.n, 8
    cx, cy, ax, ay = resolve_region(a)
    lo_x, lo_y = cx - ax, cy - ay
    step_x = (2 * ax) // (n - 1)
    step_y = (2 * ay) // (n - 1)
    pts = []
    for iy in range(n):
        for ix in range(n):
            pts += [pt(lo_x + ix * step_x, lo_y + iy * step_y)] * dwell
    print(f"[i] {n*n}점 동시 표시 ({len(pts)}pt/frame, {20000/len(pts):.0f}Hz)")
    print(f"[i] 인접 점 간격 = x {step_x} units · y {step_y} units")
    print(f"[i] 격자 전체 폭 = x {2*ax} units · y {2*ay} units")
    print("[i] ── 자로 잰 뒤 ─────────────────────────────────")
    print(f"[i]   인접 간격을 쟀으면 →  pipeline.py --pitch-mm <잰값>")
    print(f"[i]   전체 폭을 쟀으면   →  pipeline.py --grid-mm  <잰값>")
    print("[i] Ctrl+C 종료")
    while True:
        h.write(pts, pps=20000)
        time.sleep(0.05)


def cmd_sweep(h, a=None):
    """Amplitude and speed limits. Wobble means the previous value was the limit."""
    import math
    for pps in (8000, 12000, 16000, 20000):
        for a in (400, 800, 1200, 1600):
            print(f"  pps={pps} amp={a}  (3초) — 모서리 흔들림/찌그러짐 관찰")
            end = time.time() + 3
            while time.time() < end:
                h.write(square(a=a), pps=pps)
                time.sleep(0.05)
    print("[i] 안정적이었던 최대 pps/amp를 기록할 것.")


CMDS = {"detect": cmd_detect, "on": cmd_on, "square": cmd_square, "circle": cmd_circle,
        "park": cmd_park, "blink": cmd_blink, "aim": cmd_aim, "grid": cmd_grid,
        "gridall": cmd_gridall, "sweep": cmd_sweep}


def parse_args():
    ap = argparse.ArgumentParser(
        description="Helios 갈보 테스트 · 표면 조준 · 캘리브 격자 수집")
    ap.add_argument("cmd", nargs="?", default="detect", choices=list(CMDS))
    ap.add_argument("--surfaces", default="poster,mockup",
                    help="aim: 조준할 표면 이름들 (쉼표 구분)")
    ap.add_argument("--surface", default=None,
                    help="grid: 이번 sweep의 표면 이름. surfaces.csv에서 영역을 읽고 "
                         "grid_log.csv의 surface 컬럼에 기록된다")
    ap.add_argument("--plan", default="surfaces.csv", help="aim 결과 파일")
    ap.add_argument("--log", default="grid_log.csv")
    ap.add_argument("--append", action="store_true",
                    help="grid: 로그를 덮어쓰지 않고 이어 쓴다 (두 번째 표면부터 필수)")
    ap.add_argument("--cx", type=int, default=None)
    ap.add_argument("--cy", type=int, default=None)
    ap.add_argument("--amp", type=int, default=None, help="가로세로 반폭 (동시 지정)")
    ap.add_argument("--amp-y", dest="amp_y", type=int, default=None,
                    help="세로 반폭만 따로 지정")
    ap.add_argument("--n", type=int, default=5, help="격자 한 변의 점 수")
    ap.add_argument("--dwell", type=float, default=1.0, help="점당 체류 시간(s)")
    ap.add_argument("--gap", type=float, default=0.30, help="점 사이 블랭킹(s)")
    return ap.parse_args()


if __name__ == "__main__":
    a = parse_args()
    h = Helios()
    try:
        CMDS[a.cmd](h, a)
    except KeyboardInterrupt:
        print("\n[i] 중단됨.")
    finally:
        h.close()
        print("[i] 파킹 후 종료.")
