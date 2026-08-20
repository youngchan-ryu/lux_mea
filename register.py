"""Click parts on a photo to build parts.yaml.

    python3 register.py plate.jpg --edit parts.yaml --surface poster

    c r x e o l   shape: circle rect crosshair ellipse roundrect underline
    click         place at default size
    click + drag  place and size in one gesture
    [ ]           resize the part just placed
    u             undo      s  save and quit      ESC  quit without saving

Anchors are stored normalised, so they survive a change of capture resolution.
Parts linked with `link:` light up together, whichever one is spoken."""
from __future__ import annotations
import sys
import cv2
import yaml

parts = {}
R_DEFAULT = 0.03
W_DEFAULT, H_DEFAULT = 0.05, 0.02
RX_DEFAULT, RY_DEFAULT = 0.04, 0.02
MODE_KEYS = {"c": "circle", "r": "rect", "x": "crosshair",
            "e": "ellipse", "o": "roundrect", "l": "underline"}

state = {"mode": "circle", "dragging": False, "start": None, "cur": None,
         "last": None, "surface": None}


def default_shape(mode):
    if mode in ("rect", "roundrect"):
        return {"type": mode, "w_norm": W_DEFAULT, "h_norm": H_DEFAULT}
    if mode == "ellipse":
        return {"type": "ellipse", "rx_norm": RX_DEFAULT, "ry_norm": RY_DEFAULT}
    if mode == "underline":
        return {"type": "underline", "w_norm": W_DEFAULT}
    return {"type": mode, "r_norm": R_DEFAULT}


def shape_from_drag(start, cur, w, h, mode):
    """Drag vector -> shape spec, clamped to a minimum size."""
    x0, y0 = start
    x1, y1 = cur
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    if mode in ("rect", "roundrect"):
        return {"type": mode, "w_norm": round(max(dx * 2 / w, 0.01), 4),
                "h_norm": round(max(dy * 2 / h, 0.01), 4)}
    if mode == "ellipse":
        return {"type": "ellipse", "rx_norm": round(max(dx / w, 0.01), 4),
                "ry_norm": round(max(dy / h, 0.01), 4)}
    if mode == "underline":
        return {"type": "underline", "w_norm": round(max(dx * 2 / w, 0.01), 4)}
    r_norm = max((dx ** 2 + dy ** 2) ** 0.5 / w, 0.01)
    return {"type": mode, "r_norm": round(r_norm, 4)}


def draw_shape(img, anchor, shape, w, h, color):
    ax, ay = anchor
    cx, cy = int(ax * w), int(ay * h)
    t = shape.get("type", "circle")
    if t in ("rect", "roundrect"):
        rw, rh = int(shape["w_norm"] * w / 2), int(shape["h_norm"] * h / 2)
        cv2.rectangle(img, (cx - rw, cy - rh), (cx + rw, cy + rh), color, 2, cv2.LINE_AA)
    elif t == "ellipse":
        rx, ry = max(int(shape["rx_norm"] * w), 1), max(int(shape["ry_norm"] * h), 1)
        cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, color, 2, cv2.LINE_AA)
    elif t == "underline":
        hw = int(shape["w_norm"] * w / 2)
        cv2.line(img, (cx - hw, cy), (cx + hw, cy), color, 2, cv2.LINE_AA)
    else:
        r = int(shape.get("r_norm", R_DEFAULT) * w)
        cv2.circle(img, (cx, cy), r, color, 2, cv2.LINE_AA)
        if t == "crosshair":
            cv2.drawMarker(img, (cx, cy), color, cv2.MARKER_CROSS, max(r, 8), 1)
    cv2.circle(img, (cx, cy), 3, color, -1)


def redraw(base):
    img = base.copy()
    h, w = img.shape[:2]
    for pid, p in parts.items():
        same = (p.get("surface") == state["surface"])
        col = (60, 255, 60) if same else (110, 110, 110)
        draw_shape(img, p["anchor"], p["shape"], w, h, col)
        ax, ay = p["anchor"]
        c = (int(ax * w), int(ay * h))
        label = pid if same else f"{pid}({p.get('surface') or '-'})"
        cv2.putText(img, label, (c[0] + 8, c[1] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, col, 1, cv2.LINE_AA)

    if state["dragging"] and state["start"] and state["cur"]:
        x0, y0 = state["start"]
        preview = shape_from_drag(state["start"], state["cur"], w, h, state["mode"])
        draw_shape(img, (x0 / w, y0 / h), preview, w, h, (0, 220, 255))

    sfx = f"surface: {state['surface']}   " if state["surface"] else ""
    cv2.putText(img, f"{sfx}mode: {state['mode']}  (c=circle r=rect x=crosshair "
                     f"e=ellipse o=roundrect l=underline)",
                (14, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1, cv2.LINE_AA)
    n_here = sum(1 for p in parts.values() if p.get("surface") == state["surface"])
    cv2.putText(img, f"parts: {n_here}/{len(parts)}   "
                     f"[s]save [u]undo [ ][ ]resize [ESC]quit",
                (14, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    return img


def main():
    if len(sys.argv) < 2:
        print(__doc__); return
    path = sys.argv[1]
    out = "parts.yaml"
    if "--edit" in sys.argv:
        out = sys.argv[sys.argv.index("--edit") + 1]
        try:
            parts.update(yaml.safe_load(open(out))["parts"])
            print(f"[i] 기존 {len(parts)}개 로드")
        except Exception:
            pass
    if "--surface" in sys.argv:
        state["surface"] = sys.argv[sys.argv.index("--surface") + 1]
        print(f"[i] 이번에 찍는 부위는 surface: {state['surface']} 로 기록된다")
        print("    (app.py 가 이 태그를 보고 calib_<surface>.json 을 물린다)")
    else:
        tagged = {p.get("surface") for p in parts.values() if p.get("surface")}
        if tagged:
            print(f"[!] 기존 파일에 표면 태그 {sorted(tagged)} 가 있는데 --surface 를 안 줬다.")
            print("    지금 찍는 부위에는 태그가 안 붙어 app.py 가 이중표면으로 못 읽는다.")

    base = cv2.imread(path)
    if base is None:
        print(f"[!] 이미지를 못 읽음: {path}"); return
    h, w = base.shape[:2]
    if w > 1400:
        base = cv2.resize(base, (1400, int(h * 1400 / w))); h, w = base.shape[:2]

    win = "register"

    def on_mouse(ev, x, y, flags, param):
        if ev == cv2.EVENT_LBUTTONDOWN:
            state.update(dragging=True, start=(x, y), cur=(x, y))
        elif ev == cv2.EVENT_MOUSEMOVE and state["dragging"]:
            state["cur"] = (x, y)
            cv2.imshow(win, redraw(base))
        elif ev == cv2.EVENT_LBUTTONUP and state["dragging"]:
            x0, y0 = state["start"]
            dist = ((x - x0) ** 2 + (y - y0) ** 2) ** 0.5
            state["dragging"] = False
            shape = (shape_from_drag((x0, y0), (x, y), w, h, state["mode"])
                     if dist >= 4 else default_shape(state["mode"]))
            cv2.imshow(win, redraw(base))
            name = input("  부위 id (영문, 빈 줄=취소): ").strip()
            if not name:
                return
            al = input("  별칭 (쉼표 구분, 한글·영문·음차 4~6개): ").strip()
            others = [p for p in parts if p != name]
            lk = ""
            if others:
                print(f"    (등록된 부위: {', '.join(others)})")
                lk = input("  동시에 짚을 부위 id (쉼표 구분, 빈 줄=없음): ").strip()
            entry = {}
            if state["surface"]:
                entry["surface"] = state["surface"]
            link = [s.strip() for s in lk.split(",") if s.strip() and s.strip() in parts]
            if link:
                entry["link"] = link
            entry.update({
                "aliases": [a.strip() for a in al.split(",") if a.strip()],
                "anchor": [round(x0 / w, 4), round(y0 / h, 4)],
                "shape": shape,
            })
            parts[name] = entry
            state["last"] = name
            tag = f" surface={state['surface']}" if state["surface"] else ""
            print(f"  → {name} 등록 ({shape['type']}{tag}) ({len(parts)}개)\n")

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    print(__doc__)
    while True:
        cv2.imshow(win, redraw(base))
        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            print("[i] 저장 없이 종료"); break
        if k == ord("s"):
            with open(out, "w") as f:
                yaml.safe_dump({"object": "mockup_v1", "parts": parts}, f,
                               allow_unicode=True, sort_keys=False)
            print(f"[i] {out} 저장 ({len(parts)}개 부위)"); break
        if k == ord("u") and parts:
            parts.pop(next(reversed(parts))); print(f"[i] 취소 ({len(parts)}개)")
        if chr(k) in MODE_KEYS:
            state["mode"] = MODE_KEYS[chr(k)]
            print(f"  [i] 모드: {state['mode']}")
        if k in (ord("["), ord("]")) and state["last"] in parts:
            d = -0.005 if k == ord("[") else 0.005
            sh = parts[state["last"]]["shape"]
            t = sh.get("type")
            if t in ("rect", "roundrect"):
                sh["w_norm"] = max(0.01, round(sh["w_norm"] + d, 4))
                sh["h_norm"] = max(0.01, round(sh["h_norm"] + d, 4))
            elif t == "ellipse":
                sh["rx_norm"] = max(0.01, round(sh["rx_norm"] + d, 4))
                sh["ry_norm"] = max(0.01, round(sh["ry_norm"] + d, 4))
            elif t == "underline":
                sh["w_norm"] = max(0.01, round(sh["w_norm"] + d, 4))
            else:
                sh["r_norm"] = max(0.01, round(sh.get("r_norm", R_DEFAULT) + d, 4))
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
