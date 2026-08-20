"""Orchestrator: microphone -> matcher -> shapes -> galvo, plus the HUD.

    python3 app.py --sim                 simulator, no hardware
    python3 app.py                       live
    python3 app.py --no-voice            hotkeys only (1-9), useful while filming

Surfaces come from the `surface:` field in parts.yaml. If any part has one,
each surface is driven through its own calib_<surface>.json.

Keys: 1-9 pick a part, space parks the beam, q quits."""
from __future__ import annotations
import argparse
import os
import threading
import time
import numpy as np
import cv2
import yaml

import shapes as SH
from laser import make_output
from match import load_parts, load_links, shared_aliases, Matcher
from paths import data_path, data_glob, ensure_data

STATE = {"part": None, "parts": [], "text": "", "t_speech": 0.0, "lat": 0.0,
         "paused": False, "running": True}
LOG = []

_FONT = None

def build_prompt(spec, max_chars=300):
    """Whisper initial_prompt built from the part aliases.

    A short sentence with the vocabulary inside it works better than a bare comma
    list, and Whisper truncates the prompt from the front past ~224 tokens, so
    keep it short enough that it survives intact.
    """
    seen, terms = set(), []
    for pid, p in spec.items():
        for name in [pid, *p.get("aliases", [])]:
            name = name.strip()
            if name and name not in seen:
                seen.add(name)
                terms.append(name)
    vocab = ", ".join(terms)
    if len(vocab) > max_chars:
        vocab = vocab[:max_chars].rsplit(",", 1)[0]
    return f"이것은 장비 부품을 하나씩 가리키며 설명하는 발표입니다. 등장하는 부품: {vocab}."


def _font(size=22):
    global _FONT
    if _FONT is None:
        from PIL import ImageFont
        for p in ("/System/Library/Fonts/AppleSDGothicNeo.ttc",
                  "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
                  "/Library/Fonts/AppleGothic.ttf"):
            try:
                _FONT = ImageFont.truetype(p, size); break
            except Exception:
                continue
        if _FONT is None:
            _FONT = False
    return _FONT


def draw_text(img, text, org, color=(225, 225, 225), size=22):
    f = _font(size)
    if not f:
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, size / 34,
                    color, 1, cv2.LINE_AA)
        return img
    from PIL import Image, ImageDraw, ImageFont
    if f.size != size:
        f = ImageFont.truetype(f.path, size)
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(pil).text(org, text, font=f, fill=tuple(color[::-1]))
    img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return img


def build_shape(spec, part_id, img_w, img_h):
    p = spec[part_id]
    ax, ay = p["anchor"]
    cx, cy = ax * img_w, ay * img_h
    sh = dict(p.get("shape", {"type": "circle"}))
    if "r_norm" in sh:
        sh["r"] = sh.pop("r_norm") * img_w
    if "w_norm" in sh:
        sh["w"] = sh.pop("w_norm") * img_w
    if "h_norm" in sh:
        sh["h"] = sh.pop("h_norm") * img_h
    if "rx_norm" in sh:
        sh["rx"] = sh.pop("rx_norm") * img_w
    if "ry_norm" in sh:
        sh["ry"] = sh.pop("ry_norm") * img_h
    if "radius_norm" in sh:
        sh["radius"] = sh.pop("radius_norm") * img_w
    return SH.make(sh, cx, cy)


def check_anchors(spec, out, W, H):
    """Draw every part once at startup and report anything out of bounds.

    A shape whose points all clamp to one edge of the fence collapses into a line,
    which is easy to miss while presenting. Outside the fence is fatal; outside the
    calibration grid only means the polynomial is extrapolating.
    """
    import json
    grids = {}
    for s in out.profiles:
        path = data_path(f"calib_{s}.json") if s else None
        try:
            b = np.array(json.load(open(path)).get("galvo_bounds"), float)
            grids[s] = (b[0], b[1])
        except Exception:
            grids[s] = None

    bad, warn = [], []
    for pid in spec:
        surf = spec[pid].get("surface")
        i2g, fence = out._profile(surf)
        if i2g is None:
            continue
        g = np.asarray(i2g(build_shape(spec, pid, W, H)), float)
        n_out = fence.outside(g)
        if n_out:
            bad.append((pid, surf, n_out, len(g), g))
            continue
        gr = grids.get(surf if surf in grids else out.default)
        if gr is not None and ((g < gr[0]).any() or (g > gr[1]).any()):
            warn.append((pid, surf))

    if bad:
        print("\n[!] 펜스를 벗어나는 부위 — 도형이 잘리거나 선으로 뭉친다:")
        for pid, surf, n, tot, g in bad:
            f = out._profile(surf)[1]
            print(f"    {pid} [{surf or '-'}] {n}/{tot}점 밖 · "
                  f"galvo x {g[:,0].min():.0f}~{g[:,0].max():.0f} "
                  f"y {g[:,1].min():.0f}~{g[:,1].max():.0f}")
            print(f"      허용 x {f.lo[0]:.0f}~{f.hi[0]:.0f} y {f.lo[1]:.0f}~{f.hi[1]:.0f}")
        print("    → anchor 를 안쪽으로 옮기거나, 그 영역까지 덮도록 재촬영할 것")
    if warn:
        print(f"\n[!] 캘리브 격자 밖 부위 (외삽 — 조준이 부정확할 수 있다): "
              f"{[f'{p}[{s}]' for p, s in warn]}")
    if not bad and not warn:
        print("[i] 모든 부위가 펜스·격자 안에 있다 ✅")
    return bad


def hud(spec, part_id, text, lat, frame=None, bg=None, size=(1080, 720),
        group=None):
    """HUD background: live camera frame, else the --bg plate, else black."""
    w, h = size
    group = group or ([part_id] if part_id else [])
    if frame is not None:
        img = cv2.resize(frame, size)
    elif bg is not None:
        img = cv2.resize(bg, size)
    else:
        img = np.full((h, w, 3), 20, np.uint8)
    for pid, p in spec.items():
        ax, ay = p["anchor"]
        c = (int(ax * w), int(ay * h))
        hit = (pid in group)
        cv2.circle(img, c, 26 if hit else 8, (60, 255, 60) if hit else (90, 90, 90),
                   3 if hit else 1, cv2.LINE_AA)
        cv2.putText(img, pid, (c[0] + 16, c[1] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (60, 255, 60) if hit else (110, 110, 110), 1, cv2.LINE_AA)
    cv2.rectangle(img, (0, h - 74), (w, h), (12, 12, 12), -1)
    draw_text(img, text[:70], (18, h - 62), (225, 225, 225), 24)
    if len(group) > 1:
        label = " + ".join(f"{p}[{spec[p].get('surface') or '-'}]" for p in group)
    else:
        surf = spec.get(part_id, {}).get("surface") if part_id else None
        label = (part_id or "-") + (f" [{surf}]" if surf else "")
    tag = f"target: {label}    latency: {lat*1000:.0f} ms"
    draw_text(img, tag, (18, h - 30), (60, 255, 60), 20)
    return img


def speech_thread(matcher, engine, prompt, model, stop_event):
    from speech import cmd_live

    def on_text(txt):
        t0 = time.monotonic()
        pid = matcher.update(txt)
        STATE.update(text=txt, part=pid, parts=list(matcher.current_all),
                     lat=time.monotonic() - t0, t_speech=time.monotonic())
        LOG.append({"t": time.time(), "text": txt, "part": pid,
                    "parts": list(matcher.current_all)})
    try:
        cmd_live(engine, on_text=on_text, prompt=prompt, model=model, stop_event=stop_event)
    except Exception as e:
        print(f"[!] 음성 스레드 종료: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--no-voice", action="store_true")
    ap.add_argument("--surface", default=None,
                    help="표면 프리셋 (예: mockup, poster). pipeline.py 산출물 이름 규칙"
                         "(parts_<surface>.yaml · calib_<surface>.json · plate_<surface>.jpg)"
                         "을 그대로 따라 --parts/--calib/--bg 기본값을 채운다. "
                         "이 모드 저 모드 오갈 때 --surface 하나만 바꾸면 된다. "
                         "--parts/--calib/--bg를 따로 주면 그쪽이 우선한다.")
    ap.add_argument("--parts", default=None)
    ap.add_argument("--calib", default=None)
    ap.add_argument("--engine", default="auto")
    ap.add_argument("--model", default=None,
                    help="Whisper 모델. large/medium/small 축약어 또는 전체 모델 id. "
                         "기본은 large-v3-turbo. 느리면 medium→small로 낮출 것 — "
                         "python3 bench_asr.py run --parts <parts.yaml>로 적중률·지연 비교 가능")
    ap.add_argument("--tie-group", dest="tie_group", action="store_true",
                    help="link: 외에 '별칭 점수 동점'인 부위도 함께 표시. 편하지만 별칭을 "
                         "고칠 때마다 동작이 변한다 — 되도록 link: 를 쓸 것")
    ap.add_argument("--cam", type=int, default=-1)
    ap.add_argument("--bg", default=None,
                    help="배경 이미지 (목업/포스터 사진). --sim 시뮬레이터 배경으로, "
                         "카메라 프레임이 없을 때(--cam 미지정) HUD 배경으로도 쓰인다. "
                         "--surface를 주면 plate_<surface>.jpg가 기본값이 된다")
    a = ap.parse_args()

    sfx = a.surface
    a.parts = a.parts or (f"parts_{sfx}.yaml" if sfx else "parts.yaml")
    a.calib = a.calib or (f"calib_{sfx}.json" if sfx else "calib.json")
    if not a.bg:
        cand = f"plate_{sfx}.jpg" if sfx else "plate.jpg"
        a.bg = cand if os.path.exists(data_path(cand)) else None
    print(f"[i] surface={sfx or '(none)'}  parts={a.parts}  calib={a.calib}  bg={a.bg or '-'}")

    parts_path = data_path(a.parts)
    if not os.path.exists(parts_path):
        raise SystemExit(
            f"[!] {parts_path} 없음.\n"
            f"    data/ 는 추적되지 않는다 — MANIFEST.md 의 셋업 4단계로 만들거나,\n"
            f"    LUXMEA_DATA 를 기존 폴더로 지정할 것.")
    spec = yaml.safe_load(open(parts_path))["parts"]
    ids = list(spec)
    links = load_links(spec)
    matcher = Matcher(load_parts(spec), links=links, tie_group=a.tie_group)

    shown = {tuple(g) for g in links.values() if len(g) > 1}
    if shown:
        print("[i] 동시 표시 그룹 (parts.yaml 의 link:):")
        for g in sorted(shown):
            print("     " + " + ".join(f"{p}[{spec[p].get('surface') or '-'}]" for p in g))
    dup = shared_aliases(spec, links)
    if dup:
        print("[!] 같은 별칭이 여러 부위에 있는데 link 로 안 묶임 — 한쪽만 짚힌다:")
        for al, pids in dup[:8]:
            print(f"     '{al}' → {pids}   (둘 다 짚으려면 link: 로 이을 것)")

    prompt = build_prompt(spec)
    print(f"[i] Whisper 프롬프트 ({len(ids)}개 부위): {prompt}")
    bg = cv2.imread(data_path(a.bg)) if a.bg else None
    if a.bg and bg is None:
        print(f"[!] 배경 이미지를 못 읽음: {a.bg}")

    surfaces = [s for s in dict.fromkeys(p.get("surface") for p in spec.values()) if s]
    untagged = [pid for pid, p in spec.items() if not p.get("surface")]
    if surfaces:
        calib_arg = {s: data_path(f"calib_{s}.json") for s in surfaces}
        print(f"[i] 이중표면 모드 — 표면 {surfaces}")
        if untagged:
            print(f"[!] surface 태그가 없는 부위 {untagged} — '{surfaces[0]}' 로 간주된다")
    else:
        calib_arg = a.calib
        if not os.path.exists(data_path(calib_arg)):
            import glob
            found = data_glob("calib_*.json")
            print(f"[!] {calib_arg} 없음.")
            if found:
                print(f"    있는 캘리브: {found}")
                print("    → 단일표면이면:  --surface <이름>")
                print("    → 이중표면이면:  register.py ... --surface <이름> 으로 "
                      "parts.yaml 의 각 부위에 surface 태그를 붙일 것")
    out = make_output(sim=a.sim,
                      calib_json=(calib_arg if isinstance(calib_arg, dict)
                                  else data_path(calib_arg)), bg=bg)
    cap = cv2.VideoCapture(a.cam) if a.cam >= 0 else None

    W, H = 1080, 720
    import json
    sizes = {}
    for name, path in (calib_arg.items() if isinstance(calib_arg, dict)
                       else [(None, calib_arg)]):
        try:
            sz = json.load(open(data_path(path))).get("img_size")
            if sz:
                sizes[name] = (int(sz[0]), int(sz[1]))
            else:
                print(f"[!] {path}에 img_size 없음")
        except Exception:
            if not a.sim:
                print(f"[!] {path}을 못 읽음 — 실기 조준 불가")
    if sizes:
        uniq = set(sizes.values())
        W, H = next(iter(sizes.values()))
        if len(uniq) > 1:
            print(f"[!] 표면별 img_size 불일치 {sizes} — 같은 촬영에서 나온 게 맞는지 확인. "
                  f"{W}x{H} 를 사용한다")
        else:
            print(f"[i] 캘리브 이미지 좌표계 {W}x{H}")
    else:
        print("[!] img_size 를 못 구함 — 기본 1080x720 사용 (조준이 어긋날 수 있음)")

    check_anchors(spec, out, W, H)

    stop_event = threading.Event()
    voice_thread = None
    if not a.no_voice:
        voice_thread = threading.Thread(
            target=speech_thread, args=(matcher, a.engine, prompt, a.model, stop_event),
            daemon=True)
        voice_thread.start()

    print("핫키: 1~9 부위 · space 파킹 · q 종료")
    print("부위:", {i + 1: p for i, p in enumerate(ids[:9])})
    cur, t_switch = None, 0.0

    try:
        while STATE["running"]:
            want = None if STATE["paused"] else STATE["part"]
            group = [] if STATE["paused"] else [p for p in STATE["parts"] if p in spec]
            if want != cur:
                cur, t_switch = want, time.monotonic()
                if len(group) > 1:
                    n, hz = SH.total_points(*[build_shape(spec, p, W, H) for p in group])
                    print(f"[i] 동시 표시 {group} — {n}점 + 소등, 약 {hz:.0f}Hz")
            if cur is None:
                out.park()
            elif len(group) > 1:
                out.show_many([(build_shape(spec, p, W, H), spec[p].get("surface"))
                               for p in group])
            else:
                pts = build_shape(spec, cur, W, H)
                out.show(pts, spec[cur].get("surface"))

            frame = None
            if cap is not None:
                ok, f = cap.read()
                frame = f if ok else None
            cv2.imshow("HUD", hud(spec, cur, STATE["text"], STATE["lat"], frame, bg,
                                  (1080, 720), group))

            k = cv2.waitKey(30) & 0xFF
            if k == ord("q"):
                break
            elif k == ord(" "):
                STATE["paused"] = not STATE["paused"]
                print("파킹" if STATE["paused"] else "재개")
            elif ord("1") <= k <= ord("9"):
                i = k - ord("1")
                if i < len(ids):
                    STATE.update(part=ids[i], parts=[ids[i]],
                                 text=f"[수동] {ids[i]}", lat=0.0)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        if voice_thread is not None:
            voice_thread.join(timeout=3.0)
            if voice_thread.is_alive():
                print("[!] 음성 스레드가 3초 안에 안 끝남 — 강제 종료 없이 진행 (드물게 segfault 가능)")
        out.park(); out.close()
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        if LOG:
            ensure_data()
            import json
            json.dump(LOG, open(data_path("session_log.json"), "w"),
                      ensure_ascii=False, indent=1)
            print(f"[i] session_log.json 저장 ({len(LOG)}건)")


if __name__ == "__main__":
    main()
