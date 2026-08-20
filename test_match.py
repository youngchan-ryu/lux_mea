"""Regression test for matching and links. No hardware needed.

    python3 test_match.py

Runs against the real alias dictionary in parts.yaml, so editing the dictionary
breaks this before it breaks a demo."""
import yaml
from match import load_parts, load_links, best_match, score_all, Matcher

spec = yaml.safe_load(open("parts.yaml"))["parts"]
parts = load_parts(spec)
links = load_links(spec)

CASES = [    # utterance -> expected part, phrased like real presentation speech
    ("럭스 메아를 소개합니다",                       "title"),
    ("한 손으로 직접 조작하지 않아도 됩니다",         "text1"),
    ("안전성은 조준과 작동이 정해진 규칙을 따릅니다", "text2"),
    ("오차범위는 5mm 이내입니다",                    "text3"),
    ("마지막으로 클래스 기준 부상 위험",              "text4"),
    ("카메라가 좌표 보정을 합니다",                   "camera_poster"),
    ("옆에 위치한 레이저 모듈에서 빛이 나옵니다",     "laser_poster"),
    ("갈보스캐너가 거울의 각도를 바꿉니다",           "galvo_poster"),
    ("DAC가 컴퓨터의 좌표를 받습니다",                "DAC_poster"),
    ("전기회로가 좌표 정보를 전달합니다",             "electronics_poster"),
]

ok = 0
for text, want in CASES:
    m = best_match(text, parts)
    got, sc = (m[0], m[1]) if m else (None, 0.0)
    if got == want:
        ok += 1
    else:
        print(f"  ✗ {text!r} -> {got}({sc:.0f}) want {want}")
        print(f"      top={score_all(text, parts)[:3]}")
print(f"매칭 정확도: {ok}/{len(CASES)}")

PAIRS = [    # naming either side must light both surfaces("camera", "camera_poster"), ("laser", "laser_poster"),
         ("galvo", "galvo_poster"), ("DAC", "DAC_poster"),
         ("electronics", "electronics_poster")]
link_ok = 0
for a, b in PAIRS:
    ga, gb = set(links.get(a, [])), set(links.get(b, []))
    surf = {spec[a].get("surface"), spec[b].get("surface")}
    if ga == {a, b} and gb == {a, b} and surf == {"mockup", "poster"}:
        link_ok += 1
    else:
        print(f"  ✗ 링크 {a}↔{b}: {sorted(ga)} / {sorted(gb)} 표면={surf}")
print(f"링크 쌍(대칭·양 표면): {link_ok}/{len(PAIRS)}")

untagged = [pid for pid, p in spec.items() if not p.get("surface")]
print(f"표면 태그 누락: {untagged or '없음'}")

from match import shared_aliases
dup = shared_aliases(spec, links)
print(f"link 없는 중복 별칭: {[(a, p) for a, p in dup] or '없음'}")

assert ok >= len(CASES) - 1, "매칭 정확도 목표 미달"
assert link_ok == len(PAIRS), "링크 쌍이 깨졌다"
assert not untagged, "surface 태그가 없는 부위가 있다"
print("PASS")
