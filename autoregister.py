"""Propose parts.yaml entries automatically from a photo.

    python3 autoregister.py plate_poster.jpg --mode poster --surface poster
    python3 autoregister.py plate.jpg --mode parts --surface mockup --spec queries.yaml

Automation belongs to authoring only -- no vision model runs while presenting,
so this cannot make the demo less reliable. Output is a draft that a human then
checks in register.py.

Posters use OCR, because a text block is the thing being pointed at and the
recognised text doubles as the alias list. Physical parts use open-vocabulary
detection instead: printed labels sit *beside* a component, so anchoring on the
label would put the beam on the label.

OCR backend: macOS Vision (pyobjc-framework-Vision), falling back to easyocr.
Detection: transformers + torch (OWLv2)."""
from __future__ import annotations
import argparse
import os
import re
import sys

import cv2
import numpy as np
import yaml
from paths import data_path, ensure_data


def ocr_vision(path: str, langs=("ko-KR", "en-US")):
    """macOS 내장 Vision. 모델 다운로드가 없고 한국어를 지원한다."""
    from Cocoa import NSURL
    import Quartz
    import Vision

    url = NSURL.fileURLWithPath_(os.path.abspath(path))
    src = Quartz.CGImageSourceCreateWithURL(url, None)
    if src is None:
        raise RuntimeError(f"이미지를 못 엶: {path}")
    cg = Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)
    W, H = Quartz.CGImageGetWidth(cg), Quartz.CGImageGetHeight(cg)

    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg, None)
    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLanguages_(list(langs))
    req.setRecognitionLevel_(0)
    req.setUsesLanguageCorrection_(True)
    ok, err = handler.performRequests_error_([req], None)
    if not ok:
        raise RuntimeError(f"Vision 실패: {err}")

    out = []
    for obs in (req.results() or []):
        cands = obs.topCandidates_(1)
        if not cands:
            continue
        txt = cands[0].string()
        bb = obs.boundingBox()
        x0 = bb.origin.x * W
        x1 = (bb.origin.x + bb.size.width) * W
        y1 = (1.0 - bb.origin.y) * H
        y0 = (1.0 - bb.origin.y - bb.size.height) * H
        out.append((txt, x0, y0, x1, y1, float(obs.confidence())))
    return out


def ocr_easyocr(path: str, langs=("ko", "en")):
    """범용 폴백. 첫 실행 시 모델을 내려받는다(수백 MB)."""
    import easyocr
    reader = easyocr.Reader(list(langs), gpu=False)
    out = []
    for box, txt, conf in reader.readtext(path):
        p = np.array(box, float)
        out.append((txt, p[:, 0].min(), p[:, 1].min(),
                    p[:, 0].max(), p[:, 1].max(), float(conf)))
    return out


def run_ocr(path: str, engine="auto"):
    if engine in ("auto", "vision"):
        try:
            r = ocr_vision(path)
            print(f"[i] OCR 엔진 = macOS Vision  ({len(r)}줄)")
            return r
        except Exception as e:
            if engine == "vision":
                sys.exit(f"[!] Vision 실패: {e}")
            print(f"[i] Vision 사용 불가({e}) → easyocr 폴백")
    r = ocr_easyocr(path)
    print(f"[i] OCR 엔진 = easyocr  ({len(r)}줄)")
    return r


def merge_lines(lines, gap_factor=0.8, overlap_min=0.25):
    """세로로 가깝고 가로로 겹치는 줄들을 한 블록으로 묶는다.

    포스터는 '제목 + 본문 몇 줄'이 하나의 의미 단위다. 줄 단위로 등록하면
    부위가 수십 개가 되어 매칭이 흐려진다.
    """
    if not lines:
        return []
    lines = sorted(lines, key=lambda t: t[2])
    hs = [l[4] - l[2] for l in lines]
    med_h = float(np.median(hs)) if hs else 1.0

    blocks, cur = [], [lines[0]]
    for ln in lines[1:]:
        prev = cur[-1]
        gap = ln[2] - prev[4]
        ox = min(prev[3], ln[3]) - max(prev[1], ln[1])
        ow = ox / max(min(prev[3] - prev[1], ln[3] - ln[1]), 1e-6)
        if gap < med_h * gap_factor and ow > overlap_min:
            cur.append(ln)
        else:
            blocks.append(cur); cur = [ln]
    blocks.append(cur)
    return blocks


def block_box(block):
    a = np.array([[b[1], b[2], b[3], b[4]] for b in block], float)
    return a[:, 0].min(), a[:, 1].min(), a[:, 2].max(), a[:, 3].max()


_PARTICLE = re.compile(r"(은|는|이|가|을|를|의|에|에서|으로|로|와|과|도|만)$")
_STOP = {"그리고", "또는", "하는", "있는", "위한", "통해", "대한", "및", "the",
         "and", "for", "with", "our", "this", "that"}


def make_aliases(block, max_n=6):
    """블록 텍스트에서 매칭용 별칭을 만든다.

    match.py 는 별칭 길이의 합을 evidence 로 쓴다. 짧은 조각은 오히려
    엉뚱한 부위를 이기게 만들므로 2글자 미만·조사·불용어는 버린다.
    """
    title = block[0][0].strip()
    full = re.sub(r"[^0-9A-Za-z가-힣]", "", title)
    cands = []
    if len(full) >= 2:
        cands.append(full)

    for ln in block[:2]:
        for tok in re.split(r"[\s,·/()\[\]:;]+", ln[0]):
            tok = _PARTICLE.sub("", re.sub(r"[^0-9A-Za-z가-힣]", "", tok))
            if len(tok) < 2 or tok.lower() in _STOP or tok.isdigit():
                continue
            if tok not in cands:
                cands.append(tok)
    return cands[:max_n], title


def detect_parts(path: str, spec_path: str, threshold=0.2):
    """open-vocabulary detection (OWLv2). 실물 부품용.

    ★ OCR 을 쓰지 않는 이유: 인쇄 라벨은 부품 **옆**에 붙는다. 라벨 위치를
      anchor 로 쓰면 빔이 부품이 아니라 라벨을 짚는다.

    queries.yaml 형식:
        laser:
          query: laser diode module
          aliases: [레이저, 레이저모듈, 광원]
    """
    from PIL import Image
    import torch
    from transformers import Owlv2Processor, Owlv2ForObjectDetection

    spec = yaml.safe_load(open(data_path(spec_path)))
    ids = list(spec)
    queries = [spec[k]["query"] for k in ids]

    mid = "google/owlv2-base-patch16-ensemble"
    print(f"[i] 모델 로드 {mid} (첫 실행은 다운로드로 오래 걸림)")
    proc = Owlv2Processor.from_pretrained(mid)
    model = Owlv2ForObjectDetection.from_pretrained(mid)

    im = Image.open(path).convert("RGB")
    inputs = proc(text=[queries], images=im, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    ts = torch.tensor([[im.size[1], im.size[0]]])
    try:
        res = proc.post_process_grounded_object_detection(
            outputs, threshold=threshold, target_sizes=ts)[0]
    except AttributeError:
        res = proc.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=ts)[0]

    best = {}
    for box, score, label in zip(res["boxes"], res["scores"], res["labels"]):
        k = int(label)
        if k not in best or float(score) > best[k][1]:
            best[k] = ([float(v) for v in box], float(score))

    out = []
    for k, (box, score) in sorted(best.items()):
        pid = ids[k]
        out.append((pid, spec[pid].get("aliases", []), box, score,
                    spec[pid]["query"]))
        print(f"  {pid:14s} score={score:.2f}  {spec[pid]['query']}")
    missing = [ids[k] for k in range(len(ids)) if k not in best]
    if missing:
        print(f"  ⚠️ 미검출: {', '.join(missing)} — threshold 를 낮추거나 "
              f"query 문구를 바꿔볼 것")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--mode", choices=["poster", "parts"], default="poster")
    ap.add_argument("--surface", default=None,
                    help="parts.yaml 에 기록할 표면 이름 (mockup / poster)")
    ap.add_argument("--engine", default="auto", choices=["auto", "vision", "easyocr"])
    ap.add_argument("--spec", default="queries.yaml", help="--mode parts 용 질의 정의")
    ap.add_argument("--min-conf", type=float, default=0.4)
    ap.add_argument("--max-parts", type=int, default=12)
    ap.add_argument("--pad", type=float, default=0.15,
                    help="박스 여백 비율 (도형이 글자를 가리지 않도록)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    a.image = data_path(a.image)
    ensure_data()
    img = cv2.imread(a.image)
    if img is None:
        sys.exit(f"[!] 이미지를 못 읽음: {a.image}")
    H, W = img.shape[:2]
    sfx = a.surface or a.mode
    out_yaml = a.out or f"parts_{sfx}_auto.yaml"
    print(f"[i] {a.image}  {W}x{H}  mode={a.mode}  surface={sfx}")

    parts = {}
    draw = []

    if a.mode == "poster":
        lines = [l for l in run_ocr(a.image, a.engine) if l[5] >= a.min_conf]
        blocks = merge_lines(lines)
        print(f"[i] {len(lines)}줄 → {len(blocks)}블록")

        blocks.sort(key=lambda b: -(b[0][4] - b[0][2]))
        for i, blk in enumerate(blocks[:a.max_parts]):
            aliases, title = make_aliases(blk)
            if not aliases:
                continue
            x0, y0, x1, y1 = block_box(blk)
            pw, ph = (x1 - x0) * a.pad, (y1 - y0) * a.pad
            x0, y0, x1, y1 = x0 - pw, y0 - ph, x1 + pw, y1 + ph
            pid = f"sec{i+1:02d}"
            parts[pid] = {
                "surface": sfx,
                "label": title,
                "aliases": aliases,
                "anchor": [round((x0 + x1) / 2 / W, 4),
                           round((y0 + y1) / 2 / H, 4)],
                "shape": {"type": "rect",
                          "w_norm": round((x1 - x0) / W, 4),
                          "h_norm": round((y1 - y0) / H, 4)},
            }
            draw.append((pid, title, x0, y0, x1, y1))
            print(f"  {pid}  «{title[:28]}»  별칭 {aliases}")

    else:
        if not os.path.exists(data_path(a.spec)):
            sys.exit(f"[!] {a.spec} 없음. --spec 으로 질의 정의 파일을 줄 것\n"
                     f"    형식:\n    laser:\n      query: laser diode module\n"
                     f"      aliases: [레이저, 레이저모듈]")
        for pid, aliases, box, score, q in detect_parts(a.image, a.spec):
            x0, y0, x1, y1 = box
            parts[pid] = {
                "surface": sfx,
                "label": q,
                "score": round(score, 3),
                "aliases": aliases,
                "anchor": [round((x0 + x1) / 2 / W, 4),
                           round((y0 + y1) / 2 / H, 4)],
                "shape": {"type": "circle",
                          "r_norm": round(max(x1 - x0, y1 - y0) / 2 / W, 4)},
            }
            draw.append((pid, q, x0, y0, x1, y1))

    if not parts:
        sys.exit("[!] 후보를 하나도 못 찾음. --min-conf 를 낮추거나 조명을 개선할 것")

    with open(data_path(out_yaml), "w") as f:
        yaml.safe_dump({"object": f"{sfx}_auto", "parts": parts}, f,
                       allow_unicode=True, sort_keys=False)
    print(f"[i] {out_yaml} 저장 ({len(parts)}개 후보)")

    v = img.copy()
    for pid, title, x0, y0, x1, y1 in draw:
        cv2.rectangle(v, (int(x0), int(y0)), (int(x1), int(y1)), (60, 255, 60), 2)
        cv2.putText(v, pid, (int(x0), max(int(y0) - 8, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 255, 60), 2)
    rev = f"autoreg_{sfx}.png"
    cv2.imwrite(data_path(rev), v)
    print(f"[i] {rev} 저장 — 박스가 지시하려는 대상과 맞는지 **반드시 눈으로 확인**")

    print(f"\n다음: python3 register.py {a.image} --edit {out_yaml}")
    print("      (자동 후보를 불러온 뒤 클릭으로 추가·수정하고 s 로 저장)")


if __name__ == "__main__":
    main()
