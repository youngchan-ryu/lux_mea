"""Mine part vocabulary out of presentation material (script, poster).

    python3 mine_vocab.py --script script.txt --pdf poster.pdf \
                          --parts parts_mockup.yaml
    python3 mine_vocab.py --script script.txt --provider groq        # skip the menu
    python3 mine_vocab.py --script script.txt --mock mock_llm.json   # offline

Like autoregister.py this belongs to authoring only -- the LLM is never called
while presenting, so it cannot make the demo less reliable. Output is a draft
that a human then checks in register.py.

An LLM rather than morphology or regex, because the question is which nouns
name something the beam can land on: "갈보 미러" can be pointed at, "정확도"
cannot, and separating the two needs the meaning of the word.

Outputs:
    parts_<name>_auto.yaml   alias draft, hand to register.py --edit
    vocab_report.txt         what was dropped, and why

Groq or OpenAI, chosen at the prompt unless --provider says which. Groq is the
free tier and defaults to qwen/qwen3.8-27b; OpenAI defaults to gpt-5.6-terra.

Needs requests, plus pypdf to read a PDF, and the key for whichever endpoint is
chosen: GROQ_API_KEY or OPENAI_API_KEY.
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
import unicodedata

import yaml

# ─────────────────────────────────────────────────────────── prompt
SYSTEM = """당신은 발표 자료를 분석해 "레이저 포인터로 물리적으로 지시할 수 있는 부위"를
추출하는 도구입니다.

이 시스템은 발표자가 부위 이름을 말하면 레이저가 그곳을 자동으로 짚습니다.
따라서 추출 대상은 **공간상의 한 지점에 대응하는 실체**여야 합니다.

포함:
  - 물리 부품 (예: 갈보 미러, 레이저 모듈, 전원부, 커넥터)
  - 포스터·도표의 구획 (예: 문제 정의 섹션, 시스템 구성도)

제외 (매우 중요):
  - 개념·속성·수치 (정확도, 지연시간, 효율, 안전성)
  - 동작·현상 (블랭킹, 캘리브레이션, 스캐닝) — 단, 그 동작을 담당하는 *부품*은 포함
  - 사람·조직·일정

각 부위마다 **발표자가 실제로 말할 법한 별칭**을 4~6개 생성하세요.
반드시 다음 유형을 섞으십시오:
  - 정식 명칭 (갈보 미러)
  - 축약 (갈보)
  - 영문 (galvo)
  - 음차·구어 (갈바노, 갈바노미터)
  - 기능적 지칭 (거울, 스캐너)

주의: 지나치게 일반적인 단어(모듈, 신호, 부분, 장치, 시스템)는 별칭에 넣지 마십시오.
여러 부위에 동시에 해당하여 오작동을 일으킵니다.

JSON만 출력하십시오. 설명·마크다운·코드펜스 금지.
{
  "parts": [
    {
      "id": "galvo",                        // 영문 소문자·숫자·밑줄만
      "canonical": "갈보 미러",
      "aliases": ["갈보", "갈보미러", "갈바노미터", "스캐너", "거울"],
      "surface": "object",                  // object | poster
      "evidence": "여기 갈보 미러가 빔의 방향을 꺾습니다",  // 원문 인용
      "order": 1,                           // 발표 중 언급 순서
      "confidence": 0.9
    }
  ]
}"""

# ─────────────────────────────────────────────────────────── providers
PROVIDERS = {
    "groq": {
        "label": "Groq",
        "note": "무료 티어",
        "env": "GROQ_API_KEY",
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "model": "qwen/qwen3.8-27b",
        "docs": "console.groq.com/docs/models",
        "temperature": 0.2,
    },
    "openai": {
        "label": "OpenAI",
        "note": "유료",
        "env": "OPENAI_API_KEY",
        "url": "https://api.openai.com/v1/chat/completions",
        "model": "gpt-5.6-terra",
        "docs": "developers.openai.com/api/docs/models",
        "temperature": 0.2,
    },
}


def pick_provider() -> str:
    """Ask which endpoint to call. Without a terminal, fall back to whichever
    key is set, and refuse if that is ambiguous."""
    have = [n for n, p in PROVIDERS.items() if os.environ.get(p["env"])]
    if not sys.stdin.isatty():
        if len(have) == 1:
            print(f"[i] 비대화형 실행 — 키가 있는 {have[0]} 사용")
            return have[0]
        sys.exit("[!] 비대화형 실행에서는 --provider groq|openai 를 지정할 것")

    print("\n어느 API로 호출할까요?")
    for i, (name, p) in enumerate(PROVIDERS.items(), 1):
        missing = "" if os.environ.get(p["env"]) else f"   ← {p['env']} 없음"
        print(f"  {i}) {p['label']:6s} ({p['note']})  {p['model']}{missing}")
    names = list(PROVIDERS)
    while True:
        c = input("선택 [1/2]: ").strip()
        if c in ("1", "2"):
            return names[int(c) - 1]
        print("    1 또는 2 를 입력해 주세요")


# ─────────────────────────────────────────────────────────── filters
_PARTICLE = re.compile(r"(은|는|이|가|을|를|의|에|에서|으로|로|와|과|도|만)$")
_GENERIC = {
    "모듈", "신호", "부분", "장치", "시스템", "구성", "요소", "기능", "방식",
    "부품", "회로", "기판", "센서", "제어", "출력", "입력", "전체", "내부",
    "module", "signal", "system", "device", "part", "unit", "component",
}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFC", str(s)).lower()
    return re.sub(r"[^0-9a-z가-힣]", "", s)


def clean_alias(a: str) -> str:
    return _PARTICLE.sub("", norm(a))


def filter_parts(parts: list[dict], min_len=2) -> tuple[list[dict], list[str]]:
    """Drop aliases that cannot tell one part from another.

    match.py scores by the total length of the aliases that matched, so a short
    generic alias attaches to any long sentence and beats the specific part the
    speaker meant. Generating aliases makes that failure more likely, so the
    test applied here is discrimination rather than relevance: an alias claimed
    by two parts is removed from both."""
    warn = []

    # pass 1: normalise, then drop by length and generic-word list
    for p in parts:
        kept, seen = [], set()
        for a in p.get("aliases", []):
            c = clean_alias(a)
            if len(c) < min_len:
                warn.append(f"[짧음] {p['id']}: '{a}' 제거 ({len(c)}자)")
            elif c in _GENERIC:
                warn.append(f"[일반어] {p['id']}: '{a}' 제거")
            elif c in seen:
                pass
            else:
                seen.add(c)
                kept.append(a)
        p["aliases"] = kept

    # pass 2: an alias claimed by more than one part is dropped from all of them
    owner: dict[str, list[str]] = {}
    for p in parts:
        for a in p["aliases"]:
            owner.setdefault(clean_alias(a), []).append(p["id"])
    ambiguous = {k for k, v in owner.items() if len(v) > 1}
    for k in sorted(ambiguous):
        warn.append(f"[모호] '{k}' 가 {owner[k]} 에 중복 → 전부 제거")
    for p in parts:
        p["aliases"] = [a for a in p["aliases"] if clean_alias(a) not in ambiguous]

    # pass 3: warn about parts left short of aliases
    out = []
    for p in parts:
        n = len(p["aliases"])
        if n == 0:
            warn.append(f"[탈락] {p['id']}: 남은 별칭 0개 — 수동 등록 필요")
            continue
        if n < 3:
            warn.append(f"[빈약] {p['id']}: 별칭 {n}개 — 검수 시 보강 권장")
        out.append(p)
    return out, warn


def merge_existing(auto: list[dict], existing: dict) -> tuple[dict, list[str]]:
    """Merge into an existing parts.yaml. Anchors are never touched; position
    stays a manual step."""
    notes, merged = [], {}
    ex_norm = {pid: {clean_alias(a) for a in spec.get("aliases", [])}
               for pid, spec in existing.items()}

    for p in auto:
        pid = p["id"]
        if pid in existing:
            spec = dict(existing[pid])
            new = [a for a in p["aliases"] if clean_alias(a) not in ex_norm[pid]]
            if new:
                spec["aliases"] = list(spec.get("aliases", [])) + new
                notes.append(f"[보강] {pid}: +{new}")
            merged[pid] = spec
        else:
            notes.append(f"[신규] {pid}: anchor 미지정 — register.py 로 위치 등록 필요")
            merged[pid] = {
                "surface": p.get("surface", "object"),
                "aliases": p["aliases"],
                "anchor": None,                       # register.py fills this in
                "shape": {"type": "circle", "r_norm": 0.07},
                "_evidence": p.get("evidence", ""),
                "_confidence": p.get("confidence"),
            }

    for pid, spec in existing.items():                 # keep parts the mining missed
        if pid not in merged:
            merged[pid] = spec
            notes.append(f"[유지] {pid}: 자료에서 언급 없음 — 대본 누락 여부 확인")
    return merged, notes


def build_tour(auto: list[dict]) -> list[str]:
    """Mention order -> TOUR route, available whenever a script was given."""
    ordered = [p for p in auto if isinstance(p.get("order"), (int, float))]
    return [p["id"] for p in sorted(ordered, key=lambda x: x["order"])]


# ─────────────────────────────────────────────────────────── input, LLM
def read_text(script=None, pdf=None) -> str:
    chunks = []
    if script:
        chunks.append(open(script, encoding="utf-8").read())
    if pdf:
        try:
            from pypdf import PdfReader
            t = "\n".join((pg.extract_text() or "") for pg in PdfReader(pdf).pages)
            if len(t.strip()) < 50:
                print("[!] PDF 텍스트 레이어가 비어 있음 — 이미지 PDF로 보인다.")
                print("    autoregister.py 의 macOS Vision OCR 경로를 쓰거나,")
                print("    포스터 원본(텍스트 살아있는 PDF)을 사용할 것")
            chunks.append(t)
        except ImportError:
            sys.exit("[!] pypdf 필요: pip install pypdf")
    if not chunks:
        sys.exit("[!] --script 또는 --pdf 중 하나는 필요")
    return "\n\n".join(chunks)


def call_llm(text: str, provider: str, model: str, existing_ids: list[str]) -> list[dict]:
    import requests
    prov = PROVIDERS[provider]
    key = os.environ.get(prov["env"])
    if not key:
        sys.exit(f"[!] {prov['env']} 가 설정되지 않음 — export {prov['env']}=...")

    hint = ""
    if existing_ids:
        hint = ("\n\n이미 등록된 부위 id 목록입니다. 같은 대상이면 **동일한 id**를 쓰고, "
                f"새 부위만 새 id를 부여하세요:\n{', '.join(existing_ids)}")

    r = requests.post(prov["url"], timeout=90,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": model, "temperature": prov["temperature"],
              "response_format": {"type": "json_object"},
              "messages": [{"role": "system", "content": SYSTEM + hint},
                           {"role": "user", "content": text[:24000]}]})
    if r.status_code >= 400:
        sys.exit(f"[!] {prov['label']} 오류 {r.status_code}: {r.text[:300]}\n"
                 f"    모델명을 확인할 것 (--model). 목록: {prov['docs']}")
    raw = r.json()["choices"][0]["message"]["content"]
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    return json.loads(raw).get("parts", [])


# ─────────────────────────────────────────────────────────── main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script"); ap.add_argument("--pdf")
    ap.add_argument("--parts", help="기존 parts.yaml (별칭 보강 대상)")
    ap.add_argument("--provider", choices=list(PROVIDERS),
                    help="생략하면 실행할 때 고르게 한다")
    ap.add_argument("--model", default=None, help="생략하면 provider 기본 모델")
    ap.add_argument("--mock", help="LLM 응답 JSON 파일 (오프라인 검증용)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    existing = {}
    if a.parts and os.path.exists(a.parts):
        existing = (yaml.safe_load(open(a.parts)) or {}).get("parts", {}) or {}
        print(f"[i] 기존 부위 {len(existing)}개 로드")

    if a.mock:
        auto = json.load(open(a.mock)).get("parts", [])
        print(f"[i] mock 응답 사용 — 후보 {len(auto)}개")
    else:
        text = read_text(a.script, a.pdf)
        provider = a.provider or pick_provider()
        model = a.model or PROVIDERS[provider]["model"]
        print(f"[i] 입력 텍스트 {len(text)}자 → {PROVIDERS[provider]['label']} / {model} 호출")
        auto = call_llm(text, provider, model, list(existing))
        print(f"[i] 후보 {len(auto)}개 추출")

    kept, warn = filter_parts(auto)
    print(f"[i] 변별력 필터 통과 {len(kept)}/{len(auto)}개")

    merged, notes = merge_existing(kept, existing)
    tour = build_tour(kept)

    out = a.out or (a.parts.replace(".yaml", "_auto.yaml") if a.parts
                    else "parts_auto.yaml")
    doc = {"object": "auto_mined", "parts": merged}
    if tour:
        doc["tours"] = {"script_order": tour}
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, allow_unicode=True, sort_keys=False)

    lines = ["=== 자동 어휘 마이닝 리포트 ===", ""]
    lines += ["[필터]"] + (warn or ["  (없음)"]) + ["", "[병합]"] + (notes or ["  (없음)"])
    need = [p for p, s in merged.items() if s.get("anchor") is None]
    if need:
        lines += ["", "[위치 등록 필요]"] + [f"  {p}" for p in need]
    if tour:
        lines += ["", "[대본 순서 → TOUR 경로]", "  " + " → ".join(tour)]
    open("vocab_report.txt", "w", encoding="utf-8").write("\n".join(lines))

    print(f"[i] {out} 저장 · vocab_report.txt 저장")
    for w in warn[:8]:
        print("   ", w)
    if need:
        print(f"\n[!] 위치 미지정 {len(need)}개: python3 register.py --edit {out}")
    print("\n[!] 사람이 검수한 뒤, bench_asr.py 로 매칭 적중률을 재서 "
          "개선이 확인될 때만 채택할 것.")


if __name__ == "__main__":
    main()
