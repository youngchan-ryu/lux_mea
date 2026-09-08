"""Turn bench_asr.py's CSVs into a report you can draw a conclusion from.

    python3 bench_report.py
    python3 bench_report.py --runs bench_runs.csv --live bench_live.csv

Separate from collection so the statistics can be redone -- differently, or
after adding an engine -- without paying for the audio again.

Two things here are not decoration. Every engine sees the same clips, so the
comparison is paired and belongs in McNemar's test rather than two percentages
side by side; reading raw rates is what made bench_asr_result.md look like
large-v3-turbo went from 70% to 40% when the honest answer was "ten clips
cannot tell". And the confidence intervals resample *clips*, not rows, because
three repeats of one clip are not three independent observations.

No new dependencies: rapidfuzz is already required for matching, and the exact
McNemar test is a binomial, which math.comb does.
"""
from __future__ import annotations
import argparse
import csv
import math
import os
from collections import Counter, defaultdict

import numpy as np

from paths import data_path

WINDOW_S = 3.0          # a mark counts as answered if the beam arrives within this


# ------------------------------------------------------------------ stats

def bootstrap_ci(by_clip: dict, iters=2000, seed=0, alpha=0.05):
    """95% CI for a hit rate, resampling clips rather than trials.

    Repeats of the same clip share its difficulty; treating them as independent
    would shrink the interval to something the data does not support.
    """
    clips = list(by_clip)
    if not clips:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = np.arange(len(clips))
    means = np.empty(iters)
    vals = [by_clip[c] for c in clips]
    for i in range(iters):
        pick = rng.choice(idx, size=len(idx), replace=True)
        hits = tot = 0
        for j in pick:
            h, t = vals[j]
            hits += h; tot += t
        means[i] = hits / tot if tot else np.nan
    lo, hi = np.nanpercentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def mcnemar(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts.

    b = A right where B was wrong, c = the reverse. Clips both engines got
    right, or both wrong, carry no information about which is better -- that is
    exactly why the paired test is sharper than comparing two rates.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2 * tail)


def cer(ref: str, hyp: str) -> float | None:
    """Character error rate after the matcher's own normalisation.

    Normalising the same way match.py does drops spacing and punctuation, which
    the matcher ignores anyway; counting them would make an engine look worse
    for differences that cannot change which part lights up.
    """
    from rapidfuzz.distance import Levenshtein
    from match import normalize
    r, h = normalize(ref or ""), normalize(hyp or "")
    if not r:
        return None
    return Levenshtein.distance(r, h) / len(r)


def pct(xs, q):
    return float(np.percentile(xs, q)) if len(xs) else float("nan")


# ------------------------------------------------------------------- load

def engine_label(row) -> str:
    return f'{row["engine"]}:{row["model"]}' if row["model"] else row["engine"]


def read_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_marks(path):
    rows = read_rows(path)
    return sorted((float(r["t_end_s"]), r["part"]) for r in rows)


# ---------------------------------------------------------------- tier A

def tier_a(rows, out):
    if not rows:
        return
    engines = sorted({engine_label(r) for r in rows}, key=str)
    conds = sorted({r["cond"] for r in rows},
                   key=lambda c: (c != "clean", c))
    prompts = sorted({r["prompt_on"] for r in rows})

    out.append("## Tier A — 클립 단위 정확도·추론 지연\n")
    for p in prompts:
        sub_p = [r for r in rows if r["prompt_on"] == p]
        tag = "프롬프트·부스팅 ON" if p == "1" else "프롬프트·부스팅 OFF"
        out.append(f"### {tag}\n")
        out.append("| 엔진 | 조건 | n | 적중률 | 95% CI | CER | 지연 p50 | p90 | p95 |")
        out.append("|---|---|--:|--:|:--:|--:|--:|--:|--:|")
        for e in engines:
            for c in conds:
                sub = [r for r in sub_p if engine_label(r) == e and r["cond"] == c]
                if not sub:
                    continue
                by_clip = defaultdict(lambda: [0, 0])
                for r in sub:
                    by_clip[r["clip"]][0] += int(r["correct"])
                    by_clip[r["clip"]][1] += 1
                hits = sum(int(r["correct"]) for r in sub)
                acc = hits / len(sub)
                lo, hi = bootstrap_ci({k: tuple(v) for k, v in by_clip.items()})
                cers = [x for x in (cer(r["ref_text"], r["text"]) for r in sub)
                        if x is not None]
                lat = [float(r["latency_s"]) for r in sub if r["latency_s"]]
                out.append(
                    f"| {e} | {c} | {len(sub)} | {acc*100:.0f}% | "
                    f"{lo*100:.0f}–{hi*100:.0f}% | "
                    f"{np.mean(cers):.2f} | {pct(lat,50):.2f}s | "
                    f"{pct(lat,90):.2f}s | {pct(lat,95):.2f}s |")
        out.append("")

    out.append("> `clova-stream` 의 지연은 **클립 끝 기준**이라 다른 행과 의미가 다르다. "
               "음수면 오디오가 끝나기 전에 이미 답이 나왔다는 뜻이다.\n")

    # -- error taxonomy
    out.append("### 오류 분류\n")
    out.append("| 엔진 | hit | no_match | wrong_part | empty | degenerate |")
    out.append("|---|--:|--:|--:|--:|--:|")
    for e in engines:
        c = Counter(r["err"] for r in rows if engine_label(r) == e)
        out.append(f"| {e} | " + " | ".join(
            str(c.get(k, 0)) for k in
            ("hit", "no_match", "wrong_part", "empty", "degenerate")) + " |")
    out.append("")
    out.append("> `empty` 와 `degenerate` 는 모델을 키워서 고치는 실패가 아니다. "
               "전자는 VAD 가 잘못 잘랐거나 발화가 너무 짧은 것이고, 후자는 "
               "Whisper 가 잡음에 반복 환각을 낸 것이다 "
               "(`speech._is_degenerate`, `noise_guard` 가 막는 대상).\n")

    # -- paired comparison
    out.append("### 엔진 쌍 비교 (McNemar 정확검정, 페어드)\n")
    out.append("클립별 다수결로 정오를 정하고 같은 클립끼리 맞붙인다. "
               "b/c 는 한쪽만 맞힌 클립 수 — 둘 다 맞히거나 둘 다 틀린 클립은 "
               "어느 쪽이 나은지에 대해 아무 정보가 없다.\n")
    out.append("| A | B | 조건 | b (A만 맞음) | c (B만 맞음) | p |")
    out.append("|---|---|---|--:|--:|--:|")
    for c in conds + ["전체"]:
        verdict = {}
        for e in engines:
            per = defaultdict(lambda: [0, 0])
            for r in rows:
                if r["prompt_on"] != "1" or engine_label(r) != e:
                    continue
                if c != "전체" and r["cond"] != c:
                    continue
                per[r["clip"]][0] += int(r["correct"])
                per[r["clip"]][1] += 1
            verdict[e] = {k: (h * 2 >= t) for k, (h, t) in per.items()}
        for i, a in enumerate(engines):
            for bname in engines[i + 1:]:
                common = set(verdict[a]) & set(verdict[bname])
                if not common:
                    continue
                b = sum(1 for k in common if verdict[a][k] and not verdict[bname][k])
                cc = sum(1 for k in common if verdict[bname][k] and not verdict[a][k])
                p = mcnemar(b, cc)
                star = " ✅" if p < 0.05 else ""
                out.append(f"| {a} | {bname} | {c} | {b} | {cc} | {p:.3f}{star} |")
    out.append("")
    out.append("> p ≥ 0.05 면 **차이를 보였다고 말할 수 없다.** 표본을 늘리거나, "
               "차이가 없다고 결론내고 더 싸고 빠른 쪽을 쓸 것.\n")


# ---------------------------------------------------------------- tier B

def tier_b(rows, marks, out):
    if not rows:
        return
    out.append("## Tier B — 실전 반응 지연\n")
    if not marks:
        out.append("_marks.csv 가 없어 지연을 계산하지 못했다. "
                   "`python3 bench_asr.py marks --wav <발표.wav>` 후 손으로 다듬을 것._\n")
    groups = defaultdict(list)
    for r in rows:
        groups[(engine_label(r), r["config"])].append(r)

    out.append("| 엔진 | 설정 | 반응 p50 | p90 | 놓침 | 오발화/분 | 타겟 전환 |")
    out.append("|---|---|--:|--:|--:|--:|--:|")
    for (e, cfg), rs in sorted(groups.items()):
        rs.sort(key=lambda r: float(r["t_emit"]))
        span = max(float(r["t_emit"]) for r in rs) / 60.0 or 1e-9
        lats, missed = [], 0
        used = set()
        for t_end, pid in marks:
            # One emission answers one mark: without this, a part named twice
            # would score the same hit twice and hide a miss.
            hit = next((i for i, r in enumerate(rs)
                        if i not in used and r["part"] == pid
                        and t_end <= float(r["t_emit"]) <= t_end + WINDOW_S), None)
            if hit is None:
                missed += 1
            else:
                used.add(hit)
                lats.append(float(rs[hit]["t_emit"]) - t_end)
        # A change is a misfire when no mark for that part is open around it.
        false_fire = 0
        for r in rs:
            if r["changed"] != "1":
                continue
            t = float(r["t_emit"])
            if not any(m_pid == r["part"] and t_end <= t <= t_end + WINDOW_S
                       for t_end, m_pid in marks):
                false_fire += 1
        switches = sum(1 for r in rs if r["changed"] == "1")
        out.append(
            f"| {e} | {cfg or '-'} | {pct(lats,50):.2f}s | {pct(lats,90):.2f}s | "
            f"{missed}/{len(marks)} | {false_fire/span:.1f} | {switches} |")
    out.append("")
    out.append("> 반응 지연은 **부위 이름을 다 말한 순간부터 빔이 그리로 옮겨갈 때까지**다. "
               "로컬 경로에서는 여기에 `speech.SILENCE_END` 대기와 세그먼트 전체 추론이 "
               "그대로 들어가고, 스트리밍에서는 서버 EPD(`gapThreshold`, `useWordEpd`)가 "
               "그 자리를 대신한다. 둘을 같이 스윕해야 공정한 비교다.\n")


# -------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="bench_runs.csv")
    ap.add_argument("--live", default="bench_live.csv")
    ap.add_argument("--marks", default="marks.csv")
    ap.add_argument("--out", default="bench_report.md")
    a = ap.parse_args()

    runs = read_rows(data_path(a.runs))
    live = read_rows(data_path(a.live))
    marks = read_marks(data_path(a.marks))
    if not runs and not live:
        print(f"[!] {data_path(a.runs)} 도 {data_path(a.live)} 도 없다.")
        print("    먼저 python3 bench_asr.py run / live 를 돌릴 것")
        return

    out = ["# ASR 비교 리포트", ""]
    out.append(f"클립 시행 {len(runs)}건 · 실전 발화 {len(live)}건 · "
               f"정답 표시 {len(marks)}개\n")
    tier_a(runs, out)
    tier_b(live, marks, out)

    if runs:
        # Cloud audio actually sent, so credit burn is visible next to results.
        cloud = [r for r in runs if r["engine"].startswith("clova")]
        if cloud:
            secs = sum(float(r["audio_s"]) for r in cloud)
            out.append("## CLOVA 소모\n")
            out.append(f"클라우드 시행 {len(cloud)}건 · 오디오 {secs/60:.1f}분 "
                       f"({secs:.0f}초). 콘솔 사용량과 대조할 것.\n")

        worst = [r for r in runs if r["correct"] == "0" and r["prompt_on"] == "1"]
        if worst:
            out.append("## 오답 표본\n")
            out.append("| 엔진 | 조건 | 정답 | 매칭 | 전사 |")
            out.append("|---|---|---|---|---|")
            for r in worst[:25]:
                out.append(f'| {engine_label(r)} | {r["cond"]} | {r["part"]} | '
                           f'{r["matched"] or "—"} | {r["text"][:60] or "(빈 전사)"} |')
            out.append("")
            out.append("> 오답의 전사를 읽고 `parts.yaml` 별칭을 보강하는 것이 "
                       "모델을 키우는 것보다 대개 효과가 크다. 이 표가 그 목록이다.\n")

    path = data_path(a.out)
    open(path, "w", encoding="utf-8").write("\n".join(out))
    print("\n".join(out))
    print(f"\n[i] → {path}")


if __name__ == "__main__":
    main()
