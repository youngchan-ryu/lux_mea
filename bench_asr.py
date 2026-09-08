"""Collect ASR comparison data. bench_report.py turns it into a report.

Transcription accuracy is not the metric that matters -- the matcher absorbs a
lot of ASR error, so a model that transcribes worse but still lands on the
right part is not worse for us. But the old ten-clip version of this script
could not tell 70% from 40% (see bench_asr_result.md, where large-v3-turbo
scored both on the same setup), and it ran without the decoder prompt that
app.py always passes, so it measured a configuration nobody runs. Hence: more
clips, noise conditions, the prompt on by default, and every trial written to
a CSV so the statistics happen once, later, without spending API credit again.

    python3 bench_asr.py record  --parts parts_poster.yaml --per-part 6
    python3 bench_asr.py augment --parts parts_poster.yaml --snr 10,5,0
    python3 bench_asr.py run     --parts parts_poster.yaml \
                                 --engines "mlx:large,mlx:medium,clova,clova-stream"
    python3 bench_asr.py marks   --wav pres.wav --parts parts_poster.yaml
    python3 bench_asr.py live    --wav pres.wav --marks marks.csv \
                                 --parts parts_poster.yaml --engines "mlx:medium,clova-stream"

`record`/`augment`/`run` measure recognition on clips. `marks`/`live` measure
the thing a clip cannot show: how long after a word is spoken the beam actually
moves, and how often it moves to the wrong part mid-sentence.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import time

import numpy as np
import yaml

from paths import data_path, ensure_data

CLIPS = "clips"
RUNS_CSV = "bench_runs.csv"
LIVE_CSV = "bench_live.csv"

RUN_FIELDS = ["ts", "clips", "clip", "part", "cond", "speaker", "ref_text",
              "engine", "model", "prompt_on", "repeat", "latency_s",
              "first_hit_s", "audio_s", "text", "matched", "correct", "err"]
LIVE_FIELDS = ["ts", "wav", "engine", "model", "config", "t_emit", "text",
               "part", "changed"]

# Sentences to read while recording. A bare part name spoken in isolation is
# not what the model sees during a talk, and having a fixed script is also what
# makes CER computable at all.
TEMPLATES = [
    "다음은 {a} 입니다",
    "여기 보이는 것이 {a} 입니다",
    "이제 {a} 에 대해 말씀드리겠습니다",
    "{a} 를 보시면 이렇게 되어 있습니다",
    "그리고 이쪽이 {a} 부분입니다",
    "마지막으로 {a} 를 설명드리겠습니다",
]


def clips_dir_for(parts_yaml: str) -> str:
    """parts_poster.yaml -> clips_poster/, so surfaces keep separate clips."""
    base = os.path.splitext(os.path.basename(parts_yaml))[0]
    if base.startswith("parts_"):
        return f"clips_{base[len('parts_'):]}"
    if base == "parts":
        return CLIPS
    return f"clips_{base}"


def load_labels(clips_dir: str) -> dict:
    """labels.json, upgrading the old {file: part_id} form on the way in."""
    path = os.path.join(clips_dir, "labels.json")
    if not os.path.exists(path):
        return {}
    raw = json.load(open(path, encoding="utf-8"))
    out = {}
    for fn, v in raw.items():
        if isinstance(v, str):
            v = {"part": v, "text": "", "speaker": "", "cond": "clean"}
        v.setdefault("cond", "clean")
        v.setdefault("text", "")
        v.setdefault("speaker", "")
        out[fn] = v
    return out


def save_labels(clips_dir: str, labels: dict):
    json.dump(labels, open(os.path.join(clips_dir, "labels.json"), "w",
                           encoding="utf-8"), ensure_ascii=False, indent=1)


def load_spec(parts_yaml: str) -> dict:
    return yaml.safe_load(open(data_path(parts_yaml), encoding="utf-8"))["parts"]


def scripts_for(spec: dict, per_part: int, parts_yaml: str) -> dict:
    """{pid: [sentence, ...]}, from bench_script.yaml if the user wrote one.

    Falling back to templates over the aliases keeps every utterance different,
    which matters: six readings of one sentence measure the recording, not the
    vocabulary.
    """
    custom_path = data_path(f"bench_script_{os.path.splitext(os.path.basename(parts_yaml))[0]}.yaml")
    custom = {}
    if os.path.exists(custom_path):
        custom = yaml.safe_load(open(custom_path, encoding="utf-8")) or {}
        print(f"[i] 대본 사용: {custom_path}")
    out = {}
    for pid, p in spec.items():
        if pid in custom:
            out[pid] = list(custom[pid])[:per_part]
            continue
        names = [a for a in p.get("aliases", []) if a.strip()] or [pid]
        out[pid] = [TEMPLATES[i % len(TEMPLATES)].format(a=names[i % len(names)])
                    for i in range(per_part)]
    return out


# ------------------------------------------------------------------ record

def cmd_record(parts_yaml="parts.yaml", per_part=6, sec=3.5, speaker="",
               clips_dir=None):
    import sounddevice as sd, soundfile as sf
    clips_dir = data_path(clips_dir or clips_dir_for(parts_yaml))
    spec = load_spec(parts_yaml)
    os.makedirs(clips_dir, exist_ok=True)
    labels = load_labels(clips_dir)
    scripts = scripts_for(spec, per_part, parts_yaml)

    n = len(spec) * per_part
    print(f"[i] {parts_yaml} → {clips_dir}/")
    print(f"부위 {len(spec)}개 × {per_part}회 = {n}개. 화면의 문장을 그대로 읽으세요.")
    print(f"   (약 {n * 8 / 60:.0f}분. 통계가 서려면 이 정도는 필요합니다)\n")
    for pid in spec:
        for k, sentence in enumerate(scripts[pid]):
            input(f"  [{pid}] {k+1}/{per_part}  «{sentence}»  — Enter 후 읽으세요")
            a = sd.rec(int(sec * 16000), samplerate=16000, channels=1,
                       dtype="float32"); sd.wait()
            fn = f"{pid}_{k}.wav"
            sf.write(os.path.join(clips_dir, fn), a.flatten(), 16000)
            labels[fn] = {"part": pid, "text": sentence, "speaker": speaker,
                          "cond": "clean"}
            print(f"    저장 (peak={np.abs(a).max():.2f})")
    save_labels(clips_dir, labels)
    clean = sum(1 for v in labels.values() if v["cond"] == "clean")
    print(f"\n[i] 클린 클립 {clean}개 → 다음: "
          f"python3 bench_asr.py augment --parts {parts_yaml}")


# ----------------------------------------------------------------- augment

def _pink(n: int, rng) -> np.ndarray:
    """Pink noise. Room hum sits far closer to 1/f than to white."""
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.arange(len(spec)); f[0] = 1
    out = np.fft.irfft(spec / np.sqrt(f), n)
    return (out / (np.abs(out).max() + 1e-9)).astype("float32")


def _babble(clips, n, rng, k=6) -> np.ndarray:
    """Overlapping speech, the standard cheap stand-in for a crowded room.

    Built from the recordings themselves, so no external asset is needed and
    the interfering voices have the same channel as the target.
    """
    out = np.zeros(n, dtype="float32")
    if not clips:
        return out
    for _ in range(k):
        c = clips[rng.integers(len(clips))]
        if len(c) < n:
            c = np.tile(c, int(np.ceil(n / len(c))))
        off = int(rng.integers(0, max(1, len(c) - n)))
        out += c[off:off + n]
    return out / (np.abs(out).max() + 1e-9)


def _speech_rms(a: np.ndarray, frame=512) -> float:
    """RMS of the loud half of the clip -- i.e. of the speech, not the pauses.

    Using the whole-clip RMS would count the silence and quietly overstate the
    SNR by several dB.
    """
    nf = max(1, len(a) // frame)
    r = np.array([np.sqrt((a[i*frame:(i+1)*frame] ** 2).mean() + 1e-12)
                  for i in range(nf)])
    loud = r[r >= np.percentile(r, 50)]
    return float(np.sqrt((loud ** 2).mean()))


def cmd_augment(parts_yaml="parts.yaml", snrs=(10, 5, 0), noise_path=None,
                clips_dir=None):
    """Mix each clean clip down to the given SNRs, offline.

    Every engine then sees byte-identical audio in every condition, which is
    what makes the paired statistics in bench_report.py legitimate. Seeded per
    clip, so re-running reproduces the same files rather than invalidating the
    runs already collected.
    """
    import soundfile as sf
    clips_dir = data_path(clips_dir or clips_dir_for(parts_yaml))
    labels = load_labels(clips_dir)
    clean = {fn: v for fn, v in labels.items() if v["cond"] == "clean"}
    if not clean:
        print(f"[!] {clips_dir} 에 클린 클립이 없다. 먼저 record 를 실행할 것")
        return
    pool = []
    for fn in clean:
        a, _ = sf.read(os.path.join(clips_dir, fn), dtype="float32")
        pool.append(a.mean(1) if a.ndim > 1 else a)

    noise = None
    if noise_path:
        noise, sr = sf.read(data_path(noise_path), dtype="float32")
        noise = noise.mean(1) if noise.ndim > 1 else noise
        if sr != 16000:
            print(f"[!] {noise_path} 는 {sr}Hz — 16kHz 로 변환해 쓸 것")
            return
        print(f"[i] 실측 소음 사용: {noise_path} ({len(noise)/16000:.1f}s)")
    else:
        print("[i] 소음 파일 미지정 — 클립으로 babble + 핑크 노이즈 합성")
        print("    (부스 소음을 30초 녹음해 --noise 로 주면 훨씬 현실적이다)")

    made = 0
    for fn, meta in clean.items():
        a, _ = sf.read(os.path.join(clips_dir, fn), dtype="float32")
        a = a.mean(1) if a.ndim > 1 else a
        sig = _speech_rms(a)
        for snr in snrs:
            # Not hash(): str hashing is salted per process, so the mix would
            # differ between runs and the paired comparison would be comparing
            # different audio.
            key = f"{fn}|{int(snr)}".encode()
            rng = np.random.default_rng(
                int.from_bytes(hashlib.md5(key).digest()[:4], "little"))
            if noise is not None:
                off = int(rng.integers(0, max(1, len(noise) - len(a))))
                nz = np.resize(noise[off:off + len(a)], len(a)).astype("float32")
            else:
                nz = 0.7 * _babble(pool, len(a), rng) + 0.3 * _pink(len(a), rng)
            nrms = float(np.sqrt((nz ** 2).mean()) + 1e-12)
            nz *= (sig / nrms) * (10 ** (-snr / 20.0))
            mix = np.clip(a + nz, -1.0, 1.0).astype("float32")
            out = f"{os.path.splitext(fn)[0]}__snr{int(snr)}.wav"
            sf.write(os.path.join(clips_dir, out), mix, 16000)
            labels[out] = dict(meta, cond=f"snr{int(snr)}")
            made += 1
    save_labels(clips_dir, labels)
    conds = sorted({v["cond"] for v in labels.values()})
    print(f"[i] {made}개 생성. 조건 {conds}, 클립 총 {len(labels)}개")


# --------------------------------------------------------------- engines

def parse_engines(spec: str) -> list[tuple[str, str | None]]:
    """'mlx:large,clova' -> [('mlx','large'), ('clova',None)].

    A bare model id (mlx-community/...) still works, so the --models form the
    old script used keeps running.
    """
    out = []
    for tok in [t.strip() for t in spec.split(",") if t.strip()]:
        if ":" in tok:
            eng, _, mdl = tok.partition(":")
            out.append((eng.strip(), mdl.strip() or None))
        elif "/" in tok:
            out.append(("mlx" if "mlx" in tok else "faster", tok))
        else:
            out.append((tok, None))
    return out


def label_of(engine: str, model: str | None) -> str:
    return f"{engine}:{model}" if model else engine


def _classify(text: str, want: str, got: str | None) -> str:
    """Why a trial failed. 'wrong' and 'empty' need different fixes."""
    from speech import _is_degenerate
    if got == want:
        return "hit"
    if not (text or "").strip():
        return "empty"
    if _is_degenerate(text):
        return "degenerate"
    if got is None:
        return "no_match"
    return "wrong_part"


def _stream_clip(path, boost, lang, epd, parts, want):
    """Play one clip at real time into CLOVA streaming.

    Latency is measured from the end of the clip, not from the start of the
    call: a streaming engine can and should have answered before the audio
    finished, and that shows up here as a negative number. first_hit is when
    the matcher first landed on the right part, which is what the beam follows.
    """
    import clova
    from match import best_match
    from speech import FileSource
    src = FileSource(path, realtime=True)
    dur = src.duration
    hits, texts, first = [], [], None

    def on_text(t):
        el = src.elapsed()
        texts.append(t)
        hits.append(el)
        nonlocal first
        if first is None:
            r = best_match(" ".join(texts), parts)
            if r and r[0] == want:
                first = el - dur
    clova.live_stream(src, on_text, boost=clova.boostings(boost or []),
                      lang=lang or "ko", epd=epd)
    last = (hits[-1] - dur) if hits else float("nan")
    return " ".join(texts).strip(), last, first, dur


def cmd_run(parts_yaml="parts.yaml", engines="mlx:medium", repeats=3,
            conds=None, prompt_on=True, clips_dir=None, epd=None, limit=0):
    import soundfile as sf
    from match import load_parts, best_match
    from speech import ASR
    from app import alias_terms, build_prompt

    clips_dir = data_path(clips_dir or clips_dir_for(parts_yaml))
    labels = load_labels(clips_dir)
    if not labels:
        print(f"[!] {clips_dir}/labels.json 없음. 먼저:")
        print(f"    python3 bench_asr.py record --parts {parts_yaml}")
        return
    spec = load_spec(parts_yaml)
    parts = load_parts(spec)
    terms = alias_terms(spec)
    prompt = build_prompt(spec) if prompt_on else ""
    boost = terms if prompt_on else []

    items = [(fn, m) for fn, m in labels.items()
             if not conds or m["cond"] in conds]
    items.sort()
    if limit:
        items = items[:limit]
    engine_list = parse_engines(engines)
    total = len(items) * repeats * len(engine_list)
    print(f"[i] 클립 {len(items)} × 반복 {repeats} × 엔진 {len(engine_list)} = {total} 시행")
    print(f"[i] 프롬프트/부스팅 {'ON' if prompt_on else 'OFF'}  "
          f"조건 {sorted({m['cond'] for _, m in items})}")

    ensure_data()
    out_path = data_path(RUNS_CSV)
    new = not os.path.exists(out_path)
    fh = open(out_path, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(fh, RUN_FIELDS)
    if new:
        w.writeheader()

    done = 0
    for engine, model in engine_list:
        name = label_of(engine, model)
        asr = None
        if engine != "clova-stream":
            try:
                # Built once so model load time never lands in a latency figure.
                asr = ASR(engine, model, prompt=prompt, boost=boost)
            except Exception as e:
                print(f"[!] {name} 준비 실패: {e}")
                continue
        # Engines run one after another on purpose: CLOVA allows 15 concurrent
        # streams per domain, and parallelism would poison the latency numbers
        # anyway.
        for fn, meta in items:
            path = os.path.join(clips_dir, fn)
            want = meta["part"]
            for rep in range(repeats):
                try:
                    if engine == "clova-stream":
                        text, lat, first, dur = _stream_clip(
                            path, boost, model, epd, parts, want)
                    else:
                        a, _sr = sf.read(path, dtype="float32")
                        if a.ndim > 1:
                            a = a.mean(1)
                        dur = len(a) / 16000
                        t0 = time.monotonic()
                        text = asr.transcribe(a)
                        lat, first = time.monotonic() - t0, None
                except Exception as e:
                    print(f"  [!] {name} {fn}: {e}")
                    continue
                r = best_match(text, parts)
                got = r[0] if r else None
                w.writerow({
                    "ts": f"{time.time():.0f}", "clips": os.path.basename(clips_dir),
                    "clip": fn, "part": want, "cond": meta["cond"],
                    "speaker": meta.get("speaker", ""), "ref_text": meta.get("text", ""),
                    "engine": engine, "model": model or "", "prompt_on": int(prompt_on),
                    "repeat": rep, "latency_s": f"{lat:.4f}",
                    "first_hit_s": "" if first is None else f"{first:.4f}",
                    "audio_s": f"{dur:.3f}", "text": text, "matched": got or "",
                    "correct": int(got == want), "err": _classify(text, want, got)})
                done += 1
                if done % 20 == 0:
                    fh.flush()
                    print(f"    {done}/{total}")
        print(f"[i] {name} 완료")
    fh.close()
    try:
        import clova
        print(f"[i] CLOVA 소모: {clova.usage()}")
    except Exception:
        pass
    print(f"\n[i] → {out_path}\n    분석: python3 bench_report.py")


# ------------------------------------------------------------------ marks

def cmd_marks(wav, parts_yaml="parts.yaml", out="marks.csv", engine="mlx",
              model="large"):
    """Draft the ground-truth timeline for a recorded talk.

    Word timestamps from a local Whisper give the moment each part name stops
    being spoken; that is the zero point every reaction-latency number is
    measured from. Automatic only to save typing -- read the file and fix it,
    because every Tier B number inherits these times.
    """
    from match import load_parts, best_match
    parts = load_parts(load_spec(parts_yaml))
    path = data_path(wav)
    print(f"[i] {path} 단어 타임스탬프 추출 ({engine})...")
    words = []
    if engine == "mlx":
        import mlx_whisper
        from speech import MODEL_PRESETS
        r = mlx_whisper.transcribe(
            path, path_or_hf_repo=MODEL_PRESETS["mlx"].get(model, model),
            language="ko", word_timestamps=True)
        for seg in r["segments"]:
            words += [(x["word"], x["end"]) for x in seg.get("words", [])]
    else:
        from faster_whisper import WhisperModel
        segs, _ = WhisperModel(model, device="cpu", compute_type="int8").transcribe(
            path, language="ko", word_timestamps=True)
        for seg in segs:
            words += [(x.word, x.end) for x in (seg.words or [])]

    rows, last = [], {}
    for i, (wtxt, end) in enumerate(words):
        ctx = "".join(w for w, _ in words[max(0, i - 2):i + 1])
        for probe in (wtxt, ctx):
            m = best_match(probe, parts)
            if m and (end - last.get(m[0], -99) > 2.0):
                rows.append((round(end, 2), m[0]))
                last[m[0]] = end
                break
    out_path = data_path(out)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        cw = csv.writer(f)
        cw.writerow(["t_end_s", "part"])
        cw.writerows(rows)
    print(f"[i] {len(rows)}개 표시 → {out_path}")
    for t, p in rows[:15]:
        print(f"    {t:7.2f}s  {p}")
    print("\n⚠ 이 시각이 Tier B 지연의 기준점이다. 반드시 눈으로 확인하고 고칠 것.")


# ------------------------------------------------------------------- live

def parse_epd(spec: str | None):
    """'useWordEpd=false,gapThreshold=500' -> dict, for the EPD sweep."""
    if not spec:
        return None
    from clova import DEFAULT_EPD
    out = dict(DEFAULT_EPD)
    for kv in spec.split(","):
        k, _, v = kv.partition("=")
        k, v = k.strip(), v.strip()
        if not k:
            continue
        if v.lower() in ("true", "false"):
            out[k] = v.lower() == "true"
        else:
            out[k] = int(v)
    return out


def cmd_live(wav, parts_yaml="parts.yaml", engines="mlx:medium", marks=None,
             epd=None, silence_end=None, prompt_on=True, tag=""):
    """Feed a recorded talk through the live path at real time, per engine.

    This is the measurement the clip benchmark structurally cannot make. Clips
    presuppose that the utterance was already cut correctly; here the pipeline
    has to find the boundaries itself, which is where the silence wait, the
    server EPD, and every mid-sentence misfire actually live.
    """
    import speech
    from match import load_parts, load_links, Matcher
    from app import alias_terms, build_prompt

    spec = load_spec(parts_yaml)
    terms = alias_terms(spec)
    prompt = build_prompt(spec) if prompt_on else ""
    path = data_path(wav)

    ensure_data()
    out_path = data_path(LIVE_CSV)
    new = not os.path.exists(out_path)
    fh = open(out_path, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(fh, LIVE_FIELDS)
    if new:
        w.writeheader()

    if silence_end is not None:
        speech.SILENCE_END = silence_end
    cfg = tag or (json.dumps(epd, sort_keys=True) if epd else
                  (f"SILENCE_END={speech.SILENCE_END}" if silence_end else ""))

    for engine, model in parse_engines(engines):
        name = label_of(engine, model)
        print(f"\n[i] {name} — {os.path.basename(path)} 실시간 재생  {cfg}")
        matcher = Matcher(load_parts(spec), links=load_links(spec))
        src = speech.FileSource(path, realtime=True)
        rows = []

        def on_text(txt, _m=matcher, _s=src, _r=rows):
            before = _m.current
            pid = _m.update(txt)
            _r.append({"t_emit": _s.elapsed(), "text": txt, "part": pid or "",
                       "changed": int(pid != before)})
        try:
            speech.cmd_live(engine, on_text=on_text, prompt=prompt, model=model,
                            boost=terms if prompt_on else [], source=src, epd=epd)
        except Exception as e:
            print(f"  [!] {name}: {e}")
        for r in rows:
            w.writerow(dict(r, ts=f"{time.time():.0f}", wav=os.path.basename(path),
                            engine=engine, model=model or "", config=cfg,
                            t_emit=f"{r['t_emit']:.3f}"))
        fh.flush()
        print(f"  발화 {len(rows)}건 · 타겟 전환 {sum(r['changed'] for r in rows)}회")
    fh.close()
    if marks and not os.path.exists(data_path(marks)):
        print(f"[!] {marks} 없음 — bench_report.py 가 지연을 계산하려면 필요하다")
    print(f"\n[i] → {out_path}\n    분석: python3 bench_report.py")


# -------------------------------------------------------------------- cli

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["record", "augment", "run", "marks", "live"])
    ap.add_argument("--parts", default="parts.yaml")
    ap.add_argument("--clips", default=None, help="클립 디렉터리 (기본: parts 이름에서 유도)")
    ap.add_argument("--per-part", type=int, default=6)
    ap.add_argument("--sec", type=float, default=3.5)
    ap.add_argument("--speaker", default="")
    ap.add_argument("--snr", default="10,5,0")
    ap.add_argument("--noise", default=None, help="실측 소음 wav (16kHz). 없으면 합성")
    ap.add_argument("--engines", default="mlx:large,mlx:medium,clova,clova-stream",
                    help="쉼표 구분. 'mlx:large' · 'clova' · 전체 모델 id 도 가능")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--conds", default=None, help="예: clean,snr5")
    ap.add_argument("--limit", type=int, default=0, help="클립 수 제한 (연습용)")
    ap.add_argument("--no-prompt", dest="prompt_on", action="store_false",
                    help="프롬프트·부스팅을 끄고 델타 측정")
    ap.add_argument("--wav", default=None, help="live/marks 용 발표 녹음")
    ap.add_argument("--marks", default="marks.csv")
    ap.add_argument("--epd", default=None,
                    help="CLOVA EPD 덮어쓰기. 예: 'useWordEpd=false,gapThreshold=500'")
    ap.add_argument("--silence-end", type=float, default=None,
                    help="로컬 VAD 의 무음 대기(s). CLOVA gapThreshold 와 짝이 되는 축")
    ap.add_argument("--tag", default="", help="live 행에 붙일 설정 이름")
    a = ap.parse_args()

    if a.cmd == "record":
        cmd_record(a.parts, a.per_part, a.sec, a.speaker, a.clips)
    elif a.cmd == "augment":
        snrs = [float(x) for x in a.snr.split(",") if x.strip()]
        cmd_augment(a.parts, snrs, a.noise, a.clips)
    elif a.cmd == "run":
        conds = [c.strip() for c in a.conds.split(",")] if a.conds else None
        cmd_run(a.parts, a.engines, a.repeats, conds, a.prompt_on, a.clips,
                parse_epd(a.epd), a.limit)
    elif a.cmd == "marks":
        if not a.wav:
            raise SystemExit("[!] --wav 가 필요하다")
        cmd_marks(a.wav, a.parts, a.marks)
    elif a.cmd == "live":
        if not a.wav:
            raise SystemExit("[!] --wav 가 필요하다")
        cmd_live(a.wav, a.parts, a.engines, a.marks, parse_epd(a.epd),
                 a.silence_end, a.prompt_on, a.tag)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[i] 중단")
