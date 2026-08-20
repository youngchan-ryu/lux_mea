"""Compare ASR models by how often the whole pipeline picks the right part.

Transcription accuracy is not the metric that matters -- the matcher absorbs a
lot of ASR error, so a model that transcribes worse but still lands on the
right part is not worse for us.

    python3 bench_asr.py record --parts parts.yaml
    python3 bench_asr.py run    --parts parts.yaml"""
from __future__ import annotations
import json
import os
import sys
import time
import numpy as np
import yaml
from paths import data_path

CLIPS = "clips"

DEFAULT_MODELS = [
    "mlx-community/whisper-large-v3-turbo",
    "mlx-community/whisper-medium-mlx",
    "mlx-community/whisper-small-mlx",
]


def clips_dir_for(parts_yaml: str) -> str:
    """parts_poster.yaml -> clips_poster/, so surfaces keep separate clips."""
    base = os.path.splitext(os.path.basename(parts_yaml))[0]
    if base.startswith("parts_"):
        return f"clips_{base[len('parts_'):]}"
    if base == "parts":
        return CLIPS
    return f"clips_{base}"


def cmd_record(parts_yaml="parts.yaml", per_part=2, sec=3.5, clips_dir=None):
    import sounddevice as sd, soundfile as sf
    clips_dir = data_path(clips_dir or clips_dir_for(parts_yaml))
    spec = yaml.safe_load(open(data_path(parts_yaml)))["parts"]
    os.makedirs(clips_dir, exist_ok=True)
    labels = {}
    print(f"[i] {parts_yaml} → {clips_dir}/")
    print(f"부위 {len(spec)}개 × {per_part}회. 실제 발표처럼 문장으로 말하세요.\n")
    for pid, p in spec.items():
        al = ", ".join(p.get("aliases", [])[:3])
        for k in range(per_part):
            input(f"  [{pid}] ({al}) {k+1}/{per_part} — Enter 후 말하세요")
            a = sd.rec(int(sec * 16000), samplerate=16000, channels=1,
                       dtype="float32"); sd.wait()
            fn = f"{pid}_{k}.wav"
            sf.write(os.path.join(clips_dir, fn), a.flatten(), 16000)
            labels[fn] = pid
            print(f"    저장 (peak={np.abs(a).max():.2f})")
    json.dump(labels, open(os.path.join(clips_dir, "labels.json"), "w"),
              ensure_ascii=False, indent=1)
    print(f"\n[i] {len(labels)}개 클립 저장 → python3 bench_asr.py run --parts {parts_yaml}")


def cmd_run(models, parts_yaml="parts.yaml", clips_dir=None):
    import soundfile as sf
    from match import load_parts, best_match
    from speech import ASR

    clips_dir = data_path(clips_dir or clips_dir_for(parts_yaml))
    labels_path = os.path.join(clips_dir, "labels.json")
    if not os.path.exists(labels_path):
        print(f"[!] {labels_path} 없음. 먼저 record 를 실행하세요:")
        print(f"    python3 bench_asr.py record --parts {parts_yaml}")
        return
    labels = json.load(open(labels_path))
    parts = load_parts(yaml.safe_load(open(data_path(parts_yaml)))["parts"])

    rows = []
    for m in models:
        try:
            asr = ASR("mlx" if "mlx" in m else "faster", m)
        except Exception as e:
            print(f"[!] {m} 로드 실패: {e}"); continue
        hit, tot, lats, misses = 0, 0, [], []
        for fn, want in labels.items():
            a, _ = sf.read(os.path.join(clips_dir, fn), dtype="float32")
            if a.ndim > 1:
                a = a.mean(1)
            t0 = time.time()
            try:
                txt = asr.transcribe(a)
            except Exception as e:
                print(f"  [!] {fn}: {e}"); continue
            lats.append(time.time() - t0)
            r = best_match(txt, parts)
            got = r[0] if r else None
            tot += 1
            if got == want:
                hit += 1
            else:
                misses.append((want, got, txt))
        if tot:
            rows.append((m, hit / tot, float(np.median(lats)), misses))
            print(f"\n{m}\n  적중 {hit}/{tot} ({hit/tot*100:.0f}%)  "
                  f"중앙 지연 {np.median(lats):.2f}s")
            for want, got, txt in misses[:5]:
                print(f"    ✗ {want} → {got}   전사={txt!r}")

    print("\n" + "=" * 62)
    print(f"{'모델':42s} {'적중률':>7s} {'지연':>7s}")
    for m, acc, lat, _ in rows:
        print(f"{m:42s} {acc*100:6.0f}% {lat:6.2f}s")
    print("=" * 62)
    print("적중률이 같다면 더 작은 모델을 쓴다 (지연·메모리 이득).")
    print("오답의 전사 결과를 보고 parts.yaml 별칭을 보강하는 것이")
    print("모델을 키우는 것보다 대개 효과가 크다.")


def _flag_list(flag):
    """Values after a flag, stopping at the next --flag."""
    if flag not in sys.argv:
        return None
    i = sys.argv.index(flag) + 1
    out = []
    while i < len(sys.argv) and not sys.argv[i].startswith("--"):
        out.append(sys.argv[i]); i += 1
    return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    ms = _flag_list("--models") or DEFAULT_MODELS
    parts_yaml = (_flag_list("--parts") or ["parts.yaml"])[0]
    if cmd == "record":
        cmd_record(parts_yaml)
    else:
        cmd_run(ms, parts_yaml)
