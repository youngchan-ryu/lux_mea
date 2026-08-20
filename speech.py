"""Microphone -> VAD -> Whisper -> text.

    python3 speech.py devices            list audio devices
    python3 speech.py rec                record 3 s to test.wav
    python3 speech.py once test.wav      transcribe a file
    python3 speech.py live               realtime
    python3 speech.py live --model small large | medium | small

The VAD is a plain energy gate with hysteresis, calibrated from the first half
second of silence. No extra dependency, which matters more than accuracy here."""
from __future__ import annotations
import sys
import time
import numpy as np

SR = 16000
FRAME = 512
SILENCE_END = 0.3
MAX_SEG = 3.0
MIN_SEG = 0.20

MODEL_PRESETS = {
    "large": "mlx-community/whisper-large-v3-turbo",
    "medium": "mlx-community/whisper-medium-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "large-faster": "large-v3",
    "medium-faster": "medium",
    "small-faster": "small",
}


class EnergyVAD:
    """Energy gate with hysteresis, calibrated on the first 0.5 s."""

    def __init__(self, start_mult=3.0, keep_mult=1.6):
        self.noise = None
        self.start_mult, self.keep_mult = start_mult, keep_mult
        self._cal = []

    def calibrate(self, rms: float) -> bool:
        if len(self._cal) < int(0.5 * SR / FRAME):
            self._cal.append(rms)
            return False
        if self.noise is None:
            self.noise = float(np.median(self._cal)) + 1e-6
            print(f"[i] 배경 소음 기준: {self.noise:.5f}")
        return True

    def is_voice(self, rms: float, active: bool) -> bool:
        th = self.noise * (self.keep_mult if active else self.start_mult)
        return rms > th


class ASR:
    def __init__(self, engine="auto", model=None, prompt=""):
        model = MODEL_PRESETS.get(model, model)
        self.engine, self.model_name, self._m = engine, model, None
        self.prompt = prompt
        if engine == "auto":
            try:
                import mlx_whisper
                self.engine = "mlx"
            except ImportError:
                self.engine = "faster"
        if self.engine == "mlx":
            self.model_name = model or "mlx-community/whisper-medium-mlx"
        else:
            self.model_name = model or "medium"
        print(f"[i] ASR 엔진={self.engine} 모델={self.model_name}")

    def transcribe(self, audio: np.ndarray) -> str:
        if self.engine == "mlx":
            import mlx_whisper
            r = mlx_whisper.transcribe(
                audio, path_or_hf_repo=self.model_name, language="ko",
                initial_prompt=self.prompt, condition_on_previous_text=False)
            return r["text"].strip()
        else:
            from faster_whisper import WhisperModel
            if self._m is None:
                self._m = WhisperModel(self.model_name, device="cpu",
                                       compute_type="int8")
            segs, _ = self._m.transcribe(
                audio, language="ko", initial_prompt=self.prompt,
                condition_on_previous_text=False, vad_filter=False)
            return " ".join(s.text for s in segs).strip()


def cmd_devices():
    import sounddevice as sd
    print(sd.query_devices())
    print("\n기본 입력:", sd.default.device)


def cmd_rec(sec=3.0, path="test.wav"):
    import sounddevice as sd, soundfile as sf
    print(f"[i] {sec}초 녹음. 부위 이름을 또박또박 말하세요...")
    a = sd.rec(int(sec * SR), samplerate=SR, channels=1, dtype="float32")
    sd.wait()
    a = a.flatten()
    sf.write(path, a, SR)
    print(f"[i] {path} 저장. peak={np.abs(a).max():.3f} rms={np.sqrt((a**2).mean()):.4f}")
    if np.abs(a).max() < 0.02:
        print("  ⚠️ 신호가 거의 없음 — 마이크 권한/입력장치 확인")


def cmd_once(path, engine="auto"):
    import soundfile as sf
    a, sr = sf.read(path, dtype="float32")
    if a.ndim > 1:
        a = a.mean(1)
    asr = ASR(engine)
    t0 = time.time()
    txt = asr.transcribe(a)
    dt = time.time() - t0
    print(f"\n인식: {txt!r}")
    print(f"소요 {dt:.2f}s / 오디오 {len(a)/sr:.2f}s → RTF {dt/(len(a)/sr):.2f}")


def cmd_bench(path, engine="auto", n=3):
    import soundfile as sf
    a, sr = sf.read(path, dtype="float32")
    if a.ndim > 1:
        a = a.mean(1)
    asr = ASR(engine)
    asr.transcribe(a)
    ts = []
    for _ in range(n):
        t0 = time.time(); asr.transcribe(a); ts.append(time.time() - t0)
    dur = len(a) / sr
    print(f"중앙값 {np.median(ts):.2f}s (오디오 {dur:.2f}s) → RTF {np.median(ts)/dur:.2f}")
    print("RTF<0.3이면 실시간 여유 충분. >0.8이면 더 작은 모델로.")


def _is_degenerate(text: str, max_repeat_ratio: float = 0.5, min_tokens: int = 6) -> bool:
    """Catch the repeated-word hallucination Whisper produces on noise."""
    toks = text.split()
    if len(toks) < min_tokens:
        return False
    from collections import Counter
    _, n = Counter(toks).most_common(1)[0]
    return n / len(toks) >= max_repeat_ratio


def cmd_live(engine="auto", on_text=None, prompt="", model=None,
            noise_guard=1.8, stop_event=None):
    """Microphone -> VAD segments -> Whisper -> callback.

    noise_guard skips Whisper entirely when a segment never rose above the noise
    floor; that is where the repeated-word hallucinations come from. stop_event
    lets another thread stop this loop cleanly -- killing it inside PortAudio
    segfaults the interpreter.
    """
    import sounddevice as sd
    asr = ASR(engine, model=model, prompt=prompt)
    vad = EnergyVAD()
    buf, active, silence, seg_t0 = [], False, 0.0, None
    print("[i] 듣는 중. 부위 이름을 말하세요. Ctrl+C 종료.\n")

    def flush():
        nonlocal buf, active, silence, seg_t0
        if buf:
            audio = np.concatenate(buf)
            dur = len(audio) / SR
            seg_rms = float(np.sqrt((audio ** 2).mean()))
            if dur >= MIN_SEG:
                if noise_guard and vad.noise and seg_rms < vad.noise * noise_guard:
                    print(f"  (무시: 배경소음 수준  rms={seg_rms:.4f}  [세그 {dur:.1f}s])")
                else:
                    t0 = time.time()
                    txt = asr.transcribe(audio)
                    lat = time.time() - t0
                    if txt and _is_degenerate(txt):
                        print(f"  «{txt[:40]}…» 환각 의심 → 무시  [세그 {dur:.1f}s · 인식 {lat:.2f}s]")
                    elif txt:
                        print(f"  «{txt}»   [세그 {dur:.1f}s · 인식 {lat:.2f}s]")
                        if on_text:
                            on_text(txt)
        buf, active, silence, seg_t0 = [], False, 0.0, None

    with sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                        blocksize=FRAME) as stream:
        while stop_event is None or not stop_event.is_set():
            block, _ = stream.read(FRAME)
            x = block.flatten()
            rms = float(np.sqrt((x ** 2).mean()))
            if not vad.calibrate(rms):
                continue
            voiced = vad.is_voice(rms, active)
            if voiced:
                if not active:
                    active, seg_t0 = True, time.time()
                buf.append(x); silence = 0.0
            elif active:
                buf.append(x); silence += FRAME / SR
                if silence >= SILENCE_END:
                    flush()
            if active and seg_t0 and time.time() - seg_t0 > MAX_SEG:
                flush()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "devices"
    eng = "auto"
    if "--engine" in sys.argv:
        eng = sys.argv[sys.argv.index("--engine") + 1]
    mdl = None
    if "--model" in sys.argv:
        mdl = sys.argv[sys.argv.index("--model") + 1]
    try:
        if cmd == "devices":
            cmd_devices()
        elif cmd == "rec":
            cmd_rec()
        elif cmd == "once":
            cmd_once(sys.argv[2], eng)
        elif cmd == "bench":
            cmd_bench(sys.argv[2], eng)
        elif cmd == "live":
            cmd_live(eng, model=mdl)
        else:
            print(__doc__)
    except KeyboardInterrupt:
        print("\n[i] 종료")
