"""Microphone -> VAD -> Whisper -> text.

    python3 speech.py devices            list audio devices
    python3 speech.py rec                record 3 s to test.wav
    python3 speech.py once test.wav      transcribe a file
    python3 speech.py live               realtime
    python3 speech.py live --model small       large | medium | small
    python3 speech.py live --engine groq       cloud fallback, needs GROQ_API_KEY
    python3 speech.py live --engine clova-stream   CLOVA Speech, server-side EPD

The VAD is a plain energy gate with hysteresis, calibrated from the first half
second of silence. No extra dependency, which matters more than accuracy here.

`clova-stream` does not use that VAD at all: the audio goes to the server
continuously and the server decides where the utterance ends, which is the
whole point of it -- our 0.3 s silence wait disappears from the latency."""
from __future__ import annotations
import os
import sys
import threading
import time
import numpy as np

SR = 16000
FRAME = 512
SILENCE_END = 0.3
MAX_SEG = 3.0
MIN_SEG = 0.20

MODEL_PRESETS = {
    "mlx": {"large": "mlx-community/whisper-large-v3-turbo",
            "medium": "mlx-community/whisper-medium-mlx",
            "small": "mlx-community/whisper-small-mlx"},
    "faster": {"large": "large-v3", "medium": "medium", "small": "small"},
    "groq": {"large": "whisper-large-v3", "medium": "whisper-large-v3-turbo",
             "small": "whisper-large-v3-turbo"},
    # CLOVA has one model; the `model` slot carries the language instead, so
    # the shorthands fall through untouched.
    "clova": {},
    "clova-stream": {},
}

DEFAULT_MODEL = {"mlx": "mlx-community/whisper-medium-mlx",
                 "faster": "medium",
                 "groq": "whisper-large-v3-turbo",
                 "clova": "ko",
                 "clova-stream": "ko"}

CLOUD_ENGINES = ("groq", "clova", "clova-stream")


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
    """mlx and faster run locally; groq and clova are cloud engines.

    Local is the default. A cloud engine adds a network dependency, so it is
    there for when a noisy room defeats the local model, not as the normal
    path. Groq only offers whisper-large-v3 and its turbo distillation, and our
    metric is match hit rate rather than WER, so pick between them with
    bench_asr.py.

    `prompt` and `boost` are the same vocabulary in two shapes: Whisper takes a
    sentence as its decoder prompt, CLOVA takes a keyword-boosting list. Both
    come from app.alias_terms(), so the engines are fed the same words.
    """

    GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

    def __init__(self, engine="auto", model=None, prompt="", boost=None):
        self.prompt, self._m = prompt, None
        self.boost = boost
        self.engine = engine
        if engine == "auto":
            try:
                import mlx_whisper
                self.engine = "mlx"
            except ImportError:
                self.engine = "faster"
        presets = MODEL_PRESETS.get(self.engine, {})
        self.model_name = presets.get(model, model) or DEFAULT_MODEL[self.engine]
        if self.engine == "groq" and not os.environ.get("GROQ_API_KEY"):
            raise RuntimeError("GROQ_API_KEY is not set")
        if self.engine.startswith("clova") and not os.environ.get("CLOVA_SPEECH_SECRET"):
            raise RuntimeError("CLOVA_SPEECH_SECRET is not set")
        print(f"[i] ASR 엔진={self.engine} 모델={self.model_name}")

    def _clova_boost(self):
        import clova
        return clova.boostings(self.boost or [])

    def _groq(self, audio: np.ndarray) -> str:
        """numpy -> WAV in memory -> multipart upload (OpenAI-compatible)."""
        import io
        import requests
        import soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, audio, SR, format="WAV", subtype="PCM_16")
        buf.seek(0)
        r = requests.post(
            self.GROQ_URL,
            headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"},
            files={"file": ("seg.wav", buf, "audio/wav")},
            data={"model": self.model_name, "language": "ko",
                  "prompt": self.prompt, "response_format": "json",
                  "temperature": "0"},
            timeout=15,
        )
        r.raise_for_status()
        return (r.json().get("text") or "").strip()

    def transcribe(self, audio: np.ndarray) -> str:
        if self.engine == "groq":
            return self._groq(audio)
        if self.engine.startswith("clova"):
            import clova
            return clova.transcribe(audio, lang=self.model_name,
                                    boost=self._clova_boost())
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


class MicSource:
    """Microphone blocks of FRAME float32 samples, forever.

    The stream stays open across iterations so a cloud engine can reopen its
    connection without dropping audio or re-arming PortAudio.
    """

    exhausted = False

    def __init__(self, sr=SR, frame=FRAME):
        self.sr, self.frame = sr, frame
        self._stream, self._t0 = None, None

    def _open(self):
        if self._stream is None:
            import sounddevice as sd
            self._stream = sd.InputStream(samplerate=self.sr, channels=1,
                                          dtype="float32", blocksize=self.frame)
            self._stream.start()
            self._t0 = time.monotonic()

    def __iter__(self):
        self._open()
        while True:
            block, _ = self._stream.read(self.frame)
            yield block.flatten()

    def elapsed(self) -> float:
        return 0.0 if self._t0 is None else time.monotonic() - self._t0

    def close(self):
        if self._stream is not None:
            self._stream.stop(); self._stream.close()
            self._stream = None


class FileSource:
    """A WAV fed through the live path at wall-clock speed.

    This is what makes end-to-end latency measurable. Timing the live path
    needs to know exactly when a word was spoken, which a microphone cannot
    tell us; playing a file at real-time speed means the answer is the file
    position, every engine sees byte-identical input at identical times, and
    no speaker or quiet room is involved.

    realtime=False drops the pacing, for feeding a clip to a batch engine as
    fast as it will go.
    """

    def __init__(self, path, frame=FRAME, realtime=True):
        import soundfile as sf
        a, sr = sf.read(path, dtype="float32")
        if a.ndim > 1:
            a = a.mean(1)
        if sr != SR:
            raise ValueError(f"{path}: {sr}Hz — 16000Hz 로 변환해 쓸 것")
        self.audio, self.frame, self.realtime = a, frame, realtime
        self.exhausted = False
        self._t0 = None
        self._pos = 0
        self._done = threading.Event()

    @property
    def duration(self) -> float:
        return len(self.audio) / SR

    def __iter__(self):
        """Resumes where the last iteration stopped; never replays.

        A streaming engine reopens its connection by iterating the source
        again, and restarting the file there would resend audio already sent
        and reset the clock elapsed() is measured against -- so the benchmark
        would quietly record nonsense latency for the rest of the run.
        """
        if self._t0 is None:
            self._t0 = time.monotonic()
        while self._pos < len(self.audio):
            block = self.audio[self._pos:self._pos + self.frame]
            self._pos += len(block)
            if self.realtime:
                # Sleep to the frame's end time rather than for its duration,
                # so scheduling jitter cannot accumulate into clock drift.
                lag = self._pos / SR - (time.monotonic() - self._t0)
                if lag > 0:
                    time.sleep(lag)
            yield block
        self.exhausted = True
        self._done.set()

    def elapsed(self) -> float:
        return 0.0 if self._t0 is None else time.monotonic() - self._t0

    def wait(self, timeout=None) -> bool:
        return self._done.wait(timeout)

    def close(self):
        pass


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


def cmd_once(path, engine="auto", model=None):
    import soundfile as sf
    a, sr = sf.read(path, dtype="float32")
    if a.ndim > 1:
        a = a.mean(1)
    asr = ASR(engine, model)
    t0 = time.monotonic()
    txt = asr.transcribe(a)
    dt = time.monotonic() - t0
    print(f"\n인식: {txt!r}")
    print(f"소요 {dt:.2f}s / 오디오 {len(a)/sr:.2f}s → RTF {dt/(len(a)/sr):.2f}")


def cmd_bench(path, engine="auto", n=3, model=None):
    import soundfile as sf
    a, sr = sf.read(path, dtype="float32")
    if a.ndim > 1:
        a = a.mean(1)
    asr = ASR(engine, model)
    asr.transcribe(a)
    ts = []
    for _ in range(n):
        t0 = time.monotonic(); asr.transcribe(a); ts.append(time.monotonic() - t0)
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
            noise_guard=1.8, stop_event=None, boost=None, source=None,
            on_status=None, epd=None):
    """Audio source -> text -> callback. Two ways of getting there.

    `clova-stream` hands the whole stream to the server and lets its EPD decide
    where utterances end; everything else runs the local VAD and transcribes a
    segment at a time.

    noise_guard skips Whisper entirely when a segment never rose above the noise
    floor; that is where the repeated-word hallucinations come from. stop_event
    lets another thread stop this loop cleanly -- killing it inside PortAudio
    segfaults the interpreter.
    """
    own_source = source is None
    source = source or MicSource()
    try:
        if engine == "clova-stream":
            return _live_clova(source, on_text, boost, model, stop_event,
                               on_status, epd)
        return _live_vad(source, engine, on_text, prompt, model, boost,
                         noise_guard, stop_event)
    finally:
        if own_source:
            source.close()


def _live_clova(source, on_text, boost, model, stop_event, on_status, epd):
    """Continuous streaming recognition. No local VAD, no segment wait."""
    import clova
    lang = model or DEFAULT_MODEL["clova-stream"]
    print(f"[i] CLOVA 스트리밍 (언어={lang}). 말하세요. Ctrl+C 종료.\n")

    def emit(txt):
        txt = (txt or "").strip()
        if txt:
            print(f"  «{txt}»")
            if on_text:
                on_text(txt)

    clova.live_stream(source, emit, boost=clova.boostings(boost or []),
                      lang=lang, epd=epd, stop_event=stop_event,
                      on_status=on_status)


def _live_vad(source, engine, on_text, prompt, model, boost, noise_guard,
              stop_event):
    asr = ASR(engine, model=model, prompt=prompt, boost=boost)
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
                    # monotonic, not time.time(): an NTP correction or a lid
                    # closed mid-segment would otherwise print a latency in
                    # the minutes, and latency is the number we report.
                    t0 = time.monotonic()
                    txt = asr.transcribe(audio)
                    lat = time.monotonic() - t0
                    if txt and _is_degenerate(txt):
                        print(f"  «{txt[:40]}…» 환각 의심 → 무시  [세그 {dur:.1f}s · 인식 {lat:.2f}s]")
                    elif txt:
                        print(f"  «{txt}»   [세그 {dur:.1f}s · 인식 {lat:.2f}s]")
                        if on_text:
                            on_text(txt)
        buf, active, silence, seg_t0 = [], False, 0.0, None

    for x in source:
        if stop_event is not None and stop_event.is_set():
            break
        rms = float(np.sqrt((x ** 2).mean()))
        if not vad.calibrate(rms):
            continue
        voiced = vad.is_voice(rms, active)
        if voiced:
            if not active:
                active, seg_t0 = True, time.monotonic()
            buf.append(x); silence = 0.0
        elif active:
            buf.append(x); silence += len(x) / SR
            if silence >= SILENCE_END:
                flush()
        if active and seg_t0 and time.monotonic() - seg_t0 > MAX_SEG:
            flush()
    # A file source ends mid-utterance, so the tail still has to be
    # transcribed -- but not when we are shutting down: that would put one
    # more cloud round trip in front of a 3 s join that is already ticking.
    if stop_event is None or not stop_event.is_set():
        flush()


class LiveSession:
    """Runs cmd_live on a worker thread and swaps engines without restarting.

    A cloud engine at a booth will fail sooner or later -- the venue wifi, or
    just the stream's own lifespan -- and stopping the demo to restart the app
    is not an option. So the engine is switchable while running: `e` in the HUD
    toggles primary <-> fallback, and a cloud engine that gives up hands over
    to the local one by itself.

    Two rules make that safe rather than annoying:

    * A worker never restarts itself. It records why it died and exits; poll(),
      called from the main loop, starts the replacement. A thread that joined
      itself would deadlock, and this way the switch happens on one known
      thread.
    * Falling back is automatic, going back to the cloud is not. Flapping wifi
      would otherwise have the engine changing under the presenter every few
      seconds; recovery is a decision, so it is a keypress.

    Stopping goes through the same stop_event as shutdown. Killing a thread
    parked inside PortAudio segfaults the interpreter.
    """

    def __init__(self, on_text, prompt="", boost=None, primary="auto",
                 primary_model=None, fallback="auto", fallback_model=None):
        self.on_text, self.prompt, self.boost = on_text, prompt, boost
        self.primary, self.primary_model = primary, primary_model
        self.fallback, self.fallback_model = fallback, fallback_model
        self.engine = primary
        self.note = ""
        self._thread = None
        self._stop = threading.Event()
        self._failed = None

    # -- lifecycle --------------------------------------------------------
    def start(self):
        self._spawn(self.primary)
        return self

    def stop(self):
        self._halt()

    def _model_for(self, engine):
        return (self.primary_model if engine == self.primary
                else self.fallback_model)

    def _spawn(self, engine):
        self.engine = engine
        self._failed = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run,
                                        args=(engine, self._stop), daemon=True)
        self._thread.start()

    def _halt(self):
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=3.0)
        if self._thread.is_alive():
            print("[!] 음성 스레드가 3초 안에 안 끝남 — 그대로 두고 진행")
        self._thread = None

    def _run(self, engine, stop):
        def on_status(kind, msg):
            if kind == "reconnect":
                # Stream lifespan is finite by design, so this is routine.
                print(f"[i] {engine} 재연결 — {msg}")

        try:
            cmd_live(engine, on_text=self.on_text, prompt=self.prompt,
                     model=self._model_for(engine), boost=self.boost,
                     stop_event=stop, on_status=on_status)
        except Exception as e:
            if not stop.is_set():
                self._failed = (engine, e)
                print(f"[!] 음성 엔진 {engine} 중단: {e}")

    # -- control ----------------------------------------------------------
    def toggle(self):
        """primary <-> fallback. Reopening the cloud engine reconnects it."""
        nxt = self.fallback if self.engine == self.primary else self.primary
        if nxt == self.engine:
            return self.engine
        self._halt()
        self.note = ""
        print(f"[i] 음성 엔진 전환 → {nxt}")
        self._spawn(nxt)
        return nxt

    def poll(self):
        """Call every frame from the main loop. Recovers a dead worker."""
        if self._thread is None or self._thread.is_alive() or self._failed is None:
            return
        engine, err = self._failed
        self._failed = None
        self._thread = None
        if engine != self.fallback:
            self.note = f"⚠ {engine} 끊김 — e 로 복귀"
            print(f"[!] {engine} 실패 → {self.fallback} 로 전환. 복귀는 e 키.")
            self._spawn(self.fallback)
        else:
            self.note = f"✗ {engine} 실패 ({str(err)[:40]})"

    @property
    def status(self) -> str:
        """One line for the HUD."""
        return f"{self.engine}  {self.note}".strip()


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
            cmd_once(sys.argv[2], eng, mdl)
        elif cmd == "bench":
            cmd_bench(sys.argv[2], eng, model=mdl)
        elif cmd == "live":
            cmd_live(eng, model=mdl)
        else:
            print(__doc__)
    except KeyboardInterrupt:
        print("\n[i] 종료")
