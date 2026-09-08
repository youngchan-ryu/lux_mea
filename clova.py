"""NAVER Cloud CLOVA Speech (NEST): real-time gRPC streaming, and the same
stream used one segment at a time.

    export CLOVA_SPEECH_SECRET=<장문 인식 Secret Key>

    python3 clova.py check test.wav      credentials, config, response shape
    python3 clova.py stream test.wav     feed a file at real-time speed

Two engines come out of one API. `transcribe()` opens a stream, pushes a whole
VAD segment, flushes with epFlag and closes -- that is `--engine clova`, a
drop-in for the Whisper path. `live_stream()` keeps the stream open for the
whole talk -- that is `--engine clova-stream`, where the segmentation is the
server's job and our 0.3 s silence wait disappears.

Doing both over gRPC (rather than the segment one over the sync HTTP endpoint)
means one credential, one config, and -- for the benchmark -- a comparison
where the *only* difference between the two engines is where the utterance
boundary is decided.

Streaming needs the Basic long-sentence plan; the Free plan does not serve it.
"""
from __future__ import annotations
import json
import os
import queue
import sys
import threading
import time

import numpy as np

SR = 16000
SECRET = os.environ.get("CLOVA_SPEECH_SECRET")
STREAM_HOST = os.environ.get("CLOVA_STREAM_HOST", "clovaspeech-gw.ncloud.com:50051")

# 100 ms per DATA frame. speech.py reads the mic in 512-sample (32 ms) blocks,
# which would be 31 gRPC messages a second for no benefit.
STREAM_CHUNK = 1600

# A stream that survives this long before ending was a healthy one -- the
# service expires streams by design ("Lifespan expired"), so that is a
# reconnect, not a failure. Dying faster than this, repeatedly, is a failure.
HEALTHY_STREAM_S = 10.0
MAX_RECONNECT = 3

# Recommended EPD for this project. We only need a part name to appear, not a
# well-formed sentence, so we let the server close a result on a word boundary
# instead of waiting for punctuation. gap_threshold is the server-side twin of
# speech.SILENCE_END.
DEFAULT_EPD = {"skipEmptyText": True, "useWordEpd": True,
               "usePeriodEpd": False, "gapThreshold": 300}

_USAGE = {"streams": 0, "audio_s": 0.0}
_CHANNEL = None
_PB = None


class ClovaError(RuntimeError):
    """Anything that means 'stop trying this engine' -- auth, plan, config."""


def usage() -> dict:
    """Audio actually sent, for tracking credit burn during a benchmark."""
    return dict(_USAGE)


# ---------------------------------------------------------------- stubs

def _stubs():
    """nest_pb2 / nest_pb2_grpc, generating them from nest.proto if needed.

    The generated files are build output, not source, so they go to _nest/ and
    stay untracked. Generating on first use keeps 'clone and run' working.
    """
    global _PB
    if _PB is not None:
        return _PB
    here = os.path.dirname(os.path.abspath(__file__))
    gen = os.path.join(here, "_nest")
    if not os.path.exists(os.path.join(gen, "nest_pb2_grpc.py")):
        os.makedirs(gen, exist_ok=True)
        proto = os.path.join(here, "nest.proto")
        if not os.path.exists(proto):
            raise ClovaError(f"nest.proto 없음: {proto}")
        print("[i] nest.proto → gRPC 스텁 생성 중...")
        try:
            from grpc_tools import protoc
        except ImportError:
            raise ClovaError(
                "grpc-tools 가 없어 스텁을 만들 수 없다.\n"
                "    pip install grpcio grpcio-tools")
        rc = protoc.main(["protoc", f"-I={here}", f"--python_out={gen}",
                          f"--grpc_python_out={gen}", proto])
        if rc != 0:
            raise ClovaError(f"protoc 실패 (rc={rc})")
    if gen not in sys.path:
        sys.path.insert(0, gen)
    import nest_pb2, nest_pb2_grpc          # noqa: E402
    _PB = (nest_pb2, nest_pb2_grpc)
    return _PB


def _channel():
    """One TLS channel for the process; streams are cheap on top of it.

    --engine clova opens a stream per utterance, so paying for the TLS
    handshake every time would show up directly in the latency we are trying
    to measure.
    """
    global _CHANNEL
    if _CHANNEL is None:
        import grpc
        _CHANNEL = grpc.secure_channel(STREAM_HOST, grpc.ssl_channel_credentials())
    return _CHANNEL


def secret():
    """Read at call time, not import time.

    speech.ASR checks the environment when the engine starts, so caching the
    value at import would let the two disagree -- the engine would announce
    itself as ready and then every stream would fail on a stale None.
    """
    return os.environ.get("CLOVA_SPEECH_SECRET") or SECRET


def _require_secret() -> str:
    s = secret()
    if not s:
        raise ClovaError(
            "CLOVA_SPEECH_SECRET 가 설정되지 않았다.\n"
            "    콘솔 > CLOVA Speech > 도메인 의 Secret Key 를 export 할 것")
    return s


# ---------------------------------------------------------------- config

def boostings(words, weight: float = 2.0) -> list[dict]:
    """Part aliases -> the keywordBoosting.boostings list.

    `words` is one comma-joined string per entry, not an array -- see the
    boostings table in the API reference. Weight runs 0..5.0 and 0 disables
    boosting for that entry.
    """
    terms = [w.strip() for w in words if w and w.strip()]
    if not terms:
        return []
    return [{"words": ",".join(terms), "weight": float(weight)}]


def config_json(lang: str = "ko", boost=None, epd=None) -> str:
    """The Config JSON sent as the first message of every stream."""
    cfg: dict = {"transcription": {"language": lang}}
    if boost:
        # boostings (typed in here) and boostingIDs (registered on the domain)
        # cannot be combined; parts.yaml changes too often to register groups.
        cfg["keywordBoosting"] = {"boostings": boost}
    cfg["semanticEpd"] = dict(DEFAULT_EPD if epd is None else epd)
    return json.dumps(cfg, ensure_ascii=False)


def _config_problems(d: dict) -> list[str]:
    """Failure strings from a config response, '' status meaning fine.

    A rejected boosting list is the dangerous one: recognition still works, it
    is just quietly worse, which would look like the model being bad.
    """
    out = []
    cfg = d.get("config")
    if not isinstance(cfg, dict):
        return out
    for name, node in [("config", cfg), ("keywordBoosting", cfg.get("keywordBoosting")),
                       ("forbidden", cfg.get("forbidden")),
                       ("semanticEpd", cfg.get("semanticEpd"))]:
        if isinstance(node, dict):
            st = node.get("status")
        elif name == "config":
            st = None
        else:
            continue
        if st and st != "Success":
            out.append(f"{name}: {st}")
    return out


def _response_types(d: dict) -> list:
    rt = d.get("responseType") or []
    return [rt] if isinstance(rt, str) else list(rt)


def _extract(d: dict):
    """A decoded response -> (text, ep_flag, kind), or (None, False, kind).

    The reference's Recognize response table names the keys config.text /
    config.epFlag while responseType says `transcription`, which reads like the
    Config section's table was copied over. Rather than bet on either, take the
    payload from whichever top-level object actually carries a `text`.
    """
    for key in ("transcription", "recognize", "config"):
        o = d.get(key)
        if isinstance(o, dict) and "text" in o:
            return (o.get("text") or ""), bool(o.get("epFlag")), key
    if "text" in d:
        return (d.get("text") or ""), bool(d.get("epFlag")), "root"
    return None, False, ",".join(_response_types(d)) or "?"


# ---------------------------------------------------------------- streaming

def _pcm(frame: np.ndarray) -> bytes:
    """float32 [-1,1] -> headerless 16 kHz mono 16-bit PCM, what NEST wants."""
    return (np.clip(frame, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def _requests(chunks, cfg: str, ready: threading.Event, stop_event, final_seq,
              failure):
    """CONFIG, then DATA frames, then a flush.

    Blocks on `ready` before the first DATA: sending audio before the config
    response comes back earns 'ConfigRequest did not complete'. gRPC pulls this
    generator on its own thread while the caller reads responses, so waiting
    here cannot deadlock the reader that sets the event.

    Give-up reasons go into `failure` rather than being raised here. gRPC runs
    this on its own thread and turns anything raised into a bare RpcError on
    the reader side, so a raise would lose the reason and, worse, downgrade a
    hard stop into just another reconnect.
    """
    pb, _ = _stubs()
    yield pb.NestRequest(type=pb.RequestType.CONFIG, config=pb.NestConfig(config=cfg))
    if not ready.wait(timeout=10.0):
        failure[0] = ClovaError("Config 응답이 10초 안에 오지 않았다")
        return
    seq = 0
    for chunk in chunks:
        if stop_event is not None and stop_event.is_set():
            break
        seq += 1
        _USAGE["audio_s"] += len(chunk) / SR
        yield pb.NestRequest(
            type=pb.RequestType.DATA,
            data=pb.NestData(chunk=_pcm(chunk),
                             extra_contents=json.dumps({"seqId": seq, "epFlag": False})))
    # epFlag flushes the server buffer instead of waiting out its 10 s idle
    # timer. seqId is documented as "better not 0".
    final_seq[0] = seq + 1
    yield pb.NestRequest(
        type=pb.RequestType.DATA,
        data=pb.NestData(chunk=b"",
                         extra_contents=json.dumps({"seqId": seq + 1, "epFlag": True})))


def _open_stream(chunks, cfg, stop_event, on_status=None, raw=None):
    """Run one recognize stream, yielding (text, ep) for each result.

    Raises ClovaError for anything that will not get better by retrying.
    """
    import grpc
    key = _require_secret()
    pb, pb_grpc = _stubs()
    stub = pb_grpc.NestServiceStub(_channel())
    meta = (("authorization", f"Bearer {key}"),)

    ready = threading.Event()
    final_seq = [None]
    failure = [None]
    _USAGE["streams"] += 1
    responses = stub.recognize(
        _requests(chunks, cfg, ready, stop_event, final_seq, failure),
        metadata=meta)
    try:
        for res in responses:
            if raw is not None:
                raw.append(res.contents)
            d = json.loads(res.contents)
            text, ep, _kind = _extract(d)
            if not ready.is_set():
                probs = _config_problems(d)
                if probs:
                    raise ClovaError("Config 거부됨 — " + "; ".join(probs))
                # The config acknowledgement carries no text; releasing the gate
                # on anything else too means a deployment that skips the ack
                # still gets its audio instead of timing out.
                ready.set()
                if text is None and "config" in _response_types(d):
                    if on_status:
                        on_status("config-ok", "")
                    continue
            if text:
                yield text, ep
        if failure[0] is not None:
            raise failure[0]
    except grpc.RpcError as e:
        if failure[0] is not None:
            raise failure[0]     # the RPC only died because we gave up on it
        code = e.code()
        detail = (e.details() or "").strip()
        if code in (grpc.StatusCode.UNAUTHENTICATED, grpc.StatusCode.PERMISSION_DENIED):
            raise ClovaError(
                f"인증 실패 ({code.name}): {detail}\n"
                "    CLOVA_SPEECH_SECRET 이 장문 인식 도메인의 Secret Key 인지 확인할 것")
        if code == grpc.StatusCode.UNIMPLEMENTED:
            raise ClovaError(
                f"스트리밍 미지원 ({detail}) — 실시간 스트리밍은 Basic 장문 인식 "
                "플랜에서만 열린다. 콘솔에서 요금제를 확인할 것")
        if code == grpc.StatusCode.RESOURCE_EXHAUSTED:
            raise ClovaError(
                f"동시 연결 한도 초과 ({detail}) — 도메인당 15개. "
                "벤치를 병렬로 돌리고 있지 않은지 확인할 것")
        raise
    finally:
        ready.set()          # never leave the request generator parked


def _chunker(source, stop_event=None):
    """Frames from an audio source, regrouped into STREAM_CHUNK samples."""
    buf = []
    n = 0
    for frame in source:
        if stop_event is not None and stop_event.is_set():
            break
        buf.append(frame)
        n += len(frame)
        while n >= STREAM_CHUNK:
            big = np.concatenate(buf)
            yield big[:STREAM_CHUNK]
            rest = big[STREAM_CHUNK:]
            buf, n = ([rest], len(rest)) if len(rest) else ([], 0)
    if n:
        yield np.concatenate(buf)


def live_stream(source, on_text, boost=None, lang="ko", epd=None,
                stop_event=None, on_status=None):
    """Microphone (or any AudioSource) -> continuous recognition -> on_text.

    Reconnects on its own. Stream lifespan is finite by design, so a stream
    that ran a while and then ended is a normal event, not an outage -- it is
    reopened silently. Only streams that keep dying immediately raise, which is
    what tells speech.LiveSession to fall back to a local engine.
    """
    cfg = config_json(lang, boost, epd)
    fails = 0
    while stop_event is None or not stop_event.is_set():
        t0 = time.monotonic()
        try:
            for text, _ep in _open_stream(_chunker(source, stop_event), cfg,
                                          stop_event, on_status):
                on_text(text)
            if getattr(source, "exhausted", False):
                return          # a FileSource ran out; nothing left to reopen for
        except ClovaError:
            raise                       # auth / plan / config: retrying is pointless
        except Exception as e:
            if stop_event is not None and stop_event.is_set():
                return
            lived = time.monotonic() - t0
            if lived < HEALTHY_STREAM_S:
                fails += 1
                if fails > MAX_RECONNECT:
                    raise ClovaError(f"스트림 재연결 {fails}회 실패 — {e}")
            else:
                fails = 0
            if on_status:
                on_status("reconnect", str(e))
        else:
            if stop_event is not None and stop_event.is_set():
                return
            lived = time.monotonic() - t0
            fails = 0 if lived >= HEALTHY_STREAM_S else fails + 1
            if fails > MAX_RECONNECT:
                raise ClovaError("스트림이 계속 즉시 종료된다")
            if on_status:
                on_status("reconnect", f"스트림 종료 ({lived:.0f}s) — 재연결")
        time.sleep(min(0.5 * (2 ** fails), 4.0) if fails else 0.2)


def transcribe(audio: np.ndarray, lang="ko", boost=None, epd=None) -> str:
    """One VAD segment -> text. The ASR.transcribe() contract.

    Opens a stream, pushes the segment as fast as the link allows, flushes with
    epFlag and joins whatever came back. The server may split one segment into
    several results, hence the join.
    """
    cfg = config_json(lang, boost, epd)
    chunks = [audio[i:i + STREAM_CHUNK] for i in range(0, len(audio), STREAM_CHUNK)]
    out = [t for t, _ep in _open_stream(iter(chunks), cfg, None)]
    return " ".join(s.strip() for s in out if s.strip()).strip()


# ---------------------------------------------------------------- self-test

def _read_wav(path):
    import soundfile as sf
    a, sr = sf.read(path, dtype="float32")
    if a.ndim > 1:
        a = a.mean(1)
    if sr != SR:
        raise SystemExit(f"[!] {path} 는 {sr}Hz — 16000Hz 로 변환해 쓸 것")
    return a


def cmd_check(path="test.wav", words=None):
    """Prove the credentials, the config and the response shape, in that order.

    Prints raw response frames. The reference's response table and the actual
    payload disagree about where `text` lives, so seeing one real frame settles
    it faster than reading the docs again.
    """
    print(f"[1] 자격증명    SECRET={'설정됨' if secret() else '없음'}  HOST={STREAM_HOST}")
    if not secret():
        print("    ✗ export CLOVA_SPEECH_SECRET=... 후 다시 실행")
        return 1
    print("[2] gRPC 스텁")
    try:
        _stubs()
        print("    ✅ nest_pb2 / nest_pb2_grpc 준비됨")
    except ClovaError as e:
        print(f"    ✗ {e}")
        return 1

    if not os.path.exists(path):
        print(f"[!] {path} 없음 — python3 speech.py rec 로 3초 녹음해 만들 것")
        return 1
    audio = _read_wav(path)
    boost = boostings(words or [])
    cfg = config_json("ko", boost)
    print(f"[3] Config JSON\n    {cfg}")

    print(f"[4] 인식 — {path} ({len(audio)/SR:.1f}s)")
    raw = []
    t0 = time.monotonic()
    chunks = [audio[i:i + STREAM_CHUNK] for i in range(0, len(audio), STREAM_CHUNK)]
    try:
        texts = [t for t, _ in _open_stream(iter(chunks), cfg, None, raw=raw)]
    except ClovaError as e:
        print(f"    ✗ {e}")
        return 1
    except Exception as e:
        print(f"    ✗ 예상 못한 오류 {type(e).__name__}: {e}")
        return 1
    dt = time.monotonic() - t0

    print(f"\n[5] 원본 응답 프레임 {len(raw)}개 — text/epFlag 위치를 여기서 확정할 것")
    for i, c in enumerate(raw):
        print(f"    [{i}] {c}")
    print(f"\n인식: {' '.join(texts)!r}")
    print(f"소요 {dt:.2f}s / 오디오 {len(audio)/SR:.2f}s")
    print(f"소모: {usage()}")
    return 0


def cmd_stream(path, words=None):
    """Feed a WAV at real-time speed, printing results as they land."""
    from speech import FileSource
    src = FileSource(path)
    t0 = time.monotonic()

    def on_text(t):
        print(f"  [{time.monotonic()-t0:6.2f}s] «{t}»")

    def on_status(kind, msg):
        print(f"  [i] {kind} {msg}")

    print(f"[i] {path} 실시간 재생 → 스트리밍 인식")
    stop = threading.Event()
    threading.Thread(target=lambda: (src.wait(), stop.set()), daemon=True).start()
    try:
        live_stream(src, on_text, boost=boostings(words or []),
                    stop_event=stop, on_status=on_status)
    except KeyboardInterrupt:
        pass
    print(f"소모: {usage()}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    arg = sys.argv[2] if len(sys.argv) > 2 else "test.wav"
    if cmd == "check":
        raise SystemExit(cmd_check(arg))
    elif cmd == "stream":
        cmd_stream(arg)
    else:
        print(__doc__)
