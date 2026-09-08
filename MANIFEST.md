# What's in here

Setup and file reference. For what the project is and why, see
[README.md](README.md).

## Setting up a new object

Four steps. Everything except the last needs the hardware connected.

**1. Mark out the surfaces.** The beam draws a rectangle you steer with the
keyboard; put it around whatever you want to be able to point at.

```bash
python3 firstlight.py aim --surfaces poster,mockup
```

```
1-9  pick surface     0  show all at once     f  step size 20 <-> 100
wasd move centre      [ ]  resize both        , .  width      - =  height
p    print values     v  save                 q  save and quit
```

Press `0` early. If both rectangles cannot be on their targets at the same
time, the galvo cannot reach both and no amount of software fixes it — move
things and try again. Results go to `surfaces.csv`.

Selecting the region on a camera image would be the obvious interface, but it
needs the image-to-galvo map, which is exactly what has not been measured yet.

**2. Capture the grid.** Camera on a tripod, exposure and focus locked, both
surfaces in frame. Start recording, then run one sweep per surface without
stopping the recording:

```bash
python3 firstlight.py grid --surface poster
python3 firstlight.py grid --surface mockup --append
```

`--append` on everything after the first. Without it the previous log is
overwritten and the shoot is wasted.

If a surface is a 3D object, sweep a flat board standing where its front face
will be, then swap the object in afterwards and film a few seconds more for
registration. A homography is a plane-to-plane map; it needs a plane to exist.
Tape the board's position on the floor first — that alignment is the accuracy.

**3. Fit.** Measure the grid on the surface with a ruler while `gridall` holds
it up, and pass what you measured:

```bash
python3 firstlight.py gridall --surface poster    # measure, then Ctrl+C
python3 pipeline.py IMG_4250.mov grid_log.csv --pitch-mm poster=275,mockup=50
```

`--pitch-mm` is the spacing between neighbouring points; `--grid-mm` is the full
width of the top row. On a 5x5 grid they differ by a factor of four. This only
affects the millimetre figures in the report — the pointing itself works in
pixels and DAC units.

Check the held-out RMS and the coverage box in `review_<surface>.png` before
moving on.

**4. Register.** Click each part on the plate, drag to size it:

```bash
python3 register.py plate.jpg --edit parts.yaml --surface poster
```

```
c r x e o l   circle rect crosshair ellipse roundrect underline
click         default size      click + drag  place and size at once
[ ]  resize   u  undo           s  save and quit      ESC  discard
```

Pass `--surface` every time. A part without a surface tag gets driven by
whichever calibration happens to be first, which puts the beam roughly 5-6°
off — close enough to look like a calibration problem rather than a missing
field. `app.py` checks this at startup and `test_match.py` fails on it.

To have two parts light up together, give one of them `link: [other_id]`.
Links are symmetric and transitive, and they do not depend on alias scores, so
editing the dictionary later never silently changes what lights up.

## Files

### Runtime — what runs while presenting

| | |
|---|---|
| `app.py` | Orchestrator: speech thread, matching, shapes, HUD, hotkeys |
| `match.py` | Text to part id. Jamo fuzzy matching, length-weighted scoring, link groups |
| `shapes.py` | Shape generation in image coordinates |
| `laser.py` | Output: per-surface maps and fences, blanking, simultaneous display |
| `calib.py` | Galvo/image mapping — homography plus polynomial residual |
| `speech.py` | Microphone, VAD, Whisper |

### Authoring — runs before the presentation, never during

| | |
|---|---|
| `firstlight.py` | Hardware checks, `aim`, `grid`, `gridall`, `sweep` |
| `pipeline.py` | Recording plus log to plate, pairs, per-surface calibration, quality report |
| `extract_dots.py` | Laser dot detection (used by `pipeline.py`) |
| `register.py` | Click parts on a photo to build `parts.yaml` |
| `bench_asr.py` | Collect ASR comparison data: clips, noise conditions, live runs |
| `bench_report.py` | Statistics and report over what `bench_asr.py` collected |
| `probe.py` | What OpenCV actually reads from a video file (HDR debugging) |
| `test_match.py` | Regression test for matching and links |
| `launcher.py` | tkinter GUI for running `app.py` without typing flags |
| `autoregister.py` | Draft parts.yaml entries from OCR or object detection |
| `mine_vocab.py` | Draft aliases from the script or poster text, via Groq or OpenAI |
| `usbprobe.py` | Helios USB diagnosis that stops before the call that panics macOS 26.6.2 |

### Configuration

| | |
|---|---|
| `paths.py` | Resolves data file names into `data/` |
| `clova.py` | CLOVA Speech client: gRPC streaming and per-segment recognition |
| `nest.proto` | Streaming interface definition, copied from the API reference |
| `device.yaml` | Galvo limits: centre, reachable area, safe fence, blanking length |
| `libHeliosLaserDAC.dylib` | Helios DAC library, found automatically |
| `libusb-1.0.0.dylib` | libusb 1.0.26, pinned. **Do not delete** — see Notes |

### Documentation — `docs/`

Images the README points at. Tracked, unlike `data/`. The editable sources the
posters were exported from (`.pptx`, `.pdf`, the circuit `.svg`) are kept
outside the repository.

| | |
|---|---|
| `poster_en.png`, `poster_kr.png` | Exhibition poster, both editions |
| `circuits_en.png`, `circuits_kr.png` | Drive circuit diagram |
| `hero.jpg` | The photo at the top of the README |

### Data — `data/`, not tracked

Everything below is produced by the setup steps above and is specific to one
object in one room, so it is gitignored. A fresh clone has no `data/`; run the
four steps to create it, or set `LUXMEA_DATA` to a folder you already have.

Bare filenames on the command line resolve into `data/`, so `--parts parts.yaml`
finds `data/parts.yaml`. Pass a path with a separator in it to escape that.

| | |
|---|---|
| `parts.yaml` | The dictionary: aliases, anchors, shapes, `surface:`, `link:` |
| `surfaces.csv` | Aim regions per surface |
| `grid_log.csv` | Galvo coordinates logged during capture |
| `pairs_*.csv` | Galvo/image correspondences per surface |
| `calib_*.json` | Fitted maps, image size, galvo bounds |
| `plate.jpg` | Registration background — the per-pixel median of the capture |
| `review_*.png` | Detected points drawn on the plate, for checking the fit |
| `session_log.json` | What was heard and what it matched, written on exit |
| `profiles.json` | Launcher's last selection |
| `clips*/` | Recordings for `bench_asr.py`, plus their noise-mixed variants |
| `bench_runs.csv` | One row per clip x engine x condition x repeat |
| `bench_live.csv` | Utterances and target changes from a real-time run |
| `marks.csv` | When each part name finished being spoken, for latency |
| `bench_report.md` | Generated by `bench_report.py` |

## Notes

**The pinned libusb.** `libusb-1.0.0.dylib` next to the DAC library is not a
stray build artefact. Deleting it makes macOS 26.6.2 kernel panic — the whole
machine reboots — the moment `firstlight.py` or `app.py` opens the DAC.

`libHeliosLaserDAC.dylib` asks for `@rpath/libusb-1.0.0.dylib` and carries
`LC_RPATH @loader_path`, so dyld looks in this directory first. With the slot
empty it walks up to the main executable's rpaths and lands on Homebrew's
libusb 1.0.30, which calls an IOUSBLib method that the 26.6.2 kernel gets wrong
for this device's descriptor. libusb 1.0.26 never makes that call. Keeping the
1.0.26 that ships in the Helios SDK here fills the slot the way its author
intended and takes Homebrew out of the picture entirely.

Run `python3 usbprobe.py` to see which libusb actually gets loaded. It stops
before the call that panics, so it is safe to run. `handoff/MACOS-USB-PANIC.md`
has the full diagnosis.

**Changing the hardware.** `device.yaml` holds everything geometry-dependent.
The reachable area was measured by walking the beam to the edges; the safe box
is the largest axis-aligned rectangle inside it. Note the centre of that box is
not the same as the galvo's neutral position, so a symmetric amplitude cannot
use the whole range.

**Blanking length.** `blank_points` in `device.yaml` is how many dark points to
insert between shapes. The laser module takes 0.1–1 ms to actually go dark, so
24 points at 20 kpps (1.2 ms) is the safe setting. Lowering it buys refresh
rate; too low and a faint line appears between the shapes.

**Simulator.** `--sim` swaps the DAC for a window and keeps the same interface,
including the per-surface maps and fences, so most work needs no hardware.

**Launcher.** `launcher.py` needs tkinter, which a pyenv build without tcl-tk
does not have. It handles that by running itself on whatever Python can import
tkinter while starting `app.py` with the virtualenv it finds nearby, so the two
can be different interpreters. Override with `APP_PYTHON` if that guess is
wrong. When the selected parts file carries `surface:` tags it stops passing
`--calib`, since the calibration then comes from the tags.

**Speech engines.** `mlx` (Apple Silicon) and `faster` run locally; `groq` and
`clova` upload each segment to a cloud endpoint, and `clova-stream` keeps a
connection open for the whole talk. Model names accept the shorthands `large`,
`medium`, `small`, which resolve per engine; for the CLOVA engines that slot
carries the language instead (`ko`, `en`, `ja`).

Cloud engines need a key in the environment: `GROQ_API_KEY` for Groq,
`CLOVA_SPEECH_SECRET` for both CLOVA engines. Streaming additionally needs the
Basic long-sentence plan — the Free plan does not serve it. Check the setup
with `python3 clova.py check test.wav` before trusting it in a demo; it
separates a plan problem from an auth problem from a rejected config, and
prints the raw response frames.

`clova` and `clova-stream` talk to the same API with the same model and the
same keyword boosting. The only difference is who decides where an utterance
ends: our VAD, or the server. That is deliberate — it is what makes the
benchmark measure segmentation rather than two different models.

**Switching engines mid-demo.** A cloud engine will fail eventually, and
stopping the demo to restart is not an option. `e` in the HUD toggles between
`--engine` and `--fallback-engine`, and a cloud engine that gives up hands over
to the local one by itself, saying so in the HUD. It does not switch back on
its own: flapping wifi would otherwise change the engine under the presenter
every few seconds, so recovery is a keypress. A stream ending because its
lifespan expired is not a failure and reconnects silently.

**Latency.** The tuning knobs are in `speech.py`: `SILENCE_END` decides how long
to wait after speech before transcribing, `MAX_SEG` force-cuts long utterances.
Whisper inference dominates, so the model size matters far more than either.
The streaming engine has neither knob; the server-side equivalents go in
`clova.DEFAULT_EPD` (`gapThreshold` is the twin of `SILENCE_END`, and
`useWordEpd` lets a result close on a word instead of a sentence, which is all
we need to catch a part name). `bench_asr.py live --epd/--silence-end` sweeps
both so the comparison is fair.

**Comparing engines.** `bench_asr.py` collects, `bench_report.py` analyses:

```bash
python3 bench_asr.py record  --parts parts_poster.yaml --per-part 6
python3 bench_asr.py augment --parts parts_poster.yaml --snr 10,5,0
python3 bench_asr.py run     --parts parts_poster.yaml --repeats 3 \
        --engines "mlx:large,mlx:medium,clova,clova-stream"
python3 bench_asr.py marks   --wav pres.wav --parts parts_poster.yaml
python3 bench_asr.py live    --wav pres.wav --marks marks.csv \
        --parts parts_poster.yaml --engines "mlx:medium,clova-stream"
python3 bench_report.py
```

Clip accuracy and reaction latency are different questions and the two halves
answer them separately. Clips presuppose the utterance was already cut
correctly; `live` replays a recorded talk through the live path at real time
and measures how long after a part name is spoken the beam actually moves, plus
how often it moves somewhere wrong mid-sentence. Streaming can only win there.

Sample size is the point of the rewrite. The ten clips behind
`bench_asr_result.md` cannot distinguish 70% from 40% — the paired test puts
that at p ≈ 0.25 — which is why the same model scored both. Eight parts times
six utterances across four noise conditions gives ~190 paired trials per
engine, about seven minutes of recording, and the noise conditions are mixed
offline so every engine hears byte-identical audio.
