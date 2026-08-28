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
| `bench_asr.py` | Compare ASR models by end-to-end hit rate |
| `probe.py` | What OpenCV actually reads from a video file (HDR debugging) |
| `test_match.py` | Regression test for matching and links |
| `launcher.py` | tkinter GUI for running `app.py` without typing flags |
| `autoregister.py` | Draft parts.yaml entries from OCR or object detection |
| `mine_vocab.py` | Draft aliases from the script or poster text, via Groq or OpenAI |

### Configuration

| | |
|---|---|
| `paths.py` | Resolves data file names into `data/` |
| `device.yaml` | Galvo limits: centre, reachable area, safe fence, blanking length |
| `libHeliosLaserDAC.dylib` | Helios DAC library, found automatically |

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
| `clips*/` | Recordings for `bench_asr.py` |

## Notes

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

**Speech engines.** `mlx` (Apple Silicon) and `faster` run locally; `groq`
uploads each segment to a cloud endpoint and needs `GROQ_API_KEY`. Model names
accept the shorthands `large`, `medium`, `small`, which resolve per engine.

**Latency.** The tuning knobs are in `speech.py`: `SILENCE_END` decides how long
to wait after speech before transcribing, `MAX_SEG` force-cuts long utterances.
Whisper inference dominates, so the model size matters far more than either.
