"""Where runtime data lives.

Code stays in the repo; everything a session produces or consumes -- captures,
calibrations, dictionaries, logs -- lives in data/ and is not tracked. Set
LUXMEA_DATA to point somewhere else, e.g. one folder per demo object.

Bare filenames resolve into that directory; anything with a path separator or
an absolute path is used as given, so command-line arguments still work.
"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("LUXMEA_DATA") or os.path.join(ROOT, "data")


def data_path(name):
    """Resolve a data file name to a full path."""
    if not name:
        return name
    if os.path.isabs(name) or os.path.dirname(name):
        return name
    return os.path.join(DATA, name)


def data_glob(pattern):
    """glob inside the data directory, returning bare filenames."""
    import glob
    return sorted(os.path.basename(p)
                  for p in glob.glob(os.path.join(DATA, pattern)))


def ensure_data():
    os.makedirs(DATA, exist_ok=True)
    return DATA
