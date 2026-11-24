"""Fetch pinned Jittor checkpoints, initialization arrays and the upstream text encoder."""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
RELEASE = "https://github.com/Lipeilin0307/YOLOE-26s-Jittor/releases/download/v1.0.0/"
ASSETS = json.loads((ROOT / "weights/manifest.json").read_text(encoding="utf-8"))
PRESETS = {"train": ("det", "seg_init"), "demo": ("det", "TP", "text"),
           "TP": ("det", "TP"), "VP": ("det", "TP", "VP"), "PF": ("det", "TP", "PF"),
           "SEG": ("seg_init", "TP", "SEG"), "all": tuple(ASSETS)}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fetch(key, check):
    asset = ASSETS[key]
    target = ROOT / asset["path"]
    if target.is_file() and digest(target) == asset["sha256"]:
        print("OK", asset["path"], flush=True)
        return
    if check:
        raise SystemExit(f"Missing or different weights: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    url = asset.get("url", RELEASE + target.name)
    print("Downloading", url, flush=True)
    tmp = None
    try:
        with NamedTemporaryFile(dir=target.parent, suffix=".part", delete=False) as stream:
            tmp = Path(stream.name)
            with urlopen(Request(url, headers={"User-Agent": "YOLOE-Jittor-weights"}), timeout=90) as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    stream.write(block)
        if digest(tmp) != asset["sha256"]:
            raise ValueError(f"Checksum mismatch for {url}")
        tmp.replace(target)
        print("OK", asset["path"], flush=True)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preset", choices=PRESETS)
    parser.add_argument("--check", action="store_true", help="Verify local files without downloading")
    args = parser.parse_args()
    for key in PRESETS[args.preset]:
        fetch(key, args.check)
