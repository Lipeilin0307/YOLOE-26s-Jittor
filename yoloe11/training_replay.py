"""Freeze the original 11s smoke loader's outputs to replay them across Python environments."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from native_data import NativeWorker


def update_digest(digest, batch):
    for key in sorted(batch):
        value = np.ascontiguousarray(batch[key])
        digest.update(f"{key}:{value.shape}:{value.dtype}".encode())
        digest.update(value.tobytes())


def freeze(data, run, output, seed=None):
    previous = json.loads(run.read_bytes())
    seed = previous["seed"] if seed is None else seed
    cfg = dict(previous["config"], data=str(data), stage="TP", workers=2, smoke=True, epochs=1, seed=seed)
    if (cfg["size"], cfg["batch"]) != (320, 4):
        raise ValueError("Expected size 320, batch 4")
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "replay.json"
    if manifest.exists() and json.loads(manifest.read_bytes())["train"].get("seed", 20261121) != seed:
        raise ValueError("Use a separate replay directory for each training seed")
    worker = NativeWorker(cfg)
    record = {}
    try:
        for split in ("train", "val"):
            digest = hashlib.sha256()
            count = 0
            for i, batch in enumerate(worker.batches(split, 0, 200)):
                np.savez(output / f"{split}-{i:02}.npz", **batch)
                update_digest(digest, batch)
                count += len(batch["img"])
            if count != 200:
                raise ValueError(f"Incomplete {split} split")
            record[split] = dict(images=count, steps=i+1, arrays_sha256=digest.hexdigest(),
                                 manifest_sha256=cfg[split + "_sha"])
            if split == "train":
                record[split]["seed"] = seed
        manifest.write_text(json.dumps(record, indent=2) + "\n")
    finally:
        worker.close()


def batches(root, split):
    meta = json.loads((root / "replay.json").read_bytes())[split]
    digest = hashlib.sha256()
    for i in range(meta["steps"]):
        with np.load(root / f"{split}-{i:02}.npz") as f:
            batch = dict(f)
        update_digest(digest, batch)
        yield batch
    if digest.hexdigest() != meta["arrays_sha256"]:
        raise ValueError(f"Changed {split} replay arrays")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("data", "run", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    freeze(args.data, args.run, args.out, args.seed)
