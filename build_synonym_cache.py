"""Build a Jittor raw-text cache by pooling distinct LVIS names within each category."""
import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import yaml

from lvis_tp import ASSETS, JITTOR_CPU_ENV, PROTOCOL, check_text, digest, load_text, read_json


def unit_vectors(vectors):
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise ValueError("Invalid text embedding norm")
    return vectors / norms


def build(args):
    manifest = read_json(args.manifest)
    baseline_path = args.manifest.with_name("text.npy")
    lock_path = args.output.with_name("." + args.output.name + ".lock")
    if manifest["protocol"] != PROTOCOL or manifest["asset_sha256"] != ASSETS:
        raise ValueError("Unexpected LVIS protocol or benchmark assets")
    if digest(args.vocab) != ASSETS["lvis.yaml"] or digest(baseline_path) != manifest["text_sha"]:
        raise ValueError("LVIS vocabulary or baseline text cache checksum mismatch")
    if manifest["text_backend"] != "jittor-mobileclip2-b" or digest(args.encoder) != manifest["text_encoder_sha"]:
        raise ValueError("Expected the baseline's native MobileCLIP2-B encoder")
    inputs = {path.resolve() for path in (args.manifest, baseline_path, args.vocab, args.encoder)}
    artifacts = {args.output.resolve(), args.output.with_suffix(".json").resolve(), lock_path.resolve()}
    if args.output.suffix != ".npy" or inputs & artifacts:
        raise ValueError("Choose a separate .npy and .json pair for the pooled cache and provenance")

    names = yaml.safe_load(args.vocab.read_text(encoding="utf-8"))["names"]
    if set(names) != set(range(PROTOCOL["classes"])):
        raise ValueError("LVIS vocabulary IDs differ from the evaluation classes")
    original = [names[i].split("/") for i in range(len(names))]
    if [group[0] for group in original] != manifest["names"] or not all(all(group) for group in original):
        raise ValueError("LVIS names are empty or disagree with the baseline class order")
    groups = [list(dict.fromkeys(group)) for group in original]
    baseline = load_text(baseline_path)
    result = baseline.copy()
    extra_names = list(dict.fromkeys(alias for group in groups for alias in group[1:]))

    if args.device == "cpu":
        os.environ.update(JITTOR_CPU_ENV)
    import jittor as jt
    from text_encoder import BPE_SHA, MobileCLIP2

    jt.flags.use_parallel_op_compiler = 0
    jt.flags.use_cuda = int(args.device == "cuda")
    encoded = MobileCLIP2(args.encoder).encode(extra_names, batch_size=args.batch_size)
    if encoded.shape != (1, len(extra_names), 512) or encoded.dtype != np.float32:
        raise ValueError("Unexpected MobileCLIP2-B extra-name output")
    alias_vectors = dict(zip(extra_names, encoded[0]))
    for i, group in enumerate(groups):
        if len(group) > 1:
            vectors = np.stack([baseline[0, i], *(alias_vectors[name] for name in group[1:])])
            result[0, i] = unit_vectors(unit_vectors(vectors).mean(axis=0))
    check_text(result)
    multi = np.array([len(group) > 1 for group in groups])
    if result[:, ~multi].tobytes() != baseline[:, ~multi].tobytes():
        raise ValueError("Single-name category vectors changed")

    report = dict(method="L2-normalize each distinct raw MobileCLIP2-B vector, take its class mean, then L2-normalize",
                  device=args.device, batch_size=args.batch_size,
                  manifest_sha=digest(args.manifest), vocab_sha=ASSETS["lvis.yaml"], baseline_sha=manifest["text_sha"],
                  encoder_sha=manifest["text_encoder_sha"], bpe_sha=BPE_SHA, classes=len(groups),
                  name_entries=sum(map(len, original)), distinct_within_class_entries=sum(map(len, groups)),
                  unique_extra_names_encoded=len(extra_names), pooled_classes=int(multi.sum()),
                  unchanged_single_name_classes=int((~multi).sum()), single_name_vectors_bit_identical=True,
                  duplicates_removed=[{"lvis_id": i+1, "names": group} for i, group in enumerate(original)
                                      if len(group) != len(groups[i])])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as tmp:
        cache = Path(tmp) / "text.npy"
        receipt = Path(tmp) / "provenance.json"
        np.save(cache, result, allow_pickle=False)
        report["text_sha"] = digest(cache)
        receipt.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        # Keep concurrent publishers from mixing their cache and provenance.
        with lock_path.open("a+b") as lock:
            if os.name == "nt":
                import msvcrt
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            cache.replace(args.output)
            receipt.replace(args.output.with_suffix(".json"))
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--manifest", type=Path, required=True, help="Native benchmark manifest next to its baseline text.npy")
    cli.add_argument("--vocab", type=Path, required=True, help="Pinned LVIS lvis.yaml")
    cli.add_argument("--encoder", type=Path, required=True, help="Pinned mobileclip2_b.ts")
    cli.add_argument("--output", type=Path, required=True, help="New .npy cache and its sibling .json provenance")
    cli.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    cli.add_argument("--batch-size", type=int, default=16)
    build(cli.parse_args())
