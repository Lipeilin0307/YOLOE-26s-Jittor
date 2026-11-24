"""Export the published YOLOE-26s detector tensors and an independent PyTorch LVIS text reference."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
import ultralytics
import yaml
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.nn.text_model import MobileCLIPTS

from export import graph


CHECKPOINT_SHA = "48f24206bc8680d60cbbfa296b0140da849669b9515058b72f5a945142df0654"
ENCODER_SHA = "35d7f213e4d75f38514e4656ad3cb91158bd33e3805d8ac349f23b186f66982f"
LVIS_YAML_SHA = "af2b30726df6bc3019dfef2b5b1d185b352fe2474e10f620d8e851b7e52cc291"
HEAD = "model.23"
MASK_BRANCHES = ("proto", "cv5", "one2one_cv5")


def digest(path):
    with path.open("rb") as stream:
        sha = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def validate_graph(model, expected):
    head = model.model[-1]
    if (len(model.model) != 24 or type(head).__name__ != "YOLOESegment26" or head.is_fused
            or head.reg_max != 1 or head.stride.tolist() != [8, 16, 32] or model.text_model != "mobileclip2:b"):
        raise ValueError("Expected the unfused YOLOE-26s segmentation release with both detection heads")
    if not all(getattr(head, key, None) is not None for key in ("one2one_cv2", "one2one_cv3", "one2one_cv4")):
        raise ValueError("Release is missing the one-to-one detector")
    official = json.loads(json.dumps(graph(model)))  # Canonicalize tuple-valued attributes.
    official[""]["kind"] = "YOLOEModel"
    official[HEAD]["kind"] = "YOLOEDetect"
    official[HEAD]["children"] = [name for name in official[HEAD]["children"] if name not in MASK_BRANCHES]
    for node, spec in expected.items():
        if node not in official or any(official[node].get(k) != v for k, v in spec.items()):
            raise ValueError(f"Native detection graph differs at {node!r}")


def detector_state(source, template):
    keys = set(template.files)
    mask_prefixes = tuple(f"{HEAD}.{name}." for name in MASK_BRANCHES)
    missing = keys - source.keys()
    unexpected = {key for key in source if key not in keys and not key.startswith(mask_prefixes)}
    if missing or unexpected:
        raise ValueError(f"Detector key mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    result = {}
    for key in template.files:
        value = source[key].detach().cpu().numpy()
        target = template[key]
        if value.shape != target.shape or value.dtype != target.dtype:
            raise ValueError(f"Template mismatch for {key}: official {value.shape}/{value.dtype}, "
                             f"native {target.shape}/{target.dtype}")
        if key.endswith(".num_batches_tracked"):
            continue  # The native model uses fixed BN momentum.
        if not np.isfinite(value).all():
            raise ValueError(f"Nonfinite parameter or buffer at {key}")
        # YOLOE26.__init__ stores scalar logit_scale as a one-element Jittor tensor.
        result[key] = np.ascontiguousarray(value)
    return result


@torch.inference_mode()
def export(args):
    torch.set_num_threads(args.threads)
    for path, expected in ((args.weights, CHECKPOINT_SHA), (args.text_model, ENCODER_SHA), (args.lvis_yaml, LVIS_YAML_SHA)):
        if digest(path) != expected:
            raise ValueError(f"Unexpected release asset: {path}")
    categories = yaml.safe_load(args.lvis_yaml.read_bytes())["names"]
    names = [categories[i].split("/", 1)[0] for i in range(1203)]
    graph_path, template_path = args.init / "det.json", args.init / "det.npz"
    model, checkpoint = load_checkpoint(args.weights, device="cpu")
    validate_graph(model, json.loads(graph_path.read_bytes()))
    with np.load(template_path, allow_pickle=False) as template:
        state = detector_state(model.state_dict(), template)
        tensor_count = len(template.files)
    release_e2e_default = bool(model.model[-1].end2end)
    args.out.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out) as tmp:
        tmp = Path(tmp)
        native = tmp / "official-tp.npz"
        text = tmp / "pytorch-text.npy"
        np.savez(native, **state)
        native_count = len(state)
        del model, checkpoint, state
        gc.collect()
        # Use Ultralytics' tokenizer and encoder, independently of the native Jittor text tower.
        encoder = MobileCLIPTS(torch.device("cpu"), weight=str(args.text_model.resolve()))
        tokens = encoder.tokenize(names)
        raw = torch.cat([encoder.encode_text(batch) for batch in tokens.split(16)])[None].numpy()
        if raw.shape != (1, 1203, 512) or not np.allclose(np.linalg.norm(raw, axis=-1), 1, atol=1e-4):
            raise ValueError("Invalid official LVIS text features")
        np.save(text, raw)
        manifest = dict(source_sha256=CHECKPOINT_SHA, text_encoder_sha256=ENCODER_SHA,
                        lvis_yaml_sha256=LVIS_YAML_SHA, native_graph_sha256=digest(graph_path),
                        template_sha256=digest(template_path), native_sha256=digest(native),
                        pytorch_text_sha256=digest(text), names=names, compared_tensors=tensor_count,
                        native_tensors=native_count, omitted_mask_branches=list(MASK_BRANCHES),
                        omitted_batch_counters=tensor_count-native_count, release_end2end_default=release_e2e_default,
                        evaluation_head="one2one", torch=torch.__version__, ultralytics=ultralytics.__version__)
        meta = tmp / "manifest.json"
        meta.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        for path in (native, text, meta):
            path.replace(args.out / path.name)
    print(f"{args.out}: mapped {native_count} native tensors and encoded {len(names)} LVIS classes", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("weights", "init", "text-model", "lvis-yaml", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--threads", type=int, default=4)
    export(parser.parse_args())
