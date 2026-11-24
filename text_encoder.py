"""Frozen MobileCLIP2-B text tower and CLIP BPE, with native Jittor inference."""
import argparse
from collections import OrderedDict
import hashlib
import io
from pathlib import Path
import pickle
from zipfile import ZipFile

import jittor as jt
from jittor import nn
import numpy as np

from simple_tokenizer import SimpleTokenizer, default_bpe


ENCODER_SHA = "35d7f213e4d75f38514e4656ad3cb91158bd33e3805d8ac349f23b186f66982f"
BPE_SHA = "924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a"


def file_digest(path):
    sha = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def rebuild(storage, offset, shape, stride, *metadata):
    unit = storage.dtype.itemsize
    return np.ndarray(shape, dtype=storage.dtype, buffer=storage, offset=offset*unit,
                      strides=tuple(s*unit for s in stride))


class Constants(pickle.Unpickler):
    """Read only tensor storage records from the pinned, hash-checked archive."""
    def __init__(self, archive, prefix):
        self.archive = archive
        self.prefix = prefix
        super().__init__(io.BytesIO(archive.read(prefix + "/constants.pkl")))

    def find_class(self, module, name):
        allowed = {("torch", "FloatStorage"): np.dtype("<f4"), ("torch","LongStorage"):np.dtype("<i8"),
                   ("torch._utils","_rebuild_tensor_v2"):rebuild, ("collections","OrderedDict"):OrderedDict}
        if (module, name) not in allowed:
            raise ValueError(f"Unexpected constant type {module}.{name}")
        return allowed[module, name]

    def persistent_load(self, record):
        kind, dtype, key, _, count = record
        if kind != "storage" or not str(key).isdigit():
            raise ValueError("Unexpected storage record")
        return np.frombuffer(self.archive.read(f"{self.prefix}/constants/{key}"), dtype=dtype, count=count)


class MobileCLIP2:
    def __init__(self, archive, vocab=None):
        vocab = vocab or default_bpe()
        if file_digest(archive) != ENCODER_SHA or file_digest(vocab) != BPE_SHA:
            raise ValueError("MobileCLIP2-B weights or BPE vocabulary differ from the pinned release")
        with ZipFile(archive) as z:
            prefix, = [n.rsplit("/",1)[0] for n in z.namelist() if n.endswith("/constants.pkl")]
            values = Constants(z, prefix).load()
        if len(values) != 151 or tuple(values[0].shape) != (49408,512):
            raise ValueError("Unexpected MobileCLIP2-B text tower layout")
        self.weights = [jt.array(np.ascontiguousarray(x)).stop_grad() for x in values]
        self.tokenizer = SimpleTokenizer(str(vocab))

    def tokenize(self, names):
        tokens = np.zeros((len(names), 77), np.int32)
        for i, name in enumerate(names):
            ids = [49406, *self.tokenizer.encode(name)[:75], 49407]
            tokens[i,:len(ids)] = ids
        return tokens

    def forward(self, tokens):
        w = self.weights
        end = tokens.argmax(1).astype(np.int32)
        batch = len(end)
        length = int(end.max()) + 1
        x = w[0][jt.array(tokens[:,:length])] + w[1][:length]
        mask = w[7].reshape(77,77)[:length,:length]
        for layer in range(12):
            if layer:
                indices = range(16 + 12*(layer-1), 16 + 12*layer)
            else:
                indices = (2,3,5,6,8,9,10,11,12,13,14,15)
            nw,nb,qw,qb,ow,ob,fw,fb,w1,b1,w2,b2 = [w[k] for k in indices]
            h = nn.layer_norm(x, (512,), nw, nb, 1e-5)
            qkv = (h @ qw + qb).reshape(batch,length,3,8,64).permute(2,0,3,1,4)
            q,k,v = qkv[0], qkv[1], qkv[2]
            probs = nn.softmax((q @ k.transpose(-1,-2)) * .125 + mask, dim=-1)
            h = (probs @ v).transpose(1,2).reshape(batch,length,512)
            x = x + (h @ ow + ob)
            h = nn.layer_norm(x, (512,), fw, fb, 1e-5) @ w1 + b1
            h = .5*h*(1+(h*(2**-.5)).erf())
            x = x + (h @ w2 + b2)
        pooled = nn.layer_norm(x, (512,), w[148], w[149], 1e-5)[jt.arange(batch),jt.array(end)]
        y = pooled @ w[150]
        return y / (y*y).sum(-1,keepdims=True).sqrt()

    def encode(self, names, batch_size=16):
        if not names or batch_size < 1:
            raise ValueError("Provide at least one prompt and a positive batch size")
        tokens = self.tokenize(names)
        with jt.no_grad():
            rows = [self.forward(tokens[i:i+batch_size]).numpy() for i in range(0,len(tokens),batch_size)]
        values = np.concatenate(rows)[None]
        if not np.isfinite(values).all():
            raise FloatingPointError("Invalid CLIP embedding")
        return values


if __name__ == "__main__":
    import json
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", required=True, type=Path)
    p.add_argument("--names", required=True, type=Path, help="Ordered JSON list or LVIS manifest with a names list")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--append-object", action="store_true", help="Append PF object prompt for raw-text.npy")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = p.parse_args()
    jt.flags.use_cuda = int(args.device == "cuda")
    names = json.loads(args.names.read_bytes())
    if isinstance(names, dict):
        names = names["names"]
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        p.error("--names must contain an ordered list of strings")
    if args.append_object:
        names = [*names, "object"]
    np.save(args.out, MobileCLIP2(args.weights).encode(names))
