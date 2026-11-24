"""YOLOE-26s. All differentiable image, prompt and mask operations run in Jittor."""
import json
from pathlib import Path

import jittor as jt
from jittor import nn
import numpy as np


def normalized(x):
    return x / (x*x).sum(-1, keepdims=True).maximum(1e-24).sqrt()


def retrieve_vocab(features, proposal, weight, bias):
    """Evaluate the fixed vocabulary only at proposals kept by any image."""
    batch, channels, height, width = features.shape
    classes = weight.shape[0]
    keep = proposal.reshape(batch, -1).sigmoid() > 0.001
    selected = keep.any(0).numpy().astype(bool)
    positions = np.flatnonzero(selected).astype(np.int32)
    if not len(positions):
        return jt.full((batch, classes, height * width), -1e5, dtype="float32")
    x = features.reshape(batch, channels, -1).transpose(1, 2)[:, jt.array(positions), :]
    logits = nn.linear(x, weight.reshape(classes, channels), bias).transpose(1, 2)
    # Restore the coordinate layout expected by the shared box decoder.
    lookup = jt.array(np.maximum(0, np.cumsum(selected) - 1).astype(np.int32))
    return jt.where(keep.unsqueeze(1), logits[:, :, lookup], -1e5)


def parameter_enabled(stage, name):
    if name.endswith(("running_mean", "running_var", "num_batches_tracked")):
        return False
    if stage == "TP":
        return ".savpe." not in name
    if not name.startswith("model.23."):
        return False
    parts = name.split(".")
    group = parts[2].removeprefix("one2one_")
    if stage == "VP":
        return group in {"savpe", "cv4"}
    if stage == "SEG":
        return group in {"proto", "cv5"} and ".semseg." not in name
    return group == "cv3" and len(parts) == 6 and parts[4] == "2" and parts[5] in {"weight","bias"}


class YOLOE26:
    head = "model.23"

    def __init__(self, init, stage, parent=None, checkpoint=None):
        self.stage = stage
        self.training = False
        prefix = "seg" if stage == "SEG" else "det"
        self.spec = json.loads((init / (prefix + ".json")).read_text())
        self.raw = jt.array(np.load(init / "raw-text.npy")).stop_grad()
        with np.load(init / (prefix + ".npz")) as weights:
            arrays = dict(weights)
        if parent:
            with np.load(parent) as weights:
                arrays.update({k: weights[k] for k in weights.files if k in arrays})
        self.state = {k: jt.array(np.ascontiguousarray(v)).stop_grad() for k,v in arrays.items()
                      if not k.endswith("num_batches_tracked")}
        self.fused = False
        if stage == "PF":
            self.object_fuse()
        if checkpoint:
            with np.load(checkpoint) as weights:
                if set(weights.files) != set(self.state):
                    raise ValueError("Checkpoint state does not match stage")
                for k in self.state:
                    self.state[k].assign(jt.array(weights[k]))
        self.trainable = {k: v.start_grad() for k,v in self.state.items() if parameter_enabled(stage, k)}

    def bn(self, p, x):
        mean, var, w, b = (self.state[p+"."+s] for s in ("running_mean","running_var","weight","bias"))
        eps = self.spec[p]["eps"]
        if self.training and parameter_enabled(self.stage, p+".weight"):
            dtype = x.dtype
            if not jt.flags.use_cuda and dtype == "float32":
                # The CPU stem reduces over 100k samples per channel. Keep the
                # central moments and their backward reductions in wider accumulators.
                x, w, b = x.float64(), w.float64(), b.float64()
            axes = [0,2,3]
            mu = x.mean(axes)
            centered = x - mu.reshape(1,-1,1,1)
            sigma = (centered*centered).mean(axes)
            momentum = self.spec[p]["momentum"]
            n = x.shape[0]*x.shape[2]*x.shape[3]
            mean.update((1-momentum)*mean + momentum*mu.detach().cast(mean.dtype))
            var.update((1-momentum)*var + ((momentum*n/(n-1))*sigma.detach()).cast(var.dtype))
            output = centered*(w/(sigma+eps).sqrt()).reshape(1,-1,1,1) + b.reshape(1,-1,1,1)
            return output.cast(dtype)
        return nn.batch_norm(x, mean, var, w, b, training=False, eps=eps)

    def layer(self, path, x):
        m = self.spec[path]
        kind = m["kind"]
        def op(name, value):
            return self.layer(path+"."+name, value)
        if kind in ("Conv2d","ConvTranspose2d"):
            w, b = self.state[path+".weight"], self.state.get(path+".bias")
            if kind == "ConvTranspose2d":
                return nn.conv_transpose2d(x,w,b,tuple(m["stride"]),tuple(m["padding"]),tuple(m["output_padding"]),m["groups"],tuple(m["dilation"]))
            return nn.conv2d(x,w,b,m["stride"],m["padding"],m["dilation"],m["groups"])
        if kind in ("Conv", "DWConv"):
            return op("act", op("bn", op("conv", x)))
        if kind == "BatchNorm2d":
            return self.bn(path,x)
        if kind == "SiLU":
            return nn.silu(x)
        if kind == "Identity":
            return x
        if kind == "Linear":
            return nn.linear(x, self.state[path+".weight"], self.state.get(path+".bias"))
        if kind in ("Sequential","ModuleList"):
            for name in m["children"]:
                x = op(name,x)
            return x
        if kind == "Upsample":
            return nn.interpolate(x,scale_factor=m["scale_factor"],mode=m["mode"])
        if kind == "Concat":
            return jt.concat(x,m["d"])
        if kind == "Bottleneck":
            y = op("cv2",op("cv1",x))
            return x+y if m["add"] else y
        if kind == "C3k":
            return op("cv3",jt.concat([op("m",op("cv1",x)),op("cv2",x)],1))
        if kind == "C3k2":
            ys = list(op("cv1",x).chunk(2,dim=1))
            for name in self.spec[path+".m"]["children"]:
                ys.append(op("m."+name,ys[-1]))
            return op("cv2",jt.concat(ys,1))
        if kind == "SPPF":
            ys=[op("cv1",x)]
            for _ in range(m["n"]):
                ys.append(op("m",ys[-1]))
            out=op("cv2",jt.concat(ys,1))
            return out+x if m["add"] else out
        if kind == "MaxPool2d":
            if min(x.shape[-2:])<=m["kernel_size"]:
                pad=m["padding"]
                x=nn.pad(x,(pad,pad,pad,pad),mode="constant",value=-3.402823466e38)
                return nn.max_pool2d(x,m["kernel_size"],m["stride"],0)
            return nn.max_pool2d(x,m["kernel_size"],m["stride"],m["padding"])
        if kind == "C2PSA":
            a,b=op("cv1",x).chunk(2,dim=1)
            return op("cv2",jt.concat([a,op("m",b)],1))
        if kind == "PSABlock":
            y=op("attn",x)
            x=x+y if m["add"] else y
            y=op("ffn",x)
            return x+y if m["add"] else y
        if kind == "Attention":
            b,c,h,w=x.shape
            d=m["key_dim"]
            qkv=op("qkv",x).reshape(b,m["num_heads"],2*d+m["head_dim"],h*w)
            q,k,v=qkv[:,:,:d],qkv[:,:,d:2*d],qkv[:,:,2*d:]
            a=nn.softmax((q*m["scale"]).transpose(-2,-1) @ k,dim=-1)
            return op("proj",(v @ a.transpose(-2,-1)).reshape(b,c,h,w)+op("pe",v.reshape(b,c,h,w)))
        if kind == "Residual":
            return x+op("m",x)
        if kind == "SwiGLUFFN":
            a,b=op("w12",x).chunk(2,dim=-1)
            return op("w3",nn.silu(a)*b)
        if kind == "Proto26":
            f=x[0]
            for i,feat in enumerate(x[1:]):
                y=op(f"feat_refine.{i}",feat)
                f=f+nn.interpolate(y,scale_factor=2**(i+1),mode="nearest")
            feat=op("cv1",op("feat_fuse",f))
            return op("cv3",op("cv2",op("upsample",feat)))
        raise ValueError(f"Unsupported module {kind} at {path}")

    def image_features(self,x):
        values=[]
        for i in range(23):
            p=f"model.{i}"; source=self.spec[p]["f"]
            inputs=[x if j==-1 else values[j] for j in source] if isinstance(source,list) else (x if source==-1 else values[source])
            x=self.layer(p,inputs); values.append(x)
        return [values[i] for i in self.spec[self.head]["f"]]

    def text(self, raw):
        return normalized(self.layer(self.head+".reprta",raw))

    def visual(self,features,visual):
        p=self.head+".savpe."
        def fuse(tower):
            return jt.concat([self.layer(p+tower+"."+str(i),v) for i,v in enumerate(features)],1)
        y=self.layer(p+"cv4",fuse("cv2")); x=self.layer(p+"cv3",fuse("cv1"))
        b,c,h,w=x.shape; q=visual.shape[1]; g=y.shape[1]
        y=y.reshape(b,1,g,h,w).expand(b,q,g,h,w).reshape(b*q,g,h,w)
        mask=visual.reshape(b*q,1,h,w)
        y=self.layer(p+"cv6",jt.concat([y,self.layer(p+"cv5",mask)],1)).reshape(b,q,g,h*w)
        mask=mask.reshape(b,q,1,h*w)
        # For an absent prompt the reference softmax is uniform. Explicit zero logits
        # keep this case well-defined in both the CUDA forward and its backward.
        score=jt.where(mask!=0,y*mask,-1e5)
        score=jt.where(mask.sum(-1,keepdims=True)>0,score,0.)
        a=nn.softmax(score,dim=-1)
        z=a.transpose(1,2) @ x.reshape(b,g,c//g,h*w).transpose(-1,-2)
        return normalized(z.transpose(1,2).reshape(b,q,c))

    def projected(self, branch, i, pe):
        h=self.head+"."
        bn=h+branch+f"cv4.{i}."
        conv=h+branch+f"cv3.{i}.2."
        w=self.state[conv+"weight"]; b=self.state[conv+"bias"]
        norm=bn+"norm."
        s=self.state[norm+"weight"]/(self.state[norm+"running_var"]+self.spec[norm[:-1]]["eps"]).sqrt()
        t=pe[0]*self.state[bn+"logit_scale"].exp()
        wf=(t @ (s.reshape(-1,1)*w.reshape(w.shape[0],-1))).reshape(pe.shape[1],w.shape[1],1,1)
        bf=t @ ((b-self.state[norm+"running_mean"])*s+self.state[norm+"bias"])+self.state[bn+"bias"]
        return wf.detach(),bf.detach()

    def object_fuse(self):
        with jt.no_grad():
            object_pe=self.text(self.raw[:,-1:])
            vocab=self.text(self.raw[:,:80])
            for branch in ("","one2one_"):
                for i in range(3):
                    p=self.head+"."+branch+f"cv3.{i}.2."
                    for suffix, v in zip(("weight","bias"),self.projected(branch,i,vocab)):
                        self.state[p+"vocab_"+suffix]=v.stop_grad()
                    self.state[p+"weight"],self.state[p+"bias"]=self.projected(branch,i,object_pe)
        self.fused=True

    def forward(self,x,raw=None,visual=None,pe=None,retrieve=False,both=True,pf_vocab_assignment=False):
        features=self.image_features(x)
        if not self.fused and pe is None:
            pe=self.visual(features,visual) if visual is not None else self.text(self.raw[:,:80] if raw is None else raw)
        if pe is not None and pe.shape[0]!=x.shape[0]:
            pe=pe.expand(x.shape[0],pe.shape[1],pe.shape[2])
        result={}
        proto=self.layer(self.head+".proto",[f.detach() for f in features]) if self.stage=="SEG" else None
        for label,branch in (("one2many",""),("one2one","one2one_")):
            if not both and not branch: continue
            fs=[v.detach() for v in features] if branch else features
            emb=pe.detach() if branch and pe is not None else pe
            h=self.head+"."+branch
            boxes=[]; scores=[]; masks=[]
            assignment_p5 = None
            for i,f in enumerate(fs):
                b=f.shape[0]
                boxes.append(self.layer(h+f"cv2.{i}",f).reshape(b,4,-1))
                p=h+f"cv3.{i}"
                if retrieve:
                    z=self.layer(p+".1",self.layer(p+".0",f))
                    weight=self.state[p+".2.vocab_weight"]
                    bias=self.state[p+".2.vocab_bias"]
                    if i<2:
                        logits=retrieve_vocab(z,self.layer(p+".2",z),weight,bias)
                    else:
                        logits=nn.conv2d(z,weight,bias)
                elif pf_vocab_assignment and self.stage == "PF" and branch and i == 2:
                    z = self.layer(p+".1", self.layer(p+".0", f))
                    logits = self.layer(p+".2", z)
                    with jt.no_grad():
                        weight = self.state[p+".2.vocab_weight"]
                        bias = self.state[p+".2.vocab_bias"]
                        assignment_p5 = nn.conv2d(z, weight, bias).max(1, keepdims=True).reshape(b, 1, -1)
                else:
                    z=self.layer(p,f)
                    if self.fused: logits=z
                    else:
                        q=h+f"cv4.{i}."
                        z=self.bn(q+"norm",z)
                        logits=normalized(emb) @ z.reshape(b,z.shape[1],-1)
                        logits=logits*self.state[q+"logit_scale"].exp()+self.state[q+"bias"]
                scores.append(logits.reshape(b,logits.shape[1],-1))
                if proto is not None: masks.append(self.layer(h+f"cv5.{i}",f).reshape(b,32,-1))
            rec={"boxes":jt.concat(boxes,2),"scores":jt.concat(scores,2)}
            if assignment_p5 is not None:
                # TAL uses the frozen TP vocabulary at stride 32. BCE still trains the object projection.
                rec["assignment_scores"] = jt.concat([s.detach() for s in scores[:-1]] + [assignment_p5], 2)
            if masks: rec.update(mask_coefficient=jt.concat(masks,2),proto=proto.detach() if branch else proto)
            result[label]=rec
        return result
