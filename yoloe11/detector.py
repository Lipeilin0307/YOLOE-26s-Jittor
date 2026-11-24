"""Jittor text-prompt detector for the unfused official YOLOE-11s weights."""
import json

import cv2
import jittor as jt
from jittor import nn
import numpy as np

from model import YOLOE26, normalized
from objective import anchors, ciou, decode, match, softplus
from simple_tokenizer import SimpleTokenizer, default_bpe
from text_encoder import BPE_SHA, MobileCLIP2, file_digest


def image_tensor(pixels):
    """Normalize byte images with a correctly rounded float32 quotient on CPU."""
    x = jt.array(pixels).stop_grad()
    if jt.flags.use_cuda:
        return x.float32()/255.
    return (x.float64()/255.).float32()


class MobileCLIPBLT(MobileCLIP2):
    """Original MobileCLIP text tower. Only tokenization and batching are shared with MobileCLIP2."""
    def __init__(self, path):
        if file_digest(default_bpe()) != BPE_SHA:
            raise ValueError("Unexpected BPE vocabulary")
        self.tokenizer = SimpleTokenizer(default_bpe())
        with np.load(path) as state:
            self.state = {k:jt.array(state[k]).stop_grad() for k in state.files}

    def forward(self, tokens):
        state = self.state
        end = tokens.argmax(1).astype(np.int32)
        b, n = len(end), int(end.max())+1
        x = state["embedding_layer.weight"][jt.array(tokens[:,:n])]
        x = x + state["positional_embedding.pos_embed.pos_embed"].reshape(77,512)[:n]
        mask = jt.array(np.triu(np.full((n,n), -np.inf,np.float32),k=1))
        def norm(x, key):
            return nn.layer_norm(x,(512,),state[key+".weight"],state[key+".bias"],1e-5)
        def linear(x, key):
            return nn.linear(x,state[key+".weight"],state[key+".bias"])
        for i in range(12):
            att = f"transformer.{i}.pre_norm_mha."
            ff = f"transformer.{i}.pre_norm_ffn."
            qkv = linear(norm(x,att+"0"),att+"1.qkv_proj").reshape(b,n,3,8,64).permute(2,0,3,1,4)
            scores = (qkv[0]*.125) @ qkv[1].transpose(-1,-2)
            values = (nn.softmax(scores+mask,dim=-1) @ qkv[2]).transpose(1,2).reshape(b,n,512)
            x = x+linear(values,att+"1.out_proj")
            h = linear(norm(x,ff+"0"),ff+"1")
            x = x+linear(.5*h*(1+(h*(2**-.5)).erf()),ff+"4")
        pooled = norm(x,"final_layer_norm")[jt.arange(b),jt.array(end)] @ state["projection_layer"]
        return normalized(pooled)


class YOLOE11TP(YOLOE26):
    def __init__(self, init, checkpoint=None):
        self.meta=json.loads((init/"yoloe11.json").read_bytes())
        self.head=self.meta["head"]
        self.bins=self.meta["reg_max"]
        if self.bins!=16 or self.meta["strides"]!=[8,16,32] or self.head!="model.23":
            raise ValueError("Unexpected 11s TP architecture")
        super().__init__(init,"TP",checkpoint=checkpoint)

    def bn(self, p, x):
        if not self.training or jt.flags.use_cuda or x.dtype != "float32":
            return super().bn(p, x)
        # The stem reduces up to 102400 samples per channel in the smoke run.
        # Accumulate central moments and their backward reductions in float64 on CPU.
        axes = (0, 2, 3)
        wide = x.float64()
        mean = wide.mean(axes, keepdims=True)
        centered = wide - mean
        variance = centered.sqr().mean(axes, keepdims=True)
        state = self.state
        momentum = self.spec[p]["momentum"]
        count = x.shape[0] * x.shape[2] * x.shape[3]
        running_mean = state[p+".running_mean"]
        running_var = state[p+".running_var"]
        running_mean.update((1-momentum)*running_mean + momentum*mean.detach().reshape(-1).float32())
        unbiased = (variance.detach().reshape(-1) * (count/(count-1))).float32()
        running_var.update((1-momentum)*running_var + momentum*unbiased)
        weight = state[p+".weight"].float64().reshape(1,-1,1,1)
        bias = state[p+".bias"].float64().reshape(1,-1,1,1)
        return (centered*(weight/(variance+self.spec[p]["eps"]).sqrt()) + bias).float32()

    def head_outputs(self, features, pe):
        boxes, logits = [], []
        for i,f in enumerate(features):
            b=f.shape[0]
            p=self.head+"."
            boxes.append(self.layer(p+f"cv2.{i}",f).reshape(b,4*self.bins,-1))
            image=self.layer(p+f"cv3.{i}",f)
            tower=p+f"cv4.{i}."
            image=self.bn(tower+"norm",image).reshape(b,image.shape[1],-1)
            similarity=normalized(pe) @ image
            logits.append(similarity*self.state[tower+"logit_scale"].exp()+self.state[tower+"bias"])
        return dict(boxes=jt.concat(boxes,2),scores=jt.concat(logits,2))

    def forward(self, x, raw):
        return self.head_outputs(self.image_features(x),self.text(raw))

    def distances(self, rec):
        values=rec["boxes"].transpose(1,2)
        probs=nn.softmax(values.reshape(values.shape[0],values.shape[1],4,self.bins),dim=-1)
        return (probs*jt.arange(self.bins).float32()).sum(-1)

    def loss(self, rec, batch):
        height,width=batch["img"].shape[-2:]
        reg=self.distances(rec)
        logits=rec["scores"]
        pixels=decode(reg.detach().numpy(),height,width)
        bi,ai,ti,classes,weight,gtbox=match(pixels,logits.detach().numpy(),batch,height,width,10,False)
        denom=max(1.,float(weight.sum()))
        classification=softplus(logits).sum()
        box=dfl=reg.sum()*0
        if len(ai):
            ib,ia=jt.array(bi),jt.array(ai)
            wt=jt.array(weight)
            classification-= (logits[ib,jt.array(classes),ia]*wt).sum()
            grid,strides=anchors(height,width)
            center=jt.array(grid[ai]); target=jt.array(gtbox[ti]/strides[ai])
            delta=reg[ib,ia]
            pred=jt.concat([center-delta[:,:2],center+delta[:,2:]],dim=1)
            box=((1-ciou(pred,target,jt))*wt).sum()/denom
            distances=jt.concat([center-target[:,:2],target[:,2:]-center],dim=1).clamp(0,self.bins-1-.01)
            left=distances.floor().int32()
            fraction=distances-left.float32()
            values=rec["boxes"].transpose(1,2)[ib,ia].reshape(-1,self.bins)
            logp=nn.log_softmax(values,dim=1)
            rows=jt.arange(values.shape[0]); lo=left.reshape(-1)
            ce=-(logp[rows,lo].reshape(-1,4)*(1-fraction)+logp[rows,lo+1].reshape(-1,4)*fraction)
            dfl=(ce.mean(1)*wt).sum()/denom
        terms=dict(box=7.5*box,cls=.5*classification/denom,dfl=1.5*dfl)
        return sum(terms.values())*reg.shape[0],terms

    def decoded(self, rec, height, width):
        boxes=decode(self.distances(rec).numpy(),height,width)
        scores=rec["scores"].sigmoid().numpy().transpose(0,2,1)
        return boxes,scores


def suppress(boxes, scores, conf=.001, iou=.7, max_det=300):
    """Class-aware multi-label NMS. Output rows use network-pixel xyxy coordinates."""
    output=[]
    for coords, probs in zip(boxes,scores):
        anchor, cls=np.nonzero(probs>conf)
        confidence=probs[anchor,cls]
        order=np.argsort(-confidence,kind="stable")[:30000]
        anchor,cls,confidence=anchor[order],cls[order],confidence[order]
        rects=coords[anchor].copy()
        rects[:,2:]-=rects[:,:2]
        keep=[]
        for c in np.unique(cls):
            positions=np.flatnonzero(cls==c)
            picked=cv2.dnn.NMSBoxes(rects[positions].tolist(),confidence[positions].tolist(),conf,iou)
            keep.extend(positions[np.asarray(picked,dtype=int).ravel()].tolist())
        keep=np.array(sorted(keep)[:max_det],dtype=int)
        output.append(np.column_stack((coords[anchor[keep]],confidence[keep],cls[keep])).astype(np.float32))
    return output
