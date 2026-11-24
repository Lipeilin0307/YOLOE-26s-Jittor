"""YOLO26 dual assignment, CIoU/L1, prompt classification and instance-mask losses."""
from functools import lru_cache
import math
import jittor as jt
import numpy as np


@lru_cache(maxsize=12)
def anchors(h,w):
    points=[]; strides=[]
    for stride in (8,16,32):
        yy,xx=np.mgrid[:h//stride,:w//stride]
        points.append(np.stack([xx.ravel()+0.5,yy.ravel()+0.5],1).astype("float32"))
        strides.append(np.full((xx.size,1),stride,np.float32))
    return np.concatenate(points),np.concatenate(strides)


def ciou(a,b,backend):
    # Arrays/Vars end in xyxy. For gradients, alpha is the detached CIoU aspect weight.
    small,large,atan=(np.minimum,np.maximum,np.arctan) if backend is np else (jt.minimum,jt.maximum,jt.arctan)
    aw,ah=a[...,2]-a[...,0],a[...,3]-a[...,1]+1e-7
    bw,bh=b[...,2]-b[...,0],b[...,3]-b[...,1]+1e-7
    iw=large(small(a[...,2],b[...,2])-large(a[...,0],b[...,0]),0.)
    ih=large(small(a[...,3],b[...,3])-large(a[...,1],b[...,1]),0.)
    inter=iw*ih
    iou=inter/(aw*ah+bw*bh-inter+1e-7)
    cw=large(a[...,2],b[...,2])-small(a[...,0],b[...,0])
    ch=large(a[...,3],b[...,3])-small(a[...,1],b[...,1])
    dx=(a[...,0]+a[...,2]-b[...,0]-b[...,2])*0.5
    dy=(a[...,1]+a[...,3]-b[...,1]-b[...,3])*0.5
    v=(4/math.pi**2)*(atan(bw/bh)-atan(aw/ah))**2
    alpha=v/(1-iou+v+1e-7)
    if backend is jt: alpha=alpha.detach()
    return iou-(dx*dx+dy*dy)/(cw*cw+ch*ch+1e-7)-alpha*v


def decode(dist,h,w):
    grid,stride=anchors(h,w)
    return np.concatenate((grid-dist[...,:2],grid+dist[...,2:]),-1)*stride


def match(pbox,logits,batch,h,w,k,one):
    grid,stride=anchors(h,w); centers=grid*stride
    gtclasses=batch["cls"].astype(np.int32).ravel()
    xywh=batch["bboxes"]*np.asarray([w,h,w,h],np.float32)
    gtbox=np.concatenate((xywh[:,:2]-xywh[:,2:]/2,xywh[:,:2]+xywh[:,2:]/2),1)
    pairs=[]; labels=[]; anchors_idx=[]; targets=[]; strengths=[]
    for bi in range(len(pbox)):
        rows=np.where(batch["batch_idx"].ravel()==bi)[0]
        if not len(rows): continue
        c=xywh[rows,:2]; half=np.maximum(xywh[rows,2:],16.)/2
        inside=((centers[None]>((c-half)[:,None]+1e-9)) & (centers[None]<((c+half)[:,None]-1e-9))).all(-1)
        g,a=np.nonzero(inside)
        if not len(g): continue
        overlap=np.zeros(inside.shape,np.float32); metric=np.zeros_like(overlap)
        ov=np.maximum(ciou(gtbox[rows[g]],pbox[bi,a],np),0.)
        conf=1/(1+np.exp(-np.clip(logits[bi,gtclasses[rows[g]],a],-80.,80.)))
        overlap[g,a]=ov; metric[g,a]=np.sqrt(conf)*ov**6
        top=np.argpartition(-metric,k-1,axis=1)[:,:k]
        pos=np.zeros_like(inside)
        np.put_along_axis(pos,top,True,axis=1); pos &= inside
        collisions=np.where(pos.sum(0)>1)[0]
        if len(collisions):
            pos[:,collisions]=False
            pos[overlap[:,collisions].argmax(0),collisions]=True
        if one:
            best=(metric*pos).argmax(axis=1)
            final=np.zeros_like(pos); final[np.arange(len(rows)),best]=True; pos &= final
        selected=np.where(pos)
        if not len(selected[0]): continue
        m=metric*pos
        norm=(overlap*pos).max(1)/(m.max(1)+1e-9)
        g,a=selected
        pairs.extend([bi]*len(a)); anchors_idx.extend(a); targets.extend(rows[g])
        labels.extend(gtclasses[rows[g]]); strengths.extend(metric[g,a]*norm[g])
    return tuple(np.asarray(t,np.int32) for t in (pairs,anchors_idx,targets,labels))+(np.asarray(strengths,np.float32),gtbox)


def softplus(x):
    from jittor import nn
    e=x.minimum(0.).exp()
    return jt.where(x < -8.,e*(1.-e*(0.5-e/3.)),nn.softplus(x))


def mask_loss(rec,batch,bi,ai,ti,gtbox,h,w):
    proto=rec["proto"]; zero=proto.sum()*0
    if not len(ai): return zero
    mh,mw=proto.shape[-2:]
    if tuple(batch["masks"].shape[-2:])!=(mh,mw):
        from jittor import nn
        proto=nn.interpolate(proto,size=tuple(batch["masks"].shape[-2:]),mode="bilinear",align_corners=False)
        mh,mw=proto.shape[-2:]
    out=zero
    for b in np.unique(bi):
        ix=bi==b
        box=gtbox[ti[ix]]/np.asarray([w,h,w,h],np.float32)
        area=(box[:,2]-box[:,0])*(box[:,3]-box[:,1])
        bounds=box*np.asarray([mw,mh,mw,mh],np.float32)
        col=np.arange(mw)[None,:]; row=np.arange(mh)[None,:]
        keepx=(col>=bounds[:,0:1]) & (col<bounds[:,2:3])
        keepy=(row>=bounds[:,1:2]) & (row<bounds[:,3:4])
        mask=jt.array(batch["masks"][ti[ix]].astype(np.float32))
        coeff=rec["mask_coefficient"][int(b)].transpose(0,1)[jt.array(ai[ix])]
        pred=(coeff @ proto[int(b)].reshape(proto.shape[1],-1)).reshape(-1,mh,mw)
        crop=jt.array((keepy[:,:,None]&keepx[:,None,:]).astype(np.float32))
        out=out+(((softplus(pred)-pred*mask)*crop).mean([1,2])/jt.array(area)).sum()
    return out/len(ai)


def branch_loss(rec,batch,stage,one):
    h,w=batch["img"].shape[-2:]; b=batch["img"].shape[0]
    reg=rec["boxes"].transpose(1,2); logits=rec["scores"]
    assignment = rec.get("assignment_scores", logits)
    bi,ai,ti,cls,weight,gtbox=match(decode(reg.detach().numpy(),h,w),assignment.detach().numpy(),batch,h,w,7 if one else 10,one)
    den=max(float(weight.sum()),1.); cls_term=softplus(logits).sum()
    lbox=reg.sum()*0; l1=lbox
    if len(ai):
        idx=jt.array(ai); ib=jt.array(bi); wt=jt.array(weight)
        cls_term=cls_term-(logits[ib,jt.array(cls),idx]*wt).sum()
        if stage!="VP":
            grid,stride=anchors(h,w); anchor=jt.array(grid[ai]); s=jt.array(stride[ai])
            target=jt.array(gtbox[ti])/s; dist=reg[ib,idx]
            pred=jt.concat([anchor-dist[:,:2],anchor+dist[:,2:]],dim=1)
            lbox=((1-ciou(pred,target,jt))*wt).sum()/den
            gt_dist=jt.concat([anchor-target[:,:2],target[:,2:]-anchor],dim=1)
            l1=(((dist-gt_dist)*s/jt.array(np.asarray([w,h,w,h],np.float32))).abs().mean(1)*wt).sum()/den
    cls_term=cls_term*(0.5/den)
    terms={"cls":cls_term}
    if stage!="VP": terms.update(box=lbox*7.5,l1=l1*1.5)
    if stage=="SEG": terms["seg"]=7.5*mask_loss(rec,batch,bi,ai,ti,gtbox,h,w)
    return sum(terms.values())*b,terms


def objective(outputs,batch,stage,epoch,epochs):
    many,_=branch_loss(outputs["one2many"],batch,stage,False)
    one,details=branch_loss(outputs["one2one"],batch,stage,True)
    mix=max(1-epoch/max(epochs-1,1),0)*0.7+0.1
    return many*mix+one*(1-mix),details


def detections(rec,h,w,max_det=300):
    scores=rec["scores"].numpy().transpose(0,2,1)
    boxes=decode(rec["boxes"].numpy().transpose(0,2,1),h,w)
    coeff=rec["mask_coefficient"].numpy().transpose(0,2,1) if "mask_coefficient" in rec else None
    rows=[]
    for i,s in enumerate(scores):
        k=min(max_det,len(s)); anchor=np.argpartition(-s.max(1),k-1)[:k]
        v=s[anchor].ravel(); top=np.argpartition(-v,k-1)[:k]
        top=top[np.argsort(-v[top])]; a=anchor[top//s.shape[-1]]; c=top%s.shape[-1]
        prob=1/(1+np.exp(-np.clip(v[top],-80.,80.)))
        pieces=[boxes[i,a],prob[:,None],c[:,None].astype("float32")]
        if coeff is not None: pieces.append(coeff[i,a])
        rows.append(np.concatenate(pieces,1))
    return (np.stack(rows),rec["proto"].numpy()) if coeff is not None else np.stack(rows)
