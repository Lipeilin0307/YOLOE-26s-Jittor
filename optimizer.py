"""Jittor gradient accumulation, clipped MuSGD / AdamW, and exponential moving average."""
import math
import numpy as np
import jittor as jt


def parameter_group(name, ndim, spec, mu):
    """Match the detector head groups used by the Ultralytics optimizer."""
    module = name.rsplit(".", 1)[0]
    if mu and ndim in (2, 4):
        group = "muon"
    elif "bias" in name:
        group = "bias"
    elif spec.get(module, {}).get("kind") == "BatchNorm2d" or "logit_scale" in name:
        group = "bn"
    else:
        group = "weight"
    boosted = mu and name.startswith(("model.23.cv3.", "model.23.one2one_cv3.", "model.23.proto.semseg."))
    return group, 3 if boosted else 1


def rounded_bfloat16(value):
    """Round a float32 intermediate to nearest-even bfloat16, keeping FP32 GEMM accumulation."""
    if jt.flags.use_cuda:
        return value.bfloat16().float32()
    # Jittor 1.3.11's CPU bfloat16 cast drops the bottom 16 bits. Newton-Schulz
    # applies this conversion repeatedly, so even a one-ulp bias affects the update.
    flat = value.reshape(-1)
    rounded = jt.code(flat.shape, "float32", [flat], cpu_header="#include <cstring>", cpu_src="""
        for (int i = 0; i < in0_shape0; ++i) {
            uint32_t bits;
            std::memcpy(&bits, in0_p + i, sizeof(bits));
            // Propagate infinities and quiet NaNs before clearing the low bits.
            if ((bits & 0x7fffffffU) > 0x7f800000U)
                bits |= 0x00400000U;
            else
                bits += 0x7fffU + ((bits >> 16) & 1U);
            bits &= 0xffff0000U;
            std::memcpy(out0_p + i, &bits, sizeof(bits));
        }
    """)
    return rounded.reshape(value.shape)


def muon_transposes(params, spec):
    """Choose which Gram matrix to form for the same padded shapes as the reference optimizer."""
    buckets = {}
    for name, value in params.items():
        if value.ndim not in (2, 4):
            continue
        rows = value.shape[0]
        cols = math.prod(value.shape[1:])
        scale = max(1., value.shape[-2] / value.shape[-1])
        _, lr_multiplier = parameter_group(name, value.ndim, spec, True)
        buckets.setdefault((lr_multiplier, cols, scale), []).append((name, rows))
    flips = {}
    for (_, cols, _), items in buckets.items():
        items.sort(key=lambda pair: -pair[1])
        max_rows = items[0][1]
        for name, rows in items:
            if max_rows > 16 * rows:
                max_rows = rows
            flips[name] = max_rows > cols
    return flips


def orthogonal(grad, transpose=None):
    shape=tuple(grad.shape); x=grad.reshape(shape[0],-1)
    flip=x.shape[0]>x.shape[1] if transpose is None else transpose
    # Match the reference Newton-Schulz accumulator precision at iteration boundaries.
    # GEMMs accumulate in float and keep the bfloat16-rounded result.
    bf=rounded_bfloat16
    x=bf(x); x=bf(x/bf(bf((x*x).sum().sqrt())+1e-7))
    if flip: x=x.transpose(0,1)
    for _ in range(5):
        a=bf(x@x.transpose(0,1))
        b=bf(-4.775*a+2.0315*(a@a))
        x=bf(3.4445*x+b@x)
    if flip: x=x.transpose(0,1)
    return x.reshape(shape)*math.sqrt(max(1.,shape[-2]/shape[-1]))


class Optimizer:
    def __init__(self,model,epochs,batch,batches,smoke,pf_p5_bias_warmup_lr=0.1):
        self.model=model; self.params=model.trainable
        if not math.isfinite(pf_p5_bias_warmup_lr) or pf_p5_bias_warmup_lr < 0:
            raise ValueError("PF P5 bias warmup learning rate must be finite and nonnegative")
        self.pf_p5_bias_warmup_lr = pf_p5_bias_warmup_lr
        self.mu=model.stage in ("TP","SEG")
        self.muon_transposes=muon_transposes(self.params,model.spec) if self.mu else {}
        self.lr=0.00125 if self.mu else 0.002
        self.end=0.5 if self.mu else 0.01
        self.nbs=batch if smoke else 64
        nominal=max(round(self.nbs/batch),1)
        self.decay=(0.0007 if self.mu else 0.025)*batch*nominal/self.nbs
        self.accumulate=min(nominal,epochs*batches)
        self.warm=0 if smoke else round(min(3,max(epochs-1,0))*batches)
        self.epochs=epochs; self.batch=batch; self.batches=batches; self.steps=0; self.last=-1
        self.acc={}; self.vel={}; self.second={}
        self.ema={k:v.clone().stop_grad() for k,v in model.state.items()}
        for k,v in self.params.items():
            self.vel[k]=jt.zeros_like(v).stop_grad()
            if not self.mu or v.ndim in (2,4):
                self.second[k]=jt.zeros_like(v).stop_grad()

    def backward(self,loss,epoch,iteration):
        vals=list(self.params.values())
        grads=jt.grad(loss,vals)
        for k,g in zip(self.params,grads):
            g=g.detach()
            if k in self.acc: self.acc[k].update(self.acc[k]+g)
            else: self.acc[k]=g.stop_grad()
        ni=epoch*self.batches+iteration
        alpha=min(1.,ni/self.warm) if self.warm else 1.
        accumulate=max(1,round(1+alpha*(self.accumulate-1)))
        if ni-self.last<accumulate:
            jt.sync(list(self.acc.values()))
            return
        self.steps+=1; self.last=ni
        lr=self.lr*(max(1-epoch/self.epochs,0)*(1-self.end)+self.end)
        momentum=0.8+(0.9-0.8)*alpha if self.mu else 0.9
        with jt.no_grad():
            if jt.flags.use_cuda:
                norm=sum((g*g).sum() for g in self.acc.values()).sqrt()
            else:
                # Small clip coefficient errors can cross BF16 rounding boundaries
                # for thousands of entries before the first Muon matrix product.
                norm=sum(g.float64().sqr().sum() for g in self.acc.values()).sqrt().float32()
            clip=(10/(norm+1e-6)).minimum(1.)
            for k,p in self.params.items():
                grad=self.acc[k]*clip; vel=self.vel[k]
                group, multiplier=parameter_group(k,p.ndim,self.model.spec,self.mu)
                decay=self.decay if group in ("weight","muon") else 0.
                warmup_lr = 0.1 if group == "bias" else 0.
                if self.model.stage == "PF" and k == "model.23.one2one_cv3.2.2.bias":
                    warmup_lr = self.pf_p5_bias_warmup_lr
                step_lr=alpha*lr*multiplier+(1-alpha)*warmup_lr
                if self.mu:
                    weight=p
                    if group=="muon":
                        vel.update(momentum*vel+(1-momentum)*grad)
                        u=momentum*vel+(1-momentum)*grad
                        weight=weight-(0.2*step_lr)*orthogonal(u, self.muon_transposes[k])
                        vel=self.second[k]
                    g=grad+decay*weight
                    vel.update(momentum*vel+g)
                    p.update(weight-step_lr*(g+momentum*vel))
                else:
                    var=self.second[k]
                    vel.update(0.9*vel+0.1*grad)
                    var.update(0.999*var+0.001*grad*grad)
                    p.update(p*(1-step_lr*decay)-step_lr*(vel/(1-0.9**self.steps))/((var/(1-0.999**self.steps)).sqrt()+1e-8))
            d=0.9999*(1-math.exp(-self.steps/2000))
            for k,v in self.model.state.items():
                if k in self.params or k.endswith(("running_mean","running_var")):
                    self.ema[k].update(self.ema[k]*d+v*(1-d))
        self.acc.clear()
        # update() adopts the no_grad result's stop_grad flag in Jittor.
        for p in self.params.values():
            p.start_grad()
        jt.sync_all()
