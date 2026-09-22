"""
COMPUTATIONAL COUPLING --- Surrogate-gradient training of a masked recurrent SNN.

A fixed-reservoir comparison cannot test whether a structural property acts as an
*inductive bias*, because an inductive bias is a claim about learning trajectories
rather than about the function computed by a frozen network.  This module runs the
same structural ladder with the recurrent weights trained, so that each rung supplies
the connectivity **mask** and the sign pattern while the magnitudes are learned.

Constraints held fixed across rungs
-----------------------------------
* identical mask density (the ladder guarantees identical edge counts),
* Dale's principle, enforced by ``W = relu(|M| magnitudes) * mask * sign(presynaptic)``,
  so training can never flip a synapse's valence,
* identical hidden size, input encoder, optimiser, schedule, seed set and epoch budget.

The only thing that differs between conditions is which synapses exist.

The sensory/motor protocol carries over unchanged
-------------------------------------------------
The input projection reaches only a sensory subset and the readout sees only a disjoint
motor subset, exactly as in the fixed-reservoir harness.  This is not a detail.  With
every neuron driven and every neuron read, the encoder and the readout can solve the
task between them and the recurrent matrix need not carry the signal at all -- which is
the failure the fixed-reservoir protocol control exists to expose, and it does not stop
being a failure because the recurrent weights are now trained.  ``zero_recurrence``
runs the same network with the mask emptied and is the diagnostic that the trained
benchmark is sensitive to recurrent structure at all; a comparison whose
zero-recurrence control scores near the connectomes is uninformative whatever it
reports.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class FastSigmoidSpike(torch.autograd.Function):
    """Heaviside forward, fast-sigmoid surrogate backward (Zenke & Ganguli)."""

    scale = 10.0

    @staticmethod
    def forward(ctx, v):
        ctx.save_for_backward(v)
        return (v >= 0.0).to(v.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        (v,) = ctx.saved_tensors
        sg = 1.0 / (1.0 + FastSigmoidSpike.scale * v.abs()) ** 2
        return grad_out * sg


spike_fn = FastSigmoidSpike.apply


@dataclass
class TrainConfig:
    hidden: int = 256
    epochs: int = 15
    batch_size: int = 128
    lr: float = 2e-3
    leak: float = 0.9
    v_threshold: float = 1.0
    # L1 firing-rate penalty, identical across conditions.  Kept small deliberately:
    # under the sensory/motor protocol only a fraction of neurons receive direct drive,
    # so at initialisation the motor population is nearly silent and total silence is
    # already a local minimum of the task loss.  A penalty of 1.0 -- the value used
    # while the harness still drove every neuron -- makes silence the *global* minimum
    # and the network collapses to zero firing and chance accuracy on every condition
    # alike, which looks exactly like a clean null result.  Measured on C. elegans at
    # four epochs: reg_rate 1.0 gives rate 0.000 and accuracy 0.049 (chance);
    # 0.1 gives rate 0.088 and accuracy 0.130; 0.0 gives rate 0.244 and accuracy 0.219.
    reg_rate: float = 0.02
    grad_clip: float = 1.0
    seed: int = 0


class MaskedRSNN(nn.Module):
    """Recurrent LIF network whose recurrent support and valences are fixed."""

    def __init__(self, n_in: int, n_out: int, mask: np.ndarray, sign: np.ndarray,
                 cfg: TrainConfig,
                 input_nodes: Optional[np.ndarray] = None,
                 readout_nodes: Optional[np.ndarray] = None):
        super().__init__()
        self.cfg = cfg
        n = mask.shape[0]
        self.n = n
        g = torch.Generator().manual_seed(cfg.seed)

        # Encoder and readout are restricted to disjoint populations.  Both are fixed
        # buffers rather than learned gates, so training cannot widen them and quietly
        # recover the all-to-all protocol.
        in_mask = np.zeros((1, n), dtype=np.float32)
        in_mask[0, np.arange(n) if input_nodes is None else input_nodes] = 1.0
        out_mask = np.zeros(n, dtype=np.float32)
        out_mask[np.arange(n) if readout_nodes is None else readout_nodes] = 1.0
        self.register_buffer("in_mask", torch.tensor(in_mask))
        self.register_buffer("out_mask", torch.tensor(out_mask))

        self.W_in = nn.Parameter(torch.randn(n_in, n, generator=g) / np.sqrt(n_in))
        # Magnitudes are free and non-negative; the mask and the sign vector are buffers.
        init = torch.rand(n, n, generator=g) * (2.0 / max(1.0, mask.sum(axis=1).mean()))
        self.W_mag = nn.Parameter(init)
        self.register_buffer("mask", torch.tensor(mask, dtype=torch.float32))
        self.register_buffer(
            "sign", torch.tensor(np.where(sign == 0, 1, sign), dtype=torch.float32).view(-1, 1)
        )
        self.readout = nn.Linear(n, n_out)

    def recurrent(self) -> torch.Tensor:
        """Effective recurrent matrix: non-negative magnitude, fixed support and valence."""
        return torch.relu(self.W_mag) * self.mask * self.sign

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (B, T, C) -> (logits, mean firing rate per neuron)."""
        B, T, _ = x.shape
        W = self.recurrent()
        v = torch.zeros(B, self.n, device=x.device)
        s = torch.zeros(B, self.n, device=x.device)
        acc = torch.zeros(B, self.n, device=x.device)
        W_in = self.W_in * self.in_mask
        for t in range(T):
            I = x[:, t] @ W_in + s @ W
            v = self.cfg.leak * v * (1.0 - s) + I
            s = spike_fn(v - self.cfg.v_threshold)
            v = v * (1.0 - s)
            acc = acc + s
        rate = acc / T
        return self.readout(rate * self.out_mask), rate


def _synops(model: MaskedRSNN, rate: torch.Tensor) -> float:
    """Synaptic operations per sample: spikes emitted times out-degree."""
    out_deg = model.mask.sum(dim=1)
    T = 1.0
    return float((rate * out_deg).sum(dim=1).mean().item())


def train_and_eval(
    mask: np.ndarray,
    sign: np.ndarray,
    Xtr: np.ndarray, ytr: np.ndarray,
    Xte: np.ndarray, yte: np.ndarray,
    cfg: TrainConfig,
    n_classes: int = 20,
    device: str = "cpu",
    log: Optional[callable] = None,
    input_nodes: Optional[np.ndarray] = None,
    readout_nodes: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    model = MaskedRSNN(Xtr.shape[2], n_classes, mask, sign, cfg,
                       input_nodes=input_nodes,
                       readout_nodes=readout_nodes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    lossf = nn.CrossEntropyLoss()

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    Xte_t = torch.tensor(Xte, dtype=torch.float32)
    yte_t = torch.tensor(yte, dtype=torch.long)

    n = len(ytr_t)
    for ep in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            xb, yb = Xtr_t[idx].to(device), ytr_t[idx].to(device)
            logits, rate = model(xb)
            # An L1 firing-rate penalty keeps activity in a comparable range across
            # conditions, so that a synaptic-operation difference between rungs cannot
            # be manufactured by one condition simply falling silent.  The penalty is
            # identical for every condition.
            loss = lossf(logits, yb) + cfg.reg_rate * rate.mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            tot += float(loss.item()) * len(idx)
        sched.step()
        if log:
            log(ep, tot / n)

    model.eval()
    correct = 0
    syn = 0.0
    rates = 0.0
    nb = 0
    with torch.no_grad():
        for i in range(0, len(yte_t), 256):
            xb = Xte_t[i:i + 256].to(device)
            logits, rate = model(xb)
            correct += int((logits.argmax(1).cpu() == yte_t[i:i + 256]).sum())
            syn += _synops(model, rate) * len(xb)
            rates += float(rate.mean().item()) * len(xb)
            nb += len(xb)

    return dict(
        accuracy=correct / len(yte_t),
        synops_per_sample=syn / nb,
        mean_rate=rates / nb,
        n_synapses=float(mask.sum()),
        n_input_nodes=float(model.in_mask.sum().item()),
        n_readout_nodes=float(model.out_mask.sum().item()),
    )
