"""
Post-hoc probability calibration for both heads, and the pass that fits it.

A head trained under a class weight converges to a posterior tilted by exactly
that weight in logit space; each scaler here is initialized on the analytic
inverse of its own tilt, and an unfittable split still yields sane probabilities.
Fitting is unweighted: the reliability wanted is P(event | score) under the true
class balance.
"""
from typing import Dict, List, Sequence
import math

import torch
import torch.nn as nn
from tqdm import tqdm

from ..config.path_config import PLOTS_DIR
from .plots import reliability_diagram
from .utils import save_calibration


class PlattScaler(nn.Module):
    """
    Affine logit calibration: z_cal = a * z + b, then p = sigmoid(z_cal).

    A single-logit head trained with BCEWithLogitsLoss(pos_weight=w) converges,
    at the population optimum, to z = log(w) + logit(p_true): the class weight
    lands as a constant additive offset of log(w) in logit space. The intercept
    b cancels that offset; the slope a corrects residual over- or
    under-confidence. Initializing (a=1, b=-log(w)) starts the fit on the
    analytic prior correction; an unfittable split still yields sane
    probabilities.

    a is carried as exp(log_a), keeping the map monotone; every ranking score
    (ROC-AUC, PR-AUC) is invariant under calibration.
    """
    def __init__(self, prior_pos_weight: float | None = None):
        super().__init__()
        b0 = -math.log(float(prior_pos_weight)) if prior_pos_weight else 0.0
        self.log_a = nn.Parameter(torch.zeros(1))       # a = 1
        self.b = nn.Parameter(torch.full((1,), b0))     # analytic prior offset

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.log_a) * logits + self.b

    def probs(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(logits))

    @torch.enable_grad()
    def fit(self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 100):
        """ Fit (a, b) by minimizing unweighted BCE on held-out cells.

        Unweighted: the reliability fitted is P(fire | score) under the true
        class balance, not the training-time reweighting.
        """
        z = logits.detach().flatten().float()
        y = labels.detach().flatten().float()

        opt = torch.optim.LBFGS(self.parameters(), lr=0.1, max_iter=max_iter)

        def closure():
            opt.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(self.forward(z), y)
            loss.backward()
            return loss

        opt.step(closure)
        return self

    def state(self) -> Dict[str, float]:
        return {"a": float(torch.exp(self.log_a).item()), "b": float(self.b.item())}

    def load_state(self, params: Dict[str, float]):
        with torch.no_grad():
            self.log_a.fill_(math.log(float(params["a"])))
            self.b.fill_(float(params["b"]))
        return self


class CauseVectorScaler(nn.Module):
    """
    Per-class affine logit calibration for the softmax cause head:
    z'_c = a_c * z_c + b_c.

    A head trained with per-class weights w_c converges, at the population
    optimum, to q_c proportional to w_c * p_c: each class weight lands as an
    additive log(w_c) tilt on that class's logit. init_b = -log(w_c) starts
    the fit on the analytic inverse of that tilt; the unweighted fit that
    follows sees the true class balance.

    a is carried as exp(log_a); every class's scale stays positive.
    """
    def __init__(self, n_classes: int, init_b: Sequence[float] | None = None):
        super().__init__()
        b0 = torch.tensor(init_b, dtype=torch.float32) if init_b is not None else torch.zeros(n_classes)
        self.log_a = nn.Parameter(torch.zeros(n_classes))  # a = 1
        self.b = nn.Parameter(b0)                          # analytic prior offset

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.log_a) * logits + self.b

    def probs(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(logits), dim=-1)

    @torch.enable_grad()
    def fit(self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 100):
        """ Fit (a, b) by minimizing unweighted cross-entropy on held-out cells.

        Unweighted: the reliability fitted is P(cause | score) under the true
        class balance, not the training-time reweighting.
        """
        z = logits.detach().float()
        y = labels.detach().long()

        opt = torch.optim.LBFGS(self.parameters(), lr=0.1, max_iter=max_iter)

        def closure():
            opt.zero_grad()
            loss = nn.functional.cross_entropy(self.forward(z), y)
            loss.backward()
            return loss

        opt.step(closure)
        return self

    def state(self) -> Dict[str, List[float]]:
        return {"a": torch.exp(self.log_a).detach().tolist(), "b": self.b.detach().tolist()}

    def load_state(self, params: Dict[str, List[float]]):
        with torch.no_grad():
            self.log_a.copy_(torch.log(torch.tensor(params["a"], dtype=torch.float32)))
            self.b.copy_(torch.tensor(params["b"], dtype=torch.float32))
        return self


@torch.no_grad()
def expected_calibration_error(
    probs: torch.Tensor, labels: torch.Tensor, num_bins: int = 15
) -> float:
    """ Weighted gap between confidence and accuracy across equal-width bins.

    Bin by predicted probability, and in each bin compare the mean prediction
    to the observed frequency, weighted by bin population.
    """
    p = probs.detach().flatten().float()
    y = labels.detach().flatten().float()
    if p.numel() == 0:
        return float("nan")

    edges = torch.linspace(0.0, 1.0, num_bins + 1, device=p.device)
    ece = torch.zeros((), device=p.device)
    for i in range(num_bins):
        lo, hi = edges[i], edges[i + 1]
        # the last bin owns its right edge; p == 1.0 is counted
        in_bin = (p > lo) & (p <= hi) if i > 0 else (p >= lo) & (p <= hi)
        n = in_bin.sum()
        if n == 0:
            continue
        conf = p[in_bin].mean()
        acc = y[in_bin].mean()
        ece += (n.float() / p.numel()) * (conf - acc).abs()
    return float(ece.item())


def fit_calibration(model, loader, device, neg_keep_rate: float,
                    cause_weight: torch.Tensor, stage_base: str, prepare_targets):
    """ Fit a calibrator for each head on held-out cells and persist both beside
        the checkpoint.

        Scores read the same supervised-cell population as the loss and metrics,
        over the split model selection also reads; the reported ECE is mildly
        optimistic.
    """
    model.eval()
    logits_all, labels_all = [], []
    cause_logits_all, cause_labels_all = [], []
    with torch.no_grad():
        for (x_dyn, x_static), golds, masks in tqdm(loader, desc="Calibrating...", leave=False):
            x_dyn = x_dyn.to(device)
            x_static = x_static.to(device)
            golds = { k: v.to(device) for k, v in golds.items() }
            masks = { k: v.to(device) for k, v in masks.items() }

            ign_golds, cause_golds, ign_mask, cause_mask = prepare_targets(golds, masks)

            # -- raw logits: the calibrator's own intercept carries the whole
            # -- correction; no offset applied here
            ign_logits, cause_logits = model(x_dyn, x_static)
            ign_logits = ign_logits.squeeze(1)        # (B, H, W)

            logits_all.append(ign_logits[ign_mask].float().cpu())
            labels_all.append(ign_golds[ign_mask].float().cpu())
            cause_logits_all.append(
                cause_logits.permute(0, 2, 3, 1)[cause_mask].float().cpu()
            )
            cause_labels_all.append(cause_golds[cause_mask].long().cpu())

    logits = torch.cat(logits_all) if logits_all else torch.empty(0)
    labels = torch.cat(labels_all) if labels_all else torch.empty(0)
    n_cells = int(labels.numel())
    n_pos = int(labels.sum().item())

    # -- the trained head's only offset is the subsampling shift; 1/r is the
    # -- analytic intercept. The fit refines slope and whatever residual remains
    scaler = PlattScaler(prior_pos_weight=1.0 / neg_keep_rate)

    ece_before = expected_calibration_error(torch.sigmoid(logits), labels)

    # a fit needs both classes present; with none, the analytic prior stands
    if 0 < n_pos < n_cells:
        scaler.fit(logits, labels)

    probs = scaler.probs(logits)
    ece_after = expected_calibration_error(probs, labels)

    n_cause = cause_weight.numel()
    cause_logits = torch.cat(cause_logits_all) if cause_logits_all else torch.empty(0, n_cause)
    cause_labels = torch.cat(cause_labels_all) if cause_labels_all else torch.empty(0, dtype=torch.long)
    cause_scaler = CauseVectorScaler(n_cause, init_b=(-torch.log(cause_weight)).tolist())

    # -- a class with no held-out cells gets no gradient and its scale drifts on
    # -- the others alone; the analytic init stands for the whole head
    if int(torch.bincount(cause_labels, minlength=n_cause).min()) > 0:
        cause_scaler.fit(cause_logits, cause_labels)
    cause_state = cause_scaler.state()

    params = {
        **scaler.state(),
        "pos_weight": 1.0,
        "neg_keep_rate": neg_keep_rate,
        "fit_split": "eval",
        "n_cells": n_cells,
        "n_pos": n_pos,
        "ece_before": ece_before,
        "ece_after": ece_after,
        "cause_a": cause_state["a"],
        "cause_b": cause_state["b"],
    }
    calib_path = save_calibration(params, name_base=stage_base)
    print(f"[calibration] a={params['a']:.4f} b={params['b']:.4f} "
          f"ECE {ece_before:.4f} -> {ece_after:.4f}  (n_pos={n_pos}/{n_cells})")
    print(f"[calibration] cause b={[round(v, 3) for v in cause_state['b']]} "
          f"(n_cells={int(cause_labels.numel())})")
    print(f"Saved calibration >> {calib_path}")

    if n_cells > 0:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        reliability_diagram(
            probs, labels, title=f"Reliability ({stage_base})",
            save_path=str(PLOTS_DIR / f"reliability_{stage_base}.png"),
        )
    return params
