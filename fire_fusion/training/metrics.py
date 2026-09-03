"""
Streaming evaluation metrics for the two heads, plus the manager that drives
them per epoch.

Every metric accumulates into fixed-size state (counts, a confusion matrix, a
score histogram, running sums). An epoch of per-cell predictions runs to
billions of entries at the finer grid resolutions.
"""
from typing import Dict, List, Literal, Optional, Tuple
import math

import numpy as np
import torch
import torch.nn as nn

from ..analysis.archive import last_day


class Metric:
    def __init__(self):
        self.record = []
    def reset(self) -> None:
        raise NotImplementedError
    def add(self, preds: torch.Tensor, labels: torch.Tensor, mask: Optional[torch.Tensor] = None):
        raise NotImplementedError
    def compute_step(self) -> Dict:
        raise NotImplementedError
    def get_history(self):
        raise NotImplementedError



class Accuracy(Metric):
    def __init__(self):
        super().__init__()
        self.record = []
        self.ep_correct = 0
        self.ep_total = 0

    def reset(self) -> None:
        self.ep_correct = 0
        self.ep_total = 0

    @torch.no_grad()
    def add(self, preds: torch.Tensor, labels: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if mask is not None:
            preds = preds[mask]
            labels = labels[mask]

        self.ep_correct += (preds.type_as(labels) == labels).sum().item()
        self.ep_total += labels.numel()

    def compute_step(self) -> dict[str, float]:
        """ Return scores for the epoch, reset internal state, and update p/epoch record """
        acc = self.ep_correct / (self.ep_total + 1e-6)
        self.record.append(acc)

        scores = {
            f"accuracy": acc,
            f"n_samples": self.ep_total
        }
        self.reset()

        return scores
    
    def get_history(self):
        return self.record



class ConfusionMatrix(Metric):
    """
    Metric for computing mean IoU, accuracy, precision, recall, F1, and confusion matrix.

    Counts accumulate into the matrix itself; memory is O(num_classes^2), not
    O(cells seen). Every score below is a function of the matrix alone.
    """

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.num_classes = num_classes
        self.record: List[Dict] = []
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self):
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    @torch.no_grad()
    def add(self, preds: torch.Tensor, labels: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Update using predicted and ground truth class indices.

        Args:
            preds:  (B, ...) class indices (see MetricsManager._logits_to_preds)
            labels: (B, ...) ground truth class indices
            mask:   optional (B, ...) selection of the cells to score
        """
        if mask is not None:
            preds = preds[mask]
            labels = labels[mask]

        preds_np = preds.detach().cpu().numpy().astype(np.int64).ravel()
        labels_np = labels.detach().cpu().numpy().astype(np.int64).ravel()

        # cells carrying no class (cause is -1 wherever no ignition is labeled)
        # sit outside the matrix and are not scored
        keep = (
            (labels_np >= 0) & (labels_np < self.num_classes)
            & (preds_np >= 0) & (preds_np < self.num_classes)
        )
        if not keep.any():
            return

        flat = labels_np[keep] * self.num_classes + preds_np[keep]
        counts = np.bincount(flat, minlength=self.num_classes ** 2)
        self.matrix += counts.reshape(self.num_classes, self.num_classes)

    def compute_step(
        self,
        roc_auc: Optional[float] = None,
        pr_auc: Optional[float] = None,
        recall_at_prev: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Compute metrics for the epoch, append to record, and reset internal storage.

        roc_auc / pr_auc:
            Optional AUC scores for this epoch (computed elsewhere from raw logits).
        """
        cm = self.matrix
        total = float(cm.sum())

        if total == 0:
            scores = {
                "mean_iou": 0.0,
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "recall_at_prev": recall_at_prev,
            }
            self.record.append({**scores, "matrix": cm.copy()})
            self.reset()
            return scores

        # rows are ground truth, columns are predictions
        tp = np.diag(cm).astype(np.float64)
        fp = cm.sum(axis=0).astype(np.float64) - tp
        fn = cm.sum(axis=1).astype(np.float64) - tp

        def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
            """ Return num / den elementwise, giving 0 where den is 0 instead of NaN. """
            return np.divide(num, den, out=np.zeros_like(num), where=den > 0)

        iou = _ratio(tp, tp + fp + fn)
        precision = _ratio(tp, tp + fp)
        recall = _ratio(tp, tp + fn)
        f1 = _ratio(2.0 * precision * recall, precision + recall)

        scores = {
            "mean_iou": float(iou.mean()),
            "accuracy": float(tp.sum() / total),
            "precision": float(precision.mean()),
            "recall": float(recall.mean()),
            "f1": float(f1.mean()),
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "recall_at_prev": recall_at_prev,
        }

        self.record.append({**scores, "matrix": cm.copy()})
        self.reset()

        return scores

    def get_history(self) -> Tuple[np.ndarray | None, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], List[Dict]]:
        """
        Returns:
            last_cm: latest confusion matrix (or None if no epochs)
            rates:  (tpr, tnr, fpr, fnr) arrays over epochs (binary-only; empty otherwise)
            record: raw record list
        """
        if not self.record:
            empty = np.zeros(0, dtype=float)
            return None, (empty, empty, empty, empty), []

        matrices = [r["matrix"] for r in self.record]
        last_cm = matrices[-1]

        # For non-binary heads, skip rate computation but still return matrices
        if self.num_classes != 2:
            empty = np.zeros(0, dtype=float)
            return last_cm, (empty, empty, empty, empty), self.record

        # Binary rates per epoch
        tpr_list, tnr_list, fpr_list, fnr_list = [], [], [], []
        for cm in matrices:
            # cm: [[TN, FP],
            #      [FN, TP]]
            tn, fp, fn, tp = cm.ravel()
            tpr = tp / (tp + fn + 1e-6)
            tnr = tn / (tn + fp + 1e-6)
            fpr = fp / (fp + tn + 1e-6)
            fnr = fn / (fn + tp + 1e-6)
            tpr_list.append(tpr)
            tnr_list.append(tnr)
            fpr_list.append(fpr)
            fnr_list.append(fnr)

        rates = (
            np.asarray(tpr_list, dtype=float),
            np.asarray(tnr_list, dtype=float),
            np.asarray(fpr_list, dtype=float),
            np.asarray(fnr_list, dtype=float),
        )
        return last_cm, rates, self.record




class BinaryAUC(Metric):
    """
    ROC-AUC and PR-AUC for a single-logit head, accumulated as per-class
    histograms of the score.

    Threshold-free ranking scores separate a useful ignition model from one
    that answers "no fire" everywhere; accuracy reads ~1.0 at this dataset's
    class ratio. Bucketing scores holds memory at O(num_bins).
    """

    def __init__(self, num_bins: int = 4096, logit_range: Tuple[float, float] = (-16.0, 16.0)):
        """ Empty positive and negative score histograms over num_bins buckets
            spanning logit_range. """
        super().__init__()
        self.num_bins = num_bins
        self.lo, self.hi = logit_range
        self.pos = np.zeros(num_bins, dtype=np.int64)
        self.neg = np.zeros(num_bins, dtype=np.int64)

    def reset(self) -> None:
        self.pos[:] = 0
        self.neg[:] = 0

    @torch.no_grad()
    def add(self, logits: torch.Tensor, labels: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Args:
            logits: (B, 1, H, W) or (B, H, W) raw scores for the positive class
            labels: (B, H, W) ground truth in {0, 1}
            mask:   optional (B, H, W) selection of the cells to score
        """
        if logits.dim() == 4 and logits.size(1) == 1:
            logits = logits.squeeze(1)

        if mask is not None:
            logits = logits[mask]
            labels = labels[mask]
        if labels.numel() == 0:
            return

        # scores outside the range rank above/below every in-range score;
        # clamping preserves the ordering
        scores = logits.detach().float().flatten().clamp(self.lo, self.hi)
        width = (self.hi - self.lo) / self.num_bins
        bins = ((scores - self.lo) / width).long().clamp(0, self.num_bins - 1)

        bins_np = bins.cpu().numpy()
        labels_np = labels.detach().flatten().cpu().numpy()

        self.pos += np.bincount(bins_np[labels_np == 1], minlength=self.num_bins)
        self.neg += np.bincount(bins_np[labels_np != 1], minlength=self.num_bins)

    def compute_step(self) -> Dict[str, float]:
        """ ROC-AUC, PR-AUC, and recall at the n_pos-alarm operating point from the
            binned score histograms; appended to history, then reset. """
        n_pos, n_neg = int(self.pos.sum()), int(self.neg.sum())

        if n_pos == 0 or n_neg == 0:
            scores = {"roc_auc": float("nan"), "pr_auc": float("nan"),
                      "recall_at_prev": float("nan")}
            self.record.append(scores)
            self.reset()
            return scores

        # sweep the decision threshold from the highest-scoring bucket downward
        tp = np.cumsum(self.pos[::-1]).astype(np.float64)
        fp = np.cumsum(self.neg[::-1]).astype(np.float64)

        recall = tp / n_pos
        precision = tp / np.maximum(tp + fp, 1.0)
        fpr = fp / n_neg

        # trapezoid over the ROC curve, anchored at the origin
        r = np.concatenate([[0.0], recall])
        f = np.concatenate([[0.0], fpr])
        roc_auc = float(np.sum(np.diff(f) * (r[1:] + r[:-1]) / 2.0))

        # average precision: each threshold's precision weighted by the recall it adds
        pr_auc = float(np.sum(np.diff(r) * precision))

        # -- recall when exactly n_pos cells are alarmed: an operating point a
        #    0.5 threshold never reaches at this prevalence. Rank-based;
        #    invariant to monotone recalibration.
        k = min(int(np.searchsorted(tp + fp, float(n_pos))), self.num_bins - 1)
        recall_at_prev = float(recall[k])

        scores = {"roc_auc": roc_auc, "pr_auc": pr_auc,
                  "recall_at_prev": recall_at_prev}
        self.record.append(scores)
        self.reset()

        return scores

    def get_history(self) -> List[Dict]:
        return self.record



class MeanIgnorance(Metric):
    """
    Masked mean binary ignorance (log score) in bits, for a single-logit head.

    Ignorance is BCE-with-logits converted from nats to bits: the number of
    bits the model needed to be told, on average, to learn the true label:
    zero for a certain correct call, unbounded for a confident wrong one.
    Accumulated as a running sum and count; memory is O(1) per epoch.
    """

    def __init__(self):
        super().__init__()
        self.sum_bits = 0.0
        self.n_cells = 0

    def reset(self) -> None:
        self.sum_bits = 0.0
        self.n_cells = 0

    @torch.no_grad()
    def add(self, logits: torch.Tensor, labels: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Args:
            logits: (B, 1, H, W) or (B, H, W) raw scores for the positive class
            labels: (B, H, W) ground truth in {0, 1}
            mask:   optional (B, H, W) selection of the cells to score
        """
        if logits.dim() == 4 and logits.size(1) == 1:
            logits = logits.squeeze(1)

        if mask is not None:
            logits = logits[mask]
            labels = labels[mask]
        if labels.numel() == 0:
            return

        z = logits.detach().float().flatten()
        y = labels.detach().float().flatten()
        bce = nn.functional.binary_cross_entropy_with_logits(z, y, reduction="none")
        self.sum_bits += float((bce / math.log(2)).sum().item())
        self.n_cells += y.numel()

    def compute_step(self) -> Dict[str, float]:
        """ Compute mean ignorance in bits from the running sum, append to history, and reset. """
        ign = self.sum_bits / self.n_cells if self.n_cells > 0 else float("nan")
        scores = {"ign_bits": ign, "n_cells": self.n_cells}
        self.record.append(scores)
        self.reset()
        return scores

    def get_history(self) -> List[Dict]:
        return self.record



class MeanAllocationIgnorance(Metric):
    """
    Masked mean allocation ignorance in bits, for a single-logit head.

    Each sample's supervised cells are renormalized into one distribution over
    that sample's day, and the score is the surprise of the cells that ignited:
    log2 of the number of equally likely cells the field has narrowed them to.
    Uniform over m supervised cells reads log2(m); a field that concentrates
    probability where fires land reads lower.

    A constant added to a day's logits cancels in the softmax; this score is
    blind to the overall ignition rate. Samples with no ignition contribute
    nothing.
    """

    def __init__(self, events: Optional[str] = None):
        """ events=None scores every positive; 'ignition' only positives with no
            active cell within 'near' (given per call); 'spread' the rest. The
            normalizer is the whole supervised day either way. """
        super().__init__()
        self.events = events
        self.reset()

    def reset(self) -> None:
        self.sum_bits = 0.0
        self.n_events = 0
        self.sum_uniform = 0.0

    @torch.no_grad()
    def add(self, logits: torch.Tensor, labels: torch.Tensor, mask: Optional[torch.Tensor] = None,
            near: Optional[torch.Tensor] = None):
        """
        Args:
            logits: (B, 1, H, W) or (B, H, W) raw scores for the positive class
            labels: (B, H, W) ground truth in {0, 1}
            mask:   optional (B, H, W) selection of the cells to score
            near:   (B, H, W) bool, within reach of an active cell; needed when
                    the metric scores one stratum
        """
        if logits.dim() == 4 and logits.size(1) == 1:
            logits = logits.squeeze(1)

        for b in range(logits.shape[0]):
            m = (mask[b] if mask is not None
                 else torch.ones_like(labels[b], dtype=torch.bool))
            pos = (labels[b] == 1) & m
            if self.events == "ignition":
                pos = pos & ~near[b]
            elif self.events == "spread":
                pos = pos & near[b]
            k = int(pos.sum())
            if k == 0:
                continue
            log_q = torch.log_softmax(logits[b][m].detach().float().flatten(), dim=0)
            self.sum_bits += float(-log_q[pos[m]].sum().item()) / math.log(2)
            self.sum_uniform += k * math.log2(int(m.sum()))
            self.n_events += k

    def compute_step(self) -> Dict[str, float]:
        """ Mean allocation ignorance in bits and skill against a uniform baseline;
            appended to history, then reset. """
        n = self.n_events
        bits = self.sum_bits / n if n > 0 else float("nan")
        uniform = self.sum_uniform / n if n > 0 else float("nan")
        scores = {"alloc_bits": bits, "alloc_uniform_bits": uniform,
                  "alloc_skill_vs_uniform": 1.0 - bits / uniform if n > 0 else float("nan"),
                  "n_events": n}
        self.record.append(scores)
        self.reset()
        return scores

    def get_history(self) -> List[Dict]:
        return self.record


class MetricsManager:
    def __init__(
        self,
        num_classes: Tuple = (2,),
        select_by: Literal["pr_auc", "val_loss", "val_ign", "val_alloc"] = "pr_auc",
    ):
        """
        num_classes: per-head class counts, (2, 4) for binary + 4-class heads.
        select_by: what "best epoch" and early stopping key off.
            "pr_auc"    - masked PR-AUC of the ignition head (maximized)
            "val_loss"  - total validation loss (minimized), cause term included
            "val_ign"   - masked mean eval ignorance in bits (minimized); nan never selected
            "val_alloc" - masked mean eval allocation ignorance in bits (minimized)
        """
        assert num_classes and num_classes[0] == 2, \
            "head 0 is the binary ignition head; its ranking scores assume a single logit"

        self.num_classes = num_classes
        self.num_heads = len(num_classes)
        self.select_by = select_by

        self.trn_accuracies = [Accuracy() for _ in range(self.num_heads)]
        self.val_accuracies = [Accuracy() for _ in range(self.num_heads)]

        # One confusion matrix per head, with correct class count
        self.val_cm = [ConfusionMatrix(nc) for nc in self.num_classes]

        # Ranking scores for the binary ignition head
        self.val_auc = BinaryAUC()

        # Masked mean eval ignorance (bits) for the binary ignition head
        self.val_ign = MeanIgnorance()

        # Masked mean eval allocation ignorance (bits) for the same head, and
        # the ignition stratum alone; selection reads the spread-dominated whole
        self.val_alloc = MeanAllocationIgnorance()
        self.val_alloc_ign = MeanAllocationIgnorance(events="ignition")

        # Loss history: (num_loss_terms, num_epochs)
        self.trn_losses: Optional[np.ndarray] = None
        self.val_losses: Optional[np.ndarray] = None

        self.best = {
            "epoch": 0,
            "score": -float("inf") if select_by == "pr_auc" else float("inf"),
        }
        self.epoch = 1
        self.no_improve = 0

        # Last epoch_forward's console log and its scalar breakdown, mirrored
        # by the caller to TensorBoard.
        self.last_report: str = ""
        self.last_scalars: Dict[str, float] = {}

    def _is_improvement(self, score: float) -> bool:
        if self.select_by == "pr_auc":
            # an epoch whose eval split carried no positives scores nan and
            # cannot be ranked against anything
            return bool(np.isfinite(score)) and score > self.best["score"]
        if self.select_by in ("val_ign", "val_alloc"):
            # an epoch with no scored cells scores nan and cannot be ranked
            return bool(np.isfinite(score)) and score < self.best["score"]
        return score < self.best["score"]

    @staticmethod
    def _logits_to_preds(logits: torch.Tensor, n_classes: int) -> torch.Tensor:
        if logits.dim() > 1 and logits.size(1) == n_classes and n_classes > 1:
            return torch.argmax(logits, dim=1)

        if logits.dim() > 1 and logits.size(1) == 1 and n_classes == 2:
            return (logits.squeeze(1) > 0).long()

        raise ValueError(
            f"logits {tuple(logits.shape)} do not describe {n_classes} classes; "
            f"expected (B, {n_classes}, ...), or (B, 1, ...) for a binary head"
        )

    def add(
        self,
        type: Literal["train", "eval"],
        logits: List[torch.Tensor],
        golds: List[torch.Tensor],
        masks: Optional[List[torch.Tensor]] = None,
        near: Optional[torch.Tensor] = None,
    ):
        """
        One entry per output head. `near` marks cells within reach of active
        fire on the day and splits the allocation score into strata. Each mask
        selects the cells that head is supervised on; the scores describe the
        same population as the loss. Unmasked scores over every cell are
        dominated by ocean and by the no-ignition class.
        """
        assert len(logits) == self.num_heads, f"send one logit tensor for each ({self.num_heads}) output head"
        assert len(golds) == self.num_heads, f"send one golds tensor for each ({self.num_heads}) output head"
        assert masks is None or len(masks) == self.num_heads, \
            f"send one mask tensor for each ({self.num_heads}) output head"

        def mask_for(head: int) -> Optional[torch.Tensor]:
            return masks[head] if masks is not None else None

        accuracies = self.trn_accuracies if type == "train" else self.val_accuracies
        for i, acc in enumerate(accuracies):
            preds_i = self._logits_to_preds(logits[i], self.num_classes[i])
            labels_i = last_day(golds[i]).long()
            acc.add(preds_i, labels_i, mask_for(i))

        if type != "eval":
            return

        for i, cm in enumerate(self.val_cm):
            preds_i = self._logits_to_preds(logits[i], self.num_classes[i])
            labels_i = last_day(golds[i]).long()
            cm.add(preds_i, labels_i, mask_for(i))

        self.val_auc.add(logits[0], last_day(golds[0]).long(), mask_for(0))
        self.val_ign.add(logits[0], last_day(golds[0]).long(), mask_for(0))
        self.val_alloc.add(logits[0], last_day(golds[0]).long(), mask_for(0))
        if near is not None:
            self.val_alloc_ign.add(logits[0], last_day(golds[0]).long(), mask_for(0), near)

    def add_epoch_totals(
        self,
        type: Literal["train", "eval"],
        losses: np.ndarray,
    ):
        """ Append this epoch's loss-term totals as a new column of the
            (num_loss_terms, num_epochs) loss history. """
        new_col = np.asarray(losses).reshape(-1, 1)

        if type == "train":
            if self.trn_losses is None:
                self.trn_losses = new_col
            else:
                self.trn_losses = np.concatenate([self.trn_losses, new_col], axis=1)
        elif type == "eval":
            if self.val_losses is None:
                self.val_losses = new_col
            else:
                self.val_losses = np.concatenate([self.val_losses, new_col], axis=1)

    def epoch_forward(self):
        """
        Print losses for this epoch, update best score, and increment epoch counter.
        Assumes add_epoch_totals() has been called for both train and val.
        Also finalizes per-epoch accuracies and confusion matrices.
        """
        assert self.trn_losses is not None and self.val_losses is not None, "Call add_epoch_totals() before epoch_forward"

        # Finalize accuracies for this epoch (fills .record in Accuracy)
        for acc in self.trn_accuracies:
            acc.compute_step()
        for acc in self.val_accuracies:
            acc.compute_step()

        # Finalize confusion matrices for this epoch (fills .record in ConfusionMatrix).
        # The ignition head carries the epoch's ranking scores alongside its matrix.
        ign_scores = self.val_cm[0].compute_step(**self.val_auc.compute_step())
        for cm in self.val_cm[1:]:
            cm.compute_step()

        # finalized every epoch regardless of select_by; the report line
        # and TensorBoard scalar always carry it
        ign_bits = self.val_ign.compute_step()["ign_bits"]
        alloc = self.val_alloc.compute_step()
        alloc_ign = self.val_alloc_ign.compute_step()

        trn_last = self.trn_losses[:, -1]
        val_last = self.val_losses[:, -1]

        # the ignition head's ranking quality is the claim under test; total
        # validation loss also carries the sparse, high-variance cause term
        score = float(
            ign_scores["pr_auc"] if self.select_by == "pr_auc"
            else ign_bits if self.select_by == "val_ign"
            else alloc["alloc_bits"] if self.select_by == "val_alloc"
            else val_last[0]
        )

        trn_total, trn_ign, trn_cause = trn_last[:3]
        val_total, val_ign, val_cause = val_last[:3]

        report = (
            f"[Epoch {self.epoch}]\n"
            f"Train >> mL (total): {trn_total:.4f}, "
            f"mL (ign): {trn_ign:.4f}, "
            f"mL (cause): {trn_cause:.3f}\n"
            f"Eval   >> mL (total): {val_total:.4f}, "
            f"mL (ign): {val_ign:.4f}, "
            f"mL (cause): {val_cause:.3f}\n"
            f"Ignition (supervised cells) >> "
            f"PR-AUC: {ign_scores['pr_auc']:.5f}, "
            f"ROC-AUC: {ign_scores['roc_auc']:.4f}, "
            f"recall@prev: {ign_scores['recall_at_prev']:.4f}, "
            f"ignorance (bits): {ign_bits:.4f}\n"
            f"Allocation >> bits/event: {alloc['alloc_bits']:.4f} "
            f"(uniform {alloc['alloc_uniform_bits']:.4f}, "
            f"skill {alloc['alloc_skill_vs_uniform']:+.4f}, "
            f"events {alloc['n_events']}); ignition stratum "
            f"bits/event {alloc_ign['alloc_bits']:.4f} skill {alloc_ign['alloc_skill_vs_uniform']:+.4f} "
            f"events {alloc_ign['n_events']}\n"
            f"         SCORE ({self.select_by}): {score:.5f}"
        )
        print(report)

        self.last_report = report
        self.last_scalars = {
            "loss/train_total": float(trn_total), "loss/train_ign": float(trn_ign),
            "loss/train_cause": float(trn_cause),
            "loss/eval_total": float(val_total), "loss/eval_ign": float(val_ign),
            "loss/eval_cause": float(val_cause),
            "ign/pr_auc": float(ign_scores["pr_auc"]), "ign/roc_auc": float(ign_scores["roc_auc"]),
            "ign/recall_at_prev": float(ign_scores["recall_at_prev"]),
            "ign/val_bits": float(ign_bits),
            "ign/val_alloc_bits": float(alloc["alloc_bits"]),
            "ign/val_alloc_skill": float(alloc["alloc_skill_vs_uniform"]),
            "ign/val_alloc_ign_bits": float(alloc_ign["alloc_bits"]),
            "ign/val_alloc_ign_skill": float(alloc_ign["alloc_skill_vs_uniform"]),
            "score": float(score),
        }

        new_best = False
        if self._is_improvement(score):
            print(f"NEW BEST! SCORE={score:.5f}\n")
            new_best = True
            self.best["epoch"] = self.epoch
            self.best["train_loss"] = trn_last.copy()
            self.best["eval_loss"] = val_last.copy()
            self.best["score"] = score
            self.no_improve = 0
        else:
            self.no_improve += 1

        self.epoch += 1
        return score, new_best, trn_last, val_last
    
    def get_history(self):
        return self.trn_losses, self.val_losses