"""
Per-run training curves and the reliability diagram, written as PNGs under
PLOTS_DIR. Every figure is named for the run's checkpoint base; artifacts
from different experiments never overwrite one another.
"""
from typing import List, Tuple

import numpy as np
import torch
from sklearn.calibration import calibration_curve

from ..config.path_config import PLOTS_DIR
from .metrics import Accuracy


def _pyplot():
    """ Set the Agg backend at call time and return pyplot. """
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    return plt


def _finish(plt, save: bool, save_path: str | None, default: str):
    if not save:
        plt.show()
        return
    PLOTS_DIR.mkdir(exist_ok=True, parents=True)
    plt.savefig(save_path if save_path is not None else str(PLOTS_DIR / default),
                bbox_inches="tight", dpi=200)
    plt.close()


def plot_class_accuracy(
    epochs: List,
    val_ignit_acc: Accuracy,
    val_cause_acc: Accuracy,
    trn_ignit_acc: Accuracy,
    trn_cause_acc: Accuracy,
    save: bool = True,
    save_path: str | None = None,
):
    """ Plot per-epoch ignition and cause accuracy for train and val; save or show. """
    plt = _pyplot()
    plt.figure()
    plt.plot(epochs, trn_ignit_acc.record, label="Ignition acc (train)")
    plt.plot(epochs, trn_cause_acc.record, label="Cause acc (train)")
    plt.plot(epochs, val_ignit_acc.record, label="Ignition acc (val)")
    plt.plot(epochs, val_cause_acc.record, label="Cause acc (val)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation accuracy per epoch")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    _finish(plt, save, save_path, "class_accuracy.png")


def plot_loss_curves(
    epochs: List,
    trn_losses,  # (num_loss_terms, num_epochs); one line per term
    val_losses,
    term_names: Tuple[str, ...] = ("total", "ignition", "cause"),
    save: bool = True,
    save_path: str | None = None,
):
    """ Plot per-epoch train and val loss curves for each loss term; save or show. """
    trn_losses = np.atleast_2d(np.asarray(trn_losses, dtype=float))
    val_losses = np.atleast_2d(np.asarray(val_losses, dtype=float))

    def term_label(i: int) -> str:
        return term_names[i] if i < len(term_names) else f"term {i}"

    plt = _pyplot()
    plt.figure()
    for i, curve in enumerate(trn_losses):
        plt.plot(epochs, curve, label=f"Train loss ({term_label(i)})")
    for i, curve in enumerate(val_losses):
        plt.plot(epochs, curve, linestyle="--", label=f"Val loss ({term_label(i)})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss per epoch")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    _finish(plt, save, save_path, "losses.png")


def plot_rates_per_epoch(epochs: List, rates: Tuple, save: bool = True,
                         save_path: str | None = None):
    """ Plot per-epoch TPR, TNR, FPR, and FNR, and save or show the figure. """
    tpr, tnr, fpr, fnr = rates

    plt = _pyplot()
    plt.figure()
    plt.plot(epochs, tpr, label="TPR (recall)")
    plt.plot(epochs, tnr, label="TNR (specificity)")
    plt.plot(epochs, fpr, label="FPR")
    plt.plot(epochs, fnr, label="FNR")
    plt.xlabel("Epoch")
    plt.ylabel("Rate")
    plt.title("Ignition rates per epoch")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    _finish(plt, save, save_path, "rates.png")


def reliability_diagram(
    probs: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
    num_bins: int = 10,
    title: str = "Reliability diagram",
    save_path: str | None = None,
):
    """ Predicted probability against observed frequency over equal-width bins;
        the diagonal is perfect calibration. """
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)

    frac_pos, mean_pred = calibration_curve(
        labels, probs, n_bins=num_bins, strategy="uniform"
    )

    plt = _pyplot()
    plt.figure()
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.plot(mean_pred, frac_pos, marker="o", label="Model")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Empirical frequency")
    plt.title(title)
    plt.grid(True)
    plt.legend()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
        plt.close()
    else:
        plt.show()
