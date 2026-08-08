import os
import json
import math
import random
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim

from .config.path_config import MODEL_SAVE_DIR


def estimate_model_size_mb(model: torch.nn.Module) -> float:
    # -- 4 bytes per fp32 parameter; buffers and optimizer state not counted
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def set_global_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -- the name carries the experiment, so two runs never overwrite each other
def checkpoint_name(experiment: str) -> str:
    return f"{experiment}_model"


def get_device_config(maximum: int | None = None, utilization: float | None = 0.75):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # -- TF32 covers the fp32 matmuls left outside the bf16 autocast; free
    #    accuracy-wise at these magnitudes, large on Ampere+ tensor cores
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    cpus = os.cpu_count() or 1

    if utilization is not None:
        workers = math.floor(cpus * utilization)
    else:
        workers = 1
    if maximum is not None:
        workers = max(1, min(workers, maximum))

    if torch.cuda.is_available():
        print(f"Device: {device}, {torch.cuda.get_device_name(0)}")
    print(f"Using {workers}/{cpus} CPUs")
    return device, workers


def save_model(
    model: torch.nn.Module,
    name_base: str = "model",
    overwrite: bool = False,
) -> str:
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # -- overwrite takes the fixed `<name_base>.th`, for a checkpoint re-saved as
    # -- a run improves; otherwise the next free `<name_base>_<i>.th`
    if overwrite:
        output_path = MODEL_SAVE_DIR / f"{name_base}.th"
    else:
        i = 1
        while (Path(MODEL_SAVE_DIR / f"{name_base}_{i}.th").exists()):
            i += 1
        output_path = MODEL_SAVE_DIR / f"{name_base}_{i}.th"

    # -- a torch.compile wrapper prefixes keys with '_orig_mod.'; saving the
    #    plain module keeps checkpoints loadable in uncompiled processes
    torch.save(getattr(model, "_orig_mod", model).state_dict(), output_path)

    return str(output_path)


def export_to_b2(
    *paths: str | Path,
    prefix: str = "runs",
) -> bool:
    # -- a file lands at `<prefix>/<filename>`, a directory recursively under its
    # -- own relative paths, so a per-run prefix reconstructs the whole run once
    # -- the cloud box is torn down. Credentials come from the B2_* environment.
    try:
        from .dataset.fetch_cloud import B2Store
        store = B2Store()
        for p in paths:
            path = Path(p)
            if not path.exists():
                print(f"[export_to_b2] {path.name} missing locally, skipping")
                continue
            if path.is_dir():
                for f in sorted(path.rglob("*")):
                    if f.is_file():
                        rel = f.relative_to(path).as_posix()
                        store.put_file(f, f"{prefix}/{rel}", overwrite=True)
            else:
                store.put_file(path, f"{prefix}/{path.name}", overwrite=True)
        return True
    except Exception as e:
        # -- the weights are already on local disk by now, so a credentials or
        # -- network failure should not take down an otherwise complete run
        print(f"[export_to_b2] upload failed ({type(e).__name__}): {e}")
        return False


def load_model(
    model: torch.nn.Module,
    path: str | os.PathLike,
    map_location=None,
    strict: bool = True,
):
    # -- a bare name resolves under MODEL_SAVE_DIR, mirroring save_model's layout.
    # -- strict=False tolerates a checkpoint covering part of the model, such as a
    # -- backbone loaded into freshly initialized heads, and the return value lists
    # -- whichever keys were missing or unexpected.
    p = Path(path)
    if p.parent == Path("."):
        p = MODEL_SAVE_DIR / p

    state = torch.load(p, map_location=map_location)
    return getattr(model, "_orig_mod", model).load_state_dict(state, strict=strict)


def save_calibration(params: dict, name_base: str = "model") -> str:
    # -- the probabilities a checkpoint produces depend on both its weights and the
    # -- fitted calibration, so the sidecar travels under the checkpoint's name
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = MODEL_SAVE_DIR / f"{name_base}.calib.json"
    with open(output_path, "w") as f:
        json.dump(params, f, indent=2)
    return str(output_path)


def load_calibration(name_base: str = "model") -> dict | None:
    # -- a bare name resolves under MODEL_SAVE_DIR; a path ending in .calib.json is
    # -- read as given. Absent means no fit available, which the predictor answers
    # -- with the analytic prior correction.
    p = Path(name_base)
    if p.suffix != ".json":
        p = MODEL_SAVE_DIR / f"{p.name}.calib.json"
    elif p.parent == Path("."):
        p = MODEL_SAVE_DIR / p
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


class WarmupCosineAnnealingLR:
    """ 
    PyTorch CosineAnnealing learning rate, with a linear warmup step
        https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LinearLR.html
        https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
    """
    def __init__(self,
        optimizer,
        warmup_steps: int, total_steps: int,
        min_lr: float = 1e-6
    ):
        self.optimizer = optimizer

        w_steps = max(0, warmup_steps)
        base_lr = float(optimizer.param_groups[0]["lr"])

        # LinearLR scales base_lr by start_factor and requires it in (0, 1]. A
        # base_lr of 0 (the null-learning profile) leaves the ratio undefined,
        # and any factor of zero is still zero, so warm up at full scale.
        start_factor = min_lr / base_lr if base_lr > 0 else 1.0
        start_factor = float(min(1.0, max(start_factor, 1e-8)))

        warmup = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor = start_factor if w_steps > 0 else 1.0,
            total_iters = warmup_steps if warmup_steps > 0 else 1
        )
        cosine = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max = max(1, int(total_steps - warmup_steps)),
            eta_min = min_lr,
        )
        self.sched = optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[w_steps]
        )

    def step(self): self.sched.step()
    def state_dict(self): return self.sched.state_dict()
    def load_state_dict(self, sd): self.sched.load_state_dict(sd)
    def get_last_lr(self): return self.sched.get_last_lr()
    @property
    def last_epoch(self): return self.sched.last_epoch