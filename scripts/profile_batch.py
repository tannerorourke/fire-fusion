"""
Profile a few training steps of an experiment and print where the time goes.

    python scripts/profile_batch.py --experiment wa2000-s1 [--steps 4]

Builds the trainer exactly as fire_fusion.training.train does, then swaps the training
loop for a short torch.profiler run: a per-op table sorted by CUDA time, plus
a chrome trace under logs/ for chrome://tracing.
"""

import argparse
import json
from pathlib import Path

import torch
from torch import nn, optim
from torch.amp.autocast_mode import autocast
from torch.profiler import ProfilerActivity, profile, schedule

from fire_fusion.config.path_config import MODEL_DIR
from fire_fusion.training.train import WRMTrainer
from fire_fusion.training.utils import get_device_config


class ProfilingTrainer(WRMTrainer):
    steps = 4

    def train(self):
        """ Run a few training steps under torch.profiler, printing a per-op CUDA
            time table and exporting a chrome trace to logs/.
        """
        self.model.train()
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.base_lr, weight_decay=self.weight_decay,
        )

        sched = schedule(wait=1, warmup=2, active=self.steps)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     schedule=sched) as prof:
            for b_idx, ((x_dyn, x_static), golds, masks) in enumerate(self.train_loader):
                if b_idx >= 3 + self.steps:
                    break
                x_dyn = x_dyn.to(self.device)
                x_static = x_static.to(self.device)
                golds = {k: v.to(self.device) for k, v in golds.items()}
                masks = {k: v.to(self.device) for k, v in masks.items()}

                ign_golds, cause_golds, ign_mask, cause_mask = self._prepare_targets(golds, masks)
                loss_mask = self._thinned_mask(ign_golds, ign_mask)
                with autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
                    ign_logits, cause_logits = self.model(x_dyn, x_static)
                    tot_loss, _, _ = self._compute_loss(
                        ign_logits, ign_golds, cause_logits, cause_golds,
                        loss_mask, cause_mask,
                        alpha_ign=self.alpha_ign, alpha_cause=self.alpha_cause,
                    )
                tot_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                prof.step()

        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=25))
        Path("logs").mkdir(exist_ok=True)
        trace = f"logs/profile_{self.experiment}.json"
        prof.export_chrome_trace(trace)
        print(f"chrome trace >> {trace}")


def main():
    """ Parse args, build a ProfilingTrainer for the named experiment, and let
        construction run the profiled training steps.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--steps", type=int, default=4,
                    help="profiled steps, after 3 wait/warmup batches")
    args = ap.parse_args()

    params = json.load(open(MODEL_DIR / "params.json"))[args.experiment]
    mp, tp = params["model"], params["training"]
    device, num_workers = get_device_config(maximum=tp.get("max_workers", 8))

    ProfilingTrainer.steps = args.steps
    # -- alpha_cause fixed at 1.0: the auto measurement costs four extra forward
    #    passes and has no bearing on per-op timing
    ProfilingTrainer(
        mp, tp, device, num_workers,
        dataset_name=params["dataset"], experiment=args.experiment,
        seed=tp.get("seed", 0), alpha_cause=1.0,
    )


if __name__ == "__main__":
    main()
