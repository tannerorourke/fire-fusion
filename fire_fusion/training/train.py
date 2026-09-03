"""
Trainer for the two-head ConvFormer: per-cell ignition BCE, an allocation term
scoring placement alone, and cause cross-entropy, over a built fold store.

  python -m fire_fusion.training.train --experiment wa2000-s1
"""
import argparse
import json
import math
from time import perf_counter, strftime
from typing import Dict, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp.autocast_mode import autocast
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from ..analysis.allocation import SPREAD_REACH_M
from ..analysis.archive import last_day, supervised
from ..config.dataset_config import get_dataset_config
from ..config.path_config import MODEL_DIR, MODEL_SAVE_DIR, PLOTS_DIR, RUNS_DIR
from ..dataset.data_loader import init_data_loader
from ..model.model import FireFusionModel
from .calibration import fit_calibration
from .metrics import MetricsManager
from .plots import plot_class_accuracy, plot_loss_curves, plot_rates_per_epoch
from .utils import (
    checkpoint_name, estimate_model_size_mb, export_to_b2, get_device_config,
    load_model, save_model, set_global_seed, WarmupCosineAnnealingLR,
)


class WRMTrainer:
    def __init__(self,
        model_params: Dict,
        training_params: Dict,
        device: torch.device,
        num_workers: int = 0,
        dataset_name: str = "wa2000",
        experiment: str = "smoke",
        seed: int = 42,
        mode: Literal['train', 'test'] = 'train',
        init_from: str | None = None,
        freeze: Literal['none', 'main', 'heads'] = 'none',
        alpha_ign: float = 1.0,
        alpha_cause: float | str = 1.0,
        export_b2: bool = False,
        debug: bool = False
    ):
        """ Build the model, data loaders, and loss configuration, then run
            train() or test() per mode. """
        # seeded before anything draws: weight init, dropout, shuffling, and crop
        # origins all read from RNGs that are seeded here or derived from them
        self.seed = seed
        set_global_seed(seed)

        self.device = device
        self.use_amp = bool(device.type == "cuda")
        self.debug = debug

        self.dataset_name = dataset_name
        self.experiment = experiment
        self.export_b2 = export_b2
        self.stage_base = checkpoint_name(experiment)
        self.freeze = freeze
        self.alpha_ign = alpha_ign

        # -- weight on the allocation term. The per-cell BCE spends most of its
        #    gradient on the overall ignition level, which the inputs cannot
        #    resolve; the allocation term asks only which cells rank above which,
        #    given the day. Zero leaves a run on the per-cell objective alone.
        self.alpha_alloc = float(training_params.get("alpha_alloc", 0.0))

        # -- the training loss scores every positive cell but only a fraction r of
        # -- the negatives, which moves the population optimum by exactly log(r) in
        # -- logit space and nothing else; evaluation adds that offset back
        self.neg_keep_rate = float(training_params.get("neg_keep_rate", 1.0))
        self.logit_offset = math.log(self.neg_keep_rate)

        # the look-back length is a hyperparameter of the experiment, not a
        # property of the loader; both splits read the same one
        window_size = training_params.get("window_size", 10)
        window_stride = training_params.get("window_stride", 2)
        self.fold = training_params.get("fold", "full")
        # -- an event with active fire within this reach scores as spread
        res = get_dataset_config(dataset_name).resolution
        self.spread_reach_cells = max(1, int(round(SPREAD_REACH_M / res)))

        # the crop halo and alignment follow from the model geometry; the loader
        # is built from the same numbers the model is
        encoder_depth = model_params.get("encoder_depth", 1)
        attn_window = model_params["win_spatial_mixing"]["window_size"]

        self.train_loader = init_data_loader(
            "train", dataset_name, num_workers, training_params["batch_size"],
            window_size=window_size, window_stride=window_stride,
            crop_size=training_params.get("crop_size"), seed=seed,
            encoder_depth=encoder_depth, attn_window=attn_window, fold=self.fold,
        )
        self.eval_loader = init_data_loader(
            "eval", dataset_name, num_workers, training_params["batch_size"],
            window_size=window_size, window_stride=window_stride,
            seed=seed,
            encoder_depth=encoder_depth, attn_window=attn_window, fold=self.fold,
        )

        # channel counts, channel grouping, output size, cause classes, and class
        # balance come from the built dataset's manifest, not from hardcoded params
        train_set = self.train_loader.dataset
        model_params = dict(model_params)
        model_params["n_cause_classes"] = train_set.n_cause_classes
        model_params["dyn_groups"] = train_set.dyn_groups
        ign_pos_weight = train_set.ign_pos_weight
        print(f"[WRMTrainer] experiment={experiment} seed={seed}")
        print(f"[WRMTrainer] dataset={dataset_name} "
              f"channels={train_set.dyn_channels} dyn + {train_set.static_channels} static "
              f"grid={list(train_set.out_size)} "
              f"cause_classes={model_params['n_cause_classes']} "
              f"prevalence={1.0 / max(ign_pos_weight, 1e-9):.2e} (PR-AUC baseline)")

        self.model = FireFusionModel(
            train_set.dyn_channels, train_set.static_channels, mp=model_params
        ).to(self.device)

        # Warm-start from a checkpointed model, then freeze the requested group;
        # the other group specializes against a fixed representation.
        if init_from is not None:
            load_model(self.model, init_from, map_location=self.device)
            print(f"[WRMTrainer] initialized weights from {init_from}")

        if freeze != "none":
            self.model.set_frozen(
                freeze_main=(freeze == "main"),
                freeze_heads=(freeze == "heads"),
            )
        n_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.model.parameters())
        print(f"[WRMTrainer] freeze={freeze} fold={self.fold} "
              f"alpha_ign={alpha_ign} alpha_alloc={self.alpha_alloc} alpha_cause={alpha_cause} "
              f"neg_keep_rate={self.neg_keep_rate} "
              f"trainable_params={n_train}/{n_total}")

        # -- eager LayerNorm/elementwise kernels dominate step time at these
        #    activation sizes; inductor fuses them. Placed after freezing:
        #    requires_grad flips force recompiles. dynamic=False skips torch
        #    2.2's guard solver, which can spin for an hour on static shapes.
        if training_params.get("compile", False) and device.type == "cuda":
            # -- torch 2.2's recursive pattern matcher (percolate_tags) hangs on
            #    the joint fwd+bwd graph; its extra fusions are marginal here
            import torch._inductor.config as inductor_config
            inductor_config.pattern_matcher = False
            self.model = torch.compile(self.model, dynamic=False)
            print("[WRMTrainer] torch.compile enabled")

        ep = training_params["epochs"]
        self.ep_warmup, self.ep_max, self.ep_early_stop = ep[0], ep[1], ep[2]
        self.min_lr = training_params["min_lr"]
        self.base_lr = training_params["base_lr"]
        self.weight_decay = training_params["weight_decay"]
        self.grad_clip = training_params["grad_clip"]
        self.accum_steps = int(training_params.get("accum_steps", 1))

        self.bcewl_loss = nn.BCEWithLogitsLoss(reduction="none")

        # -- Inverse-frequency class weight: cause classes span two orders of
        #    magnitude; an unweighted mean cross-entropy is minimized by never
        #    predicting the rare ones. `cause_weight_beta` tempers it toward the
        #    empirical prior when the rarest class is too thin to carry full weight.
        beta = training_params.get("cause_weight_beta", 1.0)
        counts = torch.as_tensor(train_set.cause_counts, dtype=torch.float32, device=device)
        w = (counts.sum() / counts.clamp(min=1)) ** beta
        self.cause_weight = w / w.mean()
        print(f"[WRMTrainer] cause_counts={train_set.cause_counts} beta={beta} "
              f"cause_weight={[round(v, 3) for v in self.cause_weight.tolist()]}")

        # -- the two terms differ by an order of magnitude at initialization;
        # -- 'auto' puts the cause term on the ignition term's scale before the
        # -- first step
        self.alpha_cause = (
            self._measure_alpha_cause() if alpha_cause == "auto" else float(alpha_cause)
        )

        # best epoch and early stopping key off the ignition head's masked
        # ignorance, the same score the run is evaluated on
        self.mm = MetricsManager(
            num_classes=(2, train_set.n_cause_classes),
            select_by=training_params.get("select_by", "val_ign"),
        )

        if mode == "train": self.train()
        else: self.test()

    def _prepare_targets(self, golds: Dict, masks: Dict):
        """ Final-day ignition and cause targets, the ignition mask that loss and
            metrics both read, and the cause mask restricted to known-cause cells. """
        ign_golds = last_day(golds["burn_next"])
        cause_golds = last_day(golds["burn_next_cause"])
        ign_mask = supervised({k: last_day(v) for k, v in masks.items()})

        # cause is only defined where a predicted ignition carries a known cause
        cause_mask = (ign_golds == 1) & (cause_golds != -1) & ign_mask

        return ign_golds, cause_golds, ign_mask, cause_mask

    def _near_active(self, masks: Dict) -> torch.Tensor:
        """ (B, H, W) bool: cells within the spread reach of a cell burning today,
            by max-pool dilation. """
        active = last_day(masks["active"]).float().unsqueeze(1)
        r = self.spread_reach_cells
        near = F.max_pool2d(active, kernel_size=2 * r + 1, stride=1, padding=r)
        return near.squeeze(1) > 0

    def _thinned_mask(self, ign_golds: torch.Tensor, ign_mask: torch.Tensor) -> torch.Tensor:
        """ ign_mask with every positive cell kept and negatives kept at neg_keep_rate. """
        draw = torch.rand(ign_mask.shape, device=ign_mask.device) < self.neg_keep_rate
        return ign_mask & ((ign_golds == 1) | draw)

    def _measure_alpha_cause(self, n_batches: int = 4, max_batches: int = 16) -> float:
        """ Ratio of mean ignition loss to mean cause loss over a handful of training
            batches, drawing until at least two batches carry cause-labelled cells or
            max_batches is reached. Returns 1.0 when no batch carries a cause cell.
        """
        self.model.eval()
        ign_sum, ign_n, cause_sum, cause_n = 0.0, 0, 0.0, 0

        with torch.no_grad():
            # -- cause cells are sparse enough that a small crop can miss them for
            # -- several consecutive batches; keep drawing until two carried some
            for i, ((x_dyn, x_static), golds, masks) in enumerate(self.train_loader):
                if i >= n_batches and cause_n >= 2 or i >= max_batches:
                    break
                x_dyn, x_static = x_dyn.to(self.device), x_static.to(self.device)
                golds = { k: v.to(self.device) for k, v in golds.items() }
                masks = { k: v.to(self.device) for k, v in masks.items() }

                ign_golds, cause_golds, ign_mask, cause_mask = self._prepare_targets(golds, masks)
                loss_mask = self._thinned_mask(ign_golds, ign_mask)

                with autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
                    ign_logits, cause_logits = self.model(x_dyn, x_static)
                    _, ign_loss, cause_loss, alloc_loss = self._compute_loss(
                        ign_logits, ign_golds, cause_logits, cause_golds,
                        loss_mask, cause_mask, alloc_mask=ign_mask
                    )
                    if self.alpha_alloc > 0.0:
                        ign_loss = alloc_loss

                if loss_mask.any():
                    ign_sum += ign_loss.item()
                    ign_n += 1
                if cause_mask.any():
                    cause_sum += cause_loss.item()
                    cause_n += 1

        if cause_n == 0:
            print("[WRMTrainer] alpha_cause=auto saw no cause cells, using 1.0")
            return 1.0

        ratio = (ign_sum / max(ign_n, 1)) / (cause_sum / cause_n)
        print(f"[WRMTrainer] alpha_cause=auto over {ign_n} batch(es): "
              f"L_ign/L_cause={ratio:.5f}")
        return ratio

    @staticmethod
    def allocation_loss(ign_logits: torch.Tensor, ign_golds: torch.Tensor,
                        ign_mask: torch.Tensor):
        """ Negative log-likelihood of the cells that ignited under each day's
            field renormalized over that day's supervised cells.

            The softmax is taken per sample; any constant added to a day's
            logits cancels and the term is blind to the overall ignition rate.
            Days with no ignition carry no gradient.

            Returns the mean bits-equivalent NLL per event and the event count.
        """
        z = ign_logits.squeeze(1)
        total = z.new_zeros(())
        n_events = 0
        for b in range(z.shape[0]):
            m = ign_mask[b]
            pos = (ign_golds[b] == 1) & m
            k = int(pos.sum())
            if k == 0:
                continue
            # -- float32 for the normalizer: a bf16 logsumexp over 10^5 cells
            #    loses the tail the score is most sensitive to
            log_q = torch.log_softmax(z[b][m].float(), dim=0)
            total = total + -log_q[pos[m]].sum()
            n_events += k
        return total / max(n_events, 1), n_events

    def _compute_loss(self,
        ign_logits: torch.Tensor, ign_golds: torch.Tensor,  # (B, 1, H, W), (B, H, W)
        cause_logits: torch.Tensor,                         # (B, num_classes, H, W)
        cause_golds: torch.Tensor,                          # (B, H, W)
        ign_mask: torch.Tensor,                             # (B, H, W)
        cause_mask: torch.Tensor,                           # (B, H, W)
        alpha_ign: float = 1.0,
        alpha_cause: float = 1.0,
        alloc_mask: torch.Tensor | None = None,
    ):
        """ Compute BCELogitsLoss on a next-window ignition, the allocation NLL
            over the same field, and cross entropy on ignition TYPE given an
            ignition. `alloc_mask` is the unthinned supervised population; the
            normalizer is a distribution over the whole day.
        """
        ign_logits_flat = ign_logits.squeeze(1)
        ign_targets = ign_golds.float()
        ign_loss = self.bcewl_loss(
            ign_logits_flat,
            ign_targets
        )

        masked_ign_loss = ign_loss * ign_mask
        ign_loss = (
            (masked_ign_loss.sum()) /
            (ign_mask.sum() + 1e-6)
        )

        if cause_mask.any():
            cause_logits_flat = cause_logits.permute(0, 2, 3, 1)[cause_mask]
            cause_targets_flat = cause_golds[cause_mask].long()

            cause_loss = nn.functional.cross_entropy(
                cause_logits_flat,
                cause_targets_flat,
                weight=self.cause_weight,
                reduction="mean"
            )
        else:
            cause_loss = torch.tensor(0.0, device=ign_logits.device)

        if self.alpha_alloc > 0.0:
            mask = ign_mask if alloc_mask is None else alloc_mask
            alloc_loss, _ = self.allocation_loss(ign_logits, ign_golds, mask)
        else:
            alloc_loss = torch.tensor(0.0, device=ign_logits.device)

        total_loss = ((ign_loss * alpha_ign) + (alloc_loss * self.alpha_alloc)
                      + (cause_loss * alpha_cause))
        return total_loss, ign_loss, cause_loss, alloc_loss

    def train_epoch(self, epoch: int):
        """ One training epoch: forward and backward over the train loader with
            gradient accumulation, accumulating the ignition, cause and allocation
            loss totals and per-batch metrics, stepping optimizer and scheduler.
        """
        self.model.train()
        ep_total_loss: float = 0.0
        ep_ign_loss: float = 0.0
        ep_cause_loss: float = 0.0
        ep_alloc_loss: float = 0.0
        n_samples: int = 0

        n_batches = len(self.train_loader)
        self.optimizer.zero_grad(set_to_none=True)
        for b_idx, ((x_dyn, x_static), golds, masks) in enumerate(
            tqdm(self.train_loader, desc="Training...", leave=False)
        ):
            x_dyn = x_dyn.to(self.device)
            x_static = x_static.to(self.device)
            golds = { k: v.to(self.device) for k, v in golds.items() }
            masks = { k: v.to(self.device) for k, v in masks.items() }

            if epoch == 1 and self.debug:
                for name, t in (("x_dyn", x_dyn), ("x_static", x_static)):
                    print(f"[DataCheck] {name} {tuple(t.shape)} "
                          f"min/max {t.min().item():.4f}/{t.max().item():.4f} "
                          f"mean/std {t.mean().item():.4f}/{t.std().item():.4f} "
                          f"nan={torch.isnan(t).sum().item()} inf={torch.isinf(t).sum().item()}")

            ign_golds, cause_golds, ign_mask, cause_mask = self._prepare_targets(golds, masks)
            loss_mask = self._thinned_mask(ign_golds, ign_mask)

            with autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
                ign_logits, cause_logits = self.model(x_dyn, x_static)   # (B, 1, H, W), (B, num_classes, H, W)

                tot_loss, ign_loss, cause_loss, alloc_loss = self._compute_loss(
                    ign_logits, ign_golds, cause_logits, cause_golds,
                    loss_mask, cause_mask,
                    alpha_ign=self.alpha_ign, alpha_cause=self.alpha_cause,
                    alloc_mask=ign_mask
                )

            # Log total loss
            ep_total_loss += tot_loss.item()
            ep_ign_loss += ign_loss.item()
            ep_cause_loss += cause_loss.item()
            ep_alloc_loss += alloc_loss.item()
            n_samples += golds["burn_next"].size(0)

            # -- metrics read the full supervised population, not the thinned one;
            # -- training scores are comparable with the eval split
            self.mm.add('train',
                logits=[ign_logits.detach().cpu(), cause_logits.detach().cpu()],
                golds =[ign_golds.detach().cpu(), cause_golds.detach().cpu()],
                masks =[ign_mask.detach().cpu(), cause_mask.detach().cpu()]
            )

            # -- gradients accumulate over accum_steps micro-batches; the loss is
            #    scaled so their sum matches one batch of accum_steps * batch_size.
            #    Clip, step, and schedule fire once per accumulated batch.
            (tot_loss / self.accum_steps).backward()
            if (b_idx + 1) % self.accum_steps == 0 or (b_idx + 1) == n_batches:
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

        self.mm.add_epoch_totals("train",
            losses=np.array([ep_total_loss, ep_ign_loss, ep_cause_loss, ep_alloc_loss])
        )

    def eval_epoch(self):
        """ One evaluation epoch: forward over the eval loader with the subsampling
            logit offset restored, accumulating loss totals and metrics (with the
            near-active mask for allocation strata), then recording the epoch totals.
        """
        self.model.eval()
        ep_total_loss: float = 0.0
        ep_ign_loss: float = 0.0
        ep_cause_loss: float = 0.0
        ep_alloc_loss: float = 0.0
        n_samples: int = 0
        # -- no_grad rather than inference_mode: dynamo-compiled graphs reject
        #    inference tensors
        with torch.no_grad():
            for (x_dyn, x_static), golds, masks in tqdm(self.eval_loader, desc="Evaluating...", leave=False):
                x_dyn = x_dyn.to(self.device)
                x_static = x_static.to(self.device)
                golds = { k: v.to(self.device) for k, v in golds.items() }
                masks = { k: v.to(self.device) for k, v in masks.items() }

                ign_golds, cause_golds, ign_mask, cause_mask = self._prepare_targets(golds, masks)

                with autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
                    ign_logits, cause_logits = self.model(x_dyn, x_static)

                    # -- undo the subsampling shift; the reported loss and the
                    # -- selection score describe the real class balance
                    ign_logits = ign_logits + self.logit_offset

                    tot_loss, ign_loss, cause_loss, alloc_loss = self._compute_loss(
                        ign_logits, ign_golds, cause_logits, cause_golds,
                        ign_mask, cause_mask,
                        alpha_ign=self.alpha_ign, alpha_cause=self.alpha_cause
                    )

                # Log total loss for epoch
                ep_total_loss += tot_loss.item()
                ep_ign_loss += ign_loss.item()
                ep_cause_loss += cause_loss.item()
                ep_alloc_loss += alloc_loss.item()
                n_samples += golds["burn_next"].size(0)

                self.mm.add('eval',
                    logits=[ign_logits.detach().cpu(), cause_logits.detach().cpu()],
                    golds =[ign_golds.detach().cpu(), cause_golds.detach().cpu()],
                    masks =[ign_mask.detach().cpu(), cause_mask.detach().cpu()],
                    near  =self._near_active(masks).cpu(),
                )

        self.mm.add_epoch_totals("eval",
            np.array([ep_total_loss, ep_ign_loss, ep_cause_loss, ep_alloc_loss])
        )

    def train(self):
        """ The full training loop: build optimizer and scheduler, run train_epoch
            and eval_epoch per epoch with TensorBoard logging and early stopping,
            save best and final weights, fit the calibrator against the best
            weights, plot the curves, and optionally export the run to B2.
        """
        # only trainable params reach the optimizer; a frozen group is not
        # nudged by weight decay or momentum
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.base_lr,
            weight_decay=self.weight_decay
        )
        # -- the scheduler ticks once per optimizer step, not per micro-batch;
        #    its horizon shrinks by the accumulation factor
        opt_steps_per_epoch = max(1, math.ceil(len(self.train_loader) / self.accum_steps))
        self.scheduler = WarmupCosineAnnealingLR(
            self.optimizer,
            warmup_steps=self.ep_warmup * opt_steps_per_epoch,
            total_steps=self.ep_max * opt_steps_per_epoch,
            min_lr=self.min_lr
        )

        print(f"Starting training with parameters:\n"
            f"- model size: {estimate_model_size_mb(self.model):.2f}mb\n",
            f"- epochs: {self.ep_warmup} (warmup) {self.ep_max} (total) {self.ep_early_stop} (early stop)\n",
            f"- min lr: {self.min_lr}, base lr: {self.base_lr}, grad clip: {self.grad_clip}, "
            f"weight decay: {self.weight_decay}, accum steps: {self.accum_steps}\n",
        )

        # best weights take the fixed name, referenced by --init-from; the
        # final weights fall to the next free index
        stage_base = self.stage_base

        # One TensorBoard run per invocation; the timestamp keeps reruns from
        # writing into the same event stream.
        run_name = f"{self.experiment}_{strftime('%m%d-%H%M%S')}"
        writer = SummaryWriter(log_dir=str(RUNS_DIR / run_name))
        print(f"[WRMTrainer] tensorboard logdir >> {RUNS_DIR / run_name}")

        time0 = perf_counter()
        epochs_ran = 0
        for epoch in range (1, self.ep_max + 1):
            self.train_epoch(epoch)
            self.eval_epoch()

            score, new_best, trn_last, val_last = self.mm.epoch_forward()
            epochs_ran += 1

            # the console log and its scalar breakdown are collected in the
            # metrics manager; mirror both to TensorBoard for live monitoring
            for tag, val in self.mm.last_scalars.items():
                writer.add_scalar(tag, val, epoch)
            writer.add_scalar("lr", self.optimizer.param_groups[0]["lr"], epoch)
            writer.add_text("epoch_report", self.mm.last_report, epoch)
            writer.flush()

            # kept under a fixed name so the best weights survive the epochs that follow
            if new_best:
                best_path = save_model(self.model, name_base=stage_base, overwrite=True)
                print(f"Saved best weights >> {best_path}\n")

            if self.mm.no_improve > self.ep_early_stop:
                print(f"Stopped training for early stop")
                break

        writer.close()

        elapsed_min = (perf_counter() - time0) // 60
        elapsed_sec = (perf_counter() - time0) % 60
        print(f"Finished training in {elapsed_min:.0f} min {elapsed_sec:.0f} sec")
        print(f"Best score @epoch {self.mm.best['epoch']} >> score: {self.mm.best['score']:.5f}")

        final_path = save_model(self.model, name_base=stage_base)
        print(f"Saved final weights >> {final_path}")

        # The calibrator is a sidecar of `<stage_base>.th`, which holds the best
        # epoch's weights; it is fit against those weights, not the ones left in
        # memory after later epochs.
        best_path = MODEL_SAVE_DIR / f"{stage_base}.th"
        if self.mm.best["epoch"] and best_path.exists():
            load_model(self.model, best_path, map_location=self.device)
            print(f"Restored best weights (epoch {self.mm.best['epoch']}) for calibration")

        # a calibrator is only meaningful once the ignition head has trained
        if self.freeze != "heads":
            fit_calibration(self.model, self.eval_loader, self.device,
                            self.neg_keep_rate, self.cause_weight,
                            stage_base, self._prepare_targets)

        # Do some plotting and fun visualizations!
        trn_losses, val_losses = self.mm.get_history()
        trn_ignit_acc, trn_cause_acc = self.mm.trn_accuracies[0], self.mm.trn_accuracies[1]
        val_ignit_acc, val_cause_acc = self.mm.val_accuracies[0], self.mm.val_accuracies[1]
        val_ignit_cm = self.mm.val_cm[0]
        last_ign_cm, ign_rates, ign_cm_record = val_ignit_cm.get_history()

        epochs_axis = list(range(1, epochs_ran + 1))

        # Train vs. Eval
        # every plot carries the run's stage_base; artifacts from different
        # experiments never overwrite one another
        plot_class_accuracy(
            epochs_axis,
            val_ignit_acc, val_cause_acc,
            trn_ignit_acc, trn_cause_acc,
            save=True, save_path=str(PLOTS_DIR / f"class_accuracy_{stage_base}.png"),
        )
        plot_loss_curves(
            epochs_axis,
            trn_losses, val_losses,
            term_names=("total", "ignition", "cause", "allocation"),
            save=True, save_path=str(PLOTS_DIR / f"losses_{stage_base}.png"),
        )

        plot_rates_per_epoch(
            epochs_axis, ign_rates, save=True,
            save_path=str(PLOTS_DIR / f"rates_{stage_base}.png"),
        )

        # push the complete run to B2 last, once every artifact exists: weights
        # (best and final), the calibrator, all four plots, and the TensorBoard
        # run dir
        if self.export_b2:
            export_to_b2(
                MODEL_SAVE_DIR / f"{stage_base}.th",
                final_path,
                MODEL_SAVE_DIR / f"{stage_base}.calib.json",
                PLOTS_DIR / f"reliability_{stage_base}.png",
                PLOTS_DIR / f"class_accuracy_{stage_base}.png",
                PLOTS_DIR / f"losses_{stage_base}.png",
                PLOTS_DIR / f"rates_{stage_base}.png",
                RUNS_DIR / run_name,
                prefix=f"runs/{self.dataset_name}/{run_name}",
            )

    def test(self):
        return


def main():
    """ Parse CLI arguments, load the named experiment's params, apply CLI
        overrides, and run WRMTrainer. """
    parser = argparse.ArgumentParser(description="Train FireFusionNet on a built dataset")
    parser.add_argument("--experiment", default="smoke", help="params.json experiment name")
    parser.add_argument("--dataset", default=None,
                        help="override the dataset the experiment names")
    parser.add_argument("--seed", type=int, default=None,
                        help="override the experiment's seed")
    parser.add_argument("--init-from", default=None,
                        help="checkpoint to warm-start from (bare name resolves under model/saved)")
    parser.add_argument("--freeze", choices=["none", "main", "heads"], default=None,
                        help="freeze the backbone ('main') or the decoder ('heads')")
    parser.add_argument("--alpha-ign", type=float, default=None, help="ignition loss weight")
    parser.add_argument("--alpha-cause", type=float, default=None, help="cause loss weight")
    parser.add_argument("--chunk-size", type=int, default=None,
                        help="override channel_mixing.chunk_size (VRAM/throughput probe)")
    parser.add_argument("--export-b2", action="store_true",
                        help="upload final weights + calibrator to Backblaze B2 (B2_* env) when training ends")
    args = parser.parse_args()

    with open(f'{MODEL_DIR}/params.json') as file:
        data = json.load(file)
        if args.experiment not in data:
            raise SystemExit(
                f"Unknown experiment '{args.experiment}'. Options: {sorted(data)}"
            )
        params = data[args.experiment]

    model_params        = params["model"]
    training_params     = params["training"]

    # -- the attention chunk trades VRAM for kernel-launch count and is the one
    #    knob probed per GPU, not per experiment
    if args.chunk_size is not None:
        model_params["channel_mixing"]["chunk_size"] = args.chunk_size
        print(f"[train] channel_mixing.chunk_size overridden to {args.chunk_size}")

    # -- a window read decompresses a whole time-chunk of the split zarr; the
    #    loader is IO-bound before the GPU is. Each worker holds prefetched windows
    #    in pinned memory; the ceiling belongs to the experiment.
    device, num_workers = get_device_config(maximum=training_params.get("max_workers", 8))

    # an experiment names the dataset and seed it was defined against; the
    # experiment name alone reproduces the run. The flags are an escape hatch
    dataset = args.dataset if args.dataset is not None else params["dataset"]
    seed = args.seed if args.seed is not None else training_params["seed"]

    # an explicit CLI flag wins, else the experiment's own value, else the
    # hardcoded fallback
    freeze      = args.freeze if args.freeze is not None else training_params.get("freeze", "none")
    alpha_ign   = args.alpha_ign if args.alpha_ign is not None else training_params.get("alpha_ign", 1.0)
    alpha_cause = args.alpha_cause if args.alpha_cause is not None else training_params.get("alpha_cause", 1.0)

    WRMTrainer(
        model_params, training_params,
        device, num_workers,
        dataset_name = dataset,
        experiment = args.experiment,
        seed = seed,
        mode = "train",
        init_from = args.init_from,
        freeze = freeze,
        alpha_ign = alpha_ign,
        alpha_cause = alpha_cause,
        export_b2 = args.export_b2,
        debug = False
    )


if __name__ == "__main__":
    main()
