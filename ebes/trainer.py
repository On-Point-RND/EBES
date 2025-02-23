import logging
import os
from collections.abc import Iterable, Sized
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import nn
from torcheval.metrics import Mean, Metric
from tqdm.autonotebook import tqdm

from .utils.general import LoadTime
from .data.loading import Batch

logger = logging.getLogger(__name__)


class Trainer:
    """A base class for all trainers."""

    def __init__(
        self,
        *,
        model: nn.Module | None = None,
        loss: nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        train_loader: Iterable[Batch] | None = None,
        val_loader: Iterable[Batch] | None = None,
        metrics: Iterable[Metric] | None = None,
        run_name: str | None = None,
        total_iters: int | None = None,
        total_epochs: int | None = None,
        patience: int = -1,
        iters_per_epoch: int | None = 10_000,
        ckpt_dir: str | os.PathLike | None = None,
        ckpt_replace: bool = True,
        ckpt_track_metric: str = "epoch",
        ckpt_resume: str | os.PathLike | None = None,
        device: str = "cpu",
        metrics_on_train: bool = False,
    ):
        """Initialize trainer.

        Args:
            model: model to train or validate.
            loss: loss function
            optimizer: torch optimizer for training.
            lr_scheduler: torch learning rate scheduler.
            train_loader: train dataloader.
            val_loader: val dataloader.
            metrics: metrics to compute every epoch
            run_name: for runs differentiation.
            total_iters: total number of iterations to train a model.
            total_epochs: total number of epoch to train a model. Exactly one of
                `total_iters` and `total_epochs` shoud be passed.
            patience: how many epochs trainer will go without improving
                validation ckpt_track_metric. -1 means never stop
                Assumes track_metric is MAXIMIZED
            iters_per_epoch: validation and checkpointing are performed every
                `iters_per_epoch` iterations.
            ckpt_dir: path to the directory, where checkpoints are saved.
            ckpt_replace: if `replace` is `True`, only the last and the best checkpoint
                are kept in `ckpt_dir`.
            ckpt_track_metric: if `ckpt_replace` is `True`, the best checkpoint is
                determined based on `track_metric`. All metrcs except loss are assumed
                to be better if the value is higher.
            ckpt_resume: path to the checkpoint to resume training from.
            device: device to train and validate on.
            metrics_on_train: wether to compute metrics on train set.
        """
        assert (
            total_iters is None or total_epochs is None
        ), "Only one of `total_iters` and `total_epochs` shoud be passed."

        self._run_name = (
            run_name if run_name is not None else datetime.now().strftime("%F_%T")
        )

        self._metrics = {}
        if metrics is not None:
            self._metrics.update({m.__class__.__name__: m for m in metrics})

        if loss is not None:
            self._metrics.update({"loss": Mean()})

        self._total_iters = total_iters
        self._total_epochs = total_epochs
        self._patience = patience
        self._iters_per_epoch = iters_per_epoch
        self._ckpt_dir = ckpt_dir
        self._ckpt_replace = ckpt_replace
        self._ckpt_track_metric = ckpt_track_metric
        self._ckpt_resume = ckpt_resume
        self._device = device
        self._metrics_on_train = metrics_on_train

        self._model = None
        if model is not None:
            self._model = model.to(device)

        self._loss = None
        if loss is not None:
            self._loss = loss.to(device)

        self._opt = optimizer
        self._sched = lr_scheduler
        self._train_loader = train_loader
        self._val_loader = val_loader

        self._metric_values: dict[str, Any] | None = None
        self._last_iter = 0
        self._last_epoch = 0

    @property
    def model(self) -> nn.Module | None:
        return self._model

    @property
    def train_loader(self) -> Iterable[Batch] | None:
        return self._train_loader

    @property
    def val_loader(self) -> Iterable[Batch] | None:
        return self._val_loader

    @property
    def optimizer(self) -> torch.optim.Optimizer | None:
        return self._opt

    @property
    def lr_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler | None:
        return self._sched

    @property
    def run_name(self):
        return self._run_name

    @property
    def device(self) -> str:
        return self._device

    def _make_key_extractor(self, key):
        def key_extractor(p: Path) -> float:
            metrics = {}
            for it in p.stem.split("_-_"):
                kv = it.split("__")
                assert len(kv) == 2, f"Failed to parse filename: {p.name}"
                k = kv[0]
                v = -float(kv[1]) if ("loss" in k) or ("mse" in k) else float(kv[1])
                metrics[k] = v
            return metrics[key]

        return key_extractor

    def save_ckpt(self, ckpt_path: str | os.PathLike | None = None) -> None:
        """Save model, optimizer and scheduler states.

        Args:
            ckpt_path: path to checkpoints. If `ckpt_path` is a directory, the
                checkpoint will be saved there with epoch, loss an metrics in the
                filename. All scalar metrics returned from `compute_metrics` are used to
                construct a filename. If full path is specified, the checkpoint will be
                saved exectly there. If `None` `ckpt_dir` from construct is used with
                subfolder named `run_name` from Trainer's constructor.
        """

        if ckpt_path is None and self._ckpt_dir is None:
            logger.warning(
                "`ckpt_path` was not passned to `save_ckpt` and `ckpt_dir` "
                "was not set in Trainer. No checkpoint will be saved."
            )
            return

        if ckpt_path is None:
            assert self._ckpt_dir is not None
            ckpt_path = self._ckpt_dir

        ckpt_path = Path(ckpt_path)
        ckpt_path.mkdir(parents=True, exist_ok=True)

        ckpt: dict[str, Any] = {
            "last_iter": self._last_iter,
            "last_epoch": self._last_epoch,
        }
        if self._model:
            ckpt["model"] = self._model.state_dict()
        if self._opt:
            ckpt["opt"] = self._opt.state_dict()
        if self._sched:
            ckpt["sched"] = self._sched.state_dict()

        if not ckpt_path.is_dir():
            torch.save(ckpt, ckpt_path)

        assert self._metric_values

        metrics = {k: v for k, v in self._metric_values.items() if np.isscalar(v)}

        fname = f"epoch__{self._last_epoch:04d}"
        metrics_str = "_-_".join(
            f"{k}__{v:.4g}" for k, v in metrics.items() if k == self._ckpt_track_metric
        )

        if len(metrics_str) > 0:
            fname = "_-_".join((fname, metrics_str))
        fname += ".ckpt"

        torch.save(ckpt, ckpt_path / Path(fname))

        if not self._ckpt_replace:
            return

        all_ckpt = list(ckpt_path.glob("*.ckpt"))
        best_ckpt = max(all_ckpt, key=self._make_key_extractor(self._ckpt_track_metric))
        for p in all_ckpt:
            if p != best_ckpt:
                p.unlink()

    def load_ckpt(self, ckpt_fname: str | os.PathLike, strict: bool = True) -> None:
        """Load model, optimizer and scheduler states.

        Args:
            ckpt_fname: path to checkpoint.
        """

        assert self._model is not None
        ckpt = torch.load(ckpt_fname, map_location=self._device)

        if "model" in ckpt:
            msg = self._model.load_state_dict(ckpt["model"], strict=strict)
            print(msg)
        if "opt" in ckpt:
            if self._opt is None:
                logger.warning(
                    "optimizer was not passes, discarding optimizer state "
                    "in the checkpoint"
                )
            else:
                self._opt.load_state_dict(ckpt["opt"])
        if "sched" in ckpt:
            if self._sched is None:
                logger.warning(
                    "scheduler was not passes, discarding scheduler state "
                    "in the checkpoint"
                )
            else:
                self._sched.load_state_dict(ckpt["sched"])
        self._last_iter = ckpt["last_iter"]
        self._last_epoch = ckpt["last_epoch"]

    def train(self, iters: int) -> dict[str, Any]:
        assert self._opt is not None, "Set an optimizer first"
        assert self._train_loader is not None, "Set a train loader first"
        assert self._model is not None
        assert self._loss is not None

        logger.info("Epoch %04d: train started", self._last_epoch + 1)
        self._model.train()

        for metric in self._metrics.values():
            metric.reset()

        loss_ema = 0.0
        losses: list[float] = []

        total_iters = iters
        if (
            hasattr(self._train_loader, "dataset")
            and isinstance(self._train_loader.dataset, Sized)  # type: ignore
            and (total_iters > len(self._train_loader))  # type: ignore
        ):
            total_iters = len(self._train_loader)  # type: ignore
        pbar = tqdm(zip(self._train_loader, range(total_iters)), total=total_iters)

        pbar.set_description_str(f"Epoch {self._last_epoch + 1: 3}")
        for batch, i in LoadTime(pbar, disable=pbar.disable):
            batch.to(self._device)
            inp = batch
            gt = batch.pop_target()

            pred = self._model(inp)
            loss = self._loss(pred, gt)
            if torch.isnan(loss).any():
                raise ValueError("None detected in loss. Terminating training.")

            loss.backward()

            self._metrics["loss"].update(loss.detach().cpu())
            loss_np = loss.item()
            losses.append(loss_np)
            loss_ema = loss_np if i == 0 else 0.9 * loss_ema + 0.1 * loss_np
            pbar.set_postfix_str(f"Loss: {loss_ema:.4g}")

            self._opt.step()

            if self._metrics_on_train:
                if gt is not None:
                    gt = gt.detach().cpu()
                for name, metric in self._metrics.items():
                    if name != "loss":
                        pred = pred.to("cpu") if hasattr(pred, "to") else pred
                        metric.update(pred, gt)

            self._opt.zero_grad()
            self._last_iter += 1

        logger.info(
            "Epoch %04d: avg train loss = %.4g", self._last_epoch + 1, np.mean(losses)
        )
        logger.info("Epoch %04d: train finished", self._last_epoch + 1)

        return self.compute_metrics("train")

    @torch.inference_mode()
    def validate(self, loader: Iterable[Batch] | None = None) -> dict[str, Any]:
        assert self._model is not None
        if loader is None:
            if self._val_loader is None:
                raise ValueError("Either set val loader or provide loader explicitly")
            loader = self._val_loader

        logger.info("Epoch %04d: validation started", self._last_epoch + 1)

        self._model.eval()
        for metric in self._metrics.values():
            metric.reset()

        for batch in tqdm(loader):
            batch.to(self._device)
            inp = batch
            gt = batch.pop_target()
            pred = self._model(inp)

            if self._loss is not None:
                loss = self._loss(pred, gt).cpu()
                self._metrics["loss"].update(loss.cpu())

            if gt is not None:
                gt = gt.to("cpu")

            for name, metric in self._metrics.items():
                if name != "loss":
                    pred = pred.to("cpu") if hasattr(pred, "to") else pred
                    metric.update(pred, gt)

        logger.info("Epoch %04d: validation finished", self._last_epoch + 1)

        return self.compute_metrics("val")

    def compute_metrics(self, phase: Literal["train", "val"]) -> dict[str, Any]:
        """Compute and log metrics.

        The metrics are computed based on the whole epoch data, so the granularity of
        metrics is epoch, so when the metrics are not None, the epoch is not None to.

        Args:
            phase: wether the metrics were collected during train or validatoin.
        """

        self._metric_values = {}
        for name, metric in self._metrics.items():
            try:
                val = metric.compute().squeeze()
            except (RuntimeError, ValueError):
                logger.warning(f"{name} metric had RuntimeError")
                val = torch.zeros(1)
            try:
                val = val.item()
            except RuntimeError:
                val = val.numpy()
            self._metric_values.update({name: val})

        logger.info(
            f"Epoch %04d: {phase} metrics: %s",
            self._last_epoch + 1,
            str(self._metric_values),
        )

        return self._metric_values

    def run(self) -> None:
        """Train and validate model."""

        assert self._opt, "Set an optimizer to run full cycle"
        assert self._train_loader is not None, "Set a train loader to run full cycle"
        assert self._val_loader is not None, "Set a val loader to run full cycle"
        assert self._model is not None

        logger.info("run %s started", self._run_name)

        if self._ckpt_resume is not None:
            logger.info("Resuming from checkpoint '%s'", str(self._ckpt_resume))
            self.load_ckpt(self._ckpt_resume)

        self._model.to(self._device)

        if self._iters_per_epoch is None:
            logger.warning(
                "`iters_per_epoch` was not passed to the constructor. "
                "Defaulting to the length of the dataloader."
            )
            if not (
                hasattr(self._train_loader, "dataset")
                and isinstance(self._train_loader.dataset, Sized)  # type: ignore
            ):
                raise ValueError(
                    "You must explicitly set `iters_per_epoch` to use unsized loader"
                )

            self._iters_per_epoch = len(self._train_loader)  # type: ignore

        if self._total_iters is None:
            assert self._total_epochs is not None, "Set `total_iters` or `total_epochs`"
            self._total_iters = self._total_epochs * self._iters_per_epoch

        # TODO more epochs than set when dataset too small
        best_metric = float("-inf")
        patience = self._patience

        while self._last_iter < self._total_iters:
            train_iters = min(
                self._total_iters - self._last_iter,
                self._iters_per_epoch,
            )

            self._metric_values = None
            self.train(train_iters)
            if self._sched:
                self._sched.step()

            self._metric_values = None
            self.validate()

            self._last_epoch += 1
            self.save_ckpt()

            assert (
                self._metric_values is not None
                and self._ckpt_track_metric in self._metric_values
            )
            target_metric = self._metric_values[self._ckpt_track_metric]
            if self._ckpt_track_metric == "loss":
                target_metric = -1 * target_metric

            if target_metric > best_metric:
                best_metric = target_metric
                patience = self._patience
            else:
                patience -= 1
            if patience == 0:
                logger.info(
                    f"Patience has run out. Early stopping at {self._last_epoch} epoch"
                )
                break

        logger.info("run '%s' finished successfully", self._run_name)

    def best_checkpoint(self) -> Path:
        """
        Return the path to the best checkpoint
        """
        assert self._ckpt_dir is not None
        ckpt_path = Path(self._ckpt_dir)

        all_ckpt = list(ckpt_path.glob("*.ckpt"))
        best_ckpt = max(all_ckpt, key=self._make_key_extractor(self._ckpt_track_metric))

        return best_ckpt

    def load_best_model(self) -> None:
        """
        Loads the best model to self._model according to the track metric.
        """

        best_ckpt = self.best_checkpoint()
        self.load_ckpt(best_ckpt)
