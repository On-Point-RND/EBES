from collections.abc import Mapping
from pathlib import Path

from ...data.utils import build_loaders
from ...model import build_model
from ...trainer import Trainer
from ..base_runner import Runner
from ..utils import get_loss, get_metrics, get_optimizer, suggest_conf


class EvalRunner(Runner):
    def pipeline(self, config: Mapping) -> dict[str, float]:
        loaders = build_loaders(**config["data"])
        test_loaders = build_loaders(**config["test_data"])

        net = build_model(config["model"])
        opt = get_optimizer(net.parameters(), **config["optimizer"])
        metrics = get_metrics(config["metrics"], "cpu")
        loss = get_loss(**config["main_loss"])
        trainer = Trainer(
            model=net,
            loss=loss,
            optimizer=opt,
            metrics=metrics,
            train_loader=loaders["train"],
            val_loader=loaders["train_val"],
            run_name=config["run_name"],
            ckpt_dir=Path(config["log_dir"]) / config["run_name"] / "ckpt",
            device=config["device"],
            **config["trainer"],
        )
        trainer.load_best_model()
        net.to(trainer._device)
        # train_metrics = trainer.validate(loaders["full_train"])
        train_val_metrics = trainer.validate(loaders["train_val"])
        hpo_metrics = trainer.validate(loaders["hpo_val"])
        test_metrics = trainer.validate(test_loaders["test"])

        train_metrics = {}  # "train_" + k: v for k, v in train_metrics.items()}
        train_val_metrics = {"train_val_" + k: v for k, v in train_val_metrics.items()}
        test_metrics = {"test_" + k: v for k, v in test_metrics.items()}

        return dict(**hpo_metrics, **train_metrics, **train_val_metrics, **test_metrics)

    def param_grid(self, trial, config):
        suggest_conf(config["optuna"]["suggestions"], config, trial)
        return trial, config
