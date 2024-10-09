from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn

from ...data.utils import build_loaders
from ...metrics import PrimeNetAccuracy
from ...model import build_model
from ...trainer import Trainer
from ..base_runner import Runner
from ..utils import get_loss, get_metrics, get_optimizer, suggest_conf


class PrimeNetRunner(Runner):
    def pipeline(self, config: Mapping) -> dict[str, float]:
        loaders = build_loaders(**config["data"])
        test_loaders = build_loaders(**config["test_data"])

        net = build_model(config["pretrain_model"])
        opt = get_optimizer(net.parameters(), **config["optimizer"])
        metrics = [PrimeNetAccuracy()]

        class PrimeLoss(nn.Module):
            def forward(self, preds, _):
                return preds["loss"]

        trainer = Trainer(
            model=net,
            loss=PrimeLoss(),
            optimizer=opt,
            metrics=metrics,
            train_loader=loaders["primenet_train"],
            val_loader=loaders["primenet_val"],
            run_name=config["run_name"] + "/pretrain",
            ckpt_dir=Path(config["log_dir"]) / config["run_name"] / "pretrain" / "ckpt",
            device=config["device"],
            **config["pretrainer"],
        )
        trainer.run()

        net = build_model(config["model"])

        if config["pretrainer"]["total_iters"]:
            net.load_state_dict(
                torch.load(trainer.best_checkpoint(), map_location="cpu")["model"],
                strict=False,
            )
            net.eval()

        class Frozen(nn.Module):
            def __init__(self, model, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.model = model
                for param in model.parameters():
                    param.requires_grad = False
                for param in model[-1].classifier.parameters():
                    param.requires_grad = True

            def train(self, mode=True):
                if not isinstance(mode, bool):
                    raise ValueError("training mode is expected to be boolean")
                self.training = mode
                for module in self.children():
                    module.train(False)
                self.model[-1].classifier.train(mode)
                return self

            def forward(self, *args, **kwargs):
                return self.model(*args, **kwargs)

        # net = Frozen(net)
        # def get_embedding(model, seq):
        #     tokens, masks, time_steps = seq.tokens, seq.masks, seq.time
        #     time_steps = time_steps.permute(1, 0)
        #     x = torch.cat([tokens, masks], dim=-1).permute(1, 0, 2)
        #     outputs = model.bert(x, time_steps)
        #     cls_pooling = outputs["cls_pooling"]  # batch_size, hidden_size
        #     return cls_pooling
        # def get_embeddings(model, loader):
        #     model.eval()
        #     preds = []
        #     gts = []
        #     for batch in loader:
        #         batch.to(model.device)
        #         inp = batch
        #         gt = batch.pop_target()
        #         pred = get_embedding(model, inp).squeeze(1)
        #         preds.append(pred.cpu().numpy())
        #         gts.append(gt.cpu().numpy())
        #     return np.concatenate(preds), np.concatenate(gts)
        # train_x, train_y = get_embeddings(net, loaders["full_train"])
        # test_x, test_y = get_embeddings(net, test_loaders["test"])

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
        trainer.run()
        trainer.load_best_model()

        train_metrics = trainer.validate(loaders["full_train"])
        train_val_metrics = trainer.validate(loaders["train_val"])
        hpo_metrics = trainer.validate(loaders["hpo_val"])
        test_metrics = trainer.validate(test_loaders["test"])

        train_metrics = {"train_" + k: v for k, v in train_metrics.items()}
        train_val_metrics = {"train_val_" + k: v for k, v in train_val_metrics.items()}
        test_metrics = {"test_" + k: v for k, v in test_metrics.items()}

        return dict(**hpo_metrics, **train_metrics, **train_val_metrics, **test_metrics)

    def param_grid(self, trial, config):
        suggest_conf(config["optuna"]["suggestions"], config, trial)
        return trial, config
