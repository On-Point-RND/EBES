# ignoring all issues with config keys
# pyright: reportArgumentType=false

from argparse import ArgumentParser
from collections.abc import Mapping
from functools import partialmethod
from itertools import count
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from ebes.data.utils import build_loaders
from ebes.model import build_model
from ebes.pipeline.utils import get_metrics
from ebes.trainer import Trainer

METRIC_FOR_DS = {
    # "mimic3": "MulticlassAUROC",
    # "physionet2012": "MulticlassAUROC",
    # "age": "MulticlassAccuracy",
    # "x5": "MulticlassAccuracy",
    # "pendulum_cls": "MulticlassAccuracy",
    # "taobao": "MulticlassAUROC",
    "mbd": "MultiLabelMeanAUROC",
    # "arabic": "MulticlassAccuracy",
    # "electric_devices": "MulticlassAccuracy",
    # "bpi_17": "MulticlassAUROC",
}
METHODS = [
    # "mamba",
    # "gru",
    # "mlp",
    # "primenet",
    # "mtand",
    # "coles",
    # "mlem",
    # "transformer",
    "convtran",
]


def eval_ablation(exp_dir: Path, which: str, device: str):
    print(f"evaluating {exp_dir.as_posix()}")
    config = OmegaConf.load(exp_dir / "config.yaml")
    for pl in config["test_data"]["preprocessing"].values():
        tfs: list[str | Mapping[str, Any]] = pl["batch_transforms"]

        if which == "time":
            tfs.append("RandomTime")
        elif which == "permutation":
            tfs.append("RandomEventsPermutation")
        elif which == "permutation_keep_last":
            tfs.append({"RandomEventsPermutation": {"keep_last": True}})
        elif which == "none":
            pass
        else:
            raise ValueError("Unknown ablation type")

    config = OmegaConf.to_container(config, resolve=True)
    assert isinstance(config, dict)

    test_loaders = build_loaders(**config["test_data"])
    metrics = get_metrics(config["metrics"], "cpu")
    net = build_model(config["model"])

    trainer = Trainer(
        model=net,
        metrics=metrics,
        ckpt_dir=exp_dir / "ckpt",
        device=device,
    )
    trainer.load_best_model()
    try:
        test_metrics = trainer.validate(test_loaders["test"])
    except:
        print(config)
        raise
    return test_metrics


if __name__ == "__main__":
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)  # type: ignore
    parser = ArgumentParser()
    parser.add_argument(
        "which", choices=["time", "permutation", "permutation_keep_last", "none"]
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", default="0,20")
    args = parser.parse_args()

    smin, smax = args.seeds.split(",")
    smin, smax = int(smin), int(smax)
    path = Path("log/")
    res_file = path / "Ablations" / f"{args.which}_{smin}-{smax}.csv"
    for i in count(1):
        if not res_file.exists():
            break
        res_file = path / "Ablations" / f"{args.which}_{smin}-{smax}_{i}.csv"

    rows = []
    for ds_dir in path.iterdir():
        if not ds_dir.is_dir() or ds_dir.stem not in METRIC_FOR_DS:
            continue
        for method in METHODS:
            corr_dir = ds_dir / method / "correlation"
            if not corr_dir.exists():
                continue

            for seed in range(smin, smax):
                seed_dir = corr_dir / f"seed_{seed}"
                if not seed_dir.is_dir():
                    continue
                if (
                    not (seed_dir / "ckpt").exists()
                    or len(list((seed_dir / "ckpt").iterdir())) == 0
                ):
                    continue

                row = {}
                try:
                    metrics = eval_ablation(seed_dir, args.which, args.device)
                    m = metrics[METRIC_FOR_DS[ds_dir.name]]
                except Exception as e:
                    m = float("nan")
                    print(e)

                row["metric"] = m
                row["dataset"] = ds_dir.name
                row["method"] = method
                row["seed"] = seed
                rows.append(row)
                pd.DataFrame(rows).to_csv(res_file, index=False)

            torch.cuda.empty_cache()
