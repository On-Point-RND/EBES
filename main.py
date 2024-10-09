# ignoring all issues with config keys
# pyright: reportArgumentType=false

import argparse
from functools import partialmethod
from pathlib import Path
from typing import Any
from collections.abc import Mapping
import signal
import pdb

from omegaconf import OmegaConf
from tqdm import tqdm

from ebes.pipeline.base_runner import Runner


def start_debugging(_, frame):
    pdb.Pdb().set_trace(frame)


def collect_config(
    dataset, method, experiment, specify=None, gpu=None
) -> dict[str, Any]:
    data_config = OmegaConf.load(Path(f"configs/datasets/{dataset}.yaml"))
    method_config = OmegaConf.load(Path(f"configs/methods/{method}.yaml"))
    exp_config = OmegaConf.load(Path(f"configs/experiments/{experiment}.yaml"))
    configs = [data_config, method_config, exp_config]

    if specify is not None:
        specify_path = Path(f"configs/specify/{dataset}/{method}/{specify}.yaml")
        if specify_path.exists():
            configs.append(OmegaConf.load(specify_path))
        else:
            raise ValueError(f"No specification {specify}")

    config = OmegaConf.merge(*configs)
    if gpu is not None:
        assert config.runner.get("device_list") is None
        config["device"] = gpu
    return config  # type: ignore


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="age")
    parser.add_argument("-m", "--method", type=str, default="gru")
    parser.add_argument("-e", "--experiment", type=str, default="test")
    parser.add_argument("-s", "--specify", type=str, default=None)
    parser.add_argument("-g", "--gpu", type=str, default=None)
    parser.add_argument(
        "-a",
        "--ablation-type",
        choices=["none", "time", "permutation", "permutation_keep_last"],
        default="none",
    )
    parser.add_argument("--tqdm", action="store_true")
    args = parser.parse_args()

    signal.signal(signal.SIGUSR1, start_debugging)

    tqdm.__init__ = partialmethod(tqdm.__init__, disable=not args.tqdm)  # type: ignore
    config = collect_config(
        args.dataset, args.method, args.experiment, args.specify, args.gpu
    )

    if args.ablation_type != "none":
        config["run_name"] = (
            config["run_name"].rpartition("/")[0] + f"/{args.ablation_type}"
        )

    for data_sec in ("data", "test_data"):
        for pl in config[data_sec]["preprocessing"].values():
            tfs: list[str | Mapping[str, Any]] = pl["batch_transforms"]

            if args.ablation_type == "time":
                tfs.append("RandomTime")
            elif args.ablation_type == "permutation":
                tfs.append("RandomEventsPermutation")
            elif args.ablation_type == "permutation_keep_last":
                tfs.append({"RandomEventsPermutation": {"keep_last": True}})

    runner = Runner.get_runner(config["runner"]["name"])
    res = runner.run(config)
    print(res)
