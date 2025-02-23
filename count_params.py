# ignoring all issues with config keys
# pyright: reportArgumentType=false

import warnings
from glob import glob
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm

from ebes.model import build_model

DATASETS_PRETTY = {
    "mbd": "MBD",
    "x5": "Retail",
    "age": "Age",
    "taobao": "Taobao",
    "bpi_17": "BPI17",
    "physionet2012": "PhysioNet2012",
    "mimic3": "MIMIC-III",
    "pendulum_cls": "Pendulum",
    "arabic": "ArabicDigits",
    "electric_devices": "ElectricDevices",
}

METHODS_PRETTY = {
    "coles": "CoLES",
    "gru": "GRU",
    "mlem": "MLEM",
    "transformer": "Transformer",
    "mamba": "Mamba",
    "convtran": "ConvTran",
    "mtand": "mTAND",
    "primenet": "PrimeNet",
    "mlp": "MLP",
}


# Suppress the specific UserWarning
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="dropout option adds dropout after all but last recurrent layer.*",
)


def collect_config(dataset, method, specify=None) -> dict[str, Any]:
    data_config = OmegaConf.load(Path(f"configs/datasets/{dataset}.yaml"))
    method_config = OmegaConf.load(Path(f"configs/methods/{method}.yaml"))
    exp_config = OmegaConf.load(Path("configs/experiments/test.yaml"))

    if specify is None:
        specify_path = Path(f"configs/specify/{dataset}/{method}/best.yaml")
    else:
        specify_path = Path(specify)

    configs = [data_config, method_config, exp_config]

    configs.append(OmegaConf.load(specify_path))

    config = OmegaConf.merge(*configs)
    config["device"] = "cpu"
    return config  # type: ignore


def get_param_counts(dataset, method, specify=None):
    conf = collect_config(dataset, method, specify)
    model = build_model(conf["model"])
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    index = pd.MultiIndex.from_product(
        [DATASETS_PRETTY.values(), ["min", "best", "max"]], names=["Method", "Params"]
    )
    res = pd.DataFrame(index=index, columns=METHODS_PRETTY.values())
    for dataset in DATASETS_PRETTY:
        print(dataset.upper(), "STARTED")
        for method in METHODS_PRETTY:
            best_c = get_param_counts(dataset, method)
            best_c = f"{best_c:.1e}"
            optuna_counts = []
            for spec in tqdm(glob(f"log/{dataset}/{method}/optuna/*/params.txt")):
                spec = Path(spec)
                if not spec.with_name("results.csv").exists():
                    continue
                optuna_counts += [get_param_counts(dataset, method, spec)]
            min_c, max_c = f"{min(optuna_counts):.1e}", f"{max(optuna_counts):.1e}"
            res.loc[(DATASETS_PRETTY[dataset], "min"), METHODS_PRETTY[method]] = min_c
            res.loc[(DATASETS_PRETTY[dataset], "best"), METHODS_PRETTY[method]] = (
                f"\\cellcolor{{lightgray}}{best_c}"
            )
            res.loc[(DATASETS_PRETTY[dataset], "max"), METHODS_PRETTY[method]] = max_c

    print(
        res.to_latex(
            bold_rows=True,
            column_format="r" * (len(METHODS_PRETTY) + 2),
        )
    )
    res.to_csv("log/Ablations/param_counts.csv")
