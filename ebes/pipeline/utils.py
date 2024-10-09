import multiprocessing as mp
import os
from collections.abc import Iterable, Mapping
from typing import Any

import optuna
import pandas as pd
import torch
import torch.nn
from omegaconf import DictConfig
from optuna import Trial
from torcheval.metrics import Metric

from .. import losses
from .. import metrics as all_metrics


def set_start_method(logger):
    if torch.cuda.is_initialized() and mp.get_start_method() != "forkserver":
        mp.set_start_method("forkserver", force=True)
        logger.warning(
            "Switching the start method to 'forkserver' because CUDA has already been "
            "initialized. Note that any modifications to the code will be reflected "
            "in new subprocesses. Additionally, this will introduce significant "
            "overhead because creating each DataLoader's worker takes considerably "
            "more time."
        )
        return True
    return False


def set_start_method_as_fork(logger):
    if mp.get_start_method() != "fork":
        try:
            mp.set_start_method("fork", force=True)
            logger.warning("Set start method inside subprocess to 'fork' for speed up")
        except RuntimeError as e:
            logger.error(f"Failed to set start method to 'fork': {e}")


def assign_by_name(config: dict | DictConfig, name: str, value: Any):
    field = config
    for k in name.split(".")[:-1]:
        try:
            field = field[k]
        except KeyError:
            field = field[int(k)]
    field[name.split(".")[-1]] = value


def access_by_name(config: Mapping, name: str):
    field = config
    for k in name.split("."):
        try:
            field = field[k]
        except KeyError:
            field = field[int(k)]
    return field


def get_dict_from_trial_params(params: Mapping):
    config = {}
    for k, v in params.items():
        subconfig = config
        for subkey in k.split(".")[:-1]:
            subconfig[subkey] = subconfig.get(subkey, {})
            subconfig = subconfig[subkey]
        subconfig[k.split(".")[-1]] = v
    return config


def get_unique_folder_suffix(folder_path):
    folder_path = str(folder_path)
    if not os.path.exists(folder_path):
        return ""
    n = 1
    while True:
        suffix = f"({n})"
        if not os.path.exists(folder_path + suffix):
            return suffix
        n += 1


def suggest_conf(suggestions: Mapping, config: dict | DictConfig, trial: Trial):
    for name, suggestion in suggestions.items():
        value = getattr(trial, suggestion[0])(name, **suggestion[1])
        assign_by_name(config, name, value)


def get_optimizer(
    net_params: Iterable[torch.nn.Parameter],
    name: str = "Adam",
    params: Mapping[str, Any] | None = None,
):
    params = params or {}
    try:
        return getattr(torch.optim, name)(net_params, **params)
    except AttributeError:
        raise ValueError(f"Unknkown optimizer: {name}")


def get_scheduler(
    optimizer: torch.optim.Optimizer, name: str, params: Mapping[str, Any] | None = None
):
    params = params or {}
    try:
        return getattr(torch.optim.lr_scheduler, name)(optimizer, **params)
    except AttributeError:
        raise ValueError(f"Unknkown LR scheduler: {name}")


def get_metrics(
    metric_specs: list[str | Mapping[str, Any]] | None = None,
    device: torch.device | str | None = None,
) -> list[Metric]:

    if metric_specs is None:
        return []

    metrics = []
    for el in metric_specs or []:
        if isinstance(el, str):
            metrics.append(getattr(all_metrics, el)(device=device))
        else:
            params = el.get("params", {})
            metrics.append(getattr(all_metrics, el["name"])(device=device, **params))
    return metrics


def get_loss(name: str, params: Mapping[str, Any] | None = None):
    params = {**params} if params else {}
    if name[:3] == "nn.":
        loss_fn = getattr(torch.nn, name[3:])(**params)
    elif name in ["ContrastiveLoss", "InfoNCELoss"]:
        selector = getattr(losses.contrastive, params.pop("selector"))(
            params.pop("neg_count")
        )
        loss_fn = getattr(losses.contrastive, name)(pair_selector=selector, **params)
    else:
        try:
            loss_fn = getattr(losses, name)(**params)
        except AttributeError:
            raise ValueError(f"Unkown loss {name}")
    return loss_fn


def parse_n_runs(result_list: list[dict[str, float]]) -> pd.DataFrame:
    summary_df = pd.DataFrame({})
    for i, metrics in enumerate(result_list):
        for k in metrics:
            summary_df.loc[k, i] = metrics[k]
    mean_col, std_col = summary_df.mean(axis=1), summary_df.std(axis=1, ddof=0)
    mean_col, std_col = pd.DataFrame({"mean": mean_col}), pd.DataFrame({"std": std_col})
    summary_df = pd.concat([summary_df, mean_col, std_col], axis=1)
    return summary_df


def optuna_df(path="log/test", name=None) -> tuple[pd.DataFrame, optuna.Study]:
    from optuna.storages import JournalFileStorage, JournalStorage

    storage = JournalStorage(JournalFileStorage(f"{path}/study.log"))
    name = "hpo" if name is None else name
    study = optuna.load_study(study_name=name, storage=storage)
    df = study.trials_dataframe()
    df = df.drop(
        columns=[
            "number",
        ]
    )

    return df, study
