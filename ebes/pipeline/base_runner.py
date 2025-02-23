import logging
from multiprocessing import current_process
import sys
import traceback
from abc import ABC, abstractmethod
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor as Pool
from copy import deepcopy
from pathlib import Path

import optuna
import pandas as pd
import torch
import yaml
from omegaconf import DictConfig, OmegaConf
from optuna import Study
from optuna.samplers import TPESampler
from optuna.storages import JournalFileStorage, JournalStorage
from optuna.trial import Trial

from ..utils.general import log_to_file
from ..utils.reproduce import seed_everything
from .utils import (
    access_by_name,
    assign_by_name,
    get_dict_from_trial_params,
    get_unique_folder_suffix,
    parse_n_runs,
    set_start_method,
    set_start_method_as_fork,
)

logger = logging.getLogger()


class Runner(ABC):
    _registry = dict()

    def __init_subclass__(cls, /, name: str | None = None, **kwargs):
        super().__init_subclass__(**kwargs)
        name = name or cls.__name__
        if name in Runner._registry:
            raise ValueError(f"Model named {name} is already registered")
        Runner._registry[name] = cls

    @staticmethod
    def get_runner(name: str, *args, **kwargs):
        try:
            return Runner._registry[name](*args, **kwargs)
        except KeyError:
            raise RuntimeError("No runner named " + name)

    def run(
        self,
        config: Mapping,
    ) -> pd.DataFrame | Study:
        run_type = config["runner"]["run_type"]
        if run_type == "simple":
            return self.do_n_runs(config, **config["runner"]["params"])
        elif run_type == "optuna":
            assert isinstance(config, DictConfig)
            return self.run_optuna(config, **config["optuna"]["params"])
        else:
            raise NotImplementedError("Unknown run_type: " + run_type)

    def run_optuna(
        self,
        config: DictConfig,
        target_metric: str = "val_metric",
        request_list=[],
        n_startup_trials: int = 3,
        n_trials: int = 50,
        multivariate: bool = True,
        group: bool = True,
    ):
        """
        Set target_metric according to _train_eval().
        request_list - list of dicts where {key:value} is
        {trial_parameter_name:parameter_value}
        n_startup_trials == n_random_trials
        n_trials == n_trials to make in total by this function
        call(doesn't affect parallel runs).
        n_runs - better not torch it
        """

        optuna.logging.get_logger("optuna").addHandler(
            logging.StreamHandler(sys.stdout)
        )
        optuna.logging.enable_propagation()
        sampler = TPESampler(
            # seed=0, important to NOT specify, otherwise parallel scripts repeat
            multivariate=multivariate,
            group=group,  # Very usefull, allows to use conditional subsets of parameter
            n_startup_trials=n_startup_trials,
        )
        run_path = Path(config["log_dir"]) / config["run_name"]
        run_path.mkdir(exist_ok=True, parents=True)

        storage = JournalStorage(JournalFileStorage(f"{run_path}/study.log"))
        study = optuna.create_study(
            storage=storage,
            sampler=sampler,
            study_name="hpo",
            direction="maximize",
            load_if_exists=True,
        )

        for request in request_list:
            if isinstance(request, DictConfig):
                request = OmegaConf.to_container(request, resolve=True)
                assert isinstance(request, dict)
            study.enqueue_trial(request, skip_if_exists=True)  # type: ignore

        study.optimize(
            lambda trial: self._objective(trial, config, target_metric),
            n_trials=n_trials,
            catch=(RuntimeError, ValueError),
        )
        return study

    def _objective(
        self,
        trial: Trial,
        config: DictConfig,
        target_metric: str = "val_metric",
    ):
        config = deepcopy(config)
        trial, config = self.param_grid(trial, config)
        config["run_name"] = f"{config['run_name']}/{trial.number}"
        run_path = Path(config["log_dir"]) / config["run_name"]

        try:
            summary_df = self.do_n_runs(config=config, **config["runner"]["params"])
        except Exception as e:
            run_path.mkdir(exist_ok=True, parents=True)
            (run_path / "ERROR.txt").write_text(traceback.format_exc())
            with open(run_path / "params.txt", "w") as file:
                yaml.dump(
                    get_dict_from_trial_params(trial.params), file, sort_keys=False
                )
            logger.exception(e)
            raise e

        with open(run_path / "params.txt", "w") as file:
            yaml.dump(get_dict_from_trial_params(trial.params), file, sort_keys=False)

        for k in summary_df.index:
            trial.set_user_attr(f"{k}_mean", summary_df.loc[k, "mean"])
            trial.set_user_attr(f"{k}_std", summary_df.loc[k, "std"])

        return (
            summary_df.loc[target_metric, "mean"] - summary_df.loc[target_metric, "std"]
        )  # TODO *weak* maybe make customizable objective

    def param_grid(self, trial: Trial, config: DictConfig) -> tuple[Trial, DictConfig]:
        raise NotImplementedError("Implement param grid first")

    def do_n_runs(
        self, config: Mapping, n_runs: int = 3, n_workers: int = 3
    ) -> pd.DataFrame:
        """
        Do n runs with different seed in parralell
        """
        if isinstance(config, DictConfig):
            dict_config = OmegaConf.to_container(config, resolve=True)
            assert isinstance(dict_config, dict)
            config = dict_config
        config = dict(**deepcopy(config))
        config["run_name"] = config["run_name"] + get_unique_folder_suffix(
            Path(config["log_dir"]) / config["run_name"]
        )
        args = [{"config": config, "seed": seed} for seed in range(n_runs)]
        if (len(args) == 1) or (n_workers == 1):
            result_list = [self._run_with_seed(arg) for arg in args]
        else:
            set_start_method(logger)
            with Pool(n_workers) as p:
                result_list = list(p.map(self._run_with_seed, args))
        summary_df = parse_n_runs(result_list)
        summary_df.to_csv(Path(config["log_dir"]) / config["run_name"] / "results.csv")
        return summary_df

    def _run_with_seed(self, kwargs: Mapping) -> dict[str, float]:
        set_start_method_as_fork(logger)
        config = deepcopy(kwargs["config"])
        seed = kwargs["seed"]
        device_list = config["runner"].get("device_list")
        if device_list:
            proccess_id = current_process().name.split("-")[-1]
            device_id = int(proccess_id) if proccess_id.isdigit() else seed
            config["device"] = device_list[device_id % len(device_list)]
        device = config["device"]

        torch.cuda.empty_cache()
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)

        config["run_name"] = f"{config['run_name']}/seed_{seed}"

        if "common_seed" not in config["runner"]["seed_keys"]:
            logger.warning(
                "common_seed will not change during n_runs. "
                "Otherwise add 'common_seed' to config['runner']['seed_keys']"
            )
        for full_key in config["runner"]["seed_keys"]:
            value = access_by_name(config, full_key) + seed
            assign_by_name(config, full_key, value)

        metrics = self._run_and_log(config)
        metrics["memory_after"] = torch.cuda.max_memory_reserved(device) / (2**20)
        return metrics

    def _run_and_log(self, config: Mapping):
        """
        Run experiment with correct logging setup and seed settings.
        """
        log_file = Path(config["log_dir"]) / config["run_name"] / "log"
        log_file.parent.mkdir(exist_ok=True, parents=True)
        seed_everything(
            config["common_seed"],
            avoid_benchmark_noise=True,
            only_deterministic_algorithms=False,
        )
        with open(log_file.parent / "config.yaml", "w") as file:
            yaml.dump(config, file, sort_keys=False)
        with log_to_file(log_file, **config["logging"]):
            return self.pipeline(config)

    @abstractmethod
    def pipeline(self, config: Mapping) -> dict[str, float]:
        """
        Construct your pipeline.
        """
        ...
