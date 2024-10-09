import random
import numpy as np
import torch
from typing import Any


def seed_everything(
    seed: int,
    *,
    avoid_benchmark_noise: bool = False,
    only_deterministic_algorithms: bool = False,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = not avoid_benchmark_noise
    torch.use_deterministic_algorithms(only_deterministic_algorithms, warn_only=True)


def get_global_state() -> dict[str, Any]:
    state_dict = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
        "benchmark": torch.backends.cudnn.benchmark,
        "deterministic_algorithms": torch.backends.cudnn.deterministic,
    }
    return state_dict


def set_global_state(state_dict: dict):
    random.setstate(state_dict["python"])
    np.random.set_state(state_dict["numpy"])
    torch.set_rng_state(state_dict["torch"])
    torch.cuda.set_rng_state_all(state_dict["torch_cuda"])
    torch.backends.cudnn.benchmark = state_dict["benchmark"]
    torch.use_deterministic_algorithms(
        state_dict["deterministic_algorithms"], warn_only=True
    )
