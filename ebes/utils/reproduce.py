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


def spawn_generator() -> np.random.Generator:
    """Create a fresh NumPy generator seeded from PyTorch's global RNG.

    Using the global torch RNG as the entropy source makes the draws vary across
    epochs and DataLoader workers (PyTorch reseeds every worker each epoch as
    ``base_seed + worker_id``, and ``base_seed`` is redrawn from the global RNG for
    each new iterator), while staying reproducible across runs whenever the global
    seed is fixed via :func:`seed_everything`.
    """
    seed = int(torch.randint(0, 2**63 - 1, (1,)).item())
    return np.random.default_rng(seed)


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
