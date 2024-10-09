from contextlib import contextmanager
from pathlib import Path
import logging
import sys
from time import time


class LoadTime:
    def __init__(self, loader, disable=False):
        self.loader = loader
        self.full_time = 0
        self.iterator = None
        self._disable = disable

    def __iter__(self):
        self.iterator = iter(self.loader)
        return self

    def __next__(self):
        if self.iterator is None:
            raise RuntimeError(
                "Iterator not initialized. Call __iter__() before __next__()."
            )
        try:
            start = time()
            value = next(self.iterator)
            self.full_time += time() - start
            return value
        except StopIteration:
            if not self._disable:
                print(f"Data Loading took: {self.full_time} seconds")
            self.iterator = None
            self.full_time = 0
            raise


def grad_norm(named_parameters):
    total_sq_norm = 0.0
    for n, p in named_parameters:
        if p.grad is None:
            print("GRAD IS NONE", n)
        else:
            param_norm = p.grad.detach().data.norm(2)
            total_sq_norm += param_norm.item() ** 2
    return total_sq_norm**0.5


@contextmanager
def log_to_file(filename: Path, file_lvl="info", cons_lvl="warning"):
    if isinstance(file_lvl, str):
        file_lvl = getattr(logging, file_lvl.upper())
    if isinstance(cons_lvl, str):
        cons_lvl = getattr(logging, cons_lvl.upper())

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(cons_lvl)
    cfmt = logging.Formatter("{levelname:8} - {asctime} - {message}", style="{")
    ch.setFormatter(cfmt)

    fh = logging.FileHandler(filename)
    fh.setLevel(file_lvl)
    ffmt = logging.Formatter(
        "{name: ^16} - {asctime} - {message}",
        style="{",
    )
    fh.setFormatter(ffmt)
    logger = logging.getLogger()
    logger.setLevel(min(file_lvl, cons_lvl))
    logger.addHandler(fh)
    logger.addHandler(ch)

    try:
        yield
    finally:
        fh.close()
        logger.removeHandler(fh)
        logger.removeHandler(ch)
