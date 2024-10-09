""" Skeleton model with general structure"""

from torch import nn


class BaseModel(nn.Module):
    _registry = dict()

    def __init_subclass__(cls, /, name: str | None = None, **kwargs):
        super().__init_subclass__(**kwargs)
        name = name or cls.__name__
        if name in BaseModel._registry:
            raise ValueError(f"Model named {name} is already registered")
        BaseModel._registry[name] = cls

    @staticmethod
    def get_model(name: str, *args, **kwargs):
        return BaseModel._registry[name](*args, **kwargs)
