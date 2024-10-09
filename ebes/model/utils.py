from collections.abc import Mapping
from typing import Any

from torch import nn

from .basemodel import BaseModel


def build_model(model_conf: Mapping[str, Any]) -> nn.Module:

    if "name" in model_conf:
        name = model_conf["name"]
        params = model_conf.get("params", {})
        assert isinstance(name, str)

        try:
            return BaseModel.get_model(name, **params)
        except KeyError:
            pass

        if name[:3] == "nn.":
            try:
                return getattr(nn, name[3:])(**params)
            except AttributeError:
                pass

        raise ValueError(
            f"Can't find {name} neither in `torch.nn` nor in defined models"
        )

    layers = []

    def resolve_output_dim():
        for m in reversed(layers):
            try:
                return m.output_dim
            except AttributeError:
                pass
        raise AttributeError

    for i, it in model_conf.items():
        name = it["name"]
        params = it.get("params", {})

        resolved_params = dict(params)
        for k, v in params.items():
            if v == "output_dim_from_prev":  # get output dim from prevoius layers
                try:
                    v = resolve_output_dim()
                except AttributeError:
                    raise AttributeError(
                        f"Can't auto detect input dimensionality for layer {i}: "
                        "none of previous layers has an `output_dim` property"
                    )

                resolved_params[k] = v

        model = build_model({"name": name, "params": resolved_params})
        layers.append(model)

    return nn.Sequential(*layers)


class FrozenModel(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model.eval()
        for param in model.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(False)
        return self

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
