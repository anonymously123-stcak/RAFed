"""Minimal local compatibility layer for the subset of functorch used here."""

from __future__ import annotations

import copy
from typing import Callable, Iterable, List, Sequence, Tuple

import torch


def _set_nested_parameter(module: torch.nn.Module, name: str, value: torch.Tensor) -> None:
    parts = name.split(".")
    target = module
    for part in parts[:-1]:
        target = target._modules[part]
    target._parameters[parts[-1]] = value


def make_functional(module: torch.nn.Module) -> Tuple[Callable[..., torch.Tensor], List[torch.Tensor]]:
    """Return a callable that evaluates ``module`` with explicit parameters.

    The implementation keeps the original module structure intact by working on
    a cloned copy and swapping parameters in-place before each forward pass.
    This is enough for the training code in this repository, which only relies
    on ``make_functional`` and parameter-list based functional calls.
    """

    functional_module = copy.deepcopy(module)
    param_names = [name for name, _ in functional_module.named_parameters()]
    params = [p.detach().clone().requires_grad_(True) for p in functional_module.parameters()]

    def functional(param_values: Sequence[torch.Tensor], *args, **kwargs):
        for name, value in zip(param_names, param_values):
            _set_nested_parameter(functional_module, name, value)
        return functional_module(*args, **kwargs)

    return functional, params

