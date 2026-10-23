"""Strategy registry: look up, register and derive strategies.

New strategies can be added without touching the runner or evaluator:

* in YAML (``configs/experiments.yaml`` under ``strategies:``) as a variant of
  an existing strategy with option overrides, e.g.::

      graph_assisted_no_alias:
        base: graph_assisted
        description: Condition D without alias resolution
        options: {graph: {alias_resolution: false}}

* in Python with ``register_strategy(StrategySpec(...))``, optionally with a
  custom ``builder`` that returns a different compiled LangGraph workflow.
"""

from __future__ import annotations

from typing import Any

from csr_lab.config import PipelineOptions, deep_merge
from csr_lab.pipelines.base import StrategySpec
from csr_lab.pipelines.strategies import BUILT_IN

_REGISTRY: dict[str, StrategySpec] = {s.name: s for s in BUILT_IN}


def register_strategy(spec: StrategySpec, *, replace: bool = False) -> StrategySpec:
    if spec.name in _REGISTRY and not replace:
        existing = _REGISTRY[spec.name]
        if existing.options == spec.options and existing.builder == spec.builder:
            return existing
        raise ValueError(f"strategy {spec.name!r} is already registered with different settings")
    _REGISTRY[spec.name] = spec
    return spec


def get_strategy(name: str) -> StrategySpec:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown strategy {name!r}; available: {sorted(_REGISTRY)}") from None


def list_strategies() -> list[StrategySpec]:
    return [_REGISTRY[k] for k in sorted(_REGISTRY)]


def derive_strategy(name: str, base: str, overrides: dict[str, Any] | None = None, description: str = "") -> StrategySpec:
    parent = get_strategy(base)
    options = PipelineOptions.model_validate(deep_merge(parent.options.model_dump(), overrides or {}))
    return StrategySpec(
        name=name,
        description=description or f"{parent.description} (variant of {base})",
        options=options,
        builder=parent.builder,
        base=base,
    )


def register_variants(definitions: dict[str, dict[str, Any]] | None) -> list[str]:
    """Register YAML-defined variants. Variants may build on other variants."""
    names = []
    pending = dict(definitions or {})
    while pending:
        progressed = False
        for name, spec in list(pending.items()):
            base = spec.get("base")
            if not base:
                raise ValueError(f"strategy variant {name!r} needs a 'base'")
            if base in pending:
                continue
            register_strategy(derive_strategy(name, base, spec.get("options"), spec.get("description", "")))
            names.append(name)
            del pending[name]
            progressed = True
        if not progressed:
            raise ValueError(f"strategy variants have unknown or circular bases: {sorted(pending)}")
    return names


def build_workflow(spec: StrategySpec):
    if spec.builder is not None:
        return spec.builder(spec.options)
    if spec.options.workflow == "single_pass":
        from csr_lab.pipelines.baseline import build_single_pass_workflow

        return build_single_pass_workflow(spec.options)
    from csr_lab.pipelines.langgraph_workflow import build_staged_workflow

    return build_staged_workflow(spec.options)
