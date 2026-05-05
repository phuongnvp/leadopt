# leadopt/config/preset_loader.py
from __future__ import annotations

import difflib
import inspect
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union

from leadopt.actions.base import ActionOperator
from leadopt.actions.operators import (
    AddSubstituent,
    AromaticPositionalScan,
    AromaticSubstituentSwap,
    AtomMutation,
    AzaScanAromatic,
    BioisostereSwap,
    DeleteSubtree,
    FragmentationOperator,
    FunctionalGroupSwap,
    LinkerAtomSwap,
    LinkerDeleteCH2,
    LinkerInsertCH2,
    PruneTerminal,
    ReactionSMARTSOperator,
    RGroupSwap,
    RingSubstituentDelete,
    SmirksLibraryOperator,
)
from leadopt.constraints import (
    ChargeConstraint,
    Constraint,
    ConstraintSuite,
    ElementConstraint,
    HBDHBAConstraint,
    ProtectedCoreConstraint,
    ReactiveGroupConstraint,
    RingCountConstraint,
    SimilarityConstraint,
    SizeConstraint,
)
from leadopt.core.rules import RuleConfig
from leadopt.env.graph_env import GraphEnvironment
from leadopt.rl.ppo import PPOConfig
from leadopt.scoring import (
    CompositeScorer,
    DockingScorer,
    LegacyFunctionScorer,
    MPOScorer,
    QSARScorer,
    RealQSARScorer,
    RewardComposer,
    Scorer,
)

from .preset_yaml import PresetYAML, load_preset_yaml

# -----------------------------
# Registry (safe construction)
# -----------------------------

OPERATOR_REGISTRY = {
    "PruneTerminal": PruneTerminal,
    "AtomMutation": AtomMutation,
    "AddSubstituent": AddSubstituent,
    "FunctionalGroupSwap": FunctionalGroupSwap,
    "RGroupSwap": RGroupSwap,
    "LinkerInsertCH2": LinkerInsertCH2,
    "LinkerDeleteCH2": LinkerDeleteCH2,
    "AzaScanAromatic": AzaScanAromatic,
    "BioisostereSwap": BioisostereSwap,
    "DeleteSubtree": DeleteSubtree,
    "ReactionSMARTSOperator": ReactionSMARTSOperator,
    "RingSubstituentDelete": RingSubstituentDelete,
    "FragmentationOperator": FragmentationOperator,
    "AromaticPositionalScan": AromaticPositionalScan,
    "AromaticSubstituentSwap": AromaticSubstituentSwap,
    "SmirksLibraryOperator": SmirksLibraryOperator,
    "LinkerAtomSwap": LinkerAtomSwap,
}

CONSTRAINT_REGISTRY: Dict[str, Type[Constraint]] = {
    "SizeConstraint": SizeConstraint,
    "SimilarityConstraint": SimilarityConstraint,
    "ChargeConstraint": ChargeConstraint,
    "ElementConstraint": ElementConstraint,
    "RingCountConstraint": RingCountConstraint,
    "HBDHBAConstraint": HBDHBAConstraint,
    "ReactiveGroupConstraint": ReactiveGroupConstraint,
    "ProtectedCoreConstraint": ProtectedCoreConstraint,
}

SCORER_REGISTRY: Dict[str, Type[Scorer]] = {
    "LegacyFunctionScorer": LegacyFunctionScorer,
    "QSARScorer": QSARScorer,
    "RealQSARScorer": RealQSARScorer,
    "qsar_real": RealQSARScorer,
    "MPOScorer": MPOScorer,
    "DockingScorer": DockingScorer,
    "CompositeScorer": CompositeScorer,
    "composite": CompositeScorer,
}


def _maybe_inject_seed(
    scorer_type: str, scorer_params: Dict[str, Any], run_seed: int
) -> Dict[str, Any]:
    """
    Reproducibility helper: if a scorer supports a 'seed' parameter and the preset
    does not specify it, inject the unified run seed.

    This is conservative: only applies to known scorers where seed matters.
    """
    if scorer_type in {"DockingScorer"}:
        if "seed" not in scorer_params or scorer_params.get("seed", None) is None:
            out = dict(scorer_params)
            out["seed"] = int(run_seed)
            return out
    return scorer_params


# -----------------------------
# Types / results
# -----------------------------


@dataclass(frozen=True)
class LoadedPreset:
    """
    Phase 2 artifact: fully constructed components from YAML.
    Phase 3 will add run-manifest logging + YAML copy.
    Phase 4 will wire this into the main training script CLI.
    """

    preset: PresetYAML

    operators: List[ActionOperator]
    constraint_suite: Optional[ConstraintSuite]
    legality_constraint_factory: Optional[Callable[[], Constraint]]
    scorer: Scorer
    reward_composer: RewardComposer

    env_kwargs: Dict[str, Any]
    ppo_config: PPOConfig
    training_meta: Dict[str, Any]  # e.g. total_timesteps, algorithm, etc.
    # Optional model hyperparameters (e.g., for MPNNPolicy). If absent in YAML,
    # defaults are provided by the loader for downstream consumers.
    model_meta: Dict[str, Any]


class PresetLoaderError(RuntimeError):
    pass


def _constructor_signature(cls: Type[Any]) -> inspect.Signature:
    """Best-effort constructor signature for actionable error messages."""
    try:
        return inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        # Some C-extensions may not have a signature.
        return inspect.Signature()


def _signature_kwargs_help(cls: Type[Any]) -> str:
    """Human-friendly allowed kwargs summary for a class constructor."""
    sig = _constructor_signature(cls)
    if not sig.parameters:
        return "Allowed params: <unknown>"

    allowed: List[str] = []
    required: List[str] = []
    has_varkw = False
    for name, p in sig.parameters.items():
        if name == "self":
            continue
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            has_varkw = True
            continue
        if p.kind == inspect.Parameter.VAR_POSITIONAL:
            continue
        allowed.append(name)
        if p.default is inspect._empty:
            required.append(name)

    parts: List[str] = []
    if required:
        parts.append(f"Required: {sorted(required)}")
    if allowed:
        parts.append(f"Allowed: {sorted(allowed)}")
    if has_varkw:
        parts.append("Constructor accepts **kwargs (extra keys allowed).")
    return "; ".join(parts)


def _unexpected_kwargs(cls: Type[Any], params: Dict[str, Any]) -> Optional[List[str]]:
    """Return unexpected kwarg keys if constructor does not accept them."""
    sig = _constructor_signature(cls)
    if not sig.parameters:
        return None

    allowed: set[str] = set()
    has_varkw = False
    for name, p in sig.parameters.items():
        if name == "self":
            continue
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            has_varkw = True
            continue
        if p.kind == inspect.Parameter.VAR_POSITIONAL:
            continue
        allowed.add(name)

    if has_varkw:
        return []
    return sorted([k for k in params.keys() if k not in allowed])


def _import_callable(spec: str) -> Callable[..., Any]:
    """
    Import a callable from "module.submodule:callable_name".
    """
    if ":" not in spec:
        raise PresetLoaderError(
            f"Invalid callable spec '{spec}'. Expected format 'module.path:callable'."
        )
    mod_name, attr = spec.split(":", 1)
    mod = import_module(mod_name)
    try:
        fn = getattr(mod, attr)
    except AttributeError as e:
        raise PresetLoaderError(
            f"Callable '{attr}' not found in module '{mod_name}'."
        ) from e
    if not callable(fn):
        raise PresetLoaderError(f"Imported object '{spec}' is not callable.")
    return fn


def _resolve_type(type_str: str, registry: Dict[str, Type[Any]]) -> Type[Any]:
    """
    Resolve either:
      - registered short name: "PruneTerminal"
      - explicit dotted: "dotted:some.module:ClassName"
    """
    if type_str in registry:
        return registry[type_str]

    if type_str.startswith("dotted:"):
        # dotted:my_pkg.my_mod:MyClass
        dotted = type_str[len("dotted:") :]
        if ":" not in dotted:
            raise PresetLoaderError(
                f"Invalid dotted type '{type_str}'. Expected 'dotted:module.path:ClassName'."
            )
        mod_name, cls_name = dotted.split(":", 1)
        mod = import_module(mod_name)
        try:
            cls = getattr(mod, cls_name)
        except AttributeError as e:
            raise PresetLoaderError(
                f"Class '{cls_name}' not found in module '{mod_name}'."
            ) from e
        if not isinstance(cls, type):
            raise PresetLoaderError(
                f"Dotted import '{type_str}' did not resolve to a class."
            )
        return cls

    known = ", ".join(sorted(registry.keys()))
    suggestions = difflib.get_close_matches(
        type_str, sorted(registry.keys()), n=3, cutoff=0.6
    )
    hint = f" Did you mean: {suggestions}?" if suggestions else ""
    raise PresetLoaderError(f"Unknown type '{type_str}'. Known: {known}.{hint}")


def _build_operators(raw: Dict[str, Any]) -> List[ActionOperator]:
    ops_raw = raw["actions"]["operators"]
    ops: List[ActionOperator] = []
    for i, item in enumerate(ops_raw):
        t = item["type"]
        params = dict(item.get("params") or {})
        cls = _resolve_type(t, OPERATOR_REGISTRY)
        bad = _unexpected_kwargs(cls, params)
        if bad:
            raise PresetLoaderError(
                f"Operator {t} has unexpected params {bad}. {_signature_kwargs_help(cls)}"
            )
        try:
            ops.append(cls(**params))
        except TypeError as e:
            raise PresetLoaderError(
                f"Failed constructing operator {t} with params={params}: {e}. {_signature_kwargs_help(cls)}"
            ) from e
    return ops


def _build_constraint_suite(raw: Dict[str, Any]) -> Optional[ConstraintSuite]:
    constraints_raw = raw.get("constraints") or []
    if len(constraints_raw) == 0:
        return None

    constraints: List[Constraint] = []
    for i, item in enumerate(constraints_raw):
        t = item["type"]
        params = dict(item.get("params") or {})
        cls = _resolve_type(t, CONSTRAINT_REGISTRY)

        # Backwards-compat aliases for SimilarityConstraint field names.
        if cls is SimilarityConstraint:
            if "lead_smiles" not in params:
                if "reference_smiles" in params:
                    params["lead_smiles"] = params.pop("reference_smiles")
                elif "reference" in params:
                    params["lead_smiles"] = params.pop("reference")
                elif "lead" in params:
                    params["lead_smiles"] = params.pop("lead")

            if "min_sim" not in params:
                if "min_tanimoto" in params:
                    params["min_sim"] = params.pop("min_tanimoto")
                elif "min_similarity" in params:
                    params["min_sim"] = params.pop("min_similarity")

            if "max_sim" not in params:
                if "max_tanimoto" in params:
                    params["max_sim"] = params.pop("max_tanimoto")
                elif "max_similarity" in params:
                    params["max_sim"] = params.pop("max_similarity")
        bad = _unexpected_kwargs(cls, params)
        if bad:
            raise PresetLoaderError(
                f"Constraint {t} has unexpected params {bad}. {_signature_kwargs_help(cls)}"
            )
        try:
            constraints.append(cls(**params))
        except TypeError as e:
            raise PresetLoaderError(
                f"Failed constructing constraint {t} with params={params}: {e}. {_signature_kwargs_help(cls)}"
            ) from e

    return ConstraintSuite(constraints)


def _build_legality_constraint_factory(
    raw: Dict[str, Any],
) -> Optional[Callable[[], Constraint]]:
    """
    Optional action-gating constraint configured under:
      actions.legality_constraint: {type: ..., params: {...}}

    This is distinct from reward/logging constraints (top-level `constraints:`),
    and is passed to GraphEnvironment.constraint_factory so ActionSpace can gate actions.
    """
    actions = raw.get("actions") or {}
    lc = actions.get("legality_constraint", None)
    if lc is None:
        return None
    if not isinstance(lc, dict):
        raise PresetLoaderError(
            "actions.legality_constraint must be a dict if provided."
        )
    t = lc.get("type", None)
    if not isinstance(t, str) or not t:
        raise PresetLoaderError(
            "actions.legality_constraint.type must be a non-empty string."
        )
    params = lc.get("params", {}) or {}
    if not isinstance(params, dict):
        raise PresetLoaderError("actions.legality_constraint.params must be a dict.")

    cls = _resolve_type(str(t), CONSTRAINT_REGISTRY)

    bad = _unexpected_kwargs(cls, dict(params))
    if bad:
        raise PresetLoaderError(
            f"Legality constraint {t} has unexpected params {bad}. {_signature_kwargs_help(cls)}"
        )

    def _factory() -> Constraint:
        try:
            return cls(**params)
        except Exception as e:
            raise PresetLoaderError(
                f"Failed constructing legality constraint {t} with params={params}: {e}"
            ) from e

    return _factory


def _build_rule_config(raw: Dict[str, Any]) -> Optional[RuleConfig]:
    rules = (raw.get("actions") or {}).get("rules")
    if not rules:
        return None
    if not isinstance(rules, dict):
        raise PresetLoaderError("actions.rules must be a dict if provided.")

    # Only pass fields that exist on RuleConfig (prevents silent typos)
    allowed = set(RuleConfig.__dataclass_fields__.keys())
    filtered = {k: v for k, v in rules.items() if k in allowed}
    unknown = sorted(set(rules.keys()) - allowed)
    if unknown:
        raise PresetLoaderError(
            f"Unknown keys in actions.rules: {unknown}. Allowed: {sorted(allowed)}"
        )

    return RuleConfig(**filtered)


def _build_scorer(raw: Dict[str, Any], run_seed: int = 0) -> Scorer:
    scoring = raw["scoring"]
    scorer_type = scoring["type"]
    params = dict(scoring.get("params") or {})

    cls = _resolve_type(scorer_type, SCORER_REGISTRY)

    if cls is LegacyFunctionScorer:
        fn_spec = params.get("function")
        if not isinstance(fn_spec, str) or not fn_spec:
            raise PresetLoaderError(
                "LegacyFunctionScorer requires scoring.params.function = 'module.path:callable'."
            )
        score_fn = _import_callable(fn_spec)

        # Optional extras
        fail_objective = params.get("fail_objective", None)
        name = params.get("name", "LegacyFunctionScorer")
        version = params.get("version", "0")
        extra_metadata = dict(params.get("metadata") or {})

        return LegacyFunctionScorer(
            score_fn,
            fail_objective=fail_objective,
            name=name,
            version=version,
            extra_metadata=extra_metadata,
        )

    # Unified seed provenance: if a scorer supports 'seed' and the preset omits it,
    # inject the unified run seed.
    if scorer_type in {"DockingScorer"}:
        if "seed" not in params or params.get("seed", None) is None:
            params["seed"] = int(run_seed)

    # Backwards-compatible QSARScorer param aliases (Phase 7.6 presets)
    if scorer_type == "QSARScorer":
        # QSARScorer in leadopt is a deterministic RDKit-property scorer.
        # Accept a few legacy/preset-friendly aliases and ignore irrelevant keys.
        if "objective_name" in params and "objective" not in params:
            params["objective"] = params.pop("objective_name")

        # Some presets use a "model_path" concept; QSARScorer does not.
        if "model_path" in params:
            params.pop("model_path", None)
        if "artifact_path" in params:
            params.pop("artifact_path", None)

        # Not used by QSARScorer; keep presets flexible.
        params.pop("higher_is_better", None)
        params.pop("target_name", None)

    # Phase 8.A: RealQSARScorer supports nested YAML config under scoring.params
    # while the dataclass uses flat fields for stable construction.
    if scorer_type in {"qsar_real", "RealQSARScorer"}:
        model_cfg = dict(params.get("model") or {})
        features_cfg = dict(params.get("features") or {})
        cache_cfg = dict(params.get("cache") or {})

        # Allow both nested and flat forms; flat takes precedence if explicitly provided.
        if "model_path" not in params:
            params["model_path"] = model_cfg.get("path", "")
        if "input_mode" not in params:
            params["input_mode"] = model_cfg.get("input_mode", "fingerprint")

        # Fingerprint config (only used when input_mode=fingerprint)
        if "features_kind" not in params:
            params["features_kind"] = features_cfg.get("kind", "morgan")
        if "features_radius" not in params:
            params["features_radius"] = features_cfg.get("radius", 2)
        if "features_n_bits" not in params:
            params["features_n_bits"] = features_cfg.get("n_bits", 2048)
        if "features_use_chirality" not in params:
            params["features_use_chirality"] = features_cfg.get("use_chirality", True)
        if "features_use_features" not in params:
            params["features_use_features"] = features_cfg.get("use_features", False)

        # Cache config
        if "cache_enabled" not in params:
            params["cache_enabled"] = cache_cfg.get("enabled", True)
        if "cache_dir" not in params:
            params["cache_dir"] = cache_cfg.get("dir", ".leadopt_cache/qsar_real")

        # Remove nested dicts to avoid passing unexpected kwargs to the dataclass.
        params.pop("model", None)
        params.pop("features", None)
        params.pop("cache", None)

    # Phase 9.A: CompositeScorer supports nested YAML config under scoring.params.
    # It evaluates multiple sub-scorers and aggregates them into a single scalar objective.
    if scorer_type in {"composite", "CompositeScorer"}:
        aggregation = dict(params.get("aggregation") or {})
        scorers_spec = params.get("scorers")

        if not isinstance(scorers_spec, list) or len(scorers_spec) == 0:
            raise PresetLoaderError(
                "CompositeScorer requires scoring.params.scorers = [ ... ]"
            )

        mode = aggregation.get("mode", "weighted_sum")
        weights = dict(aggregation.get("weights") or {})
        normalize = bool(aggregation.get("normalize", False))

        sub_scorers = []
        for spec in scorers_spec:
            if not isinstance(spec, dict):
                raise PresetLoaderError(
                    "CompositeScorer scorers entries must be dicts."
                )
            name = spec.get("name")
            stype = spec.get("type")
            sparams = dict(spec.get("params") or {})

            if not isinstance(name, str) or not name:
                raise PresetLoaderError(
                    "CompositeScorer scorers entries require a non-empty 'name'."
                )
            if not isinstance(stype, str) or not stype:
                raise PresetLoaderError(
                    "CompositeScorer scorers entries require a non-empty 'type'."
                )

            # Reuse this function recursively to build the sub-scorer.
            sub = _build_scorer(
                {"scoring": {"type": stype, "params": sparams}}, run_seed=run_seed
            )
            sub_scorers.append((name, sub))

        # Optional extras for CompositeScorer itself
        fail_objective = params.get("fail_objective", None)
        version = params.get("version", "0")
        extra_metadata = dict(params.get("metadata") or {})

        return cls(
            scorers=sub_scorers,
            aggregation_mode=str(mode),
            weights={str(k): float(v) for k, v in weights.items()},
            normalize=normalize,
            fail_objective=(
                float(fail_objective)
                if fail_objective is not None
                else cls.fail_objective
            ),
            version=str(version),
            extra_metadata=extra_metadata,
        )

    # For future scorers: allow generic construction
    bad = _unexpected_kwargs(cls, params)
    if bad:
        raise PresetLoaderError(
            f"Scorer {scorer_type} has unexpected params {bad}. {_signature_kwargs_help(cls)}"
        )

    try:
        return cls(**params)
    except TypeError as e:
        raise PresetLoaderError(
            f"Failed constructing scorer {scorer_type} with params={params}: {e}. {_signature_kwargs_help(cls)}"
        ) from e


def _build_reward_composer(raw: Dict[str, Any]) -> RewardComposer:
    r = raw["reward"]
    t = raw.get("training", {}) or {}

    # RewardComposer is frozen => construct with kwargs
    kwargs: Dict[str, Any] = {}

    # Core fields
    if "mode" in r:
        kwargs["mode"] = r["mode"]

    # If reward.gamma not specified, default to training.gamma for consistency
    if "gamma" in r:
        kwargs["gamma"] = float(r["gamma"])
    else:
        kwargs["gamma"] = float(t.get("gamma", 0.99))

    # Penalties/bonus
    if "step_penalty" in r:
        kwargs["step_penalty"] = float(r["step_penalty"])
    if "constraint_penalty_weight" in r:
        kwargs["constraint_penalty_weight"] = float(r["constraint_penalty_weight"])
    if "compute_cost_weight" in r:
        kwargs["compute_cost_weight"] = float(r["compute_cost_weight"])

    # YAML uses "bonus" but RewardComposer expects "bonus_weight"
    if "bonus" in r:
        kwargs["bonus_weight"] = float(r["bonus"])

    try:
        return RewardComposer(**kwargs)
    except TypeError as e:
        raise PresetLoaderError(
            f"Failed constructing RewardComposer with kwargs={kwargs}. Error: {e}"
        ) from e

def _build_env_kwargs(
    raw: Dict[str, Any], rule_config: Optional[RuleConfig]
) -> Dict[str, Any]:
    actions = raw["actions"]
    training = raw.get("training", {}) or {}

    env_kwargs: Dict[str, Any] = {
        "max_steps": int(actions.get("max_steps", 8)),
        "seed": int(training.get("seed", 0)),
        "require_connected": True,
        "include_terminate": True,
        "rule_config": rule_config,
        # Keep these aligned with reward shaping when used
        "reward_mode": raw["reward"]["mode"],
        "gamma": float(training.get("gamma", 0.99)),
        "step_penalty": float(raw["reward"].get("step_penalty", 0.0)),
    }
    return env_kwargs

def _build_model_meta(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract model hyperparameters from YAML, defaulting conservatively.

    Phase 2.1 contract:
      - Only MPNNPolicy is supported
      - If absent in YAML, return defaults
      - If partially specified, merge with defaults
      - Preserve any extra keys in params for forward compatibility
    """

    defaults: Dict[str, Any] = {"hidden_dim": 128, "mp_steps": 3, "emb_dim": 32}

    model = raw.get("model", None)
    if not isinstance(model, dict):
        return dict(defaults)

    mtype = model.get("type", "MPNNPolicy")
    if mtype != "MPNNPolicy":
        raise PresetLoaderError(
            f"Unsupported model.type={mtype!r}. Only 'MPNNPolicy' is supported."
        )

    params = model.get("params", {})
    if not isinstance(params, dict):
        raise PresetLoaderError(
            f"Invalid model.params type: expected dict, got {type(params).__name__}"
        )

    out = dict(defaults)
    out.update(params)
    return out


def _build_ppo_config(raw: Dict[str, Any]) -> PPOConfig:
    t = raw.get("training", {}) or {}

    # Start with defaults, override from YAML if present
    cfg = PPOConfig()

    # Common fields
    if "gamma" in t:
        cfg.gamma = float(t["gamma"])
    if "seed" in t:
        cfg.seed = int(t["seed"])

    # Optional advanced PPO fields (allowed if provided)
    optional_fields = [
        "lam",
        "clip_ratio",
        "ent_coef",
        "vf_coef",
        "lr",
        "max_grad_norm",
        "rollout_episodes",
        "update_epochs",
        "minibatch_size",
    ]
    for k in optional_fields:
        if k in t:
            setattr(cfg, k, t[k])

    return cfg


class PresetLoader:
    """
    Phase 2 loader:
      YAML -> operators/constraints/scorer/reward/env_kwargs/ppo_config
    """

    def load(self, preset_path: Union[str, Path]) -> LoadedPreset:
        preset = load_preset_yaml(preset_path)
        raw = preset.raw

        operators = _build_operators(raw)
        constraint_suite = _build_constraint_suite(raw)
        legality_constraint_factory = _build_legality_constraint_factory(raw)
        rule_config = _build_rule_config(raw)

        # Training meta comes directly from YAML (keep it JSON-friendly and stable).
        # We also pass through optional beam-search configuration under "beam".
        training_meta = dict(raw.get("training") or {})
        if "beam" in raw and isinstance(raw.get("beam"), dict):
            training_meta["beam"] = dict(raw.get("beam") or {})
        run_seed = int(training_meta.get("seed", 0))
        scorer = _build_scorer(raw, run_seed=run_seed)
        reward_composer = _build_reward_composer(raw)
        env_kwargs = _build_env_kwargs(raw, rule_config)
        ppo_config = _build_ppo_config(raw)
        model_meta = _build_model_meta(raw)

        return LoadedPreset(
            preset=preset,
            operators=operators,
            constraint_suite=constraint_suite,
            legality_constraint_factory=legality_constraint_factory,
            scorer=scorer,
            reward_composer=reward_composer,
            env_kwargs=env_kwargs,
            ppo_config=ppo_config,
            training_meta=training_meta,
            model_meta=model_meta,
        )

    def build_env(self, loaded: LoadedPreset) -> GraphEnvironment:
        """
        Convenience: construct GraphEnvironment directly.
        """
        return GraphEnvironment(
            operators=loaded.operators,
            scorer=loaded.scorer,
            reward_composer=loaded.reward_composer,
            max_steps=loaded.env_kwargs["max_steps"],
            seed=loaded.env_kwargs["seed"],
            require_connected=loaded.env_kwargs["require_connected"],
            include_terminate=loaded.env_kwargs["include_terminate"],
            constraint_suite=loaded.constraint_suite,
            constraint_factory=loaded.legality_constraint_factory,
            rule_config=loaded.env_kwargs.get("rule_config"),
            reward_mode=loaded.env_kwargs.get("reward_mode", "potential"),
            gamma=loaded.env_kwargs.get("gamma", 0.99),
            step_penalty=loaded.env_kwargs.get("step_penalty", 0.0),
        )
