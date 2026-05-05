# leadopt/config/preset_yaml.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


class PresetValidationError(ValueError):
    """Raised when a preset YAML does not match the expected schema."""


def _err(msg: str) -> PresetValidationError:
    return PresetValidationError(msg)


def _require_dict(x: Any, path: str) -> Dict[str, Any]:
    if not isinstance(x, dict):
        raise _err(f"{path} must be a mapping/dict, got {type(x).__name__}")
    return x


def _require_list(x: Any, path: str) -> List[Any]:
    if not isinstance(x, list):
        raise _err(f"{path} must be a list, got {type(x).__name__}")
    return x


def _require_str(x: Any, path: str) -> str:
    if not isinstance(x, str):
        raise _err(f"{path} must be a string, got {type(x).__name__}")
    return x


def _require_int(x: Any, path: str) -> int:
    if not isinstance(x, int):
        raise _err(f"{path} must be an int, got {type(x).__name__}")
    return x


def _require_num(x: Any, path: str) -> float:
    if not isinstance(x, (int, float)):
        raise _err(f"{path} must be a number, got {type(x).__name__}")
    return float(x)


def _require_len3_num_list(x: Any, path: str) -> List[float]:
    xs = _require_list(x, path)
    if len(xs) != 3:
        raise _err(f"{path} must have length 3, got {len(xs)}")
    return [_require_num(v, f"{path}[{i}]") for i, v in enumerate(xs)]


def _require_bool(x: Any, path: str) -> bool:
    if not isinstance(x, bool):
        raise _err(f"{path} must be a bool, got {type(x).__name__}")
    return x


def _require_str_pair_list(x: Any, path: str) -> List[List[str]]:
    xs = _require_list(x, path)
    out: List[List[str]] = []
    for i, row in enumerate(xs):
        row = _require_list(row, f"{path}[{i}]")
        if len(row) != 2:
            raise _err(f"{path}[{i}] must have length 2, got {len(row)}")
        a = _require_str(row[0], f"{path}[{i}][0]")
        b = _require_str(row[1], f"{path}[{i}][1]")
        out.append([a, b])
    return out


def _validate_operator_params(op_type: str, params: Dict[str, Any], path: str) -> None:
    """
    Conservative operator param validation (Phase 7.5 contract freeze).

    Rules:
      - Always require params to be a dict
      - For known operators, validate types/ranges of known keys
      - Allow extra keys (forward-compatible), unless the operator is intentionally keyless
    """
    params = _require_dict(params, path)

    # Keyless operators: params should be empty (catch typos)
    keyless = {
        "PruneTerminal",
        "AzaScanAromatic",
        "FunctionalGroupSwap",
        "AtomMutation",
        "LinkerInsertCH2",
        "LinkerDeleteCH2",
    }
    if op_type in keyless:
        if len(params) != 0:
            raise _err(
                f"{path} for operator {op_type} must be empty, got keys: {sorted(params.keys())}"
            )
        return

    # DeleteSubtree
    if op_type == "DeleteSubtree":
        if "max_deleted_atoms" in params:
            v = _require_int(params["max_deleted_atoms"], f"{path}.max_deleted_atoms")
            if v <= 0:
                raise _err(f"{path}.max_deleted_atoms must be > 0, got {v}")
        return

    # RingSubstituentDelete
    if op_type == "RingSubstituentDelete":
        for k in ["max_deleted_atoms", "min_heavy_atoms", "max_fragments"]:
            if k in params:
                v = _require_int(params[k], f"{path}.{k}")
                if v <= 0:
                    raise _err(f"{path}.{k} must be > 0, got {v}")
        return

    # RGroupSwap
    if op_type == "RGroupSwap":
        if "max_sidechain_heavy_atoms" in params:
            v = _require_int(
                params["max_sidechain_heavy_atoms"], f"{path}.max_sidechain_heavy_atoms"
            )
            if v <= 0:
                raise _err(f"{path}.max_sidechain_heavy_atoms must be > 0, got {v}")
        if "library" in params:
            _require_str(params["library"], f"{path}.library")
        if "include_library_version_in_payload" in params:
            _require_bool(
                params["include_library_version_in_payload"],
                f"{path}.include_library_version_in_payload",
            )
        return

    # AddSubstituent
    if op_type == "AddSubstituent":
        if "templates" in params:
            _require_str_pair_list(params["templates"], f"{path}.templates")
        if "library" in params:
            _require_str(params["library"], f"{path}.library")
        if "include_library_version_in_payload" in params:
            _require_bool(
                params["include_library_version_in_payload"],
                f"{path}.include_library_version_in_payload",
            )
        return

    # BioisostereSwap
    if op_type == "BioisostereSwap":
        if "library" in params:
            _require_str_pair_list(params["library"], f"{path}.library")
        return

    # ReactionSMARTSOperator
    if op_type == "ReactionSMARTSOperator":
        if "reactions" not in params:
            raise _err(
                f"{path} for ReactionSMARTSOperator must define {path}.reactions"
            )
        _require_str_pair_list(params["reactions"], f"{path}.reactions")
        return

    # FragmentationOperator (Tier 2.1)
    if op_type == "FragmentationOperator":
        if "mode" in params:
            m = _require_str(params["mode"], f"{path}.mode")
            if m not in {"largest", "contains_anchor"}:
                raise _err(
                    f"{path}.mode must be 'largest' or 'contains_anchor', got {m}"
                )
        if "anchor_smarts" in params and params["anchor_smarts"] is not None:
            _require_str(params["anchor_smarts"], f"{path}.anchor_smarts")
        for k in [
            "max_cuts_per_step",
            "min_heavy_atoms",
            "max_deleted_atoms",
            "max_fragments",
        ]:
            if k in params:
                v = _require_int(params[k], f"{path}.{k}")
                if v <= 0:
                    raise _err(f"{path}.{k} must be > 0, got {v}")
        if "method" in params:
            meth = _require_str(params["method"], f"{path}.method")
            if meth.lower() not in {"brics"}:
                raise _err(f"{path}.method must be 'brics' for now, got {meth}")
        if "log_library_version" in params:
            _require_bool(params["log_library_version"], f"{path}.log_library_version")
        return

    # Unknown operator: keep permissive (params must be dict, extra keys allowed)
    return


def _validate_docking_params(params: Dict[str, Any], path: str) -> None:
    """
    Scorer-specific schema validation for DockingScorer.

    Minimal, conservative contract:
      - engine: str
      - protocol: str
      - receptor_path: str
      - either box (center+size) OR box_file
      - allow extra keys (future extensions)
    """
    params = _require_dict(params, path)

    _require_str(params.get("engine"), f"{path}.engine")
    protocol = _require_str(params.get("protocol"), f"{path}.protocol")
    if protocol not in {"standard", "aligned_local"}:
        raise _err(
            f"{path}.protocol must be one of 'standard'/'aligned_local', got {protocol}"
        )

    _require_str(params.get("receptor_path"), f"{path}.receptor_path")

    has_box = "box" in params
    has_box_file = "box_file" in params
    if not (has_box or has_box_file):
        raise _err(f"{path} must define either {path}.box or {path}.box_file")

    if has_box:
        box = _require_dict(params["box"], f"{path}.box")
        _require_len3_num_list(box.get("center"), f"{path}.box.center")
        _require_len3_num_list(box.get("size"), f"{path}.box.size")

    if has_box_file:
        _require_str(params["box_file"], f"{path}.box_file")

    # Behavioral implementation arrives in later stages; this block only validates types.
    if protocol == "aligned_local":
        if "reference_ligand_path" in params:
            _require_str(
                params["reference_ligand_path"], f"{path}.reference_ligand_path"
            )
        elif "reference_ligand" in params:
            # Backwards-compatible alias for early adopters; prefer reference_ligand_path going forward.
            _require_str(params["reference_ligand"], f"{path}.reference_ligand")
        else:
            raise _err(
                f"{path} with protocol 'aligned_local' must define {path}.reference_ligand_path (or legacy {path}.reference_ligand)"
            )

        if "reference_conformer" in params:
            _require_int(params["reference_conformer"], f"{path}.reference_conformer")

        # Optional blocks (type checks only; allow extra keys for forward-compatibility)
        for block_key in ["alignment", "anchor", "local_opt"]:
            if block_key in params:
                _require_dict(params[block_key], f"{path}.{block_key}")

    # Optional knobs (type checks only)
    if "seed" in params:
        _require_int(params["seed"], f"{path}.seed")
    if "timeout_s" in params:
        _require_num(params["timeout_s"], f"{path}.timeout_s")
    if "cache_dir" in params:
        _require_str(params["cache_dir"], f"{path}.cache_dir")
    if "budget" in params:
        _require_int(params["budget"], f"{path}.budget")
    if "engine_version" in params:
        _require_str(params["engine_version"], f"{path}.engine_version")
    if "params" in params:
        _require_dict(params["params"], f"{path}.params")


def _validate_mpo_params(params: Dict[str, Any], path: str) -> None:
    """
    Scorer-specific schema validation for MPOScorer.

    Keep this conservative and forward-compatible:
    - require the minimal MPO structure
    - allow additional keys for future extensions
    """
    params = _require_dict(params, path)

    # Required
    props = _require_list(params.get("properties"), f"{path}.properties")
    if len(props) == 0:
        raise _err(f"{path}.properties must be a non-empty list")

    for i, p in enumerate(props):
        p = _require_dict(p, f"{path}.properties[{i}]")
        _require_str(p.get("name"), f"{path}.properties[{i}].name")
        _require_num(p.get("weight"), f"{path}.properties[{i}].weight")

        # Optional: transform block (reserved for Phase 3.2/3.3 extensions)
        if "transform" in p:
            t = _require_dict(p["transform"], f"{path}.properties[{i}].transform")
            _require_str(t.get("type"), f"{path}.properties[{i}].transform.type")
            tparams = t.get("params", {})
            _require_dict(tparams, f"{path}.properties[{i}].transform.params")

    # Optional: aggregation mode (default weighted_sum)
    if "aggregation" in params:
        agg = _require_str(params["aggregation"], f"{path}.aggregation")
        if agg not in {"weighted_sum"}:
            raise _err(f"{path}.aggregation must be 'weighted_sum' for now, got {agg}")


@dataclass(frozen=True)
class PresetYAML:
    """
    Phase 1 artifact: validated YAML parsed into a plain dict.
    Phase 2 will add object construction (operators, scorer, constraints, etc.).
    """

    raw: Dict[str, Any]
    source_path: Optional[Path] = None


def validate_preset_dict(d: Dict[str, Any]) -> None:
    """
    Strict schema validation for Phase 1.
    Keep this conservative: require the core structure, allow extra keys.
    """
    d = _require_dict(d, "$")

    # Required top-level
    preset_version = _require_int(d.get("preset_version"), "$.preset_version")
    if preset_version != 1:
        raise _err(f"$.preset_version must be 1 for now, got {preset_version}")

    _require_str(d.get("name"), "$.name")

    # actions
    actions = _require_dict(d.get("actions"), "$.actions")
    ops = _require_list(actions.get("operators"), "$.actions.operators")
    if len(ops) == 0:
        raise _err("$.actions.operators must be a non-empty list")

    # Optional: legality_constraint (action gating constraint)
    if "legality_constraint" in actions:
        lc = _require_dict(
            actions["legality_constraint"], "$.actions.legality_constraint"
        )
        _require_str(lc.get("type"), "$.actions.legality_constraint.type")
        params = lc.get("params", {})
        _require_dict(params, "$.actions.legality_constraint.params")

    for i, op in enumerate(ops):
        op = _require_dict(op, f"$.actions.operators[{i}]")
        op_type = _require_str(op.get("type"), f"$.actions.operators[{i}].type")
        params = op.get("params", {})
        _require_dict(params, f"$.actions.operators[{i}].params")
        _validate_operator_params(op_type, params, f"$.actions.operators[{i}].params")

    max_steps = _require_int(actions.get("max_steps"), "$.actions.max_steps")
    if max_steps <= 0:
        raise _err(f"$.actions.max_steps must be > 0, got {max_steps}")

    # rules (optional but recommended)
    if "rules" in actions:
        rules = _require_dict(actions["rules"], "$.actions.rules")
        # keep loose in Phase 1 (types checked only if present)
        for k in ["ban_motifs", "use_pains", "use_brenk", "use_nih"]:
            if k in rules and not isinstance(rules[k], bool):
                raise _err(
                    f"$.actions.rules.{k} must be bool, got {type(rules[k]).__name__}"
                )
        for k in ["max_mw", "max_logp"]:
            if k in rules:
                _require_num(rules[k], f"$.actions.rules.{k}")

    # constraints
    constraints = _require_list(d.get("constraints"), "$.constraints")
    for i, c in enumerate(constraints):
        c = _require_dict(c, f"$.constraints[{i}]")
        _require_str(c.get("type"), f"$.constraints[{i}].type")
        params = c.get("params", {})
        _require_dict(params, f"$.constraints[{i}].params")

    # scoring
    scoring = _require_dict(d.get("scoring"), "$.scoring")
    scorer_type = _require_str(scoring.get("type"), "$.scoring.type")
    params = scoring.get("params", {})
    _require_dict(params, "$.scoring.params")

    if scorer_type == "MPOScorer":
        _validate_mpo_params(params, "$.scoring.params")
    if scorer_type == "DockingScorer":
        _validate_docking_params(params, "$.scoring.params")

    # reward
    reward = _require_dict(d.get("reward"), "$.reward")
    mode = _require_str(reward.get("mode"), "$.reward.mode")
    if mode not in {"terminal", "delta", "potential"}:
        raise _err(f"$.reward.mode must be one of terminal/delta/potential, got {mode}")

    for k in [
        "step_penalty",
        "compute_cost_weight",
        "constraint_penalty_weight",
        "bonus",
    ]:
        if k in reward:
            _require_num(reward[k], f"$.reward.{k}")

    if "complexity_weight" in reward:
        _require_num(reward["complexity_weight"], "$.reward.complexity_weight")

    # beam (optional)
    # Beam-search generation parameters (used by leadopt beam). Keeping this optional
    # preserves backward compatibility for older presets.
    if "beam" in d:
        beam = _require_dict(d.get("beam"), "$.beam")
        if "complexity_weight" in beam:
            _require_num(beam["complexity_weight"], "$.beam.complexity_weight")
        if "dock_drop_tolerance" in beam and beam["dock_drop_tolerance"] is not None:
            _require_num(beam["dock_drop_tolerance"], "$.beam.dock_drop_tolerance")
        if "hard_constraint_filter" in beam:
            _require_bool(
                beam["hard_constraint_filter"], "$.beam.hard_constraint_filter"
            )

    # model (optional)
    # Model hyperparameters may be specified in presets. If omitted, downstream
    # components should use their own defaults.
    if "model" in d:
        model = _require_dict(d.get("model"), "$.model")
        mtype = _require_str(model.get("type"), "$.model.type")
        if mtype != "MPNNPolicy":
            raise _err(f"$.model.type must be 'MPNNPolicy' for now, got {mtype}")

        mparams = model.get("params", {})
        _require_dict(mparams, "$.model.params")

        for k in ["hidden_dim", "mp_steps", "emb_dim"]:
            if k in mparams:
                v = _require_int(mparams[k], f"$.model.params.{k}")
                if v <= 0:
                    raise _err(f"$.model.params.{k} must be > 0, got {v}")

    # training (optional)
    # Training hyperparameters may be specified in presets. If omitted, defaults
    # should be used by downstream components.
    if "training" in d:
        training = _require_dict(d.get("training"), "$.training")

        # algorithm is optional; if provided, must be 'ppo' for now.
        if "algorithm" in training:
            alg = _require_str(training.get("algorithm"), "$.training.algorithm")
            if alg != "ppo":
                raise _err(
                    f"$.training.algorithm must be 'ppo' for now, got {training['algorithm']}"
                )

        # Optional numeric fields (validate if present)
        if "total_timesteps" in training:
            _require_int(training.get("total_timesteps"), "$.training.total_timesteps")
        if "gamma" in training:
            _require_num(training.get("gamma"), "$.training.gamma")
        if "seed" in training:
            _require_int(training.get("seed"), "$.training.seed")

    # logging
    logging = _require_dict(d.get("logging"), "$.logging")
    for k in [
        "save_preset_yaml",
        "log_git_commit",
        "log_scorer_metadata",
        "log_constraint_metadata",
    ]:
        if k in logging and not isinstance(logging[k], bool):
            raise _err(f"$.logging.{k} must be bool, got {type(logging[k]).__name__}")


def load_preset_yaml(path: Union[str, Path]) -> PresetYAML:
    path = Path(path)
    raw_obj = yaml.safe_load(path.read_text())
    raw = _require_dict(raw_obj, "$")
    validate_preset_dict(raw)
    return PresetYAML(raw=raw, source_path=path)
