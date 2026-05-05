from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

from ..actions.base import ActionOperator


@dataclass
class ActionVocab:
    # 0 reserved for UNK
    op2id: Dict[str, int]
    tpl2id: Dict[str, int]  # key = f"{op}|{tpl}"

    @property
    def num_ops(self) -> int:
        return max(self.op2id.values()) + 1 if self.op2id else 1

    @property
    def num_tpl(self) -> int:
        return max(self.tpl2id.values()) + 1 if self.tpl2id else 1

    def op_id(self, op: str) -> int:
        return int(self.op2id.get(op, 0))

    def tpl_id(self, op: str, tpl: Optional[str]) -> int:
        key = f"{op}|{tpl or ''}"
        return int(self.tpl2id.get(key, 0))

    def to_json(self) -> str:
        return json.dumps(
            {"op2id": self.op2id, "tpl2id": self.tpl2id}, indent=2, sort_keys=True
        )

    @classmethod
    def from_json(cls, s: str) -> "ActionVocab":
        obj = json.loads(s)
        return cls(
            op2id={k: int(v) for k, v in obj["op2id"].items()},
            tpl2id={k: int(v) for k, v in obj["tpl2id"].items()},
        )

    @classmethod
    def build(
        cls,
        operators: Sequence[ActionOperator],
        *,
        include_terminate: bool = True,
        extra_templates: Optional[Iterable[Tuple[str, str]]] = None,
    ) -> "ActionVocab":
        """
        Deterministic vocab:
          - op IDs assigned by sorted operator name
          - template IDs assigned by sorted (op, tpl) pairs
        extra_templates: iterable of (op_name, template_name) pairs if you have a known library.
        """
        names = sorted({op.name for op in operators})
        if include_terminate:
            names = sorted(set(names) | {"Terminate"})

        op2id = {"<UNK>": 0}
        for i, name in enumerate(names, start=1):
            op2id[name] = i

        # Always include the empty template for each op (tpl=None -> "")
        tpl_keys = set()
        for name in names:
            tpl_keys.add((name, ""))

        if extra_templates:
            for op, tpl in extra_templates:
                tpl_keys.add((op, tpl or ""))

        tpl_sorted = sorted(tpl_keys)
        tpl2id = {"<UNK>": 0}
        for i, (op, tpl) in enumerate(tpl_sorted, start=1):
            tpl2id[f"{op}|{tpl}"] = i

        return cls(op2id=op2id, tpl2id=tpl2id)
