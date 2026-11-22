# The interceptor: the only component allowed to change state.
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from .contracts import CONTRACTS
from .schemas import SCHEMAS
from .state import State
from .trace import Trace
from .verdicts import Check, Verdict


# Result of one proposal: committed or not, why, and the state afterwards.
@dataclass(frozen=True)
class Decision:
    committed: bool
    verdict: Verdict
    detail: str
    state: State

    @property
    def error_signal(self) -> str | None:
        if self.committed:
            return None
        return f"[{self.verdict.value}] {self.detail}"


# Short message for the first pydantic validation error.
def _schema_error(exc: ValidationError) -> str:
    first = exc.errors()[0]
    location = ".".join(str(part) for part in first["loc"]) or "<args>"
    return f"field '{location}': {first['msg']}"


# Checks schema -> precondition -> apply+postcondition, commits only if all pass, and logs every decision.
class Interceptor:
    def __init__(self, state: State, trace: Trace | None = None) -> None:
        self._state = state
        # Trace defines __len__, so an empty Trace is falsy and `trace or Trace()` would drop it.
        self.trace = trace if trace is not None else Trace()

    @property
    def state(self) -> State:
        return self._state

    def propose(self, tool: str, args: dict[str, Any]) -> Decision:
        # 0. Shape: a model can send a non-string tool or non-object args, which would raise below.
        if not isinstance(tool, str) or not isinstance(args, dict):
            return self._reject(tool, args, Check(Verdict.SCHEMA, "proposal must be a tool name and an argument object"))

        schema = SCHEMAS.get(tool)
        contract = CONTRACTS.get(tool)
        if schema is None or contract is None:
            return self._reject(tool, args, Check(Verdict.SCHEMA, f"unknown tool '{tool}'"))

        # 1. Schema: arguments must be well-formed.
        try:
            parsed = schema(**args)
        except ValidationError as exc:
            return self._reject(tool, args, Check(Verdict.SCHEMA, _schema_error(exc)))

        # 2. Precondition against the current state.
        pre = contract.precondition(self._state, parsed)
        if not pre.ok:
            return self._reject(tool, args, pre)

        # 3. Apply to a candidate state and check the postcondition; any exception rejects (fail-closed).
        try:
            candidate, produced = contract.apply(self._state, parsed)
            post = contract.postcondition(self._state, candidate, parsed, produced)
        except Exception as exc:
            return self._reject(
                tool, args, Check(Verdict.POSTCONDITION, f"contract raised: {exc!r}")
            )
        if not post.ok:
            return self._reject(tool, args, post)

        # All checks passed: commit the candidate state.
        self._state = candidate
        self.trace.append(tool, args, Verdict.OK.value, "", committed=True)
        return Decision(True, Verdict.OK, "", self._state)

    def _reject(self, tool: str, args: dict[str, Any], check: Check) -> Decision:
        self.trace.append(tool, args, check.verdict.value, check.detail, committed=False)
        return Decision(False, check.verdict, check.detail, self._state)
