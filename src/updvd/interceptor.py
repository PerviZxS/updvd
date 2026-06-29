from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from .contracts import CONTRACTS
from .schemas import SCHEMAS
from .state import State
from .trace import Trace
from .verdicts import Check, Verdict


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


def _schema_error(exc: ValidationError) -> str:
    first = exc.errors()[0]
    location = ".".join(str(part) for part in first["loc"]) or "<args>"
    return f"field '{location}': {first['msg']}"


class Interceptor:
    # The only component allowed to change state. It checks each proposed call in
    # order and commits it only if every check passes; otherwise it rejects.
    def __init__(self, state: State, trace: Trace | None = None) -> None:
        self._state = state
        self.trace = trace or Trace()

    @property
    def state(self) -> State:
        return self._state

    def propose(self, tool: str, args: dict[str, Any]) -> Decision:
        schema = SCHEMAS.get(tool)
        contract = CONTRACTS.get(tool)
        if schema is None or contract is None:
            return self._reject(tool, args, Check(Verdict.SCHEMA, f"unknown tool '{tool}'"))

        # 1. The arguments must be well-formed before any contract check runs.
        try:
            parsed = schema(**args)
        except ValidationError as exc:
            return self._reject(tool, args, Check(Verdict.SCHEMA, _schema_error(exc)))

        # 2. The precondition must hold against the current state.
        pre = contract.precondition(self._state, parsed)
        if not pre.ok:
            return self._reject(tool, args, pre)

        # 3. Apply on a copy and check the postcondition. The real state is only
        #    advanced if that copy passes, so a failed action leaves no trace.
        candidate, produced = contract.apply(self._state, parsed)
        post = contract.postcondition(self._state, candidate, parsed, produced)
        if not post.ok:
            return self._reject(tool, args, post)

        self._state = candidate
        self.trace.append(tool, args, Verdict.OK.value, "", committed=True)
        return Decision(True, Verdict.OK, "", self._state)

    def _reject(self, tool: str, args: dict[str, Any], check: Check) -> Decision:
        self.trace.append(tool, args, check.verdict.value, check.detail, committed=False)
        return Decision(False, check.verdict, check.detail, self._state)