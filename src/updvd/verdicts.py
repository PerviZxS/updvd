# Verdict codes and the result of a single check.
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


# Outcome categories; the values are written to the trace and the result files.
class Verdict(str, Enum):
    OK = "ok"
    SCHEMA = "schema_violation"
    AUTHORIZATION = "authorization_violation"
    REFERENTIAL = "referential_violation"
    LIVENESS = "liveness_violation"
    OWNERSHIP = "ownership_violation"
    POSTCONDITION = "postcondition_violation"


# A verdict plus a human-readable reason.
@dataclass(frozen=True)
class Check:
    verdict: Verdict
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.verdict is Verdict.OK


# Shared passing check.
OK = Check(Verdict.OK)
