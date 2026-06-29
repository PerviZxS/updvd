from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Verdict(str, Enum):
    OK = "ok"
    SCHEMA = "schema_violation"
    AUTHORIZATION = "authorization_violation"
    REFERENTIAL = "referential_violation"
    LIVENESS = "liveness_violation"
    OWNERSHIP = "ownership_violation"
    POSTCONDITION = "postcondition_violation"


@dataclass(frozen=True)
class Check:
    verdict: Verdict
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.verdict is Verdict.OK


OK = Check(Verdict.OK)
