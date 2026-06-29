from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

Role = Literal["reader", "writer", "admin"]


@dataclass(frozen=True)
class Record:
    record_id: int
    owner_id: int
    data: str
    deleted: bool = False


# The whole world the tools act on: who is acting and the record store. State is
# immutable, so each operation returns a new State and the old one is untouched.
@dataclass(frozen=True)
class State:
    actor_id: int
    actor_role: Role
    records: dict[int, Record] = field(default_factory=dict)
    next_id: int = 1

    def create(self, owner_id: int, data: str) -> tuple[State, int]:
        record_id = self.next_id
        records = dict(self.records)
        records[record_id] = Record(record_id, owner_id, data)
        return replace(self, records=records, next_id=record_id + 1), record_id

    def update(self, record_id: int, data: str) -> State:
        records = dict(self.records)
        records[record_id] = replace(records[record_id], data=data)
        return replace(self, records=records)

    def soft_delete(self, record_id: int) -> State:
        records = dict(self.records)
        records[record_id] = replace(records[record_id], deleted=True)
        return replace(self, records=records)