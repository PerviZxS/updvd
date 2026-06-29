from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .schemas import CreateArgs, DeleteArgs, ToolArgs, UpdateArgs
from .state import State
from .verdicts import OK, Check, Verdict


# A contract is one tool's rules: what must hold before it runs (precondition),
# how it changes state (apply), and what must hold afterwards (postcondition).
class Contract(ABC):
    name: str

    @abstractmethod
    def precondition(self, state: State, args: ToolArgs) -> Check: ...

    @abstractmethod
    def apply(self, state: State, args: ToolArgs) -> tuple[State, Any]: ...

    @abstractmethod
    def postcondition(self, before: State, after: State, args: ToolArgs, produced: Any) -> Check: ...


class CreateRecord(Contract):
    name = "create_record"

    def precondition(self, state: State, args: CreateArgs) -> Check:
        if state.actor_role not in ("writer", "admin"):
            return Check(Verdict.AUTHORIZATION, f"role '{state.actor_role}' cannot create records")
        return OK

    def apply(self, state: State, args: CreateArgs) -> tuple[State, int]:
        return state.create(args.owner_id, args.data)

    def postcondition(self, before: State, after: State, args: CreateArgs, produced: int) -> Check:
        if len(after.records) != len(before.records) + 1:
            return Check(Verdict.POSTCONDITION, "record count did not increase by one")
        record = after.records.get(produced)
        if record is None or record.owner_id != args.owner_id or record.data != args.data or record.deleted:
            return Check(Verdict.POSTCONDITION, "created record does not match the request")
        return OK


class UpdateRecord(Contract):
    name = "update_record"

    def precondition(self, state: State, args: UpdateArgs) -> Check:
        record = state.records.get(args.record_id)
        if record is None:
            return Check(Verdict.REFERENTIAL, f"record {args.record_id} does not exist")
        if record.deleted:
            return Check(Verdict.LIVENESS, f"record {args.record_id} is deleted")
        if state.actor_role != "admin" and record.owner_id != state.actor_id:
            return Check(Verdict.OWNERSHIP, f"actor {state.actor_id} does not own record {args.record_id}")
        return OK

    def apply(self, state: State, args: UpdateArgs) -> tuple[State, None]:
        return state.update(args.record_id, args.data), None

    def postcondition(self, before: State, after: State, args: UpdateArgs, produced: None) -> Check:
        if after.records[args.record_id].data != args.data:
            return Check(Verdict.POSTCONDITION, "record data was not updated")
        for key, value in before.records.items():
            if key != args.record_id and after.records.get(key) != value:
                return Check(Verdict.POSTCONDITION, f"unrelated record {key} was modified")
        return OK


class DeleteRecord(Contract):
    name = "delete_record"

    def precondition(self, state: State, args: DeleteArgs) -> Check:
        record = state.records.get(args.record_id)
        if record is None:
            return Check(Verdict.REFERENTIAL, f"record {args.record_id} does not exist")
        if record.deleted:
            return Check(Verdict.LIVENESS, f"record {args.record_id} is already deleted")
        if state.actor_role != "admin":
            return Check(Verdict.AUTHORIZATION, f"role '{state.actor_role}' cannot delete records")
        return OK

    def apply(self, state: State, args: DeleteArgs) -> tuple[State, None]:
        return state.soft_delete(args.record_id), None

    def postcondition(self, before: State, after: State, args: DeleteArgs, produced: None) -> Check:
        if not after.records[args.record_id].deleted:
            return Check(Verdict.POSTCONDITION, "record was not marked deleted")
        if len(after.records) != len(before.records):
            return Check(Verdict.POSTCONDITION, "store size changed during soft delete")
        return OK


CONTRACTS: dict[str, Contract] = {
    contract.name: contract
    for contract in (CreateRecord(), UpdateRecord(), DeleteRecord())
}