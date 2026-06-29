from __future__ import annotations

from .proposer import Proposal, ScriptedProposer
from .state import Record, State


def demo_state() -> State:
    return State(
        actor_id=1,
        actor_role="writer",
        records={
            10: Record(10, owner_id=1, data="alpha"),
            11: Record(11, owner_id=2, data="beta"),
        },
        next_id=12,
    )


SCRIPT: dict[str, list[Proposal]] = {
    "commit a valid create": [
        Proposal("create_record", {"data": "gamma", "owner_id": 1}),
    ],
    "commit a valid update": [
        Proposal("update_record", {"record_id": 10, "data": "alpha-2"}),
    ],
    "recover from a schema violation": [
        Proposal("update_record", {"record_id": "10", "data": "x"}),
        Proposal("update_record", {"record_id": 10, "data": "x"}),
    ],
    "recover from an ownership violation": [
        Proposal("update_record", {"record_id": 11, "data": "y"}),
        Proposal("update_record", {"record_id": 10, "data": "y"}),
    ],
    "fail an uncorrectable authorization violation": [
        Proposal("delete_record", {"record_id": 10}),
    ],
    "block a referential violation then recover": [
        Proposal("update_record", {"record_id": 999, "data": "z"}),
        Proposal("update_record", {"record_id": 10, "data": "z"}),
    ],
}


def demo_proposer() -> ScriptedProposer:
    return ScriptedProposer(SCRIPT)


def demo_tasks() -> list[str]:
    return list(SCRIPT.keys())