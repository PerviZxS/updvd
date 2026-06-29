from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

GENESIS = "0" * 64


@dataclass(frozen=True)
class Entry:
    seq: int
    tool: str
    args: dict[str, Any]
    verdict: str
    detail: str
    committed: bool
    prev_hash: str

    def digest(self) -> str:
        payload = json.dumps(
            {
                "seq": self.seq,
                "tool": self.tool,
                "args": self.args,
                "verdict": self.verdict,
                "detail": self.detail,
                "committed": self.committed,
                "prev_hash": self.prev_hash,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()


# A hash-chained log of every decision. Each entry stores the hash of the one
# before it, so changing any past entry breaks the chain and verify() returns
# False. This makes the log tamper-evident, not tamper-proof.
class Trace:
    def __init__(self) -> None:
        self._entries: list[Entry] = []

    def append(self, tool: str, args: dict[str, Any], verdict: str, detail: str, committed: bool) -> Entry:
        prev_hash = self._entries[-1].digest() if self._entries else GENESIS
        entry = Entry(len(self._entries), tool, args, verdict, detail, committed, prev_hash)
        self._entries.append(entry)
        return entry

    def verify(self) -> bool:
        # Walk the chain and confirm each entry points at the real previous hash.
        prev_hash = GENESIS
        for entry in self._entries:
            if entry.prev_hash != prev_hash:
                return False
            prev_hash = entry.digest()
        return True

    def to_records(self) -> list[dict[str, Any]]:
        return [
            {
                "seq": e.seq,
                "tool": e.tool,
                "args": e.args,
                "verdict": e.verdict,
                "detail": e.detail,
                "committed": e.committed,
                "prev_hash": e.prev_hash,
                "hash": e.digest(),
            }
            for e in self._entries
        ]

    @property
    def entries(self) -> list[Entry]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)