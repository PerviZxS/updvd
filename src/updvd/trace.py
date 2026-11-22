# Append-only, hash-chained decision log.
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

# prev_hash of the first entry.
GENESIS = "0" * 64


# One logged decision.
@dataclass(frozen=True)
class Entry:
    seq: int
    tool: str
    args: dict[str, Any]
    verdict: str
    detail: str
    committed: bool
    prev_hash: str

    # SHA-256 over canonical JSON (sorted keys, no whitespace) so the hash is deterministic.
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


# Each entry stores the previous entry's hash, so editing an earlier entry breaks the chain.
# This detects tampering but does not prevent it, and the newest entry is only covered once its
# hash is stored somewhere else.
class Trace:
    def __init__(self) -> None:
        self._entries: list[Entry] = []

    def append(self, tool: str, args: dict[str, Any], verdict: str, detail: str, committed: bool) -> Entry:
        prev_hash = self._entries[-1].digest() if self._entries else GENESIS
        entry = Entry(len(self._entries), tool, args, verdict, detail, committed, prev_hash)
        self._entries.append(entry)
        return entry

    # True if every entry's prev_hash matches the recomputed hash of the entry before it.
    def verify(self) -> bool:
        prev_hash = GENESIS
        for entry in self._entries:
            if entry.prev_hash != prev_hash:
                return False
            prev_hash = entry.digest()
        return True

    # JSON-ready entries, each with its own hash included.
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
