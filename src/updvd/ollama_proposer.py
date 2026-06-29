from __future__ import annotations

import json
from typing import Any

from .proposer import Proposal, Proposer

SYSTEM_PROMPT = """You control a record store by proposing one tool call at a time.

Available tools:
- create_record(data: string, owner_id: integer)
- update_record(record_id: integer, data: string)
- delete_record(record_id: integer)

Respond with a single JSON object and nothing else:
{"tool": "<tool name>", "args": {<arguments>}}

If your previous proposal was rejected, the rejection reason is provided.
Correct the proposal and respond again in the same JSON format."""


class OllamaProposer(Proposer):
    def __init__(
        self,
        model: str = "qwen3",
        host: str | None = None,
        temperature: float = 0.0,
        seed: int = 0,
    ) -> None:
        try:
            from ollama import Client
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "the ollama client is required for the live model; install it with "
                "pip install -e \".[llm]\" and ensure the ollama server is running"
            ) from exc

        self._client = Client(host=host) if host else Client()
        self._model = model
        self._options = {"temperature": temperature, "seed": seed, "keep_alive": "10m"}

    def propose(self, task: str, feedback: str | None = None) -> Proposal:
        content = f"Task: {task}"
        if feedback:
            content += f"\nPrevious proposal rejected: {feedback}"

        response = self._client.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            format="json",
            think=False,
            options=self._options,
        )
        payload = self._parse(response["message"]["content"])
        return Proposal(tool=payload.get("tool", ""), args=payload.get("args", {}))

    @staticmethod
    def _parse(content: str) -> dict[str, Any]:
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            return {"tool": "", "args": {}}
        return payload if isinstance(payload, dict) else {"tool": "", "args": {}}