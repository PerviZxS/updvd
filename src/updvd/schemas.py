# Typed argument schemas for each tool, validated before any contract runs.
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, StrictInt, StrictStr, field_validator


# Base schema: extra="forbid" rejects unknown fields; Strict* types reject coercion ("10" is not 10).
class ToolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


# Arguments for create_record.
class CreateArgs(ToolArgs):
    data: StrictStr
    owner_id: StrictInt

    @field_validator("data")
    @classmethod
    def data_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("data must not be blank")
        return value


# Arguments for update_record.
class UpdateArgs(ToolArgs):
    record_id: StrictInt
    data: StrictStr

    @field_validator("data")
    @classmethod
    def data_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("data must not be blank")
        return value


# Arguments for delete_record.
class DeleteArgs(ToolArgs):
    record_id: StrictInt


# Tool name -> argument schema.
SCHEMAS: dict[str, type[ToolArgs]] = {
    "create_record": CreateArgs,
    "update_record": UpdateArgs,
    "delete_record": DeleteArgs,
}
