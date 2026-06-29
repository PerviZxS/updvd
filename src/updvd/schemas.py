from __future__ import annotations

from pydantic import BaseModel, ConfigDict, StrictInt, StrictStr, field_validator


# Argument shapes for each tool. Strict types reject a string where an int is
# expected, and extra="forbid" rejects any unexpected field.
class ToolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CreateArgs(ToolArgs):
    data: StrictStr
    owner_id: StrictInt

    @field_validator("data")
    @classmethod
    def data_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("data must not be blank")
        return value


class UpdateArgs(ToolArgs):
    record_id: StrictInt
    data: StrictStr

    @field_validator("data")
    @classmethod
    def data_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("data must not be blank")
        return value


class DeleteArgs(ToolArgs):
    record_id: StrictInt


SCHEMAS: dict[str, type[ToolArgs]] = {
    "create_record": CreateArgs,
    "update_record": UpdateArgs,
    "delete_record": DeleteArgs,
}