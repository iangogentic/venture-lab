"""Provider-neutral chat message types shared by every LLM call."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class Role(StrEnum):
    """Who authored a chat message."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ChatMessage(BaseModel):
    """One turn in a conversation, kept deliberately narrower than the SDK's union."""

    # Forbid extras: a typo in a hand-built message should fail here, not silently
    # travel to the provider.
    model_config = ConfigDict(extra="forbid")

    role: Role
    content: str


def system(content: str) -> ChatMessage:
    """Build a system message."""
    return ChatMessage(role=Role.SYSTEM, content=content)


def user(content: str) -> ChatMessage:
    """Build a user message."""
    return ChatMessage(role=Role.USER, content=content)


def assistant(content: str) -> ChatMessage:
    """Build an assistant message."""
    return ChatMessage(role=Role.ASSISTANT, content=content)


__all__ = ["ChatMessage", "Role", "assistant", "system", "user"]
