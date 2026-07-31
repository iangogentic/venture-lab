"""LLM access: one provider-agnostic service, three adapters, one gateway.

The rest of the application imports :class:`LLM` (or :func:`get_llm`) and nothing else
from here that touches a vendor. Claude, GPT and Gemini are all reached through
OpenRouter; the ``openrouter`` package is imported only inside this subpackage, so a
native per-provider SDK can replace an adapter later without a single call-site change.
"""

from app.llm.adapters import ClaudeAdapter, GeminiAdapter, GptAdapter
from app.llm.catalog import ModelCatalog, ModelInfo
from app.llm.client import build_client, get_client
from app.llm.messages import ChatMessage, Role, assistant, system, user
from app.llm.provider import (
    ADAPTERS,
    GenerationRequest,
    GenerationResult,
    Provider,
    ProviderAdapter,
    TokenUsage,
    available,
    get_adapter,
)
from app.llm.roles import Capability, ModelTier
from app.llm.routing import (
    ModelRouter,
    ResolvedRoute,
    Route,
    RoutingSort,
    get_router,
    provider_for_model,
)
from app.llm.service import LLM, get_llm

__all__ = [
    "ADAPTERS",
    "LLM",
    "Capability",
    "ChatMessage",
    "ClaudeAdapter",
    "GeminiAdapter",
    "GenerationRequest",
    "GenerationResult",
    "GptAdapter",
    "ModelCatalog",
    "ModelInfo",
    "ModelRouter",
    "ModelTier",
    "Provider",
    "ProviderAdapter",
    "ResolvedRoute",
    "Role",
    "Route",
    "RoutingSort",
    "TokenUsage",
    "assistant",
    "available",
    "build_client",
    "get_adapter",
    "get_client",
    "get_llm",
    "get_router",
    "provider_for_model",
    "system",
    "user",
]
