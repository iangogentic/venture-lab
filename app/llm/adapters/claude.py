"""Claude adapter: Anthropic models reached through the OpenRouter gateway.

The adapter exists so nothing above ``app.llm`` knows which vendor answered. Replacing
the gateway with the native Anthropic SDK later is a change to this file alone.
"""

from typing import ClassVar

from openrouter.components import ResponseFormat

from app.llm.provider import (
    GenerationRequest,
    GenerationResult,
    Provider,
    ProviderAdapter,
    register,
)


@register
class ClaudeAdapter(ProviderAdapter):
    """Anthropic Claude models.

    Quirk compensated for: Claude returns extended-thinking output in a separate
    ``reasoning`` field rather than in ``content``. The shared normaliser reads only
    ``content``, so reasoning never leaks into a caller's markdown or parsed JSON.
    """

    provider: ClassVar[Provider] = Provider.CLAUDE
    default_model: ClassVar[str] = "anthropic/claude-sonnet-5"

    def generate(
        self,
        request: GenerationRequest,
        *,
        response_format: ResponseFormat | None = None,
    ) -> GenerationResult:
        """Run ``request`` against Claude. No provider-specific massaging is needed."""
        return self._send(request, response_format=response_format)


__all__ = ["ClaudeAdapter"]
