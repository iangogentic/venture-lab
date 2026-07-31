"""Provider adapters, one per model family, all speaking to the OpenRouter gateway."""

# Importing the concrete modules is what populates ADAPTERS, so the registry is complete
# for anyone who merely imports this package (e.g. `LLM(Provider.GEMINI)`).
from app.llm.adapters import claude, gemini, gpt  # noqa: F401
from app.llm.adapters.claude import ClaudeAdapter
from app.llm.adapters.gemini import GeminiAdapter
from app.llm.adapters.gpt import GptAdapter

__all__ = ["ClaudeAdapter", "GeminiAdapter", "GptAdapter"]
