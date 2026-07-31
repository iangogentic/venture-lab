"""Construction of the OpenRouter SDK client from application settings."""

from functools import lru_cache

from openrouter import OpenRouter

from app.config import get_settings
from app.utils.errors import ConfigurationError

_MISSING_API_KEY = "OPENROUTER_API_KEY is not set. Copy .env.example to .env and fill it in."


def build_client() -> OpenRouter:
    """Build a fresh OpenRouter client from the current settings.

    Raises:
        ConfigurationError: If no API key is configured.
    """
    settings = get_settings()
    if settings.openrouter_api_key is None:
        raise ConfigurationError(_MISSING_API_KEY)
    return OpenRouter(
        api_key=settings.openrouter_api_key.get_secret_value(),
        http_referer=settings.openrouter_http_referer,
        x_open_router_title=settings.openrouter_app_title,
        timeout_ms=settings.openrouter_timeout_ms,
    )


@lru_cache(maxsize=1)
def get_client() -> OpenRouter:
    """Return the process-wide client, building it on first use.

    Cached because the client owns an HTTP connection pool. Tests that change the
    environment must call ``get_client.cache_clear()`` alongside
    ``get_settings.cache_clear()``.
    """
    return build_client()


__all__ = ["build_client", "get_client"]
