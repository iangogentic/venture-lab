"""Typed application settings, loaded from the environment and `.env`."""

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, SecretStr
from pydantic.fields import FieldInfo
from pydantic_settings import (
    BaseSettings,
    DotEnvSettingsSource,
    EnvSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

type LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
type LLMProvider = Literal["claude", "gpt", "gemini"]
type LLMTransport = Literal["auto", "openai", "openrouter"]
type LLMSort = Literal["price", "throughput", "latency"]
type LLMCacheTTL = Literal["5m", "1h"]


class _BlankIsUnset:
    """Mixin making an empty environment variable mean "not set".

    Complex fields — the `dict` and `list` settings — are JSON-decoded by
    pydantic-settings, and an empty string is not valid JSON. Without this, a
    `.env` containing a commented-out-by-emptying `LLM_ROUTES=` fails to load at
    all, with an error naming a field the user never meant to configure. Since
    every such setting already has a working default, treating blank as absent is
    both what a reader expects and the only forgiving reading.
    """

    def prepare_field_value(
        self,
        field_name: str,
        field: FieldInfo,
        value: Any,
        value_is_complex: bool,
    ) -> Any:
        if isinstance(value, str) and not value.strip():
            return None
        return super().prepare_field_value(  # type: ignore[misc]
            field_name, field, value, value_is_complex
        )


class _EnvSource(_BlankIsUnset, EnvSettingsSource):
    """Environment variables, with blanks treated as absent."""


class _DotEnvSource(_BlankIsUnset, DotEnvSettingsSource):
    """`.env` values, with blanks treated as absent."""


class Settings(BaseSettings):
    """Runtime configuration. See `.env.example` for the supported variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- LLM transports ---
    llm_transport: LLMTransport = Field(
        default="auto",
        description="Transport for GPT calls. Auto uses native OpenAI when "
        "OPENAI_API_KEY is set, and OpenRouter otherwise.",
    )
    openai_api_key: SecretStr | None = Field(
        default=None,
        description="OpenAI API key for native GPT calls.",
    )
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="Base URL for the native OpenAI Responses API.",
    )
    openai_timeout_ms: int = Field(
        default=120_000,
        ge=1,
        description="Per-request timeout for native OpenAI calls, in milliseconds.",
    )
    openrouter_api_key: SecretStr | None = Field(
        default=None,
        description="OpenRouter API key. Required for LLM calls routed through OpenRouter.",
    )
    openrouter_http_referer: str | None = Field(
        default=None,
        description="Optional HTTP-Referer attribution header.",
    )
    openrouter_app_title: str = Field(
        default="Opportunity Engine",
        description="Optional X-Title attribution header.",
    )
    openrouter_timeout_ms: int = Field(
        default=120_000,
        ge=1,
        description="Per-request timeout for OpenRouter calls, in milliseconds.",
    )

    # --- LLM ---
    # Typed as a Literal rather than `app.llm.Provider` on purpose: the LLM layer imports
    # settings, so importing its enum back into config would be a circular import. The LLM
    # layer coerces this with `Provider(settings.llm_provider)`.
    llm_provider: LLMProvider = Field(
        default="claude",
        description="Which provider adapter to use by default.",
    )
    llm_model: str | None = Field(
        default=None,
        description="Model slug override. None means 'use the adapter's default model'.",
    )
    llm_temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. None leaves it to the provider.",
    )
    llm_max_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Cap on generated tokens. None leaves it to the provider.",
    )

    # --- Model routing (by logical role, never a slug) ---
    llm_default_capability: str = Field(
        default="synthesis",
        description="Capability used by any stage without one of its own.",
    )
    llm_capabilities: dict[str, str] = Field(
        default_factory=dict,
        description="Capability -> model tier. THE mapping to change when a better "
        'model appears: \'{"synthesis": "claude-opus"}\' moves every synthesis stage '
        "at once. Tiers: gemini-flash, gemini-pro, claude-sonnet, claude-opus, gpt.",
    )
    llm_stage_capabilities: dict[str, str] = Field(
        default_factory=dict,
        description="Stage -> capability override. Rarely needed: what a stage needs "
        "is a fact about the work, not about which model is currently best.",
    )
    llm_temperatures: dict[str, float] = Field(
        default_factory=dict,
        description="Stage -> temperature override. Unset stages use the recommended table.",
    )
    llm_max_output_tokens: dict[str, int] = Field(
        default_factory=dict,
        description="Stage -> output cap override. Unset stages use a per-stage default "
        "sized to what they emit, capped by MAX_OUTPUT_TOKENS.",
    )
    llm_fallback_tiers: list[str] = Field(
        default_factory=list,
        description="Model tiers the gateway falls back to, in order, if the primary is down.",
    )
    llm_retry_attempts: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Retries after a failed call, on top of the gateway's own failover.",
    )
    llm_sort: LLMSort | None = Field(
        default=None,
        description="How the gateway picks among providers serving one model.",
    )
    llm_use_catalogue: bool = Field(
        default=True,
        description="Resolve roles against OpenRouter's live Models API. Off, the "
        "pinned fallback slugs are used and no network call is made.",
    )
    model_cache_path: Path = Field(
        default=Path(".cache/openrouter-models.json"),
        description="Where the fetched model catalogue is cached.",
    )

    # --- Context budget ---
    max_input_tokens: int = Field(
        default=200_000,
        ge=1,
        description="Ceiling on what may be sent in one call. Enforced before the "
        "request leaves, so an over-large context fails locally rather than at the gateway.",
    )
    max_output_tokens: int = Field(
        default=12_000,
        ge=1,
        description="Ceiling on generated tokens per call.",
    )

    # --- Prompt caching ---
    llm_cache_prompts: bool = Field(
        default=True,
        description="Ask the provider to cache the static prompt prefix where supported. "
        "The system and skill prompts barely change between calls.",
    )
    llm_cache_ttl: LLMCacheTTL = Field(
        default="5m",
        description="How long a cached prefix stays warm.",
    )

    # --- Telemetry ---
    telemetry_enabled: bool = Field(
        default=True,
        description="Append one JSONL record per model call. Cheap, and the only way to "
        "answer later which stage actually cost the money.",
    )
    telemetry_path: Path = Field(
        default=Path(".telemetry/calls.jsonl"),
        description="Where call records go. Relative paths resolve inside the workspace.",
    )

    # --- Collectors (retrieval) ---
    collectors: list[str] = Field(
        default_factory=list,
        description="Which collectors to run. Empty means every one that is available.",
    )
    collector_limit: int = Field(
        default=25,
        ge=1,
        le=200,
        description="Items each collector may return per query.",
    )
    collect_preview_chars: int = Field(
        default=2_000,
        ge=200,
        description="How much of each fetched item the selection prompt sees. A couple "
        "of paragraphs is enough to judge relevance and find a quotable line; sending "
        "whole issue threads exhausts the context window on the first stage.",
    )
    collect_context_fraction: float = Field(
        default=0.6,
        gt=0.0,
        le=0.95,
        description="Share of MAX_INPUT_TOKENS the candidate list may occupy, leaving "
        "room for the prompt and the reply.",
    )
    github_token: SecretStr | None = Field(
        default=None,
        description="Optional. GitHub search works without one; a token lifts the "
        "unauthenticated rate limit from ~10 requests a minute.",
    )
    reddit_client_id: str | None = Field(
        default=None,
        description="Reddit refuses anonymous search, so this is required to use it.",
    )
    reddit_client_secret: SecretStr | None = Field(default=None)
    rss_feeds: list[str] = Field(
        default_factory=list,
        description="Feed URLs for the rss collector. RSS has no search endpoint, so "
        "these are fetched and filtered client-side.",
    )
    corpus_paths: list[str] = Field(
        default_factory=list,
        description="Local directories the filesystem collector searches — interview "
        "notes, exported threads, anything with no API.",
    )
    searxng_url: str | None = Field(
        default=None,
        description="Base URL of a self-hosted SearXNG instance for the web collector. "
        "Unset, the collector reports itself unavailable rather than failing runs.",
    )
    tavily_api_key: SecretStr | None = Field(
        default=None,
        description="Key for the tavily collector. The free tier's 1,000 monthly "
        "credits cover roughly ten to thirty full pipeline runs.",
    )
    discourse_forums: list[str] = Field(
        default_factory=list,
        description="Base URLs of Discourse forums to search — most dev-tool and SaaS "
        "product forums run Discourse, and /search.json is its designed read interface.",
    )
    stackexchange_sites: list[str] = Field(
        default_factory=lambda: ["stackoverflow"],
        description="Stack Exchange sites the stack-exchange collector searches. The "
        "API allows keyless access, so the default works with no setup.",
    )
    stackexchange_key: str | None = Field(
        default=None,
        description="Optional Stack Exchange app key. Not a secret by their design; "
        "raises the daily request quota from 300 to 10,000.",
    )
    app_store_countries: list[str] = Field(
        default_factory=lambda: ["us"],
        description="Storefront countries the app-reviews collector reads. Each "
        "country exposes its own ~500 most-recent reviews per app.",
    )

    # --- Memory (semantic dedup and cross-run recall) ---
    memory_enabled: bool = Field(
        default=True,
        description="Embed kept evidence locally, for near-duplicate dropping and "
        "`op recall`. Costs no tokens and no network beyond a one-time ~30MB model "
        "download; off, dedup falls back to exact source ids only.",
    )
    embedding_model: str = Field(
        default="minishlab/potion-base-8M",
        description="Static embedding model for the local memory, fetched once from "
        "Hugging Face. Numpy-only inference — no torch, no GPU.",
    )
    dedup_threshold: float = Field(
        default=0.92,
        gt=0.5,
        le=1.0,
        description="Cosine similarity at which two candidates count as one complaint. "
        "Raise it toward 1.0 and only near-verbatim cross-posts are dropped; lower it "
        "and paraphrases start to merge — go far enough and distinct complaints about "
        "the same product collapse into one.",
    )

    # --- Storage ---
    database_url: str = Field(
        default="sqlite:///./workspace/engine.db",
        description="SQLAlchemy URL for the run ledger. SQLite by default.",
    )
    database_echo: bool = Field(
        default=False,
        description="Echo emitted SQL to stderr. Useful when debugging.",
    )

    # --- Workspace ---
    workspace_dir: Path = Field(
        default=Path("workspace"),
        description="Root of the artifact tree. One subdirectory per pipeline stage.",
    )

    # --- Runtime ---
    log_level: LogLevel = Field(default="INFO", description="Root log level.")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Swap in the sources that read a blank value as unset.

        Priority is unchanged from pydantic-settings' default: init arguments,
        then the real environment, then `.env`, then file secrets.
        """
        return (
            init_settings,
            _EnvSource(settings_cls),
            _DotEnvSource(settings_cls),
            file_secret_settings,
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Cached so `.env` is read once per process. Call `get_settings.cache_clear()`
    in tests that need to re-read the environment.
    """
    return Settings()


__all__ = [
    "LLMCacheTTL",
    "LLMProvider",
    "LLMSort",
    "LLMTransport",
    "LogLevel",
    "Settings",
    "get_settings",
]
