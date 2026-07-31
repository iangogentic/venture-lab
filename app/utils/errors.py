"""Exception hierarchy for the application."""


class OpportunityEngineError(Exception):
    """Base class for every error raised by this package."""


class ConfigurationError(OpportunityEngineError):
    """Required configuration is missing or invalid."""


class ArtifactError(OpportunityEngineError):
    """An artifact could not be read, written, or validated."""


class CollectorError(OpportunityEngineError):
    """A collector failed, or an unknown collector was requested."""


class RateLimitedError(CollectorError):
    """A source refused on quota or rate-limit grounds.

    Its own type because it says something about the *next* request, not just
    this one: a source that is out of quota will refuse the query after this one
    too. Left as a plain `CollectorError` it reads as "that query had bad luck",
    and the caller dutifully asks the same source seven more things — which is a
    minute of round trips spent collecting refusals, and eight identical walls of
    HTML in the log. One real run did exactly that.
    """


class PipelineError(OpportunityEngineError):
    """A pipeline stage failed, or an unknown stage was requested."""


class SkillError(OpportunityEngineError):
    """A skill failed, or an unknown skill was requested."""


class LLMError(OpportunityEngineError):
    """The model provider returned an error or an unusable response."""


class StorageError(OpportunityEngineError):
    """A ledger row could not be read or written as asked."""


class MemoryUnavailableError(OpportunityEngineError):
    """The local semantic memory cannot run here — no model, no vector extension.

    Deliberately its own type rather than a `ConfigurationError`: every caller is
    expected to catch it and carry on with memory off, never to fail the run, so
    it must be catchable without also swallowing genuine configuration bugs.
    """


__all__ = [
    "ArtifactError",
    "CollectorError",
    "ConfigurationError",
    "LLMError",
    "MemoryUnavailableError",
    "OpportunityEngineError",
    "PipelineError",
    "RateLimitedError",
    "SkillError",
    "StorageError",
]
