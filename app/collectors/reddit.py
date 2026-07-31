"""Collector that searches Reddit self-posts through the application-only OAuth API.

Reddit is the one source in this batch that genuinely cannot be read anonymously.
As of 2026-07 `https://www.reddit.com/search.json` answers 403 to an unauthenticated
request however descriptive the User-Agent, and `https://oauth.reddit.com/search`
answers 403 to a bad bearer token (both verified against the live service). There is
therefore no anonymous fallback worth attempting: it would fail, and a silently empty
result is worse than a clear "configure me" message. So this collector declares
`requires_credentials` and reports itself unavailable when the client id or secret is
missing, letting a run skip Reddit with a reason instead of dying on a source it was
never configured to read.

Only self-posts become items. A link post has no body, and a headline on its own is
not something an excerpt can be checked against, so link-only hits are dropped rather
than turned into unquotable evidence.
"""

import re
import time
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, ClassVar, Final

import httpx
from pydantic import Field, ValidationError, field_validator

from app.collectors.base import Collector, CollectorConfig, SourceItem, register
from app.config import get_settings
from app.utils.errors import CollectorError, RateLimitedError
from app.utils.logging import get_logger

logger = get_logger(__name__)

USER_AGENT: Final[str] = "opportunity-engine/0.1 (+https://github.com/)"
"""Mandatory, not decorative: Reddit throttles or blocks generic User-Agents outright."""

TOKEN_URL: Final[str] = "https://www.reddit.com/api/v1/access_token"
"""Tokens are minted on www; every authenticated read happens on the oauth host."""

API_BASE: Final[str] = "https://oauth.reddit.com"
WEB_BASE: Final[str] = "https://www.reddit.com"

MAX_PAGE_SIZE: Final[int] = 100
"""Reddit caps a listing page at 100 however large `limit` is."""

MIN_PAGE_SIZE: Final[int] = 10

MAX_PAGES: Final[int] = 5
"""Bound on pagination. Link-only posts are dropped after the fact, so one page often
under-fills the requested count — but an unbounded walk would spend the rate limit on
a niche query that is never going to fill it."""

TOKEN_EXPIRY_MARGIN: Final[float] = 60.0
"""Re-authenticate a minute early so a request never races its token's expiry."""

DEFAULT_TOKEN_LIFETIME: Final[float] = 3600.0

MAX_TEXT_CHARS: Final[int] = 40_000
"""Reddit's own self-post ceiling, so this bound effectively never bites; it exists so
one malformed reply cannot blow the downstream context budget. Truncation cuts a
*prefix*, never a summary, so an excerpt taken from it is still a literal substring."""

SUBREDDIT_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9_]{2,21}$")
"""Reddit's own name grammar. Matching it also keeps arbitrary text out of the URL path."""

TOMBSTONES: Final[frozenset[str]] = frozenset({"[removed]", "[deleted]"})
"""What Reddit substitutes for a removed body or a gone account. Not evidence."""


class RedditSort(StrEnum):
    """How Reddit orders search results."""

    RELEVANCE = "relevance"
    HOT = "hot"
    TOP = "top"
    NEW = "new"
    COMMENTS = "comments"


class RedditTimeRange(StrEnum):
    """Age window applied to search results."""

    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"
    ALL = "all"


class RedditCollectorConfig(CollectorConfig):
    """Reddit-specific settings on top of the shared collector knobs."""

    subreddits: list[str] = Field(
        default_factory=list,
        description="Restrict the search to these subreddits. The useful evidence "
        "usually lives in a handful of communities rather than in global search.",
    )
    sort: RedditSort = RedditSort.RELEVANCE
    time_range: RedditTimeRange = RedditTimeRange.ALL

    @field_validator("subreddits", mode="before")
    @classmethod
    def _accept_single_name(cls, value: object) -> object:
        # One subreddit is the common case; accept the bare string for it rather than
        # letting `subreddits: "saas"` quietly validate as a list of four characters.
        return [value] if isinstance(value, str) else value


@register
class RedditCollector(Collector):
    """Search Reddit self-posts, authenticated as an application."""

    name: ClassVar[str] = "reddit"
    description: ClassVar[str] = "Reddit self-posts, via the application-only OAuth API."
    requires_credentials: ClassVar[bool] = True

    config: RedditCollectorConfig
    """Narrowed from the base's `CollectorConfig`; see `_typed_config`."""

    def __init__(self, config: CollectorConfig | None = None) -> None:
        super().__init__(self._typed_config(config))
        # Held in memory only, and never written to disk: this is a credential.
        self._token: str | None = None
        self._token_expires_at: float = 0.0

    @staticmethod
    def _typed_config(config: CollectorConfig | None) -> RedditCollectorConfig:
        """Re-read a plain `CollectorConfig`'s extras as typed Reddit settings.

        The base contract hands over a `CollectorConfig` whose extras are untyped.
        Validating once, here, is what lets the rest of this module say
        `self.config.subreddits` instead of guessing with `getattr`.
        """
        if config is None:
            return RedditCollectorConfig()
        if isinstance(config, RedditCollectorConfig):
            return config
        try:
            return RedditCollectorConfig.model_validate(config.model_dump())
        except ValidationError as exc:
            raise CollectorError(f"Invalid reddit collector configuration: {exc}") from exc

    def available(self) -> bool:
        """Whether credentials are configured. Never raises.

        One unconfigured source must not take a run down with it, so unreadable
        settings report "unavailable" here instead of exploding mid-collection.
        """
        if not self.config.enabled:
            return False
        try:
            return self._credentials() is not None
        except CollectorError as exc:
            logger.debug("reddit unavailable: %s", exc)
            return False

    def search(self, query: str, *, limit: int | None = None) -> list[SourceItem]:
        """Search Reddit for self-posts matching `query`.

        Returns at most `limit or self.config.limit` items, in the order Reddit ranked
        them. Link-only posts are skipped, so fewer items than asked for is normal and
        an empty list is a legitimate answer.

        Raises:
            CollectorError: If the collector is unconfigured, Reddit refuses the request
                (rejected credentials, rate limit), the transport fails, or a reply
                cannot be read as a listing. Never for an empty result.
        """
        wanted = self.config.limit if limit is None else limit
        if wanted <= 0:
            return []
        text = query.strip()
        if not text:
            # Reddit answers 400 to an empty `q`. Nothing was asked, so nothing was found.
            return []

        credentials = self._credentials()
        if credentials is None:
            raise CollectorError(
                "The reddit collector is not configured: set REDDIT_CLIENT_ID and "
                "REDDIT_CLIENT_SECRET (create an app at https://www.reddit.com/prefs/apps), "
                "or disable the reddit collector for this run.",
            )

        url, restrict_sr = self._search_target()
        found: list[SourceItem] = []
        seen: set[str] = set()
        after: str | None = None

        with httpx.Client(
            timeout=self.config.timeout,
            headers={"User-Agent": USER_AGENT},
        ) as client:
            for _ in range(MAX_PAGES):
                remaining = wanted - len(found)
                # Ask for more than is needed: link-only posts are dropped afterwards,
                # so a page of hits routinely yields only a handful of items.
                page_size = min(MAX_PAGE_SIZE, max(remaining * 2, MIN_PAGE_SIZE))
                payload = self._get_listing(
                    client,
                    url,
                    credentials=credentials,
                    query=text,
                    page_size=page_size,
                    restrict_sr=restrict_sr,
                    after=after,
                )
                items, after = self._parse_listing(payload)
                for item in items:
                    # Listings can repeat a post across pages when the ranking shifts.
                    if item.external_id in seen:
                        continue
                    seen.add(item.external_id)
                    found.append(item)
                    if len(found) >= wanted:
                        break
                if len(found) >= wanted or not after:
                    break

        return found[:wanted]

    # --- credentials and tokens -------------------------------------------------

    def _credentials(self) -> tuple[str, str] | None:
        """The configured client id and secret, or None if either is missing.

        Raises:
            CollectorError: If settings themselves cannot be loaded.
        """
        try:
            settings = get_settings()
        except (OSError, ValidationError) as exc:
            raise CollectorError(f"Could not load settings for reddit: {exc}") from exc
        client_id = settings.reddit_client_id
        client_secret = settings.reddit_client_secret
        if not client_id or client_secret is None:
            return None
        secret = client_secret.get_secret_value()
        if not secret:
            return None
        return client_id, secret

    def _ensure_token(self, client: httpx.Client, credentials: tuple[str, str]) -> str:
        """Return a live application-only access token, minting one if needed.

        The token is cached on the instance and reused until shortly before it expires;
        re-authenticating per search would spend the rate limit on handshakes instead
        of on evidence.
        """
        cached = self._token
        if cached is not None and time.monotonic() < self._token_expires_at:
            return cached

        client_id, client_secret = credentials
        try:
            response = client.post(
                TOKEN_URL,
                data={"grant_type": "client_credentials"},
                auth=(client_id, client_secret),
            )
        except httpx.HTTPError as exc:
            raise CollectorError(f"Could not reach Reddit to authenticate: {exc}") from exc

        self._check_status(response, what="authenticating with Reddit")

        try:
            payload: object = response.json()
        except ValueError as exc:
            raise CollectorError("Reddit's token reply was not JSON.") from exc
        if not isinstance(payload, dict):
            raise CollectorError("Reddit's token reply was not a JSON object.")

        token = payload.get("access_token")
        if not isinstance(token, str) or not token:
            # Reddit reports some failures (bad grant type, wrong app type) with HTTP 200
            # and an `error` key, so a missing token is its own case, not a status code.
            reason = payload.get("error", "no access_token in the reply")
            raise CollectorError(
                f"Reddit issued no access token ({reason}). The client-credentials flow "
                "needs an app of type 'script' or 'web app' at "
                "https://www.reddit.com/prefs/apps.",
            )

        expires_in = payload.get("expires_in")
        lifetime = DEFAULT_TOKEN_LIFETIME
        if isinstance(expires_in, int | float):
            lifetime = float(expires_in)
        self._token = token
        self._token_expires_at = time.monotonic() + max(lifetime - TOKEN_EXPIRY_MARGIN, 0.0)
        return token

    def _forget_token(self) -> None:
        self._token = None
        self._token_expires_at = 0.0

    # --- requests ---------------------------------------------------------------

    def _search_target(self) -> tuple[str, bool]:
        """The search URL and whether the results must be restricted to it."""
        names = [self._clean_subreddit(raw) for raw in self.config.subreddits]
        valid = [name for name in names if name is not None]
        if not valid:
            return f"{API_BASE}/search", False
        # `r/a+b+c/search` is how Reddit expresses a multi-subreddit listing, which keeps
        # the whole search to one request instead of one per community.
        return f"{API_BASE}/r/{'+'.join(valid)}/search", True

    @staticmethod
    def _clean_subreddit(raw: object) -> str | None:
        """Normalise one configured subreddit name, or None if it is unusable."""
        if not isinstance(raw, str):
            return None
        name = raw.strip().removeprefix("/").removeprefix("r/").strip("/")
        if not SUBREDDIT_RE.match(name):
            # One bad name must not cost the search the other communities.
            logger.debug("reddit: ignoring unusable subreddit name %r", raw)
            return None
        return name

    def _get_listing(
        self,
        client: httpx.Client,
        url: str,
        *,
        credentials: tuple[str, str],
        query: str,
        page_size: int,
        restrict_sr: bool,
        after: str | None,
        retry_on_refusal: bool = True,
    ) -> dict[str, Any]:
        """Fetch one page of search results as a decoded listing."""
        token = self._ensure_token(client, credentials)
        params: dict[str, str] = {
            "q": query,
            "limit": str(page_size),
            "sort": str(self.config.sort),
            "t": str(self.config.time_range),
            "type": "link",
            # Without raw_json Reddit HTML-escapes &, < and > in the body, and the text
            # would no longer be literally what the author wrote.
            "raw_json": "1",
        }
        if restrict_sr:
            params["restrict_sr"] = "1"
        if after:
            params["after"] = after

        try:
            response = client.get(url, params=params, headers={"Authorization": f"Bearer {token}"})
        except httpx.HTTPError as exc:
            raise CollectorError(f"Reddit search request failed: {exc}") from exc

        if response.status_code in (401, 403) and retry_on_refusal:
            # A cached token can be revoked before its stated expiry; re-authenticate
            # once before concluding the credentials themselves are bad.
            self._forget_token()
            return self._get_listing(
                client,
                url,
                credentials=credentials,
                query=query,
                page_size=page_size,
                restrict_sr=restrict_sr,
                after=after,
                retry_on_refusal=False,
            )

        self._check_status(response, what="searching Reddit")

        try:
            payload: object = response.json()
        except ValueError as exc:
            raise CollectorError("Reddit's search reply was not JSON.") from exc
        if not isinstance(payload, dict):
            raise CollectorError("Reddit's search reply was not a JSON object.")
        return payload

    @staticmethod
    def _check_status(response: httpx.Response, *, what: str) -> None:
        """Turn a refused response into a CollectorError that says what to do next."""
        status = response.status_code
        # 403, not 401, is what oauth.reddit.com actually answers to a bad bearer token
        # (verified live), so both statuses mean "these credentials were rejected".
        if status in (401, 403):
            raise CollectorError(
                f"Reddit rejected the credentials while {what} (HTTP {status}). Check "
                "REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET against the app at "
                "https://www.reddit.com/prefs/apps — the secret changes whenever the app "
                "is edited, and only 'script' and 'web app' types may use the "
                "client-credentials flow.",
            )
        if status == 429:
            retry_after = response.headers.get("retry-after")
            advice = f" Retry in {retry_after}s." if retry_after else " Wait a minute, then retry."
            raise RateLimitedError(
                f"Reddit rate limit hit while {what} (HTTP 429).{advice} Lower the "
                "collector's `limit`, or narrow `subreddits` so fewer pages are needed.",
            )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise CollectorError(f"Reddit returned HTTP {status} while {what}.") from exc

    # --- parsing ----------------------------------------------------------------

    @classmethod
    def _parse_listing(cls, payload: dict[str, Any]) -> tuple[list[SourceItem], str | None]:
        """Split a listing into its usable items and the cursor for the next page."""
        data = payload.get("data")
        if not isinstance(data, dict):
            raise CollectorError("Reddit's reply was not a listing: no `data` object.")

        # A listing with no children is an empty result, which is a finding, not a failure.
        children = data.get("children")
        rows: list[Any] = children if isinstance(children, list) else []

        items: list[SourceItem] = []
        for child in rows:
            item = cls._to_item(child)
            if item is not None:
                items.append(item)

        after = data.get("after")
        return items, after if isinstance(after, str) and after else None

    @classmethod
    def _to_item(cls, child: object) -> SourceItem | None:
        """Convert one listing child to a `SourceItem`, or None if it is unusable.

        Returns None rather than raising: one malformed record must not cost the run
        the other twenty-four.
        """
        if not isinstance(child, dict):
            return None
        if child.get("kind") != "t3":
            # t3 is a post. Anything else in a `type=link` search is not what was asked for.
            return None
        data = child.get("data")
        if not isinstance(data, dict):
            return None

        post_id = data.get("id")
        if not isinstance(post_id, str) or not post_id:
            return None

        body = data.get("selftext")
        if not isinstance(body, str):
            return None
        body = body.strip()
        # A link post has an empty body, and a headline is not quotable evidence: there
        # would be nothing for the excerpt check downstream to verify an excerpt against.
        if not body or body.lower() in TOMBSTONES:
            logger.debug("reddit: skipping post %s with no quotable body", post_id)
            return None
        # Prefix cut only — what is kept stays verbatim, so excerpts remain literal
        # substrings of it.
        body = body[:MAX_TEXT_CHARS]

        # Reddit's canonical id is the fullname (`t3_<id>`): it is what the API takes
        # back for lookups, and it stays unambiguous across kinds.
        fullname = data.get("name")
        external_id = fullname if isinstance(fullname, str) and fullname else f"t3_{post_id}"

        # `score`, `num_comments` and `subreddit` deliberately go nowhere: `SourceItem`
        # has no slot for them, and splicing them into `text` would break the verbatim
        # guarantee the excerpt check depends on.
        try:
            return SourceItem(
                collector=cls.name,
                external_id=external_id,
                text=body,
                title=cls._as_text(data.get("title")),
                url=cls._permalink(data, post_id),
                author=cls._author(data.get("author")),
                published_at=cls._published_at(data.get("created_utc")),
            )
        except ValueError as exc:
            logger.debug("reddit: skipping unusable post %s: %s", post_id, exc)
            return None

    @staticmethod
    def _as_text(value: object) -> str | None:
        return value if isinstance(value, str) and value.strip() else None

    @staticmethod
    def _author(value: object) -> str | None:
        """The post's author, or None when the account is gone."""
        if not isinstance(value, str):
            return None
        name = value.strip()
        return None if not name or name.lower() in TOMBSTONES else name

    @staticmethod
    def _permalink(data: dict[str, Any], post_id: str) -> str:
        """Where a human can verify the post."""
        permalink = data.get("permalink")
        if isinstance(permalink, str) and permalink.startswith("/"):
            return f"{WEB_BASE}{permalink}"
        # Short-link fallback: a citation that still resolves beats no citation at all.
        return f"https://redd.it/{post_id}"

    @staticmethod
    def _published_at(value: object) -> datetime | None:
        """`created_utc` as an aware UTC datetime; Evidence rejects naive ones downstream."""
        if not isinstance(value, int | float):
            return None
        if isinstance(value, bool):
            # `bool` is an `int`; a boolean timestamp is a malformed record, not 1970.
            return None
        try:
            return datetime.fromtimestamp(float(value), tz=UTC)
        except (OSError, OverflowError, ValueError):
            return None


__all__ = ["RedditCollector", "RedditCollectorConfig", "RedditSort", "RedditTimeRange"]
