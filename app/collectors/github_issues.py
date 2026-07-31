"""Collector for GitHub issues, via the public issue-search API.

Issue trackers are the closest thing to a public record of software pain: someone
hit a problem, cared enough to write it down at length, and left a permalink that
anyone can check. That makes them the best-grounded source this pipeline has —
unlike a forum post, an issue names the project, the version, and usually the
reproduction, all in the reporter's own words.

Only the issue *body* becomes `SourceItem.text`. A title is a label the reporter
wrote to be scanned, not a statement they made, so an issue with an empty body is
skipped rather than quoted: the `collect-evidence` stage would have nothing to
verify an excerpt against.
"""

import re
from datetime import UTC, datetime
from typing import Any, ClassVar

import httpx

from app.collectors.base import Collector, SourceItem, register, short_body
from app.config import get_settings
from app.utils.errors import CollectorError, RateLimitedError

_API_URL = "https://api.github.com/search/issues"
_USER_AGENT = "opportunity-engine/0.1 (+https://github.com/)"

# Pinning the API version keeps the response shape stable; GitHub ships breaking
# search changes behind this header rather than silently.
_API_VERSION = "2022-11-28"

# The search API caps `per_page` at 100 and this collector's own limit at 200, so
# two pages always suffice; the third is slack for pages thinned by empty bodies.
# Paging is bounded on purpose: anonymous search allows only ~10 requests a minute.
_MAX_PER_PAGE = 100
_MAX_PAGES = 3

# No `sort` parameter: GitHub's default is best-match relevance, and every
# alternative measured worse. `sort=comments` in particular returns the most
# *replied-to* issues in the matched set, which for a broad query means bot status
# boards and bounty spam with thousands of comments — popular, not painful.
#
# Feature requests are the opposite of what this collector is for. Labels are
# *excluded* rather than required so that repos which label nothing still appear.
_EXCLUDED_LABELS = ("-label:enhancement", '-label:"feature request"')

# A body can run to tens of thousands of characters (generated reports, pasted
# logs). Cut generously and only at a line or sentence boundary — the excerpt check
# downstream is a substring test, so a verbatim prefix stays verifiable, but a
# fragment ending mid-sentence would be quoted back as one.
_MAX_BODY_CHARS = 20_000

_REPO_API_RE = re.compile(r"/repos/([^/]+/[^/]+)/?$")
_REPO_HTML_RE = re.compile(r"^https?://github\.com/([^/]+/[^/]+)/")


@register
class GitHubIssuesCollector(Collector):
    """Search public GitHub issues for reported bugs, breakage and complaints."""

    name: ClassVar[str] = "github-issues"
    description: ClassVar[str] = "Search public GitHub issues for reported bugs and complaints."

    # A token only lifts the rate limit from ~10 to 30 searches a minute; anonymous
    # search works, so this source must never be skipped for want of credentials.
    requires_credentials: ClassVar[bool] = False

    def search(self, query: str, *, limit: int | None = None) -> list[SourceItem]:
        """Search issues for `query` and return the bodies verbatim.

        Raises:
            CollectorError: On transport failure, a refused request (including the
                rate limit), or a response that cannot be understood.
        """
        wanted = limit if limit is not None else self.config.limit
        shaped = _shape_query(query)
        per_page = min(_MAX_PER_PAGE, wanted)

        found: list[SourceItem] = []
        seen: set[str] = set()
        with httpx.Client(timeout=self.config.timeout, headers=_headers()) as client:
            for page in range(1, _MAX_PAGES + 1):
                entries = _entries(self._fetch(client, shaped, per_page=per_page, page=page))
                for entry in entries:
                    if len(found) >= wanted:
                        break
                    item = self._to_item(entry)
                    # Dedup on the readable ref: paging is not a snapshot, so an
                    # issue can appear on two pages if the index shifts between them.
                    if item is None or item.external_id in seen:
                        continue
                    seen.add(item.external_id)
                    found.append(item)
                if len(found) >= wanted or len(entries) < per_page:
                    break
        return found

    def _fetch(
        self,
        client: httpx.Client,
        query: str,
        *,
        per_page: int,
        page: int,
    ) -> dict[str, Any]:
        """Run one search request and return the decoded payload."""
        params: dict[str, str | int] = {"q": query, "per_page": per_page, "page": page}
        try:
            response = client.get(_API_URL, params=params)
        except httpx.HTTPError as exc:
            raise CollectorError(f"GitHub issue search could not be reached: {exc}") from exc

        if _is_rate_limited(response):
            raise RateLimitedError(_rate_limit_message(response))
        if response.is_error:
            raise CollectorError(
                f"GitHub issue search refused the request "
                f"(HTTP {response.status_code}): {_api_message(response)}",
            )

        try:
            payload: Any = response.json()
        except ValueError as exc:
            raise CollectorError("GitHub issue search returned a body that is not JSON") from exc
        if not isinstance(payload, dict):
            raise CollectorError(
                f"GitHub issue search returned {type(payload).__name__}, expected an object",
            )
        return payload

    def _to_item(self, entry: Any) -> SourceItem | None:
        """Convert one search hit, or return None if it is not usable evidence."""
        if not isinstance(entry, dict):
            return None

        body = entry.get("body")
        # No body, no quote: the title alone is a label, not a report.
        if not isinstance(body, str) or not body.strip():
            return None

        external_id = _issue_ref(entry)
        if external_id is None:
            return None

        try:
            return SourceItem(
                collector=self.name,
                external_id=external_id,
                text=_bounded(body),
                title=_text_field(entry, "title"),
                url=_text_field(entry, "html_url"),
                author=_login(entry.get("user")),
                published_at=_timestamp(entry.get("created_at")),
            )
        except ValueError:
            # One unusable record must not cost us the other twenty-four.
            return None


def _shape_query(query: str) -> str:
    """Bias `query` toward reported pain without narrowing it to nothing.

    Three additions, each of which a caller can override by being explicit:

    * `type:issue` drops pull requests, which search returns by default. A PR is a
      proposed fix, not a report of a problem.
    * `in:title,body` stops a hit on some passer-by's comment from dragging in an
      issue about something else. What we quote is the body, so the body is what
      has to be about the query. Measured on "CI pipeline is extremely slow" this
      cut the match set from 1915 to 593 and every dropped hit was off-topic.
    * `-label:enhancement -label:"feature request"` drops the trackers' own marker
      for "wouldn't it be nice", which is what this collector is *not* looking for.

    A caller who writes their own `type:`, `is:`, `in:` or `label:` qualifier means
    it, so the matching addition is left off rather than fought with.

    Raises:
        CollectorError: If `query` is blank — the API refuses it with HTTP 422, and
            failing here says so more usefully than the API does.
    """
    cleaned = query.strip()
    if not cleaned:
        raise CollectorError("GitHub issue search needs a non-empty query")

    lowered = cleaned.lower()
    parts = [cleaned]
    if "type:" not in lowered and "is:issue" not in lowered and "is:pr" not in lowered:
        parts.append("type:issue")
    if "in:" not in lowered:
        parts.append("in:title,body")
    if "label:" not in lowered:
        parts.extend(_EXCLUDED_LABELS)
    return " ".join(parts)


def _headers() -> dict[str, str]:
    """Request headers, carrying a token only when one is configured."""
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": _API_VERSION,
        "User-Agent": _USER_AGENT,
    }
    token = get_settings().github_token
    if token is not None:
        headers["Authorization"] = f"Bearer {token.get_secret_value()}"
    return headers


def _entries(payload: dict[str, Any]) -> list[Any]:
    """The `items` array of a search payload.

    Raises:
        CollectorError: If the payload has no usable `items` array. A search that
            matched nothing still returns `[]`, so a missing array means the
            response is not the one we know how to read.
    """
    entries = payload.get("items")
    if entries is None:
        raise CollectorError(
            f"GitHub issue search returned no 'items' array: {short_body(payload)}",
        )
    if not isinstance(entries, list):
        raise CollectorError("GitHub issue search returned a non-list 'items' field")
    return entries


def _is_rate_limited(response: httpx.Response) -> bool:
    """Whether this refusal is the rate limit rather than some other 403."""
    if response.status_code not in (403, 429):
        return False
    if response.headers.get("x-ratelimit-remaining") == "0":
        return True
    return "rate limit" in _api_message(response).lower()


def _rate_limit_message(response: httpx.Response) -> str:
    """A refusal a reader can act on, rather than a stack trace."""
    limit = response.headers.get("x-ratelimit-limit", "~10")
    reset = _reset_at(response.headers.get("x-ratelimit-reset"))
    when = f", resets at {reset}" if reset else ""
    return (
        f"GitHub search rate limit reached: {limit} requests/minute for anonymous "
        f"search{when}. Set GITHUB_TOKEN in .env to raise it to 30/minute."
    )


def _reset_at(value: str | None) -> str | None:
    """Format the rate-limit reset header as a readable UTC time."""
    if value is None:
        return None
    try:
        return datetime.fromtimestamp(int(value), tz=UTC).isoformat(timespec="seconds")
    except (OSError, OverflowError, ValueError):
        return None


def _api_message(response: httpx.Response) -> str:
    """GitHub's own explanation for a failure, or the raw body if it has none."""
    try:
        payload: Any = response.json()
    except ValueError:
        return short_body(response.text)
    if isinstance(payload, dict):
        message = payload.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()
    return short_body(payload)


def _issue_ref(entry: dict[str, Any]) -> str | None:
    """A stable, readable id: `owner/repo#123`.

    Readable because it is what a citation shows a reader, and stable because the
    repo and number outlive any URL shape GitHub might change.
    """
    number = entry.get("number")
    if not isinstance(number, int) or isinstance(number, bool):
        return None
    slug = _repo_slug(entry)
    return f"{slug}#{number}" if slug else None


def _repo_slug(entry: dict[str, Any]) -> str | None:
    """`owner/repo`, from the API URL if present and the HTML URL otherwise."""
    api_url = entry.get("repository_url")
    if isinstance(api_url, str):
        match = _REPO_API_RE.search(api_url)
        if match:
            return match.group(1)
    html_url = entry.get("html_url")
    if isinstance(html_url, str):
        match = _REPO_HTML_RE.match(html_url)
        if match:
            return match.group(1)
    return None


def _login(user: Any) -> str | None:
    """The reporter's handle, if the hit carries one."""
    if isinstance(user, dict):
        login = user.get("login")
        if isinstance(login, str):
            return login
    return None


def _text_field(entry: dict[str, Any], key: str) -> str | None:
    """A string field, or None when the API sent something else."""
    value = entry.get(key)
    return value if isinstance(value, str) else None


def _timestamp(value: Any) -> datetime | None:
    """Parse an ISO-8601 stamp as timezone-aware UTC.

    Anything naive is read as UTC — GitHub always sends `Z`, and the Evidence model
    downstream rejects naive datetimes outright.
    """
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _bounded(body: str) -> str:
    """Cap a runaway body at a whole line or sentence.

    A prefix of the body is still verbatim, so an excerpt taken from it still
    verifies; cutting mid-sentence would not be, because the truncated tail would
    later be quoted as though the reporter had stopped there.
    """
    if len(body) <= _MAX_BODY_CHARS:
        return body
    head = body[:_MAX_BODY_CHARS]
    cut = max(head.rfind("\n"), head.rfind(". "))
    # Only honour a boundary in the back half; a cut near the start would throw
    # away most of what the reporter wrote.
    return head[: cut + 1] if cut > _MAX_BODY_CHARS // 2 else head


__all__ = ["GitHubIssuesCollector"]
