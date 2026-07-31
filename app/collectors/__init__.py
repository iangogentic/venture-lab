"""Collectors: the pipeline's connection to sources outside itself.

Importing this package registers every collector, which is what lets
`collect-evidence` discover them by name. Each one searches a single source and
returns what it found verbatim; judging relevance is the stage's job, not theirs.
"""

from app.collectors.app_reviews import AppStoreReviewsCollector
from app.collectors.base import (
    COLLECTORS,
    Collector,
    CollectorConfig,
    SourceItem,
    available,
    config_from_settings,
    get_collector,
    register,
)
from app.collectors.discourse import DiscourseCollector
from app.collectors.filesystem import FilesystemCollector
from app.collectors.github_issues import GitHubIssuesCollector
from app.collectors.hackernews import HackerNewsCollector
from app.collectors.reddit import RedditCollector
from app.collectors.rss import RssCollector
from app.collectors.stackexchange import StackExchangeCollector
from app.collectors.tavily import TavilyCollector
from app.collectors.web import WebCollector

__all__ = [
    "COLLECTORS",
    "AppStoreReviewsCollector",
    "Collector",
    "CollectorConfig",
    "DiscourseCollector",
    "FilesystemCollector",
    "GitHubIssuesCollector",
    "HackerNewsCollector",
    "RedditCollector",
    "RssCollector",
    "SourceItem",
    "StackExchangeCollector",
    "TavilyCollector",
    "WebCollector",
    "available",
    "config_from_settings",
    "get_collector",
    "register",
]
