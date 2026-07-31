"""Markdown report rendering safety and fidelity tests."""

from app.venture.reporting import _safe_source_links, _text


def test_multiple_catalog_sources_render_as_independent_links() -> None:
    rendered = _safe_source_links("https://example.gov/one?a=1 | https://example.gov/two?b=2")

    assert rendered == (
        "[source 1](<https://example.gov/one?a=1>), [source 2](<https://example.gov/two?b=2>)"
    )


def test_non_url_catalog_source_is_rendered_as_text() -> None:
    assert _safe_source_links("<untrusted>") == "&lt;untrusted&gt;"


def test_model_authored_markdown_link_and_title_are_escaped() -> None:
    rendered = _text("# [Click me](https://evil.example) **trusted** `code`")

    assert rendered == (
        r"\# \[Click me\]\(https://evil.example\) "
        r"\*\*trusted\*\* \`code\`"
    )
