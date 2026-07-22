"""``web_fetch`` tool — fetch a URL and return markdown-converted content."""

from __future__ import annotations

from pydantic import BaseModel, Field

from autoship.agent.tools.base import Tool, ToolContext, ToolResult
from autoship.agent.tools.truncation import TruncationService


class WebFetchInput(BaseModel):
    """Input for the ``web_fetch`` tool."""

    url: str = Field(..., description="The URL to fetch (http:// or https://).")
    format: str = Field(
        "markdown",
        description=(
            "Output format: ``markdown`` (default, strips HTML tags), "
            "``text`` (plain text), or ``raw`` (original content)."
        ),
    )
    timeout_ms: int | None = Field(
        None,
        description="Per-call timeout override in milliseconds.",
        ge=1,
    )


_WEB_FETCH_DESCRIPTION = """\
Fetch a URL and return its content.

By default, HTML pages are converted to markdown (HTML tags stripped,
code blocks preserved). Use ``format=text`` for plain text or
``format=raw`` for the original response body.

HTTPS is enforced — HTTP URLs are automatically upgraded.

The result is truncated to 50KB / 2000 lines. When truncated, the full
output is saved to a temp file and the path is noted in the result.
"""

DEFAULT_TIMEOUT_MS = 30_000
MAX_RESPONSE_BYTES = 500_000  # 500 KB cap on what we read from the network


async def _web_fetch_execute(args: BaseModel, ctx: ToolContext) -> ToolResult:
    assert isinstance(args, WebFetchInput)

    if args.format not in {"markdown", "text", "raw"}:
        return ToolResult.failure(
            f"Invalid format: {args.format!r}. "
            "Must be one of: markdown, text, raw."
        )

    # Enforce HTTPS.
    url = args.url
    if url.startswith("http://"):
        url = "https://" + url[len("http://") :]
    elif not url.startswith("https://"):
        return ToolResult.failure(
            f"URL must start with http:// or https:// (got: {url!r})"
        )

    timeout = (args.timeout_ms or DEFAULT_TIMEOUT_MS) / 1000

    try:
        import httpx
    except ImportError as exc:
        return ToolResult.failure(
            f"httpx is not installed: {exc}. "
            "Install with: pip install httpx"
        )

    try:
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            max_redirects=5,
        ) as client:
            response = await client.get(
                url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (compatible; autoship-agent/1.0; "
                        "+https://github.com/autoship/autoship)"
                    ),
                    "Accept": "text/html,application/xhtml+xml,"
                    "application/xml;q=0.9,text/plain;q=0.8,*/*;q=0.7",
                },
            )
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            body = response.text[:MAX_RESPONSE_BYTES]
    except httpx.HTTPStatusError as exc:
        return ToolResult.failure(
            f"HTTP error fetching {url}: {exc.response.status_code} "
            f"{exc.response.reason_phrase}"
        )
    except httpx.RequestError as exc:
        return ToolResult.failure(f"Network error fetching {url}: {exc}")
    except Exception as exc:  # noqa: BLE001
        return ToolResult.failure(f"Failed to fetch {url}: {exc}")

    # Convert based on format and content type.
    if args.format == "raw" or args.format == "text" or not _is_html(content_type, body):
        output = body
    else:
        # markdown conversion
        output = _html_to_markdown(body)

    # Truncate.
    truncator = TruncationService()
    trunc = truncator.truncate(output, tool_id="web_fetch")

    metadata: dict[str, object] = {
        "url": url,
        "status_code": response.status_code,
        "content_type": content_type,
        "bytes": len(body.encode("utf-8", errors="replace")),
    }
    if trunc.truncated:
        metadata["truncated"] = True
        if trunc.output_path is not None:
            metadata["output_path"] = trunc.output_path

    return ToolResult(output=trunc.content, metadata=metadata)


def _is_html(content_type: str, body: str) -> bool:
    """Heuristic: does this look like an HTML response?"""

    if "html" in content_type.lower():
        return True
    stripped = body.lstrip()[:200].lower()
    return stripped.startswith(("<!doctype html", "<html", "<head"))


def _html_to_markdown(html: str) -> str:
    """Very basic HTML → markdown conversion.

    For a real conversion we'd use ``markdownify`` or ``html2text``, but
    to avoid adding dependencies we strip tags, preserve code blocks,
    and keep links/headings as plain text. Good enough for LLM context.
    """

    import re

    # Remove scripts and styles entirely.
    html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)

    # Preserve code blocks.
    html = re.sub(
        r"<pre[^>]*>(.*?)</pre>",
        lambda m: "\n```\n" + _strip_tags(m.group(1)) + "\n```\n",
        html,
        flags=re.DOTALL | re.IGNORECASE,
    )
    html = re.sub(
        r"<code[^>]*>(.*?)</code>",
        lambda m: "`" + _strip_tags(m.group(1)) + "`",
        html,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # Headings → markdown.
    for level in range(6, 0, -1):
        prefix = "\n" + "#" * level + " "

        def _heading_repl(m: re.Match[str], prefix: str = prefix) -> str:
            return prefix + _strip_tags(m.group(1)).strip() + "\n"

        html = re.sub(
            rf"<h{level}[^>]*>(.*?)</h{level}>",
            _heading_repl,
            html,
            flags=re.DOTALL | re.IGNORECASE,
        )

    # Paragraphs and breaks.
    html = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
    html = re.sub(r"<p[^>]*>", "\n\n", html, flags=re.IGNORECASE)
    html = re.sub(r"</p>", "\n", html, flags=re.IGNORECASE)

    # List items.
    html = re.sub(r"<li[^>]*>", "\n- ", html, flags=re.IGNORECASE)
    html = re.sub(r"</li>", "", html, flags=re.IGNORECASE)

    # Links → "text (url)".
    html = re.sub(
        r"<a\s+[^>]*href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>",
        lambda m: f"{_strip_tags(m.group(2)).strip()} ({m.group(1)})",
        html,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # Strip all remaining tags.
    text = _strip_tags(html)

    # Collapse excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _strip_tags(html: str) -> str:
    """Remove all HTML tags and decode basic entities."""

    import html as html_module
    import re

    text = re.sub(r"<[^>]+>", "", html)
    text = html_module.unescape(text)
    return text


def make_web_fetch_tool() -> Tool:
    """Construct the ``web_fetch`` tool."""

    return Tool(
        id="web_fetch",
        description=_WEB_FETCH_DESCRIPTION,
        parameters=WebFetchInput,
        execute=_web_fetch_execute,
        timeout_ms=0,  # use per-call timeout via args
    )


__all__ = ["DEFAULT_TIMEOUT_MS", "WebFetchInput", "make_web_fetch_tool"]
