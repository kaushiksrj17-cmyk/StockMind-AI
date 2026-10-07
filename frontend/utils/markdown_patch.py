"""Transparent Markdown HTML Sanitizer for Streamlit.

Resolves the CommonMark issue where indented multiline HTML strings
(4+ leading spaces inside Python functions) are misinterpreted
by the parser as preformatted code blocks (<pre><code>).
"""

from functools import wraps
from typing import Any
import streamlit as st
from streamlit.delta_generator import DeltaGenerator


def sanitize_markdown_html(body: Any) -> Any:
    """Sanitize HTML body passed to markdown methods.

    If body is a string starting with an HTML element or tag, strips leading
    indentation from every line and skips blank lines so CommonMark does not
    break the HTML block or treat indented lines as code blocks.
    """
    if not isinstance(body, str):
        return body

    stripped = body.strip()
    if stripped.startswith("<"):
        lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        return "\n".join(lines)

    return body


def patch_streamlit_markdown() -> None:
    """Patch Streamlit markdown globally to ensure clean HTML block rendering."""
    if getattr(st, "_stockmind_markdown_patched", False):
        return

    orig_dg_markdown = DeltaGenerator.markdown

    @wraps(orig_dg_markdown)
    def patched_dg_markdown(self: Any, body: Any, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("unsafe_allow_html"):
            body = sanitize_markdown_html(body)
        return orig_dg_markdown(self, body, *args, **kwargs)

    DeltaGenerator.markdown = patched_dg_markdown  # type: ignore[assignment]
    st._stockmind_markdown_patched = True
