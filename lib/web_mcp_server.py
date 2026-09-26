import os
import sys
import re
import urllib.request
import urllib.parse
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("WebReader")


def html_to_markdown(html: str) -> str:
    """Convert raw HTML string into clean markdown text."""
    # Remove script and style elements
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
    # Convert links <a href="url">text</a> -> [text](url)
    html = re.sub(r'<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', r"[\2](\1)", html, flags=re.DOTALL | re.IGNORECASE)
    # Convert headings
    html = re.sub(r'<h[1-6][^>]*>(.*?)</h[1-6]>', r"\n\n### \1\n\n", html, flags=re.DOTALL | re.IGNORECASE)
    # Convert paragraph tags to newlines
    html = re.sub(r'<p[^>]*>(.*?)</p>', r"\n\1\n", html, flags=re.DOTALL | re.IGNORECASE)
    # Convert br to newline
    html = re.sub(r'<br\s*/?>', r"\n", html, flags=re.IGNORECASE)
    # Strip remaining HTML tags
    text = re.sub(r"<[^>]+>", " ", html)
    # Unescape common HTML entities
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&quot;", '"', text)
    # Collapse consecutive whitespace & blank lines
    text = re.sub(r"\n\s*\n", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


import subprocess

@mcp.tool()
def fetch_web_page(url: str) -> str:
    """Fetch and extract text content from a web page URL. Only call this when the user provides or asks about a specific URL."""
    if not url:
        return "Error: URL parameter is empty."
    
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    raw_html = ""

    # Try fast curl subprocess first
    try:
        res = subprocess.run(
            ["curl", "-s", "-L", "-k", "--max-time", "12", "-A", user_agent, url],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if res.returncode == 0 and res.stdout.strip():
            raw_html = res.stdout
    except Exception:
        pass

    # Fallback to urllib if curl produced no output
    if not raw_html:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": user_agent})
            with urllib.request.urlopen(req, timeout=12) as resp:
                content_type = resp.headers.get_content_charset() or "utf-8"
                raw_html = resp.read().decode(content_type, errors="replace")
        except Exception as e:
            return f"Error fetching web page '{url}': {str(e)}"

    if not raw_html:
        return f"Error fetching web page '{url}': No content returned."

    parsed_md = html_to_markdown(raw_html)
    if len(parsed_md) > 15000:
        parsed_md = parsed_md[:15000] + "\n\n[Content truncated to fit LLM context size]"
    return parsed_md


if __name__ == "__main__":
    mcp.run()
