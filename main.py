import sys
import warnings
from pathlib import Path

import typer
from dotenv import load_dotenv
from langchain_core._api.deprecation import LangChainPendingDeprecationWarning

# Suppress a known LangChain deprecation warning from langgraph imports.
warnings.filterwarnings(
    "ignore",
    category=LangChainPendingDeprecationWarning,
    message=r".*default value of `allowed_objects` will change.*",
)

ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env", override=False)

sys.path.insert(0, str(ROOT_DIR))

from lib.base_mcp import create_mcp_cli
from agents import get_github_agent, get_jira_agent, get_slack_agent, get_google_chat_agent, get_telegram_agent


app = typer.Typer(
    help="Community AI MCP Agent - Unified access to GitHub, Slack, and more",
    no_args_is_help=True,
)

app.add_typer(
    create_mcp_cli("github", get_github_agent, "GitHub MCP Agent CLI"),
    name="github",
    help="GitHub MCP Agent - Interact with GitHub repositories",
)

app.add_typer(
    create_mcp_cli("jira", get_jira_agent, "Jira MCP Agent CLI"),
    name="jira",
    help="Jira MCP Agent - Interact with Jira projects",
)

app.add_typer(
    create_mcp_cli("slack", get_slack_agent, "Slack MCP Agent CLI"),
    name="slack",
    help="Slack MCP Agent - Interact with Slack workspaces",
)

app.add_typer(
    create_mcp_cli("telegram", get_telegram_agent, "Telegram MCP Agent CLI"),
    name="telegram",
    help="Telegram MCP Agent - Interact with Telegram account",
)

app.add_typer(
    create_mcp_cli("google-chat", get_google_chat_agent, "Google Chat Agent CLI"),
    name="google-chat",
    help="Google Chat Agent - Send messages to Google Chat spaces",
)

if __name__ == "__main__":
    args = sys.argv[1:] or ["--help"]
    app(args)
