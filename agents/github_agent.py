import os
from lib.base_agent import BaseAgent

__all__ = ["get_github_agent"]


def get_github_agent() -> BaseAgent:
    has_pat = bool(os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN"))
    has_bearer = bool(os.getenv("GITHUB_MCP_BEARER_TOKEN"))

    if not has_pat and not has_bearer:
        required_vars = [
            "GITHUB_PERSONAL_ACCESS_TOKEN or GITHUB_MCP_BEARER_TOKEN"
        ]
    else:
        required_vars = []

    # Use the public GitHub MCP endpoint only when a bearer token is present.
    # Personal access tokens require an explicit MCP server URL.
    default_server_url = "https://api.githubcopilot.com/mcp/"

    return BaseAgent(
        service_name="github",
        required_env_vars=required_vars,
        server_url_env="GITHUB_MCP_SERVER_URL",
        default_server_url=default_server_url,
        token_env="GITHUB_PERSONAL_ACCESS_TOKEN",
        bearer_token_env="GITHUB_MCP_BEARER_TOKEN",
    )
