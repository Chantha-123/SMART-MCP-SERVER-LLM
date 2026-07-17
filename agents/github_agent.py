import os
from typing import Any
from lib.base_agent import BaseAgent
from lib.base_transport import StdioTransportMixin

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

    transport = os.getenv("GITHUB_MCP_TRANSPORT", "streamable_http").lower()

    if transport == "stdio":
        return _create_stdio_agent(required_vars)
    else:
        return _create_http_agent(required_vars)


def _create_stdio_agent(required_vars: list[str]) -> BaseAgent:
    class GithubStdioAgent(BaseAgent, StdioTransportMixin):
        def get_connection_config(self) -> dict[str, Any]:
            env = {
                "PATH": os.environ.get("PATH", ""),
                "GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN", ""),
            }
            return self.build_stdio_config(
                "npx",
                ["-y", "@modelcontextprotocol/server-github"],
                env,
            )

    return GithubStdioAgent(
        service_name="github",
        required_env_vars=required_vars,
    )


def _create_http_agent(required_vars: list[str]) -> BaseAgent:
    default_server_url = "https://api.githubcopilot.com/mcp/"
    return BaseAgent(
        service_name="github",
        required_env_vars=required_vars,
        server_url_env="GITHUB_MCP_SERVER_URL",
        default_server_url=default_server_url,
        token_env="GITHUB_PERSONAL_ACCESS_TOKEN",
        bearer_token_env="GITHUB_MCP_BEARER_TOKEN",
    )
