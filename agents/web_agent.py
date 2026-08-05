import os
import sys
from typing import Any
from lib.base_agent import BaseAgent
from lib.base_transport import StdioTransportMixin

__all__ = ["get_web_agent"]


def get_web_agent() -> BaseAgent:
    """Create and return a Web Reader MCP agent using FastMCP lib/web_mcp_server.py."""
    class WebStdioAgent(BaseAgent, StdioTransportMixin):
        def get_connection_config(self) -> dict[str, Any]:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            script_path = os.path.join(base_dir, "lib", "web_mcp_server.py")
            env = {
                "PATH": os.environ.get("PATH", ""),
                "PYTHONPATH": os.environ.get("PYTHONPATH", "") or base_dir,
            }
            return self.build_stdio_config(
                sys.executable,
                [script_path],
                env,
            )

    return WebStdioAgent(
        service_name="web-reader",
        required_env_vars=[],
    )
