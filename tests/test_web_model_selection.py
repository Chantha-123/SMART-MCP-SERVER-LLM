import asyncio
import os
import unittest
from unittest.mock import AsyncMock, patch


class DashboardModelSelectionTests(unittest.TestCase):
    def test_dashboard_model_overrides_provider_model_from_env(self) -> None:
        # OLLAMA_MODEL from .env / the Dockerfile must not shadow the model picked in the UI
        with patch.dict(os.environ, {"OLLAMA_MODEL": "llama3.2", "MCP_DISABLE_PERSISTENCE": "1"}):
            import web_server

            with patch.object(web_server, "ensure_agent_initialized", new=AsyncMock()):
                agent = web_server.ActiveAgent("web-reader")
                asyncio.run(agent.ensure_initialized("ollama", "qwen2.5:3b", "http://ollama:11434"))
            self.assertEqual(os.environ["OLLAMA_MODEL"], "qwen2.5:3b")


if __name__ == "__main__":
    unittest.main()
