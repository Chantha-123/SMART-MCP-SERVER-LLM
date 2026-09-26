import asyncio
import os
import unittest
from unittest.mock import patch

from agents import get_web_agent
from lib.base_agent import ensure_agent_initialized
from lib.state import RuntimeState


class PersistentMCPSessionTests(unittest.TestCase):
    def test_session_stays_open_and_is_reused(self) -> None:
        async def scenario() -> None:
            env = {"LLM_PROVIDER": "ollama", "OLLAMA_BASE_URL": "http://127.0.0.1:9", "MCP_DISABLE_PERSISTENCE": "1"}
            with patch.dict(os.environ, env), \
                    patch("llm_providers.ollama.ensure_model_pulled"), patch("llm_providers.ollama.warm_up_model"):
                state = RuntimeState("web-reader")
                agent = get_web_agent()
                await ensure_agent_initialized(state, agent)
                session = state.mcp_session
                self.assertTrue(session.alive)

                # Tool calls go through the already-running server (empty URL needs no network)
                result = await state.tool_map["fetch_web_page"].ainvoke({"url": ""})
                self.assertIn("URL parameter is empty", str(result))

                # Rebuilding the LLM side keeps the same MCP session
                state.agent_executor = None
                await ensure_agent_initialized(state, agent)
                self.assertIs(state.mcp_session, session)

                await agent.disconnect(state)
                self.assertFalse(session.alive)

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
