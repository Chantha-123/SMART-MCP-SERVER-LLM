import os
import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage


class DashboardApiTests(unittest.TestCase):
    def setUp(self) -> None:
        env = {
            "GOOGLE_API_KEY": "secret-google-key-123",
            "GROQ_API_KEY": "",
            "MCP_DISABLE_PERSISTENCE": "1",
            "MCP_PRECONNECT": "false",
        }
        self.env = patch.dict(os.environ, env)
        self.env.start()
        import web_server
        self.web_server = web_server
        web_server.active_agents.clear()
        web_server.AGENT_STATUS.clear()
        # No `with`: the lifespan (MCP pre-connect) doesn't run in these tests
        self.client = TestClient(web_server.app)

    def tearDown(self) -> None:
        self.env.stop()

    def test_config_never_contains_key_values(self) -> None:
        res = self.client.get("/api/config")
        self.assertEqual(res.status_code, 200)
        self.assertNotIn("secret-google-key-123", res.text)
        data = res.json()
        self.assertNotIn("api_keys", data)
        self.assertTrue(data["server_keys"]["gemini"])
        self.assertFalse(data["server_keys"]["groq"])

    def test_status_reports_every_agent(self) -> None:
        self.web_server._set_status("jira", "ready", "Connected", tools=3)
        self.web_server._set_status("slack", "error", "missing token")
        agents = self.client.get("/api/status").json()["agents"]
        self.assertEqual(set(agents), set(self.web_server.AGENT_BUILDERS))
        self.assertEqual(agents["jira"], {"state": "ready", "message": "Connected", "tools": 3})
        self.assertEqual(agents["slack"]["state"], "error")
        self.assertEqual(agents["github"]["state"], "idle")

    def test_chat_returns_timing_separately_and_sessions_get_titles(self) -> None:
        ws = self.web_server
        with patch.object(ws.ActiveAgent, "ensure_initialized", new=AsyncMock()), \
                patch.object(ws, "stream_agent_response", new=AsyncMock(return_value=AIMessage(content="Here you go."))):
            res = self.client.post("/api/chat", json={
                "agent_name": "jira", "message": "List my jira projects", "session_id": "chat-1",
                "provider": "ollama", "model": "qwen2.5:3b", "api_key": "",
            })
        data = res.json()
        self.assertEqual(data["response"], "Here you go.")
        self.assertNotIn("Response time", data["response"])
        self.assertIsInstance(data["elapsed_seconds"], float)

        sessions = self.client.get("/api/sessions/jira").json()["sessions"]
        self.assertEqual(sessions[0]["title"], "List my jira projects")

    def test_saved_timing_lines_are_hidden_from_history(self) -> None:
        state = self.web_server.get_active_agent("slack").state
        from langchain_core.messages import HumanMessage
        state.record_message("old", HumanMessage(content="hi"))
        state.record_message("old", AIMessage(content="Hello!\n\n⏱️ *Response time: 3.20 seconds*"))
        messages = self.client.get("/api/sessions/slack/old/messages").json()["messages"]
        self.assertEqual(messages[1]["text"], "Hello!")


class IconRouteTests(unittest.TestCase):
    def test_icons_are_served_with_the_right_type(self) -> None:
        import web_server
        client = TestClient(web_server.app)
        for path, media_type in {
            "/favicon.svg": "image/svg+xml",
            "/favicon.ico": "image/png",
            "/favicon-32.png": "image/png",
            "/apple-touch-icon.png": "image/png",
        }.items():
            res = client.get(path)
            self.assertEqual(res.status_code, 200, path)
            self.assertTrue(res.headers["content-type"].startswith(media_type), path)
            self.assertGreater(len(res.content), 100, path)

    def test_page_links_to_the_icons(self) -> None:
        import web_server
        html = TestClient(web_server.app).get("/").text
        self.assertIn('rel="icon" href="/favicon.svg"', html)
        self.assertIn('rel="apple-touch-icon" href="/apple-touch-icon.png"', html)


class MessageTimeTests(unittest.TestCase):
    def test_messages_endpoint_returns_saved_times(self) -> None:
        with patch.dict(os.environ, {"MCP_DISABLE_PERSISTENCE": "1", "MCP_PRECONNECT": "false"}):
            import web_server
            from langchain_core.messages import HumanMessage
            web_server.active_agents.clear()
            state = web_server.get_active_agent("jira").state
            state.record_message("timed", HumanMessage(content="List my jira projects"))
            state.record_message("timed", AIMessage(content="MCP, SCRUM, SUP"))
            messages = TestClient(web_server.app).get("/api/sessions/jira/timed/messages").json()["messages"]
        self.assertEqual([m["sender"] for m in messages], ["user", "agent"])
        for message in messages:
            self.assertRegex(message["time"], r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")

    def test_timestamp_survives_save_and_reload(self) -> None:
        import json
        import tempfile
        from langchain_core.messages import HumanMessage
        from lib.state import RuntimeState
        with tempfile.TemporaryDirectory() as tmp, \
                patch.dict(os.environ, {"MCP_STATE_DIR": tmp, "MCP_DISABLE_PERSISTENCE": ""}):
            first = RuntimeState("jira")
            first.record_message("s1", HumanMessage(content="hi"))
            saved_at = first.chat_sessions["s1"][0].response_metadata["saved_at"]
            reloaded = RuntimeState("jira")
            self.assertEqual(reloaded.chat_sessions["s1"][0].response_metadata["saved_at"], saved_at)


if __name__ == "__main__":
    unittest.main()
