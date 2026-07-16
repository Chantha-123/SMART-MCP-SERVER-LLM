import os
import unittest
from unittest.mock import patch

from agents.teams_agent import GoogleChatAgent


class GoogleChatAgentTests(unittest.TestCase):
    def test_validate_environment_requires_required_vars(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            agent = GoogleChatAgent()
            with self.assertRaises(ValueError) as exc:
                agent.validate_environment()

        self.assertIn("GOOGLE_CHAT_CREDENTIALS_JSON", str(exc.exception))

    def test_build_space_message_payload(self) -> None:
        agent = GoogleChatAgent()
        payload = agent._build_space_message_payload("Hello")
        self.assertEqual(payload["text"], "Hello")


if __name__ == "__main__":
    unittest.main()
