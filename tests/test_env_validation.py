import os
import unittest
from unittest.mock import patch

from lib.base_agent import BaseAgent


class EnvValidationTests(unittest.TestCase):
    def test_placeholder_values_are_treated_as_missing(self) -> None:
        with patch.dict(os.environ, {"TG_APP_ID": "your_api_id_here"}, clear=True):
            agent = BaseAgent("telegram", required_env_vars=["TG_APP_ID"])
            with self.assertRaises(ValueError) as exc:
                agent.validate_environment()

        self.assertIn("Missing required environment variables", str(exc.exception))

    def test_real_values_are_accepted(self) -> None:
        with patch.dict(os.environ, {"TG_APP_ID": "123456"}, clear=True):
            agent = BaseAgent("telegram", required_env_vars=["TG_APP_ID"])
            agent.validate_environment()

    def test_telegram_agent_missing_session_file(self) -> None:
        from agents.telegram_agent import get_telegram_agent
        with patch.dict(os.environ, {
            "TG_APP_ID": "123456",
            "TG_API_HASH": "abcdef",
            "TG_SESSION_PATH": "/nonexistent/path/to/session"
        }, clear=True):
            agent = get_telegram_agent()
            with self.assertRaises(ValueError) as exc:
                agent.validate_environment()
            self.assertIn("Telegram session file not found at", str(exc.exception))
            self.assertIn("npx -y @chaindead/telegram-mcp auth --phone <your_phone_number>", str(exc.exception))
            self.assertIn("--password <your_2fa_password>", str(exc.exception))


if __name__ == "__main__":
    unittest.main()
