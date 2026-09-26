import os
import unittest
from unittest.mock import patch

from lib.base_agent import BaseAgent


class EnvValidationTests(unittest.TestCase):
    def test_placeholder_values_are_treated_as_missing(self) -> None:
        with patch.dict(os.environ, {"JIRA_API_TOKEN": "your_jira_api_token_here"}, clear=True):
            agent = BaseAgent("jira", required_env_vars=["JIRA_API_TOKEN"])
            with self.assertRaises(ValueError) as exc:
                agent.validate_environment()

        self.assertIn("Missing required environment variables", str(exc.exception))

    def test_real_values_are_accepted(self) -> None:
        with patch.dict(os.environ, {"JIRA_API_TOKEN": "123456"}, clear=True):
            agent = BaseAgent("jira", required_env_vars=["JIRA_API_TOKEN"])
            agent.validate_environment()

    def test_default_ollama_model_uses_local_working_model(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            from llm_providers.ollama import get_default_ollama_model
            self.assertEqual(get_default_ollama_model(), "llama3.2")


if __name__ == "__main__":
    unittest.main()
