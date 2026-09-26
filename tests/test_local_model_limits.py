import os
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from lib.base_agent import _local_tool_output_limit, trim_session_history, truncate_tool_output


class TruncateToolOutputTests(unittest.TestCase):
    def test_short_string_is_unchanged(self) -> None:
        self.assertEqual(truncate_tool_output("abc", 10), "abc")

    def test_long_string_is_truncated_with_note(self) -> None:
        result = truncate_tool_output("x" * 100, 10)
        self.assertTrue(result.startswith("x" * 10))
        self.assertIn("truncated", result)
        self.assertNotIn("x" * 11, result)

    def test_content_blocks_share_one_budget(self) -> None:
        blocks = [{"type": "text", "text": "a" * 8}, {"type": "text", "text": "b" * 8}, {"type": "text", "text": "c"}]
        result = truncate_tool_output(blocks, 10)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["text"], "a" * 8)
        self.assertTrue(result[1]["text"].startswith("bb"))
        self.assertIn("truncated", result[1]["text"])

    def test_budget_is_utf8_bytes_for_non_latin_text(self) -> None:
        khmer = "ក" * 100  # 3 bytes per character
        result = truncate_tool_output(khmer, 30)
        self.assertTrue(result.startswith("ក" * 10))
        self.assertNotIn("ក" * 11, result)

    def test_no_limit_returns_original(self) -> None:
        self.assertEqual(truncate_tool_output("x" * 100, None), "x" * 100)

    def test_limit_only_applies_to_ollama(self) -> None:
        with patch.dict(os.environ, {"LLM_PROVIDER": "gemini"}, clear=True):
            self.assertIsNone(_local_tool_output_limit())
        with patch.dict(os.environ, {"LLM_PROVIDER": "ollama"}, clear=True):
            self.assertEqual(_local_tool_output_limit(), 4000)


class TrimSessionHistoryTests(unittest.TestCase):
    def _turn(self, n: int) -> list:
        return [
            HumanMessage(content=f"q{n}"),
            AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": f"c{n}"}]),
            ToolMessage(content="r", tool_call_id=f"c{n}", name="t"),
            AIMessage(content=f"a{n}"),
        ]

    def test_keeps_last_turns_starting_at_human_message(self) -> None:
        history = self._turn(1) + self._turn(2) + self._turn(3)
        trimmed = trim_session_history(history, 2)
        self.assertEqual(len(trimmed), 8)
        self.assertEqual(trimmed[0].content, "q2")

    def test_short_history_is_unchanged(self) -> None:
        history = self._turn(1)
        self.assertIs(trim_session_history(history, 6), history)


class OllamaProviderTests(unittest.TestCase):
    def test_max_tokens_is_sent_in_the_field_ollama_honors(self) -> None:
        # Ollama ignores `max_completion_tokens`, which ChatOpenAI(max_tokens=...) sends
        from llm_providers import ollama

        env = {"OLLAMA_BASE_URL": "http://ollama:11434", "OLLAMA_NUM_PREDICT": "256"}
        with patch.dict(os.environ, env, clear=True), \
                patch.object(ollama, "ensure_model_pulled"), patch.object(ollama, "warm_up_model"):
            llm = ollama.get_llm()
        self.assertEqual(llm.extra_body, {"max_tokens": 256})
        self.assertIsNone(llm.max_tokens)


if __name__ == "__main__":
    unittest.main()
