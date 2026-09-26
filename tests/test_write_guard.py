import asyncio
import os
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from lib.base_agent import stream_agent_response
from lib.state import RuntimeState


class FakeExecutor:
    """Replays scripted agent runs: each run appends the given messages to the input."""

    def __init__(self, runs):
        self.runs = list(runs)
        self.calls = 0

    def _run(self, messages):
        added = self.runs[min(self.calls, len(self.runs) - 1)]
        self.calls += 1
        return {"messages": list(messages) + added}

    async def astream(self, inputs, stream_mode="values"):
        yield self._run(inputs["messages"])

    async def ainvoke(self, inputs):
        return self._run(inputs["messages"])


FAKE_SUCCESS = [AIMessage(content="I have successfully sent the message.")]
REAL_SEND = [
    AIMessage(content="", tool_calls=[{"name": "conversations_add_message", "args": {}, "id": "c1"}]),
    ToolMessage(content="ok", name="conversations_add_message", tool_call_id="c1"),
    AIMessage(content="Sent \"Good morning team\" to #all-mcpthesis."),
]


def _state(executor) -> RuntimeState:
    state = RuntimeState("slack")
    state.agent_executor = executor
    state.tool_map = {"channels_list": object(), "conversations_add_message": object()}
    return state


class WriteGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = patch.dict(os.environ, {"LLM_PROVIDER": "ollama", "MCP_DISABLE_PERSISTENCE": "1"})
        self.env.start()

    def tearDown(self) -> None:
        self.env.stop()

    def _ask(self, executor, text):
        return asyncio.run(stream_agent_response(_state(executor), [HumanMessage(content=text)]))

    def test_fake_success_is_never_reported(self) -> None:
        executor = FakeExecutor([FAKE_SUCCESS, FAKE_SUCCESS])
        answer = self._ask(executor, 'Send "Good morning team" to #all-mcpthesis')
        self.assertEqual(executor.calls, 2)
        self.assertIn("nothing was sent", answer.content)

    def test_retry_that_calls_the_tool_is_used(self) -> None:
        executor = FakeExecutor([FAKE_SUCCESS, REAL_SEND])
        answer = self._ask(executor, 'Send "Good morning team" to #all-mcpthesis')
        self.assertEqual(executor.calls, 2)
        self.assertIn("Sent", answer.content)

    def test_read_requests_are_not_retried(self) -> None:
        executor = FakeExecutor([[AIMessage(content="You have 3 channels.")]])
        answer = self._ask(executor, "How many channels do I have?")
        self.assertEqual(executor.calls, 1)
        self.assertEqual(answer.content, "You have 3 channels.")


if __name__ == "__main__":
    unittest.main()
