import os
import unittest
from unittest.mock import patch

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from lib.base_agent import BaseAgent, _tool_summary
from lib.state import RuntimeState


class SearchArgs(BaseModel):
    jql: str = Field(description="JQL query")
    limit: int = 10


def _tool(name: str) -> StructuredTool:
    def run(jql: str, limit: int = 10) -> str:
        return "ok"
    return StructuredTool.from_function(func=run, name=name, description=f"{name} tool", args_schema=SearchArgs)


class ToolSummaryTests(unittest.TestCase):
    def test_summary_lists_parameters_with_required_flags(self) -> None:
        summary = _tool_summary(_tool("jira_search"))
        self.assertEqual(summary["original_name"], "jira_search")
        self.assertEqual(summary["description"], "jira_search tool")
        params = {p["name"]: p for p in summary["parameters"]}
        self.assertTrue(params["jql"]["required"])
        self.assertFalse(params["limit"]["required"])
        self.assertEqual(params["limit"]["type"], "integer")

    def test_all_tools_are_kept_when_local_model_gets_a_subset(self) -> None:
        state = RuntimeState("jira")
        state.mcp_tools = [_tool("jira_search"), _tool("jira_get_issue"), _tool("jira_create_issue")]
        env = {"LLM_PROVIDER": "ollama", "MCP_DISABLE_PERSISTENCE": "1"}
        with patch.dict(os.environ, env), \
                patch("llm_providers.ollama.ensure_model_pulled"), patch("llm_providers.ollama.warm_up_model"):
            BaseAgent("jira").build_executor(state)
        self.assertEqual([t["original_name"] for t in state.tool_summaries], ["jira_search", "jira_get_issue"])
        self.assertEqual(len(state.all_tool_summaries), 3)


if __name__ == "__main__":
    unittest.main()
