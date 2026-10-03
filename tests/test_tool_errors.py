import asyncio
import unittest

from langchain_core.tools import StructuredTool
from mcp.shared.exceptions import McpError
from mcp.types import ErrorData

from lib.base_agent import SanitizedTool, sanitize_args

QUERY_SCHEMA = {"properties": {"query": {"type": "string"}, "page": {"type": "number"}}, "required": ["query"]}
Q_SCHEMA = {"properties": {"q": {"type": "string"}}, "required": ["q"]}


class ArgumentRepairTests(unittest.TestCase):
    def test_q_is_renamed_to_query_when_schema_expects_query(self) -> None:
        self.assertEqual(sanitize_args({"q": "user:octocat"}, QUERY_SCHEMA), {"query": "user:octocat"})

    def test_query_is_renamed_to_q_when_schema_expects_q(self) -> None:
        self.assertEqual(sanitize_args({"query": "auth"}, Q_SCHEMA), {"q": "auth"})

    def test_correct_arguments_are_unchanged(self) -> None:
        self.assertEqual(sanitize_args({"query": "x", "page": 2}, QUERY_SCHEMA), {"query": "x", "page": 2})
        both = {"properties": {"q": {}, "query": {}}}
        self.assertEqual(sanitize_args({"q": "a"}, both), {"q": "a"})


class ToolErrorTests(unittest.TestCase):
    def test_tool_exception_becomes_tool_result(self) -> None:
        async def failing(query: str) -> str:
            raise McpError(ErrorData(code=-32602, message='Invalid input: path ["query"] Required'))

        tool = SanitizedTool(StructuredTool.from_function(coroutine=failing, name="search_repositories",
                                                         description="search"), QUERY_SCHEMA)
        result = asyncio.run(tool.ainvoke({"query": "x"}))
        self.assertIn("Tool error from search_repositories", result)
        self.assertIn("Invalid input", result)

    def test_aliased_argument_reaches_the_tool(self) -> None:
        async def search(query: str) -> str:
            return f"results for {query}"

        tool = SanitizedTool(StructuredTool.from_function(coroutine=search, name="search_repositories",
                                                         description="search"), QUERY_SCHEMA)
        self.assertEqual(asyncio.run(tool._arun(q="user:octocat"))[0], "results for user:octocat")


if __name__ == "__main__":
    unittest.main()
