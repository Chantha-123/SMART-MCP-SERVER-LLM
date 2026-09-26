import json
import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from lib.base_agent import _asks_for_listing, _direct_list_answer, compact_tool_output, truncate_tool_output


def _turn(tool_content: str) -> list:
    return [
        HumanMessage(content="Search open issues in project SCRUM"),
        AIMessage(content="", tool_calls=[{"name": "jira_search", "args": {}, "id": "c1"}]),
        ToolMessage(content=tool_content, tool_call_id="c1", name="jira_search"),
    ]


class DirectListAnswerTests(unittest.TestCase):
    def test_listing_questions_are_detected(self) -> None:
        self.assertTrue(_asks_for_listing([HumanMessage(content="List my jira projects")]))
        self.assertTrue(_asks_for_listing([HumanMessage(content="Search open issues in SCRUM")]))
        self.assertFalse(_asks_for_listing([HumanMessage(content="Which issue is assigned to me?")]))

    def test_list_tool_result_is_formatted_without_llm(self) -> None:
        payload = json.dumps({"issues": [
            {"key": "SCRUM-7", "summary": "Loan balance", "status": "To Do",
             "browse_url": "https://x.atlassian.net/browse/SCRUM-7"},
        ]})
        answer = _direct_list_answer(_turn(payload), history_len=1)
        self.assertIsNotNone(answer)
        self.assertIn("[SCRUM-7](https://x.atlassian.net/browse/SCRUM-7)", answer.content)
        self.assertIn("Loan balance", answer.content)

    def test_errors_and_final_answers_go_to_the_llm(self) -> None:
        self.assertIsNone(_direct_list_answer(_turn("Error: JQL is invalid"), history_len=1))
        # A state that already ends with the model's answer isn't intercepted
        done = _turn("[]") + [AIMessage(content="No issues found.")]
        self.assertIsNone(_direct_list_answer(done, history_len=1))
        # Before any tool ran
        self.assertIsNone(_direct_list_answer(_turn("[]")[:1], history_len=1))


class JsonAwareTruncationTests(unittest.TestCase):
    def test_truncation_keeps_valid_json_and_counts_omitted_items(self) -> None:
        payload = json.dumps({"issues": [{"key": f"A-{i}", "summary": "x" * 200} for i in range(10)]})
        clipped = truncate_tool_output(payload, 1000)
        data = json.loads(clipped)
        self.assertLess(len(clipped.encode()), 1000)
        self.assertEqual(data["issues"][0]["key"], "A-0")
        self.assertIn("more not shown", data["omitted_items"])

    def test_direct_answer_mentions_omitted_items(self) -> None:
        payload = json.dumps({"issues": [{"key": f"A-{i}", "summary": "x" * 200} for i in range(10)]})
        answer = _direct_list_answer(_turn(truncate_tool_output(payload, 1000)), history_len=1)
        self.assertIn("more not shown", answer.content)
        self.assertNotIn("omitted_items", answer.content)

    def test_direct_answer_uses_untruncated_artifact(self) -> None:
        full = json.dumps({"issues": [{"key": f"A-{i}", "summary": "x" * 200} for i in range(10)]})
        turn = _turn(truncate_tool_output(full, 1000))
        turn[-1].artifact = full
        answer = _direct_list_answer(turn, history_len=1)
        self.assertIn("A-9", answer.content)
        self.assertNotIn("more not shown", answer.content)


class SlackCsvTests(unittest.TestCase):
    def test_header_only_csv_becomes_explicit_empty_result(self) -> None:
        empty = "MsgID,UserID,UserName,RealName,Channel,ThreadTs,Text,Time,Permalink,Cursor\n"
        self.assertIn("No results found", compact_tool_output(empty))
        self.assertIn("No results found", compact_tool_output("[]"))
        self.assertIn("No results found", compact_tool_output(json.dumps({"total": 0, "issues": []})))

    def test_csv_listing_is_answered_directly(self) -> None:
        channels = (
            "ID,Name,Topic,Purpose,MemberCount,Cursor\n"
            "C1,#social,,Just for fun,2,\n"
            "C2,#new-channel,,Team meetings,2,\n"
        )
        turn = _turn(channels)
        turn[0] = HumanMessage(content="List my slack channels")
        answer = _direct_list_answer(turn, history_len=1)
        self.assertIn("**#social**: Just for fun", answer.content)
        self.assertIn("**#new-channel**", answer.content)

    def test_csv_messages_show_author_time_and_text(self) -> None:
        history = (
            "MsgID,UserID,UserName,RealName,Channel,Text,Time,Cursor\n"
            "1.1,U1,bun,Bun Chantha,C2,Hello,2026-07-13T16:59:43Z,\n"
        )
        answer = _direct_list_answer(_turn(history), history_len=1)
        self.assertIn("**Bun Chantha** (2026-07-13T16:59:43Z): Hello", answer.content)


if __name__ == "__main__":
    unittest.main()
