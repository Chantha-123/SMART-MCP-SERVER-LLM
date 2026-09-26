import json
import unittest

from lib.base_agent import compact_tool_output


class CompactToolOutputTests(unittest.TestCase):
    def test_minifies_and_drops_noise(self) -> None:
        payload = {
            "issues": [{
                "id": "10006",
                "key": "SCRUM-7",
                "status": {"name": "To Do", "color": "blue-gray"},
                "priority": {"name": "Medium"},
                "assignee": {
                    "account_id": "abc",
                    "display_name": "Bun Chantha",
                    "name": "Bun Chantha",
                    "avatar_url": "https://example.com/a.png",
                },
                "labels": [],
                "parent": None,
            }]
        }
        raw = json.dumps(payload, indent=2)
        compact = compact_tool_output(raw)

        self.assertLess(len(compact), len(raw) / 2)
        issue = json.loads(compact)["issues"][0]
        self.assertEqual(issue["status"], "To Do")
        self.assertEqual(issue["priority"], "Medium")
        self.assertEqual(issue["assignee"], "Bun Chantha")
        self.assertNotIn("labels", issue)
        self.assertNotIn("parent", issue)
        self.assertNotIn("avatar_url", compact)

    def test_long_strings_are_clipped(self) -> None:
        compact = compact_tool_output(json.dumps({"description": "x" * 1000, "key": "A-1"}))
        self.assertLess(len(json.loads(compact)["description"]), 500)

    def test_non_json_text_and_content_blocks(self) -> None:
        self.assertEqual(compact_tool_output("plain text"), "plain text")
        blocks = compact_tool_output([{"type": "text", "text": '{"a": 1,  "b": null}'}])
        self.assertEqual(blocks[0]["text"], '{"a":1}')


if __name__ == "__main__":
    unittest.main()
