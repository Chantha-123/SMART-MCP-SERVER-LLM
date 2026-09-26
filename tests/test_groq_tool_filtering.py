import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from lib.base_agent import _matches_tool_name, sanitize_session_history


def test_matches_tool_name_with_prefixes():
    enabled = {"search_code", "get_issue", "search_repositories", "conversations_history"}
    
    assert _matches_tool_name("search_code", enabled) is True
    assert _matches_tool_name("git_search_code", enabled) is True
    assert _matches_tool_name("github_search_code", enabled) is True
    assert _matches_tool_name("git_get_issue", enabled) is True
    assert _matches_tool_name("git_search_repositories", enabled) is True
    assert _matches_tool_name("slack_conversations_history", enabled) is True
    
    assert _matches_tool_name("brave_search", enabled) is False
    assert _matches_tool_name("unknown_tool", enabled) is False


def test_sanitize_session_history_removes_unlisted_tools():
    valid_tools = {"search_code", "git_search_code", "get_issue"}

    history = [
        HumanMessage(content="Find code for authentication"),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "brave_search", "id": "call_1", "args": {"query": "auth"}},
                {"name": "git_search_code", "id": "call_2", "args": {"query": "auth"}},
            ],
        ),
        ToolMessage(content="brave results", name="brave_search", tool_call_id="call_1"),
        ToolMessage(content="git results", name="git_search_code", tool_call_id="call_2"),
    ]

    sanitized = sanitize_session_history(history, valid_tools)

    assert len(sanitized) == 4
    assert isinstance(sanitized[0], HumanMessage)
    
    # Check AIMessage only contains git_search_code
    ai_msg = sanitized[1]
    assert isinstance(ai_msg, AIMessage)
    assert len(ai_msg.tool_calls) == 1
    assert ai_msg.tool_calls[0]["name"] == "git_search_code"

    # Check brave_search ToolMessage was converted to HumanMessage
    tool_msg_1 = sanitized[2]
    assert isinstance(tool_msg_1, HumanMessage)
    assert "[Result for unlisted tool brave_search]" in tool_msg_1.content

    # Check git_search_code ToolMessage remains ToolMessage
    tool_msg_2 = sanitized[3]
    assert isinstance(tool_msg_2, ToolMessage)
    assert tool_msg_2.name == "git_search_code"


def test_sanitize_session_history_all_invalid():
    valid_tools = {"git_get_issue"}

    history = [
        HumanMessage(content="Search web"),
        AIMessage(
            content="",
            tool_calls=[{"name": "brave_search", "id": "call_1", "args": {}}],
        ),
    ]

    sanitized = sanitize_session_history(history, valid_tools)

    assert len(sanitized) == 2
    ai_msg = sanitized[1]
    assert isinstance(ai_msg, AIMessage)
    assert len(ai_msg.tool_calls) == 0
    assert ai_msg.content == "[Previous tool call to unlisted tool was omitted]"


def test_matches_tool_name_is_exact_not_substring():
    enabled = {"jira_search", "jira_get_issue"}

    assert _matches_tool_name("jira_search", enabled) is True
    assert _matches_tool_name("jira_get_issue", enabled) is True
    # Longer tools sharing a prefix must not be enabled (keeps local-model prompts small)
    assert _matches_tool_name("jira_search_projects", enabled) is False
    assert _matches_tool_name("jira_get_issue_sla", enabled) is False
    assert _matches_tool_name("jira_get_issue_watchers", enabled) is False


def test_old_timing_lines_are_removed_from_history():
    history = [
        HumanMessage(content="hi"),
        AIMessage(content="Hello!\n\n⏱️ *Response time: 11.28 seconds*"),
    ]
    cleaned = sanitize_session_history(history, set())
    assert cleaned[1].content == "Hello!"
