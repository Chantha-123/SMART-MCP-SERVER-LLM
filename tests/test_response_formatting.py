import json
from langchain_core.messages import AIMessage
from lib.utils import extract_message_text, format_json_payload_to_markdown


def test_format_json_array_of_dicts():
    raw_json = json.dumps([
        {"name": "SMART-MCP-SERVER-LLM"},
        {"name": "C-SHARP-School-System"},
        {"name": "Learn-html"}
    ])
    result = format_json_payload_to_markdown(raw_json)
    assert "- **SMART-MCP-SERVER-LLM**" in result
    assert "- **C-SHARP-School-System**" in result
    assert "- **Learn-html**" in result


def test_format_json_dict_with_list():
    raw_json = json.dumps({
        "repositories": [
            {"name": "repo1", "description": "test repo"},
            {"name": "repo2"}
        ]
    })
    result = format_json_payload_to_markdown(raw_json)
    assert "- **repo1**: test repo" in result
    assert "- **repo2**" in result


def test_ignore_tool_call_invocations():
    tool_call_json = json.dumps({
        "name": "search_repositories",
        "arguments": {"q": "user:Chantha-123"}
    })
    result = format_json_payload_to_markdown(tool_call_json)
    assert result == tool_call_json


def test_extract_message_text_integration():
    msg = AIMessage(content=json.dumps([{"name": "CoreX-Frontend"}]))
    text = extract_message_text(msg)
    assert "- **CoreX-Frontend**" in text


def test_format_jira_issues():
    jira_json = json.dumps({
        "issues": [
            {"key": "PROJ-101", "summary": "Fix authentication bug", "status": "In Progress"},
            {"key": "PROJ-102", "fields": {"summary": "Update UI header", "status": {"name": "To Do"}}}
        ]
    })
    result = format_json_payload_to_markdown(jira_json)
    assert "- **PROJ-101** [In Progress]: Fix authentication bug" in result
    assert "- **PROJ-102** [To Do]: Update UI header" in result


def test_format_jira_projects():
    jira_projects = json.dumps({
        "projects": [
            {"key": "CORE", "name": "Core Engine Platform"},
            {"key": "WEB", "name": "Web Dashboard Frontend"}
        ]
    })
    result = format_json_payload_to_markdown(jira_projects)
    assert "- **CORE** (Core Engine Platform)" in result
    assert "- **WEB** (Web Dashboard Frontend)" in result


def test_format_single_jira_issue():
    single_issue = json.dumps({"key": "DEV-42", "summary": "Setup database migration", "status": "Done"})
    result = format_json_payload_to_markdown(single_issue)
    assert "- **DEV-42** [Done]: Setup database migration" in result
