import os
import json
import re
import logging
from typing import Any, cast

from pydantic import PrivateAttr
from langchain_core.tools import BaseTool
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

logger = logging.getLogger(__name__)


def parse_raw_json_tool_call(content: str) -> tuple[str, dict[str, Any]] | None:
    if not isinstance(content, str) or not content.strip():
        return None

    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            name = data.get("name") or data.get("tool") or data.get("action")
            args = data.get("arguments") or data.get("parameters") or data.get("args") or {}
            if isinstance(name, str) and isinstance(args, dict):
                return name, args
    except Exception:
        pass
    return None

from .mcp_session import PersistentMCPSession
from .state import RuntimeState
from .utils import (
    build_connection_config,
    format_csv_payload_to_markdown,
    format_json_payload_to_markdown,
    sanitize_tool_name,
    schema_from_model,
)


def _is_missing_env_value(value: str | None) -> bool:
    if value is None:
        return True

    normalized = value.strip()
    if not normalized:
        return True

    lower_value = normalized.lower()
    placeholder_markers = (
        "your_",
        "changeme",
        "replace_me",
        "example",
        "placeholder",
        "api_id_here",
        "api_hash_here",
        "token_here",
        "your_api_id_here",
        "your_api_hash_here",
        "your_token_here",
    )
    return any(marker in lower_value for marker in placeholder_markers)


__all__ = [
    "BaseAgent",
    "initialize_agent",
    "ensure_agent_initialized",
    "stream_agent_response",
]


def _get_llm_provider():
    from llm_providers import gemini
    from llm_providers import groq_llm
    from llm_providers import openai
    from llm_providers import ollama

    provider = os.getenv("LLM_PROVIDER", "openai").lower()

    if provider == "openai":
        return openai.get_llm
    elif provider == "groq":
        return groq_llm.get_llm
    elif provider == "gemini":
        return gemini.get_llm
    elif provider == "ollama":
        return ollama.get_llm
    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER: {provider}. "
            f"Supported values: 'openai', 'groq', 'gemini', 'ollama'"
        )


def _sanitize_schema(d: Any) -> Any:
    if isinstance(d, dict):
        new_dict = {}
        for k, v in d.items():
            if k == "enum" and isinstance(v, list):
                new_dict[k] = [str(x) if not isinstance(x, str) else x for x in v]
            else:
                new_dict[k] = _sanitize_schema(v)
        return new_dict
    elif isinstance(d, list):
        return [_sanitize_schema(x) for x in d]
    return d


def _sanitize_tool_schemas(tools: list[Any]) -> None:
    for tool in tools:
        schema = getattr(tool, "args_schema", None)
        if schema is not None:
            if isinstance(schema, dict):
                sanitized = _sanitize_schema(schema)
                schema.clear()
                schema.update(sanitized)
            else:
                if hasattr(schema, "model_json_schema"):
                    original_method = schema.model_json_schema
                    schema.model_json_schema = classmethod(
                        lambda cls, *args, **kwargs: _sanitize_schema(original_method(*args, **kwargs))
                    )
                if hasattr(schema, "schema"):
                    original_method = schema.schema
                    schema.schema = classmethod(
                        lambda cls, *args, **kwargs: _sanitize_schema(original_method(*args, **kwargs))
                    )


def get_schema_enum(prop_schema: Any) -> list[Any] | None:
    if not isinstance(prop_schema, dict):
        return None
    if "enum" in prop_schema:
        return prop_schema["enum"]
    for combiner in ("anyOf", "oneOf"):
        if combiner in prop_schema and isinstance(prop_schema[combiner], list):
            for sub_schema in prop_schema[combiner]:
                if isinstance(sub_schema, dict) and "enum" in sub_schema:
                    return sub_schema["enum"]
    return None


def get_schema_type(prop_schema: Any) -> str | None:
    if not isinstance(prop_schema, dict):
        return None
    if "type" in prop_schema and isinstance(prop_schema["type"], str):
        return prop_schema["type"]
    for combiner in ("anyOf", "oneOf"):
        if combiner in prop_schema and isinstance(prop_schema[combiner], list):
            for sub_schema in prop_schema[combiner]:
                if isinstance(sub_schema, dict) and "type" in sub_schema and isinstance(sub_schema["type"], str):
                    return sub_schema["type"]
    return None


def sanitize_args(args: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(schema, dict) or "properties" not in schema:
        return args
    properties = schema["properties"]
    sanitized_args = {}
    for key, value in args.items():
        if key not in properties:
            sanitized_args[key] = value
            continue
        prop_schema = properties[key]
        if isinstance(prop_schema, dict):
            prop_type = get_schema_type(prop_schema)
            if prop_type in ("integer", "number"):
                if isinstance(value, float) and value.is_integer():
                    value = int(value)
                elif isinstance(value, str):
                    try:
                        val_float = float(value)
                        if val_float.is_integer():
                            value = int(val_float)
                        else:
                            value = val_float
                    except (ValueError, TypeError):
                        pass
            elif prop_type == "boolean":
                if isinstance(value, str):
                    if value.lower() in ("true", "1"):
                        value = True
                    elif value.lower() in ("false", "0"):
                        value = False

            allowed_values = get_schema_enum(prop_schema)
            if allowed_values is not None:
                if value not in allowed_values:
                    # Omit parameter to fall back to the default behavior
                    continue
        sanitized_args[key] = value
    return sanitized_args


class SanitizedTool(BaseTool):
    _original_tool: BaseTool = PrivateAttr()
    _schema_dict: dict[str, Any] = PrivateAttr()

    def __init__(self, original_tool: BaseTool, schema_dict: dict[str, Any], **kwargs: Any):
        super().__init__(
            name=original_tool.name,
            description=original_tool.description,
            args_schema=original_tool.args_schema,
            # content goes to the model; the artifact keeps the untruncated result
            # (used to answer listing questions directly, see _direct_list_answer)
            response_format="content_and_artifact",
            **kwargs
        )
        self._original_tool = original_tool
        self._schema_dict = schema_dict

    def _get_effective_schema(self) -> dict[str, Any]:
        schema = self._schema_dict or {}
        if not isinstance(schema, dict) or "properties" not in schema:
            schema = schema_from_model(getattr(self._original_tool, "args_schema", None)) or {}
        if not isinstance(schema, dict) or "properties" not in schema:
            tool_args = getattr(self._original_tool, "args", None)
            if isinstance(tool_args, dict):
                schema = {"properties": tool_args}
        return schema or {}

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        schema = self._get_effective_schema()
        sanitized_kwargs = sanitize_args(kwargs, schema)
        return self._original_tool.invoke(sanitized_kwargs), None

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        schema = self._get_effective_schema()
        sanitized_kwargs = sanitize_args(kwargs, schema)
        result = await self._original_tool.ainvoke(sanitized_kwargs)
        if _is_local_provider():
            result = compact_tool_output(result)
        return truncate_tool_output(result, _local_tool_output_limit()), result


def _is_local_provider() -> bool:
    return os.getenv("LLM_PROVIDER", "").lower() == "ollama"


def _local_tool_output_limit() -> int | None:
    """Max UTF-8 bytes of tool output fed back to a local model (None = unlimited)."""
    if not _is_local_provider():
        return None
    return int(os.getenv("LOCAL_TOOL_OUTPUT_MAX_BYTES") or 4000)


_TRUNCATION_NOTE = "\n\n[Tool output truncated to fit the local model context]"


_OMITTED_KEY = "omitted_items"


def _json_size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _largest_list(value: Any) -> list[Any] | None:
    """The biggest list in a JSON value (top level or nested in objects)."""
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        candidates = [found for item in value.values() if (found := _largest_list(item)) is not None]
        return max(candidates, key=_json_size, default=None)
    return None


def _clip_json(text: str, max_bytes: int) -> str | None:
    """Shrink a JSON payload by dropping whole items from its largest list.

    Unlike a byte cut, the result stays valid JSON (readable by the model and by the
    Markdown formatter) and says how many items were left out. Returns None when the
    text isn't JSON or can't be shrunk this way.
    """
    stripped = text.strip()
    if not stripped.startswith(("{", "[")):
        return None
    try:
        data = json.loads(stripped)
    except ValueError:
        return None
    items = _largest_list(data)
    if not items:
        return None
    omitted = 0
    while len(items) > 1 and _json_size(data) > max_bytes:
        items.pop()
        omitted += 1
    if _json_size(data) > max_bytes:
        return None
    if omitted:
        note = f"{omitted} more not shown"
        if isinstance(data, dict):
            data[_OMITTED_KEY] = note
        else:
            data.append({_OMITTED_KEY: note})
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def _clip_text(text: str, max_bytes: int) -> tuple[str, int]:
    """Clip text to max_bytes of UTF-8, returning (text, bytes used)."""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text, len(encoded)
    clipped_json = _clip_json(text, max_bytes)
    if clipped_json is not None:
        return clipped_json, len(clipped_json.encode("utf-8"))
    clipped = encoded[:max_bytes].decode("utf-8", errors="ignore")
    return clipped + _TRUNCATION_NOTE, max_bytes


def truncate_tool_output(result: Any, max_bytes: int | None) -> Any:
    """Truncate large tool results so they fit a local model's context window.

    Every token of tool output is re-read by the model on each following step, and
    prompt processing dominates latency on CPU-only Ollama. The budget is in UTF-8
    bytes because non-Latin scripts cost far more tokens per character than English.
    """
    if not max_bytes:
        return result
    if isinstance(result, str):
        return _clip_text(result, max_bytes)[0]
    if isinstance(result, list):
        remaining = max_bytes
        truncated: list[Any] = []
        for block in result:
            text = block.get("text") if isinstance(block, dict) else block if isinstance(block, str) else None
            if text is None:
                truncated.append(block)
                continue
            if remaining <= 0:
                break
            text, used = _clip_text(text, remaining)
            remaining -= used
            truncated.append({**block, "text": text} if isinstance(block, dict) else text)
        return truncated
    return result


# Fields that cost tokens but never help answer a question
_NOISE_KEYS = {
    "avatar_url", "avatar_urls", "account_id", "node_id", "gravatar_id",
    "color", "self", "icon_url", "icon_urls", "incomplete_results",
}
_WRAPPER_KEYS = {"name", "display_name", "value"}
_MAX_JSON_STRING_CHARS = 400


def _compact_json(value: Any) -> Any:
    if isinstance(value, dict):
        compact: dict[str, Any] = {}
        for key, item in value.items():
            if key in _NOISE_KEYS:
                continue
            item = _compact_json(item)
            if item is None or item == "" or item == [] or item == {}:
                continue
            compact[key] = item
        # {"display_name": "A", "name": "A"} -> drop the duplicate
        if "display_name" in compact and compact.get("name") == compact["display_name"]:
            del compact["name"]
        # {"name": "Medium"} -> "Medium" (only name-like wrappers, so keys such as
        # {"issues": [...]} keep their meaning)
        if len(compact) == 1 and next(iter(compact)) in _WRAPPER_KEYS:
            return next(iter(compact.values()))
        return compact
    if isinstance(value, list):
        return [_compact_json(item) for item in value]
    if isinstance(value, str) and len(value) > _MAX_JSON_STRING_CHARS:
        return value[:_MAX_JSON_STRING_CHARS] + "…"
    return value


_EMPTY_RESULT = "No results found (the tool returned an empty list)."
_CSV_HEADER = re.compile(r"^[A-Za-z_][A-Za-z0-9_ ]*(,[A-Za-z_][A-Za-z0-9_ ]*)+$")


def _all_lists(value: Any) -> list[list[Any]]:
    if isinstance(value, list):
        return [value]
    if isinstance(value, dict):
        return [found for item in value.values() for found in _all_lists(item)]
    return []


def _compact_text(text: str) -> str:
    stripped = text.strip()
    # A CSV with only its header row (e.g. an empty Slack channel). Small models tend
    # to invent rows for it, so say explicitly that there is nothing.
    if "\n" not in stripped and _CSV_HEADER.match(stripped):
        return _EMPTY_RESULT
    if not stripped.startswith(("{", "[")):
        return text
    try:
        data = json.loads(stripped)
    except ValueError:
        return text
    lists = _all_lists(data)
    if lists and all(not items for items in lists):
        return _EMPTY_RESULT
    return json.dumps(_compact_json(data), ensure_ascii=False, separators=(",", ":"))


def compact_tool_output(result: Any) -> Any:
    """Minify JSON tool results and drop noise fields before a local model reads them.

    Prompt processing is the bottleneck on CPU-only Ollama, and API payloads are
    mostly indentation, avatar URLs, IDs and empty fields.
    """
    if isinstance(result, str):
        return _compact_text(result)
    if isinstance(result, list):
        return [
            {**block, "text": _compact_text(block["text"])}
            if isinstance(block, dict) and isinstance(block.get("text"), str)
            else _compact_text(block) if isinstance(block, str) else block
            for block in result
        ]
    return result


_WRITE_REQUEST = re.compile(
    r"\b(send|post|reply|create|update|delete|remove|assign|comment|transition|close|edit|write)\b",
    re.IGNORECASE,
)
_WRITE_TOOL = re.compile(
    r"(add|create|update|delete|remove|send|post|assign|transition|comment|write|edit|move|link)",
    re.IGNORECASE,
)


def _last_human_text(history: list[BaseMessage]) -> str:
    for message in reversed(history):
        if isinstance(message, HumanMessage):
            return message.content if isinstance(message.content, str) else ""
    return ""


def _needs_write_tool(history: list[BaseMessage], state: RuntimeState) -> bool:
    """True if the user asks for an action and the agent has a tool that can do it."""
    if not _WRITE_REQUEST.search(_last_human_text(history)):
        return False
    return any(_WRITE_TOOL.search(name) for name in state.tool_map)


def _called_write_tool(messages: list[BaseMessage], history_len: int) -> bool:
    """Whether this turn ran a write tool (its result, success or error, is in the messages)."""
    return any(
        isinstance(message, ToolMessage) and _WRITE_TOOL.search(message.name or "")
        for message in messages[history_len:]
    )


_LISTING_REQUEST = re.compile(r"\b(list|show|search|find|get all|display)\b", re.IGNORECASE)


def _direct_list_answers_enabled() -> bool:
    return os.getenv("LOCAL_DIRECT_LIST_ANSWERS", "true").lower() not in ("0", "false", "no")


def _asks_for_listing(history: list[BaseMessage]) -> bool:
    for message in reversed(history):
        if isinstance(message, HumanMessage):
            return isinstance(message.content, str) and bool(_LISTING_REQUEST.search(message.content))
    return False


def _pop_omitted_note(text: str) -> tuple[str, str | None]:
    """Split the "N more not shown" marker added by _clip_json from a JSON payload."""
    try:
        data = json.loads(text)
    except ValueError:
        return text, None
    note = None
    if isinstance(data, dict) and _OMITTED_KEY in data:
        note = data.pop(_OMITTED_KEY)
    elif isinstance(data, list) and data and isinstance(data[-1], dict) and list(data[-1]) == [_OMITTED_KEY]:
        note = data.pop()[_OMITTED_KEY]
    if note is None:
        return text, None
    return json.dumps(data, ensure_ascii=False, separators=(",", ":")), str(note)


def _direct_list_answer(messages: list[BaseMessage], history_len: int) -> AIMessage | None:
    """Answer a listing question straight from the tool result, skipping the second LLM call.

    On CPU a local model needs ~1 minute to re-read a list payload and rewrite it as
    Markdown; the deterministic formatter does the same in milliseconds. Returns None
    unless every tool result of this turn renders as a Markdown list.
    """
    new_messages = messages[history_len:]
    if not new_messages or not isinstance(new_messages[-1], ToolMessage):
        return None
    tool_results: list[ToolMessage] = []
    for message in reversed(new_messages):
        if not isinstance(message, ToolMessage):
            break
        tool_results.append(message)

    sections = []
    for tool_message in reversed(tool_results):
        # Prefer the untruncated result: formatting it costs no model tokens
        artifact = getattr(tool_message, "artifact", None)
        content = artifact if artifact is not None else tool_message.content
        if isinstance(content, list):
            content = "\n".join(
                block.get("text", "") if isinstance(block, dict) else str(block) for block in content
            )
        if not isinstance(content, str):
            return None
        text, omitted_note = _pop_omitted_note(content)
        formatted = format_csv_payload_to_markdown(text)
        if formatted == text:
            formatted = format_json_payload_to_markdown(text)
        # Only lists ("The <items> are:"); errors, prose and single-item details go to the LLM
        if formatted == text or not formatted.startswith("The "):
            return None
        if omitted_note:
            formatted += f"\n\n_…and {omitted_note}. Ask a narrower question to see them._"
        sections.append(formatted)
    return AIMessage(content="\n\n".join(sections))


def trim_session_history(history: list[BaseMessage], max_turns: int) -> list[BaseMessage]:
    """Keep only the last `max_turns` user turns (each with its tool calls and answers).

    Cutting at a HumanMessage boundary keeps AI tool calls paired with their ToolMessages.
    """
    if max_turns <= 0:
        return history
    human_indexes = [i for i, msg in enumerate(history) if isinstance(msg, HumanMessage)]
    if len(human_indexes) <= max_turns:
        return history
    return history[human_indexes[-max_turns]:]


_TOOL_NAME_PREFIXES = ("git_", "github_", "jira_", "slack_")


def _strip_tool_prefix(name: str) -> str:
    for prefix in _TOOL_NAME_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def _matches_tool_name(tool_name: str, enabled_names: set[str]) -> bool:
    low = tool_name.lower()
    san = sanitize_tool_name(low)
    if low in enabled_names or san in enabled_names:
        return True

    # Compare with service prefixes (git_, jira_, slack_, ...) stripped on both sides.
    # Exact match only: substring matching let "jira_get_issue" enable every
    # jira_get_issue_* tool, bloating the prompt for small local models.
    stripped = _strip_tool_prefix(san)
    return any(stripped == _strip_tool_prefix(en) for en in enabled_names)


_RESPONSE_TIME_LINE = re.compile(r"\n*⏱️ \*Response time: [0-9.]+ seconds\*")


def sanitize_session_history(
    session_history: list[BaseMessage],
    valid_tool_names: set[str],
) -> list[BaseMessage]:
    """Sanitize session history by removing or converting tool calls and tool messages
    that reference tools not in valid_tool_names (preventing Groq/OpenAI validation errors).
    """
    sanitized: list[BaseMessage] = []

    for msg in session_history:
        if isinstance(msg, AIMessage):
            raw_tool_calls = getattr(msg, "tool_calls", None)
            if raw_tool_calls:
                valid_calls = []
                for call in raw_tool_calls:
                    call_name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
                    if call_name and call_name in valid_tool_names:
                        valid_calls.append(call)

                if len(valid_calls) != len(raw_tool_calls):
                    content = msg.content if isinstance(msg.content, str) else str(msg.content or "")
                    if not content and not valid_calls:
                        content = "[Previous tool call to unlisted tool was omitted]"
                    sanitized.append(AIMessage(content=content, tool_calls=valid_calls))
                else:
                    sanitized.append(msg)
            elif isinstance(msg.content, str) and _RESPONSE_TIME_LINE.search(msg.content):
                # Sessions saved by older versions include the UI timing line; models copy it
                sanitized.append(AIMessage(content=_RESPONSE_TIME_LINE.sub("", msg.content).rstrip()))
            else:
                sanitized.append(msg)
        elif isinstance(msg, ToolMessage):
            tool_name = getattr(msg, "name", None)
            if tool_name and tool_name not in valid_tool_names:
                text = f"[Result for unlisted tool {tool_name}]: {msg.content}"
                sanitized.append(HumanMessage(content=text))
            else:
                sanitized.append(msg)
        else:
            sanitized.append(msg)

    return sanitized


class BaseAgent:
    """Base class for MCP agents providing common functionality."""

    def __init__(
        self,
        service_name: str,
        required_env_vars: list[str] | None = None,
        server_url_env: str | None = None,
        default_server_url: str | None = None,
        token_env: str | None = None,
        bearer_token_env: str | None = None,
    ):
        """Initialize base agent with service-specific configuration.
        
        Args:
            service_name: Name of the service (e.g., "github", "jira", "slack")
            required_env_vars: List of required environment variables
            server_url_env: Environment variable name for server URL
            default_server_url: Default server URL if not specified
            token_env: Primary token environment variable name
            bearer_token_env: Optional bearer token environment variable name
        """
        self.service_name = service_name
        self.required_env_vars = required_env_vars or []
        self.server_url_env = server_url_env
        self.default_server_url = default_server_url
        self.token_env = token_env
        self.bearer_token_env = bearer_token_env

    def validate_environment(self) -> None:
        """Validate required environment variables."""
        missing_vars: list[str] = []
        for var in self.required_env_vars:
            if _is_missing_env_value(os.getenv(var)):
                missing_vars.append(var)

        if missing_vars:
            missing_str = ", ".join(sorted(set(missing_vars)))
            raise ValueError(
                "Missing required environment variables: "
                + missing_str
                + ". Add them to your .env file or export them in your shell."
            )

    def get_connection_config(self) -> dict[str, Any]:
        """Get the connection configuration for this service."""
        return build_connection_config(
            service_name=self.service_name,
            server_url_env=self.server_url_env or f"{self.service_name.upper()}_MCP_SERVER_URL",
            default_server_url=self.default_server_url,
            token_env=self.token_env or f"{self.service_name.upper()}_PERSONAL_ACCESS_TOKEN",
            bearer_token_env=self.bearer_token_env,
        )
    
    async def initialize(self, state: RuntimeState) -> None:
        """Connect to the MCP server (if needed) and build the agent executor."""
        await self.connect(state)
        self.build_executor(state)

    async def connect(self, state: RuntimeState) -> None:
        """Open a persistent MCP session and load its tools. No-op if already connected."""
        if state.mcp_session is not None and state.mcp_session.alive:
            return
        await self.disconnect(state)
        self.validate_environment()

        session = PersistentMCPSession(self.service_name, self.get_connection_config())
        tools = await session.start()
        _sanitize_tool_schemas(tools)
        state.mcp_session = session
        state.mcp_client = session.client
        state.mcp_tools = tools

    async def disconnect(self, state: RuntimeState) -> None:
        if state.mcp_session is not None:
            await state.mcp_session.aclose()
        state.mcp_session = None
        state.mcp_client = None
        state.mcp_tools = []
        state.agent_executor = None

    def build_executor(self, state: RuntimeState) -> None:
        """Build the LLM agent over the connected tools (cheap; redone when the LLM changes)."""
        tools = list(state.mcp_tools)

        # Filter tools based on enabled list to support low token-limit providers (like Groq free tier)
        enabled_str = os.getenv(f"{self.service_name.upper()}_ENABLED_TOOLS") or os.getenv("ENABLED_TOOLS")
        if not enabled_str and os.getenv("LLM_PROVIDER") in ("groq", "ollama"):
            default_groq_tools = {
                "github": "git_search_code,search_code,git_get_issue,get_issue,issue_read,git_search_repositories,search_repositories",
                "slack": "conversations_history,conversations_add_message",
                "jira": "jira_search,jira_get_issue,jira_get_all_projects",
                "web-reader": "fetch_web_page"
            }
            enabled_str = default_groq_tools.get(self.service_name)

        if enabled_str:
            enabled_names = {name.strip().lower() for name in enabled_str.split(",") if name.strip()}
            filtered_tools = []
            for tool in tools:
                original_name = tool.name
                if _matches_tool_name(original_name, enabled_names):
                    filtered_tools.append(tool)
            if filtered_tools:
                tools = filtered_tools
        
        state.tool_summaries = []
        state.tool_map = {}
        state.tool_details = {}
        
        wrapped_tools = []
        for tool in tools:
            original_name = tool.name
            sanitized_name = sanitize_tool_name(original_name)
            
            args_schema = schema_from_model(getattr(tool, "args_schema", None))
            if not isinstance(args_schema, dict) or "properties" not in args_schema:
                tool_args = getattr(tool, "args", None)
                if isinstance(tool_args, dict):
                    args_schema = {"properties": tool_args}
            metadata = getattr(tool, "metadata", {}) or {}
            
            # Wrap the tool in SanitizedTool to handle model parameter hallucinations (like sort enums on Groq)
            wrapped_tool = SanitizedTool(tool, args_schema or {})
            wrapped_tools.append(wrapped_tool)
            
            state.tool_map[sanitized_name] = wrapped_tool
            state.tool_details[sanitized_name] = {
                "name": sanitized_name,
                "original_name": original_name,
                "description": getattr(tool, "description", ""),
                "metadata": metadata,
                "args_schema": args_schema,
            }
            state.tool_summaries.append(
                {
                    "name": sanitized_name,
                    "original_name": original_name,
                    "description": getattr(tool, "description", ""),
                }
            )
        tools = wrapped_tools
        
        # Get LLM and create agent with tools
        get_llm = _get_llm_provider()
        llm = get_llm()
        
        valid_tool_names = list(state.tool_map.keys())
        tool_names_str = ", ".join(valid_tool_names) if valid_tool_names else "None"
        extra_instructions = ""
        if self.service_name == "github":
            extra_instructions = (
                "\nGITHUB SEARCH INSTRUCTION: When calling search_repositories or search_code, "
                "always pass a valid GitHub search query string `q` (e.g., `user:Chantha-123` or specific repo keywords). "
                "NEVER use invalid query placeholders like `user:me` or `user:my_username` because GitHub API will reject them with HTTP 422 error."
            )

        system_prompt = (
            f"You are a helpful assistant with access to the following tools: {tool_names_str}.\n"
            "CRITICAL TOOL INSTRUCTION: You MUST ONLY call tools explicitly listed in the available tools above. "
            "Never attempt to call or invent any unlisted tools (such as brave_search, web_search, python, etc.). "
            "If none of the available tools fit the request, answer directly in plain text without making any tool calls.\n"
            "NEVER claim you sent, created, updated or deleted anything unless a tool call in this conversation "
            "actually did it and succeeded. If no available tool can do what the user asks, say so plainly.\n"
            "RESPONSE FORMATTING INSTRUCTION: Always present tool results and lists of items (such as repositories, issues, messages, or channels) to the user using clean human-readable Markdown with bullet points or numbered lists. NEVER output raw JSON objects, JSON arrays, or unformatted API payloads directly as your final answer unless the user explicitly requested JSON."
            f"{extra_instructions}"
        )
        if _is_local_provider():
            # Generation runs at only a few tokens/s on CPU, so answer length is latency
            system_prompt += (
                "\nBE CONCISE: answer in as few words as possible. For lists, use one short line per item "
                "with only the most important fields (name/key, title, status, link). Skip timestamps and IDs unless asked."
            )

        # Create React agent with model, tools, and system prompt
        state.agent_executor = create_react_agent(llm, tools, prompt=system_prompt)
        state.llm = llm
        state.agent_tools = tools
        state.system_prompt = system_prompt

    async def warm_prompt_cache(self, state: RuntimeState) -> None:
        """Pre-fill a local model's prompt cache with this agent's system prompt + tools.

        On CPU, reading ~1,000 tokens of tool schemas takes 30-40 s. Ollama reuses a
        cached prompt prefix, so evaluating it once here makes the first real
        question start from the cache. Generates a single token.
        """
        if not _is_local_provider() or state.llm is None:
            return
        llm = state.llm.bind_tools(state.agent_tools).bind(extra_body={"max_tokens": 1})
        await llm.ainvoke([SystemMessage(content=state.system_prompt), HumanMessage(content="hi")])


async def initialize_agent(
    state: RuntimeState,
    agent: BaseAgent,
) -> None:
    """Async Initialize an agent using the base agent class."""
    await agent.initialize(state)


async def ensure_agent_initialized(
    state: RuntimeState,
    agent: BaseAgent,
) -> None:
    """Ensure agent is initialized, reconnecting only if the MCP session is gone."""
    if state.mcp_session is None or not state.mcp_session.alive:
        await agent.connect(state)
        state.agent_executor = None
    if state.agent_executor is None:
        agent.build_executor(state)
    if state.agent_executor is None:
        raise RuntimeError("Agent failed to initialize")


async def stream_agent_response(
    state: RuntimeState,
    session_history: list[BaseMessage],
) -> AIMessage:
    """Stream agent response for a given session history."""
    if state.agent_executor is None:
        raise RuntimeError("Agent executor is not initialized.")
    
    executor = cast(Any, state.agent_executor)
    
    valid_tool_names = set(state.tool_map.keys())
    if hasattr(state, "tool_details"):
        for details in state.tool_details.values():
            if "original_name" in details:
                valid_tool_names.add(details["original_name"])

    sanitized_history = sanitize_session_history(session_history, valid_tool_names)
    if _is_local_provider():
        # Long histories make every CPU inference step slower; only send recent turns.
        max_turns = int(os.getenv("LOCAL_HISTORY_MAX_TURNS") or 6)
        sanitized_history = trim_session_history(sanitized_history, max_turns)

    direct_lists = _is_local_provider() and _direct_list_answers_enabled() and _asks_for_listing(sanitized_history)

    # Stream graph states so a listing question can stop right after the tool step
    result: dict[str, Any] = {}
    async for result in executor.astream({"messages": sanitized_history}, stream_mode="values"):
        if direct_lists:
            direct = _direct_list_answer(result.get("messages", []), len(sanitized_history))
            if direct is not None:
                return direct

    # Small models sometimes answer "I have sent the message" without calling any tool.
    # For write requests, insist once on a real tool call, and never report a fake success.
    if _needs_write_tool(sanitized_history, state) and not _called_write_tool(result.get("messages", []), len(sanitized_history)):
        logger.info("Write request answered without a write tool call; retrying with an explicit instruction")
        nudge = HumanMessage(content=(
            "You did not call any tool, so nothing was done. Call the appropriate tool now to perform "
            "my previous request exactly. Do not answer in text before the tool has run."
        ))
        retry_history = sanitized_history + [nudge]
        result = await executor.ainvoke({"messages": retry_history})
        if not _called_write_tool(result.get("messages", []), len(retry_history)):
            return AIMessage(content=(
                "I couldn't complete that action: no tool was called, so nothing was sent or changed. "
                "Please try again, for example: Send \"Good morning team\" to #channel-name"
            ))

    # Extract the last AI message from the result
    messages = result.get("messages", [])
    last_ai_message: AIMessage | None = None
    
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            last_ai_message = message
            break
    
    if last_ai_message is None:
        raise RuntimeError("The agent did not return a response.")

    # Fallback for models (e.g. Ollama/Qwen) that output raw JSON tool calls in text content instead of tool_calls field
    if last_ai_message and not getattr(last_ai_message, "tool_calls", None):
        raw_content = getattr(last_ai_message, "content", "")
        if isinstance(raw_content, str) and raw_content.strip():
            parsed = parse_raw_json_tool_call(raw_content)
            if parsed:
                tool_name, tool_args = parsed
                sanitized_name = sanitize_tool_name(tool_name)
                target_tool = state.tool_map.get(sanitized_name) or state.tool_map.get(tool_name)
                if target_tool:
                    logger.info(f"Fallback executing raw JSON tool call from LLM text output: {tool_name}({tool_args})")
                    try:
                        tool_result = await target_tool.ainvoke(tool_args)
                        tool_msg = ToolMessage(content=str(tool_result), name=sanitized_name, tool_call_id="call_fallback_1")
                        fake_ai_call = AIMessage(
                            content="",
                            tool_calls=[{"name": sanitized_name, "args": tool_args, "id": "call_fallback_1"}]
                        )
                        followup_history = sanitized_history + [fake_ai_call, tool_msg]
                        followup_result = await executor.ainvoke({"messages": followup_history})
                        for msg in reversed(followup_result.get("messages", [])):
                            if isinstance(msg, AIMessage):
                                return msg
                    except Exception as e:
                        logger.error(f"Fallback tool execution error: {e}")

    return last_ai_message

