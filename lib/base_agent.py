import os
from typing import Any, cast

from pydantic import PrivateAttr
from langchain_core.tools import BaseTool
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

from .state import RuntimeState
from .utils import (
    build_connection_config,
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
        return self._original_tool.invoke(sanitized_kwargs)

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        schema = self._get_effective_schema()
        sanitized_kwargs = sanitize_args(kwargs, schema)
        return await self._original_tool.ainvoke(sanitized_kwargs)


def _matches_tool_name(tool_name: str, enabled_names: set[str]) -> bool:
    low = tool_name.lower()
    san = sanitize_tool_name(low)
    if low in enabled_names or san in enabled_names:
        return True

    # Strip common prefixes (e.g. git_, github_, jira_, slack_, tg_, telegram_, google_chat_, gchat_)
    stripped = low
    for prefix in ("git_", "github_", "jira_", "slack_", "tg_", "telegram_", "google_chat_", "gchat_"):
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix):]
            break
    if stripped in enabled_names:
        return True

    for en in enabled_names:
        if en in low or en in san or low in en or san in en:
            return True
    return False


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
        """Initialize the agent with MCP client and tools."""
        self.validate_environment()
        
        connection = self.get_connection_config()
        state.mcp_client = MultiServerMCPClient({self.service_name: connection})
        
        client = state.mcp_client
        if client is None:
            raise RuntimeError("Failed to initialize MCP client.")
        
        # Get tools from MCP client
        tools = await client.get_tools()
        _sanitize_tool_schemas(tools)

        # Filter tools based on enabled list to support low token-limit providers (like Groq free tier)
        enabled_str = os.getenv(f"{self.service_name.upper()}_ENABLED_TOOLS") or os.getenv("ENABLED_TOOLS")
        if not enabled_str and os.getenv("LLM_PROVIDER") in ("groq", "ollama"):
            default_groq_tools = {
                "github": "git_search_code,search_code,git_get_issue,get_issue,git_search_repositories,search_repositories",
                "slack": "conversations_history,conversations_add_message",
                "telegram": "tg_dialogs,tg_send",
                "jira": "jira_search,jira_get_issue",
                "google-chat": "list_spaces,send_message",
                "web-reader": "fetch"
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
            "If none of the available tools fit the request, answer directly in plain text without making any tool calls."
            f"{extra_instructions}"
        )

        # Create React agent with model, tools, and system prompt
        state.agent_executor = create_react_agent(llm, tools, prompt=system_prompt)


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
    """Ensure agent is initialized, initializing if necessary."""
    if state.agent_executor is None or state.mcp_client is None:
        await initialize_agent(state, agent)
    if state.agent_executor is None or state.mcp_client is None:
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

    # Use ainvoke for async execution with the agent
    result = await executor.ainvoke({"messages": sanitized_history})
    
    # Extract the last AI message from the result
    messages = result.get("messages", [])
    last_ai_message: AIMessage | None = None
    
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            last_ai_message = message
            break
    
    if last_ai_message is None:
        raise RuntimeError("The agent did not return a response.")
    return last_ai_message

