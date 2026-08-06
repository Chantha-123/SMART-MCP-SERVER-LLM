import json
import os
import sys
import re
from typing import Any, Callable

from langchain_core.messages import BaseMessage

__all__ = [
    "sanitize_tool_name",
    "truthy",
    "load_json_env",
    "build_connection_config",
    "schema_from_model",
    "extract_message_text",
    "_eprint",
]


def sanitize_tool_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", name.lower()).strip("_")


def truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def load_json_env(
    env_name: str, *, value_validator: Callable[[Any], Any] | None = None
) -> dict[str, str]:
    raw = os.getenv(env_name)
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Environment variable {env_name} must contain valid JSON."
        ) from exc
    if not isinstance(parsed, dict):
        raise ValueError(
            (
                f"Environment variable {env_name} must be a JSON object "
                "with string keys."
            )
        )
    if value_validator is None:
        return {str(k): str(v) for k, v in parsed.items()}
    validated: dict[str, str] = {}
    for key, value in parsed.items():
        validated[str(key)] = str(value_validator(value))
    return validated


def build_connection_config(
    service_name: str = "github",
    server_url_env: str = "GITHUB_MCP_SERVER_URL",
    default_server_url: str | None = None,
    token_env: str = "GITHUB_PERSONAL_ACCESS_TOKEN",
    bearer_token_env: str | None = "GITHUB_MCP_BEARER_TOKEN",
) -> dict[str, Any]:
    """
    Build connection configuration for MCP services.
    
    Args:
        service_name: Name of the service (for logging/display)
        server_url_env: Environment variable name for server URL
        default_server_url: Default server URL if not specified
        token_env: Primary token environment variable name
        bearer_token_env: Optional bearer token environment variable name
    """
    server_url = os.getenv(server_url_env)
    if server_url is None:
        if default_server_url is None:
            raise ValueError(
                f"{server_url_env} must be set when no default server URL is configured for {service_name}."
            )
        server_url = default_server_url
    server_url = server_url.strip()
    if not server_url:
        raise ValueError(f"{server_url_env} cannot be empty.")

    # Check for readonly path configuration
    readonly_path_env = f"{service_name.upper()}_MCP_USE_READONLY_PATH"
    if truthy(os.getenv(readonly_path_env)) and not server_url.endswith("/readonly"):
        server_url = server_url.rstrip("/") + "/readonly"

    transport_env = f"{service_name.upper()}_MCP_TRANSPORT"
    transport = (
        os.getenv(transport_env, "streamable_http").strip().lower()
    )
    if transport not in {"streamable_http", "sse"}:
        raise ValueError(
            f"Unsupported {transport_env}. Use 'streamable_http' or 'sse'."
        )

    headers: dict[str, str] = {}
    auth_token = os.getenv(bearer_token_env) if bearer_token_env else None
    if not auth_token:
        auth_token = os.getenv(token_env)
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    # Check for toolsets configuration
    toolsets_env = f"{service_name.upper()}_MCP_TOOLSETS"
    toolsets = os.getenv(toolsets_env)
    if toolsets:
        headers["X-MCP-Toolsets"] = toolsets.strip()

    # Check for readonly flag
    readonly_env = f"{service_name.upper()}_MCP_READONLY"
    if truthy(os.getenv(readonly_env)):
        headers["X-MCP-Readonly"] = "true"

    # Check for user agent
    user_agent_env = f"{service_name.upper()}_MCP_USER_AGENT"
    user_agent = os.getenv(user_agent_env)
    if user_agent:
        headers["User-Agent"] = user_agent.strip()

    # Check for extra headers
    extra_headers_env = f"{service_name.upper()}_MCP_EXTRA_HEADERS"
    headers.update(load_json_env(extra_headers_env))

    connection: dict[str, Any] = {"url": server_url, "transport": transport}
    if headers:
        connection["headers"] = headers

    timeout_env = f"{service_name.upper()}_MCP_TIMEOUT_SECONDS"
    timeout = os.getenv(timeout_env)
    if timeout:
        try:
            timeout_value = float(timeout)
        except ValueError as exc:
            raise ValueError(
                f"{timeout_env} must be a positive number"
            ) from exc
        if timeout_value <= 0:
            raise ValueError(
                f"{timeout_env} must be greater than zero"
            )
        connection["timeout"] = timeout_value

    return connection


def schema_from_model(model: Any) -> dict[str, Any] | None:
    if model is None:
        return None
    if isinstance(model, dict):
        return model
    for attr in ("model_json_schema", "schema"):
        schema_fn = getattr(model, attr, None)
        if callable(schema_fn):
            try:
                schema = schema_fn()
                if isinstance(schema, dict):
                    return schema
            except Exception:
                pass
    return None


def _format_dict_entry(entry: dict) -> str:
    """Helper to format a single dict item (e.g. Jira issue, Jira project, or GitHub repo) into Markdown."""
    fields = entry.get("fields") if isinstance(entry.get("fields"), dict) else {}

    key = entry.get("key") or entry.get("id") or fields.get("key")
    name = entry.get("name") or entry.get("full_name") or entry.get("title") or fields.get("summary")
    summary = entry.get("summary") or fields.get("summary") or entry.get("description") or fields.get("description")

    raw_status = entry.get("status") or fields.get("status")
    status_str = ""
    if isinstance(raw_status, dict):
        status_str = raw_status.get("name", "")
    elif isinstance(raw_status, str):
        status_str = raw_status

    if key and summary and key != summary:
        status_part = f" [{status_str}]" if status_str else ""
        return f"- **{key}**{status_part}: {summary}"
    elif key and name and key != name:
        status_part = f" [{status_str}]" if status_str else ""
        return f"- **{key}** ({name}){status_part}"
    elif name:
        desc = entry.get("description") or (summary if summary != name else None)
        status_part = f" [{status_str}]" if status_str else ""
        return f"- **{name}**{status_part}" + (f": {desc}" if desc else "")
    elif key:
        status_part = f" [{status_str}]" if status_str else ""
        return f"- **{key}**{status_part}"
    else:
        return f"- {json.dumps(entry)}"


def format_json_payload_to_markdown(text: str) -> str:
    """If text contains a raw JSON payload (e.g. array of repos/issues or dict of items),
    format it as a clean Markdown list unless it is a tool call invocation.
    """
    if not isinstance(text, str) or not text.strip():
        return text

    clean_text = text.strip()
    if clean_text.startswith("```"):
        clean_text = re.sub(r"^```(?:json)?\n?", "", clean_text)
        clean_text = re.sub(r"\n?```$", "", clean_text).strip()

    if not ((clean_text.startswith("[") and clean_text.endswith("]")) or (clean_text.startswith("{") and clean_text.endswith("}"))):
        return text

    try:
        data = json.loads(clean_text)
        # Skip if it is a tool call invocation dictionary
        if isinstance(data, dict) and ("name" in data or "tool" in data or "action" in data) and ("arguments" in data or "parameters" in data or "args" in data):
            return text

        items: list[str] = []
        header = ""

        if isinstance(data, list):
            for entry in data:
                if isinstance(entry, dict):
                    items.append(_format_dict_entry(entry))
                elif isinstance(entry, (str, int, float)):
                    items.append(f"- **{entry}**")
                else:
                    items.append(f"- {entry}")
            header = "The items are:"
        elif isinstance(data, dict):
            # Check for standard list container keys (e.g. issues, projects, values, items, repositories)
            for k in ("issues", "projects", "repositories", "values", "items", "results"):
                v = data.get(k)
                if isinstance(v, list):
                    header = f"The {k} are:"
                    for entry in v:
                        if isinstance(entry, dict):
                            items.append(_format_dict_entry(entry))
                        elif isinstance(entry, (str, int, float)):
                            items.append(f"- **{entry}**")
                        else:
                            items.append(f"- {entry}")
                    break

            if not items:
                # Single item dictionary (e.g. single Jira issue or project detail)
                if "key" in data or "summary" in data or "fields" in data:
                    header = "Details:"
                    items.append(_format_dict_entry(data))

        if items:
            return (header + "\n\n" if header else "") + "\n".join(items)
    except Exception:
        pass

    return text


def extract_message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text_chunks = [
            c.get("text", "")
            for c in content
            if isinstance(c, dict) and c.get("type") == "text"
        ]
        text = "\n".join(text_chunks) if text_chunks else str(content)
    else:
        text = str(content)

    return format_json_payload_to_markdown(text)


def _eprint(
    *args: object,
    sep: str | None = None,
    end: str | None = None,
    flush: bool = False,
) -> None:

    print(*args, file=sys.stderr, sep=sep, end=end, flush=flush)
