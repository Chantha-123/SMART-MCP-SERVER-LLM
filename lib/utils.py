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

    # Read ID/key (ignore integer database IDs for the 'key' anchor, prioritize string keys like Jira CORE-101)
    key_val = entry.get("key") or fields.get("key")
    key = key_val if isinstance(key_val, str) else None

    name = entry.get("name") or entry.get("full_name") or entry.get("title")
    summary = entry.get("summary") or fields.get("summary") or entry.get("description") or fields.get("description")
    url = entry.get("html_url") or entry.get("browse_url") or entry.get("url") or fields.get("html_url") or fields.get("url")

    raw_status = entry.get("status") or fields.get("status")
    status_str = ""
    if isinstance(raw_status, dict):
        status_str = raw_status.get("name", "")
    elif isinstance(raw_status, str):
        status_str = raw_status

    ident = ""
    if key and name and key != name:
        if url:
            ident = f"**[{key}]({url})** ({name})"
        else:
            ident = f"**{key}** ({name})"
    elif key:
        if url:
            ident = f"**[{key}]({url})**"
        else:
            ident = f"**{key}**"
    elif name:
        if url:
            ident = f"**[{name}]({url})**"
        else:
            ident = f"**{name}**"

    status_part = f" [{status_str}]" if status_str else ""
    
    if ident:
        if summary and summary != name and summary != key:
            return f"- {ident}{status_part}: {summary}"
        else:
            return f"- {ident}{status_part}"
    elif summary:
        return f"- {summary}{status_part}"
    else:
        return f"- {json.dumps(entry)}"


def _find_lists_in_dict(d: dict) -> list[tuple[str, list]]:
    """Recursively search for any non-empty lists inside a dictionary structure."""
    found = []
    for k, v in d.items():
        if isinstance(v, list) and v:
            found.append((k, v))
        elif isinstance(v, dict):
            found.extend(_find_lists_in_dict(v))
    return found


def format_json_payload_to_markdown(text: str) -> str:
    """If text contains a raw JSON payload (e.g. array of repos/issues or dict of items),
    format it as a clean Markdown list.
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
            if items:
                return (header + "\n\n" if header else "") + "\n".join(items)

        elif isinstance(data, dict):
            lists = _find_lists_in_dict(data)
            if lists:
                all_formatted = []
                for key_name, lst in lists:
                    list_items = []
                    for entry in lst:
                        if isinstance(entry, dict):
                            list_items.append(_format_dict_entry(entry))
                        elif isinstance(entry, (str, int, float)):
                            list_items.append(f"- **{entry}**")
                        else:
                            list_items.append(f"- {entry}")
                    if list_items:
                        header_title = f"The {key_name} are:"
                        all_formatted.append(header_title + "\n\n" + "\n".join(list_items))
                if all_formatted:
                    return "\n\n".join(all_formatted)

            # If no lists were found, check if it's a single item dictionary
            if "key" in data or "summary" in data or "fields" in data or "name" in data:
                return "Details:\n\n" + _format_dict_entry(data)
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
