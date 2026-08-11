import os
from pathlib import Path
from typing import Any
from lib.base_agent import BaseAgent
from lib.base_transport import StdioTransportMixin

__all__ = ["get_telegram_agent"]


def get_telegram_agent() -> BaseAgent:
    """Create and return a Telegram MCP agent.

    Requires TG_APP_ID and TG_API_HASH env variables. On first run,
    authentication must be completed once with the Telegram MCP CLI so it can
    create the session file used by the MCP server.
    """
    tg_app_id = os.getenv("TG_APP_ID")
    tg_api_hash = os.getenv("TG_API_HASH")

    if not tg_app_id or not tg_api_hash:
        required_vars = ["TG_APP_ID", "TG_API_HASH"]
    else:
        required_vars = []

    class TelegramStdioAgent(BaseAgent, StdioTransportMixin):
        def validate_environment(self) -> None:
            super().validate_environment()

            session_json_content = os.getenv("TG_SESSION_JSON")
            if session_json_content:
                # Writable temp location on server environments
                temp_dir = Path("/tmp/.telegram-mcp")
                temp_dir.mkdir(parents=True, exist_ok=True)
                session_file = temp_dir / "session.json"
                session_file.write_text(session_json_content.strip(), encoding="utf-8")
                os.environ["TG_SESSION_PATH"] = str(session_file)
            else:
                # Check relative repository root directory first (e.g. for committed session files)
                relative_dir = Path(__file__).resolve().parent.parent / ".telegram-mcp"
                session_path_env = os.getenv("TG_SESSION_PATH")
                if session_path_env:
                    session_dir = Path(session_path_env).expanduser()
                elif (relative_dir / "session.json").exists():
                    session_dir = relative_dir
                else:
                    session_dir = Path("~/.telegram-mcp").expanduser()
                session_file = session_dir / "session.json"

            if not session_file.exists():
                tg_app_id = os.getenv("TG_APP_ID", "<your_tg_app_id>")
                tg_api_hash = os.getenv("TG_API_HASH", "<your_tg_api_hash>")
                raise ValueError(
                    f"Telegram session file not found at {session_file}.\n"
                    f"On first run, you must complete authentication using the Telegram MCP CLI on your local machine.\n"
                    f"Please run the following command in your terminal (replacing <your_phone_number> with your Telegram phone number, including country code, e.g., +1234567890) and follow the prompts:\n\n"
                    f"TG_APP_ID={tg_app_id} TG_API_HASH={tg_api_hash} TG_SESSION_PATH=./session.json npx -y @chaindead/telegram-mcp auth --phone <your_phone_number>\n\n"
                    f"Note: If you have Two-Factor Authentication (2FA) enabled on your Telegram account, you must also append the password flag:\n"
                    f"  --password <your_2fa_password>\n\n"
                    f"Once authenticated, a 'session.json' file will be created. You can use it on Hugging Face Spaces in one of two ways:\n"
                    f"1. [Recommended & Safe]: Copy the contents of the generated 'session.json' file and create a new Space Secret / Environment Variable named 'TG_SESSION_JSON' in your Hugging Face Space configuration.\n"
                    f"2. [Alternative]: Commit the 'session.json' file directly to your git repository under '.telegram-mcp/session.json'. (WARNING: Only do this if your repository and Hugging Face Space are PRIVATE, otherwise your Telegram session is public)."
                )

        def get_connection_config(self) -> dict[str, Any]:
            session_json_content = os.getenv("TG_SESSION_JSON")
            if session_json_content:
                session_file = Path("/tmp/.telegram-mcp/session.json")
            else:
                relative_dir = Path(__file__).resolve().parent.parent / ".telegram-mcp"
                session_path_env = os.getenv("TG_SESSION_PATH")
                if session_path_env:
                    session_dir = Path(session_path_env).expanduser()
                elif (relative_dir / "session.json").exists():
                    session_dir = relative_dir
                else:
                    session_dir = Path("~/.telegram-mcp").expanduser()
                session_file = session_dir / "session.json"

            env: dict[str, str] = {
                "PATH": os.environ.get("PATH", ""),
                "TG_APP_ID": os.getenv("TG_APP_ID", ""),
                "TG_API_HASH": os.getenv("TG_API_HASH", ""),
                "TG_SESSION_PATH": str(session_file),
            }

            return self.build_stdio_config(
                "npx",
                ["-y", "@chaindead/telegram-mcp"],
                env,
            )

    return TelegramStdioAgent(
        service_name="telegram",
        required_env_vars=required_vars,
    )
