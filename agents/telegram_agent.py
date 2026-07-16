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

            session_dir = Path(os.getenv("TG_SESSION_PATH", "~/.telegram-mcp")).expanduser()
            session_file = session_dir / "session.json"
            if not session_file.exists():
                tg_app_id = os.getenv("TG_APP_ID", "<your_tg_app_id>")
                tg_api_hash = os.getenv("TG_API_HASH", "<your_tg_api_hash>")
                raise ValueError(
                    f"Telegram session file not found at {session_file}.\n"
                    f"On first run, you must complete authentication using the Telegram MCP CLI.\n"
                    f"Please run the following command in your terminal (replacing <your_phone_number> with your Telegram phone number, including country code, e.g., +1234567890) and follow the prompts:\n\n"
                    f"TG_APP_ID={tg_app_id} TG_API_HASH={tg_api_hash} TG_SESSION_PATH={session_file} npx -y @chaindead/telegram-mcp auth --phone <your_phone_number>\n\n"
                    f"Note: If you have Two-Factor Authentication (2FA) enabled on your Telegram account, you must also append the password flag:\n"
                    f"  --password <your_2fa_password>\n"
                )

        def get_connection_config(self) -> dict[str, Any]:
            session_dir = Path(os.getenv("TG_SESSION_PATH", "~/.telegram-mcp")).expanduser()
            session_dir.mkdir(parents=True, exist_ok=True)

            env: dict[str, str] = {
                "PATH": os.environ.get("PATH", ""),
                "TG_APP_ID": os.getenv("TG_APP_ID", ""),
                "TG_API_HASH": os.getenv("TG_API_HASH", ""),
                "TG_SESSION_PATH": str(session_dir / "session.json"),
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
