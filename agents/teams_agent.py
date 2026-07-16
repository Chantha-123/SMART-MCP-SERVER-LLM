import os
from typing import Any

from lib.base_agent import BaseAgent


class GoogleChatAgent(BaseAgent):
    """Google Chat integration scaffold for posting messages to a space."""

    def __init__(self) -> None:
        super().__init__(
            "teams",
            required_env_vars=["GOOGLE_CHAT_CREDENTIALS_JSON", "GOOGLE_CHAT_SPACE", "GOOGLE_CHAT_PROJECT_ID"],
        )

    def validate_environment(self) -> None:
        super().validate_environment()

        if not os.getenv("GOOGLE_CHAT_SPACE"):
            raise ValueError(
                "Missing required environment variables: GOOGLE_CHAT_SPACE. "
                "Set the Google Chat space name or room ID."
            )

    def _build_space_message_payload(self, message: str) -> dict[str, Any]:
        return {
            "text": message,
        }

    def get_connection_config(self) -> dict[str, Any]:
        return {
            "credentials_json": os.getenv("GOOGLE_CHAT_CREDENTIALS_JSON", ""),
            "space": os.getenv("GOOGLE_CHAT_SPACE", ""),
            "project_id": os.getenv("GOOGLE_CHAT_PROJECT_ID", ""),
        }


def get_google_chat_agent() -> BaseAgent:
    return GoogleChatAgent()
