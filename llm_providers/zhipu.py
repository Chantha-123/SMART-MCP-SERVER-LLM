import os

from pydantic import SecretStr
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel


def get_llm() -> BaseChatModel:
    api_key = os.getenv("ZHIPUAI_API_KEY")
    if not api_key:
        raise ValueError(
            "Missing required environment variable: ZHIPUAI_API_KEY"
        )

    model_id = (os.getenv("ZHIPUAI_MODEL") or os.getenv("MODEL") or "glm-4.5-air").strip()
    if not model_id:
        raise ValueError("Model ID cannot be empty.")

    base_url = (os.getenv("ZHIPUAI_BASE_URL") or "https://open.bigmodel.cn/api/paas/v4/").strip()

    return ChatOpenAI(
        model=model_id,
        api_key=SecretStr(api_key),
        base_url=base_url,
    )
