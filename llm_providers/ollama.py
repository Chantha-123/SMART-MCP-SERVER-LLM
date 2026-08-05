import os
import requests
import logging
import threading
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

# Track pulling tasks in progress to avoid double-pulling
_pulling_models = set()

def _pull_model_in_background(base_url: str, model_name: str):
    try:
        logger.info(f"Background thread starting pull for Ollama model '{model_name}'...")
        pull_response = requests.post(
            f"{base_url}/api/pull",
            json={"name": model_name, "stream": False},
            timeout=600
        )
        if pull_response.status_code == 200:
            logger.info(f"Successfully pulled Ollama model '{model_name}' in background.")
        else:
            logger.error(f"Failed to pull Ollama model '{model_name}': {pull_response.text}")
    except Exception as e:
        logger.error(f"Failed to pull Ollama model '{model_name}' in background: {e}")
    finally:
        _pulling_models.discard(model_name)

def ensure_model_pulled(base_url: str, model_name: str):
    """
    Checks if a model is installed in the local Ollama instance.
    If not, starts pulling in a background thread and raises a ValueError.
    """
    try:
        # Check tags to see if model is already pulled
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            tags = response.json()
            models = [m["name"] for m in tags.get("models", [])]
            # Ollama models might be listed with tags e.g. "llama3.1:latest"
            if model_name in models or f"{model_name}:latest" in models:
                logger.info(f"Ollama model '{model_name}' is already available.")
                return
    except Exception as e:
        logger.error(f"Failed to check local tags on Ollama: {e}")

    # Check if this model is already being pulled in the background
    if model_name in _pulling_models:
        raise ValueError(
            f"Model '{model_name}' is currently downloading in the background. "
            "Please wait a few minutes and try again."
        )

    # Start the pull in a background thread to prevent blocking FastAPI's event loop
    _pulling_models.add(model_name)
    threading.Thread(
        target=_pull_model_in_background,
        args=(base_url, model_name),
        daemon=True
    ).start()

    raise ValueError(
        f"Model '{model_name}' was not found locally. We have started downloading/pulling it "
        "for you in the background. Please wait 2-3 minutes for the download to complete and try again."
    )

def get_llm() -> BaseChatModel:
    base_url = (os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").strip()
    model_id = (os.getenv("OLLAMA_MODEL") or os.getenv("MODEL") or "llama3.1").strip()
    
    if not base_url.startswith("http"):
        base_url = f"http://{base_url}"
        
    # Remove trailing slash if present
    base_url = base_url.rstrip("/")
        
    # Auto-pull the model if it's not downloaded (handles pulling via background threads)
    ensure_model_pulled(base_url, model_id)
    
    # Return ChatOpenAI configured for Ollama OpenAI compatibility
    return ChatOpenAI(
        model=model_id,
        base_url=f"{base_url}/v1",
        api_key="ollama", # placeholder key for client initialization
        temperature=0.0
    )
