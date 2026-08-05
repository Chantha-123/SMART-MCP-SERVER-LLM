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
    If not, starts pulling in a background thread without crashing agent initialization.
    """
    try:
        # Check tags to see if model is already pulled
        response = requests.get(f"{base_url}/api/tags", timeout=3)
        if response.status_code == 200:
            tags = response.json()
            models = [m["name"] for m in tags.get("models", [])]
            
            target_lower = model_name.lower()
            target_base = target_lower.split(":")[0]
            
            for m in models:
                m_lower = m.lower()
                m_base = m_lower.split(":")[0]
                if target_lower == m_lower or f"{target_lower}:latest" == m_lower or target_base == m_base:
                    logger.info(f"Ollama model '{model_name}' (matched '{m}') is available.")
                    return
    except Exception as e:
        logger.warning(f"Could not verify tags on Ollama at {base_url}: {e}")
        return

    # Check if this model is already being pulled in the background
    if model_name not in _pulling_models:
        logger.info(f"Model '{model_name}' not detected locally. Triggering background pull...")
        _pulling_models.add(model_name)
        threading.Thread(
            target=_pull_model_in_background,
            args=(base_url, model_name),
            daemon=True
        ).start()

def get_llm() -> BaseChatModel:
    base_url = (os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").strip()
    model_id = (os.getenv("OLLAMA_MODEL") or os.getenv("MODEL") or "qwen2.5-coder:7b").strip()
    
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
