import os
import asyncio
import time
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

# Load environment variables from .env if present
load_dotenv()

from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from typing import Dict, Any, List

from lib.state import RuntimeState
from lib.base_agent import RESPONSE_TIME_LINE, ensure_agent_initialized, stream_agent_response
from agents import get_github_agent, get_jira_agent, get_slack_agent, get_web_agent
from langchain_core.messages import HumanMessage, AIMessage
from lib.utils import extract_message_text
from llm_providers.ollama import get_default_ollama_model

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start MCP servers in the background so the first chat doesn't wait for them
    if os.getenv("MCP_PRECONNECT", "true").lower() not in ("0", "false", "no"):
        for agent_name in AGENT_BUILDERS:
            asyncio.create_task(_preconnect_agent(agent_name))
    yield
    for active in active_agents.values():
        await active.agent.disconnect(active.state)


app = FastAPI(title="Community AI MCP Dashboard", lifespan=lifespan)

# Mapping from agent name to builder function
AGENT_BUILDERS = {
    "github": get_github_agent,
    "jira": get_jira_agent,
    "slack": get_slack_agent,
    "web-reader": get_web_agent,
}

PROVIDER_KEY_ENV = {
    "gemini": "GOOGLE_API_KEY",
    "openai": "OPENAI_API_KEY",
    "groq": "GROQ_API_KEY",
}

# Agent connection state for /api/status: {"state": idle|connecting|ready|error, ...}
AGENT_STATUS: Dict[str, Dict[str, Any]] = {}


def _set_status(agent_name: str, state: str, message: str = "", tools: int | None = None) -> None:
    status: Dict[str, Any] = {"state": state, "message": message}
    if tools is not None:
        status["tools"] = tools
    AGENT_STATUS[agent_name] = status


PROVIDER_MODEL_ENV = {
    "gemini": "GEMINI_MODEL",
    "openai": "OPENAI_MODEL",
    "groq": "GROQ_MODEL",
    "ollama": "OLLAMA_MODEL",
}

class ActiveAgent:
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.state = RuntimeState(service_name=agent_name)
        self.agent = AGENT_BUILDERS[agent_name]()
        self.last_provider = None
        self.last_model = None
        self.last_api_key = None
        # Serializes init between the startup pre-connect and concurrent requests
        self.lock = asyncio.Lock()

    async def preconnect(self) -> None:
        """Start the MCP server and load its tools ahead of the first chat.

        With a local (Ollama) default provider, also pre-fill the model's prompt cache
        with this agent's system prompt + tool schemas.
        """
        async with self.lock:
            await self.agent.connect(self.state)
            if os.getenv("LLM_PROVIDER", "").lower() == "ollama":
                self.agent.build_executor(self.state)
                await self.agent.warm_prompt_cache(self.state)

    async def ensure_initialized(self, provider: str, model: str, api_key: str):
        async with self.lock:
            await self._ensure_initialized(provider, model, api_key)

    async def _ensure_initialized(self, provider: str, model: str, api_key: str):
        llm_changed = (
            self.last_provider != provider
            or self.last_model != model
            or self.last_api_key != api_key
        )
        if llm_changed:
            # Only the LLM side is rebuilt; the MCP session (server process) is kept
            self.state.agent_executor = None

        # Always apply: these env vars are process-global and another agent may have
        # changed them (LLM_PROVIDER also drives the local-model limits at runtime).
        os.environ["LLM_PROVIDER"] = provider
        os.environ["MODEL"] = model
        # Providers read e.g. OLLAMA_MODEL before MODEL, so set it too or the
        # model picked in the dashboard is ignored whenever .env sets one.
        if model and model.strip() and provider in PROVIDER_MODEL_ENV:
            os.environ[PROVIDER_MODEL_ENV[provider]] = model.strip()
        if provider == "gemini" and api_key and api_key.strip():
            os.environ["GOOGLE_API_KEY"] = api_key.strip()
        elif provider == "openai" and api_key and api_key.strip():
            os.environ["OPENAI_API_KEY"] = api_key.strip()
        elif provider == "groq" and api_key and api_key.strip():
            os.environ["GROQ_API_KEY"] = api_key.strip()
        elif provider == "ollama":
            os.environ["OLLAMA_BASE_URL"] = api_key.strip() if (api_key and api_key.strip()) else os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        # Connects only if the MCP session isn't running; builds the executor if needed
        await ensure_agent_initialized(self.state, self.agent)

        self.last_provider = provider
        self.last_model = model
        self.last_api_key = api_key

active_agents: Dict[str, ActiveAgent] = {}

def get_active_agent(agent_name: str) -> ActiveAgent:
    if agent_name not in active_agents:
        active_agents[agent_name] = ActiveAgent(agent_name)
    return active_agents[agent_name]


async def _preconnect_agent(agent_name: str) -> None:
    start = time.perf_counter()
    _set_status(agent_name, "connecting", "Starting MCP server…")
    try:
        active = get_active_agent(agent_name)
        await active.preconnect()
        _set_status(agent_name, "ready", "Connected", tools=len(active.state.mcp_tools))
        print(f"[startup] {agent_name}: ready ({time.perf_counter() - start:.1f}s)")
    except Exception as e:
        # Missing credentials etc. — the agent will report the error when used
        _set_status(agent_name, "error", format_init_error(e))
        print(f"[startup] {agent_name}: not pre-connected ({format_init_error(e)[:200]})")

class ChatRequest(BaseModel):
    agent_name: str
    message: str
    session_id: str = "default"
    provider: str
    model: str
    api_key: str

class ChatResponse(BaseModel):
    response: str
    session_id: str
    elapsed_seconds: float | None = None

class ToolQueryRequest(BaseModel):
    provider: str
    model: str
    api_key: str

@app.get("/api/config")
async def get_config():
    # Read currently configured variables in .env
    return {
        "providers": ["gemini", "openai", "groq", "ollama"],
        "agents": list(AGENT_BUILDERS.keys()),
        "current_provider": os.getenv("LLM_PROVIDER", "gemini"),
        "models": {
            "gemini": os.getenv("GEMINI_MODEL", "models/gemini-3.1-flash-lite"),
            "openai": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "groq": os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            "ollama": os.getenv("OLLAMA_MODEL", get_default_ollama_model()),
        },
        # Never send key values to the browser (the dashboard may be public): only
        # whether the server has one, so an empty key field means "use the server key"
        "server_keys": {
            provider: bool((os.getenv(env_name) or "").strip())
            for provider, env_name in PROVIDER_KEY_ENV.items()
        },
        "ollama_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    }


@app.get("/api/status")
async def get_status():
    """Per-agent connection state for the dashboard's status indicators."""
    return {"agents": {name: AGENT_STATUS.get(name, {"state": "idle"}) for name in AGENT_BUILDERS}}

def _leaf_errors(e: BaseException) -> list[BaseException]:
    """Unwrap (nested) ExceptionGroups from anyio/asyncio TaskGroups down to the real errors."""
    subs = getattr(e, "exceptions", None)
    if not subs:
        return [e]
    return [leaf for sub in subs for leaf in _leaf_errors(sub)]


def format_init_error(e: Exception) -> str:
    leaves = _leaf_errors(e)
    err_msg = "; ".join(str(x) or type(x).__name__ for x in leaves)

    # A stdio MCP server that exits during startup only surfaces as "Connection closed";
    # its own error (bad token, missing permission) is printed to the container log
    if any("connection closed" in str(x).lower() for x in leaves):
        err_msg += (
            " (The MCP server stopped during startup, usually because of an invalid token or"
            " missing permissions. Its error is in the logs: docker logs community_mcp_web)"
        )

    # Provide helpful suggestion for GitHub Copilot 401 Unauthorized errors
    if "401" in err_msg and "githubcopilot" in err_msg.lower():
        err_msg += " (Authentication failed for GitHub Copilot. Your personal access token may not have Copilot access. To run a standard GitHub server instead, add 'GITHUB_MCP_TRANSPORT=stdio' to your .env file)"
        
    return err_msg

@app.post("/api/agent/{agent_name}/tools")
async def get_agent_tools(agent_name: str, req: ToolQueryRequest):
    if agent_name not in AGENT_BUILDERS:
        raise HTTPException(status_code=400, detail=f"Unsupported agent: {agent_name}")
    
    agent_wrapper = get_active_agent(agent_name)
    try:
        await agent_wrapper.ensure_initialized(req.provider, req.model, req.api_key)
    except Exception as e:
        _set_status(agent_name, "error", format_init_error(e))
        raise HTTPException(status_code=500, detail=f"Failed to load agent tools: {format_init_error(e)}")
    _set_status(agent_name, "ready", "Connected", tools=len(agent_wrapper.state.tool_summaries))
    state = agent_wrapper.state
    return {
        # Tools the current model can use (local models get a filtered subset)
        "tools": state.tool_summaries,
        # Everything the MCP server offers, for the dashboard's tools drawer
        "all_tools": state.all_tool_summaries,
        "total": len(state.all_tool_summaries),
    }

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    if req.agent_name not in AGENT_BUILDERS:
        raise HTTPException(status_code=400, detail=f"Unsupported agent: {req.agent_name}")
    
    agent_wrapper = get_active_agent(req.agent_name)
    try:
        await agent_wrapper.ensure_initialized(req.provider, req.model, req.api_key)
    except Exception as e:
        _set_status(req.agent_name, "error", format_init_error(e))
        raise HTTPException(status_code=500, detail=f"Failed to initialize agent: {format_init_error(e)}")

    # Record message to history
    human_msg = HumanMessage(content=req.message)
    session_history = agent_wrapper.state.record_message(req.session_id, human_msg)
    
    start_time = time.perf_counter()
    try:
        last_ai_msg = await stream_agent_response(agent_wrapper.state, session_history)
        elapsed_time = time.perf_counter() - start_time
        
        # Timing is returned separately (shown by the UI under the answer), never mixed
        # into the text: the model would read and imitate it on later turns
        text_response = extract_message_text(last_ai_msg)
        last_ai_msg.content = text_response
        agent_wrapper.state.record_message(req.session_id, last_ai_msg)
        return ChatResponse(
            response=text_response,
            session_id=req.session_id,
            elapsed_seconds=round(elapsed_time, 2),
        )
    except Exception as e:
        agent_wrapper.state.pop_last_message(req.session_id)
        err_msg = str(e)
        if "tool call validation failed" in err_msg or "was not in request.tools" in err_msg:
            fallback_response = f"Provider Error: The LLM model attempted to call a tool that is not in the allowed tools list for {req.agent_name}. Please check available tools or retry."
            return ChatResponse(response=fallback_response, session_id=req.session_id)
        raise HTTPException(status_code=500, detail=f"Agent error: {err_msg}")

@app.get("/api/sessions/{agent_name}")
async def list_agent_sessions(agent_name: str):
    if agent_name not in AGENT_BUILDERS:
        raise HTTPException(status_code=400, detail=f"Unsupported agent: {agent_name}")
    agent_wrapper = get_active_agent(agent_name)
    sessions = agent_wrapper.state.list_sessions()
    for session in sessions:
        # Title for the dashboard's chat list: the first question of the conversation
        history = agent_wrapper.state.chat_sessions.get(session["session_id"], [])
        first = next((m for m in history if isinstance(m, HumanMessage)), None)
        title = first.content if first is not None and isinstance(first.content, str) else ""
        session["title"] = (title[:60] + "…") if len(title) > 60 else (title or "New chat")
    return {"sessions": sessions}

@app.delete("/api/sessions/{agent_name}/{session_id}")
async def delete_agent_session(agent_name: str, session_id: str):
    if agent_name not in AGENT_BUILDERS:
        raise HTTPException(status_code=400, detail=f"Unsupported agent: {agent_name}")
    agent_wrapper = get_active_agent(agent_name)
    agent_wrapper.state.clear_session(session_id, persist=True)
    return {"status": "success"}

@app.get("/api/sessions/{agent_name}/{session_id}/messages")
async def get_session_messages(agent_name: str, session_id: str):
    if agent_name not in AGENT_BUILDERS:
        raise HTTPException(status_code=400, detail=f"Unsupported agent: {agent_name}")
    agent_wrapper = get_active_agent(agent_name)
    session_data = agent_wrapper.state.serialize_session(session_id)
    if session_data is None:
        return {"messages": []}
    
    # Format and filter messages for UI display
    formatted = []
    for msg in session_data.get("messages", []):
        msg_type = msg.get("type")
        data = msg.get("data", {})
        content = data.get("content", "")
        if msg_type in ("human", "ai") and content:
            if msg_type == "ai" and isinstance(content, str):
                # Answers saved by older versions carry the timing line in their text
                content = RESPONSE_TIME_LINE.sub("", content).rstrip()
            formatted.append({
                "sender": "user" if msg_type == "human" else "agent",
                "text": content
            })
    return {"messages": formatted}

WEB_DIR = os.path.join(os.path.dirname(__file__), "web")

# Browser tab / home-screen icons. Browsers request /favicon.ico by default, so it
# serves the 32px PNG (modern browsers accept PNG content there).
ICON_FILES = {
    "/favicon.svg": ("favicon.svg", "image/svg+xml"),
    "/favicon.ico": ("favicon-32.png", "image/png"),
    "/favicon-32.png": ("favicon-32.png", "image/png"),
    "/apple-touch-icon.png": ("apple-touch-icon.png", "image/png"),
}


def _icon_route(filename: str, media_type: str):
    async def serve_icon():
        return FileResponse(
            os.path.join(WEB_DIR, filename),
            media_type=media_type,
            headers={"Cache-Control": "public, max-age=86400"},
        )
    return serve_icon


for _path, (_filename, _media_type) in ICON_FILES.items():
    app.add_api_route(_path, _icon_route(_filename, _media_type), methods=["GET"], include_in_schema=False)


@app.get("/")
async def serve_index():
    # Return HTML index directly
    index_path = os.path.join(os.path.dirname(__file__), "web", "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Frontend index.html not found! Please compile or place index.html in the web/ directory.</h1>", status_code=404)
