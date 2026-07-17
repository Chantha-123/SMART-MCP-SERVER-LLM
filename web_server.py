import os
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Dict, Any, List

from lib.state import RuntimeState
from lib.base_agent import ensure_agent_initialized, stream_agent_response
from agents import get_github_agent, get_jira_agent, get_slack_agent, get_google_chat_agent, get_telegram_agent
from langchain_core.messages import HumanMessage, AIMessage
from lib.utils import extract_message_text

app = FastAPI(title="Community AI MCP Dashboard")

# Mapping from agent name to builder function
AGENT_BUILDERS = {
    "github": get_github_agent,
    "jira": get_jira_agent,
    "slack": get_slack_agent,
    "telegram": get_telegram_agent,
    "google-chat": get_google_chat_agent,
}

class ActiveAgent:
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.state = RuntimeState(service_name=agent_name)
        self.agent = AGENT_BUILDERS[agent_name]()
        self.last_provider = None
        self.last_model = None
        self.last_api_key = None

    async def ensure_initialized(self, provider: str, model: str, api_key: str):
        # We need to re-initialize if not initialized or if LLM parameters changed
        needs_init = (
            self.state.agent_executor is None
            or self.state.mcp_client is None
            or self.last_provider != provider
            or self.last_model != model
            or self.last_api_key != api_key
        )
        if needs_init:
            if self.state.mcp_client is not None:
                try:
                    await self.state.mcp_client.aclose()
                except Exception:
                    pass
                self.state.mcp_client = None
                self.state.agent_executor = None

            # Apply parameters to environment variables for LangChain loading
            os.environ["LLM_PROVIDER"] = provider
            os.environ["MODEL"] = model
            if provider == "gemini" and api_key and api_key.strip():
                os.environ["GOOGLE_API_KEY"] = api_key.strip()
            elif provider == "openai" and api_key and api_key.strip():
                os.environ["OPENAI_API_KEY"] = api_key.strip()
            elif provider == "groq" and api_key and api_key.strip():
                os.environ["GROQ_API_KEY"] = api_key.strip()

            # Now run agent initialization
            await ensure_agent_initialized(self.state, self.agent)
            
            self.last_provider = provider
            self.last_model = model
            self.last_api_key = api_key

active_agents: Dict[str, ActiveAgent] = {}

def get_active_agent(agent_name: str) -> ActiveAgent:
    if agent_name not in active_agents:
        active_agents[agent_name] = ActiveAgent(agent_name)
    return active_agents[agent_name]

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

class ToolQueryRequest(BaseModel):
    provider: str
    model: str
    api_key: str

@app.get("/api/config")
async def get_config():
    # Read currently configured variables in .env
    return {
        "providers": ["gemini", "openai", "groq"],
        "agents": list(AGENT_BUILDERS.keys()),
        "current_provider": os.getenv("LLM_PROVIDER", "gemini"),
        "models": {
            "gemini": os.getenv("GEMINI_MODEL", "models/gemini-3.1-flash-lite"),
            "openai": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "groq": os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        },
        "api_keys": {
            "gemini": os.getenv("GOOGLE_API_KEY", ""),
            "openai": os.getenv("OPENAI_API_KEY", ""),
            "groq": os.getenv("GROQ_API_KEY", "")
        }
    }

@app.post("/api/agent/{agent_name}/tools")
async def get_agent_tools(agent_name: str, req: ToolQueryRequest):
    if agent_name not in AGENT_BUILDERS:
        raise HTTPException(status_code=400, detail=f"Unsupported agent: {agent_name}")
    
    agent_wrapper = get_active_agent(agent_name)
    try:
        await agent_wrapper.ensure_initialized(req.provider, req.model, req.api_key)
        return {"tools": agent_wrapper.state.tool_summaries}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load agent tools: {str(e)}")

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    if req.agent_name not in AGENT_BUILDERS:
        raise HTTPException(status_code=400, detail=f"Unsupported agent: {req.agent_name}")
    
    agent_wrapper = get_active_agent(req.agent_name)
    try:
        await agent_wrapper.ensure_initialized(req.provider, req.model, req.api_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize agent: {str(e)}")

    # Record message to history
    human_msg = HumanMessage(content=req.message)
    session_history = agent_wrapper.state.record_message(req.session_id, human_msg)
    
    try:
        last_ai_msg = await stream_agent_response(agent_wrapper.state, session_history)
        agent_wrapper.state.record_message(req.session_id, last_ai_msg)
        text_response = extract_message_text(last_ai_msg)
        return ChatResponse(response=text_response, session_id=req.session_id)
    except Exception as e:
        agent_wrapper.state.pop_last_message(req.session_id)
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

@app.get("/api/sessions/{agent_name}")
async def list_agent_sessions(agent_name: str):
    if agent_name not in AGENT_BUILDERS:
        raise HTTPException(status_code=400, detail=f"Unsupported agent: {agent_name}")
    agent_wrapper = get_active_agent(agent_name)
    return {"sessions": agent_wrapper.state.list_sessions()}

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
            formatted.append({
                "sender": "user" if msg_type == "human" else "agent",
                "text": content
            })
    return {"messages": formatted}

@app.get("/")
async def serve_index():
    # Return HTML index directly
    index_path = os.path.join(os.path.dirname(__file__), "web", "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Frontend index.html not found! Please compile or place index.html in the web/ directory.</h1>", status_code=404)
