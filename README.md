# Community AI MCP Agent

A unified Python CLI for interacting with GitHub, Slack, and Jira using the Model Context Protocol (MCP) and LangChain.

## Features

- **GitHub Agent**: Repository management, issues, PRs, and code operations
- **Slack Agent**: Channel management, messaging, and workspace interactions  
- **Jira Agent**: Issue tracking, project management, and workflow automation
- **Multi-LLM Support**: Works with Gemini, Groq, OpenAI, and Lightning AI
- **Extensible Architecture**: Built on LangChain and LangGraph for easy customization

## Quick Start

### 1. Installation

```bash
# Clone the repository
cd community_chatbot/mcp_impl

# Install Python dependencies (using uv or pip)
uv sync
# or
pip install -r requirements.txt

# Install Node.js dependencies for Slack agent (requires Node.js)
cd agents
npm install
cd ..
```

### 2. Configuration

Copy the example environment file and configure your credentials:

```bash
cp .env.example .env
```

**Minimum required configuration:**

```env
# Choose your LLM provider
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your_key_here

# Enable agents as needed
GITHUB_PERSONAL_ACCESS_TOKEN=your_token_here
SLACK_MCP_XOXP_TOKEN=xoxp-your-token-here
JIRA_URL=https://your-company.atlassian.net
JIRA_USERNAME=your.email@company.com
JIRA_API_TOKEN=your_token_here
```

### 3. Usage

```bash
# Run the GitHub agent
python main.py github

# Run the Slack agent
python main.py slack

# Run the Jira agent
python main.py jira

# Get help for any agent
python main.py github --help
```

## Agent Details

### GitHub Agent

Interact with GitHub repositories using your personal access token.

**Required:**

- `GITHUB_PERSONAL_ACCESS_TOKEN` - [Create here](https://github.com/settings/tokens)

**Capabilities:**

- Repository operations (create, clone, search)
- Issue and PR management
- Code search and file operations
- Workflow automation

### Slack Agent

Connect to Slack workspaces and manage communications.

**Required:**

- `SLACK_MCP_XOXP_TOKEN` - [Create Slack app](https://api.slack.com/apps)

**Capabilities:**

- Channel and user management
- Message posting and retrieval
- Workspace information

**Optional:** Enable message posting with `SLACK_MCP_ADD_MESSAGE_TOOL=true`

### Jira Agent

Manage Jira projects and issues programmatically.

**Required:**

- `JIRA_URL` - Your Jira instance URL
- `JIRA_USERNAME` - Your email
- `JIRA_API_TOKEN` - [Create here](https://id.atlassian.com/manage-profile/security/api-tokens)

**Capabilities:**

- Issue CRUD operations
- Project and sprint management
- Advanced JQL searches
- Custom field handling

## Implementation Details

- **MCP integration:** Agents use an internal MCP-based client flow implemented in [community_chatbot/mcp_impl/lib/base_agent.py](community_chatbot/mcp_impl/lib/base_agent.py). The CLI shell for each agent is created with the helper in [community_chatbot/mcp_impl/lib/base_mcp.py](community_chatbot/mcp_impl/lib/base_mcp.py) which exposes commands like `list-tools`, `chat`, `invoke-tool`, and `health`.

- **MCP client used:** The code constructs a `MultiServerMCPClient` (from the `langchain_mcp_adapters` package) inside `BaseAgent.initialize()` to discover and load remote MCP tools. Tools discovered from the MCP endpoints are converted into LangGraph/LangChain-compatible tool definitions and used to create a React-style agent via `langgraph.prebuilt.create_react_agent`.

- **Transport options:** Agents support multiple transport modes:
 	- `stdio` — runs a local process (usually an `npx` package or a Docker image) and communicates over stdio. Examples:
  		- Slack: runs `npx slack-mcp-server --transport stdio` (see [community_chatbot/mcp_impl/agents/slack_agent.py](community_chatbot/mcp_impl/agents/slack_agent.py)).
  		- Jira: can run `ghcr.io/sooperset/mcp-atlassian:latest` (Docker) or `npx mcp-atlassian@latest` (see [community_chatbot/mcp_impl/agents/jira_agent.py](community_chatbot/mcp_impl/agents/jira_agent.py)).
 	- `sse` / `streamable_http` — connects to an HTTP/SSE MCP server endpoint. Default example endpoints used by the code:
  		- Slack HTTP default: `http://127.0.0.1:13080/sse`
  		- Jira HTTP default: `http://127.0.0.1:8080/sse`
  The HTTP transport builder is implemented in [community_chatbot/mcp_impl/lib/base_transport.py](community_chatbot/mcp_impl/lib/base_transport.py).

- **LLM & agent creation:** The LLM provider is chosen by `LLM_PROVIDER` (see [community_chatbot/mcp_impl/lib/base_agent.py](community_chatbot/mcp_impl/lib/base_agent.py)) and the repository includes provider adapters in [community_chatbot/mcp_impl/llm_providers/](community_chatbot/mcp_impl/llm_providers/) (Gemini, Groq, Lightning). The selected LLM is passed into `create_react_agent` alongside the loaded MCP tools to form the agent executor.

### GitHub Agent

- `https://github.com/github/github-mcp-server` — GitHub MCP server repository.
- `https://api.githubcopilot.com/mcp/` — default GitHub MCP endpoint used as the service URL in `agents/github_agent.py`.
- `langchain_mcp_adapters` — MCP client package used via `MultiServerMCPClient` (see `lib/base_agent.py`).
- `langgraph` / `langchain` — used to create the React-style agent (`langgraph.prebuilt.create_react_agent`).

### Slack Agent

- `https://github.com/korotovsky/slack-mcp-server` — Slack MCP server repository.
- `npx slack-mcp-server` — npm package invoked in `agents/slack_agent.py` when using the `stdio` transport (runs `slack-mcp-server --transport stdio`).
- `SLACK_*` environment variables (tokens) map to Slack credentials and standard Slack developer docs: <https://api.slack.com/>
- Default local HTTP/SSE endpoint in code: `http://127.0.0.1:13080/sse` (used when `SLACK_MCP_TRANSPORT` is set to `sse`/`streamable_http`).

### Jira Agent

- `https://github.com/sooperset/mcp-atlassian` — Jira MCP server repository.
- `ghcr.io/sooperset/mcp-atlassian:latest` — Docker image referenced in `agents/jira_agent.py` for the `stdio` Docker transport.
- `npx mcp-atlassian@latest` — npm package fallback for `stdio` (non-Docker) mode.
- Jira developer docs and API token creation: <https://id.atlassian.com/manage-profile/security/api-tokens>
- Default local HTTP/SSE endpoint in code: `http://127.0.0.1:8080/sse` (used when `JIRA_MCP_TRANSPORT` is `sse`/`streamable_http`).

## Project Structure

```
mcp_impl/
├── agents/              # Agent implementations
│   ├── github_agent.py
│   ├── jira_agent.py
│   └── slack_agent.py
├── lib/                 # Core library
│   ├── base_agent.py   # Base agent class
│   ├── base_mcp.py     # MCP integration
│   └── utils.py        # Utilities
├── llm_providers/      # LLM provider implementations
│   ├── gemini.py
│   ├── groq_llm.py
│   ├── lightning_llm.py
│   └── __init__.py
├── main.py             # CLI entry point
├── .env.example        # Configuration template
└── requirements.txt    # Python dependencies
```

## Requirements

- Python >= 3.12
- Valid API keys for your chosen LLM provider
- Agent-specific credentials (GitHub token, Slack token, Jira credentials)

## Getting API Keys

- **GitHub**: [Settings → Developer settings → Personal access tokens](https://github.com/settings/tokens)
- **Slack**: [Create a Slack app](https://api.slack.com/apps) → Install to workspace → Copy OAuth token
- **Jira**: [Account security → API tokens](https://id.atlassian.com/manage-profile/security/api-tokens)
- **Gemini**: [Google AI Studio](https://aistudio.google.com/app/apikey)
- **Groq**: [Groq Console](https://console.groq.com/keys)
- **OpenAI**: [OpenAI API Keys](https://platform.openai.com/api-keys)
# SMART-MCP-SERVER-LLM
