---
title: Community AI MCP Dashboard
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Community AI MCP Agent & Dashboard

A unified AI Agent platform with a Web Dashboard and CLI for interacting with GitHub, Slack, Jira, Telegram, Google Chat, and Web Scraping using the Model Context Protocol (MCP) and LangChain/LangGraph. Supports Gemini, Groq, OpenAI, and local Ollama.

---

## 🌟 Features

- **GitHub Agent**: Repository management, issues, PRs, and code operations.
- **Slack Agent**: Channel management, messaging, and workspace interactions.
- **Jira Agent**: Issue tracking, project management, and workflow automation.
- **Telegram & Google Chat Agents**: Community chat integrations.
- **Web Reader Agent**: Web page scraping and URL reading.
- **Multi-LLM Support**: Works with Gemini, Groq, OpenAI, and local Ollama models.
- **Web Dashboard & CLI**: Full interactive browser UI and terminal CLI interface.

---

## 🚀 Setup from Scratch

### Prerequisites
- **Python**: 3.12 or higher
- **Node.js**: 20+ (required for npm-based MCP servers like Slack and Jira)
- **Docker & Docker Compose** (Optional, for containerized deployment)

### 1. Clone & Install Dependencies

```bash
# Clone the repository and enter directory
cd community_mcp_servers

# Install Python dependencies (using uv or pip)
uv sync
# OR
pip install -r requirements.txt

# Install Node.js dependencies for agent tools
cd agents
npm install
cd ..
```

### 2. Configure Credentials (`.env`)

Copy the template environment file:

```bash
cp .env.example .env
```

Open `.env` and fill in your desired provider and agent credentials:

```env
# Choose default LLM Provider: gemini | groq | openai | ollama
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your_gemini_api_key_here

# Agent Credentials
GITHUB_PERSONAL_ACCESS_TOKEN=your_github_token_here
SLACK_MCP_XOXP_TOKEN=xoxp-your-slack-token-here
JIRA_URL=https://your-company.atlassian.net
JIRA_USERNAME=your.email@company.com
JIRA_API_TOKEN=your_jira_token_here
```

---

## 🚦 How to Start, Stop, and Restart

### Option 1: Native Python Web Server (Fastest for Development)

- **START**:
  ```bash
  python3 web_server.py
  ```
  Open browser at: `http://localhost:8000`

- **STOP**: Press `Ctrl + C` in your terminal.

- **RESTART**: Press `Ctrl + C`, then run:
  ```bash
  python3 web_server.py
  ```

---

### Option 2: Docker Compose (Containerized Setup)

- **START BOTH (Web Dashboard + Ollama)**:
  ```bash
  docker compose up -d --build
  ```
  - MCP Web Dashboard: `http://localhost:8000`
  - Ollama Service: `http://localhost:11434`

- **START ONLY OLLAMA**:
  ```bash
  docker compose up -d ollama
  ```

- **START ONLY WEB DASHBOARD**:
  ```bash
  docker compose up -d web
  ```

- **CHECK STATUS & LOGS**:
  ```bash
  # Check running containers
  docker compose ps

  # View web server logs
  docker compose logs -f web

  # View Ollama logs
  docker compose logs -f ollama
  ```

- **STOP SPECIFIC SERVICE / ALL**:
  ```bash
  # Stop only Ollama container
  docker compose stop ollama

  # Stop only Web container
  docker compose stop web

  # Stop ALL containers
  docker compose down
  ```

- **RESTART**:
  ```bash
  # Quick restart without rebuild
  docker compose restart

  # Restart specific service (e.g. web or ollama)
  docker compose restart web
  docker compose restart ollama

  # Full restart with clean rebuild
  docker compose up -d --build --force-recreate
  ```

---

### Option 3: macOS GPU Acceleration + Local Ollama (Recommended for Mac)

Docker on macOS cannot directly access Apple Silicon GPU hardware (Metal). For maximum inference speed with local Ollama:

1. **Install & Run Ollama natively on macOS**:
   Download from [ollama.com](https://ollama.com) and run in terminal:
   ```bash
   ollama pull qwen2.5-coder:7b
   ```
2. **Start Web Dashboard container**:
   ```bash
   docker compose up -d web
   ```
3. **Connect Web Dashboard to macOS Ollama**:
   - Open `http://localhost:8000`
   - Select **Ollama** as provider
   - Set **Ollama URL** to `http://host.docker.internal:11434`

---

## 💻 CLI Usage

You can also run agents directly in your command line:

```bash
# Run GitHub Agent CLI
python3 main.py github

# Run Slack Agent CLI
python3 main.py slack

# Run Jira Agent CLI
python3 main.py jira

# View help for any agent
python3 main.py github --help
```

---

## 🤗 How to Deploy to Hugging Face Spaces (All-in-One Docker + Ollama)

This project is configured with an **All-In-One Dockerfile** that runs both the **Web Dashboard** and an embedded **Ollama engine** directly inside Hugging Face Spaces.

**Benefits**:
- **0 Load on your Mac**: Runs 100% in the cloud.
- **$0 Cost**: Uses Hugging Face's free Docker Space tier.
- **Pre-loaded Model**: Automatically installs and pulls `qwen2.5-coder:7b` inside the container.

---

### Step 1: Create a Space on Hugging Face
1. Go to [huggingface.co/new-space](https://huggingface.co/new-space).
2. Choose **Space SDK**: **Docker** $\rightarrow$ **Blank**.
3. Set **Hardware**: `CPU basic (Free)`.
4. Click **Create Space**.

---

### Step 2: Push Repository to Hugging Face
Run these commands in your Mac terminal inside the project directory:

```bash
# 1. Commit updated Dockerfile
git add Dockerfile README.md
git commit -m "Configure All-in-One Dockerfile for Hugging Face Spaces with Ollama"

# 2. Add your Hugging Face Space repository as a remote
# Replace YOUR_USERNAME and YOUR_SPACE_NAME with your details
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME

# 3. Push code to Hugging Face
git push hf main
```

*(If prompted for a password, enter your Hugging Face Access Token with `write` permissions from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)).*

---

### Step 3: Access your Live App
1. Go to your Space page on Hugging Face.
2. Click the **App** tab.
3. Hugging Face will build the Docker container, start Ollama, download `llama3.2:1b`, and launch the Web Dashboard automatically.
4. Select **Ollama** as provider and start chatting for **$0 cost**!

---

## 🛠️ Project Structure

```
community_mcp_servers/
├── agents/              # Agent implementations (GitHub, Jira, Slack, etc.)
│   ├── github_agent.py
│   ├── jira_agent.py
│   ├── slack_agent.py
│   ├── telegram_agent.py
│   └── web_agent.py
├── lib/                 # Core MCP library & transports
│   ├── base_agent.py   # Base agent executor
│   ├── base_mcp.py     # MCP tool integration
│   └── base_transport.py
├── llm_providers/      # LLM provider adapters
│   ├── gemini.py
│   ├── groq_llm.py
│   ├── openai.py
│   └── ollama.py
├── web/                 # Dashboard UI frontend (index.html)
├── web_server.py        # FastAPI server backend
├── main.py             # CLI entry point
├── Dockerfile           # Production Docker build file
├── docker-compose.yml   # Multi-container setup (Web + Ollama)
├── .env.example        # Environment template
└── requirements.txt    # Python dependencies
```

---

## 🔑 Key API Links

- **GitHub**: [Personal Access Tokens](https://github.com/settings/tokens)
- **Slack**: [Slack App Management](https://api.slack.com/apps)
- **Jira**: [Atlassian API Tokens](https://id.atlassian.com/manage-profile/security/api-tokens)
- **Gemini**: [Google AI Studio](https://aistudio.google.com/app/apikey)
- **Groq**: [Groq Console](https://console.groq.com/keys)
