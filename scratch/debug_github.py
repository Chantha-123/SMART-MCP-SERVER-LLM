import asyncio
import os
import dotenv
from agents import get_github_agent
from lib.state import RuntimeState
from lib.base_agent import ensure_agent_initialized

dotenv.load_dotenv()

async def main():
    state = RuntimeState("github")
    agent = get_github_agent()
    await ensure_agent_initialized(state, agent)
    
    print("Agent executor:", state.agent_executor)
    
    # Try a simple invoke to list repositories
    prompt = "Search for repositories belonging to me, or search for repositories with a query matching my username"
    print(f"\nPrompt: {prompt}")
    
    # Run the react agent and print the history steps
    async for event in state.agent_executor.astream({"messages": [("user", prompt)]}):
        print("\n--- Event ---")
        print(event)

if __name__ == "__main__":
    asyncio.run(main())
