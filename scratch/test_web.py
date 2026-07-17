import asyncio
import os
from fastapi.testclient import TestClient
from web_server import app

client = TestClient(app)

def test_config():
    response = client.get("/api/config")
    print("Config Response:", response.status_code, response.json())

def test_telegram_tools():
    payload = {
        "provider": "gemini",
        "model": "models/gemini-3.1-flash-lite",
        "api_key": os.getenv("GOOGLE_API_KEY", "")
    }
    
    try:
        response = client.post("/api/agent/telegram/tools", json=payload)
        print("Telegram Tools Response:", response.status_code)
        if response.status_code == 200:
            print(response.json())
        else:
            print(response.text)
    except Exception as e:
        print("Exception caught:", e)

def test_telegram_chat():
    payload = {
        "agent_name": "telegram",
        "message": "list my individual chats",
        "session_id": "test_session",
        "provider": "gemini",
        "model": "models/gemini-3.1-flash-lite",
        "api_key": os.getenv("GOOGLE_API_KEY", "")
    }
    try:
        response = client.post("/api/chat", json=payload)
        print("Telegram Chat Response Status:", response.status_code)
        print("Response Body:", response.json())
    except Exception as e:
        print("Exception caught in chat:", e)

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(".env")
    
    print("Testing config...")
    test_config()
    
    print("\nTesting telegram tools...")
    test_telegram_tools()

    print("\nTesting telegram chat...")
    test_telegram_chat()
