# LangGraph Multi-Session Chatbot Platform

This project is a multi-session AI chatbot platform built with FastAPI, LangChain, and LangGraph. It features a stateful backend that can handle long conversations through summarization and supports multiple frontends.

## Features

- **Stateful Conversations**: The bot remembers previous turns in a conversation using a SQLite-backed checkpointer.
- **Conversation Summarization**: Automatically summarizes long conversations to manage context length.
- **FastAPI Backend**: A robust, asynchronous API for handling chat logic.
- **Multiple Frontends**:
    - **Terminal UI**: A command-line interface for direct interaction.
    - **Telegram Bot**: A fully functional Telegram bot.
- **Session Management**: API endpoints to create, list, and interact with distinct chat sessions.
- **Customizable Assistants**: Ability to start a new session with a custom system prompt.

## Project Structure

```
.
├── app/
│   ├── __init__.py
│   ├── main.py         # FastAPI app, API endpoints, and Terminal UI
│   ├── graph.py        # Core LangGraph agent logic
│   ├── config.py       # LLM, embeddings, and other settings
│   └── models.py       # Pydantic models for API
├── telegram_bot.py     # Telegram bot frontend
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Setup

### Prerequisites

- Python 3.8+
- An LLM API endpoint compatible with the OpenAI API format.
- (Optional) An Ollama instance for embeddings.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure the application:**
    - Open `app/config.py` and review the settings. Update the `base_url` and `api_key` for the `llm` object to point to your LLM provider.
    - Open `telegram_bot.py` and replace the placeholder `TELEGRAM_BOT_TOKEN` with your actual bot token.

## How to Run

You need to run the FastAPI backend first, and then you can run the Telegram bot in a separate terminal.

### 1. Run the Backend Server

This server hosts the API that the Telegram bot will call.

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The API documentation will be available at `http://127.0.0.1:8000/docs`.

### 2. Run the Terminal UI (Alternative to Server)

If you just want to chat directly in your terminal, you can run the `main.py` script:

```bash
python app/main.py
```

This will give you an option to select an existing session or start a new one.

### 3. Run the Telegram Bot

In a **new terminal**, run the following command:

```bash
python telegram_bot.py
```

Your Telegram bot should now be online and responding to messages.

## API Endpoints

- `GET /sessions/`: Lists all available session thread IDs.
- `POST /sessions/new`: Creates a new chat session with a default system prompt.
- `POST /assistants/new`: Creates a new session with a custom system prompt provided in the request body.
- `POST /chat/{thread_id}`: Sends a message to a specific session and streams the response back.
