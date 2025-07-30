import uuid
import sqlite3
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from . import models
from . import config
from .graph import graph_app, ensure_message_has_id

app = FastAPI(
    title="LangGraph Chatbot API",
    description="An API for interacting with a stateful LangGraph chatbot.",
    version="1.0.0"
)

# --- Helper Functions ---

def get_db_connection():
    """Establishes a connection to the SQLite database."""
    try:
        conn = sqlite3.connect(f"file:{config.SQLITE_DB_PATH}?mode=ro", uri=True)
        return conn
    except sqlite3.OperationalError:
        # This can happen if the DB doesn't exist yet.
        return None

async def stream_chat_responses(thread_id: str, human_message: HumanMessage):
    """
    Streams responses from the LangGraph app for a given thread and message.
    Yields the content of the AI's response chunks.
    """
    config_dict = {"configurable": {"thread_id": thread_id}}
    inputs = {"messages": [human_message]}

    # astream() yields the full state dictionary at each step.
    # We want to extract and stream the content of the latest AIMessage.
    async for event in graph_app.astream(inputs, config=config_dict):
        # The event dictionary holds the values of the state at that step
        all_messages = event.get("messages", [])
        if all_messages:
            latest_message = all_messages[-1]
            if isinstance(latest_message, AIMessage):
                yield latest_message.content + "\n"


# --- API Endpoints ---

@app.get("/sessions/", response_model=models.ListSessionsResponse)
def list_sessions():
    """
    Lists all available chat sessions (threads) from the database.
    """
    conn = get_db_connection()
    if not conn:
        return models.ListSessionsResponse(sessions=[])

    try:
        cursor = conn.cursor()
        # The table is named 'checkpoints' by default in SqliteSaver
        cursor.execute("SELECT DISTINCT thread_id FROM checkpoints")
        sessions = [models.SessionInfo(thread_id=row[0]) for row in cursor.fetchall()]
        return models.ListSessionsResponse(sessions=sessions)
    except sqlite3.OperationalError:
        # This can happen if the table doesn't exist yet on a fresh DB.
        return models.ListSessionsResponse(sessions=[])
    finally:
        if conn:
            conn.close()

@app.post("/sessions/new", response_model=models.SessionInfo)
def new_session():
    """
    Creates a new chat session with a default system prompt.
    """
    thread_id = str(uuid.uuid4())
    config_dict = {"configurable": {"thread_id": thread_id}}

    system_message = ensure_message_has_id(SystemMessage(content=config.DEFAULT_SYSTEM_PROMPT))

    initial_state = {
        "messages": [system_message],
        "summary": "",
        "messages_since_last_summary": 0
    }

    graph_app.update_state(config_dict, initial_state)
    return models.SessionInfo(thread_id=thread_id)

@app.post("/assistants/new", response_model=models.SessionInfo)
def new_assistant(request: models.NewAssistantRequest):
    """
    Creates a new chat session with a custom system prompt.
    """
    thread_id = str(uuid.uuid4())
    config_dict = {"configurable": {"thread_id": thread_id}}

    system_message = ensure_message_has_id(SystemMessage(content=request.system_prompt))

    initial_state = {
        "messages": [system_message],
        "summary": "",
        "messages_since_last_summary": 0
    }

    graph_app.update_state(config_dict, initial_state)
    return models.SessionInfo(thread_id=thread_id)

@app.post("/chat/{thread_id}")
async def chat(thread_id: str, request: models.ChatRequest):
    """
    Handles a chat interaction within a specific session, streaming the response.
    """
    config_dict = {"configurable": {"thread_id": thread_id}}

    # Verify the session exists
    current_state = graph_app.get_state(config_dict)
    if not current_state.values.get("messages"):
        raise HTTPException(status_code=404, detail="Session not found.")

    human_message = ensure_message_has_id(HumanMessage(content=request.message))

    # Return a streaming response
    return StreamingResponse(
        stream_chat_responses(thread_id, human_message),
        media_type="text/plain"
    )

# --- Terminal UI (for direct execution) ---

def run_terminal_chat():
    """
    Provides a terminal-based interface for chatting with the agent.
    """
    print("--- LangGraph Chatbot (Terminal Mode) ---")

    # --- Session Selection ---
    conn = get_db_connection()
    existing_threads = []
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id")
            existing_threads = [row[0] for row in cursor.fetchall()]
        except sqlite3.OperationalError:
            print("Database table not found. Starting with a new session.")
        finally:
            conn.close()

    thread_id = None
    if existing_threads:
        print("\nAvailable sessions:")
        for i, tid in enumerate(existing_threads):
            print(f"  {i+1}. {tid}")
        print(f"  {len(existing_threads)+1}. [NEW SESSION]")

        try:
            choice = int(input(f"Select a session [1-{len(existing_threads)+1}]: "))
            if 1 <= choice <= len(existing_threads):
                thread_id = existing_threads[choice-1]
                print(f"Resuming session: {thread_id}")
            elif choice != len(existing_threads)+1:
                print("Invalid choice. Starting new session.")
        except (ValueError, IndexError):
            print("Invalid input. Starting new session.")

    if not thread_id:
        thread_id = str(uuid.uuid4())
        print(f"Starting new session: {thread_id}")
        # Initialize the state for the new session
        config_dict = {"configurable": {"thread_id": thread_id}}
        system_message = ensure_message_has_id(SystemMessage(content=config.DEFAULT_SYSTEM_PROMPT))
        graph_app.update_state(config_dict, {"messages": [system_message]})


    # --- Chat Loop ---
    config_dict = {"configurable": {"thread_id": thread_id}}
    print("\nType 'exit', 'quit', or 'bye' to end.")
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("Bot: Goodbye!")
                break
            if not user_input.strip():
                continue

            human_message = ensure_message_has_id(HumanMessage(content=user_input))
            inputs = {"messages": [human_message]}

            print("Bot: Thinking...", end="", flush=True)

            # Use stream instead of astream for the synchronous terminal version
            final_response = ""
            for event in graph_app.stream(inputs, config=config_dict):
                all_messages = event.get("messages", [])
                if all_messages:
                    latest_message = all_messages[-1]
                    if isinstance(latest_message, AIMessage):
                        final_response = latest_message.content

            print(f"\rBot: {final_response}   ")

        except (KeyboardInterrupt, EOFError):
            print("\nBot: Goodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            import traceback
            traceback.print_exc()
            break

if __name__ == "__main__":
    run_terminal_chat()
