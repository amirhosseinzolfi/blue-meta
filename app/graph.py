import os
import uuid
import operator
from typing import Annotated, Literal

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, RemoveMessage
import sqlite3
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.checkpoint.sqlite import SqliteSaver

from .config import llm, embeddings, SQLITE_DB_PATH

# --- Constants for Summarization ---
MESSAGES_TO_KEEP_AFTER_SUMMARY = 2
NEW_MESSAGES_THRESHOLD_FOR_SUMMARY = 10

# --- Helper to ensure message has an ID ---
def ensure_message_has_id(message: BaseMessage) -> BaseMessage:
    if not hasattr(message, 'id') or message.id is None or not isinstance(message.id, str):
        message.id = str(uuid.uuid4())
    return message

# --- Define Graph State with Summary and New Counter ---
class AgentState(MessagesState):
    summary: str
    messages_since_last_summary: Annotated[int, operator.add]

# --- Define Nodes ---

def call_llm_node(state: AgentState) -> dict:
    print("--- Node: call_llm_node ---")
    current_messages_in_state = state['messages']
    summary = state.get("summary", "")

    messages_to_send_to_llm = []
    if summary:
        print(f"Prepending summary: '{summary[:100]}...'")
        messages_to_send_to_llm.append(ensure_message_has_id(SystemMessage(content=f"This is a summary of the conversation so far: {summary}")))

    for msg in current_messages_in_state:
        messages_to_send_to_llm.append(msg)

    if not messages_to_send_to_llm:
         messages_to_send_to_llm.append(ensure_message_has_id(HumanMessage(content="Hello.")))

    response = llm.invoke(messages_to_send_to_llm)

    return {"messages": [ensure_message_has_id(response)], "messages_since_last_summary": 1}


def should_summarize_node(state: AgentState) -> Literal["summarize_conversation_node", "__end__"]:
    print("--- Node: should_summarize_node ---")
    num_since_last_summary = state.get("messages_since_last_summary", 0)

    if num_since_last_summary >= NEW_MESSAGES_THRESHOLD_FOR_SUMMARY:
        print(f"Condition met for summarization: {num_since_last_summary} >= {NEW_MESSAGES_THRESHOLD_FOR_SUMMARY}")
        return "summarize_conversation_node"
    else:
        print(f"Condition not met ({num_since_last_summary}). Ending turn.")
        return "__end__"

def summarize_conversation_node(state: AgentState):
    print("--- Node: summarize_conversation_node ---")
    current_messages_in_state = state['messages']

    messages_to_summarize_content_from = [
        m for m in current_messages_in_state if isinstance(m, (HumanMessage, AIMessage, SystemMessage))
    ]

    existing_summary = state.get("summary", "")
    prompt_header = ("Please extend this summary with the new conversation excerpts below.\n"
                     if existing_summary
                     else "Please create a concise summary of the following conversation:\n")

    formatted_messages = "\n".join([f"{m.__class__.__name__}: {m.content}" for m in messages_to_summarize_content_from])
    full_prompt = prompt_header + formatted_messages
    if existing_summary:
        full_prompt = f"Previous Summary:\n{existing_summary}\n\n{full_prompt}"

    summary_llm_response = llm.invoke([ensure_message_has_id(HumanMessage(content=full_prompt))])
    new_summary = summary_llm_response.content.strip()
    print(f"Generated new summary: '{new_summary[:100]}...'")

    # Identify messages to remove
    num_to_remove = len(messages_to_summarize_content_from) - MESSAGES_TO_KEEP_AFTER_SUMMARY
    messages_to_remove = messages_to_summarize_content_from[:num_to_remove]

    delete_directives = [RemoveMessage(id=m.id) for m in messages_to_remove if hasattr(m, 'id') and m.id]

    # The counter should be reset. The messages we keep count towards the *next* summary cycle.
    # The number of messages we are *keeping* is the new value for the counter.
    new_counter_value = len(current_messages_in_state) - len(delete_directives)

    return {
        "summary": new_summary,
        "messages": delete_directives,
        "messages_since_last_summary": new_counter_value, # Reset counter
    }

def create_graph():
    """Creates and compiles the LangGraph agent."""
    workflow = StateGraph(AgentState)

    workflow.add_node("llm_caller", call_llm_node)
    workflow.add_node("summarize_conversation_node", summarize_conversation_node)

    workflow.set_entry_point("llm_caller")

    workflow.add_conditional_edges(
        "llm_caller",
        should_summarize_node,
        {"summarize_conversation_node": "summarize_conversation_node", "__end__": END},
    )
    workflow.add_edge("summarize_conversation_node", END)

    # The connection needs to be persistent for the lifespan of the app.
    # check_same_thread=False is crucial for multi-threaded usage in FastAPI.
    conn = sqlite3.connect(SQLITE_DB_PATH, check_same_thread=False)
    memory = SqliteSaver(conn=conn)

    app = workflow.compile(checkpointer=memory)

    return app

# A single, compiled graph instance to be used by the app
graph_app = create_graph()
