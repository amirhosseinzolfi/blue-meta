from pydantic import BaseModel
from typing import List, Optional

class ChatRequest(BaseModel):
    """Request model for sending a message to a session."""
    message: str

class ChatResponse(BaseModel):
    """Response model for receiving a message from the bot."""
    response: str

class NewAssistantRequest(BaseModel):
    """Request model for creating a new session with a custom system prompt."""
    system_prompt: str

class SessionInfo(BaseModel):
    """Data model for information about a single session."""
    thread_id: str

class ListSessionsResponse(BaseModel):
    """Response model for listing all available sessions."""
    sessions: List[SessionInfo]
