from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings

# --- LLM Configuration ---
# User-specified LLM settings
llm = ChatOpenAI(
    base_url="http://141.98.210.15:15203/v1",
    model_name="deep-seek-r1",
    temperature=0.5,
    api_key="324"
)

# --- Embeddings Configuration ---
# User-specified embeddings
try:
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    print("OllamaEmbeddings initialized successfully.")
except Exception as e:
    print(f"Warning: Could not initialize OllamaEmbeddings: {e}")
    embeddings = None

# --- Default System Prompt ---
DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant. Be concise and clear in your responses."

# --- Database Configuration ---
SQLITE_DB_PATH = "chatbot_sessions.sqlite"
