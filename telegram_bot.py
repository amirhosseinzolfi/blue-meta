import os
import httpx
import asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# --- Configuration ---
TELEGRAM_BOT_TOKEN = "8041866230:AAFtkHZhvm7_IsLOCEIKFWskycfVb5KSGuE"  # As provided by the user
FASTAPI_BASE_URL = "http://127.0.0.1:8000"

# In-memory storage for user sessions. Maps chat_id to thread_id.
# Note: This is not persistent. A proper DB would be needed for production.
user_sessions = {}

# --- Helper Functions ---

async def get_or_create_session(chat_id: int) -> str:
    """
    Retrieves the existing session ID for a user or creates a new one.
    """
    if chat_id in user_sessions:
        return user_sessions[chat_id]

    # If no session, create one via the API
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(f"{FASTAPI_BASE_URL}/sessions/new")
            response.raise_for_status()
            session_data = response.json()
            thread_id = session_data.get("thread_id")
            if thread_id:
                user_sessions[chat_id] = thread_id
                return thread_id
            else:
                return None
        except httpx.RequestError as e:
            print(f"Error creating new session for {chat_id}: {e}")
            return None

# --- Command Handlers ---

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handles the /start command. Greets the user and ensures a session is created.
    """
    chat_id = update.message.chat_id
    await update.message.reply_text("Welcome to the LangGraph Chatbot! I'm ready to chat.")

    thread_id = await get_or_create_session(chat_id)
    if thread_id:
        await update.message.reply_text(f"Your new chat session is ready. Session ID: `{thread_id}`", parse_mode='MarkdownV2')
    else:
        await update.message.reply_text("Sorry, I couldn't start a new session for you right now.")

async def new_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handles the /new command, forcing creation of a new session.
    """
    chat_id = update.message.chat_id
    # Clear old session from memory
    if chat_id in user_sessions:
        del user_sessions[chat_id]

    await update.message.reply_text("Starting a fresh chat session...")
    thread_id = await get_or_create_session(chat_id)
    if thread_id:
        await update.message.reply_text(f"Your new chat session is ready. Session ID: `{thread_id}`", parse_mode='MarkdownV2')
    else:
        await update.message.reply_text("Sorry, I couldn't start a new session for you right now.")

async def session_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handles the /session command, showing the current session ID.
    """
    chat_id = update.message.chat_id
    if chat_id in user_sessions:
        thread_id = user_sessions[chat_id]
        await update.message.reply_text(f"Your current session ID is: `{thread_id}`", parse_mode='MarkdownV2')
    else:
        await update.message.reply_text("You don't have an active session. Send a message or use /start to begin.")

# --- Message Handler ---

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handles regular text messages, sending them to the backend and returning the response.
    """
    chat_id = update.message.chat_id
    user_message = update.message.text

    # Ensure a session exists
    thread_id = await get_or_create_session(chat_id)
    if not thread_id:
        await update.message.reply_text("I'm having trouble managing our session. Please try again later.")
        return

    # Notify user that the bot is working
    await context.bot.send_chat_action(chat_id=chat_id, action='typing')

    # Stream the response from the backend
    full_response = ""
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", f"{FASTAPI_BASE_URL}/chat/{thread_id}", json={"message": user_message}) as response:
                response.raise_for_status()
                async for chunk in response.aiter_text():
                    full_response += chunk
    except httpx.RequestError as e:
        await update.message.reply_text(f"Sorry, an error occurred while talking to my brain: {e}")
        return
    except httpx.HTTPStatusError as e:
        await update.message.reply_text(f"Sorry, something went wrong: {e.response.status_code} {e.response.text}")
        return

    if full_response.strip():
        await update.message.reply_text(full_response.strip())
    else:
        await update.message.reply_text("I don't have a response for that.")


# --- Main Application Setup ---

def main() -> None:
    """
    Runs the Telegram bot.
    """
    print("Starting Telegram Bot...")

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Add handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("new", new_command))
    application.add_handler(CommandHandler("session", session_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Run the bot
    application.run_polling()

if __name__ == "__main__":
    main()
