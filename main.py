from llm_chat import LLMChat
from dotenv import load_dotenv
import os
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

load_dotenv()

api_keys = {
    'anthropic': os.getenv('ANTHROPIC_API_KEY'),
    'openai': os.getenv('OPENAI_API_KEY'),
    'google': os.getenv('GOOGLE_API_KEY')
}

# Create a dictionary to store LLMChat instances for each user
user_sessions = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /start is issued."""
    user_id = update.effective_user.id
    if user_id not in user_sessions:
        user_sessions[user_id] = LLMChat(api_keys)
    await update.message.reply_text('Hello! I am your AI assistant. You can chat with me or use these commands:\n'
                                  '/switch - Switch between AI models\n'
                                  '/clear - Clear chat history\n'
                                  '/history - Show chat history')

async def switch_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Switch between different AI models."""
    user_id = update.effective_user.id
    llm = user_sessions.get(user_id)
    if not llm:
        llm = user_sessions[user_id] = LLMChat(api_keys)
    
    models = ['claude', 'chatgpt', 'gemini']
    current_index = models.index(llm.current_model)
    next_model = models[(current_index + 1) % len(models)]
    llm.set_model(next_model)
    await update.message.reply_text(f"Switched to {next_model}")

async def clear_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Clear the chat history."""
    user_id = update.effective_user.id
    llm = user_sessions.get(user_id)
    if llm:
        llm.clear_history()
    await update.message.reply_text("History cleared")

async def show_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Display the chat history."""
    user_id = update.effective_user.id
    llm = user_sessions.get(user_id)
    if not llm or not llm.history:
        await update.message.reply_text("No conversation history.")
        return
    
    history_text = "Conversation History:\n" + "-" * 30 + "\n"
    for msg in llm.history:
        role = msg["role"].capitalize()
        content = msg["content"]
        history_text += f"{role}: {content}\n"
    history_text += "-" * 30
    await update.message.reply_text(history_text)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming messages."""
    user_id = update.effective_user.id
    if user_id not in user_sessions:
        user_sessions[user_id] = LLMChat(api_keys)
    
    llm = user_sessions[user_id]
    response = llm.chat(update.message.text)
    await update.message.reply_text(f'{llm.current_model.capitalize()}: {response}')

def main():
    """Start the bot."""
    # Get your token from @BotFather
    telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
    if not telegram_token:
        raise ValueError("Please set TELEGRAM_BOT_TOKEN in your .env file")

    application = Application.builder().token(telegram_token).build()

    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("switch", switch_model))
    application.add_handler(CommandHandler("clear", clear_history))
    application.add_handler(CommandHandler("history", show_history))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Start the bot
    application.run_polling()

if __name__ == '__main__':
    main()
