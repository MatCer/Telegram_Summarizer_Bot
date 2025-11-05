import telebot
import pandas as pd
import datetime
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get secrets from environment variables
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
USE_GOOGLE_API = os.getenv("USE_GOOGLE_API", "false").lower() == "true"

# Validate required tokens
if not TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN environment variable is required!")

if USE_GOOGLE_API and not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is required when USE_GOOGLE_API is true!")

bot = telebot.TeleBot(TOKEN, parse_mode="Markdown")

# File to store group messages
LOG_FILE = "group_messages.csv"

# Predefined time intervals
TIME_INTERVALS = {
    "12hr": 12,
    "18hr": 18,
    "1day": 24,
    "2days": 48,
    "1week": 168,
}

# Initialize AI summarization
if USE_GOOGLE_API:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel('gemini-pro')
        print("✅ Using Google Gemini API for summarization")
    except ImportError:
        raise ImportError("google-generativeai package is required. Install it with: pip install google-generativeai")
else:
    try:
        from transformers import pipeline
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        print("✅ Using BART model for summarization")
    except ImportError:
        raise ImportError("transformers package is required. Install it with: pip install transformers torch")


def summarize_with_google(text):
    """Summarize text using Google Gemini API"""
    # Google Gemini has a large context window, but truncate if extremely long to be safe
    # Gemini Pro supports ~30k tokens, so ~200k characters should be safe
    if len(text) > 200000:
        text = text[:200000]
        print("Warning: Text truncated to 200k characters for Google API")
    
    prompt = f"""Please provide a concise summary of the following group chat messages. 
    Focus on the main topics, key points, and important information discussed.
    
    Messages:
    {text}
    
    Summary:"""
    
    try:
        response = model.generate_content(prompt)
        if not response or not response.text:
            raise ValueError("Empty response from Google API")
        return response.text.strip()
    except Exception as e:
        print(f"Error with Google API: {e}")
        raise


def summarize_with_bart(text):
    """Summarize text using BART model"""
    if len(text) > 1024:
        text = text[:1024]
    result = summarizer(text, max_length=100, min_length=30, do_sample=False)
    return result[0]["summary_text"]


def summarize_text(text):
    """Unified summarization function that uses the configured AI"""
    if USE_GOOGLE_API:
        return summarize_with_google(text)
    else:
        return summarize_with_bart(text)


# Command to start the bot
@bot.message_handler(commands=["start"])
def send_welcome(message):
    ai_type = "Google Gemini API" if USE_GOOGLE_API else "BART model"
    bot.reply_to(
        message,
        f"Hello! You can summarize your messages in the group.\n\n"
        f"*Current AI:* {ai_type}\n\n"
        "Use: `/summarize <option>`\n\n"
        "*Time-based options:* \n"
        "- `12hr` (Last 12 hours)\n"
        "- `18hr` (Last 18 hours)\n"
        "- `1day` (Last 24 hours)\n"
        "- `2days` (Last 2 days)\n"
        "- `1week` (Last 7 days)\n\n"
        "*Count-based options:* \n"
        "- `last <number>` (Last N messages)\n"
        "- Example: `/summarize last 50`\n\n"
        "*Examples:*\n"
        "- `/summarize 1day` (time-based)\n"
        "- `/summarize last 100` (count-based)"
    )


# Command to summarize messages based on selected time range or count
@bot.message_handler(commands=["summarize"])
def summarize_messages(message):
    chat_id = message.chat.id
    text_parts = message.text.split()

    # Check if using count-based option (e.g., "last 50")
    is_count_based = len(text_parts) >= 3 and text_parts[1].lower() == "last"
    
    if is_count_based:
        # Count-based summarization
        try:
            count = int(text_parts[2])
            if count <= 0:
                raise ValueError("Count must be positive")
            if count > 10000:
                bot.reply_to(message, "❌ Maximum count is 10000 messages. Please use a smaller number.")
                return
            
            option_display = f"last {count} messages"
            
        except (ValueError, IndexError):
            bot.reply_to(
                message,
                "Invalid format for count-based option.\n\n"
                "Use: `/summarize last <number>`\n"
                "Example: `/summarize last 50`"
            )
            return
    elif len(text_parts) == 2 and text_parts[1] in TIME_INTERVALS:
        # Time-based summarization (existing functionality)
        hours = TIME_INTERVALS[text_parts[1]]
        option_display = text_parts[1]
    else:
        bot.reply_to(
            message,
            "Invalid format. Use:\n"
            "- `/summarize <time_option>` (e.g., `/summarize 1day`)\n"
            "- `/summarize last <number>` (e.g., `/summarize last 50`)"
        )
        return

    try:
        df = pd.read_csv(LOG_FILE)
        df["date"] = pd.to_datetime(df["date"])

        # Filter messages from the current group
        if "chat_id" not in df.columns:
            # Legacy CSV without chat_id - filter all messages
            if is_count_based:
                # For count-based, get last N messages sorted by date
                group_messages = df.sort_values("date", ascending=False).head(count)
            else:
                # Time-based: filter by time range
                end_time = datetime.datetime.now()
                start_time = end_time - datetime.timedelta(hours=hours)
                group_messages = df[df["date"] >= start_time]
        else:
            # Filter by chat_id to ensure we only summarize messages from this group
            chat_filtered = df[df["chat_id"] == chat_id]
            
            if is_count_based:
                # For count-based, get last N messages sorted by date
                group_messages = chat_filtered.sort_values("date", ascending=False).head(count)
            else:
                # Time-based: filter by time range
                end_time = datetime.datetime.now()
                start_time = end_time - datetime.timedelta(hours=hours)
                group_messages = chat_filtered[(chat_filtered["date"] >= start_time) & (chat_filtered["date"] <= end_time)]

        if group_messages.empty:
            if is_count_based:
                bot.reply_to(message, f"No messages found in this group.")
            else:
                bot.reply_to(message, "No messages found in the selected time range.")
            return

        # Sort by date ascending for proper message order
        group_messages = group_messages.sort_values("date", ascending=True)

        # Combine messages into one text block for summarization
        messages_text = " ".join(group_messages["message"].tolist())

        # Check if messages_text is empty
        if not messages_text or not messages_text.strip():
            bot.reply_to(message, "No valid messages found.")
            return

        # Summarize messages
        try:
            summary = summarize_text(messages_text)
            bot.reply_to(message, f"📊 *Summary for {option_display}:*\n\n_{summary}_")
        except Exception as e:
            print(f"Summarization error: {e}")
            bot.reply_to(message, f"❌ Error generating summary: {str(e)}")

    except FileNotFoundError:
        bot.reply_to(message, "No messages found. Ensure message logging is enabled.")
    except Exception as e:
        print(f"Error: {e}")
        bot.reply_to(message, f"❌ An error occurred: {str(e)}")


# Function to log messages in the group
@bot.message_handler(func=lambda message: True, content_types=["text"])
def log_messages(message):
    """Logs all text messages from the group."""
    if message.chat.type in ["group", "supergroup"]:
        user_id = message.from_user.id
        username = message.from_user.username or "Unknown"
        chat_id = message.chat.id
        text = message.text
        date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"Logging message: {username} ({user_id}) in chat {chat_id} - {text}")  # Debugging

        # Save message to CSV (include chat_id to support multiple groups)
        df = pd.DataFrame([[user_id, username, chat_id, text, date]], columns=["user_id", "username", "chat_id", "message", "date"])

        try:
            existing_df = pd.read_csv(LOG_FILE)
            df = pd.concat([existing_df, df], ignore_index=True)
        except FileNotFoundError:
            pass

        df.to_csv(LOG_FILE, index=False)


# Start the bot
print("Bot is running...")
bot.polling(none_stop=True)
