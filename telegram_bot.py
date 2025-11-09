import telebot
import pandas as pd
import datetime
import os
import time

import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get secrets from environment variables
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Validate required tokens
if not TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN environment variable is required!")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is required!")

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

# Initialize Google Gemini API for summarization
try:
    import google.generativeai as genai
    # Set UTF-8 encoding for stdout to handle Unicode characters
    if sys.stdout.encoding != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except (AttributeError, ValueError):
            # Fallback for older Python versions or when reconfigure is not available
            pass
    genai.configure(api_key=GOOGLE_API_KEY)
    
    # Try to find an available model - list of models to try (free tier compatible)
    model_names_to_try = [
        'gemini-2.0-flash-exp',
        'gemini-1.5-flash',
        'gemini-1.5-pro',
        'gemini-pro',
        'gemini-2.5-flash-lite'
    ]
    
    model = None
    model_name = None
    
    # First, try to list available models
    try:
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        print(f"Available models: {available_models}")
        
        # Try to find a matching model from our list
        for name in model_names_to_try:
            # Check if model name matches (could be full path like 'models/gemini-1.5-flash')
            matching = [m for m in available_models if name in m]
            if matching:
                model_name = matching[0].split('/')[-1] if '/' in matching[0] else matching[0]
                model = genai.GenerativeModel(model_name)
                print(f"[OK] Using Google Gemini API model: {model_name}")
                break
    except Exception as e:
        print(f"Could not list models: {e}")
        # Fallback: try models directly
        for name in model_names_to_try:
            try:
                model = genai.GenerativeModel(name)
                model_name = name
                print(f"[OK] Using Google Gemini API model: {name}")
                break
            except Exception:
                continue
    
    if model is None:
        raise ValueError("Could not initialize any Gemini model. Please check your API key and available models.")
        
except ImportError:
    raise ImportError("google-generativeai package is required. Install it with: pip install google-generativeai")


def summarize_text(text):
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


def safe_reply_to(message, text, **kwargs):
    """Safely reply to a message with error handling."""
    try:
        return bot.reply_to(message, text, **kwargs)
    except Exception as e:
        print(f"Error sending reply: {e}")
        # Try to send a simple message instead if reply fails
        try:
            bot.send_message(message.chat.id, text, **kwargs)
        except Exception as e2:
            print(f"Error sending message: {e2}")
        return None


def save_message_to_csv(user_id, username, chat_id, text, date_str):
    """Helper function to save a message to CSV"""
    df = pd.DataFrame([[user_id, username, chat_id, text, date_str]], 
                      columns=["user_id", "username", "chat_id", "message", "date"])
    
    try:
        # Check if file exists and has content before reading
        if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > 0:
            existing_df = pd.read_csv(LOG_FILE)
            # Check if message already exists (avoid duplicates)
            if not existing_df.empty and "message" in existing_df.columns:
                # Simple duplicate check: same chat_id, message text, and similar timestamp
                existing_df["date"] = pd.to_datetime(existing_df["date"])
                date_obj = pd.to_datetime(date_str)
                # Check for duplicates within 1 second
                mask = (existing_df["chat_id"] == chat_id) & \
                       (existing_df["message"] == text) & \
                       (abs((existing_df["date"] - date_obj).dt.total_seconds()) < 1)
                if mask.any():
                    return False  # Message already exists
            df = pd.concat([existing_df, df], ignore_index=True)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        pass
    
    df.to_csv(LOG_FILE, index=False)
    return True


# Command to start the bot
@bot.message_handler(commands=["start"])
def send_welcome(message):
    safe_reply_to(
        message,
        f"Hello! You can summarize your messages in the group using Google Gemini AI.\n\n"
        "*Commands:*\n"
        "- `/summarize <option>` - Summarize messages\n"
        "- `/sync` - Check message logging status\n\n"
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
        "- `/summarize last 100` (count-based)\n\n"
        "*Note:* Bots can only access messages sent after they were added to the group."
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
                safe_reply_to(message, "❌ Maximum count is 10000 messages. Please use a smaller number.")
                return
            
            option_display = f"last {count} messages"
            
        except (ValueError, IndexError):
            safe_reply_to(
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
        safe_reply_to(
            message,
            "Invalid format. Use:\n"
            "- `/summarize <time_option>` (e.g., `/summarize 1day`)\n"
            "- `/summarize last <number>` (e.g., `/summarize last 50`)"
        )
        return

    try:
        # Check if file exists and has content
        if not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE) == 0:
            safe_reply_to(message, "No messages found. The message log is empty. Start chatting in the group to log messages.")
            return
        
        try:
            df = pd.read_csv(LOG_FILE)
        except pd.errors.EmptyDataError:
            safe_reply_to(message, "No messages found. The message log is empty. Start chatting in the group to log messages.")
            return
        
        # Check if DataFrame is empty
        if df.empty:
            safe_reply_to(message, "No messages found. The message log is empty. Start chatting in the group to log messages.")
            return
        
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
                group_messages = df[(df["date"] >= start_time) & (df["date"] <= end_time)]
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
                safe_reply_to(message, f"No messages found in this group.")
            else:
                safe_reply_to(message, "No messages found in the selected time range.")
            return

        # Sort by date ascending for proper message order
        group_messages = group_messages.sort_values("date", ascending=True)

        # Combine messages into one text block for summarization
        messages_text = " ".join(group_messages["message"].tolist())

        # Check if messages_text is empty
        if not messages_text or not messages_text.strip():
            safe_reply_to(message, "No valid messages found.")
            return

        # Summarize messages
        try:
            summary = summarize_text(messages_text)
            safe_reply_to(message, f"📊 *Summary for {option_display}:*\n\n_{summary}_")
        except Exception as e:
            print(f"Summarization error: {e}")
            safe_reply_to(message, f"❌ Error generating summary: {str(e)}")

    except FileNotFoundError:
        safe_reply_to(message, "No messages found. Ensure message logging is enabled.")
    except Exception as e:
        print(f"Error: {e}")
        safe_reply_to(message, f"❌ An error occurred: {str(e)}")


# Command to sync/fetch recent messages
@bot.message_handler(commands=["sync"])
def sync_messages(message):
    """Provides information about message syncing."""
    if message.chat.type not in ["group", "supergroup"]:
        safe_reply_to(message, "This command can only be used in groups.")
        return
    
    chat_id = message.chat.id
    
    try:
        # Count messages in the log for this chat
        if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > 0:
            df = pd.read_csv(LOG_FILE)
            if "chat_id" in df.columns:
                chat_messages = df[df["chat_id"] == chat_id]
                message_count = len(chat_messages)
            else:
                message_count = len(df)
            
            safe_reply_to(
                message,
                f"✅ Message logging is active!\n\n"
                f"📊 *Current stats:*\n"
                f"- Messages logged in this group: {message_count}\n\n"
                f"*Note:* The bot automatically logs all new messages sent after it was added to the group. "
                f"Messages sent before the bot was added cannot be retrieved due to Telegram API restrictions."
            )
        else:
            safe_reply_to(
                message,
                "📝 Message logging is ready!\n\n"
                "The bot will automatically log all new messages sent in this group. "
                "Messages sent before the bot was added cannot be retrieved due to Telegram API restrictions."
            )
    except Exception as e:
        print(f"Sync command error: {e}")
        safe_reply_to(
            message, 
            "⚠️ Error checking message log.\n\n"
            "The bot automatically logs all new messages sent after it was added to the group. "
            "Messages sent before the bot was added cannot be retrieved due to Telegram API restrictions."
        )


# Function to log messages in the group
@bot.message_handler(func=lambda message: True, content_types=["text"])
def log_messages(message):
    """Logs all text messages from the group."""
    if message.chat.type in ["group", "supergroup"]:
        # Skip if message.text is None (e.g., for media messages)
        if message.text is None:
            return
        
        # Skip bot commands (they're handled separately)
        if message.text.startswith('/'):
            return
            
        user_id = message.from_user.id
        username = message.from_user.username or "Unknown"
        chat_id = message.chat.id
        text = message.text
        date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Safe print that handles Unicode characters
        try:
            print(f"Logging message: {username} ({user_id}) in chat {chat_id} - {text}")
        except UnicodeEncodeError:
            # Fallback for Windows console encoding issues
            safe_text = text.encode('ascii', 'replace').decode('ascii')
            print(f"Logging message: {username} ({user_id}) in chat {chat_id} - {safe_text}")

        # Save message to CSV using helper function
        save_message_to_csv(user_id, username, chat_id, text, date)


# Start the bot with error handling
def start_bot():
    """Start the bot with error handling and retry logic."""
    retry_delay = 5  # seconds
    
    while True:
        try:
            print("Bot is starting...")
            bot.polling(none_stop=True, interval=1, timeout=20)
        except KeyboardInterrupt:
            print("\n⏹️  Bot stopped by user.")
            break
        except Exception as e:
            error_str = str(e)
            # Check for 409 Conflict error (multiple bot instances)
            if "409" in error_str or "Conflict" in error_str or "terminated by other getUpdates" in error_str:
                print(f"❌ Error 409: Another bot instance may be running!")
                print("   Make sure only one instance of the bot is running.")
                print("   If you're sure only one instance is running, wait a few seconds and the bot will retry.")
            # Check for connection errors
            elif "Connection" in error_str or "RemoteDisconnected" in error_str or "Connection aborted" in error_str:
                print(f"❌ Connection error: {e}")
                print("   This may be due to network issues or Telegram API being temporarily unavailable.")
            else:
                print(f"❌ Unexpected error: {e}")
            
            print(f"   Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)


if __name__ == "__main__":
    print("Bot is running...")
    start_bot()
