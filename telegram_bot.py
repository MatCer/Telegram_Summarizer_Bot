import telebot
import pandas as pd
import datetime
import os
import time
import sqlite3
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get secrets from environment variables
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DB_FILE = "bot_data.db"

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
    
    # Use gemini-2.0-flash-exp directly
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    print(f"[OK] Using Google Gemini API model: gemini-2.0-flash-exp")
        
except ImportError:
    raise ImportError("google-generativeai package is required. Install it with: pip install google-generativeai")
except Exception as e:
    raise ValueError(f"Could not initialize Gemini model: {e}")

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS messages
                 (user_id INTEGER, username TEXT, chat_id INTEGER, 
                  message TEXT, date TIMESTAMP)''')
    conn.commit()
    conn.close()

def summarize_text(text):
    """Summarize text using Google Gemini API"""
    # Google Gemini has a large context window, but truncate if extremely long to be safe
    # Gemini Pro supports ~30k tokens, so ~200k characters should be safe
    if len(text) > 200000:
        text = text[:200000]
        print("Warning: Text truncated to 200k characters for Google API")
    
    prompt = f"""You are a punchy, energetic, snarky AI assistant with big “I can’t believe I have to deal with this group” energy — but in a fun, lovable way. Your name is REPetity, your nickname is REP. You are male.
    Your tone is witty, a little bitchy, and confidently sassy.
    You roast lightly, never cruelty. You’re sharp, not toxic.
    You ALWAYS keep summaries accurate, but you’re allowed to add spicy commentary as long as it doesn’t distort facts. You are allowed to mention specific user statements and respond to them.
    You are allowed to mention specific user statements and respond to them. > change this to " Mention specific user statements always tagging their name with @username and respond to them. All users are female.
    
    Format:
    - Give a bold TL;DR with attitude.
    - Then a few bullet points that mix facts with playful sass.
    - Keep it short and punchy.
    - Give advice on one randomly picked topic that came up.
    
    Summarize the following messages:
    {text}
    
    Summary:"""
    
    try:
        response = model.generate_content(prompt)
        if not response:
            raise ValueError("Empty response from Google API")
        # Handle different response formats
        if hasattr(response, 'text') and response.text:
            return response.text.strip()
        elif hasattr(response, 'parts') and response.parts:
            # Sometimes response comes in parts
            text_parts = [part.text for part in response.parts if hasattr(part, 'text') and part.text]
            if text_parts:
                return " ".join(text_parts).strip()
        raise ValueError("Empty or invalid response from Google API")
    except Exception as e:
        print(f"Error with Google API: {e}")
        raise


def safe_reply_to(message, text, **kwargs):
    """Safely reply to a message with error handling."""
    try:
        # Telegram has a 4096 character limit for messages
        max_length = 4096
        reply_text = text
        if len(reply_text) > max_length:
            # Truncate and add notice
            reply_text = reply_text[:max_length - 50] + "\n\n... (message truncated)"
            print(f"Warning: Message truncated to {max_length} characters")
        
        return bot.reply_to(message, reply_text, **kwargs)
    except Exception as e:
        print(f"Error sending reply: {e}")
        # Try to send a simple message instead if reply fails
        try:
            # Validate chat exists before trying to send message
            if not hasattr(message, 'chat') or not message.chat or not hasattr(message.chat, 'id'):
                print(f"Error: Cannot send message - chat information not available")
                return None
            
            # Also truncate for send_message
            send_text = text
            if len(send_text) > 4096:
                send_text = send_text[:4096 - 50] + "\n\n... (message truncated)"
            bot.send_message(message.chat.id, send_text, **kwargs)
        except Exception as e2:
            print(f"Error sending message: {e2}")
        return None


def save_message_to_csv(user_id, username, chat_id, text, date_str):
    """Helper function to save a message to CSV"""
    try:
        # Validate input
        if text is None:
            text = ""
        if not isinstance(text, str):
            text = str(text)
        
        df = pd.DataFrame([[user_id, username, chat_id, text, date_str]], 
                          columns=["user_id", "username", "chat_id", "message", "date"])
        
        # Check if file exists and has content before reading
        if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > 0:
            try:
                existing_df = pd.read_csv(LOG_FILE)
                # Check if message already exists (avoid duplicates)
                if not existing_df.empty and "message" in existing_df.columns:
                    try:
                        # Simple duplicate check: same chat_id, message text, and similar timestamp
                        existing_df["date"] = pd.to_datetime(existing_df["date"], errors='coerce')
                        date_obj = pd.to_datetime(date_str, errors='coerce')
                        
                        # Filter out NaT values (invalid dates)
                        valid_dates = existing_df["date"].notna() & pd.notna(date_obj)
                        if valid_dates.any():
                            # Check if chat_id column exists for duplicate checking
                            if "chat_id" in existing_df.columns:
                                mask = (existing_df["chat_id"] == chat_id) & \
                                       (existing_df["message"] == text) & \
                                       valid_dates & \
                                       (abs((existing_df["date"] - date_obj).dt.total_seconds()) < 1)
                            else:
                                # Legacy CSV without chat_id - check only message and date
                                mask = (existing_df["message"] == text) & \
                                       valid_dates & \
                                       (abs((existing_df["date"] - date_obj).dt.total_seconds()) < 1)
                            
                            if mask.any():
                                return False  # Message already exists
                    except Exception as e:
                        # If date parsing fails, skip duplicate check but still save
                        print(f"Warning: Could not check for duplicates: {e}")
                
                # Ensure all required columns exist
                required_columns = ["user_id", "username", "chat_id", "message", "date"]
                for col in required_columns:
                    if col not in existing_df.columns:
                        existing_df[col] = None
                
                df = pd.concat([existing_df, df], ignore_index=True)
            except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
                print(f"Warning: Could not read existing CSV, creating new one: {e}")
                # Continue with new DataFrame
            except Exception as e:
                print(f"Error reading CSV file: {e}")
                # Continue with new DataFrame
        
        # Write to CSV with error handling
        df.to_csv(LOG_FILE, index=False, encoding='utf-8')
        return True
    except Exception as e:
        print(f"Error saving message to CSV: {e}")
        return False

def save_message_to_db(user_id, username, chat_id, text, date_str):
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("INSERT INTO messages VALUES (?, ?, ?, ?, ?)",
                  (user_id, username, chat_id, text, date_str))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error saving to SQLite: {e}")
        return False

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
    # Check if message.text exists
    if not message.text:
        safe_reply_to(message, "❌ Invalid command format.")
        return
    
    # Validate chat exists
    if not hasattr(message, 'chat') or not message.chat:
        safe_reply_to(message, "❌ Error: Could not access chat information.")
        return
    
    chat_id = message.chat.id
    text_parts = message.text.split()

    # Determine selection mode (count-based vs time-based)
    is_count_based = len(text_parts) >= 3 and text_parts[1].lower() == "last"
    
    if is_count_based:
        try:
            count = int(text_parts[2])
            if count <= 0: raise ValueError
            if count > 10000:
                safe_reply_to(message, "❌ Maximum count is 10000 messages.")
                return
            option_display = f"last {count} messages"
        except (ValueError, IndexError):
            safe_reply_to(message, "Usage: `/summarize last <number>`")
            return
    elif len(text_parts) == 2 and text_parts[1] in TIME_INTERVALS:
        hours = TIME_INTERVALS[text_parts[1]]
        option_display = text_parts[1]
    else:
        safe_reply_to(message, "Invalid format. Use `/summarize 1day` or `/summarize last 50`.")
        return

    # --- SQLITE DATA RETRIEVAL ---
    try:
        conn = sqlite3.connect(DB_FILE)
        # We use pandas read_sql_query for compatibility with your existing list processing
        if is_count_based:
            # Get last X messages. Note: We sort DESC to get the limit, then reverse it later.
            query = "SELECT username, message FROM messages WHERE chat_id = ? ORDER BY date DESC LIMIT ?"
            group_messages = pd.read_sql_query(query, conn, params=(chat_id, count))
            # Reverse to maintain chronological order for Gemini
            group_messages = group_messages.iloc[::-1]
        else:
            # Get messages since X hours ago
            start_time = (datetime.datetime.now() - datetime.timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
            query = "SELECT username, message FROM messages WHERE chat_id = ? AND date >= ? ORDER BY date ASC"
            group_messages = pd.read_sql_query(query, conn, params=(chat_id, start_time))
        
        conn.close()

        if group_messages.empty:
            safe_reply_to(message, f"No messages found for {option_display}.")
            return

        # Process the messages into a single string
        # Use .fillna() to ensure we don't have crash-inducing 'None' values
        group_messages['formatted'] = (
            group_messages['username'].fillna('Unknown') + 
            ": " + 
            group_messages['message'].fillna('')
        )
        
        message_list = group_messages['formatted'].tolist()
        messages_text = " ".join(message_list)

        # Final safety check on text length before hitting Gemini API
        if len(messages_text) > 500000: # 500k chars is plenty for a summary
            messages_text = messages_text[-500000:]
            print("Warning: Input text truncated to 500k characters.")

        # --- GENERATE SUMMARY ---
        try:
            summary = summarize_text(messages_text)
            safe_reply_to(message, f"📊 *Summary for {option_display}:*\n\n_{summary}_")
            
            # Optional: Perform periodic cleanup of old messages (e.g. older than 7 days)
            cleanup_old_messages(days_to_keep=7)
            
        except Exception as e:
            print(f"Gemini Error: {e}")
            safe_reply_to(message, "❌ Error generating summary. Gemini might be busy.")

    except sqlite3.Error as e:
        print(f"Database Error: {e}")
        safe_reply_to(message, "❌ Database error. Please check logs.")
    except Exception as e:
        print(f"Unexpected Error: {e}")
        safe_reply_to(message, "❌ An unexpected error occurred.")


# Command to sync/fetch recent messages
@bot.message_handler(commands=["sync"])
def sync_messages(message):
    """Provides information about message syncing."""
    # Validate chat exists
    if not hasattr(message, 'chat') or not message.chat:
        safe_reply_to(message, "❌ Error: Could not access chat information.")
        return
    
    if message.chat.type not in ["group", "supergroup"]:
        safe_reply_to(message, "This command can only be used in groups.")
        return
    
    chat_id = message.chat.id
    
    try:
        # Count messages in the log for this chat
        if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > 0:
            try:
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
            except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
                print(f"Error reading CSV in sync command: {e}")
                safe_reply_to(
                    message,
                    "⚠️ Error reading message log file.\n\n"
                    "The bot will continue to log new messages going forward."
                )
            except Exception as e:
                print(f"Unexpected error in sync command: {e}")
                safe_reply_to(
                    message,
                    "⚠️ Error checking message log.\n\n"
                    "The bot automatically logs all new messages sent after it was added to the group."
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
    try:
        # Validate chat exists
        if not hasattr(message, 'chat') or not message.chat:
            return
        
        if message.chat.type in ["group", "supergroup"]:
            # Skip if message.text is None (e.g., for media messages)
            if message.text is None:
                return
            
            # Skip bot commands (they're handled separately)
            if message.text.startswith('/'):
                return
            
            # Check if from_user exists (may be None in channels or certain message types)
            if message.from_user is None:
                user_id = 0
                username = "Unknown"
            else:
                user_id = message.from_user.id
                username = message.from_user.username or "Unknown"
            
            chat_id = message.chat.id
            text = message.text
            
            # Safety check: truncate extremely long messages to prevent issues
            max_message_length = 100000  # 100k characters
            if len(text) > max_message_length:
                print(f"Warning: Truncating message from user {user_id} (length: {len(text)})")
                text = text[:max_message_length]
            
            date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Safe print that handles Unicode characters
            try:
                print(f"Logging message: {username} ({user_id}) in chat {chat_id} - {text}")
            except UnicodeEncodeError:
                # Fallback for Windows console encoding issues
                safe_text = text.encode('ascii', 'replace').decode('ascii')
                print(f"Logging message: {username} ({user_id}) in chat {chat_id} - {safe_text}")
            except Exception as e:
                # Silent fail for print errors
                print(f"Error printing message: {e}")

            # Save message to CSV using helper function
            if not save_message_to_db(user_id, username, chat_id, text, date):
                # Only log if it's a duplicate (save_message_to_csv returns False for duplicates)
                pass
    except Exception as e:
        # Silent fail for logging errors to prevent bot crashes
        print(f"Error in log_messages: {e}")

def cleanup_old_messages_csv(days_to_keep=7):
    """Removes messages older than the specified number of days from the CSV."""
    if not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE) == 0:
        return

    try:
        df = pd.read_csv(LOG_FILE)
        df["date"] = pd.to_datetime(df["date"], errors='coerce')
        
        # Define cutoff
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days_to_keep)
        
        # Keep only new messages
        initial_count = len(df)
        df = df[df["date"] >= cutoff_date]
        final_count = len(df)
        
        # Save back to CSV
        df.to_csv(LOG_FILE, index=False, encoding='utf-8')
        print(f"[Cleanup] Removed {initial_count - final_count} old messages.")
        
    except Exception as e:
        print(f"Cleanup error: {e}")
        
def cleanup_old_messages(days_to_keep=7):
    try:
        cutoff = (datetime.datetime.now() - datetime.timedelta(days=days_to_keep)).strftime("%Y-%m-%d %H:%M:%S")
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("DELETE FROM messages WHERE date < ?", (cutoff,))
        conn.commit()
        print(f"[Cleanup] Removed messages older than {cutoff}")
        conn.close()
    except Exception as e:
        print(f"Cleanup error: {e}")

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
    init_db()
    start_bot()
