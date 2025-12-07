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
    
    # Use gemini-2.0-flash-exp directly
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    print(f"[OK] Using Google Gemini API model: gemini-2.0-flash-exp")
        
except ImportError:
    raise ImportError("google-generativeai package is required. Install it with: pip install google-generativeai")
except Exception as e:
    raise ValueError(f"Could not initialize Gemini model: {e}")


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
        except pd.errors.ParserError as e:
            safe_reply_to(message, "❌ Error reading message log file. The file may be corrupted.")
            print(f"CSV parsing error: {e}")
            return
        except Exception as e:
            safe_reply_to(message, f"❌ Error reading message log: {str(e)}")
            print(f"Error reading CSV: {e}")
            return
        
        # Check if DataFrame is empty
        if df.empty:
            safe_reply_to(message, "No messages found. The message log is empty. Start chatting in the group to log messages.")
            return
        
        # Validate required columns exist
        required_columns = ["message", "date"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            safe_reply_to(message, f"❌ Message log is missing required columns: {', '.join(missing_columns)}")
            return
        
        # Parse dates with error handling
        try:
            df["date"] = pd.to_datetime(df["date"], errors='coerce')
            # Remove rows with invalid dates
            invalid_dates = df["date"].isna()
            if invalid_dates.any():
                print(f"Warning: Found {invalid_dates.sum()} messages with invalid dates, skipping them")
                df = df[~invalid_dates]
            
            if df.empty:
                safe_reply_to(message, "No valid messages found in the log.")
                return
        except Exception as e:
            safe_reply_to(message, f"❌ Error parsing dates in message log: {str(e)}")
            print(f"Date parsing error: {e}")
            return

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
        # Filter out None values and ensure all are strings
        message_list = group_messages["message"].tolist()
        message_list = [str(msg) if msg is not None else "" for msg in message_list]
        message_list = [msg for msg in message_list if msg.strip()]  # Remove empty messages
        
        # Safety check: limit number of messages to prevent memory issues
        max_messages = 10000
        if len(message_list) > max_messages:
            print(f"Warning: Limiting to {max_messages} messages for summarization")
            message_list = message_list[-max_messages:]  # Take the most recent messages
        
        messages_text = " ".join(message_list)

        # Check if messages_text is empty
        if not messages_text or not messages_text.strip():
            safe_reply_to(message, "No valid messages found.")
            return

        # Summarize messages
        try:
            # Check if messages_text is too large before summarizing
            if len(messages_text) > 1000000:  # 1MB limit
                safe_reply_to(message, "❌ Too many messages to summarize. Please use a smaller time range or message count.")
                return
            
            summary = summarize_text(messages_text)
            safe_reply_to(message, f"📊 *Summary for {option_display}:*\n\n_{summary}_")
            
            cleanup_old_messages(days_to_keep=7)
        except ValueError as e:
            # Handle API errors specifically
            error_msg = str(e)
            if "Empty" in error_msg or "invalid response" in error_msg.lower():
                safe_reply_to(message, "❌ Error: Could not generate summary. The API returned an empty response.")
            else:
                safe_reply_to(message, f"❌ Error generating summary: {error_msg}")
            print(f"Summarization error: {e}")
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
            if not save_message_to_csv(user_id, username, chat_id, text, date):
                # Only log if it's a duplicate (save_message_to_csv returns False for duplicates)
                pass
    except Exception as e:
        # Silent fail for logging errors to prevent bot crashes
        print(f"Error in log_messages: {e}")

def cleanup_old_messages(days_to_keep=7):
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
