import os
import logging
import re
import time # Needed for polling delay
from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk.errors import SlackApiError
from openai import OpenAI, OpenAIError

# --- Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv() # Load environment variables from .env file

# --- Constants ---
POLLING_INTERVAL_S = 2 # How often to check Run status (seconds)
RUN_TIMEOUT_S = 120 # Max time to wait for a Run to complete

# --- Initialize Slack App ---
app = App(token=os.environ.get("SLACK_BOT_TOKEN"))

# --- Initialize OpenAI Client ---
try:
    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    if not os.environ.get("OPENAI_API_KEY"):
        logging.error("OPENAI_API_KEY not found in environment variables.")
        openai_client = None
    OPENAI_ASSISTANT_ID = os.environ.get("OPENAI_ASSISTANT_ID")
    if not OPENAI_ASSISTANT_ID:
         logging.error("OPENAI_ASSISTANT_ID not found in environment variables.")
         # Assistant ID is crucial for this version
         openai_client = None # Effectively disable if no assistant ID
except Exception as e:
    logging.error(f"Error initializing OpenAI client or getting Assistant ID: {e}")
    openai_client = None
    OPENAI_ASSISTANT_ID = None

# --- State Management (In-Memory - NOT PRODUCTION READY) ---
# Stores mapping: Slack thread_ts -> OpenAI thread_id
# WARNING: This dictionary is lost if the bot restarts.
# For production, use a database (Redis, SQLite, PostgreSQL, etc.)
slack_thread_to_openai_thread = {}

# --- Helper Functions ---
def get_bot_user_id(client):
    """Fetches the bot's user ID."""
    try:
        response = client.auth_test()
        return response["user_id"]
    except SlackApiError as e:
        logging.error(f"Error fetching bot user ID: {e}")
        return None

BOT_USER_ID = get_bot_user_id(app.client)

def clean_mention(text, bot_user_id):
    """Removes the bot mention from the start of a message."""
    if bot_user_id:
        mention = f"<@{bot_user_id}>"
        pattern = rf"^\s*<@{bot_user_id}>\s*"
        return re.sub(pattern, '', text).strip()
    return text.strip()

# --- Bot Logic (Assistants API) ---

@app.event("app_mention")
def handle_mention_assistant(event, say, logger):
    """Handles mentions using the OpenAI Assistants API."""
    if not openai_client or not OPENAI_ASSISTANT_ID:
        say("Sorry, the OpenAI connection or Assistant is not configured correctly. Please check server logs.", thread_ts=event.get("ts"))
        return

    if not BOT_USER_ID:
         say("Sorry, I couldn't identify my own user ID. Please check the logs.", thread_ts=event.get("ts"))
         return

    user_message_raw = event.get("text", "")
    channel_id = event.get("channel")
    user_id = event.get("user")
    slack_thread_ts = event.get("thread_ts", event.get("ts")) # Use thread_ts if available, else message ts
    message_ts = event.get("ts") # Timestamp of the *current* message

    prompt = clean_mention(user_message_raw, BOT_USER_ID)

    logger.info(f"Received mention from {user_id} in channel {channel_id} (Slack thread: {slack_thread_ts}): '{prompt}'")

    # --- Acknowledge receipt ---
    thinking_message_ts = None
    try:
        thinking_reply = say(text="🤔 Thinking (using Assistant)...", thread_ts=slack_thread_ts)
        thinking_message_ts = thinking_reply.get('ts') if thinking_reply and thinking_reply.get('ok') else None
    except Exception as e:
        logger.error(f"Error posting thinking message: {e}")


    openai_thread_id = None
    run = None # Initialize run variable

    try:
        # --- 1. Find or Create OpenAI Thread ---
        if slack_thread_ts in slack_thread_to_openai_thread:
            openai_thread_id = slack_thread_to_openai_thread[slack_thread_ts]
            logger.info(f"Found existing OpenAI thread ID: {openai_thread_id} for Slack thread: {slack_thread_ts}")
        else:
            logger.info(f"Creating new OpenAI thread for Slack thread: {slack_thread_ts}")
            thread = openai_client.beta.threads.create()
            openai_thread_id = thread.id
            slack_thread_to_openai_thread[slack_thread_ts] = openai_thread_id
            logger.info(f"Created OpenAI thread ID: {openai_thread_id} and mapped to Slack thread: {slack_thread_ts}")
            # WARNING: In a real app, save this mapping persistently here!

        # --- 2. Add User Message to Thread ---
        logger.info(f"Adding message to OpenAI thread {openai_thread_id}: '{prompt}'")
        # Store the user message ID to potentially fetch only *newer* messages later
        user_openai_message = openai_client.beta.threads.messages.create(
            thread_id=openai_thread_id,
            role="user",
            content=prompt,
        )

        # --- 3. Run the Assistant ---
        logger.info(f"Creating Assistant Run for thread {openai_thread_id} using Assistant {OPENAI_ASSISTANT_ID}")
        run = openai_client.beta.threads.runs.create(
            thread_id=openai_thread_id,
            assistant_id=OPENAI_ASSISTANT_ID,
            # Instructions override the default Assistant instructions (optional)
            # instructions="Focus on providing short answers."
        )
        logger.info(f"Run created with ID: {run.id}, Status: {run.status}")

        # --- 4. Poll for Run Completion ---
        start_time = time.time()
        while run.status in ["queued", "in_progress", "cancelling"]:
            if time.time() - start_time > RUN_TIMEOUT_S:
                logger.warning(f"Run {run.id} timed out after {RUN_TIMEOUT_S} seconds.")
                openai_client.beta.threads.runs.cancel(thread_id=openai_thread_id, run_id=run.id) # Attempt cancellation
                raise TimeoutError("Assistant run timed out.")

            time.sleep(POLLING_INTERVAL_S)
            run = openai_client.beta.threads.runs.retrieve(thread_id=openai_thread_id, run_id=run.id)
            logger.info(f"Checking Run {run.id} status: {run.status}")

        # --- 5. Process Final Run Status ---
        if run.status == "completed":
            logger.info(f"Run {run.id} completed.")
            # --- 6. Retrieve Assistant Messages ---
            # List messages *added by this run* (or just latest)
            # Fetch in descending order to get the newest first
            messages_response = openai_client.beta.threads.messages.list(
                thread_id=openai_thread_id,
                order="desc",
                # limit=1 # Usually the latest message is the response, but sometimes there can be multiple
                # 'after=user_openai_message.id' could also work but less reliable if assistant adds multiple msgs
            )
            assistant_messages = [m for m in messages_response.data if m.run_id == run.id and m.role == "assistant"]

            if not assistant_messages:
                 logger.warning(f"Run {run.id} completed but no new assistant messages found.")
                 ai_response = "I processed your request, but didn't generate a text response."
            else:
                # Combine multiple assistant messages if necessary (usually just one)
                ai_response = "\n".join(
                    content_block.text.value
                    for msg in reversed(assistant_messages) # Reverse to get chronological order if multiple
                    for content_block in msg.content
                    if content_block.type == 'text'
                ).strip()
                logger.info(f"Retrieved Assistant response(s) for run {run.id}.")

            # --- 7. Post Response to Slack ---
            if thinking_message_ts:
                app.client.chat_update(channel=channel_id, ts=thinking_message_ts, text=ai_response)
            else:
                say(text=ai_response, thread_ts=slack_thread_ts)

        # Handle other terminal states
        elif run.status == "requires_action":
            logger.warning(f"Run {run.id} requires action (e.g., function calling) - not handled in this basic bot.")
            # In a real bot, you'd handle function calls here
            error_message = "Sorry, my current task requires actions I can't perform yet."
            if thinking_message_ts: app.client.chat_update(channel=channel_id, ts=thinking_message_ts, text=error_message)
            else: say(text=error_message, thread_ts=slack_thread_ts)

        else: # failed, cancelled, expired
            logger.error(f"Assistant Run {run.id} ended with status: {run.status}")
            # Attempt to get error details
            error_details = "An error occurred."
            if run.last_error:
                error_details = f"Run failed: {run.last_error.code} - {run.last_error.message}"
                logger.error(f"Run {run.id} Last Error: {run.last_error.code} - {run.last_error.message}")
            if thinking_message_ts: app.client.chat_update(channel=channel_id, ts=thinking_message_ts, text=error_details)
            else: say(text=error_details, thread_ts=slack_thread_ts)


    except OpenAIError as e:
        logger.error(f"OpenAI API Error during Assistant operation: {e}")
        error_message = f"Sorry, I encountered an error with the OpenAI API: {e}"
        if thinking_message_ts: app.client.chat_update(channel=channel_id, ts=thinking_message_ts, text=error_message)
        else: say(text=error_message, thread_ts=slack_thread_ts)
    except TimeoutError as e:
        logger.error(f"TimeoutError: {e}")
        error_message = "Sorry, the request took too long to process."
        if thinking_message_ts: app.client.chat_update(channel=channel_id, ts=thinking_message_ts, text=error_message)
        else: say(text=error_message, thread_ts=slack_thread_ts)
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}") # Includes traceback
        error_message = "Sorry, an unexpected error occurred. See logs for details."
        if thinking_message_ts: app.client.chat_update(channel=channel_id, ts=thinking_message_ts, text=error_message)
        else: say(text=error_message, thread_ts=slack_thread_ts)


# --- Start the Bot ---
if __name__ == "__main__":
    required_vars = ["SLACK_BOT_TOKEN", "SLACK_APP_TOKEN", "OPENAI_API_KEY", "OPENAI_ASSISTANT_ID"]
    if not all(os.environ.get(var) for var in required_vars):
        logging.error(f"Missing required environment variables. Need: {', '.join(required_vars)}")
    elif not BOT_USER_ID:
         logging.error("Failed to get bot user ID. Bot cannot start.")
    else:
        logger.info(f"Bot User ID: {BOT_USER_ID}")
        logger.info(f"Using Assistant ID: {OPENAI_ASSISTANT_ID}")
        logger.info("Starting bot in Socket Mode...")
        handler = SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
        handler.start()