"""
Step 9: Long-Term Memory with Store
Adds long-term memory capabilities using LangGraph Store for user preferences and data
that persists across conversations/threads.
Demonstrates short-term vs long-term memory distinction.
"""

from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from dataclasses import dataclass
from typing import Dict, Any
from typing_extensions import TypedDict
from dotenv import load_dotenv
from agents.models import model

load_dotenv(override=True)

# Mock calendar storage
_calendar_events: list = []


@tool
def read_calendar(date: str = None) -> str:
    """Read calendar events."""
    if not _calendar_events:
        return "No events in calendar"
    return "\n".join([f"- {e['title']} on {e['date']} at {e['time']}" for e in _calendar_events])


@tool
def write_calendar(title: str, date: str, time: str, location: str = "") -> str:
    """Create a new calendar event."""
    new_event = {"title": title, "date": date, "time": time, "location": location}
    _calendar_events.append(new_event)
    return f"Successfully created event '{title}' on {date} at {time}"


# Context schema for runtime
@dataclass
class Context:
    """Runtime context containing user_id."""
    user_id: str


# Long-term memory tools
@tool
def read_user_memory(runtime: ToolRuntime[Context]) -> str:
    """Read user information and preferences from long-term memory.
    This data persists across all conversations/threads for this user.
    
    Returns:
        User information and preferences as a string
    """
    store = runtime.store
    user_id = runtime.context.user_id
    
    # Retrieve user data from store
    user_data = store.get(("users",), user_id)
    
    if not user_data or not user_data.value:
        return f"No user information found for user {user_id}"
    
    return str(user_data.value)


@tool
def update_user_memory(runtime: ToolRuntime[Context], key: str, value: Any) -> str:
    """Update user information in long-term memory.
    This data will be available in all future conversations for this user.
    
    Args:
        key: The key to update (e.g., 'name', 'preferences', 'timezone', 'notes')
        value: The value to set
    
    Returns:
        Confirmation message
    """
    store = runtime.store
    user_id = runtime.context.user_id
    
    # Get existing user data
    user_data = store.get(("users",), user_id)
    current_info = user_data.value if user_data and user_data.value else {}
    
    # Update the specific key
    current_info[key] = value
    
    # Save back to store
    store.put(("users",), user_id, current_info)
    
    return f"Successfully updated {key} for user {user_id}"


# Create stores
# Uses the model from models.py
checkpointer = MemorySaver()  # Short-term memory (conversation history)
store = InMemoryStore()  # Long-term memory (user data across conversations)

# System prompt
SYSTEM_PROMPT = """You are a helpful calendar assistant with access to both short-term and long-term memory.

You have access to:
- read_calendar: Read calendar events
- write_calendar: Create calendar events
- read_user_memory: Read user information and preferences from long-term memory
- update_user_memory: Update user information in long-term memory

When a user first interacts with you:
1. Use read_user_memory to check if you have any information about the user
2. Use this information to personalize your responses
3. If you learn something new about the user (preferences, timezone, etc.), use update_user_memory to save it

This allows you to provide a personalized experience across all conversations."""

# Create the agent with both checkpointer (short-term) and store (long-term)
agent = create_agent(
    model=model,
    tools=[read_calendar, write_calendar, read_user_memory, update_user_memory],
    system_prompt=SYSTEM_PROMPT,
    # checkpointer=checkpointer,  # Short-term memory
    # store=store,  # Long-term memory
    context_schema=Context,
)

# if __name__ == "__main__":
#     # Example usage
#     print("=== Agent with Long-Term Memory ===\n")
    
#     # Initialize some user data
#     store.put(("users",), "user_123", {
#         "name": "Marco Perini",
#         "preferences": {"time_format": "12-hour", "reminder_style": "friendly"},
#         "timezone": "Europe/Paris",
#         "notes": "Prefers morning meetings"
#     })
    
#     # First conversation thread
#     thread_id_1 = "thread-1"
#     config_1 = {
#         "configurable": {
#             "thread_id": thread_id_1,
#         }
#     }
#     context_1 = Context(user_id="user_123")
    
#     result1 = agent.invoke({
#         "messages": [{"role": "user", "content": "What do you know about me? I'm 27 years old and italian, please update your memory to always reply in italian when talking to me."}]
#     }, config=config_1, context=context_1)
    
#     print("Thread 1 - User: What do you know about me?")
#     print(f"Agent: {result1['messages'][-1].content}\n")
    
#     # Second conversation thread (different thread, same user)
#     thread_id_2 = "thread-2"
#     config_2 = {
#         "configurable": {
#             "thread_id": thread_id_2,
#         }
#     }
#     context_2 = Context(user_id="user_123")
    
#     result2 = agent.invoke({
#         "messages": [{"role": "user", "content": "Schedule a meeting tomorrow at 10 AM."}]
#     }, config=config_2, context=context_2)
    
#     print("Thread 2 - User: Schedule a meeting tomorrow at 10 AM")
#     print(f"Agent: {result2['messages'][-1].content}\n")
    
#     print("Note:")
#     print("- Short-term memory (checkpointer): Remembers conversation within a thread")
#     print("- Long-term memory (store): Remembers user data across all threads")
#     print("- The agent can personalize responses based on stored user preferences")

