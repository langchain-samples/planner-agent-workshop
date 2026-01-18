"""
Step 2: Agent with Memory
Adds MemorySaver to enable conversational memory across invocations.
Demonstrates thread_id for conversation continuity.
"""

from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from typing import List, Dict
from dotenv import load_dotenv
from agents.models import model

load_dotenv(override=True)

# Mock calendar storage
_calendar_events: List[Dict] = []


@tool
def read_calendar(date: str = None) -> str:
    """Read calendar events. If date is provided, filter events for that date.
    
    Args:
        date: Optional date string in format 'YYYY-MM-DD'. If None, returns all events.
    
    Returns:
        A string representation of calendar events.
    """
    if date:
        filtered = [e for e in _calendar_events if e.get("date") == date]
        if not filtered:
            return f"No events found for {date}"
        return "\n".join([f"- {e['title']} on {e['date']} at {e['time']} in {e.get('location', 'N/A')}" 
                         for e in filtered])
    
    if not _calendar_events:
        return "No events in calendar"
    
    return "\n".join([f"- {e['title']} on {e['date']} at {e['time']} in {e.get('location', 'N/A')}" 
                     for e in _calendar_events])


@tool
def write_calendar(title: str, date: str, time: str, location: str = "") -> str:
    """Create a new calendar event.
    
    Args:
        title: Event title
        date: Event date in format 'YYYY-MM-DD'
        time: Event time in format 'HH:MM'
        location: Optional location
    
    Returns:
        Confirmation message
    """
    # Check for conflicts
    for event in _calendar_events:
        if event["date"] == date and event["time"] == time:
            return f"Conflict: There's already an event '{event['title']}' scheduled for {date} at {time}"
    
    new_event = {
        "title": title,
        "date": date,
        "time": time,
        "location": location
    }
    _calendar_events.append(new_event)
    return f"Successfully created event '{title}' on {date} at {time} in {location}"


# System prompt
SYSTEM_PROMPT = """You are a helpful calendar assistant. You can:
- Read calendar events using read_calendar
- Create new events using write_calendar

When scheduling events, always check the calendar first for conflicts.
Be friendly and confirm when events are successfully created."""

# Create memory saver (checkpointer)
checkpointer = MemorySaver()

# Create the agent with memory
agent = create_agent(
    model=model,
    tools=[read_calendar, write_calendar],
    system_prompt=SYSTEM_PROMPT,
    # checkpointer=checkpointer,
)

# if __name__ == "__main__":
#     # Example usage with thread_id for conversation continuity
#     print("=== Agent with Memory ===\n")
    
#     thread_id = "conversation-1"
#     config = {"configurable": {"thread_id": thread_id}}
    
#     # Schedule an event
#     result = agent.invoke({
#         "messages": [{"role": "user", "content": "Schedule a soccer game for December 20th at 11 AM in Paris"}]
#     }, config=config)
#     print("User: Schedule a soccer game for December 20th at 11 AM in Paris")
#     print(f"Agent: {result['messages'][-1].content}\n")
    
#     # Ask what it just did (now it remembers!)
#     result2 = agent.invoke({
#         "messages": [{"role": "user", "content": "What did you just do?"}]
#     }, config=config)
#     print("User: What did you just do?")
#     print(f"Agent: {result2['messages'][-1].content}\n")
#     print("Note: The agent now remembers previous interactions thanks to MemorySaver!")

