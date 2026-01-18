"""
Step 3: Human-in-the-Loop with Interrupt
Adds ask_for_help tool that uses interrupt to pause execution and wait for user input.
Demonstrates human-in-the-loop pattern for conflict resolution.
"""

from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from typing import List, Dict
from dotenv import load_dotenv
from agents.models import model

load_dotenv(override=True)

# Mock calendar storage
# Pre-populate with an existing event to demonstrate conflict handling
_calendar_events: List[Dict] = [
    {
        "title": "Meeting",
        "date": "2026-12-20",
        "time": "11:00",
        "location": "Office"
    }
]


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


@tool
def ask_for_help(message: str) -> str:
    """Ask the user for help when encountering a conflict or issue.
    This tool interrupts the agent execution and waits for user input.
    
    Args:
        message: The message/question to ask the user
    
    Returns:
        User's response (provided after interrupt)
    """
    # Interrupt execution and wait for user input
    user_input = interrupt(message)
    # This will resume when user provides input via Command(resume="...")
    return user_input


# System prompt
SYSTEM_PROMPT = """You are a helpful calendar assistant. You can:
- Read calendar events using read_calendar
- Create new events using write_calendar
- Ask for help using ask_for_help when you encounter conflicts

When scheduling events, always check the calendar first for conflicts.
If you find a conflict, use ask_for_help to ask the user what to do.
Be friendly and confirm when events are successfully created."""

# Create memory saver
checkpointer = MemorySaver()

# Create the agent with memory and interrupt capability
agent = create_agent(
    model=model,
    tools=[read_calendar, write_calendar, ask_for_help],
    system_prompt=SYSTEM_PROMPT,
    # checkpointer=checkpointer,
)

# if __name__ == "__main__":
#     # Example usage
#     print("=== Agent with Human-in-the-Loop (Interrupt) ===\n")
    
#     thread_id = "conversation-1"
#     config = {"configurable": {"thread_id": thread_id}}
    
#     # Note: Calendar already has an event on December 20th at 11 AM
#     print("Calendar already has an event: Meeting on December 20th at 11 AM\n")
    
#     # Try to schedule another event at the same time (will trigger conflict and interrupt)
#     print("User: Schedule a soccer game on December 20th at 11 AM in Paris")
#     print("(This will trigger a conflict and the agent will ask for help via interrupt)\n")
    
#     # First invocation - this will trigger the interrupt when ask_for_help is called
#     result = agent.invoke({
#         "messages": [{"role": "user", "content": "Schedule a soccer game on December 20th 2026 at 11 AM in Paris"}]
#     }, config=config)
    
#     # Check if the agent is interrupted (waiting for user input)
#     if "__interrupt__" in result:
#         print("Agent interrupted and waiting for user input...")
#         print(f"Interrupt message: {result.get('__interrupt__', 'N/A')}\n")
#     else:
#         # If no interrupt field, the interrupt() call paused execution
#         print("Agent execution paused (interrupted)...\n")
    
#     print("Resuming with user input: 'I have moved my existing event'\n")
    
#     # Second invocation - resume the agent with user input using Command
#     # The value passed to resume becomes the return value of interrupt()
#     result = agent.invoke(
#         Command(resume="I have moved my existing event"),
#         config=config
#     )
#     print(f"Agent: {result['messages'][-1].content}\n")
#     print("Note: The agent successfully resumed after the interrupt and completed the task!")

