"""
Step 4: Middleware for Tool Confirmation
Adds reschedule_calendar tool and HumanInTheLoopMiddleware to confirm tool execution.
Demonstrates prebuilt middleware for human-in-the-loop tool confirmation.
"""

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv(override=True)

# Mock calendar storage
# Pre-populate with an existing event to demonstrate conflict handling
_calendar_events: List[Dict] = [
    {
        "title": "Meeting",
        "date": "2024-12-20",
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


@tool
def reschedule_calendar(old_date: str, old_time: str, new_date: str, new_time: str) -> str:
    """Reschedule an existing calendar event to a new date and time.
    This is a powerful operation that moves events, so it requires confirmation.
    
    Args:
        old_date: Current event date in format 'YYYY-MM-DD'
        old_time: Current event time in format 'HH:MM'
        new_date: New event date in format 'YYYY-MM-DD'
        new_time: New event time in format 'HH:MM'
    
    Returns:
        Confirmation message
    """
    # Find and update the event
    for event in _calendar_events:
        if event["date"] == old_date and event["time"] == old_time:
            event["date"] = new_date
            event["time"] = new_time
            return f"Successfully rescheduled '{event['title']}' from {old_date} {old_time} to {new_date} {new_time}"
    
    return f"No event found for {old_date} at {old_time}"


# Initialize the model
model = init_chat_model("gpt-4o-mini", temperature=0)

# System prompt
SYSTEM_PROMPT = """You are a helpful calendar assistant. You can:
- Read calendar events using read_calendar
- Create new events using write_calendar
- Ask for help using ask_for_help when you encounter conflicts
- Reschedule existing events using reschedule_calendar

When scheduling events, always check the calendar first for conflicts.
If you find a conflict, use ask_for_help to ask the user if they want to reschedule the existing event.
If the user agrees to reschedule, use reschedule_calendar to move the existing event.
Be friendly and confirm when events are successfully created or rescheduled."""

# Create memory saver
checkpointer = MemorySaver()

# Create middleware for tool confirmation
# This will interrupt before executing reschedule_calendar to ask for confirmation
human_in_loop = HumanInTheLoopMiddleware(
    interrupt_on={
        "reschedule_calendar": True,  # Interrupt and allow all decisions (approve, edit, reject)
    }
)

# Create the agent with memory and middleware
agent = create_agent(
    model=model,
    tools=[read_calendar, write_calendar, ask_for_help, reschedule_calendar],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
    middleware=[human_in_loop],
)

if __name__ == "__main__":
    # Example usage
    print("=== Agent with Middleware for Tool Confirmation ===\n")
    
    thread_id = "conversation-1"
    config = {"configurable": {"thread_id": thread_id}}
    
    # Note: Calendar already has an event on December 20th at 11 AM
    print("Calendar already has an event: Meeting on December 20th at 11 AM\n")
    
    # Try to schedule another event at the same time
    print("User: Schedule a soccer game on December 20th at 11 AM in Seoul")
    print("(The agent will detect conflict, ask for help, then reschedule with middleware confirmation)\n")
    
    # First invocation - agent detects conflict and calls ask_for_help
    result = agent.invoke({
        "messages": [{"role": "user", "content": "Schedule a soccer game on December 20th at 11 AM in Seoul"}]
    }, config=config)
    
    # Check if the agent is interrupted by ask_for_help
    if "__interrupt__" in result:
        print("Agent interrupted by ask_for_help - waiting for user response...")
        print(f"Interrupt message: {result.get('__interrupt__', 'N/A')}\n")
    else:
        print("Agent execution paused (ask_for_help interrupt)...\n")
    
    print("User responds: 'Yes, reschedule it for the day after at the same time'\n")
    
    # Second invocation - resume with user's response to ask_for_help
    result = agent.invoke(
        Command(resume="Yes, reschedule it for the day after at the same time"),
        config=config
    )
    
    # Check if the agent is now interrupted by middleware before reschedule_calendar
    if "__interrupt__" in result:
        print("Agent interrupted by middleware - waiting for confirmation to execute reschedule_calendar...")
        print(f"Interrupt details: {result.get('__interrupt__', 'N/A')}\n")
    else:
        print("Agent execution paused by middleware (waiting for tool confirmation)...\n")
    
    print("Resuming with approval to execute reschedule_calendar...\n")
    
    # Third invocation - resume the agent with approval decision for reschedule_calendar
    # The middleware expects a decision format: {"decisions": [{"type": "approve"}]}
    result = agent.invoke(
        Command(resume={"decisions": [{"type": "approve"}]}),
        config=config
    )
    print(f"Agent: {result['messages'][-1].content}\n")
    print("Note: The agent used ask_for_help to ask about rescheduling, then")
    print("the HumanInTheLoopMiddleware interrupted before executing reschedule_calendar!")

