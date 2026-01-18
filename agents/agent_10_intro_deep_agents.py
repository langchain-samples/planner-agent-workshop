"""
Step 10: Introduction to Deep Agents
Demonstrates the transition from create_agent to create_deep_agent.
Deep agents come with built-in capabilities: planning (todos), file system tools, and subagents.
"""

from deepagents import create_deep_agent
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


# Create checkpointer for conversation memory
# Uses the model from models.py
checkpointer = MemorySaver()

# System prompt - deep agents benefit from detailed instructions
SYSTEM_PROMPT = """You are a helpful calendar assistant with advanced planning capabilities. You can:
- Read calendar events using read_calendar
- Create new events using write_calendar

Built-in Deep Agent Capabilities:
- write_todos: Break down complex tasks into steps (automatically available)
- File system tools: ls, read_file, write_file, edit_file (automatically available)
  * Use these to store notes, drafts, or large amounts of information
  * Files are stored in agent state and persist within the conversation thread
- Task delegation: Spawn subagents for complex subtasks (automatically available)

When handling multi-step requests:
1. Use write_todos to plan your approach
2. Check the calendar first for conflicts
3. Use file system tools to save notes or drafts if needed
4. Execute the plan step by step

Be friendly and confirm when events are successfully created."""

# Create a deep agent using create_deep_agent
# This automatically includes TodoListMiddleware, FilesystemMiddleware, and SubAgentMiddleware
agent = create_deep_agent(
    model=model,
    tools=[read_calendar, write_calendar],
    system_prompt=SYSTEM_PROMPT,
    # checkpointer=checkpointer,  # Add memory just like regular agents
)

if __name__ == "__main__":
    # Example usage
    print("=== Introduction to Deep Agents ===\n")
    print("Deep agents come with built-in capabilities:")
    print("- Planning with write_todos tool")
    print("- File system tools (ls, read_file, write_file, edit_file)")
    print("- Subagent spawning with task tool\n")

    thread_id = "deep-agent-thread-1"
    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    # Complex multi-step request
    result = agent.invoke({
        "messages": [{"role": "user", "content": "Schedule three team meetings for next week: Monday at 2 PM, Wednesday at 3 PM, and Friday at 4 PM. All in the main conference room."}]
    }, config=config)

    print("User: Schedule three team meetings for next week: Monday at 2 PM, Wednesday at 3 PM, and Friday at 4 PM")
    print(f"Agent: {result['messages'][-1].content}\n")

    # Check todos (if the agent used write_todos)
    if "todos" in result:
        print("Agent's Plan (from write_todos):")
        for todo in result["todos"]:
            status_emoji = "✅" if todo["status"] == "completed" else "⏳" if todo["status"] == "in_progress" else "📝"
            print(f"  {status_emoji} {todo['content']}")
        print()

    # Check files (if the agent used file system tools)
    if "files" in result:
        print("Files created by agent:")
        for path in result["files"].keys():
            print(f"  📄 {path}")
        print()

    # Ask about what's scheduled (memory test)
    result2 = agent.invoke({
        "messages": [{"role": "user", "content": "What meetings did we just schedule?"}]
    }, config=config)

    print("User: What meetings did we just schedule?")
    print(f"Agent: {result2['messages'][-1].content}\n")

    print("=== What's Different? ===")
    print("Deep agents (create_deep_agent) vs Regular agents (create_agent):")
    print()
    print("Deep Agents Include:")
    print("1. 📋 write_todos - For planning and tracking multi-step tasks")
    print("2. 📁 File system - ls, read_file, write_file, edit_file for context management")
    print("3. 🤖 Subagents - task tool to delegate work to specialized agents")
    print("4. 🧠 Context management - Automatic eviction of large tool results to files")
    print("5. 📝 Summary - Automatic conversation history compression for long chats")
    print()
    print("Regular Agents:")
    print("- Basic tool-calling loop")
    print("- No built-in planning or file system")
    print("- Manual middleware setup required for advanced features")
    print()
    print("Deep agents are ideal for complex, multi-step tasks that require planning,")
    print("context management, and delegation to subagents.")
