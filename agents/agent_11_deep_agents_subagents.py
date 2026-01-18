"""
Step 11: Deep Agents with Subagents
Demonstrates the built-in subagent capabilities of deep agents.
Shows how to define specialized subagents and use the task tool for delegation.
"""

from deepagents import create_deep_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from typing import List, Dict, Literal
from dotenv import load_dotenv
import os
from agents.models import model

load_dotenv(override=True)

# Mock calendar storage
_calendar_events: List[Dict] = []


@tool
def read_calendar(date: str = None) -> str:
    """Read calendar events."""
    if not _calendar_events:
        return "No events in calendar"
    if date:
        filtered = [e for e in _calendar_events if e.get("date") == date]
        if not filtered:
            return f"No events found for {date}"
        return "\n".join([f"- {e['title']} on {e['date']} at {e['time']}" for e in filtered])
    return "\n".join([f"- {e['title']} on {e['date']} at {e['time']}" for e in _calendar_events])


@tool
def write_calendar(title: str, date: str, time: str, location: str = "") -> str:
    """Create a new calendar event."""
    for event in _calendar_events:
        if event["date"] == date and event["time"] == time:
            return f"Conflict: There's already an event '{event['title']}' scheduled for {date} at {time}"

    new_event = {"title": title, "date": date, "time": time, "location": location}
    _calendar_events.append(new_event)
    return f"Successfully created event '{title}' on {date} at {time}"


# Mock web search tool
@tool
def web_search(query: str, max_results: int = 3) -> str:
    """Search the web for events and information."""
    # Mock search results
    return f"""Search results for "{query}":

1. Classical Concert at Paris Arts Center - Dec 20, 2025 at 7 PM
   Korean Symphony Orchestra performing Beethoven's 9th Symphony
   Tickets available at artscouncil.kr

2. Jazz Night at Blue Note - Dec 21, 2025 at 9 PM
   Live jazz featuring international artists
   Reservations: bluenote-Paris.com

3. K-Pop Festival - Dec 22, 2025 at 6 PM
   Outdoor festival with multiple performers
   Free entry, Han River Park"""


# Define specialized subagents
# Subagents are defined as dictionaries with name, description, system_prompt, and tools

research_subagent = {
    "name": "research-specialist",
    "description": "Conducts in-depth research on events, concerts, and activities using web search. Use when you need detailed information that requires multiple searches.",
    "system_prompt": """You are an expert researcher specialized in finding events and activities.

Your job is to:
1. Break down the research question into searchable queries
2. Use web_search to gather information
3. Save detailed findings to /research_notes.txt using write_file
4. Return a concise summary (2-3 paragraphs max)

IMPORTANT: Keep your final response under 300 words to maintain clean context.
Use the file system to store detailed research, then return only the summary.""",
    "tools": [web_search],
    "model": "gpt-4o-mini",  # Can specify different model per subagent
}

calendar_specialist_subagent = {
    "name": "calendar-specialist",
    "description": "Handles complex calendar operations including conflict resolution and multi-event scheduling. Use for scheduling multiple events or resolving conflicts.",
    "system_prompt": """You are a calendar specialist who handles complex scheduling tasks.

Your job is to:
1. Use write_todos to plan your approach
2. Check calendar for conflicts using read_calendar
3. Schedule events one by one using write_calendar
4. Handle conflicts by suggesting alternative times
5. Save scheduling notes to /scheduling_log.txt if needed

Return a summary of what was scheduled and any conflicts encountered.""",
    "tools": [read_calendar, write_calendar],
    "model": "gpt-4o-mini",
}

# Initialize checkpointer
# Uses the model from models.py
checkpointer = MemorySaver()

# System prompt for the main agent
SYSTEM_PROMPT = """You are a supervisor calendar assistant that coordinates between specialized subagents.

Available subagents (use the 'task' tool to delegate):
1. research-specialist: For finding events, concerts, and activities via web search
2. calendar-specialist: For complex calendar operations and multi-event scheduling
3. general-purpose: A general subagent with all your tools (automatically available)

When to delegate:
- Use research-specialist when you need to search for events or gather information
- Use calendar-specialist for scheduling multiple events or handling complex calendar tasks
- Use general-purpose for other multi-step tasks that would clutter your context

Workflow for "find and schedule" requests:
1. Delegate to research-specialist to find events
2. Review the research summary
3. Delegate to calendar-specialist to handle scheduling
4. Confirm with the user

This keeps your context clean while still going deep on subtasks."""

# Create the deep agent with subagents
agent = create_deep_agent(
    model=model,
    tools=[read_calendar, write_calendar, web_search],  # Main agent's tools
    system_prompt=SYSTEM_PROMPT,
    # checkpointer=checkpointer,
    subagents=[research_subagent, calendar_specialist_subagent],  # Define specialized subagents
)

if __name__ == "__main__":
    print("=== Deep Agents with Subagents ===\n")
    print("The main agent can delegate work to specialized subagents:")
    print("- research-specialist: Web search and event research")
    print("- calendar-specialist: Complex calendar operations")
    print("- general-purpose: General tasks (built-in)\n")

    thread_id = "subagent-demo-1"
    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    # Complex request that benefits from subagents
    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": "Find upcoming concerts in Paris this month and schedule the classical concert in my calendar"
        }]
    }, config=config)

    print("User: Find upcoming concerts in Paris this month and schedule the classical concert in my calendar\n")
    print(f"Agent: {result['messages'][-1].content}\n")

    # Check if files were created by subagents
    if "files" in result:
        print("📁 Files created by subagents:")
        for path in result["files"].keys():
            print(f"   {path}")
        print()

    print("=== How Subagents Work ===")
    print()
    print("The main agent uses the 'task' tool to delegate work:")
    print("  task(name='research-specialist', task='Find concerts in Paris')")
    print()
    print("Benefits of subagents:")
    print("1. 🧹 Context isolation - Subagent work doesn't clutter main agent's context")
    print("2. 🎯 Specialization - Each subagent has specific tools and instructions")
    print("3. 📦 Token efficiency - Main agent receives only final results, not intermediate steps")
    print("4. 🔄 Parallel potential - Multiple subagents can work concurrently (in async)")
    print()
    print("Subagent workflow:")
    print("1. Main agent identifies need for specialized work")
    print("2. Calls task(name='subagent-name', task='description')")
    print("3. Subagent executes with its own tools and context")
    print("4. Subagent returns concise result to main agent")
    print("5. Main agent continues with the summary")
    print()
    print("Compare this to agent_06_supervisor_multi_agent.py:")
    print("- agent_06: Manual supervisor with sub-agents wrapped as @tool functions")
    print("- agent_11: Built-in subagent system with task tool and clean delegation")
    print()
    print("The general-purpose subagent:")
    print("- Automatically available to all deep agents")
    print("- Has same tools and prompt as main agent")
    print("- Use for context isolation without specialization")
