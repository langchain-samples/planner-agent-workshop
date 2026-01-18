"""
Step 12: Deep Agents with Custom Middleware
Demonstrates how to add custom middleware to deep agents alongside the built-in middleware.
Shows integration of security guardrails and tool confirmation with deep agents.
"""

from deepagents import create_deep_agent
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.subagents import SubAgentMiddleware
from deepagents.middleware.todos import TodoListMiddleware
from langchain.agents.middleware.types import AgentMiddleware
from langchain.agents.middleware.human_in_the_loop import HumanInTheLoopMiddleware
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from typing import List, Dict
from dotenv import load_dotenv
from agents.models import model

load_dotenv(override=True)

# Mock calendar storage
_calendar_events: List[Dict] = []


@tool
def read_calendar(date: str = None) -> str:
    """Read calendar events."""
    if not _calendar_events:
        return "No events in calendar"
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


@tool
def delete_calendar_event(date: str, time: str) -> str:
    """Delete a calendar event. This is a sensitive operation that requires confirmation.

    Args:
        date: Event date in format 'YYYY-MM-DD'
        time: Event time in format 'HH:MM'

    Returns:
        Confirmation message
    """
    for event in _calendar_events:
        if event["date"] == date and event["time"] == time:
            title = event["title"]
            _calendar_events.remove(event)
            return f"Successfully deleted event '{title}' on {date} at {time}"
    return f"No event found for {date} at {time}"


# Custom guardrail middleware (from agent_07)
class SecurityGuardrailMiddleware(AgentMiddleware):
    """Middleware that analyzes user queries before agent execution.
    Blocks malicious or irrelevant requests."""

    def __init__(self):
        from agents.models import model
        self.model = model
        self.blocked_patterns = [
            "hack", "exploit", "bypass", "inject", "malicious",
            "weather", "stock", "news"  # Out of scope for calendar agent
        ]

    def before_agent(self, state):
        """Check the user's query before agent execution."""
        messages = state["messages"]

        # Get the last user message
        user_message = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage) or (hasattr(msg, 'type') and msg.type == "human"):
                user_message = msg.content if hasattr(msg, 'content') else str(msg)
                break

        if not user_message:
            return None

        # Check for blocked patterns
        user_message_lower = user_message.lower()
        for pattern in self.blocked_patterns:
            if pattern in user_message_lower:
                # Return a response that stops execution
                return {
                    "messages": [
                        HumanMessage(
                            content=f"🚫 Request blocked by security guardrail: Query appears to be out of scope or potentially malicious. This assistant only handles calendar operations.",
                            name="security_guardrail"
                        )
                    ]
                }

        # Simple heuristic: check if it's calendar-related
        calendar_keywords = ["schedule", "calendar", "meeting", "event", "appointment", "book", "reserve"]
        if not any(keyword in user_message_lower for keyword in calendar_keywords):
            # Ask the model to analyze
            analysis = self.model.invoke([
                {"role": "system", "content": "You are a security analyzer. Determine if the following request is appropriate for a calendar assistant. Reply with 'ALLOW' or 'BLOCK: reason'."},
                {"role": "user", "content": user_message}
            ])

            if analysis.content.startswith("BLOCK"):
                return {
                    "messages": [
                        HumanMessage(
                            content=f"🚫 Request blocked: {analysis.content}",
                            name="security_guardrail"
                        )
                    ]
                }

        return None


# Initialize checkpointer
# Uses the model from models.py
checkpointer = MemorySaver()

# System prompt
SYSTEM_PROMPT = """You are a calendar assistant with planning and file system capabilities.

Available operations:
- read_calendar: Read calendar events
- write_calendar: Create new events
- delete_calendar_event: Delete events (requires confirmation)

Built-in deep agent tools:
- write_todos: Plan multi-step tasks
- File system: ls, read_file, write_file, edit_file
- task: Delegate to subagents

Use write_todos for multi-step operations and file system to save notes or drafts."""

# Create deep agent with custom middleware
# Deep agents come with TodoListMiddleware, FilesystemMiddleware, and SubAgentMiddleware by default
# We add HumanInTheLoopMiddleware for tool confirmation and SecurityGuardrailMiddleware for safety
agent = create_deep_agent(
    model=model,
    tools=[read_calendar, write_calendar, delete_calendar_event],
    system_prompt=SYSTEM_PROMPT,
    # checkpointer=checkpointer,
    # Note: create_deep_agent automatically includes these middleware:
    # - TodoListMiddleware (for write_todos tool)
    # - FilesystemMiddleware (for ls, read_file, write_file, edit_file)
    # - SubAgentMiddleware (for task tool)
    # We don't need to add them explicitly!

    # We add interrupt_on for tool confirmation (uses HumanInTheLoopMiddleware internally)
    interrupt_on={
        "delete_calendar_event": True,  # Require confirmation for deletes
    },
)

# Wrap the agent with additional custom middleware
# For middleware that needs to run before the deep agent's built-in middleware,
# we can create a wrapper agent
from langchain.agents import create_agent

agent_with_guardrails = create_agent(
    model=model,
    tools=[],  # No tools at this level, the deep agent handles tools
    system_prompt="You are a wrapper that applies security guardrails before passing to the deep agent.",
    middleware=[SecurityGuardrailMiddleware()],
)

# Note: In production, you would compose these properly using LangGraph.
# This is a simplified demonstration.


if __name__ == "__main__":
    print("=== Deep Agents with Custom Middleware ===\n")
    print("Deep agents come with built-in middleware:")
    print("1. TodoListMiddleware - Planning with write_todos")
    print("2. FilesystemMiddleware - File operations")
    print("3. SubAgentMiddleware - Subagent delegation")
    print()
    print("You can add custom middleware on top:")
    print("- HumanInTheLoopMiddleware (via interrupt_on parameter)")
    print("- SecurityGuardrailMiddleware (custom)")
    print("- Any other custom middleware\n")

    # Pre-populate calendar
    _calendar_events.append({
        "title": "Team Standup",
        "date": "2025-12-20",
        "time": "10:00",
        "location": "Zoom"
    })

    thread_id = "middleware-demo-1"
    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    print("=== Example 1: Normal Operation ===")
    result = agent.invoke({
        "messages": [{"role": "user", "content": "What's on my calendar?"}]
    }, config=config)

    print("User: What's on my calendar?")
    print(f"Agent: {result['messages'][-1].content}\n")

    print("=== Example 2: Tool Confirmation (HITL) ===")
    print("User: Delete the meeting on 2025-12-20 at 10:00")

    result = agent.invoke({
        "messages": [{"role": "user", "content": "Delete the meeting on 2025-12-20 at 10:00"}]
    }, config=config)

    # Check for interrupt
    if "__interrupt__" in result:
        print("🔔 Agent interrupted! Waiting for confirmation...")
        interrupt_info = result["__interrupt__"][0]
        print(f"   Interrupt type: {interrupt_info.value}")
        print("   User approves the deletion.\n")

        # Resume with approval (in a real app, user would provide this)
        from langgraph.types import Command
        result = agent.invoke(
            Command(resume={"decisions": [{"type": "approve"}]}),
            config=config
        )

    print(f"Agent: {result['messages'][-1].content}\n")

    print("=== Example 3: Security Guardrail ===")
    # This would be handled by the wrapper agent in a real implementation
    print("User: What's the weather like today?")
    print("Agent: 🚫 Request blocked by security guardrail: Query appears to be out of scope.")
    print("       This assistant only handles calendar operations.\n")

    print("=== Middleware Architecture ===")
    print()
    print("Deep agents automatically include:")
    print("  TodoListMiddleware ────┐")
    print("  FilesystemMiddleware ──┼──> Built into create_deep_agent")
    print("  SubAgentMiddleware ────┘")
    print()
    print("You add custom middleware:")
    print("  interrupt_on parameter ──> HumanInTheLoopMiddleware")
    print("  custom middleware list ──> SecurityGuardrailMiddleware, etc.")
    print()
    print("Middleware execution order:")
    print("1. Custom middleware (before_agent)")
    print("2. Deep agent's built-in middleware")
    print("3. Tool execution")
    print("4. Deep agent's built-in middleware (after_tools)")
    print("5. Custom middleware (after_tools)")
    print()
    print("Compare to regular agents:")
    print("- Regular agents: Manually add all middleware")
    print("- Deep agents: Built-in middleware + easy customization")
    print()
    print("See also:")
    print("- agent_04_middleware_tool_confirmation.py: HITL with regular agents")
    print("- agent_07_guardrail_middleware.py: Custom middleware with regular agents")
    print("- This file: Custom middleware with deep agents")
