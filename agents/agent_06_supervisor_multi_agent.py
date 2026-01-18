"""
Step 6: Supervisor Multi-Agent Architecture
Creates a supervisor agent that uses sub-agents (calendar and web search) as tools.
Demonstrates the supervisor pattern with sub-agents as tools.
"""

from langchain.agents import create_agent
from langchain.tools import tool
from langchain.agents.middleware.human_in_the_loop import HumanInTheLoopMiddleware
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from typing import List, Dict
from dataclasses import dataclass
from datetime import datetime, timedelta
from dotenv import load_dotenv
from agents.models import model

load_dotenv(override=True)

# ========== Calendar Agent (Sub-agent) ==========
# Pre-populate with an existing event to demonstrate conflict handling
# Note: The mock search returns events at 6PM and 7PM, so we'll create a conflict at 7PM
tomorrow_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

_calendar_events: List[Dict] = [
    {
        "title": "Team Meeting",
        "date": tomorrow_date,  # Tomorrow
        "time": "19:00",  # 7PM - conflicts with "Modern Dance Show"
        "location": "Office"
    }
]

@tool
def read_calendar(date: str = None) -> str:
    """Read calendar events. If date is provided, filter events for that date."""
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
def write_calendar(title: str, date: str, time: str, location: str = "") -> str:
    """Create a new calendar event."""
    for event in _calendar_events:
        if event["date"] == date and event["time"] == time:
            return f"Conflict: There's already an event '{event['title']}' scheduled for {date} at {time}"
    
    new_event = {"title": title, "date": date, "time": time, "location": location}
    _calendar_events.append(new_event)
    return f"Successfully created event '{title}' on {date} at {time} in {location}"


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


# Create calendar sub-agent with HITL middleware
# Uses the model from models.py
calendar_checkpointer = MemorySaver()

# Create middleware for tool confirmation (like agent_04)
calendar_hitl_middleware = HumanInTheLoopMiddleware(
    interrupt_on={
        "reschedule_calendar": True,  # Interrupt and allow all decisions (approve, edit, reject)
    }
)

calendar_agent = create_agent(
    model=model,
    tools=[read_calendar, write_calendar, ask_for_help, reschedule_calendar],
    system_prompt="""You are a calendar assistant. You can:
- Read calendar events using read_calendar
- Create new events using write_calendar
- Ask for help using ask_for_help when you encounter conflicts
- Reschedule existing events using reschedule_calendar

IMPORTANT: When asked to schedule an event, you MUST attempt to write_calendar.
If write_calendar returns a conflict message, then use ask_for_help to ask the user if they want to reschedule the conflicting event.
If the user agrees to reschedule, use reschedule_calendar to move the existing event, then create the new event.

Do NOT just check the calendar and report conflicts - actually TRY to schedule the event using write_calendar.
The write_calendar tool will tell you if there's a conflict.
Be friendly and confirm when events are successfully created or rescheduled.""",
    # checkpointer=calendar_checkpointer,
    middleware=[calendar_hitl_middleware],
)


# ========== Web Search Agent (Sub-agent) ==========
@tool
def tavily_search(query: str) -> str:
    """Search the web for information about events, concerts, or other topics."""
    # Mock implementation - use same tomorrow_date as calendar
    return f"""Search results for "{query}":

1. Swan Lake Ballet - {tomorrow_date} at 18:00 (6PM)
   Description: Classical ballet performance by Paris Ballet Company
   Location: Paris Arts Center
   URL: https://example.com/ballet-swan-lake

2. Modern Dance Show - {tomorrow_date} at 19:00 (7PM)
   Description: Contemporary dance performance
   Location: National Theater
   URL: https://example.com/modern-dance
"""


# Create web search sub-agent
# Uses the model from models.py
search_agent = create_agent(
    model=model,
    tools=[tavily_search],
    system_prompt="You are a web search assistant. Search for events and return relevant information.",
)


# ========== Supervisor Agent ==========
# The supervisor uses the sub-agents as tools!
# According to LangChain docs, agents must be wrapped in @tool functions
# Uses the model from models.py
checkpointer = MemorySaver()

# Store subagent results and interrupt info for the middleware to access
_subagent_context = {"last_result": None, "calendar_config": None}

@tool(
    "calendar_agent",
    description="Handles calendar operations (read, create events, reschedule). Use this when you need to check calendar availability or schedule events."
)
def call_calendar_agent(query: str) -> str:
    """Call the calendar agent to handle calendar operations.
    Note: This may interrupt if the calendar agent needs user input."""
    calendar_config = {"configurable": {"thread_id": "calendar-conversation-1"}}
    _subagent_context["calendar_config"] = calendar_config

    result = calendar_agent.invoke({
        "messages": [{"role": "user", "content": query}]
    }, config=calendar_config)

    # Store the result for middleware to access
    _subagent_context["last_result"] = result

    # Check if calendar agent was interrupted
    if "__interrupt__" in result:
        # Return a special marker that the middleware will detect
        return "[SUBAGENT_INTERRUPT] Calendar agent is waiting for user input."

    return result["messages"][-1].content


@tool(
    "search_agent",
    description="Searches the web for events, concerts, and other activities. Use this when you need to find information about events or activities."
)
def call_search_agent(query: str) -> str:
    """Call the search agent to find events and information."""
    result = search_agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    return result["messages"][-1].content


class SubagentInterruptMiddleware(AgentMiddleware):
    """Middleware that surfaces subagent interrupts to the supervisor level.

    When a subagent (like calendar_agent) is interrupted and needs user input,
    this middleware detects it and propagates the interrupt to the supervisor agent.
    Handles multiple sequential interrupts from the same subagent.
    """

    def after_tools(self, state):
        """Check after tool execution if a subagent was interrupted."""
        messages = state['messages']

        # Check if the last tool result indicates a subagent interrupt
        last_message = messages[-1]
        if hasattr(last_message, 'content') and '[SUBAGENT_INTERRUPT]' in str(last_message.content):
            # A subagent was interrupted - propagate to supervisor
            subagent_result = _subagent_context.get("last_result")

            if subagent_result and "__interrupt__" in subagent_result:
                calendar_config = _subagent_context.get("calendar_config")

                # Handle multiple interrupts in a loop
                while "__interrupt__" in subagent_result:
                    interrupt_data = subagent_result["__interrupt__"]

                    # Surface the subagent's interrupt at the supervisor level
                    interrupt_payload = {
                        "source": "calendar_agent",
                        "subagent_interrupt": interrupt_data,
                        "message": "Calendar agent is waiting for user input",
                        "config": calendar_config,
                    }

                    # Propagate interrupt to supervisor and wait for user response
                    user_response = interrupt(interrupt_payload)

                    # Resume the subagent with the user's response
                    if calendar_config:
                        subagent_result = calendar_agent.invoke(
                            Command(resume=user_response),
                            config=calendar_config
                        )
                    else:
                        break

                # All interrupts handled, return the final result
                return {
                    "messages": [
                        HumanMessage(
                            content=f"[RESUMED] Calendar agent completed: {subagent_result['messages'][-1].content}",
                            name="subagent_handler"
                        )
                    ]
                }

        return None


supervisor_agent = create_agent(
    model=model,
    tools=[call_calendar_agent, call_search_agent],  # Wrapped sub-agents as tools!
    system_prompt="""You are a supervisor assistant that coordinates between different sub-agents:
- calendar_agent: Handles calendar operations (read, create events)
- search_agent: Searches the web for events and concerts

When a user wants to find an event and schedule it:
1. First use search_agent to find relevant events
2. Then use calendar_agent to check availability and schedule if free

Coordinate between the agents to fulfill the user's request.""",
    # checkpointer=checkpointer,
    middleware=[SubagentInterruptMiddleware()],
)


# if __name__ == "__main__":
#     # Example usage
#     print("=== Supervisor Multi-Agent Architecture ===\n")

#     # Note: Calendar already has an event tomorrow at 7 PM
#     print(f"Calendar already has an event: Team Meeting on {tomorrow_date} at 7 PM\n")

#     supervisor_thread_id = "conversation-1"
#     supervisor_config = {"configurable": {"thread_id": supervisor_thread_id}}

#     calendar_thread_id = "calendar-conversation-1"  # Match the thread_id used in call_calendar_agent
#     calendar_config = {"configurable": {"thread_id": calendar_thread_id}}
    
#     print("User: Look for a ballet for tomorrow night and schedule it in my calendar")
#     print("(The supervisor will coordinate between search_agent and calendar_agent)\n")

#     # First invocation - supervisor coordinates, calendar agent may hit conflict
#     result = supervisor_agent.invoke({
#         "messages": [{"role": "user", "content": "Look for a ballet for tomorrow night and schedule it in my calendar"}]
#     }, config=supervisor_config)

#     # Check if supervisor was interrupted (due to subagent interrupt surfacing)
#     if "__interrupt__" in result:
#         print("🔔 SUPERVISOR INTERRUPTED! Subagent interrupt was surfaced.\n")
#         interrupt_info = result["__interrupt__"][0]
#         print(f"Interrupt source: {interrupt_info.value.get('source', 'unknown')}")
#         print(f"Message: {interrupt_info.value.get('message', 'N/A')}")
#         print(f"Subagent interrupt data: {interrupt_info.value.get('subagent_interrupt', 'N/A')}\n")

#         print("User responds: 'Yes, reschedule the Team Meeting to 8 PM'\n")

#         # Resume supervisor - middleware will handle resuming the subagent
#         result = supervisor_agent.invoke(
#             Command(resume="Yes, reschedule the Team Meeting to 8 PM"),
#             config=supervisor_config
#         )

#         # Check if interrupted again (for reschedule_calendar approval)
#         if "__interrupt__" in result:
#             print("🔔 SUPERVISOR INTERRUPTED AGAIN! (for reschedule_calendar approval)\n")
#             print("Interrupt details:", result["__interrupt__"][0].value, "\n")

#             print("User approves the reschedule operation.\n")

#             # Resume with approval
#             result = supervisor_agent.invoke(
#                 Command(resume={"decisions": [{"type": "approve"}]}),
#                 config=supervisor_config
#             )

#         print(f"✅ Supervisor Agent (final): {result['messages'][-1].content}\n")
#     else:
#         print(f"Supervisor Agent: {result['messages'][-1].content}\n")
#         print("(No interrupt occurred - agent may have found a free slot)\n")
    
#     print("\n=== Advanced: Interrupt Surfacing ===")
#     print("Note: The supervisor agent has SubagentInterruptMiddleware that:")
#     print("1. Detects when a subagent (calendar_agent) is interrupted")
#     print("2. Propagates the interrupt to the supervisor level")
#     print("3. Automatically resumes the subagent with the user's response")
#     print("4. Returns the final result to the supervisor")
#     print("\nThis allows the supervisor to handle subagent interrupts transparently!")
#     print("Sub-agents are used as tools by the supervisor.")
#     print("The calendar sub-agent has both ask_for_help and HumanInTheLoopMiddleware for conflict resolution.")

#     print("\n=== How Interrupt Surfacing Works ===")
#     print("When calendar_agent.invoke() returns {'__interrupt__': [...]}:")
#     print("1. call_calendar_agent tool returns '[SUBAGENT_INTERRUPT]' marker")
#     print("2. SubagentInterruptMiddleware.after_tools() detects this marker")
#     print("3. Middleware calls interrupt() to pause supervisor execution")
#     print("4. User provides input via Command(resume=...)")
#     print("5. Middleware resumes calendar_agent with the user input")
#     print("6. Supervisor continues with the subagent's completed result")

