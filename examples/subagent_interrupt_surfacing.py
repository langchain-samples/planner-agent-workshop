"""
Example: Surfacing Subagent Interrupts to Supervisor

This example demonstrates how to propagate interrupts from a subagent to a supervisor agent.
Based on the pattern of using custom middleware to detect and surface subagent interrupts.

Key Pattern:
1. Subagent returns {'__interrupt__': [...]} when it needs user input
2. Tool wrapper detects this and returns a special marker
3. Supervisor middleware detects the marker and calls interrupt()
4. User provides input via Command(resume=...)
5. Middleware resumes the subagent and returns the result
"""

import os
from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt
from dotenv import load_dotenv

load_dotenv()

# Mock Tavily client for demonstration
class MockTavilyClient:
    def search(self, query, max_results=5, include_raw_content=False):
        return {
            "results": [
                {"title": "Weather Tokyo", "url": "https://example.com/tokyo", "content": "Tokyo weather is sunny, 22°C"}
            ]
        }

tavily_client = MockTavilyClient()
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ========== Web Search Agent (Subagent) with Approval Middleware ==========

@tool
def web_search(
    query: str,
    max_results: int = 5,
    include_raw_content: bool = False,
):
    """
    A tool to search the web for current information.
    Args:
        query: The query to search for
        max_results: The maximum number of results to return.
        include_raw_content: Whether to include the raw content of the search results.
    """
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
    )


class WebSearchApprovalMiddleware(AgentMiddleware):
    """Pause the web search agent until a human approves the tool output."""

    def before_model(self, state):
        messages = state['messages']

        last_message = messages[-1]
        if not isinstance(last_message, ToolMessage) or last_message.name != "web_search":
            return None

        interrupt_payload = {
            "prompt": "Approve or edit the latest web lookup result.",
            "tool_call_id": last_message.tool_call_id,
            "tool_name": last_message.name,
            "tool_output": last_message.content,
        }

        decision = interrupt(interrupt_payload)

        if not isinstance(decision, dict):
            raise ValueError(
                "Resume payload must be a dict with 'status' and optional 'message'."
            )

        status = decision.get("status")
        if status not in {"approved", "edit"}:
            raise ValueError("Resume payload must include status 'approved' or 'edit'.")

        review_note = decision.get("message")
        if status == "edit" and not review_note:
            raise ValueError("Edits must include a 'message' detailing the edit to make.")
        if not review_note:
            review_note = "Web search lookup approved."

        human_msg = HumanMessage(
            content=f"[{"APPROVED" if status == "approved" else "EDITED"}] {review_note}",
            name="tool_reviewer",
            additional_kwargs={
                "approval_status": status,
                "tool_call_id": last_message.tool_call_id
            },
        )
        return {"messages": [human_msg]}


# Create the web search subagent with approval middleware
search_agent = create_agent(
    model,
    system_prompt='You are a web search expert. Use the web_search tool to search the web and answer the provided query.',
    tools=[web_search],
    middleware=[WebSearchApprovalMiddleware()],
    checkpointer=MemorySaver()
)


# ========== Supervisor Agent with Subagent Interrupt Surfacing ==========

# Store subagent context for middleware to access
_subagent_context = {"last_result": None, "search_config": None}

@tool
def call_search_agent(query: str) -> str:
    """Call the search agent to find information on the web.
    Note: This may interrupt if the search agent needs approval."""
    search_config = {"configurable": {"thread_id": "search-thread-1"}}
    _subagent_context["search_config"] = search_config

    result = search_agent.invoke({
        "messages": [{"role": "user", "content": query}]
    }, config=search_config)

    # Store the result for middleware to access
    _subagent_context["last_result"] = result

    # Check if search agent was interrupted
    if "__interrupt__" in result:
        # Return a special marker that the middleware will detect
        return "[SUBAGENT_INTERRUPT] Search agent is waiting for approval."

    return result["messages"][-1].content


class SubagentInterruptMiddleware(AgentMiddleware):
    """Middleware that surfaces subagent interrupts to the supervisor level.

    When a subagent (like search_agent) is interrupted and needs user input,
    this middleware detects it and propagates the interrupt to the supervisor agent.

    This is the KEY pattern for surfacing subagent interrupts!
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
                interrupt_data = subagent_result["__interrupt__"]

                # Surface the subagent's interrupt at the supervisor level
                # The payload includes information about which subagent is waiting
                interrupt_payload = {
                    "source": "search_agent",
                    "subagent_interrupt": interrupt_data,
                    "message": "Search agent is waiting for approval",
                    "config": _subagent_context.get("search_config"),
                }

                # Propagate interrupt to supervisor
                print(f"\n[SUPERVISOR] Detected subagent interrupt, surfacing to user...")
                print(f"[SUPERVISOR] Interrupt payload: {interrupt_data}\n")

                user_response = interrupt(interrupt_payload)

                # Resume the subagent with the user's response
                search_config = _subagent_context.get("search_config")
                if search_config:
                    print(f"[SUPERVISOR] Resuming subagent with user response: {user_response}\n")

                    resumed_result = search_agent.invoke(
                        Command(resume=user_response),
                        config=search_config
                    )

                    # Update the tool message with the actual result
                    return {
                        "messages": [
                            HumanMessage(
                                content=f"[RESUMED] Search agent completed: {resumed_result['messages'][-1].content}",
                                name="subagent_handler"
                            )
                        ]
                    }

        return None


# Create supervisor agent with interrupt surfacing middleware
supervisor_agent = create_agent(
    model,
    system_prompt='You are a supervisor that coordinates web searches. Use call_search_agent to search the web.',
    tools=[call_search_agent],
    middleware=[SubagentInterruptMiddleware()],
    checkpointer=MemorySaver()
)


def main():
    """Demonstrate surfacing subagent interrupts to supervisor."""
    print("=== Subagent Interrupt Surfacing Example ===\n")
    print("This demonstrates how a supervisor can handle interrupts from subagents.\n")

    config = {"configurable": {"thread_id": "supervisor-1"}}
    question = "What is the weather in Tokyo?"

    print(f"User: {question}\n")
    print("[SUPERVISOR] Invoking supervisor agent...\n")

    # First invocation - supervisor will call search_agent, which will interrupt for approval
    initial_result = supervisor_agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config=config,
    )

    # Check if supervisor was interrupted (due to subagent interrupt)
    if '__interrupt__' in initial_result:
        interrupt_result = initial_result['__interrupt__'][0]
        print(f'[SUPERVISOR] Interrupted! Payload: {interrupt_result.value}\n')

        # Simulate user approval
        review_message = {"status": "approved", "message": "Looks good!"}
        print(f"[USER] Approving search result: {review_message}\n")

        # Resume supervisor with approval
        resume_cmd = Command(resume=review_message)
        resume_result = supervisor_agent.invoke(resume_cmd, config=config)

        print("[SUPERVISOR] Final result:")
        resume_result['messages'][-1].pretty_print()
    else:
        print("[SUPERVISOR] No interrupt (unexpected)")
        print(f"Result: {initial_result['messages'][-1].content}")

    print("\n=== How This Works ===")
    print("1. Supervisor calls call_search_agent tool")
    print("2. call_search_agent invokes search_agent.invoke()")
    print("3. search_agent executes web_search and WebSearchApprovalMiddleware interrupts")
    print("4. call_search_agent detects __interrupt__ and returns '[SUBAGENT_INTERRUPT]' marker")
    print("5. SubagentInterruptMiddleware.after_tools() detects the marker")
    print("6. Middleware calls interrupt() to pause supervisor execution")
    print("7. User provides approval via Command(resume=...)")
    print("8. Middleware resumes search_agent with the approval")
    print("9. Supervisor continues with the subagent's completed result")


if __name__ == "__main__":
    main()
