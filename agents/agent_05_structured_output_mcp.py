"""
Step 5: Structured Output Agent with MCP
Creates a web search agent using Tavily MCP server that returns structured output.
Demonstrates MCP integration and structured output with ToolStrategy.
"""

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.agents.structured_output import ToolStrategy
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import List
import os
import asyncio
from dotenv import load_dotenv

load_dotenv(override=True)

# Load Tavily API key from environment
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY must be set in .env file")

# Define the structured output schema
@dataclass
class EventInfo:
    """Information about an event or concert."""
    title: str
    short_description: str
    event_date: str  # Format: "12th May 2025 at 6PM"
    reference_url: str


@dataclass
class EventList:
    """List of events returned from search."""
    events: List[EventInfo]


# Initialize the model
model = init_chat_model("gpt-4o-mini", temperature=0)

# System prompt
SYSTEM_PROMPT = """You are a helpful event search assistant. You can search for events, concerts, 
and other activities using the tavily_search tool.

The tavily_search tool returns search results in JSON format. Each result contains:
- url: The URL of the page
- title: The title of the page/article
- content: The content/text from the page
- published_date: When the content was published (if available)

When a user asks about events, search for them and extract event information from the search results.
Return the results in the structured format:
- title: Event name (from the search result title or content)
- short_description: Brief description extracted from the content
- event_date: Date and time if mentioned in the content, format like "12th May 2025 at 6PM", or use published_date if no specific event date is found
- reference_url: URL from the search result

Return a list of events matching the user's query. Extract as much information as possible from the search results."""

# Configure ToolStrategy for structured output
# Wrap the list in a dataclass since ToolStrategy doesn't accept List[T] directly
tool_strategy = ToolStrategy(
    schema=EventList,  # Return an EventList containing a list of EventInfo objects
)

# Create MCP client for Tavily
client = MultiServerMCPClient(
    {
        "tavily": {
            "transport": "http",
            "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={TAVILY_API_KEY}",
        }
    }
)

# Create agent with async context manager for Tavily MCP server
@asynccontextmanager
async def create_agent_with_mcp():
    """Create agent with Tavily MCP server connection kept alive."""
    # Use Tavily MCP server with persistent session
    async with client.session("tavily") as session:
        # Get MCP tools
        mcp_tools = await load_mcp_tools(session)

        # Create the agent with structured output
        agent = create_agent(
            model=model,
            tools=mcp_tools,
            system_prompt=SYSTEM_PROMPT,
            response_format=tool_strategy,
        )

        yield agent

async def run_agent_async():
    """Run the agent asynchronously (required for MCP tools)."""
    async with create_agent_with_mcp() as agent:
        result = await agent.ainvoke({
            "messages": [{"role": "user", "content": "Look for concerts in Seoul tomorrow night, dec 16th 2025"}]
        })
        return result

if __name__ == "__main__":
    # Example usage
    print("=== Structured Output Agent with MCP ===\n")

    # Run agent asynchronously (required for MCP tools)
    result = asyncio.run(run_agent_async())

    print("User: Look for concerts in Seoul tomorrow night")
    print(f"Agent Response Type: {type(result['messages'][-1].content)}")
    print(f"Agent: {result['messages'][-1].content}\n")

    # Check for structured response
    if "structured_response" in result:
        print(f"Structured Response: {result['structured_response']}\n")

    print("Note: The agent returns structured output (EventList) instead of plain text.")
    print("Using Tavily MCP server for web search with persistent session.")

