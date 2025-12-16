from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain.agents import create_agent
from dotenv import load_dotenv
import os
import asyncio

load_dotenv(override=True)

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

client = MultiServerMCPClient(
    {
        "tavily": {
            "transport": "http",
            "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={TAVILY_API_KEY}",
        }
    }
)

async def main():
    # Create a session explicitly
    async with client.session("tavily") as session:  
        # Pass the session to load tools, resources, or prompts
        tools = await load_mcp_tools(session)  
        agent = create_agent(
            "openai:gpt-4o-mini",
            tools
        )
        result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": "find ai events in nyc?"}]}
            )
        print(result)

if __name__ == "__main__":
    asyncio.run(main())