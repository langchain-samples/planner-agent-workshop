# LangChain v1 Workshop - Planner Agent

A progressive workshop showcasing LangChain v1 capabilities, building from a simple agent to a sophisticated multi-agent system with memory, middleware, and MCP integration.

## Workshop Structure

This workshop consists of 9 progressive steps, each building upon the previous:

1. **agent_01_basic.py** - Basic calendar agent with tools
2. **agent_02_memory.py** - Adds conversational memory with MemorySaver
3. **agent_03_human_in_loop_interrupt.py** - Human-in-the-loop with interrupt pattern
4. **agent_04_middleware_tool_confirmation.py** - Prebuilt middleware for tool confirmation
5. **agent_05_structured_output_mcp.py** - Structured output with MCP integration
6. **agent_06_supervisor_multi_agent.py** - Supervisor pattern with sub-agents as tools
7. **agent_07_guardrail_middleware.py** - Custom middleware for security guardrails
8. **agent_08_image_handling.py** - Image analysis with custom middleware
9. **agent_09_long_term_memory.py** - Long-term memory with Store

## Setup

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

1. Install dependencies using uv:
```bash
uv sync
```

2. Create a `.env` file (copy from `.env.example`):
```bash
cp .env.example .env
```

3. Add your API keys to `.env`:
```
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

## Running the Agents

Each agent script can be run independently:

```bash
# Run with uv
uv run python agents/agent_01_basic.py
uv run python agents/agent_02_memory.py
# ... etc
```

Or activate the virtual environment and run directly:

```bash
source .venv/bin/activate  # On macOS/Linux
python agents/agent_01_basic.py
```

## Workshop Progression

### Step 1: Basic Agent
- Simple calendar agent with `read_calendar` and `write_calendar` tools
- Demonstrates the basic `create_agent` pattern
- No memory - each invocation is independent

### Step 2: Memory
- Adds `MemorySaver` for conversational memory
- Introduces `thread_id` for conversation continuity
- Agent can now remember previous interactions

### Step 3: Human-in-the-Loop (Interrupt)
- Adds `ask_for_help` tool using `interrupt`
- Agent can pause execution and wait for user input
- Useful for conflict resolution

### Step 4: Middleware for Tool Confirmation
- Adds `reschedule_calendar` tool
- Uses `HumanInTheLoopMiddleware` for tool confirmation
- Demonstrates prebuilt middleware

### Step 5: Structured Output with MCP
- Web search agent using Tavily MCP server
- Returns structured output (EventList with List[EventInfo])
- Demonstrates real MCP integration and `ToolStrategy`
- Requires TAVILY_API_KEY in .env

### Step 6: Supervisor Multi-Agent
- Supervisor agent that uses sub-agents as tools
- Calendar agent and web search agent as sub-agents
- Demonstrates multi-agent coordination
- **Advanced**: Shows how to surface subagent interrupts to supervisor level using custom middleware
- See [docs/subagent_interrupt_surfacing.md](docs/subagent_interrupt_surfacing.md) for detailed explanation

### Step 7: Guardrail Middleware
- Custom middleware for security
- Analyzes user queries before agent execution
- Blocks malicious or irrelevant requests

### Step 8: Image Handling
- Custom middleware for image analysis
- Extends state schema to accept images
- Demonstrates content blocks in LangChain v1

### Step 9: Long-Term Memory
- Adds `Store` for long-term memory
- User data persists across conversations/threads
- Distinguishes short-term (checkpointer) vs long-term (store) memory

## Deployment

The `langgraph.json` file is configured for deployment to LangSmith. The final agent (agent_09_long_term_memory.py) is set as the main graph.

To deploy:
1. Push to GitHub
2. Link repository in LangSmith
3. Select deployment type (dev/prod)
4. Agent will be deployed with Redis and Postgres instances

## Key Concepts Demonstrated

- **Agents**: Basic agent creation with `create_agent`
- **Tools**: Creating and using tools with `@tool` decorator
- **Memory**: Short-term (checkpointer) and long-term (store) memory
- **Human-in-the-Loop**: Interrupts and middleware for user interaction
- **Middleware**: Prebuilt and custom middleware for agent behavior modification
- **MCP**: Model Context Protocol integration for external tools
- **Structured Output**: Returning structured data instead of plain text
- **Multi-Agent**: Supervisor pattern with sub-agents as tools
- **Security**: Guardrails and request filtering
- **Content Blocks**: Image handling and analysis

## Advanced Patterns

### Surfacing Subagent Interrupts

When building supervisor-subagent architectures, you may want subagents to interrupt for user input (approval, clarification, etc.). The challenge is surfacing these interrupts to the supervisor level.

**Solution**: Use custom middleware that detects subagent interrupts and propagates them to the supervisor.

See:
- [docs/subagent_interrupt_surfacing.md](docs/subagent_interrupt_surfacing.md) - Detailed explanation
- [examples/subagent_interrupt_surfacing.py](examples/subagent_interrupt_surfacing.py) - Standalone example
- [agents/agent_06_supervisor_multi_agent.py](agents/agent_06_supervisor_multi_agent.py) - Implementation in workshop

## Notes

- All agents use mock calendar storage for demonstration
- Agent 05+ require TAVILY_API_KEY for MCP web search integration
- Image handling (agent 08) requires vision-capable models
- Long-term memory (agent 09) uses InMemoryStore (use DB-backed store in production)
