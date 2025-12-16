# Deep Agents Workshop Progression

This document explains the progression from LangChain v1 core agents to Deep Agents in the workshop.

## Part 1: LangChain v1 Core (Steps 1-9)

The first part of the workshop demonstrates building agents from scratch using LangChain v1's `create_agent` function. Each step adds a new capability:

1. **Basic Agent** - Tool calling with calendar tools
2. **Memory** - Conversation history with checkpointer
3. **Human-in-the-Loop** - Interrupts for user input
4. **Middleware** - Tool confirmation with prebuilt middleware
5. **MCP Integration** - External tools via Model Context Protocol
6. **Multi-Agent** - Manual supervisor pattern with sub-agents as tools
7. **Guardrails** - Custom security middleware
8. **Image Handling** - Extended state schema for images
9. **Long-Term Memory** - Persistent storage with Store

**Key Pattern**: Building up complexity incrementally, manually adding each feature.

## Part 2: Deep Agents (Steps 10-12)

The second part introduces the `deepagents` package, which provides an "agent harness" with built-in capabilities for complex tasks.

### Step 10: Introduction to Deep Agents

**What changes:**
- Replace `create_agent` with `create_deep_agent`
- Get built-in tools automatically:
  - `write_todos` - Planning and task tracking
  - `ls`, `read_file`, `write_file`, `edit_file` - File system
  - `task` - Subagent delegation

**Why it matters:**
- Deep agents are designed for complex, multi-step tasks
- Built-in context management prevents context window overflow
- Automatic planning helps agents organize work
- File system allows offloading large tool results

**Code comparison:**
```python
# Regular agent (Steps 1-9)
agent = create_agent(
    model=model,
    tools=[read_calendar, write_calendar],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
)

# Deep agent (Step 10)
agent = create_deep_agent(
    model=model,
    tools=[read_calendar, write_calendar],  # Your tools
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
    # Automatically includes: write_todos, file system tools, task tool
)
```

### Step 11: Deep Agents with Subagents

**What's demonstrated:**
- Built-in subagent system vs manual supervisor (compare with Step 6)
- Define specialized subagents with custom tools and prompts
- Use `task` tool for delegation
- Context isolation - subagent work doesn't clutter main agent

**Configuration:**
```python
research_subagent = {
    "name": "research-specialist",
    "description": "When to use this subagent",
    "system_prompt": "How this subagent should behave",
    "tools": [web_search],
    "model": "gpt-4o-mini",
}

agent = create_deep_agent(
    model=model,
    tools=[...],
    system_prompt=SYSTEM_PROMPT,
    subagents=[research_subagent],
)
```

**Comparison with Step 6:**
- Step 6: Manual supervisor, sub-agents wrapped as `@tool` functions, custom middleware for interrupt surfacing
- Step 11: Built-in subagent system, automatic context isolation, clean delegation with `task` tool

### Step 12: Deep Agents with Custom Middleware

**What's demonstrated:**
- Deep agents come with built-in middleware:
  - `TodoListMiddleware` - Planning
  - `FilesystemMiddleware` - File operations
  - `SubAgentMiddleware` - Subagent delegation
- Add custom middleware on top:
  - Use `interrupt_on` parameter for tool confirmation
  - Add custom middleware like `SecurityGuardrailMiddleware`

**Middleware architecture:**
```
Custom Middleware (before_agent)
  ↓
Built-in TodoListMiddleware
  ↓
Built-in FilesystemMiddleware
  ↓
Built-in SubAgentMiddleware
  ↓
Tool Execution
  ↓
Built-in Middleware (after_tools)
  ↓
Custom Middleware (after_tools)
```

## Key Differences: Regular Agents vs Deep Agents

### Regular Agents (`create_agent`)
- **Use for:** Simple, single-purpose agents
- **Approach:** Build up capabilities manually
- **Middleware:** Add all middleware explicitly
- **Subagents:** Manual supervisor pattern with `@tool` wrappers
- **Context:** Manual management, risk of overflow
- **Planning:** Add custom planning tools if needed
- **File System:** Build your own or use external storage

### Deep Agents (`create_deep_agent`)
- **Use for:** Complex, multi-step tasks
- **Approach:** Built-in capabilities out of the box
- **Middleware:** TodoList, Filesystem, Subagent included automatically
- **Subagents:** Built-in with `task` tool and clean delegation
- **Context:** Automatic eviction, summarization, management
- **Planning:** `write_todos` tool included
- **File System:** Built-in with pluggable backends

## When to Use Each

### Use Regular Agents When:
1. Building simple, focused agents
2. You need fine-grained control over execution
3. Lightweight applications with minimal overhead
4. Direct tool-calling patterns are sufficient
5. Learning the fundamentals of agent architecture

### Use Deep Agents When:
1. Tasks require multiple steps and planning
2. Large context management is needed
3. Tasks benefit from subagent delegation
4. Long-running research or analysis
5. Building production agents inspired by Claude Code, Deep Research

## Workshop Learning Path

The workshop is designed to:

1. **Steps 1-9**: Learn the fundamentals
   - Understand how agents work under the hood
   - Master middleware, memory, and multi-agent patterns
   - Build capabilities incrementally

2. **Steps 10-12**: Level up with Deep Agents
   - See how built-in capabilities simplify complex tasks
   - Compare manual patterns (Steps 1-9) with built-in ones (Steps 10-12)
   - Understand when to use each approach

3. **Key Insight**: Deep agents build ON TOP of LangGraph
   - Everything in Steps 1-9 is still relevant
   - Deep agents are an opinionated layer for complex tasks
   - You can mix and match: use deep agents for main logic, regular agents for specialized subagents

## Further Reading

- **Deep Agents Documentation**: See `docs/DEEP_AGENTS.md` for complete reference
- **LangChain Docs**: [docs.langchain.com](https://docs.langchain.com)
- **Deep Agents Package**: [pypi.org/project/deepagents/](https://pypi.org/project/deepagents/)

## Summary

Deep agents represent an evolution in agent design:
- **Steps 1-9** teach you how to build the pieces
- **Steps 10-12** show you an integrated harness with those pieces built-in
- Both approaches are valid; choose based on your use case
- The workshop progression helps you understand both the fundamentals AND the abstraction
