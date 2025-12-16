# Quick Reference: Subagent Interrupt Surfacing

## TL;DR

To surface interrupts from a subagent to a supervisor:

1. **Tool wrapper** detects `__interrupt__` and returns marker
2. **Middleware** detects marker and calls `interrupt()`
3. **Middleware** resumes subagent with user response

## Minimal Example

```python
from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.types import interrupt, Command

# Shared context
_ctx = {"result": None, "config": None}

# Tool wrapper
@tool
def call_subagent(query: str) -> str:
    config = {"configurable": {"thread_id": "sub-1"}}
    _ctx["config"] = config

    result = subagent.invoke({"messages": [{"role": "user", "content": query}]}, config)
    _ctx["result"] = result

    if "__interrupt__" in result:
        return "[SUBAGENT_INTERRUPT] Waiting..."
    return result["messages"][-1].content

# Middleware
class SubagentInterruptMiddleware(AgentMiddleware):
    def after_tools(self, state):
        last = state['messages'][-1]
        if '[SUBAGENT_INTERRUPT]' in str(last.content):
            result = _ctx.get("result")
            if result and "__interrupt__" in result:
                # Propagate interrupt
                user_resp = interrupt(result["__interrupt__"])

                # Resume subagent
                resumed = subagent.invoke(
                    Command(resume=user_resp),
                    config=_ctx["config"]
                )

                # Return completed result
                return {
                    "messages": [HumanMessage(
                        content=resumed['messages'][-1].content,
                        name="subagent_handler"
                    )]
                }
        return None

# Supervisor with middleware
supervisor = create_agent(
    model=model,
    tools=[call_subagent],
    middleware=[SubagentInterruptMiddleware()],
    checkpointer=checkpointer
)
```

## Usage

```python
# Invoke supervisor
result = supervisor.invoke(
    {"messages": [{"role": "user", "content": "Do task"}]},
    config={"configurable": {"thread_id": "1"}}
)

# Check for interrupt
if '__interrupt__' in result:
    # Resume with user input
    final = supervisor.invoke(
        Command(resume={"status": "approved"}),
        config={"configurable": {"thread_id": "1"}}
    )
```

## Key Pattern Elements

| Element | Purpose |
|---------|---------|
| `_ctx` dict | Share state between tool and middleware |
| `[SUBAGENT_INTERRUPT]` | Marker string for middleware to detect |
| `after_tools()` | Hook that runs after tool execution |
| `interrupt()` | Pauses execution and waits for user input |
| `Command(resume=...)` | Resumes execution with user input |

## Execution Flow

```
User → Supervisor → Tool → Subagent
                             ↓ (interrupt)
                          Returns {"__interrupt__": [...]}
                             ↓
                    Tool returns "[SUBAGENT_INTERRUPT]"
                             ↓
                    Middleware detects marker
                             ↓
                    Middleware calls interrupt()
                             ↓
                    Supervisor returns {"__interrupt__": [...]}
                             ↓
User provides input via Command(resume=...)
                             ↓
                    Middleware resumes subagent
                             ↓
                    Subagent completes
                             ↓
                    Supervisor continues
```

## Common Mistakes

❌ **Don't** try to handle the interrupt in the tool directly
```python
@tool
def call_subagent(query: str) -> str:
    result = subagent.invoke(...)
    if "__interrupt__" in result:
        # ❌ This won't work - tool execution is already done
        user_input = interrupt(...)
    return result["messages"][-1].content
```

✅ **Do** use middleware to handle the interrupt
```python
class SubagentInterruptMiddleware(AgentMiddleware):
    def after_tools(self, state):
        # ✅ Middleware can call interrupt() and resume
        if '[SUBAGENT_INTERRUPT]' in state['messages'][-1].content:
            user_input = interrupt(...)
            resumed_result = subagent.invoke(Command(resume=user_input), ...)
```

❌ **Don't** forget to store the config
```python
@tool
def call_subagent(query: str) -> str:
    config = {"configurable": {"thread_id": "sub-1"}}
    result = subagent.invoke(..., config)
    # ❌ Middleware won't be able to resume without config
```

✅ **Do** store both result and config in shared context
```python
@tool
def call_subagent(query: str) -> str:
    config = {"configurable": {"thread_id": "sub-1"}}
    _ctx["config"] = config  # ✅ Store for middleware
    result = subagent.invoke(..., config)
    _ctx["result"] = result  # ✅ Store for middleware
```

## See Also

- [Full Documentation](./subagent_interrupt_surfacing.md)
- [Example Implementation](../examples/subagent_interrupt_surfacing.py)
- [Workshop Agent 06](../agents/agent_06_supervisor_multi_agent.py)
