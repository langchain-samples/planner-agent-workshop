# Surfacing Subagent Interrupts to Supervisor

## Problem

When building multi-agent systems with a supervisor pattern, you often want subagents to pause and wait for user input (e.g., for approval, clarification, or conflict resolution). However, when a subagent calls `interrupt()`, it only pauses that subagent - the supervisor doesn't automatically know about it.

## Challenge

```python
# Subagent is interrupted
subagent_result = subagent.invoke({"messages": [...]})
# subagent_result = {"__interrupt__": [...], "messages": [...]}

# But the supervisor has already moved on!
# How do we surface this interrupt to the supervisor level?
```

## Solution: Interrupt Surfacing Pattern

The solution uses a combination of:
1. **Tool wrapper** that detects subagent interrupts
2. **Custom middleware** that propagates the interrupt to the supervisor
3. **Automatic resumption** of the subagent with user input

### Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Supervisor Agent                    │
│  ┌────────────────────────────────────────────────┐ │
│  │    SubagentInterruptMiddleware                 │ │
│  │  (after_tools: detect [SUBAGENT_INTERRUPT])   │ │
│  └────────────────────────────────────────────────┘ │
│                       │                              │
│                       ▼                              │
│         ┌──────────────────────────┐                │
│         │  call_subagent (tool)    │                │
│         │  - Invoke subagent       │                │
│         │  - Detect __interrupt__  │                │
│         │  - Return marker         │                │
│         └──────────────────────────┘                │
└─────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│                   Subagent                          │
│  ┌────────────────────────────────────────────────┐ │
│  │    SubagentMiddleware                          │ │
│  │  (e.g., approval middleware)                   │ │
│  │  - Calls interrupt()                           │ │
│  │  - Returns __interrupt__ in result             │ │
│  └────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

### Implementation Steps

#### 1. Create a Tool Wrapper for the Subagent

The tool wrapper invokes the subagent and detects when it's interrupted:

```python
# Store context for middleware to access
_subagent_context = {"last_result": None, "subagent_config": None}

@tool
def call_subagent(query: str) -> str:
    """Call the subagent to perform a task."""
    config = {"configurable": {"thread_id": "subagent-thread-1"}}
    _subagent_context["subagent_config"] = config

    # Invoke the subagent
    result = subagent.invoke({
        "messages": [{"role": "user", "content": query}]
    }, config=config)

    # Store result for middleware
    _subagent_context["last_result"] = result

    # Check if subagent was interrupted
    if "__interrupt__" in result:
        # Return a special marker that middleware will detect
        return "[SUBAGENT_INTERRUPT] Subagent is waiting for input."

    return result["messages"][-1].content
```

**Key Points:**
- Store the subagent result in a shared context
- Return a special marker `[SUBAGENT_INTERRUPT]` that the middleware can detect
- The marker is important because it's what the LLM sees, and what the middleware can check

#### 2. Create Custom Middleware to Surface the Interrupt

The middleware runs after tool execution and detects the special marker:

```python
class SubagentInterruptMiddleware(AgentMiddleware):
    """Surfaces subagent interrupts to the supervisor level."""

    def after_tools(self, state):
        """Check after tool execution if a subagent was interrupted."""
        messages = state['messages']
        last_message = messages[-1]

        # Detect the special marker
        if hasattr(last_message, 'content') and '[SUBAGENT_INTERRUPT]' in str(last_message.content):
            # Get the subagent's interrupt data
            subagent_result = _subagent_context.get("last_result")

            if subagent_result and "__interrupt__" in subagent_result:
                interrupt_data = subagent_result["__interrupt__"]

                # Create payload for supervisor-level interrupt
                interrupt_payload = {
                    "source": "subagent",
                    "subagent_interrupt": interrupt_data,
                    "message": "Subagent is waiting for input",
                }

                # Propagate interrupt to supervisor level
                user_response = interrupt(interrupt_payload)

                # Resume the subagent with user's response
                config = _subagent_context.get("subagent_config")
                resumed_result = subagent.invoke(
                    Command(resume=user_response),
                    config=config
                )

                # Return the completed result
                return {
                    "messages": [
                        HumanMessage(
                            content=f"[RESUMED] Subagent completed: {resumed_result['messages'][-1].content}",
                            name="subagent_handler"
                        )
                    ]
                }

        return None
```

**Key Points:**
- Use `after_tools()` hook to run after tool execution
- Detect the `[SUBAGENT_INTERRUPT]` marker in the tool's output
- Call `interrupt()` to pause the supervisor
- Automatically resume the subagent with the user's response
- Return the final result to continue supervisor execution

#### 3. Create the Supervisor with the Middleware

```python
supervisor_agent = create_agent(
    model=model,
    tools=[call_subagent],
    middleware=[SubagentInterruptMiddleware()],
    checkpointer=MemorySaver()
)
```

### Usage Example

```python
config = {"configurable": {"thread_id": "supervisor-1"}}

# First invocation - will interrupt when subagent needs input
result = supervisor_agent.invoke(
    {"messages": [{"role": "user", "content": "Search for weather"}]},
    config=config,
)

# Check if interrupted
if '__interrupt__' in result:
    interrupt_data = result['__interrupt__'][0]
    print(f'Supervisor interrupted: {interrupt_data.value}')

    # User provides input
    user_response = {"status": "approved", "message": "Looks good!"}

    # Resume supervisor - middleware will handle subagent resumption
    final_result = supervisor_agent.invoke(
        Command(resume=user_response),
        config=config
    )

    print(f"Final result: {final_result['messages'][-1].content}")
```

## Execution Flow

1. **User** → Supervisor: "Search for weather"
2. **Supervisor** → Subagent tool: "Search for weather"
3. **Subagent tool** → Subagent: Invoke with query
4. **Subagent middleware**: Calls `interrupt()` → Returns `{"__interrupt__": [...]}`
5. **Subagent tool**: Detects `__interrupt__` → Returns `"[SUBAGENT_INTERRUPT] ..."`
6. **Supervisor middleware** (`after_tools`): Detects marker → Calls `interrupt()`
7. **Supervisor**: Returns `{"__interrupt__": [...]}` to user
8. **User** → Supervisor: `Command(resume=user_response)`
9. **Supervisor middleware**: Resumes subagent with `Command(resume=user_response)`
10. **Subagent**: Completes execution
11. **Supervisor middleware**: Returns completed result
12. **Supervisor**: Continues with final answer

## Key Patterns

### Pattern 1: Shared Context

Use a module-level dictionary to share state between the tool and middleware:

```python
_subagent_context = {"last_result": None, "config": None}
```

This is necessary because the tool and middleware run in different contexts.

### Pattern 2: Special Marker

Use a special string marker in the tool's return value:

```python
return "[SUBAGENT_INTERRUPT] Subagent waiting..."
```

This allows the middleware to detect that a subagent was interrupted without complex state management.

### Pattern 3: Transparent Resumption

The middleware automatically resumes the subagent, making the interrupt transparent to the supervisor's main logic:

```python
user_response = interrupt(interrupt_payload)
resumed_result = subagent.invoke(Command(resume=user_response), config=config)
return {"messages": [HumanMessage(content=resumed_result['messages'][-1].content)]}
```

## Benefits

1. **Encapsulation**: Subagent interrupts are handled transparently
2. **Flexibility**: Each subagent can have its own interrupt logic
3. **Composability**: Multiple levels of supervisor-subagent relationships
4. **User Experience**: Single interrupt point for the user

## Limitations

1. **Nested Interrupts**: If multiple subagents interrupt simultaneously, only one is handled
2. **State Management**: Requires careful management of shared context
3. **Thread Safety**: The shared context pattern is not thread-safe

## Real-World Use Cases

1. **Approval Workflows**: Search agent needs approval before returning results
2. **Conflict Resolution**: Calendar agent needs user input to resolve scheduling conflicts
3. **Multi-Step Processes**: Research agent needs clarification mid-research
4. **Human-in-the-Loop**: Any subagent that requires human input or approval

## See Also

- [agent_06_supervisor_multi_agent.py](../agents/agent_06_supervisor_multi_agent.py) - Full implementation
- [examples/subagent_interrupt_surfacing.py](../examples/subagent_interrupt_surfacing.py) - Standalone example
