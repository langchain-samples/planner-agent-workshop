# Fixes for Agents 07 & 08 - Import Corrections

## Agent 07: Guardrail Middleware

### Issues Fixed

**Import Issues:**
1. `AgentMiddleware` was imported from wrong location
2. `AgentState` import was unnecessary

**Before:**
```python
from langchain.agents import create_agent, AgentMiddleware
from langchain.agents import AgentState

def before_model(self, state: AgentState, runtime) -> Dict[str, Any] | None:
```

**After:**
```python
from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentMiddleware

def before_model(self, state) -> Dict[str, Any] | None:
```

### Changes Made

1. **Fixed AgentMiddleware import** ✅
   - Changed: `from langchain.agents import create_agent, AgentMiddleware`
   - To: `from langchain.agents import create_agent` and `from langchain.agents.middleware.types import AgentMiddleware`

2. **Removed unused AgentState import** ✅
   - Removed: `from langchain.agents import AgentState`
   - Not needed since we're not using it

3. **Simplified method signature** ✅
   - Changed: `def before_model(self, state: AgentState, runtime)`
   - To: `def before_model(self, state)`
   - The `runtime` parameter isn't used and type annotation isn't necessary

## Agent 08: Image Handling Middleware

### Issues Fixed

**Import Issues:**
1. `AgentMiddleware` was imported from wrong location
2. `AgentState` import was incorrect
3. `ImageState` incorrectly extended `AgentState`

**Before:**
```python
from langchain.agents import create_agent, AgentMiddleware
from langchain.agents import AgentState

class ImageState(AgentState):
    image: Optional[str] = None

def before_model(self, state: ImageState, runtime) -> Dict[str, Any] | None:
```

**After:**
```python
from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentMiddleware
from typing_extensions import TypedDict

class ImageState(TypedDict, total=False):
    messages: list
    image: Optional[str]

def before_model(self, state) -> Dict[str, Any] | None:
```

### Changes Made

1. **Fixed AgentMiddleware import** ✅
   - Changed: `from langchain.agents import create_agent, AgentMiddleware`
   - To: `from langchain.agents import create_agent` and `from langchain.agents.middleware.types import AgentMiddleware`

2. **Removed AgentState import** ✅
   - Removed: `from langchain.agents import AgentState`
   - Not available in the expected location

3. **Fixed ImageState definition** ✅
   - Changed: `class ImageState(AgentState):`
   - To: `class ImageState(TypedDict, total=False):`
   - Added `messages: list` field (required for agent state)
   - Used `TypedDict` which is the correct base for custom state schemas

4. **Simplified method signature** ✅
   - Changed: `def before_model(self, state: ImageState, runtime)`
   - To: `def before_model(self, state)`

## Why These Changes Were Needed

### AgentMiddleware Import Location

In LangChain v1, middleware classes are organized in a specific module structure:
- ❌ `from langchain.agents import AgentMiddleware` (doesn't exist)
- ✅ `from langchain.agents.middleware.types import AgentMiddleware` (correct)

This is consistent with the pattern we saw in agent 06.

### Custom State Schema

When creating custom state schemas in LangChain v1:
1. Use `TypedDict` as the base class
2. Always include `messages: list` field
3. Mark additional fields as optional if they're not always present
4. Use `total=False` to make all fields optional except those explicitly typed as required

**Example:**
```python
from typing_extensions import TypedDict
from typing import Optional

class CustomState(TypedDict, total=False):
    messages: list  # Required for agent
    custom_field: Optional[str]  # Your custom field
```

### Method Signatures

The middleware hooks (`before_model`, `after_tools`, etc.) should have simple signatures:
- Just `self` and `state`
- No type annotation for `state` needed (middleware handles any state type)
- No `runtime` parameter needed

## Testing

Both agents now compile successfully:

```bash
uv run python -m py_compile agents/agent_07_guardrail_middleware.py
uv run python -m py_compile agents/agent_08_image_handling.py
```

### Test Agent 07 (Guardrail)

```bash
uv run python agents/agent_07_guardrail_middleware.py
```

Expected behavior:
- Normal calendar queries work fine
- Malicious queries (prompt injection attempts) are blocked
- Guardrail middleware runs before the agent

### Test Agent 08 (Image Handling)

```bash
uv run python agents/agent_08_image_handling.py
```

Expected behavior:
- Demonstrates image handling setup
- Shows how custom state can include additional fields
- Middleware can process images before agent execution

## Key Takeaways

1. **Always import AgentMiddleware from the correct location:**
   ```python
   from langchain.agents.middleware.types import AgentMiddleware
   ```

2. **Use TypedDict for custom state schemas:**
   ```python
   class MyState(TypedDict, total=False):
       messages: list
       my_field: Optional[Any]
   ```

3. **Keep middleware method signatures simple:**
   ```python
   def before_model(self, state):
       # Your logic here
       return None  # or return updates
   ```

4. **State updates are merged:**
   When you return `{"messages": [...]}` from middleware, it's merged with existing state, not replaced.

## Related Documentation

- [Agent 06 Fixes](./FIXES_AGENT_06.md) - Similar import fixes
- [Subagent Interrupt Surfacing](./subagent_interrupt_surfacing.md) - Custom middleware patterns
- [Quick Reference](./QUICK_REFERENCE.md) - Middleware patterns
