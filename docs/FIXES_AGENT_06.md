# Fixes for Agent 06 - Subagent Interrupt Surfacing

## Problem

The agent wasn't triggering interrupts because:

1. **Calendar agent was too passive** - It was checking for conflicts but not actually *attempting* to schedule events
2. **Date was hardcoded** - Calendar event date didn't match search results
3. **Single interrupt handling** - Middleware only handled one interrupt, not sequential interrupts

## Root Cause

When the supervisor asked the calendar agent to schedule an event, the calendar agent would:
1. Check the calendar for conflicts
2. Find conflicts
3. **Report back without trying to schedule** ❌

This meant `write_calendar` never got called, so no conflict was detected, and `ask_for_help` was never triggered.

## Fixes Applied

### Fix 1: Made Calendar Agent More Proactive

**Before:**
```python
system_prompt="""...
When scheduling events, always check the calendar first for conflicts.
If you find a conflict, use ask_for_help to ask the user...
"""
```

**After:**
```python
system_prompt="""...
IMPORTANT: When asked to schedule an event, you MUST attempt to write_calendar.
If write_calendar returns a conflict message, then use ask_for_help...

Do NOT just check the calendar and report conflicts - actually TRY to schedule the event using write_calendar.
The write_calendar tool will tell you if there's a conflict.
"""
```

**Why:** The agent now attempts `write_calendar` first, which triggers the conflict detection, which then prompts it to use `ask_for_help`.

### Fix 2: Dynamic Date Generation

**Before:**
```python
_calendar_events: List[Dict] = [
    {
        "title": "Team Meeting",
        "date": "2024-12-21",  # Hardcoded
        "time": "19:00",
        "location": "Office"
    }
]
```

**After:**
```python
from datetime import datetime, timedelta

tomorrow_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

_calendar_events: List[Dict] = [
    {
        "title": "Team Meeting",
        "date": tomorrow_date,  # Dynamic
        "time": "19:00",
        "location": "Office"
    }
]
```

**Why:** "Tomorrow" now actually means tomorrow, not a hardcoded date.

### Fix 3: Updated Mock Search with Structured Data

**Before:**
```python
return f"""Search results for "{query}":

1. Swan Lake Ballet - Tomorrow night at 6PM
   Description: Classical ballet performance by Seoul Ballet Company
   URL: https://example.com/ballet-swan-lake
"""
```

**After:**
```python
return f"""Search results for "{query}":

1. Swan Lake Ballet - {tomorrow_date} at 18:00 (6PM)
   Description: Classical ballet performance by Seoul Ballet Company
   Location: Seoul Arts Center
   URL: https://example.com/ballet-swan-lake
"""
```

**Why:** Provides specific date and time in a format the calendar agent can use.

### Fix 4: Handle Multiple Sequential Interrupts

**Before:**
```python
def after_tools(self, state):
    if '[SUBAGENT_INTERRUPT]' in str(last_message.content):
        user_response = interrupt(interrupt_payload)
        resumed_result = calendar_agent.invoke(Command(resume=user_response), config)
        # ❌ If resumed_result has __interrupt__ again, it's not handled
        return {"messages": [...]}
```

**After:**
```python
def after_tools(self, state):
    if '[SUBAGENT_INTERRUPT]' in str(last_message.content):
        # Handle multiple interrupts in a loop
        while "__interrupt__" in subagent_result:
            user_response = interrupt(interrupt_payload)
            subagent_result = calendar_agent.invoke(Command(resume=user_response), config)
            # ✅ Loop continues if there's another interrupt
        return {"messages": [...]}
```

**Why:** The calendar agent may be interrupted multiple times:
1. First by `ask_for_help` (interrupt from tool)
2. Then by `HumanInTheLoopMiddleware` (interrupt for `reschedule_calendar` approval)

The middleware now handles all interrupts in sequence before returning the final result.

### Fix 5: Improved Example Code

**Before:**
- Invoked supervisor
- Then separately invoked calendar agent directly
- Confusing flow that didn't demonstrate interrupt surfacing

**After:**
- Invokes supervisor once
- Checks for supervisor interrupts (surfaced from subagent)
- Handles multiple interrupts in sequence
- Clear demonstration of interrupt surfacing

## Expected Behavior Now

When you run the agent:

1. **Supervisor** asks search agent for ballet events
2. **Search agent** returns Swan Lake (6PM) and Modern Dance (7PM)
3. **Supervisor** asks calendar agent to schedule Swan Lake
4. **Calendar agent** tries `write_calendar` for Swan Lake at 6PM → ✅ Success
5. **Supervisor** asks calendar agent to schedule Modern Dance
6. **Calendar agent** tries `write_calendar` for Modern Dance at 7PM → ❌ Conflict!
7. **Calendar agent** calls `ask_for_help` → 🔔 **INTERRUPT #1**
8. **SubagentInterruptMiddleware** detects interrupt → Surfaces to supervisor
9. **Supervisor** returns `{"__interrupt__": [...]}` to user
10. **User** provides response: "Yes, reschedule Team Meeting to 8 PM"
11. **Middleware** resumes calendar agent with user response
12. **Calendar agent** calls `reschedule_calendar` → 🔔 **INTERRUPT #2** (from HumanInTheLoopMiddleware)
13. **Middleware** surfaces second interrupt to supervisor
14. **User** approves: `{"decisions": [{"type": "approve"}]}`
15. **Middleware** resumes calendar agent with approval
16. **Calendar agent** completes reschedule and schedules Modern Dance
17. **Supervisor** receives final result and responds to user

## Testing

Run the agent:
```bash
uv run python agents/agent_06_supervisor_multi_agent.py
```

You should see:
- Supervisor coordinating between agents
- 🔔 SUPERVISOR INTERRUPTED! messages when subagent interrupts are surfaced
- Multiple interrupts handled in sequence
- Final successful scheduling

## Key Takeaway

For interrupt surfacing to work:
1. **Subagent must actually interrupt** - Make sure the subagent's logic triggers `interrupt()` or `ask_for_help`
2. **Tool wrapper must detect it** - Check for `__interrupt__` in subagent result
3. **Middleware must propagate it** - Call `interrupt()` at supervisor level
4. **Handle multiple interrupts** - Use a loop to handle sequential interrupts

The most common mistake is the subagent never actually calling `interrupt()` because it's being too conservative or passive in its logic.
