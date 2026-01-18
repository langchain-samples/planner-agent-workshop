"""
Step 8: Image Handling
Adds image analysis capability by including images directly in the prompt.
Demonstrates content blocks and image handling in LangChain v1.
"""

from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
from agents.models import model

load_dotenv(override=True)

# Mock calendar storage
_calendar_events: list = []


@tool
def read_calendar(date: str = None) -> str:
    """Read calendar events."""
    if not _calendar_events:
        return "No events in calendar"
    return "\n".join([f"- {e['title']} on {e['date']} at {e['time']}" for e in _calendar_events])


@tool
def write_calendar(title: str, date: str, time: str, location: str = "") -> str:
    """Create a new calendar event."""
    new_event = {"title": title, "date": date, "time": time, "location": location}
    _calendar_events.append(new_event)
    return f"Successfully created event '{title}' on {date} at {time}"


# System prompt
# Uses the model from models.py (should be vision-capable for image analysis)
SYSTEM_PROMPT = """You are a helpful calendar assistant. You can:
- Read calendar events using read_calendar
- Create new events using write_calendar
- Analyze images of calendars or event information

If a user provides an image, analyze it directly and help them accordingly."""

# Create memory saver
checkpointer = MemorySaver()

# Create the agent
agent = create_agent(
    model=model,
    tools=[read_calendar, write_calendar],
    system_prompt=SYSTEM_PROMPT,
    # checkpointer=checkpointer,
)

# if __name__ == "__main__":
#     # Example usage
#     print("=== Agent with Image Handling ===\n")
    
#     thread_id = "image-conversation-1"
#     config = {"configurable": {"thread_id": thread_id}}
    
#     # Request with image
#     print("User: [Provides image of calendar]")
#     print("What events are shown in this image?")
    
#     # Load image from file and encode it as base64
#     import base64
#     with open("assets/kpop-flyer.png", "rb") as image_file:
#         image_data = image_file.read()
#         image_base64 = base64.b64encode(image_data).decode('utf-8')
#         image_data_uri = f"data:image/png;base64,{image_base64}"
    
#     # Include the image directly in the message content using content blocks
#     result = agent.invoke({
#         "messages": [{
#             "role": "user", 
#             "content": [
#                 {"type": "text", "text": "What events are shown in this image? schedule them"},
#                 {"type": "image_url", "image_url": {"url": image_data_uri}}
#             ]
#         }]
#     }, config=config)

#     print(f"Agent: {result['messages'][-1].content}")
    
#     print("\nNote: Images are included directly in the message content:")
#     print("1. Images are encoded as base64 data URIs")
#     print("2. Images are included in the message using content blocks")
#     print("3. The vision-capable model analyzes the image directly")
#     print("4. The agent can answer questions about the image content")

