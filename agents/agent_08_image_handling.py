"""
Step 8: Image Handling with Custom Middleware
Adds image analysis capability using custom middleware that processes images before agent execution.
Demonstrates content blocks and image handling in LangChain v1.
"""

from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentMiddleware
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from typing import Any, Dict, Optional
from typing_extensions import TypedDict
from dotenv import load_dotenv

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


# Custom state to include images
class ImageState(TypedDict, total=False):
    """Extended state that includes optional image input."""
    messages: list  # Required field for agent state
    image: Optional[str]  # Base64 encoded image or image URL


# Custom middleware for image analysis
class ImageAnalysisMiddleware(AgentMiddleware):
    """Middleware that analyzes images and provides summaries before agent execution."""
    
    state_schema = ImageState
    
    def __init__(self, vision_model):
        super().__init__()
        self.vision_model = vision_model
    
    def before_model(self, state) -> Dict[str, Any] | None:
        """Analyze image if present and add summary to messages."""
        image = state.get("image")
        if not image:
            return None
        
        # Analyze the image using vision model
        # In production, you would decode the image and send it to the vision model
        # For this example, we'll create a mock analysis
        analysis_prompt = "Analyze this image and provide a detailed description of what you see."
        
        # In real implementation:
        # vision_response = self.vision_model.invoke([
        #     {"type": "text", "text": analysis_prompt},
        #     {"type": "image_url", "image_url": {"url": image}}
        # ])
        
        # Mock response for demonstration
        image_summary = "Image analysis: This appears to be a calendar screenshot showing events for December 20th, including a meeting at 11 AM and a soccer game at 3 PM."
        
        # Add the image analysis as a system message to help the agent
        analysis_message = {
            "role": "system",
            "content": f"User provided an image. Image analysis: {image_summary}"
        }
        
        # Prepend the analysis to messages
        current_messages = state.get("messages", [])
        return {
            "messages": [analysis_message] + current_messages
        }


# Initialize models
main_model = init_chat_model("gpt-4o-mini", temperature=0)
# In production, use a vision-capable model like gpt-4o or claude-3-opus
vision_model = init_chat_model("gpt-4o-mini", temperature=0)

# Create image analysis middleware
image_middleware = ImageAnalysisMiddleware(vision_model)

# System prompt
SYSTEM_PROMPT = """You are a helpful calendar assistant. You can:
- Read calendar events using read_calendar
- Create new events using write_calendar
- Analyze images of calendars or event information

If a user provides an image, use the image analysis provided to understand the content and help them accordingly."""

# Create memory saver
checkpointer = MemorySaver()

# Create the agent with image handling middleware
agent = create_agent(
    model=main_model,
    tools=[read_calendar, write_calendar],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
    middleware=[image_middleware],
    state_schema=ImageState,  # Use custom state schema
)

if __name__ == "__main__":
    # Example usage
    print("=== Agent with Image Handling ===\n")
    
    thread_id = "image-conversation-1"
    config = {"configurable": {"thread_id": thread_id}}
    
    # Request with image (mock)
    print("User: [Provides image of calendar]")
    print("What events are shown in this image?")
    
    # In production, you would pass the actual image:
    # result = agent.invoke({
    #     "messages": [{"role": "user", "content": "What events are shown in this image?"}],
    #     "image": "data:image/png;base64,..."  # or image URL
    # }, config=config)
    
    print("\nNote: The ImageAnalysisMiddleware:")
    print("1. Detects if an image is provided in the state")
    print("2. Analyzes the image using a vision model")
    print("3. Adds the analysis as context for the agent")
    print("4. The agent can then answer questions about the image content")

