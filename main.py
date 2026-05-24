"""
Dental Appointment System — powered by LangGraph
"""

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage, AIMessageChunk
from dental_agent.agent import dental_graph

# Detect which API key is available
xai_key = os.getenv("XAI_API_KEY", "")
groq_key = os.getenv("GROQ_API_KEY", "")

if xai_key:
    POWERED_BY = "LangGraph + Grok-4 (xAI)"
elif groq_key:
    POWERED_BY = "LangGraph + Groq (Llama 3.3 70B)"
else:
    POWERED_BY = "LangGraph"

BANNER = f"""
╔══════════════════════════════════════════════════════════╗
║         Dental Appointment Management System             ║
║         Powered by {POWERED_BY:<38}║
╚══════════════════════════════════════════════════════════╝
Examples:
  • Show available slots for an orthodontist
  • Book patient 1000082 with Emily Johnson on 5/10/2026 9:00
  • Cancel appointment for patient 1000082 at 5/10/2026 9:00
  • Reschedule patient 1000082 from 5/10/2026 9:00 to 5/12/2026 10:00
  • What appointments does patient 1000048 have?

Type 'quit' to exit.
"""


def run():
    print(BANNER)
    history = []

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "bye"}:
            print("Goodbye!")
            break

        history.append(HumanMessage(content=user_input))

        print("\nAgent: ", end="", flush=True)
        final_messages = None

        try:
            for event_type, data in dental_graph.stream(
                {"messages": history},
                stream_mode=["messages", "values"],
                config={"recursion_limit": 20},
            ):
                if event_type == "messages":
                    chunk, meta = data
                    # Stream tokens only from the agent (not tool results)
                    if (
                        isinstance(chunk, AIMessageChunk)
                        and chunk.content
                        and not getattr(chunk, "tool_calls", None)
                    ):
                        print(chunk.content, end="", flush=True)
                elif event_type == "values":
                    final_messages = data.get("messages", [])
        except Exception as exc:
            print(f"\nError: {exc}")
            history.pop()  # Remove the HumanMessage to avoid consecutive user messages
            continue

        print()
        if final_messages:
            history = final_messages


if __name__ == "__main__":
    run()
