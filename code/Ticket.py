from langchain.agents import create_agent
from pydantic import BaseModel
from typing import Literal
from langchain.agents.structured_output import ToolStrategy


class Ticket(BaseModel):
    priority: Literal["low", "medium", "high"]
    category: Literal["billing", "technical", "account", "other"]
    summary: str


agent = create_agent(
    model="ollama:gemma4:31b-cloud",
    tools=[],
    response_format=ToolStrategy(Ticket),
    system_prompt="Classify the customer support message.",
)

query = input("Customer message: ")
result = agent.invoke(
    {"messages": [{"role": "user", "content": query}]}
)

t = result["structured_response"]
print(f"[{t.priority.upper()}] {t.category} - {t.summary}")