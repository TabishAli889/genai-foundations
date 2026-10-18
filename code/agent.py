from datetime import datetime
from pydantic import BaseModel, Field   
from typing import Optional
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy


def get_time() -> str:
    """Return the current local time in HH:MM (24-hour) format."""
    return datetime.now().strftime("%H:%M")


class AgentResponse(BaseModel):
    is_greeting: bool = Field(
        description="True if the user's message is a greeting, otherwise False."
    )
    time: Optional[str] = Field(
        default=None,
        description="Current time returned by get_time. Only fill this if is_greeting is True.",
    )
    answer: str = Field(
        description=(
            "Your final reply. If is_greeting is True, greet the user "
            "(good morning/afternoon/evening/night) based on the time. "
            "Otherwise, answer the question normally."
        )
    )

agent = create_agent(
    model="ollama:gemma4:31b-cloud",
    tools=[get_time],
    response_format=ToolStrategy(AgentResponse),
    system_prompt=(
        "You are a helpful AI agent. "
        "If the user is greeting you, set is_greeting=true, call the get_time tool, "
        "put the result in time, and greet them according to the time of day. "
        "Otherwise set is_greeting=false, leave time empty, and just answer normally. "
        "Never guess the time; always use the get_time tool."
    ),
)

query = input("Query: ")
result = agent.invoke({"messages": [{"role": "user", "content": query}]})

response: AgentResponse = result["structured_response"]

if response.is_greeting:
    print(f"Time: {response.time}")
    print(f"Greeting: {response.answer}")
else:
    print(response.answer)

#-----------------------------------

# Hi
# Time: 10:19
# Greeting: Hi good morning
# What is NLP?
# Nlp is ....

#-----------------------------------

# 1. Memory --- short-term memory (check pointer)
# 2. Streaming --- AI message chunk + Tool message
# 3. Middleware ... HITL
# 4. MAS 

#-----------------------------------

# 1. Simple Agent (without tool or chatbot)
# 2. Agent with tool
# 3. Agent with structured output
# 4. Agent with short-term memory (check pointer)
# 5. Agent with other tools (file search, web search)

#-----------------------------------

# RAG (file search) -- 2 step RAG
# Agentic RAG with Agent like file search tool

#-----------------------------------

# Langchain/Langraph -- environment
# OpenAI Agents SDKs 2.0 -- sanbox

#-----------------------------------

# Agent = LLM
# Agent = LLM + Tools
# Agent = LLM + Tool + Memory
# Agent = LLM + Harness

#-----------------------------------

# Agent Idea: Support Ticket Triage

# Query: Customer message
# [HIGH] billing summary

# - priority: low, medium, high
# -category: billing, technical, account, other
# -summary:

# System prompt: Classify the customer support message