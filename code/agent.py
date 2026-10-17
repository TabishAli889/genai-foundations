from datetime import datetime
from langchain.agents import create_agent

def get_time() -> str:
    """Return current local time"""
    return datetime.now().strftime(format="%H:%T")

agent = create_agent(
    model = "ollama:gemma4:31b-cloud",
    tools = [get_time],
    system_prompt = "You are helpful AI assistant"
)

query = input("Query: ")
result = agent.invoke(
    input={"messages": [{"role": "user", "content": query}]}
)
print(result["messages"][-1].text)
