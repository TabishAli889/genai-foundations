# init_chat_model - GenAI - Chatbot - RAG
# Provider Name + Model ? openai:gpt-6
#create_agent - Agent AI - Agent - 

from datetime import datetime

from langchain.chat_models import init_chat_model
from langchain.tools import tool

# llm = init_chat_model(
#     model="gpt-6",
#     model_provider="openai",
# )

llm = init_chat_model(
    model="gemma4:31b-cloud",
    model_provider="ollama",
)

#langchain_openai 

history = [
    {"role": "system", "content": "You are helpful AI assitant"},   # System Message
]

@tool
def get_time() -> str:
    """Return current local time"""
    return datetime.now().strftime(format="%H:%T")

#tool bind
#Agentic loop

while True:
    question = input("You: ").strip()

    if question in {"exit", "bye", "quit"}:
        break

    history.append({"role": "user", "content": question})

    print("Bot: ", end="", flush=True)
    full= None
    completed = True

    try:
        for chunk in llm.stream(input=history):
            print(chunk.text, end="", flush=True)   # AI Message Chunck
            full = chunk if full is None else full + chunk
        completed = True
        print()
    except KeyboardInterrupt:
        print("\n[interupt]")
    except Exception as err:
        print(f"\n[error {err}]")

    if full is not None and full.text:
        history.append(full)
    else:
        history.pop()