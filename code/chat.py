import os
from pyexpat import model
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
#from langchain.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv() #Read .env file

model_name = os.getenv(key="MODEL_NAME")
if not model_name:
    raise SystemError("MODEL_NAME is not set in .env file (e.g., 'openai:gpt-6').")

llm = init_chat_model(
    model=os.getenv("MODEL_NAME"),
   # temperature=os.getenv(key="TEMPERATURE"),
    api_key=os.getenv(key="API_KEY")
)

question = input("You: ")
response = llm.invoke(input=question)
print("Bot:", response.text)

value = [5, 6, 7, 8, {"key": "value"}]

SYSTEM_PROMPT = "You are a helpful assistant."

history = [
    {"role": "system", "content": SYSTEM_PROMPT},  #System message

]

while True:
    try:
        print()
        question = input("You: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye!")
        break

    if not question:
        continue

    if question.lower() in ("exit", "quit", "bye"):
        print("Goodbye!")
        break

    history.append({"role": "user", "content": question})   #User message
    print("Bot: ", end="", flush=True)
   # response = llm.invoke(input=history)    #AI message

    full = None  # None | AIMessageChunk
    try:
        for chunk in llm.stream(input=history):
                print(chunk.text, end="", flush=True)  # print the chunk as it arrives
                full = chunk if full is None else full + chunk
        print()

    except KeyboardInterrupt:
        print("\n[Interrupted]")
    except Exception as e:
        print(f"\n[Error: {e}]")

    if not full and full.text: # pyright: ignore[reportOptionalMemberAccess]
        history.append(full)  # pyright: ignore[reportArgumentType] # AI message
    else:
        history.pop()  # remove the last user message if no response was generated

