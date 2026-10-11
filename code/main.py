import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
#from langchain.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv() #Read .env file

llm = init_chat_model(
    model=os.getenv("MODEL_NAME"),
    temperature=os.getenv(key="TEMPERATURE"),
    api_key=os.getenv("API_KEY")
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
    question = input("You: ").strip()

    if not question:
        continue

    if question.lower() in ("exit", "quit", "bye"):
        print("Goodbye!")
        break

    history.append({"role": "user", "content": question})   #User message
    response = llm.invoke(input=history)    #AI message
    
    history.append(response.text)

    print(f"Bot: {response.text}")

