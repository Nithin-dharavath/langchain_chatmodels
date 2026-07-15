from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3
)

chat_history = [
    SystemMessage(content="you are a helpful AI assistance")
]

while True:
    user_input = input("you : ")
    if user_input == "exit" :
        break
    chat_history.append(HumanMessage(content=user_input))
    result = model.invoke(chat_history)
    chat_history.append(result)
    print("AI :", result.content)

print(chat_history)