from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3
)

chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{user_input}")
])

chat_history = []

# Read chat history from file
with open("chat_history.txt", "r") as f:
    for line in f:
        role, message = line.strip().split("|", 1)

        if role == "human":
            chat_history.append(HumanMessage(content=message))

        elif role == "ai":
            chat_history.append(AIMessage(content=message))

# Create prompt
prompt = chat_template.invoke({
    "chat_history": chat_history,
    "user_input": "What is my refund status?"
})

print(prompt)

# Get AI response
result = model.invoke(prompt)

print("\nAI:", result.content)