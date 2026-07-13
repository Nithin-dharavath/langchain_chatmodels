from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

load_dotenv()

chat = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=1.5,
    max_completion_tokens=30
)

response = chat.invoke([
    HumanMessage(content="Explain LangChain in one sentence.")
])

print(response.content)