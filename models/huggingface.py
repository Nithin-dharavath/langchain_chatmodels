from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    max_new_tokens=100,
)

chat_model = ChatHuggingFace(llm=llm)

response = chat_model.invoke("What is the capital of India?")

print(response.content)