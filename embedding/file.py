from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

docs = (
    "delhi is captial of india",
    "kolkata is captial of west bengal",
    "hyderabad is captial of telangana",
    "bangalore is captial of karnataka",
    "chennai is captial of tamil nadu"
)

embedding_vector = embeddings.embed_documents(docs)

print(str(embedding_vector))