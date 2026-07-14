from langchain_huggingface import HuggingFaceembeddings
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


load_dotenv()

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

documents = (
    "delhi is captial of india",
    "kolkata is captial of west bengal",
    "hyderabad is captial of telangana",
    "bangalore is captial of karnataka",
    "chennai is captial of tamil nadu",
    "pune is captial of maharashtra",
    "jaipur is captial of rajasthan",
    "lucknow is captial of uttar pradesh",
)

query = "what is the captial of tealangana"

doc_vectors = embedding_model.embed_documents(documents)
query_vector = embedding_model.embed_query(query)

similarity_scores = cosine_similarity([query_vector], doc_vectors)[0]

