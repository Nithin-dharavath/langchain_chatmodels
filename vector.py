from dotenv import load_dotenv
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

doc1 = Document(
    page_content="Virat Kohli is the key batsman of the team.",
    metadata={"team": "Royal Challengers Bengaluru"}
)

doc2 = Document(
    page_content="Rohit Sharma is the greatest IPL captain.",
    metadata={"team": "Mumbai Indians"}
)

doc3 = Document(
    page_content="Jadeja is the all-rounder player of the team.",
    metadata={"team": "CSK"}
)

doc4 = Document(
    page_content="Jasprit Bumrah is the greatest death-over specialist.",
    metadata={"team": "Mumbai Indians"}
)

doc5 = Document(
    page_content="AB de Villiers is one of the greatest players for RCB.",
    metadata={"team": "Royal Challengers Bengaluru"}
)

docs = [doc1, doc2, doc3, doc4, doc5]

vector_store = Chroma(
    collection_name="sample",
    persist_directory="chroma_db",
    embedding_function=embeddings
)

vector_store.add_documents(docs)

result = vector_store.similarity_search_with_score(
    query="Who is the best player of RCB?",
    k=2
)

print(result)