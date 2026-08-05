from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from dotenv import load_dotenv

load_dotenv()

# ----------------------------------
# STEP 1 : INDEXING
# ----------------------------------

video_id = "Gfr50f6ZBvo"

try:
    ytt_api = YouTubeTranscriptApi()

    transcript_list = ytt_api.fetch(
        video_id,
        languages=["en"]
    ).to_raw_data()

    transcript = " ".join(
        chunk["text"] for chunk in transcript_list
    )

except TranscriptsDisabled:
    print("No captions available for this video.")
    exit()

# ----------------------------------
# STEP 2 : TEXT SPLITTING
# ----------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

chunks = splitter.create_documents([transcript])

# ----------------------------------
# STEP 3 : EMBEDDINGS + VECTOR STORE
# ----------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = FAISS.from_documents(
    chunks,
    embeddings
)

# ----------------------------------
# STEP 4 : RETRIEVER
# ----------------------------------

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# ----------------------------------
# STEP 5 : PROMPT
# ----------------------------------

prompt = PromptTemplate(
    template="""
You are a helpful AI assistant.

Answer the user's question ONLY using the transcript context.

If the answer is not available in the transcript,
reply only with:

"I don't know."

Transcript:
{context}

Question:
{question}
""",
    input_variables=["context", "question"],
)

# ----------------------------------
# STEP 6 : LLM
# ----------------------------------

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.4
)

# ----------------------------------
# STEP 7 : FORMAT RETRIEVED DOCUMENTS
# ----------------------------------

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ----------------------------------
# STEP 8 : Langchain CHAIN
# ----------------------------------

parallel_chain = RunnableParallel(
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    }
)

parser = StrOutputParser()

final_chain = ( parallel_chain | prompt | llm | parser)

# ----------------------------------
# STEP 9 : ASK QUESTION
# ----------------------------------

question = input("type the question")

response = final_chain.invoke(question)

print(response)