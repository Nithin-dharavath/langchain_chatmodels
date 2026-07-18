from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

loader = DirectoryLoader(
    path="rag/doc_loader",
    glob="*.txt",
    loader_cls=TextLoader
)

docs = loader.load()

text = "\n\n".join(doc.page_content for doc in docs)

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3
)

prompt = PromptTemplate(
    template="Summarize the content in under 10 lines.\n\n{text}",
    input_variables=["text"]
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"text": text})

print(result)