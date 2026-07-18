from langchain_groq import ChatGroq
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

groq_model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.4
)

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation"
)

model2 = ChatHuggingFace(llm=llm)

llm2 = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model3 = ChatHuggingFace(llm=llm2)

prompt1 = PromptTemplate(
    template="Give 5 advantages of {topic}.",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Give 2 disadvantages of {topic}.",
    input_variables=["topic"]
)

prompt3 = PromptTemplate(
    template="""Advantages{advantages} Disadvantages:{disadvantages} Write a concise summary balancing both.""",
    input_variables=["advantages", "disadvantages"]
)

parser = StrOutputParser()

chain_parallel = RunnableParallel(
    advantages=prompt1 | groq_model | parser,
    disadvantages=prompt2 | model2 | parser
)

chain = chain_parallel | prompt3 | model3 | parser

result = chain.invoke({"topic": "technology automation"})

print(result)