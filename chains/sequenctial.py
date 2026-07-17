from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.
)

template1 = PromptTemplate(
    template = "give the detailed report on {topic}",
    input_variables=["topic"]
)

template2 = PromptTemplate(
    template = "give the 5 line summary on the {text}",
    input_variables = ["text"]
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({"topic" : "unemployement in india"})

print(result)
