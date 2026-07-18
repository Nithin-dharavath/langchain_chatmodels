from typing import Literal
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_core.output_parsers import (StrOutputParser,PydanticOutputParser)

load_dotenv()

groq_model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3
)

parser = StrOutputParser()

class Feedback(BaseModel):
    feedback: Literal["positive", "negative"] = Field(
        description="Whether the feedback is positive or negative."
    )

pydantic_parser = PydanticOutputParser(
    pydantic_object=Feedback
)

prompt1 = PromptTemplate(
    template="""Determine whether the following customer feedback is positive or negative.{text} {format_instructions}""",
    input_variables=["text"],
    partial_variables={
        "format_instructions":
        pydantic_parser.get_format_instructions()
    },
)

classified_chain = (prompt1 | groq_model | pydantic_parser)

prompt2 = PromptTemplate(
    template="""Write an appropriate response to this positive feedback.{text}""",
    input_variables=["text"],
)

prompt3 = PromptTemplate(
    template="""Write an appropriate response to this negative feedback.{text}""",
    input_variables=["text"],
)

branch_chain = RunnableBranch(
    (lambda x: x.feedback == "positive", prompt2 | groq_model | parser),
    (lambda x: x.feedback == "negative", prompt3 | groq_model | parser),
    RunnableLambda(
        lambda _: "Could not determine feedback."
    ),
)

chain = classified_chain | branch_chain

result = chain.invoke({"text": "I love this product! It has changed my life for the better."})

print(result)