from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

class Facts(BaseModel):
    fact1: str = Field(description="Fact 1")
    fact2: str = Field(description="Fact 2")
    fact3: str = Field(description="Fact 3")


llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

parser = PydanticOutputParser(pydantic_object = Facts)

template = PromptTemplate(
    template="""Give 3 facts {topic}\n {format_instructions}""",
    input_variables=["topic"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

chain = template | model | parser
response = chain.invoke({"topic" : "instagram"})
print(response)