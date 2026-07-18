from langchain_community.document_loaders import WebBaseLoader, PlaywrightURLLoader
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

url = "https://www.flipkart.com/nothing-phone-4a-black-128-gb/p/itm4d8bff10483ce?pid=MOBHZAWYHMRFGNFF&lid=LSTMOBHZAWYHMRFGNFFEN8HH5&pageUID=1784390192320"
loader = PlaywrightURLLoader(urls=[url])
docs  = loader.load()

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3
)

prompt = PromptTemplate(
    template = "give the answer for the question \n {question} \n based on the content of the webpage \n {text}",
    input_variables = ["question", "text"]
)

chain = prompt | model | StrOutputParser()
response = chain.invoke({"question" : "what is the product about", "text" : docs[0].page_content })
print(response)