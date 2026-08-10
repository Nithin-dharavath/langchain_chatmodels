# #search tool web

from langchain_community.tools import DuckDuckGoSearchRun

search_tool = DuckDuckGoSearchRun()

result = search_tool.invoke("what is the latest news today in tech?")

print(result)

#shell tool _ system works cl

from langchain_community.tools import ShellTool

shell_tool = ShellTool()

system_result = shell_tool.invoke("dir")

print(system_result)

## custum tool using the @decarator ##

from langchain_core.tools import tool

#step - 1 define function
def addition(a, b):
    """add the a and b"""
    return (a+b)
#step-2 add type hinting
def addition(a:int, b:int) -> int:
    """add the a and b"""
    return (a+b)
#step-3 add the tool decorator
@tool
def addition(a:int, b:int) -> int:
    """add the a and b"""
    return (a+b)

result = addition.invoke({"a" : 50, "b" : 10})

print(addition.args_schema)

##method -2 :suing the structuredTool##
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

class additionInput(BaseModel):
    a : int = Field(required=True, description="the frist number to multiply")
    b : int = Field(required=True, description="add the second number")

def addition(a, b):
    return(a+b)

addition_tool = StructuredTool.from_function(
    func=addition,
    name="addition",
    description="add the both a and b",
    args_schema=additionInput
)

result = addition_tool.invoke({"a" : 3, "b" : 5})

print(result)


###method-3 using the basemodel##

class additionInput(BaseModel):
    a : int = Field(required=True, description="the frist number to multiply")
    b : int = Field(required=True, description="add the second number")

class additiontTool(BaseModel):
    name: str = "addition"
    description : str = "add the both a and b numbers"
    args_schema: Type(BaseModel) = additionInput

    def run(self, a: int, b: int) -> int:
        return a+b

addition_tool = additiontTool()

result = addition_tool.invoke({"a" : 3, "b" : 5})

##tool kit ##

class mathToolkit:
    def get_tools(self):
        return [addition, addition_tool]

tool_kit = mathToolkit()
tools = tool_kit.get_tools()

for tool in tools:
    print(tool.name, tool.description)
