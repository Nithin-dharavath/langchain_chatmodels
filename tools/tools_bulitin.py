# # #search tool web

# from langchain_community.tools import DuckDuckGoSearchRun

# search_tool = DuckDuckGoSearchRun()

# result = search_tool.invoke("what is the latest news today in tech?")

# print(result)

# #shell tool _ system works cl

# from langchain_community.tools import ShellTool

# shell_tool = ShellTool()

# system_result = shell_tool.invoke("dir")

# print(system_result)


#custum tool 

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