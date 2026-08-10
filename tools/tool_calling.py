from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()


@tool
def subtraction(a: int, b: int) -> int:
    """Subtract b from a."""
    return a - b


groq_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2
)

llm_with_tools = groq_llm.bind_tools([subtraction])


messages = [
    HumanMessage(content="Subtract 10 from 15")
]


ai_message = llm_with_tools.invoke(messages)

for tool_call in ai_message.tool_calls:

    if tool_call["name"] == "subtraction":

        result = subtraction.invoke(tool_call["args"])

        messages.append(ai_message)

        messages.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"]
            )
        )

final_response = llm_with_tools.invoke(messages)

print(final_response.content)