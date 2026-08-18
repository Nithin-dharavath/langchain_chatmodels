from langchain_groq import ChatGroq
from langchain_core.tools import tool, InjectedToolArg
from typing import Annotated
from langchain_core.messages import HumanMessage, ToolMessage
from dotenv import load_dotenv
import requests

load_dotenv()


@tool
def get_conversion_factor(base_currency: str, targeted_currency: str) -> float:
    """
    Fetch the currency conversion factor between the base currency
    and target currency.
    """

    url = f"https://v6.exchangerate-api.com/v6/d4dd805f95bb631bdc8f981a/pair/{base_currency}/{targeted_currency}"

    response = requests.get(url)

    return response.json()["conversion_rate"]


@tool
def conversion_multiply(
    amount: float,
    conversion_rate: Annotated[float, InjectedToolArg]
) -> float:
    """Multiply the amount by the currency conversion factor."""

    return amount * conversion_rate


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.2
)

llm_with_tools = llm.bind_tools(
    [get_conversion_factor, conversion_multiply]
)


messages = [
    HumanMessage(
        content="What is the conversion factor between USD and INR? "
                "Based on that, convert the 10 USD to INR."
    )
]


# 1. Ask the LLM
aimessage = llm_with_tools.invoke(messages)

messages.append(aimessage)


# 2. Process the first tool call
for tool_call in aimessage.tool_calls:

    if tool_call["name"] == "get_conversion_factor":

        tool_message1 = get_conversion_factor.invoke(
            tool_call["args"]
        )

        conversion_rate = tool_message1

        messages.append(
            ToolMessage(
                content=str(tool_message1),
                tool_call_id=tool_call["id"]
            )
        )


# 3. Ask the LLM again with the conversion factor
aimessage = llm_with_tools.invoke(messages)

messages.append(aimessage)


# 4. Process the multiplication tool
for tool_call in aimessage.tool_calls:

    if tool_call["name"] == "conversion_multiply":

        tool_call["args"]["conversion_rate"] = conversion_rate

        tool_message2 = conversion_multiply.invoke(
            tool_call["args"]
        )

        messages.append(
            ToolMessage(
                content=str(tool_message2),
                tool_call_id=tool_call["id"]
            )
        )


# 5. Ask the LLM for the final answer
response = llm_with_tools.invoke(messages).content

print(response)