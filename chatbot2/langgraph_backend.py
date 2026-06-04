import truststore

truststore.inject_into_ssl()

from dotenv import load_dotenv

load_dotenv()

import os

os.environ["LANGCHAIN_PROJECT"] = "Agentic AI Chatbot"

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)

############################# TOOLS #############################
search_tool = DuckDuckGoSearchRun(regin="us-en")


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """Performs basic arithmetic operations on two numbers. Supported operations are: add, subtract, multiply, divide."""
    if operation == "add":
        result = first_num + second_num
    elif operation == "subtract":
        result = first_num - second_num
    elif operation == "multiply":
        result = first_num * second_num
    elif operation == "divide":
        if second_num == 0:
            return {"error": "Division by zero"}
        result = first_num / second_num
    else:
        return {"error": "Error: Invalid operation"}

    return {
        "first_num": first_num,
        "second_num": second_num,
        "operation": operation,
        "result": result,
    }


tools = [search_tool, calculator]
llm_with_tools = llm.bind_tools(tools)


############################## State and Grapg Defination ##############################
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


############################## Graph Nodes ###############################
def chat_node(state: ChatState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


tool_node = ToolNode(tools)


############################### SQLITE CONNECTION #############################
connection = sqlite3.connect("chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=connection)
graph = StateGraph(ChatState)


################################## NODES AND EDGES #############################
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)


def fetch_thread_ids():
    threads = set()
    for checkpoint in checkpointer.list(None):
        threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(threads)
