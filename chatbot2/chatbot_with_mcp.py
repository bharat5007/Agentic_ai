import truststore
from dotenv import load_dotenv
import os


from langgraph.graph import StateGraph, START
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import asyncio
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

truststore.inject_into_ssl()
os.environ["LANGCHAIN_PROJECT"] = "Agentic AI Chatbot"
load_dotenv()


llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)

############################## TOOLS ##############################
client = MultiServerMCPClient(
    {
        "arith": {
            "transport": "stdio",
            "command": "python3",
            "args": ["/Users/bharat/Desktop/mcp-math-server/main.py"],
        },
        "expense": {
            "transport": "streamable_http",
            "url": "https://splendid-gold-dingo.fastmcp.app/mcp",
        },
    }
)


############################## GRAPH STATE ##############################
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


async def build_graph():
    ############################## MCP TOOLS ##############################
    tools = await client.get_tools()
    llm_with_tools = llm.bind_tools(tools)

    ############################## Graph Nodes ##############################
    async def chat_node(state: ChatState):
        messages = state["messages"]
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}

    tool_node = ToolNode(tools)

    ############################## GRAPH INITIALIZATION ##############################
    graph = StateGraph(ChatState)

    ############################## NODES AND EDGES ##############################
    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")

    chatbot = graph.compile()
    return chatbot


async def main():
    chatbot = await build_graph()
    query = "Find the modulus of 12321 and give answer like a cricket commentator"
    result = await chatbot.ainvoke({"messages": [HumanMessage(content=query)]})
    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
