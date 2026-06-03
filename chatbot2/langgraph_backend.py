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

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}

connection = sqlite3.connect('chatbot.db', check_same_thread=False)

checkpointer = SqliteSaver(conn=connection)
graph = StateGraph(ChatState)

graph.add_node('chat_node', chat_node)

graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer=checkpointer)

def fetch_thread_ids():
    threads = set()
    for checkpoint in checkpointer.list(None):
        threads.add(checkpoint.config['configurable']['thread_id'])
    return list(threads)