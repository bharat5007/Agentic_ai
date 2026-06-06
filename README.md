# Agentic AI

A hands-on learning repository for building agentic AI systems with **LangGraph**, **LangChain**, and related tooling. Covers everything from basic graph workflows to production-ready chatbots with memory, RAG, MCP, and human-in-the-loop patterns.

---

## Structure

```
agentic_ai/
├── workflow_basic.ipynb        # Intro to LangGraph nodes and edges
├── conditional_workflow.ipynb  # Conditional edges and branching logic
├── iterative_workflow.ipynb    # Loops and iterative graph execution
├── parallel_workflow.ipynb     # Fan-out / fan-in parallel node execution
├── sub_graphs.ipynb            # Composing parent and child subgraphs
├── human_in_loop.ipynb         # Human approval gate using interrupt()
├── chatbot.ipynb               # Stateful chatbot with InMemorySaver
│
├── chatbot2/
│   ├── langgraph_backend.py    # Chatbot backend: tools + SQLite checkpointing
│   ├── streamlit_frontend.py   # Streamlit UI for the chatbot
│   ├── chatbot_with_mcp.py     # Chatbot with MCP tool servers (stdio + HTTP)
│   ├── chatbot_with_rag.ipynb  # RAG chatbot over a PDF using FAISS + HuggingFace
│   ├── short_term_memory.ipynb # Thread-scoped memory: InMemorySaver & PostgresSaver
│   ├── long_term_memory.ipynb  # Cross-thread persistent memory patterns
│   └── islr.pdf                # Reference PDF used in RAG examples
│
└── langsmith/
    ├── simple_llm.py           # Basic LLM call with LangSmith tracing
    ├── sequential_chain.py     # Sequential LangChain chains
    ├── rag_v1.py → rag_v4.py   # Iterative RAG implementations
    └── ai_agent.py             # ReAct agent with web search + weather tools
```

---

## Topics Covered

| Topic | File(s) |
|---|---|
| Basic graph workflows | `workflow_basic.ipynb`, `conditional_workflow.ipynb` |
| Parallel & iterative execution | `parallel_workflow.ipynb`, `iterative_workflow.ipynb` |
| Subgraphs | `sub_graphs.ipynb` |
| Stateful chatbot with memory | `chatbot.ipynb`, `chatbot2/short_term_memory.ipynb` |
| Long-term memory | `chatbot2/long_term_memory.ipynb` |
| Postgres checkpointing | `chatbot2/short_term_memory.ipynb` |
| Tool use (search + calculator) | `chatbot2/langgraph_backend.py` |
| Streamlit UI | `chatbot2/streamlit_frontend.py` |
| MCP tool servers | `chatbot2/chatbot_with_mcp.py` |
| RAG over PDF | `chatbot2/chatbot_with_rag.ipynb` |
| Human-in-the-loop | `human_in_loop.ipynb` |
| LangSmith tracing | `langsmith/` |
| ReAct agent | `langsmith/ai_agent.py` |

---

## Setup

### Prerequisites

- Python 3.12
- PostgreSQL (for Postgres checkpointing examples)

### Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # or install from notebooks as needed
```

### Environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=...
OPENAI_API_KEY=...          # langsmith/ai_agent.py only
LANGCHAIN_API_KEY=...       # LangSmith tracing
LANGCHAIN_TRACING_V2=true
```

### Postgres setup (short_term_memory.ipynb)

```bash
psql -U <your-pg-user> -c "CREATE ROLE agentic_ai WITH SUPERUSER LOGIN PASSWORD 'agentic_ai';"
psql -U <your-pg-user> -c "CREATE DATABASE agentic_ai OWNER agentic_ai;"
```

Then use this connection string in the notebook:

```python
DB_URI = "postgresql://agentic_ai:agentic_ai@localhost:5432/agentic_ai"
```

### Run the Streamlit chatbot

```bash
cd chatbot2
streamlit run streamlit_frontend.py
```

---

## Tech Stack

- [LangGraph](https://github.com/langchain-ai/langgraph) — graph-based agent orchestration
- [LangChain](https://github.com/langchain-ai/langchain) — LLM tooling and chains
- [Groq](https://groq.com/) — LLM inference (llama-3.3-70b-versatile)
- [FAISS](https://github.com/facebookresearch/faiss) — vector store for RAG
- [HuggingFace Embeddings](https://huggingface.co/) — local sentence embeddings
- [LangSmith](https://smith.langchain.com/) — tracing and observability
- [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) — external tool servers
- [Streamlit](https://streamlit.io/) — chatbot frontend
- SQLite / PostgreSQL — conversation checkpointing
