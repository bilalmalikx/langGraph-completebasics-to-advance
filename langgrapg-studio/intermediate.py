import os
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage

# ------------------- ENV LOAD -------------------
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") or ""

# ------------------- Custom State -------------------
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# ------------------- LLM -------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ------------------- Graph Creation -------------------
def make_default_graph():
    graph_workflow = StateGraph(State)  # Pass State type

    # Simple node: takes messages from state, returns new AI message
    def callmodel(state: State):
        return {"messages": [llm.invoke(state["messages"])]}

    # Add nodes
    graph_workflow.add_node("agent", callmodel)

    # Edges: START -> agent -> END
    graph_workflow.add_edge(START, "agent")
    graph_workflow.add_edge("agent", END)

    # Compile and return
    return graph_workflow.compile()

# ------------------- Run -------------------
agent = make_default_graph()
