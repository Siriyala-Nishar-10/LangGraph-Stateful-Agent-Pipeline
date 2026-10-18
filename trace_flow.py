"""W1D3: trace how data flows through a minimal LangGraph pipeline."""
from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    question: str
    notes: str
    answer: str


def research(state: State) -> dict:
    return {"notes": f"facts about {state['question']}"}


def summarise(state: State) -> dict:
    return {"answer": f"summary of: {state['notes']}"}


builder = StateGraph(State)
builder.add_node("research", research)
builder.add_node("summarise", summarise)
builder.add_edge(START, "research")
builder.add_edge("research", "summarise")
builder.add_edge("summarise", END)
app = builder.compile()

if __name__ == "__main__":
    inputs = {"question": "what is LangGraph?"}
    print("INPUT:", inputs)
    for update in app.stream(inputs, stream_mode="updates"):
        print("NODE UPDATE:", update)
    print("OUTPUT:", app.invoke(inputs))
