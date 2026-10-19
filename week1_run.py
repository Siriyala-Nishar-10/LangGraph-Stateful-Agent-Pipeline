"""W1D5: run a LangGraph pipeline driven by the sample dataset, printing state at each step."""
import os
from typing import TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

df = pd.read_csv("data/sample_dataset.csv")
descriptions = dict(zip(df["node"], df["description"]))
MAX_LOOPS = int(df.loc[df["node"] == "critique", "max_retries"].max())


class State(TypedDict):
    question: str
    notes: str
    summary: str
    score: int
    loops: int


def research(state: State) -> dict:
    return {"notes": f"{descriptions['research']} (pass {state['loops'] + 1}): {state['question']}"}


def summarise(state: State) -> dict:
    return {"summary": f"{descriptions['summarise']}: {state['notes']}"}


def critique(state: State) -> dict:
    # Placeholder scorer: the summary improves by 2 points on each extra pass
    loops = state["loops"] + 1
    return {"score": 3 + 2 * loops, "loops": loops}


def route(state: State) -> str:
    if state["score"] >= 7 or state["loops"] >= MAX_LOOPS:
        return END
    return "research"


builder = StateGraph(State)
builder.add_node("research", research)
builder.add_node("summarise", summarise)
builder.add_node("critique", critique)
builder.add_edge(START, "research")
builder.add_edge("research", "summarise")
builder.add_edge("summarise", "critique")
builder.add_conditional_edges("critique", route, [END, "research"])
app = builder.compile()

if __name__ == "__main__":
    start = {"question": "What is LangGraph?", "notes": "", "summary": "", "score": 0, "loops": 0}
    rows = []
    for i, update in enumerate(app.stream(start, stream_mode="updates"), start=1):
        node, changes = next(iter(update.items()))
        print(f"Step {i} | node: {node} | update: {changes}")
        rows.append({"step": i, "node": node, "update": str(changes)})

    final = app.invoke(start)
    print("\nFINAL STATE:", final)

    os.makedirs("results", exist_ok=True)
    pd.DataFrame(rows).to_csv("results/week1_results.csv", index=False)
    print("Saved results/week1_results.csv")
