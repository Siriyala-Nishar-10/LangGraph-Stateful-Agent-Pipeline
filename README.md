# LangGraph Stateful Agent Pipeline

Cynaris Solutions AI/ML Internship, by Siriyala Nishar.

A stateful multi-step agent pipeline built with LangGraph, featuring conditional branching, memory, and human-in-the-loop checkpoints for business workflow automation.

## Week 1 Day 1: Setup Notes

### Environment

| Item                | Value           |
| ------------------- | --------------- |
| OS                  | Windows `11`    |
| Python              | `Python 3.14.7` |
| Editor              | VS Code         |
| Virtual environment | `.venv`         |

### Setup Commands

```powershell
mkdir ai-ml-internship
cd ai-ml-internship
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install langgraph langchain-core langchain-openai
python -c "import langgraph; print('LangGraph ready')"
```

### Installed Versions

| Package          | Version  |
| ---------------- | -------- |
| langgraph        | `1.2.16` |
| langchain-core   | `1.6.12` |
| langchain-openai | `1.7.2`  |

### Verification

Running the import check printed:

```
LangGraph ready
```

Screenshot of the running setup: `docs/day1_running.png`

![Day 1 setup running](docs/day1_running.png)

## Codebase Structure (W1D2)

Explored the installed `langgraph` package (v1.2.16, Python 3.14), located at
`.venv/Lib/site-packages/langgraph`.

### Folder structure

- `graph/`: defines `StateGraph`, nodes, edges and conditional branching
- `pregel/`: the execution engine that runs a compiled graph step by step
- `channels/`: state channels that store and merge values between nodes
- `checkpoint/`: saving and restoring graph state (persistence and memory)
- `store/`: long-term key-value memory shared across threads
- `prebuilt/`: ready-made components such as ReAct agents and tool nodes
- `cache/`: caching of node results
- `stream/`: streaming of output while the graph runs
- `func/`: functional API (`@entrypoint` / `@task`) as an alternative to graphs
- `managed/`: managed values such as `is_last_step`
- `utils/`: helpers for configs and runnables
- `_internal/`: private helpers (caching, retries, timeouts, serialization)

### Entry point

`langgraph/graph/__init__.py` exposes `StateGraph`, which is defined in
`graph/state.py`. A library has no `main.py`, so this is the API entry point.

### Key modules

1. `graph/state.py`: the `StateGraph` class used to build a pipeline
2. `pregel/main.py`: the `Pregel` runtime that executes the compiled graph
3. `checkpoint/`: persistence that lets a graph pause and resume (needed for
   human-in-the-loop in this project)

### Configuration

LangGraph has no config file. It reads environment variables instead:

- `LANGGRAPH_DEFAULT_RECURSION_LIMIT` (`_internal/_config.py`): max steps per run
- `LANGGRAPH_STRICT_MSGPACK` (`checkpoint/serde/_msgpack.py`): stricter checkpoint
  deserialization
- `LANGGRAPH_AES_KEY` (`checkpoint/serde/encrypted.py`): key for encrypted checkpoints
- `LANGGRAPH_API_KEY`, `LANGSMITH_API_KEY`, `LANGCHAIN_API_KEY` (`pregel/remote.py`):
  authentication for remote graphs

In this project, API keys such as `OPENAI_API_KEY` go in a `.env` file, which is
git-ignored.

## Data Flow (W1D3)

### Input

A Python dict matching the graph's `State` schema, e.g. `{"question": "what is LangGraph?"}`.

### Processing

`StateGraph.compile()` builds a runnable graph. When it runs, each node receives
the current state and returns a partial update. LangGraph merges that update into
the state (via channels) and then follows the edges to the next node.

### Output

The final state dict, e.g. `{"question": ..., "notes": ..., "answer": ...}`.

### Diagram

```
  input dict
  {"question": ...}
        |
        v
   +---------+
   |  START  |
   +---------+
        |
        v
   +----------+      returns {"notes": ...}
   | research | ------------------------+
   +----------+                          |
        |                                v
        |                      +------------------+
        |                      |   shared STATE   |  <-- persisted by
        |                      | question|notes|ans|      checkpointer
        |                      +------------------+
        v                                ^
   +-----------+     returns {"answer": ...}
   | summarise | ------------------------+
   +-----------+
        |
        v
   +---------+
   |   END   |
   +---------+
        |
        v
  final state dict
```

### Core concept: graph-based agent control flow

- **Nodes**: Python functions that read the state and return updates
- **Edges**: fixed transitions between nodes (`add_edge`)
- **Conditional routing**: `add_conditional_edges` picks the next node based on the
  state (used later for branching in this project)
- **Persistent state**: a checkpointer saves the state after each step, so a run can
  pause for human approval and resume

  ### Sample run (`python trace_flow.py`)

````
INPUT: {'question': 'what is LangGraph?'}
NODE UPDATE: {'research': {'notes': 'facts about what is LangGraph?'}}
NODE UPDATE: {'summarise': {'answer': 'summary of: facts about what is LangGraph?'}}
OUTPUT: {'question': 'what is LangGraph?', 'notes': 'facts about what is LangGraph?', 'answer': 'summary of: facts about what is LangGraph?'}
```git add README.md trace_flow.py
````

## Sample Data (W1D4)

`data/sample_dataset.csv` defines a research → summarise → critique loop as a graph.
Each row is one edge: the node, its next node, the condition for taking the edge,
a retry limit, and whether human approval is needed.

Loaded with pandas via `load_data.py`: 6 rows, 7 columns.

![Data loaded](docs/day4_data_loaded.png)

## Week 1 Output (W1D5)

This project runs a LangGraph StateGraph with three nodes (research, summarise, critique)
and a conditional edge that loops back to research until the critique score reaches 7.
It uses `data/sample_dataset.csv`, which defines the loop's steps, conditions and retry limit.
Running `python week1_run.py` prints the state at each step and saves the step log to
`results/week1_results.csv` (screenshot: `docs/week1_output.png`).

## Exploratory Data Analysis (W2D1)

`notebooks/week2_day1_eda.ipynb` explores `data/sample_dataset.csv`.

### EDA summary

- The dataset has 6 rows and 7 columns; each row is one edge of the research → summarise → critique loop.
- Only 3 columns are numeric or boolean (`step_id`, `max_retries`, `requires_approval`); the rest are text.
- `max_retries` ranges from 0 to 3 (mean 1.5); the critique steps have the highest value (3).
- The target `requires_approval` is imbalanced: 5 rows are False and 1 is True (only `send_report`).
- No column has missing values, so none is above the 5% threshold.

### Top 3 correlations

| Pair                                 | Correlation |
| ------------------------------------ | ----------- |
| `step_id` vs `requires_approval`     | 0.65        |
| `max_retries` vs `requires_approval` | -0.53       |
| `step_id` vs `max_retries`           | 0.19        |

### Limitation

With only 6 rows and 3 numeric columns, these correlations are not statistically
meaningful. The dataset will need to grow before modelling work in later days.
