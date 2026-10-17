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

## Next Steps

- W1D3: Map the data flow
