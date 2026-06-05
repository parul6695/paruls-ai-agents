# Parul's AI Agents

**Agentic AI engineering portfolio** — design, build, and deploy autonomous multi-agent systems end to end.

> Production-oriented stack: orchestration, tool use, memory, and real-world integrations — not just prompt demos.

---

## Tech stack

| Layer | Tools |
|-------|--------|
| **Orchestration** | OpenAI Agents SDK · CrewAI · LangGraph · AutoGen |
| **Models** | OpenAI · Anthropic · Google Gemini |
| **Interfaces** | Gradio · Jupyter |
| **Integrations** | MCP (Model Context Protocol) · HTTP APIs · PDF/RAG |
| **Runtime** | Python 3.12 · [uv](https://docs.astral.sh/uv/) |

---

## Featured projects

_Add links and one-line descriptions as you ship work — recruiters skim this section first._

| Project | Stack | Description |
|---------|-------|-------------|
| _Coming soon_ | — | Your first agent build goes here |

---

## Repository layout

| Directory | Purpose |
|-----------|---------|
| `1_foundations/` | Core patterns, notebooks, Gradio prototypes |
| `2_openai/` | OpenAI Agents SDK — tools, handoffs, guardrails |
| `3_crew/` | CrewAI multi-agent crews |
| `4_langgraph/` | Stateful graphs, checkpoints, human-in-the-loop |
| `5_autogen/` | AutoGen agent teams |
| `6_mcp/` | Custom MCP servers and tool wiring |
| `contributions/` | Standalone showcase projects |
| `me/` | Profile context for personal / career agents |

---

## Quick start

```bash
cd paruls-ai-agents
uv sync
cp .env.example .env   # add API keys
```

Open `1_foundations/getting_started.ipynb` in Cursor, select kernel **`.venv (Python 3.12.x)`**, and run all cells.

Optional (CrewAI):

```bash
uv tool install crewai==0.130.0 --python 3.12
```

---

## Publish to GitHub

```bash
git add .
git commit -m "Initial commit: Parul's AI Agents portfolio"
gh repo create paruls-ai-agents --public --description "Agentic AI portfolio — multi-agent systems with OpenAI SDK, CrewAI, LangGraph, AutoGen, MCP" --source=. --push
```

Pin the repo on your GitHub profile and link it from your resume and LinkedIn.

---

## About

**Parul** — builder of autonomous AI agents.  
_Update this section with your headline, LinkedIn, and email._
