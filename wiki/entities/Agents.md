---
tags: [llm, workflow, automation]
date: 2026-04-08
aliases: [Agentic Workflows, Multi-Agent Systems]
---

# Agentic Workflows & Multi-Agent Systems

Agentic workflows transition LLMs from static text generators to computation engines capable of multi-step planning, tool use, and self-correction. They solve the fragility of zero-shot generation by providing the model with "Test-Time Compute" (time to think and iterate).

## Core Mechanisms

*   **State & Persistence:** APIs are stateless. Agents require Summary Memory (compressing history) and Checkpointing (saving the execution graph state to a database like SQLite).
*   **Human-in-the-loop:** The ability to pause execution, seek human approval, and resume from a saved checkpoint.
*   **Single-Agent Workflows:** Moving away from autonomous "black box" agents toward deterministic State Machines. Complex tasks are defined as Directed Acyclic Graphs (DAGs), where the LLM acts as a router between predefined nodes. Frameworks like LangGraph dominate this space.
*   **Reasoning Patterns:**
    *   **ReAct:** Loop of Reasoning + Action.
    *   **Plan-and-Solve:** Breaking a problem into sub-tasks before executing.
    *   **Reflexion:** Analyzing errors (e.g., from an API call) and retrying.

## Multi-Agent Systems

Simulating human organizational structures to tackle broader tasks. Agents have specific roles, System Prompts, and tool access.

*   **Topologies:** Sequential (pipelines), Hierarchical (manager to worker to reviewer), or Debate (optimizing through argument).
*   **Frameworks:** AutoGen (code-focused), CrewAI (role-based), Swarm (lightweight handoffs).

## Tool Calling & Sandboxes

The crucial bridge between the LLM and the real world.
*   **Schema Design:** Defining API inputs meticulously using JSON Schema or Pydantic.
*   **Error Handling:** Returning 400/500 errors back to the LLM as an `Observation` to trigger an automated retry loop.
*   **Secure Execution:** Operating code-generation agents within secure, fast-booting micro-VMs (e.g., E2B) to prevent catastrophic prompt injections on host machines.

See also: [[LLM Study Plan]]
