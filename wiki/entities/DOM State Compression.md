---
tags:
  - web-agent
  - architecture
aliases:
  - UI State Compression
date: 2026-04-09
sources: ["[[raw/web_agent/web_agent_abstract.md]]"]
---

# DOM State Compression

**DOM State Compression** (often referred to as UI State Compression) is a key architectural technique used by a [[Web Agent]] to interact efficiently and privately with web interfaces.

## The Problem
When sending web pages to Large Language Models (LLMs), the raw HTML (`document.outerHTML`) is often massive (megabytes in size), contains thousands of useless styling nodes, and carries highly sensitive user data. This creates context window overflows, high latency, and severe privacy risks.

## The Solution
Instead of sending the raw HTML, a lightweight JavaScript script (`agent.js`) traverses the Document Object Model (DOM) and strips out everything that isn't functionally interactive or informative. 

The compression process typically involves:
1. **Filtering Non-Interactables**: Removing `div` wrappers, inline styles, CSS, and hidden elements.
2. **Identifying Actionable Targets**: Locating buttons, input fields, dropdowns, and anchor tags (often looking for visual cues like `cursor: pointer` or accessibility attributes).
3. **Data Masking**: Anonymizing or hashing sensitive text before transmission.
4. **JSON Construction**: Compiling the remaining critical elements into a heavily condensed JSON array (e.g., `[{"id": 1, "type": "btn", "text": "Submit"}]`), drastically reducing the payload (often from 5MB down to 5KB).

This condensed JSON is then sent to a cloud-hosted LLM (like GPT-4 or Claude). The LLM processes the state, determines the next logical action based on the user's prompt, and returns a machine-readable command (e.g., `{"action": "click", "id": 1}`) that the frontend script subsequently executes via [[Action Execution - JS vs CDP|CDP hardware simulation]].

## Real-World Examples
Frameworks like **[[Browser-Use Implementation|Browser-Use]]** (now via pure CDP snapshotting), **[[Taxy.ai Implementation|Taxy.ai]]**, and **Natbot** all rely heavily on variations of this state compression to achieve state-of-the-art Web Agent performance.