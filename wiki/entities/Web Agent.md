---
tags:
  - web-agent
  - ai
aliases:
  - Web Agents
date: 2026-04-09
sources: ["[[raw/web_agent/web_agent_abstract.md]]"]
---

# Web Agent

A **Web Agent** is a specialized type of [[Agents|AI Agent]] designed to navigate, interpret, and manipulate web interfaces autonomously or semi-autonomously. 

Unlike traditional web scrapers (like simple Python scripts or Puppeteer routines), a Web Agent relies on an LLM to "see" the page (usually by ingesting the DOM or a compressed version of it), reason about the user's intent, and execute discrete actions like clicking buttons or filling out forms.

## Key Architectures

According to recent research on [[Web Agent Architecture]], the most robust architecture for Web Agents today is a **Client-Cloud Collaborative** model. This avoids sending the entire `document.outerHTML` (which is bulky and privacy-invasive) or running massive LLMs locally on the client.

Instead, a lightweight script (like an injected `agent.js`) extracts only interactive elements from the DOM into a tiny JSON object—a process known as [[DOM State Compression]]. This compressed state is sent to a powerful cloud LLM (e.g., GPT-4 or Claude), which returns structured instructions (like `{"action": "click", "id": 1}`). The frontend script then executes the command, typically utilizing [[Action Execution - JS vs CDP|CDP to bypass anti-bot protections]].

## Notable Implementations
- **[[Browser-Use Implementation|Browser-Use]]**: A highly capable Python framework utilizing pure Chrome DevTools Protocol (`cdp_use`) for stealthy DOM snapshots.
- **[[Taxy.ai Implementation|Taxy.ai]]**: A Chrome extension running this paradigm entirely within the browser UI.
- **Natbot**: A foundational, minimalist implementation by Nat Friedman.