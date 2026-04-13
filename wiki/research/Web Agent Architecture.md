---
tags:
  - web-agent
  - architecture
  - frontend
  - ai
aliases:
  - Web Agent Architecture
date: 2026-04-09
sources:
  - raw/web_agent/web_agent_abstract.md
---

# Web Agent Architecture: 4 Approaches

This page summarizes a brainstorming document detailing architectural designs for an AI-powered [[Web Agent]]. The goal is to allow a Large Language Model (LLM) to interact intelligently with web pages, balancing context windows, privacy, latency, and capabilities.

## The 4 Architectural Paradigms

### 1. Raw Full HTML Passing
- **Mechanism**: The frontend sends the entire `document.outerHTML` to a backend LLM to find targets and decide actions.
- **Pros**: Trivial frontend integration (one line of code).
- **Cons**: 
  - Massive context consumption (several MBs per request).
  - Infinite loops caused by self-referential scripts (the HTML includes the agent's own script).
  - Severe privacy violations (sending raw, authenticated DOM to a cloud server).

### 2. Headless Browser Scraper (Puppeteer)
- **Mechanism**: A backend server launches a headless browser (like Puppeteer/Playwright) to visit the target URL independently.
- **Pros**: Zero impact on user client performance, avoids self-reference.
- **Cons**: Cannot bypass authentication walls natively. It operates outside the user's authenticated session, seeing only public pages.

### 3. Pure Client-Side Edge Computing
- **Mechanism**: Local LLMs (e.g., Llama-3-8B) run directly in the browser via WebAssembly or WebGPU.
- **Pros**: Ultimate privacy (zero data leaves the client), zero network latency.
- **Cons**: Excessive hardware requirements, massive initial model download overhead (GBs), and smaller models generally lack the intelligence for complex DOM reasoning.

### 4. Client-Cloud Collaborative / [[DOM State Compression]] (SOTA)
- **Mechanism**: The optimal current solution. A lightweight frontend agent extracts only actionable elements (buttons, inputs) and filters out styling/invisibles. It compiles a compact JSON (e.g., 5KB) to send to a powerful cloud LLM (GPT-4/Claude), which responds with a discrete action (e.g., `{"action": "click", "id": 1}`). The frontend executes the action.
- **Pros**: Inherits user authentication naturally, ultra-low token usage, extremely fast, and highly privacy-preserving (sensitive data can be masked locally before transmission).

## Key Open-Source Implementations

Several projects showcase the SOTA [[DOM State Compression]] paradigm:

- **Browser-Use** (`browser-use/browser-use`): Currently one of the most popular Web Agent frameworks. While Python-based on the backend, it relies heavily on elegant JavaScript injection (`buildDomTree.js`) to parse and compress the DOM structure.
- **Taxy.ai** (`TaxyAI/browser-extension`): A Chrome extension showcasing pure frontend state extraction and execution architecture. It tags interactive elements with numeric labels for GPT-4 to interact with.
- **Natbot** (`nat/natbot`): Created by Nat Friedman. The original MVP demonstrating a minimalist approach: extract elements with `cursor: pointer` or `href`, strip HTML, and send raw text to the model.

## Implementation Advice
For production integration, inject an `agent.js` script into the frontend to generate the Action Space JSON. Send user queries and the JSON state to a backend LLM, receive the specific action payload, and execute the JS command (`document.getElementById().click()`) entirely on the client.