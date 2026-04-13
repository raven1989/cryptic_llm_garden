---
tags:
  - web-agent
  - code-review
  - implementation
aliases:
  - Taxy.ai
date: 2026-04-09
sources:
  - raw/web_agent/taxyai/
---

# Taxy.ai Implementation Deep Dive

**Taxy.ai** is an open-source Chrome extension that implements a SOTA [[DOM State Compression]] architecture. This document outlines how it extracts the page state and how it executes cloud-generated LLM commands. 

The source code analyzed is located in `raw/web_agent/taxyai/`.

## 1. State Extraction: "The Perception Phase"

Web pages are inherently noisy. An element might exist in the HTML but be completely hidden from the user (`opacity: 0`, `display: none`). Taxy.ai solves this by using a two-pass extraction process running inside the browser context:

### Pass A: Annotation (`src/pages/Content/getAnnotatedDOM.ts`)
1. The script traverses the live DOM.
2. For *every* node, it calls `window.getComputedStyle()`.
3. It checks for actual visibility and interactivity (e.g., is it an `<input>`, does it have an `onClick` handler, or does the computed style have `cursor: pointer`?).
4. Interactive elements are stored in a JavaScript array.
5. The live DOM nodes are tagged with a unique index (e.g., `data-id="42"`).

### Pass B: Simplification (`src/helpers/simplifyDom.ts`)
1. The annotated DOM is cloned.
2. The script prunes the clone aggressively. Invisible nodes are deleted.
3. Structural wrappers (like empty `<div>` tags) without text or interaction are removed, flattening the tree.
4. All CSS and styling attributes are stripped. Only semantic attributes (`aria-label`, `placeholder`, `role`) are preserved.

**Result**: A 5MB messy web page is compressed into a tiny snippet of pseudo-HTML where interactive targets look like `<button id="42">Submit</button>`.

## 2. LLM Reasoning: "The Brain Phase"

The tiny compressed snippet is sent to a cloud LLM (like GPT-4). Because all layout noise is gone, the LLM can easily reason about the layout. Based on the user's prompt, it returns a strictly formatted JSON action, such as:
`{"action": "click", "elementId": 42}`

## 3. Action Execution: "The Motor Phase"

This is arguably the most crucial engineering detail in Taxy.ai (`src/helpers/domActions.ts`). 

Modern web apps (React, Vue) and anti-bot systems often ignore simple JavaScript `element.click()` calls because the event lacks the `isTrusted: true` hardware flag. Taxy.ai circumvents this entirely by simulating physical hardware interactions:

1. **Mapping:** It takes the `elementId` (`42`) and looks up the corresponding live HTML element stored during Pass A.
2. **Tagging:** It injects a unique CSS attribute (`[data-taxy-id="xyz"]`) onto that live element.
3. **CDP (Chrome DevTools Protocol):** It uses the `chrome.debugger` API to ask the browser for the literal `(X, Y)` pixel coordinates of the center of that tagged element (`DOM.getBoxModel`).
4. **Hardware Simulation:** It fires raw `Input.dispatchMouseEvent` and `Input.dispatchKeyEvent` events via the debugger directly to those coordinates. 

Because these events are dispatched at the protocol level, the webpage receives perfect, unblockable hardware-level mouse clicks and keystrokes, identical to a human user.