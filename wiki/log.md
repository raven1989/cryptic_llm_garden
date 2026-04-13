# Operation Log

An append-only chronological record of what happened and when. Useful for tracking the evolution of the wiki.

## [2026-04-08] Initialization
Wiki structure and schema created.
## [2026-04-08] Ingest | LLM Algorithm Engineer Study Plan
Ingested four raw outline files (`raw/Outline.md`, `raw/Outline_Layer1.md`, `raw/Outline_Layer2.md`, `raw/Outline_Layer3.md`). Created a high-level study plan (`wiki/personal/LLM Study Plan.md`) and extracted key concepts into new entity pages: `[[Transformers]]`, `[[RAG]]`, `[[Agents]]`, `[[Fine-tuning]]`, and `[[Vector Database]]`. Updated the index.
## [2026-04-08] Query Synthesis | My LLM Learning Sequence
Synthesized an execution plan from a user query about the `[[LLM Study Plan]]`. Filed the answer back into the wiki as a new page (`wiki/personal/My LLM Learning Sequence.md`) and updated the index.

## [2026-04-09] Ingest | Web Agent Architecture
Ingested the raw brainstorm document (`raw/web_agent/web_agent_abstract.md`). Created a summary page at `wiki/research/Web Agent Architecture.md`. Extracted and created new entity pages for `[[Web Agent]]` and `[[DOM State Compression]]` in `wiki/entities/`. Updated `wiki/index.md` with new links.

## [2026-04-09] Ingest | Taxy.ai Codebase
Cloned and moved the Taxy.ai open-source repository to `raw/web_agent/taxyai/`. Analyzed the codebase and synthesized a new research page at `wiki/research/Taxy.ai Implementation.md` detailing its two-pass DOM extraction (`getAnnotatedDOM.ts`, `simplifyDom.ts`) and hardware-level action execution via CDP (`domActions.ts`). Updated `wiki/index.md` and added cross-links to existing entity pages.

## [2026-04-09] Ingest | Browser-Use Codebase
Cloned and moved the `browser-use/browser-use` repository to `raw/web_agent/browser-use/`. Analyzed its modernized architecture, discovering a shift away from JS injection toward pure Python Chrome DevTools Protocol (`cdp_use`) for stealthy DOM snapshotting and hardware-simulated mouse actions. Created `wiki/research/Browser-Use Implementation.md`. Updated the index and entity definitions accordingly.

## [2026-04-09] Synthesis | Action Execution (JS vs CDP)
Synthesized our conversation regarding the limitations of JavaScript execution in modern web environments into a new research page at `wiki/research/Action Execution - JS vs CDP.md`. Outlined the importance of `isTrusted` flags, React state management, and z-index visual overlays when building Web Agents. Updated `wiki/entities/Web Agent.md` and `wiki/entities/DOM State Compression.md` with links to this new concept.