# Phase 0 Decisions

**Status:** Static audit complete; no tests or notebooks executed.

1. The Markdown specification is controlling because the earlier named master-prompt DOCX is not present under the supplied Downloads path.
2. The source archive `isl-engine.zip` is the implementation baseline. It was restored because the attached working repository had no source or commits. Its nested Git metadata was not copied. The smaller debug ZIP is not a replacement.
3. Existing isolated modules are preserved and classified as candidates, not production-ready. No old metric, class count, test pass or live accuracy is accepted without original data/model/evaluation evidence.
4. Continuous translation is a separate product path and must not be approximated by concatenated isolated labels. It is unavailable until licensed paired data and a real trained model exist.
5. The current 170-value pose/hand model path remains a baseline. It omits face/non-manual input and quality confidence. A continuous RGB or multimodal comparison remains necessary before concluding landmarks are enough.
6. The attached architecture's CTC-English, RAG/agent, distributed service and named cloud-provider proposals are not adopted by default. Detailed resolutions are in `ARCHITECTURE_DECISION_RECORDS.md` and `IMPLEMENTATION_ROADMAP.md`.
7. Dataset use is gated by source-specific rights review and exact release reconciliation. No large download or external data scraping was performed.
8. The user explicitly asked to proceed through Phases 0–8, so implementation continues. Missing data, artifact and Colab authorization block actual model training/acceptance, not the architecture, adapters or safe engineering work.
9. RAM and Colab details remain unknown. Local Python 3.14.8 has no project dependencies; use an isolated supported version before execution.

## Phase 0 exit status

**Passed for repository and document inventory.** The source files, key entry points, dependency/deployment gaps, notebooks, architecture conflicts, environment constraints, data access state and implementation backlog are documented. Notebook restart, package setup, test execution, dataset decoding, model loading, training and deployment remain unverified and are not falsely marked as passed.
