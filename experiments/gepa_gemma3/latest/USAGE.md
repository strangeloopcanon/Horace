# Using the Evolved Prompt

- System/Instruction prompt: see `optimized_prompt.txt`.
- Input format: the model receives an `instruction` and a `context` string.
- Output format: return a JSON object with a single key `answer`, e.g. `{"answer": "..."}`.
- Word count: keep within the task's specified window (typically 180–210 words).
- Motifs: include all motifs verbatim from the context.