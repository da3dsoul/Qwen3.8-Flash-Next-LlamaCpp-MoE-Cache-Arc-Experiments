# Methodology notes (scratch — folded into ../long-context-120k-benchmark-report.md)

Prompt provenance: generated in place from the sibling project
`../Qwen3.8-vLLM-KVarN-MTP-Experiments/scripts/` (MIT, same author), run from
that project's own directory; nothing was copied into this repo except the
generated artifacts (`staging/work/bench120k/messy.py`, `full_prompt.txt`),
which are outputs, not sibling source.

```
cd ../Qwen3.8-vLLM-KVarN-MTP-Experiments/scripts
python3 generate_messy.py 153 > .../staging/work/bench120k/messy.py
python3 build_prompt.py    .../messy.py .../full_prompt.txt
```

Input characteristics (verified this pass):
- `messy.py`: 410,744 chars, 14,7xx lines, 1,229 top-level defs/classes
- 153 near-identical entity blocks (7 functions + 1 handler class each)
- planted anti-patterns: 153 bare `except:`, 154 mutable default args,
  459 `== None`, 153 `== True`, 460 `for i in range(len(...))`,
  153 `global` statements, 153 Python-2 `has_key` calls, manual `open()`
- `full_prompt.txt`: 411,558 chars

Tokenization (this project's own model, not the sibling's 27B):
`llama-tokenize -m .../UD-IQ3_XXS-00001-of-00003.gguf -f full_prompt.txt --ids`
-> **116,226 tokens** raw. The sibling reported 116,226 raw / 116,277 after
chat templating for `Qwen3.8-27B` — the two tokenizers agree exactly on this
input, so the sibling's size calibration carries over unchanged.
