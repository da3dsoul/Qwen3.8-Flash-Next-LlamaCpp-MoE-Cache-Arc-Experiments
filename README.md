# Qwen3.8-Flash-Next on Arc via llama.cpp MoE expert cache

Goal: run Alibaba's **Qwen3.8-Flash-Next** (~125B MoE params, ~6B active/token,
plus a separate ~51B-parameter n-gram/PLE lookup table) on an **Intel Arc Pro
B70 (32GB VRAM)** at usable speed, by porting a CUDA-only llama.cpp fork's
**MoE expert cache** (hot routed experts pinned in a VRAM slot pool, cold ones
in host RAM, LRU/frequency-decay eviction, device-side admission planning) to
llama.cpp's SYCL backend.

This is a *different* model and a *different* serving stack from this
account's other Qwen3.8 projects: `qwen38-27b-rtx3090` / `qwen38-27b-b70-vllm`
/ `Qwen3.8-vLLM-KVarN-MTP-Experiments` all serve the dense
`Qwen3.8-27B-W4A16-AutoRound` checkpoint on vLLM with the custom **KVarN**
attention/KV-cache patch. That investigation concluded the MoE expert cache
does **not** apply there — the served model is dense (no experts at all),
already fits fully in VRAM, and the technique conflicts with the MTP
speculative decoding that deployment depends on. Full writeup:
`../coding-agent/docs/06-moe-cache-research/`.

This project exists because the *reason* it didn't apply there — "our model
has no experts to cache" — doesn't hold for Qwen3.8-Flash-Next, which is
genuinely a large sparse MoE. On a 32GB card (vs. the 12GB RTX 3060 the
technique was demoed on) the hot-expert slot pool can be much larger, which
should mean a much higher cache hit rate and a correspondingly bigger win —
that's the bet this project is testing.

**Status: planning.** See `PLAN.md` for the phased implementation plan and
`docs/` for the technical grounding it's built on. Nothing has been
implemented yet.

## Layout

- `PLAN.md` — the phased build-out plan (read this first if you're picking
  this project up).
- `docs/00-background.md` — self-contained technical background: the expert
  cache mechanism, the model architecture, the target hardware, and the risks
  already known from a related project's driver-stack history.
- `docs/research/` — deep-dive research feeding the plan (SYCL backend
  feasibility, model support status on llama.cpp).
