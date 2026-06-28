"""
Benchmark native Multi-Token Prediction (MTP) self-speculative decoding.

MTP lets a model speculatively decode against itself using its built-in MTP
head -- no separate draft model. It is currently supported for Qwen3-Next.

Enabling MTP (the official Qwen3-Next config does not advertise the head):

    # convert from the original checkpoint with the MTP head preserved
    python -c "import json,pathlib; \
p=pathlib.Path('Qwen3-Next-80B-A3B-Instruct/config.json'); \
c=json.loads(p.read_text()); c['mtp_num_hidden_layers']=1; \
p.write_text(json.dumps(c, indent=2))"
    mlx_lm.convert --hf-path Qwen3-Next-80B-A3B-Instruct \
        --mlx-path Qwen3-Next-80B-A3B-Instruct-4bit-mtp -q --q-bits 4

Then benchmark baseline vs. MTP decode throughput (prefill excluded), verify
the greedy output is byte-identical to standard generation, and report the
fraction of tokens accepted from the MTP drafter:

    python benchmarks/mtp_benchmark.py --model Qwen3-Next-80B-A3B-Instruct-4bit-mtp

Measured on Apple M5 Max (128 GB), Qwen3-Next-80B-A3B-Instruct, 256 tokens,
greedy, decode-only (mean over code/prose/reasoning/list prompts):

    quant   baseline   MTP      speedup   accept   identical
    4-bit   101 tok/s  116 tok/s  1.15x    ~47%     yes
    8-bit    81 tok/s   94 tok/s  1.16x    ~48%     yes

The speedup is exact (lossless) but modest: Qwen3-Next ships a single MTP layer
(one draft token), and on Apple Silicon the 2-token verify pass costs more than
a 1-token decode (see ml-explore/mlx#3553), which caps the gain.

Optional: fused gate/up expert projections (ml-explore/mlx-lm#956)
-----------------------------------------------------------------
Qwen3-Next's MoE runs the routed experts' gate and up projections as two separate
gathered matmuls over the same input and indices. Setting ``fuse_gate_up: true``
(config, or ``--fuse-gate-up`` here) concatenates them at load time into one
gathered matmul -- one fewer kernel dispatch per MoE layer per token. It is
opt-in, token-exact, and runtime memory-neutral (the fused weight *replaces* the
two; only a ~1.7 GB transient appears during the load-time concatenate).

Measured on Apple M5 Max (128 GB), Qwen3-Next-80B-A3B-Instruct-4bit, decode-only:

    mode             unfused    fused    delta    identical
    plain decode     97 tok/s   106 tok/s  +8.5%   yes
    MTP decode      117 tok/s   119 tok/s  +1.5%   yes

The plain-decode gain matches #956's reported +8.6% on Qwen3-30B-A3B. Stacked on
MTP it adds only ~+1.5%: the MTP round already does more work per step (2-token
verify + draft head), so the per-dispatch saving is largely amortized. For max
absolute throughput, enable both (118.7 tok/s, 1.22x over the stock baseline).
"""

import argparse
import time

from mlx_lm.generate import stream_generate
from mlx_lm.utils import load

DEFAULT_PROMPTS = {
    "code": "Write a complete Python implementation of an LRU cache with get/put in O(1), including a short docstring and an example.",
    "prose": "Write three paragraphs about the history and cultural impact of the printing press.",
    "reasoning": "A train leaves city A at 60 mph. Another leaves city B, 300 miles away, at 40 mph toward A. Work through, step by step, when and where they meet.",
    "list": "List the first 30 prime numbers, one per line, with no commentary.",
}


def run(model, tokenizer, prompt, mtp, max_tokens):
    messages = [{"role": "user", "content": prompt}]
    ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    tokens, from_draft, t_first, gen_tps, peak = [], [], None, 0.0, 0.0
    for resp in stream_generate(
        model, tokenizer, ids, max_tokens=max_tokens, mtp=mtp, temp=0.0
    ):
        if t_first is None:
            t_first = time.perf_counter()
        tokens.append(resp.token)
        from_draft.append(resp.from_draft)
        gen_tps, peak = resp.generation_tps, resp.peak_memory
    t_end = time.perf_counter()
    decode_tps = (
        (len(tokens) - 1) / (t_end - t_first) if t_first and len(tokens) > 1 else 0.0
    )
    return {
        "tokens": tokens,
        "n": len(tokens),
        "decode_tps": decode_tps,
        "from_draft_frac": sum(from_draft) / max(len(from_draft), 1),
        "peak_gb": peak,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Path/repo of an MTP-enabled model.")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument(
        "--fuse-gate-up",
        action="store_true",
        help="Fuse routed-expert gate/up projections at load (mlx-lm#956).",
    )
    args = p.parse_args()

    print(f"[load] {args.model}" + (" [fuse_gate_up]" if args.fuse_gate_up else ""))
    model_config = {"fuse_gate_up": True} if args.fuse_gate_up else None
    model, tokenizer = load(args.model, model_config=model_config)
    if not hasattr(model, "mtp_forward"):
        raise SystemExit(
            "Model has no MTP head. Convert with mtp_num_hidden_layers=1 (see module docstring)."
        )

    # Warm up both code paths.
    for m in (False, True):
        for _ in stream_generate(
            model, tokenizer, tokenizer.encode("Hello"), max_tokens=4, mtp=m, temp=0.0
        ):
            pass

    print(
        f"\n{'prompt':<10} {'mode':<6} {'tok':>5} {'decode tok/s':>13} {'draft%':>7} {'match':>6}"
    )
    print("-" * 56)
    speedups, matches, peak = [], [], 0.0
    for name, prompt in DEFAULT_PROMPTS.items():
        base = run(model, tokenizer, prompt, False, args.max_tokens)
        mtp = run(model, tokenizer, prompt, True, args.max_tokens)
        k = min(base["n"], mtp["n"])
        match = base["tokens"][:k] == mtp["tokens"][:k]
        speedup = mtp["decode_tps"] / base["decode_tps"] if base["decode_tps"] else 0.0
        peak = max(peak, mtp["peak_gb"])
        speedups.append(speedup)
        matches.append(match)
        print(
            f"{name:<10} {'base':<6} {base['n']:>5} {base['decode_tps']:>13.1f} {'-':>7} {'-':>6}"
        )
        print(
            f"{name:<10} {'mtp':<6} {mtp['n']:>5} {mtp['decode_tps']:>13.1f} {mtp['from_draft_frac']*100:>6.0f}% {str(match):>6}  ({speedup:.2f}x)"
        )

    print("\n=== SUMMARY ===")
    print(f"mean decode speedup (MTP vs baseline): {sum(speedups)/len(speedups):.2f}x")
    print(f"all greedy outputs byte-identical to baseline: {all(matches)}")
    print(f"peak memory: {peak:.1f} GB")


if __name__ == "__main__":
    main()
