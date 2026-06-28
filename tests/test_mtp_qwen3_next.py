# Copyright © 2025 Apple Inc.

import importlib
import unittest

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_lm.generate import generate_step, mtp_generate_step
from mlx_lm.models.cache import make_prompt_cache

_TINY_CONFIG = {
    "model_type": "qwen3_next",
    "hidden_size": 64,
    "num_hidden_layers": 4,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "linear_num_value_heads": 4,
    "linear_num_key_heads": 2,
    "linear_key_head_dim": 16,
    "linear_value_head_dim": 16,
    "linear_conv_kernel_dim": 4,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "decoder_sparse_step": 1,
    "shared_expert_intermediate_size": 64,
    "mlp_only_layers": [],
    "moe_intermediate_size": 64,
    "rms_norm_eps": 1e-6,
    "vocab_size": 256,
    "rope_theta": 1000.0,
    "partial_rotary_factor": 0.25,
    "max_position_embeddings": 128,
    # full_attention_interval=2 gives a mix of GatedDeltaNet (linear)
    # and full-attention layers, exercising the SSM rollback path.
    "full_attention_interval": 2,
    "tie_word_embeddings": True,
    "mtp_num_hidden_layers": 1,
}


def _make_qwen3_next_mtp_model(**overrides):
    """Create a tiny Qwen3-Next model with a native MTP head for testing."""
    module = importlib.import_module("mlx_lm.models.qwen3_next")
    args = module.ModelArgs.from_dict({**_TINY_CONFIG, **overrides})
    model = module.Model(args)
    model.set_dtype(mx.float32)
    mx.eval(model.parameters())
    return model


class TestQwen3NextMTP(unittest.TestCase):
    """Native MTP (Multi-Token Prediction) speculative decoding for Qwen3-Next.

    Uses a tiny synthetic model (4 layers, hidden=64, vocab=256) with
    mtp_num_hidden_layers=1 and full_attention_interval=2, mixing GatedDeltaNet
    (recurrent) and full-attention layers. Because speculative decoding is
    weight-agnostic, the greedy-identity test below is a strong end-to-end
    correctness check that does not require the real 80B checkpoint.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = _make_qwen3_next_mtp_model()

    def test_mtp_module_exists(self):
        self.assertTrue(hasattr(self.model, "mtp_forward"))
        self.assertTrue(hasattr(self.model, "make_mtp_cache"))
        self.assertTrue(hasattr(self.model, "mtp"))
        self.assertEqual(len(self.model.mtp.layers), 1)
        # The MTP block is a full-attention layer (matches the checkpoint).
        self.assertFalse(self.model.mtp.layers[0].is_linear)

    def test_make_mtp_cache(self):
        mtp_cache = self.model.make_mtp_cache()
        self.assertEqual(len(mtp_cache), 1)
        self.assertTrue(mtp_cache[0].is_trimmable())

    def test_return_hidden_shapes(self):
        inputs = mx.array([[0, 1, 2]])
        cache = make_prompt_cache(self.model)
        out, hidden = self.model(inputs, cache=cache, return_hidden=True)
        self.assertEqual(out.shape, (1, 3, 256))
        self.assertEqual(hidden.shape, (1, 3, 64))

    def test_hidden_is_pre_norm(self):
        """return_hidden must return the PRE-norm backbone hidden state."""
        inputs = mx.array([[0, 1, 2]])
        cache = make_prompt_cache(self.model)
        _, hidden = self.model(inputs, cache=cache, return_hidden=True)
        normed = self.model.model.norm(hidden)
        self.assertFalse(mx.allclose(hidden, normed, atol=1e-5).item())

    def test_mtp_forward_shape(self):
        hidden = mx.random.normal((1, 1, 64))
        next_ids = mx.array([[5]])
        mtp_cache = self.model.make_mtp_cache()
        logits = self.model.mtp_forward(hidden, next_ids, mtp_cache)
        self.assertEqual(logits.shape, (1, 1, 256))

    def test_mtp_generate_identity(self):
        """The most important test: greedy MTP decoding must reproduce the exact
        token sequence of standard generation. Any bug in the draft/verify loop,
        the GatedDeltaNet state rollback, or the MTP cache alignment would make
        the two diverge."""
        prompt = mx.array([0, 1, 2, 3], dtype=mx.uint32)
        n_tokens = 16

        std_cache = make_prompt_cache(self.model)
        std_tokens = []
        for i, (tok, _) in enumerate(
            generate_step(prompt, self.model, prompt_cache=std_cache)
        ):
            std_tokens.append(int(tok))
            if i + 1 >= n_tokens:
                break

        mtp_tokens = []
        for tok, _, _ in mtp_generate_step(prompt, self.model, max_tokens=n_tokens):
            mtp_tokens.append(int(tok))
            if len(mtp_tokens) >= n_tokens:
                break

        self.assertEqual(
            std_tokens,
            mtp_tokens,
            f"Token mismatch:\n std={std_tokens}\n mtp={mtp_tokens}",
        )

    def test_mtp_generate_identity_with_logits_processor(self):
        """Greedy identity must also hold under a context-sensitive logits
        processor, which stresses prev_tokens bookkeeping across the verify and
        draft passes."""
        prompt = mx.array([0, 1, 2, 3], dtype=mx.uint32)
        n_tokens = 16

        def context_processor(tokens, logits):
            if tokens is None or tokens.size == 0:
                return logits
            target = (int(tokens[-1].item()) + 1) % logits.shape[-1]
            boost = mx.zeros(logits.shape[-1])
            return logits + boost.at[target].add(10.0)

        std_cache = make_prompt_cache(self.model)
        std_tokens = []
        for i, (tok, _) in enumerate(
            generate_step(
                prompt,
                self.model,
                prompt_cache=std_cache,
                logits_processors=[context_processor],
            )
        ):
            std_tokens.append(int(tok))
            if i + 1 >= n_tokens:
                break

        mtp_tokens = []
        for tok, _, _ in mtp_generate_step(
            prompt,
            self.model,
            max_tokens=n_tokens,
            logits_processors=[context_processor],
        ):
            mtp_tokens.append(int(tok))
            if len(mtp_tokens) >= n_tokens:
                break

        self.assertEqual(std_tokens, mtp_tokens)

    def test_mtp_probabilistic_acceptance_completes(self):
        """The temp>0 probabilistic-acceptance + residual-sampling path must run
        to completion (with and without filters)."""
        prompt = mx.array([0, 1, 2, 3], dtype=mx.uint32)
        n_tokens = 16
        for kwargs in [{"temp": 0.7}, {"temp": 0.7, "top_k": 8}]:
            tokens = []
            for tok, _, _ in mtp_generate_step(
                prompt, self.model, max_tokens=n_tokens, **kwargs
            ):
                tokens.append(int(tok))
                if len(tokens) >= n_tokens:
                    break
            self.assertEqual(len(tokens), n_tokens, f"kwargs={kwargs}")


class TestQwen3NextFusedGateUp(unittest.TestCase):
    """Opt-in SwitchGLU gate/up fusion (mlx-lm#956). Fusing the two routed-expert
    projections into one gathered matmul must be numerically identical: the same
    weights, just concatenated along the output axis and split after the matmul."""

    def test_fused_module_layout(self):
        fused = _make_qwen3_next_mtp_model(fuse_gate_up=True)
        moe = fused.model.layers[0].mlp.switch_mlp
        self.assertTrue(moe.fuse_gate_up)
        self.assertTrue(hasattr(moe, "gate_up_proj"))
        self.assertFalse(hasattr(moe, "gate_proj"))
        # 2x the hidden dim packed into the single fused projection.
        self.assertEqual(moe.gate_up_proj.output_dims, 2 * moe.down_proj.input_dims)

    def test_sanitize_concatenates_gate_up(self):
        fused = _make_qwen3_next_mtp_model(fuse_gate_up=True)
        unfused = _make_qwen3_next_mtp_model()
        weights = dict(tree_flatten(unfused.parameters()))
        sanitized = fused.sanitize({k: v for k, v in weights.items()})
        # Backbone + the MTP layer both get fused; no separate gate/up survive.
        self.assertTrue(any(".switch_mlp.gate_up_proj.weight" in k for k in sanitized))
        self.assertTrue(
            any(
                "mtp.layers.0.mlp.switch_mlp.gate_up_proj.weight" in k
                for k in sanitized
            )
        )
        self.assertFalse(any(".switch_mlp.gate_proj.weight" in k for k in sanitized))

    def test_fused_is_numerically_identical(self):
        """Load identical weights into an unfused and a fused model; backbone and
        MTP-head logits must match bit-for-bit (concatenate-then-split is exact)."""
        unfused = _make_qwen3_next_mtp_model()
        weights = dict(tree_flatten(unfused.parameters()))

        fused = _make_qwen3_next_mtp_model(fuse_gate_up=True)
        fused.load_weights(list(fused.sanitize({**weights}).items()))
        mx.eval(fused.parameters())

        inputs = mx.array([[0, 1, 2, 3, 4, 5, 6, 7]])
        self.assertTrue(
            mx.array_equal(unfused(inputs), fused(inputs)).item(),
            "fused backbone logits differ from unfused",
        )

        hidden = mx.random.normal((1, 1, 64))
        next_ids = mx.array([[5]])
        lu = unfused.mtp_forward(hidden, next_ids, unfused.make_mtp_cache())
        lf = fused.mtp_forward(hidden, next_ids, fused.make_mtp_cache())
        self.assertTrue(
            mx.array_equal(lu, lf).item(), "fused MTP logits differ from unfused"
        )


if __name__ == "__main__":
    unittest.main()
