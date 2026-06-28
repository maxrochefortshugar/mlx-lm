# Copyright © 2025 Apple Inc.

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import sum_gradients

from .activations import swiglu
from .base import (BaseModelArgs, create_attention_mask, create_ssm_mask,
                   scaled_dot_product_attention)
from .cache import ArraysCache, KVCache
from .gated_delta import gated_delta_update
from .rope_utils import initialize_rope
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    linear_num_value_heads: int
    linear_num_key_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    num_experts: int
    num_experts_per_tok: int
    decoder_sparse_step: int
    shared_expert_intermediate_size: int
    mlp_only_layers: List[int]
    moe_intermediate_size: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int
    rope_theta: float
    partial_rotary_factor: float
    max_position_embeddings: int
    head_dim: int
    norm_topk_prob: bool = False
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    full_attention_interval: int = 4
    # Number of native Multi-Token Prediction (MTP) layers. The official
    # Qwen3-Next checkpoints ship exactly one MTP layer but do not advertise it
    # in config.json, so this defaults to 0 (MTP off, weights dropped at load,
    # preserving the behaviour of existing community conversions). Set it to 1
    # to build the MTP head and enable native self-speculative decoding.
    mtp_num_hidden_layers: int = 0
    # Fuse the routed experts' gate and up projections into a single gathered
    # matmul (one fewer gather per MoE layer per token). Opt-in and off by
    # default for back-compat with existing conversions; weights are fused at
    # load time in ``sanitize`` so no extra memory is held. See mlx-lm#956.
    fuse_gate_up: bool = False


@partial(mx.compile, shapeless=True)
def _precise_swiglu(h, gate, x):
    gate = nn.silu(gate.astype(mx.float32))
    x = x.astype(mx.float32)
    return (gate * x).astype(h.dtype)


class Qwen3NextRMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones(hidden_size)

    def __call__(
        self, hidden_states: mx.array, gate: mx.array | None = None
    ) -> mx.array:
        x = mx.fast.rms_norm(hidden_states, self.weight, self.eps)
        if gate is not None:
            return _precise_swiglu(hidden_states, gate, x)
        else:
            return x.astype(hidden_states.dtype)


class Qwen3NextAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_key_value_heads = args.num_key_value_heads
        self.num_attention_heads = args.num_attention_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            args.hidden_size,
            self.num_attention_heads * self.head_dim * 2,
            bias=args.attention_bias,
        )
        self.k_proj = nn.Linear(
            args.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.v_proj = nn.Linear(
            args.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim,
            args.hidden_size,
            bias=args.attention_bias,
        )

        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)

        self.rope = initialize_rope(
            int(self.head_dim * args.partial_rotary_factor),
            base=args.rope_theta,
            traditional=False,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        q_proj_output = self.q_proj(x)
        queries, gate = mx.split(
            q_proj_output.reshape(B, L, self.num_attention_heads, -1), 2, axis=-1
        )
        gate = gate.reshape(B, L, -1)

        keys, values = self.k_proj(x), self.v_proj(x)

        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys.reshape(B, L, self.num_key_value_heads, -1)).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(
            0, 2, 1, 3
        )

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.o_proj(output * mx.sigmoid(gate))


class Qwen3NextMLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class Qwen3NextGatedDeltaNet(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        if self.num_v_heads % self.num_k_heads != 0:
            raise ValueError(
                f"num_v_heads ({self.num_v_heads}) must be divisible by num_k_heads ({self.num_k_heads})"
            )

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_norm_epsilon = config.rms_norm_eps

        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=0,
        )

        self.in_proj_qkvz = nn.Linear(
            self.hidden_size, self.key_dim * 2 + self.value_dim * 2, bias=False
        )
        self.in_proj_ba = nn.Linear(self.hidden_size, self.num_v_heads * 2, bias=False)

        self.dt_bias = mx.ones(self.num_v_heads)

        A = mx.random.uniform(low=0, high=16, shape=(self.num_v_heads,))
        self.A_log = mx.log(A)

        self.norm = Qwen3NextRMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)

        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    def fix_query_key_value_ordering(
        self, mixed_qkvz: mx.array, mixed_ba: mx.array
    ) -> mx.array:
        nk, dn, nv, dv = (
            self.num_k_heads,
            self.head_k_dim,
            self.num_v_heads,
            self.head_v_dim,
        )
        mixed_qkvz = mixed_qkvz.reshape(*mixed_qkvz.shape[:-1], nk, -1)
        mixed_ba = mixed_ba.reshape(*mixed_ba.shape[:-1], nk, -1)
        q, k, v, z = mx.split(mixed_qkvz, [dn, 2 * dn, 2 * dn + nv // nk * dv], axis=-1)
        b, a = mx.split(mixed_ba, [nv // nk], axis=-1)
        return (
            q,
            k,
            v.reshape(*v.shape[:2], -1, dv),
            z.reshape(*z.shape[:2], -1, dv),
            b.reshape(*b.shape[:2], nv),
            a.reshape(*a.shape[:2], nv),
        )

    def _process_chunk(
        self,
        mixed_qkv: mx.array,
        a: mx.array,
        b: mx.array,
        conv_state: mx.array,
        ssm_state: Optional[mx.array],
        mask: Optional[mx.array] = None,
        lengths: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """Advance the conv window and recurrent state over one chunk of tokens.

        Returns ``(out, new_conv_state, new_ssm_state)`` where ``out`` is the
        pre-gate-norm Gated DeltaNet output for the chunk and the two states are
        the carry to feed into the next chunk. Factoring this out lets the
        caller process the confirmed and draft portions of a speculative step
        separately and snapshot the carry in between for exact rollback.
        """
        B, S_chunk = mixed_qkv.shape[:2]
        conv_input = mx.concatenate([conv_state, mixed_qkv], axis=1)

        n_keep = self.conv_kernel_size - 1
        if lengths is not None:
            ends = mx.clip(lengths, 0, S_chunk)
            positions = (ends[:, None] + mx.arange(n_keep))[..., None]
            new_conv_state = mx.take_along_axis(conv_input, positions, axis=1)
        else:
            new_conv_state = mx.contiguous(conv_input[:, -n_keep:, :])

        conv_out = nn.silu(self.conv1d(conv_input))

        q, k, v = [
            t.reshape(B, S_chunk, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]

        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

        out, new_ssm_state = gated_delta_update(
            q,
            k,
            v,
            a,
            b,
            self.A_log,
            self.dt_bias,
            ssm_state,
            mask,
            use_kernel=not self.training,
        )
        return out, new_conv_state, new_ssm_state

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        n_confirmed: int = 0,
    ) -> mx.array:
        B, S, _ = inputs.shape
        q, k, v, z, b, a = self.fix_query_key_value_ordering(
            self.in_proj_qkvz(inputs), self.in_proj_ba(inputs)
        )

        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros(
                (B, self.conv_kernel_size - 1, self.conv_dim),
                dtype=inputs.dtype,
            )
        ssm_state = cache[1] if cache else None

        mixed_qkv = mx.concatenate(
            [q.reshape(B, S, -1), k.reshape(B, S, -1), v.reshape(B, S, -1)], axis=-1
        )
        if mask is not None:
            mixed_qkv = mx.where(mask[..., None], mixed_qkv, 0)

        if 0 < n_confirmed < S:
            # Speculative verify step. Process ALL tokens in one unsplit pass so
            # the conv1d + recurrent kernels get M=S amortization (a split into
            # per-token chunks roughly doubles their dispatch/state-IO cost). The
            # rollback carry -- the conv/ssm state after just the confirmed prefix,
            # needed only if the draft is rejected -- is computed in a second pass
            # that mlx leaves UNEVALUATED on acceptance: the caller clears
            # rollback_state (the sole reference) before anything forces it, so
            # accepted rounds (the common case) pay nothing for it. This is
            # numerically identical to a hard split because the recurrent carry is
            # float32, so one M=S pass equals S sequential M=1 steps bit-for-bit.
            out, conv_f, ssm_f = self._process_chunk(
                mixed_qkv, a, b, conv_state, ssm_state, mask
            )
            if cache is not None:
                mask_c = mask[:, :n_confirmed] if mask is not None else None
                _, conv_c, ssm_c = self._process_chunk(
                    mixed_qkv[:, :n_confirmed],
                    a[:, :n_confirmed],
                    b[:, :n_confirmed],
                    conv_state,
                    ssm_state,
                    mask_c,
                )
                cache.rollback_state = (conv_c, ssm_c)
        else:
            lengths = cache.lengths if cache is not None else None
            out, conv_f, ssm_f = self._process_chunk(
                mixed_qkv, a, b, conv_state, ssm_state, mask, lengths=lengths
            )

        if cache is not None:
            cache[0] = conv_f
            cache[1] = ssm_f
            cache.advance(S)

        out = self.norm(out, z)
        return self.out_proj(out.reshape(B, S, -1))


class Qwen3NextSparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        intermediate_size = args.moe_intermediate_size
        shared_expert_intermediate_size = args.shared_expert_intermediate_size

        self.norm_topk_prob = args.norm_topk_prob
        self.num_experts = num_experts = args.num_experts
        self.top_k = args.num_experts_per_tok

        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.switch_mlp = SwitchGLU(
            dim, intermediate_size, num_experts, fuse_gate_up=args.fuse_gate_up
        )

        self.shared_expert = Qwen3NextMLP(dim, shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(dim, 1, bias=False)

        self.sharding_group = None

    def __call__(
        self,
        x: mx.array,
    ) -> mx.array:
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)

        gates = self.gate(x)
        gates = mx.softmax(gates, axis=-1, precise=True)

        k = self.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)

        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)

        shared_y = self.shared_expert(x)
        shared_y = mx.sigmoid(self.shared_expert_gate(x)) * shared_y

        y = y + shared_y

        if self.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.sharding_group)

        return y


class Qwen3NextDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.is_linear = (layer_idx + 1) % args.full_attention_interval != 0
        if self.is_linear:
            self.linear_attn = Qwen3NextGatedDeltaNet(args)
        else:
            self.self_attn = Qwen3NextAttention(args)

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        if (layer_idx not in args.mlp_only_layers) and (
            args.num_experts > 0 and (layer_idx + 1) % args.decoder_sparse_step == 0
        ):
            self.mlp = Qwen3NextSparseMoeBlock(args)
        else:
            self.mlp = Qwen3NextMLP(args.hidden_size, args.intermediate_size)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        n_confirmed: int = 0,
    ) -> mx.array:
        if self.is_linear:
            r = self.linear_attn(
                self.input_layernorm(x), mask, cache, n_confirmed=n_confirmed
            )
        else:
            r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out


class Qwen3NextMTP(nn.Module):
    """Native Multi-Token Prediction head for Qwen3-Next.

    Given the backbone's pre-norm hidden state ``h_t`` for position ``t`` and
    the next token ``x_{t+1}``, it predicts the token at ``t+2`` (one step ahead
    of the backbone), enabling self-speculative decoding with no separate draft
    model. The official checkpoint ships a single full-attention + MoE
    transformer layer here, structurally identical to a non-linear
    ``Qwen3NextDecoderLayer``, plus an EAGLE-style fusion: the separately
    RMS-normed hidden state and token embedding are concatenated and projected
    back to ``hidden_size`` by ``fc``. The shared ``embed_tokens``/``lm_head``
    (applied by the caller) turn the output into logits.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.pre_fc_norm_hidden = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.pre_fc_norm_embedding = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.fc = nn.Linear(args.hidden_size * 2, args.hidden_size, bias=False)
        # Force is_linear=False (full attention + MoE) so the block builds
        # self_attn, matching the checkpoint's mtp.layers.* parameter names.
        self.layers = [
            Qwen3NextDecoderLayer(args, layer_idx=args.full_attention_interval - 1)
            for _ in range(args.mtp_num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        next_token_ids: mx.array,
        embed_tokens: nn.Embedding,
        cache: Optional[Any] = None,
    ) -> mx.array:
        embeds = embed_tokens(next_token_ids)
        e = self.pre_fc_norm_embedding(embeds)
        h = self.pre_fc_norm_hidden(hidden_states)
        fused = self.fc(mx.concatenate([e, h], axis=-1))

        if cache is None:
            cache = [None] * len(self.layers)

        mask = create_attention_mask(fused, cache[0])
        for layer, c in zip(self.layers, cache):
            fused = layer(fused, mask, c)

        return self.norm(fused)


class Qwen3NextModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            Qwen3NextDecoderLayer(args=args, layer_idx=i)
            for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.ssm_idx = 0
        self.fa_idx = args.full_attention_interval - 1

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
        n_confirmed: int = 0,
    ) -> mx.array:
        if input_embeddings is not None:
            hidden_states = input_embeddings
        else:
            hidden_states = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        fa_mask = create_attention_mask(hidden_states, cache[self.fa_idx])
        ssm_mask = create_ssm_mask(hidden_states, cache[self.ssm_idx])

        for layer, c in zip(self.layers, cache):
            mask = ssm_mask if layer.is_linear else fa_mask
            hidden_states = layer(
                hidden_states, mask=mask, cache=c, n_confirmed=n_confirmed
            )

        # Return pre-norm hidden states. The final norm is applied by ``Model``
        # so the MTP head can consume the un-normed backbone hidden state, which
        # is what it was trained on.
        return hidden_states


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Qwen3NextModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        if args.mtp_num_hidden_layers > 0:
            self.mtp = Qwen3NextMTP(args)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
        return_hidden: bool = False,
        n_confirmed: int = 0,
    ) -> mx.array:
        hidden = self.model(
            inputs, cache, input_embeddings=input_embeddings, n_confirmed=n_confirmed
        )
        normed = self.model.norm(hidden)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(normed)
        else:
            out = self.lm_head(normed)
        if return_hidden:
            # Hidden is pre-norm: the MTP head was trained on the un-normed
            # backbone hidden state.
            return out, hidden
        return out

    def mtp_forward(
        self,
        hidden_states: mx.array,
        next_token_ids: mx.array,
        mtp_cache: Any,
    ) -> mx.array:
        """Run the native MTP head and apply the shared output projection.

        Args:
            hidden_states: Backbone pre-norm hidden state, shape ``(B, N, H)``.
                ``N == 1`` during decode, ``N > 1`` during prompt prefill.
            next_token_ids: The token ids one position ahead, shape ``(B, N)``.
            mtp_cache: KVCache entries for the MTP transformer layer(s), from
                :meth:`make_mtp_cache`.

        Returns:
            Logits of shape ``(B, N, vocab_size)``.
        """
        mtp_out = self.mtp(
            hidden_states, next_token_ids, self.model.embed_tokens, mtp_cache
        )
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(mtp_out)
        return self.lm_head(mtp_out)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [ArraysCache(size=2) if l.is_linear else KVCache() for l in self.layers]

    def make_mtp_cache(self):
        """Return a fresh KVCache per MTP layer, or ``[]`` when MTP is absent."""
        if hasattr(self, "mtp"):
            return [KVCache() for _ in self.mtp.layers]
        return []

    def _fuse_gate_up_weights(self, weights):
        """Concatenate the routed experts' gate/up projections into a single
        ``gate_up_proj`` (gate rows first, then up rows) so ``SwitchGLU`` can run
        one gathered matmul instead of two. Concatenation is along the output
        axis (axis=1), which is valid for affine-quantized weights because the
        quantization groups run along the input axis -- every output row keeps
        its own scales/biases, so the result is numerically identical. Replaces
        the two projections in-place, so no extra memory is held."""
        if not self.args.fuse_gate_up:
            return weights
        prefixes = [
            f"model.layers.{l}.mlp.switch_mlp"
            for l in range(self.args.num_hidden_layers)
        ]
        if hasattr(self, "mtp"):
            prefixes += [
                f"mtp.layers.{l}.mlp.switch_mlp"
                for l in range(self.args.mtp_num_hidden_layers)
            ]
        for p in prefixes:
            if f"{p}.gate_proj.weight" not in weights:
                continue
            for sub in ("weight", "scales", "biases"):
                gk, uk = f"{p}.gate_proj.{sub}", f"{p}.up_proj.{sub}"
                if gk in weights and uk in weights:
                    weights[f"{p}.gate_up_proj.{sub}"] = mx.concatenate(
                        [weights.pop(gk), weights.pop(uk)], axis=1
                    )
        return weights

    def sanitize(self, weights):
        if "model.layers.0.mlp.experts.0.up_proj.weight" not in weights:
            # Already-converted (SwitchGLU format) checkpoint: weights are in
            # MLX layout and norms are already shifted. Only intervene to drop
            # the MTP weights when this model has no MTP head to receive them.
            if not hasattr(self, "mtp"):
                weights = {k: v for k, v in weights.items() if "mtp." not in k}
            return self._fuse_gate_up_weights(weights)

        # Raw HF checkpoint path.
        if not hasattr(self, "mtp"):
            weights = {
                key: value for key, value in weights.items() if "mtp." not in key
            }
        elif not any("mtp." in k for k in weights):
            raise ValueError(
                "mtp_num_hidden_layers > 0 but the checkpoint contains no MTP "
                "weights. Set mtp_num_hidden_layers=0 to disable the MTP head."
            )

        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        # Stack per-expert MoE weights into SwitchGLU format for the backbone
        # layers and, when present, the MTP layer(s) (same expert layout).
        expert_prefixes = [
            f"model.layers.{l}.mlp" for l in range(self.args.num_hidden_layers)
        ]
        if hasattr(self, "mtp"):
            expert_prefixes += [
                f"mtp.layers.{l}.mlp" for l in range(self.args.mtp_num_hidden_layers)
            ]
        for prefix in expert_prefixes:
            if f"{prefix}.experts.0.up_proj.weight" not in weights:
                continue
            for n in ["up_proj", "down_proj", "gate_proj"]:
                to_join = [
                    weights.pop(f"{prefix}.experts.{e}.{n}.weight")
                    for e in range(self.args.num_experts)
                ]
                weights[f"{prefix}.switch_mlp.{n}.weight"] = mx.stack(to_join)

        norm_keys = (
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
            # MTP-specific norms (the reused decoder-layer norms above already
            # match by suffix).
            ".pre_fc_norm_hidden.weight",
            ".pre_fc_norm_embedding.weight",
            "mtp.norm.weight",
        )
        for k, v in weights.items():
            if "conv1d.weight" in k and v.shape[-1] != 1:
                weights[k] = v.moveaxis(2, 1)
            if any(k.endswith(sfx) for sfx in norm_keys):
                if v.ndim == 1:
                    weights[k] = v + 1.0
        return self._fuse_gate_up_weights(weights)

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if path.endswith("mlp.gate") or path.endswith("shared_expert_gate"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate
