# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pascal (sm_60/sm_61) attention backend: pure-PyTorch SDPA over the paged KV
cache.

Every other CUDA attention backend in this tree is unusable below sm_70:
  * flash_attn_v100 is nvcuda::wmma (HMMA) throughout — no tensor cores on
    Pascal;
  * every Triton attention kernel bottoms out in tl.dot, which Triton cannot
    lower for sm_60 (verified on a P100: ``PassManager::run failed``);
  * vllm_flash_attn / FlashInfer are sm_75+.

This backend trades speed for portability: the KV cache update stays on the
CUDA core _C op (reshape_and_cache_flash compiles for any arch), and the
attention itself is computed with plain torch matmul/softmax in fp32, one
request at a time. It is deliberately the simplest thing that is CORRECT;
optimize only after outputs are validated.

Reuses TritonAttentionMetadata/-Builder for scheduling metadata (those are
plain tensor plumbing, no tl.dot), with CPU twins of query_start_loc/seq_lens
attached at build time the same way flash_attn_v100 does, so forward never
syncs the GPU to read loop bounds.
"""

from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm.logger import init_logger
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import AttentionCGSupport, AttentionType
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
    TritonAttentionMetadata,
    TritonAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.pascal_decode_attn import paged_decode_attention
from vllm.v1.attention.backends.utils import CommonAttentionMetadata

logger = init_logger(__name__)


@dataclass
class PascalSDPAMetadata(TritonAttentionMetadata):
    # CPU twins so the per-request loop never forces a device sync.
    query_start_loc_cpu: torch.Tensor | None = None
    seq_lens_cpu: torch.Tensor | None = None


class PascalSDPAMetadataBuilder(TritonAttentionMetadataBuilder):
    # Only the all-qlen==1 decode path is graph-safe: it runs one triton kernel
    # whose launch shape is (batch, heads) and which reads each sequence's
    # length from device memory.  Prefill and mixed batches still walk a
    # host-driven python loop over requests, so they must stay eager.
    # (TritonAttentionMetadataBuilder claims ALWAYS; that is not true here.)
    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        **kwargs,
    ):
        m = super().build(common_prefix_len, common_attn_metadata, **kwargs)
        md = PascalSDPAMetadata(
            **{
                f: getattr(m, f)
                for f in TritonAttentionMetadata.__dataclass_fields__
            }
        )
        md.query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        md.seq_lens_cpu = common_attn_metadata.seq_lens_cpu
        return md


class PascalSDPABackend(TritonAttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        return "PASCAL_SDPA"

    @staticmethod
    def get_impl_cls() -> type["PascalSDPAImpl"]:
        return PascalSDPAImpl

    @staticmethod
    def get_builder_cls() -> type["PascalSDPAMetadataBuilder"]:
        return PascalSDPAMetadataBuilder

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major >= 6

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        return False


class PascalSDPAImpl(TritonAttentionImpl):
    """Pure-torch paged attention. fp32 math, per-request loop."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.alibi_slopes is not None:
            raise NotImplementedError("PASCAL_SDPA: alibi not supported")
        if self.logits_soft_cap:
            raise NotImplementedError("PASCAL_SDPA: soft cap not supported")
        if getattr(self, "sinks", None) is not None:
            raise NotImplementedError("PASCAL_SDPA: sinks not supported")

    # ------------------------------------------------------------------
    # KV cache update: CUDA core op from _C, portable to every arch.
    # ------------------------------------------------------------------
    def do_kv_cache_update(
        self,
        layer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ):
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return
        if self.kv_cache_dtype != "auto":
            raise NotImplementedError(
                "PASCAL_SDPA: only kv_cache_dtype=auto is supported"
            )
        # Pure-torch cache write.  reshape_and_cache_flash lives in
        # _C_stable_libtorch, which cannot be built here (torch 2.7 ships no
        # stable-ABI headers), so torch.ops._C_cache_ops does not exist at all
        # in this build.
        #
        # slot_mapping encodes block * block_size + offset; padding rows carry
        # -1, which the C++ kernel skips.  Here they are clamped into slot 0 --
        # vLLM's reserved null block (BlockPool pops block 0 and marks it
        # is_null, so no request is ever handed it) whose contents are never
        # read back.  Clamping rather than masking is deliberate: selecting the
        # valid rows needs `bool(mask.all())` on the host, which syncs the GPU
        # every single step *and* makes the write shape depend on data -- either
        # one alone makes the decode step impossible to capture in a CUDA graph.
        key_cache, value_cache = kv_cache.unbind(1)
        block_size = key_cache.shape[1]
        n = slot_mapping.shape[0]
        slots = slot_mapping.clamp_min(0)
        blk = torch.div(slots, block_size, rounding_mode="floor")
        off = slots - blk * block_size
        key_cache[blk, off] = key[:n]
        value_cache[blk, off] = value[:n]

    def fused_rope_kvcache_supported(self):
        return False

    # ------------------------------------------------------------------
    # Attention
    # ------------------------------------------------------------------
    def _sdpa_one(
        self,
        q: torch.Tensor,   # [qlen, H, D] fp32
        k: torch.Tensor,   # [ctx, H, D] fp32 (kv heads already expanded)
        v: torch.Tensor,   # [ctx, H, D] fp32
        causal: bool,
    ) -> torch.Tensor:     # [qlen, H, D] fp32
        qlen, H, D = q.shape
        ctx = k.shape[0]
        qh = q.permute(1, 0, 2)                       # [H, qlen, D]
        kh = k.permute(1, 2, 0)                       # [H, D, ctx]
        scores = torch.bmm(qh, kh) * self.scale       # [H, qlen, ctx]
        if causal and qlen > 1:
            # query j (absolute position ctx-qlen+j) sees kv positions <= that
            qpos = torch.arange(
                ctx - qlen, ctx, device=scores.device
            ).unsqueeze(1)
            kpos = torch.arange(ctx, device=scores.device).unsqueeze(0)
            scores.masked_fill_((kpos > qpos).unsqueeze(0), float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out = torch.bmm(probs, v.permute(1, 0, 2))    # [H, qlen, D]
        return out.permute(1, 0, 2)

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: PascalSDPAMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "PASCAL_SDPA: fused output quantization not supported"
            )
        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        num_actual_tokens = attn_metadata.num_actual_tokens

        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return self._forward_encoder_attention(
                query[:num_actual_tokens],
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                output[:num_actual_tokens],
                attn_metadata,
                layer,
            )

        assert attn_metadata.use_cascade is False
        key_cache, value_cache = kv_cache.unbind(1)
        # [num_blocks, block_size, KVH, D]
        block_size = key_cache.shape[1]

        cu_cpu = attn_metadata.query_start_loc_cpu
        seq_cpu = attn_metadata.seq_lens_cpu
        assert cu_cpu is not None and seq_cpu is not None
        block_table = attn_metadata.block_table
        n_rep = self.num_heads // self.num_kv_heads

        num_seqs = seq_cpu.shape[0]

        # ---- batched decode fast path: every sequence has qlen == 1 -------
        # One triton kernel for the whole batch.  Its launch shape is
        # (batch, heads) and the per-sequence context length is read from the
        # device-side seq_lens *inside* the kernel, so nothing here depends on
        # host data -- which is what makes the decode step capturable into a
        # CUDA graph.  (See pascal_decode_attn.py for why sm_60 needs its own
        # kernel rather than any of the tl.dot-based ones.)
        if num_actual_tokens == num_seqs and int(cu_cpu[num_seqs]) == num_seqs:
            paged_decode_attention(
                query[:num_seqs],
                key_cache,
                value_cache,
                block_table[:num_seqs],
                attn_metadata.seq_lens[:num_seqs],
                output[:num_seqs],
                self.scale,
            )
            return output
        for i in range(num_seqs):
            q0 = int(cu_cpu[i])
            q1 = int(cu_cpu[i + 1])
            qlen = q1 - q0
            if qlen <= 0:
                continue
            ctx = int(seq_cpu[i])
            nblk = (ctx + block_size - 1) // block_size
            blocks = block_table[i, :nblk].to(torch.long)
            k = key_cache.index_select(0, blocks).view(
                -1, self.num_kv_heads, self.head_size
            )[:ctx]
            v = value_cache.index_select(0, blocks).view(
                -1, self.num_kv_heads, self.head_size
            )[:ctx]
            if n_rep > 1:
                k = k.repeat_interleave(n_rep, dim=1)
                v = v.repeat_interleave(n_rep, dim=1)
            o = self._sdpa_one(
                query[q0:q1].float(),
                k.float(),
                v.float(),
                causal=True,
            )
            output[q0:q1] = o.to(output.dtype)
        return output

    def _forward_encoder_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: PascalSDPAMetadata,
        layer: torch.nn.Module,
    ) -> torch.Tensor:
        # Bidirectional attention over each encoder sequence, no KV cache.
        cu = attn_metadata.query_start_loc_cpu
        if cu is None:
            cu = attn_metadata.query_start_loc.cpu()
        n_rep = self.num_heads // self.num_kv_heads
        for i in range(cu.shape[0] - 1):
            q0, q1 = int(cu[i]), int(cu[i + 1])
            if q1 <= q0:
                continue
            k = key[q0:q1]
            v = value[q0:q1]
            if n_rep > 1:
                k = k.repeat_interleave(n_rep, dim=1)
                v = v.repeat_interleave(n_rep, dim=1)
            o = self._sdpa_one(
                query[q0:q1].float(), k.float(), v.float(), causal=False
            )
            output[q0:q1] = o.to(output.dtype)
        return output
