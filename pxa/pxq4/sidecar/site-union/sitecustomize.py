# Attaches a stderr handler to the pxq4_vllm logger tree so its INFO lines are
# visible in container logs (vLLM's logging config only handles the "vllm" namespace).
import logging, sys
_lg = logging.getLogger("pxq4_vllm")
if not _lg.handlers:
    _h = logging.StreamHandler(sys.stderr)
    _h.setFormatter(logging.Formatter("PXQ4LOG %(levelname)s %(name)s: %(message)s"))
    _lg.addHandler(_h)
    _lg.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# PASCAL PORT: torch 2.7 compatibility shims for torch.accelerator APIs that
# the 1cat fork (written against torch 2.10) calls. Loaded in every vllm
# process via PYTHONPATH. Each shim maps to the torch.cuda equivalent.
# ---------------------------------------------------------------------------
try:
    import torch as _pxa_torch

    _acc = _pxa_torch.accelerator
    if not hasattr(_acc, "empty_cache"):
        _acc.empty_cache = _pxa_torch.cuda.empty_cache
    if not hasattr(_acc, "device_index"):
        _acc.device_index = _pxa_torch.cuda.device
    if not hasattr(_acc, "reset_peak_memory_stats"):
        _acc.reset_peak_memory_stats = _pxa_torch.cuda.reset_peak_memory_stats
    if not hasattr(_acc, "max_memory_allocated"):
        _acc.max_memory_allocated = _pxa_torch.cuda.max_memory_allocated
    if not hasattr(_acc, "memory_allocated"):
        _acc.memory_allocated = _pxa_torch.cuda.memory_allocated
    if not hasattr(_acc, "memory_reserved"):
        _acc.memory_reserved = _pxa_torch.cuda.memory_reserved
    for _name in (
        "memory_stats", "memory_summary", "mem_get_info", "memory_snapshot",
        "max_memory_reserved", "reset_accumulated_memory_stats",
        "reset_max_memory_allocated", "synchronize",
    ):
        if not hasattr(_acc, _name) and hasattr(_pxa_torch.cuda, _name):
            setattr(_acc, _name, getattr(_pxa_torch.cuda, _name))
except Exception:
    pass
# ---------------------------------------------------------------------------
# PXQ4_CUDAGRAPH_TRACE: env-gated instrumentation of cudagraph dispatch.
# Default OFF; enable with PXQ4_CUDAGRAPH_TRACE=1. Logs, per dispatcher
# instance (the target worker and the drafter each own one):
#   - INIT: resolved cudagraph mode, uniform_decode_query_len, and the full
#     key sets (FULL and PIECEWISE) so captured-graph population is visible
#   - DUMP every PXQ4_CUDAGRAPH_TRACE_INTERVAL dispatches: aggregated
#     (dispatcher, runtime mode, input tokens, uniform, padded size) counts
# ---------------------------------------------------------------------------
import os as _pxa_os

if _pxa_os.getenv("PXQ4_CUDAGRAPH_TRACE", "") not in ("", "0"):
    try:
        import collections as _pxa_collections
        import sys as _pxa_sys
        import threading as _pxa_threading

        from vllm.config import CUDAGraphMode as _PXA_CGM
        from vllm.v1.cudagraph_dispatcher import (
            CudagraphDispatcher as _PXA_Dispatcher,
        )

        _pxa_trace_lock = _pxa_threading.Lock()
        _pxa_disp_ids: dict = {}
        _pxa_counts: "_pxa_collections.Counter" = _pxa_collections.Counter()
        _pxa_interval = int(
            _pxa_os.getenv("PXQ4_CUDAGRAPH_TRACE_INTERVAL", "2000")
        )
        _pxa_ncalls = [0]

        def _pxa_disp_id(disp):
            with _pxa_trace_lock:
                return _pxa_disp_ids.setdefault(id(disp), len(_pxa_disp_ids))

        _pxa_orig_init = _PXA_Dispatcher.initialize_cudagraph_keys

        def _pxa_init(self, cudagraph_mode, uniform_decode_query_len=1):
            r = _pxa_orig_init(self, cudagraph_mode, uniform_decode_query_len)
            try:
                did = _pxa_disp_id(self)
                full = sorted(
                    (d.num_tokens, d.num_reqs, d.attention_context_bucket)
                    for d in self.cudagraph_keys[_PXA_CGM.FULL]
                )
                pw = sorted(
                    d.num_tokens for d in self.cudagraph_keys[_PXA_CGM.PIECEWISE]
                )
                print(
                    f"CGTRACE pid={_pxa_os.getpid()} disp{did} INIT "
                    f"mode={cudagraph_mode} udql={uniform_decode_query_len} "
                    f"FULL({len(full)})={full} PIECEWISE({len(pw)})={pw}",
                    file=_pxa_sys.stderr,
                    flush=True,
                )
            except Exception:
                pass
            return r

        _PXA_Dispatcher.initialize_cudagraph_keys = _pxa_init

        _pxa_orig_dispatch = _PXA_Dispatcher.dispatch

        def _pxa_dispatch(self, num_tokens, *args, **kw):
            mode, desc = _pxa_orig_dispatch(self, num_tokens, *args, **kw)
            try:
                uniform = kw.get(
                    "uniform_decode", args[0] if args else False
                )
                did = _pxa_disp_id(self)
                with _pxa_trace_lock:
                    _pxa_counts[
                        (did, str(mode), num_tokens, bool(uniform),
                         desc.num_tokens, bool(desc.uniform))
                    ] += 1
                    _pxa_ncalls[0] += 1
                    dump = _pxa_ncalls[0] % _pxa_interval == 0
                    items = (
                        sorted(_pxa_counts.items(), key=lambda kv: -kv[1])[:24]
                        if dump
                        else None
                    )
                if items:
                    print(
                        f"CGTRACE pid={_pxa_os.getpid()} DUMP "
                        f"n={_pxa_ncalls[0]} "
                        + " | ".join(
                            f"disp{k[0]} {k[1]} in={k[2]} unif={k[3]} "
                            f"pad={k[4]} punif={k[5]} x{v}"
                            for k, v in items
                        ),
                        file=_pxa_sys.stderr,
                        flush=True,
                    )
            except Exception:
                pass
            return mode, desc

        _PXA_Dispatcher.dispatch = _pxa_dispatch
        print(
            f"CGTRACE pid={_pxa_os.getpid()} armed interval={_pxa_interval}",
            file=_pxa_sys.stderr,
            flush=True,
        )
    except Exception:
        pass

# ---------------------------------------------------------------------------
# PXQ4_ENGINE_LOOP_TRACE: measures the gap BETWEEN EngineCore step calls
# (loop/IPC/output overhead) vs time INSIDE the step. Env-gated, default off.
# ---------------------------------------------------------------------------
import os as _pxa_os2

if _pxa_os2.getenv("PXQ4_ENGINE_LOOP_TRACE", "") not in ("", "0"):
    try:
        import sys as _pxa_sys2
        import time as _pxa_time2

        from vllm.v1.engine.core import EngineCore as _PXA_EngineCore

        _state = {"last_end": None, "n": 0, "gap": 0.0, "inside": 0.0,
                  "gap_max": 0.0, "in_max": 0.0}
        _every = int(_pxa_os2.getenv("PXQ4_ENGINE_LOOP_TRACE_EVERY", "64"))

        def _wrap(name):
            orig = getattr(_PXA_EngineCore, name)

            def wrapped(self, *a, **kw):
                t0 = _pxa_time2.perf_counter()
                if _state["last_end"] is not None:
                    g = t0 - _state["last_end"]
                    _state["gap"] += g
                    _state["gap_max"] = max(_state["gap_max"], g)
                r = orig(self, *a, **kw)
                t1 = _pxa_time2.perf_counter()
                _state["last_end"] = t1
                _state["inside"] += t1 - t0
                _state["in_max"] = max(_state["in_max"], t1 - t0)
                _state["n"] += 1
                if _state["n"] % _every == 0:
                    print(
                        f"ENGLOOP n={_state['n']} "
                        f"inside_avg_ms={_state['inside']/_every*1000:.2f} "
                        f"inside_max_ms={_state['in_max']*1000:.1f} "
                        f"gap_avg_ms={_state['gap']/_every*1000:.2f} "
                        f"gap_max_ms={_state['gap_max']*1000:.1f}",
                        file=_pxa_sys2.stderr, flush=True,
                    )
                    _state["gap"] = _state["inside"] = 0.0
                    _state["gap_max"] = _state["in_max"] = 0.0
                return r

            return wrapped

        for _n in ("step", "step_with_batch_queue"):
            if hasattr(_PXA_EngineCore, _n):
                setattr(_PXA_EngineCore, _n, _wrap(_n))
        print("ENGLOOP armed", file=_pxa_sys2.stderr, flush=True)
    except Exception:
        pass

# ---------------------------------------------------------------------------
# PXQ4_GREEDY_VERIFY_FAST: env-gated all-greedy spec-verify sampling fast path.
# When every request is greedy, drafts are one-hot (draft_probs None), and no
# logprobs are requested, the verify sampling only needs argmax per row. The
# stock path materializes [num_draft, V] fp32, runs processors/constraints on
# it, argmaxes it, and runs the full sampler on the bonus row. This path does
# ONE fp16 argmax over all rows and feeds the greedy accept kernel directly.
# Default OFF; enable with PXQ4_GREEDY_VERIFY_FAST=1. Falls back to the stock
# path for any condition it does not cover, and on any error arms itself off.
# ---------------------------------------------------------------------------
import os as _pxa_os3

if _pxa_os3.getenv("PXQ4_GREEDY_VERIFY_FAST", "") not in ("", "0"):
    try:
        import sys as _pxa_sys3

        import torch as _pxa_torch3
        from vllm.v1.sample import rejection_sampler as _pxa_rej

        _pxa_orig_forward = _pxa_rej.RejectionSampler.forward
        _pxa_gvf_state = {"disabled": False, "hits": 0, "misses": 0}

        def _pxa_gvf_forward(self, metadata, draft_probs, logits, sampling_metadata):
            st = _pxa_gvf_state
            if (
                st["disabled"]
                or draft_probs is not None
                or not sampling_metadata.all_greedy
                or sampling_metadata.max_num_logprobs is not None
                or getattr(sampling_metadata, "logprob_token_ids", None)
                or getattr(self, "synthetic_mode", False)
                or getattr(sampling_metadata, "bad_words_token_ids", None)
                or getattr(sampling_metadata, "allowed_token_ids_mask", None)
                    is not None
                or not getattr(sampling_metadata, "no_penalties", False)
            ):
                st["misses"] += 1
                return _pxa_orig_forward(
                    self, metadata, draft_probs, logits, sampling_metadata
                )
            try:
                self._last_target_candidate_ids = None
                # One argmax over every sampled row (targets + bonus together).
                flat_argmax = logits.argmax(dim=-1)
                bonus_token_ids = flat_argmax[metadata.bonus_logits_indices].view(
                    -1, 1
                )
                target_argmax = flat_argmax[metadata.target_logits_indices]

                batch_size = len(metadata.num_draft_tokens)
                output_token_ids = _pxa_torch3.full(
                    (batch_size, metadata.max_spec_len + 1),
                    _pxa_rej.PLACEHOLDER_TOKEN_ID,
                    dtype=_pxa_torch3.int32,
                    device=logits.device,
                )
                _pxa_rej.rejection_greedy_sample_kernel[(batch_size,)](
                    output_token_ids,
                    metadata.cu_num_draft_tokens,
                    metadata.draft_token_ids,
                    target_argmax,
                    bonus_token_ids,
                    None,  # is_greedy: all greedy
                    metadata.max_spec_len,
                    None,  # uniform_probs
                    None,  # synthetic_conditional_rates
                    SYNTHETIC_MODE=False,
                )
                st["hits"] += 1
                if st["hits"] == 1:
                    print(
                        "PXQ4_GREEDY_VERIFY_FAST active (first hit)",
                        file=_pxa_sys3.stderr,
                        flush=True,
                    )
                return _pxa_rej.SamplerOutput(
                    sampled_token_ids=output_token_ids, logprobs_tensors=None
                )
            except Exception as exc:
                st["disabled"] = True
                print(
                    f"PXQ4_GREEDY_VERIFY_FAST disabled after error: {exc!r}",
                    file=_pxa_sys3.stderr,
                    flush=True,
                )
                return _pxa_orig_forward(
                    self, metadata, draft_probs, logits, sampling_metadata
                )

        _pxa_rej.RejectionSampler.forward = _pxa_gvf_forward
        print("PXQ4_GREEDY_VERIFY_FAST armed", file=_pxa_sys3.stderr, flush=True)
    except Exception:
        pass

# ---------------------------------------------------------------------------
# VLLM_SM70_META_CACHE: env-gated steady-decode attention-metadata cache.
# Measured (V100 lab, np4 MTP, single stream): the target execute step spends
# ~7.5 ms/step rebuilding attention metadata = 3x GDN builds (~2.1 ms each,
# kv_cache_gid 0..2) + 1x FlashAttnV100 build (~1.1 ms). In steady pure-spec
# decode the produced metadata objects are slices of PERSISTENT buffers whose
# only per-step content is (a) GDN spec state block ids / accepted counts /
# slot selectors and (b) the Flash small-query decode expansion. So:
#   - GDN: when the batch shape (num_reqs, num_actual_tokens, max_query_len,
#     query_start_loc, draft-token layout, align-mode state table shape) is
#     unchanged AND the previous build took the pure-spec full-cudagraph
#     branch into the builder's own persistent buffers, reuse the metadata
#     object and refresh ONLY the three per-step buffer rows
#     (spec_state_indices <- current_state_block_ids, num_accepted_tokens,
#     spec_state_slot_selectors). Align mode does not read the block table in
#     this branch, so no block-boundary hazard exists; the fresh
#     current_state_block_ids row IS the per-step state addressing.
#   - FlashAttnV100 (target verify, max_query_len>1): when every tensor field
#     still aliases the same persistent buffers (data_ptr identity), reuse the
#     object, update max_seq_len, and re-run only the per-step helpers
#     (_update_smallq_decode_metadata / shape hints / active partitions).
# Any regime change (prefill, ddtree, cascade, resized batch, new buffers)
# misses and falls back to the stock build, which re-snapshots.
# Default OFF; enable with VLLM_SM70_META_CACHE=1. On any error the cache
# disarms permanently and the stock path takes over.
# ---------------------------------------------------------------------------
import os as _pxa_os4

if _pxa_os4.getenv("VLLM_SM70_META_CACHE", "") not in ("", "0"):
    try:
        import sys as _pxa_sys4

        import torch as _pxa_torch4
        from vllm.v1.attention.backends.flash_attn_v100 import (
            FlashAttnV100MetadataBuilder as _PXA_FAB,
        )
        from vllm.v1.attention.backends.gdn_attn import (
            GDNAttentionMetadataBuilder as _PXA_GDNB,
        )

        # Mode: "1"/"both" = both caches; "gdn" / "fa" = one side only;
        # "verify" = hit paths run AND the stock build still executes, with
        # any divergence between the two logged (output stays stock-correct).
        _pxa_mc_mode = _pxa_os4.getenv("VLLM_SM70_META_CACHE", "1").strip().lower()
        _pxa_mc_verify = _pxa_mc_mode == "verify"
        _pxa_mc_gdn_on = _pxa_mc_mode in ("1", "both", "gdn", "verify", "true")
        _pxa_mc_fa_on = _pxa_mc_mode in ("1", "both", "fa", "verify", "true")

        _pxa_mc_state = {"disabled": False, "gdn_hits": 0, "fa_hits": 0}

        def _pxa_mc_vlog(msg):
            print(f"VLLM_SM70_META_CACHE VERIFY {msg}", file=_pxa_sys4.stderr, flush=True)

        def _pxa_mc_disarm(exc):
            _pxa_mc_state["disabled"] = True
            print(
                f"VLLM_SM70_META_CACHE disabled after error: {exc!r}",
                file=_pxa_sys4.stderr,
                flush=True,
            )

        # ------------------------- GDN builder -------------------------
        _pxa_gdn_orig_build = _PXA_GDNB.build

        def _pxa_gdn_key(self, m, ndt, csbi):
            return (
                int(m.num_reqs),
                int(m.num_actual_tokens),
                int(m.max_query_len),
                tuple(m.query_start_loc_cpu.tolist()),
                tuple(ndt.tolist()),
                tuple(csbi.shape),
                m.query_start_loc.data_ptr(),
            )

        def _pxa_gdn_build(self, common_prefix_len, common_attn_metadata, *a, **kw):
            st = _pxa_mc_state
            m = common_attn_metadata
            nat = kw.get("num_accepted_tokens")
            sss = kw.get("spec_state_slot_selectors")
            ndt = kw.get("num_decode_draft_tokens_cpu")
            csbi = kw.get("current_state_block_ids")
            eligible = (
                not st["disabled"]
                and _pxa_mc_gdn_on
                and not a
                and common_prefix_len == 0
                and not kw.get("for_cudagraph_capture", False)
                and not kw.get("fast_build", False)
                and kw.get("ddtree_parent_ids") is None
                and kw.get("spec_sequence_masks_cpu") is None
                and nat is not None
                and sss is not None
                and ndt is not None
                and csbi is not None
                and getattr(self, "_ddtree_fast_common_buffers", None) is None
                and bool(getattr(self, "use_full_cuda_graph", False))
            )
            if eligible:
                try:
                    cached = getattr(self, "_pxa_meta_cache", None)
                    if cached is not None and cached[0] == _pxa_gdn_key(
                        self, m, ndt, csbi
                    ):
                        md = cached[1]
                        nsd = md.num_spec_decodes
                        w = self.num_spec_state_tokens + 1
                        self.spec_state_indices_tensor[:nsd].copy_(
                            csbi[:nsd, :w], non_blocking=True
                        )
                        self.num_accepted_tokens[:nsd].copy_(
                            nat[:nsd], non_blocking=True
                        )
                        self.spec_state_slot_selectors[:nsd].copy_(
                            sss[:nsd], non_blocking=True
                        )
                        # current_state_block_ids is an async H2D view of a
                        # pinned staging buffer the runner REFILLS for the
                        # next kv-cache group within this same step. The
                        # stock build serialized that reuse via its
                        # `spec_query_start_loc[-1].item()` assert; without
                        # an equivalent wait the next group's CPU refill
                        # races the in-flight H2D and corrupts state ids
                        # (reproduced). Pay the same wait the stock path
                        # pays before returning.
                        _pxa_ev = _pxa_torch4.cuda.Event()
                        _pxa_ev.record()
                        _pxa_ev.synchronize()
                        st["gdn_hits"] += 1
                        if st["gdn_hits"] == 1:
                            print(
                                "VLLM_SM70_META_CACHE active (first GDN hit)",
                                file=_pxa_sys4.stderr,
                                flush=True,
                            )
                        if not _pxa_mc_verify:
                            return md
                        # verify mode: snapshot the fast-path buffer state,
                        # run the stock build (truth), compare, return truth.
                        f_idx = self.spec_state_indices_tensor[:nsd].clone()
                        f_nat = self.num_accepted_tokens[:nsd].clone()
                        f_sss = self.spec_state_slot_selectors[:nsd].clone()
                        f_qsl = self.spec_query_start_loc[: nsd + 1].clone()
                        f_tok = self.spec_token_indx[
                            : md.spec_token_indx.numel()
                        ].clone()
                        f_msk = self.spec_sequence_masks[:nsd].clone()
                        md2 = _pxa_gdn_orig_build(
                            self, common_prefix_len, common_attn_metadata, *a, **kw
                        )
                        pairs = [
                            ("spec_state_indices",
                             f_idx, self.spec_state_indices_tensor[:nsd]),
                            ("num_accepted_tokens",
                             f_nat, self.num_accepted_tokens[:nsd]),
                            ("spec_state_slot_selectors",
                             f_sss, self.spec_state_slot_selectors[:nsd]),
                            ("spec_query_start_loc",
                             f_qsl, self.spec_query_start_loc[: nsd + 1]),
                            ("spec_token_indx",
                             f_tok,
                             self.spec_token_indx[: md.spec_token_indx.numel()]),
                            ("spec_sequence_masks",
                             f_msk, self.spec_sequence_masks[:nsd]),
                        ]
                        for name, mine, truth in pairs:
                            if not _pxa_torch4.equal(mine, truth):
                                _pxa_mc_vlog(
                                    f"GDN MISMATCH {name}: fast="
                                    f"{mine.detach().cpu().tolist()} truth="
                                    f"{truth.detach().cpu().tolist()}"
                                )
                        scal = [
                            ("num_spec_decodes", md.num_spec_decodes,
                             md2.num_spec_decodes),
                            ("num_spec_decode_tokens", md.num_spec_decode_tokens,
                             md2.num_spec_decode_tokens),
                            ("num_actual_tokens", md.num_actual_tokens,
                             md2.num_actual_tokens),
                            ("num_prefills", md.num_prefills, md2.num_prefills),
                            ("num_decodes", md.num_decodes, md2.num_decodes),
                        ]
                        for name, mine_v, truth_v in scal:
                            if mine_v != truth_v:
                                _pxa_mc_vlog(
                                    f"GDN SCALAR MISMATCH {name}: "
                                    f"fast={mine_v} truth={truth_v}"
                                )
                        if st["gdn_hits"] % 512 == 1:
                            _pxa_mc_vlog(f"GDN checked hit #{st['gdn_hits']}")
                        return md2
                except Exception as exc:
                    _pxa_mc_disarm(exc)
            md = _pxa_gdn_orig_build(
                self, common_prefix_len, common_attn_metadata, *a, **kw
            )
            try:
                snap = None
                if eligible and not st["disabled"]:
                    if (
                        md.num_prefills == 0
                        and md.num_decodes == 0
                        and md.num_spec_decodes >= 1
                        and md.num_spec_decodes == int(m.num_reqs)
                        and md.non_spec_state_indices_tensor is None
                        and md.chunk_indices is None
                        and md.nums_dict is None
                        and md.has_initial_state is None
                        and md.ddtree_parent_ids is None
                        and md.spec_state_indices_tensor.data_ptr()
                        == self.spec_state_indices_tensor.data_ptr()
                        and md.num_accepted_tokens is not None
                        and md.num_accepted_tokens.data_ptr()
                        == self.num_accepted_tokens.data_ptr()
                        and md.spec_state_slot_selectors is not None
                        and md.spec_state_slot_selectors.data_ptr()
                        == self.spec_state_slot_selectors.data_ptr()
                        and md.spec_state_indices_tensor.shape[1]
                        == self.num_spec_state_tokens + 1
                        and csbi.shape[1] >= self.num_spec_state_tokens + 1
                        and bool((ndt >= 0).all().item())
                    ):
                        snap = (_pxa_gdn_key(self, m, ndt, csbi), md)
                self._pxa_meta_cache = snap
            except Exception:
                self._pxa_meta_cache = None
            return md

        _PXA_GDNB.build = _pxa_gdn_build

        # --------------------- FlashAttnV100 builder ---------------------
        # build_for_cudagraph_capture funnels through self.build(0, cm) via
        # the Triton parent, with the SAME persistent buffers as runtime, so
        # a capture build could otherwise be snapshotted (it is later mutated:
        # seq_lens.fill_, flash_v100_cudagraph_capture=True) or hit the cache.
        # Guard the whole capture window and drop any snapshot it leaves.
        _pxa_fa_orig_build = _PXA_FAB.build
        _pxa_fa_orig_capture = _PXA_FAB.build_for_cudagraph_capture

        def _pxa_fa_capture(self, common_attn_metadata):
            self._pxa_in_capture = True
            try:
                return _pxa_fa_orig_capture(self, common_attn_metadata)
            finally:
                self._pxa_in_capture = False
                self._pxa_meta_cache = None

        _PXA_FAB.build_for_cudagraph_capture = _pxa_fa_capture

        def _pxa_fa_cpu_seq_lens(m):
            slc = getattr(m, "_seq_lens_cpu", None)
            if slc is None:
                slc = getattr(m, "seq_lens_cpu_upper_bound", None)
            return slc

        def _pxa_fa_key(self, m):
            slc = _pxa_fa_cpu_seq_lens(m)
            return (
                int(m.num_reqs),
                int(m.num_actual_tokens),
                int(m.max_query_len),
                m.query_start_loc.data_ptr(),
                m.seq_lens.data_ptr(),
                m.block_table_tensor.data_ptr(),
                tuple(m.block_table_tensor.shape),
                m.slot_mapping.data_ptr(),
                int(m.slot_mapping.numel()),
                m.query_start_loc_cpu.data_ptr(),
                slc.data_ptr() if slc is not None else 0,
                bool(getattr(m, "causal", True)),
            )

        def _pxa_fa_build(self, common_prefix_len, common_attn_metadata, *a, **kw):
            st = _pxa_mc_state
            m = common_attn_metadata
            eligible = (
                not st["disabled"]
                and _pxa_mc_fa_on
                and not a
                and not getattr(self, "_pxa_in_capture", False)
                and common_prefix_len == 0
                and not kw.get("fast_build", False)
                and kw.get("ddtree_parent_ids") is None
                and kw.get("ddtree_num_tree_tokens_cpu") is None
                and int(m.max_query_len) > 1
                and _pxa_fa_cpu_seq_lens(m) is not None
            )
            if eligible:
                try:
                    cached = getattr(self, "_pxa_meta_cache", None)
                    if cached is not None and cached[0] == _pxa_fa_key(self, m):
                        md = cached[1]
                        md.max_seq_len = m.max_seq_len
                        self._update_smallq_decode_metadata(
                            md,
                            m,
                            workspace_seq_capacity_cap=(
                                int(getattr(m, "max_seq_len", 0) or 0) or None
                            ),
                        )
                        self._attach_decode_shape_hints(md, m)
                        self._update_decode_active_num_partitions(md, stage="build")
                        st["fa_hits"] += 1
                        if st["fa_hits"] == 1:
                            print(
                                "VLLM_SM70_META_CACHE active (first FA hit)",
                                file=_pxa_sys4.stderr,
                                flush=True,
                            )
                        if not _pxa_mc_verify:
                            return md
                        ntok = int(md.num_actual_tokens)
                        nrq = int(m.num_reqs)
                        f_bt = (
                            self._smallq_decode_block_table[:ntok].clone()
                            if self._smallq_decode_block_table is not None
                            else None
                        )
                        f_sl = (
                            self._smallq_decode_seq_lens[:ntok].clone()
                            if self._smallq_decode_seq_lens is not None
                            else None
                        )
                        f_scalars = (
                            md.max_seq_len,
                            getattr(md, "smallq_decode_max_seq_len_hint", None),
                            getattr(
                                md, "smallq_decode_workspace_seq_capacity_hint", None
                            ),
                            getattr(md, "smallq_decode_partition_size_hint", None),
                        )
                        md2 = _pxa_fa_orig_build(
                            self, common_prefix_len, common_attn_metadata, *a, **kw
                        )
                        t_scalars = (
                            md2.max_seq_len,
                            getattr(md2, "smallq_decode_max_seq_len_hint", None),
                            getattr(
                                md2, "smallq_decode_workspace_seq_capacity_hint", None
                            ),
                            getattr(md2, "smallq_decode_partition_size_hint", None),
                        )
                        if f_scalars != t_scalars:
                            _pxa_mc_vlog(
                                f"FA SCALAR MISMATCH fast={f_scalars} "
                                f"truth={t_scalars}"
                            )
                        if f_bt is not None and not _pxa_torch4.equal(
                            f_bt, self._smallq_decode_block_table[:ntok]
                        ):
                            _pxa_mc_vlog("FA MISMATCH smallq_decode_block_table")
                        if f_sl is not None and not _pxa_torch4.equal(
                            f_sl, self._smallq_decode_seq_lens[:ntok]
                        ):
                            _pxa_mc_vlog(
                                f"FA MISMATCH smallq_decode_seq_lens fast="
                                f"{f_sl.detach().cpu().tolist()[:16]} truth="
                                f"{self._smallq_decode_seq_lens[:ntok].detach().cpu().tolist()[:16]}"
                            )
                        for fname in (
                            "query_start_loc", "seq_lens", "block_table",
                            "slot_mapping",
                        ):
                            if getattr(md, fname).data_ptr() != getattr(
                                md2, fname
                            ).data_ptr():
                                _pxa_mc_vlog(f"FA PTR MISMATCH {fname}")
                        if st["fa_hits"] % 512 == 1:
                            _pxa_mc_vlog(f"FA checked hit #{st['fa_hits']}")
                        self._pxa_meta_cache = (_pxa_fa_key(self, m), md2)
                        return md2
                except Exception as exc:
                    _pxa_mc_disarm(exc)
            md = _pxa_fa_orig_build(
                self, common_prefix_len, common_attn_metadata, *a, **kw
            )
            try:
                snap = None
                if eligible and not st["disabled"]:
                    if (
                        not md.use_cascade
                        and getattr(md, "ddtree_parent_ids", None) is None
                        and md.query_start_loc.data_ptr()
                        == m.query_start_loc.data_ptr()
                        and md.seq_lens.data_ptr() == m.seq_lens.data_ptr()
                        and md.block_table.data_ptr()
                        == m.block_table_tensor.data_ptr()
                        and md.slot_mapping.data_ptr() == m.slot_mapping.data_ptr()
                    ):
                        snap = (_pxa_fa_key(self, m), md)
                self._pxa_meta_cache = snap
            except Exception:
                self._pxa_meta_cache = None
            return md

        _PXA_FAB.build = _pxa_fa_build
        print("VLLM_SM70_META_CACHE armed", file=_pxa_sys4.stderr, flush=True)
    except Exception:
        pass

# ---------------------------------------------------------------------------
# VLLM_SM70_DRAFT_GREEDY_FAST: env-gated drafter greedy-sample fast path.
# The MTP proposer's per-round sample calls (first_sample + loopN_sample,
# ~2.4 ms each) run model.compute_logits: full lm_head matmul + TP all-gather
# of the whole [rows, vocab] logits tensor + vocab trim, then argmax. When the
# batch is all-greedy with no penalties/logprobs/bad-words (same gate as
# PXQ4_GREEDY_VERIFY_FAST) the draft token only needs the argmax, so use the
# existing vocab-parallel local-argmax reduction (model.get_top_tokens):
# each TP rank argmaxes its own fp16 logits shard and only (value, index)
# pairs are gathered — communication O(2*tp) instead of O(vocab). Tie-breaks
# resolve to the lowest global vocab index on both paths, so the chosen draft
# token is identical; the target verifies every draft token regardless.
# Default OFF; enable with VLLM_SM70_DRAFT_GREEDY_FAST=1. Falls back to the
# stock path for any condition it does not cover, and on any error arms off.
# ---------------------------------------------------------------------------
import os as _pxa_os5

if _pxa_os5.getenv("VLLM_SM70_DRAFT_GREEDY_FAST", "") not in ("", "0"):
    try:
        import sys as _pxa_sys5

        from vllm.v1.spec_decode import llm_base_proposer as _pxa_lbp

        _pxa_orig_sample_draft = _pxa_lbp.SpecDecodeBaseProposer._sample_draft_tokens
        _pxa_dgf_state = {"disabled": False, "hits": 0}

        def _pxa_dgf_sample(
            self, hidden_states, sampling_metadata, logits=None, spec_step_idx=0
        ):
            st = _pxa_dgf_state
            if (
                st["disabled"]
                or logits is not None
                or self._static_draft_vocab is not None
                or not sampling_metadata.all_greedy
                or sampling_metadata.max_num_logprobs is not None
                or getattr(sampling_metadata, "logprob_token_ids", None)
                or getattr(sampling_metadata, "bad_words_token_ids", None)
                or getattr(sampling_metadata, "allowed_token_ids_mask", None)
                is not None
                or not getattr(sampling_metadata, "no_penalties", False)
                or not hasattr(self.model, "get_top_tokens")
                or getattr(self.model, "draft_id_to_target_id", None) is not None
            ):
                return _pxa_orig_sample_draft(
                    self, hidden_states, sampling_metadata, logits, spec_step_idx
                )
            try:
                token_ids = self._get_top_tokens_for_step(hidden_states, spec_step_idx)
                st["hits"] += 1
                if st["hits"] == 1:
                    print(
                        "VLLM_SM70_DRAFT_GREEDY_FAST active (first hit)",
                        file=_pxa_sys5.stderr,
                        flush=True,
                    )
                return token_ids, None
            except Exception as exc:
                st["disabled"] = True
                print(
                    f"VLLM_SM70_DRAFT_GREEDY_FAST disabled after error: {exc!r}",
                    file=_pxa_sys5.stderr,
                    flush=True,
                )
                return _pxa_orig_sample_draft(
                    self, hidden_states, sampling_metadata, logits, spec_step_idx
                )

        _pxa_lbp.SpecDecodeBaseProposer._sample_draft_tokens = _pxa_dgf_sample
        print("VLLM_SM70_DRAFT_GREEDY_FAST armed", file=_pxa_sys5.stderr, flush=True)
    except Exception:
        pass
