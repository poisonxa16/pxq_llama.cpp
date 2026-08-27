#!/usr/bin/env python3
"""Convert Qwen4Exp (qwen4_exp) HF safetensors to a qwen4exp GGUF.

Scope: the text model only, matching what the engine loads. The vision tower
(model.visual.*) and the multi-token-prediction module (mtp.*) are skipped and
counted; a vision projector would be a separate mmproj file.

Layout rules below were established by byte-comparison against a known-good
reference GGUF, not by inference. See --explain.

Memory: per_layer_token_embd is 128 separate checkpoint shards totalling ~95 GiB
in bf16. They are NEVER concatenated in RAM -- a streaming writer feeds them to
the output file one shard at a time. Concatenating them is what took the host
down once already; the reference implementation's own comment notes the
concatenation peaks near 300 GB of RSS.
"""

import argparse, json, os, struct, sys, glob
from pathlib import Path
import numpy as np

# ---------------------------------------------------------------- layout rules
# v-heads: HF stores 48 value heads grouped k-head-major (16 groups of 3);
# the engine wants the transpose. gguf[i] = hf[PERM48[i]].
PERM48 = np.arange(48).reshape(16, 3).T.ravel()

# Every RMSNorm weight is stored by HF as (w - 1): the module computes x*(1+w).
# The engine applies a plain x*w, so the converter must add 1.0.
# linear_attn.norm is the sole exception -- it already stores the full weight.
NORM_PLUS_ONE = True


def bf16_bits_to_f32(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16_bits(f: np.ndarray) -> np.ndarray:
    u32 = np.ascontiguousarray(f, dtype=np.float32).view(np.uint32)
    # round-to-nearest-even on the truncated 16 bits
    rounded = (u32 + 0x7FFF + ((u32 >> 16) & 1)) >> 16
    return rounded.astype(np.uint16)


class SafeTensorsStore:
    """Random access to tensors across many safetensors shards, lazily mmapped."""

    def __init__(self, folder: Path):
        self.index = {}          # tensor name -> (path, data_start, info)
        self._maps = {}
        files = sorted(folder.glob("model-*-of-*.safetensors"))
        if not files:
            raise SystemExit(f"no safetensors shards under {folder}")
        for fn in files:
            with open(fn, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(n))
            for k, v in hdr.items():
                if k == "__metadata__":
                    continue
                self.index[k] = (fn, 8 + n, v)
        self.n_files = len(files)

    def _mm(self, path):
        m = self._maps.get(path)
        if m is None:
            m = np.memmap(path, dtype=np.uint8, mode="r")
            self._maps[path] = m
        return m

    def has(self, name): return name in self.index

    def raw_u16(self, name):
        """bf16 tensor as a shaped uint16 view -- no copy, no dtype conversion."""
        path, base, info = self.index[name]
        if info["dtype"] != "BF16":
            raise TypeError(f"{name} is {info['dtype']}, expected BF16")
        s, e = info["data_offsets"]
        buf = self._mm(path)[base + s: base + e]
        return np.frombuffer(buf, dtype="<u2").reshape(info["shape"])

    def f32(self, name):
        return bf16_bits_to_f32(self.raw_u16(name))

    def i64(self, name):
        path, base, info = self.index[name]
        assert info["dtype"] in ("I64", "INT64"), (name, info["dtype"])
        s, e = info["data_offsets"]
        buf = self._mm(path)[base + s: base + e]
        return np.frombuffer(buf, dtype="<i8").reshape(info["shape"])

    def shape(self, name): return tuple(self.index[name][2]["shape"])


class ShardStream:
    """Duck-types just enough of np.ndarray for GGUFWriter.write_tensors_to_file:
    it exposes .nbytes and .tofile(), streaming N shards back-to-back."""

    def __init__(self, store, names, nbytes):
        self.store = store
        self.names = names
        self.nbytes = nbytes

    def tofile(self, fout):
        written = 0
        for i, nm in enumerate(self.names):
            a = self.store.raw_u16(nm)
            a.tofile(fout)
            written += a.nbytes
            if (i + 1) % 16 == 0 or i + 1 == len(self.names):
                print(f"      per_layer_token_embd: shard {i+1}/{len(self.names)} "
                      f"({written / 2**30:.1f} GiB)", flush=True)
        assert written == self.nbytes, (written, self.nbytes)



def add_vocab(w, model_dir, n_vocab, tcfg_text, gguf):
    """GPT-2 style BPE vocab out of tokenizer.json."""
    tj = json.loads((model_dir / "tokenizer.json").read_text(encoding="utf-8"))
    tok_cfg = {}
    p = model_dir / "tokenizer_config.json"
    if p.exists():
        tok_cfg = json.loads(p.read_text(encoding="utf-8"))

    model = tj["model"]
    vocab = model["vocab"]
    id2tok = {v: k for k, v in vocab.items()}
    added = {int(a["id"]): a for a in tj.get("added_tokens", [])}

    T = gguf.TokenType
    tokens, toktypes = [], []
    n_pad = 0
    for i in range(n_vocab):
        if i in added:
            tokens.append(added[i]["content"])
            toktypes.append(T.CONTROL if added[i].get("special") else T.USER_DEFINED)
        elif i in id2tok:
            tokens.append(id2tok[i]); toktypes.append(T.NORMAL)
        else:
            tokens.append(f"[PAD{i}]"); toktypes.append(T.UNUSED); n_pad += 1

    merges_raw = model.get("merges", [])
    merges = []
    for m in merges_raw:
        merges.append(m if isinstance(m, str) else " ".join(m))

    w.add_tokenizer_model("gpt2")
    # Must name a pre-tokenizer the engine actually implements; the reference
    # build of this model uses the qwen35 split regex.
    w.add_tokenizer_pre("qwen35")
    w.add_token_list(tokens)
    w.add_token_types(toktypes)
    w.add_token_merges(merges)

    eos = tok_cfg.get("eos_token")
    if isinstance(eos, dict): eos = eos.get("content")
    eos_id = vocab.get(eos) if eos else None
    if eos_id is None:
        eos_id = next((i for i, a in added.items() if a["content"] == eos), None)
    w.add_eos_token_id(eos_id if eos_id is not None else tcfg_text["eos_token_id"])
    w.add_bos_token_id(tcfg_text["bos_token_id"])
    w.add_pad_token_id(tcfg_text.get("pad_token_id") or tcfg_text["bos_token_id"])
    w.add_add_bos_token(bool(tok_cfg.get("add_bos_token", False)))

    ct = tok_cfg.get("chat_template")
    if ct is None and (model_dir / "chat_template.jinja").exists():
        ct = (model_dir / "chat_template.jinja").read_text(encoding="utf-8")
    if ct:
        w.add_chat_template(ct)

    print(f"vocab: {len(tokens)} tokens ({n_pad} pad), {len(merges)} merges, "
          f"eos={eos_id}, chat_template={'yes' if ct else 'NO'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--outfile", type=Path, required=True)
    ap.add_argument("--split-max-size", type=int, default=0,
                    help="bytes per output shard (0 = single file)")
    ap.add_argument("--kv-only", type=Path, default=None,
                    help="build only the KV header and diff it against this reference GGUF")
    ap.add_argument("--dry-run", action="store_true",
                    help="build the full tensor plan and KV header, write nothing")
    ap.add_argument("--gguf-py", type=Path, default=None)
    ap.add_argument("--tokenizer-dir", type=Path, default=None,
                    help="where tokenizer.json lives (default: model_dir)")
    ap.add_argument("--name", default="Qwen3.8 Flash Next")
    ap.add_argument("--size-label", default="56B")
    args = ap.parse_args()

    sys.path.insert(0, str(args.gguf_py or (Path(__file__).parent / "gguf-py")))
    import gguf
    from gguf.constants import GGMLQuantizationType

    cfg = json.loads((args.model_dir / "config.json").read_text())
    t = cfg["text_config"]

    n_layer   = t["num_hidden_layers"]
    n_embd    = t["hidden_size"]
    hc        = t["hc_count"]
    hc_dim    = hc * n_embd
    n_v_heads = t["linear_num_value_heads"]
    v_head_d  = t["linear_value_head_dim"]
    layer_types = t["layer_types"]
    assert len(layer_types) == n_layer
    assert n_v_heads == 48, "PERM48 is derived for 48 value heads"

    store = SafeTensorsStore(args.model_dir)
    print(f"indexed {len(store.index)} tensors across {store.n_files} shards")

    # ---- locate the PLE module by where its tensors actually live.
    # config's ple_layer_ids is NOT the module index (it says [2]; the module is
    # at layers.1, which is also what the reference GGUF records).
    ple_layers = sorted({
        int(k.split(".")[3]) for k in store.index
        if k.startswith("model.language_model.layers.") and ".ple." in k
    })
    if len(ple_layers) != 1:
        raise SystemExit(f"expected exactly one PLE layer, found {ple_layers}")
    ple_il = ple_layers[0]
    print(f"PLE module found at layer {ple_il} "
          f"(config ple_layer_ids={t.get('ple_layer_ids')} -- not the module index)")

    P = f"model.language_model.layers.{ple_il}.ple."
    mult   = store.i64(P + "ple_embedding.layer_multipliers")
    offs   = store.i64(P + "ple_embedding.ngram_heads_offsets")
    vocabs = store.i64(P + "ple_embedding.ngram_heads_vocab_sizes")

    shard_names = []
    i = 0
    while store.has(P + f"ple_embedding.ngram_embedding.shard_{i}.weight"):
        shard_names.append(P + f"ple_embedding.ngram_embedding.shard_{i}.weight")
        i += 1
    n_ple_shards = len(shard_names)
    if n_ple_shards != t.get("split_ngram_parts", n_ple_shards):
        raise SystemExit(f"found {n_ple_shards} ngram shards, config says "
                         f"{t.get('split_ngram_parts')}")
    ple_rows = sum(store.shape(s)[0] for s in shard_names)
    ple_dim  = store.shape(shard_names[0])[1]
    print(f"PLE table: {n_ple_shards} shards, {ple_rows} rows x {ple_dim} "
          f"({ple_rows * ple_dim * 2 / 2**30:.1f} GiB bf16, streamed)")

    w = gguf.GGUFWriter(path=None, arch="qwen4exp",
                        split_max_size=args.split_max_size)

    # KV keys and their GGUF types are written literally, matched against a
    # known-good reference header. The bundled constants.py is older than
    # gguf_writer.py in this tree (several add_* helpers reference Keys entries
    # that do not exist), so the helpers are not safe to rely on here.
    U32, I32, F32V, STR, ARR = (gguf.GGUFValueType.UINT32, gguf.GGUFValueType.INT32,
                                gguf.GGUFValueType.FLOAT32, gguf.GGUFValueType.STRING,
                                gguf.GGUFValueType.ARRAY)
    U64 = gguf.GGUFValueType.UINT64

    def kv(key, val, vtype, sub=None):
        w.add_key_value(key, val, vtype, sub_type=sub)

    A = "qwen4exp."
    kv("general.type", "model", STR)
    kv("general.name", args.name, STR)
    kv("general.size_label", f"{t['num_experts']}x{args.size_label}", STR)

    kv(A + "block_count", n_layer, U32)
    kv(A + "context_length", t["max_position_embeddings"], U32)
    kv(A + "embedding_length", n_embd, U32)
    kv(A + "attention.head_count", t["num_attention_heads"], U32)
    kv(A + "attention.head_count_kv", t["num_key_value_heads"], U32)
    kv(A + "attention.key_length", t["head_dim"], U32)
    kv(A + "attention.value_length", t["head_dim"], U32)
    kv(A + "attention.layer_norm_rms_epsilon", t["rms_norm_eps"], F32V)
    kv(A + "expert_count", t["num_experts"], U32)
    kv(A + "expert_used_count", t["num_experts_per_tok"], U32)
    kv(A + "expert_feed_forward_length", t["moe_intermediate_size"], U32)
    kv(A + "expert_shared_feed_forward_length", t["shared_expert_intermediate_size"], U32)

    rp = t["rope_parameters"]
    kv(A + "rope.freq_base", rp["rope_theta"], F32V)
    kv(A + "rope.dimension_count", int(t["head_dim"] * rp["partial_rotary_factor"]), U32)
    kv(A + "rope.dimension_sections", list(rp["mrope_section"]) + [0], ARR, I32)

    kv(A + "ssm.conv_kernel", t["linear_conv_kernel_dim"], U32)
    kv(A + "ssm.state_size", t["linear_key_head_dim"], U32)
    kv(A + "ssm.group_count", t["linear_num_key_heads"], U32)
    kv(A + "ssm.time_step_rank", n_v_heads, U32)
    kv(A + "ssm.inner_size", n_v_heads * v_head_d, U32)

    kv(A + "full_attention_interval", t["full_attention_interval"], U32)
    kv(A + "hyper_connection.count", hc, U32)
    kv(A + "hyper_connection.low_rank", t["hc_lowrank"], U32)

    kv(A + "attention.indexer.head_count", t["indexer_n_heads"], U32)
    kv(A + "attention.indexer.key_length", t["indexer_head_dim"], U32)
    kv(A + "attention.indexer.top_k", t["indexer_budget"], U32)
    kv(A + "attention.compress_ratios",
       [t["indexer_compress_ratio"] if lt == "full_attention" else 0 for lt in layer_types],
       ARR, I32)

    kv(A + "embedding_length_per_layer_input", ple_dim, U32)
    kv(A + "ple.layers", [ple_il], ARR, I32)
    kv(A + "ple.ngram_size", t["ngram_size"], U32)
    kv(A + "ple.heads_per_ngram", t["heads_per_ngram"], U32)
    kv(A + "ple.conv_kernel", t["ple_conv_kernel_size"], U32)
    kv(A + "ple.eos_token_id", t["eos_token_id"], U32)
    kv(A + "ple.image_token_id", cfg["image_token_id"], U32)
    # UINT64 arrays, read straight from the checkpoint -- not derived. The
    # engine's loader reads these as uint64; both get_arr overloads used to
    # throw on that type, which is why this arch could not load at all.
    kv(A + "ple.layer_multipliers", [int(x) for x in mult], ARR, U64)
    kv(A + "ple.head_offsets", [int(x) for x in offs], ARR, U64)
    kv(A + "ple.head_vocab_sizes", [int(x) for x in vocabs], ARR, U64)

    # ------------------------------------------------------------------ vocab
    add_vocab(w, args.tokenizer_dir or args.model_dir, t["vocab_size"], t, gguf)

    w.add_quantization_version(2)
    w.add_file_type(gguf.LlamaFileType.MOSTLY_BF16)

    if args.kv_only is not None:
        ref = gguf.GGUFReader(str(args.kv_only))
        mine = {}
        for d in w.kv_data:
            for k, v in d.items():
                mine[k] = v.value
        skip = ("general.quantization_version", "general.file_type",
                "general.description", "general.quantized_by", "general.repo_url",
                "general.tags", "split.", "quantize.imatrix", "general.sampling")
        same = diff = missing = 0
        print("\n=== KV header vs %s ===" % args.kv_only.name)
        for k, f in ref.fields.items():
            if k.startswith("GGUF.") or any(k.startswith(x) for x in skip):
                continue
            if k.startswith("tokenizer.") and k not in (
                    "tokenizer.ggml.model", "tokenizer.ggml.pre",
                    "tokenizer.ggml.eos_token_id", "tokenizer.ggml.bos_token_id",
                    "tokenizer.ggml.padding_token_id", "tokenizer.ggml.add_bos_token"):
                continue
            if k not in mine:
                print("  MISSING   %s" % k); missing += 1; continue
            rv = f.contents()
            mv = mine[k]
            if isinstance(rv, (list, tuple)) or isinstance(mv, (list, tuple)):
                a = [int(x) if isinstance(x, (int, np.integer)) else x for x in (rv or [])]
                b = [int(x) if isinstance(x, (int, np.integer)) else x for x in (mv or [])]
                eq = a == b
            elif isinstance(rv, float) or isinstance(mv, float):
                eq = abs(float(rv) - float(mv)) <= 1e-9 * max(1.0, abs(float(rv)))
            else:
                eq = str(rv) == str(mv)
            if eq:
                same += 1
            else:
                diff += 1
                print("  DIFFERS   %-46s ref=%s  mine=%s" % (k, str(rv)[:40], str(mv)[:40]))
        extra = [k for k in mine if k not in ref.fields and not any(k.startswith(x) for x in skip)]
        for k in extra:
            print("  EXTRA     %-46s mine=%s" % (k, str(mine[k])[:40]))
        print("\n  matched %d, differ %d, missing %d, extra %d" % (same, diff, missing, len(extra)))
        return

    # ------------------------------------------------------------- tensors
    stats = {"f32": 0, "bf16": 0, "skipped_visual": 0, "skipped_mtp": 0}
    claimed = set()

    def add_f32(name, arr):
        a = np.ascontiguousarray(arr, dtype=np.float32)
        w.add_tensor(name, a)
        stats["f32"] += 1

    def add_bf16(name, u16):
        a = np.ascontiguousarray(u16)
        # NB: write_ti_data_to_file reverses shape itself -- pass numpy order.
        w.add_tensor_info(name, a.shape, np.dtype(np.uint16),
                          a.nbytes, raw_dtype=GGMLQuantizationType.BF16)
        w.tensors[-1][name].tensor = a
        stats["bf16"] += 1

    def src(name):
        claimed.add(name)
        return name

    def norm(name, hf_name, plus_one=True):
        v = store.f32(src(hf_name)).astype(np.float32)
        add_f32(name, v + 1.0 if (plus_one and NORM_PLUS_ONE) else v)

    LM = "model.language_model."

    # globals
    for cand in (LM + "embed_tokens.weight", "model.embed_tokens.weight"):
        if store.has(cand):
            add_bf16("token_embd.weight", store.raw_u16(src(cand))); break
    else:
        raise SystemExit("embed_tokens not found")
    for cand in ("lm_head.weight", LM + "lm_head.weight"):
        if store.has(cand):
            add_bf16("output.weight", store.raw_u16(src(cand))); break
    else:
        raise SystemExit("lm_head not found")

    HCM = LM + "hyper_connection_mixer."
    norm("output_hc_norm.weight", HCM + "hc_norm.weight")
    add_bf16("output_hc_down.weight", store.raw_u16(src(HCM + "input_mix_weight_down.weight")))
    add_bf16("output_hc_up.weight",   store.raw_u16(src(HCM + "input_mix_weight_up.weight")))

    # the streamed PLE table
    ple_nbytes = ple_rows * ple_dim * 2
    w.add_tensor_info("per_layer_token_embd.weight", (ple_rows, ple_dim),
                      np.dtype(np.uint16), ple_nbytes,
                      raw_dtype=GGMLQuantizationType.BF16)
    w.tensors[-1]["per_layer_token_embd.weight"].tensor = ShardStream(
        store, [src(s) for s in shard_names], ple_nbytes)
    stats["bf16"] += 1

    for il in range(n_layer):
        L = f"{LM}layers.{il}."
        B = f"blk.{il}."

        for hf_pfx, gg_pfx in (("attn_hyper_connection.", "hc_attn_"),
                               ("mlp_hyper_connection.",  "hc_ffn_")):
            norm(B + gg_pfx + "norm.weight", L + hf_pfx + "hc_norm.weight")
            add_bf16(B + gg_pfx + "down.weight",
                     store.raw_u16(src(L + hf_pfx + "input_mix_weight_down.weight")))
            add_bf16(B + gg_pfx + "up.weight",
                     store.raw_u16(src(L + hf_pfx + "input_mix_weight_up.weight")))
            add_f32(B + gg_pfx + "inject.weight",
                    store.f32(src(L + hf_pfx + "block_inject_weight.weight")))

        # MoE
        add_f32(B + "ffn_gate_inp.weight", store.f32(src(L + "mlp.gate.weight")))
        add_f32(B + "ffn_gate_inp_shexp.weight",
                store.f32(src(L + "mlp.shared_expert_gate.weight")).reshape(-1))
        gu = store.raw_u16(src(L + "mlp.experts.gate_up_proj"))
        half = gu.shape[1] // 2
        add_bf16(B + "ffn_gate_exps.weight", gu[:, :half, :])
        add_bf16(B + "ffn_up_exps.weight",   gu[:, half:, :])
        add_bf16(B + "ffn_down_exps.weight", store.raw_u16(src(L + "mlp.experts.down_proj")))
        for a, b in (("gate", "gate"), ("up", "up"), ("down", "down")):
            add_bf16(B + f"ffn_{a}_shexp.weight",
                     store.raw_u16(src(L + f"mlp.shared_expert.{b}_proj.weight")))

        if layer_types[il] == "linear_attention":
            A = L + "linear_attn."
            qkv = store.raw_u16(src(A + "in_proj_qkv.weight"))
            n_kv = t["linear_num_key_heads"] * t["linear_key_head_dim"]
            v = qkv[2 * n_kv:].reshape(n_v_heads, v_head_d, -1)[PERM48].reshape(-1, qkv.shape[1])
            add_bf16(B + "attn_qkv.weight", np.concatenate([qkv[:2 * n_kv], v], 0))
            add_bf16(B + "attn_gate.weight",
                     store.raw_u16(src(A + "in_proj_z.weight"))
                          .reshape(n_v_heads, v_head_d, -1)[PERM48].reshape(-1, n_embd))
            out = store.raw_u16(src(A + "out_proj.weight"))
            add_bf16(B + "ssm_out.weight",
                     out.reshape(n_embd, n_v_heads, v_head_d)[:, PERM48, :].reshape(n_embd, -1))
            # A_log -> -exp(A_log), then permute
            add_f32(B + "ssm_a", (-np.exp(store.f32(src(A + "A_log")).astype(np.float32)))[PERM48])
            add_f32(B + "ssm_dt.bias", store.f32(src(A + "dt_bias")).astype(np.float32)[PERM48])
            add_f32(B + "ssm_alpha.weight", store.f32(src(A + "in_proj_a.weight"))[PERM48])
            add_f32(B + "ssm_beta.weight",  store.f32(src(A + "in_proj_b.weight"))[PERM48])
            conv = store.f32(src(A + "conv1d.weight")).reshape(-1, t["linear_conv_kernel_dim"])
            cv = conv[2 * n_kv:].reshape(n_v_heads, v_head_d, -1)[PERM48].reshape(-1, conv.shape[1])
            add_f32(B + "ssm_conv1d.weight", np.concatenate([conv[:2 * n_kv], cv], 0))
            # the sole norm that is NOT offset by one
            norm(B + "ssm_norm.weight", A + "norm.weight", plus_one=False)
        else:
            S = L + "self_attn."
            for gg, hf_n in (("attn_q", "q_proj"), ("attn_k", "k_proj"),
                             ("attn_v", "v_proj"), ("attn_output", "o_proj")):
                add_bf16(B + gg + ".weight", store.raw_u16(src(S + hf_n + ".weight")))
            norm(B + "attn_q_norm.weight", S + "q_norm.weight")
            norm(B + "attn_k_norm.weight", S + "k_norm.weight")
            qk = store.raw_u16(src(S + "indexer.index_qk_proj.weight"))
            n_iq = t["indexer_n_heads"] * t["indexer_head_dim"]
            add_bf16(B + "indexer.q_proj.weight", qk[:n_iq])
            add_bf16(B + "indexer.k_proj.weight", qk[n_iq:])
            norm(B + "indexer.q_norm.weight", S + "indexer.q_layernorm.weight")
            norm(B + "indexer.k_norm.weight", S + "indexer.k_layernorm.weight")

        if il == ple_il:
            add_bf16(B + "ple_key.weight",   store.raw_u16(src(P + "key_proj.weight")))
            add_bf16(B + "ple_value.weight", store.raw_u16(src(P + "value_proj.weight")))
            add_f32(B + "ple_conv1d.weight",
                    store.f32(src(P + "conv1d.weight")).reshape(-1, t["ple_conv_kernel_size"]))
            for nm in ("conv", "key", "query"):
                norm(B + f"ple_norm_{nm}.weight", P + f"norm_{nm}.weight")

    # ------------------------------------------------------- coverage report
    for k in store.index:
        if k.startswith("model.visual."):
            stats["skipped_visual"] += 1
        elif k.startswith("mtp."):
            stats["skipped_mtp"] += 1
    unclaimed = [k for k in store.index
                 if k not in claimed
                 and not k.startswith("model.visual.")
                 and not k.startswith("mtp.")]
    n_out = sum(len(d) for d in w.tensors)
    print(f"\ntensors written : {n_out}  (bf16 {stats['bf16']}, f32 {stats['f32']})")
    print(f"skipped         : {stats['skipped_visual']} vision (separate mmproj), "
          f"{stats['skipped_mtp']} mtp (not modelled by the engine)")
    if unclaimed:
        print(f"\nUNCONSUMED text tensors ({len(unclaimed)}) -- these would be "
              f"silently dropped, refusing:")
        for k in sorted(unclaimed)[:40]:
            print(f"   {k}  {store.shape(k)}")
        raise SystemExit("refusing to write an incomplete model")

    if args.dry_run:
        print("\n--dry-run: plan built, KV header complete, nothing written")
        return

    w.write_header_to_file(path=args.outfile)
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=True)
    w.close()
    print(f"\nwrote {args.outfile}")


if __name__ == "__main__":
    main()
