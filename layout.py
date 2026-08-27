import json, struct, glob, sys
import numpy as np
sys.path.insert(0, "<local-path>")
from gguf import GGUFReader

IDX = {}
def build_index():
    for fn in sorted(glob.glob("<local-path>*-of-00131.safetensors")):
        try:
            with open(fn, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(n))
        except Exception:
            continue
        for k, v in hdr.items():
            if k != "__metadata__":
                IDX[k] = (fn, 8 + n, v)
build_index()

def hf(name):
    if name not in IDX: return None
    fn, base, info = IDX[name]
    s, e = info["data_offsets"]
    with open(fn, "rb") as f:
        f.seek(base + s); raw = f.read(e - s)
    u16 = np.frombuffer(raw, dtype="<u2")
    return (u16.astype(np.uint32) << 16).view(np.float32).reshape(info["shape"])

r = GGUFReader("<local-path>")
GG = {t.name: t for t in r.tensors}
def gguf(name):
    t = GG.get(name)
    if t is None: return None, None
    from gguf.constants import GGMLQuantizationType
    ty = t.tensor_type.name if hasattr(t.tensor_type, "name") else str(t.tensor_type)
    d = t.data
    if t.tensor_type == GGMLQuantizationType.BF16:
        u16 = np.frombuffer(d.tobytes(), dtype="<u2")
        return (u16.astype(np.uint32) << 16).view(np.float32), ty
    return np.array(d, dtype=np.float32), ty

PERM48 = np.arange(48).reshape(16, 3).T.ravel()

def report(label, g, cands):
    if g is None:
        print("  %-34s GGUF tensor absent" % label); return
    best = None
    for lbl, c in cands:
        if c is None: continue
        if c.shape != g.shape:
            print("  %-34s %-18s SHAPE %s vs gguf %s" % (label, lbl, c.shape, g.shape)); continue
        d = float(np.max(np.abs(c - g)))
        if best is None or d < best[1]: best = (lbl, d)
        print("  %-34s %-18s max|diff| = %.3e" % (label, lbl, d))
    if best: print("      -> BEST: %s (%.3e)" % best)
    print()

print("=== F32 tensors: exact checks ===\n")

# 1. ssm_alpha / ssm_beta  [48,2560] -> gguf 2560x48
for gn, hn in [("blk.0.ssm_alpha.weight", "model.language_model.layers.0.linear_attn.in_proj_a.weight"),
               ("blk.0.ssm_beta.weight",  "model.language_model.layers.0.linear_attn.in_proj_b.weight")]:
    g, ty = gguf(gn); h = hf(hn)
    if g is None or h is None: print("  skip", gn); continue
    g = g.reshape(48, 2560)
    report("%s [%s]" % (gn, ty), g, [("identity", h), ("perm48 dim0", h[PERM48])])

# 2. ssm_conv1d  [10240,1,4] -> gguf 4x10240  (q 2048 | k 2048 | v 6144)
g, ty = gguf("blk.0.ssm_conv1d.weight"); h = hf("model.language_model.layers.0.linear_attn.conv1d.weight")
if g is not None and h is not None:
    g = g.reshape(10240, 4); h = h.reshape(10240, 4)
    def permute_qkv(x, pq, pk, pv):
        q, k, v = x[:2048], x[2048:4096], x[4096:]
        if pq: q = q.reshape(16, 128)[np.arange(16)].reshape(2048, -1) if False else q
        if pv: v = v.reshape(48, 128, -1)[PERM48].reshape(6144, -1)
        if pk: k = k.reshape(16, 128, -1).reshape(2048, -1)
        return np.concatenate([q, k, v], 0)
    report("blk.0.ssm_conv1d [%s]" % ty, g, [
        ("identity",      h),
        ("perm v-heads",  permute_qkv(h, False, False, True)),
    ])

# 3. indexer: index_qk_proj [640,2560] -> q_proj 2560x512 + k_proj 2560x128
gq, tq = gguf("blk.3.indexer.q_proj.weight"); gk, tk = gguf("blk.3.indexer.k_proj.weight")
h = hf("model.language_model.layers.3.self_attn.indexer.index_qk_proj.weight")
if h is not None and gq is not None:
    gq = gq.reshape(512, 2560); gk = gk.reshape(128, 2560)
    report("blk.3.indexer.q_proj [%s]" % tq, gq, [("qk[0:512]", h[:512]), ("qk[128:640]", h[128:640])])
    report("blk.3.indexer.k_proj [%s]" % tk, gk, [("qk[512:640]", h[512:640]), ("qk[0:128]", h[:128])])

# 4. attn q/k norms (per-head-dim, expect identity)
for gn, hn in [("blk.3.attn_q_norm.weight", "model.language_model.layers.3.self_attn.q_norm.weight"),
               ("blk.3.attn_k_norm.weight", "model.language_model.layers.3.self_attn.k_norm.weight"),
               ("blk.3.indexer.q_norm.weight", "model.language_model.layers.3.self_attn.indexer.q_layernorm.weight"),
               ("blk.3.indexer.k_norm.weight", "model.language_model.layers.3.self_attn.indexer.k_layernorm.weight")]:
    g, ty = gguf(gn); h = hf(hn)
    if g is None or h is None: print("  skip %s" % gn); continue
    report("%s [%s]" % (gn, ty), g.ravel(), [("identity", h.ravel())])

# 5. hyper-connection F32 pieces
for gn, hn in [("blk.0.hc_attn_norm.weight",   "model.language_model.layers.0.attn_hyper_connection.hc_norm.weight"),
               ("blk.0.hc_attn_inject.weight", "model.language_model.layers.0.attn_hyper_connection.block_inject_weight.weight"),
               ("blk.0.hc_ffn_norm.weight",    "model.language_model.layers.0.mlp_hyper_connection.hc_norm.weight"),
               ("blk.1.ple_norm_key.weight",   "model.language_model.layers.1.ple.norm_key.weight"),
               ("blk.1.ple_conv1d.weight",     "model.language_model.layers.1.ple.conv1d.weight")]:
    g, ty = gguf(gn); h = hf(hn)
    if g is None or h is None: print("  skip %s" % gn); continue
    report("%s [%s]" % (gn, ty), g.ravel(), [("identity", h.ravel())])

# 6. MoE router gate (F32 in gguf)
g, ty = gguf("blk.0.ffn_gate_inp.weight"); h = hf("model.language_model.layers.0.mlp.gate.weight")
if g is not None and h is not None:
    report("blk.0.ffn_gate_inp [%s]" % ty, g.reshape(512, 2560), [("identity", h)])
g, ty = gguf("blk.0.ffn_gate_inp_shexp.weight"); h = hf("model.language_model.layers.0.mlp.shared_expert_gate.weight")
if g is not None and h is not None:
    report("blk.0.ffn_gate_inp_shexp [%s]" % ty, g.ravel(), [("identity", h.ravel())])
