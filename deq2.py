import json, struct, glob, sys
import numpy as np
sys.path.insert(0, "<local-path>")
from gguf import GGUFReader
from gguf.constants import GGMLQuantizationType
from gguf.quants import dequantize

IDX = {}
for fn in sorted(glob.glob("<local-path>*-of-00131.safetensors")):
    try:
        with open(fn, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]; hdr = json.loads(f.read(n))
    except Exception: continue
    for k, v in hdr.items():
        if k != "__metadata__": IDX[k] = (fn, 8 + n, v)

def hf(name):
    if name not in IDX: return None
    fn, base, info = IDX[name]; s, e = info["data_offsets"]
    with open(fn, "rb") as f:
        f.seek(base + s); raw = f.read(e - s)
    u16 = np.frombuffer(raw, dtype="<u2")
    return (u16.astype(np.uint32) << 16).view(np.float32).reshape(info["shape"])

r = GGUFReader("<local-path>")
GG = {t.name: t for t in r.tensors}

def gdeq(name, expert_slice=None, n_experts=512):
    """Dequantize a tensor, optionally only one expert's contiguous byte slice."""
    t = GG.get(name)
    if t is None: return None, None
    ty = t.tensor_type
    d = t.data
    if expert_slice is not None:
        nb = d.shape[0]
        assert nb % n_experts == 0, (name, nb, n_experts)
        per = nb // n_experts
        d = d[expert_slice * per:(expert_slice + 1) * per]
    if ty == GGMLQuantizationType.F32:
        return np.array(d, dtype=np.float32), ty.name
    if ty == GGMLQuantizationType.BF16:
        u16 = np.frombuffer(d.tobytes(), dtype="<u2")
        return (u16.astype(np.uint32) << 16).view(np.float32), ty.name
    return dequantize(d, ty).astype(np.float32), ty.name

PERM48 = np.arange(48).reshape(16, 3).T.ravel()

def cmp(label, g, cands):
    print("  %s" % label, flush=True)
    best = None
    for lbl, c in cands:
        if c is None: continue
        cf = np.ascontiguousarray(c).reshape(-1)
        if cf.size != g.size:
            print("      %-26s SIZE %d vs %d" % (lbl, cf.size, g.size)); continue
        rel = float(np.abs(cf - g).mean()) / (float(np.abs(g).mean()) + 1e-12)
        if best is None or rel < best[1]: best = (lbl, rel)
        print("      %-26s rel_err = %8.4f" % (lbl, rel), flush=True)
    if best: print("      -> BEST: %s (%.4f)" % best, flush=True)
    print(flush=True)

# --- ssm_out: HF [2560,6144] ; permute along the 6144 INPUT dim
g, ty = gdeq("blk.0.ssm_out.weight"); h = hf("model.language_model.layers.0.linear_attn.out_proj.weight")
if g is not None and h is not None:
    cmp("blk.0.ssm_out [%s]" % ty, g.ravel(),
        [("identity", h), ("perm48 input dim", h.reshape(2560, 48, 128)[:, PERM48, :])])

# --- attn_gate: HF in_proj_z [6144,2560]
g, ty = gdeq("blk.0.attn_gate.weight"); h = hf("model.language_model.layers.0.linear_attn.in_proj_z.weight")
if g is not None and h is not None:
    cmp("blk.0.attn_gate [%s]" % ty, g.ravel(),
        [("identity", h), ("perm48 rows", h.reshape(48, 128, 2560)[PERM48])])

# --- attn_qkv: HF in_proj_qkv [10240,2560], permute only v rows (4096:)
g, ty = gdeq("blk.0.attn_qkv.weight"); h = hf("model.language_model.layers.0.linear_attn.in_proj_qkv.weight")
if g is not None and h is not None:
    v = h[4096:].reshape(48, 128, 2560)[PERM48].reshape(6144, 2560)
    cmp("blk.0.attn_qkv [%s]" % ty, g.ravel(),
        [("identity", h), ("perm48 v-rows only", np.concatenate([h[:4096], v], 0))])

# --- MoE expert split order (expert 0 only)
h = hf("model.language_model.layers.0.mlp.experts.gate_up_proj")
if h is not None:
    g, ty = gdeq("blk.0.ffn_gate_exps.weight", expert_slice=0)
    if g is not None:
        cmp("blk.0.ffn_gate_exps[e0] [%s]" % ty, g.ravel(),
            [("gate_up[0,:640,:]", h[0, :640, :]), ("gate_up[0,640:,:]", h[0, 640:, :])])
    g, ty = gdeq("blk.0.ffn_up_exps.weight", expert_slice=0)
    if g is not None:
        cmp("blk.0.ffn_up_exps[e0] [%s]" % ty, g.ravel(),
            [("gate_up[0,640:,:]", h[0, 640:, :]), ("gate_up[0,:640,:]", h[0, :640, :])])

h = hf("model.language_model.layers.0.mlp.experts.down_proj")
if h is not None:
    g, ty = gdeq("blk.0.ffn_down_exps.weight", expert_slice=0)
    if g is not None:
        cmp("blk.0.ffn_down_exps[e0] [%s]" % ty, g.ravel(),
            [("identity", h[0]), ("transposed", h[0].T)])

# --- full-attn q_proj (packed output gate): direct or de-interleaved?
g, ty = gdeq("blk.3.attn_q.weight"); h = hf("model.language_model.layers.3.self_attn.q_proj.weight")
if g is not None and h is not None:
    cmp("blk.3.attn_q [%s]" % ty, g.ravel(),
        [("identity", h),
         ("deinterleave 2x6144", h.reshape(6144, 2, 2560).transpose(1, 0, 2))])

for gn, hn in [("blk.3.attn_k.weight", "model.language_model.layers.3.self_attn.k_proj.weight"),
               ("blk.3.attn_v.weight", "model.language_model.layers.3.self_attn.v_proj.weight"),
               ("blk.3.attn_output.weight", "model.language_model.layers.3.self_attn.o_proj.weight"),
               ("blk.0.hc_attn_down.weight", "model.language_model.layers.0.attn_hyper_connection.input_mix_weight_down.weight"),
               ("blk.0.hc_attn_up.weight", "model.language_model.layers.0.attn_hyper_connection.input_mix_weight_up.weight"),
               ("blk.1.ple_key.weight", "model.language_model.layers.1.ple.key_proj.weight"),
               ("blk.1.ple_value.weight", "model.language_model.layers.1.ple.value_proj.weight"),
               ("blk.0.ffn_gate_shexp.weight", "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight"),
               ("blk.0.ffn_down_shexp.weight", "model.language_model.layers.0.mlp.shared_expert.down_proj.weight")]:
    g, ty = gdeq(gn); h = hf(hn)
    if g is None or h is None:
        print("  skip %s" % gn, flush=True); continue
    cmp("%s [%s]" % (gn, ty), g.ravel(), [("identity", h)])
