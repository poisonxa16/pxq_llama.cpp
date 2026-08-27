import json, struct, glob, sys
import numpy as np
sys.path.insert(0, "<local-path>")
from gguf import GGUFReader
from gguf.constants import GGMLQuantizationType

IDX = {}
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
    if t is None: return None
    if t.tensor_type == GGMLQuantizationType.BF16:
        u16 = np.frombuffer(t.data.tobytes(), dtype="<u2")
        return (u16.astype(np.uint32) << 16).view(np.float32)
    return np.array(t.data, dtype=np.float32)

PAIRS = [
    ("blk.3.attn_q_norm.weight",     "model.language_model.layers.3.self_attn.q_norm.weight"),
    ("blk.3.attn_k_norm.weight",     "model.language_model.layers.3.self_attn.k_norm.weight"),
    ("blk.3.indexer.q_norm.weight",  "model.language_model.layers.3.self_attn.indexer.q_layernorm.weight"),
    ("blk.3.indexer.k_norm.weight",  "model.language_model.layers.3.self_attn.indexer.k_layernorm.weight"),
    ("blk.0.hc_attn_norm.weight",    "model.language_model.layers.0.attn_hyper_connection.hc_norm.weight"),
    ("blk.0.hc_ffn_norm.weight",     "model.language_model.layers.0.mlp_hyper_connection.hc_norm.weight"),
    ("blk.1.ple_norm_key.weight",    "model.language_model.layers.1.ple.norm_key.weight"),
    ("blk.1.ple_norm_conv.weight",   "model.language_model.layers.1.ple.norm_conv.weight"),
    ("blk.1.ple_norm_query.weight",  "model.language_model.layers.1.ple.norm_query.weight"),
    ("output_hc_norm.weight",        "model.language_model.hyper_connection_mixer.hc_norm.weight"),
    ("blk.0.ssm_norm.weight",        "model.language_model.layers.0.linear_attn.norm.weight"),
]
print("%-32s %12s %12s %12s   verdict" % ("tensor", "max|g-h|", "max|g-(h+1)|", "n_elem"))
for gn, hn in PAIRS:
    g = gguf(gn); h = hf(hn)
    if g is None: print("%-32s  GGUF absent" % gn); continue
    if h is None: print("%-32s  HF absent" % gn); continue
    g = g.ravel(); h = h.ravel().astype(np.float32)
    if g.shape != h.shape:
        print("%-32s  SHAPE %s vs %s" % (gn, g.shape, h.shape)); continue
    d0 = float(np.max(np.abs(g - h)))
    d1 = float(np.max(np.abs(g - (h + 1.0))))
    verdict = "PLUS-ONE" if d1 < d0 and d1 < 1e-5 else ("identity" if d0 < 1e-5 else "NEITHER")
    print("%-32s %12.3e %12.3e %12d   %s" % (gn, d0, d1, g.size, verdict))
