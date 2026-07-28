#!/usr/bin/env python3
# Graft the 35B MTP layer (ALL blk.<N>.* incl nextn.*) from a donor GGUF onto a base GGUF.
# Parameterized version of the proven graft_mtp_35b_sft.py. block_count N -> N+1,
# nextn_predict_layers=1. Usage: graft_mtp_generic.py BASE DONOR OUT [mtp_blk=40]
import sys
sys.path.insert(0, "<local-path>")
from gguf import GGUFReader, GGUFWriter, GGUFValueType

BASE, DONOR, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
MTP_BLK = int(sys.argv[4]) if len(sys.argv) > 4 else 40

base = GGUFReader(BASE)
donor = GGUFReader(DONOR)
arch = str(bytes(base.fields["general.architecture"].parts[base.fields["general.architecture"].data[0]]), "utf-8")
print("arch:", arch)
w = GGUFWriter(OUT, arch)
bc_key = f"{arch}.block_count"
SKIP = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture", bc_key}

def add_field(key, field):
    t = field.types[0]
    if t == GGUFValueType.ARRAY:
        vals = [field.contents(i) for i in range(len(field.data))]
        w.add_array(key, vals); return
    v = field.contents()
    if t == GGUFValueType.UINT32: w.add_uint32(key, int(v))
    elif t == GGUFValueType.INT32: w.add_int32(key, int(v))
    elif t == GGUFValueType.FLOAT32: w.add_float32(key, float(v))
    elif t == GGUFValueType.UINT64: w.add_uint64(key, int(v))
    elif t == GGUFValueType.FLOAT64: w.add_float64(key, float(v))
    elif t == GGUFValueType.BOOL: w.add_bool(key, bool(v))
    elif t == GGUFValueType.STRING: w.add_string(key, str(v))
    elif t in (GGUFValueType.UINT8, GGUFValueType.UINT16): w.add_uint32(key, int(v))
    elif t in (GGUFValueType.INT8, GGUFValueType.INT16): w.add_int32(key, int(v))
    else: print("  !! unhandled KV", key, t)

n = 0
for key, field in base.fields.items():
    if key in SKIP: continue
    add_field(key, field); n += 1
w.add_uint32(bc_key, MTP_BLK + 1)
w.add_uint32(f"{arch}.nextn_predict_layers", 1)
print(f"copied {n} KV + block_count={MTP_BLK+1} + nextn_predict_layers=1")
for t in base.tensors:
    w.add_tensor(t.name, t.data, raw_dtype=t.tensor_type)
nb = 0
pref = f"blk.{MTP_BLK}."
for t in donor.tensors:
    if t.name.startswith(pref):
        w.add_tensor(t.name, t.data, raw_dtype=t.tensor_type); nb += 1
        print("  ", t.name)
print(f"tensors: {len(base.tensors)} base + {nb} donor blk.{MTP_BLK}")
assert nb == 20, f"expected 20 donor tensors, got {nb}"
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(progress=True)
w.close()
print("GRAFT DONE ->", OUT)
