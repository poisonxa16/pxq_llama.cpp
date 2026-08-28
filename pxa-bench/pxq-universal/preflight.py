"""Preflight a PXQU tier map against real tensor geometry BEFORE spending hours.

The first build of a Flash-Next tier died at tensor 59/1224 after writing 57 GiB
because blk.1.ple_conv1d.weight has ne0=4 and, with no matching rule, fell
through to the PXQU default type MXFP4 -- a block-32 codec -- and
GGML_ASSERT(n_per_row % kBlockSize == 0) aborted. This check makes that failure
take 2 seconds.

Usage:
    python3 preflight.py <map.tiers> <tensors.txt>

  <map.tiers>   a PXQU tier map (see docs/PXQU-CONVERT.md); one `regex=type`
                rule per line, `#` comments ignored.
  <tensors.txt> a tensor listing for the model you intend to quantize, one
                tensor per line in the form

                    <name>  <ne0>x<ne1>[x...]  <type>  off=<offset>

                Any dump that prints those four fields will do. To generate one
                from a GGUF with the bundled gguf-py:

                    PYTHONPATH=gguf-py python3 -c 'import sys, gguf
                    for t in gguf.GGUFReader(sys.argv[1]).tensors:
                        print(t.name, "x".join(map(str, t.shape)),
                              t.tensor_type.name.lower(), "off=%d" % t.data_offset)' model.gguf > tensors.txt

Both paths may also be supplied as the environment variables PXQU_MAP and
PXQU_TENSORS instead of positional arguments.
"""
import os, re, sys

BLOCK32 = {"mxfp4","q8_0","q4_k","q5_k","q6_k","q4_0","q5_0","q6_0","iq4_nl",
           "iq1_s","iq2_xxs","pxq1","pxq2","pxq3","pxq4","pxq4hq","pxq6"}
PASSTHRU = {"f32","f16","bf16"}

args = sys.argv[1:]
MAP     = args[0] if len(args) > 0 else os.environ.get("PXQU_MAP", "")
TENSORS = args[1] if len(args) > 1 else os.environ.get("PXQU_TENSORS", "")
if not MAP or not TENSORS:
    sys.exit("usage: preflight.py <map.tiers> <tensors.txt>   "
             "(or set PXQU_MAP / PXQU_TENSORS)")

rules = []
for l in open(MAP):
    l = l.strip()
    if not l or l.startswith("#") or "=" not in l: continue
    rx, t = l.rsplit("=", 1)
    rules.append((re.compile(rx), t.lower()))

fail = []
n = 0
for line in open(TENSORS):
    line = line.strip()
    if not line or line.startswith("#"): continue
    m = re.match(r"^(\S+)\s+([0-9x]+)\s+(\S+)\s+off=", line)
    if not m: continue
    n += 1
    name = m.group(1); ne = [int(x) for x in m.group(2).split("x")]
    hit = None
    for rx, t in rules:
        if rx.search(name): hit = t; break
    eff = hit if hit else "mxfp4(DEFAULT)"
    base = eff.replace("(DEFAULT)", "")
    if ne[0] % 32 != 0 and base in BLOCK32:
        fail.append((name, "x".join(map(str, ne)), eff))

print(f"checked {n} tensors against {len(rules)} rules")
if fail:
    print(f"\n*** PREFLIGHT FAILED: {len(fail)} tensor(s) would abort a block-32 codec ***")
    for name, shape, t in fail[:12]:
        print(f"  {name:<44} ne={shape:<14} -> {t}   (ne0 % 32 != 0)")
    sys.exit(1)
print("PREFLIGHT OK: no tensor with ne0 %% 32 != 0 is targeted at a block-32 codec")
