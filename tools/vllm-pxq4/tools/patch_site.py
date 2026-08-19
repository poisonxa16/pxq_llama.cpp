import re
# ops.py: register fake for linear_out
p = "/mnt/models/pxa-int-v6/site/pxq4_vllm/ops.py"
src = open(p).read()
anchor = """    @torch.library.register_fake("pxq4::mmv_out")
    def _mmv_out_meta(out, x, slabs, anchor):  # noqa: ANN001, ANN202
        torch._check(slabs.dim() == 3 and anchor.dim() == 2)
        torch._check(x.dim() == 2 and out.dim() == 2)
        torch._check(x.shape[1] == slabs.shape[1] * 32)
        torch._check(out.shape[0] == x.shape[0])
        torch._check(out.shape[1] == slabs.shape[0] * 64)
        return None
"""
add = """
    if hasattr(torch.ops.pxq4, "linear_out"):
        @torch.library.register_fake("pxq4::linear_out")
        def _linear_out_meta(out, x, slabs, anchor):  # noqa: ANN001, ANN202
            torch._check(slabs.dim() == 3 and anchor.dim() == 2)
            torch._check(x.dim() == 2 and out.dim() == 2)
            torch._check(x.shape[1] == slabs.shape[1] * 32)
            torch._check(out.shape[0] == x.shape[0])
            torch._check(out.shape[1] == slabs.shape[0] * 64)
            return None
"""
if "pxq4::linear_out" not in src:
    assert anchor in src
    src = src.replace(anchor, anchor + add, 1)
    open(p, "w").write(src)
    print("ops.py patched")
else:
    print("ops.py already patched")

# linear.py: apply() prefers linear_out
p = "/mnt/models/pxa-int-v6/site/pxq4_vllm/linear.py"
src = open(p).read()
old = "        if layer.pxq4_use_mmv and M <= layer.pxq4_mmv_max_m:"
new = """        if getattr(layer, "pxq4_linear_op", False):
            # v7 single-op dispatcher: the mmv-vs-dequant+GEMM policy runs in C++ per
            # call. The op is opaque to torch.compile, so no branch is baked per compile
            # range (the Python branch below was baked as mmv for the whole (1,2048)
            # prefill range by backed/no-guard dynamic shapes: 13.8x served prefill loss).
            torch.ops.pxq4.linear_out(out, x2, layer.pxq4_slabs, layer.pxq4_anchor)
        elif layer.pxq4_use_mmv and M <= layer.pxq4_mmv_max_m:"""
if "pxq4_linear_op" not in src:
    assert old in src
    src = src.replace(old, new, 1)
    m = re.search(r"layer\.pxq4_use_mmv = [^\n]+", src)
    assert m, "pxq4_use_mmv assignment not found"
    old2 = m.group(0)
    new2 = old2 + '\n        layer.pxq4_linear_op = hasattr(torch.ops, "pxq4") and hasattr(torch.ops.pxq4, "linear_out")'
    src = src.replace(old2, new2, 1)
    open(p, "w").write(src)
    print("linear.py patched; use_mmv line:", old2.strip())
else:
    print("linear.py already patched")
