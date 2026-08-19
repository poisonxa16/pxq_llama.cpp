p = "/mnt/models/pxa-int-v6/site/pxq4_vllm/linear.py"
src = open(p).read()
anchor = """            torch.ops.pxq4.mmv_out(out_warm, x_warm, layer.pxq4_slabs, layer.pxq4_anchor)
            del x_warm, out_warm"""
add = """

        # Warm the v7 dequant arena (linear_out's large-M path) for this layer's
        # N*K, eagerly and pre-capture, for the same reason as the partials
        # arena above: the arena refuses to grow under cuda-graph capture, and
        # if capture sizes ever exceed mmv_max_m the captured graph takes the
        # dequant+GEMM branch. One dummy call per layer keeps the maximum.
        if getattr(layer, "pxq4_linear_op", False):
            m = int(layer.pxq4_mmv_max_m or 8) + 1
            dev = layer.pxq4_slabs.device
            x_warm = torch.zeros((m, layer.pxq4_K), dtype=torch.float16, device=dev)
            out_warm = torch.empty((m, layer.pxq4_N), dtype=torch.float16, device=dev)
            torch.ops.pxq4.linear_out(out_warm, x_warm, layer.pxq4_slabs, layer.pxq4_anchor)
            del x_warm, out_warm"""
if "Warm the v7 dequant arena" not in src:
    assert anchor in src
    src = src.replace(anchor, anchor + add, 1)
    open(p, "w").write(src)
    print("warm patch applied")
else:
    print("already applied")
