#!/usr/bin/env python3
"""Gate: the K-chunk-split mmv (libpxq4_sm70 v2 decode path) is bit-identical to the
monolithic k_pxq4_mmv, executed on the REAL kernel source via the host simulator.

Run by build_hostsim.sh after test_pxq4_kernel_ref.py. Needs ./libpxq4_hostsim.so.
"""
import ctypes, os, sys
import numpy as np

lib = ctypes.CDLL(sys.argv[1] if len(sys.argv) > 1 else
                  os.path.join(os.path.dirname(os.path.abspath(__file__)), "libpxq4_hostsim.so"))
rng = np.random.default_rng(7)
fails = 0
# (panels, kslabs, M): covers nfix 8 and 16, every real K class of the model at reduced panel
# count (the fold is per-panel, so panel count only multiplies cases), M across the mmv range.
for (panels, kslabs, M) in [(2, 48, 1), (2, 136, 1), (1, 160, 3), (2, 544, 2), (3, 40, 8)]:
    nfix = lib.pxq4_hostsim_canon_nfix(kslabs)
    slabs = rng.integers(0, 256, size=(panels, kslabs, 1088), dtype=np.uint8)
    anchor = (rng.standard_normal((panels, 64)) * 0.05).astype(np.float16)
    x = (rng.standard_normal((M, kslabs * 32)) * 0.1).astype(np.float16)
    out_a = np.zeros((M, panels * 64), dtype=np.float16)
    out_b = np.ones((M, panels * 64), dtype=np.float16)
    part = np.zeros(M * panels * nfix * 256, dtype=np.float32)
    c = lambda a: a.ctypes.data_as(ctypes.c_void_p)
    for vecx in (1, 0):
        lib.pxq4_hostsim_mmv_f16(c(slabs), c(anchor.view(np.uint16)), c(x.view(np.uint16)),
                                 c(out_a.view(np.uint16)), M, panels, kslabs, vecx)
        lib.pxq4_hostsim_mmv_split_f16(c(slabs), c(anchor.view(np.uint16)), c(x.view(np.uint16)),
                                       c(part), c(out_b.view(np.uint16)), M, panels, kslabs, vecx)
        eq = np.array_equal(out_a.view(np.uint16), out_b.view(np.uint16))
        print(f"panels={panels} kslabs={kslabs} M={M} vecx={vecx} nfix={nfix} bitexact={eq}")
        if not eq:
            fails += 1
print("mmv split parity:", "FAIL" if fails else "PASS")
sys.exit(1 if fails else 0)
