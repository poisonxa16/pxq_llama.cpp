// ggml-pxq2-tables.h -- PXQ2 frozen numeric tables. Format spec: this header; conversion: docs/PXQU-CONVERT.md.
//
// PXQ2 = 2-bit codes into the co-fit LM4 4-entry book + the PROVEN PXQ6 E16-row two-level
// scales, UNCHANGED:
//   per-ROW fp16 anchor (128 B header per 64-row panel = 2 B/row via ggml row_meta_size)
//   + one 4-bit sub-scale per 16-elem block through the frozen PXQ6 SUB16 LUT
//   (two nibbles in the one scale byte per 32 elems -> slab scale SoA stays 64 B).
//   dequant contract (parity-locked, identical to PXQ6):
//     eff = fp32(anchor_fp16) * PXQ6_SUB16[s4]      (fp32 mul, once per 16-block)
//     w   = eff * fp32(book[c])                     (fp32 mul; GEMM snaps __float2half_rn(w))
//   codes: 2 bits/elem, 4 codes/byte -> 8 code B per 32-elem row-block; packed as two LE
//   uint32 words per block, word h covers elems 16h..16h+15, elem j at bits 2*(j&15)
//   (identical to the PXQ3 low bit-plane -> shared extraction logic).
//   slab = 64 B scale SoA + 64 x 8 B code rows = 576 B; panel = 128 B anchor hdr + kslabs slabs.
//   bpw = 9*8/32 + 16/K = 2.25 + 16/K  (2.2656 @ K=2048 gate/up, 2.28125 @ K=512 down).
//   Measured (lab, uniform-2bit, PROD imatrix, rng-42 eval protocol): wrel 0.3020488.
//
// SOURCES:
//   sub-scale LUT: PXQ6_SUB16_INIT from ggml-pxq6-tables.h REUSED VERBATIM. ⚠ That table's
//   range was frozen for books with absmax == 1.0 (max(SUB16) = 0.98779): the reconstruction
//   ceiling of the composition is  max|w| = anchor * max(SUB16) * max|book|,  with anchor ==
//   the row absmax. A book with absmax < 1 therefore CLIPS every row's peak weights.
//   book: LM4 Lloyd refit on real DS4-Flash expert weights with absmax PINNED TO 1.0
//   (DS4-FP8-REQUANT-2026-08-02.md §6). fp16-snapped, strictly ascending.
//
// HISTORY — the original book here was the co-fit LM4 from books.json "b2_e16"
// (sha256 e3ef27d5...), absmax 0.70557, "NO zero entry and absmax != 1 by design (Lloyd
// centroids of absmax-normalized data)". That design premise is only valid if SUB16 is
// refit for the book's range; reusing PXQ6's SUB16 verbatim capped reconstruction at
// 0.6970 x row absmax — every row's peak weights clipped ~30%, 18.8% of squared error
// concentrated in the top-1% weights, degenerate output on DS4-Flash while GLOBAL wrel
// looked *better* than the coherent community arm (0.3296 vs 0.3535). Files built with
// the old book carry it in their `pxa.pxq2.book` KV; decode them with
// PXA_PXQ2_BOOK=-0.70556640625,-0.1876220703125,0.186767578125,0.70263671875
// The min-|v| entry is index 2 (PXQ2_ZIDX) -- used for all-zero blocks.
#pragma once

#define PXQ2_QK          32
#define PXQ2_TYPE_SIZE   9       // 1 scale byte + 8 code bytes per 32-elem row-block
#define PXQ2_BM          64
#define PXQ2_SLAB_BYTES  576     // 64 B scale SoA + 64 rows x 8 B codes
#define PXQ2_HDR_BYTES   128     // 64 x fp16 row anchors at the head of every 64-row panel
#define PXQ2_ROW_META    2       // ggml row_meta_size: 2 B/row == 128 B / 64-row panel
#define PXQ2_ZIDX        2       // argmin |book| -- code written for exactly-zero blocks

// LM4 Lloyd refit on DS4-Flash expert weights, absmax pinned to 1.0 so the shared SUB16
// range is reachable (ceiling = 0.98779 * 1.0 = 0.98779 x anchor, matching PXQ4/PXQ6/PXQ1).
// Measured on DS4 experts vs the shipped book: global wrel 0.3207 -> 0.3108, top-1% rel
// 0.2874 -> 0.2734 (unweighted-MSE protocol, DS4-FP8-REQUANT-2026-08-02.md §6).
// fp16-snapped, strictly ascending: -1.0, -0.338623046875, 0.19189453125, 0.8857421875
#define PXQ2_BOOK_INIT { \
    -0x1.6940000000000p-1f, -0x1.8040000000000p-3f, 0x1.7e80000000000p-3f, 0x1.67c0000000000p-1f }

// ---------------------------------------------------------------------------------------------
// v2 book (PXA_PXQ_CEIL_V2, 2026-08-09) -- THE CEILING FIX. The frozen LM4 book rescaled by
// 1/max|book| = 1/0.70556640625 so max|book| == 1.0 exactly, then fp16-snapped. A pure rescale:
// same Lloyd grid shape, same ZIDX (2), same sign straddle -- only the constant that was
// silently LOST between the SUB16 ceiling (frozen for a max-1.0 book) and this book's 0.7056
// absmax moves into the book itself. Restores the representable ceiling from 0.697*anchor
// (a row's own absmax unrepresentable at ANY sub-scale) to 0.987793*anchor = PXQ4/PXQ6 parity.
// Provenance: files quantized with it bake pxa.pxq2.version = 2 and these values in
// pxa.pxq2.book. v1 files keep decoding with the v1 table above; this table is active ONLY
// when PXA_PXQ_CEIL_V2=1 is set on BOTH quantizer and runtime (an older/shipped build can
// decode v2 files via its existing PXA_PXQ2_BOOK override with these exact values).
// fp16-snapped, strictly ascending: -1.0, -0.265869140625, 0.2646484375, 0.99560546875
#define PXQ2_BOOK_V2_INIT { \
    -0x1.0000000000000p+0f, -0x1.1040000000000p-2f, 0x1.0f00000000000p-2f, 0x1.fdc0000000000p-1f }

// ---------------------------------------------------------------------------------------------
// v3 book (PXA_PXQ2_V3, 2026-08-10) -- THE MODEL-FAMILY REFIT BOOK (LM4R). The frozen LM4 was
// co-fit on a different lab distribution (books.json b2_e16); this is a fresh Lloyd refit of
// the 4 entries on the target family's own pxq2-tier expert weights (a 122B-A10B qwen35moe Q8 reference),
// with the EXACT quantizer search inside the Lloyd loop (joint s+c re-derived every iteration,
// uniform weights == the imatrix-free path). The fit moves the small-positive entry hard
// toward zero (0.187 -> 0.1007, 58%% code usage): pxq2-tier experts carry near-zero mass the
// frozen LM4 cannot represent (26.9%% of held-out elements reconstruct better as 0 than as
// their nearest v1 entry), and a near-zero entry also frees the sub-scale search upward for
// the peaks (s=0-pinned groups 27.6 -> 25.8%%, clipped groups 75.3 -> 71.0%%).
// HELD-OUT EVAL (10 pxq2 tensors x 4 experts, disjoint from the 8x4 fit set; plain E2 --
// the error family that predicted the ppl sign in both controlled A/Bs of this campaign):
// pooled -7.9%% raw / -9.3%% population-weighted (tier map: 45 down / 16 gate-up tensors);
// down-class -11.5%%, gate/up +4.2%%; uniform across |w| bands (bulk -6.3%%, mid -8.7%%,
// top99-99.9 -15.8%%, peak99.9+ -16.7%%) -- an ERROR REDUCTION, not a band redistribution.
// The PINNED-EXACT-ZERO variant (one entry forced to 0.0) was fit and evaluated by the same
// machinery and FAILED its pre-registered gate (-4.5%% population-weighted but gate/up class
// +18.2%% and non-robust): a 4-entry book cannot afford to spend an entry on exact 0; 0.1007
// at 58%% usage is the data's compromise. (Campaign record 2026-08-10, zbook-fit.)
// The quantizer derives the zero-fill code dynamically (argmin |book|, pxq2_zidx_q): index 2
// for v1/v2 AND v3, so stock output stays byte-identical and the constant above stays true.
// Provenance: files quantized with it bake pxa.pxq2.version = 3 and these values in
// pxa.pxq2.book. Active ONLY when PXA_PXQ2_V3=1 on BOTH quantizer and runtime (or
// PXA_PXQ2_BOOK with these exact values on an older build). PXQ2-only: PXQ3 books are not
// touched by this gate. Wins over PXA_PXQ_CEIL_V2 for PXQ2; PXA_PXQ2_BOOK still wins over both.
// fp16-snapped, strictly ascending: -0.791015625, -0.2712402344, 0.1007080078, 0.6655273438
#define PXQ2_BOOK_V3_INIT { \
    -0x1.9500000000000p-1f, -0x1.15c0000000000p-2f, 0x1.9c80000000000p-4f, 0x1.54c0000000000p-1f }

    -0x1.0000000000000p+0f, -0x1.5ac0000000000p-2f, 0x1.8900000000000p-3f, 0x1.c580000000000p-1f }
