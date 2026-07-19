// ggml-pxq2-tables.h -- PXQ2 frozen numeric tables (spec: PXQ-UNIVERSAL-2026-07-17.md).
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
// SOURCES (sha256-locked; do NOT edit values by hand):
//   books.json ("b2_e16")  e3ef27d550d4538654bf46c9ca8dac39ff181af31871da336a48b9e54755089f
//   (internal calibration lab artifact; the sha256 pins the exact book used)
//   sub-scale LUT: PXQ6_SUB16_INIT from ggml-pxq6-tables.h REUSED VERBATIM (the SUB16 LUT is
//   codebook-agnostic -- measured bit-identical after fp16 snap on the LM8 refit, checked for
//   LM4 in gate B0). PXQ2 defines NO sub table of its own.
// Book is the alternating-co-fit LM4 (kept round 2), fp16-snapped, emitted as exact fp32 hex.
// NO zero entry and absmax != 1 by design (Lloyd centroids of absmax-normalized data);
// the min-|v| entry is index 2 (PXQ2_ZIDX) -- used for all-zero blocks.
#pragma once

#define PXQ2_QK          32
#define PXQ2_TYPE_SIZE   9       // 1 scale byte + 8 code bytes per 32-elem row-block
#define PXQ2_BM          64
#define PXQ2_SLAB_BYTES  576     // 64 B scale SoA + 64 rows x 8 B codes
#define PXQ2_HDR_BYTES   128     // 64 x fp16 row anchors at the head of every 64-row panel
#define PXQ2_ROW_META    2       // ggml row_meta_size: 2 B/row == 128 B / 64-row panel
#define PXQ2_ZIDX        2       // argmin |book| -- code written for exactly-zero blocks

// LM4 co-fit book (books.json b2_e16, val_wrel 0.303996; full-eval wrel 0.302049).
// fp16-snapped, strictly ascending: -0.70556640625, -0.1876220703125, 0.186767578125, 0.70263671875
#define PXQ2_BOOK_INIT { \
    -0x1.6940000000000p-1f, -0x1.8040000000000p-3f, 0x1.7e80000000000p-3f, 0x1.67c0000000000p-1f }
