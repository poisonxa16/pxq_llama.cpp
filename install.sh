#!/usr/bin/env bash
# pxq_llama / pxa-vllm installer.
# Reads the card, picks the supported path, and refuses to guess.
#
# Environment:
#   PXA_DIST_DIR  directory holding the release assets (wheels + constraints
#                 files) downloaded from the releases page. Default: ./dist
set -euo pipefail

DIST="${PXA_DIST_DIR:-./dist}"
RELEASES="https://github.com/poisonxa16/pxq_llama/releases"

say() { printf "%s\n" "$*"; }
die() { printf "error: %s\n" "$*" >&2; exit 1; }

command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found - install the NVIDIA driver first."

mapfile -t CC < <(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d "[:blank:]" | sort -u)
[ "${#CC[@]}" -gt 0 ] || die "no CUDA devices visible."

say "Detected compute capability: ${CC[*]}"
if [ "${#CC[@]}" -gt 1 ]; then
  say
  say "NOTE: mixed compute capabilities on this host. A vLLM wheel is built"
  say "against ONE capability and ONE torch. Mixed rigs should use the"
  say "llama.cpp engine, which handles a heterogeneous set in one process."
fi

PRIMARY="${CC[0]}"
case "$PRIMARY" in
  6.0)
    say
    say "Pascal (sm_60). Supported path: pxa-vllm sm60 variant."
    say "  torch 2.7.1+cu126 - the last torch shipping sm_60 cubins."
    say
    say "  The sm60 wheel and its constraints file are release assets."
    say "  Download them into \$PXA_DIST_DIR (currently: $DIST) from:"
    say "    $RELEASES"
    say
    say "  Install WITH the shipped constraints file, never on its own:"
    say "    pip install -c $DIST/constraints-sm60.txt torch==2.7.1+cu126 \\"
    say "        --index-url https://download.pytorch.org/whl/cu126"
    say "    pip install -c $DIST/constraints-sm60.txt $DIST/pxa_vllm-*-sm60-*.whl"
    if [ ! -d "$DIST" ]; then
      say
      say "  NOTE: $DIST does not exist yet. Create it and unpack the release"
      say "  assets there, or point PXA_DIST_DIR at wherever you put them."
    fi
    ;;
  7.0)
    say
    say "Volta (sm_70). Supported path: pxa-vllm sm70 variant."
    say
    say "  Measured on a 2x Tesla V100-PCIE-16GB pair (TP=2), dense 27B PXQ4:"
    say "    50.98-51.46 tok/s single-stream, 132-136 tok/s aggregate at 8."
    say "  Recipe, the full sweep and the traps: docs/PXA-SM70-SERVING.md"
    say "  Launcher:                            scripts/pxa-serve-sm70.sh"
    say
    say "  The sm70 wheel and its constraints file are release assets."
    say "  Download them into \$PXA_DIST_DIR (currently: $DIST) from:"
    say "    $RELEASES"
    say
    say "    pip install -c $DIST/constraints-sm70.txt $DIST/pxa_vllm-*-sm70-*.whl"
    say
    say "  The llama.cpp engine also runs on Volta and is the better choice for"
    say "  single-stream and for mixed-capability hosts: see BUILD-FROM-SOURCE.md"
    ;;
  *)
    say
    say "Compute capability $PRIMARY is outside the two variants we build."
    say "The llama.cpp engine builds and runs on modern cards normally:"
    say "  see BUILD-FROM-SOURCE.md"
    ;;
esac

say
say "Engine (all cards): BUILD-FROM-SOURCE.md, or a prebuilt llama-* binary"
say "                    from $RELEASES"
say "Adaptive launcher:  tools/pxa-launch.py - picks engine, image and params per card."
