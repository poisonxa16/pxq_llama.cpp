#!/bin/bash
# A/B harness for the PXQ6 (pxq6r) tie semantics. Builds the two translation
# units side by side and runs the comparison; see main.cpp for what it asserts.
#
#   ./run.sh              # build + run
#   OUT=/path ./run.sh    # keep the intermediates somewhere else
set -e

cd "$(dirname "$0")"
OUT="${OUT:-$(mktemp -d)}"

g++ -O2 -std=c++17 -mf16c -c tu_old.cpp -o "$OUT/o.o"
g++ -O2 -std=c++17 -mf16c -c tu_new.cpp -o "$OUT/n.o"
g++ -O2 -std=c++17 main.cpp "$OUT/o.o" "$OUT/n.o" -o "$OUT/tieab" -lm
"$OUT/tieab"
