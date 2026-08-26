// PASCAL PORT torch-2.7 compat: torch/headeronly/* landed in torch 2.8 as
// re-exports of c10 types into torch::headeronly. Recreate exactly the names
// this tree uses, backed by the c10 originals that torch 2.7 ships.
#pragma once
#include <c10/util/Exception.h>
#ifndef STD_TORCH_CHECK
  #define STD_TORCH_CHECK(...) TORCH_CHECK(__VA_ARGS__)
#endif
