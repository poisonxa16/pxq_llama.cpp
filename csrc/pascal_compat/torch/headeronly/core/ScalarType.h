#pragma once
#include "../util/compat_common.h"
#include <c10/core/ScalarType.h>
namespace torch { namespace headeronly {
using c10::ScalarType;
template <typename T> using CppTypeToScalarType = c10::CppTypeToScalarType<T>;
} }
