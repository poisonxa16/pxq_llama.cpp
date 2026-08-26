# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch


def _decode_e4m3fn(bits: torch.Tensor) -> torch.Tensor:
    values = bits.to(torch.int32)
    sign = torch.where(values & 0x80 != 0, -1.0, 1.0)
    exponent = values >> 3 & 0x0F
    mantissa = values & 0x07
    subnormal = mantissa.float() * 2.0**-9
    normal = (1.0 + mantissa.float() * 0.125) * torch.exp2(exponent.float() - 7.0)
    decoded = torch.where(exponent == 0, subnormal, normal) * sign
    return torch.where(
        (exponent == 15) & (mantissa == 7),
        torch.full_like(decoded, float("nan")),
        decoded,
    )


def _encode_e4m3fn(values: torch.Tensor) -> torch.Tensor:
    sign = (values.view(torch.int32) >> 24) & 0x80
    magnitude = values.abs().clamp(max=448.0)
    subnormal = torch.round(magnitude * 512.0).to(torch.int32).clamp(0, 8)

    safe = magnitude.clamp(min=2.0**-126)
    unbiased = torch.floor(torch.log2(safe)).to(torch.int32)
    scale = torch.exp2(unbiased.float())
    mantissa = torch.round((safe / scale - 1.0) * 8.0).to(torch.int32)
    carry = mantissa >= 8
    exponent = unbiased + 7 + carry.to(torch.int32)
    mantissa = torch.where(carry, 0, mantissa)
    overflow = (exponent > 15) | ((exponent == 15) & (mantissa > 6))
    normal = (exponent.clamp(1, 15) << 3) | mantissa.clamp(0, 7)
    normal = torch.where(overflow, 0x7E, normal)
    encoded = torch.where(magnitude < 2.0**-6, subnormal, normal)
    encoded = torch.where(magnitude == 0, 0, encoded)
    return (sign | encoded).to(torch.uint8)


def test_software_decode_matches_torch_for_all_e4m3fn_bytes():
    bits = torch.arange(256, dtype=torch.uint8)
    expected = bits.view(torch.float8_e4m3fn).float()
    actual = _decode_e4m3fn(bits)
    torch.testing.assert_close(actual, expected, equal_nan=True)


def test_software_encode_matches_torch_on_boundaries_and_midpoints():
    positive_bits = torch.arange(0x7F, dtype=torch.uint8)
    representable = positive_bits.view(torch.float8_e4m3fn).float()
    finite = representable[torch.isfinite(representable)]
    midpoints = (finite[:-1] + finite[1:]) * 0.5
    probes = torch.cat(
        [
            -finite.flip(0),
            finite,
            -midpoints.flip(0),
            midpoints,
            torch.tensor([-1000.0, -448.0, -0.0, 0.0, 448.0, 1000.0]),
        ]
    )
    expected = probes.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(torch.uint8)
    actual = _encode_e4m3fn(probes)
    torch.testing.assert_close(actual, expected)
