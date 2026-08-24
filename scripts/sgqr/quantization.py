#!/usr/bin/env python3
"""Reference packed symmetric quantization used by SGQR.

The reference backend prioritizes mathematical transparency and correct storage
accounting. INT4 values are physically packed two per byte. It intentionally
does not claim fused-kernel speedups: selected matrices are dequantized only when
their remnant group executes, then passed to ordinary PyTorch linear operators.
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


_SUPPORTED_BITS = (4, 8)


def _validate_bits(bits: int) -> None:
    if bits not in _SUPPORTED_BITS:
        raise ValueError(f"SGQR reference quantization supports bits={_SUPPORTED_BITS}, got {bits}")


def pack_int4(values: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Pack signed integers in [-7, 7] into two nibbles per uint8 byte.

    Zero is encoded as nibble 8. If the input has an odd number of values, the
    high nibble of the final byte is padding and is removed by ``unpack_int4``.
    """
    if values.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise TypeError(f"expected integer tensor, got {values.dtype}")
    flat = values.reshape(-1).to(torch.int16)
    if bool(((flat < -7) | (flat > 7)).any()):
        raise ValueError("INT4 reference format accepts only values in [-7, 7]")
    encoded = (flat + 8).to(torch.uint8)
    original_numel = int(encoded.numel())
    if original_numel % 2:
        encoded = torch.cat([encoded, torch.tensor([8], dtype=torch.uint8, device=encoded.device)])
    low = encoded[0::2]
    high = encoded[1::2] << 4
    return (low | high).contiguous(), original_numel


def unpack_int4(packed: torch.Tensor, original_numel: int) -> torch.Tensor:
    """Unpack the format produced by :func:`pack_int4` into int8 values."""
    if packed.dtype != torch.uint8:
        raise TypeError(f"packed INT4 tensor must be uint8, got {packed.dtype}")
    if original_numel < 0 or original_numel > packed.numel() * 2:
        raise ValueError("invalid original_numel for packed payload")
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    interleaved = torch.stack((low, high), dim=-1).reshape(-1)[:original_numel]
    return (interleaved.to(torch.int16) - 8).to(torch.int8)


class PackedSymmetricMatrix(nn.Module):
    """Row-wise symmetric INT8 or physically packed INT4 matrix.

    ``scales`` has shape ``[rows, 1]``. The logical matrix is reconstructed as
    ``qweight * scale``. Keeping quantization row-wise makes the format explicit,
    deterministic and suitable for the gate/up/down submatrices stored inside
    each SGQR MLP group.
    """

    def __init__(
        self,
        *,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        shape: tuple[int, int],
        bits: int,
        original_numel: int,
    ) -> None:
        super().__init__()
        _validate_bits(bits)
        rows, cols = (int(shape[0]), int(shape[1]))
        if rows <= 0 or cols <= 0:
            raise ValueError(f"matrix dimensions must be positive, got {shape}")
        if original_numel != rows * cols:
            raise ValueError("original_numel does not match logical shape")
        if tuple(scales.shape) != (rows, 1):
            raise ValueError(f"scales must have shape {(rows, 1)}, got {tuple(scales.shape)}")
        expected_dtype = torch.uint8 if bits == 4 else torch.int8
        if qweight.dtype != expected_dtype:
            raise TypeError(f"bits={bits} payload must use {expected_dtype}, got {qweight.dtype}")

        self.bits = int(bits)
        self.rows = rows
        self.cols = cols
        self.original_numel = int(original_numel)
        self.register_buffer("qweight", qweight.contiguous(), persistent=True)
        self.register_buffer("scales", scales.contiguous(), persistent=True)

    @classmethod
    def from_float(
        cls,
        weight: torch.Tensor,
        *,
        bits: int,
        scale_dtype: torch.dtype = torch.float16,
    ) -> "PackedSymmetricMatrix":
        _validate_bits(bits)
        if weight.ndim != 2:
            raise ValueError(f"expected 2-D weight matrix, got shape {tuple(weight.shape)}")
        if not weight.is_floating_point():
            raise TypeError(f"weight must be floating point, got {weight.dtype}")

        source = weight.detach().float()
        qmax = 7 if bits == 4 else 127
        max_abs = source.abs().amax(dim=1, keepdim=True)
        scales = max_abs / float(qmax)
        # Exact zero rows use scale 1 and therefore reconstruct exactly to zero.
        scales = torch.where(max_abs > 0, scales, torch.ones_like(scales))
        q = torch.round(source / scales).clamp(-qmax, qmax).to(torch.int8)

        if bits == 4:
            qweight, original_numel = pack_int4(q)
        else:
            qweight = q.contiguous()
            original_numel = int(q.numel())

        return cls(
            qweight=qweight,
            scales=scales.to(scale_dtype),
            shape=(int(weight.shape[0]), int(weight.shape[1])),
            bits=bits,
            original_numel=original_numel,
        )

    def quantized_values(self) -> torch.Tensor:
        if self.bits == 4:
            q = unpack_int4(self.qweight, self.original_numel)
        else:
            q = self.qweight
        return q.reshape(self.rows, self.cols)

    def dequantize(
        self,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        target_device = torch.device(device) if device is not None else self.qweight.device
        q = self.quantized_values().to(device=target_device, dtype=torch.float32)
        scales = self.scales.to(device=target_device, dtype=torch.float32)
        out = q * scales
        return out.to(dtype=dtype or self.scales.dtype)

    @property
    def logical_parameter_count(self) -> int:
        return self.rows * self.cols

    @property
    def packed_weight_bytes(self) -> int:
        return int(self.qweight.numel() * self.qweight.element_size())

    @property
    def scale_bytes(self) -> int:
        return int(self.scales.numel() * self.scales.element_size())

    @property
    def storage_bytes(self) -> int:
        return self.packed_weight_bytes + self.scale_bytes

    def storage_manifest(self) -> dict[str, int]:
        return {
            "bits": self.bits,
            "rows": self.rows,
            "cols": self.cols,
            "logical_parameters": self.logical_parameter_count,
            "packed_weight_bytes": self.packed_weight_bytes,
            "scale_bytes": self.scale_bytes,
            "total_bytes": self.storage_bytes,
        }

    def extra_repr(self) -> str:
        return (
            f"shape=({self.rows}, {self.cols}), bits={self.bits}, "
            f"packed={self.packed_weight_bytes}B, scales={self.scale_bytes}B"
        )


def total_storage_bytes(matrices: Iterable[PackedSymmetricMatrix]) -> int:
    return sum(matrix.storage_bytes for matrix in matrices)
