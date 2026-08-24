#!/usr/bin/env python3
"""Blockwise weight quantization with genuinely packed storage, laid out so that
a CSGQR remnant group is a *contiguous row slice*.

Why the layout matters
----------------------
CSGQR routes over structured groups of the MLP intermediate dimension. The
requirement from the design is explicit: an inactive group must be excluded
*before* dequantization and the matmul, or nothing is saved. That is a storage
layout problem, not a masking problem.

Qwen3's MLP is::

    F(x) = W_down @ ( silu(W_gate x) * (W_up x) )

with ``bias=False`` on all three projections (verified against
``transformers.models.qwen3.modeling_qwen3.Qwen3MLP``). ``W_gate`` and ``W_up``
are ``[intermediate, hidden]``, so a group of intermediate units is a row slice
of each. ``W_down`` is ``[hidden, intermediate]``, so the same group is a
*column* slice -- which is not contiguous, and for 4-bit packing would not even
be byte-aligned.

So this module stores ``W_down`` **transposed**, as ``[intermediate, hidden]``.
All three matrices then share one layout, ``[intermediate, hidden]``, quantized
blockwise along ``hidden``, and group ``g`` is rows ``[g*G, (g+1)*G)`` of each --
a contiguous, byte-aligned range whose scales are a contiguous slice too. The
down projection is then computed as ``u_g @ down_qT[I_g]`` instead of
``u @ W_down.T``, which is the same arithmetic.

What is actually stored
-----------------------
``qweight`` is ``uint8``. At 8 bits that is one value per byte; at 4 bits, two
values per byte (low nibble first), so a row of 1024 values occupies 512 bytes.
Nothing full-precision is retained -- ``PackedMatrix`` never holds the original
weight, and ``packed_bytes()`` reports what is really resident so that a memory
claim can be checked against ``torch.cuda.memory_allocated`` rather than a
parameter count.

Numerical choices, stated explicitly
------------------------------------
* Scale/zero search runs in float32 regardless of the weight dtype, because
  bf16 has 8 mantissa bits and computing a range in it visibly quantizes the
  scale itself.
* Scales are stored float16 (the usual convention) and counted in the byte
  total. Zero points are uint8.
* An all-zero block would give ``scale = 0`` and a division by zero. Such
  blocks get ``scale = 1`` and quantize to exactly zero, which is exact rather
  than merely safe.
* Symmetric 4-bit uses the range ``[-7, 7]`` stored biased by +8, so the
  representable set is ``[1, 15]`` and zero maps to 8 exactly. The asymmetric
  form uses the full ``[0, 15]``.
* A block shorter than ``block`` (when the input dim is not a multiple of it)
  is quantized on its real length; the padding nibble of an odd-length row is
  written as zero and never read back.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class QuantSpec:
    """A quantization configuration.

    ``block`` is the number of consecutive input-dimension weights sharing one
    scale (and zero point). ``bits=16`` is a pass-through used by the
    full-precision donor arm, so every arm can travel the same code path and a
    difference between arms is never a difference of implementation.
    """

    bits: int = 4
    block: int = 128
    symmetric: bool = True

    def __post_init__(self) -> None:
        if self.bits not in (2, 3, 4, 8, 16):
            raise ValueError("bits must be one of 2, 3, 4, 8, 16; got " + str(self.bits))
        if self.bits != 16 and self.block < 1:
            raise ValueError("block must be >= 1; got " + str(self.block))

    @property
    def is_passthrough(self) -> bool:
        return self.bits == 16

    @property
    def levels(self) -> int:
        return 1 << self.bits

    @property
    def values_per_byte(self) -> int:
        """How many quantized values fit in one uint8.

        Only 4 and 8 bits get real packing. 2- and 3-bit are stored one value
        per byte: they exist here for the Tier-3 bitwidth sweep, and inventing a
        3-bit bit-packer whose storage claim we never validate would be worse
        than storing them loosely and saying so.
        """
        return 2 if self.bits == 4 else 1

    def name(self) -> str:
        if self.is_passthrough:
            return "fp16"
        kind = "sym" if self.symmetric else "asym"
        return "int" + str(self.bits) + "-b" + str(self.block) + "-" + kind


def _quantize_blocks(weight: Tensor, spec: QuantSpec) -> tuple[Tensor, Tensor, Tensor | None]:
    """Quantize ``[rows, cols]`` blockwise along ``cols``.

    Returns ``(codes, scales, zeros)`` where ``codes`` is int64 in the unsigned
    storage range (biased for the symmetric form), ``scales`` is float32
    ``[rows, n_blocks]``, and ``zeros`` is uint8 ``[rows, n_blocks]`` for the
    asymmetric form or ``None`` for the symmetric one.
    """
    rows, cols = weight.shape
    block = min(spec.block, cols)
    n_blocks = (cols + block - 1) // block
    work = weight.to(torch.float32)

    # Pad to a whole number of blocks with NaN, so padding cannot influence the
    # min/max of the final block; it is masked out below.
    pad = n_blocks * block - cols
    if pad:
        work = torch.cat(
            [work, torch.full((rows, pad), float("nan"), dtype=torch.float32,
                              device=work.device)], dim=1)
    grid = work.view(rows, n_blocks, block)
    valid = ~torch.isnan(grid)
    safe = torch.where(valid, grid, torch.zeros_like(grid))

    if spec.symmetric:
        # amax over real entries only.
        amax = torch.where(valid, grid.abs(), torch.zeros_like(grid)).amax(dim=2)
        qmax = (1 << (spec.bits - 1)) - 1          # 7 at 4 bits, 127 at 8
        bias = 1 << (spec.bits - 1)                # 8 at 4 bits, 128 at 8
        dead = amax <= 0
        scales = torch.where(dead, torch.ones_like(amax), amax / qmax)
        codes = torch.round(safe / scales.unsqueeze(2)).clamp(-qmax, qmax)
        codes = (codes + bias).to(torch.int64)
        zeros = None
    else:
        big = torch.finfo(torch.float32).max
        lo = torch.where(valid, grid, torch.full_like(grid, big)).amin(dim=2)
        hi = torch.where(valid, grid, torch.full_like(grid, -big)).amax(dim=2)
        # Include zero in the represented interval so an exactly-zero weight
        # stays exactly zero after the round trip.
        lo = torch.minimum(lo, torch.zeros_like(lo))
        hi = torch.maximum(hi, torch.zeros_like(hi))
        qmax = spec.levels - 1
        span = hi - lo
        dead = span <= 0
        scales = torch.where(dead, torch.ones_like(span), span / qmax)
        zero_f = torch.round(-lo / scales).clamp(0, qmax)
        codes = torch.round(safe / scales.unsqueeze(2)) + zero_f.unsqueeze(2)
        codes = codes.clamp(0, qmax).to(torch.int64)
        zeros = zero_f.to(torch.uint8)

    # Zero out the codes sitting on padding so the packed bytes are deterministic.
    codes = torch.where(valid, codes, torch.zeros_like(codes))
    codes = codes.view(rows, n_blocks * block)[:, :cols].contiguous()
    return codes, scales, zeros


def _pack(codes: Tensor, spec: QuantSpec) -> Tensor:
    """Pack unsigned codes ``[rows, cols]`` into uint8 ``[rows, bytes_per_row]``."""
    rows, cols = codes.shape
    if spec.values_per_byte == 1:
        return codes.to(torch.uint8).contiguous()
    # 4-bit: low nibble carries the even index, high nibble the odd one.
    n_bytes = (cols + 1) // 2
    padded = codes
    if cols % 2:
        padded = torch.cat(
            [codes, torch.zeros((rows, 1), dtype=codes.dtype, device=codes.device)], dim=1)
    pair = padded.view(rows, n_bytes, 2)
    packed = (pair[:, :, 0] | (pair[:, :, 1] << 4)).to(torch.uint8)
    return packed.contiguous()


def _unpack(packed: Tensor, cols: int, spec: QuantSpec) -> Tensor:
    """Inverse of :func:`_pack`. Returns int16 codes ``[rows, cols]``."""
    if spec.values_per_byte == 1:
        return packed[:, :cols].to(torch.int16)
    rows = packed.shape[0]
    lo = (packed & 0x0F).to(torch.int16)
    hi = (packed >> 4).to(torch.int16)
    out = torch.stack([lo, hi], dim=2).view(rows, -1)
    return out[:, :cols]


class PackedMatrix(nn.Module):
    """A frozen, packed, blockwise-quantized ``[rows, cols]`` matrix.

    Registered as buffers rather than parameters: buffers move with ``.to()``
    and are saved by ``state_dict``, but an optimizer constructed over
    ``model.parameters()`` cannot see them, so no optimizer state can be created
    for the donor by accident. ``requires_grad`` is False for integer dtypes by
    construction -- autograd refuses to track them -- which is a stronger
    guarantee than setting a flag.
    """

    def __init__(self, weight: Tensor, spec: QuantSpec, *,
                 compute_dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        if weight.dim() != 2:
            raise ValueError("expected a 2-D weight; got shape " + str(tuple(weight.shape)))
        self.spec = spec
        self.rows, self.cols = int(weight.shape[0]), int(weight.shape[1])
        self.compute_dtype = compute_dtype

        if spec.is_passthrough:
            # The FP donor arm. Stored at compute dtype, still a buffer.
            self.register_buffer("dense", weight.detach().to(compute_dtype).contiguous())
            self.block = self.cols
            self.n_blocks = 1
            return

        codes, scales, zeros = _quantize_blocks(weight.detach(), spec)
        self.block = min(spec.block, self.cols)
        self.n_blocks = (self.cols + self.block - 1) // self.block
        self.register_buffer("qweight", _pack(codes, spec))
        self.register_buffer("scales", scales.to(torch.float16).contiguous())
        if zeros is None:
            self.zeros = None
        else:
            self.register_buffer("zeros", zeros.contiguous())

    # -- accounting --------------------------------------------------------
    def packed_bytes(self) -> dict[str, int]:
        """Bytes genuinely resident for this matrix, itemized.

        This is what a memory claim must be checked against. It is deliberately
        not a parameter count: quantization does not remove parameters, it
        changes how many bytes each one costs.
        """
        if self.spec.is_passthrough:
            n = self.dense.numel() * self.dense.element_size()
            return {"weight": n, "scales": 0, "zeros": 0, "total": n}
        w = self.qweight.numel() * self.qweight.element_size()
        s = self.scales.numel() * self.scales.element_size()
        z = 0 if self.zeros is None else self.zeros.numel() * self.zeros.element_size()
        return {"weight": w, "scales": s, "zeros": z, "total": w + s + z}

    def dense_bytes(self) -> int:
        """What the same matrix would cost unquantized at ``compute_dtype``."""
        return self.rows * self.cols * torch.finfo(self.compute_dtype).bits // 8

    # -- reconstruction ----------------------------------------------------
    def dequantize(self, row_lo: int = 0, row_hi: int | None = None) -> Tensor:
        """Reconstruct rows ``[row_lo, row_hi)`` at ``compute_dtype``.

        Only the requested rows are touched: the packed bytes, the scales and
        the zero points are all sliced before any arithmetic happens. This is
        the property that makes group routing a real compute saving rather than
        a mask applied after the fact.
        """
        row_hi = self.rows if row_hi is None else row_hi
        if not (0 <= row_lo <= row_hi <= self.rows):
            raise ValueError("row range [" + str(row_lo) + ", " + str(row_hi)
                             + ") outside [0, " + str(self.rows) + ")")
        if self.spec.is_passthrough:
            return self.dense[row_lo:row_hi]
        if row_lo == row_hi:
            return torch.empty((0, self.cols), dtype=self.compute_dtype,
                               device=self.qweight.device)

        packed = self.qweight[row_lo:row_hi]
        codes = _unpack(packed, self.cols, self.spec)
        k = codes.shape[0]

        scales = self.scales[row_lo:row_hi].to(torch.float32)
        if self.spec.symmetric:
            shifted = codes.to(torch.float32) - float(1 << (self.spec.bits - 1))
        else:
            zeros = self.zeros[row_lo:row_hi].to(torch.float32)
            shifted = codes.to(torch.float32) - zeros.repeat_interleave(
                self.block, dim=1)[:, :self.cols]
        wide = scales.repeat_interleave(self.block, dim=1)[:, :self.cols]
        return (shifted * wide).to(self.compute_dtype).view(k, self.cols)

    def extra_repr(self) -> str:
        return (str(self.rows) + "x" + str(self.cols) + ", " + self.spec.name()
                + ", " + format(self.packed_bytes()["total"], ",") + " B")


def quantization_error(weight: Tensor, spec: QuantSpec,
                       compute_dtype: torch.dtype = torch.bfloat16) -> dict[str, float]:
    """Round-trip fidelity for one matrix, as plain numbers.

    ``rel_fro`` is the relative Frobenius error and is the one to quote;
    ``cosine`` saturates near 1 and hides differences that matter.
    """
    packed = PackedMatrix(weight, spec, compute_dtype=compute_dtype)
    recon = packed.dequantize().to(torch.float32)
    ref = weight.detach().to(torch.float32)
    diff = recon - ref
    denom = ref.norm().item() or 1.0
    return {
        "rel_fro": diff.norm().item() / denom,
        "max_abs": diff.abs().max().item(),
        "cosine": torch.nn.functional.cosine_similarity(
            recon.flatten(), ref.flatten(), dim=0).item(),
        "packed_bytes": packed.packed_bytes()["total"],
        "dense_bytes": packed.dense_bytes(),
        "compression": packed.dense_bytes() / max(packed.packed_bytes()["total"], 1),
    }
