from __future__ import annotations

"""Flower-safe transport for ciphertext coefficients larger than int64.

NumPy object arrays cannot be serialized by Flower because Flower calls
``np.save(..., allow_pickle=False)``.  Ciphertext coefficients for the selected
69-bit modulus are therefore encoded as two unsigned 64-bit limbs before they
cross the Flower boundary.
"""

from typing import Sequence

import numpy as np

_LIMB_BITS = 64
_LIMB_MASK = (1 << _LIMB_BITS) - 1
_LIMB_COUNT = 2


def encode_bigint_coefficients(
    coefficients: Sequence[int] | np.ndarray,
    *,
    modulus: int,
) -> np.ndarray:
    """Encode non-negative coefficients modulo ``modulus`` as uint64 limbs.

    The returned array has shape ``(coefficient_count, 2)``.  Column 0 is the
    low 64-bit limb and column 1 is the high limb.  This is a regular numeric
    ndarray, so Flower can serialize it with ``allow_pickle=False``.
    """
    flat = np.asarray(coefficients, dtype=object).reshape(-1)
    encoded = np.empty((flat.size, _LIMB_COUNT), dtype=np.uint64)

    for index, raw_value in enumerate(flat):
        value = int(raw_value)
        if not 0 <= value < modulus:
            raise ValueError(
                "Ciphertext coefficient is outside the canonical interval: "
                f"index={index}, value={value}, modulus={modulus}"
            )
        encoded[index, 0] = np.uint64(value & _LIMB_MASK)
        encoded[index, 1] = np.uint64(value >> _LIMB_BITS)

    return encoded


def decode_bigint_coefficients(
    encoded: np.ndarray,
    *,
    expected_count: int,
    modulus: int,
) -> np.ndarray:
    """Decode a Flower-transmitted two-limb uint64 array to Python integers."""
    array = np.asarray(encoded)
    expected_shape = (expected_count, _LIMB_COUNT)

    if array.dtype != np.uint64:
        raise TypeError(
            "Ciphertext transport array must have dtype uint64, "
            f"received {array.dtype}"
        )
    if array.shape != expected_shape:
        raise ValueError(
            "Unexpected ciphertext transport shape: "
            f"received={array.shape}, expected={expected_shape}"
        )

    decoded = np.empty(expected_count, dtype=object)
    for index, (low_raw, high_raw) in enumerate(array):
        value = int(low_raw) | (int(high_raw) << _LIMB_BITS)
        if value >= modulus:
            raise ValueError(
                "Decoded ciphertext coefficient is outside the modulus: "
                f"index={index}, value={value}, modulus={modulus}"
            )
        decoded[index] = value

    return decoded
