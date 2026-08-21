from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from config import ERROR_STDDEV, SMUDGING_BOUND

def mod_int(value: int, modulus: int) -> int:
    return int(value) % modulus


def centered_int(value: int, modulus: int) -> int:
    reduced = int(value) % modulus
    return reduced - modulus if reduced > modulus // 2 else reduced


def mod_array(values: Sequence[int] | np.ndarray, modulus: int) -> np.ndarray:
    array = np.asarray(values)
    result = [int(value) % modulus for value in array.reshape(-1)]
    return np.asarray(result, dtype=object).reshape(array.shape)


def centered_array(values: Sequence[int] | np.ndarray, modulus: int) -> np.ndarray:
    array = np.asarray(values)
    result = [centered_int(int(value), modulus) for value in array.reshape(-1)]
    return np.asarray(result, dtype=object).reshape(array.shape)


def next_power_of_two(value: int) -> int:
    if value < 1:
        raise ValueError("value must be positive")
    return 1 << (value - 1).bit_length()


def gadget_digit_count(modulus: int, base: int) -> int:
    count = 0
    value = 1
    while value < modulus:
        value *= base
        count += 1
    return count


def infinity_norm(value: "RingElement") -> int:
    """Infinity norm of centered ring coefficients."""
    centered = value.centered_coefficients()
    return int(np.max(np.abs(centered))) if len(centered) else 0


def negacyclic_product_bound(
    ring_degree: int,
    left_bound: int,
    right_bound: int,
) -> int:
    """Public bound ||a*b||_inf <= N ||a||_inf ||b||_inf."""
    return int(ring_degree) * int(left_bound) * int(right_bound)


# =============================================================================
# NEGACYCLIC RING R_q = Z_q[x]/(x^N+1)
# =============================================================================


@dataclass(frozen=True)
class Ring:
    degree: int
    modulus: int

    def element(self, coefficients: Sequence[int] | np.ndarray) -> "RingElement":
        return RingElement(self, coefficients)

    def zero(self) -> "RingElement":
        return self.element(np.zeros(self.degree, dtype=object))

    def one(self) -> "RingElement":
        values = np.zeros(self.degree, dtype=object)
        values[0] = 1
        return self.element(values)

    def random_uniform(self, rng: np.random.Generator) -> "RingElement":
        """Sample uniformly from Z_q, including when q exceeds int64."""
        bit_length = self.modulus.bit_length()
        byte_length = (bit_length + 7) // 8
        excess_bits = 8 * byte_length - bit_length
        mask = (1 << (8 - excess_bits)) - 1 if excess_bits else 0xFF

        values: list[int] = []
        while len(values) < self.degree:
            raw = bytearray(rng.bytes(byte_length))
            raw[0] &= mask
            candidate = int.from_bytes(raw, byteorder="big", signed=False)
            if candidate < self.modulus:
                values.append(candidate)

        return self.element(values)

    def sample_ternary(self, rng: np.random.Generator) -> "RingElement":
        values = rng.integers(-1, 2, size=self.degree, dtype=np.int64)
        return self.element(values)

    def sample_error(self, rng: np.random.Generator) -> "RingElement":
        values = np.rint(
            rng.normal(0.0, ERROR_STDDEV, size=self.degree)
        ).astype(np.int64)
        return self.element(values)

    def sample_smudging(self, rng: np.random.Generator) -> "RingElement":
        values = rng.integers(
            -SMUDGING_BOUND,
            SMUDGING_BOUND + 1,
            size=self.degree,
            dtype=np.int64,
        )
        return self.element(values)


@dataclass(frozen=True)
class RingElement:
    ring: Ring
    coefficients: np.ndarray

    def __init__(
        self,
        ring: Ring,
        coefficients: Sequence[int] | np.ndarray,
    ) -> None:
        array = np.asarray(coefficients, dtype=object).reshape(-1)
        padded = np.zeros(ring.degree, dtype=object)

        for index, value in enumerate(array):
            target = index % ring.degree
            sign = -1 if (index // ring.degree) % 2 else 1
            padded[target] = (
                int(padded[target]) + sign * int(value)
            ) % ring.modulus

        object.__setattr__(self, "ring", ring)
        object.__setattr__(self, "coefficients", padded)

    def _check(self, other: "RingElement") -> None:
        if self.ring != other.ring:
            raise ValueError("Ring mismatch")

    def add(self, other: "RingElement") -> "RingElement":
        self._check(other)
        values = [
            (int(a) + int(b)) % self.ring.modulus
            for a, b in zip(self.coefficients, other.coefficients)
        ]
        return self.ring.element(values)

    def sub(self, other: "RingElement") -> "RingElement":
        self._check(other)
        values = [
            (int(a) - int(b)) % self.ring.modulus
            for a, b in zip(self.coefficients, other.coefficients)
        ]
        return self.ring.element(values)

    def neg(self) -> "RingElement":
        return self.ring.element(
            [(-int(value)) % self.ring.modulus for value in self.coefficients]
        )

    def scalar_mul(self, scalar: int) -> "RingElement":
        return self.ring.element(
            [
                (int(value) * int(scalar)) % self.ring.modulus
                for value in self.coefficients
            ]
        )

    def mul(self, other: "RingElement") -> "RingElement":
        self._check(other)
        n = self.ring.degree
        q = self.ring.modulus
        result = [0] * n

        for i, a_raw in enumerate(self.coefficients):
            a = int(a_raw)
            if a == 0:
                continue
            for j, b_raw in enumerate(other.coefficients):
                b = int(b_raw)
                if b == 0:
                    continue

                exponent = i + j
                if exponent < n:
                    result[exponent] += a * b
                else:
                    result[exponent - n] -= a * b

        return self.ring.element([value % q for value in result])

    def automorphism(self, odd_exponent: int) -> "RingElement":
        """
        Apply sigma_k(f)(x)=f(x^k), where k is odd modulo 2N.
        """
        n = self.ring.degree
        modulus_2n = 2 * n

        if odd_exponent % 2 == 0:
            raise ValueError("Cyclotomic automorphism exponent must be odd")

        result = [0] * n
        for index, value_raw in enumerate(self.coefficients):
            value = int(value_raw)
            exponent = (index * odd_exponent) % modulus_2n
            if exponent >= n:
                result[exponent - n] -= value
            else:
                result[exponent] += value

        return self.ring.element(result)

    def centered_coefficients(self) -> np.ndarray:
        return centered_array(self.coefficients, self.ring.modulus)

    def switch_modulus(self, target_ring: Ring) -> "RingElement":
        if target_ring.degree != self.ring.degree:
            raise ValueError("Ring degree mismatch during modulus switch")
        return target_ring.element(self.centered_coefficients())


# =============================================================================
# CRT/SIMD ENCODING
# =============================================================================


