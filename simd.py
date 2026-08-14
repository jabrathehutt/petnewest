from __future__ import annotations

from typing import List, Sequence

import numpy as np

from ring import Ring, RingElement, centered_array, mod_array

def modular_matrix_inverse(matrix: np.ndarray, modulus: int) -> np.ndarray:
    n = matrix.shape[0]
    if matrix.shape != (n, n):
        raise ValueError("Matrix must be square")

    augmented: List[List[int]] = []
    for row_index in range(n):
        row = [int(value) % modulus for value in matrix[row_index]]
        row.extend(1 if row_index == column else 0 for column in range(n))
        augmented.append(row)

    for column in range(n):
        pivot = next(
            (
                row
                for row in range(column, n)
                if augmented[row][column] % modulus != 0
            ),
            None,
        )
        if pivot is None:
            raise ValueError("Matrix is not invertible modulo the plaintext modulus")

        augmented[column], augmented[pivot] = (
            augmented[pivot],
            augmented[column],
        )

        inverse = pow(augmented[column][column], -1, modulus)
        augmented[column] = [
            (value * inverse) % modulus
            for value in augmented[column]
        ]

        for row in range(n):
            if row == column:
                continue
            factor = augmented[row][column] % modulus
            if factor == 0:
                continue
            augmented[row] = [
                (left - factor * right) % modulus
                for left, right in zip(augmented[row], augmented[column])
            ]

    return np.asarray(
        [row[n:] for row in augmented],
        dtype=np.int64,
    )


class SIMDEncoder:
    """
    Two-row batching for power-of-two cyclotomic rings.

    For R_p=Z_p[x]/(x^N+1), p=1 mod 2N, the roots are ordered as

        zeta^(5^j),  j=0,...,N/2-1
        zeta^(-5^j), j=0,...,N/2-1.

    sigma_(5^r) rotates each row by r positions.
    """

    def __init__(self, ring_p: Ring):
        self.ring = ring_p
        self.n = ring_p.degree
        self.p = ring_p.modulus

        if self.n < 2 or self.n & (self.n - 1):
            raise ValueError("Ring degree must be a power of two")
        if (self.p - 1) % (2 * self.n) != 0:
            raise ValueError("Plaintext modulus must satisfy p = 1 mod 2N")

        self.row_size = self.n // 2
        primitive_generator = 3  # primitive root modulo 65537
        zeta = pow(
            primitive_generator,
            (self.p - 1) // (2 * self.n),
            self.p,
        )

        first_exponents = [
            pow(5, index, 2 * self.n)
            for index in range(self.row_size)
        ]
        second_exponents = [
            (-exponent) % (2 * self.n)
            for exponent in first_exponents
        ]
        self.slot_exponents = first_exponents + second_exponents
        self.roots = [
            pow(zeta, exponent, self.p)
            for exponent in self.slot_exponents
        ]

        vandermonde = np.zeros((self.n, self.n), dtype=np.int64)
        for row, root in enumerate(self.roots):
            value = 1
            for column in range(self.n):
                vandermonde[row, column] = value
                value = (value * root) % self.p

        self.vandermonde = vandermonde
        self.inverse_vandermonde = modular_matrix_inverse(
            vandermonde,
            self.p,
        )

    def encode(self, slots: Sequence[int] | np.ndarray) -> RingElement:
        values = np.asarray(slots).reshape(-1)
        if len(values) > self.n:
            raise ValueError(f"At most {self.n} slots can be encoded")

        padded = np.zeros(self.n, dtype=np.int64)
        padded[: len(values)] = mod_array(values, self.p)

        coefficients = []
        for row in self.inverse_vandermonde:
            total = 0
            for weight, value in zip(row, padded):
                total += int(weight) * int(value)
            coefficients.append(total % self.p)

        return self.ring.element(coefficients)

    def decode(self, plaintext: RingElement) -> np.ndarray:
        if plaintext.ring != self.ring:
            raise ValueError("Plaintext ring mismatch")

        slots = []
        for row in self.vandermonde:
            total = 0
            for weight, coefficient in zip(row, plaintext.coefficients):
                total += int(weight) * int(coefficient)
            slots.append(total % self.p)

        return centered_array(slots, self.p)

    def rotation_exponent(self, offset: int) -> int:
        return pow(5, offset % self.row_size, 2 * self.n)
