from __future__ import annotations

import math
import secrets
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from bgv import (
    Ciphertext,
    EvaluationKeys,
    KeySwitchKey,
    PublicKey,
)
from config import GADGET_BASE, PAILLIER_BITS, P_MODULUS
from ring import (
    Ring,
    RingElement,
    centered_int,
    gadget_digit_count,
    infinity_norm,
    negacyclic_product_bound,
)
from simd import SIMDEncoder

def is_probable_prime(value: int, rounds: int = 24) -> bool:
    if value < 2:
        return False
    small_primes = (
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47
    )
    for prime in small_primes:
        if value == prime:
            return True
        if value % prime == 0:
            return False

    d = value - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1

    for _ in range(rounds):
        base = secrets.randbelow(value - 3) + 2
        x = pow(base, d, value)
        if x in (1, value - 1):
            continue
        for _ in range(s - 1):
            x = pow(x, 2, value)
            if x == value - 1:
                break
        else:
            return False
    return True


def random_prime(bits: int) -> int:
    while True:
        candidate = secrets.randbits(bits)
        candidate |= (1 << (bits - 1)) | 1
        if is_probable_prime(candidate):
            return candidate


def lcm(left: int, right: int) -> int:
    return abs(left * right) // math.gcd(left, right)


@dataclass(frozen=True)
class PaillierPublicKey:
    n: int
    n_squared: int
    g: int

    def encrypt_signed(self, message: int) -> int:
        encoded = int(message) % self.n
        while True:
            randomness = secrets.randbelow(self.n - 1) + 1
            if math.gcd(randomness, self.n) == 1:
                break
        return (
            pow(self.g, encoded, self.n_squared)
            * pow(randomness, self.n, self.n_squared)
        ) % self.n_squared

    def add(self, left: int, right: int) -> int:
        return (left * right) % self.n_squared

    def scalar_mul(self, ciphertext: int, scalar: int) -> int:
        if scalar >= 0:
            return pow(ciphertext, scalar, self.n_squared)
        inverse = pow(ciphertext, -1, self.n_squared)
        return pow(inverse, -scalar, self.n_squared)


@dataclass(frozen=True)
class PaillierPrivateKey:
    public_key: PaillierPublicKey
    lambda_value: int
    mu: int

    def decrypt_signed(self, ciphertext: int) -> int:
        n = self.public_key.n
        n_squared = self.public_key.n_squared
        value = pow(ciphertext, self.lambda_value, n_squared)
        l_value = (value - 1) // n
        message = (l_value * self.mu) % n
        return message - n if message > n // 2 else message


def generate_paillier_keypair(bits: int) -> tuple[PaillierPublicKey, PaillierPrivateKey]:
    prime_bits = bits // 2
    while True:
        p = random_prime(prime_bits)
        q = random_prime(prime_bits)
        if p != q:
            break

    n = p * q
    n_squared = n * n
    g = n + 1
    lambda_value = lcm(p - 1, q - 1)
    l_value = (pow(g, lambda_value, n_squared) - 1) // n
    mu = pow(l_value, -1, n)

    public = PaillierPublicKey(n=n, n_squared=n_squared, g=g)
    private = PaillierPrivateKey(
        public_key=public,
        lambda_value=lambda_value,
        mu=mu,
    )
    return public, private


def encrypted_negacyclic_product(
    left_small: RingElement,
    encrypted_right_coefficients: Sequence[int],
    paillier_public_key: PaillierPublicKey,
) -> List[int]:
    """
    Compute Paillier encryptions of left_small * right.

    left_small has centered ternary coefficients. The right coefficients stay
    encrypted throughout the computation.
    """
    n = left_small.ring.degree
    left = left_small.centered_coefficients()

    outputs = [
        paillier_public_key.encrypt_signed(0)
        for _ in range(n)
    ]

    for left_index, scalar_raw in enumerate(left):
        scalar = int(scalar_raw)
        if scalar == 0:
            continue

        for right_index, encrypted_value in enumerate(encrypted_right_coefficients):
            exponent = left_index + right_index
            sign = 1
            target = exponent
            if exponent >= n:
                target = exponent - n
                sign = -1

            contribution = paillier_public_key.scalar_mul(
                encrypted_value,
                sign * scalar,
            )
            outputs[target] = paillier_public_key.add(
                outputs[target],
                contribution,
            )

    return outputs


def distributed_square_shares(
    ring: Ring,
    local_secrets: Mapping[int, RingElement],
    rng: np.random.Generator,
) -> Dict[int, RingElement]:
    """
    Produce additive shares h_i satisfying sum_i h_i = s^2 mod q, without
    reconstructing s=sum_i s_i.

    For every pair i<j, party j Paillier-encrypts s_j coefficient-wise.
    Party i homomorphically computes s_i*s_j, masks it by a uniformly random
    ring element r_ij, and party j decrypts only s_i*s_j-r_ij. The two parties
    receive additive shares of 2*s_i*s_j. Local squares are added directly.
    """
    print("[DKG] Generating Paillier setup keys for private cross-products...")

    paillier: Dict[int, tuple[PaillierPublicKey, PaillierPrivateKey]] = {
        participant: generate_paillier_keypair(PAILLIER_BITS)
        for participant in local_secrets
    }

    shares: Dict[int, RingElement] = {
        participant: secret.mul(secret)
        for participant, secret in local_secrets.items()
    }

    participants = sorted(local_secrets)

    encrypted_secret_coefficients: Dict[int, List[int]] = {}
    for participant in participants:
        public, _ = paillier[participant]
        centered = local_secrets[participant].centered_coefficients()
        encrypted_secret_coefficients[participant] = [
            public.encrypt_signed(int(value))
            for value in centered
        ]

    for left_index, left_party in enumerate(participants):
        for right_party in participants[left_index + 1 :]:
            public, private = paillier[right_party]

            encrypted_product = encrypted_negacyclic_product(
                local_secrets[left_party],
                encrypted_secret_coefficients[right_party],
                public,
            )

            mask = ring.random_uniform(rng)
            masked_ciphertexts: List[int] = []

            for encrypted_value, mask_value_raw in zip(
                encrypted_product,
                mask.coefficients,
            ):
                mask_value = centered_int(int(mask_value_raw), ring.modulus)
                encrypted_negative_mask = public.encrypt_signed(-mask_value)
                masked_ciphertexts.append(
                    public.add(
                        encrypted_value,
                        encrypted_negative_mask,
                    )
                )

            decrypted_masked = ring.element(
                [
                    private.decrypt_signed(ciphertext) % ring.modulus
                    for ciphertext in masked_ciphertexts
                ]
            )

            shares[left_party] = shares[left_party].add(
                mask.scalar_mul(2)
            )
            shares[right_party] = shares[right_party].add(
                decrypted_masked.scalar_mul(2)
            )

    print("[DKG] Additive shares of s^2 generated without reconstructing s")
    return shares


# =============================================================================
# DISTRIBUTED BGV SETUP
# =============================================================================




class CanonicalEvaluationKeyGenerator:
    """
    Generate textbook single-modulus BGV key-switching keys.

    rlk[j] = Enc_s(B^j * s^2)
    rk[k][j] = Enc_s(B^j * sigma_k(s))

    The s^2 source is represented by additive square shares obtained through
    the private Paillier-assisted setup protocol. The collective secret s is
    never assembled by this class.
    """

    def __init__(
        self,
        ring_q: Ring,
        encoder: SIMDEncoder,
        participant_ids: Sequence[int],
        seed: int,
    ) -> None:
        self.ring_q = ring_q
        self.encoder = encoder
        self.participant_ids = tuple(participant_ids)
        self.rng = np.random.default_rng(seed)
        self.digit_count = gadget_digit_count(ring_q.modulus, GADGET_BASE)

    def _encrypt_target_share(
        self,
        public_key: PublicKey,
        target: RingElement,
        label: str,
    ) -> Ciphertext:
        u = self.ring_q.sample_ternary(self.rng)
        e0 = self.ring_q.sample_error(self.rng)
        e1 = self.ring_q.sample_error(self.rng)

        c0 = (
            public_key.b.mul(u)
            .add(e0.scalar_mul(P_MODULUS))
            .add(target)
        )
        c1 = public_key.a.mul(u).add(e1.scalar_mul(P_MODULUS))

        noise_bound = (
            infinity_norm(e0)
            + negacyclic_product_bound(
                self.ring_q.degree,
                infinity_norm(e1),
                public_key.collective_secret_bound,
            )
            + negacyclic_product_bound(
                self.ring_q.degree,
                public_key.public_key_error_bound,
                infinity_norm(u),
            )
        )
        return Ciphertext(
            c0,
            c1,
            message_bound=infinity_norm(target),
            noise_bound=noise_bound,
            label=label,
        )

    def generate(
        self,
        public_key: PublicKey,
        local_secrets: Mapping[int, RingElement],
    ) -> tuple[Dict[int, RingElement], EvaluationKeys]:
        square_shares = distributed_square_shares(
            self.ring_q,
            local_secrets,
            self.rng,
        )

        relin_components: List[Ciphertext] = []
        power = 1
        for digit_index in range(self.digit_count):
            combined: Ciphertext | None = None
            for participant in self.participant_ids:
                contribution = self._encrypt_target_share(
                    public_key,
                    square_shares[participant].scalar_mul(power),
                    f"rlk_{digit_index}_share_{participant}",
                )
                combined = contribution if combined is None else combined.add(contribution)
            if combined is None:
                raise AssertionError("Empty relinearization-key generation")
            relin_components.append(combined)
            power *= GADGET_BASE

        rotation_offsets: List[int] = []
        offset = 1
        while offset < self.encoder.row_size:
            rotation_offsets.append(offset)
            offset *= 2

        rotation_keys: Dict[int, KeySwitchKey] = {}
        for offset in rotation_offsets:
            exponent = self.encoder.rotation_exponent(offset)
            components: List[Ciphertext] = []
            power = 1
            for digit_index in range(self.digit_count):
                combined = None
                for participant in self.participant_ids:
                    target = (
                        local_secrets[participant]
                        .automorphism(exponent)
                        .scalar_mul(power)
                    )
                    contribution = self._encrypt_target_share(
                        public_key,
                        target,
                        f"rk_{offset}_{digit_index}_share_{participant}",
                    )
                    combined = contribution if combined is None else combined.add(contribution)
                if combined is None:
                    raise AssertionError("Empty rotation-key generation")
                components.append(combined)
                power *= GADGET_BASE

            rotation_keys[offset] = KeySwitchKey(
                tuple(components),
                source_label=f"sigma_{exponent}(s)",
            )

        evaluation_keys = EvaluationKeys(
            relinearization_key=KeySwitchKey(
                tuple(relin_components),
                source_label="s^2",
            ),
            rotation_keys=rotation_keys,
        )
        return square_shares, evaluation_keys
