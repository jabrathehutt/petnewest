from __future__ import annotations

from typing import List, Mapping

import numpy as np

import config as cfg
from bgv import Ciphertext, DegreeTwoCiphertext, ManualBGV, ThresholdDecryptor
from ring import Ring, RingElement, infinity_norm, next_power_of_two
from simd import SIMDEncoder
from flower_client import DatasetMetadata

def distributed_secret_evaluation(
    ciphertext: Ciphertext,
    local_secret_contributions: Mapping[int, RingElement],
) -> RingElement:
    """
    Compute c0+c1*s as a sum of local products c1*s_i.

    The collective secret polynomial is never assembled.
    """
    result = ciphertext.c0
    for secret_share in local_secret_contributions.values():
        result = result.add(ciphertext.c1.mul(secret_share))
    return result


def distributed_degree_two_evaluation(
    ciphertext: DegreeTwoCiphertext,
    local_secret_contributions: Mapping[int, RingElement],
    square_shares: Mapping[int, RingElement],
) -> RingElement:
    """
    Compute d0+d1*s+d2*s^2 from distributed shares of s and s^2.
    """
    result = ciphertext.d0
    for secret_share in local_secret_contributions.values():
        result = result.add(ciphertext.d1.mul(secret_share))
    for square_share in square_shares.values():
        result = result.add(ciphertext.d2.mul(square_share))
    return result


def extract_exact_noise(
    evaluated_representative: RingElement,
    expected_plaintext_q: RingElement,
) -> RingElement:
    """
    From c0+c1*s = m+p*eta, recover eta for testing.

    This is used only by the self-test auditor. It does not participate in
    decryption and is never sent to the server in the Flower protocol.
    """
    residual = evaluated_representative.sub(
        expected_plaintext_q
    ).centered_coefficients()

    noise_coefficients: List[int] = []
    for value_raw in residual:
        value = int(value_raw)
        if value % cfg.P_MODULUS != 0:
            raise AssertionError(
                "Exact BGV equation failed: residual is not divisible by p"
            )
        noise_coefficients.append(value // cfg.P_MODULUS)

    return evaluated_representative.ring.element(noise_coefficients)


def audit_ciphertext_equation(
    name: str,
    ciphertext: Ciphertext,
    expected_plaintext_q: RingElement,
    local_secret_contributions: Mapping[int, RingElement],
) -> RingElement:
    evaluated = distributed_secret_evaluation(
        ciphertext,
        local_secret_contributions,
    )
    exact_noise = extract_exact_noise(
        evaluated,
        expected_plaintext_q,
    )

    actual_noise_norm = infinity_norm(exact_noise)
    print(
        f"[Exact audit:{name}] "
        f"c0+c1*s = m+p*eta; "
        f"||eta||_inf={actual_noise_norm}, "
        f"public_bound={ciphertext.noise_bound}"
    )

    if actual_noise_norm > ciphertext.noise_bound:
        raise AssertionError(
            f"Noise bound underestimated for {name}: "
            f"{actual_noise_norm}>{ciphertext.noise_bound}"
        )

    return exact_noise


# =============================================================================
# INITIALIZATION AND SELF-TESTS
# =============================================================================


def initialize_dimensions(
    metadata: DatasetMetadata,
) -> None:

    cfg.RUNTIME.model_dimension = metadata.feature_count + 1

    # Only the first batching row is used. It must hold the complete model.
    cfg.RUNTIME.ring_degree = next_power_of_two(
        max(8, 2 * cfg.RUNTIME.model_dimension)
    )
    cfg.RUNTIME.row_size = cfg.RUNTIME.ring_degree // 2

    if (cfg.P_MODULUS - 1) % (2 * cfg.RUNTIME.ring_degree) != 0:
        raise ValueError(
            "Chosen plaintext modulus does not split the selected ring"
        )

    cfg.RUNTIME.authorized_query = np.zeros(
        cfg.RUNTIME.ring_degree,
        dtype=np.int64,
    )
    cfg.RUNTIME.authorized_query[
        : min(2, cfg.RUNTIME.model_dimension)
    ] = 1


def run_crypto_self_test(
    ring_q: Ring,
    ring_p: Ring,
    encoder: SIMDEncoder,
    bgv: ManualBGV,
    decryptor: ThresholdDecryptor,
    local_secret_contributions: Mapping[int, RingElement],
    square_shares: Mapping[int, RingElement],
) -> None:
    if cfg.RUNTIME.ring_degree is None or cfg.RUNTIME.model_dimension is None:
        raise RuntimeError("Dimensions are not initialized")

    print("\n=== CRYPTOGRAPHIC SELF-TEST AND EXACT BGV AUDIT ===")

    message = np.zeros(cfg.RUNTIME.ring_degree, dtype=np.int64)
    query = np.zeros(cfg.RUNTIME.ring_degree, dtype=np.int64)

    message[: min(4, cfg.RUNTIME.model_dimension)] = np.array(
        [3, -2, 5, 1][: min(4, cfg.RUNTIME.model_dimension)],
        dtype=np.int64,
    )
    query[: min(4, cfg.RUNTIME.model_dimension)] = np.array(
        [1, 1, 0, 0][: min(4, cfg.RUNTIME.model_dimension)],
        dtype=np.int64,
    )

    message_p = encoder.encode(message)
    query_p = encoder.encode(query)
    message_q = ring_q.element(message_p.centered_coefficients())
    query_q = ring_q.element(query_p.centered_coefficients())

    encrypted_message = bgv.encrypt_slots(
        message,
        label="selftest_message",
    )
    encrypted_query = bgv.encrypt_slots(
        query,
        label="selftest_query",
    )

    audit_ciphertext_equation(
        "Enc(M)",
        encrypted_message,
        message_q,
        local_secret_contributions,
    )
    audit_ciphertext_equation(
        "Enc(y)",
        encrypted_query,
        query_q,
        local_secret_contributions,
    )

    # Raw multiplication equation:
    # d0+d1*s+d2*s^2 =
    # (M+p eta_M)(y+p eta_y).
    raw_product = bgv.multiply(
        encrypted_message,
        encrypted_query,
    )
    raw_evaluation = distributed_degree_two_evaluation(
        raw_product,
        local_secret_contributions,
        square_shares,
    )
    left_evaluation = distributed_secret_evaluation(
        encrypted_message,
        local_secret_contributions,
    )
    right_evaluation = distributed_secret_evaluation(
        encrypted_query,
        local_secret_contributions,
    )
    expected_raw_evaluation = left_evaluation.mul(
        right_evaluation
    )

    if not np.array_equal(
        raw_evaluation.coefficients,
        expected_raw_evaluation.coefficients,
    ):
        raise AssertionError(
            "Raw multiplication equation "
            "d0+d1*s+d2*s^2=(M+pη_M)(y+pη_y) failed"
        )
    print(
        "[Exact audit:Mult] "
        "d0+d1*s+d2*s^2=(M+p*eta_M)(y+p*eta_y)"
    )

    # Canonical relinearization must preserve the raw secret-key evaluation.
    relinearized = bgv.relinearize(raw_product)
    relin_evaluation = distributed_secret_evaluation(
        relinearized,
        local_secret_contributions,
    )

    relin_delta = relin_evaluation.sub(
        raw_evaluation
    ).centered_coefficients()
    if any(int(value) % cfg.P_MODULUS != 0 for value in relin_delta):
        raise AssertionError(
            "Relinearization introduced a non-p-multiple error"
        )
    print(
        "[Exact audit:Relin] "
        "canonical Enc_s(B^j*s^2) key switching preserved plaintext modulo p"
    )

    # Every canonical automorphism key must satisfy
    # Dec_s(Rot_k(ct)) = sigma_k(Dec_s(ct)) modulo p.
    rotated_once = bgv.rotate(
        relinearized,
        1,
    )
    rotated_eval = distributed_secret_evaluation(
        rotated_once,
        local_secret_contributions,
    )
    expected_rotated_eval = relin_evaluation.automorphism(
        encoder.rotation_exponent(1)
    )
    rotation_delta = rotated_eval.sub(
        expected_rotated_eval
    ).centered_coefficients()
    if any(int(value) % cfg.P_MODULUS != 0 for value in rotation_delta):
        raise AssertionError(
            "Automorphism key switching introduced a non-p-multiple error"
        )
    print(
        "[Exact audit:Rotation] "
        "rk_j=Enc_s(B^j*sigma_k(s)) preserves sigma_k plaintext modulo p"
    )

    evaluated = bgv.rotate_and_add_first_row(
        relinearized
    )
    expected_scalar = int(
        np.dot(
            message[:cfg.RUNTIME.model_dimension],
            query[:cfg.RUNTIME.model_dimension],
        )
    )

    for active in (
        (1, 2, 3),
        (1, 4, 5),
        (2, 3, 5),
    ):
        decrypted_message = decryptor.decrypt_slots(
            encrypted_message,
            active,
        )
        if not np.array_equal(
            decrypted_message[:cfg.RUNTIME.model_dimension],
            message[:cfg.RUNTIME.model_dimension],
        ):
            raise AssertionError(
                f"LISS decryption failed for qualified set {active}"
            )

        decrypted_scalar = int(
            decryptor.decrypt_slots(
                evaluated,
                active,
            )[0]
        )
        if decrypted_scalar != expected_scalar:
            raise AssertionError(
                f"Encrypted inner product failed for {active}: "
                f"{decrypted_scalar}!={expected_scalar}"
            )

    try:
        decryptor.decrypt_slots(
            encrypted_message,
            (1, 2),
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "An unauthorized two-party set decrypted successfully"
        )

    bgv.print_noise_budget(
        encrypted_message,
        cfg.THRESHOLD,
        "selftest_encryption",
    )
    bgv.print_noise_budget(
        evaluated,
        cfg.THRESHOLD,
        "selftest_inner_product",
    )

    print(
        "[Self-test] Balanced gadget decomposition, canonical rlk/rk, "
        "exact equations, noise bounds, and 3-of-5 LISS all passed"
    )


# =============================================================================
# MAIN
# =============================================================================


