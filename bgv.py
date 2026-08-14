from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Sequence, Tuple

import numpy as np

from config import GADGET_BASE, P_MODULUS, SMUDGING_BOUND
from liss import LISSShare, ReplicatedThresholdLISS
from ring import (
    Ring,
    RingElement,
    centered_array,
    gadget_digit_count,
    infinity_norm,
    negacyclic_product_bound,
)
from simd import SIMDEncoder

@dataclass(frozen=True)
class Ciphertext:
    c0: RingElement
    c1: RingElement
    # Public conservative bounds; these are not plaintexts or secret noise.
    message_bound: int = 0
    noise_bound: int = 0
    label: str = "ciphertext"

    def __post_init__(self) -> None:
        if self.c0.ring != self.c1.ring:
            raise ValueError("Ciphertext components must lie in the same ring")
        if self.message_bound < 0 or self.noise_bound < 0:
            raise ValueError("Ciphertext bounds must be non-negative")

    @property
    def ring(self) -> Ring:
        return self.c0.ring

    def add(self, other: "Ciphertext") -> "Ciphertext":
        return Ciphertext(
            self.c0.add(other.c0),
            self.c1.add(other.c1),
            message_bound=self.message_bound + other.message_bound,
            noise_bound=self.noise_bound + other.noise_bound,
            label=f"({self.label}+{other.label})",
        )


@dataclass(frozen=True)
class DegreeTwoCiphertext:
    d0: RingElement
    d1: RingElement
    d2: RingElement
    message_bound: int
    noise_bound: int
    label: str = "raw_product"


@dataclass(frozen=True)
class DecryptionAudit:
    """Exact data from one simulated threshold-decryption execution."""

    slots: np.ndarray
    plaintext_coefficient_bound: int
    exact_noise_bound: int
    representative_bound: int


@dataclass(frozen=True)
class PublicKey:
    a: RingElement
    b: RingElement
    public_key_error_bound: int
    collective_secret_bound: int


@dataclass(frozen=True)
class KeySwitchKey:
    """
    Canonical textbook key-switching key.

    Component j encrypts B^j * s_old under the current secret s:
        k_{0,j} + k_{1,j}s = B^j s_old + p rho_j.
    """
    components: Tuple[Ciphertext, ...]
    source_label: str


@dataclass(frozen=True)
class EvaluationKeys:
    relinearization_key: KeySwitchKey
    rotation_keys: Mapping[int, KeySwitchKey]


def decompose_ring_element(
    value: RingElement,
    base: int,
    digit_count: int,
) -> List[RingElement]:
    """
    Exact balanced coefficient decomposition.

    For every centered coefficient x, compute digits d_j in
    [-base/2, base/2) such that

        x = sum_j d_j base^j

    over the integers. Ring reconstruction is therefore exact modulo q.
    """
    if base < 2 or base % 2:
        raise ValueError("Balanced gadget base must be an even integer")

    ring = value.ring
    digit_polynomials = [
        np.zeros(ring.degree, dtype=np.int64)
        for _ in range(digit_count)
    ]

    centered = value.centered_coefficients()

    for coefficient_index, coefficient_raw in enumerate(centered):
        remaining = int(coefficient_raw)

        for digit_index in range(digit_count):
            digit = ((remaining + base // 2) % base) - base // 2
            digit_polynomials[digit_index][coefficient_index] = digit
            remaining = (remaining - digit) // base

        if remaining != 0:
            raise OverflowError(
                "Gadget digit count is insufficient for centered decomposition"
            )

    digits = [
        ring.element(polynomial)
        for polynomial in digit_polynomials
    ]

    # Exact internal reconstruction check.
    reconstructed = ring.zero()
    power = 1
    for digit in digits:
        reconstructed = reconstructed.add(digit.scalar_mul(power))
        power *= base

    if not np.array_equal(
        reconstructed.coefficients,
        value.coefficients,
    ):
        raise AssertionError("Balanced gadget decomposition did not reconstruct")

    return digits


# =============================================================================
# REPLICATED 3-OF-5 LISS
# =============================================================================




class ManualBGV:
    def __init__(
        self,
        ring_q: Ring,
        ring_p: Ring,
        encoder: SIMDEncoder,
        public_key: PublicKey,
        evaluation_keys: EvaluationKeys,
        seed: int,
    ):
        self.ring_q = ring_q
        self.ring_p = ring_p
        self.encoder = encoder
        self.public_key = public_key
        self.evaluation_keys = evaluation_keys
        self.rng = np.random.default_rng(seed)
        self.digit_count = gadget_digit_count(
            ring_q.modulus,
            GADGET_BASE,
        )
        self.public_key_error_bound = public_key.public_key_error_bound
        self.collective_secret_bound = public_key.collective_secret_bound

    def encrypt_slots(
        self,
        slots: Sequence[int] | np.ndarray,
        label: str = "encryption",
    ) -> Ciphertext:
        plaintext_p = self.encoder.encode(slots)
        # Use the centered coefficient lift from R_p into R_q.  This is
        # congruent to the encoded plaintext modulo p, but it makes the
        # decomposition c0+c1*s = m+p*eta use the same centered plaintext
        # representative as the protocol-design coefficient bounds.
        plaintext_q = self.ring_q.element(
            plaintext_p.centered_coefficients()
        )

        u = self.ring_q.sample_ternary(self.rng)
        e0 = self.ring_q.sample_error(self.rng)
        e1 = self.ring_q.sample_error(self.rng)

        c0 = (
            self.public_key.b.mul(u)
            .add(e0.scalar_mul(P_MODULUS))
            .add(plaintext_q)
        )
        c1 = self.public_key.a.mul(u).add(
            e1.scalar_mul(P_MODULUS)
        )

        n = self.ring_q.degree
        exact_public_bound = (
            infinity_norm(e0)
            + negacyclic_product_bound(
                n,
                infinity_norm(e1),
                self.collective_secret_bound,
            )
            + negacyclic_product_bound(
                n,
                self.public_key_error_bound,
                infinity_norm(u),
            )
        )

        return Ciphertext(
            c0,
            c1,
            message_bound=infinity_norm(plaintext_q),
            noise_bound=exact_public_bound,
            label=label,
        )

    @staticmethod
    def aggregate(ciphertexts: Sequence[Ciphertext]) -> Ciphertext:
        if not ciphertexts:
            raise ValueError("Cannot aggregate an empty ciphertext sequence")
        result = ciphertexts[0]
        for ciphertext in ciphertexts[1:]:
            result = result.add(ciphertext)
        return result

    @staticmethod
    def multiply(
        left: Ciphertext,
        right: Ciphertext,
    ) -> DegreeTwoCiphertext:
        n = left.ring.degree
        message_bound = negacyclic_product_bound(
            n,
            left.message_bound,
            right.message_bound,
        )
        noise_bound = (
            negacyclic_product_bound(
                n,
                left.message_bound,
                right.noise_bound,
            )
            + negacyclic_product_bound(
                n,
                right.message_bound,
                left.noise_bound,
            )
            + P_MODULUS
            * negacyclic_product_bound(
                n,
                left.noise_bound,
                right.noise_bound,
            )
        )

        return DegreeTwoCiphertext(
            d0=left.c0.mul(right.c0),
            d1=left.c0.mul(right.c1).add(left.c1.mul(right.c0)),
            d2=left.c1.mul(right.c1),
            message_bound=message_bound,
            noise_bound=noise_bound,
            label=f"{left.label}*{right.label}",
        )

    def key_switch(
        self,
        c0: RingElement,
        c1_under_old_secret: RingElement,
        key: KeySwitchKey,
    ) -> Ciphertext:
        digits = decompose_ring_element(
            c1_under_old_secret,
            GADGET_BASE,
            self.digit_count,
        )
        if len(key.components) != len(digits):
            raise ValueError("Key-switch gadget length mismatch")

        new_c0 = c0
        new_c1 = self.ring_q.zero()
        switching_noise_bound = 0

        for digit, component in zip(digits, key.components):
            new_c0 = new_c0.add(digit.mul(component.c0))
            new_c1 = new_c1.add(digit.mul(component.c1))
            switching_noise_bound += negacyclic_product_bound(
                self.ring_q.degree,
                infinity_norm(digit),
                component.noise_bound,
            )

        return Ciphertext(
            new_c0,
            new_c1,
            message_bound=0,
            noise_bound=switching_noise_bound,
            label=f"keyswitch[{key.source_label}]",
        )

    def relinearize(
        self,
        ciphertext: DegreeTwoCiphertext,
    ) -> Ciphertext:
        digits = decompose_ring_element(
            ciphertext.d2,
            GADGET_BASE,
            self.digit_count,
        )
        key = self.evaluation_keys.relinearization_key

        if len(key.components) != len(digits):
            raise ValueError("Relinearization-key gadget length mismatch")

        c0 = ciphertext.d0
        c1 = ciphertext.d1
        relinearization_noise = 0

        for digit, component in zip(digits, key.components):
            c0 = c0.add(digit.mul(component.c0))
            c1 = c1.add(digit.mul(component.c1))
            relinearization_noise += negacyclic_product_bound(
                self.ring_q.degree,
                infinity_norm(digit),
                component.noise_bound,
            )

        return Ciphertext(
            c0,
            c1,
            message_bound=ciphertext.message_bound,
            noise_bound=ciphertext.noise_bound + relinearization_noise,
            label=f"Relin({ciphertext.label})",
        )

    def rotate(
        self,
        ciphertext: Ciphertext,
        offset: int,
    ) -> Ciphertext:
        key = self.evaluation_keys.rotation_keys.get(offset)
        if key is None:
            raise ValueError(f"No rotation key for offset {offset}")

        exponent = self.encoder.rotation_exponent(offset)
        transformed_c0 = ciphertext.c0.automorphism(exponent)
        transformed_c1 = ciphertext.c1.automorphism(exponent)

        switched = self.key_switch(
            transformed_c0,
            transformed_c1,
            key,
        )
        return Ciphertext(
            switched.c0,
            switched.c1,
            message_bound=ciphertext.message_bound,
            noise_bound=ciphertext.noise_bound + switched.noise_bound,
            label=f"Rot_{offset}({ciphertext.label})",
        )

    def rotate_and_add_first_row(
        self,
        ciphertext: Ciphertext,
    ) -> Ciphertext:
        accumulated = ciphertext
        offset = 1

        while offset < self.encoder.row_size:
            accumulated = accumulated.add(
                self.rotate(accumulated, offset)
            )
            offset *= 2

        return accumulated

    @staticmethod
    def representative_bound(
        ciphertext: Ciphertext,
        active_party_count: int,
    ) -> int:
        threshold_noise = active_party_count * SMUDGING_BOUND
        return (
            ciphertext.message_bound
            + P_MODULUS
            * (ciphertext.noise_bound + threshold_noise)
        )

    @staticmethod
    def print_noise_budget(
        ciphertext: Ciphertext,
        active_party_count: int,
        purpose: str,
    ) -> None:
        representative = ManualBGV.representative_bound(
            ciphertext,
            active_party_count,
        )
        half_q = ciphertext.ring.modulus // 2
        remaining = half_q - representative
        print(
            f"[Noise:{purpose}] message_bound={ciphertext.message_bound}, "
            f"noise_bound={ciphertext.noise_bound}, "
            f"|representative|<={representative}, "
            f"q/2={half_q}, remaining_margin={remaining}"
        )
        if remaining <= 0:
            print(
                f"[Noise:{purpose}:generic] NOT CERTIFIED: the generic "
                "recursive envelope exceeds q/2. This does not report an "
                "observed decryption failure; the exact execution audit is "
                "reported separately."
            )

    def print_protocol_modulus_report(
        self,
        aggregate_ciphertext: Ciphertext,
        encrypted_query: Ciphertext,
        evaluated_query_ciphertext: Ciphertext,
        aggregate_audit: DecryptionAudit,
        query_audit: DecryptionAudit,
        query_slots: Sequence[int] | np.ndarray,
        active_party_count: int,
        model_dimension: int,
        public_client_slot_bound: int,
        client_count: int,
        purpose: str,
    ) -> dict:
        """Print a domain-correct modulus report without changing the protocol.

        The report deliberately separates four notions:

        1. an a-priori application bound in the SIMD slot domain;
        2. round-specific observed slot values;
        3. public, conservative coefficient/noise envelopes propagated by the
           implementation; and
        4. exact post-execution representatives measured by the simulator.

        Only slot-domain quantities determine the plaintext-modulus condition.
        Coefficient-domain quantities determine the BGV representative bound
        modulo q.  For the symbolic q condition, the implementation uses the
        raw plaintext representative propagated through multiplication and
        rotate-and-add together with the correspondingly propagated noise.
        The canonical encoded output is used only by the exact execution audit.
        """
        if active_party_count < 1:
            raise ValueError("active_party_count must be positive")
        if model_dimension < 1:
            raise ValueError("model_dimension must be positive")
        if public_client_slot_bound < 0:
            raise ValueError("public_client_slot_bound must be non-negative")
        if client_count < 1:
            raise ValueError("client_count must be positive")

        aggregate_slots = np.asarray(aggregate_audit.slots, dtype=np.int64)
        y_slots = np.asarray(query_slots, dtype=np.int64)
        if aggregate_slots.size < model_dimension or y_slots.size < model_dimension:
            raise ValueError("Insufficient SIMD slots for modulus report")

        model_values = aggregate_slots[:model_dimension]
        query_values = y_slots[:model_dimension]

        # ------------------------------------------------------------------
        # A-priori plaintext-modulus condition: SIMD slot domain.
        # ------------------------------------------------------------------
        b_client_slot_public = int(public_client_slot_bound)
        b_m_slot_public = int(client_count) * b_client_slot_public
        b_y_slot_public = (
            int(np.max(np.abs(query_values))) if query_values.size else 0
        )
        query_l1 = int(np.sum(np.abs(query_values)))
        b_ip_slot_public_dimension = (
            model_dimension * b_m_slot_public * b_y_slot_public
        )
        b_ip_slot_public_query = b_m_slot_public * query_l1

        minimum_p_dimension = 2 * max(
            b_m_slot_public,
            b_ip_slot_public_dimension,
        ) + 1
        minimum_p_query_specific = 2 * max(
            b_m_slot_public,
            b_ip_slot_public_query,
        ) + 1

        # ------------------------------------------------------------------
        # Observed plaintext values: still slot-domain, but not parameter proof.
        # ------------------------------------------------------------------
        b_m_slot_observed = (
            int(np.max(np.abs(model_values))) if model_values.size else 0
        )
        b_y_slot_observed = b_y_slot_public
        actual_inner_product = int(np.dot(model_values, query_values))
        b_ip_slot_observed_bound = b_m_slot_observed * query_l1
        minimum_p_observed = 2 * max(
            b_m_slot_observed,
            abs(actual_inner_product),
        ) + 1

        # ------------------------------------------------------------------
        # Ciphertext modulus q: one internally consistent raw-representative
        # decomposition.  The symbolic tracker follows the plaintext
        # polynomial produced by the actual multiplication/rotation circuit,
        # before reduction to the canonical representative modulo p.
        # ------------------------------------------------------------------
        b_m_coeff_canonical = int(aggregate_audit.plaintext_coefficient_bound)
        b_y_coeff_canonical = int(encrypted_query.message_bound)
        b_ip_coeff_canonical = int(query_audit.plaintext_coefficient_bound)
        b_ip_coeff_raw = int(evaluated_query_ciphertext.message_bound)

        b_eta_m_public = int(aggregate_ciphertext.noise_bound)
        b_eta_y_public = int(encrypted_query.noise_bound)
        b_eta_ip_public = int(evaluated_query_ciphertext.noise_bound)
        threshold_noise = int(active_party_count) * int(SMUDGING_BOUND)
        n = self.ring_q.degree

        # This is the generic inequality ||ab||_inf <= N||a||_inf||b||_inf.
        # It is intentionally retained only as a stress diagnostic.
        generic_product_coefficient_envelope = negacyclic_product_bound(
            n,
            b_m_coeff_canonical,
            b_y_coeff_canonical,
        )
        generic_m_eta_y = negacyclic_product_bound(
            n,
            b_m_coeff_canonical,
            b_eta_y_public,
        )
        generic_y_eta_m = negacyclic_product_bound(
            n,
            b_y_coeff_canonical,
            b_eta_m_public,
        )
        generic_eta_m_eta_y = negacyclic_product_bound(
            n,
            b_eta_m_public,
            b_eta_y_public,
        )
        generic_raw_product_noise_envelope = (
            generic_m_eta_y
            + generic_y_eta_m
            + P_MODULUS * generic_eta_m_eta_y
        )

        aggregate_public_runtime_representative = (
            b_m_coeff_canonical
            + P_MODULUS * (b_eta_m_public + threshold_noise)
        )
        # IMPORTANT: pair the recursively propagated operation noise with
        # the recursively propagated RAW plaintext representative.  Pairing
        # the operation noise with the canonical output polynomial would omit
        # the quotient kappa in F(M*y)=Encode(<M,y>)+p*kappa.
        query_public_runtime_representative = (
            b_ip_coeff_raw
            + P_MODULUS * (b_eta_ip_public + threshold_noise)
        )
        minimum_q_public_runtime = 2 * max(
            aggregate_public_runtime_representative,
            query_public_runtime_representative,
        ) + 1

        # Exact simulator audit for this execution. It is evidence of execution
        # correctness, not a replacement for an a-priori parameter proof.
        minimum_q_observed = 2 * max(
            int(aggregate_audit.representative_bound),
            int(query_audit.representative_bound),
        ) + 1

        selected_p = int(P_MODULUS)
        selected_q = int(self.ring_q.modulus)

        print(
            f"[Modulus:{purpose}:p_public] "
            f"B_client_slot={b_client_slot_public}, clients={client_count}, "
            f"B_M_slot_public={b_m_slot_public}, B_y_slot={b_y_slot_public}, "
            f"||y||_1={query_l1}"
        )
        print(
            f"[Modulus:{purpose}:p_registered_query] query_bound="
            f"{b_ip_slot_public_query}, minimum_p="
            f"{minimum_p_query_specific}, selected_p={selected_p}, "
            f"margin={selected_p-minimum_p_query_specific}, status=PASS"
        )
        print(
            f"[Modulus:{purpose}:p_dense_comparison] dense_query_bound="
            f"{b_ip_slot_public_dimension}, minimum_p_dense="
            f"{minimum_p_dimension}, selected_p={selected_p}, status="
            f"{'PASS' if selected_p >= minimum_p_dimension else 'NOT SUPPORTED'}"
        )
        print(
            f"[Modulus:{purpose}:p_observed] B_M_slot_observed="
            f"{b_m_slot_observed}, actual_inner_product={actual_inner_product}, "
            f"B_M_slot*||y||_1={b_ip_slot_observed_bound}, "
            f"minimum_p_observed={minimum_p_observed}, selected_p={selected_p}, "
            f"margin={selected_p-minimum_p_observed}"
        )
        print(
            f"[Modulus:{purpose}:coeff_centered] B_M_coeff_centered="
            f"{b_m_coeff_canonical}, B_y_coeff_centered={b_y_coeff_canonical}, "
            f"B_IP_coeff_canonical={b_ip_coeff_canonical}, "
            f"B_IP_coeff_raw_envelope={b_ip_coeff_raw}, "
            f"B_eta_M={b_eta_m_public}, B_eta_y={b_eta_y_public}, "
            f"B_eta_IP_raw={b_eta_ip_public}"
        )
        print(
            f"[Modulus:{purpose}:generic_diagnostic] "
            f"generic_product_coefficient_envelope="
            f"{generic_product_coefficient_envelope}, "
            f"generic_raw_product_noise_envelope="
            f"{generic_raw_product_noise_envelope}"
        )
        print(
            f"[Modulus:{purpose}:q_generic_diagnostic] aggregate_bound="
            f"{aggregate_public_runtime_representative}, query_bound="
            f"{query_public_runtime_representative}, minimum_q="
            f"{minimum_q_public_runtime}, selected_q={selected_q}, "
            f"margin={selected_q-minimum_q_public_runtime}, status="
            f"{'CERTIFIED' if selected_q >= minimum_q_public_runtime else 'NOT CERTIFIED'}"
        )
        print(
            f"[Modulus:{purpose}:q_exact_execution] aggregate_exact_noise="
            f"{aggregate_audit.exact_noise_bound}, query_exact_noise="
            f"{query_audit.exact_noise_bound}, aggregate_representative="
            f"{aggregate_audit.representative_bound}, query_representative="
            f"{query_audit.representative_bound}, minimum_q="
            f"{minimum_q_observed}, selected_q={selected_q}, "
            f"margin={selected_q-minimum_q_observed}, status=PASS"
        )

        if selected_p < minimum_p_query_specific:
            raise RuntimeError(
                "The selected plaintext modulus does not satisfy the public "
                "query-specific slot-domain bound."
            )
        if b_m_slot_observed > b_m_slot_public:
            raise RuntimeError(
                "The observed aggregate exceeds the declared public aggregate "
                "slot bound."
            )
        if selected_q < minimum_q_public_runtime:
            raise RuntimeError(
                "The selected ciphertext modulus does not satisfy the "
                "conservative raw-representative correctness bound: "
                f"selected_q={selected_q}, required_q>={minimum_q_public_runtime}."
            )
        if selected_q < minimum_q_observed:
            raise RuntimeError(
                "The exact executed representative reaches or exceeds q/2; "
                "decryption correctness is not certified for this run."
            )

        return {
            "minimum_p_dense_public": minimum_p_dimension,
            "minimum_p_query_specific_public": minimum_p_query_specific,
            "p_public_margin": selected_p - minimum_p_query_specific,
            "minimum_p_observed": minimum_p_observed,
            "p_observed_margin": selected_p - minimum_p_observed,
            "minimum_q_public_runtime": minimum_q_public_runtime,
            "q_public_runtime_margin": selected_q - minimum_q_public_runtime,
            "minimum_q_observed": minimum_q_observed,
            "q_observed_margin": selected_q - minimum_q_observed,
            "B_client_slot_public": b_client_slot_public,
            "B_M_slot_public": b_m_slot_public,
            "B_M_slot_observed": b_m_slot_observed,
            "B_y_slot": b_y_slot_observed,
            "query_l1": query_l1,
            "B_M_coeff_canonical": b_m_coeff_canonical,
            "B_y_coeff_canonical": b_y_coeff_canonical,
            "B_IP_coeff_canonical": b_ip_coeff_canonical,
            "B_IP_coeff_raw_envelope": b_ip_coeff_raw,
            "generic_product_coefficient_envelope": (
                generic_product_coefficient_envelope
            ),
        }

    def encrypted_inner_product(
        self,
        aggregate_ciphertext: Ciphertext,
        encrypted_query: Ciphertext,
    ) -> Ciphertext:
        raw_product = self.multiply(
            aggregate_ciphertext,
            encrypted_query,
        )
        relinearized = self.relinearize(raw_product)
        return self.rotate_and_add_first_row(relinearized)


# =============================================================================
# LISS THRESHOLD DECRYPTION WITH FRESH SMUDGING
# =============================================================================


class ThresholdDecryptor:
    def __init__(
        self,
        ring_q: Ring,
        ring_p: Ring,
        encoder: SIMDEncoder,
        liss: ReplicatedThresholdLISS,
        shares: Mapping[int, LISSShare],
        seed: int,
    ):
        self.ring_q = ring_q
        self.ring_p = ring_p
        self.encoder = encoder
        self.liss = liss
        self.shares = shares
        self.rng = np.random.default_rng(seed)

    def decrypt_slots_with_audit(
        self,
        ciphertext: Ciphertext,
        active_ids: Sequence[int],
    ) -> DecryptionAudit:
        selected = self.liss.choose_units(active_ids, self.shares)
        combined = self.ring_q.zero()

        for participant in active_ids:
            selected_units = selected.get(participant, [])
            if not selected_units:
                continue

            aggregate_share = self.ring_q.zero()
            for unit in selected_units:
                aggregate_share = aggregate_share.add(unit)

            smudging = self.ring_q.sample_smudging(self.rng)
            partial = ciphertext.c1.mul(aggregate_share).add(
                smudging.scalar_mul(P_MODULUS)
            )
            combined = combined.add(partial)

        representative = ciphertext.c0.add(combined)
        centered_q = representative.centered_coefficients()
        plaintext_coefficients = [
            int(value) % P_MODULUS for value in centered_q
        ]
        plaintext = self.ring_p.element(plaintext_coefficients)
        slots = self.encoder.decode(plaintext)

        # Re-encode the recovered slot vector to its canonical plaintext
        # polynomial and extract the exact executed eta from v=m+p*eta.
        canonical_p = self.encoder.encode(slots)
        canonical_q = self.ring_q.element(
            canonical_p.centered_coefficients()
        )
        residual = representative.sub(canonical_q).centered_coefficients()
        noise_coefficients = []
        for value_raw in residual:
            value = int(value_raw)
            if value % P_MODULUS != 0:
                raise AssertionError(
                    "Threshold-decryption audit failed: v-m is not divisible by p"
                )
            noise_coefficients.append(value // P_MODULUS)

        exact_noise = self.ring_q.element(noise_coefficients)
        return DecryptionAudit(
            slots=slots,
            plaintext_coefficient_bound=infinity_norm(canonical_q),
            exact_noise_bound=infinity_norm(exact_noise),
            representative_bound=infinity_norm(representative),
        )

    def decrypt_slots(
        self,
        ciphertext: Ciphertext,
        active_ids: Sequence[int],
    ) -> np.ndarray:
        return self.decrypt_slots_with_audit(ciphertext, active_ids).slots

