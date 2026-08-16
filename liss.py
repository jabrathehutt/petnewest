from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from ring import Ring, RingElement

ShareLabel = Tuple[int, ...]
LISSShare = Dict[ShareLabel, RingElement]


class ReplicatedThresholdLISS:
    """
    Coefficient-wise replicated Linear Integer Secret Sharing (LISS)
    for a t-of-n threshold access structure.

    Let B_1, ..., B_L be the maximal unauthorized sets, i.e. all
    (t-1)-subsets of the participants. For an integer secret coefficient s,
    choose bounded integer masks r_1, ..., r_{L-1} and define

        z_{B_j} = r_j                  for j < L
        z_{B_L} = s - sum_j r_j.

    Component z_B is assigned to every participant outside B. Any qualified
    set Q with |Q| >= t therefore contains at least one holder of every
    component. Reconstruction uses one selected copy of every component,
    each with integer coefficient 1.

    The same integer construction is applied coefficient-wise to the BGV
    secret polynomial and then embedded into R_q. This is a proof-of-concept
    adaptation of LISS to Ring-BGV; it is not a claim that Damgard--Thorbek
    directly specify this Ring-BGV threshold-decryption construction.
    """

    def __init__(
        self,
        ring: Ring,
        participant_ids: Sequence[int],
        threshold: int,
    ) -> None:
        ids = tuple(int(pid) for pid in participant_ids)

        if len(set(ids)) != len(ids):
            raise ValueError("Participant identifiers must be unique")
        if threshold < 1 or threshold > len(ids):
            raise ValueError("Invalid LISS threshold")

        self.ring = ring
        self.participant_ids = ids
        self.threshold = int(threshold)
        self.labels: Tuple[ShareLabel, ...] = tuple(
            tuple(group)
            for group in combinations(
                self.participant_ids,
                self.threshold - 1,
            )
        )

        if not self.labels:
            raise AssertionError("Threshold LISS must contain at least one label")

    def _safe_mask_bound(
        self,
        dealer_count: int,
        max_secret_coefficient: int,
    ) -> int:
        """
        Choose bounded integer masks so that the exact integer shares remain
        comfortably inside the centered interval (-q/2,q/2), even after the
        corresponding shares from all dealers are added.
        """
        if dealer_count < 1:
            raise ValueError("dealer_count must be positive")
        if max_secret_coefficient < 0:
            raise ValueError("max_secret_coefficient must be non-negative")

        random_component_count = len(self.labels) - 1
        if random_component_count == 0:
            return 0

        half_q = self.ring.modulus // 2
        secret_budget = dealer_count * max_secret_coefficient
        available = half_q - secret_budget - 1

        if available <= 0:
            raise ValueError(
                "Ciphertext modulus is too small to embed the integer LISS shares"
            )

        # Factor 2 leaves centered-representation headroom.
        denominator = 2 * dealer_count * random_component_count
        bound = available // denominator

        # NumPy's integer RNG is signed-int64 based.
        bound = min(bound, (2**63 - 2) // 2)

        if bound < 2:
            raise ValueError(
                "Ciphertext modulus leaves insufficient room for bounded "
                "integer LISS masks"
            )

        return int(bound)

    def _sample_integer_vector(
        self,
        rng: np.random.Generator,
        bound: int,
    ) -> np.ndarray:
        if bound < 0:
            raise ValueError("Integer mask bound must be non-negative")
        if bound == 0:
            return np.zeros(self.ring.degree, dtype=object)

        values = rng.integers(
            -bound,
            bound + 1,
            size=self.ring.degree,
            dtype=np.int64,
        )
        return np.asarray([int(v) for v in values], dtype=object)

    def _embed_integer_vector(
        self,
        values: Sequence[int] | np.ndarray,
    ) -> RingElement:
        """
        Embed exact bounded integers into R_q while preserving their centered
        representatives.
        """
        ints = [int(v) for v in np.asarray(values, dtype=object).reshape(-1)]
        if len(ints) != self.ring.degree:
            raise ValueError("Integer LISS vector has wrong ring degree")

        half_q = self.ring.modulus // 2
        if any(abs(v) >= half_q for v in ints):
            raise OverflowError(
                "Integer LISS share coefficient reaches the centered q boundary"
            )

        element = self.ring.element(ints)
        centered = [int(v) for v in element.centered_coefficients()]
        if centered != ints:
            raise AssertionError(
                "Integer LISS embedding changed the centered representative"
            )
        return element

    def _share_secret_with_bound(
        self,
        secret: RingElement,
        rng: np.random.Generator,
        mask_bound: int,
    ) -> Dict[int, LISSShare]:
        if secret.ring != self.ring:
            raise ValueError("Secret ring mismatch")

        secret_integer = np.asarray(
            [int(v) for v in secret.centered_coefficients()],
            dtype=object,
        )

        components_integer: Dict[ShareLabel, np.ndarray] = {}
        running = np.zeros(self.ring.degree, dtype=object)

        for label in self.labels[:-1]:
            random_component = self._sample_integer_vector(rng, mask_bound)
            components_integer[label] = random_component
            running = running + random_component

        final_component = secret_integer - running
        components_integer[self.labels[-1]] = final_component

        reconstructed_integer = np.zeros(self.ring.degree, dtype=object)
        for component in components_integer.values():
            reconstructed_integer = reconstructed_integer + component

        if not np.array_equal(reconstructed_integer, secret_integer):
            raise AssertionError("Integer LISS dealer sharing failed to reconstruct")

        components: Dict[ShareLabel, RingElement] = {
            label: self._embed_integer_vector(component)
            for label, component in components_integer.items()
        }

        shares: Dict[int, LISSShare] = {
            participant: {}
            for participant in self.participant_ids
        }

        for label, component in components.items():
            for participant in self.participant_ids:
                if participant not in label:
                    shares[participant][label] = component

        return shares

    def share_secret(
        self,
        secret: RingElement,
        rng: np.random.Generator,
    ) -> Dict[int, LISSShare]:
        """
        Share one ring secret coefficient-wise using bounded integer masks.
        """
        if secret.ring != self.ring:
            raise ValueError("Secret ring mismatch")

        max_secret = max(
            (abs(int(v)) for v in secret.centered_coefficients()),
            default=0,
        )
        mask_bound = self._safe_mask_bound(
            dealer_count=1,
            max_secret_coefficient=max_secret,
        )
        return self._share_secret_with_bound(secret, rng, mask_bound)

    def distributed_share_sum(
        self,
        local_secrets: Mapping[int, RingElement],
        rng: np.random.Generator,
    ) -> Dict[int, LISSShare]:
        """
        LISS-share every local s_i independently and add corresponding units.
        By linearity, the result shares s = sum_i s_i without using s as an
        input to the sharing procedure.
        """
        if not local_secrets:
            raise ValueError("No local secrets supplied")

        unknown = set(local_secrets) - set(self.participant_ids)
        if unknown:
            raise ValueError(f"Unknown LISS dealers: {sorted(unknown)}")

        for secret in local_secrets.values():
            if secret.ring != self.ring:
                raise ValueError("Local secret ring mismatch")

        dealer_count = len(local_secrets)
        max_secret = max(
            (
                abs(int(value))
                for secret in local_secrets.values()
                for value in secret.centered_coefficients()
            ),
            default=0,
        )
        mask_bound = self._safe_mask_bound(
            dealer_count=dealer_count,
            max_secret_coefficient=max_secret,
        )

        result: Dict[int, LISSShare] = {
            participant: {
                label: self.ring.zero()
                for label in self.labels
                if participant not in label
            }
            for participant in self.participant_ids
        }

        for local_secret in local_secrets.values():
            dealer_shares = self._share_secret_with_bound(
                local_secret,
                rng,
                mask_bound,
            )
            for receiver, units in dealer_shares.items():
                for label, value in units.items():
                    result[receiver][label] = result[receiver][label].add(value)

        # Simulator-only algebraic assertion. Normal threshold decryption still
        # consumes distributed share units and does not reconstruct s.
        expected = self.ring.zero()
        for local_secret in local_secrets.values():
            expected = expected.add(local_secret)

        reconstructed = self.reconstruct(self.participant_ids, result)
        if not np.array_equal(
            reconstructed.coefficients,
            expected.coefficients,
        ):
            raise AssertionError(
                "Distributed integer LISS shares do not reconstruct "
                "the collective BGV secret"
            )

        return result

    def choose_units(
        self,
        active_ids: Sequence[int],
        shares: Mapping[int, LISSShare],
    ) -> Dict[int, List[RingElement]]:
        """
        Select exactly one available copy of every LISS component.

        The public reconstruction coefficient for each selected component is 1.
        """
        active = tuple(int(pid) for pid in active_ids)

        if len(set(active)) != len(active):
            raise ValueError("Active participant identifiers must be unique")
        if any(pid not in self.participant_ids for pid in active):
            raise ValueError("Active set contains an unknown participant")
        if len(active) < self.threshold:
            raise ValueError(
                f"Need at least {self.threshold} active parties, got {len(active)}"
            )

        selected: Dict[int, List[RingElement]] = {
            participant: []
            for participant in active
        }

        for label in self.labels:
            owner = next(
                (
                    participant
                    for participant in active
                    if label in shares[participant]
                ),
                None,
            )
            if owner is None:
                raise ValueError(
                    f"Qualified set cannot reconstruct LISS component {label}"
                )
            selected[owner].append(shares[owner][label])

        return selected

    def reconstruct(
        self,
        active_ids: Sequence[int],
        shares: Mapping[int, LISSShare],
    ) -> RingElement:
        """
        Test/helper reconstruction from a qualified set.

        Threshold decryption should use choose_units() instead, so the
        collective secret need not be reconstructed during normal execution.
        """
        selected = self.choose_units(active_ids, shares)
        result = self.ring.zero()

        for participant in active_ids:
            for unit in selected.get(int(participant), []):
                result = result.add(unit)

        return result

    def contributing_party_count(
        self,
        active_ids: Sequence[int],
        shares: Mapping[int, LISSShare],
    ) -> int:
        """
        Number of active parties assigned at least one selected share unit.
        """
        selected = self.choose_units(active_ids, shares)
        return sum(1 for units in selected.values() if units)
