from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from ring import Ring, RingElement

# A share label is the minimal qualified set (block) to which the share unit
# belongs.  A participant holds one share unit for every block containing it.
ShareLabel = Tuple[int, ...]
LISSShare = Dict[ShareLabel, RingElement]


class ThresholdLISS:
    """
    Coefficient-wise Linear Integer Secret Sharing (LISS) for a t-of-n
    threshold access structure.

    The concrete proof-of-concept construction is an integer monotone-span
    realization built from the minimal qualified sets.  For every t-subset

        S = (P_{i_1}, ..., P_{i_t})

    one additive integer-sharing block of the same secret is created:

        sk_{S,i_1} = r_{S,1}
        ...
        sk_{S,i_{t-1}} = r_{S,t-1}
        sk_{S,i_t} = s - sum_j r_{S,j}.

    The t rows of one block therefore sum to the target vector
    epsilon=(1,0,...,0), so a qualified set containing S reconstructs with
    public integer coefficients d_{S,i}=1 for i in S.  Any set of at least t
    participants contains at least one such minimal qualified block; a set
    with fewer than t participants contains none.

    The same construction is applied coefficient-wise to the Ring-BGV secret
    polynomial and the bounded integer share units are then embedded in R_q.

    This is a concrete threshold-specific LISS realization for the thesis
    prototype.  It uses the integer-linearity and integer reconstruction
    properties of LISS; it is not claimed to be the general construction from
    Damgard--Thorbek.
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

        # Minimal qualified sets.  For the configured 3-of-5 structure there
        # are C(5,3)=10 blocks.
        self.blocks: Tuple[ShareLabel, ...] = tuple(
            tuple(group)
            for group in combinations(self.participant_ids, self.threshold)
        )
        if not self.blocks:
            raise AssertionError("Threshold LISS must contain at least one block")

        # Public integer-span representation used for documentation/auditing.
        # One secret column plus (t-1) independent random columns per block.
        self.random_columns_per_block = max(0, self.threshold - 1)
        self.column_count = 1 + len(self.blocks) * self.random_columns_per_block
        self.row_count = len(self.blocks) * self.threshold
        self.epsilon = np.zeros(self.column_count, dtype=object)
        self.epsilon[0] = 1

        self.distribution_matrix = np.zeros(
            (self.row_count, self.column_count),
            dtype=object,
        )
        self.row_owners: List[int] = []
        self.row_blocks: List[ShareLabel] = []

        row_index = 0
        for block_index, block in enumerate(self.blocks):
            random_start = 1 + block_index * self.random_columns_per_block

            if self.threshold == 1:
                self.distribution_matrix[row_index, 0] = 1
                self.row_owners.append(block[0])
                self.row_blocks.append(block)
                row_index += 1
                continue

            for position, participant in enumerate(block):
                row = self.distribution_matrix[row_index]
                if position < self.threshold - 1:
                    row[random_start + position] = 1
                else:
                    row[0] = 1
                    for random_offset in range(self.threshold - 1):
                        row[random_start + random_offset] = -1

                self.row_owners.append(participant)
                self.row_blocks.append(block)
                row_index += 1

        if row_index != self.row_count:
            raise AssertionError("LISS matrix row count mismatch")

        # Verify the integer reconstruction identity block-by-block.
        for block_index, _ in enumerate(self.blocks):
            start = block_index * self.threshold
            stop = start + self.threshold
            block_sum = np.sum(
                self.distribution_matrix[start:stop],
                axis=0,
                dtype=object,
            )
            if not np.array_equal(block_sum, self.epsilon):
                raise AssertionError(
                    "LISS block rows do not reconstruct the target vector"
                )

    @property
    def share_unit_count(self) -> int:
        return self.row_count

    @property
    def minimal_qualified_set_count(self) -> int:
        return len(self.blocks)

    def _safe_mask_bound(
        self,
        dealer_count: int,
        max_secret_coefficient: int,
    ) -> int:
        """
        Choose bounded integer masks so that exact integer share coefficients
        remain comfortably inside (-q/2,q/2), including after corresponding
        dealer share units are added.
        """
        if dealer_count < 1:
            raise ValueError("dealer_count must be positive")
        if max_secret_coefficient < 0:
            raise ValueError("max_secret_coefficient must be non-negative")

        random_term_count = self.threshold - 1
        if random_term_count == 0:
            return 0

        half_q = self.ring.modulus // 2
        secret_budget = dealer_count * max_secret_coefficient
        available = half_q - secret_budget - 1
        if available <= 0:
            raise ValueError(
                "Ciphertext modulus is too small to embed the integer LISS shares"
            )

        # The final unit in a block contains the secret minus (t-1) masks.
        # The extra factor 2 leaves centered-representation headroom.
        denominator = 2 * dealer_count * random_term_count
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
        ints = [int(v) for v in np.asarray(values, dtype=object).reshape(-1)]
        if len(ints) != self.ring.degree:
            raise ValueError("Integer LISS vector has wrong ring degree")

        half_q = self.ring.modulus // 2
        if any(abs(v) >= half_q for v in ints):
            raise OverflowError(
                "Integer LISS share coefficient reaches the centered q boundary"
            )

        element = self.ring.element(np.asarray(ints, dtype=object))
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

        shares_integer: Dict[int, Dict[ShareLabel, np.ndarray]] = {
            participant: {}
            for participant in self.participant_ids
        }

        for block in self.blocks:
            if self.threshold == 1:
                shares_integer[block[0]][block] = secret_integer.copy()
                continue

            masks = [
                self._sample_integer_vector(rng, mask_bound)
                for _ in range(self.threshold - 1)
            ]
            running = np.zeros(self.ring.degree, dtype=object)

            for participant, mask in zip(block[:-1], masks):
                shares_integer[participant][block] = mask
                running = running + mask

            final_share = secret_integer - running
            shares_integer[block[-1]][block] = final_share

            reconstructed = np.zeros(self.ring.degree, dtype=object)
            for participant in block:
                reconstructed = reconstructed + shares_integer[participant][block]
            if not np.array_equal(reconstructed, secret_integer):
                raise AssertionError(
                    "Integer LISS block failed to reconstruct dealer secret"
                )

        return {
            participant: {
                block: self._embed_integer_vector(value)
                for block, value in units.items()
            }
            for participant, units in shares_integer.items()
        }

    def share_secret(
        self,
        secret: RingElement,
        rng: np.random.Generator,
    ) -> Dict[int, LISSShare]:
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
        LISS-share every local s_i independently and add corresponding share
        units.  By linearity, the result is a sharing of s=sum_i s_i without
        ever supplying the reconstructed collective secret to the sharing
        algorithm.
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
                block: self.ring.zero()
                for block in self.blocks
                if participant in block
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
                for block, value in units.items():
                    result[receiver][block] = result[receiver][block].add(value)

        # Simulator-only algebraic assertion.
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

    def selected_block(self, active_ids: Sequence[int]) -> ShareLabel:
        """
        Return a canonical minimal qualified subset contained in active_ids.

        Larger qualified sets therefore also reconstruct: the combiner simply
        uses one contained t-subset and sets the reconstruction coefficient of
        every other active participant to zero.
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

        active_set = set(active)
        for block in self.blocks:
            if set(block).issubset(active_set):
                return block

        raise ValueError("Qualified set contains no reconstructing LISS block")

    def reconstruction_coefficients(
        self,
        active_ids: Sequence[int],
    ) -> Dict[int, int]:
        """
        Public participant-level reconstruction coefficients d_{Q,i} for the
        canonical selected minimal qualified block.

        In this concrete threshold LISS, d_{Q,i}=1 for the t participants of
        the selected block and d_{Q,i}=0 for any additional active members.
        """
        block = set(self.selected_block(active_ids))
        return {
            int(pid): (1 if int(pid) in block else 0)
            for pid in active_ids
        }

    def reconstruction_plan(
        self,
        active_ids: Sequence[int],
        shares: Mapping[int, LISSShare],
    ) -> Dict[int, Tuple[RingElement, int]]:
        """
        Return the concrete share unit and public integer coefficient used for
        each active participant.  Only the selected t-subset has nonzero
        coefficients.
        """
        block = self.selected_block(active_ids)
        coefficients = self.reconstruction_coefficients(active_ids)

        plan: Dict[int, Tuple[RingElement, int]] = {}
        for participant in active_ids:
            pid = int(participant)
            coefficient = int(coefficients[pid])
            if coefficient == 0:
                continue
            if pid not in shares or block not in shares[pid]:
                raise ValueError(
                    f"Participant {pid} does not hold required LISS share "
                    f"for block {block}"
                )
            plan[pid] = (shares[pid][block], coefficient)

        if len(plan) != self.threshold:
            raise AssertionError("LISS reconstruction plan has wrong size")
        return plan

    def reconstruct(
        self,
        active_ids: Sequence[int],
        shares: Mapping[int, LISSShare],
    ) -> RingElement:
        """
        Test/helper reconstruction from a qualified set.
        """
        plan = self.reconstruction_plan(active_ids, shares)
        result = self.ring.zero()
        for share_unit, coefficient in plan.values():
            result = result.add(share_unit.scalar_mul(coefficient))
        return result

    def reconstruction_weight(
        self,
        active_ids: Sequence[int],
    ) -> int:
        """
        Sum of absolute public integer reconstruction coefficients:
            Lambda_LISS(Q) = sum_i |d_{Q,i}|.
        """
        coefficients = self.reconstruction_coefficients(active_ids)
        return sum(abs(int(value)) for value in coefficients.values())

    def contributing_party_count(
        self,
        active_ids: Sequence[int],
        shares: Mapping[int, LISSShare],
    ) -> int:
        return len(self.reconstruction_plan(active_ids, shares))

