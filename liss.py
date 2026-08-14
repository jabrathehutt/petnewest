from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from ring import Ring, RingElement

ShareLabel = Tuple[int, ...]
LISSShare = Dict[ShareLabel, RingElement]

class ReplicatedThresholdLISS:
    """
    Replicated integer-linear secret sharing for a t-of-n access structure.

    For every maximal unauthorized set B of size t-1, a random component z_B
    is created and assigned to all parties outside B. The components sum to
    the shared ring secret. Every qualified set contains at least one holder
    of each component and reconstructs by summing one selected copy.
    """

    def __init__(
        self,
        ring: Ring,
        participant_ids: Sequence[int],
        threshold: int,
    ):
        if threshold < 1 or threshold > len(participant_ids):
            raise ValueError("Invalid LISS threshold")

        self.ring = ring
        self.participant_ids = tuple(participant_ids)
        self.threshold = threshold
        self.labels: Tuple[ShareLabel, ...] = tuple(
            tuple(group)
            for group in combinations(
                self.participant_ids,
                self.threshold - 1,
            )
        )

    def share_secret(
        self,
        secret: RingElement,
        rng: np.random.Generator,
    ) -> Dict[int, LISSShare]:
        if secret.ring != self.ring:
            raise ValueError("Secret ring mismatch")

        components: Dict[ShareLabel, RingElement] = {}
        running = self.ring.zero()

        for label in self.labels[:-1]:
            random_component = self.ring.random_uniform(rng)
            components[label] = random_component
            running = running.add(random_component)

        components[self.labels[-1]] = secret.sub(running)

        shares: Dict[int, LISSShare] = {
            participant: {}
            for participant in self.participant_ids
        }

        for label, component in components.items():
            for participant in self.participant_ids:
                if participant not in label:
                    shares[participant][label] = component

        return shares

    def distributed_share_sum(
        self,
        local_secrets: Mapping[int, RingElement],
        rng: np.random.Generator,
    ) -> Dict[int, LISSShare]:
        result: Dict[int, LISSShare] = {
            participant: {
                label: self.ring.zero()
                for label in self.labels
                if participant not in label
            }
            for participant in self.participant_ids
        }

        for local_secret in local_secrets.values():
            dealer_shares = self.share_secret(local_secret, rng)
            for receiver, units in dealer_shares.items():
                for label, value in units.items():
                    result[receiver][label] = result[receiver][label].add(value)

        return result

    def choose_units(
        self,
        active_ids: Sequence[int],
        shares: Mapping[int, LISSShare],
    ) -> Dict[int, List[RingElement]]:
        active = tuple(active_ids)
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


# =============================================================================
# MINIMAL PAILLIER FOR DISTRIBUTED s^2 SHARING
# =============================================================================


