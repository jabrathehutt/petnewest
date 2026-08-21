from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np

from bgv import EvaluationKeys, PublicKey
from config import P_MODULUS
from evaluation_keys import CanonicalEvaluationKeyGenerator
from liss import LISSShare, ThresholdLISS
from ring import Ring, RingElement, infinity_norm
from simd import SIMDEncoder


@dataclass(frozen=True)
class DKGResult:
    public_key: PublicKey
    local_secret_contributions: Dict[int, RingElement]
    square_shares: Dict[int, RingElement]
    liss_shares: Dict[int, LISSShare]
    evaluation_keys: EvaluationKeys
    liss: ThresholdLISS


class DistributedBGVSetup:
    """
    Distributed setup matching the thesis equations:

        b_i = -(a*s_i + p*e_i) mod q
        b   = sum_i b_i
        s   = sum_i s_i   (implicit; never reconstructed by a participant)

    Each local s_i is shared with the coefficient-wise threshold LISS in
    liss.py. By linearity, adding corresponding dealer share units yields a
    threshold sharing of the collective secret sum_i s_i with public
    integer reconstruction coefficients.

    Evaluation keys are generated separately in evaluation_keys.py.
    """

    def __init__(
        self,
        ring_q: Ring,
        encoder: SIMDEncoder,
        participant_ids: Sequence[int],
        threshold: int,
        seed: int,
    ) -> None:
        self.ring_q = ring_q
        self.encoder = encoder
        self.participant_ids = tuple(participant_ids)
        self.threshold = threshold
        self.rng = np.random.default_rng(seed)
        self.liss = ThresholdLISS(
            ring_q,
            participant_ids,
            threshold,
        )

    def generate(self) -> DKGResult:
        print("\n=== PHASE 2: DISTRIBUTED BGV SETUP ===")
        a = self.ring_q.random_uniform(self.rng)

        local_secrets: Dict[int, RingElement] = {}
        local_errors: Dict[int, RingElement] = {}
        local_public_parts: Dict[int, RingElement] = {}

        for participant in self.participant_ids:
            secret = self.ring_q.sample_ternary(self.rng)
            error = self.ring_q.sample_error(self.rng)
            local_secrets[participant] = secret
            local_errors[participant] = error
            local_public_parts[participant] = (
                a.mul(secret)
                .add(error.scalar_mul(P_MODULUS))
                .neg()
            )

        b = self.ring_q.zero()
        aggregate_error = self.ring_q.zero()
        for participant in self.participant_ids:
            b = b.add(local_public_parts[participant])
            aggregate_error = aggregate_error.add(local_errors[participant])

        public_key = PublicKey(
            a=a,
            b=b,
            public_key_error_bound=infinity_norm(aggregate_error),
            collective_secret_bound=max(
                1,
                sum(infinity_norm(secret) for secret in local_secrets.values()),
            ),
        )

        liss_shares = self.liss.distributed_share_sum(
            local_secrets,
            self.rng,
        )

        key_generator = CanonicalEvaluationKeyGenerator(
            ring_q=self.ring_q,
            encoder=self.encoder,
            participant_ids=self.participant_ids,
            seed=int(self.rng.integers(0, 2**31 - 1)),
        )
        square_shares, evaluation_keys = key_generator.generate(
            public_key,
            local_secrets,
        )

        print(
            "[DKG] Collective public key, coefficient-wise integer LISS shares, "
            "canonical rlk, and canonical rk generated"
        )
        print(
            "[DKG] Collective secret s was not reconstructed by any "
            "simulated protocol participant"
        )

        return DKGResult(
            public_key=public_key,
            local_secret_contributions=local_secrets,
            square_shares=square_shares,
            liss_shares=liss_shares,
            evaluation_keys=evaluation_keys,
            liss=self.liss,
        )
