from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
METADATA_FILE = BASE_DIR / "federated_metadata.json"

CLIENT_NAMES = ("A", "B", "C", "D", "E")
CLIENT_IDS = tuple(range(1, 6))
CLIENT_COUNT = 5
THRESHOLD = 3

NUM_ROUNDS = 5
LOCAL_EPOCHS = 3
BATCH_SIZE = 256
LEARNING_RATE = 0.03
WEIGHT_DECAY = 1e-4
TEST_FRACTION = 0.20
RANDOM_SEED = 2026

# 69-bit prime satisfying q ≡ 1 (mod 2N) for N=64.
# This value is deliberately above the conservative raw-representative
# correctness bound produced by the encrypted inner-product circuit.
# Python arbitrary-precision integers are used by ring.py because q > 2**63.
Q_MODULUS = 295_147_905_179_352_836_353
P_MODULUS = 65_537
SCALING_FACTOR = 256
GADGET_BASE = 2**11
ERROR_STDDEV = 1.0
SMUDGING_BOUND = 2
PAILLIER_BITS = 384

# Public application bound used before the protocol runs. Every client must
# satisfy |round(S * update_j)| <= PUBLIC_CLIENT_SLOT_BOUND for every model
# coordinate. The client rejects an update that violates this assumption.
# With five clients and the registered query y=(1,1,0,...,0), 3000 gives
# B_M_public_slot=15000 and B_IP_public_slot=30000, both below p/2=32768.5.
PUBLIC_CLIENT_SLOT_BOUND = 3_000


@dataclass
class RuntimeState:
    model_dimension: Optional[int] = None
    ring_degree: Optional[int] = None
    row_size: Optional[int] = None
    authorized_query: Optional[np.ndarray] = None

    def require_dimensions(self) -> tuple[int, int, int]:
        if (
            self.model_dimension is None
            or self.ring_degree is None
            or self.row_size is None
        ):
            raise RuntimeError("Runtime dimensions have not been initialized")
        return self.model_dimension, self.ring_degree, self.row_size


RUNTIME = RuntimeState()
