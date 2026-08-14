from __future__ import annotations

from typing import List, Sequence

import numpy as np
import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

import config as cfg
from bgv import Ciphertext, ManualBGV, ThresholdDecryptor
from ring import Ring
from transport import decode_bigint_coefficients

class ThesisProtocolStrategy(fl.server.strategy.FedAvg):
    def __init__(
        self,
        ring_q: Ring,
        bgv: ManualBGV,
        decryptor: ThresholdDecryptor,
        encrypted_query: Ciphertext,
        committee_ids: Sequence[int],
    ):
        super().__init__(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=cfg.CLIENT_COUNT,
            min_evaluate_clients=cfg.CLIENT_COUNT,
            min_available_clients=cfg.CLIENT_COUNT,
        )

        self.ring_q = ring_q
        self.bgv = bgv
        self.decryptor = decryptor
        self.encrypted_query = encrypted_query
        self.committee_ids = tuple(committee_ids)

        if cfg.RUNTIME.model_dimension is None:
            raise RuntimeError("cfg.RUNTIME.model_dimension is not initialized")

        self.global_model = np.zeros(
            cfg.RUNTIME.model_dimension,
            dtype=np.float64,
        )
        self.round_history: List[dict] = []
        self.evaluation_history: List[dict] = []

    def initialize_parameters(self, client_manager):
        return ndarrays_to_parameters([self.global_model])

    def aggregate_fit(
        self,
        server_round,
        results,
        failures,
    ):
        print("\n" + "=" * 78)
        print(f"SERVER ROUND {server_round}/{cfg.NUM_ROUNDS}")
        print("=" * 78)

        if not results:
            return (
                ndarrays_to_parameters([self.global_model]),
                {},
            )

        ciphertexts: List[Ciphertext] = []

        for _, fit_result in results:
            arrays = parameters_to_ndarrays(
                fit_result.parameters
            )
            if len(arrays) != 2:
                raise ValueError(
                    "Each client must return c0 and c1"
                )
            ciphertexts.append(
                Ciphertext(
                    self.ring_q.element(
                        decode_bigint_coefficients(
                            arrays[0],
                            expected_count=self.ring_q.degree,
                            modulus=self.ring_q.modulus,
                        )
                    ),
                    self.ring_q.element(
                        decode_bigint_coefficients(
                            arrays[1],
                            expected_count=self.ring_q.degree,
                            modulus=self.ring_q.modulus,
                        )
                    ),
                    message_bound=int(
                        fit_result.metrics.get("message_bound", 0)
                    ),
                    noise_bound=int(
                        fit_result.metrics.get("noise_bound", 0)
                    ),
                    label=str(
                        fit_result.metrics.get("client_name", "client")
                    ),
                )
            )

        aggregate_ciphertext = self.bgv.aggregate(
            ciphertexts
        )
        print(
            f"[Server] Homomorphically aggregated "
            f"{len(ciphertexts)} ciphertexts"
        )

        # Demonstrate that any 3-of-5 qualified set can be selected.
        start = (server_round - 1) % cfg.CLIENT_COUNT
        active_committee = tuple(
            self.committee_ids[
                (start + offset) % cfg.CLIENT_COUNT
            ]
            for offset in range(cfg.THRESHOLD)
        )
        print(
            f"[Threshold] Active qualified set: "
            f"{active_committee}"
        )
        self.bgv.print_noise_budget(
            aggregate_ciphertext,
            len(active_committee),
            "aggregate",
        )

        # ---------------------------------------------------------------------
        # Model path: threshold-decrypt aggregate and perform FedAvg.
        # ---------------------------------------------------------------------
        aggregate_audit = self.decryptor.decrypt_slots_with_audit(
            aggregate_ciphertext,
            active_committee,
        )
        aggregate_slots = aggregate_audit.slots

        if cfg.RUNTIME.model_dimension is None:
            raise RuntimeError("cfg.RUNTIME.model_dimension is not initialized")

        quantized_sum = aggregate_slots[:cfg.RUNTIME.model_dimension]
        average_update = (
            quantized_sum.astype(np.float64)
            / (cfg.SCALING_FACTOR * len(ciphertexts))
        )

        old_model = self.global_model.copy()
        self.global_model = (
            self.global_model + average_update
        )

        # ---------------------------------------------------------------------
        # Authorized-query path:
        # ct-ct multiply -> rlk -> genuine automorphisms/rk -> threshold decrypt.
        # ---------------------------------------------------------------------
        query_result_ciphertext = (
            self.bgv.encrypted_inner_product(
                aggregate_ciphertext,
                self.encrypted_query,
            )
        )
        self.bgv.print_noise_budget(
            query_result_ciphertext,
            len(active_committee),
            "authorized_query",
        )

        if cfg.RUNTIME.model_dimension is None:
            raise RuntimeError(
                "cfg.RUNTIME.model_dimension is not initialized"
            )

        query_audit = self.decryptor.decrypt_slots_with_audit(
            query_result_ciphertext,
            active_committee,
        )
        query_slots = query_audit.slots

        if cfg.RUNTIME.authorized_query is None:
            raise RuntimeError("cfg.RUNTIME.authorized_query is not initialized")

        protocol_modulus_bound = self.bgv.print_protocol_modulus_report(
            aggregate_ciphertext=aggregate_ciphertext,
            encrypted_query=self.encrypted_query,
            evaluated_query_ciphertext=query_result_ciphertext,
            aggregate_audit=aggregate_audit,
            query_audit=query_audit,
            query_slots=cfg.RUNTIME.authorized_query,
            active_party_count=len(active_committee),
            model_dimension=cfg.RUNTIME.model_dimension,
            public_client_slot_bound=cfg.PUBLIC_CLIENT_SLOT_BOUND,
            client_count=len(ciphertexts),
            purpose=f"round_{server_round}",
        )

        encoded_scalar = int(query_slots[0])
        decoded_scalar = (
            encoded_scalar
            / (cfg.SCALING_FACTOR * len(ciphertexts))
        )

        validation_scalar = float(
            np.dot(
                average_update,
                cfg.RUNTIME.authorized_query[:cfg.RUNTIME.model_dimension],
            )
        )

        print(
            f"[Model path] Average update: "
            f"{np.round(average_update, 6)}"
        )
        print(
            f"[Model path] New global model: "
            f"{np.round(self.global_model, 6)}"
        )
        print(
            f"[Query path] Authorized scalar: "
            f"{decoded_scalar:.8f}"
        )
        print(
            f"[Query path] Plaintext validation: "
            f"{validation_scalar:.8f}"
        )
        print(
            f"[Query path] Absolute error: "
            f"{abs(decoded_scalar - validation_scalar):.8e}"
        )

        self.round_history.append(
            {
                "round": server_round,
                "active_committee": active_committee,
                "old_model": old_model,
                "average_update": average_update.copy(),
                "new_model": self.global_model.copy(),
                "authorized_scalar": decoded_scalar,
                "validation_scalar": validation_scalar,
                "protocol_modulus_bound": protocol_modulus_bound,
            }
        )

        return (
            ndarrays_to_parameters([self.global_model]),
            {},
        )

    def aggregate_evaluate(
        self,
        server_round,
        results,
        failures,
    ):
        if not results:
            return None

        total_examples = sum(
            result.num_examples
            for _, result in results
        )
        if total_examples <= 0:
            return None

        loss = sum(
            result.loss * result.num_examples
            for _, result in results
        ) / total_examples

        accuracy = sum(
            float(
                result.metrics.get(
                    "accuracy",
                    0.0,
                )
            )
            * result.num_examples
            for _, result in results
        ) / total_examples

        self.evaluation_history.append(
            {
                "round": server_round,
                "loss": float(loss),
                "accuracy": float(accuracy),
            }
        )

        print(
            f"[Evaluation] loss={loss:.6f}, "
            f"accuracy={accuracy:.4f}"
        )

        return (
            float(loss),
            {"accuracy": float(accuracy)},
        )


# =============================================================================
# EXACT TEXTBOOK BGV EQUATION AUDITOR (SELF-TEST ONLY)
# =============================================================================


