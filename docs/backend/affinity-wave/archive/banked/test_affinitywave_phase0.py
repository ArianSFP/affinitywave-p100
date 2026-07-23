#!/usr/bin/env python3

import importlib.util
import sqlite3
import struct
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("affinitywave_phase0", HERE / "affinitywave_phase0.py")
AW = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = AW
SPEC.loader.exec_module(AW)


def synthetic_observations(records_per_layer=4):
    result = []
    for layer in range(AW.N_LAYER):
        for record in range(records_per_layer):
            counts = [0] * AW.N_EXPERT
            for expert in range(AW.N_EXPERT):
                counts[expert] = AW.N_USED if expert == (layer + record) % AW.N_EXPERT else 0
            result.append(AW.RouteObservation(layer, 1, tuple(counts)))
    return result


class AffinityWaveTests(unittest.TestCase):
    def test_awtr_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "routes.awtr"
            routes = tuple(tuple(range(AW.N_USED)) for _ in range(3))
            with path.open("wb") as handle:
                handle.write(AW.AWTR_MAGIC)
                handle.write(AW.AWTR_HEADER.pack(7, len(routes), AW.N_USED))
                handle.write(struct.pack(f"<{len(routes) * AW.N_USED}H", *(e for row in routes for e in row)))
            loaded = AW.load_awtr(path)
            self.assertEqual(loaded[0].layer, 7)
            self.assertEqual(loaded[0].routes, routes)
            self.assertEqual(sum(loaded[0].counts), 3 * AW.N_USED)

    def test_manifest_is_deterministic_and_capacity_constrained(self):
        calibration, holdout = AW.split_calibration_holdout(synthetic_observations())
        digest = "a" * 64
        first, _, _ = AW.build_manifest(calibration, holdout, digest, "test", 16, 1)
        second, _, _ = AW.build_manifest(calibration, holdout, digest, "test", 16, 1)
        self.assertEqual(first, second)
        for layer in first["placement"]:
            owners = layer["primary_owner"]
            self.assertEqual([owners.count(rank) for rank in range(AW.N_GPU)], [64] * AW.N_GPU)
            self.assertEqual(len(layer["replicated_experts"]), 16)
            self.assertEqual(
                [sum(owners[expert] == rank for expert in layer["replicated_experts"]) for rank in range(AW.N_GPU)],
                [4] * AW.N_GPU,
            )

    def test_trace_cost_exclusions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.sqlite"
            connection = sqlite3.connect(path)
            connection.executescript(
                """
                CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
                    start INTEGER NOT NULL, end INTEGER NOT NULL, deviceId INTEGER NOT NULL,
                    shortName INTEGER NOT NULL
                );
                """
            )
            names = ["dense", "ncclDevKernel_AllReduce_X", "moe_gemm_q8_plan", "k_get_rows_float", "moe_plan_build"]
            connection.executemany("INSERT INTO StringIds VALUES (?,?)", enumerate(names, 1))
            for device in range(AW.N_GPU):
                start = 0
                for index, _name in enumerate(names, 1):
                    connection.execute(
                        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?)",
                        (start, start + 1_000_000_000, device, index),
                    )
                    start += 1_000_000_000
            connection.commit()
            connection.close()
            costs = AW.extract_trace_costs(path, passes=1, pfold_savings_ms=0)
            self.assertEqual(costs.kernel_s_per_gpu, (5.0,) * AW.N_GPU)
            self.assertEqual(costs.fixed_lower_s_per_gpu, (1.0,) * AW.N_GPU)

    def test_simulator_has_43_diagonals(self):
        calibration, holdout = AW.split_calibration_holdout(synthetic_observations())
        _, owners, hot = AW.build_manifest(calibration, holdout, "b" * 64, "test", 16, 0)
        trace = AW.TraceCosts(
            passes=1,
            span_s=1.0,
            kernel_s_per_gpu=(1.0,) * 4,
            nccl_s_per_gpu=(0.0,) * 4,
            expert_s_per_gpu=(0.0,) * 4,
            get_rows_s_per_gpu=(0.0,) * 4,
            plan_s_per_gpu=(0.0,) * 4,
            fixed_lower_s_per_gpu=(1.0,) * 4,
            top_kernels=(),
        )
        result = AW.simulate_wave(
            holdout, owners, hot, trace, chunk_tokens=1, service_tflops=5.5,
            peer_gbps=10.0, request_bytes=8192, response_bytes=4096,
            gdn_state_bytes=0, kv_key_length=0, kv_value_length=0,
        )
        self.assertEqual(result["diagonals"], 43)
        self.assertGreaterEqual(result["seconds"], 43 / 40)


if __name__ == "__main__":
    unittest.main()
