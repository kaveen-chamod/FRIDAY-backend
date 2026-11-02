"""Reproducible local benchmark for F.R.I.D.A.Y.'s bounded predictive engine.

Run from the project root:
    python -m benchmarks.benchmark_predictive_engine
"""
from __future__ import annotations

import asyncio
import json
import os
import platform
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.predictive_engine import (
    SimulationCapacityError,
    SimulationJobManager,
    run_prediction,
)


_SPEC = {
    "action": "benchmark-model",
    "input_distributions": {
        "latency_jitter_ms": {"type": "uniform", "low": 0.0, "high": 50.0},
        "battery_voltage": {"type": "normal", "mean": 4.0, "stddev": 0.1},
        "obstacle_distance_m": {"type": "uniform", "low": 0.1, "high": 2.0},
    },
    "success_probability": 0.98,
    "success_conditions": [
        {"variable": "battery_voltage", "operator": ">=", "value": 3.5},
        {"variable": "obstacle_distance_m", "operator": ">=", "value": 0.2},
    ],
    "catastrophic_probability": 0.0005,
    "epistemic_uncertainty": 0.2,
    "model_source": "benchmark fixture",
    "model_version": "1",
}


async def _concurrency_benchmark() -> dict:
    manager = SimulationJobManager(
        max_concurrent_jobs=2,
        max_queued_jobs=10,
        timeout_seconds=10,
        max_iterations=1_000,
    )
    await manager.start()
    try:
        started = time.perf_counter()
        jobs = await asyncio.gather(*[
            manager.submit(_SPEC, profile="QUICK", seed=index)
            for index in range(4)
        ])
        snapshots = []
        for submitted in jobs:
            while True:
                status = await manager.get_job(submitted["job_id"])
                if status["status"] in {"completed", "failed", "timed_out"}:
                    snapshots.append(status)
                    break
                await asyncio.sleep(0.01)
        return {
            "job_count": len(jobs),
            "concurrent_workers": manager.max_concurrent_jobs,
            "elapsed_seconds": time.perf_counter() - started,
            "completed": sum(job["status"] == "completed" for job in snapshots),
            "resource_snapshot": await manager.resource_snapshot(),
        }
    finally:
        await manager.shutdown()


async def _saturation_benchmark() -> dict:
    manager = SimulationJobManager(
        max_concurrent_jobs=2,
        max_queued_jobs=10,
        timeout_seconds=10,
        max_iterations=1_000,
    )
    await manager.start()
    accepted = []
    rejected = 0
    started = time.perf_counter()
    try:
        for index in range(13):
            try:
                accepted.append(await manager.submit(
                    _SPEC, profile="QUICK", seed=10_000 + index
                ))
            except SimulationCapacityError:
                rejected += 1
        initial = await manager.resource_snapshot()
        for submitted in accepted:
            while True:
                status = await manager.get_job(submitted["job_id"])
                if status["status"] in {"completed", "failed", "timed_out"}:
                    break
                await asyncio.sleep(0.01)
        return {
            "accepted_jobs": len(accepted),
            "rejected_jobs": rejected,
            "queue_length_at_saturation": initial["queued_jobs"],
            "elapsed_seconds": time.perf_counter() - started,
            "resource_snapshot": await manager.resource_snapshot(),
        }
    finally:
        await manager.shutdown()


def main() -> None:
    results = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        },
        "single_process_batch_benchmarks": [],
    }
    for iterations in (1_000, 5_000, 10_000):
        started = time.perf_counter()
        result = run_prediction(
            _SPEC,
            profile="DEEP",
            seed=20261102,
            max_iterations=iterations,
            batch_size=250,
            stop_on_convergence=False,
        )
        results["single_process_batch_benchmarks"].append({
            "requested_iterations": iterations,
            "actual_iterations": result["sample_size"],
            "elapsed_seconds": time.perf_counter() - started,
            "worker_cpu_seconds": result["execution"]["cpu_seconds"],
            "sampled_process_rss_bytes": result["execution"]["max_sampled_rss_bytes"],
            "sampling_converged": result["sampling_converged"],
        })
    results["bounded_process_pool_concurrency"] = asyncio.run(
        _concurrency_benchmark()
    )
    results["worker_and_queue_saturation"] = asyncio.run(
        _saturation_benchmark()
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
