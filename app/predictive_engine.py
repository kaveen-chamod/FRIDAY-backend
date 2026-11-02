"""Bounded, reproducible Monte Carlo analysis for caller-supplied models.

The estimates produced here describe only the supplied distributions and rules.
They are not measured real-world probabilities or proof of physical safety.
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import math
import os
import random
import sys
import time
import uuid
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Mapping


_LOG = logging.getLogger("friday.predictive")
_MAX_ITERATIONS = 10_000
_MAX_INPUTS = 32
_MAX_FAILURE_MODES = 32
_MAX_CONDITIONS = 32
_MAX_TRACKED_JOBS = 1_000
_JOB_RETENTION_SECONDS = 3_600
_WILSON_Z_95 = 1.959963984540054
_ALLOWED_OPERATORS = {"<", "<=", ">", ">=", "==", "!="}

_PROFILES = {
    "QUICK": {"max_iterations": 1_000, "target_width": 0.10},
    "STANDARD": {"max_iterations": 5_000, "target_width": 0.05},
    "DEEP": {"max_iterations": 10_000, "target_width": 0.02},
}


class SimulationInputError(ValueError):
    """The supplied simulation model is incomplete or invalid."""


class SimulationCapacityError(RuntimeError):
    """The bounded simulation service cannot accept another job."""


class SimulationServiceClosed(RuntimeError):
    """The simulation job service is not accepting jobs."""


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SimulationInputError(f"{label} must be a finite number.")
    result = float(value)
    if not math.isfinite(result) or abs(result) > 1_000_000_000_000:
        raise SimulationInputError(
            f"{label} must be finite and between -1e12 and 1e12."
        )
    return result


def _normalize_distribution(
    value: Any, label: str, *, probability: bool = False
) -> dict[str, float | str]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = _finite_number(value, label)
        if probability and not 0.0 <= number <= 1.0:
            raise SimulationInputError(f"{label} must be between 0 and 1.")
        return {"type": "constant", "value": number}
    if not isinstance(value, Mapping):
        raise SimulationInputError(f"{label} must be a number or distribution object.")

    kind = str(value.get("type", "")).strip().lower()
    if kind == "constant":
        number = _finite_number(value.get("value"), f"{label}.value")
        if probability and not 0.0 <= number <= 1.0:
            raise SimulationInputError(f"{label} must be between 0 and 1.")
        result: dict[str, float | str] = {"type": kind, "value": number}
    elif kind == "uniform":
        low = _finite_number(value.get("low"), f"{label}.low")
        high = _finite_number(value.get("high"), f"{label}.high")
        if low > high:
            raise SimulationInputError(f"{label}.low cannot exceed {label}.high.")
        if probability and (low < 0.0 or high > 1.0):
            raise SimulationInputError(f"{label} must remain between 0 and 1.")
        result = {"type": kind, "low": low, "high": high}
    elif kind == "normal":
        mean = _finite_number(value.get("mean"), f"{label}.mean")
        stddev = _finite_number(value.get("stddev"), f"{label}.stddev")
        if stddev < 0:
            raise SimulationInputError(f"{label}.stddev cannot be negative.")
        minimum = value.get("minimum")
        maximum = value.get("maximum")
        if (minimum is None) != (maximum is None):
            raise SimulationInputError(
                f"{label} must specify both minimum and maximum, or neither."
            )
        result = {"type": kind, "mean": mean, "stddev": stddev}
        if minimum is not None:
            minimum_value = _finite_number(minimum, f"{label}.minimum")
            maximum_value = _finite_number(maximum, f"{label}.maximum")
            if minimum_value > maximum_value:
                raise SimulationInputError(
                    f"{label}.minimum cannot exceed {label}.maximum."
                )
            if probability and (minimum_value < 0.0 or maximum_value > 1.0):
                raise SimulationInputError(f"{label} must remain between 0 and 1.")
            result.update({"minimum": minimum_value, "maximum": maximum_value})
        elif probability:
            raise SimulationInputError(
                f"{label} normal distribution must bound values within 0 and 1."
            )
    else:
        raise SimulationInputError(
            f"{label}.type must be constant, uniform, or normal."
        )
    return result


def _distribution_can_be_positive(distribution: Mapping[str, Any]) -> bool:
    kind = distribution["type"]
    if kind == "constant":
        return float(distribution["value"]) > 0.0
    if kind == "uniform":
        return float(distribution["high"]) > 0.0
    if "maximum" in distribution:
        return float(distribution["maximum"]) > 0.0
    return (
        float(distribution["mean"])
        + 3.0 * float(distribution["stddev"])
        > 0.0
    )


def _normalize_conditions(
    value: Any, label: str, input_names: set[str]
) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > _MAX_CONDITIONS:
        raise SimulationInputError(
            f"{label} must be a list of at most {_MAX_CONDITIONS} conditions."
        )
    result: list[dict[str, Any]] = []
    for index, condition in enumerate(value):
        prefix = f"{label}[{index}]"
        if not isinstance(condition, Mapping):
            raise SimulationInputError(f"{prefix} must be an object.")
        name = condition.get("variable")
        operator = condition.get("operator")
        if not isinstance(name, str) or name not in input_names:
            raise SimulationInputError(f"{prefix}.variable must name a declared input.")
        if operator not in _ALLOWED_OPERATORS:
            raise SimulationInputError(f"{prefix}.operator is not supported.")
        if "value" not in condition:
            raise SimulationInputError(f"{prefix}.value is required.")
        result.append({
            "variable": name,
            "operator": operator,
            "value": _finite_number(condition["value"], f"{prefix}.value"),
        })
    return result


def normalize_action_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize a data-only simulation model; no code is evaluated."""
    if not isinstance(spec, Mapping):
        raise SimulationInputError("The action specification must be an object.")
    action = spec.get("action")
    if not isinstance(action, str) or not action.strip() or len(action) > 256:
        raise SimulationInputError("action must be a non-empty string of at most 256 characters.")

    raw_inputs = spec.get("input_distributions", {})
    if not isinstance(raw_inputs, Mapping) or len(raw_inputs) > _MAX_INPUTS:
        raise SimulationInputError(
            f"input_distributions must contain at most {_MAX_INPUTS} variables."
        )
    inputs: dict[str, dict[str, float | str]] = {}
    for name, distribution in raw_inputs.items():
        if not isinstance(name, str) or not name.strip() or len(name) > 64:
            raise SimulationInputError("Input variable names must be 1 to 64 characters.")
        inputs[name] = _normalize_distribution(
            distribution, f"input_distributions.{name}"
        )

    input_names = set(inputs)
    success_conditions = _normalize_conditions(
        spec.get("success_conditions"), "success_conditions", input_names
    )
    catastrophic_conditions = _normalize_conditions(
        spec.get("catastrophic_conditions"), "catastrophic_conditions", input_names
    )

    raw_modes = spec.get("failure_modes", [])
    if not isinstance(raw_modes, list) or len(raw_modes) > _MAX_FAILURE_MODES:
        raise SimulationInputError(
            f"failure_modes must contain at most {_MAX_FAILURE_MODES} entries."
        )
    failure_modes: list[dict[str, Any]] = []
    mode_names: set[str] = set()
    for index, mode in enumerate(raw_modes):
        prefix = f"failure_modes[{index}]"
        if not isinstance(mode, Mapping):
            raise SimulationInputError(f"{prefix} must be an object.")
        name = mode.get("name")
        if not isinstance(name, str) or not name.strip() or len(name) > 96:
            raise SimulationInputError(f"{prefix}.name must be 1 to 96 characters.")
        if name in mode_names:
            raise SimulationInputError(f"Failure mode {name!r} is duplicated.")
        mode_names.add(name)
        probability = _normalize_distribution(
            mode.get("probability", 0.0), f"{prefix}.probability", probability=True
        )
        conditions = _normalize_conditions(
            mode.get("conditions"), f"{prefix}.conditions", input_names
        )
        if not conditions and not _distribution_can_be_positive(probability):
            raise SimulationInputError(
                f"{prefix} must define a non-zero probability or conditions."
            )
        failure_modes.append({
            "name": name,
            "probability": probability,
            "conditions": conditions,
        })

    epistemic_uncertainty = spec.get("epistemic_uncertainty")
    if epistemic_uncertainty is not None:
        epistemic_uncertainty = _finite_number(
            epistemic_uncertainty, "epistemic_uncertainty"
        )
        if not 0.0 <= epistemic_uncertainty <= 1.0:
            raise SimulationInputError(
                "epistemic_uncertainty must be between 0 and 1."
            )
    model_source = spec.get("model_source", "unspecified")
    model_version = spec.get("model_version", "unspecified")
    if not isinstance(model_source, str) or len(model_source) > 256:
        raise SimulationInputError("model_source must be a string of at most 256 characters.")
    if not isinstance(model_version, str) or len(model_version) > 64:
        raise SimulationInputError("model_version must be a string of at most 64 characters.")

    return {
        "action": action.strip(),
        "input_distributions": inputs,
        "success_probability": _normalize_distribution(
            spec.get("success_probability", 1.0),
            "success_probability",
            probability=True,
        ),
        "success_conditions": success_conditions,
        "catastrophic_probability": _normalize_distribution(
            spec.get("catastrophic_probability", 0.0),
            "catastrophic_probability",
            probability=True,
        ),
        "catastrophic_conditions": catastrophic_conditions,
        "failure_modes": failure_modes,
        "epistemic_uncertainty": epistemic_uncertainty,
        "model_source": model_source.strip() or "unspecified",
        "model_version": model_version.strip() or "unspecified",
    }


def _sample_distribution(
    distribution: Mapping[str, Any], rng: random.Random
) -> float:
    kind = distribution["type"]
    if kind == "constant":
        return float(distribution["value"])
    if kind == "uniform":
        return rng.uniform(float(distribution["low"]), float(distribution["high"]))
    value = rng.gauss(float(distribution["mean"]), float(distribution["stddev"]))
    if "minimum" in distribution:
        value = min(
            float(distribution["maximum"]),
            max(float(distribution["minimum"]), value),
        )
    return value


def _matches_conditions(
    conditions: list[Mapping[str, Any]], values: Mapping[str, float]
) -> bool:
    for condition in conditions:
        left = values[condition["variable"]]
        right = float(condition["value"])
        operator = condition["operator"]
        if operator == "<" and not left < right:
            return False
        if operator == "<=" and not left <= right:
            return False
        if operator == ">" and not left > right:
            return False
        if operator == ">=" and not left >= right:
            return False
        if operator == "==" and not left == right:
            return False
        if operator == "!=" and not left != right:
            return False
    return True


def _current_rss_bytes() -> int | None:
    """Best-effort resident memory measurement on Windows, Linux, and macOS."""
    try:
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        process = ctypes.windll.kernel32.GetCurrentProcess()
        get_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_info.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        get_info.restype = wintypes.BOOL
        if get_info(process, ctypes.byref(counters), counters.cb):
            return int(counters.WorkingSetSize)
    except (OSError, ImportError, AttributeError):
        pass
    try:
        if os.path.exists("/proc/self/statm"):
            with open("/proc/self/statm", encoding="ascii") as stream:
                resident_pages = int(stream.read().split()[1])
            return resident_pages * os.sysconf("SC_PAGE_SIZE")
        import resource

        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return peak if sys.platform == "darwin" else peak * 1024
    except (OSError, ImportError, IndexError, ValueError):
        return None


def _simulate_batch(
    spec: Mapping[str, Any], seed: int, sample_count: int
) -> dict[str, Any]:
    rng = random.Random(seed)
    successes = 0
    catastrophes = 0
    mode_counts: Counter[str] = Counter()
    cpu_start = time.process_time_ns()
    peak_rss = _current_rss_bytes()
    for _ in range(sample_count):
        values = {
            name: _sample_distribution(distribution, rng)
            for name, distribution in spec["input_distributions"].items()
        }
        catastrophic = (
            rng.random()
            < _sample_distribution(spec["catastrophic_probability"], rng)
        )
        catastrophic = catastrophic or (
            bool(spec["catastrophic_conditions"])
            and _matches_conditions(spec["catastrophic_conditions"], values)
        )

        failed = (
            rng.random() >= _sample_distribution(spec["success_probability"], rng)
            or not _matches_conditions(spec["success_conditions"], values)
        )
        for mode in spec["failure_modes"]:
            mode_failed = (
                rng.random() < _sample_distribution(mode["probability"], rng)
                or (
                    bool(mode["conditions"])
                    and _matches_conditions(mode["conditions"], values)
                )
            )
            if mode_failed:
                failed = True
                mode_counts[mode["name"]] += 1
        if catastrophic:
            catastrophes += 1
            failed = True
        if not failed:
            successes += 1
    return {
        "samples": sample_count,
        "successes": successes,
        "catastrophes": catastrophes,
        "failure_modes": dict(mode_counts),
        "cpu_seconds": (time.process_time_ns() - cpu_start) / 1_000_000_000,
        "rss_bytes": max(peak_rss or 0, _current_rss_bytes() or 0) or None,
    }


def wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    """Return a two-sided 95% Wilson score interval for a binomial estimate."""
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("Wilson interval requires 0 <= successes <= trials and trials > 0.")
    z2 = _WILSON_Z_95**2
    p = successes / trials
    denominator = 1.0 + z2 / trials
    center = (p + z2 / (2.0 * trials)) / denominator
    margin = (
        _WILSON_Z_95
        * math.sqrt((p * (1.0 - p) / trials) + (z2 / (4.0 * trials**2)))
        / denominator
    )
    if successes == 0:
        return 0.0, min(1.0, center + margin)
    if successes == trials:
        return max(0.0, center - margin), 1.0
    return max(0.0, center - margin), min(1.0, center + margin)


def _build_result(
    spec: Mapping[str, Any],
    *,
    profile: str,
    seed: int,
    totals: Mapping[str, Any],
    target_width: float,
    max_iterations: int,
    timed_out: bool = False,
) -> dict[str, Any]:
    samples = int(totals["samples"])
    successes = int(totals["successes"])
    failures = samples - successes
    catastrophes = int(totals["catastrophes"])
    success_probability = successes / samples if samples else 0.0
    failure_probability = failures / samples if samples else 1.0
    catastrophic_risk = catastrophes / samples if samples else 1.0
    success_ci = wilson_interval(successes, samples) if samples else (0.0, 1.0)
    failure_ci = wilson_interval(failures, samples) if samples else (0.0, 1.0)
    catastrophic_ci = (
        wilson_interval(catastrophes, samples) if samples else (0.0, 1.0)
    )
    statistical_width = success_ci[1] - success_ci[0]
    catastrophic_width = catastrophic_ci[1] - catastrophic_ci[0]
    sampling_converged = max(statistical_width, catastrophic_width) <= target_width
    epistemic = spec["epistemic_uncertainty"]
    uncertainty_score = max(
        statistical_width,
        epistemic if epistemic is not None else 1.0,
    )

    if catastrophes == samples and samples:
        recommendation = "BLOCKED"
    elif catastrophic_risk > 0.20 or success_probability < 0.60:
        recommendation = "STRONGLY_DISCOURAGED"
    elif (
        success_probability >= 0.85
        and catastrophic_risk <= 0.05
        and sampling_converged
        and epistemic is not None
        and epistemic <= 0.10
        and spec["model_source"] != "unspecified"
        and spec["model_version"] != "unspecified"
    ):
        recommendation = "RECOMMENDED"
    else:
        recommendation = "PROCEED_WITH_CAUTION"

    mode_counts = totals["failure_modes"]
    ranked_modes = []
    for name, count in sorted(
        mode_counts.items(), key=lambda item: (-item[1], item[0])
    )[:10]:
        mode_ci = wilson_interval(int(count), samples) if samples else (0.0, 1.0)
        ranked_modes.append({
            "name": name,
            "observed_count": int(count),
            "estimated_probability": int(count) / samples if samples else 0.0,
            "confidence_interval_95": {
                "lower": mode_ci[0],
                "upper": mode_ci[1],
            },
        })

    warnings = [
        "Monte Carlo estimates are conditional on the supplied model; they are not verified real-world probabilities.",
        "Input variables and failure-mode events are sampled independently; correlations must be represented explicitly in the model.",
    ]
    if epistemic is None:
        warnings.append(
            "Epistemic/model uncertainty was not supplied; recommendation is capped at PROCEED_WITH_CAUTION."
        )
    if not sampling_converged and samples >= max_iterations:
        warnings.append(
            "The iteration budget was exhausted before the requested sampling confidence-interval width was reached."
        )
    if catastrophes == 0:
        warnings.append(
            "No catastrophic sample was observed; this does not establish zero catastrophic risk."
        )
    if timed_out:
        warnings.append(
            "The time limit was reached; the returned estimates use partial samples only."
        )

    return {
        "action": spec["action"],
        "profile": profile,
        "seed": seed,
        "sample_size": samples,
        "observed_failure_count": failures,
        "observed_catastrophic_count": catastrophes,
        "success_probability": success_probability,
        "failure_probability": failure_probability,
        "catastrophic_risk_score": catastrophic_risk,
        "confidence_intervals_95": {
            "success_probability": {
                "lower": success_ci[0],
                "upper": success_ci[1],
            },
            "failure_probability": {
                "lower": failure_ci[0],
                "upper": failure_ci[1],
            },
            "catastrophic_risk_score": {
                "lower": catastrophic_ci[0],
                "upper": catastrophic_ci[1],
            },
        },
        "uncertainty_score": min(1.0, uncertainty_score),
        "uncertainty_components": {
            "aleatoric_sampling_interval_width": statistical_width,
            "catastrophic_sampling_interval_width": catastrophic_width,
            "epistemic_model_uncertainty": epistemic,
        },
        "sampling_converged": sampling_converged,
        "convergence_target_interval_width": target_width,
        "failure_modes": ranked_modes,
        "recommendation": recommendation,
        "probability_basis": (
            "Monte Carlo estimate under caller-supplied distributions; not a measured real-world probability."
        ),
        "model": {
            "source": spec["model_source"],
            "version": spec["model_version"],
        },
        "model_assumptions": {
            "input_distributions": copy.deepcopy(spec["input_distributions"]),
            "success_probability": copy.deepcopy(spec["success_probability"]),
            "success_conditions": copy.deepcopy(spec["success_conditions"]),
            "catastrophic_probability": copy.deepcopy(
                spec["catastrophic_probability"]
            ),
            "catastrophic_conditions": copy.deepcopy(
                spec["catastrophic_conditions"]
            ),
            "failure_modes": copy.deepcopy(spec["failure_modes"]),
            "sampling": "Independent draws unless represented by shared explicit rules.",
        },
        "execution": {
            "cpu_seconds": float(totals["cpu_seconds"]),
            "max_sampled_rss_bytes": totals["peak_worker_rss_bytes"],
            "timed_out": timed_out,
        },
        "warnings": warnings,
    }


def run_prediction(
    action_spec: Mapping[str, Any],
    *,
    profile: str = "STANDARD",
    seed: int | None = None,
    max_iterations: int | None = None,
    batch_size: int = 250,
    stop_on_convergence: bool = True,
) -> dict[str, Any]:
    """Run bounded, adaptive Monte Carlo in the current process.

    Production request handlers should submit work through SimulationJobManager
    so CPU-bound batches run in a bounded process pool.
    """
    normalized = normalize_action_spec(action_spec)
    profile_name = str(profile).upper()
    if profile_name not in _PROFILES:
        raise SimulationInputError("profile must be QUICK, STANDARD, or DEEP.")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 1_000:
        raise SimulationInputError("batch_size must be between 1 and 1000.")
    if not isinstance(stop_on_convergence, bool):
        raise SimulationInputError("stop_on_convergence must be a boolean.")
    profile_settings = _PROFILES[profile_name]
    maximum = profile_settings["max_iterations"] if max_iterations is None else max_iterations
    if (
        isinstance(maximum, bool)
        or not isinstance(maximum, int)
        or not 1_000 <= maximum <= min(_MAX_ITERATIONS, profile_settings["max_iterations"])
    ):
        raise SimulationInputError(
            f"max_iterations must be between 1000 and {profile_settings['max_iterations']} for {profile_name}."
        )
    if seed is None:
        seed = random.SystemRandom().getrandbits(63)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise SimulationInputError("seed must be an integer between 0 and 2^63-1.")

    seed_source = random.Random(seed)
    totals: dict[str, Any] = {
        "samples": 0,
        "successes": 0,
        "catastrophes": 0,
        "failure_modes": Counter(),
        "cpu_seconds": 0.0,
        "peak_worker_rss_bytes": None,
    }
    target = profile_settings["target_width"]
    initial_samples = min(1_000, maximum)
    while totals["samples"] < maximum:
        count = min(batch_size, maximum - totals["samples"])
        batch = _simulate_batch(normalized, seed_source.getrandbits(63), count)
        _merge_batch(totals, batch)
        partial = _build_result(
            normalized,
            profile=profile_name,
            seed=seed,
            totals=totals,
            target_width=target,
            max_iterations=maximum,
        )
        if (
            stop_on_convergence
            and totals["samples"] >= initial_samples
            and partial["sampling_converged"]
        ):
            break
    return _build_result(
        normalized,
        profile=profile_name,
        seed=seed,
        totals=totals,
        target_width=target,
        max_iterations=maximum,
    )


def _merge_batch(totals: dict[str, Any], batch: Mapping[str, Any]) -> None:
    totals["samples"] += int(batch["samples"])
    totals["successes"] += int(batch["successes"])
    totals["catastrophes"] += int(batch["catastrophes"])
    totals["failure_modes"].update(batch["failure_modes"])
    totals["cpu_seconds"] += float(batch["cpu_seconds"])
    rss = batch["rss_bytes"]
    if rss is not None:
        previous = totals["peak_worker_rss_bytes"]
        totals["peak_worker_rss_bytes"] = max(int(previous or 0), int(rss))


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass
class _SimulationJob:
    job_id: str
    spec: dict[str, Any]
    profile: str
    seed: int
    created_at: float
    created_at_iso: str
    idempotency_key: str | None = None
    request_hash: str | None = None
    status: str = "queued"
    samples_completed: int = 0
    result: dict[str, Any] | None = None
    error: str | None = None
    cancel_requested: bool = False
    updated_at: float = field(default_factory=time.monotonic)

    def public_view(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "profile": self.profile,
            "samples_completed": self.samples_completed,
            "created_at": self.created_at_iso,
            "error": self.error,
            "result": copy.deepcopy(self.result),
        }


class SimulationJobManager:
    """Fixed-worker, bounded-queue job service for asynchronous integrations."""

    def __init__(
        self,
        *,
        max_concurrent_jobs: int = 2,
        max_queued_jobs: int = 10,
        timeout_seconds: float = 10.0,
        batch_size: int = 250,
        max_iterations: int = 10_000,
        max_memory_bytes: int = 256 * 1024 * 1024,
        executor_factory: Callable[..., Any] = ProcessPoolExecutor,
        on_completed: Callable[[dict[str, Any]], Mapping[str, Any] | None] | None = None,
    ) -> None:
        if not 1 <= max_concurrent_jobs <= 2:
            raise ValueError("max_concurrent_jobs must be 1 or 2.")
        if not 1 <= max_queued_jobs <= 100:
            raise ValueError("max_queued_jobs must be between 1 and 100.")
        if not math.isfinite(timeout_seconds) or not 0.1 <= timeout_seconds <= 60:
            raise ValueError("timeout_seconds must be between 0.1 and 60.")
        if not 1 <= batch_size <= 1_000:
            raise ValueError("batch_size must be between 1 and 1000.")
        if not 1_000 <= max_iterations <= _MAX_ITERATIONS:
            raise ValueError(f"max_iterations must be between 1000 and {_MAX_ITERATIONS}.")
        if not 16 * 1024 * 1024 <= max_memory_bytes <= 2 * 1024 * 1024 * 1024:
            raise ValueError("max_memory_bytes must be between 16 MiB and 2 GiB.")
        self.max_concurrent_jobs = max_concurrent_jobs
        self.max_queued_jobs = max_queued_jobs
        self.timeout_seconds = timeout_seconds
        self.batch_size = batch_size
        self.max_iterations = max_iterations
        self.max_memory_bytes = max_memory_bytes
        self._executor_factory = executor_factory
        self._on_completed = on_completed
        self._queue: asyncio.Queue[str | None] | None = None
        self._executor: Any = None
        self._workers: list[asyncio.Task] = []
        self._jobs: dict[str, _SimulationJob] = {}
        self._idempotency: dict[str, tuple[str, str, float]] = {}
        self._closed = False
        self._lock: asyncio.Lock | None = None
        self._total_cpu_seconds = 0.0
        self._total_samples = 0
        self._active_jobs = 0

    async def start(self) -> None:
        if self._workers:
            return
        if self._closed:
            raise SimulationServiceClosed("A closed simulation service cannot be restarted.")
        self._queue = asyncio.Queue(maxsize=self.max_queued_jobs)
        self._lock = asyncio.Lock()
        self._executor = self._executor_factory(
            max_workers=self.max_concurrent_jobs
        )
        self._workers = [
            asyncio.create_task(self._worker(), name=f"friday-simulation-{index}")
            for index in range(self.max_concurrent_jobs)
        ]

    async def submit(
        self,
        action_spec: Mapping[str, Any],
        *,
        profile: str = "STANDARD",
        seed: int | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        if self._closed or self._queue is None:
            raise SimulationServiceClosed("Simulation service is not running.")
        current_rss = _current_rss_bytes()
        if current_rss is not None and current_rss > self.max_memory_bytes:
            raise SimulationCapacityError(
                "Simulation rejected because process memory is above its configured limit."
            )
        profile_name = str(profile).upper()
        if profile_name not in _PROFILES:
            raise SimulationInputError("profile must be QUICK, STANDARD, or DEEP.")
        normalized = normalize_action_spec(action_spec)
        effective_limit = min(
            self.max_iterations, _PROFILES[profile_name]["max_iterations"]
        )
        seed_was_supplied = seed is not None
        if seed is None:
            seed = random.SystemRandom().getrandbits(63)
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
            raise SimulationInputError("seed must be an integer between 0 and 2^63-1.")
        request_hash = _canonical_hash({
            "spec": normalized,
            "profile": profile_name,
            "seed": seed if seed_was_supplied else None,
            "seed_was_supplied": seed_was_supplied,
            "max_iterations": effective_limit,
        })
        if idempotency_key is not None:
            if not isinstance(idempotency_key, str) or not 1 <= len(idempotency_key) <= 128:
                raise SimulationInputError("idempotency_key must be 1 to 128 characters.")
        self._prune_jobs()
        if idempotency_key is not None:
            previous = self._idempotency.get(idempotency_key)
            if previous:
                prior_hash, prior_job_id, _ = previous
                if prior_hash != request_hash:
                    raise SimulationInputError(
                        "idempotency_key was already used for a different simulation request."
                    )
                existing = self._jobs.get(prior_job_id)
                if existing:
                    return existing.public_view()
        job = _SimulationJob(
            job_id=uuid.uuid4().hex,
            spec=normalized,
            profile=profile_name,
            seed=seed,
            created_at=time.monotonic(),
            created_at_iso=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            idempotency_key=idempotency_key,
            request_hash=request_hash,
        )
        try:
            self._queue.put_nowait(job.job_id)
        except asyncio.QueueFull as exc:
            raise SimulationCapacityError(
                "Simulation queue is full; retry after a running job completes."
            ) from exc
        self._jobs[job.job_id] = job
        if idempotency_key is not None:
            self._idempotency[idempotency_key] = (
                request_hash,
                job.job_id,
                job.created_at,
            )
        self._prune_jobs()
        return job.public_view()

    async def get_job(self, job_id: str) -> dict[str, Any] | None:
        self._prune_jobs()
        job = self._jobs.get(job_id)
        return job.public_view() if job else None

    async def cancel(self, job_id: str) -> dict[str, Any] | None:
        job = self._jobs.get(job_id)
        if job is None:
            return None
        if job.status in {"completed", "failed", "cancelled", "timed_out"}:
            return job.public_view()
        job.cancel_requested = True
        job.updated_at = time.monotonic()
        if job.status == "queued":
            job.status = "cancelled"
        return job.public_view()

    async def resource_snapshot(self) -> dict[str, Any]:
        rss = _current_rss_bytes()
        return {
            "active_jobs": self._active_jobs,
            "queued_jobs": self._queue.qsize() if self._queue else 0,
            "max_concurrent_jobs": self.max_concurrent_jobs,
            "max_queued_jobs": self.max_queued_jobs,
            "process_cpu_seconds": time.process_time_ns() / 1_000_000_000,
            "process_rss_bytes": rss,
            "max_memory_bytes": self.max_memory_bytes,
            "completed_samples": self._total_samples,
            "worker_cpu_seconds": self._total_cpu_seconds,
        }

    async def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        for job in self._jobs.values():
            if job.status in {"queued", "running"}:
                job.cancel_requested = True
                if job.status == "queued":
                    job.status = "cancelled"
        if self._queue is not None:
            for _ in self._workers:
                await self._queue.put(None)
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        if self._executor is not None:
            executor = self._executor
            self._executor = None
            await asyncio.to_thread(executor.shutdown, wait=True, cancel_futures=True)

    async def _worker(self) -> None:
        assert self._queue is not None
        while True:
            job_id = await self._queue.get()
            try:
                if job_id is None:
                    return
                job = self._jobs.get(job_id)
                if job is None or job.status == "cancelled":
                    continue
                await self._execute_job(job)
            except asyncio.CancelledError:
                raise
            except Exception:
                _LOG.exception("Simulation worker failed.")
                if job_id is not None and (failed_job := self._jobs.get(job_id)):
                    failed_job.status = "failed"
                    failed_job.error = "Simulation worker failed."
                    failed_job.updated_at = time.monotonic()
            finally:
                self._queue.task_done()

    async def _execute_job(self, job: _SimulationJob) -> None:
        assert self._executor is not None
        loop = asyncio.get_running_loop()
        settings = _PROFILES[job.profile]
        iteration_limit = min(self.max_iterations, settings["max_iterations"])
        seed_source = random.Random(job.seed)
        totals: dict[str, Any] = {
            "samples": 0,
            "successes": 0,
            "catastrophes": 0,
            "failure_modes": Counter(),
            "cpu_seconds": 0.0,
            "peak_worker_rss_bytes": None,
        }
        deadline = loop.time() + self.timeout_seconds
        job.status = "running"
        job.updated_at = time.monotonic()
        self._active_jobs += 1
        timed_out = False
        memory_exceeded = False
        initial_samples = min(1_000, iteration_limit)
        try:
            while totals["samples"] < iteration_limit:
                if job.cancel_requested or self._closed:
                    job.status = "cancelled"
                    break
                remaining = iteration_limit - totals["samples"]
                batch_count = min(self.batch_size, remaining)
                batch_seed = seed_source.getrandbits(63)
                future = loop.run_in_executor(
                    self._executor,
                    _simulate_batch,
                    job.spec,
                    batch_seed,
                    batch_count,
                )
                time_left = deadline - loop.time()
                if time_left <= 0:
                    timed_out = True
                    break
                try:
                    batch = await asyncio.wait_for(future, timeout=time_left)
                except asyncio.TimeoutError:
                    timed_out = True
                    break
                _merge_batch(totals, batch)
                job.samples_completed = totals["samples"]
                job.updated_at = time.monotonic()
                self._total_samples += int(batch["samples"])
                self._total_cpu_seconds += float(batch["cpu_seconds"])
                if (
                    batch["rss_bytes"] is not None
                    and int(batch["rss_bytes"]) > self.max_memory_bytes
                ):
                    memory_exceeded = True
                    break
                if job.cancel_requested or self._closed:
                    job.status = "cancelled"
                    break
                partial = _build_result(
                    job.spec,
                    profile=job.profile,
                    seed=job.seed,
                    totals=totals,
                    target_width=settings["target_width"],
                    max_iterations=iteration_limit,
                )
                if (
                    totals["samples"] >= initial_samples
                    and partial["sampling_converged"]
                ):
                    break
            job.result = _build_result(
                job.spec,
                profile=job.profile,
                seed=job.seed,
                totals=totals,
                target_width=settings["target_width"],
                max_iterations=iteration_limit,
                timed_out=timed_out,
            )
            if timed_out:
                job.status = "timed_out"
            elif memory_exceeded:
                job.status = "failed"
                job.error = "Worker memory limit exceeded."
                job.result["warnings"].append(
                    "The worker memory limit was exceeded; returned statistics are partial."
                )
            elif job.status != "cancelled":
                if job.result is not None and self._on_completed is not None:
                    callback_event = {
                        "prediction_id": job.job_id,
                        "action": job.spec["action"],
                        "profile": job.profile,
                        "seed": job.seed,
                        "sample_size": job.result["sample_size"],
                        "model": copy.deepcopy(job.result["model"]),
                        "result": {
                            "success_probability": job.result["success_probability"],
                            "failure_probability": job.result["failure_probability"],
                            "catastrophic_risk_score": job.result[
                                "catastrophic_risk_score"
                            ],
                        },
                    }
                    try:
                        calibration = await asyncio.to_thread(
                            self._on_completed, callback_event
                        )
                        if isinstance(calibration, Mapping):
                            job.result["reflection_calibration"] = dict(calibration)
                    except Exception:
                        _LOG.exception(
                            "Could not record completed simulation in reflection ledger."
                        )
                job.status = "completed"
        finally:
            self._active_jobs = max(0, self._active_jobs - 1)
            job.updated_at = time.monotonic()

    def _prune_jobs(self) -> None:
        now = time.monotonic()
        expired = [
            job_id
            for job_id, job in self._jobs.items()
            if job.status in {"completed", "failed", "cancelled", "timed_out"}
            and now - job.updated_at > _JOB_RETENTION_SECONDS
        ]
        for job_id in expired:
            job = self._jobs.pop(job_id, None)
            if job and job.idempotency_key:
                self._idempotency.pop(job.idempotency_key, None)
        while len(self._jobs) >= _MAX_TRACKED_JOBS:
            terminal = [
                job for job in self._jobs.values()
                if job.status in {"completed", "failed", "cancelled", "timed_out"}
            ]
            if not terminal:
                raise SimulationCapacityError(
                    "Simulation job history is full; wait for retained jobs to expire."
                )
            oldest = min(terminal, key=lambda item: item.updated_at)
            self._jobs.pop(oldest.job_id, None)
            if oldest.idempotency_key:
                self._idempotency.pop(oldest.idempotency_key, None)
        if len(self._idempotency) > _MAX_TRACKED_JOBS:
            self._idempotency = {
                key: value
                for key, value in self._idempotency.items()
                if value[1] in self._jobs
            }


def manager_from_environment() -> SimulationJobManager:
    """Create the bounded default manager using validated environment limits."""
    try:
        max_workers = int(os.environ.get("FRIDAY_SIMULATION_MAX_WORKERS", "2"))
        max_queued = int(os.environ.get("FRIDAY_SIMULATION_MAX_QUEUED", "10"))
        timeout = float(os.environ.get("FRIDAY_SIMULATION_TIMEOUT_SECONDS", "10"))
        batch_size = int(os.environ.get("FRIDAY_SIMULATION_BATCH_SIZE", "250"))
        max_iterations = int(os.environ.get("FRIDAY_SIMULATION_MAX_ITERATIONS", "10000"))
        memory_limit_mb = int(os.environ.get("FRIDAY_SIMULATION_MAX_MEMORY_MB", "256"))
    except ValueError as exc:
        raise RuntimeError("Simulation resource limits must be numeric.") from exc
    if not 16 <= memory_limit_mb <= 2048:
        raise RuntimeError("FRIDAY_SIMULATION_MAX_MEMORY_MB must be between 16 and 2048.")
    return SimulationJobManager(
        max_concurrent_jobs=max_workers,
        max_queued_jobs=max_queued,
        timeout_seconds=timeout,
        batch_size=batch_size,
        max_iterations=max_iterations,
        max_memory_bytes=memory_limit_mb * 1024 * 1024,
    )
