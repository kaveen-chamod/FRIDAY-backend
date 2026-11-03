"""Auditable prediction/outcome ledger with conservative calibration updates."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from memory.memory_manager import redact_sensitive_content


_LOG = logging.getLogger("friday.reflection")
_SCHEMA_VERSION = 1
_MAX_LEDGER_BYTES = 16 * 1024 * 1024
_MAX_RECORD_BYTES = 4_096
_MAX_RECORDS = _MAX_LEDGER_BYTES // 128
_MIN_VERIFIED_EXAMPLES = 20
_MAX_CALIBRATION_BIAS = 0.20
_MAX_CALIBRATION_STEP = 0.05


class ReflectionStorageError(OSError):
    """The reflection ledger cannot be read or safely extended."""


class ReflectionInputError(ValueError):
    """A reflection record is invalid or references an unknown prediction."""


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _probability(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReflectionInputError(f"{label} must be a probability.")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ReflectionInputError(f"{label} must be between 0 and 1.")
    return number


def _safe_text(value: Any, limit: int) -> str:
    return redact_sensitive_content(str(value or ""))[:limit]


class SelfReflectionLedger:
    """Append-only ledger; only internal independently verified data calibrates."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        outcome_verifier: Callable[[str, bool, str], bool] | None = None,
    ) -> None:
        state_root = Path(
            os.environ.get("FRIDAY_STATE_DIR")
            or Path(__file__).resolve().parents[1] / "memory"
        ).expanduser().resolve()
        self.path = (
            Path(path).expanduser().resolve(strict=False)
            if path is not None
            else state_root / "reflection_ledger.jsonl"
        )
        self._lock = threading.RLock()
        self._outcome_verifier = outcome_verifier
        self._loaded = False
        self._records: list[dict[str, Any]] = []
        self._predictions: dict[str, dict[str, Any]] = {}
        self._calibration_history: dict[int, float] = {1: 0.0}
        self._calibration_version = 1
        self._calibration_bias = 0.0

    def record_simulation_prediction(self, event: Mapping[str, Any]) -> dict[str, Any]:
        """Record a hypothetical result and return advisory-only calibrated metrics."""
        if not isinstance(event, Mapping):
            raise ReflectionInputError("Simulation completion must be an object.")
        prediction_id = event.get("prediction_id")
        if not isinstance(prediction_id, str) or not 1 <= len(prediction_id) <= 128:
            raise ReflectionInputError("prediction_id must contain 1 to 128 characters.")
        result = event.get("result")
        if not isinstance(result, Mapping):
            raise ReflectionInputError("A completed simulation result is required.")
        predicted_success = _probability(
            result.get("success_probability"), "success_probability"
        )
        predicted_failure = _probability(
            result.get("failure_probability"), "failure_probability"
        )
        predicted_risk = _probability(
            result.get("catastrophic_risk_score"), "catastrophic_risk_score"
        )
        sample_size = event.get("sample_size")
        if isinstance(sample_size, bool) or not isinstance(sample_size, int) or sample_size < 0:
            raise ReflectionInputError("sample_size must be a non-negative integer.")

        model = event.get("model")
        model = model if isinstance(model, Mapping) else {}
        features = {
            "profile": _safe_text(event.get("profile"), 16),
            "sample_size": sample_size,
            "model_source": _safe_text(model.get("source"), 100),
            "model_version": _safe_text(model.get("version"), 64),
            "seed": (
                event.get("seed")
                if isinstance(event.get("seed"), int)
                and not isinstance(event.get("seed"), bool)
                else None
            ),
        }
        record = {
            "schema_version": _SCHEMA_VERSION,
            "event_type": "prediction",
            "prediction_id": prediction_id,
            "timestamp": _timestamp(),
            "action": _safe_text(event.get("action"), 256),
            "source": "hypothetical_simulation",
            "verification_status": "not_real_world_evidence",
            "predicted_probabilities": {
                "success": predicted_success,
                "failure": predicted_failure,
                "catastrophic_risk": predicted_risk,
            },
            "features": features,
        }
        with self._lock:
            self._ensure_loaded()
            existing = self._predictions.get(prediction_id)
            if existing is None:
                self._append_locked(record)
            elif (
                existing["predicted_probabilities"] != record["predicted_probabilities"]
                or existing.get("action") != record["action"]
            ):
                raise ReflectionInputError(
                    "prediction_id was reused for a different result."
                )
            return {
                "version": self._calibration_version,
                "bias": self._calibration_bias,
                "calibrated_success_probability": self._clamp(
                    predicted_success + self._calibration_bias
                ),
                "basis": "supplemental calibration from independently verified outcomes only",
                "does_not_change_recommendation": True,
            }

    def record_user_feedback(self, prediction_id: str, actual_success: bool) -> dict[str, Any]:
        if not isinstance(actual_success, bool):
            raise ReflectionInputError("actual_success must be a boolean.")
        return self._record_outcome(
            prediction_id,
            float(actual_success),
            source="user_feedback",
            verification_status="unverified",
        )

    def record_sensor_observation(
        self, prediction_id: str, actual_success: bool, observation_ref: str
    ) -> dict[str, Any]:
        if not isinstance(actual_success, bool):
            raise ReflectionInputError("actual_success must be a boolean.")
        if (
            not isinstance(observation_ref, str)
            or not observation_ref.strip()
            or len(observation_ref) > 256
        ):
            raise ReflectionInputError("A sensor observation reference is required.")
        return self._record_outcome(
            prediction_id,
            float(actual_success),
            source="sensor_observation",
            verification_status="observed_unverified",
            evidence_ref=observation_ref,
        )

    def _record_verified_outcome(
        self, prediction_id: str, actual_success: bool, verification_ref: str
    ) -> dict[str, Any]:
        """Internal integration point; intentionally not exposed by an API route."""
        if not isinstance(actual_success, bool):
            raise ReflectionInputError("actual_success must be a boolean.")
        if not isinstance(verification_ref, str) or not verification_ref.strip():
            raise ReflectionInputError("Independent verification evidence is required.")
        if len(verification_ref) > 256:
            raise ReflectionInputError("Independent verification reference is too long.")
        if self._outcome_verifier is None:
            raise ReflectionInputError(
                "No trusted independent-outcome verifier is configured."
            )
        try:
            verified = self._outcome_verifier(
                prediction_id, actual_success, verification_ref
            )
        except Exception as exc:
            _LOG.exception("Independent outcome verification failed.")
            raise ReflectionInputError(
                "Independent outcome verification is unavailable."
            ) from exc
        if verified is not True:
            raise ReflectionInputError(
                "The supplied outcome was not independently verified."
            )
        return self._record_outcome(
            prediction_id,
            float(actual_success),
            source="independently_verified",
            verification_status="verified",
            evidence_ref=verification_ref,
            calibrate=True,
        )

    def _record_outcome(
        self,
        prediction_id: str,
        actual: float,
        *,
        source: str,
        verification_status: str,
        evidence_ref: str | None = None,
        calibrate: bool = False,
    ) -> dict[str, Any]:
        if not isinstance(prediction_id, str) or not 1 <= len(prediction_id) <= 128:
            raise ReflectionInputError("prediction_id must contain 1 to 128 characters.")
        with self._lock:
            self._ensure_loaded()
            prediction = self._predictions.get(prediction_id)
            if prediction is None:
                raise ReflectionInputError("Outcome must reference a recorded prediction.")
            existing = next(
                (
                    row for row in self._records
                    if row.get("event_type") == "outcome"
                    and row.get("prediction_id") == prediction_id
                    and row.get("source") == source
                ),
                None,
            )
            if existing is not None:
                if existing["actual_outcome"] != actual:
                    raise ReflectionInputError(
                        "A different outcome from this source is already recorded."
                    )
                return {"status": "duplicate", "calibration_update": None}

            probability = prediction["predicted_probabilities"]["success"]
            outcome = {
                "schema_version": _SCHEMA_VERSION,
                "event_type": "outcome",
                "outcome_id": uuid.uuid4().hex,
                "prediction_id": prediction_id,
                "timestamp": _timestamp(),
                "source": source,
                "verification_status": verification_status,
                "actual_outcome": actual,
                "absolute_error": abs(probability - actual),
                "evidence_ref_hash": (
                    hashlib.sha256(evidence_ref.encode("utf-8")).hexdigest()
                    if evidence_ref is not None else None
                ),
            }
            self._append_locked(outcome)
            update = self._update_calibration_locked() if calibrate else None
            return {
                "status": "recorded",
                "source": source,
                "verification_status": verification_status,
                "absolute_error": abs(probability - actual),
                "calibration_update": update,
            }

    def report(self) -> dict[str, Any]:
        with self._lock:
            self._ensure_loaded()
            outcomes = [row for row in self._records if row.get("event_type") == "outcome"]
            counts = {
                source: sum(row.get("source") == source for row in outcomes)
                for source in (
                    "user_feedback",
                    "sensor_observation",
                    "independently_verified",
                )
            }
            verified = [
                row for row in outcomes
                if row.get("source") == "independently_verified"
                and row.get("verification_status") == "verified"
            ]
            brier = self._brier_score(verified)
            drift = (
                sum(
                    float(row["actual_outcome"])
                    - self._predictions[row["prediction_id"]]["predicted_probabilities"]["success"]
                    for row in verified
                ) / len(verified)
                if verified else None
            )
            return {
                "prediction_count": len(self._predictions),
                "outcome_counts_by_source": counts,
                "independently_verified_count": len(verified),
                "brier_score": brier,
                "calibration_drift": drift,
                "calibration": {
                    "version": self._calibration_version,
                    "success_probability_bias": self._calibration_bias,
                    "history": [
                        {"version": version, "bias": bias}
                        for version, bias in sorted(self._calibration_history.items())
                    ],
                    "applies_to": "supplemental probability estimate only",
                    "changes_authorization_or_safety_limits": False,
                },
                "learned_heuristics": [],
                "learning_policy": (
                    "Only independently verified outcomes are eligible for calibration; "
                    "simulations and unverified feedback are not ground truth."
                ),
            }

    def _update_calibration_locked(self) -> dict[str, Any] | None:
        verified = [
            row for row in self._records
            if row.get("event_type") == "outcome"
            and row.get("source") == "independently_verified"
            and row.get("verification_status") == "verified"
            and row.get("prediction_id") in self._predictions
        ]
        if len(verified) < _MIN_VERIFIED_EXAMPLES:
            return None
        training: list[dict[str, Any]] = []
        holdout: list[dict[str, Any]] = []
        for row in verified:
            bucket = int(
                hashlib.sha256(row["prediction_id"].encode("utf-8")).hexdigest()[:8],
                16,
            ) % 5
            (holdout if bucket == 0 else training).append(row)
        if len(training) < 10 or len(holdout) < 4:
            return None

        residuals = [
            float(row["actual_outcome"])
            - self._predictions[row["prediction_id"]]["predicted_probabilities"]["success"]
            for row in training
        ]
        proposed_step = max(
            -_MAX_CALIBRATION_STEP,
            min(_MAX_CALIBRATION_STEP, sum(residuals) / len(residuals)),
        )
        candidate = max(
            -_MAX_CALIBRATION_BIAS,
            min(_MAX_CALIBRATION_BIAS, self._calibration_bias + proposed_step),
        )
        if math.isclose(candidate, self._calibration_bias, abs_tol=1e-12):
            return None
        current_brier = self._brier_score(holdout, bias=self._calibration_bias)
        candidate_brier = self._brier_score(holdout, bias=candidate)
        if current_brier is None or candidate_brier is None or candidate_brier > current_brier:
            return {
                "status": "rejected",
                "reason": "holdout performance did not improve or remain equal",
                "current_holdout_brier": current_brier,
                "candidate_holdout_brier": candidate_brier,
            }

        new_version = self._calibration_version + 1
        self._append_locked({
            "schema_version": _SCHEMA_VERSION,
            "event_type": "calibration_update",
            "timestamp": _timestamp(),
            "version": new_version,
            "previous_version": self._calibration_version,
            "bias": candidate,
            "training_count": len(training),
            "holdout_count": len(holdout),
            "previous_holdout_brier": current_brier,
            "candidate_holdout_brier": candidate_brier,
        })
        self._calibration_version = new_version
        self._calibration_bias = candidate
        self._calibration_history[new_version] = candidate
        _LOG.info("Activated validated reflection calibration version %d.", new_version)
        return {
            "status": "activated",
            "version": new_version,
            "bias": candidate,
            "previous_holdout_brier": current_brier,
            "candidate_holdout_brier": candidate_brier,
        }

    def _rollback_calibration(self, target_version: int) -> dict[str, Any]:
        if isinstance(target_version, bool) or not isinstance(target_version, int):
            raise ReflectionInputError("target_version must be an integer.")
        with self._lock:
            self._ensure_loaded()
            target_bias = self._calibration_history.get(target_version)
            if target_bias is None or target_version >= self._calibration_version:
                raise ReflectionInputError("A prior calibration version is required.")
            new_version = self._calibration_version + 1
            self._append_locked({
                "schema_version": _SCHEMA_VERSION,
                "event_type": "calibration_rollback",
                "timestamp": _timestamp(),
                "version": new_version,
                "rollback_to": target_version,
                "bias": target_bias,
            })
            self._calibration_version = new_version
            self._calibration_bias = target_bias
            self._calibration_history[new_version] = target_bias
            return {"status": "rolled_back", "version": new_version, "bias": target_bias}

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        try:
            size = self.path.stat().st_size
        except FileNotFoundError:
            self._loaded = True
            return
        except OSError as exc:
            raise ReflectionStorageError("Reflection ledger could not be inspected.") from exc
        if size > _MAX_LEDGER_BYTES:
            raise ReflectionStorageError("Reflection ledger exceeds its configured size limit.")
        try:
            content = self.path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise ReflectionStorageError("Reflection ledger could not be read.") from exc
        records: list[dict[str, Any]] = []
        for line in content.splitlines():
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ReflectionStorageError("Reflection ledger contains an invalid record.") from exc
            if (
                not isinstance(record, dict)
                or record.get("schema_version") != _SCHEMA_VERSION
            ):
                raise ReflectionStorageError("Reflection ledger contains an unsupported record.")
            records.append(record)
        if len(records) > _MAX_RECORDS:
            raise ReflectionStorageError("Reflection ledger contains too many records.")
        self._records = records
        for record in records:
            if record.get("event_type") == "prediction":
                prediction_id = record.get("prediction_id")
                probabilities = record.get("predicted_probabilities")
                if not isinstance(prediction_id, str) or not isinstance(probabilities, dict):
                    raise ReflectionStorageError("Reflection ledger prediction is malformed.")
                self._predictions[prediction_id] = record
            elif record.get("event_type") in {"calibration_update", "calibration_rollback"}:
                version = record.get("version")
                bias = record.get("bias")
                if (
                    isinstance(version, int)
                    and not isinstance(version, bool)
                    and isinstance(bias, (int, float))
                    and math.isfinite(float(bias))
                    and abs(float(bias)) <= _MAX_CALIBRATION_BIAS
                ):
                    self._calibration_version = version
                    self._calibration_bias = float(bias)
                    self._calibration_history[version] = float(bias)
        self._loaded = True

    def _append_locked(self, record: dict[str, Any]) -> None:
        if len(self._records) >= _MAX_RECORDS:
            raise ReflectionStorageError("Reflection record count exceeded its configured bound.")
        encoded = (
            json.dumps(record, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
        if len(encoded) > _MAX_RECORD_BYTES:
            raise ReflectionInputError("Reflection record exceeds its size limit.")
        try:
            current_size = self.path.stat().st_size
        except FileNotFoundError:
            current_size = 0
        except OSError as exc:
            raise ReflectionStorageError("Reflection ledger size could not be checked.") from exc
        if current_size + len(encoded) > _MAX_LEDGER_BYTES:
            raise ReflectionStorageError("Reflection ledger is full; no record was written.")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
            try:
                written = os.write(descriptor, encoded)
                if written != len(encoded):
                    raise ReflectionStorageError("Reflection ledger append was incomplete.")
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except ReflectionStorageError:
            raise
        except OSError as exc:
            raise ReflectionStorageError("Reflection ledger append failed.") from exc
        self._records.append(record)
        if record.get("event_type") == "prediction":
            self._predictions[record["prediction_id"]] = record

    def _brier_score(
        self, outcomes: list[dict[str, Any]], *, bias: float = 0.0
    ) -> float | None:
        if not outcomes:
            return None
        total = 0.0
        valid = 0
        for row in outcomes:
            prediction = row.get("prediction_id")
            actual = row.get("actual_outcome")
            if not isinstance(actual, (int, float)) or isinstance(actual, bool):
                continue
            prediction_record = self._predictions.get(prediction)
            if prediction_record is None:
                continue
            predicted_success = prediction_record["predicted_probabilities"]["success"]
            total += (
                self._clamp(float(predicted_success) + bias) - float(actual)
            ) ** 2
            valid += 1
        return total / valid if valid else None

    @classmethod
    def _clamp(cls, value: float) -> float:
        return max(0.0, min(1.0, value))
