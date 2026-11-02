"""Universal digital-twin core with explicit, validated domain adapters.

No adapter is enabled by default. A twin result is a preflight only and never
dispatches hardware or replaces authorization and confirmation controls.
"""
from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import re
import secrets
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

from app.access_control import assert_safe_write_target
from app.predictive_engine import (
    SimulationCapacityError,
    SimulationInputError,
    SimulationJobManager,
    SimulationServiceClosed,
)
from app.task_classifier import has_explicit_simulation_intent


_LOG = logging.getLogger("friday.digital_twin")
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_MAX_STATE_BYTES = 2 * 1024 * 1024
_DEVICE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
_DEFAULT_TELEMETRY_AGE_SECONDS = 30.0
_FUTURE_TELEMETRY_SKEW_SECONDS = 5.0


class DigitalTwinError(ValueError):
    """Twin configuration, state, or input is invalid."""


@runtime_checkable
class DigitalTwinAdapter(Protocol):
    """Trusted, versioned domain model implemented in application code."""

    domain: str
    model_version: str
    validated: bool

    def validate_device(
        self, specification: Mapping[str, Any], state: Mapping[str, Any]
    ) -> bool: ...

    def build_action_spec(
        self,
        specification: Mapping[str, Any],
        state: Mapping[str, Any],
        command: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...

    def allow_dispatch(self, result: Mapping[str, Any]) -> bool: ...

    def build_text_request(self, text: str) -> Mapping[str, Any] | None: ...


class DigitalTwinEngine:
    """Persistent device shadow and fail-closed simulation coordinator."""

    def __init__(
        self,
        *,
        predictive_service: SimulationJobManager | None = None,
        state_path: Path | None = None,
        adapters: tuple[DigitalTwinAdapter, ...] = (),
        telemetry_max_age_seconds: float = _DEFAULT_TELEMETRY_AGE_SECONDS,
    ) -> None:
        if (
            not isinstance(telemetry_max_age_seconds, (int, float))
            or isinstance(telemetry_max_age_seconds, bool)
            or not 1.0 <= float(telemetry_max_age_seconds) <= 3_600.0
        ):
            raise ValueError("telemetry_max_age_seconds must be between 1 and 3600.")
        state_root = Path(
            os.environ.get("FRIDAY_STATE_DIR") or _PROJECT_ROOT / "memory"
        ).expanduser()
        self.state_path = Path(state_path or state_root / "digital_twins.json")
        self.predictive_service = predictive_service
        self.telemetry_max_age_seconds = float(telemetry_max_age_seconds)
        self._lock = threading.RLock()
        self._adapters: dict[str, DigitalTwinAdapter] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread_id: int | None = None
        self._state = self._load_state()
        for adapter in adapters:
            self.register_adapter(adapter)

    def bind_loop(self, loop: asyncio.AbstractEventLoop | None) -> None:
        self._loop = loop
        self._loop_thread_id = threading.get_ident() if loop is not None else None

    def register_adapter(self, adapter: DigitalTwinAdapter) -> None:
        domain = getattr(adapter, "domain", None)
        model_version = getattr(adapter, "model_version", None)
        if (
            not isinstance(domain, str)
            or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", domain)
            or not isinstance(model_version, str)
            or not model_version.strip()
            or getattr(adapter, "validated", None) is not True
        ):
            raise DigitalTwinError(
                "Only code-reviewed, versioned, validated adapters may be registered."
            )
        required = (
            "validate_device",
            "build_action_spec",
            "allow_dispatch",
            "build_text_request",
        )
        if any(not callable(getattr(adapter, method, None)) for method in required):
            raise DigitalTwinError("Adapter is missing a required typed operation.")
        with self._lock:
            if domain in self._adapters:
                raise DigitalTwinError(f"Adapter domain {domain!r} is already registered.")
            self._adapters[domain] = adapter

    def register_device(
        self,
        device_id: str,
        domain: str,
        specification: Mapping[str, Any],
        state: Mapping[str, Any],
        *,
        telemetry_at: str,
        telemetry_source: str,
    ) -> None:
        if not isinstance(device_id, str) or not _DEVICE_ID_RE.fullmatch(device_id):
            raise DigitalTwinError("device_id must be a safe identifier of at most 80 characters.")
        if not isinstance(specification, Mapping) or not isinstance(state, Mapping):
            raise DigitalTwinError("Device specification and state must be objects.")
        if not isinstance(telemetry_source, str) or not telemetry_source.strip():
            raise DigitalTwinError("A verified telemetry source is required.")
        parsed_timestamp = self._parse_timestamp(telemetry_at)
        with self._lock:
            adapter = self._adapters.get(domain)
            if adapter is None:
                raise DigitalTwinError(
                    f"NOT_SUPPORTED: No validated adapter is registered for {domain!r}."
                )
            copied_spec = copy.deepcopy(dict(specification))
            copied_state = copy.deepcopy(dict(state))
            self._validate_json_size(copied_spec, "Device specification")
            self._validate_json_size(copied_state, "Device state")
            if adapter.validate_device(copied_spec, copied_state) is not True:
                raise DigitalTwinError("Device specification or initial state was rejected.")
            self._state["devices"][device_id] = {
                "domain": domain,
                "model_version": adapter.model_version,
                "specification": copied_spec,
                "state": copied_state,
                "telemetry_at": parsed_timestamp.isoformat(),
                "telemetry_source": telemetry_source.strip()[:128],
            }
            self._write_state()

    def synchronize_telemetry(
        self,
        device_id: str,
        state: Mapping[str, Any],
        *,
        telemetry_at: str,
        source_authenticated: bool,
    ) -> None:
        if source_authenticated is not True:
            raise DigitalTwinError("Unauthenticated telemetry is rejected.")
        if not isinstance(state, Mapping):
            raise DigitalTwinError("Telemetry state must be an object.")
        parsed_timestamp = self._parse_timestamp(telemetry_at)
        now = time.time()
        if parsed_timestamp.timestamp() > now + _FUTURE_TELEMETRY_SKEW_SECONDS:
            raise DigitalTwinError("Telemetry timestamp is in the future.")
        with self._lock:
            device = self._state["devices"].get(device_id)
            if device is None:
                raise DigitalTwinError("Telemetry references an unregistered device.")
            adapter = self._adapters.get(device["domain"])
            if adapter is None:
                raise DigitalTwinError("Telemetry cannot be validated without its adapter.")
            copied = copy.deepcopy(dict(state))
            self._validate_json_size(copied, "Telemetry state")
            if adapter.validate_device(device["specification"], copied) is not True:
                raise DigitalTwinError("Telemetry state failed adapter validation.")
            current_at = self._parse_timestamp(device["telemetry_at"]).timestamp()
            if parsed_timestamp.timestamp() < current_at:
                raise DigitalTwinError("Older telemetry cannot replace newer device state.")
            device["state"] = copied
            device["telemetry_at"] = parsed_timestamp.isoformat()
            self._write_state()

    def status(self) -> dict[str, Any]:
        now = time.time()
        result: dict[str, Any] = {
            "status": "ok",
            "devices": [],
            "supported_domains": sorted(self._adapters),
            "note": "No domain is supported without a registered validated adapter.",
        }
        with self._lock:
            for device_id, device in sorted(self._state["devices"].items()):
                adapter = self._adapters.get(device["domain"])
                telemetry_at = self._parse_timestamp(device["telemetry_at"])
                age = now - telemetry_at.timestamp()
                fresh = (
                    -_FUTURE_TELEMETRY_SKEW_SECONDS
                    <= age
                    <= self.telemetry_max_age_seconds
                )
                result["devices"].append({
                    "device_id": device_id,
                    "domain": device["domain"],
                    "model_version": device["model_version"],
                    "model_validated": bool(adapter and adapter.validated is True),
                    "telemetry_at": device["telemetry_at"],
                    "telemetry_age_seconds": max(0.0, age),
                    "telemetry_fresh": fresh,
                    "state": copy.deepcopy(device["state"]),
                })
        return result

    def device_configuration(self, device_id: str) -> dict[str, Any]:
        """Return a defensive copy of one registered device for trusted adapters."""
        if not isinstance(device_id, str) or not _DEVICE_ID_RE.fullmatch(device_id):
            raise DigitalTwinError("Invalid device_id.")
        with self._lock:
            device = self._state["devices"].get(device_id)
            if device is None:
                raise DigitalTwinError("Digital-twin device was not found.")
            return copy.deepcopy(device)

    async def simulate_command(
        self,
        device_id: str,
        command: Mapping[str, Any],
        *,
        profile: str = "STANDARD",
        seed: int | None = None,
    ) -> dict[str, Any]:
        try:
            snapshot, adapter, telemetry_at = self._validated_snapshot(device_id)
        except DigitalTwinError as exc:
            status = (
                "STALE_TELEMETRY"
                if str(exc).startswith("STALE_TELEMETRY:")
                else "NOT_SUPPORTED"
            )
            return {
                "status": status,
                "dispatch_allowed": False,
                "detail": str(exc),
                "device_id": device_id if isinstance(device_id, str) else None,
            }
        if isinstance(command, Mapping) is False:
            raise DigitalTwinError("A device command must be an object.")
        copied_command = copy.deepcopy(dict(command))
        self._validate_json_size(copied_command, "Device command")
        try:
            spec = adapter.build_action_spec(
                snapshot["specification"], snapshot["state"], copied_command
            )
        except Exception as exc:
            _LOG.exception("Twin adapter could not construct a simulation model.")
            return {
                "status": "NOT_SUPPORTED",
                "dispatch_allowed": False,
                "detail": f"Adapter could not model this command: {type(exc).__name__}.",
            }
        if not isinstance(spec, Mapping):
            return {
                "status": "NOT_SUPPORTED",
                "dispatch_allowed": False,
                "detail": "Adapter did not produce a structured, data-only simulation model.",
            }
        if self.predictive_service is None:
            return {
                "status": "UNAVAILABLE",
                "dispatch_allowed": False,
                "detail": "The bounded predictive service is not running.",
            }
        try:
            job = await self.predictive_service.submit(
                spec, profile=profile, seed=seed
            )
        except SimulationCapacityError as exc:
            return {
                "status": "BUSY",
                "dispatch_allowed": False,
                "detail": str(exc),
            }
        except (SimulationInputError, SimulationServiceClosed) as exc:
            return {
                "status": "NOT_SUPPORTED",
                "dispatch_allowed": False,
                "detail": str(exc),
            }
        job.update({
            "status": "QUEUED",
            "dispatch_allowed": False,
            "device_id": device_id,
            "domain": snapshot["domain"],
            "model_version": adapter.model_version,
            "telemetry_at": telemetry_at,
            "dispatch_note": "Simulation never authorizes or dispatches hardware.",
        })
        return job

    async def simulate_explicit_request(self, text: str) -> dict[str, Any]:
        if not has_explicit_simulation_intent(text):
            return {
                "status": "NOT_REQUESTED",
                "dispatch_allowed": False,
            }
        if len(text) > 16_000:
            raise DigitalTwinError("Simulation request is too large.")
        adapters = tuple(self._adapters.values())
        for adapter in adapters:
            try:
                request = adapter.build_text_request(text)
            except Exception:
                _LOG.exception("Twin adapter rejected a text simulation request.")
                continue
            if request is None:
                continue
            if not isinstance(request, Mapping):
                _LOG.error("Twin adapter returned a malformed natural-language request.")
                continue
            device_id = request.get("device_id")
            command = request.get("command")
            if not isinstance(device_id, str) or not isinstance(command, Mapping):
                _LOG.error("Twin adapter returned an incomplete natural-language request.")
                continue
            submission = await self.simulate_command(
                device_id,
                command,
                profile=str(request.get("profile") or "STANDARD"),
            )
            if submission.get("status") != "QUEUED":
                return submission
            job_id = submission.get("job_id")
            if not isinstance(job_id, str) or self.predictive_service is None:
                return {
                    "status": "FAILED",
                    "dispatch_allowed": False,
                    "detail": "The bounded simulation job could not be retrieved.",
                }
            deadline = asyncio.get_running_loop().time() + (
                self.predictive_service.timeout_seconds + 1.0
            )
            while asyncio.get_running_loop().time() < deadline:
                job = await self.predictive_service.get_job(job_id)
                if job is None:
                    return {
                        "status": "FAILED",
                        "dispatch_allowed": False,
                        "detail": "The simulation result expired or was unavailable.",
                    }
                if job["status"] in {
                    "completed", "failed", "cancelled", "timed_out"
                }:
                    return {
                        **job,
                        "status": job["status"].upper(),
                        "dispatch_allowed": False,
                        "device_id": device_id,
                        "model_version": submission.get("model_version"),
                        "dispatch_note": (
                            "Simulation is an estimate only and never authorizes or "
                            "dispatches physical hardware."
                        ),
                    }
                await asyncio.sleep(0.025)
            await self.predictive_service.cancel(job_id)
            return {
                "status": "TIMED_OUT",
                "dispatch_allowed": False,
                "detail": "Simulation exceeded its bounded wait and was cancelled.",
            }
        return {
            "status": "NOT_SUPPORTED",
            "dispatch_allowed": False,
            "detail": (
                "No validated domain adapter can model this request. "
                "No simulation result was generated."
            ),
        }

    async def assess_dispatch(
        self, device_id: str, command: Mapping[str, Any]
    ) -> dict[str, Any]:
        try:
            snapshot, adapter, expected_telemetry = self._validated_snapshot(device_id)
        except DigitalTwinError as exc:
            return {
                "status": (
                    "STALE_TELEMETRY"
                    if str(exc).startswith("STALE_TELEMETRY:")
                    else "NOT_SUPPORTED"
                ),
                "dispatch_allowed": False,
                "detail": str(exc),
            }
        expected_state_hash = self._state_fingerprint(snapshot["state"])
        submission = await self.simulate_command(
            device_id, command, profile="DEEP"
        )
        if submission.get("status") != "QUEUED":
            return submission
        if self.predictive_service is None:
            return {
                "status": "UNAVAILABLE",
                "dispatch_allowed": False,
                "detail": "The bounded predictive service is not running.",
            }
        job_id = submission.get("job_id")
        deadline = asyncio.get_running_loop().time() + (
            self.predictive_service.timeout_seconds + 1.0
        )
        while asyncio.get_running_loop().time() < deadline:
            if not self._telemetry_snapshot_still_valid(
                device_id, expected_telemetry, expected_state_hash
            ):
                await self.predictive_service.cancel(str(job_id))
                return {
                    "status": "STALE_STATE",
                    "dispatch_allowed": False,
                    "detail": "Twin state changed during simulation; command rejected.",
                }
            job = await self.predictive_service.get_job(str(job_id))
            if job is None:
                return {
                    "status": "FAILED",
                    "dispatch_allowed": False,
                    "detail": "Simulation result expired or was unavailable.",
                }
            if job["status"] in {"completed", "failed", "cancelled", "timed_out"}:
                result = job.get("result")
                allowed = (
                    job["status"] == "completed"
                    and isinstance(result, Mapping)
                    and result.get("recommendation") == "RECOMMENDED"
                    and adapter.allow_dispatch(result) is True
                    and self._telemetry_snapshot_still_valid(
                        device_id, expected_telemetry, expected_state_hash
                    )
                )
                return {
                    "status": "SIMULATED" if allowed else "REJECTED",
                    "dispatch_allowed": allowed,
                    "simulation": result,
                    "detail": (
                        "A passing simulation is only a preflight; existing security "
                        "and confirmation checks remain mandatory."
                        if allowed
                        else "Simulation did not satisfy the adapter's dispatch policy."
                    ),
                }
            await asyncio.sleep(0.025)
        await self.predictive_service.cancel(str(job_id))
        return {
            "status": "TIMED_OUT",
            "dispatch_allowed": False,
            "detail": "Twin preflight exceeded its bounded wait and was cancelled.",
        }

    def preflight_mqtt_command(
        self, device_id: str, command: Mapping[str, Any]
    ) -> dict[str, Any]:
        loop = self._loop
        if (
            loop is None
            or loop.is_closed()
            or threading.get_ident() == self._loop_thread_id
        ):
            return {
                "status": "UNAVAILABLE",
                "dispatch_allowed": False,
                "detail": "Synchronous MQTT preflight requires the managed worker thread.",
            }
        timeout = (
            self.predictive_service.timeout_seconds + 2.0
            if self.predictive_service is not None
            else 2.0
        )
        future = asyncio.run_coroutine_threadsafe(
            self.assess_dispatch(device_id, command), loop
        )
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            future.cancel()
            return {
                "status": "TIMED_OUT",
                "dispatch_allowed": False,
                "detail": "Twin preflight timed out; MQTT dispatch was rejected.",
            }
        except Exception as exc:
            future.cancel()
            _LOG.exception("Twin preflight failed; rejecting physical actuation.")
            return {
                "status": "FAILED",
                "dispatch_allowed": False,
                "detail": f"Twin preflight failed ({type(exc).__name__}); actuation rejected.",
            }

    def _validated_snapshot(
        self, device_id: str
    ) -> tuple[dict[str, Any], DigitalTwinAdapter, str]:
        if not isinstance(device_id, str) or not _DEVICE_ID_RE.fullmatch(device_id):
            raise DigitalTwinError("Invalid device_id.")
        with self._lock:
            device = self._state["devices"].get(device_id)
            if device is None:
                raise DigitalTwinError(
                    "NOT_SUPPORTED: No configured digital twin exists for this device."
                )
            adapter = self._adapters.get(device["domain"])
            if (
                adapter is None
                or adapter.validated is not True
                or adapter.model_version != device["model_version"]
            ):
                raise DigitalTwinError(
                    "NOT_SUPPORTED: No matching validated domain adapter is available."
                )
            telemetry_at = str(device["telemetry_at"])
            age = time.time() - self._parse_timestamp(telemetry_at).timestamp()
            if not -_FUTURE_TELEMETRY_SKEW_SECONDS <= age <= self.telemetry_max_age_seconds:
                raise DigitalTwinError(
                    "STALE_TELEMETRY: Twin state is missing, future-dated, or stale."
                )
            snapshot = copy.deepcopy(device)
        return snapshot, adapter, telemetry_at

    def _telemetry_snapshot_still_valid(
        self,
        device_id: str,
        expected_telemetry: str,
        expected_state_hash: str,
    ) -> bool:
        try:
            snapshot, _, telemetry_at = self._validated_snapshot(device_id)
        except DigitalTwinError:
            return False
        return (
            telemetry_at == expected_telemetry
            and snapshot["model_version"]
            == self._adapters[snapshot["domain"]].model_version
            and self._state_fingerprint(snapshot["state"]) == expected_state_hash
        )

    @staticmethod
    def _state_fingerprint(state: Mapping[str, Any]) -> str:
        return json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False)

    def _load_state(self) -> dict[str, Any]:
        try:
            if not self.state_path.is_file():
                return {"version": 1, "devices": {}}
            if self.state_path.stat().st_size > _MAX_STATE_BYTES:
                raise DigitalTwinError("Digital twin state file exceeds its size limit.")
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {"version": 1, "devices": {}}
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DigitalTwinError("Digital twin state is unreadable; refusing unsafe defaults.") from exc
        if (
            not isinstance(data, dict)
            or data.get("version") != 1
            or not isinstance(data.get("devices"), dict)
        ):
            raise DigitalTwinError("Digital twin state has an unsupported schema.")
        return data

    def _write_state(self) -> None:
        encoded = json.dumps(
            self._state,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        if len(encoded.encode("utf-8")) > _MAX_STATE_BYTES:
            raise DigitalTwinError("Digital twin state exceeds its storage limit.")
        target = self.state_path.expanduser().resolve(strict=False)
        assert_safe_write_target(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(
            target.name + "." + secrets.token_hex(8) + ".tmp"
        )
        assert_safe_write_target(temporary)
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _validate_json_size(value: Mapping[str, Any], label: str) -> None:
        try:
            encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise DigitalTwinError(f"{label} must contain finite JSON values.") from exc
        if len(encoded.encode("utf-8")) > 32_768:
            raise DigitalTwinError(f"{label} exceeds 32 KiB.")

    @staticmethod
    def _parse_timestamp(value: str) -> datetime:
        if not isinstance(value, str) or len(value) > 64:
            raise DigitalTwinError("Telemetry timestamp must be an ISO 8601 string.")
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise DigitalTwinError("Telemetry timestamp is invalid.") from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise DigitalTwinError("Telemetry timestamp must include a timezone.")
        return parsed.astimezone(timezone.utc)


_DEFAULT_ENGINE_LOCK = threading.RLock()
_DEFAULT_ENGINE: DigitalTwinEngine | None = None


def bind_default_engine(engine: DigitalTwinEngine | None) -> None:
    global _DEFAULT_ENGINE
    with _DEFAULT_ENGINE_LOCK:
        _DEFAULT_ENGINE = engine


def get_default_engine() -> DigitalTwinEngine | None:
    with _DEFAULT_ENGINE_LOCK:
        return _DEFAULT_ENGINE
