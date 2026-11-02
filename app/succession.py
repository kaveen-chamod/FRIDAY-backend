from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
import tempfile
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from app.access_control import assert_safe_write_target
from core import security


_LOG = logging.getLogger("friday.succession")
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_ITERATIONS = 300_000
_GRANTABLE_ROLES = {"BROTHER", "SISTER"}
_GUARDIAN_ROLES = {"FATHER", "MOTHER"}
_DEFAULT_IDLE_HOURS = 72
_DEFAULT_SUCCESSION_GRACE_HOURS = 24


def _positive_int_env(name: str, default: int, maximum: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a positive integer.") from exc
    if not 1 <= parsed <= maximum:
        raise RuntimeError(f"{name} must be between 1 and {maximum}.")
    return parsed


def _new_state() -> dict[str, Any]:
    return {
        "version": 1,
        "phase": "ACTIVE",
        "activity": {"MASTER": None, "BROTHER": None, "SISTER": None},
        "emergency_credentials": {},
        "claim": None,
        "temporary_master_profile_id": None,
        "manual_emergency": False,
    }


class SuccessionManager:
    """Persistent, fail-closed succession state and temporary role grants."""

    def __init__(
        self,
        path: Path | None = None,
        *,
        inactivity_hours: int | None = None,
        grace_hours: int | None = None,
    ) -> None:
        root = Path(
            os.environ.get("FRIDAY_STATE_DIR") or _PROJECT_ROOT / "memory"
        )
        self.path = Path(path or root / "succession_state.json")
        self.inactivity_seconds = (
            inactivity_hours * 3600 if inactivity_hours is not None
            else _positive_int_env(
                "FRIDAY_MASTER_INACTIVITY_HOURS", _DEFAULT_IDLE_HOURS, 24 * 365
            ) * 3600
        )
        self.grace_seconds = (
            grace_hours * 3600 if grace_hours is not None
            else _positive_int_env(
                "FRIDAY_SUCCESSION_GRACE_HOURS",
                _DEFAULT_SUCCESSION_GRACE_HOURS,
                24 * 30,
            ) * 3600
        )
        if self.inactivity_seconds <= 0 or self.grace_seconds <= 0:
            raise ValueError("Succession inactivity and grace periods must be positive.")
        self._lock = threading.RLock()
        self._attempt_lock = threading.Lock()
        self._emergency_attempts: dict[str, deque[float]] = defaultdict(deque)
        self._emergency_locked_until: dict[str, float] = {}

    def heartbeat(self, role: str, profile_id: str, *, now: float | None = None) -> None:
        role = role.upper()
        if role not in {"MASTER", "FATHER", "MOTHER", "BROTHER", "SISTER"}:
            raise ValueError("Only verified family members can send succession heartbeats.")
        if not profile_id or len(profile_id) > 128:
            raise ValueError("Invalid voice profile identifier.")
        with self._lock:
            state = self._read()
            if role in state["activity"]:
                state["activity"][role] = {
                    "profile_id": profile_id,
                    "at": float(time.time() if now is None else now),
                }
            self._audit("heartbeat", "recorded", f"role={role}")
            self._write(state)

    def enroll_emergency_credential(
        self, role: str, profile_id: str, passphrase: str
    ) -> None:
        role = role.upper()
        if role not in _GRANTABLE_ROLES:
            raise ValueError("Emergency passphrases may only be enrolled for Brother or Sister.")
        if not profile_id or len(profile_id) > 128:
            raise ValueError("Invalid voice profile identifier.")
        if not isinstance(passphrase, str) or len(passphrase) < 16 or len(passphrase) > 1024:
            raise ValueError("Emergency Passphrase must contain 16 to 1024 characters.")
        salt = secrets.token_bytes(16)
        digest = hashlib.pbkdf2_hmac(
            "sha256", passphrase.encode("utf-8"), salt, _ITERATIONS
        )
        with self._lock:
            state = self._read()
            state["emergency_credentials"][profile_id] = {
                "role": role,
                "salt": salt.hex(),
                "digest": digest.hex(),
                "iterations": _ITERATIONS,
            }
            self._audit("emergency_credential_enrollment", "configured", f"role={role}")
            self._write(state)

    def mark_manual_emergency(self, master_profile_id: str) -> None:
        if not master_profile_id:
            raise ValueError("A verified Master profile is required.")
        with self._lock:
            state = self._read()
            state["manual_emergency"] = True
            state["manual_emergency_by"] = master_profile_id
            state["manual_emergency_at"] = time.time()
            self._audit("manual_emergency", "authorized", "Master-triggered emergency.")
            self._write(state)

    def claim(
        self,
        role: str,
        profile_id: str,
        emergency_passphrase: str,
        *,
        manual_emergency: bool = False,
        master_profile_id: str | None = None,
        now: float | None = None,
    ) -> dict[str, str]:
        role = role.upper()
        timestamp = float(time.time() if now is None else now)
        if role not in _GRANTABLE_ROLES or not profile_id:
            raise ValueError("Only a verified Brother or Sister may claim succession.")
        with self._lock:
            state = self._read()
            if manual_emergency:
                if not master_profile_id:
                    raise PermissionError("Manual emergency trigger requires verified Master authorization.")
                state["manual_emergency"] = True
                state["manual_emergency_by"] = master_profile_id
                state["manual_emergency_at"] = timestamp
            eligibility = self._eligibility(state, timestamp)
            if role == "BROTHER" and not eligibility["brother_eligible"]:
                raise PermissionError("Brother succession is not currently eligible.")
            if role == "SISTER" and not eligibility["sister_eligible"]:
                raise PermissionError(
                    "Sister fallback requires confirmed Master and Brother inactivity."
                )
            credential = state["emergency_credentials"].get(profile_id)
            if not credential or credential["role"] != role:
                raise PermissionError("No matching emergency credential is enrolled.")
            if self._emergency_is_locked(profile_id):
                raise PermissionError("Emergency Passphrase verification is temporarily rate-limited.")
            if not self._verify_emergency_credential(credential, emergency_passphrase):
                self._record_emergency_failure(profile_id)
                self._audit("succession_claim", "denied", f"role={role}; invalid emergency passphrase")
                raise PermissionError("Emergency Passphrase verification failed.")
            with self._attempt_lock:
                self._emergency_attempts.pop(profile_id, None)
                self._emergency_locked_until.pop(profile_id, None)
            claim_id = secrets.token_urlsafe(24)
            state["claim"] = {
                "claim_id": claim_id,
                "profile_id": profile_id,
                "role": role,
                "created_at": timestamp,
                "guardian_approval": None,
            }
            state["phase"] = f"AWAITING_{role}_GUARDIAN_APPROVAL"
            self._audit("succession_claim", "pending_guardian", f"role={role}")
            self._write(state)
            return {"claim_id": claim_id, "status": state["phase"]}

    def approve_claim(
        self,
        claim_id: str,
        guardian_role: str,
        guardian_profile_id: str,
        *,
        now: float | None = None,
    ) -> dict[str, str]:
        role = guardian_role.upper()
        if role not in _GUARDIAN_ROLES or not guardian_profile_id:
            raise PermissionError("A verified Father or Mother approval is required.")
        timestamp = float(time.time() if now is None else now)
        with self._lock:
            state = self._read()
            claim = state.get("claim")
            if not claim or not hmac.compare_digest(claim.get("claim_id", ""), claim_id):
                raise PermissionError("No matching succession claim is pending.")
            if timestamp - claim["created_at"] > self.grace_seconds:
                state["claim"] = None
                state["phase"] = "ELIGIBLE_FOR_CLAIM"
                self._write(state)
                self._audit("succession_claim", "expired", f"role={claim['role']}")
                raise PermissionError("Succession claim expired; initiate a new claim.")
            claim["guardian_approval"] = {
                "role": role,
                "profile_id": guardian_profile_id,
                "at": timestamp,
            }
            state["temporary_master_profile_id"] = claim["profile_id"]
            state["phase"] = f"TEMPORARY_MASTER_{claim['role']}"
            state["claim"] = None
            self._audit(
                "succession_promotion", "authorized",
                f"role={claim['role']}; guardian={role}",
            )
            self._write(state)
            return {
                "status": state["phase"],
                "profile_id": state["temporary_master_profile_id"],
            }

    def reclaim(self, master_profile_id: str, *, now: float | None = None) -> None:
        if not master_profile_id:
            raise ValueError("A verified Master profile is required.")
        with self._lock:
            state = self._read()
            state["temporary_master_profile_id"] = None
            state["claim"] = None
            state["manual_emergency"] = False
            state["manual_emergency_by"] = None
            state["manual_emergency_at"] = None
            state["phase"] = "ACTIVE"
            state["activity"]["MASTER"] = {
                "profile_id": master_profile_id,
                "at": float(time.time() if now is None else now),
            }
            self._audit("succession_reclaim", "authorized", "Master reclaimed control.")
            self._write(state)

    def effective_role(self, profile_id: str, enrolled_role: str) -> str:
        if enrolled_role == "MASTER":
            return "MASTER"
        with self._lock:
            state = self._read()
            if (
                profile_id
                and state.get("temporary_master_profile_id")
                and hmac.compare_digest(
                    state["temporary_master_profile_id"], profile_id
                )
            ):
                return "MASTER"
        return enrolled_role

    def status(self, *, now: float | None = None) -> dict[str, Any]:
        timestamp = float(time.time() if now is None else now)
        with self._lock:
            state = self._read()
            old_phase = state["phase"]
            eligibility = self._eligibility(state, timestamp)
            phase = state["phase"]
            if phase == "ACTIVE" and eligibility["master_inactive"]:
                phase = "ELIGIBLE_FOR_CLAIM"
                state["phase"] = phase
                state["eligible_since"] = eligibility["succession_started_at"]
            if (
                not state.get("temporary_master_profile_id")
                and not state.get("claim")
                and eligibility["sister_eligible"]
                and phase not in {"SANCTUARY_LOCKDOWN"}
            ):
                phase = "SISTER_ELIGIBLE"
                state["phase"] = phase
            if (
                phase == "SISTER_ELIGIBLE"
                and eligibility["sister_inactive"]
                and eligibility["elapsed_since_succession"]
                    >= self.grace_seconds * 2
                and not state.get("claim")
                and not state.get("temporary_master_profile_id")
            ):
                phase = "SANCTUARY_LOCKDOWN"
                state["phase"] = phase
            if old_phase != phase:
                self._audit(
                    "succession_state", phase,
                    "Automatic succession state transition.",
                )
                self._write(state)
            return {
                "phase": phase,
                "master_inactive": eligibility["master_inactive"],
                "brother_inactive": eligibility["brother_inactive"],
                "sister_inactive": eligibility["sister_inactive"],
                "brother_eligible": eligibility["brother_eligible"],
                "sister_eligible": eligibility["sister_eligible"],
                "claim_pending": bool(state.get("claim")),
                "temporary_master_active": bool(state.get("temporary_master_profile_id")),
                "inactivity_threshold_seconds": self.inactivity_seconds,
                "succession_grace_seconds": self.grace_seconds,
            }

    def is_locked_down(self) -> bool:
        try:
            with self._lock:
                return self._read()["phase"] == "SANCTUARY_LOCKDOWN"
        except Exception:
            _LOG.exception("Succession state unreadable; treating service as locked down.")
            return True

    def _eligibility(self, state: dict[str, Any], now: float) -> dict[str, bool]:
        activity = state["activity"]
        master_last = _activity_time(activity.get("MASTER"))
        brother_last = _activity_time(activity.get("BROTHER"))
        sister_last = _activity_time(activity.get("SISTER"))
        manual_at = state.get("manual_emergency_at")
        if isinstance(manual_at, (int, float)):
            succession_started_at = float(manual_at)
        elif master_last is not None:
            succession_started_at = master_last + self.inactivity_seconds
        else:
            succession_started_at = None
        master_inactive = bool(
            state.get("manual_emergency")
            or master_last is not None and now - master_last >= self.inactivity_seconds
        )
        brother_inactive = bool(
            brother_last is None
            or now - brother_last >= self.inactivity_seconds
        )
        sister_inactive = bool(
            sister_last is None
            or now - sister_last >= self.inactivity_seconds
        )
        elapsed_since_succession = (
            max(0.0, now - succession_started_at)
            if succession_started_at is not None and master_inactive else 0.0
        )
        return {
            "master_inactive": master_inactive,
            "brother_inactive": brother_inactive,
            "sister_inactive": sister_inactive,
            "brother_eligible": master_inactive,
            "sister_eligible": (
                master_inactive and brother_inactive
                and elapsed_since_succession >= self.grace_seconds
            ),
            "succession_started_at": succession_started_at,
            "elapsed_since_succession": elapsed_since_succession,
        }

    @staticmethod
    def _verify_emergency_credential(credential: dict[str, Any], passphrase: str) -> bool:
        if not isinstance(passphrase, str) or not passphrase or len(passphrase) > 1024:
            return False
        try:
            salt = bytes.fromhex(credential["salt"])
            expected = bytes.fromhex(credential["digest"])
            iterations = credential["iterations"]
            if iterations != _ITERATIONS:
                raise ValueError("Unsupported emergency credential format.")
        except (ValueError, KeyError, TypeError) as exc:
            raise RuntimeError("Stored emergency credential is invalid.") from exc
        actual = hashlib.pbkdf2_hmac(
            "sha256", passphrase.encode("utf-8"), salt, iterations
        )
        return hmac.compare_digest(actual, expected)

    def _emergency_is_locked(self, profile_id: str) -> bool:
        now = time.monotonic()
        with self._attempt_lock:
            if now < self._emergency_locked_until.get(profile_id, 0.0):
                return True
            attempts = self._emergency_attempts[profile_id]
            while attempts and now - attempts[0] > 300:
                attempts.popleft()
            return False

    def _record_emergency_failure(self, profile_id: str) -> None:
        now = time.monotonic()
        with self._attempt_lock:
            attempts = self._emergency_attempts[profile_id]
            attempts.append(now)
            if len(attempts) >= 5:
                self._emergency_locked_until[profile_id] = now + 900
                attempts.clear()
                _LOG.warning("Emergency Passphrase temporarily rate-limited for profile %s.", profile_id)

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return _new_state()
        try:
            state = json.loads(self.path.read_text(encoding="utf-8"))
            if (
                not isinstance(state, dict)
                or state.get("version") != 1
                or state.get("phase") not in {
                    "ACTIVE", "ELIGIBLE_FOR_CLAIM", "SISTER_ELIGIBLE",
                    "SANCTUARY_LOCKDOWN",
                }
                and not str(state.get("phase", "")).startswith(
                    ("AWAITING_BROTHER_", "AWAITING_SISTER_", "TEMPORARY_MASTER_")
                )
                or not isinstance(state.get("activity"), dict)
                or not isinstance(state.get("emergency_credentials"), dict)
            ):
                raise ValueError("Unsupported succession state format.")
            return state
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise RuntimeError("Succession state is unreadable; operation denied.") from exc

    def _write(self, state: dict[str, Any]) -> None:
        assert_safe_write_target(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=self.path.parent,
                prefix=self.path.name + ".", suffix=".tmp", delete=False,
            ) as stream:
                temporary = Path(stream.name)
                assert_safe_write_target(temporary)
                json.dump(state, stream, ensure_ascii=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            if os.name != "nt":
                temporary.chmod(0o600)
            os.replace(temporary, self.path)
        except OSError as exc:
            raise RuntimeError("Could not persist succession state; operation denied.") from exc
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @staticmethod
    def _audit(event: str, decision: str, detail: str) -> None:
        security.audit(
            event, "succession", security.TIER_DANGEROUS, decision, detail
        )


def _activity_time(record: Any) -> float | None:
    if not isinstance(record, dict):
        return None
    value = record.get("at")
    return float(value) if isinstance(value, (int, float)) else None
