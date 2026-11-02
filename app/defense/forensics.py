"""Append-only, bounded SHA-256 chain for sanitized defense incidents."""
from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from app.access_control import assert_safe_write_target
from memory.memory_manager import redact_sensitive_content


_GENESIS = "0" * 64
_MAX_LEDGER_BYTES = 32 * 1024 * 1024
_MAX_RECORD_BYTES = 16 * 1024
_MAX_RECORDS = 100_000


class ForensicsError(OSError):
    """Incident ledger is corrupt, full, or unavailable."""


class IncidentLedger:
    def __init__(self, path: str | Path | None = None) -> None:
        root = Path(os.environ.get("FRIDAY_STATE_DIR") or Path(__file__).resolve().parents[2] / "memory")
        self.path = Path(path) if path is not None else root / "defense_incidents.jsonl"
        self._lock = threading.RLock()

    def append(self, incident: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(incident, Mapping):
            raise ForensicsError("Incident must be an object.")
        with self._lock:
            previous_hash, count = self._verified_tail()
            if count >= _MAX_RECORDS:
                raise ForensicsError("Incident ledger record limit reached.")
            record = {
                "incident_id": str(incident.get("incident_id") or ""),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "category": redact_sensitive_content(str(incident.get("category") or ""))[:64],
                "threat_score": float(incident.get("threat_score", 0.0)),
                "target_asset": redact_sensitive_content(str(incident.get("target_asset") or ""))[:80],
                "evidence_hashes": self._evidence_hashes(incident.get("evidence", [])),
                "actions_taken": self._sanitize_actions(incident.get("actions_taken", [])),
                "rollback_status": redact_sensitive_content(
                    str(incident.get("rollback_status") or "not_applicable")
                )[:80],
                "previous_hash": previous_hash,
            }
            if not record["incident_id"] or not 0.0 <= record["threat_score"] <= 1.0:
                raise ForensicsError("Incident fields are invalid.")
            record["record_hash"] = self._hash_record(record)
            encoded = (
                json.dumps(record, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)
                + "\n"
            ).encode("utf-8")
            if len(encoded) > _MAX_RECORD_BYTES:
                raise ForensicsError("Incident record exceeds its size limit.")
            current_size = self.path.stat().st_size if self.path.exists() else 0
            if current_size + len(encoded) > _MAX_LEDGER_BYTES:
                raise ForensicsError("Incident ledger is full.")
            try:
                assert_safe_write_target(self.path)
                self.path.parent.mkdir(parents=True, exist_ok=True)
                descriptor = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
                try:
                    offset = 0
                    while offset < len(encoded):
                        written = os.write(descriptor, encoded[offset:])
                        if written <= 0:
                            raise ForensicsError("Incident ledger append was incomplete.")
                        offset += written
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            except ForensicsError:
                raise
            except OSError as exc:
                raise ForensicsError("Incident ledger append failed.") from exc
            return dict(record)

    def list_incidents(self, *, offset: int = 0, limit: int = 50) -> dict[str, Any]:
        if (
            isinstance(offset, bool)
            or not isinstance(offset, int)
            or offset < 0
            or isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 100
        ):
            raise ValueError("offset must be non-negative and limit must be 1 to 100.")
        with self._lock:
            records = self._read_verified()
            page = records[offset : offset + limit]
            return {
                "total": len(records),
                "offset": offset,
                "limit": limit,
                "incidents": page,
                "ledger_valid": True,
            }

    def verify(self) -> bool:
        with self._lock:
            self._read_verified()
            return True

    def _verified_tail(self) -> tuple[str, int]:
        records = self._read_verified()
        return (records[-1]["record_hash"] if records else _GENESIS, len(records))

    def _read_verified(self) -> list[dict[str, Any]]:
        try:
            size = self.path.stat().st_size
        except FileNotFoundError:
            return []
        except OSError as exc:
            raise ForensicsError("Incident ledger could not be inspected.") from exc
        if size > _MAX_LEDGER_BYTES:
            raise ForensicsError("Incident ledger exceeds its configured size limit.")
        try:
            content = self.path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise ForensicsError("Incident ledger could not be read.") from exc
        records: list[dict[str, Any]] = []
        previous_hash = _GENESIS
        for line in content.splitlines():
            if not line:
                continue
            if len(records) >= _MAX_RECORDS:
                raise ForensicsError("Incident ledger record limit reached.")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ForensicsError("Incident ledger contains invalid JSON.") from exc
            if (
                not isinstance(record, dict)
                or record.get("previous_hash") != previous_hash
                or record.get("record_hash") != self._hash_record(record)
            ):
                raise ForensicsError("Incident ledger hash chain validation failed.")
            previous_hash = record["record_hash"]
            records.append(record)
        return records

    @staticmethod
    def _hash_record(record: Mapping[str, Any]) -> str:
        body = {key: value for key, value in record.items() if key != "record_hash"}
        canonical = json.dumps(body, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _evidence_hashes(evidence: Any) -> list[str]:
        if not isinstance(evidence, list) or len(evidence) > 32:
            raise ForensicsError("Incident evidence must be a list of at most 32 items.")
        result = []
        for item in evidence:
            sanitized = redact_sensitive_content(str(item))[:2_048]
            result.append(hashlib.sha256(sanitized.encode("utf-8")).hexdigest())
        return result

    @staticmethod
    def _sanitize_actions(actions: Any) -> list[dict[str, Any]]:
        if not isinstance(actions, list) or len(actions) > 16:
            raise ForensicsError("Incident actions must be a list of at most 16 items.")
        result = []
        for action in actions:
            if not isinstance(action, Mapping):
                raise ForensicsError("Incident action record is invalid.")
            result.append({
                key: redact_sensitive_content(str(value))[:256]
                for key, value in action.items()
                if key in {
                    "action_id",
                    "action",
                    "target_asset",
                    "indicator_hash",
                    "applied_at",
                    "expires_at",
                    "rollback_status",
                    "rollback_callable",
                    "status",
                    "rollback_reason",
                }
            })
        return result
