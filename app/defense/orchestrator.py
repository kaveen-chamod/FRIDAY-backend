"""Reversible, application-bound SAFE_AUTO mitigations and expiry rollback."""
from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import logging
import threading
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Mapping

from app.defense.asset_registry import AuthorizedAssetRegistry, SAFE_AUTO_ACTIONS
from app.defense.forensics import ForensicsError, IncidentLedger
from app.defense.threat_matrix import ThreatMatrix, ThreatEvaluationError


_LOG = logging.getLogger("friday.defense.orchestrator")
_HIGH_IMPACT_ACTIONS = {
    "ISOLATE_SERVER",
    "NETWORK_FIREWALL_CHANGE",
    "CLOUD_CONFIGURATION_CHANGE",
    "DELETE_ACCOUNT",
    "DELETE_DATABASE",
    "DELETE_MEMORY",
    "MODIFY_SECURITY_POLICY",
    "UNSEAL_SECRET",
    "SHUTDOWN_SERVER",
    "REBOOT_SERVER",
}
_MAX_ACTIVE_ACTIONS = 2_000
_MAX_RATE_KEYS = 4_096
_REQUEST_RATE_WINDOW_SECONDS = 60.0
_RATE_LIMIT_PER_MINUTE = 30
_DEFAULT_LOCK = threading.Lock()
_DEFAULT_ORCHESTRATOR: "DefenseOrchestrator | None" = None


class DefenseActionError(ValueError):
    """A mitigation cannot safely be applied or reversed."""


class DefenseOrchestrator:
    def __init__(
        self,
        registry: AuthorizedAssetRegistry,
        threat_matrix: ThreatMatrix,
        incident_ledger: IncidentLedger,
        *,
        alert_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.registry = registry
        self.threat_matrix = threat_matrix
        self.incident_ledger = incident_ledger
        self.alert_callback = alert_callback
        self._lock = threading.RLock()
        self._actions: dict[str, dict[str, Any]] = {}
        self._rate_limits: dict[str, float] = {}
        self._blocked_ips: dict[str, float] = {}
        self._revoked_indicators: dict[str, float] = {}
        self._request_times: dict[str, deque[float]] = defaultdict(deque)
        self._threats: deque[dict[str, Any]] = deque(maxlen=1_000)
        self._rollback_task: asyncio.Task | None = None
        self._rollback_stop: asyncio.Event | None = None
        self._previous_audit_level: int | None = None
        self._audit_level_expiry = 0.0

    async def start(self, *, poll_seconds: float = 1.0) -> None:
        if not 0.1 <= poll_seconds <= 60:
            raise ValueError("poll_seconds must be between 0.1 and 60.")
        if self._rollback_task is not None and not self._rollback_task.done():
            return
        self._rollback_stop = asyncio.Event()
        self._rollback_task = asyncio.create_task(
            self._rollback_loop(poll_seconds),
            name="friday-defense-expiry-rollback",
        )

    async def stop(self) -> None:
        task = self._rollback_task
        if task is None:
            return
        if self._rollback_stop is not None:
            self._rollback_stop.set()
        await task
        self._rollback_task = None
        self._rollback_stop = None

    def evaluate_event(self, event: Mapping[str, Any]) -> dict[str, Any]:
        """Correlate authorized telemetry and apply only registered SAFE_AUTO actions."""
        assessment = self.threat_matrix.evaluate(event)
        asset_id = assessment["target_asset"]
        indicator = event.get("indicator")
        evidence = event.get("evidence", [])
        if not isinstance(evidence, list) or len(evidence) > 32:
            raise ThreatEvaluationError("evidence must be a list of at most 32 items.")
        self.incident_ledger.verify()
        actions: list[dict[str, Any]] = []
        created_action_ids: list[str] = []
        if assessment["safe_auto_actions"] and indicator is not None:
            for action_name in assessment["safe_auto_actions"]:
                try:
                    action = self.apply_safe_action(
                        asset_id,
                        action_name,
                        str(indicator),
                        autonomous=True,
                        record_incident=False,
                        _created_action_ids=created_action_ids,
                        _dispatch_alert=False,
                    )
                    actions.append(action)
                except DefenseActionError:
                    _LOG.warning(
                        "SAFE_AUTO action %s was refused for asset %s.",
                        action_name,
                        asset_id,
                    )
        if assessment["master_action_required"]:
            actions.append({
                "action_id": uuid.uuid4().hex,
                "target_asset": asset_id,
                "action": "NETWORK_FIREWALL_CHANGE",
                "status": "AWAITING_MASTER_AUTH",
                "autonomous": False,
                "rollback_status": "not_applied",
            })
        incident = {
            "incident_id": uuid.uuid4().hex,
            "category": assessment["category"],
            "threat_score": assessment["threat_score"],
            "target_asset": asset_id,
            "evidence": evidence,
            "actions_taken": actions,
            "rollback_status": (
                "scheduled" if any(item.get("expires_at") for item in actions)
                else "not_applicable"
            ),
        }
        try:
            forensic = self.incident_ledger.append(incident)
        except Exception:
            with self._lock:
                for action_id in created_action_ids:
                    action = self._actions.get(action_id)
                    if action is None:
                        continue
                    if action["rollback_status"] == "scheduled":
                        self._rollback_action_locked(action, "forensic_failure")
                    else:
                        self._actions.pop(action_id, None)
            _LOG.exception("Threat assessment could not be written to the incident ledger.")
            raise
        result = {**assessment, "incident_id": incident["incident_id"], "actions_taken": actions}
        with self._lock:
            self._threats.append({
                **result,
                "timestamp_utc": forensic["timestamp_utc"],
                "evidence_hashes": forensic["evidence_hashes"],
            })
        if assessment["master_alert_required"] and self.alert_callback is not None:
            try:
                dispatch_action = next(
                    (
                        item for item in actions
                        if item["action"] == "DISPATCH_SECURITY_ALERT"
                    ),
                    None,
                )
                if dispatch_action is not None:
                    self.alert_callback(
                        f"F.R.I.D.A.Y. defense alert for {asset_id}; "
                        f"indicator {dispatch_action['indicator_hash'][:16]}."
                    )
                else:
                    self.alert_callback(
                        "F.R.I.D.A.Y. defense alert: "
                        f"{assessment['classification']} ({assessment['category']}) "
                        f"for authorized asset {asset_id}; score "
                        f"{assessment['threat_score']:.2f}."
                    )
            except Exception:
                _LOG.exception("Master security alert callback failed.")
        return result

    def apply_safe_action(
        self,
        target_asset: str,
        action: str,
        indicator: str,
        *,
        ttl_seconds: int | None = None,
        autonomous: bool = False,
        record_incident: bool = True,
        _created_action_ids: list[str] | None = None,
        _dispatch_alert: bool = True,
    ) -> dict[str, Any]:
        if isinstance(action, str) and action in _HIGH_IMPACT_ACTIONS:
            raise DefenseActionError("High-impact actions are never autonomously executable.")
        if not isinstance(action, str) or action not in SAFE_AUTO_ACTIONS:
            raise DefenseActionError("Action is not on the SAFE_AUTO allowlist.")
        asset = self.registry.require(target_asset)
        if action not in asset["permitted_defense_actions"]:
            raise DefenseActionError("Action is not permitted for this registered asset.")
        if not isinstance(indicator, str) or not indicator.strip() or len(indicator) > 512:
            raise DefenseActionError("A bounded mitigation indicator is required.")
        if record_incident:
            self.incident_ledger.verify()
        if autonomous and action in {"REVOKE_SUSPECT_SESSION"}:
            raise DefenseActionError(
                "Session revocation requires a verified token indicator; autonomous telemetry cannot supply it."
            )
        indicator = indicator.strip()
        now = time.time()
        expires_at: float | None = None
        if action == "RATE_LIMIT_IP":
            indicator = self._normalize_ip(indicator)
            ttl = 900 if ttl_seconds is None else ttl_seconds
            if isinstance(ttl, bool) or not isinstance(ttl, int) or not 900 <= ttl <= 3_600:
                raise DefenseActionError("IP rate-limit TTL must be between 15 and 60 minutes.")
            expires_at = now + ttl
        elif action == "BLOCK_INDICATOR_TEMP":
            indicator = self._normalize_ip(indicator)
            ttl = 900 if ttl_seconds is None else ttl_seconds
            if isinstance(ttl, bool) or not isinstance(ttl, int) or not 1 <= ttl <= 1_800:
                raise DefenseActionError("Temporary indicator block TTL cannot exceed 30 minutes.")
            expires_at = now + ttl
        elif action == "REVOKE_SUSPECT_SESSION":
            if autonomous:
                raise DefenseActionError("Session revocation must be manually verified.")
            ttl = 1_800 if ttl_seconds is None else ttl_seconds
            if isinstance(ttl, bool) or not isinstance(ttl, int) or not 1 <= ttl <= 1_800:
                raise DefenseActionError("Session revocation TTL cannot exceed 30 minutes.")
            expires_at = now + ttl
            indicator = hashlib.sha256(indicator.encode("utf-8")).hexdigest()
        elif action == "DISPATCH_SECURITY_ALERT":
            ttl = None
            indicator = hashlib.sha256(indicator.encode("utf-8")).hexdigest()
        elif action == "ELEVATE_AUDIT_LOGGING":
            ttl = 900 if ttl_seconds is None else ttl_seconds
            if isinstance(ttl, bool) or not isinstance(ttl, int) or not 60 <= ttl <= 3_600:
                raise DefenseActionError("Audit logging TTL must be between 1 and 60 minutes.")
            expires_at = now + ttl

        with self._lock:
            existing = next(
                (
                    active for active in self._actions.values()
                    if active["target_asset"] == target_asset
                    and active["action"] == action
                    and active["indicator"] == indicator
                    and active["rollback_status"] in {"scheduled", "not_applicable"}
                ),
                None,
            )
            if existing is not None and expires_at is None:
                return self._public_action(existing)
            if existing is not None and expires_at is not None:
                monotonic_expiry = max(
                    existing.get("expires_monotonic") or 0.0,
                    time.monotonic() + (expires_at - now),
                )
                existing["expires_monotonic"] = monotonic_expiry
                existing["expires_at"] = datetime.fromtimestamp(
                    time.time() + (monotonic_expiry - time.monotonic()), timezone.utc
                ).isoformat(timespec="seconds")
                if action == "RATE_LIMIT_IP":
                    self._rate_limits[indicator] = max(
                        self._rate_limits.get(indicator, 0.0), monotonic_expiry
                    )
                elif action == "BLOCK_INDICATOR_TEMP":
                    self._blocked_ips[indicator] = max(
                        self._blocked_ips.get(indicator, 0.0), monotonic_expiry
                    )
                elif action == "REVOKE_SUSPECT_SESSION":
                    self._revoked_indicators[indicator] = max(
                        self._revoked_indicators.get(indicator, 0.0), monotonic_expiry
                    )
                return self._public_action(existing)
            if len(self._actions) >= _MAX_ACTIVE_ACTIONS:
                for old_id, old_action in list(self._actions.items()):
                    if old_action["rollback_status"] in {"rolled_back", "not_applicable"}:
                        del self._actions[old_id]
                        if len(self._actions) < _MAX_ACTIVE_ACTIONS:
                            break
                if len(self._actions) >= _MAX_ACTIVE_ACTIONS:
                    raise DefenseActionError("Active defense action capacity is full.")
            expires_monotonic = (
                time.monotonic() + (expires_at - now) if expires_at is not None else None
            )
            if action == "RATE_LIMIT_IP":
                self._rate_limits[indicator] = max(
                    self._rate_limits.get(indicator, 0.0), expires_monotonic or 0.0
                )
            elif action == "BLOCK_INDICATOR_TEMP":
                self._blocked_ips[indicator] = max(
                    self._blocked_ips.get(indicator, 0.0), expires_monotonic or 0.0
                )
            elif action == "REVOKE_SUSPECT_SESSION":
                self._revoked_indicators[indicator] = max(
                    self._revoked_indicators.get(indicator, 0.0), expires_monotonic or 0.0
                )
            elif action == "ELEVATE_AUDIT_LOGGING":
                if self._previous_audit_level is None:
                    defense_logger = logging.getLogger("friday.defense")
                    self._previous_audit_level = defense_logger.level
                self._audit_level_expiry = max(self._audit_level_expiry, expires_at or now)
                logging.getLogger("friday.defense").setLevel(logging.DEBUG)

            action_id = uuid.uuid4().hex
            applied_at = datetime.fromtimestamp(now, timezone.utc).isoformat(timespec="seconds")
            expiry_iso = (
                datetime.fromtimestamp(expires_at, timezone.utc).isoformat(timespec="seconds")
                if expires_at is not None else None
            )
            record = {
                "action_id": action_id,
                "target_asset": target_asset,
                "action": action,
                "indicator": indicator,
                "indicator_hash": hashlib.sha256(indicator.encode("utf-8")).hexdigest(),
                "applied_at": applied_at,
                "expires_at": expiry_iso,
                "rollback_callable": f"internal.rollback:{action_id}",
                "rollback_status": "scheduled" if expires_at is not None else "not_applicable",
                "autonomous": autonomous,
                "expires_monotonic": expires_monotonic,
            }
            self._actions[action_id] = record
            if _created_action_ids is not None:
                _created_action_ids.append(action_id)

        public_action = self._public_action(record)
        if record_incident:
            try:
                self.incident_ledger.append({
                    "incident_id": uuid.uuid4().hex,
                    "category": f"MITIGATION_{action}",
                    "threat_score": 0.0,
                    "target_asset": target_asset,
                    "evidence": [indicator],
                    "actions_taken": [public_action],
                    "rollback_status": public_action["rollback_status"],
                })
            except ForensicsError:
                with self._lock:
                    self._rollback_action_locked(record, "audit_failure")
                raise
        if action == "DISPATCH_SECURITY_ALERT" and _dispatch_alert and self.alert_callback:
            try:
                self.alert_callback(
                    f"F.R.I.D.A.Y. defense alert for {target_asset}; "
                    f"indicator {record['indicator_hash'][:16]}."
                )
            except Exception:
                _LOG.exception("Master security alert callback failed.")
        return public_action

    def manual_mitigate(
        self,
        target_asset: str,
        action: str,
        indicator: str,
        *,
        ttl_seconds: int | None,
        master_authorized: bool,
    ) -> dict[str, Any]:
        self.registry.require(target_asset)
        if isinstance(action, str) and action in _HIGH_IMPACT_ACTIONS:
            if not isinstance(indicator, str) or not indicator.strip() or len(indicator) > 512:
                raise DefenseActionError("A bounded mitigation indicator is required.")
            incident = self.incident_ledger.append({
                "incident_id": uuid.uuid4().hex,
                "category": "HIGH_IMPACT_REQUEST",
                "threat_score": 0.0,
                "target_asset": target_asset,
                "evidence": [indicator],
                "actions_taken": [{
                    "action_id": uuid.uuid4().hex,
                    "action": action,
                    "target_asset": target_asset,
                    "status": "AWAITING_MASTER_AUTH",
                    "rollback_status": "not_applied",
                }],
                "rollback_status": "not_applied",
            })
            if not master_authorized:
                return {
                    "status": "AWAITING_MASTER_AUTH",
                    "action": action,
                    "target_asset": target_asset,
                    "incident_id": incident["incident_id"],
                }
            return {
                "status": "AWAITING_MASTER_AUTH",
                "action": action,
                "target_asset": target_asset,
                "incident_id": incident["incident_id"],
                "message": "No high-impact executor is configured; no change was made.",
            }
        if not master_authorized:
            raise DefenseActionError("Master authorization is required for manual mitigation.")
        return self.apply_safe_action(
            target_asset,
            action,
            indicator,
            ttl_seconds=ttl_seconds,
            autonomous=False,
        )

    def rollback(self, action_id: str) -> dict[str, Any]:
        with self._lock:
            action = self._actions.get(action_id)
            if action is None:
                raise DefenseActionError("Defense action was not found.")
            if action["rollback_status"] in {"rolled_back", "not_applicable"}:
                return self._public_action(action)
            self.incident_ledger.append({
                "incident_id": uuid.uuid4().hex,
                "category": "MITIGATION_ROLLBACK",
                "threat_score": 0.0,
                "target_asset": action["target_asset"],
                "evidence": [action["indicator"]],
                "actions_taken": [{
                    **self._public_action(action),
                    "rollback_status": "rolled_back",
                    "rollback_reason": "manual",
                }],
                "rollback_status": "rolled_back",
            })
            self._rollback_action_locked(action, "manual")
            return self._public_action(action)

    def check_request(self, client_ip: str | None, *, now: float | None = None) -> str | None:
        if not client_ip:
            return None
        try:
            address = str(ipaddress.ip_address(client_ip))
        except ValueError:
            return None
        current = time.monotonic() if now is None else now
        with self._lock:
            for mapping in (self._blocked_ips, self._rate_limits):
                expiry = mapping.get(address)
                if expiry is not None and expiry <= current:
                    mapping.pop(address, None)
            if address in self._blocked_ips:
                return "blocked"
            if address not in self._rate_limits:
                return None
            if address not in self._request_times and len(self._request_times) >= _MAX_RATE_KEYS:
                oldest = min(
                    self._request_times,
                    key=lambda key: self._request_times[key][0] if self._request_times[key] else float("inf"),
                )
                del self._request_times[oldest]
            events = self._request_times[address]
            while events and current - events[0] >= _REQUEST_RATE_WINDOW_SECONDS:
                events.popleft()
            if len(events) >= _RATE_LIMIT_PER_MINUTE:
                return "rate_limited"
            events.append(current)
            return None

    def is_indicator_revoked(self, raw_indicator: str | None) -> bool:
        if not isinstance(raw_indicator, str) or not raw_indicator:
            return False
        digest = hashlib.sha256(raw_indicator.encode("utf-8")).hexdigest()
        with self._lock:
            expiry = self._revoked_indicators.get(digest)
            if expiry is not None and expiry <= time.monotonic():
                self._revoked_indicators.pop(digest, None)
                return False
            return expiry is not None

    def active_status(self) -> dict[str, Any]:
        with self._lock:
            now = time.monotonic()
            active_rate_limits = {
                ip: expiry for ip, expiry in self._rate_limits.items() if expiry > now
            }
            active_blocks = {
                ip: expiry for ip, expiry in self._blocked_ips.items() if expiry > now
            }
            active_revocations = {
                digest: expiry
                for digest, expiry in self._revoked_indicators.items()
                if expiry > now
            }
            actions = [
                self._public_action(record)
                for record in self._actions.values()
                if record["rollback_status"] == "scheduled"
                and (
                    record.get("expires_monotonic") is None
                    or record["expires_monotonic"] > now
                )
            ]
            return {
                "active_indicators": {
                    "rate_limited_ip_hashes": [
                        hashlib.sha256(ip.encode("ascii")).hexdigest()[:16]
                        for ip in active_rate_limits
                    ],
                    "blocked_ip_hashes": [
                        hashlib.sha256(ip.encode("ascii")).hexdigest()[:16]
                        for ip in active_blocks
                    ],
                    "revoked_session_hashes": [
                        digest[:16] for digest in active_revocations
                    ],
                },
                "temporary_actions": actions,
                "recent_threats": list(self._threats)[-100:],
                "correlation": self.threat_matrix.active_correlation_summary(),
            }

    async def _rollback_loop(self, poll_seconds: float) -> None:
        assert self._rollback_stop is not None
        while not self._rollback_stop.is_set():
            await asyncio.to_thread(self._expire)
            try:
                await asyncio.wait_for(
                    self._rollback_stop.wait(), timeout=poll_seconds
                )
            except asyncio.TimeoutError:
                continue

    def _expire(self) -> None:
        with self._lock:
            expired = self._expire_locked(time.monotonic())
        for action in expired:
            try:
                self.incident_ledger.append({
                    "incident_id": uuid.uuid4().hex,
                    "category": "MITIGATION_ROLLBACK",
                    "threat_score": 0.0,
                    "target_asset": action["target_asset"],
                    "evidence": [action["indicator"]],
                    "actions_taken": [{
                        **self._public_action(action),
                        "rollback_status": "rolled_back",
                        "rollback_reason": "expired",
                    }],
                    "rollback_status": "rolled_back",
                })
            except ForensicsError:
                _LOG.exception(
                    "Expired mitigation rollback could not be recorded in the incident ledger."
                )

    def _expire_locked(self, now: float) -> list[dict[str, Any]]:
        expired = [
            record for record in self._actions.values()
            if record["rollback_status"] == "scheduled"
            and record.get("expires_monotonic") is not None
            and record["expires_monotonic"] <= now
        ]
        for record in expired:
            self._rollback_action_locked(record, "expired")
        return expired

    def _rollback_action_locked(self, record: dict[str, Any], reason: str) -> None:
        action = record["action"]
        indicator = record["indicator"]
        record["rollback_status"] = "rolled_back"
        record["rollback_reason"] = reason
        now = time.monotonic()
        if action == "RATE_LIMIT_IP":
            remaining = self._remaining_action_expiries(
                action, now, record["action_id"], indicator
            )
            if remaining:
                self._rate_limits[indicator] = max(remaining)
            else:
                self._rate_limits.pop(indicator, None)
                self._request_times.pop(indicator, None)
        elif action == "BLOCK_INDICATOR_TEMP":
            remaining = self._remaining_action_expiries(
                action, now, record["action_id"], indicator
            )
            if remaining:
                self._blocked_ips[indicator] = max(remaining)
            else:
                self._blocked_ips.pop(indicator, None)
        elif action == "REVOKE_SUSPECT_SESSION":
            remaining = self._remaining_action_expiries(
                action, now, record["action_id"], indicator
            )
            if remaining:
                self._revoked_indicators[indicator] = max(remaining)
            else:
                self._revoked_indicators.pop(indicator, None)
        elif action == "ELEVATE_AUDIT_LOGGING":
            if (
                self._previous_audit_level is not None
                and not self._remaining_action_expiries(
                    action, now, record["action_id"]
                )
            ):
                logging.getLogger("friday.defense").setLevel(self._previous_audit_level)
                self._previous_audit_level = None
                self._audit_level_expiry = 0.0

    def _remaining_action_expiries(
        self,
        action: str,
        now: float,
        excluding_action_id: str,
        indicator: str | None = None,
    ) -> list[float]:
        return [
            record["expires_monotonic"]
            for record in self._actions.values()
            if record["action_id"] != excluding_action_id
            and record["action"] == action
            and (indicator is None or record["indicator"] == indicator)
            and record["rollback_status"] == "scheduled"
            and record.get("expires_monotonic") is not None
            and record["expires_monotonic"] > now
        ]

    @staticmethod
    def _normalize_ip(value: str) -> str:
        try:
            ip = ipaddress.ip_address(value)
        except ValueError as exc:
            raise DefenseActionError("IP mitigation indicators must be literal IP addresses.") from exc
        if ip.is_unspecified or ip.is_multicast:
            raise DefenseActionError("Unspecified and multicast addresses cannot be mitigated.")
        return str(ip)

    @staticmethod
    def _public_action(record: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in record.items()
            if key not in {"expires_monotonic", "indicator"}
        }


def bind_default_orchestrator(
    orchestrator: DefenseOrchestrator | None,
) -> None:
    global _DEFAULT_ORCHESTRATOR
    with _DEFAULT_LOCK:
        _DEFAULT_ORCHESTRATOR = orchestrator


def is_revoked_speaker_token(token: str | None) -> bool:
    with _DEFAULT_LOCK:
        orchestrator = _DEFAULT_ORCHESTRATOR
    return bool(orchestrator and orchestrator.is_indicator_revoked(token))
