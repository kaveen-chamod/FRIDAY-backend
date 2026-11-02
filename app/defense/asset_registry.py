"""Master-enrolled registry of explicitly authorized defensive assets."""
from __future__ import annotations

import ipaddress
import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from app.access_control import assert_safe_write_target


_ASSET_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
_ASSET_TYPES = {"server", "network", "cloud", "application", "iot_device"}
_CRITICALITIES = {"LOW", "MEDIUM", "HIGH", "MISSION_CRITICAL"}
SAFE_AUTO_ACTIONS = {
    "RATE_LIMIT_IP",
    "BLOCK_INDICATOR_TEMP",
    "REVOKE_SUSPECT_SESSION",
    "DISPATCH_SECURITY_ALERT",
    "ELEVATE_AUDIT_LOGGING",
}
_MAX_ASSETS = 1_000
_MAX_REGISTRY_BYTES = 2 * 1024 * 1024
_HOSTNAME_LABEL = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$")


class AssetRegistryError(ValueError):
    """An asset or registry operation is invalid."""


class AuthorizedAssetRegistry:
    def __init__(self, path: str | Path | None = None) -> None:
        root = Path(os.environ.get("FRIDAY_STATE_DIR") or Path(__file__).resolve().parents[2] / "memory")
        self.path = Path(path) if path is not None else root / "authorized_assets.json"
        self._lock = threading.RLock()
        self._assets = self._load()

    def register(
        self,
        asset: Mapping[str, Any],
        *,
        ownership_verified: bool,
    ) -> dict[str, Any]:
        if not isinstance(asset, Mapping):
            raise AssetRegistryError("Asset registration must be an object.")
        if ownership_verified is not True:
            raise AssetRegistryError("Master verification is required to register ownership.")
        asset_id = asset.get("asset_id")
        asset_type = asset.get("asset_type")
        if not isinstance(asset_id, str) or not _ASSET_ID.fullmatch(asset_id):
            raise AssetRegistryError("asset_id must be a safe identifier of at most 80 characters.")
        if not isinstance(asset_type, str) or asset_type not in _ASSET_TYPES:
            raise AssetRegistryError("asset_type is not supported.")
        sources = asset.get("allowed_telemetry_sources", [])
        if not isinstance(sources, list) or len(sources) > 32:
            raise AssetRegistryError("allowed_telemetry_sources must contain at most 32 entries.")
        normalized_sources = sorted({self._normalize_source(source) for source in sources})
        actions = asset.get("permitted_defense_actions", [])
        if (
            not isinstance(actions, list)
            or len(actions) > len(SAFE_AUTO_ACTIONS)
            or any(not isinstance(action, str) or action not in SAFE_AUTO_ACTIONS for action in actions)
        ):
            raise AssetRegistryError("permitted_defense_actions must use only SAFE_AUTO actions.")
        criticality = asset.get("criticality", "MEDIUM")
        if not isinstance(criticality, str) or criticality.upper() not in _CRITICALITIES:
            raise AssetRegistryError("criticality must be LOW, MEDIUM, HIGH, or MISSION_CRITICAL.")
        record = {
            "asset_id": asset_id,
            "asset_type": asset_type,
            "ownership_verified": True,
            "allowed_telemetry_sources": normalized_sources,
            "permitted_defense_actions": sorted(set(actions)),
            "criticality": criticality.upper(),
            "registered_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "status": "active",
        }
        with self._lock:
            if asset_id in self._assets:
                raise AssetRegistryError("asset_id is already registered.")
            if len(self._assets) >= _MAX_ASSETS:
                raise AssetRegistryError("The authorized asset registry is full.")
            self._assets[asset_id] = record
            try:
                self._save_locked()
            except Exception:
                del self._assets[asset_id]
                raise
            return dict(record)

    def get(self, asset_id: str) -> dict[str, Any] | None:
        with self._lock:
            item = self._assets.get(asset_id)
            return dict(item) if item else None

    def require(self, asset_id: str) -> dict[str, Any]:
        item = self.get(asset_id)
        if item is None or item.get("ownership_verified") is not True:
            raise AssetRegistryError("The requested asset is not registered and verified.")
        return item

    def list_assets(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(self._assets[key]) for key in sorted(self._assets)]

    def telemetry_source_allowed(self, asset_id: str, source: str) -> bool:
        item = self.require(asset_id)
        candidate = self._normalize_source(source)
        for configured in item["allowed_telemetry_sources"]:
            try:
                if "/" in configured and ipaddress.ip_address(candidate) in ipaddress.ip_network(configured):
                    return True
            except ValueError:
                pass
            if configured == candidate:
                return True
        return False

    @staticmethod
    def _normalize_source(source: Any) -> str:
        if not isinstance(source, str) or not source.strip() or len(source) > 253:
            raise AssetRegistryError("Telemetry sources must be IPs, CIDRs, or hostnames.")
        value = source.strip().rstrip(".")
        try:
            if "/" in value:
                network = ipaddress.ip_network(value, strict=False)
                if network.prefixlen == 0:
                    raise AssetRegistryError(
                        "A telemetry source cannot authorize the entire address space."
                    )
                return str(network)
            return str(ipaddress.ip_address(value))
        except ValueError:
            labels = value.split(".")
            if (
                len(labels) < 2
                or any(not _HOSTNAME_LABEL.fullmatch(label) for label in labels)
            ):
                raise AssetRegistryError("Telemetry hostname is invalid.")
            return value.lower()

    def _load(self) -> dict[str, dict[str, Any]]:
        try:
            size = self.path.stat().st_size
        except FileNotFoundError:
            return {}
        except OSError as exc:
            raise AssetRegistryError("Authorized asset registry cannot be inspected.") from exc
        if size > _MAX_REGISTRY_BYTES:
            raise AssetRegistryError("Authorized asset registry exceeds its size limit.")
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AssetRegistryError("Authorized asset registry is unreadable.") from exc
        if not isinstance(value, dict) or value.get("schema_version") != 1:
            raise AssetRegistryError("Authorized asset registry schema is invalid.")
        assets = value.get("assets")
        if not isinstance(assets, list) or len(assets) > _MAX_ASSETS:
            raise AssetRegistryError("Authorized asset registry contents are invalid.")
        loaded: dict[str, dict[str, Any]] = {}
        for row in assets:
            if not isinstance(row, dict):
                raise AssetRegistryError("Authorized asset record is invalid.")
            asset_id = row.get("asset_id")
            sources = row.get("allowed_telemetry_sources")
            actions = row.get("permitted_defense_actions")
            registered_at = row.get("registered_at")
            if (
                not isinstance(asset_id, str)
                or not _ASSET_ID.fullmatch(asset_id)
                or row.get("ownership_verified") is not True
                or asset_id in loaded
                or not isinstance(row.get("asset_type"), str)
                or row.get("asset_type") not in _ASSET_TYPES
                or not isinstance(sources, list)
                or len(sources) > 32
                or not isinstance(actions, list)
                or len(actions) > len(SAFE_AUTO_ACTIONS)
                or any(
                    not isinstance(action, str) or action not in SAFE_AUTO_ACTIONS
                    for action in actions
                )
                or not isinstance(row.get("criticality"), str)
                or row.get("criticality") not in _CRITICALITIES
                or not isinstance(registered_at, str)
                or not registered_at
                or len(registered_at) > 64
                or row.get("status") != "active"
            ):
                raise AssetRegistryError("Authorized asset record failed validation.")
            try:
                normalized_sources = sorted(
                    {self._normalize_source(source) for source in sources}
                )
            except AssetRegistryError as exc:
                raise AssetRegistryError(
                    "Authorized asset source record failed validation."
                ) from exc
            loaded[asset_id] = {
                "asset_id": asset_id,
                "asset_type": row["asset_type"],
                "ownership_verified": True,
                "allowed_telemetry_sources": normalized_sources,
                "permitted_defense_actions": sorted(set(actions)),
                "criticality": row["criticality"],
                "registered_at": registered_at,
                "status": "active",
            }
        return loaded

    def _save_locked(self) -> None:
        data = {
            "schema_version": 1,
            "assets": [self._assets[key] for key in sorted(self._assets)],
        }
        encoded = json.dumps(data, ensure_ascii=True, indent=2, allow_nan=False)
        if len(encoded.encode("utf-8")) > _MAX_REGISTRY_BYTES:
            raise AssetRegistryError("Authorized asset registry exceeds its size limit.")
        assert_safe_write_target(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        try:
            with temporary.open("x", encoding="utf-8") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        except OSError as exc:
            raise AssetRegistryError("Authorized asset registry could not be saved.") from exc
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
