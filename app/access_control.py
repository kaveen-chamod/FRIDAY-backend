from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import threading
import time
from collections import deque
from contextvars import ContextVar
from pathlib import Path
from typing import Any


_LOG = logging.getLogger("friday.access")
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG_ROOT = _PROJECT_ROOT / "config"
_CONFIG_ROOT = Path(
    os.environ.get("FRIDAY_CONFIG_DIR") or _DEFAULT_CONFIG_ROOT
).expanduser().resolve()
MASTER_AUTH_FILE = _CONFIG_ROOT / "master_auth.json"
SPEAKER_KEY_FILE = _CONFIG_ROOT / "speaker_token.key"
_CONFIG_ROOTS = {_DEFAULT_CONFIG_ROOT.resolve(), _CONFIG_ROOT}
_PROTECTED_CORE_FILES = frozenset(
    (_PROJECT_ROOT / "core" / name).resolve()
    for name in ("security.py", "confirm.py", "path_guard.py", "sandbox.py")
)
_PBKDF2_ITERATIONS = 600_000
_MIN_PASSPHRASE_LENGTH = 16
_SPEAKER_TOKEN_TTL = 120
_AUTH_LOCK = threading.RLock()
_ATTEMPT_LOCK = threading.Lock()
_FAILED_ATTEMPTS: deque[float] = deque()
_LOCKED_UNTIL = 0.0
_ATTEMPT_WINDOW_SECONDS = 300.0
_MAX_FAILED_ATTEMPTS = 5
_speaker_secret_cache: str | None = None

ROLES = ("MASTER", "FATHER", "MOTHER", "BROTHER", "SISTER", "GUEST")
_role_context: ContextVar[str] = ContextVar("friday_role", default="GUEST")
_passphrase_context: ContextVar[str | None] = ContextVar(
    "friday_master_passphrase", default=None
)

_GUEST_TOOLS = frozenset({"weather_report", "web_search"})
_STANDARD_TOOLS = _GUEST_TOOLS | frozenset({
    "youtube_video", "spotify_control", "flight_finder",
})
_MOTHER_TOOLS = _STANDARD_TOOLS | frozenset({
    "iot_control", "reminder", "remember_intention",
    "proactive_planner", "save_memory", "recall_memory", "memory_health",
})
_PARENT_ROLES = frozenset({"FATHER", "MOTHER"})


class MasterAuthError(ValueError):
    pass


def current_role() -> str:
    role = _role_context.get()
    return role if role in ROLES else "GUEST"


def current_master_passphrase() -> str | None:
    return _passphrase_context.get()


def enter_request_context(role: str, passphrase: str | None = None):
    normalized = role.upper() if isinstance(role, str) else "GUEST"
    if normalized not in ROLES:
        normalized = "GUEST"
    return (
        _role_context.set(normalized),
        _passphrase_context.set(passphrase),
    )


def leave_request_context(tokens) -> None:
    role_token, passphrase_token = tokens
    _passphrase_context.reset(passphrase_token)
    _role_context.reset(role_token)


def role_for_tool(name: str, *, mother_home_access: bool = True) -> bool:
    role = current_role()
    if role == "MASTER":
        return True
    if role in {"BROTHER", "SISTER"}:
        return name in _STANDARD_TOOLS
    if role in _PARENT_ROLES:
        return name in (_MOTHER_TOOLS if mother_home_access else _STANDARD_TOOLS)
    return name in _GUEST_TOOLS


def assert_safe_write_target(path: str | os.PathLike, *, internal_secret: bool = False) -> Path:
    """Block core-security and config writes outside narrowly named auth stores."""
    target = Path(path).expanduser().resolve(strict=False)
    if target in _PROTECTED_CORE_FILES:
        raise PermissionError("Writes to protected core security files are forbidden.")
    if not any(_is_relative_to(target, root) for root in _CONFIG_ROOTS):
        return target
    allowed_secrets = {MASTER_AUTH_FILE.resolve(), SPEAKER_KEY_FILE.resolve()}
    if internal_secret and target in allowed_secrets:
        return target
    raise PermissionError("Writes to root configuration files are forbidden.")


def _is_relative_to(target: Path, parent: Path) -> bool:
    try:
        target.relative_to(parent)
        return True
    except ValueError:
        return False


def permitted_declarations(declarations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        item for item in declarations
        if isinstance(item, dict) and role_for_tool(str(item.get("name", "")))
    ]


def require_master_passphrase(passphrase: str | None) -> None:
    if not verify_master_passphrase(passphrase):
        _LOG.warning("Master authorization denied.")
        raise MasterAuthError("Valid Master Passphrase verification is required.")


def configure_master_passphrase(passphrase: str) -> None:
    if not isinstance(passphrase, str) or len(passphrase) < _MIN_PASSPHRASE_LENGTH:
        raise ValueError("Master Passphrase must contain at least 16 characters.")
    if len(passphrase) > 1024:
        raise ValueError("Master Passphrase is too long.")

    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", passphrase.encode("utf-8"), salt, _PBKDF2_ITERATIONS
    )
    record = {
        "version": 1,
        "algorithm": "pbkdf2_sha256",
        "iterations": _PBKDF2_ITERATIONS,
        "salt": base64.b64encode(salt).decode("ascii"),
        "digest": base64.b64encode(digest).decode("ascii"),
    }
    with _AUTH_LOCK:
        if MASTER_AUTH_FILE.exists():
            raise MasterAuthError(
                "Master Passphrase is already configured; setup cannot replace it."
            )
        MASTER_AUTH_FILE.parent.mkdir(parents=True, exist_ok=True)
        assert_safe_write_target(MASTER_AUTH_FILE, internal_secret=True)
        temporary = MASTER_AUTH_FILE.with_name(
            MASTER_AUTH_FILE.name + "." + secrets.token_hex(8) + ".tmp"
        )
        assert_safe_write_target(temporary, internal_secret=True)
        try:
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(record, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, MASTER_AUTH_FILE)
        finally:
            temporary.unlink(missing_ok=True)


def master_passphrase_configured() -> bool:
    return MASTER_AUTH_FILE.is_file()


def verify_master_passphrase(passphrase: str | None) -> bool:
    global _LOCKED_UNTIL
    if not isinstance(passphrase, str) or not passphrase or len(passphrase) > 1024:
        return False
    now = time.monotonic()
    with _ATTEMPT_LOCK:
        if now < _LOCKED_UNTIL:
            return False
        while _FAILED_ATTEMPTS and now - _FAILED_ATTEMPTS[0] > _ATTEMPT_WINDOW_SECONDS:
            _FAILED_ATTEMPTS.popleft()
    with _AUTH_LOCK:
        try:
            record = json.loads(MASTER_AUTH_FILE.read_text(encoding="utf-8"))
            if (
                record.get("version") != 1
                or record.get("algorithm") != "pbkdf2_sha256"
                or record.get("iterations") != _PBKDF2_ITERATIONS
            ):
                raise ValueError("Unsupported Master Passphrase record.")
            salt = base64.b64decode(record["salt"], validate=True)
            expected = base64.b64decode(record["digest"], validate=True)
        except FileNotFoundError:
            return False
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Master Passphrase configuration is unreadable.") from exc

    actual = hashlib.pbkdf2_hmac(
        "sha256", passphrase.encode("utf-8"), salt, _PBKDF2_ITERATIONS
    )
    verified = hmac.compare_digest(actual, expected)
    with _ATTEMPT_LOCK:
        if verified:
            _FAILED_ATTEMPTS.clear()
            _LOCKED_UNTIL = 0.0
        else:
            now = time.monotonic()
            _FAILED_ATTEMPTS.append(now)
            if len(_FAILED_ATTEMPTS) >= _MAX_FAILED_ATTEMPTS:
                _LOCKED_UNTIL = now + _ATTEMPT_WINDOW_SECONDS
                _FAILED_ATTEMPTS.clear()
                _LOG.warning("Master authorization temporarily rate-limited.")
    return verified


def issue_speaker_token(
    profile_id: str, role: str, signing_secret: str, *, now: int | None = None
) -> str:
    if role not in ROLES or role == "GUEST" or not profile_id:
        raise ValueError("A verified enrolled speaker is required.")
    issued_at = int(time.time() if now is None else now)
    payload = {
        "profile_id": profile_id,
        "role": role,
        "iat": issued_at,
        "exp": issued_at + _SPEAKER_TOKEN_TTL,
    }
    encoded = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).rstrip(b"=").decode("ascii")
    signature = hmac.new(
        signing_secret.encode("utf-8"), encoded.encode("ascii"), hashlib.sha256
    ).digest()
    return f"{encoded}.{base64.urlsafe_b64encode(signature).rstrip(b'=').decode('ascii')}"


def load_speaker_signing_secret() -> str:
    """Load a server-only signer, never reusing the client-visible API token."""
    global _speaker_secret_cache
    with _AUTH_LOCK:
        if _speaker_secret_cache:
            return _speaker_secret_cache
        configured = os.environ.get("FRIDAY_SPEAKER_TOKEN_SECRET", "").strip()
        if configured:
            if len(configured) < 32:
                raise RuntimeError(
                    "FRIDAY_SPEAKER_TOKEN_SECRET must contain at least 32 characters."
                )
            _speaker_secret_cache = configured
            return configured
        try:
            encoded = SPEAKER_KEY_FILE.read_text(encoding="ascii").strip()
        except FileNotFoundError:
            try:
                assert_safe_write_target(SPEAKER_KEY_FILE, internal_secret=True)
                SPEAKER_KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
                key = secrets.token_bytes(32)
                encoded = base64.b64encode(key).decode("ascii")
                descriptor = os.open(
                    SPEAKER_KEY_FILE,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
            except FileExistsError:
                try:
                    encoded = SPEAKER_KEY_FILE.read_text(encoding="ascii").strip()
                except (OSError, UnicodeDecodeError) as exc:
                    raise RuntimeError(
                        "Could not read the local speaker-token signing key."
                    ) from exc
            except OSError as exc:
                raise RuntimeError(
                    "Could not create the local speaker-token signing key."
                ) from exc
            else:
                try:
                    with os.fdopen(descriptor, "w", encoding="ascii") as stream:
                        stream.write(encoded)
                        stream.flush()
                        os.fsync(stream.fileno())
                except OSError as exc:
                    try:
                        SPEAKER_KEY_FILE.unlink(missing_ok=True)
                    except OSError:
                        _LOG.exception("Could not remove incomplete speaker-token key.")
                    raise RuntimeError(
                        "Could not persist the local speaker-token signing key."
                    ) from exc
        except (OSError, UnicodeDecodeError) as exc:
            raise RuntimeError("Could not read the local speaker-token signing key.") from exc
        try:
            raw_key = base64.b64decode(encoded, validate=True)
        except ValueError as exc:
            raise RuntimeError("The local speaker-token signing key is invalid.") from exc
        if len(raw_key) != 32:
            raise RuntimeError("The local speaker-token signing key has an invalid length.")
        _speaker_secret_cache = base64.urlsafe_b64encode(raw_key).decode("ascii")
        return _speaker_secret_cache


def verify_speaker_token(
    token: str | None, signing_secret: str, *, now: int | None = None
) -> dict[str, str] | None:
    if not isinstance(token, str) or len(token) > 2048:
        return None
    try:
        encoded, encoded_signature = token.split(".", 1)
        signature = base64.urlsafe_b64decode(
            encoded_signature + "=" * (-len(encoded_signature) % 4)
        )
        expected = hmac.new(
            signing_secret.encode("utf-8"), encoded.encode("ascii"), hashlib.sha256
        ).digest()
        if not hmac.compare_digest(signature, expected):
            return None
        payload_bytes = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
        payload = json.loads(payload_bytes.decode("utf-8"))
        current_time = int(time.time() if now is None else now)
        if (
            payload["role"] not in ROLES
            or payload["role"] == "GUEST"
            or not isinstance(payload["profile_id"], str)
            or not payload["profile_id"]
            or not isinstance(payload["iat"], int)
            or not isinstance(payload["exp"], int)
            or payload["iat"] > current_time + 30
            or payload["exp"] <= current_time
            or payload["exp"] - payload["iat"] > _SPEAKER_TOKEN_TTL
        ):
            return None
        return {"profile_id": payload["profile_id"], "role": payload["role"]}
    except (ValueError, TypeError, KeyError, UnicodeDecodeError, json.JSONDecodeError):
        return None
