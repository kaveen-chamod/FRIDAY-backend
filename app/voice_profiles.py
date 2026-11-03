from __future__ import annotations

import io
import base64
import json
import logging
import os
import secrets
import statistics
import threading
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from app.access_control import ROLES
from core import voice_id


_LOG = logging.getLogger("friday.voice")
_PROFILE_FILE = Path(
    os.environ.get("FRIDAY_CONFIG_DIR")
    or Path(__file__).resolve().parents[1] / "config"
).expanduser().resolve() / "family_voice.dat"
_SAMPLE_RATE = 16_000
_MIN_SAMPLES = 2
_MAX_SAMPLES = 8
_MAX_SAMPLE_BYTES = 10 * 1024 * 1024
_FAMILY_THRESHOLD = 0.75
_MASTER_THRESHOLD = 0.82
_CLOUD_PROFILE_MAGIC = b"FRVP1"
_CLOUD_PROFILE_KEY_ENV = "FRIDAY_VOICE_PROFILE_KEY"


class VoiceProfileStore:
    """Encrypted, local family voice templates with fail-closed matching."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path or _PROFILE_FILE)
        self._lock = threading.RLock()

    def list_metadata(self) -> list[dict[str, str]]:
        with self._lock:
            return [
                {
                    "profile_id": profile["profile_id"],
                    "name": profile["name"],
                    "role": profile["role"],
                    "created_at": profile["created_at"],
                }
                for profile in self._read()["profiles"]
            ]

    def contains(self, profile_id: str) -> bool:
        with self._lock:
            return any(
                item["profile_id"] == profile_id
                for item in self._read()["profiles"]
            )

    def get_metadata(self, profile_id: str) -> dict[str, str] | None:
        with self._lock:
            for item in self._read()["profiles"]:
                if item["profile_id"] == profile_id:
                    return {
                        "profile_id": item["profile_id"],
                        "name": item["name"],
                        "role": item["role"],
                        "created_at": item["created_at"],
                    }
        return None

    def enroll(self, name: str, role: str, audio_samples: list[bytes]) -> dict[str, str]:
        clean_name = name.strip() if isinstance(name, str) else ""
        role = role.upper() if isinstance(role, str) else ""
        if not clean_name or len(clean_name) > 80:
            raise ValueError("Profile name must contain 1 to 80 characters.")
        if role not in ROLES or role == "GUEST":
            raise ValueError("Choose MASTER, FATHER, MOTHER, BROTHER, or SISTER.")
        if not isinstance(audio_samples, list) or not (
            _MIN_SAMPLES <= len(audio_samples) <= _MAX_SAMPLES
        ):
            raise ValueError("Enroll with 2 to 8 separate WAV voice samples.")

        embeddings: list[list[float]] = []
        pitch_values: list[float] = []
        formant_values: list[list[float]] = []
        for sample in audio_samples:
            waveform = _decode_wav(sample)
            embeddings.append(voice_id._embedding(waveform).astype(float).tolist())
            pitch, formants = _acoustic_features(waveform)
            pitch_values.append(pitch)
            formant_values.append(formants)

        common_formants = [
            statistics.median(values[index] for values in formant_values)
            for index in range(3)
        ]
        profile = {
            "profile_id": secrets.token_urlsafe(18),
            "name": clean_name,
            "role": role,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "embeddings": embeddings,
            "pitch_hz": statistics.median(pitch_values),
            "formants_hz": common_formants,
        }
        with self._lock:
            record = self._read()
            if any(item["name"].casefold() == clean_name.casefold()
                   for item in record["profiles"]):
                raise ValueError("A voice profile with that name already exists.")
            if len(record["profiles"]) >= 32:
                raise ValueError("The maximum of 32 enrolled profiles has been reached.")
            record["profiles"].append(profile)
            self._write(record)
        return {
            "profile_id": profile["profile_id"],
            "name": clean_name,
            "role": role,
            "created_at": profile["created_at"],
        }

    def delete(self, profile_id: str) -> bool:
        if not profile_id or len(profile_id) > 128:
            raise ValueError("Invalid profile identifier.")
        with self._lock:
            record = self._read()
            remaining = [
                item for item in record["profiles"]
                if item["profile_id"] != profile_id
            ]
            if len(remaining) == len(record["profiles"]):
                return False
            record["profiles"] = remaining
            self._write(record)
            return True

    def identify(self, audio: bytes) -> dict[str, Any]:
        waveform = _decode_wav(audio)
        with self._lock:
            profiles = self._read()["profiles"]
        if not profiles:
            return {"profile_id": None, "name": "Guest", "role": "GUEST", "similarity": 0.0}
        embedding = voice_id._embedding(waveform)
        pitch, formants = _acoustic_features(waveform)

        matches: list[tuple[float, dict[str, Any]]] = []
        for profile in profiles:
            vectors = np.asarray(profile["embeddings"], dtype=np.float32)
            if vectors.ndim != 2 or vectors.shape[1] != embedding.size:
                continue
            scores = vectors @ embedding
            score = float(np.mean(scores))
            if (
                score > _threshold_for(profile["role"])
                and _acoustic_sanity_check(
                    pitch, formants, profile["pitch_hz"], profile["formants_hz"]
                )
            ):
                matches.append((score, profile))

        matches.sort(key=lambda item: item[0], reverse=True)
        if not matches or (
            len(matches) > 1 and matches[0][0] - matches[1][0] < 0.04
        ):
            return {"profile_id": None, "name": "Guest", "role": "GUEST", "similarity": 0.0}
        score, profile = matches[0]
        return {
            "profile_id": profile["profile_id"],
            "name": profile["name"],
            "role": profile["role"],
            "similarity": score,
        }

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"version": 1, "profiles": []}
        try:
            payload = _unprotect_profile(self.path.read_bytes())
            record = json.loads(payload.decode("utf-8"))
            if record.get("version") != 1 or not isinstance(record.get("profiles"), list):
                raise ValueError("Unknown voice-profile database format.")
            for item in record["profiles"]:
                if (
                    not isinstance(item, dict)
                    or item.get("role") not in ROLES
                    or not isinstance(item.get("profile_id"), str)
                    or not isinstance(item.get("embeddings"), list)
                ):
                    raise ValueError("Invalid voice-profile database record.")
            return record
        except (
            OSError, UnicodeDecodeError, ValueError, TypeError, AttributeError,
            InvalidTag,
        ) as exc:
            raise RuntimeError("Encrypted voice-profile data is unreadable.") from exc

    def _write(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        protected = _protect_profile(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )
        temporary = self.path.with_name(self.path.name + ".tmp")
        try:
            with temporary.open("wb") as stream:
                stream.write(protected)
                stream.flush()
                os.fsync(stream.fileno())
            if os.name != "nt":
                temporary.chmod(0o600)
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)


def _threshold_for(role: str) -> float:
    return _MASTER_THRESHOLD if role == "MASTER" else _FAMILY_THRESHOLD


def _decode_wav(payload: bytes) -> np.ndarray:
    if not isinstance(payload, bytes) or not payload or len(payload) > _MAX_SAMPLE_BYTES:
        raise ValueError("Provide a non-empty WAV sample no larger than 10 MB.")
    try:
        with wave.open(io.BytesIO(payload), "rb") as wav:
            if (
                wav.getnchannels() != 1
                or wav.getsampwidth() != 2
                or wav.getframerate() != _SAMPLE_RATE
                or wav.getcomptype() != "NONE"
            ):
                raise ValueError("Voice samples must be mono, 16-bit PCM WAV at 16 kHz.")
            frame_count = wav.getnframes()
            if not int(_SAMPLE_RATE * 1.5) <= frame_count <= _SAMPLE_RATE * 20:
                raise ValueError("Speak for 1.5 to 20 seconds in each voice sample.")
            frames = wav.readframes(frame_count)
    except (wave.Error, EOFError) as exc:
        raise ValueError("Voice samples must be valid PCM WAV audio.") from exc
    waveform = np.frombuffer(frames, dtype="<i2").copy()
    if not np.isfinite(waveform).all():
        raise ValueError("Voice sample contains invalid audio values.")
    return waveform


def _acoustic_features(waveform: np.ndarray) -> tuple[float, list[float]]:
    audio = np.asarray(waveform, dtype=np.float64).reshape(-1) / 32768.0
    if audio.size < int(_SAMPLE_RATE * 1.5) or not np.isfinite(audio).all():
        raise ValueError("Voice sample is too short or invalid.")

    frame_size = 640
    hop_size = 320
    pitches: list[float] = []
    for start in range(0, len(audio) - frame_size + 1, hop_size):
        frame = audio[start:start + frame_size]
        if float(np.sqrt(np.mean(frame * frame))) < 0.004:
            continue
        frame = (frame - np.mean(frame)) * np.hamming(frame_size)
        correlation = np.correlate(frame, frame, mode="full")[frame_size - 1:]
        first_lag = int(_SAMPLE_RATE / 400)
        last_lag = min(int(_SAMPLE_RATE / 60), len(correlation) - 1)
        if last_lag <= first_lag or correlation[0] <= 0:
            continue
        lag = first_lag + int(np.argmax(correlation[first_lag:last_lag + 1]))
        if correlation[lag] / correlation[0] >= 0.30:
            pitches.append(_SAMPLE_RATE / lag)
    if not pitches:
        raise ValueError("Voice sample does not contain a stable voiced pitch.")

    emphasized = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
    segment = emphasized[:min(len(emphasized), _SAMPLE_RATE * 2)]
    segment = segment * np.hamming(len(segment))
    order = 16
    correlations = np.correlate(segment, segment, mode="full")[len(segment) - 1:]
    matrix = np.empty((order, order), dtype=np.float64)
    for row in range(order):
        for column in range(order):
            matrix[row, column] = correlations[abs(row - column)]
    try:
        coefficients = np.linalg.solve(matrix, correlations[1:order + 1])
    except np.linalg.LinAlgError as exc:
        raise ValueError("Voice sample does not contain stable formant characteristics.") from exc
    roots = np.roots(np.r_[1.0, -coefficients])
    candidates: list[float] = []
    for root in roots:
        if np.imag(root) <= 0:
            continue
        frequency = np.angle(root) * _SAMPLE_RATE / (2 * np.pi)
        bandwidth = -0.5 * (_SAMPLE_RATE / (2 * np.pi)) * np.log(abs(root))
        if 200 <= frequency <= 4000 and 0 < bandwidth < 400:
            candidates.append(float(frequency))
    candidates.sort()
    if len(candidates) < 3:
        raise ValueError("Voice sample does not contain stable formant characteristics.")
    return float(statistics.median(pitches)), candidates[:3]


def _acoustic_sanity_check(
    pitch: float, formants: list[float], reference_pitch: float,
    reference_formants: list[float],
) -> bool:
    if not (
        isinstance(reference_pitch, (int, float))
        and reference_pitch > 0
        and len(formants) >= 3
        and len(reference_formants) >= 3
    ):
        return False
    pitch_ratio = max(pitch, reference_pitch) / min(pitch, reference_pitch)
    matching_formants = sum(
        abs(left - right) / max(left, right) <= 0.40
        for left, right in zip(formants[:3], reference_formants[:3])
        if left > 0 and right > 0
    )
    return pitch_ratio <= 1.8 and matching_formants >= 2


def voice_backend_ready() -> bool:
    return voice_id.backend_ready()


def _cloud_profile_key() -> bytes | None:
    encoded = os.environ.get(_CLOUD_PROFILE_KEY_ENV, "").strip()
    if not encoded:
        return None
    try:
        key = base64.b64decode(encoded, altchars=b"-_", validate=True)
    except (ValueError, TypeError, UnicodeError) as exc:
        raise RuntimeError(
            f"{_CLOUD_PROFILE_KEY_ENV} must be a URL-safe base64-encoded 32-byte key."
        ) from exc
    if len(key) != 32:
        raise RuntimeError(
            f"{_CLOUD_PROFILE_KEY_ENV} must encode exactly 32 bytes."
        )
    return key


def _protect_profile(payload: bytes) -> bytes:
    key = _cloud_profile_key()
    if key is None:
        return voice_id._protect(payload)
    nonce = secrets.token_bytes(12)
    return _CLOUD_PROFILE_MAGIC + nonce + AESGCM(key).encrypt(
        nonce, payload, _CLOUD_PROFILE_MAGIC
    )


def _unprotect_profile(payload: bytes) -> bytes:
    if not payload.startswith(_CLOUD_PROFILE_MAGIC):
        return voice_id._unprotect(payload)
    key = _cloud_profile_key()
    if key is None:
        raise RuntimeError(
            f"{_CLOUD_PROFILE_KEY_ENV} is required to decrypt cloud voice profiles."
        )
    nonce_start = len(_CLOUD_PROFILE_MAGIC)
    nonce = payload[nonce_start:nonce_start + 12]
    ciphertext = payload[nonce_start + 12:]
    if len(nonce) != 12 or len(ciphertext) < 16:
        raise ValueError("Encrypted voice-profile data is truncated.")
    return AESGCM(key).decrypt(nonce, ciphertext, _CLOUD_PROFILE_MAGIC)
