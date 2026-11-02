from __future__ import annotations

import io
import json
import os
import re
import subprocess
import tempfile
import wave

import static_ffmpeg
from static_ffmpeg.run import get_or_fetch_platform_executables_else_raise


static_ffmpeg.add_paths()
_STATIC_FFMPEG, _STATIC_FFPROBE = (
    get_or_fetch_platform_executables_else_raise()
)
_STATIC_FFMPEG_DIR = os.path.dirname(_STATIC_FFMPEG)
_PATH_ENTRIES = os.environ.get("PATH", "").split(os.pathsep)
if not _PATH_ENTRIES or os.path.normcase(os.path.abspath(_PATH_ENTRIES[0])) != os.path.normcase(
    os.path.abspath(_STATIC_FFMPEG_DIR)
):
    os.environ["PATH"] = os.pathsep.join(
        [_STATIC_FFMPEG_DIR, os.environ.get("PATH", "")]
    )

_MAX_AUDIO_BYTES = 25 * 1024 * 1024
_MAX_DECODE_SECONDS = 120
_FFMPEG_TIMEOUT_SECONDS = 45
_MAX_PCM_WAV_BYTES = 16000 * 2 * _MAX_DECODE_SECONDS + 65536


def ffmpeg_executable() -> str:
    if not os.path.isfile(_STATIC_FFMPEG):
        raise RuntimeError(
            "Static FFmpeg is unavailable; reinstall the static-ffmpeg dependency."
        )
    return _STATIC_FFMPEG


def ffprobe_executable() -> str:
    if not os.path.isfile(_STATIC_FFPROBE):
        raise RuntimeError(
            "Static FFprobe is unavailable; reinstall the static-ffmpeg dependency."
        )
    return _STATIC_FFPROBE


def ffmpeg_available() -> bool:
    return os.path.isfile(_STATIC_FFMPEG)


def ffprobe_available() -> bool:
    return os.path.isfile(_STATIC_FFPROBE)


def opus_codec_support() -> dict[str, bool]:
    result = subprocess.run(
        [ffmpeg_executable(), "-hide_banner", "-codecs"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=10,
        check=False,
        text=True,
    )
    if result.returncode:
        raise RuntimeError("FFmpeg could not report its codec capabilities.")
    for line in result.stdout.splitlines():
        columns = line.split()
        if len(columns) >= 2 and columns[1] == "opus":
            flags = columns[0]
            return {"decoder": "D" in flags, "encoder": "E" in flags}
    return {"decoder": False, "encoder": False}


def verify_static_ffmpeg() -> dict[str, str | bool]:
    """Confirm both bundled executables and the required Opus codec are usable."""
    ffmpeg = ffmpeg_executable()
    ffprobe = ffprobe_executable()
    for executable in (ffmpeg, ffprobe):
        result = subprocess.run(
            [executable, "-hide_banner", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
            text=True,
        )
        if result.returncode:
            raise RuntimeError(f"Audio executable failed its version check: {executable}")
    codecs = opus_codec_support()
    if not codecs["decoder"] or not codecs["encoder"]:
        raise RuntimeError(
            "The configured FFmpeg build must provide both Opus decoding and encoding."
        )
    return {
        "ffmpeg": ffmpeg,
        "ffprobe": ffprobe,
        "opus_decoder": codecs["decoder"],
        "opus_encoder": codecs["encoder"],
    }


def verify_audio_pipeline() -> dict[str, str | bool]:
    """Run a short in-memory round-trip to catch broken codec/runtime installs."""
    tools = verify_static_ffmpeg()
    silence = b"\x00\x00" * 1600
    ogg = encode_ogg_opus(silence, "audio/pcm;rate=16000")
    wav_audio = decode_to_pcm_wav(ogg, mime_type="audio/ogg")
    with wave.open(io.BytesIO(wav_audio), "rb") as wav:
        if (
            wav.getnchannels() != 1
            or wav.getsampwidth() != 2
            or wav.getframerate() != 16000
            or wav.getnframes() == 0
        ):
            raise RuntimeError("Static FFmpeg audio round-trip validation failed.")
    return tools


def _bounded_timeout(timeout: int) -> int:
    try:
        return max(1, min(int(timeout), _FFMPEG_TIMEOUT_SECONDS))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Audio conversion timeout must be a positive integer.") from exc


def probe_audio(audio: bytes, *, timeout: int = _FFMPEG_TIMEOUT_SECONDS) -> dict:
    if not isinstance(audio, bytes) or not audio or len(audio) > _MAX_AUDIO_BYTES:
        raise ValueError("Audio payload is empty or exceeds the 25 MB limit.")
    timeout = _bounded_timeout(timeout)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="friday-audio-", suffix=".input", delete=False
        ) as temporary:
            temporary.write(audio)
            temporary.flush()
            temporary_path = temporary.name
        command = [
            ffprobe_executable(), "-v", "error",
            "-show_entries",
            "format=format_name,duration:stream=codec_name,codec_type,sample_rate,channels",
            "-of", "json", "-i", temporary_path,
        ]
        try:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ValueError("Audio inspection exceeded its time limit.") from exc
        if result.returncode:
            detail = result.stderr.decode("utf-8", errors="replace").strip()[-1000:]
            raise ValueError(
                f"Audio payload is invalid or unsupported: {detail or 'FFprobe rejected the stream'}"
            )
        try:
            metadata = json.loads(result.stdout)
            streams = metadata["streams"]
            audio_stream = next(
                stream for stream in streams if stream.get("codec_type") == "audio"
            )
            duration = float(metadata["format"]["duration"])
        except (
            KeyError, StopIteration, TypeError, ValueError, json.JSONDecodeError
        ) as exc:
            raise ValueError(
                "Audio payload has no valid audio stream or duration."
            ) from exc
        if duration <= 0:
            raise ValueError("Audio payload has an invalid duration.")
        if duration > _MAX_DECODE_SECONDS:
            raise ValueError("Audio payload exceeds the 120 second duration limit.")
        return {
            "stream": audio_stream,
            "format": metadata["format"],
            "duration": duration,
        }
    finally:
        if temporary_path is not None:
            os.unlink(temporary_path)


def decode_to_pcm_wav(
    audio: bytes,
    *,
    mime_type: str = "audio/unknown",
    timeout: int = _FFMPEG_TIMEOUT_SECONDS,
) -> bytes:
    """Decode bounded audio to mono 16 kHz, signed 16-bit PCM WAV."""
    if not isinstance(audio, bytes) or not audio:
        raise ValueError("Audio payload is empty.")
    if len(audio) > _MAX_AUDIO_BYTES:
        raise ValueError("Audio payload exceeds the 25 MB limit.")
    if not isinstance(mime_type, str) or not mime_type.lower().startswith("audio/"):
        raise ValueError("Audio payload must use an audio MIME type.")
    if mime_type.lower().split(";", 1)[0].strip() in {"audio/ogg", "audio/opus"}:
        if not audio.startswith(b"OggS"):
            raise ValueError("Telegram voice payload is not a valid Ogg container.")

    timeout = _bounded_timeout(timeout)
    source = probe_audio(audio, timeout=timeout)
    if mime_type.lower().split(";", 1)[0].strip() in {"audio/ogg", "audio/opus"}:
        if (
            "ogg" not in source["format"]["format_name"].split(",")
            or source["stream"].get("codec_name") != "opus"
        ):
            raise ValueError("Telegram voice must contain Ogg/Opus audio.")
    output_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="friday-audio-", suffix=".wav", delete=False
        ) as temporary:
            output_path = temporary.name
        command = [
            ffmpeg_executable(), "-nostdin", "-y", "-hide_banner", "-loglevel",
            "error", "-threads", "1", "-filter_threads", "1",
            "-protocol_whitelist", "pipe", "-i", "pipe:0", "-t",
            str(_MAX_DECODE_SECONDS), "-vn", "-ac", "1", "-ar", "16000",
            "-c:a", "pcm_s16le", "-f", "wav", output_path,
        ]
        try:
            result = subprocess.run(
                command,
                input=audio,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ValueError("Audio decoding exceeded its time limit.") from exc
        if result.returncode:
            detail = result.stderr.decode("utf-8", errors="replace").strip()[-1000:]
            raise ValueError(
                f"Audio decoding failed: {detail or 'unsupported audio format'}"
            )
        with open(output_path, "rb") as decoded_stream:
            decoded_audio = decoded_stream.read(_MAX_PCM_WAV_BYTES + 1)
        if len(decoded_audio) > _MAX_PCM_WAV_BYTES:
            raise ValueError("Decoded audio exceeds the 120 second duration limit.")
        try:
            with wave.open(io.BytesIO(decoded_audio), "rb") as wav:
                if (
                    wav.getnchannels() != 1
                    or wav.getsampwidth() != 2
                    or wav.getframerate() != 16000
                    or wav.getnframes() > 16000 * _MAX_DECODE_SECONDS
                    or wav.getnframes() <= 0
                ):
                    raise ValueError("Audio decoder produced unexpected PCM parameters.")
        except (wave.Error, EOFError) as exc:
            raise ValueError(
                "Audio decoder did not produce a valid PCM WAV stream."
            ) from exc
        return decoded_audio
    finally:
        if output_path is not None:
            os.unlink(output_path)


def encode_ogg_opus(
    audio: bytes,
    mime_type: str,
    *,
    timeout: int = _FFMPEG_TIMEOUT_SECONDS,
) -> bytes:
    """Encode WAV or Gemini raw 16-bit mono PCM as a validated Ogg Opus note."""
    if not isinstance(audio, bytes) or not audio or len(audio) > _MAX_AUDIO_BYTES:
        raise ValueError("Audio payload is empty or exceeds the 25 MB limit.")
    if not isinstance(mime_type, str) or not mime_type.lower().startswith("audio/"):
        raise ValueError("Audio payload must use an audio MIME type.")

    timeout = _bounded_timeout(timeout)
    command = [
        ffmpeg_executable(), "-nostdin", "-hide_banner", "-loglevel", "error",
        "-threads", "1", "-filter_threads", "1", "-protocol_whitelist", "pipe",
    ]
    if audio.startswith(b"RIFF") and audio[8:12] == b"WAVE":
        source = probe_audio(audio, timeout=timeout)
        command.extend(["-i", "pipe:0"])
    else:
        if not re.search(r"(?:rate|sample_rate)\s*=\s*\d+", mime_type, re.I):
            raise ValueError("Raw PCM MIME type must include its sample rate.")
        rate_match = re.search(
            r"(?:rate|sample_rate)\s*=\s*(\d+)", mime_type, re.I
        )
        sample_rate = int(rate_match.group(1)) if rate_match else 0
        if not 8000 <= sample_rate <= 96000:
            raise ValueError("Raw PCM sample rate is outside supported bounds.")
        if len(audio) % 2:
            raise ValueError("Raw 16-bit PCM payload has an incomplete sample.")
        duration = len(audio) / (sample_rate * 2)
        if duration <= 0 or duration > _MAX_DECODE_SECONDS:
            raise ValueError("Raw PCM duration is outside supported bounds.")
        command.extend([
            "-f", "s16le", "-ar", str(sample_rate), "-ac", "1", "-i", "pipe:0",
        ])
        source = {"duration": duration}
    if source["duration"] > _MAX_DECODE_SECONDS:
        raise ValueError("Audio payload exceeds the 120 second duration limit.")
    command.extend([
        "-t", str(_MAX_DECODE_SECONDS), "-vn", "-ac", "1", "-c:a", "libopus",
        "-b:a", "32k", "-application", "voip", "-f", "ogg", "pipe:1",
    ])
    try:
        result = subprocess.run(
            command,
            input=audio,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Audio encoding exceeded its time limit.") from exc
    if result.returncode:
        detail = result.stderr.decode("utf-8", errors="replace").strip()[-1000:]
        raise RuntimeError(
            f"Static FFmpeg could not encode Ogg Opus: {detail or 'encoder unavailable'}"
        )
    encoded = result.stdout
    if not encoded.startswith(b"OggS"):
        raise RuntimeError("Static FFmpeg did not produce an Ogg voice note.")
    metadata = probe_audio(encoded, timeout=timeout)
    stream = metadata["stream"]
    if (
        "ogg" not in metadata["format"]["format_name"].split(",")
        or stream.get("codec_name") != "opus"
        or stream.get("channels") != 1
    ):
        raise RuntimeError("Encoded audio is not a mono Ogg/Opus voice note.")
    return encoded
