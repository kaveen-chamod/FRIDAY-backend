from __future__ import annotations

import io
import os
import shutil
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app import voice


class StaticAudioRuntimeTests(unittest.TestCase):
    def test_ffmpeg_ffprobe_and_opus_encoder_decoder_are_available(self):
        runtime = voice.verify_static_ffmpeg()
        self.assertTrue(Path(runtime["ffmpeg"]).is_file())
        self.assertTrue(Path(runtime["ffprobe"]).is_file())
        self.assertTrue(runtime["opus_decoder"])
        self.assertTrue(runtime["opus_encoder"])
        self.assertTrue(voice.ffmpeg_available())
        self.assertTrue(voice.ffprobe_available())
        self.assertEqual(
            os.path.normcase(shutil.which("ffmpeg")),
            os.path.normcase(runtime["ffmpeg"]),
        )
        self.assertEqual(
            os.path.normcase(shutil.which("ffprobe")),
            os.path.normcase(runtime["ffprobe"]),
        )

    def test_opus_ogg_round_trip_has_required_pcm_format(self):
        sample_rate = 24_000
        samples = (
            np.sin(2 * np.pi * 440 * np.arange(sample_rate // 3) / sample_rate)
            * 8000
        ).astype("<i2")
        encoded = voice.encode_ogg_opus(
            samples.tobytes(), f"audio/pcm;rate={sample_rate}"
        )
        self.assertTrue(encoded.startswith(b"OggS"))
        metadata = voice.probe_audio(encoded)
        self.assertIn("ogg", metadata["format"]["format_name"].split(","))
        self.assertEqual(metadata["stream"]["codec_name"], "opus")
        self.assertEqual(metadata["stream"]["channels"], 1)

        decoded = voice.decode_to_pcm_wav(encoded, mime_type="audio/ogg")
        with wave.open(io.BytesIO(decoded), "rb") as wav:
            self.assertEqual(wav.getframerate(), 16_000)
            self.assertEqual(wav.getnchannels(), 1)
            self.assertEqual(wav.getsampwidth(), 2)
            self.assertGreater(wav.getnframes(), 0)

    def test_startup_audio_self_test_runs_real_conversion(self):
        result = voice.verify_audio_pipeline()
        self.assertTrue(result["opus_decoder"])
        self.assertTrue(result["opus_encoder"])

    def test_invalid_or_corrupted_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            voice.decode_to_pcm_wav(b"")
        with self.assertRaises(ValueError):
            voice.decode_to_pcm_wav(b"not an ogg", mime_type="audio/ogg")
        with self.assertRaises(ValueError):
            voice.decode_to_pcm_wav(b"not audio", mime_type="text/plain")
        with self.assertRaises(ValueError):
            voice.decode_to_pcm_wav(b"corrupt audio payload")
        with self.assertRaises(ValueError):
            voice.encode_ogg_opus(b"\0\0", "audio/pcm")
        with self.assertRaises(ValueError):
            voice.encode_ogg_opus(b"\0", "audio/pcm;rate=16000")

    def test_input_size_and_duration_limits_are_enforced(self):
        with self.assertRaises(ValueError):
            voice.decode_to_pcm_wav(b"x" * (25 * 1024 * 1024 + 1))
        too_long_pcm = b"\0\0" * (16000 * 120 + 1)
        with self.assertRaisesRegex(ValueError, "duration"):
            voice.encode_ogg_opus(too_long_pcm, "audio/pcm;rate=16000")

    def test_ffmpeg_timeouts_become_controlled_errors(self):
        timeout = subprocess.TimeoutExpired("ffmpeg", 1)
        with (
            patch(
                "app.voice.probe_audio",
                return_value={
                    "duration": 0.1,
                    "stream": {"codec_name": "opus"},
                    "format": {"format_name": "ogg"},
                },
            ),
            patch("app.voice.subprocess.run", side_effect=timeout),
        ):
            with self.assertRaisesRegex(ValueError, "decoding exceeded"):
                voice.decode_to_pcm_wav(b"OggS" + b"x", mime_type="audio/ogg")

        with patch("app.voice.subprocess.run", side_effect=timeout):
            with self.assertRaisesRegex(RuntimeError, "encoding exceeded"):
                voice.encode_ogg_opus(b"\0\0" * 100, "audio/pcm;rate=16000")

    def test_audio_tempfiles_are_removed_after_conversion(self):
        samples = b"\0\0" * 1600
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(voice.tempfile, "tempdir", directory):
                before = list(Path(directory).iterdir())
                ogg = voice.encode_ogg_opus(samples, "audio/pcm;rate=16000")
                voice.decode_to_pcm_wav(ogg, mime_type="audio/ogg")
                with self.assertRaises(ValueError):
                    voice.probe_audio(b"corrupted")
                self.assertEqual(list(Path(directory).iterdir()), before)


if __name__ == "__main__":
    unittest.main()
