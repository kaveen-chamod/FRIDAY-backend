from __future__ import annotations

import base64
import json
import tempfile
import threading
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app import access_control
from app.agent import FridayAgent
from app import tool_maker
from app.tool_maker import ToolMaker
from app.voice_profiles import (
    VoiceProfileStore,
    _acoustic_sanity_check,
    _threshold_for,
)
from core import security


_VALID_TOOL = '''from app.modules.base import BaseModule

class SafeExample(BaseModule):
    TOOL = {
        "name": "safe_example",
        "description": "A test proposal",
        "parameters": {"type": "OBJECT", "properties": {}},
    }

    def run(self, parameters):
        return "ok"
'''

_SPECIFICATION = json.dumps({
    "name": "safe_example",
    "description": "A test proposal",
    "parameters": {"type": "OBJECT", "properties": {}},
    "expected_output": "A short string.",
    "candidate_dependencies": [],
    "duplicate_of": None,
})


class FamilyRoleTests(unittest.TestCase):
    def test_cloud_voice_profiles_are_encrypted_with_configured_key(self):
        key = base64.urlsafe_b64encode(b"k" * 32).decode("ascii")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profiles.dat"
            store = VoiceProfileStore(path)
            record = {
                "version": 1,
                "profiles": [{
                    "profile_id": "profile-1",
                    "name": "Master",
                    "role": "MASTER",
                    "created_at": "2025-01-01T00:00:00+00:00",
                    "embeddings": [[0.1, 0.2]],
                }],
            }
            with patch.dict("os.environ", {"FRIDAY_VOICE_PROFILE_KEY": key}):
                store._write(record)
                self.assertNotIn(b'"Master"', path.read_bytes())
                self.assertEqual(store.list_metadata()[0]["role"], "MASTER")

    def test_guest_and_family_tool_matrix(self):
        tokens = access_control.enter_request_context("GUEST")
        try:
            self.assertTrue(access_control.role_for_tool("weather_report"))
            self.assertTrue(access_control.role_for_tool("web_search"))
            self.assertFalse(access_control.role_for_tool("recall_memory"))
            self.assertFalse(access_control.role_for_tool("iot_control"))
        finally:
            access_control.leave_request_context(tokens)

        expectations = {
            "FATHER": (
                ("reminder", "iot_control", "recall_memory"),
                ("computer_control", "memory_export"),
            ),
            "MOTHER": (
                ("reminder", "iot_control", "recall_memory"),
                ("computer_control", "memory_export"),
            ),
            "BROTHER": (("youtube_video", "web_search"), ("iot_control",)),
            "SISTER": (("spotify_control", "web_search"), ("memory_export",)),
        }
        for role, (allowed, denied) in expectations.items():
            tokens = access_control.enter_request_context(role)
            try:
                self.assertTrue(all(access_control.role_for_tool(name) for name in allowed))
                self.assertFalse(any(access_control.role_for_tool(name) for name in denied))
            finally:
                access_control.leave_request_context(tokens)

    def test_father_and_mother_receive_parental_greetings(self):
        agent = object.__new__(FridayAgent)
        with patch("app.agent.get_memory_provider_access_enabled", return_value=False):
            for role, expected in (
                ("FATHER", "සුබ දවසක් තාත්තා"),
                ("MOTHER", "සුබ දවසක් අම්මා"),
            ):
                tokens = access_control.enter_request_context(role)
                try:
                    self.assertIn(expected, agent._system_instruction())
                finally:
                    access_control.leave_request_context(tokens)

    def test_speaker_token_is_signed_not_client_role_input(self):
        secret = "server-side-api-token-with-at-least-32-characters"
        self.assertIsNone(access_control.verify_speaker_token("MASTER", secret))
        token = access_control.issue_speaker_token(
            "profile-1", "MASTER", secret, now=1000
        )
        self.assertEqual(
            access_control.verify_speaker_token(token, secret, now=1001),
            {"profile_id": "profile-1", "role": "MASTER"},
        )
        self.assertIsNone(
            access_control.verify_speaker_token(token + "x", secret, now=1001)
        )
        self.assertIsNone(
            access_control.verify_speaker_token(token, "different-server-secret", now=1001)
        )

    def test_speaker_token_signer_is_separate_from_client_api_token(self):
        api_token = "client-visible-api-token-with-32-characters"
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(access_control, "SPEAKER_KEY_FILE", Path(directory) / "signer.key"),
                patch.object(access_control, "_speaker_secret_cache", None),
                patch.dict("os.environ", {"FRIDAY_SPEAKER_TOKEN_SECRET": ""}),
            ):
                signer = access_control.load_speaker_signing_secret()
                token = access_control.issue_speaker_token("p", "SISTER", signer)
                self.assertIsNotNone(access_control.verify_speaker_token(token, signer))
                self.assertIsNone(access_control.verify_speaker_token(token, api_token))

    def test_master_passphrase_is_hashed_and_challenge_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            auth_file = Path(directory) / "master_auth.json"
            with patch.object(access_control, "MASTER_AUTH_FILE", auth_file):
                access_control._FAILED_ATTEMPTS.clear()
                access_control._LOCKED_UNTIL = 0.0
                access_control.configure_master_passphrase("a-long-test-passphrase")
                stored = json.loads(auth_file.read_text(encoding="utf-8"))
                self.assertNotIn("a-long-test-passphrase", auth_file.read_text(encoding="utf-8"))
                self.assertEqual(stored["algorithm"], "pbkdf2_sha256")
                self.assertFalse(access_control.verify_master_passphrase("wrong-secret"))
                self.assertTrue(access_control.verify_master_passphrase("a-long-test-passphrase"))
                with self.assertRaises(access_control.MasterAuthError):
                    access_control.configure_master_passphrase("another-long-secret")
            access_control._FAILED_ATTEMPTS.clear()
            access_control._LOCKED_UNTIL = 0.0

    def test_unknown_speaker_falls_back_to_guest(self):
        with tempfile.TemporaryDirectory() as directory:
            store = VoiceProfileStore(Path(directory) / "profiles.dat")
            audio = np.zeros(24_000, dtype="<i2").tobytes()
            buffer = __import__("io").BytesIO()
            with wave.open(buffer, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16_000)
                wav.writeframes(audio)
            speaker = store.identify(buffer.getvalue())
        self.assertEqual(speaker["role"], "GUEST")
        self.assertIsNone(speaker["profile_id"])

    def test_profile_listing_never_returns_embeddings(self):
        with tempfile.TemporaryDirectory() as directory:
            store = VoiceProfileStore(Path(directory) / "profiles.dat")
            sample = b"sample"
            with (
                patch("app.voice_profiles._decode_wav", return_value=np.ones(24_000)),
                patch("app.voice_profiles.voice_id._embedding", return_value=np.ones(4)),
                patch("app.voice_profiles._acoustic_features", return_value=(150.0, [500.0, 1500.0, 2500.0])),
                patch("app.voice_profiles.voice_id._protect", side_effect=lambda value: value),
                patch("app.voice_profiles.voice_id._unprotect", side_effect=lambda value: value),
            ):
                store.enroll("Family member", "SISTER", [sample, sample])
                profiles = store.list_metadata()
            self.assertEqual(profiles[0]["role"], "SISTER")
            self.assertNotIn("embeddings", profiles[0])
            self.assertNotIn("pitch_hz", profiles[0])

    def test_embedding_thresholds_and_secondary_acoustic_sanity_check(self):
        self.assertEqual(_threshold_for("SISTER"), 0.75)
        self.assertEqual(_threshold_for("MASTER"), 0.82)
        self.assertFalse(_acoustic_sanity_check(
            150.0, [1800.0, 2500.0, 3500.0], 150.0, [500.0, 1400.0, 2400.0]
        ))

    def test_dangerous_tool_requires_master_passphrase_before_guard(self):
        class FakeBridge:
            def __init__(self):
                self.messages = []

            def write_log(self, message):
                self.messages.append(message)

        agent = object.__new__(FridayAgent)
        agent.bridge = FakeBridge()
        agent._registry_lock = threading.RLock()
        agent.actions = None
        agent.plugins = None
        tokens = access_control.enter_request_context("MASTER", None)
        try:
            with (
                patch.object(security, "tier_for", return_value=security.TIER_DANGEROUS),
                patch.object(security, "guard") as guard,
                patch.object(security, "audit"),
            ):
                response = agent._dispatch("security_scanner", {})
            self.assertIn("AWAITING_MASTER_AUTH", response)
            guard.assert_not_called()
        finally:
            access_control.leave_request_context(tokens)

    def test_mother_home_control_needs_master_passphrase_for_dangerous_action(self):
        from types import SimpleNamespace

        class FakeBridge:
            def write_log(self, _message):
                pass

        agent = object.__new__(FridayAgent)
        agent.bridge = FakeBridge()
        agent._registry_lock = threading.RLock()
        agent.actions = SimpleNamespace(has=lambda name: name == "iot_control")
        agent.plugins = SimpleNamespace(has=lambda _name: False)
        tokens = access_control.enter_request_context("MOTHER", "master secret")
        try:
            with (
                patch.object(security, "tier_for", return_value=security.TIER_DANGEROUS),
                patch.object(access_control, "verify_master_passphrase", return_value=True),
                patch.object(security, "guard", return_value="confirmation pending") as guard,
            ):
                self.assertEqual(agent._dispatch("iot_control", {}), "confirmation pending")
                guard.assert_called_once()
        finally:
            access_control.leave_request_context(tokens)


class ToolMakerGateTests(unittest.TestCase):
    def test_static_validation_rejects_system_network_and_protected_imports(self):
        ToolMaker._validate(_VALID_TOOL)
        for source in (
            _VALID_TOOL.replace("from app.modules.base import BaseModule",
                                "import socket\nfrom app.modules.base import BaseModule"),
            _VALID_TOOL.replace("from app.modules.base import BaseModule",
                                "from subprocess import run\nfrom app.modules.base import BaseModule"),
            _VALID_TOOL.replace("from app.modules.base import BaseModule",
                                "from core.security import guard\nfrom app.modules.base import BaseModule"),
        ):
            with self.subTest(source=source[:60]), self.assertRaises(ValueError):
                ToolMaker._validate(source)

    def test_core_security_files_remain_protected_from_writes(self):
        from core.path_guard import check, protected_reason

        for filename in (
            "security.py", "confirm.py", "path_guard.py", "sandbox.py",
        ):
            target = tool_maker.PROJECT_ROOT / "core" / filename
            with self.subTest(filename=filename):
                self.assertIsNotNone(protected_reason(target))
                with self.assertRaises(PermissionError):
                    check(target)
        with self.assertRaises(PermissionError):
            access_control.assert_safe_write_target(
                tool_maker.PROJECT_ROOT / "config" / "api_keys.json"
            )

    def test_analysis_staging_and_manifest_are_review_only(self):
        outputs = iter((_SPECIFICATION, _VALID_TOOL))
        maker = ToolMaker(lambda _prompt: next(outputs))
        with tempfile.TemporaryDirectory(dir=tool_maker.PROJECT_ROOT) as directory:
            root = Path(directory)
            with (
                patch("app.tool_maker.STAGING_DIR", root / "staging"),
                patch("app.tool_maker.MANIFEST_FILE", root / "tool_manifest.json"),
            ):
                staged = maker.stage("Find a useful capability")
                manifest = json.loads((root / "tool_manifest.json").read_text(encoding="utf-8"))
                self.assertTrue(staged.is_file())
                self.assertEqual(manifest["tools"][0]["status"], "staged")
                self.assertEqual(manifest["tools"][0]["validation"]["unit_tests"], "passed")
                self.assertEqual(
                    manifest["tools"][0]["validation"]["sandbox_execution"],
                    "passed",
                )
                self.assertEqual(
                    manifest["tools"][0]["validation"]["deployment"],
                    "disabled_pending_os_sandbox",
                )

    def test_static_repair_is_limited_to_three_retries(self):
        outputs = iter((_SPECIFICATION, "bad syntax", "bad syntax", "bad syntax", "bad syntax"))
        calls = []

        def generate(prompt):
            calls.append(prompt)
            return next(outputs)

        maker = ToolMaker(generate)
        with tempfile.TemporaryDirectory(dir=tool_maker.PROJECT_ROOT) as directory:
            with (
                patch("app.tool_maker.STAGING_DIR", Path(directory) / "staging"),
                patch("app.tool_maker.MANIFEST_FILE", Path(directory) / "tool_manifest.json"),
            ):
                with self.assertRaisesRegex(ValueError, "3 repair attempts"):
                    maker.stage("A safe local calculation")
        self.assertEqual(len(calls), 5)

    def test_sandbox_repair_is_limited_to_three_retries(self):
        outputs = iter((_SPECIFICATION, _VALID_TOOL, _VALID_TOOL, _VALID_TOOL, _VALID_TOOL))
        calls = []

        def generate(prompt):
            calls.append(prompt)
            return next(outputs)

        maker = ToolMaker(generate)
        with tempfile.TemporaryDirectory(dir=tool_maker.PROJECT_ROOT) as directory:
            with (
                patch("app.tool_maker.STAGING_DIR", Path(directory) / "staging"),
                patch("app.tool_maker.MANIFEST_FILE", Path(directory) / "tool_manifest.json"),
                patch.object(
                    ToolMaker, "_sandbox_test",
                    side_effect=RuntimeError("test sandbox failure"),
                ),
            ):
                with self.assertRaisesRegex(ValueError, "3 repair attempts"):
                    maker.stage("A safe local calculation")
        self.assertEqual(len(calls), 5)


class FamilyApiRouteTests(unittest.TestCase):
    def test_headless_voice_and_master_routes_exist(self):
        from app.main import app

        paths = {route.path for route in app.routes}
        self.assertTrue({
            "/api/master/setup",
            "/api/master/verify",
            "/api/voice/enroll",
            "/api/voice/profiles",
            "/api/voice/profiles/{profile_id}",
            "/api/voice/verify",
            "/api/succession/heartbeat",
            "/api/succession/enroll",
            "/api/succession/claim",
            "/api/succession/guardian-approve",
            "/api/succession/reclaim",
            "/api/succession/status",
            "/api/telegram-webhook",
        }.issubset(paths))

    def test_admin_helper_requires_master_voice_token(self):
        from types import SimpleNamespace

        from fastapi import HTTPException

        from app.main import _require_master_speaker

        agent = SimpleNamespace(
            _verified_speaker=lambda token: (
                {"role": "MASTER"} if token == "valid-master-token" else None
            )
        )
        request = SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(agent=agent))
        )
        with patch("app.main.security.audit"):
            with self.assertRaises(HTTPException):
                _require_master_speaker(request, "mother-token")
        _require_master_speaker(request, "valid-master-token")


if __name__ == "__main__":
    unittest.main()
