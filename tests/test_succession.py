from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.succession import SuccessionManager


class SuccessionLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "succession_state.json"
        self.manager = SuccessionManager(
            self.path, inactivity_hours=1, grace_hours=1
        )
        self.audit = patch("app.succession.security.audit")
        self.audit.start()
        self.addCleanup(self.audit.stop)

    def test_master_heartbeat_inactivity_claim_parent_approval_and_reclaim(self):
        manager = self.manager
        manager.heartbeat("MASTER", "master-profile", now=100)
        manager.heartbeat("BROTHER", "brother-profile", now=100)
        manager.enroll_emergency_credential(
            "BROTHER", "brother-profile", "brother emergency phrase"
        )

        before_deadline = manager.status(now=3699)
        self.assertEqual(before_deadline["phase"], "ACTIVE")
        eligible = manager.status(now=3700)
        self.assertEqual(eligible["phase"], "ELIGIBLE_FOR_CLAIM")
        self.assertTrue(eligible["master_inactive"])
        self.assertTrue(eligible["brother_eligible"])

        pending = manager.claim(
            "BROTHER",
            "brother-profile",
            "brother emergency phrase",
            now=3700,
        )
        self.assertEqual(pending["status"], "AWAITING_BROTHER_GUARDIAN_APPROVAL")
        with self.assertRaises(PermissionError):
            manager.approve_claim(
                pending["claim_id"], "BROTHER", "brother-profile", now=3701
            )

        promoted = manager.approve_claim(
            pending["claim_id"], "FATHER", "father-profile", now=3701
        )
        self.assertEqual(promoted["status"], "TEMPORARY_MASTER_BROTHER")
        self.assertEqual(
            manager.effective_role("brother-profile", "BROTHER"), "MASTER"
        )

        manager.reclaim("master-profile", now=3702)
        self.assertEqual(
            manager.effective_role("brother-profile", "BROTHER"), "BROTHER"
        )
        self.assertEqual(manager.status(now=3702)["phase"], "ACTIVE")

    def test_sister_fallback_requires_master_and_brother_inactivity(self):
        manager = self.manager
        manager.heartbeat("MASTER", "master-profile", now=100)
        manager.heartbeat("BROTHER", "brother-profile", now=100)
        manager.heartbeat("SISTER", "sister-profile", now=100)
        manager.enroll_emergency_credential(
            "SISTER", "sister-profile", "sister emergency phrase"
        )

        status = manager.status(now=7299)
        self.assertFalse(status["sister_eligible"])
        status = manager.status(now=7300)
        self.assertTrue(status["master_inactive"])
        self.assertTrue(status["brother_inactive"])
        self.assertTrue(status["sister_eligible"])
        self.assertEqual(status["phase"], "SISTER_ELIGIBLE")

        pending = manager.claim(
            "SISTER",
            "sister-profile",
            "sister emergency phrase",
            now=7300,
        )
        promoted = manager.approve_claim(
            pending["claim_id"], "MOTHER", "mother-profile", now=7301
        )
        self.assertEqual(promoted["status"], "TEMPORARY_MASTER_SISTER")
        self.assertEqual(
            manager.effective_role("sister-profile", "SISTER"), "MASTER"
        )

    def test_sanctuary_lockdown_when_successors_do_not_claim(self):
        manager = self.manager
        manager.heartbeat("MASTER", "master-profile", now=100)
        self.assertEqual(manager.status(now=3700)["phase"], "ELIGIBLE_FOR_CLAIM")
        self.assertEqual(manager.status(now=7300)["phase"], "SISTER_ELIGIBLE")
        status = manager.status(now=10900)
        self.assertEqual(status["phase"], "SANCTUARY_LOCKDOWN")
        self.assertTrue(manager.is_locked_down())

    def test_manual_emergency_and_wrong_passphrase_fail_closed(self):
        manager = self.manager
        manager.enroll_emergency_credential(
            "BROTHER", "brother-profile", "brother emergency phrase"
        )
        with self.assertRaises(PermissionError):
            manager.claim(
                "BROTHER",
                "brother-profile",
                "wrong phrase",
                manual_emergency=True,
                master_profile_id="master-profile",
                now=100,
            )
        self.assertFalse(manager.is_locked_down())
        claim = manager.claim(
            "BROTHER",
            "brother-profile",
            "brother emergency phrase",
            manual_emergency=True,
            master_profile_id="master-profile",
            now=101,
        )
        self.assertTrue(claim["claim_id"])

    def test_corrupt_state_fails_closed(self):
        self.path.write_text("{", encoding="utf-8")
        self.assertTrue(self.manager.is_locked_down())
        with self.assertRaises(RuntimeError):
            self.manager.status()


if __name__ == "__main__":
    unittest.main()
