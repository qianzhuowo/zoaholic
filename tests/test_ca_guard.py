"""Offline CA guard tests; fixtures contain public certificates, never keys."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ca_guard", ROOT / "scripts/ca_guard.py")
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)
FIXTURE = ROOT / "tests/fixtures/emudhra-public-roots.pem"


class CAGuardTests(unittest.TestCase):
    def test_fixture_fingerprints(self):
        self.assertEqual(guard.fingerprints(FIXTURE), guard.BLOCKED)

    def test_pem_without_readable_ca_name_is_rejected(self):
        self.assertNotIn("emSign ECC Root", FIXTURE.read_text())
        with self.assertRaises(ValueError):
            guard.assert_clean_file(FIXTURE)

    def test_renamed_ca_is_disabled_and_operation_is_idempotent(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "renamed.crt").write_bytes(FIXTURE.read_bytes())
            result, changed = guard.disabled_config("# comment\nrenamed.crt\n", root)
            self.assertEqual(changed, 1)
            self.assertEqual(result, "# comment\n!renamed.crt\n")
            self.assertEqual(guard.disabled_config(result, root), (result, 0))

    def test_hash_directory_is_independent_of_clean_bundle(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "abcd1234.0").write_bytes(FIXTURE.read_bytes())
            with self.assertRaises(ValueError):
                guard.assert_clean_directory(root)

    def test_empty_directory_and_malformed_bundle_fail(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with self.assertRaises(ValueError):
                guard.assert_clean_directory(root)
            p = root / "bad.pem"
            p.write_text("not a certificate")
            with self.assertRaises(ValueError):
                guard.assert_clean_file(p)

    def test_missing_enabled_certificate_fails(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                guard.disabled_config("missing.crt", Path(d))

    def test_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                guard.disabled_config("../outside.crt", Path(d))


if __name__ == "__main__":
    unittest.main()
