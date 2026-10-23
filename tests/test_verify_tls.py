"""Offline regression tests for TLS acceptance classification (no network)."""
import importlib.util
from pathlib import Path
import socket
import ssl
import sys
import unittest
import urllib.error
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
# The acceptance script is normally invoked directly from scripts/.
sys.path.insert(0, str(ROOT / "scripts"))
try:
    spec = importlib.util.spec_from_file_location("verify_tls", ROOT / "scripts/verify_tls.py")
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
finally:
    sys.path.pop(0)


def certificate_error(code):
    error = ssl.SSLCertVerificationError(1, "certificate verify failed")
    error.verify_code = code
    return error


class TLSAcceptanceTests(unittest.TestCase):
    def test_wrapped_trust_error_is_recognized(self):
        self.assertEqual(probe.verification_error(urllib.error.URLError(certificate_error(20))), 20)

    def test_unrelated_errors_are_not_evidence_of_distrust(self):
        for error in (TimeoutError(), socket.gaierror(), urllib.error.URLError("offline"),
                      certificate_error(10), certificate_error(62)):
            with self.subTest(error=repr(error)):
                self.assertIsNone(probe.verification_error(error))

    def test_exception_cycle_terminates(self):
        error = RuntimeError("cycle")
        error.__cause__ = error
        self.assertIsNone(probe.verification_error(error))

    def test_redirects_are_not_followed(self):
        self.assertIsNone(probe.NoRedirect().redirect_request(None, None, 302, "Found", {}, "https://other.invalid"))

    def test_http_response_including_redirect_means_original_tls_was_accepted(self):
        for status in (302, 403, 500):
            with self.subTest(status=status), patch.object(probe.urllib.request, "build_opener") as factory:
                factory.return_value.open.side_effect = urllib.error.HTTPError(
                    "https://" + probe.BAD[0], status, "response", {}, None)
                result = probe.check(("urllib", probe.BAD[0]))
                self.assertIsInstance(factory.call_args.args[0], probe.NoRedirect)
                self.assertEqual(result["outcome"], "accepted")
                self.assertFalse(result["ok"])

    def test_network_failure_is_inconclusive_and_trust_rejection_is_expected_only_for_bad_hosts(self):
        for host in (probe.BAD[0], probe.GOOD[0]):
            for error, outcome, ok in (
                (TimeoutError(), "inconclusive", False),
                (urllib.error.URLError(certificate_error(20)), "trust_rejected", host in probe.BAD),
            ):
                with self.subTest(host=host, outcome=outcome), patch.object(probe.urllib.request, "build_opener") as factory:
                    factory.return_value.open.side_effect = error
                    result = probe.check(("urllib", host))
                    self.assertEqual((result["outcome"], result["ok"]), (outcome, ok))

    def test_production_and_debug_images_have_final_ca_gate(self):
        for name in ("Dockerfile", "Dockerfile.debug"):
            text = (ROOT / name).read_text(encoding="utf-8")
            self.assertGreater(text.index("RUN python scripts/ca_guard.py --apply-system --verify"),
                               text.rindex("COPY "))


if __name__ == "__main__":
    unittest.main()
