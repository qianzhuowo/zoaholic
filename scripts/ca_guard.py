"""Build-time CA distrust and offline verification for Debian-based images.

Run --apply-system before --verify. Never disables TLS validation or adds roots.
Public certificate fingerprints: https://emudhra.we-have-a-problem.com/roots.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import logging
from pathlib import Path
import re
import ssl
import subprocess

logger = logging.getLogger(__name__)
BLOCKED = frozenset({
    "86a1ecba089c4a8d3bbe2734c612ba341d813e043cf9e8a862cd5c57a36bbe6b",
    "d7a7a0fb5d7e2731d771e9484ebcdef71d5f0c3e0a2948782bc83ee0ea699ef4",
    "7dd78d5f4f13459a83dff9abbba62edbaf6f2d102bf257fd712f4d9f2746ed8d",
    "f2edaaff3d9b701fb4d14c9f285f8c0bad90fc3bcc1fa07d47f4718edaba34a5",
})
PEM = re.compile(r"-----BEGIN CERTIFICATE-----.*?-----END CERTIFICATE-----", re.S)
HASH_NAME = re.compile(r"[0-9a-fA-F]{8}\.\d+$")


def fingerprints(path: Path) -> set[str]:
    text = path.read_text(encoding="ascii")
    blocks = PEM.findall(text)
    if not blocks:
        raise ValueError(f"No PEM certificates in {path}")
    # ssl parses the bundle as well: malformed ASN.1 must not pass based on hashes.
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.load_verify_locations(cafile=str(path))
    return {hashlib.sha256(ssl.PEM_cert_to_DER_cert(block)).hexdigest() for block in blocks}


def assert_clean_file(path: Path) -> int:
    values = fingerprints(path)
    bad = values & BLOCKED
    if bad:
        raise ValueError(f"Blocked CA in {path}: {sorted(bad)}")
    return len(values)


def assert_clean_directory(directory: Path) -> int:
    if not directory.is_dir():
        raise ValueError(f"Missing CA directory: {directory}")
    entries = [p for p in directory.iterdir() if HASH_NAME.fullmatch(p.name)]
    if not entries:
        raise ValueError(f"Empty CA hash directory: {directory}")
    for entry in entries:
        # A dangling symlink raises instead of silently skipping stale entries.
        assert_clean_file(entry)
    return len(entries)


def disabled_config(text: str, certificate_root: Path) -> tuple[str, int]:
    lines, changed = [], 0
    for line in text.splitlines():
        name = line.strip()
        if name and not name.startswith(("#", "!")):
            path = (certificate_root / name).resolve()
            if not path.is_relative_to(certificate_root.resolve()):
                raise ValueError("CA configuration path escapes certificate root")
            if fingerprints(path) & BLOCKED:
                line = "!" + name
                changed += 1
        lines.append(line)
    return "\n".join(lines) + "\n", changed


def apply_system() -> None:
    config = Path("/etc/ca-certificates.conf")
    text, changed = disabled_config(config.read_text(), Path("/usr/share/ca-certificates"))
    config.write_text(text)
    # Also remove stale OpenSSL hash links, not just rebuild the concatenated file.
    subprocess.run(["update-ca-certificates", "--fresh"], check=True)
    logger.info("Disabled %d configured CA certificates", changed)


def verify() -> None:
    import certifi

    version = importlib.metadata.version("certifi")
    parts = tuple(int(p) for p in version.split("."))
    if parts < (2026, 10, 19):
        raise ValueError(f"certifi too old: {version}")
    paths = ssl.get_default_verify_paths()
    bundles = {Path("/etc/ssl/certs/ca-certificates.crt"), Path(certifi.where())}
    if paths.cafile:
        bundles.add(Path(paths.cafile))
    directories = {Path("/etc/ssl/certs")}
    if paths.capath:
        directories.update(Path(p) for p in paths.capath.split(":") if p)
    for bundle in sorted(bundles):
        logger.info("Verified bundle %s (%d certificates)", bundle, assert_clean_file(bundle))
    for directory in sorted(directories):
        logger.info("Verified capath %s (%d hash entries)", directory, assert_clean_directory(directory))
    logger.info("CA fingerprint checks passed; explicit custom CA and network paths require separate validation")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply-system", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if not (args.apply_system or args.verify):
        parser.error("Choose --apply-system and/or --verify")
    logging.basicConfig(level=logging.INFO)
    if args.apply_system:
        apply_system()
    if args.verify:
        verify()


if __name__ == "__main__":
    main()
