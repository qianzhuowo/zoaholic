"""Restore ONLY a recorded CA build-file deployment; never restart services.

Run on the server as root with a root-only backup directory. Refuses to overwrite
files edited since deployment. This is not a database or container rollback.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

ALLOWED = frozenset({"Dockerfile", ".dockerignore", "scripts/ca_guard.py", "scripts/verify_tls.py",
                     "tests/test_ca_guard.py", "tests/fixtures/emudhra-public-roots.pem",
                     "docs/docker-ca-security.md"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backup", type=Path)
    parser.add_argument("--apply", action="store_true", help="Without this flag only checks preconditions")
    args = parser.parse_args()
    root = Path("/opt/Zoaholic").resolve()
    manifest = json.loads((args.backup / "file-deployment.json").read_text())
    for item in manifest:
        relative = item["path"]
        if relative not in ALLOWED:
            raise ValueError("Unexpected file in rollback manifest")
        path = root / relative
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("Unsafe destination")
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["deployed_sha256"]:
            raise ValueError(f"Refusing to overwrite later changes: {relative}")
        if item["existed"]:
            old = args.backup / "files-before" / relative
            if hashlib.sha256(old.read_bytes()).hexdigest() != item["old_sha256"]:
                raise ValueError(f"Backup checksum mismatch: {relative}")
    if args.apply:
        for item in manifest:
            path = root / item["path"]
            if item["existed"]:
                temp = path.with_name(path.name + ".ca-restore.tmp")
                temp.write_bytes((args.backup / "files-before" / item["path"]).read_bytes())
                os.chmod(temp, item["old_mode"])
                os.replace(temp, path)
            else:
                path.unlink()


if __name__ == "__main__":
    main()
