"""Online TLS acceptance check; exit nonzero on acceptance or inconclusive failure."""
import concurrent.futures
import json
import socket
import ssl
import sys
import urllib.error
import urllib.request

import certifi
import httpx
from ca_guard import verify

BAD = [f"{kind}-g3{generation}{cross}-emudhra.we-have-a-problem.com"
       for kind in ("dv", "ov", "ev") for generation in ("b", "c") for cross in ("", "-aaa")]
GOOD = ["github.com", "pypi.org"]


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """Evaluate the original host's handshake, not a redirect target's trust."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def verification_error(error):
    pending, seen = [error], set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, ssl.SSLCertVerificationError) and getattr(current, "verify_code", None) in (2, 18, 19, 20, 21, 23, 27, 28):
            return current.verify_code
        for attr in ("reason", "__cause__", "__context__", "certificate_error"):
            try:
                nested = getattr(current, attr, None)
            except Exception:
                continue
            if isinstance(nested, BaseException):
                pending.append(nested)
        pending.extend(x for x in getattr(current, "args", ()) if isinstance(x, BaseException))
    return None


def check(args):
    mode, host = args
    result = {"mode": mode, "host": host}
    try:
        if mode == "httpx":
            with httpx.Client(timeout=12) as client:
                with client.stream("GET", "https://" + host, follow_redirects=False):
                    pass
        elif mode == "urllib":
            try:
                opener = urllib.request.build_opener(NoRedirect())
                with opener.open("https://" + host, timeout=12):
                    pass
            except urllib.error.HTTPError:
                pass  # An HTTP error response still proves successful TLS.
        else:
            if mode == "capath-only":
                context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
                context.load_verify_locations(capath="/etc/ssl/certs")
            elif mode == "certifi":
                context = ssl.create_default_context(cafile=certifi.where())
            else:
                context = ssl.create_default_context()
            with socket.create_connection((host, 443), timeout=12) as raw:
                with context.wrap_socket(raw, server_hostname=host):
                    pass
        result.update(outcome="accepted", ok=host in GOOD)
    except Exception as error:
        code = verification_error(error)
        result.update(outcome="trust_rejected" if code is not None else "inconclusive",
                      verify_code=code, error=type(error).__name__, ok=host in BAD and code is not None)
    return result


def main():
    verify()
    modes = ("ssl-default", "capath-only", "certifi", "urllib", "httpx")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(check, [(mode, host) for mode in modes for host in BAD + GOOD]))
    ok = len(results) == 70 and all(r["ok"] for r in results)
    sys.stdout.write(json.dumps({"ok": ok, "results": results}) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
