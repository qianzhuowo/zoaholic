import os
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("ZOAHOLIC_SKIP_DIALECT_ROUTE_REGISTRATION", "1")


try:
    import h2.exceptions  # type: ignore # noqa: F401
except ModuleNotFoundError:
    h2_module = types.ModuleType("h2")
    exceptions_module = types.ModuleType("h2.exceptions")

    class H2Error(Exception):
        pass

    exceptions_module.ProtocolError = H2Error
    exceptions_module.StreamClosedError = H2Error
    exceptions_module.ConnectionError = H2Error
    h2_module.exceptions = exceptions_module

    sys.modules["h2"] = h2_module
    sys.modules["h2.exceptions"] = exceptions_module
