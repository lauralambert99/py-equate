import importlib
import sys


def test_fresh_import_succeeds():
    for name in [m for m in sys.modules if m.startswith("pyequate")]:
        del sys.modules[name]
    pq = importlib.import_module("pyequate")
    for name in pq.__all__:
        assert hasattr(pq, name), name
