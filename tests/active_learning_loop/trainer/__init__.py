import pytest

try:
    import flare_pp  # noqa
except ImportError:
    pytest.skip("Skipping FLARE tests:  optional FLARE dependencies not installed.", allow_module_level=True)
