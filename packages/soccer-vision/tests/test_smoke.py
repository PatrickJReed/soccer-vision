"""Smoke tests — package imports and version is set."""

import soccer_vision


def test_package_imports() -> None:
    assert soccer_vision is not None


def test_version_is_set() -> None:
    assert soccer_vision.__version__ == "0.1.0"
