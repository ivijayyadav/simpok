from typing import NamedTuple


class Device(NamedTuple):
    available: bool
    name: str | None
    reason: str | None


def get_device() -> Device:
    import mojo.importer  # noqa: F401

    from .mojo_module import _get_device

    available, name, reason = _get_device()
    return Device(available, name or None, reason or None)
