import mojo.importer # noqa: F401
from .mojo_module import _get_device

def get_device():
    found, name = _get_device()
    return (found, name)