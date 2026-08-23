from std.python import PythonObject
from std.python import Python
from std.python.bindings import PythonModuleBuilder
from _core_mojo import get_device
from std.os import abort


@export
def PyInit_mojo_module() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("mojo_module")
        m.def_function[_get_device](
            "_get_device", docstring="Probe the default accelerator"
        )
        return m.finalize()
    except e:
        abort(String("error creating Python Mojo module: ", e))


def _get_device() raises -> PythonObject:
    var available, name, reason = get_device()
    return Python.tuple(
        PythonObject(available), PythonObject(name), PythonObject(reason)
    )
