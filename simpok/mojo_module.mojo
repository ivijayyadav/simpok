from std.python import PythonObject
from std.python import Python
from std.python.bindings import PythonModuleBuilder
from _core_mojo import __get_device
from std.os import abort

@export
def PyInit_mojo_module() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("mojo_module")
        m.def_function[_get_device]("_get_device", docstring="Check GPU")
        return m.finalize()
    except e:
        abort(String("error creating Python Mojo module:", e))

def _get_device() raises -> PythonObject:
    var found, name = __get_device()
    return Python.tuple(PythonObject(found), PythonObject(name))


    
