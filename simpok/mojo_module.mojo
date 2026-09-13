from std.python import PythonObject
from std.python import Python
from std.python.bindings import PythonModuleBuilder
from _core_mojo import get_device, run_tdgl, run_chc
from std.os import abort


@export
def PyInit_mojo_module() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("mojo_module")
        m.def_function[_get_device](
            "_get_device", docstring="Probe the default accelerator"
        )
        m.def_function[_tdgl_run](
            "_tdgl_run", docstring="Evolve a 2d TDGL field in place"
        )
        m.def_function[_chc_run](
            "_chc_run", docstring="Evolve a 2d CHC field in place"
        )
        return m.finalize()
    except e:
        abort(String("error creating Python Mojo module: ", e))


def _get_device() raises -> PythonObject:
    var available, name, reason = get_device()
    return Python.tuple(
        PythonObject(available), PythonObject(name), PythonObject(reason)
    )


def _tdgl_run(
    field: PythonObject, snaps: PythonObject, params: PythonObject
) raises -> PythonObject:
    var backend = run_tdgl(
        Int(py=field.ctypes.data),
        Int(py=snaps.ctypes.data),
        Int(py=params[0]),
        Int(py=params[1]),
        Int(py=params[2]),
        Float64(py=params[3]),
        Float64(py=params[4]),
        Float64(py=params[5]),
        Float64(py=params[6]),
        Int(py=params[7]),
        Int(py=params[8]),
        Bool(py=params[9]),
        Int(py=params[10]),
    )
    return PythonObject(backend)


def _chc_run(
    field: PythonObject, snaps: PythonObject, params: PythonObject
) raises -> PythonObject:
    var backend = run_chc(
        Int(py=field.ctypes.data),
        Int(py=snaps.ctypes.data),
        Int(py=params[0]),
        Int(py=params[1]),
        Int(py=params[2]),
        Float64(py=params[3]),
        Float64(py=params[4]),
        Float64(py=params[5]),
        Int(py=params[6]),
        Int(py=params[7]),
        Bool(py=params[8]),
        Int(py=params[9]),
    )
    return PythonObject(backend)
