"""
Implemented struct and function common to all the all files
"""

from std.sys import has_accelerator
from max.gpu.host import DeviceContext

def __get_device() -> Tuple[Bool, String]:
    comptime if not has_accelerator():
        print("No GPU found.\n")
        return (False, "Not Found")
    else:
        try:
            var ctx = DeviceContext()
            var name = ctx.name()
            return (True, name)
        except e:
            print("GPU present but unavailable: ", String(e), "\n")
            return (False, "Not Found")