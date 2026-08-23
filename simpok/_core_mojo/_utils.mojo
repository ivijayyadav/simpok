"""
Implemented struct and function common to all the all files
"""

from std.sys import has_accelerator
from max.gpu.host import DeviceContext


def get_device() -> Tuple[Bool, String, String]:
    comptime if not has_accelerator():
        return (
            False, String(""), String("no supported accelerator at build time")
        )
    else:
        try:
            if DeviceContext.number_of_devices() == 0:
                return (False, String(""), String("no accelerator detected"))
            var ctx = DeviceContext()
            return (True, ctx.name(), String(""))
        except e:
            return (False, String(""), String(e))
