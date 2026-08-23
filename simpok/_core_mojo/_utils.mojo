from std.math import sqrt, log, cos
from std.sys import has_accelerator
from max.gpu.host import DeviceContext

comptime DEVICE_AUTO = 0
comptime DEVICE_ACCEL = 1
comptime DEVICE_CPU = 2

comptime TPB = 16
comptime GOLDEN = 0x9E3779B97F4A7C15
comptime TWO_PI = 6.283185307179586


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


def mix(x: UInt64) -> UInt64:
    var z = x + GOLDEN
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def gaussian[dt: DType](
    seed: UInt64, step: UInt64, site: UInt64
) -> Scalar[dt]:
    comptime assert dt.is_floating_point(), "dt must be a float type"
    comptime shift: UInt64 = 40 if dt == DType.float32 else 11
    comptime scale = (
        1.0 / 16777216.0 if dt == DType.float32 else 1.1102230246251565e-16
    )
    var h1 = mix(seed ^ mix(step * GOLDEN ^ site))
    var h2 = mix(h1)
    var u1 = (Scalar[dt](h1 >> shift) + 1.0) * Scalar[dt](scale)
    var u2 = Scalar[dt](h2 >> shift) * Scalar[dt](scale)
    var phase = cos(Float32(TWO_PI) * Float32(u2))
    return sqrt(-2.0 * log(u1)) * Scalar[dt](phase)
