from std.math import ceildiv, sqrt
from std.memory import unsafe_memcpy
from std.memory.alloc import alloc, Layout
from std.sys import has_accelerator, size_of
from std.sys.info import has_apple_gpu_accelerator
from std.gpu import global_idx
from max.gpu.host import DeviceContext
from layout import TileTensor, TensorLayout, row_major

from ._utils import gaussian, TPB, DEVICE_ACCEL, DEVICE_CPU


def _mu[dt: DType](
    c: Scalar[dt],
    xp: Scalar[dt],
    xm: Scalar[dt],
    yp: Scalar[dt],
    ym: Scalar[dt],
    inv_dx2: Scalar[dt],
) -> Scalar[dt]:
    return -c + c * c * c - (xp + xm + yp + ym - 4.0 * c) * inv_dx2


def _update[dt: DType](
    c: Scalar[dt],
    xp: Scalar[dt],
    xm: Scalar[dt],
    yp: Scalar[dt],
    ym: Scalar[dt],
    xpp: Scalar[dt],
    xmm: Scalar[dt],
    ypp: Scalar[dt],
    ymm: Scalar[dt],
    pp: Scalar[dt],
    pm: Scalar[dt],
    mp: Scalar[dt],
    mm: Scalar[dt],
    step_dt: Scalar[dt],
    inv_dx2: Scalar[dt],
) -> Scalar[dt]:
    var mc = _mu[dt](c, xp, xm, yp, ym, inv_dx2)
    var mxp = _mu[dt](xp, xpp, c, pp, pm, inv_dx2)
    var mxm = _mu[dt](xm, c, xmm, mp, mm, inv_dx2)
    var myp = _mu[dt](yp, pp, mp, ypp, c, inv_dx2)
    var mym = _mu[dt](ym, pm, mm, c, ymm, inv_dx2)
    return c + step_dt * (mxp + mxm + myp + mym - 4.0 * mc) * inv_dx2


def _divnoise[dt: DType](
    n: Int, i: Int, j: Int, im: Int, jm: Int, seed: UInt64, step: UInt64
) -> Scalar[dt]:
    var here = UInt64(2 * (i * n + j))
    return (
        gaussian[dt](seed, step, here)
        - gaussian[dt](seed, step, UInt64(2 * (i * n + jm)))
        + gaussian[dt](seed, step, here + 1)
        - gaussian[dt](seed, step, UInt64(2 * (im * n + j) + 1))
    )


def _step_kernel[dt: DType, LT: TensorLayout](
    psi: TileTensor[dt, LT, MutAnyOrigin],
    nxt: TileTensor[dt, LT, MutAnyOrigin],
    n: Int32,
    step_dt: Scalar[dt],
    inv_dx2: Scalar[dt],
    noise_amp: Scalar[dt],
    seed: UInt64,
    step: UInt64,
):
    comptime assert psi.flat_rank == 2, "field must be 2d"
    comptime assert nxt.flat_rank == 2, "field must be 2d"
    var nn = Int(n)
    var j = Int(global_idx.x)
    var i = Int(global_idx.y)
    if i >= nn or j >= nn:
        return

    var ip = i + 1 if i + 1 < nn else i + 1 - nn
    var im = i - 1 if i >= 1 else i - 1 + nn
    var jp = j + 1 if j + 1 < nn else j + 1 - nn
    var jm = j - 1 if j >= 1 else j - 1 + nn
    var ipp = i + 2 if i + 2 < nn else i + 2 - nn
    var imm = i - 2 if i >= 2 else i - 2 + nn
    var jpp = j + 2 if j + 2 < nn else j + 2 - nn
    var jmm = j - 2 if j >= 2 else j - 2 + nn

    var v = _update[dt](
        rebind[Scalar[dt]](psi[i, j]),
        rebind[Scalar[dt]](psi[ip, j]),
        rebind[Scalar[dt]](psi[im, j]),
        rebind[Scalar[dt]](psi[i, jp]),
        rebind[Scalar[dt]](psi[i, jm]),
        rebind[Scalar[dt]](psi[ipp, j]),
        rebind[Scalar[dt]](psi[imm, j]),
        rebind[Scalar[dt]](psi[i, jpp]),
        rebind[Scalar[dt]](psi[i, jmm]),
        rebind[Scalar[dt]](psi[ip, jp]),
        rebind[Scalar[dt]](psi[ip, jm]),
        rebind[Scalar[dt]](psi[im, jp]),
        rebind[Scalar[dt]](psi[im, jm]),
        step_dt,
        inv_dx2,
    )
    if noise_amp != 0.0:
        v += noise_amp * _divnoise[dt](nn, i, j, im, jm, seed, step)
    nxt[i, j] = rebind[nxt.ElementType](v)


def _step_cpu[dt: DType](
    psi: Pointer[Scalar[dt], MutAnyOrigin],
    nxt: Pointer[Scalar[dt], MutAnyOrigin],
    n: Int,
    step_dt: Scalar[dt],
    inv_dx2: Scalar[dt],
    noise_amp: Scalar[dt],
    seed: UInt64,
    step: UInt64,
):
    for i in range(n):
        var ip = i + 1 if i + 1 < n else i + 1 - n
        var im = i - 1 if i >= 1 else i - 1 + n
        var ipp = i + 2 if i + 2 < n else i + 2 - n
        var imm = i - 2 if i >= 2 else i - 2 + n
        for j in range(n):
            var jp = j + 1 if j + 1 < n else j + 1 - n
            var jm = j - 1 if j >= 1 else j - 1 + n
            var jpp = j + 2 if j + 2 < n else j + 2 - n
            var jmm = j - 2 if j >= 2 else j - 2 + n
            var v = _update[dt](
                psi[unsafe_offset=i * n + j],
                psi[unsafe_offset=ip * n + j],
                psi[unsafe_offset=im * n + j],
                psi[unsafe_offset=i * n + jp],
                psi[unsafe_offset=i * n + jm],
                psi[unsafe_offset=ipp * n + j],
                psi[unsafe_offset=imm * n + j],
                psi[unsafe_offset=i * n + jpp],
                psi[unsafe_offset=i * n + jmm],
                psi[unsafe_offset=ip * n + jp],
                psi[unsafe_offset=ip * n + jm],
                psi[unsafe_offset=im * n + jp],
                psi[unsafe_offset=im * n + jm],
                step_dt,
                inv_dx2,
            )
            if noise_amp != 0.0:
                v += noise_amp * _divnoise[dt](n, i, j, im, jm, seed, step)
            nxt[unsafe_offset=i * n + j] = v


def _run_cpu[dt: DType](
    field: Pointer[Scalar[dt], MutAnyOrigin],
    snaps_addr: Int,
    n: Int,
    nsteps: Int,
    nevery: Int,
    step_dt: Scalar[dt],
    inv_dx2: Scalar[dt],
    noise_amp: Scalar[dt],
    seed: UInt64,
    step0: Int,
) raises -> String:
    comptime esize = size_of[Scalar[dt]]()
    var size = n * n
    var owned_a = alloc(Layout[Scalar[dt]](count=size)).into_managed()
    var owned_b = alloc(Layout[Scalar[dt]](count=size)).into_managed()
    var a = Pointer[Scalar[dt], MutAnyOrigin](
        unsafe_from_address=Int(owned_a.unsafe_ptr())
    )
    var b = Pointer[Scalar[dt], MutAnyOrigin](
        unsafe_from_address=Int(owned_b.unsafe_ptr())
    )
    unsafe_memcpy(dest=a, src=field, count=size)

    var k = 0
    for s in range(1, nsteps + 1):
        _step_cpu[dt](
            a, b, n, step_dt, inv_dx2, noise_amp, seed, UInt64(step0 + s - 1),
        )
        swap(a, b)
        if s % nevery == 0:
            var dst = Pointer[Scalar[dt], MutAnyOrigin](
                unsafe_from_address=snaps_addr + k * size * esize
            )
            unsafe_memcpy(dest=dst, src=a, count=size)
            k += 1

    unsafe_memcpy(dest=field, src=a, count=size)
    _ = owned_a^
    _ = owned_b^
    return String("cpu")


def _run_accel[dt: DType](
    field: Pointer[Scalar[dt], MutAnyOrigin],
    snaps_addr: Int,
    n: Int,
    nsteps: Int,
    nevery: Int,
    step_dt: Scalar[dt],
    inv_dx2: Scalar[dt],
    noise_amp: Scalar[dt],
    seed: UInt64,
    step0: Int,
) raises -> String:
    comptime esize = size_of[Scalar[dt]]()
    var size = n * n
    var ctx = DeviceContext()
    var a = ctx.enqueue_create_buffer[dt](size)
    var b = ctx.enqueue_create_buffer[dt](size)
    ctx.enqueue_copy(dst_buf=a, src_ptr=field)

    var layout = row_major(n, n)
    comptime kern = _step_kernel[dt, type_of(layout)]
    var grid = (ceildiv(n, TPB), ceildiv(n, TPB))

    var k = 0
    for s in range(1, nsteps + 1):
        ctx.enqueue_function[kern](
            TileTensor(a, layout),
            TileTensor(b, layout),
            Int32(n),
            step_dt,
            inv_dx2,
            noise_amp,
            seed,
            UInt64(step0 + s - 1),
            grid_dim=grid,
            block_dim=(TPB, TPB),
        )
        swap(a, b)
        if s % nevery == 0:
            var dst = Pointer[Scalar[dt], MutAnyOrigin](
                unsafe_from_address=snaps_addr + k * size * esize
            )
            ctx.enqueue_copy(dst_ptr=dst, src_buf=a)
            k += 1

    ctx.enqueue_copy(dst_ptr=field, src_buf=a)
    ctx.synchronize()
    return String("accelerator")


def _dispatch[dt: DType](
    field_addr: Int,
    snaps_addr: Int,
    n: Int,
    nsteps: Int,
    nevery: Int,
    step_dt: Float64,
    dx: Float64,
    eps: Float64,
    seed: Int,
    step0: Int,
    device: Int,
) raises -> String:
    var field = Pointer[Scalar[dt], MutAnyOrigin](
        unsafe_from_address=field_addr
    )
    var inv_dx2 = 1.0 / (dx * dx)
    var amp = sqrt(2.0 * eps * step_dt) * inv_dx2 if eps > 0.0 else 0.0
    var sdt = Scalar[dt](step_dt)
    var sidx = Scalar[dt](inv_dx2)
    var samp = Scalar[dt](amp)
    var useed = UInt64(seed)

    comptime no_f64_on_accel = (
        has_apple_gpu_accelerator() and dt == DType.float64
    )

    comptime if not has_accelerator():
        if device == DEVICE_ACCEL:
            raise Error(
                "accelerator requested but this build has no supported"
                " accelerator"
            )
        return _run_cpu[dt](
            field, snaps_addr, n, nsteps, nevery, sdt, sidx, samp, useed, step0,
        )
    else:
        comptime if no_f64_on_accel:
            if device == DEVICE_ACCEL:
                raise Error(
                    "accelerator requested but it has no float64"
                    " support; use dtype=float32"
                )
            return _run_cpu[dt](
                field, snaps_addr, n, nsteps, nevery, sdt, sidx, samp, useed,
                step0,
            )
        else:
            if device == DEVICE_CPU:
                return _run_cpu[dt](
                    field, snaps_addr, n, nsteps, nevery, sdt, sidx, samp,
                    useed, step0,
                )
            if DeviceContext.number_of_devices() == 0:
                if device == DEVICE_ACCEL:
                    raise Error(
                        "accelerator requested but none was detected"
                        " at runtime"
                    )
                return _run_cpu[dt](
                    field, snaps_addr, n, nsteps, nevery, sdt, sidx, samp,
                    useed, step0,
                )
            return _run_accel[dt](
                field, snaps_addr, n, nsteps, nevery, sdt, sidx, samp, useed,
                step0,
            )


def run_chc(
    field_addr: Int,
    snaps_addr: Int,
    n: Int,
    nsteps: Int,
    nevery: Int,
    step_dt: Float64,
    dx: Float64,
    eps: Float64,
    seed: Int,
    step0: Int,
    single: Bool,
    device: Int,
) raises -> String:
    if single:
        return _dispatch[DType.float32](
            field_addr, snaps_addr, n, nsteps, nevery, step_dt, dx, eps, seed,
            step0, device,
        )
    return _dispatch[DType.float64](
        field_addr, snaps_addr, n, nsteps, nevery, step_dt, dx, eps, seed,
        step0, device,
    )
