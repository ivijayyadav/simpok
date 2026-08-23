from std.math import ceildiv, sqrt
from std.memory import unsafe_memcpy
from std.memory.alloc import alloc, Layout
from std.sys import has_accelerator, size_of
from std.gpu import global_idx
from max.gpu.host import DeviceContext
from layout import TileTensor, TensorLayout, row_major

from ._utils import gaussian, TPB


def _update[dt: DType](
    c: Scalar[dt],
    up: Scalar[dt],
    dn: Scalar[dt],
    lf: Scalar[dt],
    rt: Scalar[dt],
    step_dt: Scalar[dt],
    inv_dx2: Scalar[dt],
    h: Scalar[dt],
) -> Scalar[dt]:
    var lap = (up + dn + lf + rt - 4.0 * c) * inv_dx2
    return c + step_dt * (c - c * c * c + h + lap)


def _step_kernel[dt: DType, LT: TensorLayout](
    psi: TileTensor[dt, LT, MutAnyOrigin],
    nxt: TileTensor[dt, LT, MutAnyOrigin],
    n: Int32,
    step_dt: Scalar[dt],
    inv_dx2: Scalar[dt],
    h: Scalar[dt],
    noise_amp: Scalar[dt],
    seed: UInt64,
    step: UInt64,
):
    comptime assert psi.flat_rank == 2, "field must be 2d"
    comptime assert nxt.flat_rank == 2, "field must be 2d"
    var nn = Int(n)
    var j = global_idx.x
    var i = global_idx.y
    if i >= nn or j >= nn:
        return

    var ip = i + 1 if i + 1 < nn else 0
    var im = i - 1 if i > 0 else nn - 1
    var jp = j + 1 if j + 1 < nn else 0
    var jm = j - 1 if j > 0 else nn - 1

    var v = _update[dt](
        rebind[Scalar[dt]](psi[i, j]),
        rebind[Scalar[dt]](psi[ip, j]),
        rebind[Scalar[dt]](psi[im, j]),
        rebind[Scalar[dt]](psi[i, jp]),
        rebind[Scalar[dt]](psi[i, jm]),
        step_dt,
        inv_dx2,
        h,
    )
    if noise_amp != 0.0:
        v += noise_amp * gaussian[dt](seed, step, UInt64(i * nn + j))
    nxt[i, j] = rebind[nxt.ElementType](v)


def _step_cpu[dt: DType](
    psi: Pointer[Scalar[dt], MutAnyOrigin],
    nxt: Pointer[Scalar[dt], MutAnyOrigin],
    n: Int,
    step_dt: Scalar[dt],
    inv_dx2: Scalar[dt],
    h: Scalar[dt],
    noise_amp: Scalar[dt],
    seed: UInt64,
    step: UInt64,
):
    for i in range(n):
        var ip = i + 1 if i + 1 < n else 0
        var im = i - 1 if i > 0 else n - 1
        for j in range(n):
            var jp = j + 1 if j + 1 < n else 0
            var jm = j - 1 if j > 0 else n - 1
            var v = _update[dt](
                psi[unsafe_offset=i * n + j],
                psi[unsafe_offset=ip * n + j],
                psi[unsafe_offset=im * n + j],
                psi[unsafe_offset=i * n + jp],
                psi[unsafe_offset=i * n + jm],
                step_dt,
                inv_dx2,
                h,
            )
            if noise_amp != 0.0:
                v += noise_amp * gaussian[dt](seed, step, UInt64(i * n + j))
            nxt[unsafe_offset=i * n + j] = v


def _run_cpu[dt: DType](
    field: Pointer[Scalar[dt], MutAnyOrigin],
    snaps_addr: Int,
    n: Int,
    nsteps: Int,
    nevery: Int,
    step_dt: Scalar[dt],
    inv_dx2: Scalar[dt],
    h: Scalar[dt],
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
            a, b, n, step_dt, inv_dx2, h, noise_amp, seed,
            UInt64(step0 + s - 1),
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


def _run_gpu[dt: DType](
    field: Pointer[Scalar[dt], MutAnyOrigin],
    snaps_addr: Int,
    n: Int,
    nsteps: Int,
    nevery: Int,
    step_dt: Scalar[dt],
    inv_dx2: Scalar[dt],
    h: Scalar[dt],
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
            h,
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
    return String("gpu")


def _dispatch[dt: DType](
    field_addr: Int,
    snaps_addr: Int,
    n: Int,
    nsteps: Int,
    nevery: Int,
    step_dt: Float64,
    dx: Float64,
    h: Float64,
    eps: Float64,
    seed: Int,
    step0: Int,
) raises -> String:
    var field = Pointer[Scalar[dt], MutAnyOrigin](
        unsafe_from_address=field_addr
    )
    var inv_dx2 = 1.0 / (dx * dx)
    var amp = sqrt(2.0 * eps * step_dt * inv_dx2) if eps > 0.0 else 0.0
    var sdt = Scalar[dt](step_dt)
    var sidx = Scalar[dt](inv_dx2)
    var sh = Scalar[dt](h)
    var samp = Scalar[dt](amp)
    var useed = UInt64(seed)

    comptime if not has_accelerator():
        return _run_cpu[dt](
            field, snaps_addr, n, nsteps, nevery, sdt, sidx, sh,
            samp, useed, step0,
        )
    else:
        if DeviceContext.number_of_devices() == 0:
            return _run_cpu[dt](
                field, snaps_addr, n, nsteps, nevery, sdt, sidx, sh,
                samp, useed, step0,
            )
        return _run_gpu[dt](
            field, snaps_addr, n, nsteps, nevery, sdt, sidx, sh,
            samp, useed, step0,
        )


def run_tdgl(
    field_addr: Int,
    snaps_addr: Int,
    n: Int,
    nsteps: Int,
    nevery: Int,
    step_dt: Float64,
    dx: Float64,
    h: Float64,
    eps: Float64,
    seed: Int,
    step0: Int,
    single: Bool,
) raises -> String:
    if single:
        return _dispatch[DType.float32](
            field_addr, snaps_addr, n, nsteps, nevery, step_dt, dx, h,
            eps, seed, step0,
        )
    return _dispatch[DType.float64](
        field_addr, snaps_addr, n, nsteps, nevery, step_dt, dx, h,
        eps, seed, step0,
    )
