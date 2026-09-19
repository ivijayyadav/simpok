from std.math import ceildiv, sqrt
from std.memory import unsafe_memcpy
from std.memory.alloc import alloc, Layout
from std.sys import has_accelerator, size_of
from std.sys.info import has_apple_gpu_accelerator
from std.gpu import global_idx
from max.gpu.host import DeviceContext
from layout import TileTensor, TensorLayout, row_major

from ._utils import gaussian, TPB, DEVICE_ACCEL, DEVICE_CPU


def _wall[dt: DType](
    p: Scalar[dt], h1: Scalar[dt], y1: Scalar[dt], y2: Scalar[dt]
) -> Scalar[dt]:
    return (h1 + y1 * p) / y2


def _mu[dt: DType](
    c: Scalar[dt],
    up: Scalar[dt],
    dn: Scalar[dt],
    lf: Scalar[dt],
    rt: Scalar[dt],
    v: Scalar[dt],
    inv_dx2: Scalar[dt],
) -> Scalar[dt]:
    return -c + c * c * c - 0.5 * (up + dn + lf + rt - 4.0 * c) * inv_dx2 + v


def _divnoise[dt: DType](
    nx: Int,
    i: Int,
    j: Int,
    jm: Int,
    at_wall: Bool,
    at_far: Bool,
    seed: UInt64,
    step: UInt64,
) -> Scalar[dt]:
    var here = UInt64(2 * (i * nx + j))
    var v = gaussian[dt](seed, step, here) - gaussian[dt](
        seed, step, UInt64(2 * (i * nx + jm))
    )
    if not at_far:
        v += gaussian[dt](seed, step, here + 1)
    if not at_wall:
        v -= gaussian[dt](seed, step, UInt64(2 * ((i - 1) * nx + j) + 1))
    return v


def _mu_kernel[dt: DType, LT: TensorLayout, LV: TensorLayout](
    psi: TileTensor[dt, LT, MutAnyOrigin],
    mu: TileTensor[dt, LT, MutAnyOrigin],
    pot: TileTensor[dt, LV, MutAnyOrigin],
    nz: Int32,
    nx: Int32,
    inv_dx2: Scalar[dt],
    h1: Scalar[dt],
    y1: Scalar[dt],
    y2: Scalar[dt],
):
    comptime assert psi.flat_rank == 2, "field must be 2d"
    comptime assert mu.flat_rank == 2, "field must be 2d"
    comptime assert pot.flat_rank == 1, "potential must be 1d"
    var rz = Int(nz)
    var rx = Int(nx)
    var j = Int(global_idx.x)
    var i = Int(global_idx.y)
    if i >= rz or j >= rx:
        return

    var jp = j + 1 if j + 1 < rx else 0
    var jm = j - 1 if j > 0 else rx - 1
    var c = rebind[Scalar[dt]](psi[i, j])
    var up = c if i == 0 else rebind[Scalar[dt]](psi[i - 1, j])
    var dn = rebind[Scalar[dt]](psi[i + 1, j]) if i < rz - 1 else _wall[dt](
        c, h1, y1, y2
    )

    mu[i, j] = rebind[mu.ElementType](
        _mu[dt](
            c,
            up,
            dn,
            rebind[Scalar[dt]](psi[i, jm]),
            rebind[Scalar[dt]](psi[i, jp]),
            rebind[Scalar[dt]](pot[i]),
            inv_dx2,
        )
    )


def _step_kernel[dt: DType, LT: TensorLayout](
    psi: TileTensor[dt, LT, MutAnyOrigin],
    mu: TileTensor[dt, LT, MutAnyOrigin],
    nxt: TileTensor[dt, LT, MutAnyOrigin],
    nz: Int32,
    nx: Int32,
    step_dt: Scalar[dt],
    inv_dx2: Scalar[dt],
    noise_amp: Scalar[dt],
    seed: UInt64,
    step: UInt64,
):
    comptime assert psi.flat_rank == 2, "field must be 2d"
    comptime assert nxt.flat_rank == 2, "field must be 2d"
    var rz = Int(nz)
    var rx = Int(nx)
    var j = Int(global_idx.x)
    var i = Int(global_idx.y)
    if i >= rz or j >= rx:
        return

    var jp = j + 1 if j + 1 < rx else 0
    var jm = j - 1 if j > 0 else rx - 1
    var m = rebind[Scalar[dt]](mu[i, j])
    var div = (
        rebind[Scalar[dt]](mu[i, jp]) + rebind[Scalar[dt]](mu[i, jm]) - 2.0 * m
    )
    if i > 0:
        div += rebind[Scalar[dt]](mu[i - 1, j]) - m
    if i < rz - 1:
        div += rebind[Scalar[dt]](mu[i + 1, j]) - m

    var v = rebind[Scalar[dt]](psi[i, j]) + step_dt * div * inv_dx2
    if noise_amp != 0.0:
        v += noise_amp * _divnoise[dt](
            rx, i, j, jm, i == 0, i == rz - 1, seed, step
        )
    nxt[i, j] = rebind[nxt.ElementType](v)


def _mu_cpu[dt: DType](
    psi: Pointer[Scalar[dt], MutAnyOrigin],
    mu: Pointer[Scalar[dt], MutAnyOrigin],
    pot: Pointer[Scalar[dt], MutAnyOrigin],
    nz: Int,
    nx: Int,
    inv_dx2: Scalar[dt],
    h1: Scalar[dt],
    y1: Scalar[dt],
    y2: Scalar[dt],
):
    for i in range(nz):
        var v = pot[unsafe_offset=i]
        for j in range(nx):
            var jp = j + 1 if j + 1 < nx else 0
            var jm = j - 1 if j > 0 else nx - 1
            var c = psi[unsafe_offset=i * nx + j]
            var up = c if i == 0 else psi[unsafe_offset=(i - 1) * nx + j]
            var dn = psi[
                unsafe_offset=(i + 1) * nx + j
            ] if i < nz - 1 else _wall[dt](c, h1, y1, y2)
            mu[unsafe_offset=i * nx + j] = _mu[dt](
                c,
                up,
                dn,
                psi[unsafe_offset=i * nx + jm],
                psi[unsafe_offset=i * nx + jp],
                v,
                inv_dx2,
            )


def _step_cpu[dt: DType](
    psi: Pointer[Scalar[dt], MutAnyOrigin],
    mu: Pointer[Scalar[dt], MutAnyOrigin],
    nxt: Pointer[Scalar[dt], MutAnyOrigin],
    nz: Int,
    nx: Int,
    step_dt: Scalar[dt],
    inv_dx2: Scalar[dt],
    noise_amp: Scalar[dt],
    seed: UInt64,
    step: UInt64,
):
    for i in range(nz):
        for j in range(nx):
            var jp = j + 1 if j + 1 < nx else 0
            var jm = j - 1 if j > 0 else nx - 1
            var m = mu[unsafe_offset=i * nx + j]
            var div = (
                mu[unsafe_offset=i * nx + jp]
                + mu[unsafe_offset=i * nx + jm]
                - 2.0 * m
            )
            if i > 0:
                div += mu[unsafe_offset=(i - 1) * nx + j] - m
            if i < nz - 1:
                div += mu[unsafe_offset=(i + 1) * nx + j] - m

            var v = psi[unsafe_offset=i * nx + j] + step_dt * div * inv_dx2
            if noise_amp != 0.0:
                v += noise_amp * _divnoise[dt](
                    nx, i, j, jm, i == 0, i == nz - 1, seed, step
                )
            nxt[unsafe_offset=i * nx + j] = v


def _run_cpu[dt: DType](
    field: Pointer[Scalar[dt], MutAnyOrigin],
    pot: Pointer[Scalar[dt], MutAnyOrigin],
    snaps_addr: Int,
    nz: Int,
    nx: Int,
    nsteps: Int,
    nevery: Int,
    step_dt: Scalar[dt],
    inv_dx2: Scalar[dt],
    h1: Scalar[dt],
    y1: Scalar[dt],
    y2: Scalar[dt],
    noise_amp: Scalar[dt],
    seed: UInt64,
    step0: Int,
) raises -> String:
    comptime esize = size_of[Scalar[dt]]()
    var size = nz * nx
    var owned_a = alloc(Layout[Scalar[dt]](count=size)).into_managed()
    var owned_b = alloc(Layout[Scalar[dt]](count=size)).into_managed()
    var owned_m = alloc(Layout[Scalar[dt]](count=size)).into_managed()
    var a = Pointer[Scalar[dt], MutAnyOrigin](
        unsafe_from_address=Int(owned_a.unsafe_ptr())
    )
    var b = Pointer[Scalar[dt], MutAnyOrigin](
        unsafe_from_address=Int(owned_b.unsafe_ptr())
    )
    var m = Pointer[Scalar[dt], MutAnyOrigin](
        unsafe_from_address=Int(owned_m.unsafe_ptr())
    )
    unsafe_memcpy(dest=a, src=field, count=size)

    var k = 0
    for s in range(1, nsteps + 1):
        _mu_cpu[dt](a, m, pot, nz, nx, inv_dx2, h1, y1, y2)
        _step_cpu[dt](
            a, m, b, nz, nx, step_dt, inv_dx2, noise_amp, seed,
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
    _ = owned_m^
    return String("cpu")


def _run_accel[dt: DType](
    field: Pointer[Scalar[dt], MutAnyOrigin],
    pot: Pointer[Scalar[dt], MutAnyOrigin],
    snaps_addr: Int,
    nz: Int,
    nx: Int,
    nsteps: Int,
    nevery: Int,
    step_dt: Scalar[dt],
    inv_dx2: Scalar[dt],
    h1: Scalar[dt],
    y1: Scalar[dt],
    y2: Scalar[dt],
    noise_amp: Scalar[dt],
    seed: UInt64,
    step0: Int,
) raises -> String:
    comptime esize = size_of[Scalar[dt]]()
    var size = nz * nx
    var ctx = DeviceContext()
    var a = ctx.enqueue_create_buffer[dt](size)
    var b = ctx.enqueue_create_buffer[dt](size)
    var m = ctx.enqueue_create_buffer[dt](size)
    var v = ctx.enqueue_create_buffer[dt](nz)
    ctx.enqueue_copy(dst_buf=a, src_ptr=field)
    ctx.enqueue_copy(dst_buf=v, src_ptr=pot)

    var layout = row_major(nz, nx)
    var vlayout = row_major(nz)
    comptime mkern = _mu_kernel[dt, type_of(layout), type_of(vlayout)]
    comptime skern = _step_kernel[dt, type_of(layout)]
    var grid = (ceildiv(nx, TPB), ceildiv(nz, TPB))

    var k = 0
    for s in range(1, nsteps + 1):
        ctx.enqueue_function[mkern](
            TileTensor(a, layout),
            TileTensor(m, layout),
            TileTensor(v, vlayout),
            Int32(nz),
            Int32(nx),
            inv_dx2,
            h1,
            y1,
            y2,
            grid_dim=grid,
            block_dim=(TPB, TPB),
        )
        ctx.enqueue_function[skern](
            TileTensor(a, layout),
            TileTensor(m, layout),
            TileTensor(b, layout),
            Int32(nz),
            Int32(nx),
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
    return String("gpu")


def _dispatch[dt: DType](
    field_addr: Int,
    pot_addr: Int,
    snaps_addr: Int,
    nz: Int,
    nx: Int,
    nsteps: Int,
    nevery: Int,
    step_dt: Float64,
    dx: Float64,
    h1: Float64,
    g: Float64,
    gamma: Float64,
    eps: Float64,
    seed: Int,
    step0: Int,
    device: Int,
) raises -> String:
    var field = Pointer[Scalar[dt], MutAnyOrigin](
        unsafe_from_address=field_addr
    )
    var pot = Pointer[Scalar[dt], MutAnyOrigin](unsafe_from_address=pot_addr)
    var inv_dx2 = 1.0 / (dx * dx)
    var amp = sqrt(2.0 * eps * step_dt) * inv_dx2 if eps > 0.0 else 0.0
    var y1 = gamma / dx
    var sdt = Scalar[dt](step_dt)
    var sidx = Scalar[dt](inv_dx2)
    var sh1 = Scalar[dt](h1)
    var sy1 = Scalar[dt](y1)
    var sy2 = Scalar[dt](y1 - g)
    var samp = Scalar[dt](amp)
    var useed = UInt64(seed)

    comptime no_f64_on_accel = (
        has_apple_gpu_accelerator() and dt == DType.float64
    )

    comptime if not has_accelerator():
        if device == DEVICE_ACCEL:
            raise Error(
                "device='gpu' requested but this build has no supported"
                " GPU"
            )
        return _run_cpu[dt](
            field, pot, snaps_addr, nz, nx, nsteps, nevery, sdt, sidx, sh1,
            sy1, sy2, samp, useed, step0,
        )
    else:
        comptime if no_f64_on_accel:
            if device == DEVICE_ACCEL:
                raise Error(
                    "device='gpu' requested but this GPU has no float64"
                    " support (use dtype=float32)"
                )
            return _run_cpu[dt](
                field, pot, snaps_addr, nz, nx, nsteps, nevery, sdt, sidx, sh1,
                sy1, sy2, samp, useed, step0,
            )
        else:
            if device == DEVICE_CPU:
                return _run_cpu[dt](
                    field, pot, snaps_addr, nz, nx, nsteps, nevery, sdt, sidx,
                    sh1, sy1, sy2, samp, useed, step0,
                )
            if DeviceContext.number_of_devices() == 0:
                if device == DEVICE_ACCEL:
                    raise Error(
                        "device='gpu' requested but no GPU was detected"
                        " at runtime"
                    )
                return _run_cpu[dt](
                    field, pot, snaps_addr, nz, nx, nsteps, nevery, sdt, sidx,
                    sh1, sy1, sy2, samp, useed, step0,
                )
            return _run_accel[dt](
                field, pot, snaps_addr, nz, nx, nsteps, nevery, sdt, sidx, sh1,
                sy1, sy2, samp, useed, step0,
            )


def run_sdsd(
    field_addr: Int,
    pot_addr: Int,
    snaps_addr: Int,
    nz: Int,
    nx: Int,
    nsteps: Int,
    nevery: Int,
    step_dt: Float64,
    dx: Float64,
    h1: Float64,
    g: Float64,
    gamma: Float64,
    eps: Float64,
    seed: Int,
    step0: Int,
    single: Bool,
    device: Int,
) raises -> String:
    if single:
        return _dispatch[DType.float32](
            field_addr, pot_addr, snaps_addr, nz, nx, nsteps, nevery, step_dt,
            dx, h1, g, gamma, eps, seed, step0, device,
        )
    return _dispatch[DType.float64](
        field_addr, pot_addr, snaps_addr, nz, nx, nsteps, nevery, step_dt, dx,
        h1, g, gamma, eps, seed, step0, device,
    )
