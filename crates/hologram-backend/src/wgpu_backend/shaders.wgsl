// WGSL compute kernels for the wgpu backend (spec IX.4).
// One entry point per major op family; the backend selects the right
// pipeline by KernelCall variant. Buffers are typed as `array<f32>`;
// non-f32 dtypes route through the CPU backend.

@group(0) @binding(0) var<storage, read>      a_buf: array<f32>;
@group(0) @binding(1) var<storage, read>      b_buf: array<f32>;
@group(0) @binding(2) var<storage, read_write> out_buf: array<f32>;

struct Params {
    n: u32,
    m: u32,
    k: u32,
    pad0: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn add_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
    out_buf[i] = a_buf[i] + b_buf[i];
}

@compute @workgroup_size(64)
fn sub_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
    out_buf[i] = a_buf[i] - b_buf[i];
}

@compute @workgroup_size(64)
fn mul_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
    out_buf[i] = a_buf[i] * b_buf[i];
}

@compute @workgroup_size(64)
fn relu_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
    out_buf[i] = max(a_buf[i], 0.0);
}

@compute @workgroup_size(64)
fn sigmoid_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
    out_buf[i] = 1.0 / (1.0 + exp(-a_buf[i]));
}

@compute @workgroup_size(64)
fn tanh_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
    let x = a_buf[i];
    let ex = exp(2.0 * x);
    out_buf[i] = (ex - 1.0) / (ex + 1.0);
}

@compute @workgroup_size(8, 8)
fn matmul_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let j = gid.y;
    if (i >= params.m || j >= params.n) { return; }
    var acc = 0.0;
    for (var kk: u32 = 0u; kk < params.k; kk = kk + 1u) {
        acc = acc + a_buf[i * params.k + kk] * b_buf[kk * params.n + j];
    }
    out_buf[i * params.n + j] = acc;
}
