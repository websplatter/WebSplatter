// Oriented-quad Gaussian render shader — ply_precision=f16 variant

const CUTOFF = log(255.);

override DISABLE_OPACITY_RADIUS: u32 = 0u;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
};

struct Splat {
    v_0: u32,
    v_1: u32,
    center_ndc: u32,
};

@group(0) @binding(0) var<storage, read> points_2d : array<Splat>;
@group(0) @binding(1) var<storage, read> color : array<vec2<u32>>;
@group(0) @binding(2) var<storage, read> indices : array<u32>;

@vertex
fn vs_main(
    @builtin(vertex_index) vid: u32,
    @builtin(instance_index) iid: u32
) -> VertexOutput {
    let idx = indices[iid];
    let vertex = points_2d[idx];

    let rg = unpack2x16float(color[idx].x);
    let ba = unpack2x16float(color[idx].y);
    let rgba = vec4<f32>(rg, ba);

    let v0 = unpack2x16float(vertex.v_0);
    let v1 = unpack2x16float(vertex.v_1);
    let center = unpack2x16float(vertex.center_ndc);

    // Opacity-adaptive radius_scale (disabled → fixed worst-case radius)
    var rs: f32;
    if DISABLE_OPACITY_RADIUS == 1u {
        rs = sqrt(CUTOFF);
    } else {
        rs = sqrt(log(max(1.0, 255.0 * rgba.a)));
    }

    let corner = vec2<f32>(f32((vid & 1u) == 0u) * 2.0 - 1.0, f32(vid < 2u) * 2.0 - 1.0);
    let ndc = center + v0 * corner.x * rs + v1 * corner.y * rs;

    return VertexOutput(
        vec4<f32>(ndc, 0.0, 1.0),
        rgba,
        corner,
    );
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let sigma_sq = dot(in.uv, in.uv);
    let opacity = in.color.a;

    var cutoff_val: f32;
    if DISABLE_OPACITY_RADIUS == 1u {
        cutoff_val = CUTOFF;
    } else {
        cutoff_val = log(max(1.0, 255.0 * opacity));
    }
    let alpha = min(0.99, opacity * exp(-cutoff_val * sigma_sq));

    if alpha < 1.0 / 255.0 {
        return vec4<f32>(0.0);
    }
    return vec4<f32>(in.color.rgb, 1.0) * alpha;
}
