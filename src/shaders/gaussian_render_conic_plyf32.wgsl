// Oriented-quad Gaussian render shader — ply_precision=f32 variant

const CUTOFF = log(255.);

override DISABLE_OPACITY_RADIUS: u32 = 0u;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
};

struct Splat {
    v_0_x: f32, v_0_y: f32,
    v_1_x: f32, v_1_y: f32,
    center_ndc_x: f32, center_ndc_y: f32,
};

@group(0) @binding(0) var<storage, read> points_2d : array<Splat>;
@group(0) @binding(1) var<storage, read> color : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> indices : array<u32>;

@vertex
fn vs_main(
    @builtin(vertex_index) vid: u32,
    @builtin(instance_index) iid: u32
) -> VertexOutput {
    let idx = indices[iid];
    let vertex = points_2d[idx];

    let rgba = color[idx];
    let v0 = vec2<f32>(vertex.v_0_x, vertex.v_0_y);
    let v1 = vec2<f32>(vertex.v_1_x, vertex.v_1_y);
    let center = vec2<f32>(vertex.center_ndc_x, vertex.center_ndc_y);

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
