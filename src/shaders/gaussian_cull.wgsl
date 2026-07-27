// Culling & index compaction pass — ply_precision=f16 variant

struct GeneralInfo{
  keys_size : atomic<u32>, dispatch_x: u32, dispatch_y: u32, dispatch_z: u32,
  l0_x : u32, l0_y : u32, l0_z : u32, l0_t : u32,
  l1_x : u32, l1_y : u32, l1_z : u32, l1_t : u32,
};

const WG_SIZE = 256u;
const CUTOFF = log(255.);

override DISABLE_AABB_CULL: u32 = 0u;
override DISABLE_OPACITY_RADIUS: u32 = 0u;

struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

struct Gaussian {
    xy: u32,
    zw: u32,
    cov01: u32,
    cov23: u32,
    cov45: u32,
};

struct Splat {
    v_0: u32,        // pack2x16float — eigenvector 0 in NDC (unbaked)
    v_1: u32,        // pack2x16float — eigenvector 1 in NDC (unbaked)
    center_ndc: u32,  // pack2x16float — center in NDC
};

struct RenderSettings {
    canvas_size: vec2<u32>,
    max_sh_deg: u32,
    cur_sh_deg: u32,
    gaussian_scaling: f32,
    kernel_size: f32,
    mip_spatting: u32,
    walltime: f32,
}

struct SHSolver {
    dir_xy: u32,
    dir_z_opacity: u32,
    idx: u32,
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> render_settings: RenderSettings;

@group(1) @binding(0) var<storage, read> gaussians : array<Gaussian>;
@group(1) @binding(1) var<storage, read_write> points_2d : array<Splat>;

@group(2) @binding(0) var<storage, read_write> sort_infos: GeneralInfo;
@group(2) @binding(1) var<storage, read_write> sort_depths : array<u32>;
@group(2) @binding(2) var<storage, read_write> sort_indices : array<u32>;
@group(2) @binding(3) var<storage, read_write> sh_solvers : array<SHSolver>;

var<workgroup> scan0: array<u32, WG_SIZE>;
var<workgroup> scan1: array<u32, WG_SIZE>;
var<workgroup> group_base: u32;

@compute @workgroup_size(WG_SIZE)
fn preprocess_cull(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    var alive = 0u;
    var depth: f32;
    var v_0: vec2<f32>;
    var v_1: vec2<f32>;
    var v_center: vec2<f32>;
    var opacity: f32;
    var dir: vec3<f32>;

    let idx = gid.x;
    if idx < arrayLength(&gaussians) {
        let vertex = gaussians[idx];
        let xyzw = vec4<f32>(unpack2x16float(vertex.xy), unpack2x16float(vertex.zw));
        let xyz = xyzw.xyz;
        opacity = xyzw.w;

        var camspace = camera.view * vec4<f32>(xyz, 1.0);
        let pos2d = camera.proj * camspace;

        let z = pos2d.z / pos2d.w;
        if camspace.z > 0.2 && z > 0.0 && z < 1.0 {
            let focal: vec2<f32> = camera.focal;
            let viewport: vec2<f32> = camera.viewport;
            let scaling: f32 = render_settings.gaussian_scaling;

            let cov01 = unpack2x16float(vertex.cov01);
            let cov23 = unpack2x16float(vertex.cov23);
            let cov45 = unpack2x16float(vertex.cov45);
            let Vrk = mat3x3<f32>(
                cov01.x, cov01.y, cov23.x,
                cov01.y, cov23.y, cov45.x,
                cov23.x, cov45.x, cov45.y,
            ) * scaling * scaling;

            var t = camspace.xyz;
            let tan_fovx = viewport.x / (2.0 * focal.x);
            let tan_fovy = viewport.y / (2.0 * focal.y);
            let limx = 1.3 * tan_fovx;
            let limy = 1.3 * tan_fovy;
            t.x = clamp(t.x / t.z, -limx, limx) * t.z;
            t.y = clamp(t.y / t.z, -limy, limy) * t.z;

            let J: mat3x3<f32> = mat3x3<f32>(
                focal.x / t.z,
                0.0,
                -(focal.x * t.x) / (t.z * t.z),
                0.0,
                focal.y / t.z,
                -(focal.y * t.y) / (t.z * t.z),
                0.0,
                0.0,
                0.0
            );

            let W = transpose(mat3x3<f32>(camera.view[0].xyz, camera.view[1].xyz, camera.view[2].xyz));
            let T = W * J;
            let cov = transpose(T) * Vrk * T;

            let kernel_size: f32 = render_settings.kernel_size;
            if bool(render_settings.mip_spatting) {
                let det_0: f32 = max(1e-6, cov[0][0] * cov[1][1] - cov[0][1] * cov[0][1]);
                let det_1: f32 = max(1e-6, (cov[0][0] + kernel_size) * (cov[1][1] + kernel_size) - cov[0][1] * cov[0][1]);
                var coef: f32 = sqrt(det_0 / (det_1 + 1e-6) + 1e-6);
                if (det_0 <= 1e-6 || det_1 <= 1e-6) {
                    coef = 0.0;
                }
                opacity *= coef;
            }

            if opacity > 1.0 / 255.0 {
                opacity = min(opacity, 1.0);

                let diagonal1 = cov[0][0] + kernel_size;
                let offDiagonal = cov[0][1];
                let diagonal2 = cov[1][1] + kernel_size;

                let mid = 0.5 * (diagonal1 + diagonal2);
                let eigRadius = length(vec2<f32>((diagonal1 - diagonal2) / 2.0, offDiagonal));
                let lambda1 = mid + eigRadius;
                let lambda2 = max(mid - eigRadius, 0.1);

                // Guard against degenerate isotropic case
                var diagonalVector: vec2<f32>;
                if eigRadius < 1e-6 {
                    diagonalVector = vec2<f32>(1.0, 0.0);
                } else {
                    diagonalVector = normalize(vec2<f32>(offDiagonal, lambda1 - diagonal1));
                }

                // NDC ↔ pixel: Δndc = Δpixel * (2/W, -2/H) because NDC Y-up, pixel Y-down
                let pixel_to_ndc = vec2<f32>(2.0, -2.0) / viewport;
                v_0 = sqrt(2.0 * lambda1) * diagonalVector * pixel_to_ndc;
                v_1 = sqrt(2.0 * lambda2) * vec2<f32>(diagonalVector.y, -diagonalVector.x) * pixel_to_ndc;
                v_center = pos2d.xy / pos2d.w;

                var radius_scale: f32;
                if DISABLE_OPACITY_RADIUS == 1u {
                    radius_scale = sqrt(CUTOFF);
                } else {
                    radius_scale = sqrt(log(255.0 * opacity));
                }

                let camera_pos = vec3<f32>(camera.view_inv[3].xyz);
                dir = normalize(xyz - camera_pos);

                let zfar = -camera.proj[3][2] / (camera.proj[2][2] - 1.0);
                depth = zfar - pos2d.z;

                let half_extent = (abs(v_0) + abs(v_1)) * radius_scale;
                if DISABLE_AABB_CULL == 1u {
                    alive = 1u;
                } else {
                    let min_pos = v_center - half_extent;
                    let max_pos = v_center + half_extent;
                    if max_pos.x > -1.0 && max_pos.y > -1.0 && min_pos.x < 1.0 && min_pos.y < 1.0 {
                        alive = 1u;
                    }
                }
            }
        }
    }
    scan0[lid.x] = alive;

    workgroupBarrier();

    if (lid.x >= 1u) { scan1[lid.x] = scan0[lid.x] + scan0[lid.x - 1u]; } else { scan1[lid.x] = scan0[lid.x]; } workgroupBarrier();
    if (lid.x >= 2u) { scan0[lid.x] = scan1[lid.x] + scan1[lid.x - 2u]; } else { scan0[lid.x] = scan1[lid.x]; } workgroupBarrier();
    if (lid.x >= 4u) { scan1[lid.x] = scan0[lid.x] + scan0[lid.x - 4u]; } else { scan1[lid.x] = scan0[lid.x]; } workgroupBarrier();
    if (lid.x >= 8u) { scan0[lid.x] = scan1[lid.x] + scan1[lid.x - 8u]; } else { scan0[lid.x] = scan1[lid.x]; } workgroupBarrier();
    if (lid.x >= 16u) { scan1[lid.x] = scan0[lid.x] + scan0[lid.x - 16u]; } else { scan1[lid.x] = scan0[lid.x]; } workgroupBarrier();
    if (lid.x >= 32u) { scan0[lid.x] = scan1[lid.x] + scan1[lid.x - 32u]; } else { scan0[lid.x] = scan1[lid.x]; } workgroupBarrier();
    if (lid.x >= 64u) { scan1[lid.x] = scan0[lid.x] + scan0[lid.x - 64u]; } else { scan1[lid.x] = scan0[lid.x]; } workgroupBarrier();
    if (lid.x >= 128u) { scan0[lid.x] = scan1[lid.x] + scan1[lid.x - 128u]; } else { scan0[lid.x] = scan1[lid.x]; } workgroupBarrier();

    if (lid.x == 0u) {
        let group_cnt  = scan0[WG_SIZE - 1u];
        if (group_cnt != 0u) {
            group_base = atomicAdd(&sort_infos.keys_size, group_cnt);
        }
    }
    workgroupBarrier();

    if (alive == 1u) {
        let store_idx = group_base + scan0[lid.x] - 1u;
        sh_solvers[store_idx] = SHSolver(
            pack2x16float(dir.xy),
            pack2x16float(vec2<f32>(dir.z, opacity)),
            idx,
        );
        points_2d[store_idx] = Splat(
            pack2x16float(v_0),
            pack2x16float(v_1),
            pack2x16float(v_center),
        );
        sort_depths[store_idx] = bitcast<u32>(depth);
        sort_indices[store_idx] = store_idx;
    }
}
