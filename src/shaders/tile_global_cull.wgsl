const WG_SIZE = 256u;
const TILE_SIZE = 16u;
const CUTOFF = log(255.);

// Culling & index compaction pass extracted from preprocess.wgsl
// This file only contains the minimal definitions required for the cull pass.

struct GeneralInfo{
  keys_size : atomic<u32>, preprocess_dispatch_x: u32, preprocess_dispatch_y: u32, preprocess_dispatch_z: u32,
  total_tile_depth_pair: atomic<u32>, sort_dispatch_x: u32, sort_dispatch_y: u32, sort_dispatch_z: u32,
  l0_x : u32, l0_y : u32, l0_z : u32, l0_t : u32, // t0
  l1_x : u32, l1_y : u32, l1_z : u32, l1_t : u32, // t1
};

struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

struct Gaussian {
    // // (3+1)x f16 packed as u32
    // pos_opacity: array<u32,2>,
    // // 6x f16 sparse cov matrix
    // cov: array<u32,3>
    xy: u32,
    zw: u32,
    cov01: u32,
    cov23: u32,
    cov45: u32,
};

struct Splat {
    // 4x f16 packed as u32
    // vec of the quad
    v_0: u32,
    v_1: u32,
    // 2x f16 packed as u32
    // center of the quad
    pos: u32,
};

// Updated compact RenderSettings layout (32 bytes)
struct RenderSettings {
    canvas_size: vec2<u32>, // width, height
    max_sh_deg: u32,
    cur_sh_deg: u32,
    gaussian_scaling: f32,
    kernel_size: f32,
    mip_spatting: u32,
    walltime: f32,
}

struct SHSolver {
    // dir_opacity: vec4<f16>,
    dir_xy: u32,
    dir_z_opacity: u32,
    idx: u32,
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> render_settings: RenderSettings;

// Group 1: screen-size independent (persistent) GPU data
@group(1) @binding(0) var<storage, read> gaussians : array<Gaussian>;
@group(1) @binding(1) var<storage, read_write> points_2d : array<Splat>;
@group(1) @binding(2) var<storage, read_write> sort_pairs : array<vec2<u32>>; // .x tile_id, .y depth bits
@group(1) @binding(3) var<storage, read_write> sort_infos : GeneralInfo;      // atomic counters & dispatch info
@group(1) @binding(4) var<storage, read_write> sh_solvers : array<SHSolver>;  // compact alive records
@group(1) @binding(5) var<storage, read_write> tile_list : array<u32>;        // original per-tile indices (paired with sort_pairs)

// Group 2: screen-size dependent atomics (recreated on resize)
@group(2) @binding(0) var<storage, read_write> tile_counts : array<atomic<u32>>;


var<workgroup> scan0: array<u32, WG_SIZE>; // for prefix sum
var<workgroup> scan1: array<u32, WG_SIZE>; // for prefix sum
var<workgroup> group_base: u32;
// PASS 1: culling & index compaction
@compute @workgroup_size(WG_SIZE)
fn preprocess_cull(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    var alive = 0u;
    var depth: f32;
    var v_0: vec2<f32>;
    var v_1: vec2<f32>;
    var v_center: vec2<f32>;
    var opacity: f32;
    var dir: vec3<f32>;
    var radius_scale: f32;

    let idx = gid.x;
    if idx < arrayLength(&gaussians) {
        let vertex = gaussians[idx];
        let xyzw = vec4<f32>(unpack2x16float(vertex.xy), unpack2x16float(vertex.zw));
        let xyz = xyzw.xyz;
        opacity = xyzw.w;
        let cov01 = unpack2x16float(vertex.cov01);
        let cov23 = unpack2x16float(vertex.cov23);
        let cov45 = unpack2x16float(vertex.cov45);

        var camspace = camera.view * vec4<f32>(xyz, 1.0);
        let pos2d = camera.proj * camspace;
    
        let bounds = 1.2 * pos2d.w;
        let z = pos2d.z / pos2d.w;
        // if z > 0. && z < 1. && pos2d.x >= -bounds && pos2d.x <= bounds && pos2d.y >= -bounds && pos2d.y <= bounds {
        // if pos2d.w > 0. && z > 0. && z < 1. {
        if z > 0. && z < 1. && pos2d.x >= -bounds && pos2d.x <= bounds && pos2d.y >= -bounds && pos2d.y <= bounds {
            let focal: vec2<f32> = camera.focal;
            let viewport: vec2<f32> = camera.viewport;
            let scaling: f32 = render_settings.gaussian_scaling;

            let Vrk = mat3x3<f32>(
                cov01.x, cov01.y, cov23.x,
                cov01.y, cov23.y, cov45.x,
                cov23.x, cov45.x, cov45.y,
            ) * scaling * scaling;
            let J: mat3x3<f32> = mat3x3<f32>(
                focal.x / camspace.z,
                0.0,
                -(focal.x * camspace.x) / (camspace.z * camspace.z),
                0.0,
                -focal.y / camspace.z,
                (focal.y * camspace.y) / (camspace.z * camspace.z),
                0.0,
                0.0,
                0.0
            );

            let W = transpose(mat3x3<f32>(camera.view[0].xyz, camera.view[1].xyz, camera.view[2].xyz));
            let T = W * J; // DEBUG: extremely large values observed!
            let cov = transpose(T) * Vrk * T;

            // let kernel_size = KERNEL_SIZE;
            let kernel_size: f32 = render_settings.kernel_size;
            if bool(render_settings.mip_spatting) {
                // according to Mip-Splatting by Yu et al. 2023
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
                let radius = length(vec2<f32>((diagonal1 - diagonal2) / 2.0, offDiagonal));
                // eigenvalues of the 2D screen space splat
                let lambda1 = mid + radius;
                let lambda2 = max(mid - radius, 0.1);

                let diagonalVector = normalize(vec2<f32>(offDiagonal, lambda1 - diagonal1));
                
                // scaled eigenvectors in screen space 
                v_0 = sqrt(2.0 * lambda1) * diagonalVector / viewport;
                v_1 = sqrt(2.0 * lambda2) * vec2<f32>(diagonalVector.y, -diagonalVector.x) / viewport;
                v_center = pos2d.xy / pos2d.w;

                radius_scale = sqrt(log(255.0 * opacity));
                // let extent = sqrt(v_0 * v_0 + v_1 * v_1) * radius_scale * 2.0;
                // let extent = sqrt(v_0 * v_0 + v_1 * v_1) * CUTOFF * 2.0;
                let extent: vec2<f32> = (abs(v_0) + abs(v_1)) * radius_scale * 2.0;

                let min_pos = v_center - extent;
                let max_pos = v_center + extent;
                if max_pos.x > -1.0 && max_pos.y > -1.0 && min_pos.x < 1.0 && min_pos.y < 1.0 {
                    alive = 1u;
                    let camera_pos = vec3<f32>(camera.view_inv[3].xyz);
                    dir = normalize(xyz - camera_pos);
                    let zfar = -camera.proj[3][2] / (camera.proj[2][2] - 1.0);
                    depth = zfar - pos2d.z;
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
    // var addend256: u32 = 0u; if (lid.x >= 256u) { addend256 = scan[lid.x - 256u]; } workgroupBarrier(); scan[lid.x] = scan[lid.x] + addend256; workgroupBarrier();

    // Only perform one global atomic when this workgroup has survivors; then broadcast the base index
    if (lid.x == 0u) {
        let group_cnt  = scan0[WG_SIZE - 1u];      // Total survivors in this workgroup
        if (group_cnt != 0u) {
            group_base = atomicAdd(&sort_infos.keys_size, group_cnt);
        }
    }
    workgroupBarrier();

    // Write out compacted indices in order
    if (alive == 1u) {
        let store_idx = group_base + scan0[lid.x] - 1u;
        sh_solvers[store_idx] = SHSolver(
            // vec4<f16>(vec3<f16>(dir), f16(opacity)),
            pack2x16float(dir.xy),
            pack2x16float(vec2<f32>(dir.z, opacity)),
            idx,
        );
        points_2d[store_idx] = Splat(
            pack2x16float(v_0),
            pack2x16float(v_1),
            pack2x16float(v_center),
        );

        let e = (abs(v_0) + abs(v_1)) * radius_scale * 2;
        let size_size = vec4<u32>(render_settings.canvas_size, render_settings.canvas_size);
        let pixel_min_max = vec4<u32>((vec4<f32>(v_center - e, v_center + e) * 0.5 + vec4<f32>(0.5, 0.5, 0.5, 0.5)) * vec4<f32>(size_size));
        let bounds_min_max = clamp(pixel_min_max, vec4<u32>(0u), size_size - vec4<u32>(1u));
        let tiles_xy = (render_settings.canvas_size + TILE_SIZE - 1u) / TILE_SIZE;

        let tile_ranges = bounds_min_max / vec4<u32>(TILE_SIZE, TILE_SIZE, TILE_SIZE, TILE_SIZE);
        let nx = tiles_xy.x;
        let tile_depth_pair_count: u32 = (tile_ranges.z - tile_ranges.x + 1u) * (tile_ranges.w - tile_ranges.y + 1u);
        let tile_depth_pair_offset: u32 = atomicAdd(&sort_infos.total_tile_depth_pair, tile_depth_pair_count);
        var tmp_i = 0u;
        for (var ty = tile_ranges.y; ty <= tile_ranges.w; ty++) {
            for (var tx = tile_ranges.x; tx <= tile_ranges.z; tx++) {
                let tile_id = ty * nx + tx;
                let dst_id = tile_depth_pair_offset + tmp_i;
                atomicAdd(&tile_counts[tile_id], 1u);
                sort_pairs[dst_id] = vec2<u32>(tile_id, bitcast<u32>(depth) ^ 0xffffffffu); // bitwise NOT to sort descending by depth
                tile_list[dst_id] = store_idx;
                tmp_i += 1u;
            }
        }
    }
}