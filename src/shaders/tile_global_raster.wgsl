const CUTOFF = log(255.);

struct Splat {
    // 4x f16 packed as u32
    // vec of the quad
    v_0: u32,
    v_1: u32,
    // 2x f16 packed as u32
    // center of the quad
    pos: u32,
};

struct RenderSettings {
    canvas_size: vec2<u32>, // width, height
    max_sh_deg: u32,
    cur_sh_deg: u32,
    gaussian_scaling: f32,
    kernel_size: f32,
    mip_spatting: u32,
    walltime: f32,
};

struct GeneralInfo {
    keys_size: u32,  // Total number of keys (for context)
    dispatch_x: u32,  // Number of workgroups along x for histogram source
    dispatch_y: u32,  // (digits) normally RS_RADIX_SIZE or batched digits
    dispatch_z: u32,
    l0_x: u32,  // Mirrors grid dims for L0 (informational)
    l0_y: u32,
    l0_z: u32,
    l0_t: u32,  // Number of L0 tiles per digit
    l1_x: u32,  // Mirrors grid dims for L1 (informational)
    l1_y: u32,
    l1_z: u32,
    l1_t: u32,  // Number of L1 tiles over L0 tiles per digit
};


// Group 0: static (not re-created on resize) now also holds tile_list (persistent)
@group(0) @binding(0) var<uniform> render_settings: RenderSettings;
@group(0) @binding(1) var<storage, read> sort_infos: GeneralInfo;          // provides keys_size
@group(0) @binding(2) var<storage, read> points_2d : array<Splat>;
@group(0) @binding(3) var<storage, read> color : array<u32>;
@group(0) @binding(4) var<storage, read> depth : array<u32>;            // (optional / reserved)
@group(0) @binding(5) var<storage, read> tile_list    : array<u32>;        // flattened per-tile sorted indices (persistent)

// Group 1: resize-only tile_offsets
@group(1) @binding(0) var<storage, read> tile_offsets : array<u32>;        // length = numTiles+1 (recreated on resize)

// Group 2: resize-only texture target
@group(2) @binding(0) var out_texture: texture_storage_2d<rgba8unorm, write>;

// TODO: 
fn fast_inverse_2x2(T: mat2x2<f32>) -> mat2x2<f32> {
    let a = T[0][0];
    let b = T[0][1];
    let c = T[1][0];
    let d = T[1][1];
    let det = a * d - b * c;
    let inv_det = 1.0 / det;
    return mat2x2<f32>(
        vec2<f32>(d, -b) * inv_det,
        vec2<f32>(-c, a) * inv_det
    );
}

const TILE_SIZE: u32 = 16u;   // must match binning pass

@compute @workgroup_size(TILE_SIZE, TILE_SIZE, 1)
fn tile_based_main(
    @builtin(global_invocation_id)    global_id: vec3<u32>,
    @builtin(workgroup_id)            wid: vec3<u32>,
    @builtin(num_workgroups)          num_wg: vec3<u32>,
    @builtin(local_invocation_index)  local_idx: u32,
) {
    if global_id.x >= render_settings.canvas_size.x || global_id.y >= render_settings.canvas_size.y { return; }
    // let in_bounds = global_id.x < render_settings.canvas_size.x && global_id.y < render_settings.canvas_size.y;

    // Identify tile for this pixel (each workgroup == one tile by construction)
    let tile_id = wid.y * num_wg.x + wid.x;
    let num_tiles = num_wg.x * num_wg.y;
    if tile_id >= num_tiles { return; }

    let start: u32 = tile_offsets[tile_id];
    let end: u32 = tile_offsets[tile_id + 1u];
    // let len: u32 = end - start;
    var accumulate_opacity: f32 = 1.0;
    var final_color = vec3<f32>(0.0, 0.0, 0.0);

    // Direct iteration (may be unsorted)
    let screen_pos = vec2<f32>(global_id.xy) + vec2<f32>(0.5, 0.5);
    let clip_pos = screen_pos / vec2<f32>(render_settings.canvas_size) * 2.0 - 1.0;
    for (var i: u32 = start; i < end; i++) {
        let idx = tile_list[i];
        let p = points_2d[idx];
        let color = unpack4x8unorm(color[idx]);

        let center = unpack2x16float(p.pos);
        let T = mat2x2<f32>(unpack2x16float(p.v_0), unpack2x16float(p.v_1));
        let inv_T = fast_inverse_2x2(T);
        let diff = clip_pos - center;
        let d = inv_T * diff / 2.0;
        let power = dot(d, d);
        if power > CUTOFF { continue; }

        let opacity = min(0.999, color.a * exp(-power));
        if opacity < 0.001 { continue; }

        final_color += vec3<f32>(color.rgb) * opacity * accumulate_opacity;
        accumulate_opacity *= 1.0 - opacity;
        if accumulate_opacity < 0.001 {
            break;
        }
    }

    textureStore(out_texture, vec2<u32>(global_id.x, render_settings.canvas_size.y - global_id.y - 1u), vec4<f32>(final_color, 1.0));
}