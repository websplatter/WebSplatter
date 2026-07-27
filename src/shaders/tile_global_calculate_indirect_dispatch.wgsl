const WG_SIZE = 256u;
const TILE_SIZE = 16u;
override RS_RADIX_LOG2 = 8u;  // 2 bit radices
override RS_RADIX_SIZE = 1u << RS_RADIX_LOG2;    // 4 entries into the radix table

struct GeneralInfo{
  keys_size : u32, preprocess_dispatch_x: u32, preprocess_dispatch_y: u32, preprocess_dispatch_z: u32,
  total_tile_depth_pair: u32, sort_dispatch_x: u32, sort_dispatch_y: u32, sort_dispatch_z: u32,
  l0_x : u32, l0_y : u32, l0_z : u32, l0_t : u32, // t0
  l1_x : u32, l1_y : u32, l1_z : u32, l1_t : u32, // t1
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
};

@group(0) @binding(0) var<storage, read_write> infos  : GeneralInfo;
@group(0) @binding(1) var<uniform>    render_settings : RenderSettings;

@group(1) @binding(0) var<storage, read> tile_counts : array<u32>; // length = numTiles
@group(1) @binding(1) var<storage, read_write> tile_offsets : array<u32>; // length = numTiles + 1

@compute @workgroup_size(1)
fn write_dispatch_triples(
    @builtin(global_invocation_id) gid: vec3<u32>,
    // @builtin(workgroup_id)        wid: vec3<u32>,
    // @builtin(local_invocation_id) lid: vec3<u32>
) {
    if (gid.x == 0u) {
        // Histogram/Scatter dispatch X (elements divided by WG_SIZE)
        infos.preprocess_dispatch_x = (infos.keys_size + WG_SIZE - 1u) / WG_SIZE;
        infos.preprocess_dispatch_y = 1u;
        infos.preprocess_dispatch_z = 1u;

        infos.sort_dispatch_x = (infos.total_tile_depth_pair + WG_SIZE - 1u) / WG_SIZE;
        infos.sort_dispatch_y = 1u;
        infos.sort_dispatch_z = 1u;

        // Two-level tile counts
        let t0 = (infos.sort_dispatch_x + WG_SIZE - 1u) / WG_SIZE;
        let t1 = (t0 + WG_SIZE - 1u) / WG_SIZE;

        // Triples for L0/L1 plus t0/t1
        infos.l0_x = t0; infos.l0_y = RS_RADIX_SIZE; infos.l0_z = 1u; infos.l0_t = t0;
        infos.l1_x = t1; infos.l1_y = RS_RADIX_SIZE; infos.l1_z = 1u; infos.l1_t = t1;

        let tiles_xy = (render_settings.canvas_size + TILE_SIZE - 1u) / TILE_SIZE;
        let num_tiles = tiles_xy.x * tiles_xy.y;
        var sum: u32 = 0u;
        for (var t: u32 = 0u; t < num_tiles; t++) {
            let c = tile_counts[t];
            tile_offsets[t] = sum; // exclusive
            sum += c;
        }
        // store total at element num_tiles
        tile_offsets[num_tiles] = sum;
    }
}