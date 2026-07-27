// Pair-based radix sort: local histogram pass using updated GeneralInfo layout.
// Keys: selected field (depth or tile id) provided in keys_src
// Payload: vec2<u32> pairs kept in parallel buffer (scatter stage handles)

override PASS_ID = 0u;  // Pass ID for current radix sort pass
const WG_SIZE = 256u;
override RS_RADIX_LOG2 = 8u;  // 8 bit radices
override RS_RADIX_SIZE = 1u << RS_RADIX_LOG2;    // 256 entries into the radix table

struct GeneralInfo{
  keys_size : u32, preprocess_dispatch_x: u32, preprocess_dispatch_y: u32, preprocess_dispatch_z: u32,
  total_tile_depth_pair: u32, sort_dispatch_x: u32, sort_dispatch_y: u32, sort_dispatch_z: u32,
  l0_x : u32, l0_y : u32, l0_z : u32, l0_t : u32,
  l1_x : u32, l1_y : u32, l1_z : u32, l1_t : u32,
};

@group(0) @binding(0) var<storage, read> infos : GeneralInfo;
@group(0) @binding(1) var<storage, read> pairs_src : array<vec2<u32>>; // full pair array (moved to group0/binding2)
@group(0) @binding(2) var<storage, read_write> wg_histograms : array<u32>; // [digit][wg]

// Override: which component of the pair to sort on this phase.
// 0u -> use .x component; 1u -> use .y component.
override SORT_COMPONENT = 0u;

var<workgroup> local_histogram : array<atomic<u32>, RS_RADIX_SIZE>;

@compute @workgroup_size(WG_SIZE)
fn local_histogram_pass(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) wgs: vec3<u32>
) {
    if lid.x < RS_RADIX_SIZE { atomicStore(&local_histogram[lid.x], 0u); }
    workgroupBarrier();

    let pos = wid.x * WG_SIZE + lid.x;
    let n = infos.total_tile_depth_pair;
    if (pos < n) {
        // var key : u32 = p.x; // default
        // if (SORT_COMPONENT == 1u) { key = p.y; }
        let key: u32 = select(pairs_src[pos].x, pairs_src[pos].y, SORT_COMPONENT == 1u);
        let digit = extractBits(key, PASS_ID * RS_RADIX_LOG2, RS_RADIX_LOG2);
        atomicAdd(&local_histogram[digit], 1u);
    }
    workgroupBarrier();

    if lid.x < RS_RADIX_SIZE {
        wg_histograms[wid.x + lid.x * infos.sort_dispatch_x] = atomicLoad(&local_histogram[lid.x]);
    }
}
