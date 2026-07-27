// Pair-based radix sort scatter stage.
// Payload comparison objects are pairs_src/pairs_dst (tile_id, depth_bits)
// keys_src/keys_dst store indices (permutation) accompanying the pair ordering.

override PASS_ID = 0u;
const WG_SIZE = 256u;
const WORDS_PER_WG : u32 = WG_SIZE / 32u; // 8 words for 256
override RS_RADIX_LOG2 = 8u;
override RS_RADIX_SIZE = 1u << RS_RADIX_LOG2; // 256
// Keep an override for SORT_COMPONENT (mirrors histogram shader API); not used directly here yet.
override SORT_COMPONENT = 0u;

struct GeneralInfo{
  keys_size : u32, preprocess_dispatch_x: u32, preprocess_dispatch_y: u32, preprocess_dispatch_z: u32,
  total_tile_depth_pair: u32, sort_dispatch_x: u32, sort_dispatch_y: u32, sort_dispatch_z: u32,
  l0_x : u32, l0_y : u32, l0_z : u32, l0_t : u32,
  l1_x : u32, l1_y : u32, l1_z : u32, l1_t : u32,
};

@group(0) @binding(0) var<storage, read> infos: GeneralInfo;
@group(0) @binding(1) var<storage, read> digit_base : array<u32>;
// Reordered bindings: pairs first (2,3) then keys (4,5)
@group(0) @binding(2) var<storage, read> pairs_src : array<vec2<u32>>;
@group(0) @binding(3) var<storage, read_write> pairs_dst : array<vec2<u32>>;
@group(0) @binding(4) var<storage, read> tile_list_src : array<u32>;
@group(0) @binding(5) var<storage, read_write> tile_list_dst : array<u32>;
@group(0) @binding(6) var<storage, read> wg_prefixes : array<u32>;

struct BinWords { words: array<atomic<u32>, WORDS_PER_WG + 1> }
var<workgroup> bin_flags : array<BinWords, RS_RADIX_SIZE>;

@compute @workgroup_size(WG_SIZE)
fn scatter_elements(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    // zero selected digit row words (manually unrolled 8 words)
    atomicStore(&bin_flags[lid.x].words[0], 0u);
    atomicStore(&bin_flags[lid.x].words[1], 0u);
    atomicStore(&bin_flags[lid.x].words[2], 0u);
    atomicStore(&bin_flags[lid.x].words[3], 0u);
    atomicStore(&bin_flags[lid.x].words[4], 0u);
    atomicStore(&bin_flags[lid.x].words[5], 0u);
    atomicStore(&bin_flags[lid.x].words[6], 0u);
    atomicStore(&bin_flags[lid.x].words[7], 0u);
    workgroupBarrier();

    let pos = wid.x * WG_SIZE + lid.x;
    let n = infos.total_tile_depth_pair;
    var key: u32 = 0u; var digit: u32 = 0u;
    if (pos < n) {
        key = select(pairs_src[pos].x, pairs_src[pos].y, SORT_COMPONENT == 1u);
        digit = extractBits(key, PASS_ID * RS_RADIX_LOG2, RS_RADIX_LOG2);
        let myWord = lid.x >> 5u;
        let myBit = 1u << (lid.x & 31u);
        atomicOr(&bin_flags[digit].words[myWord], myBit);
    }
    workgroupBarrier();

    if (pos < n) {
        let myWord = lid.x >> 5u;
        let myBit = 1u << (lid.x & 31u);
        var rank_in_row: u32 = 0u;
        for (var w = 0u; w < myWord; w++) {
            let bits = atomicLoad(&bin_flags[digit].words[w]);
            rank_in_row += countOneBits(bits);
        }
        let cur = atomicLoad(&bin_flags[digit].words[myWord]);
        rank_in_row += countOneBits(cur & (myBit - 1u));
        let global_pos = digit_base[digit] + wg_prefixes[digit * infos.sort_dispatch_x + wid.x] + rank_in_row;
        tile_list_dst[global_pos] = tile_list_src[pos];
        pairs_dst[global_pos] = pairs_src[pos];
    }
}
