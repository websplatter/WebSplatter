import { PlyLoader } from './ply-loader.ts';
import preprocess_f16_wgsl from './shaders/preprocess_f16.wgsl';
import preprocess_f16_plyf32_wgsl from './shaders/preprocess_f16_plyf32.wgsl';
import preprocess_f32_wgsl from './shaders/preprocess_f32.wgsl';
import preprocess_f32_plyf32_wgsl from './shaders/preprocess_f32_plyf32.wgsl';
import render_conic_wgsl from './shaders/gaussian_render_conic.wgsl';
import render_conic_plyf32_wgsl from './shaders/gaussian_render_conic_plyf32.wgsl';
import calcIndirectDispatchWGSL from './shaders/gaussian_calculate_indirect_dispatch.wgsl';
import gaussian_cull_wgsl from './shaders/gaussian_cull.wgsl';
import gaussian_cull_plyf32_wgsl from './shaders/gaussian_cull_plyf32.wgsl';
import scatterWGSL from './shaders/gaussian_sort_scatter.wgsl';
import localHistogramWGSL from './shaders/gaussian_sort_local_histogram.wgsl';
import blellochWGSL from './shaders/gaussian_sort_blelloch.wgsl';
import { log } from './utils/simple-console.ts';

const C_WORKGROUP_SIZE_PREPROCESS = 256;
import { RENDER_SETTINGS_SIZE, initRenderSettings, writeRenderSettings, setGaussianScaling, setCurSHDegree } from './render-settings.ts';
import { GaussianRenderers, Renderers } from './splat-app.ts';
import { PadTo4Bytes } from './utils/util.ts';
const C_SIZE_RENDER_SETTINGS_BUFFER = RENDER_SETTINGS_SIZE;
// NOTE: We record 8 timestamps per frame:
//  0: cull begin
//  1: cull end
//  2: preprocess begin
//  3: preprocess end
//  4: sort (radix round 0 local histogram) begin
//  5: sort (final scatter) end
//  6: render begin
//  7: render end
// The old value was 7 which caused ring buffer misalignment (we accessed index 7
// while only reserving 0..6). This produced NaN / negative timings.
const C_TIMESTAMP_COUNT = 8; // Total number of timestamps used in the pipeline
// Splat size depends on ply_precision:
// f32: 6 x f32 = 24 bytes (v_0 as vec2<f32>, v_1 as vec2<f32>, center_ndc as vec2<f32>)
// f16: 3 x u32 = 12 bytes (v_0 packed, v_1 packed, center_ndc packed)
const C_SIZE_2D_SPLAT_F32 = 24;
const C_SIZE_2D_SPLAT_F16 = 12;
// Color size: f32 = 16 bytes (vec4<f32>), f16 = 8 bytes (2 x u32 packed)
const C_SIZE_COLOR_F32 = 16;
const C_SIZE_COLOR_F16 = 8;

// ========================= Inlined sorter (from sort.ts) =========================
interface PingPongBuffers {
    sort_indices_buffer: GPUBuffer; // payload (indices)
    sort_depths_buffer: GPUBuffer;  // keys (depths)
}

interface RadixPassPipelines {
    localHistogram: GPUComputePipeline;
    scatterElements: GPUComputePipeline;
}

interface HierarchicalBlellochPipelines {
    l0TileScan: GPUComputePipeline;
    l1TileScanOnL0: GPUComputePipeline;
    l1ScanSums: GPUComputePipeline;
    addL1ToL0: GPUComputePipeline;
    addL0ToElems: GPUComputePipeline;
    computeDigitBase: GPUComputePipeline;
    prefixBindGroupLayout: GPUBindGroupLayout;
}

interface SortPipelines {
    localHistogramBindGroupLayout: GPUBindGroupLayout;
    scatterBindGroupLayout: GPUBindGroupLayout;
    passes: RadixPassPipelines[];
    hierarchicalBlelloch: HierarchicalBlellochPipelines;
}

// Constants for sorter
const RADIX_BITS_PER_PASS = 8;
const RADIX_SIZE = 1 << RADIX_BITS_PER_PASS; // 256
const WORKGROUP_SIZE = 256;
const TOTAL_PASSES = 32 / RADIX_BITS_PER_PASS; // 4 passes

function createPingPongBuffers(adjusted: number, device: GPUDevice): PingPongBuffers {
    return {
        sort_indices_buffer: device.createBuffer({
            label: 'ping-pong payload (indices)',
            size: adjusted * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        }),
        sort_depths_buffer: device.createBuffer({
            label: 'ping-pong keys (depths)',
            size: adjusted * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        }),
    };
}

function createHierarchicalBlelloch(device: GPUDevice, prefixModule: GPUShaderModule): HierarchicalBlellochPipelines {
    const prefixBindGroupLayout = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        ],
    });
    const prefixLayout = device.createPipelineLayout({ bindGroupLayouts: [prefixBindGroupLayout] });
    const mk = (entryPoint: string) => device.createComputePipeline({
        layout: prefixLayout,
        compute: {
            module: prefixModule,
            entryPoint,
            constants: {
                WG_SIZE: WORKGROUP_SIZE,
                // RS_RADIX_SIZE: RADIX_SIZE
            }
        },
    });
    return {
        l0TileScan: mk('prefix_l0_tile_scan'),
        l1TileScanOnL0: mk('prefix_l1_tile_scan_on_l0_sums'),
        l1ScanSums: mk('prefix_scan_l1_sums'),
        addL1ToL0: mk('prefix_add_l1_to_l0_offsets'),
        addL0ToElems: mk('prefix_add_l0_to_elements'),
        computeDigitBase: mk('compute_digit_base'),
        prefixBindGroupLayout,
    };
}

function createRadixPassPipelines(device: GPUDevice, histogramModule: GPUShaderModule, scatterModule: GPUShaderModule) {
    const localHistogramBindGroupLayout = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        ],
    });
    const scatterBindGroupLayout = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        ],
    });
    const histogramLayout = device.createPipelineLayout({ bindGroupLayouts: [localHistogramBindGroupLayout] });
    const scatterLayout = device.createPipelineLayout({ bindGroupLayouts: [scatterBindGroupLayout] });
    const passes: RadixPassPipelines[] = [];
    for (let i = 0; i < TOTAL_PASSES; i++) {
        const constants = { PASS_ID: i, RS_RADIX_LOG2: RADIX_BITS_PER_PASS, RS_RADIX_SIZE: RADIX_SIZE };
        passes.push({
            localHistogram: device.createComputePipeline({
                layout: histogramLayout,
                compute: { module: histogramModule, entryPoint: 'local_histogram_pass', constants },
            }),
            scatterElements: device.createComputePipeline({
                layout: scatterLayout,
                compute: { module: scatterModule, entryPoint: 'scatter_elements', constants },
            }),
        });
    }
    return { passes, localHistogramBindGroupLayout, scatterBindGroupLayout };
}

function buildPipelines(device: GPUDevice): SortPipelines {
    const histogramModule = device.createShaderModule({ label: 'local histogram', code: localHistogramWGSL });
    const scatterModule = device.createShaderModule({ label: 'scatter', code: scatterWGSL });
    const prefixModule = device.createShaderModule({ label: 'blelloch prefix', code: blellochWGSL });
    const hierarchicalBlelloch = createHierarchicalBlelloch(device, prefixModule);
    const radix = createRadixPassPipelines(device, histogramModule, scatterModule);
    return {
        localHistogramBindGroupLayout: radix.localHistogramBindGroupLayout,
        scatterBindGroupLayout: radix.scatterBindGroupLayout,
        passes: radix.passes,
        hierarchicalBlelloch,
    };
}

// ======================= End inlined sorter (from sort.ts) =======================

/**
 * Implements the full rendering pipeline for Gaussian Splatting.
 * This class handles preprocessing, sorting, and rendering the splats.
 * It also includes performance metric tracking using timestamp queries.
 */
export class GaussianRenderer implements GaussianRenderers {
    private readonly device: GPUDevice;
    private pc: PlyLoader;
    private readonly presentationFormat: GPUTextureFormat;

    // --- Core Resources ---
    public camera_buffer: GPUBuffer;
    public render_settings_buffer: GPUBuffer;
    private draw_indirect_buffer: GPUBuffer;
    private splat_2d_buffer: GPUBuffer;

    // --- Timestamp Query Resources (only when timestamp-query is enabled) ---
    private readonly timeQueryEnabled: boolean;
    private querySet!: GPUQuerySet;
    private resolveBuffer!: GPUBuffer;
    private resultBuffer!: GPUBuffer;

    // --- Timestamp Ring Buffer Controls ---
    private readonly queriesPerFrame: number = C_TIMESTAMP_COUNT;
    private readonly queryCapacityFrames: number = 200; // 100 × N
    private readonly sort_prefixBindGroup: GPUBindGroup;
    private readonly sort_pipelines: SortPipelines;
    private readonly sort_localHistogramBindGroups: GPUBindGroup[];
    private readonly sort_scatterBindGroups: GPUBindGroup[];
    private get totalQueryCount(): number { return this.queriesPerFrame * this.queryCapacityFrames; }
    // Ring write/read tracking
    private lastFrame: number = 0;   // total frames rendered since start (monotonic)
    private frameCount: number = 0;  // frames rendered since last read

    // --- Child Modules & Pipelines ---
    // private readonly sorter: SortStuff;
    private preprocessPipeline: GPUComputePipeline;
    private cullPipeline: GPUComputePipeline;
    private renderPipeline: GPURenderPipeline;
    private indirectPipeline: GPUComputePipeline;


    private sort_info_buffer: GPUBuffer;
    private sort_ping_pong: PingPongBuffers[];

    // --- Bind Groups ---
    private crsBg: GPUBindGroup;
    private gsBg: GPUBindGroup;
    private cullBg2: GPUBindGroup;
    private preprocessBg1: GPUBindGroup;
    private renderSplatsBindGroup: GPUBindGroup;
    private indirectBindGroup: GPUBindGroup;

    // Buffers for culling/compaction
    private sh_color_rgba_buffer: GPUBuffer;
    private sh_solvers_buffer: GPUBuffer;
    // Per-splat sizes (vary by ply_precision)
    private readonly splatSize: number;
    private readonly colorSize: number;
    // One-shot perf dialog flag
    private showPerfDialogNext: boolean = false;
    // One-shot CPU-side reorder trigger flags
    private requestReorderNextFrame: boolean = false;
    private reorderInFlight: boolean = false;
    // One-shot CSV download trigger for next readPerfMetrics
    private downloadOnceNextRead: boolean = false;
    private downloadOnceFileName: string = 'fps_metrics';

    private allFrameTimes: number[] = [];


    constructor(
        pc: PlyLoader,
        device: GPUDevice,
        presentationFormat: GPUTextureFormat,
        camera_buffer: GPUBuffer,
        features_list: GPUFeatureName[],
        disableAabbCull = false,
        disableOpacityRadius = false,
        enableF16Feature = true,
    ) {
        this.pc = pc;
        this.device = device;
        const f16_enabled = features_list.includes('shader-f16') && enableF16Feature;
        this.timeQueryEnabled = features_list.includes('timestamp-query');

        const plyF16 = (this.pc.ply_precision ?? 'f16') === 'f16';
        this.splatSize = plyF16 ? C_SIZE_2D_SPLAT_F16 : C_SIZE_2D_SPLAT_F32;
        this.colorSize = plyF16 ? C_SIZE_COLOR_F16 : C_SIZE_COLOR_F32;
        log(`🚀 enable_f16_feature: ${f16_enabled}`);
        log(`💾 ply_precision: ${this.pc.ply_precision ?? 'f16'}`);
        const cullName = plyF16 ? 'gaussian_cull.wgsl' : 'gaussian_cull_plyf32.wgsl';
        const preprocessName = f16_enabled
            ? (plyF16 ? 'preprocess_f16.wgsl' : 'preprocess_f16_plyf32.wgsl')
            : (plyF16 ? 'preprocess_f32.wgsl' : 'preprocess_f32_plyf32.wgsl');
        const renderName = plyF16 ? 'gaussian_render_conic.wgsl' : 'gaussian_render_conic_plyf32.wgsl';
        log(`📦 shaders: ${cullName}, ${preprocessName}, ${renderName}`);
        this.presentationFormat = presentationFormat;
        this.camera_buffer = camera_buffer;

        device.addEventListener('uncapturederror', (event) => {
            // Re-surface the error.
            console.error('A WebGPU error was not captured:', event.error);
        });

        if (this.timeQueryEnabled) {
            this._setupTimestampQueries();
        }
        this._setupBuffers(pc.sh_degree);

        const adjustedKeyCount = (Math.floor((this.pc.num_points + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE) + 1) * WORKGROUP_SIZE;
        const maxNumWorkgroups = Math.ceil(adjustedKeyCount / WORKGROUP_SIZE);
        console.log(`keys count adjusted: ${adjustedKeyCount}`);
        console.log(`key size: ${this.pc.num_points}`);

        // sort_info_buffer (uint32 * 12): keys_size + dispatch triples & hierarchical prefix meta
        const sortInfoBuffer = device.createBuffer({
            label: 'sort info',
            size: 16 * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | GPUBufferUsage.INDIRECT,
        });

        this.sort_pipelines = buildPipelines(device);
        const ping_pong: PingPongBuffers[] = [
            createPingPongBuffers(adjustedKeyCount, device),
            createPingPongBuffers(adjustedKeyCount, device),
        ];

        // Intermediate buffers
        const wgHistograms = device.createBuffer({
            label: 'workgroup histograms',
            size: maxNumWorkgroups * RADIX_SIZE * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
        const wgPrefixes = device.createBuffer({
            label: 'workgroup prefixes',
            size: maxNumWorkgroups * RADIX_SIZE * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
        const digitBase = device.createBuffer({
            label: 'digit base',
            size: RADIX_SIZE * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });

        const T0Max = Math.ceil(maxNumWorkgroups / WORKGROUP_SIZE);
        const T1Max = Math.ceil(T0Max / WORKGROUP_SIZE); // Constraint: T1 <= 256
        const l0Sums = device.createBuffer({ label: 'prefix l0 sums', size: T0Max * RADIX_SIZE * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
        const l0Offsets = device.createBuffer({ label: 'prefix l0 offsets', size: T0Max * RADIX_SIZE * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
        const l1Sums = device.createBuffer({ label: 'prefix l1 sums', size: T1Max * RADIX_SIZE * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
        const l1Offsets = device.createBuffer({ label: 'prefix l1 offsets', size: T1Max * RADIX_SIZE * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });

        this.sort_prefixBindGroup = device.createBindGroup({
            label: 'prefix 2L bind group',
            layout: this.sort_pipelines.hierarchicalBlelloch.prefixBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: sortInfoBuffer } },
                { binding: 1, resource: { buffer: wgHistograms } },
                { binding: 2, resource: { buffer: wgPrefixes } },
                { binding: 3, resource: { buffer: l0Sums } },
                { binding: 4, resource: { buffer: l0Offsets } },
                { binding: 5, resource: { buffer: l1Sums } },
                { binding: 6, resource: { buffer: l1Offsets } },
                { binding: 7, resource: { buffer: digitBase } },
            ],
        });

        this.sort_localHistogramBindGroups = [
            device.createBindGroup({
                label: 'localHistogram src=0',
                layout: this.sort_pipelines.localHistogramBindGroupLayout,
                entries: [
                    { binding: 0, resource: { buffer: sortInfoBuffer } },
                    { binding: 1, resource: { buffer: ping_pong[0].sort_depths_buffer } },
                    { binding: 2, resource: { buffer: wgHistograms } },
                ],
            }),
            device.createBindGroup({
                label: 'localHistogram src=1',
                layout: this.sort_pipelines.localHistogramBindGroupLayout,
                entries: [
                    { binding: 0, resource: { buffer: sortInfoBuffer } },
                    { binding: 1, resource: { buffer: ping_pong[1].sort_depths_buffer } },
                    { binding: 2, resource: { buffer: wgHistograms } },
                ],
            }),
        ];

        this.sort_scatterBindGroups = [
            device.createBindGroup({
                label: 'scatter 0->1',
                layout: this.sort_pipelines.scatterBindGroupLayout,
                entries: [
                    { binding: 0, resource: { buffer: sortInfoBuffer } },
                    { binding: 1, resource: { buffer: digitBase } },
                    { binding: 2, resource: { buffer: ping_pong[0].sort_depths_buffer } },
                    { binding: 3, resource: { buffer: ping_pong[1].sort_depths_buffer } },
                    { binding: 4, resource: { buffer: ping_pong[0].sort_indices_buffer } },
                    { binding: 5, resource: { buffer: ping_pong[1].sort_indices_buffer } },
                    { binding: 6, resource: { buffer: wgPrefixes } },
                ],
            }),
            device.createBindGroup({
                label: 'scatter 1->0',
                layout: this.sort_pipelines.scatterBindGroupLayout,
                entries: [
                    { binding: 0, resource: { buffer: sortInfoBuffer } },
                    { binding: 1, resource: { buffer: digitBase } },
                    { binding: 2, resource: { buffer: ping_pong[1].sort_depths_buffer } },
                    { binding: 3, resource: { buffer: ping_pong[0].sort_depths_buffer } },
                    { binding: 4, resource: { buffer: ping_pong[1].sort_indices_buffer } },
                    { binding: 5, resource: { buffer: ping_pong[0].sort_indices_buffer } },
                    { binding: 6, resource: { buffer: wgPrefixes } },
                ],
            }),
        ];

        this.sort_info_buffer = sortInfoBuffer;
        this.sort_ping_pong = ping_pong;

        const camera_renderSettings_Bgl = this.device.createBindGroupLayout({
            label: 'camera + renderSettings',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ],
        });
        const gaussians_splats_Bgl = this.device.createBindGroupLayout({
            label: 'gaussians + splats',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            ],
        });
        const cullBgl2 = this.device.createBindGroupLayout({
            label: 'cullBgl2',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            ],
        });
        const preprocessBgl2 = this.device.createBindGroupLayout({
            label: 'preprocessBgl2',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            ],
        });
        this.crsBg = this.device.createBindGroup({
            label: 'camera + renderSettings',
            layout: camera_renderSettings_Bgl,
            entries: [
                { binding: 0, resource: { buffer: this.camera_buffer } },
                { binding: 1, resource: { buffer: this.render_settings_buffer } },
            ],
        });
        this.gsBg = this.device.createBindGroup({
            label: 'gaussians + splats',
            layout: gaussians_splats_Bgl,
            entries: [
                { binding: 0, resource: { buffer: this.pc.gaussian_3d_buffer } },
                { binding: 1, resource: { buffer: this.splat_2d_buffer } },
            ],
        });
        this.cullBg2 = this.device.createBindGroup({
            label: 'cullBg2',
            layout: cullBgl2,
            entries: [
                { binding: 0, resource: { buffer: this.sort_info_buffer } },
                { binding: 1, resource: { buffer: this.sort_ping_pong[0].sort_depths_buffer } },
                { binding: 2, resource: { buffer: this.sort_ping_pong[0].sort_indices_buffer } },
                { binding: 3, resource: { buffer: this.sh_solvers_buffer } },
            ],
        });
        // @group(2) @binding(0) var<storage, read> sort_infos: GeneralInfo;
        // @group(2) @binding(1) var<storage, read> sh_coefs : array<f16>;
        // @group(2) @binding(2) var<storage, read> sh_solvers : array<SHSolver>;
        this.preprocessBg1 = this.device.createBindGroup({
            label: 'preprocessBg2',
            layout: preprocessBgl2,
            entries: [
                { binding: 0, resource: { buffer: this.sort_info_buffer } },
                { binding: 1, resource: { buffer: this.pc.sh_buffer } },
                { binding: 2, resource: { buffer: this.sh_solvers_buffer } },
                { binding: 3, resource: { buffer: this.sh_color_rgba_buffer } },
            ],
        });

        // Indirect dispatch pipeline for sort parameters (decoupled from sort)
        const indirectModule = this.device.createShaderModule({ code: calcIndirectDispatchWGSL });
        this.indirectPipeline = this.device.createComputePipeline({
            label: 'indirect dispatch calc',
            layout: 'auto',
            compute: { module: indirectModule, entryPoint: 'write_dispatch_triples', constants: { RS_RADIX_SIZE: 256 } },
        });

        // Bind group for indirect dispatch calc (expects sort_info_buffer at binding 0)
        this.indirectBindGroup = this.device.createBindGroup({
            label: 'indirect dispatch bind group',
            layout: this.indirectPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.sort_info_buffer } },
                { binding: 1, resource: { buffer: this.draw_indirect_buffer } },
            ],
        });

        const cullShaderCode = plyF16 ? gaussian_cull_wgsl : gaussian_cull_plyf32_wgsl;
        const preprocessCullShader = this.device.createShaderModule({ code: cullShaderCode });

        // Culling & compaction pipeline
        this.cullPipeline = this.device.createComputePipeline({
            label: 'preprocess_cull',
            layout: this.device.createPipelineLayout({
                bindGroupLayouts: [
                    camera_renderSettings_Bgl, // 0: camera + render settings
                    gaussians_splats_Bgl, // 1: cull gaussians
                    cullBgl2, // 2: cull sort/active indices
                ]
            }),
            compute: {
                module: preprocessCullShader,
                entryPoint: 'preprocess_cull',
                constants: {
                    DISABLE_AABB_CULL: disableAabbCull ? 1 : 0,
                    DISABLE_OPACITY_RADIUS: disableOpacityRadius ? 1 : 0,
                },
            },
        });

        const preprocessCode = f16_enabled
            ? (plyF16 ? preprocess_f16_wgsl : preprocess_f16_plyf32_wgsl)
            : (plyF16 ? preprocess_f32_wgsl : preprocess_f32_plyf32_wgsl);
        const preprocessShader = this.device.createShaderModule({ code: preprocessCode });

        this.preprocessPipeline = this.device.createComputePipeline({
            label: 'preprocess',
            layout: this.device.createPipelineLayout({
                bindGroupLayouts: [
                    camera_renderSettings_Bgl,
                    preprocessBgl2,
                ]
            }),
            compute: {
                module: preprocessShader,
                entryPoint: 'preprocess',
            },
        });

        const renderCode = plyF16 ? render_conic_wgsl : render_conic_plyf32_wgsl;
        const renderShader = this.device.createShaderModule({ code: renderCode });
        const renderConstants = {
            DISABLE_OPACITY_RADIUS: disableOpacityRadius ? 1 : 0,
        };
        this.renderPipeline = this.device.createRenderPipeline({
            label: 'render',
            layout: 'auto',
            vertex: {
                module: renderShader,
                entryPoint: 'vs_main',
                constants: renderConstants,
            },
            fragment: {
                module: renderShader,
                entryPoint: 'fs_main',
                constants: renderConstants,
                targets: [{
                    format: 'rgba16float', // HDR target to avoid clamping during blending
                    blend: {
                        color: { operation: 'add', srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
                        alpha: { operation: 'add', srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
                    },
                }],
            },
            primitive: {
                topology: 'triangle-strip',
                cullMode: 'none',
            },
        });

        this.renderSplatsBindGroup = this.device.createBindGroup({
            label: 'gaussian splats rendering',
            layout: this.renderPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.splat_2d_buffer } },
                { binding: 1, resource: { buffer: this.sh_color_rgba_buffer } },
                { binding: 2, resource: { buffer: this.sort_ping_pong[0].sort_indices_buffer } },
            ],
        });
    }

    /**
     * Debug helper: read first N sorted indices and print them.
     */
    public async debugReadSortedIndices(count: number = 30): Promise<void> {
        const maxCount = Math.max(0, Math.min(count, this.pc.num_points));
        const bytes = maxCount * Uint32Array.BYTES_PER_ELEMENT;
        if (bytes === 0) {
            console.log('[DEBUG] No indices to read.');
            return;
        }
        const readBuffer = this.device.createBuffer({
            size: bytes,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        const encoder = this.device.createCommandEncoder();
        encoder.copyBufferToBuffer(
            this.sort_ping_pong[0].sort_indices_buffer,
            0,
            readBuffer,
            0,
            bytes
        );
        this.device.queue.submit([encoder.finish()]);
        await readBuffer.mapAsync(GPUMapMode.READ);
        const arr = new Uint32Array(readBuffer.getMappedRange());
        console.log('[DEBUG] Sorted indices (first', maxCount, '):', Array.from(arr));
        readBuffer.unmap();
    }

    /**
     * Read back real GPU buffers for one pixel and return only splats that overlap that pixel.
     * Uses the actual sorted draw order, compacted points_2d buffer, and colors buffer.
     */
    public async debugDumpPixelInputs(px: number, py: number, width: number, height: number, limit: number = 10000): Promise<any> {
        const num = this.pc.num_points;
        if (!num) return { alive: 0, items: [] };

        const bytesSortInfo = 16 * 4;
        const readSortInfo = this.device.createBuffer({ size: bytesSortInfo, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        const enc0 = this.device.createCommandEncoder();
        enc0.copyBufferToBuffer(this.sort_info_buffer, 0, readSortInfo, 0, bytesSortInfo);
        this.device.queue.submit([enc0.finish()]);
        await this.device.queue.onSubmittedWorkDone();
        await readSortInfo.mapAsync(GPUMapMode.READ);
        const infoU32 = new Uint32Array(readSortInfo.getMappedRange());
        const alive = Math.min(infoU32[0] ?? 0, num);
        readSortInfo.unmap();
        readSortInfo.destroy();
        if (!alive) return { alive: 0, items: [] };

        const SH_SOLVER_STRIDE = 12;
        const COLOR_STRIDE = this.colorSize;
        const readSortedIdx = this.device.createBuffer({ size: alive * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        const readPoints = this.device.createBuffer({ size: alive * this.splatSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        const readColors = this.device.createBuffer({ size: alive * COLOR_STRIDE, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        const readSolvers = this.device.createBuffer({ size: alive * SH_SOLVER_STRIDE, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

        const enc = this.device.createCommandEncoder();
        enc.copyBufferToBuffer(this.sort_ping_pong[0].sort_indices_buffer, 0, readSortedIdx, 0, alive * 4);
        enc.copyBufferToBuffer(this.splat_2d_buffer, 0, readPoints, 0, alive * this.splatSize);
        enc.copyBufferToBuffer(this.sh_color_rgba_buffer, 0, readColors, 0, alive * COLOR_STRIDE);
        enc.copyBufferToBuffer(this.sh_solvers_buffer, 0, readSolvers, 0, alive * SH_SOLVER_STRIDE);
        this.device.queue.submit([enc.finish()]);
        await this.device.queue.onSubmittedWorkDone();

        await Promise.all([
            readSortedIdx.mapAsync(GPUMapMode.READ),
            readPoints.mapAsync(GPUMapMode.READ),
            readColors.mapAsync(GPUMapMode.READ),
            readSolvers.mapAsync(GPUMapMode.READ),
        ]);

        const sortedIdx = new Uint32Array(readSortedIdx.getMappedRange());
        const pointsView = new DataView(readPoints.getMappedRange());
        const colorsView = new DataView(readColors.getMappedRange());
        const solversView = new DataView(readSolvers.getMappedRange());

        const w = width || 1;
        const h = height || 1;
        const targetNdcX = ((2 * px + 1) / w) - 1;
        const targetNdcY = 1 - ((2 * py + 1) / h);

        const halfToFloat = (h: number): number => {
            const s = (h & 0x8000) ? -1 : 1;
            const e = (h >> 10) & 0x1f;
            const f = h & 0x03ff;
            if (e === 0) return f ? s * Math.pow(2, -14) * (f / 1024) : s * 0;
            if (e === 31) return f ? NaN : s * Infinity;
            return s * Math.pow(2, e - 15) * (1 + f / 1024);
        };
        const unpack2x16 = (packed: number): [number, number] => {
            const lo = packed & 0xffff;
            const hi = (packed >>> 16) & 0xffff;
            return [halfToFloat(lo), halfToFloat(hi)];
        };

        const items: any[] = [];
        for (let drawOrder = 0; drawOrder < alive; drawOrder++) {
            const compactIdx = sortedIdx[drawOrder] >>> 0;
            if (compactIdx >= alive) continue;

            const pOff = compactIdx * this.splatSize;
            const cOff = compactIdx * COLOR_STRIDE;
            const sOff = compactIdx * SH_SOLVER_STRIDE;

            // Read oriented-quad Splat struct (layout depends on ply_precision)
            let v0: [number, number], v1: [number, number], centerNdc: [number, number];
            if (this.pc.ply_precision === 'f16') {
                // f16 layout: 3 x u32 = 12 bytes (v_0, v_1, center_ndc packed as f16x2)
                const v0f = new Float16Array(pointsView.buffer, pointsView.byteOffset + pOff + 0, 2);
                const v1f = new Float16Array(pointsView.buffer, pointsView.byteOffset + pOff + 4, 2);
                const cf = new Float16Array(pointsView.buffer, pointsView.byteOffset + pOff + 8, 2);
                v0 = [Number(v0f[0]), Number(v0f[1])];
                v1 = [Number(v1f[0]), Number(v1f[1])];
                centerNdc = [Number(cf[0]), Number(cf[1])];
            } else {
                // f32 layout: 6 x f32 = 24 bytes
                v0 = [pointsView.getFloat32(pOff + 0, true), pointsView.getFloat32(pOff + 4, true)];
                v1 = [pointsView.getFloat32(pOff + 8, true), pointsView.getFloat32(pOff + 12, true)];
                centerNdc = [pointsView.getFloat32(pOff + 16, true), pointsView.getFloat32(pOff + 20, true)];
            }

            // Read color (layout depends on ply_precision)
            let rg: [number, number], ba: [number, number];
            if (this.pc.ply_precision === 'f16') {
                const c = new Float16Array(colorsView.buffer, colorsView.byteOffset + cOff, 4);
                rg = [Number(c[0]), Number(c[1])];
                ba = [Number(c[2]), Number(c[3])];
            } else {
                rg = [colorsView.getFloat32(cOff + 0, true), colorsView.getFloat32(cOff + 4, true)];
                ba = [colorsView.getFloat32(cOff + 8, true), colorsView.getFloat32(cOff + 12, true)];
            }
            const origIdx = solversView.getUint32(sOff + 8, true) >>> 0;
            const gsStride = this.pc.gs_stride;
            const gOff = origIdx * gsStride;
            const covCpu = [
                Number(this.pc.gaussian_cpu[gOff + 4]),
                Number(this.pc.gaussian_cpu[gOff + 5]),
                Number(this.pc.gaussian_cpu[gOff + 6]),
                Number(this.pc.gaussian_cpu[gOff + 7]),
                Number(this.pc.gaussian_cpu[gOff + 8]),
                Number(this.pc.gaussian_cpu[gOff + 9]),
            ];

            const rgba = [rg[0], rg[1], ba[0], ba[1]];
            const opacity = rgba[3];
            if (!(opacity > 1 / 255)) continue;

            // Compute radius_scale from opacity
            const rs = Math.sqrt(Math.log(Math.max(1, 255 * opacity)));

            // Convert target pixel to NDC
            const targetPxX = px + 0.5;
            const targetPxY = py + 0.5;
            const pixelNdcX = targetPxX * 2.0 / w - 1.0;
            const pixelNdcY = 1.0 - targetPxY * 2.0 / h;
            const diffX = pixelNdcX - centerNdc[0];
            const diffY = pixelNdcY - centerNdc[1];

            // Invert effective 2x2 matrix [v0*rs | v1*rs] to get UV
            const effV0x = v0[0] * rs, effV0y = v0[1] * rs;
            const effV1x = v1[0] * rs, effV1y = v1[1] * rs;
            const det = effV0x * effV1y - effV0y * effV1x;
            if (Math.abs(det) < 1e-12) continue;
            const inv_det = 1.0 / det;
            const uv_x = (effV1y * diffX - effV1x * diffY) * inv_det;
            const uv_y = (-effV0y * diffX + effV0x * diffY) * inv_det;
            const sigma_sq = uv_x * uv_x + uv_y * uv_y;
            if (sigma_sq > 2.0) continue; // outside quad (corners at sigma_sq=2)

            const alpha = Math.min(0.99, opacity * Math.exp(-Math.log(Math.max(1, 255 * opacity)) * sigma_sq));
            if (alpha <= 1 / 255) continue;

            items.push({
                drawOrder,
                compactIdx,
                origIdx,
                covCpu,
                centerNdc,
                v0, v1,
                rgba,
                uv: [uv_x, uv_y],
                sigma_sq,
                alpha,
            });
            if (items.length >= limit) break;
        }

        readSortedIdx.unmap(); readPoints.unmap(); readColors.unmap(); readSolvers.unmap();
        readSortedIdx.destroy(); readPoints.destroy(); readColors.destroy(); readSolvers.destroy();

        return {
            px, py, width: w, height: h, alive,
            items,
        };
    }

    /**
     * Executes a single frame of rendering.
     * @param encoder - The command encoder for this frame.
     * @param texture_view - The canvas texture view to render to.
     */
    public frame(encoder: GPUCommandEncoder, texture_view: GPUTextureView): void {
        // Base offset into the query ring for this frame: (lastFrame + frameCount) % capacity
        const tq = this.timeQueryEnabled;
        const offsetFrameIndex = (this.lastFrame + this.frameCount) % this.queryCapacityFrames;
        const base = offsetFrameIndex * this.queriesPerFrame;
        // First do culling and index compaction, then write indirect dispatch triples based on keys_size
        {
            encoder.clearBuffer(this.sort_info_buffer, 0, 4);
            const pass = encoder.beginComputePass({
                label: 'cull',
                timestampWrites: tq ? {
                    querySet: this.querySet,
                    beginningOfPassWriteIndex: base + 0, // T0: cull start
                    endOfPassWriteIndex: base + 1, // T1: cull end
                } : undefined,
            });
            pass.setPipeline(this.cullPipeline);
            pass.setBindGroup(0, this.crsBg);
            pass.setBindGroup(1, this.gsBg);
            pass.setBindGroup(2, this.cullBg2);
            const wgCount = Math.ceil(this.pc.num_points / C_WORKGROUP_SIZE_PREPROCESS);
            pass.dispatchWorkgroups(wgCount, 1, 1);
            pass.end();
        }
        {
            const pass = encoder.beginComputePass({ label: 'calculate indirect dispatch' });
            pass.setPipeline(this.indirectPipeline);
            pass.setBindGroup(0, this.indirectBindGroup);
            pass.dispatchWorkgroups(1, 1, 1);
            pass.end();
        }
        // Execute preprocess using indirect dispatch
        {
            const pass = encoder.beginComputePass({
                label: 'preprocess',
                timestampWrites: tq ? {
                    querySet: this.querySet,
                    beginningOfPassWriteIndex: base + 2, // T2: preprocess start
                    endOfPassWriteIndex: base + 3, // T3: preprocess end
                } : undefined,
            });
            pass.setPipeline(this.preprocessPipeline);
            pass.setBindGroup(0, this.crsBg);
            pass.setBindGroup(1, this.preprocessBg1);
            // dispatch using indirect triples written by the indirect pass
            pass.dispatchWorkgroupsIndirect(this.sort_info_buffer, 4);
            pass.end();
        }
        for (let round = 0; round < TOTAL_PASSES; round++) {
            const flip = round & 1;
            const radixPipelines = this.sort_pipelines.passes[round];
            const histogramBG = this.sort_localHistogramBindGroups[flip];
            const scatterBG = this.sort_scatterBindGroups[flip];

            // Upsweep / Local histogram
            {
                const pass = encoder.beginComputePass({ label: `upsweep_round${round}` , timestampWrites: (tq && round == 0) ? {
                    querySet: this.querySet,
                    beginningOfPassWriteIndex: base + 4, // T4: l0TileScan start
                } : undefined });
                pass.setPipeline(radixPipelines.localHistogram);
                pass.setBindGroup(0, histogramBG);
                pass.dispatchWorkgroupsIndirect(this.sort_info_buffer, 4);
                pass.end();
            }
            // Hierarchical prefix (6 sub-passes)
            {
                const pass = encoder.beginComputePass({ label: `prefix_round${round} - l0TileScan` });
                pass.setPipeline(this.sort_pipelines.hierarchicalBlelloch.l0TileScan);
                pass.setBindGroup(0, this.sort_prefixBindGroup);
                pass.dispatchWorkgroupsIndirect(this.sort_info_buffer, 16);
                pass.end();
            }
            {
                const pass = encoder.beginComputePass({ label: `prefix_round${round} - l1TileScanOnL0` });
                pass.setPipeline(this.sort_pipelines.hierarchicalBlelloch.l1TileScanOnL0);
                pass.setBindGroup(0, this.sort_prefixBindGroup);
                pass.dispatchWorkgroupsIndirect(this.sort_info_buffer, 32);
                pass.end();
            }
            {
                const pass = encoder.beginComputePass({ label: `prefix_round${round} - l1ScanSums` });
                pass.setPipeline(this.sort_pipelines.hierarchicalBlelloch.l1ScanSums);
                pass.setBindGroup(0, this.sort_prefixBindGroup);
                pass.dispatchWorkgroups(1, RADIX_SIZE, 1);
                pass.end();
            }
            {
                const pass = encoder.beginComputePass({ label: `prefix_round${round} - addL1ToL0` });
                pass.setPipeline(this.sort_pipelines.hierarchicalBlelloch.addL1ToL0);
                pass.setBindGroup(0, this.sort_prefixBindGroup);
                pass.dispatchWorkgroupsIndirect(this.sort_info_buffer, 32);
                pass.end();
            }
            {
                const pass = encoder.beginComputePass({ label: `prefix_round${round} - addL0ToElems` });
                pass.setPipeline(this.sort_pipelines.hierarchicalBlelloch.addL0ToElems);
                pass.setBindGroup(0, this.sort_prefixBindGroup);
                pass.dispatchWorkgroupsIndirect(this.sort_info_buffer, 16);
                pass.end();
            }
            {
                const pass = encoder.beginComputePass({ label: `prefix_round${round} - computeDigitBase` });
                pass.setPipeline(this.sort_pipelines.hierarchicalBlelloch.computeDigitBase);
                pass.setBindGroup(0, this.sort_prefixBindGroup);
                pass.dispatchWorkgroups(1, 1, 1);
                pass.end();
            }
            // Scatter
            {
                const pass = encoder.beginComputePass({ label: `scatter_round${round}` , timestampWrites: (tq && round == TOTAL_PASSES - 1) ? {
                    querySet: this.querySet,
                    endOfPassWriteIndex: base + 5, // T5: scatter end
                } : undefined });
                pass.setPipeline(radixPipelines.scatterElements);
                pass.setBindGroup(0, scatterBG);
                pass.dispatchWorkgroupsIndirect(this.sort_info_buffer, 4);
                pass.end();
            }
        }
        {
            const pass = encoder.beginRenderPass({
                label: 'render',
                colorAttachments: [
                    {
                        view: texture_view,
                        loadOp: 'clear',
                        storeOp: 'store',
                        clearValue: [0, 0, 0, 0]
                    }
                ],
                timestampWrites: tq ? {
                    querySet: this.querySet,
                    beginningOfPassWriteIndex: base + 6, // T6: render start
                    endOfPassWriteIndex: base + 7, // T7: render end
                } : undefined,
            });
            pass.setPipeline(this.renderPipeline);
            pass.setBindGroup(0, this.renderSplatsBindGroup);
            pass.drawIndirect(this.draw_indirect_buffer, 0);
            pass.end();
        }

        // account this frame for the current interval
        if (tq) this.frameCount++;
    }

    /**
     * Read per-frame stage breakdown for eval=3 mode.
     * Returns arrays of per-frame timings for each pipeline stage.
     * Requires timestamp-query to be enabled.
     */
    public async readBreakdown(): Promise<{
        cull_ms: number[],
        preprocess_ms: number[],
        sort_ms: number[],
        render_ms: number[],
    }> {
        if (!this.timeQueryEnabled) {
            throw new Error('readBreakdown() requires timestamp-query feature');
        }

        const encoder = this.device.createCommandEncoder({ label: 'breakdown resolve' });
        encoder.resolveQuerySet(this.querySet, 0, this.totalQueryCount, this.resolveBuffer, 0);
        encoder.copyBufferToBuffer(this.resolveBuffer, 0, this.resultBuffer, 0, this.totalQueryCount * 8);
        this.device.queue.submit([encoder.finish()]);
        await this.device.queue.onSubmittedWorkDone();

        await this.resultBuffer.mapAsync(GPUMapMode.READ);
        const times = new BigInt64Array(this.resultBuffer.getMappedRange());

        const framesToRead = Math.min(this.frameCount, this.queryCapacityFrames);
        const startSlot = (this.lastFrame + this.frameCount - framesToRead) % this.queryCapacityFrames;

        const cull_ms: number[] = [];
        const preprocess_ms: number[] = [];
        const sort_ms: number[] = [];
        const render_ms: number[] = [];

        for (let f = 0; f < framesToRead; f++) {
            const slot = (startSlot + f) % this.queryCapacityFrames;
            const b = slot * this.queriesPerFrame;
            // Validate timestamps
            if (times[b] === 0n || times[b + 7] === 0n || times[b + 7] < times[b]) continue;
            cull_ms.push(Number(times[b + 1] - times[b + 0]) / 1e6);
            preprocess_ms.push(Number(times[b + 3] - times[b + 2]) / 1e6);
            sort_ms.push(Number(times[b + 5] - times[b + 4]) / 1e6);
            render_ms.push(Number(times[b + 7] - times[b + 6]) / 1e6);
        }

        this.resultBuffer.unmap();
        // Reset counters
        this.lastFrame += this.frameCount;
        this.frameCount = 0;

        return { cull_ms, preprocess_ms, sort_ms, render_ms };
    }

    /**
     * Reads the performance metrics from the last completed frame.
     * This function is async and will resolve when the data is available on the CPU.
     */
    public async readPerfMetrics(): Promise<void> {
        if (!this.timeQueryEnabled) {
            console.warn('[readPerfMetrics] timestamp-query not enabled, skipping');
            return;
        }
        const encoder = this.device.createCommandEncoder({
            label: 'timestamp resolve encoder'
        });
        // Resolve the entire query ring buffer into resolveBuffer
        encoder.resolveQuerySet(
            this.querySet,
            0,
            this.totalQueryCount,
            this.resolveBuffer,
            0
        );

        // Copy the resolved data from resolveBuffer to resultBuffer for reading
        encoder.copyBufferToBuffer(
            this.resolveBuffer, 0,
            this.resultBuffer, 0,
            this.totalQueryCount * 8
        );
        this.device.queue.submit([encoder.finish()]);
        await this.device.queue.onSubmittedWorkDone();

        // Stages layout within one frame worth of timestamps
        const stages: [string, number, number][] = [
            ['Total', 7, 0],
            ['Culling', 1, 0],
            ['Preprocess', 3, 2],
            ['Sort', 5, 4],
            ['Render', 7, 6],
        ];

        // If no new frames since last output, skip
        if (this.frameCount <= 0) {
            return;
        }

        // Wait for resultBuffer mapping to complete
        await this.resultBuffer.mapAsync(GPUMapMode.READ);
        const times = new BigInt64Array(this.resultBuffer.getMappedRange());

        // Determine how many frames to average this time
        const framesToRead = Math.min(this.frameCount, this.queryCapacityFrames);
        // Oldest frame slot in this window within the ring
        const startSlot = (this.lastFrame + this.frameCount - framesToRead) % this.queryCapacityFrames;

        const stageSamples: number[][] = Array.from({ length: stages.length }, () => []);
        let validFrames = 0;
        for (let f = 0; f < framesToRead; f++) {
            const slot = (startSlot + f) % this.queryCapacityFrames;
            const baseIdx = slot * this.queriesPerFrame;
            // Determine whether all required timestamps for this frame were written (non-zero)
            let frameValid = true;
            for (let si = 0; si < stages.length; si++) {
                const [_name, endIndex, startIndex] = stages[si];
                if (times[baseIdx + startIndex] === 0n || times[baseIdx + endIndex] === 0n || times[baseIdx + endIndex] < times[baseIdx + startIndex]) {
                    frameValid = false;
                    break;
                }
            }
            if (!frameValid) {
                // Print once for debugging (can be commented out if desired)
                if ((slot % 60) === 0) {
                    console.debug('[timestamp] frame slot', slot, 'contains unwritten (0) timestamps, skipped in stats');
                }
                continue;
            }
            validFrames++;
            for (let si = 0; si < stages.length; si++) {
                const [_name, endIndex, startIndex] = stages[si];
                const startTime = Number(times[baseIdx + startIndex]);
                const endTime = Number(times[baseIdx + endIndex]);
                stageSamples[si].push((endTime - startTime) / 1e6); // ms
            }
        }
        if (validFrames === 0) {
            this.resultBuffer.unmap();
            console.warn('[timestamp] No complete frames available (some timestamps are 0). It may be the first frame or the GPU is still filling.');
            return;
        }

        // Accumulate total times
        this.allFrameTimes.push(...stageSamples[0]);

        const measuredTimes: [string, number][] = [];
        let totalP99 = 0;
        let totalStd = 0;
        let totalAvg = 0;

        for (let si = 0; si < stages.length; si++) {
            const name = stages[si][0];
            const samples = stageSamples[si];
            // Compute mean for current batch for stages other than Total, or keep as is?
            // User specifically asked about the big array logic which implies Total stats should be over allFrameTimes.
            // However, usually detailed stage breakdown (Culling, Preprocess etc) is also interesting. 
            // If I change 'Total' mean to be over all history, it might mismatch with current batch 'Culling' mean.
            // The user request was "calculate std and p99 and average on this big array".
            
            let mean = 0;
            if (name === 'Total') {
                 // Use allFrameTimes for statistics
                 const allSamples = this.allFrameTimes;
                 mean = allSamples.reduce((a, b) => a + b, 0) / allSamples.length;
                 
                 const sorted = [...allSamples].sort((a, b) => a - b);
                 totalP99 = sorted[Math.floor(sorted.length * 0.99)] || 0;
                 
                 const variance = allSamples.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / allSamples.length;
                 totalStd = Math.sqrt(variance);

                 totalAvg = mean;
            } else {
                 // Keep current batch average for sub-stages? Or accumulate them too? 
                 // The user prompt specifically mentioned "save all frame times to a big array" (using the code snippet selection which showed total frame times).
                 // It's safer to only modify Total stats to use the accumulated array as requested, to avoid memory explosion if we accumulate everything, 
                 // although we are already accumulating Total.
                 // Let's stick to modifying Total stats.
                 mean = samples.reduce((a, b) => a + b, 0) / samples.length;
            }
            measuredTimes.push([name, mean]);
        }

        // Update counters: lastFrame += all frames we produced since last read; reset frameCount
        this.lastFrame += this.frameCount;
        this.frameCount = 0;

        const renderMethod = this.constructor.name;
        const output = `[TIMESTAMP - ${renderMethod}]\n` + 
            measuredTimes.map(([name, time]) => `${name}: ${time.toFixed(3)}ms`).join('\n') + 
            `\nTotal P99: ${totalP99.toFixed(3)}ms` +
            `\nTotal STD: ${totalStd.toFixed(3)}ms` +
            `\nTotal AVG: ${totalAvg.toFixed(3)}ms` +
            `\nStats computed over ${this.allFrameTimes.length} frames (cumulative)` +
            `\n${this.lastFrame} frames rendered since start`;

        console.log(output);
        // Print all total frame times for plotting
        console.log("All Frame Times (Total, ms):", JSON.stringify(this.allFrameTimes));

        // If armed, download a one-shot CSV of the current averaged metrics
        if (this.downloadOnceNextRead) {
            this.downloadOnceNextRead = false;
            const csvHeader = 'Stage,ms\n';
            const csvBody = measuredTimes.map(([name, time]) => `${name},${time.toFixed(3)}`).join('\n');
            const csvContent = 'data:text/csv;charset=utf-8,' + encodeURIComponent(csvHeader + csvBody);
            const link = document.createElement('a');
            link.href = csvContent;
            link.download = `${this.downloadOnceFileName}.csv`;
            document.body.appendChild(link);
            link.click();
            link.remove();
        }

        // If requested, show a one-time dialog with the latest metrics
        if (this.showPerfDialogNext) {
            this.showPerfDialogNext = false;
            try {
                // Use a simple alert dialog to display metrics
                alert(output);
            } catch {
                // Fallback: ensure at least console output exists
                console.warn('Unable to show dialog; metrics printed to console.');
            }
        }

        // Unmap
        this.resultBuffer.unmap();
    }

    // ========================================================================
    // PRIVATE SETUP METHODS
    // ========================================================================

    /**
     * Initializes GPU resources for timestamp queries.
     */
    private _setupTimestampQueries(): void {
        // Create a timestamp query set (expanded to 100 × per-frame usage)
        this.querySet = this.device.createQuerySet({
            type: 'timestamp',
            count: this.totalQueryCount,
        });

        const bytes = this.totalQueryCount * 8; // 64-bit per query
        // Create a GPU buffer to receive results from the query set
        this.resolveBuffer = this.device.createBuffer({
            size: bytes,
            usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
        });

        // Create a mappable GPU buffer to read data from resolveBuffer to the CPU
        this.resultBuffer = this.device.createBuffer({
            size: bytes,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
    }

    /**
     * Creates various GPU buffers for settings and indirect drawing.
     */
    private _setupBuffers(max_sh_deg: number): void {
        this.render_settings_buffer = this.device.createBuffer({
            label: 'render settings',
            size: C_SIZE_RENDER_SETTINGS_BUFFER,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        const canvas = document.querySelector('canvas');
        const width = canvas ? canvas.width : 1;
        const height = canvas ? canvas.height : 1;
        initRenderSettings({ width, height, max_sh_deg: max_sh_deg });
        writeRenderSettings(this.device, this.render_settings_buffer);


        this.splat_2d_buffer = this.device.createBuffer({
            label: '2d gaussians buffer',
            size: this.pc.num_points * this.splatSize,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        this.draw_indirect_buffer = this.device.createBuffer({
            label: 'draw indirect',
            size: 4 * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | GPUBufferUsage.INDIRECT,
        });
        // Set vertexCount to 4 for the triangle strip, instanceCount will be copied from sorter
        this.device.queue.writeBuffer(this.draw_indirect_buffer, 0, new Uint32Array([4, 0, 0, 0]));

        // Buffer to hold SHSolver outputs from cull pass (dir_opacity + idx)
        // Layout size per element: pad to 12 bytes for safe alignment
        const SH_SOLVER_STRIDE = 12;
        this.sh_solvers_buffer = this.device.createBuffer({
            label: 'sh_solvers',
            size: this.pc.num_points * SH_SOLVER_STRIDE,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        this.sh_color_rgba_buffer = this.device.createBuffer({
            label: "sh_color_rgba",
            size: this.pc.num_points * this.colorSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
    }

    // ========================================================================
    // PRIVATE RUNTIME METHODS
    // ========================================================================

    /**
     * Request showing a dialog the next time perf metrics are read.
     * This sets a one-shot flag consumed by readPerfMetrics().
     */
    public requestPerfDialog(): void {
        this.showPerfDialogNext = true;
    }

    /**
     * One-shot: request downloading the next readPerfMetrics result as CSV.
     * If a file name is provided, it will be sanitized and used (without extension).
     */
    public requestDownloadMetrics(fileName?: string): void {
        if (fileName && fileName.trim().length > 0) {
            const sanitized = fileName.trim().replace(/[^a-zA-Z0-9_\-]/g, '_');
            this.downloadOnceFileName = sanitized.length > 0 ? sanitized : this.downloadOnceFileName;
        } else {
            // give it a timestamped default to avoid overwriting
            const ts = new Date();
            const iso = `${ts.getFullYear()}${String(ts.getMonth() + 1).padStart(2, '0')}${String(ts.getDate()).padStart(2, '0')}_${String(ts.getHours()).padStart(2, '0')}${String(ts.getMinutes()).padStart(2, '0')}${String(ts.getSeconds()).padStart(2, '0')}`;
            this.downloadOnceFileName = `fps_metrics_${iso}`;
        }
        this.downloadOnceNextRead = true;
    }

    /**
     * Public API: schedule a CPU-side reorder after the next submitted frame finishes.
     */
    public requestReorder(): void {
        this.requestReorderNextFrame = true;
    }

    /**
     * To be called by the app right after queue.submit of a frame.
     * If armed by requestReorder(), performs one-shot readback of sorted indices,
     * builds permutation (alive first, invisible to the back), reorders CPU mirrors,
     * and writes back to GPU buffers.
     */
    public async maybeReorderAfterSubmit(): Promise<void> {
        // TODO: check this
        if (!this.requestReorderNextFrame || this.reorderInFlight) return;
        this.reorderInFlight = true;
        this.requestReorderNextFrame = false;
        try {
            const num = this.pc.num_points;
            if (!num) return;
            log(`Reordering ${num} splats...`);
            // Read back: sorted indices (length = alive), sh_solvers (compact->original map), and keys_size from sort_info
            const bytesSortInfo = 16 * 4; // GeneralInfo size we care (at least keys_size at offset 0)
            const readSortInfo = this.device.createBuffer({ size: bytesSortInfo, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
            const readSortedIdx = this.device.createBuffer({ size: num * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
            const SH_SOLVER_STRIDE = 12; // bytes
            const readSolvers = this.device.createBuffer({ size: num * SH_SOLVER_STRIDE, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

            const enc = this.device.createCommandEncoder();
            enc.copyBufferToBuffer(this.sort_info_buffer, 0, readSortInfo, 0, bytesSortInfo);
            enc.copyBufferToBuffer(this.sort_ping_pong[0].sort_indices_buffer, 0, readSortedIdx, 0, num * 4);
            enc.copyBufferToBuffer(this.sh_solvers_buffer, 0, readSolvers, 0, num * SH_SOLVER_STRIDE);
            this.device.queue.submit([enc.finish()]);
            await this.device.queue.onSubmittedWorkDone();

            await Promise.all([
                readSortInfo.mapAsync(GPUMapMode.READ),
                readSortedIdx.mapAsync(GPUMapMode.READ),
                readSolvers.mapAsync(GPUMapMode.READ),
            ]);

            const infoU32 = new Uint32Array(readSortInfo.getMappedRange());
            const alive = Math.min(infoU32[0] ?? 0, num);
            const sortedCompacted = new Uint32Array(readSortedIdx.getMappedRange(), 0, alive);
            const solversView = new DataView(readSolvers.getMappedRange());

            // Build permutation: first alive (mapped to original ids), then the rest
            const aliveOriginal = new Uint32Array(alive);
            for (let i = 0; i < alive; i++) {
                const compact = sortedCompacted[i] >>> 0;
                const orig = solversView.getUint32(compact * SH_SOLVER_STRIDE + 8, true);
                aliveOriginal[i] = orig >>> 0;
            }
            const seen = new Uint8Array(num);
            for (let i = 0; i < alive; i++) seen[aliveOriginal[i]] = 1;
            const perm = new Uint32Array(num);
            for (let i = 0; i < alive; i++) perm[i] = aliveOriginal[i];
            let p = alive;
            for (let j = 0; j < num; j++) if (seen[j] === 0) perm[p++] = j;

            // Reorder CPU mirrors
            const gsStride = this.pc.gs_stride;
            const shStride = this.pc.sh_stride;
            const ScalarArray = (this.pc.ply_precision === 'f32' ? Float32Array : Float16Array) as any;
            const bytesPerScalar = this.pc.ply_precision === 'f32' ? 4 : 2;
            const newGauss = new ScalarArray(PadTo4Bytes(num * gsStride * bytesPerScalar) / bytesPerScalar);
            const newSH = new ScalarArray(PadTo4Bytes(num * shStride * bytesPerScalar) / bytesPerScalar);
            for (let i = 0; i < num; i++) {
                const src = perm[i] >>> 0;
                newGauss.set(this.pc.gaussian_cpu.subarray(src * gsStride, (src + 1) * gsStride), i * gsStride);
                newSH.set(this.pc.sh_cpu.subarray(src * shStride, (src + 1) * shStride), i * shStride);
            }
            this.pc.gaussian_cpu = newGauss;
            this.pc.sh_cpu = newSH;

            // Write back to GPU
            this.device.queue.writeBuffer(this.pc.gaussian_3d_buffer, 0, this.pc.ply_precision === 'f32' ? new Float32Array(newGauss.buffer) : new Uint16Array(newGauss.buffer));
            this.device.queue.writeBuffer(this.pc.sh_buffer, 0, this.pc.ply_precision === 'f32' ? new Float32Array(newSH.buffer) : new Uint16Array(newSH.buffer));

            // Cleanup
            readSortInfo.unmap(); readSortedIdx.unmap(); readSolvers.unmap();
            readSortInfo.destroy(); readSortedIdx.destroy(); readSolvers.destroy();
            log(`Reordered ${num} splats, moved ${alive} alive splats to the front.`);
        } catch (e) {
            console.warn('[reorder] failed:', e);
        } finally {
            this.reorderInFlight = false;
        }
    }
}
