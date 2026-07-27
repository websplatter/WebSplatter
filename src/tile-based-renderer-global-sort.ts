import { PlyLoader } from './ply-loader.ts';
import preprocess_f16_wgsl from './shaders/preprocess_f16.wgsl';
import rasterization_wgsl from './shaders/tile_global_raster.wgsl';
import display_wgsl from './shaders/display.wgsl';
import calcIndirectDispatchWGSL from './shaders/tile_global_calculate_indirect_dispatch.wgsl';
import tile_cull_wgsl from './shaders/tile_global_cull.wgsl';
import pairScatterWGSL from './shaders/pair_sort_scatter.wgsl';
import pairLocalHistogramWGSL from './shaders/pair_sort_local_histogram.wgsl';
import pairBellochScanWGSL from './shaders/pair_sort_belloch.wgsl';
import { log } from './utils/simple-console.ts';

import { RENDER_SETTINGS_SIZE, initRenderSettings, writeRenderSettings } from './render-settings.ts';
import { GaussianRenderers } from './splat-app.ts';

const WG_SIZE = 256;
const TILE_SIZE = 16;

const C_SIZE_RENDER_SETTINGS_BUFFER = RENDER_SETTINGS_SIZE;
const C_TIMESTAMP_COUNT = 10;
const C_SIZE_2D_SPLAT = 12;
const RADIX_BITS_PER_PASS = 8;
const RADIX_SIZE = 1 << RADIX_BITS_PER_PASS;
const TOTAL_PASSES = 32 / RADIX_BITS_PER_PASS;

/**
 * Implements the full rendering pipeline for Gaussian Splatting.
 * This class handles preprocessing, sorting, and rendering the splats.
 * It also includes performance metric tracking using timestamp queries.
 */
export class TileBasedRendererGlobalSort implements GaussianRenderers {
    private readonly device: GPUDevice;
    private pc: PlyLoader;
    private readonly presentationFormat: GPUTextureFormat;
    private readonly canvas: HTMLCanvasElement;

    public camera_buffer: GPUBuffer;
    public render_settings_buffer: GPUBuffer;
    private splat_2d_buffer: GPUBuffer;

    private querySet: GPUQuerySet | null = null;
    private resolveBuffer: GPUBuffer | null = null;
    private resultBuffer: GPUBuffer | null = null;
    private timestampEnabled: boolean = false;

    private readonly queriesPerFrame: number = C_TIMESTAMP_COUNT;
    private readonly queryCapacityFrames: number = 200;
    private get totalQueryCount(): number { return this.queriesPerFrame * this.queryCapacityFrames; }
    private lastFrame: number = 0;
    private frameCount: number = 0;

    private aliveInfoBuffer: GPUBuffer;
    private preprocessPipeline: GPUComputePipeline;
    private cullPipeline: GPUComputePipeline;
    private rasterPipeline: GPUComputePipeline;
    private displayPipeline: GPURenderPipeline;
    private indirectPipeline: GPUComputePipeline;
    private rasterBgl1: GPUBindGroupLayout;

    private rsPingPongPairs: [GPUBuffer, GPUBuffer];
    private rsPingPongIndicies: [GPUBuffer, GPUBuffer];
    private rsWgHistograms: GPUBuffer;
    private rsWgPrefixes: GPUBuffer;
    private rsDigitBase: GPUBuffer;
    private rsL0Sums: GPUBuffer; private rsL0Offsets: GPUBuffer;
    private rsL1Sums: GPUBuffer; private rsL1Offsets: GPUBuffer;
    private rsPrefixBindGroup: GPUBindGroup;
    private rsLocalHistogramPairsBindGroups: GPUBindGroup[] = [];
    private rsScatterBindGroups: GPUBindGroup[] = [];
    private rsPipelines: { passes: { localHistogramComp0: GPUComputePipeline, localHistogramComp1: GPUComputePipeline, scatterElements0: GPUComputePipeline, scatterElements1: GPUComputePipeline }[], hierarchical: any };

    private crsBg: GPUBindGroup;
    private cullBg1: GPUBindGroup;
    private cullBg2: GPUBindGroup;
    private preprocessBg1: GPUBindGroup;
    private rasterBindGroup0: GPUBindGroup;
    private rasterBindGroup1: GPUBindGroup;
    private indirectBindGroup0: GPUBindGroup;
    private indirectBindGroup1: GPUBindGroup;

    private sh_color_rgba_buffer: GPUBuffer;
    private sh_solvers_buffer: GPUBuffer;
    private depths_buffer: GPUBuffer;
    private tileCountsBuffer: GPUBuffer;
    private tileOffsetsBuffer: GPUBuffer;
    private tileInfo: { tilesX: number, tilesY: number, capacity: number } = { tilesX: 0, tilesY: 0, capacity: 0 };

    private showPerfDialogNext: boolean = false;
    private requestReorderNextFrame: boolean = false;
    private reorderInFlight: boolean = false;
    private downloadOnceNextRead: boolean = false;
    private downloadOnceFileName: string = 'fps_metrics';

    constructor(
        pc: PlyLoader,
        device: GPUDevice,
        presentationFormat: GPUTextureFormat,
        camera_buffer: GPUBuffer,
        features_list: GPUFeatureName[]
    ) {
        const f16_enabled = features_list.includes('shader-f16');
        const time_query_enabled = features_list.includes('timestamp-query');

        if (f16_enabled) {
            log('🚀 using shader-f16');
        }

        if (time_query_enabled) {
            log('⏰ using timestamp-query');
        }

        this.canvas = document.querySelector('canvas') as HTMLCanvasElement;
        this.pc = pc;
        this.device = device;
        this.presentationFormat = presentationFormat;
        this.camera_buffer = camera_buffer;
        this.timestampEnabled = time_query_enabled;

        this._setupTimestampQueries();
        this._setupBuffers(pc.sh_degree);
        this._setupTileBuffers();
        this._initPairRadix();

        const camera_renderSettings_Bgl = this.device.createBindGroupLayout({
            label: 'camera + renderSettings',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ],
        });

        const cullGroup1Bgl = this.device.createBindGroupLayout({
            label: 'cull group1 persistent',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            ],
        });
        const cullGroup2Bgl = this.device.createBindGroupLayout({
            label: 'cull group2 resize (tile_counts)',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
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
        this.preprocessBg1 = this.device.createBindGroup({
            label: 'preprocessBg2',
            layout: preprocessBgl2,
            entries: [
                { binding: 0, resource: { buffer: this.aliveInfoBuffer } },
                { binding: 1, resource: { buffer: this.pc.sh_buffer } },
                { binding: 2, resource: { buffer: this.sh_solvers_buffer } },
                { binding: 3, resource: { buffer: this.sh_color_rgba_buffer } },
            ],
        });
        this.cullBg1 = this.device.createBindGroup({
            label: 'cull group1',
            layout: cullGroup1Bgl,
            entries: [
                { binding: 0, resource: { buffer: this.pc.gaussian_3d_buffer } },
                { binding: 1, resource: { buffer: this.splat_2d_buffer } },
                { binding: 2, resource: { buffer: this.rsPingPongPairs[0] } },
                { binding: 3, resource: { buffer: this.aliveInfoBuffer } },
                { binding: 4, resource: { buffer: this.sh_solvers_buffer } },
                { binding: 5, resource: { buffer: this.rsPingPongIndicies[0] } },
            ]
        });
        this.cullBg2 = this.device.createBindGroup({
            label: 'cull group2',
            layout: cullGroup2Bgl,
            entries: [
                { binding: 0, resource: { buffer: this.tileCountsBuffer } },
            ]
        });

        const indirectModule = this.device.createShaderModule({ code: calcIndirectDispatchWGSL });
        this.indirectPipeline = this.device.createComputePipeline({
            label: 'indirect dispatch calc',
            layout: 'auto',
            compute: { module: indirectModule, entryPoint: 'write_dispatch_triples', constants: { RS_RADIX_SIZE: 256 } },
        });

        this.indirectBindGroup0 = this.device.createBindGroup({
            label: 'indirect dispatch bind group 0',
            layout: this.indirectPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.aliveInfoBuffer } },
                { binding: 1, resource: { buffer: this.render_settings_buffer } },
            ],
        });
        this.indirectBindGroup1 = this.device.createBindGroup({
            label: 'indirect dispatch bind group 1',
            layout: this.indirectPipeline.getBindGroupLayout(1),
            entries: [
                { binding: 0, resource: { buffer: this.tileCountsBuffer } },
                { binding: 1, resource: { buffer: this.tileOffsetsBuffer } },
            ],
        });

        const preprocessCullShader = this.device.createShaderModule({ code: tile_cull_wgsl });

        this.cullPipeline = this.device.createComputePipeline({
            label: 'preprocess_cull',
            layout: this.device.createPipelineLayout({
                bindGroupLayouts: [camera_renderSettings_Bgl, cullGroup1Bgl, cullGroup2Bgl]
            }),
            compute: { module: preprocessCullShader, entryPoint: 'preprocess_cull' },
        });

        const preprocessShader = this.device.createShaderModule({ code: preprocess_f16_wgsl });

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

        const rasterShader = this.device.createShaderModule({ code: rasterization_wgsl });
        const rasterStaticBgl = this.device.createBindGroupLayout({
            label: 'raster static layout (g0)',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            ]
        });
        this.rasterBgl1 = this.device.createBindGroupLayout({
            label: 'raster tile layout (g1)',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            ]
        });
        const rasterTextureBgl = this.device.createBindGroupLayout({
            label: 'raster texture layout (g2)',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba8unorm', viewDimension: '2d' } },
            ]
        });
        const rasterPl = this.device.createPipelineLayout({
            bindGroupLayouts: [
                rasterStaticBgl,
                this.rasterBgl1,
                rasterTextureBgl,
            ]
        });
        this.rasterPipeline = this.device.createComputePipeline({
            label: 'software raster',
            layout: rasterPl,
            compute: {
                module: rasterShader,
                entryPoint: 'tile_based_main',
            },
        });

        const displayShader = this.device.createShaderModule({ code: display_wgsl });
        this.displayPipeline = this.device.createRenderPipeline({
            label: 'display',
            layout: 'auto',
            vertex: {
                module: displayShader,
                entryPoint: 'vs_main',
            },
            fragment: {
                module: displayShader,
                entryPoint: 'fs_main',
                targets: [{ format: this.presentationFormat }],
            },
            primitive: {
                topology: 'triangle-list',
            },
        });

        this.rasterBindGroup0 = this.device.createBindGroup({
            label: 'raster static BG (g0)',
            layout: rasterStaticBgl,
            entries: [
                { binding: 0, resource: { buffer: this.render_settings_buffer } },
                { binding: 1, resource: { buffer: this.aliveInfoBuffer } },
                { binding: 2, resource: { buffer: this.splat_2d_buffer } },
                { binding: 3, resource: { buffer: this.sh_color_rgba_buffer } },
                { binding: 4, resource: { buffer: this.depths_buffer } },
                { binding: 5, resource: { buffer: this.rsPingPongIndicies[1] } },
            ],
        });
        this.rasterBindGroup1 = this.device.createBindGroup({
            label: 'tile raster bind group 1',
            layout: this.rasterBgl1,
            entries: [
                { binding: 0, resource: { buffer: this.tileOffsetsBuffer } },
            ],
        });

        // @ts-ignore
        window.debugPrintTileSplatCounts = this.debugPrintTileSplatCounts.bind(this);
    }

    /**
     * Debug helper: read first N sorted indices and print them.
     */
    public async debugReadSortedIndices(count: number = 30): Promise<void> {
        console.log('[DEBUG] sorter removed');
    }

    public async debugDumpPixelInputs(px: number, py: number, width: number, height: number, limit?: number) {
        throw new Error('debugDumpPixelInputs not implemented');
    }

    /**
     * Executes a single frame of rendering.
     * @param encoder - The command encoder for this frame.
     * @param texture_view - The canvas texture view to render to.
     */
    public frame(encoder: GPUCommandEncoder, texture_view: GPUTextureView): void {
        const offsetFrameIndex = (this.lastFrame + this.frameCount) % this.queryCapacityFrames;
        const base = offsetFrameIndex * this.queriesPerFrame;
        {
            encoder.clearBuffer(this.aliveInfoBuffer, 0, 20);
            encoder.clearBuffer(this.tileCountsBuffer, 0);
            const pass = encoder.beginComputePass({
                label: 'cull',
                ...(this.timestampEnabled && this.querySet ? {
                    timestampWrites: {
                        querySet: this.querySet,
                        beginningOfPassWriteIndex: base + 0,
                        endOfPassWriteIndex: base + 1,
                    },
                } : {}),
            });
            pass.setPipeline(this.cullPipeline);
            pass.setBindGroup(0, this.crsBg);
            pass.setBindGroup(1, this.cullBg1);
            pass.setBindGroup(2, this.cullBg2);
            const wgCount = Math.ceil(this.pc.num_points / WG_SIZE);
            pass.dispatchWorkgroups(wgCount, 1, 1);
            pass.end();
        }
        {
            const pass = encoder.beginComputePass({ label: 'calculate indirect dispatch' });
            pass.setPipeline(this.indirectPipeline);
            pass.setBindGroup(0, this.indirectBindGroup0);
            pass.setBindGroup(1, this.indirectBindGroup1);
            pass.dispatchWorkgroups(1, 1, 1);
            pass.end();
        }
        {
            const pass = encoder.beginComputePass({
                label: 'preprocess',
                ...(this.timestampEnabled && this.querySet ? {
                    timestampWrites: {
                        querySet: this.querySet,
                        beginningOfPassWriteIndex: base + 2,
                        endOfPassWriteIndex: base + 3,
                    },
                } : {}),
            });
            pass.setPipeline(this.preprocessPipeline);
            pass.setBindGroup(0, this.crsBg);
            pass.setBindGroup(1, this.preprocessBg1);
            pass.dispatchWorkgroupsIndirect(this.aliveInfoBuffer, 4);
            pass.end();
        }

        this._pairRadixSort(encoder);

        {
            const pass = encoder.beginComputePass({
                label: 'software raster',
                ...(this.timestampEnabled && this.querySet ? {
                    timestampWrites: {
                        querySet: this.querySet,
                        beginningOfPassWriteIndex: base + 4,
                        endOfPassWriteIndex: base + 5,
                    },
                } : {}),
            });
            pass.setPipeline(this.rasterPipeline);
            pass.setBindGroup(0, this.rasterBindGroup0);
            pass.setBindGroup(1, this.rasterBindGroup1);
            pass.setBindGroup(2, this.device.createBindGroup({
                layout: this.rasterPipeline.getBindGroupLayout(2),
                entries: [
                    {
                        binding: 0,
                        resource: texture_view,
                    }
                ]
            }));
            const width = this.canvas.width;
            const height = this.canvas.height;
            pass.dispatchWorkgroups(
                Math.ceil(width / TILE_SIZE),
                Math.ceil(height / TILE_SIZE),
                1
            );
            pass.end();
        }

        this._resolveTimestamps(encoder);
        this.frameCount++;
    }

    /**
     * Reads the performance metrics from the last completed frame.
     * This function is async and will resolve when the data is available on the CPU.
     */
    public async readPerfMetrics(): Promise<void> {
        const stages: [string, number, number][] = [
            ['Total', 5, 0],
            ['Culling', 1, 0],
            ['Preprocess', 3, 2],
            ['Sort', 4, 3],
            ['Rasterization', 5, 4],
        ];

        if (!this.timestampEnabled || !this.resultBuffer || this.frameCount <= 0) {
            return;
        }

        await this.resultBuffer.mapAsync(GPUMapMode.READ);
        const times = new BigInt64Array(this.resultBuffer.getMappedRange());

        const framesToRead = Math.min(this.frameCount, this.queryCapacityFrames);
        const startSlot = (this.lastFrame + this.frameCount - framesToRead) % this.queryCapacityFrames;

        const sums = new Array<number>(stages.length).fill(0);
        for (let f = 0; f < framesToRead; f++) {
            const slot = (startSlot + f) % this.queryCapacityFrames;
            const baseIdx = slot * this.queriesPerFrame;
            for (let si = 0; si < stages.length; si++) {
                const [_name, endIndex, startIndex] = stages[si];
                const startTime = Number(times[baseIdx + startIndex]);
                const endTime = Number(times[baseIdx + endIndex]);
                sums[si] += (endTime - startTime) / 1e6;
            }
        }
        const measuredTimes: [string, number][] = stages.map(([name], i) => [name, sums[i] / framesToRead]);
        this.lastFrame += this.frameCount;
        this.frameCount = 0;

        const renderMethod = this.constructor.name;
        const output = `[TIMESTAMP - ${renderMethod}]\n` + measuredTimes.map(([name, time]) => `${name}: ${time.toFixed(3)}ms`).join('\n') + `\n${this.lastFrame} frames rendered\n`;

        console.log(output);

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

        if (this.showPerfDialogNext) {
            this.showPerfDialogNext = false;
            try {
                alert(output);
            } catch {
                console.warn('Unable to show dialog; metrics printed to console.');
            }
        }

        this.resultBuffer.unmap();
    }

    private _initPairRadix(): void {
        const capacity = this.tileInfo.capacity || 1;
        this.rsPingPongPairs = [0, 1].map(i => this.device.createBuffer({ label: `pair_pairs_${i}`, size: capacity * (4 + 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST })) as [GPUBuffer, GPUBuffer];
        this.rsPingPongIndicies = [0, 1].map(i => this.device.createBuffer({ label: `pair_keys_${i}`, size: capacity * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST })) as [GPUBuffer, GPUBuffer];
        const RADIX_SIZE = 256;
        const maxWG = Math.ceil(capacity / WG_SIZE);
        this.rsWgHistograms = this.device.createBuffer({ label: 'pair_wg_hist', size: maxWG * RADIX_SIZE * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
        this.rsWgPrefixes = this.device.createBuffer({ label: 'pair_wg_prefix', size: maxWG * RADIX_SIZE * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
        this.rsDigitBase = this.device.createBuffer({ label: 'pair_digit_base', size: RADIX_SIZE * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
        const T0Max = Math.ceil(maxWG / WG_SIZE);
        const T1Max = Math.ceil(T0Max / WG_SIZE);
        this.rsL0Sums = this.device.createBuffer({ label: 'pair_l0_sums', size: T0Max * RADIX_SIZE * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
        this.rsL0Offsets = this.device.createBuffer({ label: 'pair_l0_offs', size: T0Max * RADIX_SIZE * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
        this.rsL1Sums = this.device.createBuffer({ label: 'pair_l1_sums', size: T1Max * RADIX_SIZE * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
        this.rsL1Offsets = this.device.createBuffer({ label: 'pair_l1_offs', size: T1Max * RADIX_SIZE * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
        const histogramModule = this.device.createShaderModule({ code: pairLocalHistogramWGSL });
        const scatterModule = this.device.createShaderModule({ code: pairScatterWGSL });
        const prefixModule = this.device.createShaderModule({ code: pairBellochScanWGSL });
        const prefixBGL = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            ]
        });
        const localHistBGL = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            ]
        });
        const scatterBGL = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            ]
        });
        const histogramLayout = this.device.createPipelineLayout({ bindGroupLayouts: [localHistBGL] });
        const scatterLayout = this.device.createPipelineLayout({ bindGroupLayouts: [scatterBGL] });
        const prefixLayout = this.device.createPipelineLayout({ bindGroupLayouts: [prefixBGL] });
        const mkPrefix = (ep: string) => this.device.createComputePipeline({ layout: prefixLayout, compute: { module: prefixModule, entryPoint: ep } });
        const hierarchical = {
            l0TileScan: mkPrefix('prefix_l0_tile_scan'),
            l1TileScanOnL0: mkPrefix('prefix_l1_tile_scan_on_l0_sums'),
            l1ScanSums: mkPrefix('prefix_scan_l1_sums'),
            addL1ToL0: mkPrefix('prefix_add_l1_to_l0_offsets'),
            addL0ToElems: mkPrefix('prefix_add_l0_to_elements'),
            computeDigitBase: mkPrefix('compute_digit_base'),
            prefixBindGroupLayout: prefixBGL,
        };
        const passes: { localHistogramComp0: GPUComputePipeline, localHistogramComp1: GPUComputePipeline, scatterElements0: GPUComputePipeline, scatterElements1: GPUComputePipeline }[] = [];
        for (let i = 0; i < TOTAL_PASSES; i++) {
            const baseConsts = { PASS_ID: i, RS_RADIX_LOG2: RADIX_BITS_PER_PASS, RS_RADIX_SIZE: RADIX_SIZE } as any;
            const histComp0 = this.device.createComputePipeline({ layout: histogramLayout, compute: { module: histogramModule, entryPoint: 'local_histogram_pass', constants: { ...baseConsts, SORT_COMPONENT: 0 } } });
            const histComp1 = this.device.createComputePipeline({ layout: histogramLayout, compute: { module: histogramModule, entryPoint: 'local_histogram_pass', constants: { ...baseConsts, SORT_COMPONENT: 1 } } });
            const scatter0 = this.device.createComputePipeline({ layout: scatterLayout, compute: { module: scatterModule, entryPoint: 'scatter_elements', constants: { ...baseConsts, SORT_COMPONENT: 0 } } });
            const scatter1 = this.device.createComputePipeline({ layout: scatterLayout, compute: { module: scatterModule, entryPoint: 'scatter_elements', constants: { ...baseConsts, SORT_COMPONENT: 1 } } });
            passes.push({ localHistogramComp0: histComp0, localHistogramComp1: histComp1, scatterElements0: scatter0, scatterElements1: scatter1 });
        }
        this.rsPipelines = { passes, hierarchical };
        this.rsPrefixBindGroup = this.device.createBindGroup({
            layout: prefixBGL,
            entries: [
                { binding: 0, resource: { buffer: this.aliveInfoBuffer } },
                { binding: 1, resource: { buffer: this.rsWgHistograms } },
                { binding: 2, resource: { buffer: this.rsWgPrefixes } },
                { binding: 3, resource: { buffer: this.rsL0Sums } },
                { binding: 4, resource: { buffer: this.rsL0Offsets } },
                { binding: 5, resource: { buffer: this.rsL1Sums } },
                { binding: 6, resource: { buffer: this.rsL1Offsets } },
                { binding: 7, resource: { buffer: this.rsDigitBase } },
            ]
        });
        this.rsLocalHistogramPairsBindGroups = [0, 1].map(i => this.device.createBindGroup({
            layout: localHistBGL,
            entries: [
                { binding: 0, resource: { buffer: this.aliveInfoBuffer } },
                { binding: 1, resource: { buffer: this.rsPingPongPairs[i] } },
                { binding: 2, resource: { buffer: this.rsWgHistograms } },
            ]
        }));
        this.rsScatterBindGroups = [0, 1].map(i => this.device.createBindGroup({
            layout: scatterBGL, entries: [
                { binding: 0, resource: { buffer: this.aliveInfoBuffer } },
                { binding: 1, resource: { buffer: this.rsDigitBase } },
                { binding: 2, resource: { buffer: this.rsPingPongPairs[i] } },
                { binding: 3, resource: { buffer: this.rsPingPongPairs[1 - i] } },
                { binding: 4, resource: { buffer: this.rsPingPongIndicies[i] } },
                { binding: 5, resource: { buffer: this.rsPingPongIndicies[1 - i] } },
                { binding: 6, resource: { buffer: this.rsWgPrefixes } },
            ]
        }));
    }

    private _pairRadixSort(encoder: GPUCommandEncoder): void {
        this._runPairSortPhase(encoder, false, TOTAL_PASSES);
        this._runPairSortPhase(encoder, true, TOTAL_PASSES);
    }
    private _runPairSortPhase(encoder: GPUCommandEncoder, tilePhase: boolean, total_passes: number) {
        for (let round = 0; round < total_passes; round++) {
            const flip = round & 1;
            const p = this.rsPipelines.passes[round];
            {
                const pass = encoder.beginComputePass({ label: (tilePhase ? 'tile' : 'depth') + `_hist_${round}` });
                pass.setPipeline(tilePhase ? p.localHistogramComp0 : p.localHistogramComp1);
                pass.setBindGroup(0, this.rsLocalHistogramPairsBindGroups[flip]);
                pass.dispatchWorkgroupsIndirect(this.aliveInfoBuffer, 20);
                pass.end();
            }
            {
                const pass = encoder.beginComputePass({ label: (tilePhase ? 'tile' : 'depth') + `_prefix_${round}` });
                pass.setBindGroup(0, this.rsPrefixBindGroup);
                pass.setPipeline(this.rsPipelines.hierarchical.l0TileScan); pass.dispatchWorkgroupsIndirect(this.aliveInfoBuffer, 32);
                pass.setPipeline(this.rsPipelines.hierarchical.l1TileScanOnL0); pass.dispatchWorkgroupsIndirect(this.aliveInfoBuffer, 48);
                pass.setPipeline(this.rsPipelines.hierarchical.l1ScanSums); pass.dispatchWorkgroups(1, RADIX_SIZE, 1);
                pass.setPipeline(this.rsPipelines.hierarchical.addL1ToL0); pass.dispatchWorkgroupsIndirect(this.aliveInfoBuffer, 48);
                pass.setPipeline(this.rsPipelines.hierarchical.addL0ToElems); pass.dispatchWorkgroupsIndirect(this.aliveInfoBuffer, 32);
                pass.setPipeline(this.rsPipelines.hierarchical.computeDigitBase); pass.dispatchWorkgroups(1, 1, 1);
                pass.end();
            }
            {
                const pass = encoder.beginComputePass({ label: (tilePhase ? 'tile' : 'depth') + `_scatter_${round}` });
                pass.setPipeline(tilePhase ? p.scatterElements0 : p.scatterElements1);
                pass.setBindGroup(0, this.rsScatterBindGroups[flip]);
                pass.dispatchWorkgroupsIndirect(this.aliveInfoBuffer, 20);
                pass.end();
            }
        }
    }

    private _setupTimestampQueries(): void {
        if (!this.timestampEnabled) return;

        this.querySet = this.device.createQuerySet({
            type: 'timestamp',
            count: this.totalQueryCount,
        });

        const bytes = this.totalQueryCount * 8;
        this.resolveBuffer = this.device.createBuffer({
            size: bytes,
            usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
        });

        this.resultBuffer = this.device.createBuffer({
            size: bytes,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
    }

    private _setupBuffers(max_sh_deg: number): void {
        this.aliveInfoBuffer = this.device.createBuffer({
            label: 'alive_info',
            size: 16 * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | GPUBufferUsage.INDIRECT,
        });
        this.render_settings_buffer = this.device.createBuffer({
            label: 'render settings',
            size: C_SIZE_RENDER_SETTINGS_BUFFER,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        const width = this.canvas.width;
        const height = this.canvas.height;
        initRenderSettings({ width, height, max_sh_deg: max_sh_deg });
        writeRenderSettings(this.device, this.render_settings_buffer);

        this.splat_2d_buffer = this.device.createBuffer({
            label: '2d gaussians buffer',
            size: this.pc.num_points * C_SIZE_2D_SPLAT,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE,
        });

        const SH_SOLVER_STRIDE = 12;
        this.sh_solvers_buffer = this.device.createBuffer({
            label: 'sh_solvers',
            size: this.pc.num_points * SH_SOLVER_STRIDE,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        this.sh_color_rgba_buffer = this.device.createBuffer({
            label: "sh_color_rgba",
            size: this.pc.num_points * Uint32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        this.depths_buffer = this.device.createBuffer({
            label: 'depths',
            size: this.pc.num_points * Float32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
    }

    private _setupTileBuffers(): void {
        const w = this.canvas.width;
        const h = this.canvas.height;
        const TILE_SIZE = 16;
        const tilesX = Math.ceil(w / TILE_SIZE);
        const tilesY = Math.ceil(h / TILE_SIZE);
        const numTiles = tilesX * tilesY;
        const capacity = Math.max(this.pc.num_points * 8, 1);
        this.tileInfo = { tilesX, tilesY, capacity };

        const countsSize = numTiles * 4;
        const offsetsSize = (numTiles + 1) * 4;

        this.tileCountsBuffer?.destroy();
        this.tileOffsetsBuffer?.destroy();

        this.tileCountsBuffer = this.device.createBuffer({
            label: 'tileCounts',
            size: countsSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.tileOffsetsBuffer = this.device.createBuffer({
            label: 'tileOffsets',
            size: offsetsSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
    }

    public onResize(): void {
        const width = this.canvas.width;
        const height = this.canvas.height;

        try {
            initRenderSettings({ width, height, max_sh_deg: this.pc.sh_degree });
            writeRenderSettings(this.device, this.render_settings_buffer);
        } catch (_) { }

        this._setupTileBuffers();

        this.cullBg2 = this.device.createBindGroup({
            label: 'cull group2 (tile_counts resized)',
            layout: this.cullPipeline.getBindGroupLayout(2),
            entries: [
                { binding: 0, resource: { buffer: this.tileCountsBuffer } },
            ]
        });

        this.indirectBindGroup1 = this.device.createBindGroup({
            label: 'indirect dispatch bind group 1',
            layout: this.indirectPipeline.getBindGroupLayout(1),
            entries: [
                { binding: 0, resource: { buffer: this.tileCountsBuffer } },
                { binding: 1, resource: { buffer: this.tileOffsetsBuffer } },
            ],
        });

        this.rasterBindGroup1 = this.device.createBindGroup({
            label: 'tile raster bind group 1',
            layout: this.rasterBgl1,
            entries: [
                { binding: 0, resource: { buffer: this.tileOffsetsBuffer } },
            ],
        });
    }

    private _resolveTimestamps(encoder: GPUCommandEncoder): void {
        if (!this.timestampEnabled || !this.querySet || !this.resolveBuffer || !this.resultBuffer) return;

        encoder.resolveQuerySet(
            this.querySet,
            0,
            this.totalQueryCount,
            this.resolveBuffer,
            0
        );

        encoder.copyBufferToBuffer(
            this.resolveBuffer, 0,
            this.resultBuffer, 0,
            this.totalQueryCount * 8
        );
    }

    public requestPerfDialog(): void {
        this.showPerfDialogNext = true;
    }

    public async readBreakdown(): Promise<{ cull_ms: number[], preprocess_ms: number[], sort_ms: number[], render_ms: number[] }> {
        throw new Error('readBreakdown() is not supported by TileBasedRendererGlobalSort');
    }

    public requestDownloadMetrics(fileName?: string): void {
        if (fileName && fileName.trim().length > 0) {
            const sanitized = fileName.trim().replace(/[^a-zA-Z0-9_\-]/g, '_');
            this.downloadOnceFileName = sanitized.length > 0 ? sanitized : this.downloadOnceFileName;
        } else {
            const ts = new Date();
            const iso = `${ts.getFullYear()}${String(ts.getMonth() + 1).padStart(2, '0')}${String(ts.getDate()).padStart(2, '0')}_${String(ts.getHours()).padStart(2, '0')}${String(ts.getMinutes()).padStart(2, '0')}${String(ts.getSeconds()).padStart(2, '0')}`;
            this.downloadOnceFileName = `fps_metrics_${iso}`;
        }
        this.downloadOnceNextRead = true;
    }

    public requestReorder(): void {
        this.requestReorderNextFrame = true;
    }

    public async debugPrintTileSplatCounts(maxTiles: number = 128): Promise<void> {
        try {
            const tilesX = this.tileInfo.tilesX;
            const tilesY = this.tileInfo.tilesY;
            const numTiles = tilesX * tilesY;
            if (numTiles === 0) {
                console.warn('[tile-debug] no tiles');
                return;
            }
            const byteSize = (numTiles + 1) * 4;
            const readBuffer = this.device.createBuffer({
                label: 'tileOffsetsReadback',
                size: byteSize,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
            });
            const enc = this.device.createCommandEncoder();
            enc.copyBufferToBuffer(this.tileOffsetsBuffer, 0, readBuffer, 0, byteSize);
            this.device.queue.submit([enc.finish()]);
            await this.device.queue.onSubmittedWorkDone();
            await readBuffer.mapAsync(GPUMapMode.READ);
            const data = new Uint32Array(readBuffer.getMappedRange());
            const counts: number[] = new Array(numTiles);
            const counts_sorted: number[] = new Array(numTiles);
            for (let t = 0; t < numTiles; t++) {
                counts[t] = data[t + 1] - data[t];
                counts_sorted[t] = counts[t];
            }
            counts_sorted.sort();
            readBuffer.unmap();
            readBuffer.destroy();
            let maxVal = 0, minVal = Number.MAX_SAFE_INTEGER, sum = 0;
            let mediumVal = counts_sorted[Math.floor(numTiles / 2)];
            for (let c of counts) { maxVal = Math.max(maxVal, c); minVal = Math.min(minVal, c); sum += c; }
            const avg = sum / numTiles;
            console.log(`[tile-debug] tiles=${numTiles} (${tilesX}x${tilesY}) total_splats=${sum} avg=${avg.toFixed(2)} med=${mediumVal} min=${minVal} max=${maxVal}`);
            const toPrint = Math.min(numTiles, maxTiles);
            let lines: string[] = [];
            for (let i = 0; i < toPrint; i++) {
                const y = Math.floor(i / tilesX);
                const x = i % tilesX;
                lines.push(`(${x},${y}):${counts[i]}`);
            }
            console.log('[tile-debug] first ' + toPrint + ' tiles => ' + lines.join(' '));
        } catch (e) {
            console.warn('[tile-debug] failed', e);
        }
    }

    public async maybeReorderAfterSubmit(): Promise<void> {
        if (!this.requestReorderNextFrame || this.reorderInFlight) return;
        this.reorderInFlight = true;
        this.requestReorderNextFrame = false;
        try {
            const num = this.pc.num_points;
            if (!num) return;
            log(`Reordering ${num} splats...`);
            const bytesSortInfo = 16 * 4;
            const readSortInfo = this.device.createBuffer({ size: bytesSortInfo, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
            const readSortedIdx = this.device.createBuffer({ size: num * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
            const SH_SOLVER_STRIDE = 12;
            const readSolvers = this.device.createBuffer({ size: num * SH_SOLVER_STRIDE, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

            const enc = this.device.createCommandEncoder();
            enc.copyBufferToBuffer(this.aliveInfoBuffer, 0, readSortInfo, 0, bytesSortInfo);
            enc.copyBufferToBuffer(this.sh_solvers_buffer, 0, readSortedIdx, 0, num * 4);
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

            const gsStride = this.pc.gs_stride;
            const shStride = this.pc.sh_stride;
            const newGauss = new Float16Array(num * gsStride);
            const newSH = new Float16Array(num * shStride);
            for (let i = 0; i < num; i++) {
                const src = perm[i] >>> 0;
                newGauss.set(this.pc.gaussian_cpu.subarray(src * gsStride, (src + 1) * gsStride), i * gsStride);
                newSH.set(this.pc.sh_cpu.subarray(src * shStride, (src + 1) * shStride), i * shStride);
            }
            this.pc.gaussian_cpu = newGauss;
            this.pc.sh_cpu = newSH;

            this.device.queue.writeBuffer(this.pc.gaussian_3d_buffer, 0, new Uint16Array(newGauss.buffer));
            this.device.queue.writeBuffer(this.pc.sh_buffer, 0, new Uint16Array(newSH.buffer));

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
