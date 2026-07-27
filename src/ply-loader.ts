import { ReorderMethod } from './reorder-types';
import { log, error, time, timeLog, logProgress, showPLYSpinner, hidePLYSpinner, setPLYSpinnerStatus } from './utils/simple-console';
import { PadTo4Bytes } from './utils/util';

export type PrecisionMode = 'f16' | 'f32';

export type PLYObject = {
    sh_degree: number;
    num_points: number;
    ply_precision: PrecisionMode;
    gaussian_3d_buffer: GPUBuffer;
    sh_buffer: GPUBuffer;
    gaussian_cpu: Float16Array | Float32Array;
    sh_cpu: Float16Array | Float32Array;
    gs_stride: number;
    sh_stride: number;
}

const c_size_f16 = 2;   // byte size of f16
// const c_size_float = 4;   // byte size of f32

// WGSL struct Splat layout (f16) in storage buffer:
// v_0: vec2<f16>  (4B)
// v_1: vec2<f16>  (4B)
// pos: vec2<f16>  (4B)
// Note: alignment/padding may lead to 16B stride depending on implementation.
const c_size_2d_splat = 16; // bytes per Splat (GPU storage buffer stride)
const gs_stride = 10; // 3 for position, 1 for opacity, 6 for cov
const c_size_3d_gaussian = gs_stride * c_size_f16;


export type PlyLoader = PLYObject;


export async function loadFromFile(input: File, device: GPUDevice, sh_degree: number, sort: ReorderMethod | null, memoryPrecision: PrecisionMode = 'f16'): Promise<PLYObject> {
    log(`loading ply file from File... : ${input.name}`);
    showPLYSpinner('downloading PLY...');
    const arrayBuffer = await input.arrayBuffer();
    try {
    return await processPLYInWorker(arrayBuffer, device, sh_degree, sort, memoryPrecision);
    } finally {
        hidePLYSpinner();
    }
}

export async function loadFromURL(url: string, device: GPUDevice, sh_degree: number, sort: ReorderMethod | null, memoryPrecision: PrecisionMode = 'f16'): Promise<PLYObject> {
    log(`loading ply file from URL... : ${url}`);
    showPLYSpinner('downloading PLY...');
    try {
        time();
    // Resolve to absolute URL so the worker (whose base is assets/) fetches correctly
    const absoluteUrl = new URL(url, window.location.href).href;
    // Let worker handle streaming download and progress updates
    return await processPLYInWorker({ url: absoluteUrl }, device, sh_degree, sort, memoryPrecision);
    } finally {
        hidePLYSpinner();
    }
}

type PLYInput = ArrayBuffer | { url: string };

async function processPLYInWorker(input: PLYInput, device: GPUDevice, sh_degree: number, sort: ReorderMethod | null, memoryPrecision: PrecisionMode): Promise<PLYObject> {
    return new Promise((resolve, reject) => {
        const worker = new Worker(new URL('./workers/ply-worker.ts', import.meta.url), { type: 'module' });

    worker.onmessage = (e: MessageEvent<any>) => {
            const msg = e.data;
            if (msg?.type === 'error') {
                error(`PLY worker error: ${msg.message ?? 'unknown error'}`);
                worker.terminate();
                reject(new Error(msg.message ?? 'Worker error'));
                return;
            } else if (msg?.type === 'download_progress') {
                // Show detailed download info: total MB, loaded MB, speed MB/s
                const totalBytes: number | undefined = msg.totalBytes;
                const loadedMB = msg.loadedBytes / (1024 * 1024);
                const totalMB = totalBytes ? (totalBytes / (1024 * 1024)) : undefined;
                const speedMBs = (msg.speedBps ?? 0) / (1024 * 1024);
                const percent = totalBytes ? Math.min(99, Math.floor((msg.loadedBytes / totalBytes) * 100)) : undefined;
                const totalLine = totalMB ? `total ${totalMB.toFixed(1)} MB` : 'total -- MB';
                const progressLine = totalMB && percent !== undefined
                    ? `${loadedMB.toFixed(1)} MB downloaded (${percent}%)`
                    : `${loadedMB.toFixed(1)} MB downloaded`;
                const speedLine = `${speedMBs.toFixed(2)} MB/s`;
                setPLYSpinnerStatus(`downloading PLY ...\n${totalLine}, ${progressLine}\n${speedLine}`);
                // if (percent !== undefined && typeof percent === 'number' && totalBytes) {
                //     logProgress(percent, "Downloading PLY");
                // }
                return;
            } else if (msg?.type === 'fetched') {
                log(`💾 Fetched (${msg.byteLength} bytes)`);
                timeLog('Download');
                setPLYSpinnerStatus('parsing PLY...');
                time();
                return;
            } else if (msg?.type === 'buffer_info') {
                log(`Using ${msg.useShared ? '🚀SharedArrayBuffer' : '🐢ArrayBuffer'} for buffers <->`);
                return;
            } else if (msg?.type === 'sorting') {
                const method = msg.method ?? (msg.sort ? `${msg.sort} sorting` : 'sorting');
                const bits = msg.bitsPerAxis ? ` (${msg.bitsPerAxis} bits/axis)` : '';
                const pts = typeof msg.points === 'number' ? ` ${msg.points} splats` : '';
                setPLYSpinnerStatus(`${method}${bits}${pts}\nreordering...`);
                return;
            } else if (msg?.type === 'parse_progress') {
                const total: number = msg.total ?? 0;
                const read: number = msg.read ?? 0;
                const percent = total > 0 ? Math.floor((read / total) * 100) : 0;
                setPLYSpinnerStatus(`parsing PLY ...\n${read}/${total} splats (${percent}%)`);
                // logProgress(percent, "Parsing PLY");
                return;
            } else if (msg?.type === 'done') {
                const num_points: number = msg.num_points;
                const final_sh_degree: number = msg.final_sh_degree;
                const gs_stride_recv: number = msg.gs_stride;
                const sh_stride_recv: number = msg.sh_stride;
                // Build typed views regardless of SAB/AB
                const gaussian_cpu = msg.ply_precision === 'f32' ? new Float32Array(msg.gaussian_cpu) : new Float16Array(msg.gaussian_cpu);
                const sh_cpu = msg.ply_precision === 'f32' ? new Float32Array(msg.sh_cpu) : new Float16Array(msg.sh_cpu);
                log(`🪐 Total splats: ${num_points}`);

                const gaussian_3d_buffer = device.createBuffer({
                    label: 'ply input 3d gaussians data buffer',
                    size: PadTo4Bytes(gaussian_cpu.byteLength),
                    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
                });
                device.queue.writeBuffer(gaussian_3d_buffer, 0, gaussian_cpu);

                const sh_buffer = device.createBuffer({
                    label: 'ply input spherical harmonics data buffer',
                    size: PadTo4Bytes(sh_cpu.byteLength),
                    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
                });
                device.queue.writeBuffer(sh_buffer, 0, sh_cpu);

                worker.terminate();

                timeLog('Parse');

                resolve({
                    num_points,
                    ply_precision: msg.ply_precision ?? memoryPrecision,
                    gaussian_3d_buffer,
                    sh_buffer,
                    sh_degree: final_sh_degree,
                    gaussian_cpu: gaussian_cpu as Float16Array | Float32Array,
                    sh_cpu: sh_cpu as Float16Array | Float32Array,
                    gs_stride: gs_stride_recv,
                    sh_stride: sh_stride_recv,
                });
            }
        };
        worker.onerror = (err) => {
            worker.terminate();
            reject(err);
        };
        if (input instanceof ArrayBuffer) {
            // Local file path: switch label to parsing when we hand off to worker
            setPLYSpinnerStatus('parsing PLY...');
            worker.postMessage({ type: 'start', plyBuffer: input, sh_degree, sort, memoryPrecision }, [input]);
        } else {
            worker.postMessage({ type: 'start_url', url: input.url, sh_degree, sort, memoryPrecision });
        }
    });
}