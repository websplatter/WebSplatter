/// <reference lib="webworker" />

import { mat3, Quat, quat, Vec3, vec3 } from 'wgpu-matrix';
import type { ReorderMethod } from '../reorder-types';
import { INRIAV1PlyParser } from '../utils/loaders/ply/INRIAV1PlyParser';
import { PadTo4Bytes, sigmoid } from '../utils/util';

const SH_C0 = Math.sqrt(1 / (4 * Math.PI));

const BaseFieldNamesToRead = ['scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3', 'x', 'y', 'z', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity', 'red', 'green', 'blue', 'f_rest_0'] as const;
const BaseFieldsToReadIndexes = BaseFieldNamesToRead.map((_, i) => i);
const [
    SCALE_0, SCALE_1, SCALE_2, ROT_0, ROT_1, ROT_2, ROT_3, X, Y, Z,
    F_DC_0, F_DC_1, F_DC_2, OPACITY, RED, GREEN, BLUE,
] = BaseFieldsToReadIndexes as unknown as number[];

const gs_stride = 10; // 3 pos, 1 opacity, 6 cov

function build_cov(rot: Quat, scale: Vec3): number[] {
    const r = mat3.fromQuat(rot);
    const s = mat3.identity();
    s[0] = scale[0];
    s[5] = scale[1];
    s[10] = scale[2];
    const l = mat3.mul(r, s);
    const m = mat3.mul(l, mat3.transpose(l));
    return [m[0], m[1], m[2], m[5], m[6], m[10]];
}

// Message types
type PrecisionMode = 'f16' | 'f32';
type StartMsg = { type: 'start'; plyBuffer: ArrayBuffer; sh_degree: number; sort: ReorderMethod | null; memoryPrecision?: PrecisionMode };
type StartUrlMsg = { type: 'start_url'; url: string; sh_degree: number; sort: ReorderMethod | null; memoryPrecision?: PrecisionMode };

declare const self: DedicatedWorkerGlobalScope;

function processBuffer(plyBuffer: ArrayBuffer, sh_degree: number, sort: ReorderMethod | null, memoryPrecision: PrecisionMode = 'f16') {
    const header = INRIAV1PlyParser.decodeHeaderFromBuffer(plyBuffer);
    const num_points = header.vertexCount;
    const splat_data = INRIAV1PlyParser.findSplatData(plyBuffer, header);

    const final_sh_degree = Math.min(sh_degree, header.sphericalHarmonicsDegree);
    const num_coefs = (final_sh_degree + 1) * (final_sh_degree + 1);
    const sh_stride = 3 * num_coefs; // f16 elements per point

    const scalarBytes = memoryPrecision === 'f32' ? 4 : 2;
    const byteSizeG = PadTo4Bytes(num_points * gs_stride * scalarBytes);
    const byteSizeSh = PadTo4Bytes(num_points * sh_stride * scalarBytes);
    const useShared = self.crossOriginIsolated === true;
    self.postMessage({ type: 'buffer_info', useShared });
    const gbuf: ArrayBuffer | SharedArrayBuffer = useShared
        ? new SharedArrayBuffer(byteSizeG)
        : new ArrayBuffer(byteSizeG);
    const shbuf: ArrayBuffer | SharedArrayBuffer = useShared
        ? new SharedArrayBuffer(byteSizeSh)
        : new ArrayBuffer(byteSizeSh);
    const gaussian = memoryPrecision === 'f32' ? new Float32Array(gbuf) : new Float16Array(gbuf);
    const sh = memoryPrecision === 'f32' ? new Float32Array(shbuf) : new Float16Array(shbuf);

    let rawSplat: number[] = [];
    let offset_g = 0;
    let offset_sh = 0;
    let lastTick = performance.now();
    const tickIntervalMs = 200;

    // Track bounding box for Morton normalization
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

    for (let row = 0; row < num_points; row++) {
        INRIAV1PlyParser.readSplat(splat_data, header, row, 0, rawSplat);

    gaussian[offset_g + 0] = rawSplat[X];
    gaussian[offset_g + 1] = rawSplat[Y];
    gaussian[offset_g + 2] = rawSplat[Z];
    // update bbox
    if (rawSplat[X] < minX) minX = rawSplat[X]; if (rawSplat[X] > maxX) maxX = rawSplat[X];
    if (rawSplat[Y] < minY) minY = rawSplat[Y]; if (rawSplat[Y] > maxY) maxY = rawSplat[Y];
    if (rawSplat[Z] < minZ) minZ = rawSplat[Z]; if (rawSplat[Z] > maxZ) maxZ = rawSplat[Z];
        gaussian[offset_g + 3] = sigmoid(rawSplat[OPACITY]);
        const rot = quat.create(rawSplat[ROT_1], rawSplat[ROT_2], rawSplat[ROT_3], rawSplat[ROT_0]);
        quat.normalize(rot, rot);
        const scale = vec3.create(Math.exp(rawSplat[SCALE_0]), Math.exp(rawSplat[SCALE_1]), Math.exp(rawSplat[SCALE_2]));
        const cov = build_cov(rot, scale);
        gaussian.set(cov, offset_g + 4);

        if (rawSplat[F_DC_0] !== undefined) {
            sh[offset_sh + 0] = rawSplat[F_DC_0];
            sh[offset_sh + 1] = rawSplat[F_DC_1];
            sh[offset_sh + 2] = rawSplat[F_DC_2];
        } else if (rawSplat[RED] !== undefined) {
            sh[offset_sh + 0] = (rawSplat[RED] / 255.0 - 0.5) / SH_C0;
            sh[offset_sh + 1] = (rawSplat[GREEN] / 255.0 - 0.5) / SH_C0;
            sh[offset_sh + 2] = (rawSplat[BLUE] / 255.0 - 0.5) / SH_C0;
        } else {
            sh[offset_sh + 0] = 0;
            sh[offset_sh + 1] = 0;
            sh[offset_sh + 2] = 0;
        }

        // higher orders
        let orderBase = 1;
        for (let d = 1; d <= final_sh_degree; d++) {
            const degreeFields = (header as any).sphericalHarmonicsDegreeFields?.[d] as number[] | undefined;
            if (!degreeFields) break;
            const termCount = 2 * d + 1;
            for (let j = 0; j < termCount; j++) {
                for (let c = 0; c < 3; c++) {
                    const srcId = degreeFields[c * termCount + j];
                    sh[offset_sh + (orderBase + j) * 3 + c] = rawSplat[srcId];
                }
            }
            orderBase += termCount;
        }

        offset_g += gs_stride;
        offset_sh += sh_stride;

        const read = row + 1;
        const now = performance.now();
        if (now - lastTick >= tickIntervalMs) {
            // new detailed parse progress
            self.postMessage({ type: 'parse_progress', total: num_points, read });
            lastTick = now;
        }
    }

    // Morton 3D sort (21 bits per axis -> 63-bit Morton code using BigInt)
    const spanX = Math.max(1e-9, maxX - minX);
    const spanY = Math.max(1e-9, maxY - minY);
    const spanZ = Math.max(1e-9, maxZ - minZ);
    const BITS = 21;
    const MAXQ = (1 << BITS) - 1;

    function quant(v: number, minV: number, span: number): number {
        const t = (v - minV) / span;
        // clamp to [0,1]
        const u = t < 0 ? 0 : (t > 1 ? 1 : t);
        return Math.min(MAXQ, Math.max(0, Math.floor(u * MAXQ))) >>> 0;
    }

    // 21-bit spreading using 64-bit masks (BigInt)
    function part1by2_21(x: bigint): bigint {
        x &= 0x1fffffn; // 21 bits
        x = (x ^ (x << 32n)) & 0x1f00000000ffffn;
        x = (x ^ (x << 16n)) & 0x1f0000ff0000ffn;
        x = (x ^ (x << 8n))  & 0x100f00f00f00f00fn;
        x = (x ^ (x << 4n))  & 0x10c30c30c30c30c3n;
        x = (x ^ (x << 2n))  & 0x1249249249249249n;
        return x;
    }
    function morton3D64(xi: bigint, yi: bigint, zi: bigint): bigint {
        return part1by2_21(xi) | (part1by2_21(yi) << 1n) | (part1by2_21(zi) << 2n);
    }

    const sortKind = (sort ?? 'none');
    if (sortKind === 'hilbert') {
        throw new Error('Sort method "hilbert" is not implemented');
    }
    if (sortKind === 'peano') {
        throw new Error('Sort method "peano" is not implemented');
    }
    const doSort = sortKind === 'morton';
    function makeBuf(byteSize: number, shared: boolean): ArrayBuffer | SharedArrayBuffer {
        return shared ? new SharedArrayBuffer(byteSize) : new ArrayBuffer(byteSize);
    }
    function postDone(gauss: ArrayBuffer | SharedArrayBuffer, shbuf: ArrayBuffer | SharedArrayBuffer, useSharedLocal: boolean) {
        if (useSharedLocal) {
            self.postMessage({
                type: 'done',
                num_points,
                final_sh_degree,
                gs_stride,
                sh_stride,
                gaussian_cpu: gauss,
                sh_cpu: shbuf,
            });
        } else {
            self.postMessage(
                {
                    type: 'done',
                    num_points,
                    final_sh_degree,
                    gs_stride,
                    sh_stride,
                    ply_precision: memoryPrecision,
                    gaussian_cpu: gauss,
                    sh_cpu: shbuf,
                },
                [gauss as ArrayBuffer, shbuf as ArrayBuffer]
            );
        }
    }

    if (doSort) {
        // One-shot progress notification: sorting begins, include sort name for future extensibility
        self.postMessage({ type: 'sorting', sort: 'morton', method: 'Morton 3D', bitsPerAxis: BITS, points: num_points });
        const codes = new BigUint64Array(num_points);
        for (let i = 0; i < num_points; i++) {
            const px = gaussian[i * gs_stride + 0];
            const py = gaussian[i * gs_stride + 1];
            const pz = gaussian[i * gs_stride + 2];
            const xi = BigInt(quant(px, minX, spanX));
            const yi = BigInt(quant(py, minY, spanY));
            const zi = BigInt(quant(pz, minZ, spanZ));
            codes[i] = morton3D64(xi, yi, zi);
        }
        // build permutation 0..N-1 and sort by codes
        const perm = new Uint32Array(num_points);
        for (let i = 0; i < num_points; i++) perm[i] = i;
        const permArr = Array.from(perm as unknown as number[]);
        permArr.sort((a, b) => (codes[a] < codes[b] ? -1 : (codes[a] > codes[b] ? 1 : 0)));

        // Reorder into new buffers
    const newGaussBuf = makeBuf(PadTo4Bytes(num_points * gs_stride * scalarBytes), useShared);
    const newSHBuf    = makeBuf(PadTo4Bytes(num_points * sh_stride * scalarBytes), useShared);
        const gaussian_sorted = memoryPrecision === 'f32' ? new Float32Array(newGaussBuf) : new Float16Array(newGaussBuf);
        const sh_sorted = memoryPrecision === 'f32' ? new Float32Array(newSHBuf) : new Float16Array(newSHBuf);
        for (let i = 0; i < num_points; i++) {
            const src = permArr[i] >>> 0;
            gaussian_sorted.set(gaussian.subarray(src * gs_stride, (src + 1) * gs_stride), i * gs_stride);
            sh_sorted.set(sh.subarray(src * sh_stride, (src + 1) * sh_stride), i * sh_stride);
        }
    postDone(gaussian_sorted.buffer, sh_sorted.buffer, useShared);
    } else {
        // No sorting: return as-is
    postDone(gaussian.buffer, sh.buffer, useShared);
    }
}

self.onmessage = async (e: MessageEvent<StartMsg | StartUrlMsg>) => {
    const msg = e.data as StartMsg | StartUrlMsg;
    try {
        if (msg && msg.type === 'start') {
            processBuffer(msg.plyBuffer, msg.sh_degree, msg.sort, msg.memoryPrecision ?? 'f16');
            return;
        }
        if (msg && msg.type === 'start_url') {
            const response = await fetch(msg.url, { cache: 'no-store' });
            if (!response.ok) throw new Error(`Failed to fetch ${msg.url}: ${response.status} ${response.statusText}`);

            const total = parseInt(response.headers.get('content-length') ?? '0', 10);
            if (!response.body || !response.body.getReader) {
                // Fallback: no streaming
                const ab = await response.arrayBuffer();
                self.postMessage({ type: 'download_progress', loadedBytes: ab.byteLength, totalBytes: ab.byteLength, speedBps: Infinity });
                self.postMessage({ type: 'fetched', byteLength: ab.byteLength });
                processBuffer(ab, msg.sh_degree, msg.sort, msg.memoryPrecision ?? 'f16');
                return;
            }

            const reader = response.body.getReader();
            const chunks: Uint8Array[] = [];
            let loaded = 0;
            let lastTickLoaded = 0;
            let lastTick = performance.now();
            const tickIntervalMs = 120;

            // initial 0 progress
            self.postMessage({ type: 'download_progress', loadedBytes: 0, totalBytes: Number.isFinite(total) && total > 0 ? total : undefined, speedBps: 0 });

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                if (value) {
                    chunks.push(value);
                    loaded += value.byteLength;
                }
                const now = performance.now();
                if (now - lastTick >= tickIntervalMs) {
                    const deltaBytes = loaded - lastTickLoaded;
                    const deltaSecs = (now - lastTick) / 1000;
                    const speedBps = deltaSecs > 0 ? deltaBytes / deltaSecs : 0;
                    self.postMessage({ type: 'download_progress', loadedBytes: loaded, totalBytes: Number.isFinite(total) && total > 0 ? total : undefined, speedBps });
                    lastTickLoaded = loaded;
                    lastTick = now;
                }
            }

            // concat
            const buf = new Uint8Array(loaded);
            let offset = 0;
            for (const c of chunks) {
                buf.set(c, offset);
                offset += c.byteLength;
            }

            // final tick
            const now = performance.now();
            const deltaBytes = loaded - lastTickLoaded;
            const deltaSecs = (now - lastTick) / 1000;
            const speedBps = deltaSecs > 0 ? deltaBytes / Math.max(deltaSecs, 1e-6) : 0;
            self.postMessage({ type: 'download_progress', loadedBytes: loaded, totalBytes: Number.isFinite(total) && total > 0 ? total : undefined, speedBps });

            const ab = buf.buffer as ArrayBuffer;
            self.postMessage({ type: 'fetched', byteLength: ab.byteLength });
            processBuffer(ab, msg.sh_degree, msg.sort, msg.memoryPrecision ?? 'f16');
            return;
        }
    } catch (err: any) {
        self.postMessage({ type: 'error', message: String(err?.message ?? err) });
    }
};
