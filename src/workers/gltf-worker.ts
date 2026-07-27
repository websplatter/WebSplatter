/// <reference lib="webworker" />

import { mat3, Quat, quat, Vec3, vec3 } from 'wgpu-matrix';
import { loadSpz } from '@spz-loader/core';
import type { ReorderMethod } from '../reorder-types';
import { PadTo4Bytes } from '../utils/util';

const gs_stride = 10; // 3 pos, 1 opacity, 6 cov
const SH_C0 = 0.28209479177387814;

declare const self: DedicatedWorkerGlobalScope;

// ---------- GLB parsing ----------

interface GltfJson {
    accessors: GltfAccessor[];
    bufferViews: GltfBufferView[];
    meshes: GltfMesh[];
    extensionsUsed?: string[];
}

interface GltfAccessor {
    bufferView?: number;
    byteOffset?: number;
    componentType: number;
    count: number;
    type: string;
    normalized?: boolean;
}

interface GltfBufferView {
    buffer: number;
    byteOffset?: number;
    byteLength: number;
    byteStride?: number;
}

interface GltfMesh {
    primitives: GltfPrimitive[];
}

interface GltfPrimitive {
    attributes: Record<string, number>;
    extensions?: {
        KHR_gaussian_splatting?: {
            extensions?: {
                KHR_gaussian_splatting_compression_spz_2?: {
                    bufferView: number;
                };
            };
            [key: string]: any;
        };
    };
}

// glTF component type constants
const COMPONENT_BYTE = 5120;
const COMPONENT_UNSIGNED_BYTE = 5121;
const COMPONENT_SHORT = 5122;
const COMPONENT_UNSIGNED_SHORT = 5123;
const COMPONENT_FLOAT = 5126;

function componentByteSize(componentType: number): number {
    switch (componentType) {
        case COMPONENT_BYTE:
        case COMPONENT_UNSIGNED_BYTE:
            return 1;
        case COMPONENT_SHORT:
        case COMPONENT_UNSIGNED_SHORT:
            return 2;
        case COMPONENT_FLOAT:
            return 4;
        default:
            throw new Error(`Unknown glTF componentType: ${componentType}`);
    }
}

function typeElementCount(type: string): number {
    switch (type) {
        case 'SCALAR': return 1;
        case 'VEC2': return 2;
        case 'VEC3': return 3;
        case 'VEC4': return 4;
        case 'MAT2': return 4;
        case 'MAT3': return 9;
        case 'MAT4': return 16;
        default: throw new Error(`Unknown glTF accessor type: ${type}`);
    }
}

/** Read a single float-valued element from an accessor at index `i`, component `c`. */
function readAccessorFloat(
    bin: DataView,
    accessor: GltfAccessor,
    bufferView: GltfBufferView,
    index: number,
    component: number,
): number {
    const elemCount = typeElementCount(accessor.type);
    const compSize = componentByteSize(accessor.componentType);
    const stride = bufferView.byteStride ?? (elemCount * compSize);
    const base = (bufferView.byteOffset ?? 0) + (accessor.byteOffset ?? 0) + index * stride + component * compSize;

    let raw: number;
    switch (accessor.componentType) {
        case COMPONENT_FLOAT:
            raw = bin.getFloat32(base, true);
            break;
        case COMPONENT_BYTE:
            raw = bin.getInt8(base);
            if (accessor.normalized) raw = Math.max(raw / 127.0, -1.0);
            break;
        case COMPONENT_UNSIGNED_BYTE:
            raw = bin.getUint8(base);
            if (accessor.normalized) raw = raw / 255.0;
            break;
        case COMPONENT_SHORT:
            raw = bin.getInt16(base, true);
            if (accessor.normalized) raw = Math.max(raw / 32767.0, -1.0);
            break;
        case COMPONENT_UNSIGNED_SHORT:
            raw = bin.getUint16(base, true);
            if (accessor.normalized) raw = raw / 65535.0;
            break;
        default:
            throw new Error(`Unsupported componentType ${accessor.componentType}`);
    }
    return raw;
}

// ---------- Covariance from rotation + scale ----------

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

// ---------- Morton 3D sort (same as ply-worker) ----------

const BITS = 21;
const MAXQ = (1 << BITS) - 1;

function quant(v: number, minV: number, span: number): number {
    const t = (v - minV) / span;
    const u = t < 0 ? 0 : (t > 1 ? 1 : t);
    return Math.min(MAXQ, Math.max(0, Math.floor(u * MAXQ))) >>> 0;
}

function part1by2_21(x: bigint): bigint {
    x &= 0x1fffffn;
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

// ---------- Attribute lookup helpers ----------

const NS = 'KHR_gaussian_splatting:';

/** Look up an attribute by trying namespaced then un-namespaced names. */
function findAttr(attrs: Record<string, number>, name: string): number | undefined {
    // Spec format: KHR_gaussian_splatting:OPACITY
    if (attrs[`${NS}${name}`] !== undefined) return attrs[`${NS}${name}`];
    // Fallback: un-namespaced (draft/legacy)
    if (attrs[name] !== undefined) return attrs[name];
    return undefined;
}

// ---------- Sort & post helpers ----------

interface BBox {
    minX: number; maxX: number;
    minY: number; maxY: number;
    minZ: number; maxZ: number;
}

function sortAndPost(
    gaussian: Float16Array,
    sh: Float16Array,
    num_points: number,
    sh_stride_val: number,
    final_sh_degree: number,
    sort: ReorderMethod | null,
    useShared: boolean,
    bbox: BBox,
): void {
    const { minX, maxX, minY, maxY, minZ, maxZ } = bbox;
    const spanX = Math.max(1e-9, maxX - minX);
    const spanY = Math.max(1e-9, maxY - minY);
    const spanZ = Math.max(1e-9, maxZ - minZ);

    const sortKind = sort ?? 'none';
    if (sortKind === 'hilbert') throw new Error('Sort method "hilbert" is not implemented');
    if (sortKind === 'peano') throw new Error('Sort method "peano" is not implemented');
    const doSort = sortKind === 'morton';

    function makeBuf(byteSize: number, shared: boolean): ArrayBuffer | SharedArrayBuffer {
        return shared ? new SharedArrayBuffer(byteSize) : new ArrayBuffer(byteSize);
    }

    function postDone(gauss: ArrayBuffer | SharedArrayBuffer, shOut: ArrayBuffer | SharedArrayBuffer) {
        if (useShared) {
            self.postMessage({
                type: 'done',
                num_points,
                final_sh_degree,
                gs_stride,
                sh_stride: sh_stride_val,
                gaussian_cpu: gauss,
                sh_cpu: shOut,
            });
        } else {
            self.postMessage(
                {
                    type: 'done',
                    num_points,
                    final_sh_degree,
                    gs_stride,
                    sh_stride: sh_stride_val,
                    gaussian_cpu: gauss,
                    sh_cpu: shOut,
                },
                [gauss as ArrayBuffer, shOut as ArrayBuffer]
            );
        }
    }

    if (doSort) {
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
        const perm = new Uint32Array(num_points);
        for (let i = 0; i < num_points; i++) perm[i] = i;
        const permArr = Array.from(perm as unknown as number[]);
        permArr.sort((a, b) => (codes[a] < codes[b] ? -1 : (codes[a] > codes[b] ? 1 : 0)));

        const newGaussBuf = makeBuf(PadTo4Bytes(num_points * gs_stride * 2), useShared);
        const newSHBuf = makeBuf(PadTo4Bytes(num_points * sh_stride_val * 2), useShared);
        const gaussian_sorted = new Float16Array(newGaussBuf);
        const sh_sorted = new Float16Array(newSHBuf);
        for (let i = 0; i < num_points; i++) {
            const src = permArr[i] >>> 0;
            gaussian_sorted.set(gaussian.subarray(src * gs_stride, (src + 1) * gs_stride), i * gs_stride);
            sh_sorted.set(sh.subarray(src * sh_stride_val, (src + 1) * sh_stride_val), i * sh_stride_val);
        }
        postDone(gaussian_sorted.buffer, sh_sorted.buffer);
    } else {
        postDone(gaussian.buffer, sh.buffer);
    }
}

// ---------- Main processing ----------

function parseGLB(buffer: ArrayBuffer): { json: GltfJson; bin: DataView } {
    const view = new DataView(buffer);
    const magic = view.getUint32(0, true);
    if (magic !== 0x46546C67) {
        throw new Error('Not a valid GLB file (bad magic)');
    }
    const version = view.getUint32(4, true);
    if (version !== 2) {
        throw new Error(`Unsupported glTF version: ${version}`);
    }

    // Chunk 0: JSON
    let offset = 12;
    const jsonChunkLength = view.getUint32(offset, true);
    const jsonChunkType = view.getUint32(offset + 4, true);
    if (jsonChunkType !== 0x4E4F534A) {
        throw new Error('First GLB chunk is not JSON');
    }
    const jsonBytes = new Uint8Array(buffer, offset + 8, jsonChunkLength);
    const jsonStr = new TextDecoder().decode(jsonBytes);
    const json = JSON.parse(jsonStr) as GltfJson;

    // Chunk 1: BIN
    offset += 8 + jsonChunkLength;
    const binChunkLength = view.getUint32(offset, true);
    const binChunkType = view.getUint32(offset + 4, true);
    if (binChunkType !== 0x004E4942) {
        throw new Error('Second GLB chunk is not BIN');
    }
    const binOffset = offset + 8;
    const bin = new DataView(buffer, binOffset, binChunkLength);

    return { json, bin };
}

async function processGLB_SPZ(
    json: GltfJson,
    bin: DataView,
    spzBufViewIdx: number,
    sh_degree: number,
    sort: ReorderMethod | null,
) {
    // Extract SPZ blob from BIN chunk
    const spzBv = json.bufferViews[spzBufViewIdx];
    const spzOffset = spzBv.byteOffset ?? 0;
    const spzLength = spzBv.byteLength;
    const spzBytes = new Uint8Array(bin.buffer, bin.byteOffset + spzOffset, spzLength);

    // Decompress SPZ
    const cloud = await loadSpz(spzBytes);
    const num_points = cloud.numPoints;

    // Determine SH degree
    const final_sh_degree = Math.min(sh_degree, Math.max(0, cloud.shDegree));
    const num_coefs = (final_sh_degree + 1) * (final_sh_degree + 1);
    const sh_stride_val = 3 * num_coefs;
    const num_higher_coefs = num_coefs - 1; // excluding DC

    self.postMessage({ type: 'parse_progress', total: num_points, read: 0 });

    // Allocate output buffers
    const byteSizeG = PadTo4Bytes(num_points * gs_stride * 2);
    const byteSizeSh = PadTo4Bytes(num_points * sh_stride_val * 2);
    const useShared = self.crossOriginIsolated === true;
    self.postMessage({ type: 'buffer_info', useShared });

    const gbuf = useShared ? new SharedArrayBuffer(byteSizeG) : new ArrayBuffer(byteSizeG);
    const shbuf = useShared ? new SharedArrayBuffer(byteSizeSh) : new ArrayBuffer(byteSizeSh);
    const gaussian = new Float16Array(gbuf);
    const sh = new Float16Array(shbuf);

    // Bounding box for Morton sort
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

    let offset_g = 0;
    let offset_sh = 0;
    let lastTick = performance.now();
    const tickIntervalMs = 200;

    for (let i = 0; i < num_points; i++) {
        // Positions (direct copy from Float32)
        const px = cloud.positions[i * 3 + 0];
        const py = cloud.positions[i * 3 + 1];
        const pz = cloud.positions[i * 3 + 2];
        gaussian[offset_g + 0] = px;
        gaussian[offset_g + 1] = py;
        gaussian[offset_g + 2] = pz;

        if (px < minX) minX = px; if (px > maxX) maxX = px;
        if (py < minY) minY = py; if (py > maxY) maxY = py;
        if (pz < minZ) minZ = pz; if (pz > maxZ) maxZ = pz;

        // Opacity: loadSpz already applied sigmoid, direct copy
        gaussian[offset_g + 3] = cloud.alphas[i];

        // Rotation: loadSpz returns XYZW, normalize then build_cov
        const rx = cloud.rotations[i * 4 + 0];
        const ry = cloud.rotations[i * 4 + 1];
        const rz = cloud.rotations[i * 4 + 2];
        const rw = cloud.rotations[i * 4 + 3];
        const rot = quat.create(rx, ry, rz, rw);
        quat.normalize(rot, rot);

        // Scale: loadSpz already applied exp(), do NOT exp() again
        const sx = cloud.scales[i * 3 + 0];
        const sy = cloud.scales[i * 3 + 1];
        const sz = cloud.scales[i * 3 + 2];
        const scale = vec3.create(sx, sy, sz);

        const cov = build_cov(rot, scale);
        gaussian.set(cov, offset_g + 4);

        // SH DC: reverse color transform  (color - 0.5) / SH_C0  → raw f_dc
        sh[offset_sh + 0] = (cloud.colors[i * 3 + 0] - 0.5) / SH_C0;
        sh[offset_sh + 1] = (cloud.colors[i * 3 + 1] - 0.5) / SH_C0;
        sh[offset_sh + 2] = (cloud.colors[i * 3 + 2] - 0.5) / SH_C0;

        // Higher-degree SH: direct copy from cloud.sh
        // cloud.sh layout: per-point, coefficient-major RGB triplets
        if (num_higher_coefs > 0 && cloud.sh.length > 0) {
            const srcBase = i * num_higher_coefs * 3;
            for (let c = 0; c < num_higher_coefs * 3; c++) {
                sh[offset_sh + 3 + c] = cloud.sh[srcBase + c];
            }
        }

        offset_g += gs_stride;
        offset_sh += sh_stride_val;

        const now = performance.now();
        if (now - lastTick >= tickIntervalMs) {
            self.postMessage({ type: 'parse_progress', total: num_points, read: i + 1 });
            lastTick = now;
        }
    }

    sortAndPost(gaussian, sh, num_points, sh_stride_val, final_sh_degree, sort, useShared,
        { minX, maxX, minY, maxY, minZ, maxZ });
}

async function processGLB(buffer: ArrayBuffer, sh_degree: number, sort: ReorderMethod | null) {
    const { json, bin } = parseGLB(buffer);

    // Validate extension
    if (!json.extensionsUsed?.includes('KHR_gaussian_splatting')) {
        throw new Error('GLB does not contain KHR_gaussian_splatting extension');
    }

    // Find first primitive with KHR_gaussian_splatting
    let primitive: GltfPrimitive | null = null;
    for (const mesh of json.meshes) {
        for (const prim of mesh.primitives) {
            if (prim.extensions?.KHR_gaussian_splatting) {
                primitive = prim;
                break;
            }
        }
        if (primitive) break;
    }
    if (!primitive) {
        throw new Error('No mesh primitive with KHR_gaussian_splatting extension found');
    }

    // Check for SPZ compression
    const spzExt = primitive.extensions?.KHR_gaussian_splatting?.extensions?.KHR_gaussian_splatting_compression_spz_2;
    if (spzExt) {
        return processGLB_SPZ(json, bin, spzExt.bufferView, sh_degree, sort);
    }

    const attrs = primitive.attributes;

    function getAccessor(idx: number): { accessor: GltfAccessor; bufferView: GltfBufferView } {
        const accessor = json.accessors[idx];
        if (accessor.bufferView === undefined) {
            throw new Error(`Accessor ${idx} has no bufferView (SPZ stub?)`);
        }
        const bufferView = json.bufferViews[accessor.bufferView];
        return { accessor, bufferView };
    }

    // POSITION accessor (required)
    if (attrs.POSITION === undefined) {
        throw new Error('Missing POSITION attribute');
    }
    const posInfo = getAccessor(attrs.POSITION);
    const num_points = posInfo.accessor.count;

    // Determine available SH degree from attributes
    // Spec: SH_DEGREE_0_COEF_0 (DC), SH_DEGREE_1_COEF_0..2, SH_DEGREE_2_COEF_0..4, SH_DEGREE_3_COEF_0..6
    let maxAvailableSHDegree = -1;
    if (findAttr(attrs, 'SH_DEGREE_0_COEF_0') !== undefined) {
        maxAvailableSHDegree = 0;
        for (let d = 1; d <= 4; d++) {
            if (findAttr(attrs, `SH_DEGREE_${d}_COEF_0`) !== undefined) {
                maxAvailableSHDegree = d;
            } else {
                break;
            }
        }
    }

    const final_sh_degree = Math.min(sh_degree, Math.max(0, maxAvailableSHDegree));
    const num_coefs = (final_sh_degree + 1) * (final_sh_degree + 1);
    const sh_stride_val = 3 * num_coefs;

    self.postMessage({ type: 'parse_progress', total: num_points, read: 0 });

    // Allocate output buffers
    const byteSizeG = PadTo4Bytes(num_points * gs_stride * 2);
    const byteSizeSh = PadTo4Bytes(num_points * sh_stride_val * 2);
    const useShared = self.crossOriginIsolated === true;
    self.postMessage({ type: 'buffer_info', useShared });

    const gbuf = useShared ? new SharedArrayBuffer(byteSizeG) : new ArrayBuffer(byteSizeG);
    const shbuf = useShared ? new SharedArrayBuffer(byteSizeSh) : new ArrayBuffer(byteSizeSh);
    const gaussian = new Float16Array(gbuf);
    const sh = new Float16Array(shbuf);

    // Resolve accessor infos for each attribute
    const opacityIdx = findAttr(attrs, 'OPACITY');
    const scaleIdx = findAttr(attrs, 'SCALE');
    const rotIdx = findAttr(attrs, 'ROTATION');

    const opacityInfo = opacityIdx !== undefined ? getAccessor(opacityIdx) : null;
    const scaleInfo = scaleIdx !== undefined ? getAccessor(scaleIdx) : null;
    const rotInfo = rotIdx !== undefined ? getAccessor(rotIdx) : null;

    // SH DC accessor — try spec name first, then COLOR_0 fallback
    const shDcIdx = findAttr(attrs, 'SH_DEGREE_0_COEF_0') ?? attrs.COLOR_0;
    const shDcInfo = shDcIdx !== undefined ? getAccessor(shDcIdx) : null;

    // Per-coefficient SH accessors for higher degrees
    type SHCoefInfo = { accessor: GltfAccessor; bufferView: GltfBufferView };
    const shCoefInfos: SHCoefInfo[][] = []; // shCoefInfos[d-1][j]
    for (let d = 1; d <= final_sh_degree; d++) {
        const termCount = 2 * d + 1;
        const coefs: SHCoefInfo[] = [];
        for (let j = 0; j < termCount; j++) {
            const idx = findAttr(attrs, `SH_DEGREE_${d}_COEF_${j}`);
            if (idx === undefined) break;
            coefs.push(getAccessor(idx));
        }
        if (coefs.length === termCount) {
            shCoefInfos.push(coefs);
        } else {
            break;
        }
    }

    // Bounding box for Morton sort
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

    let offset_g = 0;
    let offset_sh = 0;
    let lastTick = performance.now();
    const tickIntervalMs = 200;

    for (let i = 0; i < num_points; i++) {
        // Position (direct copy)
        const px = readAccessorFloat(bin, posInfo.accessor, posInfo.bufferView, i, 0);
        const py = readAccessorFloat(bin, posInfo.accessor, posInfo.bufferView, i, 1);
        const pz = readAccessorFloat(bin, posInfo.accessor, posInfo.bufferView, i, 2);
        gaussian[offset_g + 0] = px;
        gaussian[offset_g + 1] = py;
        gaussian[offset_g + 2] = pz;

        if (px < minX) minX = px; if (px > maxX) maxX = px;
        if (py < minY) minY = py; if (py > maxY) maxY = py;
        if (pz < minZ) minZ = pz; if (pz > maxZ) maxZ = pz;

        // Opacity: glTF stores post-sigmoid [0,1] — NO sigmoid needed
        if (opacityInfo) {
            gaussian[offset_g + 3] = readAccessorFloat(bin, opacityInfo.accessor, opacityInfo.bufferView, i, 0);
        } else {
            gaussian[offset_g + 3] = 1.0;
        }

        // Rotation: glTF KHR_gaussian_splatting uses (x,y,z,w)
        let rot: Quat;
        if (rotInfo) {
            const rx = readAccessorFloat(bin, rotInfo.accessor, rotInfo.bufferView, i, 0);
            const ry = readAccessorFloat(bin, rotInfo.accessor, rotInfo.bufferView, i, 1);
            const rz = readAccessorFloat(bin, rotInfo.accessor, rotInfo.bufferView, i, 2);
            const rw = readAccessorFloat(bin, rotInfo.accessor, rotInfo.bufferView, i, 3);
            rot = quat.create(rx, ry, rz, rw);
        } else {
            rot = quat.create(0, 0, 0, 1);
        }
        quat.normalize(rot, rot);

        // Scale: log-space -> exp()
        let scale: Vec3;
        if (scaleInfo) {
            const sx = Math.exp(readAccessorFloat(bin, scaleInfo.accessor, scaleInfo.bufferView, i, 0));
            const sy = Math.exp(readAccessorFloat(bin, scaleInfo.accessor, scaleInfo.bufferView, i, 1));
            const sz = Math.exp(readAccessorFloat(bin, scaleInfo.accessor, scaleInfo.bufferView, i, 2));
            scale = vec3.create(sx, sy, sz);
        } else {
            scale = vec3.create(1, 1, 1);
        }

        const cov = build_cov(rot, scale);
        gaussian.set(cov, offset_g + 4);

        // SH DC (degree 0)
        if (shDcInfo) {
            sh[offset_sh + 0] = readAccessorFloat(bin, shDcInfo.accessor, shDcInfo.bufferView, i, 0);
            sh[offset_sh + 1] = readAccessorFloat(bin, shDcInfo.accessor, shDcInfo.bufferView, i, 1);
            sh[offset_sh + 2] = readAccessorFloat(bin, shDcInfo.accessor, shDcInfo.bufferView, i, 2);
        } else {
            sh[offset_sh + 0] = 0;
            sh[offset_sh + 1] = 0;
            sh[offset_sh + 2] = 0;
        }

        // Higher-degree SH: each coefficient is a separate VEC3 accessor
        let orderBase = 1;
        for (let dIdx = 0; dIdx < shCoefInfos.length; dIdx++) {
            const coefs = shCoefInfos[dIdx];
            for (let j = 0; j < coefs.length; j++) {
                const ci = coefs[j];
                sh[offset_sh + (orderBase + j) * 3 + 0] = readAccessorFloat(bin, ci.accessor, ci.bufferView, i, 0);
                sh[offset_sh + (orderBase + j) * 3 + 1] = readAccessorFloat(bin, ci.accessor, ci.bufferView, i, 1);
                sh[offset_sh + (orderBase + j) * 3 + 2] = readAccessorFloat(bin, ci.accessor, ci.bufferView, i, 2);
            }
            orderBase += coefs.length;
        }

        offset_g += gs_stride;
        offset_sh += sh_stride_val;

        const now = performance.now();
        if (now - lastTick >= tickIntervalMs) {
            self.postMessage({ type: 'parse_progress', total: num_points, read: i + 1 });
            lastTick = now;
        }
    }

    sortAndPost(gaussian, sh, num_points, sh_stride_val, final_sh_degree, sort, useShared,
        { minX, maxX, minY, maxY, minZ, maxZ });
}

// ---------- Message handling ----------

type StartMsg = { type: 'start'; glbBuffer: ArrayBuffer; sh_degree: number; sort: ReorderMethod | null };
type StartUrlMsg = { type: 'start_url'; url: string; sh_degree: number; sort: ReorderMethod | null };

self.onmessage = async (e: MessageEvent<StartMsg | StartUrlMsg>) => {
    const msg = e.data as StartMsg | StartUrlMsg;
    try {
        if (msg && msg.type === 'start') {
            await processGLB(msg.glbBuffer, msg.sh_degree, msg.sort);
            return;
        }
        if (msg && msg.type === 'start_url') {
            const response = await fetch(msg.url, { cache: 'no-store' });
            if (!response.ok) throw new Error(`Failed to fetch ${msg.url}: ${response.status} ${response.statusText}`);

            const total = parseInt(response.headers.get('content-length') ?? '0', 10);
            if (!response.body || !response.body.getReader) {
                const ab = await response.arrayBuffer();
                self.postMessage({ type: 'download_progress', loadedBytes: ab.byteLength, totalBytes: ab.byteLength, speedBps: Infinity });
                self.postMessage({ type: 'fetched', byteLength: ab.byteLength });
                await processGLB(ab, msg.sh_degree, msg.sort);
                return;
            }

            const reader = response.body.getReader();
            const chunks: Uint8Array[] = [];
            let loaded = 0;
            let lastTickLoaded = 0;
            let lastTick = performance.now();
            const tickIntervalMs = 120;

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

            const buf = new Uint8Array(loaded);
            let offset = 0;
            for (const c of chunks) {
                buf.set(c, offset);
                offset += c.byteLength;
            }

            const now = performance.now();
            const deltaBytes = loaded - lastTickLoaded;
            const deltaSecs = (now - lastTick) / 1000;
            const speedBps = deltaSecs > 0 ? deltaBytes / Math.max(deltaSecs, 1e-6) : 0;
            self.postMessage({ type: 'download_progress', loadedBytes: loaded, totalBytes: Number.isFinite(total) && total > 0 ? total : undefined, speedBps });

            const ab = buf.buffer as ArrayBuffer;
            self.postMessage({ type: 'fetched', byteLength: ab.byteLength });
            await processGLB(ab, msg.sh_degree, msg.sort);
            return;
        }
    } catch (err: any) {
        self.postMessage({ type: 'error', message: String(err?.message ?? err) });
    }
};
