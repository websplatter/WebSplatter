// Note: You can import your separate WGSL shader files like this.

// import fragWGSL from './shaders/red.frag.wgsl';

import { mat4, vec3 } from 'wgpu-matrix';
import { loadFromFile, loadFromURL, PLYObject, PrecisionMode } from './ply-loader.ts';
import { loadGltfFromFile, loadGltfFromURL } from './gltf-loader.ts';

type ModelFormat = 'ply' | 'gltf';

function detectFormat(filename: string): ModelFormat {
    const lower = filename.toLowerCase();
    if (lower.endsWith('.glb') || lower.endsWith('.gltf')) return 'gltf';
    return 'ply';
}
import { ReorderMethod, normalizeReorder } from './reorder-types';
import { Pane } from 'tweakpane';
import { GaussianRenderer } from './gaussian-renderer';
import { TileBasedRendererGlobalSort } from './tile-based-renderer-global-sort.ts';
// import { TileBasedRendererLocalSort as TileBasedRenderer } from './tile-based-renderer-local-sort.ts';
import { PointCloudRenderer } from './point-cloud-renderer.ts';
import { Camera, CameraPreset, load_camera_presets } from './camera';
import { setGaussianScaling, setMipSplatting, setCurSHDegree, setKernelSize } from './render-settings.ts';
import { CameraControl } from './camera-control';
// import { TileBasedRendererLocalSort } from './tile-based-renderer-local-sort.ts';

export interface Renderers {
    frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => void,
    camera_buffer: GPUBuffer,
    readPerfMetrics: () => Promise<void>,
}

export interface GaussianRenderers extends Renderers {
    debugDumpPixelInputs: (px: number, py: number, width: number, height: number, limit?: number) => Promise<any>,
    maybeReorderAfterSubmit: () => Promise<void>,
    render_settings_buffer: GPUBuffer,
    requestReorder: () => void,
    requestDownloadMetrics: (filename: string) => void,
    requestPerfDialog: () => void,
    readBreakdown: () => Promise<{ cull_ms: number[], preprocess_ms: number[], sort_ms: number[], render_ms: number[] }>,
}

function promptUserForFile(accept: string): Promise<File | null> {
    return new Promise((resolve) => {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = accept;
        input.style.display = 'none';
        input.onchange = () => resolve(input.files?.[0] ?? null);
        document.body.appendChild(input);
        input.click();
        // optional cleanup
        setTimeout(() => document.body.removeChild(input), 1000);
    });
}


function setupUi(
    device: GPUDevice,
    reorder: ReorderMethod | null,
    onFileSelected: (pointcloud: PLYObject) => void,
    onUrlSelected: (modelUrl: string, shDegree: number, reorder: ReorderMethod | null, rendererParam: string | null) => void
) {
    const loadPanel = document.getElementById('ui-panel-container')!;
    const shDegreeInput = document.getElementById('sh-degree-input') as HTMLInputElement;
    const reorderSelect = document.getElementById('sort-select') as HTMLSelectElement;
    const rendererSelect = document.getElementById('renderer-select') as HTMLSelectElement | null;
    const animationCheckbox = document.getElementById('animation-checkbox') as HTMLInputElement | null;
    const loadButton = document.getElementById('load-button') as HTMLButtonElement;
    const linksContainer = document.getElementById('quick-links') as HTMLDivElement;
    const _setupParams = new URLSearchParams(window.location.search);
    const plyPrecisionParam = (_setupParams.get('ply_precision') ?? 'f16') as PrecisionMode;

    loadButton.onclick = async () => {
        const file = await promptUserForFile('.ply,.glb,.gltf');
        if (file) {
            const shDegree = parseInt(shDegreeInput.value) || 3;
            const selectedReorder = normalizeReorder(reorderSelect.value);
            loadPanel.style.display = 'none';
            document.getElementById('eval-panel')?.style.setProperty('display', 'none');
            const format = detectFormat(file.name);
            const pointcloud = format === 'gltf'
                ? await loadGltfFromFile(file, device, shDegree, selectedReorder)
                : await loadFromFile(file, device, shDegree, selectedReorder, plyPrecisionParam);
            onFileSelected(pointcloud);
        }
    };

    const updateLinks = () => {
        const currentShDegree = shDegreeInput.value || '3';
        const reorderStr = reorderSelect.value as ReorderMethod; // use current dropdown selection
        const rendererStr = rendererSelect ? rendererSelect.value : 'gaussian';
        const animationStr = animationCheckbox ? (animationCheckbox.checked ? '1' : '0') : '1';
        linksContainer.innerHTML = '';
        const scenes = [
            { label: 'Bicycle', path: 'scenes/bicycle/bicycle_30000.ply', cam: 'scenes/bicycle/cameras.json' },
            { label: 'Bicycle Cleaned', path: 'scenes/bicycle/bicycle_30000.cleaned.ply', cam: 'scenes/bicycle/cameras.json' },
            { label: 'Bonsai', path: 'scenes/bonsai/bonsai_30000.ply', cam: 'scenes/bonsai/cameras.json' },
            { label: 'Garden', path: 'scenes/garden/garden_30000.ply', cam: 'scenes/garden/cameras.json' },
            { label: 'Train', path: 'scenes/train/train_30000.ply', cam: 'scenes/train/cameras.json' },
            { label: 'Truck', path: 'scenes/truck/truck_30000.ply', cam: 'scenes/truck/cameras.json' },
            { label: 'Van Gogh Room', path: 'scenes/van_gogh_room/van_gogh_room.ply', cam: 'scenes/van_gogh_room/cameras.json' },
        ] as const;
        for (const s of scenes) {
            const url = `?model_url=${encodeURIComponent(s.path)}&camera_url=${encodeURIComponent(s.cam)}&clip_sh_degree=${encodeURIComponent(currentShDegree)}&sort=${encodeURIComponent(reorderStr)}&renderer=${encodeURIComponent(rendererStr)}&animation=${animationStr}`;
            const linkButton = document.createElement('a');
            linkButton.textContent = `🔗 ${s.label}`;
            linkButton.href = url;
            linkButton.className = 'ui-link';
            linkButton.target = '_blank';
            linksContainer.appendChild(linkButton);
        }
    };

    shDegreeInput.addEventListener('input', updateLinks);
    reorderSelect.addEventListener('change', updateLinks);
    rendererSelect?.addEventListener('change', updateLinks);
    animationCheckbox?.addEventListener('change', updateLinks);
    updateLinks();

    const paramsa = new URLSearchParams(window.location.search);
    const model_url = paramsa.get('model_url');
    if (model_url) {
        loadPanel.style.display = 'none';
        document.getElementById('eval-panel')?.style.setProperty('display', 'none');
        const sh_degree_param = paramsa.get('clip_sh_degree') ?? '3';
        const shDegree = parseInt(sh_degree_param);
        const rendererParam = paramsa.get('renderer');
        if (rendererSelect && rendererParam) {
            rendererSelect.value = rendererParam;
        }
        onUrlSelected(model_url, shDegree, reorder, rendererParam);
    }
}


// ---- Eval mode: render test views one by one ----
// Playwright drives the loop externally via __eval_* globals.
async function evalRenderOneView(
    canvas: HTMLCanvasElement,
    context: GPUCanvasContext,
    device: GPUDevice,
    camera: Camera,
    renderer: Renderers,
    preset: CameraPreset,
    downscale: number,
): Promise<Uint8Array> {
    camera.set_eval_preset(preset, downscale);

    // Re-configure the context for the new canvas size
    context.configure({
        device,
        format: 'rgba8unorm',
        alphaMode: 'opaque',
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC | GPUTextureUsage.STORAGE_BINDING,
    });

    // Create offscreen rgba16float render target for HDR blending (avoids clamping)
    const w = canvas.width;
    const h = canvas.height;
    const hdrTexture = device.createTexture({
        size: { width: w, height: h },
        format: 'rgba16float',
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    const hdrView = hdrTexture.createView();

    // Render a few warmup frames
    for (let warmup = 0; warmup < 5; warmup++) {
        const encoder = device.createCommandEncoder();
        renderer.frame(encoder, hdrView);
        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();
    }

    // Final render to HDR target
    const bytesPerRow = Math.ceil(w * 8 / 256) * 256; // rgba16float = 8 bytes/pixel

    const encoder = device.createCommandEncoder();
    renderer.frame(encoder, hdrView);

    // Create staging buffer for readback (rgba16float = 8 bytes/pixel)
    const stagingBuffer = device.createBuffer({
        size: bytesPerRow * h,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    encoder.copyTextureToBuffer(
        { texture: hdrTexture },
        { buffer: stagingBuffer, bytesPerRow },
        { width: w, height: h },
    );

    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();

    // Read back float pixel data and convert to u8 with clamping
    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const rawData = new Uint16Array(stagingBuffer.getMappedRange());

    // Helper: decode float16 to float32
    const f16ToF32 = (bits: number): number => {
        const s = (bits & 0x8000) ? -1 : 1;
        const e = (bits >> 10) & 0x1f;
        const f = bits & 0x03ff;
        if (e === 0) return f ? s * Math.pow(2, -14) * (f / 1024) : 0;
        if (e === 31) return f ? NaN : s * Infinity;
        return s * Math.pow(2, e - 15) * (1 + f / 1024);
    };

    // Convert rgba16float to rgba8unorm with clamp
    const pixelData = new Uint8Array(w * h * 4);
    const u16PerRow = bytesPerRow / 2; // bytesPerRow in u16 elements
    for (let row = 0; row < h; row++) {
        for (let col = 0; col < w; col++) {
            const srcIdx = row * u16PerRow + col * 4;
            const dstIdx = (row * w + col) * 4;
            for (let ch = 0; ch < 4; ch++) {
                const f = f16ToF32(rawData[srcIdx + ch]);
                pixelData[dstIdx + ch] = Math.max(0, Math.min(255, Math.round(f * 255)));
            }
        }
    }
    hdrTexture.destroy();
    stagingBuffer.unmap();
    stagingBuffer.destroy();

    const params = new URLSearchParams(window.location.search);
    const debugPx = params.get('debug_px');
    const debugPy = params.get('debug_py');
    const debugCamera = params.get('debug_camera');
    if (debugPx !== null && debugPy !== null && (!debugCamera || debugCamera === (preset.img_name ?? ''))) {
        const px = parseInt(debugPx, 10);
        const py = parseInt(debugPy, 10);
        const limit = parseInt(params.get('debug_limit') ?? '2000', 10);
        if (!Number.isNaN(px) && !Number.isNaN(py) && 'debugDumpPixelInputs' in (renderer as any)) {
            console.log(`[eval-debug] dumping pixel inputs for ${preset.img_name} @ (${px}, ${py}) ...`);
            (window as any).__debug_pixel_inputs = {
                camera: preset.img_name ?? '',
                px,
                py,
                width: w,
                height: h,
                pixel: Array.from(pixelData.slice((py * w + px) * 4, (py * w + px) * 4 + 4)),
                runtimeCamera: {
                    position: Array.from((camera as any).position ?? []),
                    rotation: Array.from((camera as any).rotation ?? []),
                    view_matrix: Array.from((camera as any).view_matrix ?? []),
                    proj_matrix: Array.from((camera as any).proj_matrix ?? []),
                    focal: Array.from((camera as any).focal ?? []),
                    viewport: Array.from((camera as any).viewport ?? []),
                },
                dump: await (renderer as GaussianRenderers).debugDumpPixelInputs(px, py, w, h, limit),
            };
            console.log(`[eval-debug] dumped ${((window as any).__debug_pixel_inputs.dump?.items ?? []).length} pixel input items`);
        }
    }

    return pixelData;
}

// Helper: convert RGBA pixel data to PNG data URL via OffscreenCanvas
async function pixelsToPngDataUrl(pixels: Uint8Array, w: number, h: number): Promise<string> {
    const imageData = new ImageData(new Uint8ClampedArray(pixels.buffer), w, h);
    const offscreen = new OffscreenCanvas(w, h);
    const ctx2d = offscreen.getContext('2d')!;
    ctx2d.putImageData(imageData, 0, 0);
    const blob = await offscreen.convertToBlob({ type: 'image/png' });
    return new Promise<string>((resolve) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve(reader.result as string);
        reader.readAsDataURL(blob);
    });
}

async function runEvalMode(
    canvas: HTMLCanvasElement,
    context: GPUCanvasContext,
    device: GPUDevice,
    camera: Camera,
    renderer: Renderers,
    cameras: CameraPreset[],
    downscale: number,
) {
    const results: { name: string, dataUrl: string }[] = [];
    (window as any).__eval_progress = { current: 0, total: cameras.length, done: false, results };

    for (let i = 0; i < cameras.length; i++) {
        const preset = cameras[i];
        const pixels = await evalRenderOneView(canvas, context, device, camera, renderer, preset, downscale);
        const w = canvas.width;
        const h = canvas.height;
        const dataUrl = await pixelsToPngDataUrl(pixels, w, h);
        const name = preset.img_name ?? `frame_${i}`;
        results.push({ name, dataUrl });
        console.log(`[eval] Captured ${i + 1}/${cameras.length}: ${name} (${w}x${h})`);
        (window as any).__eval_progress.current = i + 1;
    }

    (window as any).__eval_progress.done = true;
    console.log(`[eval] Done! Captured ${results.length} frames.`);
}

// ---- Auto-upload helper for eval=2/3 ----
async function uploadBenchResults(
    server: string, device: string, scene: string, config: string,
    payload: Record<string, unknown>,
) {
    const body = {
        id: crypto.randomUUID(),
        device,
        viewer: 'websplatter',
        scene,
        config,
        ua: navigator.userAgent,
        source_url: window.location.href,
        ...payload,
    };
    try {
        const resp = await fetch(`${server}/api/bench`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        const json = await resp.json();
        console.log(`[bench-upload] ${resp.ok ? 'OK' : 'FAIL'}`, json);
        return resp.ok;
    } catch (err) {
        console.warn('[bench-upload] Upload failed (server unreachable?):', err);
        return false;
    }
}

// ---- eval=2: Raw end-to-end performance (no timestamp-query) ----
// Uses JS-side performance.now() to measure frame times without GPU observation overhead.
async function runEvalRawMode(
    device: GPUDevice,
    context: GPUCanvasContext,
    renderer: Renderers,
    warmupFrames: number,
    measuredFrames: number,
    benchCtx?: { server: string; device: string; scene: string; config: string },
) {
    (window as any).__eval_progress = { current: 0, total: warmupFrames + measuredFrames, done: false, results: null };

    // Warmup phase
    for (let i = 0; i < warmupFrames; i++) {
        const encoder = device.createCommandEncoder();
        const texture_view = context.getCurrentTexture().createView();
        renderer.frame(encoder, texture_view);
        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();
        (window as any).__eval_progress.current = i + 1;
    }
    console.log(`[eval=2] Warmup done (${warmupFrames} frames)`);

    // Measured phase: collect per-frame wall-clock times
    const frameTimes: number[] = [];
    for (let i = 0; i < measuredFrames; i++) {
        const t0 = performance.now();
        const encoder = device.createCommandEncoder();
        const texture_view = context.getCurrentTexture().createView();
        renderer.frame(encoder, texture_view);
        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();
        const t1 = performance.now();
        frameTimes.push(t1 - t0);
        (window as any).__eval_progress.current = warmupFrames + i + 1;
    }

    // Compute statistics
    const sorted = [...frameTimes].sort((a, b) => a - b);
    const median = sorted[Math.floor(sorted.length / 2)];
    const p99 = sorted[Math.floor(sorted.length * 0.99)];
    const mean = frameTimes.reduce((a, b) => a + b, 0) / frameTimes.length;
    const variance = frameTimes.reduce((acc, v) => acc + (v - mean) ** 2, 0) / frameTimes.length;
    const std = Math.sqrt(variance);
    const fps = 1000 / median;

    const results = { median, p99, std, fps, mean, frame_count: measuredFrames, frame_times: frameTimes };
    (window as any).__eval_progress.results = results;
    (window as any).__eval_progress.done = true;

    console.log(`[eval=2] Raw performance results:`);
    console.log(`  median: ${median.toFixed(3)} ms`);
    console.log(`  p99:    ${p99.toFixed(3)} ms`);
    console.log(`  std:    ${std.toFixed(3)} ms`);
    console.log(`  fps:    ${fps.toFixed(1)}`);
    console.log(`  mean:   ${mean.toFixed(3)} ms`);

    // Auto-upload to bench server
    if (benchCtx) {
        const ok = await uploadBenchResults(benchCtx.server, benchCtx.device, benchCtx.scene, benchCtx.config, {
            frames: frameTimes,
            std,
        });
        if (ok) alert(`[eval=2] Bench results uploaded successfully (${benchCtx.scene}/${benchCtx.config})`);
    }
}

// ---- eval=3: Breakdown mode (requires timestamp-query) ----
// Uses GPU timestamp queries to measure per-stage pipeline timings.
async function runEvalBreakdownMode(
    device: GPUDevice,
    context: GPUCanvasContext,
    renderer: GaussianRenderers,
    warmupFrames: number,
    measuredFrames: number,
    benchCtx?: { server: string; device: string; scene: string; config: string },
) {
    (window as any).__eval_progress = { current: 0, total: warmupFrames + measuredFrames, done: false, results: null };

    // Warmup phase
    for (let i = 0; i < warmupFrames; i++) {
        const encoder = device.createCommandEncoder();
        const texture_view = context.getCurrentTexture().createView();
        renderer.frame(encoder, texture_view);
        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();
        (window as any).__eval_progress.current = i + 1;
    }
    // Drain warmup timestamps so they don't pollute measurement
    await renderer.readBreakdown();
    console.log(`[eval=3] Warmup done (${warmupFrames} frames)`);

    // Measured phase
    for (let i = 0; i < measuredFrames; i++) {
        const encoder = device.createCommandEncoder();
        const texture_view = context.getCurrentTexture().createView();
        renderer.frame(encoder, texture_view);
        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();
        (window as any).__eval_progress.current = warmupFrames + i + 1;
    }

    // Read back all stage breakdowns
    const breakdown = await renderer.readBreakdown();

    const stat = (arr: number[]) => {
        if (arr.length === 0) return { median: 0, p99: 0, std: 0, mean: 0 };
        const s = [...arr].sort((a, b) => a - b);
        const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
        const variance = arr.reduce((acc, v) => acc + (v - mean) ** 2, 0) / arr.length;
        return {
            median: s[Math.floor(s.length / 2)],
            p99: s[Math.floor(s.length * 0.99)],
            std: Math.sqrt(variance),
            mean,
        };
    };

    const results = {
        frame_count: breakdown.cull_ms.length,
        preprocess_ms: stat(breakdown.preprocess_ms),
        cull_ms: stat(breakdown.cull_ms),
        sort_ms: stat(breakdown.sort_ms),
        render_ms: stat(breakdown.render_ms),
        raw: breakdown,
    };
    (window as any).__eval_progress.results = results;
    (window as any).__eval_progress.done = true;

    console.log(`[eval=3] Breakdown results (${results.frame_count} frames):`);
    for (const stage of ['cull_ms', 'preprocess_ms', 'sort_ms', 'render_ms'] as const) {
        const s = results[stage];
        console.log(`  ${stage}: median=${s.median.toFixed(3)} p99=${s.p99.toFixed(3)} std=${s.std.toFixed(3)} mean=${s.mean.toFixed(3)}`);
    }

    // Auto-upload to bench server
    if (benchCtx) {
        const ok = await uploadBenchResults(benchCtx.server, benchCtx.device, benchCtx.scene, benchCtx.config, {
            preprocess_ms: results.preprocess_ms,
            cull_ms: results.cull_ms,
            sort_ms: results.sort_ms,
            render_ms: results.render_ms,
        });
        if (ok) alert(`[eval=3] Bench results uploaded successfully (${benchCtx.scene}/${benchCtx.config})`);
    }
}

export default async function init(
    canvas: HTMLCanvasElement,
    context: GPUCanvasContext,
    device: GPUDevice,
    features_list: GPUFeatureName[]
) {
    const camera = new Camera(canvas, device);
    const control = new CameraControl(camera);

    const paramsa = new URLSearchParams(window.location.search);
    const model_url = paramsa.get('model_url');
    const camera_url = paramsa.get('camera_url') ?? 'scenes/bicycle/cameras.json';
    const sh_degree_param = paramsa.get('clip_sh_degree') ?? '3';
    // URL sort method name (supports 'morton'; use 'none'/'0'/'false' to disable)
    const reorderParam = paramsa.get('sort') ?? null;
    // Canonical sort for all downstream calls (allow null to disable)
    const reorder: ReorderMethod | null = normalizeReorder(reorderParam);

    const animationParam = paramsa.get('animation');
    const evalMode = parseInt(paramsa.get('eval') ?? '0');
    const evalFrames = parseInt(paramsa.get('eval_frames') ?? '500');
    const evalWarmup = parseInt(paramsa.get('eval_warmup') ?? '100');
    const disableCull = paramsa.get('disable_cull') === '1';
    const disableRadius = paramsa.get('disable_radius') === '1';
    const evalDownscale = parseInt(paramsa.get('downscale') ?? '1');
    const kernelSizeParam = paramsa.get('kernel_size');
    const plyPrecisionParam = (paramsa.get('ply_precision') ?? 'f16') as PrecisionMode;
    const enableF16FeatureParam = (paramsa.get('enable_f16_feature') ?? '1') === '1';
    // Bench upload context (used by eval=2/3 auto-upload)
    const benchDevice = paramsa.get('bench_device') ?? 'unknown';
    const benchScene = paramsa.get('bench_scene') ?? 'unknown';
    // Auto-derive ablation config name from feature flags
    const benchConfig = paramsa.get('bench_config')
        ?? (disableCull && disableRadius ? '-CULL-RADIUS'
          : disableCull ? '-CULL'
          : disableRadius ? '-RADIUS'
          : 'FULL');
    const benchServer = paramsa.get('bench_server') ?? 'http://100.64.0.66:9876';

    let userInteracting = false;
    const onPointerDown = () => { userInteracting = true; };
    const onPointerUp = () => { userInteracting = false; };
    canvas.addEventListener('pointerdown', onPointerDown);
    window.addEventListener('pointerup', onPointerUp);

    const observer = new ResizeObserver(() => {
        if (evalMode) return; // Don't override canvas size in eval mode
        const dpr = window.devicePixelRatio || 1;
        canvas.width = Math.ceil(dpr * canvas.clientWidth);
        canvas.height = Math.ceil(dpr * canvas.clientHeight);

        camera.on_update_canvas();
    });
    if (!evalMode) {
        observer.observe(canvas);
    }

    const presentation_format = "rgba16float";
    context.configure({
        device,
        format: presentation_format,
        alphaMode: 'opaque',
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.STORAGE_BINDING,
    });
    let doRotate = !evalMode && animationParam === '1';

    control.enabled = !doRotate;

    const cameras = await load_camera_presets(camera_url);

    function applyCameraPreset(preset: CameraPreset): void {
        if (evalMode) {
            camera.set_eval_preset(preset, evalDownscale);
        } else {
            camera.set_preset(preset);
        }
    }

    // const url_base = '/scenes/bonsai';
    // const model_url = `${url_base}/bonsai_30000.ply`;
    applyCameraPreset(cameras[0]);

    // Tweakpane: easily adding tweak control for parameters.
    const params = {
        gaussian_scaling: 1,
        renderer: 'gaussian',
        animate: doRotate,
        sh_degree: parseInt(sh_degree_param),
        metrics_filename: 'fps_metrics',
        mip_splatting: false,
        // Default kernel size for mip-splatting and ellipse inflation
        kernel_size: kernelSizeParam !== null ? parseFloat(kernelSizeParam) : 0.3,
    };

    const rendererParamFromUrl = paramsa.get('renderer');

    const pane = new Pane({
        title: 'Config',
        expanded: true,
        container: document.getElementById('ui-panel-container')!,
    });

    pane.addInput(params, 'animate', { label: 'Animate' }).on('change', (e) => {
        doRotate = e.value;
        control.enabled = !e.value;
    });

    setupUi(
        device,
        reorder,
        (pointcloud) => init(pointcloud),
        async (modelUrl, shDegree, reorder, rendererParam) => {
            const format = detectFormat(modelUrl);
            const pointcloud = format === 'gltf'
                ? await loadGltfFromURL(modelUrl, device, shDegree, reorder)
                : await loadFromURL(modelUrl, device, shDegree, reorder, plyPrecisionParam);
            if (rendererParam) {
                params.renderer = rendererParam as any;
            }
            init(pointcloud);
        }
    );

    async function init(pointcloud: PLYObject) {
        // ---- Lazy (on-demand) renderer instantiation ----
        let gaussian_renderer: GaussianRenderer | null = null;
        let pointcloud_renderer: PointCloudRenderer | null = null;
        let tile_based_renderer: TileBasedRendererGlobalSort | null = null;

        function getGaussian(): GaussianRenderer {
            if (!gaussian_renderer) {
                gaussian_renderer = new GaussianRenderer(
                    pointcloud,
                    device,
                    presentation_format,
                    camera.uniform_buffer,
                    features_list,
                    disableCull,
                    disableRadius,
                    enableF16FeatureParam,
                );
                // Initial kernel size via helper
                setKernelSize(params.kernel_size, device, gaussian_renderer.render_settings_buffer);
            }
            return gaussian_renderer;
        }
        function getPointCloud(): PointCloudRenderer {
            return (pointcloud_renderer ||= new PointCloudRenderer(pointcloud, device, presentation_format, camera.uniform_buffer));
        }
        function getTileBased(): TileBasedRendererGlobalSort {
            if (!tile_based_renderer) {
                tile_based_renderer = new TileBasedRendererGlobalSort(pointcloud, device, presentation_format, camera.uniform_buffer, features_list);
            }
            return tile_based_renderer;
        }

        type RendererKey = 'pointcloud' | 'gaussian' | 'tile_based';
        function getRenderer(key: RendererKey): Renderers {
            switch (key) {
                case 'gaussian': return getGaussian();
                case 'pointcloud': return getPointCloud();
                case 'tile_based': return getTileBased();
            }
        }

        // If renderer is specified in URL, override default
        if (rendererParamFromUrl) {
            params.renderer = rendererParamFromUrl as any;
        }
        let renderer: Renderers = getRenderer(params.renderer as RendererKey);

        observer.observe(canvas);
        {
            // Renderer switcher (restore pointcloud browsing mode)
            pane
                .addInput(params, 'renderer', {
                    options: {
                        pointcloud: 'pointcloud',
                        gaussian: 'gaussian',
                        tile_based: 'tile_based',
                    },
                })
                .on('change', (e: any) => {
                    const key = e.value as RendererKey;
                    renderer = getRenderer(key);
                });

            const intermediate_gaussian_scale = new Float32Array([1]);
            pane.addInput(
                params,
                'gaussian_scaling',
                { min: 0, max: 1 }
            ).on('change', (e) => {
                intermediate_gaussian_scale[0] = e.value;
                setGaussianScaling(e.value, device, (renderer as GaussianRenderers).render_settings_buffer);
            });

            // Toggle mip-splatting (RenderSettings.mip_spatting at index 11)
            pane.addInput(params, 'mip_splatting', { label: 'Mip splatting' })
                .on('change', (e) => {
                    setMipSplatting(e.value, device, (renderer as GaussianRenderers).render_settings_buffer);
                });

            // Max SH degree control (clamped to loaded SH degree)
            const maxSHSupported = pointcloud.sh_degree;
            const maxShParam = { value: maxSHSupported };
            pane.addInput(maxShParam, 'value', { label: 'SH degree', min: 0, max: maxSHSupported, step: 1 })
                .on('change', (e) => {
                    const clamped = Math.max(0, Math.min(maxSHSupported, e.value | 0));
                    setCurSHDegree(clamped, device, (renderer as GaussianRenderers).render_settings_buffer);
                });

            // Kernel size control
            pane.addInput(params, 'kernel_size', { label: 'Kernel size', min: 0, max: 2, step: 0.01 })
                .on('change', (e) => {
                    setKernelSize(e.value, device, (renderer as GaussianRenderers).render_settings_buffer);
                });

            // Filename input for metrics CSV
            pane.addInput(params, 'metrics_filename', { label: 'Metrics filename' });

            // Add a button to download FPS metrics CSV on next sampling
            pane.addButton({ title: '💾 Download FPS metrics' }).on('click', () => {
                (renderer as GaussianRenderers).requestDownloadMetrics(params.metrics_filename);
            });

            // Add a button to show perf metrics in a dialog on next sampling
            pane.addButton({ title: '🔍 Show perf metrics' }).on('click', () => {
                (renderer as GaussianRenderers).requestPerfDialog();
            });

            // Add a button to request CPU-side reorder after next frame completes
            pane.addButton({ title: '🚀 Compact visible Gaussians' }).on('click', () => {
                (renderer as GaussianRenderers).requestReorder();
            });
        }
        document.addEventListener('keydown', (event) => {
            switch (event.key) {
                case '0':
                case '1':
                case '2':
                case '3':
                case '4':
                case '5':
                case '6':
                case '7':
                case '8':
                case '9':
                    const i = parseInt(event.key);
                    console.log(`set to camera preset ${i}`);
                    applyCameraPreset(cameras[i]);
                    break;
            }
        });

        // ---- Eval mode: run eval and skip interactive loop ----
        if (evalMode) {
            observer.disconnect(); // Prevent ResizeObserver from overriding canvas size
            if (evalMode === 1) {
                console.log(`[eval] Starting eval=1 (image capture) with ${cameras.length} cameras, downscale=${evalDownscale}`);
                await runEvalMode(canvas, context, device, camera, renderer, cameras, evalDownscale);
            } else if (evalMode === 2) {
                console.log(`[eval] Starting eval=2 (raw perf) warmup=${evalWarmup} frames=${evalFrames}`);
                applyCameraPreset(cameras[0]);
                const bCtx = { server: benchServer, device: benchDevice, scene: benchScene, config: benchConfig };
                await runEvalRawMode(device, context, renderer, evalWarmup, evalFrames, bCtx);
            } else if (evalMode === 3) {
                console.log(`[eval] Starting eval=3 (breakdown) warmup=${evalWarmup} frames=${evalFrames}`);
                applyCameraPreset(cameras[0]);
                const bCtx = { server: benchServer, device: benchDevice, scene: benchScene, config: benchConfig };
                await runEvalBreakdownMode(device, context, renderer as GaussianRenderers, evalWarmup, evalFrames, bCtx);
            }
            return; // Don't start interactive loop
        }

        let last_time = performance.now();
        // let rendered_frames = 0;
        let lastPerfLogMs = 0;

        async function frame() {
            const now = performance.now();
            const dt = (now - last_time)/1000;
            last_time = now;

            control.update(dt);

            if (doRotate && !userInteracting) { // animation=1 or default
                const speed = 0.2; // radians per second
                const angle = speed * dt;
                const m = mat4.rotationY(angle);
                vec3.transformMat4(camera.position, m, camera.position);
                
                const mInv = mat4.rotationY(-angle);
                mat4.mul(camera.rotation, mInv, camera.rotation);
                camera.update_buffer();
            }

            const encoder = device.createCommandEncoder();
            const texture_view = context.getCurrentTexture().createView();
            renderer.frame(encoder, texture_view);
            device.queue.submit([encoder.finish()]);
            // if (now - lastPerfLogMs >= 2000) {
            //     renderer.readPerfMetrics();
            //     lastPerfLogMs = now;
            // }
            requestAnimationFrame(frame);
        }
        requestAnimationFrame(frame);
    }

}
