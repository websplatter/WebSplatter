import { Camera, load_camera_presets } from '../camera';
import { CameraControl } from '../camera-control';
import { GaussianRenderer } from '../gaussian-renderer';
import { loadGltfFromURL } from '../gltf-loader';

const DEFAULT_MODEL_URL = '/demo/scenes/van_gogh_room/van_gogh_room_spz.glb';
const DEFAULT_CAMERA_URL = 'scenes/van_gogh_room/cameras.json';

export default async function init(
    canvas: HTMLCanvasElement,
    context: GPUCanvasContext,
    device: GPUDevice,
    features: GPUFeatureName[],
): Promise<void> {
    const params = new URLSearchParams(window.location.search);
    const modelUrl = params.get('model_url') ?? DEFAULT_MODEL_URL;
    const cameraUrl = params.get('camera_url') ?? DEFAULT_CAMERA_URL;
    const shDegree = Math.max(0, Math.min(4, Number.parseInt(params.get('clip_sh_degree') ?? '0', 10) || 0));

    const camera = new Camera(canvas, device);
    const controls = new CameraControl(camera);

    const resize = () => {
        const dpr = window.devicePixelRatio || 1;
        const width = Math.max(1, Math.ceil(canvas.clientWidth * dpr));
        const height = Math.max(1, Math.ceil(canvas.clientHeight * dpr));
        if (canvas.width === width && canvas.height === height) return;
        canvas.width = width;
        canvas.height = height;
        camera.on_update_canvas();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(canvas);
    resize();

    context.configure({
        device,
        format: 'rgba16float',
        alphaMode: 'opaque',
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.STORAGE_BINDING,
    });

    const cameras = await load_camera_presets(cameraUrl);
    if (cameras.length === 0) throw new Error('No camera presets are available.');
    camera.set_preset(cameras[0]);

    const pointcloud = await loadGltfFromURL(modelUrl, device, shDegree, null);
    const renderer = new GaussianRenderer(
        pointcloud,
        device,
        'rgba16float',
        camera.uniform_buffer,
        features,
        false,
        false,
        true,
    );

    let previousTime = performance.now();
    function frame(now: number): void {
        const deltaSeconds = (now - previousTime) / 1000;
        previousTime = now;
        controls.update(deltaSeconds);

        const encoder = device.createCommandEncoder();
        renderer.frame(encoder, context.getCurrentTexture().createView());
        device.queue.submit([encoder.finish()]);
        requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
}
