import './style.css';
import init from './splat-app';
import { assert } from '../utils/util';
import { error } from './simple-console';

async function start(): Promise<void> {
    if (!navigator.gpu) throw new Error('WebGPU is not supported in this browser.');

    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('No WebGPU adapter is available.');

    const features: GPUFeatureName[] = [];
    if (adapter.features.has('shader-f16')) features.push('shader-f16');

    const device = await adapter.requestDevice({
        requiredLimits: {
            maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
            maxBufferSize: adapter.limits.maxBufferSize,
            maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
            maxStorageBuffersPerShaderStage: 10,
        },
        requiredFeatures: features,
    });

    const canvas = document.querySelector<HTMLCanvasElement>('#webgpu-canvas');
    assert(canvas !== null);
    const context = canvas.getContext('webgpu');
    assert(context !== null);
    await init(canvas, context, device, features);
}

start().catch((cause: unknown) => {
    const message = cause instanceof Error ? cause.message : String(cause);
    error(message);
});
