import './style.css'
import init from './splat-app.ts';
import { assert } from './utils/util';

(async () => {
  if (navigator.gpu === undefined) {
    const h = document.querySelector('#title') as HTMLElement;
    h.innerText = 'WebGPU is not supported in this browser.';
    return;
  }
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: 'high-performance',
  });
  if (adapter === null) {
    const h = document.querySelector('#title') as HTMLElement;
    h.innerText = 'No adapter is available for WebGPU.';
    return;
  }
  const evalParam = new URLSearchParams(window.location.search).get('eval');
  const features_list: GPUFeatureName[] = [];
  // eval=2 (Raw mode): must NOT request timestamp-query to avoid GPU observation overhead
  // eval=3 (Breakdown mode) and all other modes: request timestamp-query
  if (evalParam === '3' && adapter.features.has('timestamp-query')) {
    features_list.push('timestamp-query');
  }
  // if support f16
  if (adapter.features.has('shader-f16')) {
    features_list.push('shader-f16');
  }
  // console.log(adapter.limits.maxComputeWorkgroupStorageSize);
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxStorageBuffersPerShaderStage: 10, // 16 storage buffers per shader stage
    },
    requiredFeatures: features_list
  });

  const canvas = document.querySelector<HTMLCanvasElement>('#webgpu-canvas');
  assert(canvas !== null);
  const context = canvas.getContext('webgpu') as GPUCanvasContext;

  init(canvas, context, device, features_list);
})();