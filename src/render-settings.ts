// Helper utilities for the compact 32-byte RenderSettings uniform buffer
// Layout (32 bytes):
// 0  u32 canvas_size.x
// 4  u32 canvas_size.y
// 8  u32 max_sh_deg
// 12 u32 cur_sh_deg
// 16 f32 gaussian_scaling
// 20 f32 kernel_size
// 24 u32 mip_spatting
// 28 f32 walltime

export const RENDER_SETTINGS_SIZE = 32; // bytes

export const RenderSettingsValues = new ArrayBuffer(RENDER_SETTINGS_SIZE);
export const RenderSettingsViews = {
  canvas_size: new Uint32Array(RenderSettingsValues, 0, 2),
  max_sh_deg: new Uint32Array(RenderSettingsValues, 8, 1),
  cur_sh_deg: new Uint32Array(RenderSettingsValues, 12, 1),
  gaussian_scaling: new Float32Array(RenderSettingsValues, 16, 1),
  kernel_size: new Float32Array(RenderSettingsValues, 20, 1),
  mip_spatting: new Uint32Array(RenderSettingsValues, 24, 1),
  walltime: new Float32Array(RenderSettingsValues, 28, 1),
};

export function initRenderSettings(params: {
  width: number;
  height: number;
  gaussian_scaling?: number;
  max_sh_deg: number;
  cur_sh_deg?: number;
  mip_spatting?: boolean | number;
  kernel_size?: number;
  walltime?: number;
}) {
  RenderSettingsViews.canvas_size[0] = params.width >>> 0;
  RenderSettingsViews.canvas_size[1] = params.height >>> 0;
  RenderSettingsViews.max_sh_deg[0] = params.max_sh_deg >>> 0;
  RenderSettingsViews.cur_sh_deg[0] = (params.cur_sh_deg ?? params.max_sh_deg) >>> 0;
  RenderSettingsViews.gaussian_scaling[0] = params.gaussian_scaling ?? 1.0;
  RenderSettingsViews.kernel_size[0] = params.kernel_size ?? 0.3;
  RenderSettingsViews.mip_spatting[0] = (typeof params.mip_spatting === 'boolean' ? (params.mip_spatting ? 1 : 0) : (params.mip_spatting ?? 0)) >>> 0;
  RenderSettingsViews.walltime[0] = params.walltime ?? 0.0;
}

export function writeRenderSettings(device: GPUDevice, buffer: GPUBuffer) {
  device.queue.writeBuffer(buffer, 0, RenderSettingsValues);
}

// Individual setters (optionally flush immediately)
function flush(device: GPUDevice | null, buffer: GPUBuffer | null, commit: boolean) {
  if (commit && device && buffer) writeRenderSettings(device, buffer);
}

export function setCanvasSize(w: number, h: number, device?: GPUDevice, buffer?: GPUBuffer, commit: boolean = true) {
  RenderSettingsViews.canvas_size[0] = w >>> 0;
  RenderSettingsViews.canvas_size[1] = h >>> 0;
  flush(device ?? null, buffer ?? null, commit);
}
export function setCurSHDegree(v: number, device?: GPUDevice, buffer?: GPUBuffer, commit: boolean = true) {
  RenderSettingsViews.cur_sh_deg[0] = v >>> 0;
  flush(device ?? null, buffer ?? null, commit);
}
export function setGaussianScaling(v: number, device?: GPUDevice, buffer?: GPUBuffer, commit: boolean = true) {
  RenderSettingsViews.gaussian_scaling[0] = v;
  flush(device ?? null, buffer ?? null, commit);
}
export function setMipSplatting(enabled: boolean, device?: GPUDevice, buffer?: GPUBuffer, commit: boolean = true) {
  RenderSettingsViews.mip_spatting[0] = enabled ? 1 : 0;
  flush(device ?? null, buffer ?? null, commit);
}
export function setKernelSize(v: number, device?: GPUDevice, buffer?: GPUBuffer, commit: boolean = true) {
  RenderSettingsViews.kernel_size[0] = v;
  flush(device ?? null, buffer ?? null, commit);
}
export function setWalltime(v: number, device?: GPUDevice, buffer?: GPUBuffer, commit: boolean = true) {
  RenderSettingsViews.walltime[0] = v;
  flush(device ?? null, buffer ?? null, commit);
}
