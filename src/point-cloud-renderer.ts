import { PlyLoader } from './ply-loader.ts';
import pointcloud_wgsl from './shaders/point_cloud.wgsl';
import { Renderers } from './splat-app.ts';

// This renderer only has one pass, so we only need two timestamps (start and end).
const C_TIMESTAMP_COUNT = 2;

/**
 * Implements a simple renderer for visualizing a PlyLoader object.
 * This class handles the setup of pipelines and resources for drawing points
 * and includes performance metric tracking.
 */
export class PointCloudRenderer implements Renderers {
    private device: GPUDevice;
    private pc: PlyLoader;
    private presentationFormat: GPUTextureFormat;

    // --- Core Resources ---
    public camera_buffer: GPUBuffer;

    // --- Timestamp Query Resources ---
    private querySet: GPUQuerySet;
    private resolveBuffer: GPUBuffer;
    private resultBuffer: GPUBuffer;

    // --- Rendering Pipeline ---
    private renderPipeline: GPURenderPipeline;

    // --- Bind Groups ---
    private cameraBindGroup: GPUBindGroup;
    private gaussianBindGroup: GPUBindGroup;

    constructor(
        pc: PlyLoader,
        device: GPUDevice,
        presentationFormat: GPUTextureFormat,
        camera_buffer: GPUBuffer
    ) {
        this.pc = pc;
        this.device = device;
        this.presentationFormat = presentationFormat;
        this.camera_buffer = camera_buffer;

        this._setupTimestampQueries();
        this._setupPipeline();
        this._setupBindGroups();
    }

    /**
     * Executes a single frame of rendering for the point cloud.
     * @param encoder - The command encoder for this frame.
     * @param texture_view - The canvas texture view to render to.
     */
    public frame(encoder: GPUCommandEncoder, texture_view: GPUTextureView): void {
        const pass = encoder.beginRenderPass({
            label: 'point cloud render',
            colorAttachments: [{
                view: texture_view,
                loadOp: 'clear',
                storeOp: 'store',
                clearValue: [0.0, 0.0, 0.0, 1.0],
            }],
            // Add timestamp writes to the render pass
            timestampWrites: {
                querySet: this.querySet,
                beginningOfPassWriteIndex: 0, // T0
                endOfPassWriteIndex: 1,       // T1
            },
        });

        pass.setPipeline(this.renderPipeline);
        pass.setBindGroup(0, this.cameraBindGroup);
        pass.setBindGroup(1, this.gaussianBindGroup);
        pass.draw(this.pc.num_points);
        pass.end();

        // Resolve the timestamps after the pass
        this._resolveTimestamps(encoder);
    }

    /**
     * Reads the performance metrics from the last completed frame.
     */
    public async readPerfMetrics(): Promise<void> {
        await this.resultBuffer.mapAsync(GPUMapMode.READ);
        const times = new BigInt64Array(this.resultBuffer.getMappedRange());
        
        const renderTime = Number(times[1] - times[0]) / 1_000_000; // to ms
        const renderMethod = this.constructor.name;
        console.log(`[TIMESTAMP - ${renderMethod}] Point Cloud Render: ${renderTime.toFixed(3)}ms`);

        this.resultBuffer.unmap();
    }

    // ========================================================================
    // PRIVATE SETUP METHODS
    // ========================================================================

    /**
     * Initializes GPU resources for timestamp queries.
     */
    private _setupTimestampQueries(): void {
        this.querySet = this.device.createQuerySet({
            type: "timestamp",
            count: C_TIMESTAMP_COUNT,
        });

        this.resolveBuffer = this.device.createBuffer({
            size: C_TIMESTAMP_COUNT * 8,
            usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
        });

        this.resultBuffer = this.device.createBuffer({
            size: C_TIMESTAMP_COUNT * 8,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
    }
    
    /**
     * Creates the render pipeline for drawing the point cloud.
     */
    private _setupPipeline(): void {
        // This method's body is unchanged
        const renderShader = this.device.createShaderModule({ code: pointcloud_wgsl });
        this.renderPipeline = this.device.createRenderPipeline({
            label: 'point cloud render pipeline',
            layout: 'auto',
            vertex: { module: renderShader, entryPoint: 'vs_main' },
            fragment: { module: renderShader, entryPoint: 'fs_main', targets: [{ format: this.presentationFormat }] },
            primitive: { topology: 'point-list' },
        });
    }

    /**
     * Creates bind groups to link GPU resources to the render pipeline.
     */
    private _setupBindGroups(): void {
        // This method's body is unchanged
        this.cameraBindGroup = this.device.createBindGroup({
            label: 'point cloud camera',
            layout: this.renderPipeline.getBindGroupLayout(0),
            entries: [{ binding: 0, resource: { buffer: this.camera_buffer } }],
        });

        this.gaussianBindGroup = this.device.createBindGroup({
            label: 'point cloud gaussians',
            layout: this.renderPipeline.getBindGroupLayout(1),
            entries: [{ binding: 0, resource: { buffer: this.pc.gaussian_3d_buffer } }],
        });
    }

    /**
     * Records commands to resolve and copy timestamp query data.
     */
    private _resolveTimestamps(encoder: GPUCommandEncoder): void {
        encoder.resolveQuerySet(
            this.querySet, 0, C_TIMESTAMP_COUNT, this.resolveBuffer, 0
        );
        encoder.copyBufferToBuffer(
            this.resolveBuffer, 0, this.resultBuffer, 0, C_TIMESTAMP_COUNT * 8
        );
    }
}