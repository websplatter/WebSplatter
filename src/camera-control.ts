import { vec3, mat3, mat4, quat } from 'wgpu-matrix';
import { Camera } from './camera';

export class CameraControl {
  element: HTMLCanvasElement;
  enabled: boolean = true;
  constructor(private camera: Camera) {
    this.register_element(camera.canvas);
  }

  register_element(value: HTMLCanvasElement) {
    if (this.element && this.element != value) {
      this.element.removeEventListener('pointerdown', this.downCallback.bind(this));
      this.element.removeEventListener('pointermove', this.moveCallback.bind(this));
      this.element.removeEventListener('pointerup', this.upCallback.bind(this));
      this.element.removeEventListener('wheel', this.wheelCallback.bind(this));
    }

    this.element = value;
    this.element.addEventListener('pointerdown', this.downCallback.bind(this));
    this.element.addEventListener('pointermove', this.moveCallback.bind(this));
    this.element.addEventListener('pointerup', this.upCallback.bind(this));
    this.element.addEventListener('wheel', this.wheelCallback.bind(this));
    this.element.addEventListener('contextmenu', (e) => { e.preventDefault(); });
  }

  private panning = false;
  private rotating = false;
  private lastX: number;
  private lastY: number;

  downCallback(event: PointerEvent) {
    if (!this.enabled) return;
    if (!event.isPrimary) {
      return;
    }

    if (event.button === 0) {
      this.rotating = true;
      this.panning = false;
    } else {
      this.rotating = false;
      this.panning = true;
    }
    this.lastX = event.pageX;
    this.lastY = event.pageY;
  }
  moveCallback(event: PointerEvent) {
    if (!this.enabled) return;
    if (!(this.rotating || this.panning)) {
      return;
    }

    const xDelta = event.pageX - this.lastX;
    const yDelta = event.pageY - this.lastY;
    this.lastX = event.pageX;
    this.lastY = event.pageY;

    if (this.rotating) {
      this.rotate(xDelta, yDelta);
    } else if (this.panning) {
      this.pan(xDelta, yDelta);
    }
  }
  upCallback(event: PointerEvent) {
    this.rotating = false;
    this.panning = false;
    event.preventDefault();
  }
  wheelCallback(event: WheelEvent) {
    if (!this.enabled) return;
    event.preventDefault();
    const delta = vec3.mulScalar(this.camera.look, -event.deltaY * 0.001);
    vec3.add(delta, this.camera.position, this.camera.position);
    this.camera.update_buffer();
  }

  rotate(xDelta: number, yDelta: number) {
    // const r = mat4.identity();
    // mat4.rotateY(r, -xDelta, r);
    // mat4.rotateX(r, yDelta, r);
    const r = mat4.fromQuat(quat.fromEuler(yDelta * 0.01, -xDelta * 0.01, 0, 'xyz'));

    mat4.mul(r, this.camera.rotation, this.camera.rotation);

    this.camera.update_buffer();
  }

  pan(xDelta: number, yDelta: number) {
    const d = vec3.copy(this.camera.up);
    vec3.mulScalar(d, -yDelta * 0.01, d);
    vec3.add(d, this.camera.position, this.camera.position);
    vec3.copy(this.camera.right, d);
    vec3.mulScalar(d, -xDelta * 0.01, d);
    vec3.add(d, this.camera.position, this.camera.position);
    this.camera.update_buffer();
  }
};