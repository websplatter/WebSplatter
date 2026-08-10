import { vec3, mat4, quat } from 'wgpu-matrix';
import { Camera } from './camera';

export class CameraControl {
  element: HTMLCanvasElement;
  private _enabled: boolean = true;
  private readonly pressedKeys = new Set<string>();
  private readonly movementSpeed = 1.0;
  private readonly boostMultiplier = 3.0;
  private readonly rotationSpeed = 1.0;

  get enabled(): boolean {
    return this._enabled;
  }

  set enabled(value: boolean) {
    this._enabled = value;
    if (!value) {
      this.clearKeys();
      this.rotating = false;
      this.panning = false;
    }
  }

  constructor(private camera: Camera) {
    this.register_element(camera.canvas);
  }

  register_element(value: HTMLCanvasElement) {
    if (this.element && this.element != value) {
      this.element.removeEventListener('pointerdown', this.downCallback.bind(this));
      this.element.removeEventListener('pointermove', this.moveCallback.bind(this));
      this.element.removeEventListener('pointerup', this.upCallback.bind(this));
      this.element.removeEventListener('wheel', this.wheelCallback.bind(this));
      this.element.removeEventListener('keydown', this.keyDownCallback.bind(this));
      this.element.removeEventListener('keyup', this.keyUpCallback.bind(this));
      this.element.removeEventListener('blur', this.clearKeys.bind(this));
    }

    this.element = value;
    this.element.addEventListener('pointerdown', this.downCallback.bind(this));
    this.element.addEventListener('pointermove', this.moveCallback.bind(this));
    this.element.addEventListener('pointerup', this.upCallback.bind(this));
    this.element.addEventListener('wheel', this.wheelCallback.bind(this));
    this.element.addEventListener('keydown', this.keyDownCallback.bind(this));
    this.element.addEventListener('keyup', this.keyUpCallback.bind(this));
    this.element.addEventListener('blur', this.clearKeys.bind(this));
    this.element.addEventListener('contextmenu', (e) => { e.preventDefault(); });
    this.element.tabIndex = 0;
    window.addEventListener('blur', this.clearKeys.bind(this));
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) this.clearKeys();
    });
  }

  private panning = false;
  private rotating = false;
  private lastX: number;
  private lastY: number;

  downCallback(event: PointerEvent) {
    this.element.focus({ preventScroll: true });
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

  private isControlKey(code: string): boolean {
    return code === 'KeyW' || code === 'KeyA' || code === 'KeyS' || code === 'KeyD'
      || code === 'ArrowUp' || code === 'ArrowLeft' || code === 'ArrowDown' || code === 'ArrowRight'
      || code === 'KeyQ' || code === 'KeyE';
  }

  private keyDownCallback(event: KeyboardEvent) {
    if (event.code === 'ShiftLeft' || event.code === 'ShiftRight') {
      this.pressedKeys.add(event.code);
      return;
    }
    if (!this.enabled || !this.isControlKey(event.code)) return;
    if (event.ctrlKey || event.metaKey || event.altKey) return;
    this.pressedKeys.add(event.code);
    event.preventDefault();
  }

  private keyUpCallback(event: KeyboardEvent) {
    if (event.code === 'ShiftLeft' || event.code === 'ShiftRight') {
      this.pressedKeys.delete(event.code);
      return;
    }
    if (!this.isControlKey(event.code)) return;
    this.pressedKeys.delete(event.code);
    if (!event.ctrlKey && !event.metaKey && !event.altKey) event.preventDefault();
  }

  private clearKeys() {
    this.pressedKeys.clear();
  }

  update(deltaSeconds: number): boolean {
    if (!this.enabled || this.pressedKeys.size === 0) return false;

    const forward = Number(this.pressedKeys.has('KeyW')) - Number(this.pressedKeys.has('KeyS'));
    // camera.right uses the renderer's view-space convention, whose positive direction is screen-left.
    const strafe = Number(this.pressedKeys.has('KeyA')) - Number(this.pressedKeys.has('KeyD'));
    const pitch = Number(this.pressedKeys.has('ArrowDown')) - Number(this.pressedKeys.has('ArrowUp'));
    const yaw = Number(this.pressedKeys.has('ArrowLeft')) - Number(this.pressedKeys.has('ArrowRight'));
    const roll = Number(this.pressedKeys.has('KeyQ')) - Number(this.pressedKeys.has('KeyE'));
    if (forward === 0 && strafe === 0 && pitch === 0 && yaw === 0 && roll === 0) return false;

    const dt = Math.min(Math.max(deltaSeconds, 0), 0.1);
    if (forward !== 0 || strafe !== 0) {
      const length = Math.hypot(forward, strafe);
      const boosted = this.pressedKeys.has('ShiftLeft') || this.pressedKeys.has('ShiftRight');
      const distance = this.movementSpeed * (boosted ? this.boostMultiplier : 1) * dt / length;
      this.camera.position[0] += (this.camera.look[0] * forward + this.camera.right[0] * strafe) * distance;
      this.camera.position[1] += (this.camera.look[1] * forward + this.camera.right[1] * strafe) * distance;
      this.camera.position[2] += (this.camera.look[2] * forward + this.camera.right[2] * strafe) * distance;
    }
    if (pitch !== 0 || yaw !== 0 || roll !== 0) {
      const rotation = mat4.fromQuat(quat.fromEuler(
        pitch * this.rotationSpeed * dt,
        yaw * this.rotationSpeed * dt,
        roll * this.rotationSpeed * dt,
        'xyz',
      ));
      mat4.mul(rotation, this.camera.rotation, this.camera.rotation);
    }
    this.camera.update_buffer();
    return true;
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
