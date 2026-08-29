import { CameraControl } from './camera-control';

export function isDebugInteractionEnabled(params: URLSearchParams): boolean {
  return params.get('debug') === '1';
}

interface DebugInteractionOptions {
  canvas: HTMLCanvasElement;
  controls: CameraControl;
  onResetCamera: () => void;
}

function createButton(label: string, title: string): HTMLButtonElement {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'debug-hud__button';
  button.textContent = label;
  button.title = title;
  return button;
}

export function setupDebugInteractions(options: DebugInteractionOptions): () => void {
  const hud = document.createElement('aside');
  hud.className = 'debug-hud';
  hud.setAttribute('aria-label', 'Debug camera controls');

  const header = document.createElement('div');
  header.className = 'debug-hud__header';

  const title = document.createElement('strong');
  title.textContent = 'Debug controls';

  const status = document.createElement('span');
  status.className = 'debug-hud__status';
  status.setAttribute('role', 'status');

  const actions = document.createElement('div');
  actions.className = 'debug-hud__actions';

  const captureButton = createButton('Capture mouse', 'Toggle FPS-style mouse capture (`)');
  const resetButton = createButton('Reset view', 'Return to the first camera preset');
  const fullscreenButton = createButton('Fullscreen', 'Toggle browser fullscreen');

  const help = document.createElement('p');
  help.className = 'debug-hud__help';
  help.textContent = 'Drag rotate · Right-drag pan · Wheel zoom · ` capture · WASD move · Space/Ctrl fly relative to view · Shift boost · Q/E roll · R reset · Esc release';

  header.append(title, status);
  actions.append(captureButton, resetButton, fullscreenButton);
  hud.append(header, actions, help);
  document.body.append(hud);

  const syncPointerState = () => {
    const captured = options.controls.isPointerCaptured();
    status.textContent = captured ? 'Mouse captured' : 'Mouse free';
    status.classList.toggle('debug-hud__status--active', captured);
    captureButton.textContent = captured ? 'Release mouse' : 'Capture mouse';
  };

  const syncFullscreenState = () => {
    fullscreenButton.textContent = document.fullscreenElement ? 'Exit fullscreen' : 'Fullscreen';
  };

  const toggleFullscreen = async () => {
    if (document.fullscreenElement) {
      await document.exitFullscreen();
    } else {
      await options.canvas.requestFullscreen();
    }
  };

  captureButton.addEventListener('click', () => {
    options.canvas.focus({ preventScroll: true });
    options.controls.togglePointerCapture();
  });
  resetButton.addEventListener('click', options.onResetCamera);
  fullscreenButton.addEventListener('click', () => {
    void toggleFullscreen();
  });
  document.addEventListener('pointerlockchange', syncPointerState);
  document.addEventListener('fullscreenchange', syncFullscreenState);

  syncPointerState();
  syncFullscreenState();

  return () => {
    document.removeEventListener('pointerlockchange', syncPointerState);
    document.removeEventListener('fullscreenchange', syncFullscreenState);
    hud.remove();
  };
}
