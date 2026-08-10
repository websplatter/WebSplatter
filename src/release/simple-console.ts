const status = document.querySelector<HTMLDivElement>('#log');
let spinnerOverlay: HTMLDivElement | null = null;
let spinnerLabel: HTMLDivElement | null = null;

function ensureSpinner(): HTMLDivElement {
    if (spinnerOverlay) return spinnerOverlay;
    spinnerOverlay = document.createElement('div');
    spinnerOverlay.className = 'ply-spinner-overlay';

    const spinner = document.createElement('div');
    spinner.className = 'ply-spinner';
    spinnerOverlay.appendChild(spinner);

    spinnerLabel = document.createElement('div');
    spinnerLabel.className = 'ply-spinner-label';
    spinnerOverlay.appendChild(spinnerLabel);
    spinnerOverlay.style.display = 'none';
    document.body.appendChild(spinnerOverlay);
    return spinnerOverlay;
}

export function showPLYSpinner(message?: string): void {
    ensureSpinner();
    if (spinnerLabel && message) spinnerLabel.textContent = message;
    if (spinnerOverlay) spinnerOverlay.style.display = 'flex';
}

export function setPLYSpinnerStatus(message: string): void {
    ensureSpinner();
    if (spinnerLabel) spinnerLabel.textContent = message;
}

export function hidePLYSpinner(): void {
    if (spinnerOverlay) spinnerOverlay.style.display = 'none';
}

export function log(_message: string): void {}

export function error(message: string): void {
    console.error(message);
    if (status) status.textContent = message;
}

export function time(): void {}
export function timeLog(_label?: string): void {}
export function logColored(_message: string, _color: string): void {}
export function logProgress(_percent: number, _header?: string): void {}
