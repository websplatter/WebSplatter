const html_log = document.querySelector('#log') as HTMLDivElement;

let spinnerOverlay: HTMLDivElement | null = null;
let spinnerLabel: HTMLDivElement | null = null;

function ensureSpinner() {
    if (spinnerOverlay) return spinnerOverlay;
    spinnerOverlay = document.createElement('div');
    spinnerOverlay.className = 'ply-spinner-overlay';

    const spinner = document.createElement('div');
    spinner.className = 'ply-spinner';
    spinnerOverlay.appendChild(spinner);

    // status label under the spinner
    spinnerLabel = document.createElement('div');
    spinnerLabel.className = 'ply-spinner-label';
    spinnerOverlay.appendChild(spinnerLabel);

    spinnerOverlay.style.display = 'none';
    document.body.appendChild(spinnerOverlay);
    return spinnerOverlay;
}

export function showPLYSpinner(status?: string) {
    ensureSpinner();
    if (spinnerLabel && status) spinnerLabel.textContent = status;
    if (spinnerOverlay) spinnerOverlay.style.display = 'flex';
}

export function setPLYSpinnerStatus(status: string) {
    ensureSpinner();
    if (spinnerLabel) spinnerLabel.textContent = status;
}

export function hidePLYSpinner() {
    if (!spinnerOverlay) return;
    spinnerOverlay.style.display = 'none';
}

function appendLog(text: string, style?: Partial<CSSStyleDeclaration>) {
    const p = document.createElement('p');
    p.innerText = text;
    if (style) Object.assign(p.style, style);
    html_log.appendChild(p);
}

export async function log(msg: string) {
    console.log(msg);
    appendLog(msg);
}

export async function error(msg: string) {
    console.error(msg);
    appendLog(msg, { color: 'red', backgroundColor: 'rgba(255, 0, 0, 0.1)' });
}

let t: number;

export function time() {
    t = performance.now();
}

export function timeLog(s?: string) {
    const d = performance.now() - t;
    log(`⏱️ ${s} Time: ${d.toFixed(0)} ms`);
}

// Colored log helper
export function logColored(msg: string, color: string) {
    console.log(msg);
    appendLog(msg, { color, fontWeight: '600' });
}

// Progress log: white -> bright green gradient
export function logProgress(percent: number, header?: string) {
    const t = Math.max(0, Math.min(1, percent / 100));
    const r_from = 255;
    const g_from = 255;
    const b_from = 255;
    const r_to = 146;
    const g_to = 175;
    const b_to = 214;
    const r = Math.round(r_from + (r_to - r_from) * t);
    const g = Math.round(g_from + (g_to - g_from) * t);
    const b = Math.round(b_from + (b_to - b_from) * t);
    const color = `rgb(${r}, ${g}, ${b})`;
    if (header) {
        logColored(`${header}: ${percent}%`, color);
    } else {
        logColored(`${percent}%`, color);
    }
}