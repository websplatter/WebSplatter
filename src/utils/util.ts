/**
 * Asserts `condition` is true. Otherwise, throws an `Error` with the provided message.
 */

export const clamp = function(val, min, max) {
    return Math.max(Math.min(val, max), min);
};

export const getSphericalHarmonicsComponentCountForDegree = (sphericalHarmonicsDegree = 0) => {
  // Total components for RGB across degrees 1..D is 3 * D * (D + 2)
  // D=0 -> 0, D=1 -> 9, D=2 -> 24, D=3 -> 45, etc.
  const d = Math.max(0, Math.floor(sphericalHarmonicsDegree));
  if (d === 0) return 0;
  return 3 * d * (d + 2);
};

export function assert(
    condition: boolean,
    msg?: string | (() => string)
  ): asserts condition {
    if (!condition) {
      throw new Error(msg && (typeof msg === 'string' ? msg : msg()));
    }
  }

/** If the argument is an Error, throw it. Otherwise, pass it back. */
export function assertOK<T>(value: Error | T): T {
if (value instanceof Error) {
    throw value;
}
return value;
}

/**
 * Assert this code is unreachable. Unconditionally throws an `Error`.
 */
export function unreachable(msg?: string): never {
throw new Error(msg);
}

/** Round `n` up to the next multiple of `alignment` (inclusive). */
export function align(n: number, alignment: number): number {
  assert(Number.isInteger(n) && n >= 0, 'n must be a non-negative integer');
  assert(Number.isInteger(alignment) && alignment > 0, 'alignment must be a positive integer');
  return Math.ceil(n / alignment) * alignment;
}

export function sigmoid(x: number): number {
  if (x >= 0.) {
    return 1. / (1. + Math.exp(-x))
  }
  return  Math.exp(x) / (1. + Math.exp(x))
}

export function PadTo4Bytes(size: number): number {
    return (size + 3) & (~3);
}