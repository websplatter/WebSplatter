// Shared sort types and helpers

export type ReorderMethod = 'none' | 'morton' | 'hilbert' | 'peano';

// Normalize a possibly user-provided value. Returns a canonical SortMethod or null.
export function normalizeReorder(input: string | null | undefined): ReorderMethod | null {
  if (input == null) return null;
  const s = String(input).toLowerCase();
  if (s === '0' || s === 'false' || s === 'none') return 'none';
  if (s === 'morton') return 'morton';
  if (s === 'hilbert') return 'hilbert';
  if (s === 'peano') return 'peano';
  return null;
}

export function isReorderEnabled(sort: ReorderMethod | null): boolean {
  return !!sort && sort !== 'none';
}
