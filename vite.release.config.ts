import { defineConfig } from 'vite';
import rawPlugin from 'vite-raw-plugin';
import { fileURLToPath } from 'node:url';

function stripBetween(source: string, start: string, end: string): string {
  const startIndex = source.indexOf(start);
  const endIndex = source.indexOf(end, startIndex);
  if (startIndex < 0 || endIndex < 0) {
    throw new Error(`Release strip marker not found: ${start.slice(0, 60)}`);
  }
  return source.slice(0, startIndex) + source.slice(endIndex);
}

function replaceRequired(source: string, search: string, replacement: string): string {
  const first = source.indexOf(search);
  if (first < 0 || source.indexOf(search, first + search.length) >= 0) {
    throw new Error(`Release replacement must match exactly once: ${search.slice(0, 60)}`);
  }
  return source.slice(0, first) + replacement + source.slice(first + search.length);
}

function releaseRendererPlugin() {
  return {
    name: 'websplatter-release-renderer',
    enforce: 'pre' as const,
    transform(source: string, id: string) {
      if (!id.endsWith('/src/gaussian-renderer.ts')) return null;

      let code = stripBetween(
        source,
        '    // --- Timestamp Query Resources',
        '    // --- Child Modules & Pipelines',
      );
      code = stripBetween(
        code,
        '    // One-shot perf dialog flag',
        '    constructor(',
      );
      code = stripBetween(
        code,
        '    /**\n     * Read back real GPU buffers',
        '    /**\n     * Executes a single frame',
      );
      code = stripBetween(
        code,
        '    /**\n     * Read per-frame stage breakdown',
        '    private _setupBuffers',
      );

      const runtimeMethods = code.indexOf('    // ========================================================================\n    // PRIVATE RUNTIME METHODS');
      const classEnd = code.lastIndexOf('\n}');
      if (runtimeMethods < 0 || classEnd < runtimeMethods) {
        throw new Error('Release runtime method markers not found.');
      }
      code = code.slice(0, runtimeMethods) + code.slice(classEnd);

      code = replaceRequired(
        code,
        "import { log } from './utils/simple-console.ts';",
        "import { log } from './release/simple-console.ts';",
      );
      code = replaceRequired(code, "        this.timeQueryEnabled = features_list.includes('timestamp-query');\n", '');
      code = replaceRequired(
        code,
        "        if (this.timeQueryEnabled) {\n            this._setupTimestampQueries();\n        }\n",
        '',
      );
      code = replaceRequired(code, '        const tq = this.timeQueryEnabled;', '        const tq = false;');
      return { code, map: null };
    },
  };
}

const releaseConsole = fileURLToPath(new URL('./src/release/simple-console.ts', import.meta.url));

export default defineConfig({
  root: 'release',
  base: './',
  build: {
    outDir: '../dist-release',
    emptyOutDir: true,
    target: 'es2022',
    minify: 'esbuild',
    cssMinify: true,
    sourcemap: false,
    reportCompressedSize: true,
  },
  esbuild: {
    drop: ['debugger'],
    pure: ['console.log', 'console.debug', 'console.info', 'console.time', 'console.timeEnd'],
  },
  resolve: {
    alias: [
      { find: /^\.\/utils\/simple-console(?:\.ts)?$/, replacement: releaseConsole },
    ],
  },
  plugins: [
    releaseRendererPlugin(),
    rawPlugin({ fileRegex: /\.wgsl$/ }),
  ],
});
