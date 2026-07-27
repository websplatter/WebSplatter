import rawPlugin from 'vite-raw-plugin';
import fs from 'fs';

const tlsCert = process.env.TLS_CERT;
const tlsKey = process.env.TLS_KEY;
const httpsConfig = tlsCert && tlsKey ? {
  cert: fs.readFileSync(tlsCert),
  key: fs.readFileSync(tlsKey),
} : undefined;

export default {
  base: './',
  build: {
    outDir: 'docs',
  },
  plugins: [
    rawPlugin({
      fileRegex: /\.wgsl$/,
    }),
  ],
  server: {
    https: httpsConfig,
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
};
