import { defineConfig } from 'astro/config';

export default defineConfig({
  site: 'https://geetpurohit.com',
  output: 'static',
  compressHTML: true,
  prefetch: true,
  devToolbar: { enabled: false },
});
