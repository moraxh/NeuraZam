// @ts-check
import { defineConfig } from 'astro/config';

import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
    vite: {
      server: {
          watch: { usePolling: true }
      },

      plugins: [tailwindcss()],
    },
});