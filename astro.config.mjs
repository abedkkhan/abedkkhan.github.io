import { defineConfig } from 'astro/config';

export default defineConfig({
  site: 'https://abedkkhan.github.io',
  markdown: {
    syntaxHighlight: 'shiki',
    shikiConfig: {
      // VS Code "Dark+" style — same palette as the screenshot
      theme: 'dark-plus',
      wrap: false,
    },
  },
});
