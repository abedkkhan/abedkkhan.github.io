import { defineConfig } from 'astro/config';

export default defineConfig({
  site: 'https://abedkkhan.github.io',
  markdown: {
    // Plain code blocks — styled in global.css as a retro amber terminal
    syntaxHighlight: false,
  },
});
