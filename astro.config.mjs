import { defineConfig } from 'astro/config';

export default defineConfig({
  site: 'https://abedk.dev',
  markdown: {
    // Plain code blocks — styled in global.css as a retro amber terminal
    syntaxHighlight: false,
  },
});
