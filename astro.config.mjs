import { defineConfig } from 'astro/config';
import { createHighlighter } from 'shiki';
import { visit } from 'unist-util-visit';
import { fromHtml } from 'hast-util-from-html';

// one highlighter instance, reused for every inline code node
const highlighter = await createHighlighter({
  themes: ['dark-plus'],
  langs: ['python'],
});

// rehype plugin: highlight inline <code> (not inside <pre>) as python
function rehypeInlineCode() {
  return (tree) => {
    visit(tree, 'element', (node, _index, parent) => {
      if (node.tagName !== 'code') return;
      if (parent && parent.type === 'element' && parent.tagName === 'pre') return;

      const first = node.children?.[0];
      if (!first || first.type !== 'text' || typeof first.value !== 'string') return;
      const text = first.value;
      if (!text.trim()) return;

      const html = highlighter.codeToHtml(text, { lang: 'python', theme: 'dark-plus' });
      // extract inner spans from <pre ...><code ...>INNER</code></pre>
      const match = html.match(/<code[^>]*>([\s\S]*?)<\/code>/);
      if (!match) return;
      const inner = match[1];

      const frag = fromHtml(inner, { fragment: true });
      node.children = frag.children;
      node.properties = node.properties || {};
      const existing = Array.isArray(node.properties.className) ? node.properties.className : [];
      node.properties.className = [...existing, 'inline-code'];
    });
  };
}

export default defineConfig({
  site: 'https://abedkkhan.github.io',
  markdown: {
    syntaxHighlight: 'shiki',
    shikiConfig: {
      theme: 'dark-plus',
      wrap: false,
    },
    rehypePlugins: [rehypeInlineCode],
  },
});
