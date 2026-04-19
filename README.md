# abedkkhan.github.io — Personal Website

Personal website and blog for Abed K. Built with Astro, deployed on GitHub Pages.

**Live URL:** https://abedkkhan.github.io/
**Repo:** https://github.com/abedkkhan/abedkkhan.github.io
**Framework:** [Astro](https://astro.build/) v4 (static site generator)

---

## Tech Stack

- **Astro** — static site generator, zero JS by default
- **Markdown** — all blog posts are `.md` files
- **GitHub Actions** — auto-builds and deploys on every push
- **GitHub Pages** — free hosting at `abedkkhan.github.io`
- **No CMS, no database** — everything is just files in this repo

---

## Project Structure

```
my_own_website/
├── public/                     # Static files served as-is
│   ├── profile.png             # Profile photo (replace to update)
│   ├── favicon.svg
│   └── blog/
│       └── <slug>/             # Images for each blog post
│           ├── image_1.png
│           └── ...
├── src/
│   ├── content/
│   │   └── blog/               # All blog posts live here as .md files
│   │       └── my-post.md
│   ├── layouts/
│   │   └── Base.astro          # HTML shell, loads global.css
│   ├── pages/
│   │   ├── index.astro         # Homepage (about, links, research, blogs)
│   │   └── blog/
│   │       └── [...slug].astro # Blog post template (auto-generated)
│   └── styles/
│       └── global.css          # All styling — dark terminal theme
├── .github/
│   └── workflows/
│       └── deploy.yml          # GitHub Actions deploy pipeline
├── astro.config.mjs            # Astro config (site URL, markdown settings)
└── package.json
```

---

## Day-to-Day Workflow

All changes follow the same pattern:

```bash
# 1. Make changes locally (edit files, add blog posts, etc.)

# 2. Commit and push
git add .
git commit -m "your message"
git push

# 3. Done — GitHub Actions builds and deploys automatically in ~30 seconds
```

Check deploy status at: https://github.com/abedkkhan/abedkkhan.github.io/actions

---

## How to Add a New Blog Post

### Step 1 — Create the markdown file

Create a new file in `src/content/blog/` named with a URL-friendly slug:

```
src/content/blog/my-post-title.md
```

The filename becomes the URL: `https://abedkkhan.github.io/blog/my-post-title/`

### Step 2 — Add frontmatter at the top

```markdown
---
title: "Your Full Post Title Here"
description: "One sentence summary of the post"
pubDate: 2024-06-15
readTime: "7 min read"
tags: ["tag1", "tag2"]
---

Your post content starts here...
```

- `pubDate` — use the original publication date to preserve it (format: YYYY-MM-DD)
- `readTime` — optional, shows up in the blog listing
- `tags` — optional array, not displayed yet but useful for future filtering
- `description` — used for SEO/meta tags

### Step 3 — Add images (if any)

Create a folder in `public/blog/` matching your slug:

```
public/blog/my-post-title/
    image_1.png
    image_2.png
```

Reference them in your markdown as:

```markdown
![Alt text](/blog/my-post-title/image_1.png)
```

### Step 4 — Push

```bash
git add .
git commit -m "add blog post: my post title"
git push
```

The post automatically appears in the blog listing on the homepage, sorted by date.

---

## How to Update the Homepage

Open `src/pages/index.astro`. Everything is clearly labeled.

### Update the About section

Find the `<!-- About -->` comment and edit the paragraph text directly in the HTML.

### Add a new research paper / patent / dataset

Find the `research` array near the top of `index.astro`:

```js
const research = [
  {
    title: 'Paper title',
    venue: 'arXiv, 2025',
    url: 'https://arxiv.org/abs/...',
    kind: 'arxiv',   // options: 'arxiv', 'patent', 'dataset'
  },
  // add new entries here
];
```

### Update links (email, twitter, github, linkedin)

Find the `links` array near the top of `index.astro` and edit the URLs.

---

## How to Update the Profile Photo

Replace `public/profile.png` with your new photo, keeping the same filename. Push to deploy.

The image displays at 16:9 aspect ratio, max 460px wide.

---

## Styling

All visual styling is in `src/styles/global.css`.

Key design tokens (colors, fonts) are CSS variables defined at the top:

```css
:root {
  --bg: #0b0b0c;           /* main background */
  --accent: #f59e0b;        /* amber — used for highlights */
  --fg: #e5e5e5;            /* main text */
  --fg-dim: #a1a1aa;        /* secondary text */
  --mono: 'JetBrains Mono', ...;
}
```

Code blocks in blog posts use an old-school amber terminal style (black background, `#f0a020` text, Courier font). This is intentional — defined in the `.prose pre` and `.prose code` rules.

---

## Local Development

```bash
npm install       # first time only
npm run dev       # starts local server at http://localhost:4321
npm run build     # build to /dist (same as what GitHub Actions does)
```

---

## Deployment

Deployments are fully automatic via GitHub Actions (`.github/workflows/deploy.yml`).

- Every `git push` to `main` triggers a build
- Astro builds the site to `/dist`
- GitHub Pages serves the `/dist` output
- No manual steps needed

If a deploy fails, check: https://github.com/abedkkhan/abedkkhan.github.io/actions

---

## Custom Domain (Future)

To add a custom domain (e.g. `abedk.dev`):

1. Buy the domain (Cloudflare Registrar recommended — at-cost pricing)
2. In this repo, create a file `public/CNAME` containing just:
   ```
   abedk.dev
   ```
3. In GitHub repo Settings → Pages → Custom domain, enter your domain
4. At your domain registrar, add these DNS records:
   ```
   A    @    185.199.108.153
   A    @    185.199.109.153
   A    @    185.199.110.153
   A    @    185.199.111.153
   ```
5. Wait ~10 minutes for DNS propagation. GitHub will issue a free SSL cert automatically.

---

## Blog Posts Migration Status

Posts originally published on Hashnode (aabidkarim.hashnode.dev). Migrated here to preserve original dates.

| # | Title | Status |
|---|-------|--------|
| 1 | How basic concept of Calculus (Derivative) has a Key Role in Training Neural Networks? | Done |
| 2-13 | Remaining posts | Pending |

---

## Context for Claude / AI Sessions

If you're an AI assistant helping maintain this site, here is everything you need to know:

- **Owner:** Abed K, AI research engineer at 55mV Research Lab
- **Framework:** Astro v4, static, no JS framework
- **Deployed:** GitHub Pages via GitHub Actions, auto-deploy on push to `main`
- **Live URL:** https://abedkkhan.github.io/
- **GitHub repo:** https://github.com/abedkkhan/abedkkhan.github.io
- **Local path:** `/Users/abidkarim/Desktop/my_own_website/`
- **Theme:** Dark terminal aesthetic, amber accent (`#f59e0b`), mono font throughout, retro code blocks
- **Blog posts:** Add `.md` files to `src/content/blog/`, images go in `public/blog/<slug>/`
- **No syntax highlighting:** Disabled in `astro.config.mjs` — code blocks are styled manually in CSS as amber terminal
- **Homepage edits:** Everything is in `src/pages/index.astro` — research array, links array, about text
- **Deploy command:** `git add . && git commit -m "message" && git push` — that's it
