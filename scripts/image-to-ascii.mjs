#!/usr/bin/env node
/**
 * Convert an image file to dense ASCII art.
 *
 * Usage:
 *   node scripts/image-to-ascii.mjs <input-image> [width] [charset]
 *
 * Examples:
 *   node scripts/image-to-ascii.mjs wave.jpg
 *   node scripts/image-to-ascii.mjs portrait.png 120
 *   node scripts/image-to-ascii.mjs robot.png 100 dense
 *
 * Output is printed to stdout — pipe it to a file:
 *   node scripts/image-to-ascii.mjs wave.jpg 120 > art.txt
 */

import sharp from 'sharp';
import { readFileSync } from 'node:fs';

const CHARSETS = {
  // densest to lightest — classic image-to-ASCII mapping
  dense:    '@%#*+=-:. ',
  detailed: '$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,"^`\'. ',
  blocks:   '█▓▒░ ',
  soft:     '#*+-. ',
};

const args = process.argv.slice(2);
if (args.length < 1) {
  console.error('usage: node scripts/image-to-ascii.mjs <image> [width=110] [charset=dense]');
  process.exit(1);
}

const input = args[0];
const width = parseInt(args[1] ?? '110', 10);
const charsetName = args[2] ?? 'dense';
const charset = CHARSETS[charsetName] ?? charsetName; // allow raw charset string too

// Monospace characters are ~2x taller than wide, so compress height
const aspectCompensation = 0.5;

const img = sharp(readFileSync(input)).greyscale();
const meta = await img.metadata();
const srcW = meta.width ?? width;
const srcH = meta.height ?? width;
const height = Math.round((srcH / srcW) * width * aspectCompensation);

const { data } = await img
  .resize(width, height, { fit: 'fill' })
  .raw()
  .toBuffer({ resolveWithObject: true });

const n = charset.length - 1;
let out = '';
for (let y = 0; y < height; y++) {
  let row = '';
  for (let x = 0; x < width; x++) {
    const brightness = data[y * width + x] / 255; // 0..1 (0=dark, 1=light)
    // Map bright pixels to lighter chars (end of charset is lighter)
    const idx = Math.round(brightness * n);
    row += charset[idx];
  }
  out += row + '\n';
}

process.stdout.write(out);
