import { defineCollection, z } from 'astro:content';

const blog = defineCollection({
  type: 'content',
  schema: z.object({
    title: z.string(),
    description: z.string().optional(),
    pubDate: z.coerce.date(),
    originalUrl: z.string().url().optional(),
    tags: z.array(z.string()).default([]),
    readTime: z.string().optional(),
  }),
});

export const collections = { blog };
