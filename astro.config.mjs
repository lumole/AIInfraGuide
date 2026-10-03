import { defineConfig } from 'astro/config';
import tailwind from '@astrojs/tailwind';
import sitemap from '@astrojs/sitemap';
import rehypeSlug from 'rehype-slug';
import rehypeAutolinkHeadings from 'rehype-autolink-headings';
import rehypeExternalLinks from 'rehype-external-links';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

const isVercel = process.env.VERCEL === '1';
const vercelSite = process.env.VERCEL_PROJECT_PRODUCTION_URL
  ? `https://${process.env.VERCEL_PROJECT_PRODUCTION_URL}`
  : 'https://all-infra-guide.vercel.app';

export default defineConfig({
  site: isVercel ? vercelSite : 'https://caomaolufei.github.io',
  base: isVercel ? '/' : '/AIInfraGuide',
  integrations: [tailwind(), sitemap()],
  experimental: {
    contentLayer: true,
  },
  markdown: {
    shikiConfig: {
      themes: {
        light: 'github-light',
        dark: 'github-dark',
      },
      defaultColor: 'light',
      wrap: false,
    },
    remarkPlugins: [remarkMath],
    rehypePlugins: [
      rehypeSlug,
      [rehypeAutolinkHeadings, { behavior: 'wrap' }],
      [rehypeExternalLinks, { target: '_blank', rel: ['nofollow', 'noopener'] }],
      rehypeKatex,
    ],
  },
});
