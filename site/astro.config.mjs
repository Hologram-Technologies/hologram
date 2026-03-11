import { defineConfig } from 'astro/config'
import starlight from '@astrojs/starlight'

export default defineConfig({
  site: 'https://uor-foundation.github.io',
  base: '/hologram',
  server: {
    allowedHosts: ["bore.pub", "localhost"]
  },
  integrations: [
    starlight({
      title: 'Hologram',
      description: 'O(1) neural network inference via precomputed lookup tables and KV-dispatch',
      social: [
        {
          icon: 'github',
          label: 'GitHub',
          href: 'https://github.com/UOR-Foundation/hologram',
        },
      ],
      sidebar: [
        { label: 'Getting Started', link: '/guides/getting-started/' },
        {
          label: 'Guides',
          items: [
            { label: 'Custom Ops', link: '/guides/custom-ops/' },
            { label: 'Calculator Demo', link: '/demo/calculator/' },
            { label: 'Compression Demo', link: '/demo/compression/' },
          ],
        },
        { label: 'Configuration', link: '/configuration/' },
        { label: 'Architecture', link: '/architecture/' },
        {
          label: 'Reference',
          items: [
            { label: 'LUT Tables & Ring Algebra', link: '/reference/lut-ring/' },
            { label: 'Expression Graph', link: '/reference/graph/' },
            { label: 'Executor & KV-Dispatch', link: '/reference/executor/' },
            { label: 'Archive Format', link: '/reference/archive/' },
            { label: 'Compiler Pipeline', link: '/reference/compiler/' },
            { label: 'Async & Streaming', link: '/reference/async/' },
            { label: 'C & WASM Bindings', link: '/reference/bindings/' },
          ],
        },
      ],
      editLink: {
        baseUrl: 'https://github.com/UOR-Foundation/hologram/edit/main/site/',
      },
    }),
  ],
})
