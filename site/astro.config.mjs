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
      description: 'A content-addressed, UOR-native tensor runtime',
      social: [
        {
          icon: 'github',
          label: 'GitHub',
          href: 'https://github.com/Hologram-Technologies/hologram',
        },
      ],
      sidebar: [
        { label: 'Getting Started', link: '/guides/getting-started/' },
        {
          label: 'Demos',
          items: [
            { label: 'Calculator Demo', link: '/demo/calculator/' },
            { label: 'Compression Demo', link: '/demo/compression/' },
          ],
        },
        { label: 'Configuration', link: '/configuration/' },
        { label: 'Architecture', link: '/architecture/' },
        {
          label: 'Reference',
          items: [
            { label: 'LUT Materialization', link: '/reference/lut-ring/' },
            { label: 'Tensor Graph', link: '/reference/graph/' },
            { label: 'Executor', link: '/reference/executor/' },
            { label: 'Archive Format', link: '/reference/archive/' },
            { label: 'Compiler Pipeline', link: '/reference/compiler/' },
            { label: 'C ABI Bindings', link: '/reference/bindings/' },
          ],
        },
      ],
      editLink: {
        baseUrl: 'https://github.com/Hologram-Technologies/hologram/edit/main/site/',
      },
    }),
  ],
})
