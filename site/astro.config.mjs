import { defineConfig } from 'astro/config'
import starlight from '@astrojs/starlight'

export default defineConfig({
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
        { label: 'Overview', link: '/' },
        { label: 'Architecture', link: '/architecture/' },
        {
          label: 'Crates',
          items: [
            { label: 'hologram-core', link: '/crates/core/' },
            { label: 'hologram-graph', link: '/crates/graph/' },
            { label: 'hologram-archive', link: '/crates/archive/' },
            { label: 'hologram-exec', link: '/crates/exec/' },
            { label: 'hologram-compiler', link: '/crates/compiler/' },
            { label: 'hologram-async', link: '/crates/async/' },
            { label: 'hologram-ffi', link: '/crates/ffi/' },
          ],
        },
        {
          label: 'Guides',
          items: [
            { label: 'Getting Started', link: '/guides/getting-started/' },
            { label: 'Custom Ops', link: '/guides/custom-ops/' },
          ],
        },
      ],
      editLink: {
        baseUrl: 'https://github.com/UOR-Foundation/hologram/edit/main/site/',
      },
    }),
  ],
})
