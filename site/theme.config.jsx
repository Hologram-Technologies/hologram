export default {
  logo: <strong>Hologram</strong>,
  project: {
    link: 'https://github.com/UOR-Foundation/hologram',
  },
  docsRepositoryBase: 'https://github.com/UOR-Foundation/hologram/tree/main/site',
  useNextSeoProps() {
    return { titleTemplate: '%s – Hologram' }
  },
  head: (
    <>
      <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      <meta name="description" content="O(1) neural network inference via precomputed lookup tables and KV-dispatch" />
    </>
  ),
  footer: {
    text: 'Hologram — UOR Foundation',
  },
  sidebar: {
    defaultMenuCollapseLevel: 1,
  },
}
