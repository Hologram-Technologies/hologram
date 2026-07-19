# hologram-tck

> The reusable KappaStore conformance battery every Hologram storage backend must pass.

The Technology Compatibility Kit: one shared suite of trait-level ST/SPINE invariants
that hold for *any* `KappaStore` substrate. Every backend (mem, native, OPFS, bare-metal)
runs the *same* battery, so conformance is defined once and validated against each
substrate identically — the TR substrate-tripling discipline applied at the trait level.

The kit asserts only properties that follow from the `KappaStore` contract itself.
Backend-specific behaviour (zero-copy `Arc` identity for the reference store, reachability
`gc`) is covered here where it is a documented trait guarantee, and by each backend directly
where it is not. `no_std` by default (`std` feature on by default).

## What it provides

- `store_battery(store)` — runs the full trait-level battery; panics on the first violation (call from a `#[test]`; the store must start empty).
- `idempotency` — identical `(axis, bytes)` yields the same κ-label with no duplicate write (ST-2).
- `eviction_tolerant_get` — an unstored κ is `Ok(None)`, not an error; there is no delete primitive (SPINE-5).
- `unknown_axis_fails_loud` — an unwired σ-axis (e.g. `md5`) errors instead of silently coercing (SPINE-6).
- `axis_polymorphic_round_trip` — `put_axis`/`get_axis` round-trips across the five uor-addr axes (AS).
- `zero_copy_get_returns_arc_handle` — consecutive `get(κ)` share one `Arc<[u8]>`, no per-call copy (SP floor).
- `pin_unpin` — pin establishes a root, unpin removes it, and unpin of a non-pinned κ errors.
- `content_roundtrip` — bytes put are bytes got, and the κ re-derives to itself.
- `MemKappaStore` — re-export of the reference in-memory store (its home is `hologram-space`), the differential oracle every real backend is compared against.

Part of the [hologram](../../README.md) workspace.
