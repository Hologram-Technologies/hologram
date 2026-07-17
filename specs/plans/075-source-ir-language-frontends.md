# Plan 075: Source IR and Multi-Language Frontends

**Status:** In Progress
**Created:** 2026-06-01
**Primary files:** `crates/hologram-compiler/src/source/`,
`crates/hologram-compiler/src/lib.rs`

## Problem

`hologram-compiler::source::parse` currently hand-parses a small
line-oriented language directly into `hologram_graph::Graph`. That made sense
as a bootstrap path, but it now has three structural problems:

1. There is no source-level IR. Every syntax rule must know how to allocate
   graph nodes, intern shapes, encode constants, and attach op metadata.
2. The grammar is tokenized with `split_whitespace`, so diagnostics, spans,
   attributes, richer tensor literals, and nested expressions are hard to add
   without making the parser brittle.
3. The parser is the only frontend. Supporting Python, TypeScript, Rust, or a
   richer Hologram DSL would either duplicate graph-building logic or add
   language-specific behavior directly to `Graph`.

The root cause is that parsing, semantic source validation, and graph lowering
are one step. They need to become separate stages.

## Goals

- Introduce a common source IR that all source languages lower into before
  graph construction.
- Replace the current hand-rolled Hologram text parser with a real parser.
- Preserve the existing `source::parse(&str) -> Result<Graph, CompileError>`
  API as the compatibility path for current tests, CLI, FFI, benches, and docs.
- Add explicit language-fronted entry points so Python, TypeScript, Rust, and
  future frontends lower through the same validation and graph-building code.
- Support embedded Hologram graph regions inside host-language files by
  extracting only explicit graph-builder regions and ignoring unrelated host
  code.
- Keep compile-time shape/dtype resolution in the compiler path. Source
  frontends must not push shape inference or algorithm selection into runtime.
- Preserve `hologram-compiler`'s default `no_std + alloc` posture.
- Preserve the runtime's zero-copy and O(1) execution story: source parsing and
  lowering may allocate while compiling, but must not introduce graph/runtime
  surfaces that require per-dispatch parsing, name lookup, dynamic shape
  inference, virtual dispatch, or data copying.

## Non-Goals

- Full general-purpose Python, TypeScript, or Rust execution.
- Dynamic control flow in source languages before graph-level control-flow
  semantics exist.
- Runtime tracing of arbitrary host-language code.
- Moving `Graph` into the source layer or making source syntax part of the
  `.holo` archive wire format.

## Architecture

Add a source frontend boundary inside `hologram-compiler`:

```text
source text
  -> language parser
  -> SourceDocument / SourceGraph
  -> selected SourceProgram
  -> source semantic validation
  -> GraphBuilder / Graph lowering
  -> Compiler
  -> .holo archive
```

The key rule is that only the Source IR lowerer knows how to build a `Graph`.
Language parsers produce IR; they do not allocate graph nodes directly.

### Proposed Module Layout

```text
crates/hologram-compiler/src/source/
├── mod.rs          # public compatibility API + language dispatch
├── document.rs     # SourceDocument, SourceGraph, SourceParseOptions
├── ir.rs           # SourceProgram, SourceStmt, SourceExpr, SourceType, spans
├── diagnostic.rs   # source spans + structured parse/semantic errors
├── lower.rs        # SourceProgram -> Graph
├── op_table.rs     # OpKind lookup from OpKind::name()
├── frontend.rs     # SourceFrontend trait
└── frontends/
    ├── mod.rs          # frontend exports
    ├── hologram.rs     # native Hologram DSL frontend (nom, no_std + alloc)
    ├── hologram/
    │   └── legacy.rs   # compatibility parser for the original line grammar
    ├── python.rs       # optional Python AST frontend (std feature)
    ├── typescript.rs   # optional TypeScript AST frontend (std feature)
    ├── rust.rs         # optional Rust AST frontend (std feature)
    └── ...             # future Go/C/PHP/etc. adapters
```

`source.rs` should become `source/mod.rs` during the refactor so each stage is
testable in isolation.

### Source IR Shape

The IR should model source semantics, not graph storage:

```rust
pub struct SourceProgram {
    pub items: Vec<SourceItem>,
}

pub enum SourceItem {
    Input(SourceInput),
    Const(SourceConst),
    Let(SourceBinding),
    Output(SourceOutput),
}

pub struct SourceBinding {
    pub name: Symbol,
    pub expr: SourceExpr,
    pub ty: Option<SourceType>,
    pub span: SourceSpan,
}

pub enum SourceExpr {
    Ref(Symbol),
    TensorLiteral(SourceTensorLiteral),
    OpCall(SourceOpCall),
}

pub struct SourceOpCall {
    pub op: hologram_graph::OpKind,
    pub inputs: Vec<Symbol>,
    pub attrs: SourceAttrs,
    pub span: SourceSpan,
}
```

The lowerer then owns:

- symbol resolution (`name -> InputSource`)
- dtype defaulting and validation
- shape interning
- constant byte encoding
- sparse op-attribute attachment (`ConvAttrs`, `GemmAttrs`, `ReduceAttrs`,
  `GatherAttrs`, `AttentionAttrs`, etc.)
- named graph input/output registration
- Graph construction order and duplicate-name checks

### Source Documents and Embedded Graph Regions

Frontends parse whole source files into `SourceDocument`, not directly into a
single `SourceProgram`. A document can contain zero, one, or many explicit graph
regions:

```rust
pub struct SourceDocument {
    graphs: Vec<SourceGraph>,
}

pub struct SourceGraph {
    pub name: Option<String>,
    pub program: SourceProgram,
    pub span: SourceSpan,
}

pub struct SourceParseOptions {
    graph: Option<String>,
}
```

Selection rules are deliberately small:

- one graph and no `--graph`: compile that graph
- multiple graphs and no `--graph`: reject as ambiguous
- `--graph <name>`: compile the unique graph with that name
- no matching or duplicate matching graph: reject before graph lowering

The native Hologram frontend currently produces one anonymous graph. Host
language frontends should produce named `SourceGraph`s for explicit Hologram
builder regions and ignore unrelated Python/TypeScript/Rust code. This is the
DX boundary for embedding Hologram-specific work inside existing source files
without compiling or executing the rest of the file.

### Graph Inference Policy

Inference is allowed only when the host-language AST contains unambiguous use
of Hologram's closed builder API. The compiler must not guess intent from
ordinary tensor-looking host code.

Initial policy:

- **Default extraction:** explicit graph regions and recognized builder API
  functions are accepted.
- **Safe inference:** a host-language function whose body calls the known
  builder parameter (`h.input`, `h.const`, `h.ops.<op>`, `h.output`) may become
  a named `SourceGraph` even without a decorator.
- **Ignored code:** imports, application functions, classes, and helper code
  outside recognized graph candidates are ignored by the compiler.
- **Strict candidate parsing:** once a function is recognized as a graph
  candidate, unsupported statements inside that function fail loudly instead of
  being interpreted or skipped.
- **Rejected inference:** ordinary Python/TypeScript/Rust tensor expressions
  such as `x @ w`, loops, reflection, imports, and arbitrary calls are not
  compiled unless a later ADR defines a pure tensor subset.

This keeps the inference story AST-only, non-executing, and compatible with the
same `SourceDocument -> SourceProgram -> Graph` path used by explicit regions.

### Parser Strategy

Do not make `nom` the parser for every supported language. Use a common
frontend adapter boundary:

```rust
pub trait SourceFrontend {
    const INFO: SourceFrontendInfo;

    fn parse_ir(&self, source: &str) -> Result<SourceProgram, CompileError>;
}
```

Each frontend owns its parser choice and returns the same `SourceProgram` IR.
The lowerer remains shared. Frontend metadata (`SourceFrontendInfo`) owns the
language name aliases and file extensions, so CLI/tooling resolution is driven
by the frontend registry rather than by separate hard-coded extension tables.

Parser choices:

| Frontend | Parser strategy | Feature posture |
|----------|-----------------|-----------------|
| Native Hologram DSL | `nom` parser over the project-owned grammar; PEG is acceptable only if it preserves the default no-std build | default, `no_std + alloc` |
| Python | `rustpython-parser` AST parser for a restricted builder subset | `std`, feature-gated (`frontend-python`) |
| TypeScript | SWC TypeScript parser for a restricted builder subset | `std`, feature-gated (`frontend-typescript`) |
| Rust | `syn` parser for a restricted builder subset | `std`, feature-gated (`frontend-rust`) |
| Go / C / PHP / future languages | dedicated adapter using an established parser for that language | feature-gated |

`nom` is still useful for Hologram-owned grammars and small adapter-specific
configuration grammars, but general-purpose languages should use parsers that
understand their lexical and syntactic edge cases. The critical invariant is
not parser uniformity; it is that every frontend lowers into `SourceProgram`
and never allocates graph nodes directly.

PEG-like grammars are a good fit for project-owned languages when grammar
readability and unambiguous syntax matter more than parser micro-control. They
should be considered for the native v2 DSL if the chosen crate satisfies:

- default `hologram-compiler --no-default-features` support
- borrowed/span-addressed token handling or bounded owned allocation
- clear diagnostics with line/column spans
- no build-time or proc-macro dependency that makes embedded/wasm builds
  brittle

Tree-sitter is useful, but for a different role. It is excellent for editor
tooling, incremental parsing, syntax highlighting, tolerant parsing, and
possibly std-only host-language adapters when a mature grammar exists. It
should not be the default compiler-core parser because the compiler does not
need incremental reparsing, and the runtime/embedded path should not inherit a
tree-sitter dependency. A practical split is:

- compiler core: no-std native parser + shared `SourceProgram` lowerer
- developer tooling: optional tree-sitter grammars for editor integration and
  richer diagnostics
- host languages: feature-gated adapters, using tree-sitter only when it is the
  best available parser for that language and remains outside the default build

## Zero-Copy and O(1) Performance Contract

The source frontend is allowed to allocate during compilation, but it must not
weaken the runtime architecture. The output of this plan is still a `.holo`
archive of resolved `KernelCall`s; execution must not know which source
language produced it.

Design constraints:

- **Borrow first while parsing.** Native parsing should keep identifiers,
  diagnostics, and token text as source spans or interned symbols until graph
  lowering needs owned graph names. Avoid cloning strings per token.
- **Single-pass lowering.** Lower `SourceProgram -> Graph` in declaration order
  with one symbol table and one append-only graph build. Complexity target:
  `O(items + edges + literal_bytes)`, no repeated graph scans.
- **No runtime source metadata dependency.** Spans, AST nodes, source-language
  tags, and symbol names are compiler diagnostics. They must not be required by
  `hologram-exec`, `hologram-compute`, or `KernelCall` dispatch. Optional
  provenance can be emitted as archive extensions, never as execution input.
- **Resolve all names before archive build.** Source identifiers lower to
  `InputSource` / `NodeId` / `ConstantId` during compilation. No string lookup,
  dynamic attribute map, or source-language dispatch may survive into the
  execution schedule.
- **Parse constants directly to bytes.** Tensor literals should be validated
  against dtype and shape while writing the final little-endian byte buffer.
  Avoid `Vec<f32> -> Vec<u8>` double materialization for large constants.
- **External constants for large weights.** Source IR should include a future
  `ConstRef` / `ExternalTensor` form for file, archive-section, or κ-addressed
  tensors so host-language frontends do not force large model weights through
  source text. Inline literals are for tests and small examples.
- **Canonicalize once.** Op names, aliases, dtypes, shapes, and attributes are
  normalized in source lowering. Backend lowering must see ordinary `Graph`
  state, not frontend-specific variants.
- **No algorithm selection in source frontends.** A frontend may request an op
  and attributes; choosing padded/unpadded, packed/unpacked, fused/unfused, or
  backend-specific variants remains compiler/load-time work that emits concrete
  `KernelCall` variants.
- **No virtual frontend hooks in execution.** Language support is a compiler
  concern. Do not add callback registries, trait-object kernels, dynamic op
  tables, or host-language evaluators to the runtime path.

Validation should include source-lowering microbenchmarks for large linear
graphs and large constants, plus negative checks that no `source::*` types are
referenced from `hologram-exec`, `hologram-compute`, or `hologram-archive`
execution structures.

## Native Hologram DSL

The existing line grammar remains valid in phase 1:

```text
input  x :2x3
const  w :3x4 = 1,2,3,4,5,6,7,8,9,10,11,12
op     matmul x w :2x4 as=y
output y
```

Then introduce a v2 grammar that is easier for humans and host-language
frontends to target:

```text
input x: f32[2, 3]
const w: f32[3, 4] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
let y: f32[2, 4] = matmul(x, w)
let z: f32[2] = reduce_sum(y, axes=[1], keepdims=true)
output y
```

Named call attributes are parsed into `SourceAttrs` and validated against the
op kind before lowering. Supported forms include scalar booleans/integers/floats
(`keepdims=true`, `axis=-1`, `alpha=0.5`) and fixed numeric lists
(`axes=[1, 2]`, `stride=[2, 2]`, `pads=[1, 1, 1, 1]`, `kernel=[3, 3]`).

Legacy syntax should parse through the same `SourceProgram` as v2 syntax, so
compatibility is tested at the IR boundary and at the compiled-archive boundary.

## Host-Language Frontend Contract

Python, TypeScript, and Rust frontends should parse explicit graph-building
subsets. They must not execute arbitrary source code. A host-language file may
contain ordinary application code; the frontend extracts only recognized
Hologram graph regions into `SourceDocument` and leaves the rest of the file
semantically invisible to the compiler.

### Python

Accepted shape is an AST pattern such as:

```python
def graph(h):
    x = h.input("x", dtype="f32", shape=[2, 3])
    w = h.const("w", dtype="f32", shape=[3, 4], values=[...])
    y = h.ops.matmul(x, w, shape=[2, 4])
    h.output("y", y)
```

The frontend parses calls and literals from the Python AST and lowers them into
`SourceProgram`. No imports, loops, reflection, or user code execution happen
in the compiler. The first implemented subset accepts top-level functions that
use a builder parameter and statements of the form:

- `x = h.input("x", dtype="f32", shape=[...])`
- `w = h.const("w", dtype="f32", shape=[...], values=[...])`
- `y = h.ops.<op>(x, w, shape=[...])`
- `h.output("y", y)` or `h.output(y)`

Other statements inside an inferred graph function currently fail loudly.
Unrelated host-language code outside inferred graph functions is ignored.

### TypeScript

Accepted shape should mirror Python:

```ts
export function graph(h: Hologram) {
  const x = h.input("x", { dtype: "f32", shape: [2, 3] })
  const w = h.const("w", { dtype: "f32", shape: [3, 4], values: [...] })
  const y = h.ops.matmul(x, w, { shape: [2, 4] })
  h.output("y", y)
}
```

The frontend parses the AST and accepts only the explicit Hologram builder
surface. Plain or exported functions using the builder parameter become named
graph regions. Object-literal options carry `shape`, `dtype`, constant
`values`, and op attributes. Unsupported statements inside an inferred graph
function fail loudly with source-position diagnostics.

### Rust

Accepted shape is a declarative builder subset parsed with `syn`, not compiled
and run:

```rust
fn graph(h: &mut HologramBuilder) {
    let x = h.input("x", dtype("f32"), shape([2, 3]));
    let w = h.constant("w", shape([3, 4]), values([...]));
    let y = h.ops().matmul(x, w, shape([2, 4]));
    h.output("y", y);
}
```

Helper-call options carry `shape`, `dtype`, constant `values`, and op attrs.
The Rust frontend rejects arbitrary expressions until there is a clear semantic
model for them.

## Public API

Preserve:

```rust
pub fn source::parse(source: &str) -> Result<Graph, CompileError>;
pub fn compile_from_source(
    source: &str,
    level: WittLevel,
    target: BackendKind,
) -> Result<CompilationOutput, CompileError>;
```

Add:

```rust
pub enum SourceLanguage {
    Hologram,
    Python,
    TypeScript,
    Rust,
}

pub fn source::parse_ir(
    source: &str,
    language: SourceLanguage,
) -> Result<SourceProgram, CompileError>;

pub fn source::parse_document(
    source: &str,
    language: SourceLanguage,
) -> Result<SourceDocument, CompileError>;

pub fn source::parse_ir_with_options(
    source: &str,
    language: SourceLanguage,
    options: &SourceParseOptions,
) -> Result<SourceProgram, CompileError>;

pub fn source::lower_ir(program: &SourceProgram) -> Result<Graph, CompileError>;

pub fn compile_from_source_language(
    source: &str,
    language: SourceLanguage,
    level: WittLevel,
    target: BackendKind,
) -> Result<CompilationOutput, CompileError>;
```

The CLI should grow `--source-language` with extension-based auto-detection and
`--graph <name>` for multi-graph source documents, while keeping the current
default as native Hologram source.

## Phases

### Phase 1: Source IR and Compatibility Lowerer

- [x] Define `source::ir` types with spans, dtype, shape, attributes, tensor
  literals, and op calls.
- [x] Implement `source::lower::lower_ir(&SourceProgram) -> Graph`.
- [x] Move current name resolution, shape interning, constant insertion, and
  output creation out of the parser and into the lowerer.
- [x] Add a symbol table / interner so identifiers are resolved once and stored
  as compact source symbols until graph lowering.
- [x] Parse inline tensor literals directly into final constant bytes and
  validate shape/value count before insertion into `ConstantStore`.
- [x] Attach `SourceAttrs` to the existing graph sparse attribute tables during
  source lowering.
- [x] Keep `source::parse(&str)` compiling legacy source by routing
  `legacy parser -> SourceProgram -> Graph`.
- [x] Add duplicate-name, unresolved-symbol, invalid-output-source, constant
  shape/value-count, and sparse-attribute tests at the IR/lowering boundary.
- [ ] Add op-arity tests once the source IR arity contract is pinned for ops
  whose operands can be represented as attributes or constants.

**Acceptance:** Existing source tests pass unchanged, new tests prove the
lowerer can build a `Graph` without invoking the text parser, and source
lowering is one pass over the IR plus literal bytes.

### Phase 2: Native DSL Parser

- [x] Replace `split_whitespace` parsing with a `nom` parser for the legacy
  grammar.
- [x] Add a v2 native grammar subset with typed tensors, bracket shapes,
  tensor literals, `let`, and call expressions.
- [x] Add named attributes to the v2 native grammar.
- [x] Parse op attributes into `SourceAttrs`, then attach them through the
  existing sparse graph attribute tables during lowering.
- [x] Replace the hand-maintained `parse_op_kind` list with a central lookup
  from `OpKind::ALL`; generate `OpKind`, `OpKind::ALL`, and `OpKind::name()`
  from one catalog declaration.
- [x] Add parse-span diagnostics that report line/column and the rejected token.
- [ ] Keep native parser token data borrowed or span-addressed until lowering;
  no per-token owned string churn in the parser.

**Acceptance:** Legacy and v2 native DSL programs that describe the same graph
produce the same graph fingerprint / compiled archive for representative unary,
binary, matmul, layout, reduction, convolution, attention, and quantization
cases.

### Phase 3: API and Tooling Integration

- [x] Add `SourceLanguage`, `parse_ir`, `lower_ir`, and
  `compile_from_source_language`.
- [x] Add a `SourceFrontend` adapter boundary so each language owns its parser
  choice while lowering into the same `SourceProgram`.
- [x] Add `SourceDocument`, `SourceGraph`, and `SourceParseOptions` so
  frontends can extract graph regions from larger host-language source files
  before selecting a single `SourceProgram` for lowering.
- [x] Move language aliases and filename extensions into `SourceFrontendInfo`
  metadata implemented by each frontend; CLI detection delegates to the source
  frontend registry.
- [x] Update `hologram-cli compile` with `--source-language` and extension
  detection (`.txt`, `.py`, `.ts`, `.tsx`, `.rs`).
- [x] Add `hologram-cli compile --graph <name>` so embedded/multi-graph source
  documents can select the graph region to compile.
- [ ] Update `hologram-ffi` only after the C ABI design for language selection
  is explicit; default FFI behavior remains native Hologram source.
- [x] Update docs in `README.md`, `specs/docs/architecture.md`, and the site
  compiler/getting-started/configuration pages so the source path is documented
  as `source -> SourceDocument -> SourceProgram -> Graph`.
- [x] Add negative dependency checks or tests proving `source::*` types do not
  leak into runtime/backend/archive execution structs.

**Acceptance:** Current callers of `compile_from_source` keep working, while
CLI smoke tests can compile native v2 source through the new language-aware
entry point.

### Phase 4: Python Frontend

- [x] Add a Python frontend behind an explicit `frontend-python` feature.
- [x] Parse a restricted Hologram builder subset from Python AST.
- [x] Infer graph candidates from unambiguous builder usage while ignoring
  unrelated host-language code.
- [x] Reject unsupported Python constructs inside graph candidates.
- [x] Document Python compile flags, accepted builder calls, graph selection,
  and the AST-only/no-execution boundary.
- [x] Add source-position diagnostics for rejected Python AST nodes.
- [x] Add Python op-attribute parsing beyond `shape`/`dtype`.
- [x] Add equivalence tests comparing Python source against native DSL source
  for the same graph.

**Acceptance:** Python source can express inputs, constants, op calls with
attributes, and outputs without executing Python code.

### Phase 5: TypeScript Frontend

- [x] Add a TypeScript frontend behind an explicit `frontend-typescript`
  feature.
- [x] Parse a restricted Hologram builder subset from TypeScript AST.
- [x] Reject unsupported TS constructs with span diagnostics.
- [x] Add equivalence tests comparing TypeScript source against native DSL
  source for the same graph.

**Acceptance:** TypeScript source can express the same initial graph subset as
Python through the same `SourceProgram` structure.

### Phase 6: Rust Frontend

- [x] Add a Rust frontend behind an explicit `frontend-rust` feature.
- [x] Parse a restricted builder subset with `syn`.
- [x] Reject unsupported Rust constructs with span diagnostics.
- [x] Add equivalence tests comparing Rust source against native DSL source for
  the same graph.

**Acceptance:** Rust source can express the same initial graph subset as Python
and TypeScript without compiling or executing user Rust.

### Phase 7: Hardening and Documentation

- [x] Add a source frontend conformance suite: one graph expressed in every
  supported frontend must lower to the same IR and compile to the same archive.
- [x] Add source-lowering microbenchmarks for large graphs and large constants,
  tracking parse allocations, lowering time, and archive equivalence.
- [x] Add an external tensor / constant-reference design note before allowing
  large host-language examples to inline weights in source text.
- [ ] Add no-std build coverage for `hologram-compiler --no-default-features`
  with the native parser enabled.
- [ ] Add feature-gated std build coverage for each host-language frontend.
- [x] Document unsupported constructs and the exact accepted builder subset for
  each host language.
- [ ] Update `AGENTS.md` if the source frontend architecture becomes a standing
  convention.

**Acceptance:** `cargo test --workspace`, `cargo clippy --workspace -- -D
warnings`, and `cargo fmt --all --check` pass with the default feature set; each
frontend feature has a targeted compile/test job.

## Risks and Decisions

- **Parser dependencies:** Keep the native parser no-std. Host-language parsers
  can be std-only and feature-gated. Do not force `nom` onto Python,
  TypeScript, Rust, Go, C, PHP, or other general-purpose languages; each
  frontend should use a parser that understands that language's real syntax.
- **Host-language ambiguity:** Do not infer arbitrary host-language semantics.
  The first supported shape is an explicit Hologram builder subset.
- **Embedding boundary:** Host-language frontends parse complete files but only
  recognized Hologram graph regions become `SourceGraph`s. Ordinary host code
  is ignored rather than interpreted.
- **Graph drift:** Source op lookup now uses the closed `OpKind::ALL` catalog.
  `OpKind`, `OpKind::ALL`, and `OpKind::name()` are generated from one catalog
  declaration, so parser and dispatch coverage consume the same list.
- **Attributes:** The source IR must represent op attributes from day one;
  otherwise language frontends will immediately diverge on conv, reduction,
  gather, attention, quantization, and layout ops.
- **Diagnostics:** `CompileError::SourceParse(&'static str)` is too small for
  rich parser errors. Add structured source diagnostics while preserving a
  simple `CompileError` display path for existing callers.
- **Zero-copy boundaries:** Inline source constants necessarily become owned
  graph/archive bytes. For large weights, source languages should reference an
  external tensor artifact so bytes are loaded, packed, and archived once rather
  than copied through source literals. See
  [External Tensor References](../docs/external-tensor-references.md).
- **O(1) runtime preservation:** The Source IR must fully disappear before
  backend lowering. If a proposed frontend feature requires runtime name
  lookup, runtime shape inference, host-language execution, or dynamic op
  dispatch, it belongs outside this plan or requires a separate ADR.

## Completion Criteria

- [x] Existing line-oriented source remains accepted.
- [x] Native v2 source lowers through `SourceProgram`.
- [ ] Python, TypeScript, and Rust frontends lower into the same `SourceProgram`
  without executing source code.
- [ ] Equivalent programs across languages produce equivalent graphs and
  archives.
- [x] Source frontend implementation keeps runtime dispatch source-agnostic:
  no source-language tags, parser spans, dynamic attribute maps, or frontend
  callbacks are required by execution.
- [x] Large-constant handling has a zero-copy-oriented path or an explicit
  design note for external tensor references.
- [ ] Documentation and sprint tracking describe the new source pipeline.
