We want to create a faster execution environment using this workspace (create a workspace project here) that utilizes the computation platform of https://crates.io/crates/uor-foundation. This is a v2 version of our original work which is available at `../categorical-x` describes a lot of layers, including a compiler which exposes an `OperationGraph` for serialization, optimizations, archiving, etc. etc. This project is intended on being the next version that strips away any hacks or intermediate representations. However, we want to use the structure of uor-foundation to turn execution into KV-lookups.

Our initial target is to be able to create a super fast calculator that runs at lightning speed with benchmarks to demonstrate this capability. We want to use LUT fusion tables as well as fast activations across all operations.

We'll iterate through this project, so keep the scope focused and tight.

We'll need to add an AGENTS.md that describes utilizing the best-practices in rust development. We want to ensure we prefer using traits for complex types, structs with the builder pattern for functions that require lots of inputs. We'll also want to make sure we have clean and clear boundaries about what subcrates are required. We also want to ensure we write comprehensive tests for e2e and unit tests for all functions.

We'll also want to include benchmarks.

Can you identify gaps in this restructuring plan using the knowledge of the work already in the original work and help us create a step-by-step refactor of the project and keep the scope tight and focused at each step?

---

We should have O(1) execution time, zero-copy, memory-mapped lookups, using rkyv serialziation, and parallelization (with rayon, SIMD, and tokio).

---

We published a new version of `uor-foundation`. Please refresh this plan with the latest from github's `main` branch. 

We don't want to embed `holo-calc` here, that is an example project. Can we create this as an example, NOT embedded in this library.

Let's explore this without using SIMD, rayon, and tokio parallelization. The math itself should be as fast as possible. However, if you disagree, let us know.

In the Graph, layers need to be optimized for when and where they execute. We have the concept of subgraphs in v1 as well as parallelized subgraph execution. In part we do this so that we can utilize networked nodes (another step) execution and storage. We want to make sure we have this concept in this refactor.

Also those Graph nodes and edges need to be able to be automatically optimized to run when all their dependencies have been satisfied. That is we should be able to create an `OperationGraph` that is highly optimized for fast runtime execution.

The only serialization we want to support is rkyv utilizing memory-mapped data.

For the future iterations, I want  you to document those in a SPRINT.md document. For every active sprint we want to keep those in the `specs/SPRINT.md` that contains the sprint and checkboxes to keep track of what is complete and what is not. When we are complete with a sprint, it needs to be archived in `specs/sprints/<number>-<title>.md`. Add this to @AGENTS.md.

---

We want to make sure this runs on CPUs and WASM targets to start.

The `uor-foundation` crate overcomes the 8-bit scaling notation. Please confirm this.

We'll also need to be able to support running AI-models which can be VERY large weights. We'll want to be able to use the O(1) KV LUTs to target that data. In addition, we'll need the archive format for the `.holo` files because we can store subgraphs (or their IDs for looking them up across networks) as well. That's a hard requirement.

I want the organization to contain subdirectories with common functionality so we don't have too many dangling files in the root of each crate.

We'll also want to make sure this bundle can be TINY and have it able to run on constrained devices, like ESP32 and RaspberryPis as well as full servers/ci and development machines.

---

For those future tasks, can you please plan those out please so we don't drop those tasks?

I want to prefer macros for repeated implementations, trait-based operations, where necessary. I also don't want functions to take more than 3 arguments. If they do, I want you to make them accept structs that implement the builder pattern.

We don't need to support backwards compatibility at all. I don't want to keep V1 and V2 formats for `holo` archives.

We will have to distribute and schedule execution as well as storage with the network crates, which is a key for the subgraphs and execution.

We'll want to expose all available public functions at the root crate so consumers of this library just have to use it as the dependency. We don't want users of this crate to have to import subcrates.

Does the `holo-archive` implement headers with execution entrypoints?

---

For `hologram-ai` (a new greenfield version, which is not yet implemented), we'll want to be able to support complex ONNX operations and utilize a way for those model operations to be created by a consumer so we can support all ONNX operations. The same goes for gguf and ggml models as well.

I need you to save this plan in `specs/plans`, create a new @specs/SPRINT.md with the current tasks and keep it up to date as we work.

---

Are you reimplementing the `hologram-network` crate? We'll have to bring over the orchestration crate and resolver if we do that, for now...