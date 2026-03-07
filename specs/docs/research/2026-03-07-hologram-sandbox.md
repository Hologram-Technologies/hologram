# Hologram Sandbox Architecture Summary

Date: 2026-03-07  
Status: Research  
Related Repos:
- ../hologram
- ../hologram-ai
- ../hologram-sandbox (future)

---

# Overview

`hologram-sandbox` is a proposed repository that provides **isolated execution environments for Hologram workloads**.

The key architectural principle is:

Hologram defines the **execution model**, while `hologram-sandbox` provides **execution environments**.

This repository will consume Hologram execution plans and execute them inside sandboxed environments.

---

# Repository Roles

## hologram

Defines the execution environment and compute model.

Responsibilities include:

- graph representation
- node semantics
- execution plans
- memory and data contracts
- artifact references
- capability requirements
- backend-neutral execution interfaces

Hologram must remain:

- sandbox agnostic
- virtualization agnostic
- platform neutral

---

## hologram-sandbox

Provides isolated execution environments capable of running Hologram workloads.

Responsibilities include:

- process sandbox execution
- WASM runtime execution
- microVM execution
- sandbox lifecycle management
- sandbox pooling
- snapshot restore
- guest/host communication

---

# Architectural Principle

Dependency direction must always be:

```
hologram
  ↓ defines execution contracts
hologram-sandbox
  ↓ implements isolated execution environments
platform backends
```

`hologram` must never depend on `hologram-sandbox`.

---

# Sandbox Types

The sandbox runtime should support multiple isolation models.

## Process Sandbox

Host OS process execution.

Advantages:

- lowest overhead
- easiest developer experience
- fast startup

---

## WASM Sandbox

Execution inside a WASM runtime.

Advantages:

- strong isolation
- portable execution
- small runtime footprint

---

## MicroVM Sandbox

Execution inside lightweight virtual machines.

Possible backends:

- KVM
- HVF
- WHPX

Advantages:

- strong isolation
- multi-tenant safety
- kernel-level isolation

---

# Sandbox Lifecycle

All sandbox types should follow a consistent lifecycle model.

```
create
prepare
execute
collect outputs
reuse or teardown
```

This lifecycle must be consistent across sandbox types.

---

# Pooling and Snapshotting

Sandbox startup latency should be minimized.

Preferred strategies:

- sandbox pooling
- snapshot restore
- pre-initialized environments
- warm execution paths

Cold boot should be avoided where possible.

---

# Data and Memory Model

Sandbox execution must respect Hologram memory contracts.

Possible transport modes:

1. Shared memory regions
2. Structured shared buffers
3. Serialized fallback transport

Zero-copy should be used where isolation allows it.

---

# Guest Communication

Guest environments should communicate with the host using structured protocols.

Possible mechanisms:

- vsock
- virtio-serial
- shared memory
- job-manifest disks

Avoid SSH-based orchestration.

Guest agents should remain minimal.

---

# Platform Strategy

Initial platform support should include:

Linux
- process sandbox
- WASM sandbox
- KVM microVM

macOS
- process sandbox
- WASM sandbox
- optional HVF development backend

Windows
- process sandbox
- WASM sandbox
- optional WHPX development backend

Edge Linux
- process sandbox
- optional WASM

MCU-class targets are out of scope.

---

# Open Questions

- Should WASM be treated as a first-class backend or plugin system?
- How should capability negotiation work between Hologram and sandbox runtimes?
- How should artifact staging work for sandbox environments?
- How much responsibility should sandbox runtimes have for scheduling vs the Hologram runtime?