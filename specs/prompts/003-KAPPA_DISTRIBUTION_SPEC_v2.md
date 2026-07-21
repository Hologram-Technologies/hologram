# κ-Distribution Protocol Specification
## Universal Content-Addressed Blob Store
Status: Draft Specification
Version: 2.0.0
Date: 2026-07-20
---
## 1. Abstract
κ-Distribution is a protocol for storing, retrieving, addressing,
composing, replicating, and garbage collecting content-addressed
blobs across any number of nodes, any storage substrate, and any
network topology including intermittently connected and long-
duration-disconnected deployments.
Every object in the store is a blob identified by a κ-label — a
self-describing, algorithm-prefixed content hash. There are no
special object types. Manifests, edges, witnesses, schemas, pins,
admission filters, and application content are all blobs, all
addressed by κ-labels, all stored and retrieved through the same
operations.
Tags are mutable pointers from human-readable names to immutable
κ-labels. Tags are the only mutable state in the protocol.
The addressing convention encodes organizational hierarchy to the
left of a colon separator, resource identity to the right, and
version after an @ delimiter. Organizational depth grows leftward
via DNS-style subdomains. Resource specificity grows rightward via
path segments.
The protocol defines operations, consistency guarantees, garbage
collection semantics, replication behavior, admission control,
failure modes, and a conformance test framework. It does not
prescribe a storage backend, a consensus protocol, a replication
transport, an encryption mechanism, or an authorization framework.
Any system that implements the specified operations and upholds
the specified invariants is a conforming κ-Distribution registry.
---
## 2. Terminology and Conventions
### 2.1 Key Words
The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL
NOT", "SHOULD", "SHOULD NOT", "RECOMMENDED", "MAY", and
"OPTIONAL" in this document are to be interpreted as described in
RFC 2119.
### 2.2 Glossary
    κ-label         Content address. An ASCII string of the form
                    <algorithm>:<lowercase-hex-digest>. The algorithm
                    names the hash function. The digest is the
                    hexadecimal encoding of the hash output.
                    Lengths: 71 bytes (sha256, blake3), 73 bytes
                    (sha3-256), 74 bytes (keccak256), 135 bytes
                    (sha512). A κ-label is immutable, deterministic,
                    and self-describing.

    σ-axis          The hash algorithm component of a κ-label.
                    One of: sha256, blake3, sha3-256, keccak256,
                    sha512.

    blob            Arbitrary bytes stored in the system. Every
                    blob has exactly one κ-label per σ-axis. The
                    same bytes produce the same κ-label on any
                    node at any time.

    tag             A mutable binding from (path, name) to a
                    κ-label. Tags are the only mutable state.

    edge            A typed directed relationship between two
                    κ-labels. An edge is itself a blob with its
                    own κ-label.

    manifest        A blob whose content references other κ-labels.
                    Not a distinct protocol type.

    witness         A blob containing the derivation trace of a
                    κ-label computation.

    schema          A blob containing validation rules for content
                    written to a scoped path.

    pin             A blob marking a κ-label as protected from
                    garbage collection.

    finalizer       A pin carrying a controller name. Blocks GC
                    until explicitly released.

    consensus       An ordering authority that linearizes tag
    domain          updates within a scope. The protocol requires
                    consensus domains exist but does not prescribe
                    their implementation or scoping.

    sweep           A garbage collection pass that walks
                    reachability from roots and evicts unreachable
                    blobs.

    registry        Any endpoint implementing this protocol.

    federation      Cross-registry content resolution via κ-label
                    query, including multi-hop relay.

    verify-on-      Client-side re-hash of received content to
    receipt         confirm κ-label integrity, independent of
                    server trust.

    upload          A stateful session for chunked blob upload.
    session         The session URL is opaque to the client.
### 2.3 Notation
Path segments in braces are variable:
    {org-hierarchy}:{resource-path}@{version}
The vertical bar separates alternatives:
    accept | reject
HTTP header names follow RFC 9110 conventions.
---
## 3. Object Model
### 3.1 Blobs
A blob is the atomic unit of storage. A blob consists of:
    content     arbitrary bytes, 0 to 2^63 - 1 bytes
    κ-label     the content address, computed from the content
A blob is immutable. Once stored, its content never changes. The
κ-label is deterministic: the same content always produces the
same κ-label under the same σ-axis.
There is exactly one kind of object in κ-Distribution: a blob.
All higher-level constructs — manifests, edges, witnesses, schemas,
pins, admission filters, tag snapshots — are blobs. They differ
in what their content means, not in how they are stored, addressed,
or retrieved.
**Exact-bytes invariant.** The registry MUST store blobs as the
exact byte representation provided by the client. No
normalization, re-encoding, field stripping, or whitespace
modification is permitted. Any modification would invalidate the
κ-label. This invariant applies to all blobs regardless of
content type.
A blob MAY have additional κ-labels under different σ-axes. The
same bytes addressed as `sha256:abc...` and as `blake3:def...`
are the same content with two addresses. Cross-axis κ-labels MAY
be computed eagerly at ingestion or lazily on demand when a query
arrives for an axis not computed at ingestion.
### 3.2 κ-Labels
A κ-label is an ASCII string:
    <algorithm> ":" <lowercase-hex-digest>
The algorithm is one of the recognized σ-axis tokens:
    Token       Digest bytes   κ-label bytes   Standard
    -----       ------------   -------------   --------
    sha256      32             71              FIPS 180-4
    blake3      32             71              BLAKE3 spec
    sha3-256    32             73              FIPS 202
    keccak256   32             74              Keccak
    sha512      64             135             FIPS 180-4
A κ-label MUST satisfy:
- Total length is between 71 and 135 bytes inclusive
- Every byte is in the ASCII range 0x21-0x7E
- Exactly one colon separator exists
- The prefix before the colon is a recognized σ-axis token
- The suffix after the colon consists entirely of lowercase
  hexadecimal digits (0-9, a-f)
- The hex digit count equals twice the digest byte count for
  the token
Construction is fallible. Implementations MUST reject κ-labels
that violate any constraint.
The σ-axis token set is extensible. Future versions of this
specification MAY add tokens. Implementations MUST reject
unrecognized tokens and SHOULD NOT attempt to process them.
### 3.3 Tags
A tag is a mutable binding:
    (path, name) → κ-label
The path is a κ-address (§4) without the version component. The
name is a UTF-8 string of 1 to 256 bytes. Tag names MUST NOT
contain the characters `/` (path separator) or `@` (version
delimiter).
Tags are the only mutable state in the protocol. Creating,
updating, or deleting a tag is a mutation that requires ordering
within a consensus domain (§7).
Tag names MAY be ISO 8601 timestamps with nanosecond precision:
    2026-07-20T14:30:00.123456789
When tags are timestamps, the tag list for a path is a timeline.
Temporal queries are range-filtered tag list operations:
    "Value at time T"       → resolve tag named T
    "Values between T1-T2"  → list tags in range [T1, T2]
    "Most recent value"     → list tags, descending, limit 1
A tag snapshot is the set of all tag bindings for a path,
serialized as a blob with its own κ-label. The tag snapshot
captures mutable state as immutable content at a point in time.
Implementations MAY produce tag snapshots at configurable
intervals.
### 3.4 Edges
An edge expresses a typed directed relationship between two
κ-labels. An edge is a blob. Its content is the canonical
serialization of:
    source      κ-label of the origin blob
    relation    UTF-8 string naming the relationship type
    target      κ-label of the destination blob
    metadata    CBOR-encoded bytes (RFC 8949 §4.2 deterministic
                encoding with NFC normalization of string values),
                may be empty
The canonical serialization is defined in Appendix C. It is
deterministic: the same (source, relation, target, metadata)
always produces the same bytes, which always produces the same
κ-label for the edge blob.
**σ-axis for edge κ-labels.** The edge blob's κ-label MUST use
the same σ-axis as the source κ-label. If the source is
`sha256:...`, the edge is hashed with SHA-256. If the source and
target use different σ-axes, the edge κ-label follows the source.
This is consistent with the σ-axis homogeneity requirement in
composition operations (§5.4).
An edge blob is stored like any other blob. It is retrievable by
its own κ-label. The registry MUST additionally maintain an index
that allows querying edges by source, by target, by relation, or
by any combination.
Defined relation types (conventions, not exhaustive):
    owns          owner → owned
                  GC: target is reachable if source is reachable
    derives-from  child → parent
                  content lineage, schema evolution
    composed-of   composed → operand
                  composition provenance
    witness-of    witness → witnessed blob
    schema-for    schema → path scope (encoded in metadata)
    pins          pin blob → protected blob
    refers-to     referrer → subject (generalized referrers)
    filter-for    filter blob → path scope (encoded in metadata)
Any UTF-8 string is a valid relation type.
### 3.5 Manifests
A manifest is a blob whose content references other κ-labels.
The protocol does not define a manifest format. JSON, CBOR,
protobuf, custom binary, or any serialization is valid. The
protocol treats a manifest identically to any other blob.
Container images, Helm charts, LLM model weight packages,
genomic dataset bundles, state snapshots, firmware archives, and
WASM module bundles are all manifests: blobs that describe sets
of other blobs.
A manifest MAY be linked to its referenced blobs by edges
(relation: "refers-to") for graph traversal. This linkage is
OPTIONAL — applications that parse the manifest format discover
references directly. Applications that want format-agnostic
traversal SHOULD create edges.
### 3.6 Witnesses
A witness is a blob containing the serialized derivation trace
of a κ-label computation. The witness records the steps the
canonicalization pipeline took to produce the κ-label. Replaying
the witness re-derives the κ-label without re-hashing the
original content.
**Self-describing witness format.** A witness blob MUST begin
with a 4-byte header encoding the label width N and fingerprint
width FP as two u16 little-endian values:
    offset 0: label_width    (u16 LE, e.g. 71 for sha256)
    offset 2: fingerprint_width (u16 LE, e.g. 32 for sha256)
    offset 4: trace data     (implementation-defined)
This header allows a consumer to determine the witness's
parametric type without external metadata.
A witness blob is linked to the blob it attests by an edge
(relation: "witness-of").
### 3.7 Schemas
A schema is a blob whose content defines validation rules for
content written to a specific path scope. The schema format is
not prescribed — OpenAPI v3, JSON Schema, WASM bytecode, or any
format that can express accept/reject decisions is valid.
A schema blob is linked to its path scope by an edge (relation:
"schema-for") with the scope encoded in the edge metadata.
**Schema validation is an admission predicate, not a κ-label
modifier.** A schema-validated blob receives the same κ-label
it would receive without the schema. The schema gates admission;
it does not participate in addressing. Content that fails
validation is rejected and never enters the store.
Schema evolution: updating a schema produces a new schema blob
linked to the old one by a "derives-from" edge. Content validated
against the old schema retains its original κ-label.
Schemas are self-hosting: the schema defining valid schema
structure is itself a blob (Appendix E).
### 3.8 Pins and Finalizers
A pin is a blob whose content names a κ-label to protect from
garbage collection, a TTL, and optional controller/session
identifiers:
    protected   κ-label of the blob to protect
    ttl         seconds until expiry (0 = permanent)
    controller  controller name (if finalizer)
    session     upload session identifier (if upload protection)
A pin blob is linked to the protected blob by an edge (relation:
"pins"). TTL 0 is permanent until explicitly superseded. A
finalizer pin blocks GC until the named controller releases it.
An upload session pin protects partially-uploaded content and
expires on upload completion or session timeout.
Pin state changes (create, expire, release) produce new blobs
linked by "derives-from" edges. Pin history is auditable.
---
## 4. Addressing
### 4.1 The κ-Address Convention
Every resource is addressable by a structured string encoding
organizational ownership, resource identity, and version:
    <org-hierarchy> ":" <resource-path> "@" <version>
**Left of the colon: organizational hierarchy.**
DNS-style subdomain depth. Each level leftward increases
specificity of ownership:
    cern.ch
    atlas.cern.ch
    atlas.cern.ch/calibration
    atlas.cern.ch/calibration/v1
The left side determines consensus domain scope, authorization
boundaries, and registry routing. The protocol does not constrain
depth — 4, 5, 6, or more levels are valid. Deeper hierarchy
enables finer delegation of ownership.
**Right of the colon: resource path.**
Path segments identifying the specific resource:
    calibrations/run3/alignment
    datasets/muon-pairs/filtered
    images/production/node
**After @: version.**
A tag name or a κ-label:
    @latest                                 tag name
    @v2.1.0                                 tag name
    @2026-07-20T14:30:00.123456789          timestamp tag
    @sha256:abcdef0123456789...             κ-label (direct)
When the version contains a colon, it is a κ-label and resolves
directly. When it does not, it is a tag name and resolves via
tag lookup.
**Examples:**
    atlas.cern.ch/calibration/v1:calibrations/run3/alignment@latest
    braincraft.io/images/v1:production/rekindle-node@v2.1.0
    iot.fleet.example.com/sensors/v1:pacific/buoy-7@sha256:abc...
    nasa.gov/jpl/deep-space/v1:voyager2/telemetry@2026-01-15T00:00:00
    home.lab/minecraft/v1:servers/survival@latest
### 4.2 Formal Grammar
    kappa-address   = org-hierarchy ":" resource-path "@" version
    org-hierarchy   = org-domain *("/" path-segment)
    org-domain      = label *("." label)
    label           = 1*(ALPHA / DIGIT / "-")
    path-segment    = 1*(ALPHA / DIGIT / "-" / "_")
    resource-path   = path-segment *("/" path-segment)
    version         = tag-name / kappa-label
    tag-name        = 1*256(VCHAR) ; excludes "/" and "@"
    kappa-label     = axis-token ":" hex-digest
    axis-token      = "sha256" / "blake3" / "sha3-256"
                    / "keccak256" / "sha512"
    hex-digest      = 2*128(HEXDIG)
### 4.3 Resolution Path
    κ-address
        |
        v
    parse: extract org-hierarchy, resource-path, version
        |
        v
    if version contains ':' (κ-label):
        resolve directly from store
    else (tag name):
        resolve via tag lookup → κ-label → blob
        |
        v
    client verifies on receipt:
        re-hash content, compare against κ-label, reject on mismatch
### 4.4 Registries
A registry is any endpoint implementing this protocol. A single
κ-address MAY be resolvable from multiple registries. Content
retrieved from any registry is verified by re-hash — the registry
is untrusted for content integrity.
Registries MAY be a multi-petabyte distributed cluster, a
single-node server, an in-memory test store, a CDN edge cache,
an embedded device, or any system that responds to the operations
in §5. The protocol does not distinguish between them.
### 4.5 Cross-Axis Enrichment
A blob stored under one σ-axis MAY be enriched with κ-labels
under other axes. Enrichment MAY be eager (compute all axes at
ingestion) or lazy (compute on demand).
When a query arrives for a κ-label whose axis is not in the
index, the registry MAY retrieve by stored axis, re-hash under
the requested axis, verify, and return. The re-hash result MAY
be stored as an additional index entry.
### 4.6 Multi-Label Push
A single blob MAY be pushed with multiple κ-labels simultaneously
via query parameters:
    PUT /v2/{path}/blobs/{κ-label}?also={κ-label-2}&also={κ-label-3}
The registry MUST verify all provided κ-labels against the
content. If any do not match, the entire push is rejected. If all
match, the blob is stored once and all κ-labels are indexed.
Registries SHOULD support at least 10 κ-labels per push. Registries
MAY return 414 if the URI is too long. This enables multi-axis
enrichment in a single round trip.
---
## 5. Operations
### 5.1 Blob Operations
**put(κ-label, content) → result**
Store content at the given κ-label.
The registry MUST verify that the κ-label matches the hash of
the content. If it does not, the registry MUST reject the write.
If the κ-label already exists and the stored content matches,
the operation succeeds (idempotent). If the κ-label already
exists and the stored content differs, the operation fails
(hash collision — probability 2^-128 for SHA-256).
If a schema is registered for the write path, the registry MUST
validate content against the schema before computing the κ-label.
Rejected content never enters the store.
If admission filters are registered for the write path, the
registry MUST execute all matching filters. If any filter rejects,
the write is rejected. If any filter fails (crash, timeout), the
write is rejected (fail closed).
The stored bytes MUST be identical to the submitted bytes. No
normalization is permitted (§3.1 exact-bytes invariant).
**get(κ-label) → content | absent**
Retrieve content by κ-label. The registry MUST NOT return content
that does not match the κ-label.
**exists(κ-label) → boolean**
Check existence without retrieving content.
**remove(κ-label) → result**
Mark the blob as GC-eligible. Content with outstanding pins,
tags, or inbound edges from reachable blobs is retained.
**list(prefix) → κ-labels**
Enumerate κ-labels matching a string prefix (including the σ-axis
prefix). Paginated.
### 5.2 Tag Operations
**tag_set(path, name, κ-label) → result**
Bind a tag to a κ-label at the given path. The referenced κ-label
MUST exist — tags MUST NOT point to absent content. If the tag
exists, the binding is atomically replaced. tag_set is
linearizable within the tag's consensus domain (§7).
**tag_get(path, name) → κ-label | absent**
Resolve a tag to a κ-label.
**tag_list(path) → (name, κ-label) pairs**
List all tags at a path. Paginated. MUST support ascending and
descending order. MUST support range filtering (after/before) for
timestamp tags. Pagination uses a cursor (the last tag name from
the prior page), not a numerical index.
**tag_delete(path, name) → result**
Remove a tag binding. Does NOT delete the content.
**tag_set_if(path, name, κ-label, expected) → result**
Compare-and-swap: set the tag only if the current value matches
expected. Fails with a conflict error if the current value has
changed. Used for optimistic concurrency control.
### 5.3 Edge Operations
**edge_put(source, relation, target, metadata) → edge-κ-label**
Create an edge blob from the canonical serialization (Appendix C).
The edge κ-label uses the source's σ-axis. The source MUST exist.
The target MAY be absent (lazy replication). If the exact edge
exists, idempotent. Returns the edge κ-label.
**edge_get(κ-label, relation, direction) → edges**
Query edges involving a κ-label.
    direction = outbound: κ-label is the source
    direction = inbound:  κ-label is the target
    direction = both:     κ-label is either
Optionally filter by relation type. Paginated.
**edge_remove(edge-κ-label) → result**
Remove an edge. The source and target blobs are not affected.
### 5.4 Composition Operations
**compose(operation, operands) → composed-κ, witness-κ**
Apply a categorical composition to operand κ-labels.
**σ-axis homogeneity.** All operands MUST share the same σ-axis.
Cross-axis composition is rejected. The composed κ-label inherits
the operands' axis.
Operations:
    g2    commutative binary product: lex-sort operands, concat, hash
    f4    involution quotient: lex-min of raw and complement
    e6    degree partition: tag by first byte mod 9
    e7    S₄ orbit: lex-min across 24 quarter-permutations
    e8    identity embedding: hash identity bytes
Canonical forms are defined in Appendix D.
The registry MUST store the composed blob, create "composed-of"
edges from the composed κ-label to each operand, generate a
witness blob, and create a "witness-of" edge. The composition
response MUST include the operation type. The "composed-of" edge
metadata MUST include the operation type to disambiguate the
F4/E7/E8 digest coincidence case (Appendix D.6).
### 5.5 Bulk Operations
**put_batch(entries) → results**
Atomically store multiple blobs. All succeed or all fail.
**tag_set_batch(bindings) → result**
Atomically set multiple tags within a single consensus domain.
### 5.6 Streaming Operations
**put_stream(κ-label, reader) → result**
Store content from a streaming reader larger than available
memory. The registry MUST compute the κ-label incrementally
as bytes arrive, using a streaming hash state. The buffer size
for streaming reads SHOULD be 16 KiB (optimal for SIMD hash
implementations on modern CPUs).
After the stream completes, the registry MUST compare the
computed κ-label against the provided κ-label. If they do not
match, the partially written content is discarded and the
operation fails.
The streaming hash state MUST be checkpointable for upload
resumption. If a connection drops mid-stream, the registry
SHOULD be able to resume from the last checkpointed hash state
rather than re-hashing from the beginning.
**get_stream(κ-label) → reader**
Retrieve content as a streaming reader. The caller reads bytes
incrementally without buffering the entire blob in memory.
Registries SHOULD support Range requests (RFC 9110 §14) on
blob GET to enable partial retrieval and parallel download of
large blobs.
### 5.7 Conditional Operations
**tag_set_if(path, name, κ-label, expected) → result**
Compare-and-swap for tag updates. Sets the tag only if the
current value matches expected. If the current value does not
match, the operation fails with a conflict error. If the tag
does not exist, expected MUST be a sentinel value indicating
absence.
Used for optimistic concurrency control: read the current tag,
compute new state, write back only if unchanged. On conflict,
the client re-reads and retries. This pattern avoids distributed
locking.
tag_set_if is linearizable within the tag's consensus domain.
### 5.8 Cross-Namespace Mount
**mount(source-path, κ-label, target-path) → result**
Make a blob available under a different namespace without
re-uploading. The registry verifies the blob exists under
source-path and creates a reference under target-path.
The mounted blob is the same blob — same κ-label, same content,
same storage. The mount creates an additional namespace binding,
not a copy. De-mounting (removing the target-path binding) does
not affect the source-path binding.
If the registry cannot verify the blob at source-path (does not
exist, insufficient authorization), the mount fails and the
client falls back to a normal upload.
If source-path is omitted, the registry MAY search across all
namespaces for the κ-label. This creates an information
disclosure risk: the search may reveal the existence of a blob
in a namespace the requesting identity is not authorized to
access. Registries SHOULD evaluate this risk before enabling
source-omitted mounts.
### 5.7 Invariants
1. **Content immutability.** Same κ-label always returns same
   content.
2. **Exact-bytes storage.** Stored bytes are identical to
   submitted bytes. No normalization.
3. **Content-before-tag.** Tags MUST NOT bind to absent κ-labels.
4. **Verify-on-put.** Registry re-hashes on put and rejects
   mismatches.
5. **Idempotent put.** Same content, same κ-label, no error.
6. **Edge source existence.** edge_put MUST verify source exists.
7. **Edge target tolerance.** edge_put MUST accept absent targets.
8. **Everything is a blob.** Edges, witnesses, schemas, pins,
   filters — all blobs, all invariants apply.
9. **σ-axis homogeneity.** Composition operands share one axis.
10. **Schema is admission only.** Schema validation does not
    alter the κ-label.
11. **Unknown field preservation.** Content is stored and served
    without stripping any fields (follows from exact-bytes).
---
## 6. Protocol Binding: HTTP
This section defines an HTTP binding. Other bindings (gRPC, RDMA,
Unix domain socket) are valid. The HTTP binding is the reference
binding for conformance testing.
### 6.1 Global HTTP Behavior
**Redirects.** Registries MAY respond to any request with a
redirect per RFC 9110 §15.4. Clients SHOULD follow redirects.
Clients MUST NOT forward Authorization headers across host
boundaries on redirect. The success status codes in this
specification are those returned after all redirects have been
followed.
**Rate limiting.** Registries MAY return 429 Too Many Requests
for any operation. The response SHOULD include a Retry-After
header per RFC 6585 §4 and RFC 9110 §10.2.3.
**Warnings.** Registries MAY include `Warning: 299 - "<message>"`
headers per RFC 9111 §5.5 for non-fatal issues (deprecated
endpoints, impending GC, expiring capabilities). Warning data
MUST NOT exceed 4096 bytes total. Clients SHOULD report warnings
to users. Clients MUST NOT take automated action based on
warnings.
**Content-Type.** Clients SHOULD set Content-Type on PUT requests.
Registries that receive a Content-Type MUST store it as metadata
and echo it on subsequent GET responses for the same κ-label. If
no Content-Type was set on PUT, registries MUST respond with
`application/octet-stream` on GET.
### 6.2 Version Check
    GET /v2/
    Response: 200 OK
    Body: {"kappa-distribution": "2.0.0"}
404 indicates the endpoint does not implement κ-Distribution.
### 6.3 Blob Endpoints
**Check existence:**
    HEAD /v2/{path}/blobs/{κ-label}
    Response: 200 OK
    Headers:
        Content-Length: <byte-count>
        X-Kappa-Label: <κ-label>
        X-Kappa-Axis: <σ-axis>
    Response: 404 Not Found
The X-Kappa-Label header is REQUIRED on all responses that
identify a blob. This is the server's assertion of the content
address. Clients SHOULD verify by re-hashing independently.
Content-Length is REQUIRED on HEAD to allow pre-allocation.
**Retrieve:**
    GET /v2/{path}/blobs/{κ-label}
    Response: 200 OK
    Headers:
        Content-Length: <byte-count>
        X-Kappa-Label: <κ-label>
        X-Kappa-Axis: <σ-axis>
        Content-Type: <stored-media-type | application/octet-stream>
    Body: content bytes
    Response: 404 Not Found
Registries SHOULD support Range requests per RFC 9110 §14.
**Push (monolithic):**
    PUT /v2/{path}/blobs/{κ-label}
    Body: content bytes
    Response: 201 Created (new blob stored)
    Headers:
        X-Kappa-Label: <κ-label>
        Location: /v2/{path}/blobs/{κ-label}
    Response: 200 OK (blob already exists, idempotent)
    Response: 409 Conflict (κ-label does not match content)
    Response: 422 Unprocessable Entity (schema/filter rejected)
**Push (multi-label):**
    PUT /v2/{path}/blobs/{κ-label}?also={κ-label-2}&also={κ-label-3}
All κ-labels verified. 409 if any mismatch. Registries SHOULD
support at least 10 labels. Registries MAY return 414 if URI
is too long.
**Unpin:**
    DELETE /v2/{path}/blobs/{κ-label}
    Response: 202 Accepted (GC-eligible)
    Response: 409 Conflict (outstanding finalizer pins)
### 6.4 Chunked Upload Protocol
The chunked upload protocol is a stateful session with recovery
semantics. The session URL is opaque — clients MUST NOT assemble
it manually. It MAY contain critical query parameters. It MAY
point to a different server.
**Phase 1 — Start session:**
    POST /v2/{path}/blobs/uploads/
    Response: 202 Accepted
    Headers:
        Location: <upload-url>         (opaque, server-generated)
        X-Kappa-Upload-Session: <uuid>
The registry MUST create a temporary upload session pin (§3.8)
protecting partial content from GC.
If the registry specifies a minimum chunk size:
    X-Kappa-Chunk-Min-Length: <bytes>
All chunks except the final SHOULD be at least this size.
**Phase 2 — Upload chunks:**
    PATCH <upload-url>
    Headers:
        Content-Range: <start>-<end>
    Body: chunk bytes
    Response: 202 Accepted
    Headers:
        Location: <upload-url>   (MAY change between chunks)
        Range: 0-<last-byte-received>
Chunks MUST be uploaded sequentially. The first byte of chunk N
MUST equal the last byte of chunk N-1 plus 1. Out-of-order
chunks MUST receive 416 Range Not Satisfiable.
**Phase 2a — Recovery after failure:**
    GET <upload-url>
    Response: 204 No Content
    Headers:
        Location: <upload-url>   (current, MAY have changed)
        Range: 0-<last-byte-received>
After a 416 or connection loss, the client GETs the upload URL
to discover the server's current state. The response tells the
client which bytes the server has, and the Location header
provides the current upload URL for the next chunk.
**Phase 3 — Complete upload:**
    PUT <upload-url>?kappa=<κ-label>
    Body: final chunk (may be empty)
    Response: 201 Created
    Headers:
        Location: /v2/{path}/blobs/{κ-label}
        X-Kappa-Label: <κ-label>
    Response: 409 Conflict (κ-label does not match assembled content)
The κ-label in the query parameter is the hash of the ENTIRE
blob, not the final chunk. The registry MUST verify against the
complete assembled content.
**Phase 2b — Cancel upload:**
    DELETE <upload-url>
    Response: 204 No Content
Cancels the session. Partial content is discarded. The upload
session pin is released.
### 6.5 Tag Endpoints
**Resolve:**
    GET /v2/{path}/manifests/{version}
    When {version} contains ':' → direct retrieval by κ-label
    When {version} has no ':'  → tag lookup, then retrieval
    Response: 200 OK
    Headers:
        X-Kappa-Label: <resolved-κ-label>
        Content-Type: <stored-media-type | application/octet-stream>
    Body: content bytes
    Response: 404 Not Found
**Bind tag:**
    PUT /v2/{path}/manifests/{tag}
    Body: content bytes
    Response: 201 Created
    Headers:
        X-Kappa-Label: <computed-κ-label>
    Response: 422 Unprocessable Entity (schema/filter rejected)
The registry MUST:
1. Execute admission filters for the path (if registered)
2. Validate against registered schema (if registered)
3. Compute the κ-label from the content
4. Store the blob (idempotent if exists)
5. Bind the tag via consensus
6. Return the κ-label in X-Kappa-Label
**Multi-tag bind:**
    PUT /v2/{path}/manifests/{tag}?tag={tag2}&tag={tag3}
Binds multiple tags to the same content in one consensus round.
Registries SHOULD support at least 10 tags per request.
**List tags:**
    GET /v2/{path}/tags/list
    Query parameters:
        n:      max results (default 100)
        last:   pagination cursor (tag name, NOT a number)
        order:  "asc" | "desc" (default "asc")
        after:  ISO 8601 lower bound (timestamp tag range)
        before: ISO 8601 upper bound (timestamp tag range)
    Response: 200 OK
    Headers:
        Link: <next-page-url>; rel="next"  (if more pages)
    Body: {"tags": [{"name": "latest", "kappa": "sha256:..."}]}
The `last` parameter MUST be a tag name value, not a numerical
index. Tag names MUST be returned in ASCIIbetical order per the
requested order.
When `n=0`, the response MUST be an empty list and MUST NOT
include a Link header.
**Delete tag:**
    DELETE /v2/{path}/manifests/{tag}
    Response: 202 Accepted
    Response: 404 Not Found
    Response: 405 Method Not Allowed (deletion disabled)
### 6.6 Edge Endpoints
**Query edges:**
    GET /v2/{path}/edges/{κ-label}
    Query parameters:
        relation:   filter by type
        direction:  "inbound" | "outbound" | "both" (default "outbound")
        n:          max results
        last:       pagination cursor
    Response: 200 OK
    Body:
    {
        "edges": [
            {
                "edge_kappa": "sha256:...",
                "source": "sha256:...",
                "relation": "derives-from",
                "target": "sha256:...",
                "metadata": {}
            }
        ]
    }
**Create edge:**
    PUT /v2/{path}/edges/
    Body:
    {
        "source": "sha256:...",
        "relation": "derives-from",
        "target": "sha256:...",
        "metadata": {}
    }
    Response: 201 Created
    Headers:
        X-Kappa-Label: <edge-blob-κ-label>
    Response: 200 OK (edge exists, idempotent)
**Delete edge:**
    DELETE /v2/{path}/edges/{edge-κ-label}
    Response: 202 Accepted
    Response: 404 Not Found
### 6.7 Cross-Namespace Linking
**Mount blob from another namespace:**
    POST /v2/{path}/blobs/uploads/?mount={κ-label}&from={source-path}
    Response: 201 Created (mount successful)
    Headers:
        Location: /v2/{path}/blobs/{κ-label}
    Response: 202 Accepted (mount not available, proceed with upload)
    Headers:
        Location: <upload-url>
If the registry can verify the blob exists under `source-path`,
it makes it available under the target `path` without re-upload.
If the `from` parameter is omitted, the registry MAY search
across all namespaces for the κ-label. Registries SHOULD evaluate
the information disclosure risk of cross-namespace search before
enabling this behavior.
If the registry does not support mounting or the blob is not
found at the source path, the registry responds with 202 and the
client proceeds with a normal upload.
### 6.8 Witness Endpoint
    GET /v2/{path}/witnesses/{κ-label}
    Response: 200 OK
    Headers:
        Content-Type: application/octet-stream
    Body: witness blob content (self-describing format per §3.6)
    Response: 404 Not Found
The κ-label parameter is the κ-label of the witnessed blob.
The registry resolves the witness via edge query (relation:
"witness-of", direction: inbound).
### 6.9 Composition Endpoint
    POST /v2/{path}/compose/{operation}
    Body: {"operands": ["sha256:...", "sha256:..."]}
    Response: 200 OK
    Body:
    {
        "composed": "sha256:...",
        "witness": "sha256:...",
        "operands": ["sha256:...", "sha256:..."],
        "operation": "g2"
    }
    Response: 422 Unprocessable Entity (σ-axis mismatch)
### 6.10 Schema Endpoints
**Register:**
    PUT /v2/{path}/schemas/{scope}
    Body: schema content
    Response: 201 Created
    Headers:
        X-Kappa-Label: <schema-blob-κ-label>
**Retrieve:**
    GET /v2/{path}/schemas/{scope}
    Response: 200 OK
**List:**
    GET /v2/{path}/schemas/
    Response: 200 OK
    Body: {"schemas": [{"scope": "...", "kappa": "sha256:..."}]}
### 6.11 GC Endpoints
**Pin:**
    POST /v2/{path}/gc/pin
    Body: {"kappa": "sha256:...", "ttl": 3600, "controller": ""}
    Response: 201 Created
    Headers:
        X-Kappa-Label: <pin-blob-κ-label>
**Unpin:**
    POST /v2/{path}/gc/unpin
    Body: {"pin_kappa": "sha256:..."}
    Response: 200 OK
    Response: 409 Conflict (outstanding finalizer)
**Trigger sweep:**
    POST /v2/{path}/gc/sweep
    Response: 202 Accepted
    Body: {"sweep_id": "<uuid>"}
**Status:**
    GET /v2/{path}/gc/status
    Response: 200 OK
    Body:
    {
        "last_sweep": "2026-07-20T14:00:00Z",
        "duration_seconds": 90,
        "objects_scanned": 1000000,
        "objects_evicted": 42000,
        "bytes_reclaimed": 1073741824,
        "pending_finalizers": [
            {"kappa": "sha256:...", "controller": "name"}
        ]
    }
### 6.12 Admission Filter Endpoints
**Register filter:**
    PUT /v2/{path}/filters/{scope}
    Body: filter content (WASM bytecode, executable, etc.)
    Response: 201 Created
    Headers:
        X-Kappa-Label: <filter-blob-κ-label>
**List filters:**
    GET /v2/{path}/filters/
    Response: 200 OK
    Body: {"filters": [{"scope": "...", "kappa": "sha256:..."}]}
**Remove filter:**
    DELETE /v2/{path}/filters/{filter-κ-label}
    Response: 202 Accepted
The filter blob is unlinked, not deleted. It remains for audit.
**Filter interface.** A filter receives content bytes and returns
accept or reject:
    input:  content bytes
    output: {"accept": true} | {"accept": false, "reason": "..."}
Multiple filters matching a scope MUST all accept. If any rejects,
the write is rejected. If any fails (crash, timeout), the write
is rejected (fail closed). The filter runtime is not prescribed.
### 6.13 Discovery Endpoint
    GET /v2/_discovery/
    Response: 200 OK
    Body:
    {
        "hierarchies": [
            {
                "org": "atlas.cern.ch/calibration/v1",
                "resources": ["calibrations", "alignments"]
            }
        ]
    }
### 6.14 Health Endpoints
    GET /v2/_health/live      200 if process alive
    GET /v2/_health/ready     200 if backend connected, consensus available
    GET /v2/_health/startup   200 if initialization complete
### 6.15 Extension System
Registries MAY implement non-standard operations via extension
endpoints. Extension paths MUST be prefixed with `_` followed by
the extension namespace:
    /v2/_<extension>/<component>/<module>
    /v2/{path}/_<extension>/<component>/<module>
Reserved namespaces: `_kappa` (protocol-defined endpoints),
`_system` (system-level operations). All other `_`-prefixed
namespaces are available for extensions.
**Extension discovery:**
    GET /v2/_kappa/ext/discover
    Response: 200 OK
    Body:
    {
        "extensions": [
            {
                "name": "_example/metrics/v1",
                "description": "Prometheus metrics",
                "endpoints": ["/v2/_example/metrics/v1/scrape"]
            }
        ]
    }
Extension versioning SHOULD use Accept/Content-Type headers. If
fundamentally changed, introduce a new component or module path.
### 6.16 Error Response Format
Error responses MUST use:
    {
        "errors": [
            {
                "code": "<UPPERCASE_WITH_UNDERSCORES>",
                "message": "<human readable>",
                "detail": <arbitrary JSON | null>
            }
        ]
    }
The `code` field MUST contain only uppercase ASCII letters and
underscores. The `message` field is OPTIONAL. The `detail` field
is OPTIONAL and MAY be arbitrary JSON.
Defined error codes:
    BLOB_UNKNOWN            blob not found
    BLOB_UPLOAD_INVALID     upload session invalid
    BLOB_UPLOAD_UNKNOWN     upload session not found
    DIGEST_INVALID          κ-label does not match content
    SCHEMA_VIOLATION        content fails schema validation
    FILTER_REJECTED         admission filter rejected content
    FILTER_FAILED           admission filter execution failed
    TAG_UNKNOWN             tag not found
    TAG_CONTENT_ABSENT      tag_set for absent κ-label
    TAG_CONFLICT            tag_set_if expected value mismatch
    EDGE_UNKNOWN            edge not found
    EDGE_SOURCE_ABSENT      edge source κ-label absent
    AXIS_MISMATCH           composition operands differ in σ-axis
    NAME_INVALID            invalid resource path
    UNAUTHORIZED            authentication required
    DENIED                  access denied
    UNSUPPORTED             operation not supported
    TOOMANYREQUESTS         rate limit exceeded
    FINALIZER_OUTSTANDING   unpin blocked by finalizer
    SIZE_INVALID            content length mismatch
### 6.17 Endpoint Summary
    Method  Path                                       Cons.  Level
    ------  ----                                       -----  -----
    GET     /v2/                                       no     1
    HEAD    /v2/{p}/blobs/{κ}                          no     1
    GET     /v2/{p}/blobs/{κ}                          no     1
    PUT     /v2/{p}/blobs/{κ}                          no     1
    POST    /v2/{p}/blobs/uploads/                     no     1
    PATCH   <upload-url>                               no     1
    GET     <upload-url>                               no     1
    PUT     <upload-url>?kappa={κ}                     no     1
    DELETE  <upload-url>                               no     1
    DELETE  /v2/{p}/blobs/{κ}                          no     1
    POST    /v2/{p}/blobs/uploads/?mount={κ}&from={p}  no     1
    GET     /v2/{p}/manifests/{v}                      no     2
    PUT     /v2/{p}/manifests/{t}                      yes    2
    DELETE  /v2/{p}/manifests/{t}                      yes    2
    GET     /v2/{p}/tags/list                          no     2
    GET     /v2/{p}/edges/{κ}                          no     3
    PUT     /v2/{p}/edges/                             no     3
    DELETE  /v2/{p}/edges/{κ}                          no     3
    GET     /v2/{p}/witnesses/{κ}                      no     4
    POST    /v2/{p}/compose/{op}                       no     4
    PUT     /v2/{p}/schemas/{scope}                    yes    4
    GET     /v2/{p}/schemas/{scope}                    no     4
    GET     /v2/{p}/schemas/                           no     4
    POST    /v2/{p}/gc/pin                             no     5
    POST    /v2/{p}/gc/unpin                           no     5
    POST    /v2/{p}/gc/sweep                           no     5
    GET     /v2/{p}/gc/status                          no     5
    PUT     /v2/{p}/filters/{scope}                    yes    5
    GET     /v2/{p}/filters/                           no     5
    DELETE  /v2/{p}/filters/{κ}                        yes    5
    GET     /v2/_discovery/                            no     2
    GET     /v2/_health/live                           no     1
    GET     /v2/_health/ready                          no     1
    GET     /v2/_health/startup                        no     1
    GET     /v2/_kappa/ext/discover                    no     1
    35 endpoints. 5 require consensus. 30 do not.
---
## 7. Consistency and Consensus
### 7.1 Content Consistency
Content is eventually consistent across replicas. A blob stored
at one registry becomes available at others through replication.
Content is immutable and infinitely cacheable — a CDN, edge
cache, reverse proxy, or local disk cache MAY serve any blob
indefinitely without staleness concerns. Cache invalidation is
unnecessary because the κ-label is deterministic: the same
κ-label always maps to the same bytes. A cache entry keyed by
κ-label is valid forever.
Content writes are idempotent: two writers pushing the same
content produce the same κ-label and the second put is a no-op.
No coordination is required between concurrent writers of the
same content.
### 7.2 Tag Consistency
Tags are sequentially consistent within a consensus domain. All
clients observing the same domain see tag updates in the same
order. If client A observes tag T bound to κ₁ and later bound
to κ₂, every other client in the same domain that observes both
bindings observes κ₁ before κ₂.
Tag reads outside the consensus domain (e.g., from a replica
that has not yet received the update) MAY return a stale value.
This is the standard eventual consistency tradeoff — the
staleness window is bounded by the replication lag.
### 7.3 Consensus Domains
The protocol requires that consensus domains exist. Each tag
belongs to exactly one domain. tag_set, tag_delete, and
tag_set_if are linearizable within a domain.
The protocol does not prescribe domain scoping. Deployments MAY
scope by org-hierarchy (all tags under `atlas.cern.ch/...` in
one domain), by resource path (each resource type in its own
domain), by geographic region, or by any other partition.
Scoping determines the contention surface: tags in different
domains update without coordination. Narrower domains enable
higher aggregate throughput at the cost of no cross-domain
ordering guarantees.
### 7.4 Consensus Interface
The protocol requires that the consensus mechanism provides:
    linearizable tag_set(path, name, κ-label)
    linearizable tag_delete(path, name)
    linearizable tag_set_if(path, name, κ-label, expected)
The consensus protocol is a deployment choice. Single-node mutex,
Raft, Paxos, RDMA-based consensus, or any mechanism that
provides linearizable read-write on tag bindings is valid. The
protocol does not prescribe leader election, quorum size, log
format, or membership changes.
### 7.5 Content-Before-Tag
Blobs MUST exist before tags bind to them. Blob durability is
confirmed before the tag update is submitted to consensus.
Combined guarantee: once a tag update is acknowledged by
consensus, the referenced content is guaranteed to be readable
from the local store. This guarantee holds even if the
consensus acknowledgment arrives before replication delivers
the content to other registries — each registry independently
confirms content existence before binding its local tags.
### 7.6 Edge Consistency
Edge creation is idempotent and does not require consensus.
Two registries independently creating the same edge (same
source, relation, target, metadata) produce the same edge
blob with the same κ-label. Duplicate edge creation is a
no-op. Edge deletions do not require consensus — an edge
deleted at one registry persists at others until replication
propagates the deletion or GC evicts the unreachable edge blob.
---
## 8. Replication and Federation
### 8.1 Replication Model
Content replicates between registries via whatever mechanism the
deployment provides. The protocol does not define a replication
transport. It defines one invariant: content received via
replication MUST be verified by re-hash against its κ-label. The
replication transport is untrusted.
The registry MAY implement replication as:
- Background async push from writer to replicas
- Pull-based polling by replicas
- Gossip-based dissemination
- External orchestration (rsync, object store replication)
- Any combination
The protocol does not distinguish these. The verify-on-receipt
invariant is the only constraint.
### 8.2 Metadata-Before-Content
Tags and edges MAY replicate faster than content blobs. A
registry that receives a tag pointing to an absent κ-label MUST
accept the tag binding. Content requests for absent κ-labels
return absent. Content populates via continued replication or
federation fetch.
This ordering is intentional: it allows lightweight metadata
to propagate quickly while heavy content follows. A consumer
that resolves a tag and receives an absent κ-label knows the
content exists somewhere in the federation but has not yet
arrived locally. The consumer MAY wait for replication or
trigger a federation fetch.
### 8.3 Federation Fetch and Multi-Hop Relay
When a registry does not hold a requested κ-label, it MAY query
peer registries. Peers MAY relay the query to their own peers.
This enables resolution across networks where the requesting
registry cannot directly reach the registry holding the content.
Multi-hop resolution:
    A requests κ from local store → absent
    A queries peer B → B checks local → absent
    B queries peer C (A cannot reach C) → C has κ
    C returns content to B
    B re-hashes → matches → stores locally (idempotent put)
    B returns content to A
    A re-hashes → matches → stores locally (idempotent put)
    A returns content to client
    Client re-hashes → matches → verified
Every hop verifies by re-hash. A compromised or buggy
intermediate registry that returns wrong content is detected
at the next hop. The final client verifies independently of
all intermediate hops.
Federation fetch results are cached locally via idempotent put.
Subsequent requests for the same κ-label are served locally
without traversing the federation.
Federation fetch is transparent to the client. The client sends
one GET and receives one response. The server handles all
peer queries and relay internally. The latency includes all
transit round trips.
### 8.4 Conflict Resolution
When two registries independently update the same tag during a
period of disconnection or replication delay, both updates
eventually propagate to all registries. A receiving registry
detects a conflict when an arriving tag update's previous value
does not match the local current value.
Resolution is deterministic and requires no cross-registry
coordination:
    1. Compare HLC timestamps of the conflicting updates
    2. The update with the strictly higher timestamp wins
    3. If timestamps are equal: the update with the
       lexicographically greater κ-label wins
Properties:
- Deterministic: all registries arrive at the same winner
  without communication
- Commutative: the order updates are received does not affect
  the outcome
- Idempotent: applying the same update twice has no effect
Both the winning and losing κ-labels remain in the store as
blobs. Only the tag pointer resolves. No content is deleted
by conflict resolution.
Applications that need to detect conflicts MAY observe conflict
events through the registry's notification mechanism. The
protocol does not prescribe the notification mechanism.
**Site-scoped tag pattern.** The preferred pattern for conflict
avoidance is site-scoped tags:
    Site A writes: tag "latest@site-a" → κ₁
    Site B writes: tag "latest@site-b" → κ₂
A controller reconciles site-scoped tags into a global "latest"
using application-specific merge logic. This moves conflict
resolution from the protocol (deterministic but generic) to the
application (domain-aware).
### 8.5 Long-Duration Disconnection
The protocol makes no assumption about maximum disconnection
duration. A registry offline for seconds, hours, days, weeks,
months, or years reconnects and synchronizes through the same
mechanism:
    1. Tags accumulated during disconnection replicate
    2. Conflicts resolve deterministically (§8.4)
    3. Content backfills on demand via federation fetch (§8.3)
    4. Edges replicate and may arrive before content (§8.2)
    5. No manual intervention required
An Antarctic sensor station offline for six months, a deep-space
probe reconnecting after years, and a CDN edge node offline for
six seconds use the same reconnection protocol. The consistency
model and conflict resolution are designed for arbitrary
replication delay.
### 8.6 Client Fallback Procedures
For every optional feature, the protocol defines the client
fallback:
- Chunked upload not supported → monolithic PUT
- Multi-label push not supported → sequential single-label PUTs
- Multi-tag bind not supported → sequential tag_sets
- Mount not supported (202 response) → proceed with normal upload
- Cross-axis enrichment not supported → client computes and pushes
- Federation fetch not supported → client queries alternative
  registries directly
Clients MUST implement the fallback path for every optional
feature they use.
---
## 9. Garbage Collection
### 9.1 Roots
GC roots: every κ-label that is pinned (unexpired) or tagged.
### 9.2 Reachability
A κ-label is reachable if it is a root or if it is the target of
an "owns" or "composed-of" edge from a reachable κ-label. Edge
blobs are reachable if source or target is reachable.
### 9.3 Sweep Algorithm
1. Snapshot roots (pins + tags) at sweep start
2. Walk edges from roots along "owns" and "composed-of"
3. Mark all visited κ-labels reachable
4. Mark edge blobs reachable if source or target is reachable
5. Evict unmarked κ-labels, log evictions
### 9.4 Snapshot Isolation
Pins added after snapshot protect on the NEXT sweep. Upload
session pins active at snapshot protect during the current sweep.
### 9.5 Parallelism
Sweeps MAY partition the κ-label space by hash prefix, namespace,
or any sharding dimension. Partitions sweep independently.
### 9.6 Finalizers
Outstanding finalizers block eviction. Stuck finalizers are
reported in GC status. Operators or recovery controllers clear
them.
### 9.7 Sweep Scheduling
Not prescribed. Deployments choose frequency. Sweeps MUST NOT
block reads or writes. Eviction MAY be batched asynchronously.
---
## 10. Admission Control
### 10.1 Filter Model
A filter is a blob. Its content is executable logic. It is linked
to a path scope by a "filter-for" edge. It receives content bytes
and returns accept or reject.
### 10.2 Filter Matching
Filters scope by path prefix. Multiple matching filters MUST all
accept. If any rejects or fails, the write is rejected.
### 10.3 Filter Lifecycle
Filters are blobs. Updating a filter stores a new blob and
updates the filter tag. Old filter blobs remain for audit.
---
## 11. Security
### 11.1 Verify-on-Receipt
κ-labels provide content integrity independent of transport
security and server trust. A client that retrieves a blob and
re-hashes it detects any corruption, tampering, or substitution
regardless of how the blob arrived. This property is strictly
stronger than transport encryption: TLS verifies the channel;
κ-labels verify the content. A correctly-TLS'd connection to a
compromised server delivers authenticated garbage. κ-label
verification detects the garbage.
Verify-on-receipt is REQUIRED for clients (not SHOULD). Every
blob received MUST be re-hashed and compared against the
requested κ-label. This is a departure from OCI Distribution,
which specifies verification as SHOULD. The SHOULD-level was
insufficient — OCI upgraded Docker-Content-Digest from SHOULD to
MUST after discovering clients were silently skipping verification.
### 11.2 Tag Integrity
Tags are mutable and consensus-protected. Redirecting a tag
requires compromising a consensus quorum (a majority of nodes in
the consensus domain). An attacker controlling fewer than a
majority cannot redirect tags.
For deployments requiring stronger guarantees, signed tags
provide consensus-independent verification: the tag value is a
κ-label of a signed manifest containing the content κ-label and
the signer's identity. A client verifying the signature confirms
the tag binding independent of the consensus layer.
### 11.3 Edge Integrity
Edges are blobs with κ-labels. A tampered edge is detectable by
re-hashing the edge blob and comparing against its κ-label.
Composition edges carry witness blobs. Forging a composition edge
requires forging a witness that replays correctly — the verifier
detects the forgery by replaying the witness.
Application-created edges (e.g., "owns", "derives-from") are not
witness-protected. Their integrity relies on the same mechanism
as any blob: re-hash on retrieval detects tampering. However,
edge creation is not consensus-protected, so a compromised
registry can inject false edges. Applications that need edge
integrity guarantees beyond re-hash SHOULD sign edge content.
### 11.4 Content Provenance
A κ-label is intrinsic provenance. It proves the content existed
in its exact form at the time the κ-label was computed. Combined
with a witness, it proves how the canonical form was derived —
which pipeline stages ran, what intermediate values were produced.
Combined with a signed tag, it proves who asserted the content's
identity. Combined with a composition witness, it proves which
operands produced the composed value and which operation was used.
This provenance chain is self-contained. No external attestation
service, no centralized certificate authority, no blockchain
anchor is required. Any party with the blob, its κ-label, and
its witness can independently verify provenance.
### 11.5 Replay Protection
A client receiving a blob with a valid κ-label knows the content
is authentic (matches the address) but not necessarily current.
An attacker serving old blobs with valid κ-labels is undetectable
at the content layer.
Mitigation: access content through tags when freshness matters.
Tag resolution reflects the most recent consensus-committed
state. A tag-resolved κ-label is both authentic (re-hash) and
current (consensus-committed).
For use cases where the κ-label is the only identifier (e.g.,
replication), replay protection comes from the conflict
resolution mechanism (§8.4) — the tag with the higher timestamp
wins, so an old κ-label served in place of a new one is overridden
when the correct tag update arrives.
### 11.6 Multi-Hop Trust
In federation fetch (§8.3), content traverses multiple registries.
Each hop verifies by re-hash. A compromised intermediate registry
cannot tamper with content without the next hop detecting the
mismatch. This provides end-to-end content integrity across
untrusted relay chains without pre-established trust relationships.
The relay chain does not provide confidentiality. Each relay sees
the content in cleartext. Deployments requiring confidentiality
across untrusted relays MUST encrypt content before κ-label
computation (§11.9).
### 11.7 Pin Manipulation
Pin creation and deletion MUST be authorized. Unauthorized pin
creation enables GC denial-of-service (pin flooding — creating
pins faster than GC can process them, exhausting storage).
Unauthorized pin deletion enables premature eviction of protected
content. Pin creation SHOULD be rate-limited per identity.
### 11.8 Authorization
The protocol requires that operations are authorized before
execution. It does not define the authorization framework. It
defines the authorization scope: org-hierarchy + resource path.
An identity authorized at a given scope can operate within it.
An identity not authorized MUST receive a denial.
The registry MUST NOT disclose the existence of content or tags
in unauthorized scopes. A query for content in an unauthorized
scope MUST return the same response as a query for non-existent
content (404, not 403) to prevent existence probing.
### 11.9 Encryption
Encryption is not defined by this protocol. Content is stored as
provided by the client. Encryption at rest, encryption in transit,
and content-level encryption are all deployment concerns. If
content-level encryption is desired, the client MUST encrypt
before κ-label computation — the κ-label covers the ciphertext,
not the plaintext. A verifier re-hashing the encrypted blob
confirms the ciphertext integrity.
### 11.10 Denial of Service
The protocol surfaces subject to denial of service:
- Blob upload: mitigated by rate limiting, upload session TTL,
  and the registry's right to reject large uploads
- Tag updates: mitigated by consensus throughput limiting
- Edge creation: mitigated by rate limiting
- Federation fetch: mitigated by query depth limits and
  circuit breakers on peer query timeouts
- GC sweep: mitigated by sweep scheduling and the principle
  that sweeps MUST NOT block reads or writes
---
## 12. Failure Modes
### 12.1 Node Down
Content uniquely held by a down node is temporarily unavailable.
Federation fetch resolves from peers if any hold a copy. If no
peer holds a copy, the content is unavailable until the node
recovers. Tag reads return the last committed value from the
consensus domain. Content reads for available κ-labels continue
normally on any operational node.
Recovery: the down node rejoins, its content becomes available,
and any κ-labels that were federation-fetched from peers during
the outage are now locally redundant on both the recovered node
and the fetching peers.
### 12.2 Consensus Quorum Lost
Tag updates in the affected consensus domain stall. No new tags
are committed. Content reads continue — content does not require
consensus. Tag reads return the last committed value. Content
writes continue — blobs are idempotent and coordination-free.
Recovery: quorum is restored (by recovering nodes or
reconfiguring the domain). Pending tag updates are committed
or rejected depending on the consensus protocol's recovery
semantics.
### 12.3 Corrupted Blob
A blob whose content does not match its κ-label is corrupt.
Detection: any get that re-hashes on retrieval detects the
mismatch. The registry MUST return an error rather than corrupt
content.
Recovery: federation fetch retrieves an uncorrupted copy from a
peer. The corrupt entry is replaced. The κ-label itself is not
invalidated — it correctly identifies the uncorrupted content
that the corrupt entry should have been.
### 12.4 Split-Brain Network Partition
Registries on opposite sides of a partition operate independently.
Tags diverge — each side commits updates to its local consensus
domain. Content written on one side is unavailable on the other
(no replication path during partition).
Recovery: partition heals, replication resumes, tag conflicts
resolve deterministically (§8.4). Both sides converge to the
same tag state. Content written during the partition replicates
to the other side. No data loss.
### 12.5 GC Race
Snapshot isolation (§9.4) prevents premature eviction. A blob
pinned after the sweep snapshot is protected on the next sweep.
An upload session pinned before the snapshot is protected during
the current sweep.
Worst case: a blob is evicted and a concurrent writer expected
it to exist for a tag binding. The writer's tag_set fails
(content-before-tag violation). The writer re-pushes the blob
(idempotent if still available from a peer) and retries.
### 12.6 Stuck Finalizer
A finalizer whose controller has crashed or is permanently
unavailable blocks eviction of its protected blob indefinitely.
Detection: GC status endpoint reports outstanding finalizers
with controller name, protected κ-label, and pin age.
Recovery: operator or recovery controller issues a finalizer
release (new pin blob with released state). Automated recovery
MAY use a finalizer TTL: if a finalizer is older than a
configured threshold, it MAY be automatically released.
### 12.7 Federation Fetch Exhaustion
All reachable peers queried, none holds the content. The registry
returns absent to the client. No partial content is returned.
The client MAY retry with a different peer set, wait for
replication to deliver the content, or fail.
Mitigation: registries SHOULD implement circuit breakers on
federation queries to prevent cascading timeouts when a large
subgraph is unavailable.
### 12.8 Admission Filter Failure
A filter execution that crashes, times out, or exceeds resource
limits causes the write to be rejected. Fail closed — no
content enters the store on filter failure. The failure is
logged with the filter κ-label, the content κ-label (if
computable), and the failure reason.
Recovery: fix the filter, push a new filter blob, update the
filter tag. Old content that was rejected MAY be re-pushed.
### 12.9 Cross-Axis Resolution Failure
Content stored under axis A is requested under axis B. On-demand
re-hash requires retrieval (axis A), re-hash (axis B), and
verification. If the content is unavailable (node down, not
replicated), cross-axis resolution fails. Returns absent. Retry
when the content becomes available under its original axis.
### 12.10 Upload Session Timeout
A chunked upload session whose client has not sent a chunk
within the session TTL is abandoned. Partial content is
discarded. The upload session pin is released. The client
starts a new session if it wishes to retry.
### 12.11 Upload Out-of-Order Chunk
A PATCH with Content-Range starting at a byte offset that does
not equal the next expected byte returns 416 Range Not
Satisfiable. The client MUST use the recovery endpoint (GET on
the upload URL) to discover the server's current byte range and
resume from the correct offset.
### 12.12 κ-Label Mismatch on Complete
The closing PUT of a chunked upload includes the expected κ-label
of the full assembled content. If the assembled content hashes to
a different κ-label, the upload is rejected with 409. All partial
data is discarded. The upload session pin is released. The client
SHOULD verify its own hash computation and retry.
---
## 13. Conformance
### 13.1 Conformance Levels
    Level 1: Blob operations
             put, get, exists, remove, list, chunked upload
             with recovery, upload cancellation, mount
    Level 2: Level 1 + Tags
             tag_set, tag_get, tag_list, tag_delete, tag_set_if
    Level 3: Level 2 + Edges
             edge_put, edge_get, edge_remove (edges as blobs)
    Level 4: Level 3 + Composition, Witnesses, Schemas
    Level 5: Level 4 + GC, Admission, Federation, Multi-hop
### 13.2 Conformance Test Harness Architecture
The conformance test harness is a compiled test binary. Its
architecture follows the patterns proven by the OCI Distribution
conformance suite.
**Environment-variable configuration.** The harness is configured
entirely by environment variables. No code changes are required
to test different registries:
    KAPPA_REGISTRY_URL       base URL of the registry under test
    KAPPA_NAMESPACE          org-hierarchy:resource-path for test data
    KAPPA_AUTH_TOKEN         bearer token (if required)
    KAPPA_TEST_LEVELS        comma-separated levels to test (1,2,3,4,5)
    KAPPA_UPLOAD_CHUNK_SIZE  chunk size for upload tests (default 1 MiB)
    KAPPA_TEARDOWN_ORDER     "tags-first" | "blobs-first" (default "tags-first")
    KAPPA_TIMEOUT            per-operation timeout (default 30s)
**Dual report output.** The harness produces:
- JUnit XML for CI/CD integration (machine-parseable pass/fail)
- HTML report for human review (expandable per-test HTTP traces,
  progress meter, pass/fail/skip counts per level)
**Credential redaction.** Authorization tokens, passwords, and
sensitive headers are redacted from all log and report output.
HTTP traces show `Authorization: [REDACTED]` rather than the
actual token value.
**Category isolation.** Each conformance level is independently
runnable. A read-only registry passes Level 1 blob retrieval
tests without implementing Level 2 tag operations. The harness
MUST NOT assume that passing a lower level implies running a
higher level.
**Test data isolation.** Each level uses distinct test data
(different blob content, different tag names, different edge
relations) to avoid cross-contamination between levels.
**Warn for SHOULD, fail for MUST.** Operations marked SHOULD in
the spec emit warnings on non-compliance. Operations marked MUST
emit failures. The report distinguishes warnings from failures.
A registry with warnings but no failures is conformant.
**Setup/teardown symmetry.** Each level has setup (populate
registry with test data) and teardown (clean up test data). The
teardown order is configurable because different registries have
different deletion constraints. The harness MUST succeed even if
teardown is skipped (test data has TTL pins that expire).
**Idempotency verification.** For every put operation, the harness
pushes the same blob twice and verifies both succeed (idempotent).
This is a Level 1 MUST test, not an optional pattern.
**Unknown field preservation.** The harness pushes a blob
containing fields not defined by any schema, retrieves it, and
verifies byte-for-byte identity. This tests the exact-bytes
invariant (§3.1) and forward compatibility.
**Upload recovery path.** The harness deliberately sends an
out-of-order chunk, receives the expected 416, performs the GET
recovery (§6.4 Phase 2a), and resumes from the correct offset.
This tests the full upload session recovery flow.
### 13.3 Test Vectors
**Blob vectors (Level 1):**
    Input: empty bytes (0 bytes)
    sha256: sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
    blake3: blake3:af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262
    Input: UTF-8 "hello" (5 bytes)
    sha256: sha256:2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824
    blake3: blake3:ea8f163db38682925e4491c5e58d4bb3506ef8c14eb78a86e908c5624a67200f
    Tests:
    - put(κ, content) → get(κ) → content matches
    - put(κ, content) → put(κ, content) → idempotent
    - put(κ, wrong_content) → 409 rejected
    - put(κ, content) → exists(κ) → true
    - exists(absent_κ) → false
    - get(absent_κ) → 404
    - stored bytes == submitted bytes (unknown field preservation)
**Chunked upload vectors (Level 1):**
    Tests:
    - POST → PATCH chunk 1 → PATCH chunk 2 → PUT → 201
    - POST → PATCH chunk 2 (out of order) → 416
    - POST → PATCH chunk 1 → GET session → 204 with Range
    - POST → DELETE session → 204 (cancellation)
    - POST → PATCH → PUT with wrong κ → 409
**Mount vectors (Level 1):**
    Tests:
    - POST mount existing blob → 201
    - POST mount non-existent blob → 202 (fall back to upload)
**Tag vectors (Level 2):**
    Tests:
    - tag_set → tag_get → κ matches
    - tag_set update → tag_get → new κ
    - tag_set for absent κ → rejected
    - tag_delete → tag_get → absent
    - tag_list → all (name, κ) pairs, ASCIIbetical order
    - tag_set_if(expected=current) → succeeds
    - tag_set_if(expected=wrong) → conflict
    - tag_list with n=0 → empty, no Link header
    - tag_list pagination: n=2, last=<tag> → correct continuation
**Edge vectors (Level 3):**
    Tests:
    - edge_put → edge blob retrievable by own κ-label via get()
    - edge_get(source, outbound) → includes target
    - edge_get(target, inbound) → includes source
    - edge_put with absent source → rejected
    - edge_put with absent target → accepted (lazy)
    - edge κ-label uses source's σ-axis
**Composition vectors (Level 4):**
    Tests:
    - g2(κ_A, κ_B) == g2(κ_B, κ_A) (commutative)
    - composed blob stored and retrievable
    - composed-of edges created
    - witness-of edge created
    - cross-axis operands → 422 rejected
    - edge metadata includes operation type
**GC vectors (Level 5):**
    Tests:
    - pinned + owned → both survive sweep
    - unpinned unreachable → evicted after sweep
    - tagged → survives sweep
    - untagged → evicted after sweep
    - finalizer blocks eviction
    - finalizer release → next sweep evicts
**Conflict vectors (Level 5):**
    Tests:
    - site A tags κ₁ at T1, site B tags κ₂ at T2, T2 > T1
      → all sites converge to κ₂
    - T1 == T2 → converge to max(κ₁, κ₂) lexicographically
    - both κ₁ and κ₂ remain as blobs
---
## Appendix A: Edge Blob Canonical Form
    field           encoding
    -----           --------
    source          κ-label as UTF-8 bytes
    separator       0x00
    relation        UTF-8 bytes, NFC-normalized
    separator       0x00
    target          κ-label as UTF-8 bytes
    separator       0x00
    metadata_len    u32 big-endian
    metadata        CBOR deterministic (RFC 8949 §4.2) with
                    NFC normalization of all string values
Null byte separators are unambiguous: κ-labels and relation
strings contain no null bytes.
NFC normalization of relation and metadata strings ensures
that semantically identical edges with different Unicode
normalization forms produce the same canonical bytes and
therefore the same κ-label.
The edge blob's κ-label is computed under the source's σ-axis.
---
## Appendix B: Composition Canonical Forms
### B.1 CS-G2 — Commutative Binary Product
    input:   κ-labels A, B (same axis, equal width N)
    canon:   if A ≤ B: A || B, else B || A
    output:  hash(canon) under shared axis
    property: g2(A, B) == g2(B, A)
### B.2 CS-F4 — Involution Quotient
    input:   κ-label A
    canon:   decode digest, compute bitwise complement,
             emit axis + lex-min(digest, complement)
    property: f4(A) == f4(mirror(A))
### B.3 CS-E6 — Degree Partition
    input:   κ-label A
    canon:   tag = 0x05 if first_byte % 9 ∈ [0,7], else 0x06
             [tag] || A bytes
    property: 8:1 population ratio
### B.4 CS-E7 — S₄ Orbit
    input:   κ-label A (digest divisible by 4)
    canon:   divide digest into 4 quarters, enumerate 24 S₄
             permutations, emit axis + lex-min permutation
    property: quarter permutation invariance
### B.5 CS-E8 — Identity Embedding
    input:   κ-label A
    canon:   A bytes (identity)
    property: distinguished from operand by operation type
### B.6 F4/E7/E8 Digest Coincidence
An operand already in canonical form under F4 (raw ≤ complement)
or E7 (already lex-min of orbit) produces the same digest as E8.
The κ-labels are distinguished by the operation type recorded in
the "composed-of" edge metadata, not by the digest. Consumers
MUST use the edge metadata to determine which operation produced
the composed κ-label.
---
## Appendix C: Witness Blob Format
A witness blob begins with a self-describing header:
    offset 0:  label_width      (u16 LE, e.g. 71)
    offset 2:  fingerprint_width (u16 LE, e.g. 32)
    offset 4:  trace_event_count (u16 LE)
    offset 6:  trace_data       (variable, implementation-defined)
The header enables consumers to determine the witness's parametric
type (label width N, fingerprint width FP) without external
metadata.
Witness verification: replay the trace through the derivation
pipeline, re-derive the fingerprint, compare against the stored
fingerprint. If they match, the κ-label derivation is confirmed
without re-hashing the original content.
---
## Appendix D: Schema Self-Hosting
The schema for schemas is a blob at:
    _system/schemas/v1:definition@latest
    {
        "type": "object",
        "required": ["scope", "format", "validation"],
        "properties": {
            "scope": {"type": "string"},
            "format": {
                "type": "string",
                "enum": ["openapi-v3","json-schema","wasm","native"]
            },
            "validation": {}
        }
    }
This schema has a κ-label. It validates itself.
---
## Appendix E: CLI Experience Reference (Informative)
    $ echo "hello" | kap push atlas.cern.ch/data/v1:test
    sha256:2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824
    $ kap tag atlas.cern.ch/data/v1:test@v1.0 sha256:2cf24...
    $ kap pull atlas.cern.ch/data/v1:test@v1.0
    hello
    $ kap pull sha256:2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7...
    hello
    $ kap tags atlas.cern.ch/data/v1:test
    v1.0    sha256:2cf24...
    latest  sha256:2cf24...
    $ kap edge create sha256:aaa... derives-from sha256:bbb...
    edge sha256:ccc...
    $ kap edges sha256:aaa... --relation derives-from
    sha256:aaa... --[derives-from]--> sha256:bbb...
    $ kap compose g2 sha256:aaa... sha256:bbb...
    composed: sha256:ddd...  witness: sha256:eee...
    $ kap gc sweep atlas.cern.ch/data/v1:test
    sweep started: 8f3a...
    $ kap schema register atlas.cern.ch/data/v1:datasets/ schema.json
    schema: sha256:fff...
    $ kap filter register atlas.cern.ch/data/v1:datasets/ filter.wasm
    filter: sha256:ggg...
---
## Appendix F: Multi-Hop Federation Worked Example
Three registries: A, B, C. A↔B connected. B↔C connected.
A↔C not connected. Content κ = sha256:xyz on C only.
    1. Client → A: GET /v2/org/v1/blobs/sha256:xyz
    2. A local → absent
    3. A → B: GET /v2/org/v1/blobs/sha256:xyz
    4. B local → absent
    5. B → C: GET /v2/org/v1/blobs/sha256:xyz
    6. C local → found, returns content
    7. B re-hashes → matches sha256:xyz → stores locally
    8. B returns to A
    9. A re-hashes → matches sha256:xyz → stores locally
    10. A returns to client
    11. Client re-hashes → matches → verified
Future requests to A or B served locally.
---
## Appendix G: Long-Duration Disconnection Worked Example
Submarine registry offline 90 days.
During disconnection:
    Day 1:  sub pushes κ₁, tags experiment/run1@latest
    Day 30: sub pushes κ₂, tags experiment/run1@latest (κ₁ → κ₂)
    Day 60: sub pushes κ₃, tags experiment/run2@latest
Shore during same period:
    Day 15: shore tags experiment/run1@latest → κ₄
    Day 45: shore tags experiment/run1@latest → κ₅
On reconnection (Day 91):
    1. Replicate tags: experiment/run1@latest:
       sub says κ₂ (Day 30), shore says κ₅ (Day 45)
       κ₅ wins (higher timestamp)
    2. Content backfill: sub fetches κ₄, κ₅ from shore
       Shore fetches κ₁, κ₂, κ₃ from sub
       All verified by re-hash
    3. Result:
       experiment/run1@latest → κ₅ (shore wins)
       experiment/run2@latest → κ₃ (no conflict)
       All κ₁-κ₅ available on both registries
       No data loss. No manual intervention.
---
## Appendix H: OCI Distribution Compatibility (Informative)
Implementors MAY provide OCI Distribution v1.1.1 compatibility.
    OCI concept              κ-Distribution equivalent
    -----------              -------------------------
    digest (sha256:...)      κ-label (identical for sha256)
    blob                     blob
    manifest                 blob (semantic, not structural)
    tag                      tag
    referrers API            edge query (refers-to, inbound)
    catalog                  discovery endpoint
OCI namespace mapping:
    OCI: /v2/{name}/blobs/{digest}
    κ:   /v2/_oci/{name}/blobs/{κ-label}
`_oci` is a reserved org-hierarchy prefix for OCI-native content.
SHA-256 OCI digests are byte-identical to sha256 κ-labels.
OCI referrers responses are assembled from edge queries into
the OCI Image Index format.
---
## Appendix I: Deployment Topology Patterns (Informative)
**Single node, any backend:**
    client ←→ registry ←→ storage
No consensus needed. Single-node mutex suffices.
**Multi-instance cluster:**
    clients ←→ load balancer ←→ instances ←→ distributed store
                                    ↕
                               consensus
**Multi-site, async replication:**
    site A ←→ site B ←→ site C
Conflict resolution on reconnect. Federation fetch on demand.
**Hub and spoke:**
    edge₁ ←→ hub ←→ edge₂
Writes at hub. Reads at nearest edge.
**Partial mesh:**
    A ←→ B ←→ C
         ↕
    D ←→ E
Multi-hop federation for A→C through B.
**Hybrid cloud-edge:**
    cloud (durable) ←→ edge₁ (intermittent) ←→ edge₂ (offline months)
---
## Appendix J: Prior Art (Informative)
**OCI Distribution:** κ-Distribution generalizes OCI's
blob+manifest+tag model. OCI clients work through compatibility
mapping (Appendix H).
**IPFS/IPLD:** Shares content-addressing. Differs in tag model
(mutable with consensus), edge model (typed, κ-labeled), and GC
(pin-based reachability).
**Git:** Content-addressed with four object types. κ-Distribution
has one (blob) with relationships expressed as edges.
**Nix store:** Input-addressed (derivation hash). κ-Distribution
is content-addressed (output hash). GC models parallel (roots +
reachability).
**etcd:** κ-Distribution's tags provide equivalent KV semantics.
Content storage separates from consensus, scaling independently.
**PURL:** The κ-address convention is informed by PURL structure
but extends it with DNS-style org hierarchy for ownership
delegation.
---
## Appendix K: Versioning and Evolution
**Version format:** `<major>.<minor>.<patch>`
Patch: bug fixes, no behavioral changes. Minor: additive features,
backwards compatible. Major: breaking changes.
Major releases require ≥3 release candidates spaced ≥1 week.
**Extension mechanism:** §6.15.
**κ-label stability:** The κ-label format is fixed across all
versions. A κ-label computed by any version is identical to the
same κ-label computed by any other version. Content addressing
is the invariant that never changes.
---
## Appendix L: Embedded and Constrained Registries (Informative)
κ-Distribution registries MAY run on embedded devices, IoT
sensors, and constrained environments. Implementors should note
that κ-label computation via canonical-form hashing may have
platform constraints:
Realizations requiring heap allocation (JSON with key sorting,
XML, CBOR, GGUF, ONNX) require an allocator.
Realizations operating without heap (S-expression, ASN.1 subset,
ring element, code AST) are suitable for bare-metal and
no_alloc targets.
Constrained registries MAY implement only Level 1 conformance
(blob operations) and serve as leaf nodes in a federation
topology, relying on upstream registries for tags, edges, and
GC.

---

## Appendix M: MUST / SHOULD / MAY Summary
### M.1 MUST (Hard Requirements)
**Content:**
- MUST verify κ-label on put (server-side re-hash)
- MUST return X-Kappa-Label header on all blob responses
- MUST return Content-Length on HEAD
- MUST store blobs as exact bytes provided (no normalization)
- MUST NOT return content that does not match its κ-label
- MUST reject put with mismatched κ-label
**Tags:**
- MUST NOT bind tag to absent κ-label
- MUST linearize tag_set within consensus domain
- MUST return tag list in ASCIIbetical order
- MUST NOT include Link header when n=0 on tag list
- MUST use last as cursor value, not numerical index
**Edges:**
- MUST verify source κ-label exists on edge_put
- MUST accept absent target on edge_put
- MUST use source σ-axis for edge κ-label
- MUST require σ-axis homogeneity in composition
**Upload:**
- MUST return opaque Location on upload POST
- MUST return 416 on out-of-order chunk
- MUST return 204 with Range on upload GET (recovery)
- MUST verify full-content κ-label on upload PUT
**GC:**
- MUST honor pins and finalizers during sweep
- MUST snapshot roots at sweep start (isolation)
- MUST NOT block reads or writes during sweep
**Protocol:**
- MUST NOT forward Authorization across host boundaries
- MUST return errors in the defined JSON format
- MUST use uppercase-plus-underscore error codes
### M.2 SHOULD (Strong Recommendations)
- SHOULD support Range requests on blob GET
- SHOULD support at least 10 labels in multi-label push
- SHOULD support at least 10 tags in multi-tag bind
- SHOULD include Retry-After on 429 responses
- SHOULD include X-Kappa-Chunk-Min-Length if minimum applies
- SHOULD follow redirects
- SHOULD verify response κ-label by re-hash (clients)
- SHOULD implement fallback paths for optional features
- SHOULD rate-limit pin creation per identity
- SHOULD use 16 KiB buffer for streaming hash
- SHOULD support chunked upload recovery (GET on session)
- SHOULD report Warning headers to users
- SHOULD evaluate information disclosure risk of mount search
### M.3 MUST NOT (Hard Prohibitions)
- MUST NOT return content that does not match its κ-label
- MUST NOT normalize, re-encode, or strip fields from blobs
- MUST NOT bind tags to absent κ-labels
- MUST NOT compose κ-labels across σ-axes
- MUST NOT forward Authorization across host boundaries
- MUST NOT send more than 4096 bytes of Warning data
- MUST NOT take automated action based on Warning headers
- MUST NOT disclose existence of unauthorized content
- MUST NOT block reads or writes during GC sweep
### M.4 MAY (Permitted Optionals)
- MAY redirect any request per RFC 9110 §15.4
- MAY implement deletion or disable it (405 response)
- MAY support monolithic PUT without chunked upload
- MAY support multi-label push
- MAY support multi-tag bind
- MAY support cross-namespace mount
- MAY support source-omitted mount with risk evaluation
- MAY include Warning headers for non-fatal issues
- MAY implement federation fetch
- MAY implement multi-hop relay
- MAY implement cross-axis on-demand enrichment
- MAY implement admission filters
- MAY produce tag snapshots automatically
- MAY use automated finalizer TTL for stuck finalizer recovery

---

## Appendix N: References
### N.1 Normative References
    [RFC2119]   Bradner, S., "Key words for use in RFCs to
                Indicate Requirement Levels", BCP 14, RFC 2119,
                March 1997.
    [RFC6585]   Nottingham, M., Fielding, R., "Additional HTTP
                Status Codes", RFC 6585, April 2012.
    [RFC8949]   Bormann, C., Hoffman, P., "Concise Binary Object
                Representation (CBOR)", RFC 8949, December 2020.
    [RFC9110]   Fielding, R., et al., "HTTP Semantics", RFC 9110,
                June 2022.
    [RFC9111]   Fielding, R., et al., "HTTP Caching", RFC 9111,
                June 2022.
    [UAX15]     Unicode Consortium, "Unicode Normalization Forms",
                Unicode Standard Annex #15.
    [FIPS180-4] NIST, "Secure Hash Standard (SHS)", FIPS PUB
                180-4, August 2015.
    [FIPS202]   NIST, "SHA-3 Standard", FIPS PUB 202, August 2015.
    [BLAKE3]    O'Connor, J., et al., "BLAKE3: One function, fast
                everywhere", 2020.
### N.2 Informative References
    [OCI-DIST]  Open Container Initiative, "OCI Distribution
                Specification v1.1.1", 2024.
    [PURL]      Package URL Project, "purl-spec",
                github.com/package-url/purl-spec.
    [IPFS]      Benet, J., "IPFS - Content Addressed, Versioned,
                P2P File System", arXiv:1407.3561, 2014.
    [KINE]      Rancher Labs, "kine", github.com/k3s-io/kine.
    [SCITT]     IETF, "Supply Chain Integrity, Transparency, and
                Trust", draft-ietf-scitt-architecture.
