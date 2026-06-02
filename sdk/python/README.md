# Hologram Python SDK

The Python package exposes a chainable graph builder on top of the stable
`hologram-ffi` source-builder ABI. Wheels build and bundle the Rust
`hologram-ffi` cdylib; `hologram._hologram` loads that library and implements
the SDK `NativeBinding` protocol.

```python
import hologram as hg

g = hg.Graph("encoder")
x = g.input("x", dtype=hg.f32, shape=[2, 3])
w = g.const_ref("w", dtype=hg.f32, shape=[3, 2], file="weights.bin", blake3="0" * 64)
y = x.matmul(w, shape=[2, 2]).relu()
archive = g.output("y", y).compile()

with hg.Session.load(archive) as session:
    assert session.input_dtype(0) == hg.f32
    assert session.output_dtype(0) == hg.f32
    assert session.extension("missing") is None
    outputs = session.execute({"x": input_bytes})
    y_bytes = outputs["y"]
```

Native Hologram `.txt` source can be compiled without constructing a
`Graph` object:

```python
archive = hg.compile_source("input x :1\nop relu x :1 as=y\noutput y\n")
archive = hg.compile_source_file("graph.txt")
```

Native and SDK-side failures raise typed `hg.HologramError` subclasses with a
stable `.code` field:

```python
try:
    hg.Session.load(b"not-a-holo")
except hg.ArchiveLoadError as exc:
    assert exc.code == 7
```

When the native compiler can locate source position, errors also expose
`.line`, `.column`, and `.rejected`:

```python
try:
    bad = hg.Graph("bad")
    w = bad.const_ref("w", shape=[1], file="weights.bin", blake3="not-hex")
    bad.output("w", w).compile()
except hg.ExternalTensorError as exc:
    print(exc.code, exc.line, exc.column, exc.rejected)
```

The public error classes are `ParseError`, `GraphError`,
`UnsupportedOpError`, `UnknownOpError`, `BadAttrError`, `ShapeError`,
`ExternalTensorError`, `ArchiveLoadError`, `ExecutionError`,
`AbiMismatchError`, `InvalidArgumentError`, `UnsupportedDTypeError`, and
`CompileError`.

`compile()` requires a native binding module that implements the SDK
`NativeBinding` protocol. Tests can pass a fake binding directly:

```python
archive = g.output("y", y).compile(native=my_binding)
```

`const_ref` is compile-time only. The native compiler reads the declared file
range, verifies the BLAKE3 digest, and embeds the bytes into the archive before
`Session.load(...)`. Set `HOLOGRAM_EXTERNAL_TENSOR_ROOT` to force all resolved
external tensor paths under an explicit compile root.

Build a local wheel from the repository root:

```bash
python3 -m pip wheel sdk/python --no-deps -w /tmp/hologram-wheel
python3 -m pip install /tmp/hologram-wheel/*.whl
python3 sdk/python/scripts/smoke-installed.py
```
