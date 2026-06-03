from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from ._generated import OPS, REQUIRED_FEATURES, f32, op_attrs, op_call
from .errors import AbiMismatchError, BadAttrError, GraphError, HologramNativeUnavailable, UnknownOpError

Shape = Sequence[int]
TensorLike = "Tensor | str"


class LowLevelBuilder(Protocol):
    def input(self, name: str, *, dtype: int, shape: Shape | None = None) -> str: ...

    def const(self, name: str, *, dtype: int, shape: Shape, values: Sequence[float]) -> str: ...

    def const_ref(
        self,
        name: str,
        *,
        dtype: int,
        shape: Shape,
        file: str,
        blake3: str,
        byte_len: int | None = None,
        byte_offset: int = 0,
    ) -> str: ...

    def op(self, op: str, inputs: Sequence[str], *, as_: str, **attrs: Any) -> str: ...

    def output(self, name: str, *, source: str | None = None) -> None: ...

    def compile(self) -> bytes: ...


class NativeBinding(Protocol):
    def source_builder(self) -> LowLevelBuilder: ...

    def feature_supported(self, feature: str) -> bool: ...


@dataclass(frozen=True)
class Tensor:
    graph: "Graph"
    name: str

    def op(self, op: str, *inputs: TensorLike, **attrs: Any) -> "Tensor":
        return self.graph.op(op, (self, *inputs), **attrs)

    def __getattr__(self, op: str):
        if op not in OPS:
            raise AttributeError(op)
        return lambda *inputs, **attrs: self.op(op, *inputs, **attrs)


@dataclass(frozen=True)
class _Item:
    kind: str
    name: str
    dtype: int = f32
    shape: tuple[int, ...] = ()
    values: tuple[float, ...] = ()
    inputs: tuple[str, ...] = ()
    op: str = ""
    attrs: Mapping[str, Any] = field(default_factory=dict)
    path: str = ""
    blake3: str = ""
    byte_len: int | None = None
    byte_offset: int = 0


class Graph:
    def __init__(self, name: str | None = None, native: NativeBinding | None = None):
        self.name = name
        self._native = native
        self._items: list[_Item] = []
        self._outputs: list[tuple[str, str]] = []
        self._names: set[str] = set()
        self._next_tmp = 0

    def input(self, name: str, *, dtype: int = f32, shape: Shape | None = None) -> Tensor:
        self._add(_Item("input", name, dtype=dtype, shape=_shape(shape)))
        return Tensor(self, name)

    def const(self, name: str, *, shape: Shape, values: Sequence[float], dtype: int = f32) -> Tensor:
        self._add(_Item("const", name, dtype=dtype, shape=_shape(shape), values=tuple(values)))
        return Tensor(self, name)

    def const_ref(self, name: str, *, dtype: int = f32, shape: Shape, file: str, blake3: str, byte_len: int | None = None, byte_offset: int = 0) -> Tensor:
        self._add(_Item("const_ref", name, dtype=dtype, shape=_shape(shape), path=file, blake3=blake3, byte_len=byte_len, byte_offset=byte_offset))
        return Tensor(self, name)

    def op(self, op: str, inputs: Sequence[TensorLike], *, shape: Shape | None = None, dtype: int = f32, as_: str | None = None, **attrs: Any) -> Tensor:
        _validate_op(op, attrs)
        name = as_ if as_ is not None else self._tmp()
        all_attrs = dict(attrs)
        if shape is not None:
            all_attrs["shape"] = list(shape)
        if dtype != f32:
            all_attrs["dtype"] = dtype
        self._add(_Item("op", name, dtype=dtype, shape=_shape(shape), inputs=_names(inputs), op=op, attrs=all_attrs))
        return Tensor(self, name)

    def output(self, name: str, tensor: TensorLike | None = None) -> "Graph":
        self._outputs.append((name, name if tensor is None else _name(tensor)))
        return self

    def emit(self, builder: LowLevelBuilder) -> LowLevelBuilder:
        for item in self._items:
            _emit_item(builder, item)
        for name, source in self._outputs:
            _emit_output(builder, name, source)
        return builder

    def compile(self, native: NativeBinding | None = None) -> bytes:
        binding = native or self._native or _load_native()
        _check_features(binding)
        return self.emit(binding.source_builder()).compile()

    def _add(self, item: _Item) -> None:
        if item.name in self._names:
            raise GraphError(f"duplicate tensor name: {item.name}")
        self._names.add(item.name)
        self._items.append(item)

    def _tmp(self) -> str:
        name = f"_t{self._next_tmp}"
        self._next_tmp += 1
        return name


def _emit_item(builder: LowLevelBuilder, item: _Item) -> None:
    if item.kind == "input":
        builder.input(item.name, dtype=item.dtype, shape=item.shape or None)
    elif item.kind == "const":
        builder.const(item.name, dtype=item.dtype, shape=item.shape, values=item.values)
    elif item.kind == "const_ref":
        builder.const_ref(item.name, dtype=item.dtype, shape=item.shape, file=item.path, blake3=item.blake3, byte_len=item.byte_len, byte_offset=item.byte_offset)
    else:
        op_call(builder, item.name, item.op, item.inputs, **dict(item.attrs))


def _emit_output(builder: LowLevelBuilder, name: str, source: str) -> None:
    try:
        builder.output(name, source=source)
    except TypeError as exc:
        if name != source:
            raise AbiMismatchError("low-level builder does not support output aliases") from exc
        builder.output(name)


def _validate_op(op: str, attrs: Mapping[str, Any]) -> None:
    if op not in OPS:
        raise UnknownOpError(op)
    allowed = set(op_attrs(op))
    unknown = sorted(name for name in attrs if name not in allowed)
    if unknown:
        raise BadAttrError(f"{op}: unsupported attrs: {', '.join(unknown)}")


def _check_features(binding: NativeBinding) -> None:
    missing = [feature for feature in REQUIRED_FEATURES if not binding.feature_supported(feature)]
    if missing:
        raise AbiMismatchError(f"native binding missing features: {', '.join(missing)}")


def _load_native() -> NativeBinding:
    try:
        from . import _hologram as native
    except ImportError as exc:
        raise HologramNativeUnavailable("install the native hologram package to compile graphs") from exc
    try:
        native.assert_compatible()
    except Exception as exc:
        raise HologramNativeUnavailable(str(exc)) from exc
    return native


def _shape(shape: Shape | None) -> tuple[int, ...]:
    return () if shape is None else tuple(int(dim) for dim in shape)


def _names(inputs: Sequence[TensorLike]) -> tuple[str, ...]:
    return tuple(_name(input) for input in inputs)


def _name(value: TensorLike) -> str:
    return value.name if isinstance(value, Tensor) else value
