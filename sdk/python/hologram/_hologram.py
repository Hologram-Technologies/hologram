from __future__ import annotations

import ctypes
import os
import platform
import struct
from pathlib import Path
from typing import Sequence

from ._generated import REQUIRED_FEATURES, f32
from .errors import (
    ABI_MISMATCH,
    ARCHIVE_LOAD,
    BAD_ATTR,
    EXECUTION,
    EXTERNAL_TENSOR,
    INVALID_ARGUMENT,
    AbiMismatchError,
    ExternalTensorError,
    InvalidArgumentError,
    NativeError,
    error_from_code,
)

ABI_VERSION = 1
ARCHIVE_FORMAT_VERSION = 2
INITIAL_ARCHIVE_CAPACITY = 16 * 1024


class HologramString(ctypes.Structure):
    _fields_ = [("ptr", ctypes.POINTER(ctypes.c_uint8)), ("len", ctypes.c_size_t)]


class HologramShape(ctypes.Structure):
    _fields_ = [("dims", ctypes.POINTER(ctypes.c_uint64)), ("rank", ctypes.c_size_t)]


class HologramTensorDesc(ctypes.Structure):
    _fields_ = [("name", HologramString), ("dtype_id", ctypes.c_uint8), ("shape", HologramShape)]


class HologramConstDesc(ctypes.Structure):
    _fields_ = [
        ("tensor", HologramTensorDesc),
        ("bytes", ctypes.POINTER(ctypes.c_uint8)),
        ("byte_len", ctypes.c_size_t),
    ]


class HologramExternalTensorDesc(ctypes.Structure):
    _fields_ = [
        ("tensor", HologramTensorDesc),
        ("path", HologramString),
        ("byte_offset", ctypes.c_uint64),
        ("byte_len", ctypes.c_uint64),
        ("content_hash", ctypes.c_uint8 * 32),
    ]


class HologramSourceOp(ctypes.Structure):
    _fields_ = [
        ("output", HologramString),
        ("op", HologramString),
        ("inputs", ctypes.POINTER(HologramString)),
        ("input_count", ctypes.c_size_t),
        ("shape", HologramShape),
    ]


_LIB = None


def abi_version() -> int:
    return int(_lib().hologram_abi_version())


def archive_format_version() -> int:
    return int(_lib().hologram_archive_format_version())


def feature_supported(feature: str) -> bool:
    feature_str, _feature_bytes = _string(feature)
    result = int(_lib().hologram_feature_supported(feature_str))
    if result < 0:
        raise _last_error("feature_supported")
    return result == 1


def last_error_code() -> int:
    return int(_lib().hologram_last_error_code())


def last_error_message() -> str | None:
    message = _lib().hologram_last_error_message()
    return None if message is None else message.decode("utf-8", "replace")


def last_error_line() -> int | None:
    line = int(_lib().hologram_last_error_line())
    return line or None


def last_error_column() -> int | None:
    column = int(_lib().hologram_last_error_column())
    return column or None


def last_error_rejected() -> str | None:
    rejected = _lib().hologram_last_error_rejected()
    return None if rejected is None else rejected.decode("utf-8", "replace")


def assert_compatible() -> None:
    if abi_version() != ABI_VERSION:
        raise AbiMismatchError(f"unsupported Hologram ABI {abi_version()}")
    if archive_format_version() != ARCHIVE_FORMAT_VERSION:
        raise AbiMismatchError(f"unsupported Hologram archive format {archive_format_version()}")
    missing = [feature for feature in REQUIRED_FEATURES if not feature_supported(feature)]
    if missing:
        raise AbiMismatchError(f"native binding missing features: {', '.join(missing)}")


def source_builder() -> "SourceBuilder":
    assert_compatible()
    return SourceBuilder()


def compile_source(source: str | bytes | bytearray | memoryview) -> bytes:
    assert_compatible()
    return _compile_source(_source_bytes(source))


def compile_source_file(path: str | os.PathLike[str]) -> bytes:
    return compile_source(Path(path).read_bytes())


def session_load(archive: bytes | bytearray | memoryview) -> "Session":
    return Session.load(archive)


class SourceBuilder:
    def __init__(self):
        self._ptr = _lib().hologram_source_builder_new()
        if not self._ptr:
            raise _last_error("source_builder_new")

    def input(self, name: str, *, dtype: int = f32, shape: Sequence[int] | None = None) -> str:
        desc, _refs = _tensor_desc(name, dtype, shape)
        _check(_lib().hologram_source_builder_input(self._ptr, ctypes.byref(desc)), "input")
        return name

    def const(self, name: str, *, dtype: int = f32, shape: Sequence[int], values: Sequence[float]) -> str:
        desc, refs = _tensor_desc(name, dtype, shape)
        data = _f32_bytes(values)
        data_ptr = _bytes_ptr(data)
        refs.append(data_ptr)
        const = HologramConstDesc(desc, data_ptr, len(data))
        _check(_lib().hologram_source_builder_const(self._ptr, ctypes.byref(const)), "const")
        return name

    def const_ref(
        self,
        name: str,
        *,
        dtype: int = f32,
        shape: Sequence[int],
        file: str,
        blake3: str,
        byte_len: int | None = None,
        byte_offset: int = 0,
    ) -> str:
        desc, refs = _tensor_desc(name, dtype, shape)
        path, path_bytes = _string(file)
        refs.append(path_bytes)
        content_hash = _content_hash(blake3)
        ref = HologramExternalTensorDesc(desc, path, byte_offset, _byte_len(shape, byte_len), content_hash)
        _check(_lib().hologram_source_builder_const_ref(self._ptr, ctypes.byref(ref)), "const_ref")
        return name

    def op(self, op: str, inputs: Sequence[str], *, as_: str, **attrs) -> str:
        shape = attrs.pop("shape", None)
        if attrs:
            raise error_from_code(BAD_ATTR, f"Python builder does not support op attrs: {', '.join(sorted(attrs))}")
        call, _refs = _source_op(as_, op, inputs, shape)
        _check(_lib().hologram_source_builder_op(self._ptr, ctypes.byref(call)), "op")
        return as_

    def output(self, name: str, *, source: str | None = None) -> None:
        actual = name if source is None else source
        name_str, _name_bytes = _string(name)
        if actual == name:
            _check(_lib().hologram_source_builder_output(self._ptr, name_str), "output")
            return
        source_str, _source_bytes = _string(actual)
        _check(_lib().hologram_source_builder_output_alias(self._ptr, name_str, source_str), "output")

    def compile(self) -> bytes:
        try:
            return _compile(self._ptr)
        finally:
            self.free()

    def free(self) -> None:
        if self._ptr:
            _lib().hologram_source_builder_free(self._ptr)
            self._ptr = None

    def __del__(self):
        self.free()


class Session:
    def __init__(self, handle: int):
        self._handle = handle

    @classmethod
    def load(cls, archive: bytes | bytearray | memoryview) -> "Session":
        assert_compatible()
        data = bytes(archive)
        archive_buf = _bytes_ptr(data)
        handle = int(_lib().hologram_session_load(archive_buf, len(data)))
        if handle < 0:
            raise _native_error(ARCHIVE_LOAD, "session load failed")
        return cls(handle)

    @property
    def input_count(self) -> int:
        return _session_int(self._handle, _lib().hologram_session_input_count, "input_count")

    @property
    def output_count(self) -> int:
        return _session_int(self._handle, _lib().hologram_session_output_count, "output_count")

    @property
    def kernel_count(self) -> int:
        return _session_int(self._handle, _lib().hologram_session_kernel_count, "kernel_count")

    @property
    def archive_fingerprint(self) -> bytes:
        self._require_open()
        out = (ctypes.c_uint8 * 32)()
        _check(_lib().hologram_session_archive_fingerprint(self._handle, out), "archive_fingerprint")
        return bytes(out)

    def input_name(self, index: int) -> str:
        return _session_string(self._handle, index, _lib().hologram_session_input_name, "input_name")

    def output_name(self, index: int) -> str:
        return _session_string(self._handle, index, _lib().hologram_session_output_name, "output_name")

    def input_shape(self, index: int) -> tuple[int, ...]:
        return _session_shape(self._handle, index, _lib().hologram_session_input_shape, "input_shape")

    def output_shape(self, index: int) -> tuple[int, ...]:
        return _session_shape(self._handle, index, _lib().hologram_session_output_shape, "output_shape")

    def output_byte_len(self, index: int) -> int:
        result = int(_lib().hologram_session_output_byte_len(self._handle, index))
        if result < 0:
            raise _native_error(INVALID_ARGUMENT, "output_byte_len failed")
        return result

    def input_dtype(self, index: int) -> int:
        return _session_int_index(self._handle, index, _lib().hologram_session_input_dtype, "input_dtype")

    def output_dtype(self, index: int) -> int:
        return _session_int_index(self._handle, index, _lib().hologram_session_output_dtype, "output_dtype")

    def extension(self, key: str) -> bytes | None:
        self._require_open()
        return _session_extension(self._handle, key)

    def execute(self, inputs) -> dict[str, bytes]:
        ordered = self._ordered_inputs(inputs)
        return _execute_session(self._handle, ordered, self._output_names())

    def close(self) -> None:
        if self._handle is None:
            return
        handle = self._handle
        self._handle = None
        _check(_lib().hologram_session_close(handle), "session_close")

    def _ordered_inputs(self, inputs) -> list[bytes]:
        if _is_bytes_like(inputs):
            return _checked_input_list([bytes(inputs)], self.input_count)
        if hasattr(inputs, "keys"):
            return [_mapping_input(inputs, self.input_name(i)) for i in range(self.input_count)]
        return _checked_input_list([bytes(value) for value in inputs], self.input_count)

    def _output_names(self) -> list[str]:
        return [self.output_name(i) or str(i) for i in range(self.output_count)]

    def _require_open(self) -> None:
        if self._handle is None:
            raise InvalidArgumentError("session is closed")

    def __enter__(self) -> "Session":
        self._require_open()
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _compile(builder) -> bytes:
    capacity = INITIAL_ARCHIVE_CAPACITY
    while True:
        out = (ctypes.c_uint8 * capacity)()
        required = int(_lib().hologram_source_builder_compile(builder, out, capacity))
        if required < 0:
            raise _last_error("compile")
        if required <= capacity:
            return bytes(out[:required])
        capacity = required


def _compile_source(data: bytes) -> bytes:
    source = _bytes_ptr(data)
    capacity = INITIAL_ARCHIVE_CAPACITY
    while True:
        out = (ctypes.c_uint8 * capacity)()
        required = int(_lib().hologram_compile_source(source, len(data), out, capacity))
        if required < 0:
            raise _last_error("compile_source")
        if required <= capacity:
            return bytes(out[:required])
        capacity = required


def _source_bytes(source: str | bytes | bytearray | memoryview) -> bytes:
    return source.encode("utf-8") if isinstance(source, str) else bytes(source)


def _tensor_desc(name: str, dtype: int, shape: Sequence[int] | None):
    name_str, name_bytes = _string(name)
    shape_desc, shape_array = _shape(shape)
    return HologramTensorDesc(name_str, dtype, shape_desc), [name_bytes, shape_array]


def _source_op(output: str, op: str, inputs: Sequence[str], shape: Sequence[int] | None):
    output_str, output_bytes = _string(output)
    op_str, op_bytes = _string(op)
    input_refs = [_string(input) for input in inputs]
    input_array = (HologramString * len(input_refs))(*(item[0] for item in input_refs))
    shape_desc, shape_array = _shape(shape)
    refs = [output_bytes, op_bytes, input_array, shape_array, *(item[1] for item in input_refs)]
    return HologramSourceOp(output_str, op_str, input_array, len(input_refs), shape_desc), refs


def _shape(shape: Sequence[int] | None):
    dims = () if shape is None else tuple(int(dim) for dim in shape)
    if len(dims) == 0:
        return HologramShape(None, 0), None
    array = (ctypes.c_uint64 * len(dims))(*dims)
    return HologramShape(array, len(dims)), array


def _string(value: str):
    data = value.encode("utf-8")
    array = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
    return HologramString(array, len(data)), array


def _f32_bytes(values: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def _bytes_ptr(data: bytes):
    return (ctypes.c_uint8 * len(data)).from_buffer_copy(data)


def _byte_len(shape: Sequence[int], explicit: int | None) -> int:
    if explicit is not None:
        return int(explicit)
    total = 4
    for dim in shape:
        total *= int(dim)
    return total


def _content_hash(blake3: str):
    try:
        digest = bytes.fromhex(blake3)
    except ValueError as exc:
        raise ExternalTensorError("const_ref blake3 must be hex") from exc
    if len(digest) != 32:
        raise ExternalTensorError("const_ref blake3 must be 32 bytes")
    return (ctypes.c_uint8 * 32).from_buffer_copy(digest)


def _check(result: int, context: str) -> None:
    if int(result) < 0:
        raise _last_error(context)


def _last_error(context: str) -> NativeError:
    return error_from_code(
        last_error_code(),
        last_error_message() or context,
        line=last_error_line(),
        column=last_error_column(),
        rejected=last_error_rejected(),
    )


def _native_error(code: int, fallback: str) -> NativeError:
    message = last_error_message()
    native_code = last_error_code()
    return error_from_code(
        native_code or code,
        message or fallback,
        line=last_error_line(),
        column=last_error_column(),
        rejected=last_error_rejected(),
    )


def _session_int(handle: int | None, func, context: str) -> int:
    if handle is None:
        raise InvalidArgumentError("session is closed")
    result = int(func(handle))
    if result < 0:
        raise _native_error(INVALID_ARGUMENT, context)
    return result


def _session_int_index(handle: int | None, index: int, func, context: str) -> int:
    if handle is None:
        raise InvalidArgumentError("session is closed")
    result = int(func(handle, index))
    if result < 0:
        raise _native_error(INVALID_ARGUMENT, context)
    return result


def _session_string(handle: int | None, index: int, func, context: str) -> str:
    if handle is None:
        raise InvalidArgumentError("session is closed")
    capacity = 64
    while True:
        out = (ctypes.c_uint8 * capacity)()
        required = int(func(handle, index, out, capacity))
        if required < 0:
            raise _native_error(INVALID_ARGUMENT, context)
        if required <= capacity:
            return bytes(out[:required]).decode("utf-8", "replace")
        capacity = required


def _session_shape(handle: int | None, index: int, func, context: str) -> tuple[int, ...]:
    if handle is None:
        raise InvalidArgumentError("session is closed")
    capacity = 8
    while True:
        out = (ctypes.c_uint64 * capacity)()
        required = int(func(handle, index, out, capacity))
        if required < 0:
            raise _native_error(INVALID_ARGUMENT, context)
        if required <= capacity:
            return tuple(int(out[i]) for i in range(required))
        capacity = required


def _session_extension(handle: int | None, key: str) -> bytes | None:
    if handle is None:
        raise InvalidArgumentError("session is closed")
    key_str, key_bytes = _string(key)
    capacity = 256
    while True:
        out = (ctypes.c_uint8 * capacity)()
        required = int(_lib().hologram_session_extension(handle, key_str.ptr, key_str.len, out, capacity))
        if required < 0:
            return None
        if required <= capacity:
            return bytes(out[:required])
        capacity = required


def _execute_session(handle: int | None, inputs: list[bytes], output_names: list[str]) -> dict[str, bytes]:
    if handle is None:
        raise InvalidArgumentError("session is closed")
    input_buffers = [_bytes_ptr(data) for data in inputs]
    output_buffers = [_output_buffer(handle, i) for i in range(len(output_names))]
    result = _lib().hologram_session_execute(
        handle,
        _pointer_array(input_buffers),
        _size_array([len(data) for data in inputs]),
        len(inputs),
        _pointer_array(output_buffers),
        _size_array([len(buf) for buf in output_buffers]),
        len(output_buffers),
    )
    if int(result) < 0:
        raise _native_error(EXECUTION, "session execute failed")
    return dict(zip(output_names, (bytes(buf) for buf in output_buffers), strict=True))


def _mapping_input(inputs, name: str) -> bytes:
    if name not in inputs:
        raise InvalidArgumentError(f"missing input: {name}")
    return bytes(inputs[name])


def _checked_input_list(values: list[bytes], expected: int) -> list[bytes]:
    if len(values) != expected:
        raise InvalidArgumentError("input count mismatch")
    return values


def _is_bytes_like(value) -> bool:
    return isinstance(value, (bytes, bytearray, memoryview))


def _output_buffer(handle: int, index: int):
    length = int(_lib().hologram_session_output_byte_len(handle, index))
    if length < 0:
        raise _native_error(INVALID_ARGUMENT, "output_byte_len failed")
    return (ctypes.c_uint8 * length)()


def _pointer_array(buffers):
    if not buffers:
        return None
    ptrs = [ctypes.cast(buffer, ctypes.POINTER(ctypes.c_uint8)) for buffer in buffers]
    return (ctypes.POINTER(ctypes.c_uint8) * len(ptrs))(*ptrs)


def _size_array(values: Sequence[int]):
    if not values:
        return None
    return (ctypes.c_size_t * len(values))(*(int(value) for value in values))


def _lib():
    global _LIB
    if _LIB is None:
        _LIB = ctypes.CDLL(str(_library_path()))
        _configure(_LIB)
    return _LIB


def _configure(lib) -> None:
    lib.hologram_abi_version.restype = ctypes.c_uint32
    lib.hologram_archive_format_version.restype = ctypes.c_uint32
    lib.hologram_feature_supported.argtypes = [HologramString]
    lib.hologram_feature_supported.restype = ctypes.c_int32
    lib.hologram_last_error_code.restype = ctypes.c_int32
    lib.hologram_last_error_message.restype = ctypes.c_char_p
    lib.hologram_last_error_line.restype = ctypes.c_size_t
    lib.hologram_last_error_column.restype = ctypes.c_size_t
    lib.hologram_last_error_rejected.restype = ctypes.c_char_p
    lib.hologram_source_builder_new.restype = ctypes.c_void_p
    lib.hologram_source_builder_free.argtypes = [ctypes.c_void_p]
    lib.hologram_source_builder_free.restype = None
    lib.hologram_source_builder_input.argtypes = [ctypes.c_void_p, ctypes.POINTER(HologramTensorDesc)]
    lib.hologram_source_builder_input.restype = ctypes.c_int32
    lib.hologram_source_builder_const.argtypes = [ctypes.c_void_p, ctypes.POINTER(HologramConstDesc)]
    lib.hologram_source_builder_const.restype = ctypes.c_int32
    lib.hologram_source_builder_const_ref.argtypes = [ctypes.c_void_p, ctypes.POINTER(HologramExternalTensorDesc)]
    lib.hologram_source_builder_const_ref.restype = ctypes.c_int32
    lib.hologram_source_builder_op.argtypes = [ctypes.c_void_p, ctypes.POINTER(HologramSourceOp)]
    lib.hologram_source_builder_op.restype = ctypes.c_int32
    lib.hologram_source_builder_output.argtypes = [ctypes.c_void_p, HologramString]
    lib.hologram_source_builder_output.restype = ctypes.c_int32
    lib.hologram_source_builder_output_alias.argtypes = [ctypes.c_void_p, HologramString, HologramString]
    lib.hologram_source_builder_output_alias.restype = ctypes.c_int32
    lib.hologram_source_builder_compile.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
    lib.hologram_source_builder_compile.restype = ctypes.c_int32
    lib.hologram_compile_source.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
    ]
    lib.hologram_compile_source.restype = ctypes.c_int32
    lib.hologram_session_load.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
    lib.hologram_session_load.restype = ctypes.c_int32
    lib.hologram_session_input_count.argtypes = [ctypes.c_int32]
    lib.hologram_session_input_count.restype = ctypes.c_int32
    lib.hologram_session_output_count.argtypes = [ctypes.c_int32]
    lib.hologram_session_output_count.restype = ctypes.c_int32
    lib.hologram_session_kernel_count.argtypes = [ctypes.c_int32]
    lib.hologram_session_kernel_count.restype = ctypes.c_int32
    lib.hologram_session_output_byte_len.argtypes = [ctypes.c_int32, ctypes.c_size_t]
    lib.hologram_session_output_byte_len.restype = ctypes.c_int32
    lib.hologram_session_input_dtype.argtypes = [ctypes.c_int32, ctypes.c_size_t]
    lib.hologram_session_input_dtype.restype = ctypes.c_int32
    lib.hologram_session_output_dtype.argtypes = [ctypes.c_int32, ctypes.c_size_t]
    lib.hologram_session_output_dtype.restype = ctypes.c_int32
    lib.hologram_session_archive_fingerprint.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_uint8)]
    lib.hologram_session_archive_fingerprint.restype = ctypes.c_int32
    byte_ptr = ctypes.POINTER(ctypes.c_uint8)
    lib.hologram_session_execute.argtypes = [
        ctypes.c_int32,
        ctypes.POINTER(byte_ptr),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
        ctypes.POINTER(byte_ptr),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
    ]
    lib.hologram_session_execute.restype = ctypes.c_int32
    lib.hologram_session_close.argtypes = [ctypes.c_int32]
    lib.hologram_session_close.restype = ctypes.c_int32
    lib.hologram_session_input_name.argtypes = [ctypes.c_int32, ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
    lib.hologram_session_input_name.restype = ctypes.c_int32
    lib.hologram_session_output_name.argtypes = [ctypes.c_int32, ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
    lib.hologram_session_output_name.restype = ctypes.c_int32
    lib.hologram_session_input_shape.argtypes = [ctypes.c_int32, ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
    lib.hologram_session_input_shape.restype = ctypes.c_int32
    lib.hologram_session_output_shape.argtypes = [ctypes.c_int32, ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
    lib.hologram_session_output_shape.restype = ctypes.c_int32
    lib.hologram_session_extension.argtypes = [
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
    ]
    lib.hologram_session_extension.restype = ctypes.c_int32


def _library_path() -> Path:
    env_path = os.environ.get("HOLOGRAM_FFI_LIBRARY")
    if env_path:
        return Path(env_path)
    for path in _library_candidates():
        if path.exists():
            return path
    raise NativeError(0, "unable to load bundled hologram native library")


def _library_candidates():
    here = Path(__file__).resolve().parent
    yield here / python_library_name()
    for parent in here.parents:
        yield parent / "target" / "release" / rust_library_name()


def rust_library_name() -> str:
    system = platform.system()
    if system == "Darwin":
        return "libhologram_ffi.dylib"
    if system == "Windows":
        return "hologram_ffi.dll"
    return "libhologram_ffi.so"


def python_library_name() -> str:
    system = platform.system()
    if system == "Darwin":
        return "_hologram_ffi.dylib"
    if system == "Windows":
        return "_hologram_ffi.dll"
    return "_hologram_ffi.so"
