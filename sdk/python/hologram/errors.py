from __future__ import annotations

PARSE = 1
GRAPH = 2
UNSUPPORTED_OP = 3
BAD_ATTR = 4
SHAPE = 5
EXTERNAL_TENSOR = 6
ARCHIVE_LOAD = 7
EXECUTION = 8
ABI_MISMATCH = 9
INVALID_ARGUMENT = 10
UNSUPPORTED_DTYPE = 11
COMPILE = 12


class HologramError(RuntimeError):
    def __init__(
        self,
        message: str,
        code: int | None = None,
        *,
        line: int | None = None,
        column: int | None = None,
        rejected: str | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.native_message = message
        self.line = line
        self.column = column
        self.rejected = rejected


class HologramNativeUnavailable(HologramError):
    pass


class NativeError(HologramError):
    DEFAULT_CODE = 0

    def __init__(
        self,
        code: int | str,
        message: str | None = None,
        *,
        line: int | None = None,
        column: int | None = None,
        rejected: str | None = None,
    ):
        if message is None:
            message = str(code)
            code = self.DEFAULT_CODE
        super().__init__(message, code, line=line, column=column, rejected=rejected)


class ParseError(NativeError):
    DEFAULT_CODE = PARSE


class GraphError(NativeError):
    DEFAULT_CODE = GRAPH


class UnsupportedOpError(NativeError):
    DEFAULT_CODE = UNSUPPORTED_OP


class UnknownOpError(UnsupportedOpError):
    pass


class BadAttrError(NativeError):
    DEFAULT_CODE = BAD_ATTR


class ShapeError(NativeError):
    DEFAULT_CODE = SHAPE


class ExternalTensorError(NativeError):
    DEFAULT_CODE = EXTERNAL_TENSOR


class ArchiveLoadError(NativeError):
    DEFAULT_CODE = ARCHIVE_LOAD


class ExecutionError(NativeError):
    DEFAULT_CODE = EXECUTION


class AbiMismatchError(NativeError):
    DEFAULT_CODE = ABI_MISMATCH


class InvalidArgumentError(NativeError):
    DEFAULT_CODE = INVALID_ARGUMENT


class UnsupportedDTypeError(NativeError):
    DEFAULT_CODE = UNSUPPORTED_DTYPE


class CompileError(NativeError):
    DEFAULT_CODE = COMPILE


def error_from_code(
    code: int,
    message: str,
    *,
    line: int | None = None,
    column: int | None = None,
    rejected: str | None = None,
) -> NativeError:
    kind = _ERROR_TYPES.get(code, NativeError)
    return kind(code, message, line=line, column=column, rejected=rejected)


_ERROR_TYPES = {
    PARSE: ParseError,
    GRAPH: GraphError,
    UNSUPPORTED_OP: UnsupportedOpError,
    BAD_ATTR: BadAttrError,
    SHAPE: ShapeError,
    EXTERNAL_TENSOR: ExternalTensorError,
    ARCHIVE_LOAD: ArchiveLoadError,
    EXECUTION: ExecutionError,
    ABI_MISMATCH: AbiMismatchError,
    INVALID_ARGUMENT: InvalidArgumentError,
    UNSUPPORTED_DTYPE: UnsupportedDTypeError,
    COMPILE: CompileError,
}
