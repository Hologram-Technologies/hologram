use std::ffi::CStr;
use std::os::raw::c_char;
use std::sync::{Mutex, OnceLock};

use hologram_ffi::{
    hologram_abi_version, hologram_archive_format_version, hologram_compile_source,
    hologram_error_message, hologram_feature_supported, hologram_last_error_code,
    hologram_last_error_column, hologram_last_error_line, hologram_last_error_rejected,
    hologram_session_archive_fingerprint, hologram_session_close, hologram_session_execute,
    hologram_session_extension, hologram_session_input_count, hologram_session_input_dtype,
    hologram_session_input_name, hologram_session_input_shape, hologram_session_kernel_count,
    hologram_session_load, hologram_session_output_byte_len, hologram_session_output_count,
    hologram_session_output_dtype, hologram_session_output_name, hologram_session_output_shape,
    hologram_source_builder_compile, hologram_source_builder_const,
    hologram_source_builder_const_ref, hologram_source_builder_free, hologram_source_builder_input,
    hologram_source_builder_new, hologram_source_builder_op, hologram_source_builder_output,
    hologram_source_builder_output_alias, HologramConstDesc, HologramExternalTensorDesc,
    HologramShape, HologramSourceBuilder, HologramSourceOp, HologramString, HologramTensorDesc,
};
use napi::bindgen_prelude::{Buffer, Result};
use napi::{Error, Status};
use napi_derive::napi;

const INITIAL_ARCHIVE_CAPACITY: usize = 16 * 1024;

#[napi(object)]
pub struct TensorDesc {
    pub name: String,
    pub dtype: u32,
    pub shape: Option<Vec<u32>>,
}

#[napi(object)]
pub struct ConstDesc {
    pub tensor: TensorDesc,
    pub bytes: Buffer,
}

#[napi(object)]
pub struct ConstRefDesc {
    pub tensor: TensorDesc,
    pub file: String,
    pub byte_offset: u32,
    pub byte_len: u32,
    pub blake3: String,
}

#[napi(object)]
pub struct OpDesc {
    pub output: String,
    pub op: String,
    pub inputs: Vec<String>,
    pub shape: Option<Vec<u32>>,
}

fn builders() -> &'static Mutex<Vec<Option<usize>>> {
    static BUILDERS: OnceLock<Mutex<Vec<Option<usize>>>> = OnceLock::new();
    BUILDERS.get_or_init(|| Mutex::new(Vec::new()))
}

#[napi(js_name = "abiVersion")]
pub fn abi_version() -> u32 {
    hologram_abi_version()
}

#[napi(js_name = "archiveFormatVersion")]
pub fn archive_format_version() -> u32 {
    hologram_archive_format_version()
}

#[napi(js_name = "featureSupported")]
pub fn feature_supported(feature: String) -> i32 {
    unsafe { hologram_feature_supported(hstr(&feature)) }
}

#[napi(js_name = "lastErrorCode")]
pub fn last_error_code() -> i32 {
    hologram_last_error_code()
}

#[napi(js_name = "lastErrorMessage")]
pub fn last_error_message() -> Option<String> {
    error_message()
}

#[napi(js_name = "lastErrorLine")]
pub fn last_error_line() -> u32 {
    hologram_last_error_line() as u32
}

#[napi(js_name = "lastErrorColumn")]
pub fn last_error_column() -> u32 {
    hologram_last_error_column() as u32
}

#[napi(js_name = "lastErrorRejected")]
pub fn last_error_rejected() -> Option<String> {
    error_string(hologram_last_error_rejected())
}

#[napi(js_name = "sourceBuilderNew")]
pub fn source_builder_new() -> Result<u32> {
    let ptr = hologram_source_builder_new();
    if ptr.is_null() {
        return Err(last_error("sourceBuilderNew"));
    }
    let mut table = lock_builders()?;
    table.push(Some(ptr as usize));
    Ok((table.len() - 1) as u32)
}

#[napi(js_name = "sourceBuilderFree")]
pub fn source_builder_free(handle: u32) -> Result<()> {
    let Some(ptr) = take_builder(handle)? else {
        return Ok(());
    };
    unsafe { hologram_source_builder_free(ptr) };
    Ok(())
}

#[napi(js_name = "sourceBuilderInput")]
pub fn source_builder_input(handle: u32, desc: TensorDesc) -> Result<i32> {
    with_builder(handle, |builder| {
        let shape = shape_storage(desc.shape);
        let ffi = tensor_desc(&desc.name, desc.dtype, &shape);
        unsafe { hologram_source_builder_input(builder, &ffi) }
    })
}

#[napi(js_name = "sourceBuilderConst")]
pub fn source_builder_const(handle: u32, desc: ConstDesc) -> Result<i32> {
    with_builder(handle, |builder| {
        let shape = shape_storage(desc.tensor.shape);
        let tensor = tensor_desc(&desc.tensor.name, desc.tensor.dtype, &shape);
        let bytes = desc.bytes.as_ref();
        let ffi = HologramConstDesc {
            tensor,
            bytes: bytes.as_ptr(),
            byte_len: bytes.len(),
        };
        unsafe { hologram_source_builder_const(builder, &ffi) }
    })
}

#[napi(js_name = "sourceBuilderConstRef")]
pub fn source_builder_const_ref(handle: u32, desc: ConstRefDesc) -> Result<i32> {
    let content_hash = parse_blake3(&desc.blake3)?;
    with_builder(handle, |builder| {
        let shape = shape_storage(desc.tensor.shape);
        let tensor = tensor_desc(&desc.tensor.name, desc.tensor.dtype, &shape);
        let ffi = HologramExternalTensorDesc {
            tensor,
            path: hstr(&desc.file),
            byte_offset: u64::from(desc.byte_offset),
            byte_len: u64::from(desc.byte_len),
            content_hash,
        };
        unsafe { hologram_source_builder_const_ref(builder, &ffi) }
    })
}

#[napi(js_name = "sourceBuilderOp")]
pub fn source_builder_op(handle: u32, desc: OpDesc) -> Result<i32> {
    with_builder(handle, |builder| {
        let shape = shape_storage(desc.shape);
        let inputs = desc
            .inputs
            .iter()
            .map(|input| hstr(input))
            .collect::<Vec<_>>();
        let ffi = HologramSourceOp {
            output: hstr(&desc.output),
            op: hstr(&desc.op),
            inputs: inputs.as_ptr(),
            input_count: inputs.len(),
            shape: shape_ref(&shape),
        };
        unsafe { hologram_source_builder_op(builder, &ffi) }
    })
}

#[napi(js_name = "sourceBuilderOutput")]
pub fn source_builder_output(handle: u32, name: String) -> Result<i32> {
    with_builder(handle, |builder| unsafe {
        hologram_source_builder_output(builder, hstr(&name))
    })
}

#[napi(js_name = "sourceBuilderOutputAlias")]
pub fn source_builder_output_alias(handle: u32, name: String, source: String) -> Result<i32> {
    with_builder(handle, |builder| unsafe {
        hologram_source_builder_output_alias(builder, hstr(&name), hstr(&source))
    })
}

#[napi(js_name = "sourceBuilderCompile")]
pub fn source_builder_compile(handle: u32) -> Result<Buffer> {
    let ptr = builder_ptr(handle)?;
    let archive = compile_archive(ptr)?;
    source_builder_free(handle)?;
    Ok(Buffer::from(archive))
}

#[napi(js_name = "compileSource")]
pub fn compile_source(source: Buffer) -> Result<Buffer> {
    compile_source_archive(source.as_ref()).map(Buffer::from)
}

#[napi(js_name = "sessionLoad")]
pub fn session_load(archive: Buffer) -> Result<u32> {
    let handle = unsafe { hologram_session_load(archive.as_ptr(), archive.len()) };
    if handle < 0 {
        return Err(last_error("sessionLoad"));
    }
    Ok(handle as u32)
}

#[napi(js_name = "sessionInputCount")]
pub fn session_input_count(handle: u32) -> Result<i32> {
    session_value(
        unsafe { hologram_session_input_count(handle as i32) },
        "sessionInputCount",
    )
}

#[napi(js_name = "sessionOutputCount")]
pub fn session_output_count(handle: u32) -> Result<i32> {
    session_value(
        unsafe { hologram_session_output_count(handle as i32) },
        "sessionOutputCount",
    )
}

#[napi(js_name = "sessionKernelCount")]
pub fn session_kernel_count(handle: u32) -> Result<i32> {
    session_value(
        unsafe { hologram_session_kernel_count(handle as i32) },
        "sessionKernelCount",
    )
}

#[napi(js_name = "sessionOutputByteLen")]
pub fn session_output_byte_len(handle: u32, index: u32) -> Result<i32> {
    output_len(handle, index as usize)
}

#[napi(js_name = "sessionInputDType")]
pub fn session_input_dtype(handle: u32, index: u32) -> Result<i32> {
    session_value(
        unsafe { hologram_session_input_dtype(handle as i32, index as usize) },
        "sessionInputDType",
    )
}

#[napi(js_name = "sessionOutputDType")]
pub fn session_output_dtype(handle: u32, index: u32) -> Result<i32> {
    session_value(
        unsafe { hologram_session_output_dtype(handle as i32, index as usize) },
        "sessionOutputDType",
    )
}

#[napi(js_name = "sessionArchiveFingerprint")]
pub fn session_archive_fingerprint(handle: u32) -> Result<Buffer> {
    let mut out = [0u8; 32];
    let result = unsafe { hologram_session_archive_fingerprint(handle as i32, out.as_mut_ptr()) };
    session_value(result, "sessionArchiveFingerprint")?;
    Ok(Buffer::from(out.to_vec()))
}

#[napi(js_name = "sessionInputName")]
pub fn session_input_name(handle: u32, index: u32) -> Result<String> {
    session_name(handle, index as usize, true)
}

#[napi(js_name = "sessionOutputName")]
pub fn session_output_name(handle: u32, index: u32) -> Result<String> {
    session_name(handle, index as usize, false)
}

#[napi(js_name = "sessionInputShape")]
pub fn session_input_shape(handle: u32, index: u32) -> Result<Vec<u32>> {
    session_shape(handle, index as usize, true)
}

#[napi(js_name = "sessionOutputShape")]
pub fn session_output_shape(handle: u32, index: u32) -> Result<Vec<u32>> {
    session_shape(handle, index as usize, false)
}

#[napi(js_name = "sessionExtension")]
pub fn session_extension(handle: u32, key: String) -> Result<Option<Buffer>> {
    session_extension_bytes(handle, &key).map(|bytes| bytes.map(Buffer::from))
}

#[napi(js_name = "sessionExecute")]
pub fn session_execute(handle: u32, inputs: Vec<Buffer>) -> Result<Vec<Buffer>> {
    let mut outputs = output_buffers(handle)?;
    let input_ptrs = inputs
        .iter()
        .map(|input| input.as_ptr())
        .collect::<Vec<_>>();
    let input_lens = inputs.iter().map(|input| input.len()).collect::<Vec<_>>();
    let mut output_ptrs = outputs
        .iter_mut()
        .map(|output| output.as_mut_ptr())
        .collect::<Vec<_>>();
    let output_lens = outputs.iter().map(Vec::len).collect::<Vec<_>>();
    let result = unsafe {
        hologram_session_execute(
            handle as i32,
            input_ptrs.as_ptr(),
            input_lens.as_ptr(),
            inputs.len(),
            output_ptrs.as_mut_ptr(),
            output_lens.as_ptr(),
            outputs.len(),
        )
    };
    session_value(result, "sessionExecute")?;
    Ok(outputs.into_iter().map(Buffer::from).collect())
}

#[napi(js_name = "sessionClose")]
pub fn session_close(handle: u32) -> Result<i32> {
    session_value(
        unsafe { hologram_session_close(handle as i32) },
        "sessionClose",
    )
}

fn with_builder<F>(handle: u32, f: F) -> Result<i32>
where
    F: FnOnce(*mut HologramSourceBuilder) -> i32,
{
    let result = f(builder_ptr(handle)?);
    if result < 0 {
        return Err(last_error("source builder"));
    }
    Ok(result)
}

fn builder_ptr(handle: u32) -> Result<*mut HologramSourceBuilder> {
    let table = lock_builders()?;
    let ptr = table
        .get(handle as usize)
        .and_then(|slot| *slot)
        .ok_or_else(|| Error::new(Status::InvalidArg, "invalid source builder handle"))?;
    Ok(ptr as *mut HologramSourceBuilder)
}

fn take_builder(handle: u32) -> Result<Option<*mut HologramSourceBuilder>> {
    let mut table = lock_builders()?;
    let ptr = table
        .get_mut(handle as usize)
        .and_then(Option::take)
        .map(|ptr| ptr as *mut HologramSourceBuilder);
    Ok(ptr)
}

fn lock_builders() -> Result<std::sync::MutexGuard<'static, Vec<Option<usize>>>> {
    builders()
        .lock()
        .map_err(|_| Error::new(Status::GenericFailure, "source builder table poisoned"))
}

fn compile_archive(builder: *const HologramSourceBuilder) -> Result<Vec<u8>> {
    let mut capacity = INITIAL_ARCHIVE_CAPACITY;
    loop {
        let mut out = vec![0u8; capacity];
        let required =
            unsafe { hologram_source_builder_compile(builder, out.as_mut_ptr(), out.len()) };
        if required < 0 {
            return Err(last_error("sourceBuilderCompile"));
        }
        let required = required as usize;
        if required <= capacity {
            out.truncate(required);
            return Ok(out);
        }
        capacity = required;
    }
}

fn compile_source_archive(source: &[u8]) -> Result<Vec<u8>> {
    let mut capacity = INITIAL_ARCHIVE_CAPACITY;
    loop {
        let mut out = vec![0u8; capacity];
        let required = unsafe {
            hologram_compile_source(source.as_ptr(), source.len(), out.as_mut_ptr(), out.len())
        };
        if required < 0 {
            return Err(last_error("compileSource"));
        }
        let required = required as usize;
        if required <= capacity {
            out.truncate(required);
            return Ok(out);
        }
        capacity = required;
    }
}

fn session_value(result: i32, context: &str) -> Result<i32> {
    if result < 0 {
        return Err(last_error(context));
    }
    Ok(result)
}

fn output_len(handle: u32, index: usize) -> Result<i32> {
    session_value(
        unsafe { hologram_session_output_byte_len(handle as i32, index) },
        "sessionOutputByteLen",
    )
}

fn output_buffers(handle: u32) -> Result<Vec<Vec<u8>>> {
    let count = session_output_count(handle)? as usize;
    (0..count)
        .map(|i| output_len(handle, i).map(|len| vec![0u8; len as usize]))
        .collect()
}

fn session_name(handle: u32, index: usize, input: bool) -> Result<String> {
    let mut capacity = 64usize;
    loop {
        let mut out = vec![0u8; capacity];
        let required = copy_name(handle, index, input, &mut out)?;
        if required <= capacity {
            out.truncate(required);
            return String::from_utf8(out)
                .map_err(|_| Error::new(Status::InvalidArg, "session name utf8"));
        }
        capacity = required;
    }
}

fn copy_name(handle: u32, index: usize, input: bool, out: &mut [u8]) -> Result<usize> {
    let result = if input {
        unsafe { hologram_session_input_name(handle as i32, index, out.as_mut_ptr(), out.len()) }
    } else {
        unsafe { hologram_session_output_name(handle as i32, index, out.as_mut_ptr(), out.len()) }
    };
    session_value(result, "sessionName").map(|required| required as usize)
}

fn session_shape(handle: u32, index: usize, input: bool) -> Result<Vec<u32>> {
    let mut capacity = 8usize;
    loop {
        let mut out = vec![0u64; capacity];
        let rank = copy_shape(handle, index, input, &mut out)?;
        if rank <= capacity {
            out.truncate(rank);
            return Ok(out.into_iter().map(|dim| dim as u32).collect());
        }
        capacity = rank;
    }
}

fn copy_shape(handle: u32, index: usize, input: bool, out: &mut [u64]) -> Result<usize> {
    let result = if input {
        unsafe { hologram_session_input_shape(handle as i32, index, out.as_mut_ptr(), out.len()) }
    } else {
        unsafe { hologram_session_output_shape(handle as i32, index, out.as_mut_ptr(), out.len()) }
    };
    session_value(result, "sessionShape").map(|rank| rank as usize)
}

fn session_extension_bytes(handle: u32, key: &str) -> Result<Option<Vec<u8>>> {
    let mut capacity = 256usize;
    loop {
        match copy_extension(handle, key, capacity) {
            ExtensionRead::Missing => return Ok(None),
            ExtensionRead::Bytes(bytes) => return Ok(Some(bytes)),
            ExtensionRead::Grow(required) => capacity = required,
        }
    }
}

enum ExtensionRead {
    Missing,
    Bytes(Vec<u8>),
    Grow(usize),
}

fn copy_extension(handle: u32, key: &str, capacity: usize) -> ExtensionRead {
    let mut out = vec![0u8; capacity];
    let required = unsafe { read_extension(handle, key, &mut out) };
    if required < 0 {
        return ExtensionRead::Missing;
    }
    extension_result(out, required as usize, capacity)
}

unsafe fn read_extension(handle: u32, key: &str, out: &mut [u8]) -> i32 {
    unsafe {
        hologram_session_extension(
            handle as i32,
            key.as_ptr(),
            key.len(),
            out.as_mut_ptr(),
            out.len(),
        )
    }
}

fn extension_result(mut out: Vec<u8>, required: usize, capacity: usize) -> ExtensionRead {
    if required > capacity {
        return ExtensionRead::Grow(required);
    }
    out.truncate(required);
    ExtensionRead::Bytes(out)
}

fn tensor_desc(name: &str, dtype: u32, shape: &[u64]) -> HologramTensorDesc {
    HologramTensorDesc {
        name: hstr(name),
        dtype_id: dtype as u8,
        shape: shape_ref(shape),
    }
}

fn shape_storage(shape: Option<Vec<u32>>) -> Vec<u64> {
    shape
        .unwrap_or_default()
        .into_iter()
        .map(u64::from)
        .collect()
}

fn shape_ref(shape: &[u64]) -> HologramShape {
    HologramShape {
        dims: shape.as_ptr(),
        rank: shape.len(),
    }
}

fn hstr(value: &str) -> HologramString {
    HologramString {
        ptr: value.as_ptr(),
        len: value.len(),
    }
}

fn parse_blake3(hex: &str) -> Result<[u8; 32]> {
    if hex.len() != 64 {
        return Err(Error::new(
            Status::InvalidArg,
            "blake3 must be 64 hex characters",
        ));
    }
    let mut out = [0u8; 32];
    for i in 0..32 {
        out[i] = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16)
            .map_err(|_| Error::new(Status::InvalidArg, "blake3 must be hex"))?;
    }
    Ok(out)
}

fn last_error(context: &str) -> Error {
    Error::new(
        Status::GenericFailure,
        error_message().unwrap_or_else(|| context.into()),
    )
}

fn error_message() -> Option<String> {
    error_string(hologram_error_message())
}

fn error_string(ptr: *const c_char) -> Option<String> {
    if ptr.is_null() {
        return None;
    }
    unsafe { cstr(ptr).to_str().ok().map(str::to_owned) }
}

unsafe fn cstr<'a>(ptr: *const c_char) -> &'a CStr {
    CStr::from_ptr(ptr)
}
