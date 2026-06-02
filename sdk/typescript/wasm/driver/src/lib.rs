use std::ffi::CStr;
use std::os::raw::c_char;
use std::sync::{Mutex, OnceLock};

use hologram_ffi::{
    hologram_abi_version, hologram_archive_format_version, hologram_error_message,
    hologram_feature_supported, hologram_last_error_code, hologram_last_error_column,
    hologram_last_error_line, hologram_last_error_rejected, hologram_compile_source,
    hologram_source_builder_compile, hologram_source_builder_const, hologram_source_builder_const_ref,
    hologram_source_builder_free, hologram_source_builder_input, hologram_source_builder_new,
    hologram_source_builder_op, hologram_source_builder_output, hologram_source_builder_output_alias,
    hologram_session_archive_fingerprint, hologram_session_close, hologram_session_execute,
    hologram_session_extension, hologram_session_input_count, hologram_session_input_dtype,
    hologram_session_input_name, hologram_session_input_shape, hologram_session_kernel_count,
    hologram_session_load, hologram_session_output_byte_len, hologram_session_output_count,
    hologram_session_output_dtype, hologram_session_output_name, hologram_session_output_shape,
    HologramConstDesc, HologramExternalTensorDesc, HologramShape, HologramSourceBuilder,
    HologramSourceOp, HologramString, HologramTensorDesc,
};
use js_sys::{Array, Reflect, Uint8Array};
use wasm_bindgen::prelude::*;

const INITIAL_ARCHIVE_CAPACITY: usize = 16 * 1024;

#[wasm_bindgen]
pub struct HologramWasmDriver;

fn builders() -> &'static Mutex<Vec<Option<usize>>> {
    static BUILDERS: OnceLock<Mutex<Vec<Option<usize>>>> = OnceLock::new();
    BUILDERS.get_or_init(|| Mutex::new(Vec::new()))
}

#[wasm_bindgen]
impl HologramWasmDriver {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self
    }

    #[wasm_bindgen(js_name = abiVersion)]
    pub fn abi_version(&self) -> u32 {
        hologram_abi_version()
    }

    #[wasm_bindgen(js_name = archiveFormatVersion)]
    pub fn archive_format_version(&self) -> u32 {
        hologram_archive_format_version()
    }

    #[wasm_bindgen(js_name = featureSupported)]
    pub fn feature_supported(&self, feature: String) -> i32 {
        unsafe { hologram_feature_supported(hstr(&feature)) }
    }

    #[wasm_bindgen(js_name = lastErrorCode)]
    pub fn last_error_code(&self) -> i32 {
        hologram_last_error_code()
    }

    #[wasm_bindgen(js_name = lastErrorMessage)]
    pub fn last_error_message(&self) -> Option<String> {
        error_message()
    }

    #[wasm_bindgen(js_name = lastErrorLine)]
    pub fn last_error_line(&self) -> u32 {
        hologram_last_error_line() as u32
    }

    #[wasm_bindgen(js_name = lastErrorColumn)]
    pub fn last_error_column(&self) -> u32 {
        hologram_last_error_column() as u32
    }

    #[wasm_bindgen(js_name = lastErrorRejected)]
    pub fn last_error_rejected(&self) -> Option<String> {
        error_string(hologram_last_error_rejected())
    }

    #[wasm_bindgen(js_name = sourceBuilderNew)]
    pub fn source_builder_new(&self) -> Result<u32, JsValue> {
        let ptr = hologram_source_builder_new();
        if ptr.is_null() {
            return Err(last_error("sourceBuilderNew"));
        }
        let mut table = lock_builders()?;
        table.push(Some(ptr as usize));
        Ok((table.len() - 1) as u32)
    }

    #[wasm_bindgen(js_name = sourceBuilderFree)]
    pub fn source_builder_free(&self, handle: u32) -> Result<(), JsValue> {
        let Some(ptr) = take_builder(handle)? else {
            return Ok(());
        };
        unsafe { hologram_source_builder_free(ptr) };
        Ok(())
    }

    #[wasm_bindgen(js_name = sourceBuilderInput)]
    pub fn source_builder_input(&self, handle: u32, desc: JsValue) -> Result<i32, JsValue> {
        with_builder(handle, |builder| {
            let name = string_prop(&desc, "name")?;
            let dtype = number_prop(&desc, "dtype")? as u32;
            let shape = shape_prop(&desc)?;
            let ffi = tensor_desc(&name, dtype, &shape);
            Ok(unsafe { hologram_source_builder_input(builder, &ffi) })
        })
    }

    #[wasm_bindgen(js_name = sourceBuilderConst)]
    pub fn source_builder_const(&self, handle: u32, desc: JsValue) -> Result<i32, JsValue> {
        with_builder(handle, |builder| {
            let tensor = prop(&desc, "tensor")?;
            let name = string_prop(&tensor, "name")?;
            let dtype = number_prop(&tensor, "dtype")? as u32;
            let shape = shape_prop(&tensor)?;
            let bytes = bytes_prop(&desc, "bytes")?;
            let tensor = tensor_desc(&name, dtype, &shape);
            let ffi = HologramConstDesc {
                tensor,
                bytes: bytes.as_ptr(),
                byte_len: bytes.len(),
            };
            Ok(unsafe { hologram_source_builder_const(builder, &ffi) })
        })
    }

    #[wasm_bindgen(js_name = sourceBuilderConstRef)]
    pub fn source_builder_const_ref(&self, handle: u32, desc: JsValue) -> Result<i32, JsValue> {
        let hash = parse_blake3(&string_prop(&desc, "blake3")?)?;
        with_builder(handle, |builder| {
            let tensor = prop(&desc, "tensor")?;
            let name = string_prop(&tensor, "name")?;
            let dtype = number_prop(&tensor, "dtype")? as u32;
            let shape = shape_prop(&tensor)?;
            let file = string_prop(&desc, "file")?;
            let tensor = tensor_desc(&name, dtype, &shape);
            let ffi = HologramExternalTensorDesc {
                tensor,
                path: hstr(&file),
                byte_offset: number_prop(&desc, "byteOffset")? as u64,
                byte_len: number_prop(&desc, "byteLen")? as u64,
                content_hash: hash,
            };
            Ok(unsafe { hologram_source_builder_const_ref(builder, &ffi) })
        })
    }

    #[wasm_bindgen(js_name = sourceBuilderOp)]
    pub fn source_builder_op(&self, handle: u32, desc: JsValue) -> Result<i32, JsValue> {
        with_builder(handle, |builder| {
            let output = string_prop(&desc, "output")?;
            let op = string_prop(&desc, "op")?;
            let inputs = strings_prop(&desc, "inputs")?;
            let shape = shape_prop(&desc)?;
            let ffi_inputs = inputs.iter().map(|input| hstr(input)).collect::<Vec<_>>();
            let ffi = HologramSourceOp {
                output: hstr(&output),
                op: hstr(&op),
                inputs: ffi_inputs.as_ptr(),
                input_count: ffi_inputs.len(),
                shape: shape_ref(&shape),
            };
            Ok(unsafe { hologram_source_builder_op(builder, &ffi) })
        })
    }

    #[wasm_bindgen(js_name = sourceBuilderOutput)]
    pub fn source_builder_output(&self, handle: u32, name: String) -> Result<i32, JsValue> {
        with_builder(handle, |builder| {
            Ok(unsafe { hologram_source_builder_output(builder, hstr(&name)) })
        })
    }

    #[wasm_bindgen(js_name = sourceBuilderOutputAlias)]
    pub fn source_builder_output_alias(
        &self,
        handle: u32,
        name: String,
        source: String,
    ) -> Result<i32, JsValue> {
        with_builder(handle, |builder| {
            Ok(
                unsafe {
                    hologram_source_builder_output_alias(builder, hstr(&name), hstr(&source))
                },
            )
        })
    }

    #[wasm_bindgen(js_name = sourceBuilderCompile)]
    pub fn source_builder_compile(&self, handle: u32) -> Result<Uint8Array, JsValue> {
        let archive = compile_archive(builder_ptr(handle)?)?;
        Ok(Uint8Array::from(archive.as_slice()))
    }

    #[wasm_bindgen(js_name = compileSource)]
    pub fn compile_source(&self, source: Uint8Array) -> Result<Uint8Array, JsValue> {
        let bytes = source.to_vec();
        let archive = compile_source_archive(&bytes)?;
        Ok(Uint8Array::from(archive.as_slice()))
    }

    #[wasm_bindgen(js_name = sessionLoad)]
    pub fn session_load(&self, archive: Uint8Array) -> Result<u32, JsValue> {
        let bytes = archive.to_vec();
        let handle = unsafe { hologram_session_load(bytes.as_ptr(), bytes.len()) };
        if handle < 0 {
            return Err(last_error("sessionLoad"));
        }
        Ok(handle as u32)
    }

    #[wasm_bindgen(js_name = sessionInputCount)]
    pub fn session_input_count(&self, handle: u32) -> Result<i32, JsValue> {
        session_value(unsafe { hologram_session_input_count(handle as i32) }, "sessionInputCount")
    }

    #[wasm_bindgen(js_name = sessionOutputCount)]
    pub fn session_output_count(&self, handle: u32) -> Result<i32, JsValue> {
        session_value(
            unsafe { hologram_session_output_count(handle as i32) },
            "sessionOutputCount",
        )
    }

    #[wasm_bindgen(js_name = sessionKernelCount)]
    pub fn session_kernel_count(&self, handle: u32) -> Result<i32, JsValue> {
        session_value(
            unsafe { hologram_session_kernel_count(handle as i32) },
            "sessionKernelCount",
        )
    }

    #[wasm_bindgen(js_name = sessionOutputByteLen)]
    pub fn session_output_byte_len(&self, handle: u32, index: u32) -> Result<i32, JsValue> {
        output_len(handle, index as usize)
    }

    #[wasm_bindgen(js_name = sessionInputDType)]
    pub fn session_input_dtype(&self, handle: u32, index: u32) -> Result<i32, JsValue> {
        session_value(
            unsafe { hologram_session_input_dtype(handle as i32, index as usize) },
            "sessionInputDType",
        )
    }

    #[wasm_bindgen(js_name = sessionOutputDType)]
    pub fn session_output_dtype(&self, handle: u32, index: u32) -> Result<i32, JsValue> {
        session_value(
            unsafe { hologram_session_output_dtype(handle as i32, index as usize) },
            "sessionOutputDType",
        )
    }

    #[wasm_bindgen(js_name = sessionArchiveFingerprint)]
    pub fn session_archive_fingerprint(&self, handle: u32) -> Result<Uint8Array, JsValue> {
        let mut out = [0u8; 32];
        let result = unsafe { hologram_session_archive_fingerprint(handle as i32, out.as_mut_ptr()) };
        session_value(result, "sessionArchiveFingerprint")?;
        Ok(Uint8Array::from(out.as_slice()))
    }

    #[wasm_bindgen(js_name = sessionInputName)]
    pub fn session_input_name(&self, handle: u32, index: u32) -> Result<String, JsValue> {
        session_name(handle, index as usize, true)
    }

    #[wasm_bindgen(js_name = sessionOutputName)]
    pub fn session_output_name(&self, handle: u32, index: u32) -> Result<String, JsValue> {
        session_name(handle, index as usize, false)
    }

    #[wasm_bindgen(js_name = sessionInputShape)]
    pub fn session_input_shape(&self, handle: u32, index: u32) -> Result<Array, JsValue> {
        Ok(numbers(session_shape(handle, index as usize, true)?))
    }

    #[wasm_bindgen(js_name = sessionOutputShape)]
    pub fn session_output_shape(&self, handle: u32, index: u32) -> Result<Array, JsValue> {
        Ok(numbers(session_shape(handle, index as usize, false)?))
    }

    #[wasm_bindgen(js_name = sessionExtension)]
    pub fn session_extension(&self, handle: u32, key: String) -> Result<JsValue, JsValue> {
        match session_extension_bytes(handle, &key)? {
            Some(bytes) => Ok(Uint8Array::from(bytes.as_slice()).into()),
            None => Ok(JsValue::NULL),
        }
    }

    #[wasm_bindgen(js_name = sessionExecute)]
    pub fn session_execute(&self, handle: u32, inputs: JsValue) -> Result<Array, JsValue> {
        let inputs = byte_arrays(inputs)?;
        let outputs = execute_session(handle, &inputs)?;
        let result = Array::new();
        for output in outputs {
            result.push(&Uint8Array::from(output.as_slice()));
        }
        Ok(result)
    }

    #[wasm_bindgen(js_name = sessionClose)]
    pub fn session_close(&self, handle: u32) -> Result<i32, JsValue> {
        session_value(unsafe { hologram_session_close(handle as i32) }, "sessionClose")
    }
}

fn with_builder<F>(handle: u32, f: F) -> Result<i32, JsValue>
where
    F: FnOnce(*mut HologramSourceBuilder) -> Result<i32, JsValue>,
{
    let result = f(builder_ptr(handle)?)?;
    if result < 0 {
        return Err(last_error("source builder"));
    }
    Ok(result)
}

fn builder_ptr(handle: u32) -> Result<*mut HologramSourceBuilder, JsValue> {
    let table = lock_builders()?;
    let ptr = table
        .get(handle as usize)
        .and_then(|slot| *slot)
        .ok_or_else(|| js_error("invalid source builder handle"))?;
    Ok(ptr as *mut HologramSourceBuilder)
}

fn take_builder(handle: u32) -> Result<Option<*mut HologramSourceBuilder>, JsValue> {
    let mut table = lock_builders()?;
    Ok(table
        .get_mut(handle as usize)
        .and_then(Option::take)
        .map(|ptr| ptr as *mut HologramSourceBuilder))
}

fn lock_builders() -> Result<std::sync::MutexGuard<'static, Vec<Option<usize>>>, JsValue> {
    builders()
        .lock()
        .map_err(|_| js_error("source builder table poisoned"))
}

fn compile_archive(builder: *const HologramSourceBuilder) -> Result<Vec<u8>, JsValue> {
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

fn compile_source_archive(source: &[u8]) -> Result<Vec<u8>, JsValue> {
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

fn session_value(result: i32, context: &str) -> Result<i32, JsValue> {
    if result < 0 {
        return Err(last_error(context));
    }
    Ok(result)
}

fn output_len(handle: u32, index: usize) -> Result<i32, JsValue> {
    session_value(
        unsafe { hologram_session_output_byte_len(handle as i32, index) },
        "sessionOutputByteLen",
    )
}

fn output_buffers(handle: u32) -> Result<Vec<Vec<u8>>, JsValue> {
    let count = session_value(
        unsafe { hologram_session_output_count(handle as i32) },
        "sessionOutputCount",
    )? as usize;
    (0..count)
        .map(|i| output_len(handle, i).map(|len| vec![0u8; len as usize]))
        .collect()
}

fn session_name(handle: u32, index: usize, input: bool) -> Result<String, JsValue> {
    let mut capacity = 64usize;
    loop {
        let mut out = vec![0u8; capacity];
        let required = copy_name(handle, index, input, &mut out)?;
        if required <= capacity {
            out.truncate(required);
            return String::from_utf8(out).map_err(|_| js_error("session name utf8"));
        }
        capacity = required;
    }
}

fn copy_name(handle: u32, index: usize, input: bool, out: &mut [u8]) -> Result<usize, JsValue> {
    let result = if input {
        unsafe { hologram_session_input_name(handle as i32, index, out.as_mut_ptr(), out.len()) }
    } else {
        unsafe { hologram_session_output_name(handle as i32, index, out.as_mut_ptr(), out.len()) }
    };
    session_value(result, "sessionName").map(|required| required as usize)
}

fn session_shape(handle: u32, index: usize, input: bool) -> Result<Vec<u64>, JsValue> {
    let mut capacity = 8usize;
    loop {
        let mut out = vec![0u64; capacity];
        let rank = copy_shape(handle, index, input, &mut out)?;
        if rank <= capacity {
            out.truncate(rank);
            return Ok(out);
        }
        capacity = rank;
    }
}

fn copy_shape(handle: u32, index: usize, input: bool, out: &mut [u64]) -> Result<usize, JsValue> {
    let result = if input {
        unsafe { hologram_session_input_shape(handle as i32, index, out.as_mut_ptr(), out.len()) }
    } else {
        unsafe { hologram_session_output_shape(handle as i32, index, out.as_mut_ptr(), out.len()) }
    };
    session_value(result, "sessionShape").map(|rank| rank as usize)
}

fn session_extension_bytes(handle: u32, key: &str) -> Result<Option<Vec<u8>>, JsValue> {
    let mut capacity = 256usize;
    loop {
        match copy_extension(handle, key, capacity)? {
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

fn copy_extension(handle: u32, key: &str, capacity: usize) -> Result<ExtensionRead, JsValue> {
    let mut out = vec![0u8; capacity];
    let required = unsafe { read_extension(handle, key, &mut out) };
    if required < 0 {
        return Ok(ExtensionRead::Missing);
    }
    extension_result(out, required as usize, capacity)
}

unsafe fn read_extension(handle: u32, key: &str, out: &mut [u8]) -> i32 {
    unsafe { hologram_session_extension(
        handle as i32,
        key.as_ptr(),
        key.len(),
        out.as_mut_ptr(),
        out.len(),
    ) }
}

fn extension_result(
    mut out: Vec<u8>,
    required: usize,
    capacity: usize,
) -> Result<ExtensionRead, JsValue> {
    if required > capacity {
        return Ok(ExtensionRead::Grow(required));
    }
    out.truncate(required);
    Ok(ExtensionRead::Bytes(out))
}

fn byte_arrays(value: JsValue) -> Result<Vec<Vec<u8>>, JsValue> {
    let array = Array::from(&value);
    let mut out = Vec::with_capacity(array.length() as usize);
    for value in array.iter() {
        out.push(Uint8Array::new(&value).to_vec());
    }
    Ok(out)
}

fn execute_session(handle: u32, inputs: &[Vec<u8>]) -> Result<Vec<Vec<u8>>, JsValue> {
    let mut outputs = output_buffers(handle)?;
    let input_ptrs = inputs.iter().map(|input| input.as_ptr()).collect::<Vec<_>>();
    let input_lens = inputs.iter().map(Vec::len).collect::<Vec<_>>();
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
    Ok(outputs)
}

fn tensor_desc(name: &str, dtype: u32, shape: &[u64]) -> HologramTensorDesc {
    HologramTensorDesc {
        name: hstr(name),
        dtype_id: dtype as u8,
        shape: shape_ref(shape),
    }
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

fn prop(value: &JsValue, name: &str) -> Result<JsValue, JsValue> {
    Reflect::get(value, &JsValue::from_str(name)).map_err(|_| js_error("object property read"))
}

fn string_prop(value: &JsValue, name: &str) -> Result<String, JsValue> {
    prop(value, name)?
        .as_string()
        .ok_or_else(|| js_error("expected string property"))
}

fn number_prop(value: &JsValue, name: &str) -> Result<f64, JsValue> {
    prop(value, name)?
        .as_f64()
        .ok_or_else(|| js_error("expected number property"))
}

fn bytes_prop(value: &JsValue, name: &str) -> Result<Vec<u8>, JsValue> {
    Ok(Uint8Array::new(&prop(value, name)?).to_vec())
}

fn shape_prop(value: &JsValue) -> Result<Vec<u64>, JsValue> {
    let shape = prop(value, "shape")?;
    if shape.is_null() || shape.is_undefined() {
        return Ok(Vec::new());
    }
    let array = Array::from(&shape);
    let mut out = Vec::with_capacity(array.length() as usize);
    for value in array.iter() {
        out.push(
            value
                .as_f64()
                .ok_or_else(|| js_error("shape dim must be a number"))? as u64,
        );
    }
    Ok(out)
}

fn strings_prop(value: &JsValue, name: &str) -> Result<Vec<String>, JsValue> {
    let array = Array::from(&prop(value, name)?);
    let mut out = Vec::with_capacity(array.length() as usize);
    for value in array.iter() {
        out.push(
            value
                .as_string()
                .ok_or_else(|| js_error("expected string array"))?,
        );
    }
    Ok(out)
}

fn numbers(values: Vec<u64>) -> Array {
    values
        .into_iter()
        .map(|value| JsValue::from_f64(value as f64))
        .collect()
}

fn parse_blake3(hex: &str) -> Result<[u8; 32], JsValue> {
    if hex.len() != 64 {
        return Err(js_error("blake3 must be 64 hex characters"));
    }
    let mut out = [0u8; 32];
    for i in 0..32 {
        out[i] = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16)
            .map_err(|_| js_error("blake3 must be hex"))?;
    }
    Ok(out)
}

fn last_error(context: &str) -> JsValue {
    js_error(&error_message().unwrap_or_else(|| context.into()))
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

fn js_error(message: &str) -> JsValue {
    JsValue::from_str(message)
}
