//! Source IR to graph lowering.

use alloc::string::String;
use alloc::vec::Vec;

use crate::error::CompileError;
#[cfg(feature = "std")]
use crate::source::SourceExternalTensorLocation;
use crate::source::{
    SourceAttrs, SourceBinding, SourceConst, SourceExpr, SourceExternalConst, SourceExternalTensor,
    SourceInput, SourceItem, SourceOpCall, SourceOutput, SourceProgram, SourceSymbol, SourceType,
};
use core::convert::TryFrom;
use hashbrown::HashMap;
use hologram_graph::constant::ConstantEntry;
use hologram_graph::node::Node;
use hologram_graph::registry::{DTypeId, ShapeDescriptor, ShapeId};
use hologram_graph::{Graph, GraphOp, InputSource, NodeId};
use smallvec::SmallVec;

const DTYPE_F32: u8 = 8;

/// Lower a source program into the graph IR.
pub fn lower_ir(program: &SourceProgram) -> Result<Graph, CompileError> {
    lower_program(program.clone())
}

pub(crate) fn lower_program(program: SourceProgram) -> Result<Graph, CompileError> {
    let mut lowering = Lowering::new();
    lowering.lower(program)?;
    Ok(lowering.finish())
}

struct Lowering {
    graph: Graph,
    names: HashMap<SourceSymbol, InputSource>,
    symbols: Vec<String>,
}

impl Lowering {
    fn new() -> Self {
        Self {
            graph: Graph::new(),
            names: HashMap::new(),
            symbols: Vec::new(),
        }
    }

    fn lower(&mut self, program: SourceProgram) -> Result<(), CompileError> {
        let (items, symbols) = program.into_parts();
        self.symbols = symbols;
        for item in items {
            self.lower_item(item)?;
        }
        Ok(())
    }

    fn lower_item(&mut self, item: SourceItem) -> Result<(), CompileError> {
        match item {
            SourceItem::Input(input) => self.lower_input(input),
            SourceItem::Const(constant) => self.lower_const(constant),
            SourceItem::ExternalConst(constant) => self.lower_external_const(constant),
            SourceItem::Binding(binding) => self.lower_binding(binding),
            SourceItem::Output(output) => self.lower_output(output),
        }
    }

    fn lower_input(&mut self, input: SourceInput) -> Result<(), CompileError> {
        let name = self.symbol_name(input.name)?;
        let node = self.input_node(&input.ty);
        let id = self.graph.add_node(node);
        self.graph.add_named_input(id, name);
        self.bind(input.name, InputSource::Node(id))
    }

    fn lower_const(&mut self, constant: SourceConst) -> Result<(), CompileError> {
        let SourceConst {
            name, ty, literal, ..
        } = constant;
        validate_const(&ty, literal.value_count, literal.bytes.len())?;
        let shape = required_shape(&ty, "const: missing shape")?;
        let shape = self.graph.shape_registry_mut().intern(shape);
        let entry = const_entry(literal.bytes, ty.dtype, shape);
        let id = self.graph.constants_mut().insert(entry);
        self.bind(name, InputSource::Constant(id))
    }

    fn lower_external_const(&mut self, constant: SourceExternalConst) -> Result<(), CompileError> {
        let SourceExternalConst {
            name,
            ty,
            reference,
            ..
        } = constant;
        let expected = expected_byte_len(&ty)?;
        validate_external_const(&reference, expected)?;
        let bytes = load_external_tensor(&reference, expected)?;
        let shape = required_shape(&ty, "external const: missing shape")?;
        let shape = self.graph.shape_registry_mut().intern(shape);
        let entry = const_entry(bytes, ty.dtype, shape);
        let id = self.graph.constants_mut().insert(entry);
        self.bind(name, InputSource::Constant(id))
    }

    fn lower_binding(&mut self, binding: SourceBinding) -> Result<(), CompileError> {
        let id = match &binding.expr {
            SourceExpr::OpCall(call) => self.lower_call(call)?,
            _ => return Err(CompileError::SourceParse("binding: unsupported expr")),
        };
        self.bind_optional(binding.name, id)
    }

    fn lower_call(&mut self, call: &SourceOpCall) -> Result<NodeId, CompileError> {
        let node = self.call_node(call)?;
        let id = self.graph.add_node(node);
        self.apply_attrs(id, &call.attrs);
        Ok(id)
    }

    fn lower_output(&mut self, output: SourceOutput) -> Result<(), CompileError> {
        let name = output.name;
        let port_name = self.symbol_name(output.port_name.unwrap_or(name))?;
        let src = self.node_source(name)?;
        let id = self.graph.add_node(output_node(src));
        self.graph.add_named_output(id, port_name);
        Ok(())
    }

    fn bind_optional(
        &mut self,
        name: Option<SourceSymbol>,
        id: NodeId,
    ) -> Result<(), CompileError> {
        if let Some(name) = name {
            self.bind(name, InputSource::Node(id))?;
        }
        Ok(())
    }

    fn bind(&mut self, name: SourceSymbol, source: InputSource) -> Result<(), CompileError> {
        if self.names.contains_key(&name) {
            return Err(CompileError::SourceParse("source: duplicate name"));
        }
        self.names.insert(name, source);
        Ok(())
    }

    fn finish(self) -> Graph {
        self.graph
    }
}

impl Lowering {
    fn input_node(&mut self, ty: &SourceType) -> Node {
        Node {
            op: GraphOp::Input,
            inputs: SmallVec::new(),
            output_dtype: ty.dtype,
            output_shape: self.shape_id(ty),
        }
    }

    fn call_node(&mut self, call: &SourceOpCall) -> Result<Node, CompileError> {
        Ok(Node {
            op: GraphOp::Op(call.op),
            inputs: self.inputs(&call.inputs)?,
            output_dtype: DTypeId(DTYPE_F32),
            output_shape: optional_shape_id(&mut self.graph, &call.ty),
        })
    }

    fn inputs(&self, names: &[SourceSymbol]) -> Result<SmallVec<[InputSource; 4]>, CompileError> {
        let mut inputs = SmallVec::new();
        for &name in names {
            inputs.push(self.source(name)?);
        }
        Ok(inputs)
    }

    fn source(&self, name: SourceSymbol) -> Result<InputSource, CompileError> {
        self.names
            .get(&name)
            .copied()
            .ok_or(CompileError::SourceParse("op: unresolved input"))
    }

    fn node_source(&self, name: SourceSymbol) -> Result<NodeId, CompileError> {
        match self.names.get(&name) {
            Some(InputSource::Node(id)) => Ok(*id),
            _ => Err(CompileError::SourceParse("output: unknown/!node source")),
        }
    }

    fn symbol_name(&self, symbol: SourceSymbol) -> Result<String, CompileError> {
        self.symbols
            .get(symbol.0 as usize)
            .cloned()
            .ok_or(CompileError::SourceParse("source: bad symbol"))
    }

    fn shape_id(&mut self, ty: &SourceType) -> ShapeId {
        shape_id(&mut self.graph, ty)
    }

    fn apply_attrs(&mut self, id: NodeId, attrs: &SourceAttrs) {
        if let Some(attrs) = attrs.quant {
            self.graph.set_quant_attrs(id, attrs);
        }
        if let Some(attrs) = attrs.conv {
            self.graph.set_conv_attrs(id, attrs);
        }
        if let Some(attrs) = attrs.lrn {
            self.graph.set_lrn_attrs(id, attrs);
        }
        self.apply_more_attrs(id, attrs);
    }

    fn apply_more_attrs(&mut self, id: NodeId, attrs: &SourceAttrs) {
        if let Some(attrs) = attrs.gemm {
            self.graph.set_gemm_attrs(id, attrs);
        }
        if let Some(attrs) = attrs.norm {
            self.graph.set_norm_attrs(id, attrs);
        }
        if let Some(attrs) = attrs.reduce {
            self.graph.set_reduce_attrs(id, attrs);
        }
        self.apply_tail_attrs(id, attrs);
    }

    fn apply_tail_attrs(&mut self, id: NodeId, attrs: &SourceAttrs) {
        if let Some(attrs) = attrs.gather {
            self.graph.set_gather_attrs(id, attrs);
        }
        if let Some(attrs) = attrs.attention {
            self.graph.set_attention_attrs(id, attrs);
        }
    }
}

fn optional_shape_id(graph: &mut Graph, ty: &Option<SourceType>) -> ShapeId {
    match ty {
        Some(ty) => shape_id(graph, ty),
        None => ShapeId(0),
    }
}

fn shape_id(graph: &mut Graph, ty: &SourceType) -> ShapeId {
    match &ty.shape {
        Some(shape) => graph.shape_registry_mut().intern(shape.clone()),
        None => ShapeId(0),
    }
}

fn required_shape(ty: &SourceType, err: &'static str) -> Result<ShapeDescriptor, CompileError> {
    ty.shape.clone().ok_or(CompileError::SourceParse(err))
}

fn expected_values(shape: &ShapeDescriptor) -> Result<usize, CompileError> {
    usize::try_from(shape.total_elements())
        .map_err(|_| CompileError::SourceParse("const: shape too large"))
}

fn validate_const(
    ty: &SourceType,
    value_count: usize,
    byte_len: usize,
) -> Result<(), CompileError> {
    let expected = expected_values(&required_shape(ty, "const: missing shape")?)?;
    if expected != value_count {
        return Err(CompileError::SourceParse("const: value count mismatch"));
    }
    if expected_byte_len(ty)? != byte_len {
        return Err(CompileError::SourceParse("const: byte count mismatch"));
    }
    Ok(())
}

fn expected_byte_len(ty: &SourceType) -> Result<usize, CompileError> {
    let values = expected_values(&required_shape(ty, "const: missing shape")?)?;
    values
        .checked_mul(dtype_width(ty.dtype)?)
        .ok_or(CompileError::SourceParse("const: byte count overflow"))
}

fn dtype_width(dtype: DTypeId) -> Result<usize, CompileError> {
    match dtype.0 {
        DTYPE_F32 => Ok(4),
        _ => Err(CompileError::SourceParse("const: unsupported dtype")),
    }
}

fn validate_external_const(
    reference: &SourceExternalTensor,
    expected: usize,
) -> Result<(), CompileError> {
    let byte_len = usize::try_from(reference.byte_len)
        .map_err(|_| CompileError::SourceParse("external const: byte len too large"))?;
    if byte_len != expected {
        return Err(CompileError::SourceParse(
            "external const: byte len mismatch",
        ));
    }
    Ok(())
}

#[cfg(feature = "std")]
fn load_external_tensor(
    reference: &SourceExternalTensor,
    expected: usize,
) -> Result<Vec<u8>, CompileError> {
    let bytes = match &reference.location {
        SourceExternalTensorLocation::File(path) => read_external_file(path, reference)?,
    };
    validate_external_bytes(reference, &bytes, expected)?;
    Ok(bytes)
}

#[cfg(not(feature = "std"))]
fn load_external_tensor(
    _reference: &SourceExternalTensor,
    _expected: usize,
) -> Result<Vec<u8>, CompileError> {
    Err(CompileError::SourceParse("external const: requires std"))
}

#[cfg(feature = "std")]
fn read_external_file(
    path: &str,
    reference: &SourceExternalTensor,
) -> Result<Vec<u8>, CompileError> {
    use std::io::{Read, Seek, SeekFrom};

    let path = resolve_external_path(path)?;
    let mut file = std::fs::File::open(path)
        .map_err(|_| CompileError::SourceParse("external const: open file"))?;
    file.seek(SeekFrom::Start(reference.byte_offset))
        .map_err(|_| CompileError::SourceParse("external const: seek file"))?;
    let mut bytes = Vec::new();
    file.take(reference.byte_len)
        .read_to_end(&mut bytes)
        .map_err(|_| CompileError::SourceParse("external const: read file"))?;
    Ok(bytes)
}

#[cfg(feature = "std")]
fn resolve_external_path(path: &str) -> Result<std::path::PathBuf, CompileError> {
    let raw = std::path::Path::new(path);
    let root = external_tensor_root()?;
    let candidate = if raw.is_absolute() {
        raw.to_path_buf()
    } else {
        root.join(raw)
    };
    let candidate = candidate
        .canonicalize()
        .map_err(|_| CompileError::SourceParse("external const: resolve file"))?;
    if external_root_enforced() && !candidate.starts_with(&root) {
        return Err(CompileError::SourceParse("external const: outside root"));
    }
    Ok(candidate)
}

#[cfg(feature = "std")]
fn external_tensor_root() -> Result<std::path::PathBuf, CompileError> {
    let root = match std::env::var_os("HOLOGRAM_EXTERNAL_TENSOR_ROOT") {
        Some(root) => std::path::PathBuf::from(root),
        None => std::env::current_dir()
            .map_err(|_| CompileError::SourceParse("external const: current dir"))?,
    };
    root.canonicalize()
        .map_err(|_| CompileError::SourceParse("external const: resolve root"))
}

#[cfg(feature = "std")]
fn external_root_enforced() -> bool {
    std::env::var_os("HOLOGRAM_EXTERNAL_TENSOR_ROOT").is_some()
}

#[cfg(feature = "std")]
fn validate_external_bytes(
    reference: &SourceExternalTensor,
    bytes: &[u8],
    expected: usize,
) -> Result<(), CompileError> {
    use hologram_host::HologramHasher;
    use prism::vocabulary::Hasher;

    if bytes.len() != expected {
        return Err(CompileError::SourceParse("external const: short read"));
    }
    let actual = HologramHasher::initial().fold_bytes(bytes).finalize();
    if actual != reference.content_hash {
        return Err(CompileError::SourceParse(
            "external const: content hash mismatch",
        ));
    }
    Ok(())
}

fn const_entry(bytes: Vec<u8>, dtype: DTypeId, shape: ShapeId) -> ConstantEntry {
    ConstantEntry {
        bytes,
        dtype,
        shape,
    }
}

fn output_node(src: NodeId) -> Node {
    Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(src)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: ShapeId(0),
    }
}
