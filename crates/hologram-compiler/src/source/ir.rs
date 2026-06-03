//! Common source IR for compiler frontends.

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use hashbrown::HashMap;
use hologram_graph::registry::{DTypeId, ShapeDescriptor};
use hologram_graph::{
    AttentionAttrs, ConvAttrs, GatherAttrs, GemmAttrs, LrnAttrs, NormAttrs, QuantAttrs, ReduceAttrs,
};

/// Compact source symbol handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SourceSymbol(pub u32);

/// Half-open source byte span.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SourceSpan {
    /// Byte offset from the start of the source.
    pub start: usize,
    /// Span length in bytes.
    pub len: usize,
}

impl SourceSpan {
    /// Return an empty span when legacy callers have no precise span.
    pub const fn empty() -> Self {
        Self { start: 0, len: 0 }
    }
}

/// Source-level tensor type.
#[derive(Debug, Clone)]
pub struct SourceType {
    /// Element dtype.
    pub dtype: DTypeId,
    /// Optional concrete tensor shape.
    pub shape: Option<ShapeDescriptor>,
}

impl SourceType {
    /// Build a tensor type with the f32 dtype used by the legacy language.
    pub const fn f32(shape: Option<ShapeDescriptor>) -> Self {
        Self {
            dtype: DTypeId(8),
            shape,
        }
    }
}

/// Parsed tensor literal bytes plus element count.
#[derive(Debug, Clone)]
pub struct SourceTensorLiteral {
    /// Final little-endian bytes for the target dtype.
    pub bytes: Vec<u8>,
    /// Number of scalar values in the literal.
    pub value_count: usize,
}

impl SourceTensorLiteral {
    /// Build a tensor literal from final bytes and element count.
    pub fn new(bytes: Vec<u8>, value_count: usize) -> Self {
        Self { bytes, value_count }
    }
}

/// Source-level reference to tensor bytes outside the parsed source text.
#[derive(Debug, Clone)]
pub struct SourceExternalTensor {
    /// External byte location.
    pub location: SourceExternalTensorLocation,
    /// Byte offset within the external object.
    pub byte_offset: u64,
    /// Number of tensor bytes to read.
    pub byte_len: u64,
    /// Expected BLAKE3 digest over the referenced byte range.
    pub content_hash: [u8; 32],
}

impl SourceExternalTensor {
    /// Build a file-backed external tensor reference.
    pub fn file(
        path: impl Into<String>,
        byte_offset: u64,
        byte_len: u64,
        content_hash: [u8; 32],
    ) -> Self {
        Self {
            location: SourceExternalTensorLocation::File(path.into()),
            byte_offset,
            byte_len,
            content_hash,
        }
    }
}

/// External tensor byte location.
#[derive(Debug, Clone)]
pub enum SourceExternalTensorLocation {
    /// File path resolved by the compiler host process.
    File(String),
}

/// Source program before graph lowering.
#[derive(Debug, Default, Clone)]
pub struct SourceProgram {
    items: Vec<SourceItem>,
    symbols: Vec<String>,
    symbol_ids: HashMap<String, SourceSymbol>,
}

impl SourceProgram {
    /// Create an empty source program.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append an item to this program.
    pub fn push(&mut self, item: SourceItem) {
        self.items.push(item);
    }

    /// Intern a source-visible name and return its compact symbol.
    pub fn intern(&mut self, name: &str) -> SourceSymbol {
        if let Some(symbol) = self.symbol_ids.get(name) {
            return *symbol;
        }
        let symbol = SourceSymbol(self.symbols.len() as u32);
        let name = String::from(name);
        self.symbols.push(name.clone());
        self.symbol_ids.insert(name, symbol);
        symbol
    }

    /// Return the source text for a symbol.
    pub fn symbol_name(&self, symbol: SourceSymbol) -> Option<&str> {
        self.symbols
            .get(symbol.0 as usize)
            .map(|name| name.as_str())
    }

    /// Return program items in source order.
    pub fn items(&self) -> &[SourceItem] {
        &self.items
    }

    pub(crate) fn into_parts(self) -> (Vec<SourceItem>, Vec<String>) {
        (self.items, self.symbols)
    }
}

/// Source-level declaration or statement.
#[derive(Debug, Clone)]
pub enum SourceItem {
    /// Graph input declaration.
    Input(SourceInput),
    /// Constant tensor declaration.
    Const(SourceConst),
    /// File/address-backed constant tensor declaration.
    ExternalConst(SourceExternalConst),
    /// Named or anonymous expression binding.
    Binding(SourceBinding),
    /// Graph output declaration.
    Output(SourceOutput),
}

/// Source input declaration.
#[derive(Debug, Clone)]
pub struct SourceInput {
    /// Source-visible input name.
    pub name: SourceSymbol,
    /// Declared type.
    pub ty: SourceType,
    /// Source span.
    pub span: SourceSpan,
}

impl SourceInput {
    /// Build a source input declaration.
    pub fn new(name: SourceSymbol, ty: SourceType) -> Self {
        Self {
            name,
            ty,
            span: SourceSpan::empty(),
        }
    }
}

/// Source constant declaration.
#[derive(Debug, Clone)]
pub struct SourceConst {
    /// Source-visible constant name.
    pub name: SourceSymbol,
    /// Declared type.
    pub ty: SourceType,
    /// Literal bytes.
    pub literal: SourceTensorLiteral,
    /// Source span.
    pub span: SourceSpan,
}

impl SourceConst {
    /// Build a source constant declaration.
    pub fn new(name: SourceSymbol, ty: SourceType, literal: SourceTensorLiteral) -> Self {
        Self {
            name,
            ty,
            literal,
            span: SourceSpan::empty(),
        }
    }
}

/// Source external constant declaration.
#[derive(Debug, Clone)]
pub struct SourceExternalConst {
    /// Source-visible constant name.
    pub name: SourceSymbol,
    /// Declared type.
    pub ty: SourceType,
    /// External tensor reference.
    pub reference: SourceExternalTensor,
    /// Source span.
    pub span: SourceSpan,
}

impl SourceExternalConst {
    /// Build an external constant declaration.
    pub fn new(name: SourceSymbol, ty: SourceType, reference: SourceExternalTensor) -> Self {
        Self {
            name,
            ty,
            reference,
            span: SourceSpan::empty(),
        }
    }
}

/// Source expression binding.
#[derive(Debug, Clone)]
pub struct SourceBinding {
    /// Optional source-visible name for the expression result.
    pub name: Option<SourceSymbol>,
    /// Bound expression.
    pub expr: SourceExpr,
    /// Source span.
    pub span: SourceSpan,
}

impl SourceBinding {
    /// Build an op-call binding.
    pub fn op(name: Option<SourceSymbol>, call: SourceOpCall) -> Self {
        Self {
            name,
            expr: SourceExpr::OpCall(Box::new(call)),
            span: SourceSpan::empty(),
        }
    }
}

/// Source graph output declaration.
#[derive(Debug, Clone)]
pub struct SourceOutput {
    /// Name of a prior node-valued binding.
    pub name: SourceSymbol,
    /// Optional semantic output-port name.
    pub port_name: Option<SourceSymbol>,
    /// Source span.
    pub span: SourceSpan,
}

impl SourceOutput {
    /// Build an output declaration.
    pub fn new(name: SourceSymbol) -> Self {
        Self {
            name,
            port_name: None,
            span: SourceSpan::empty(),
        }
    }

    /// Build an output declaration with a semantic port name.
    pub fn with_port_name(name: SourceSymbol, port_name: SourceSymbol) -> Self {
        Self {
            name,
            port_name: Some(port_name),
            span: SourceSpan::empty(),
        }
    }
}

/// Source expression.
#[derive(Debug, Clone)]
pub enum SourceExpr {
    /// Reference to a prior source symbol.
    Ref(SourceSymbol),
    /// Inline tensor literal.
    TensorLiteral(SourceTensorLiteral),
    /// Canonical operation call.
    OpCall(Box<SourceOpCall>),
}

/// Source operation call.
#[derive(Debug, Clone)]
pub struct SourceOpCall {
    /// Canonical op kind.
    pub op: hologram_graph::OpKind,
    /// Source names of operands.
    pub inputs: Vec<SourceSymbol>,
    /// Optional result type.
    pub ty: Option<SourceType>,
    /// Source-level attributes.
    pub attrs: SourceAttrs,
    /// Source span.
    pub span: SourceSpan,
}

impl SourceOpCall {
    /// Build a source op call.
    pub fn new(
        op: hologram_graph::OpKind,
        inputs: Vec<SourceSymbol>,
        ty: Option<SourceType>,
    ) -> Self {
        Self {
            op,
            inputs,
            ty,
            attrs: SourceAttrs::default(),
            span: SourceSpan::empty(),
        }
    }
}

/// Source-level op attributes.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SourceAttrs {
    /// Optional quantization attributes.
    pub quant: Option<QuantAttrs>,
    /// Optional convolution attributes.
    pub conv: Option<ConvAttrs>,
    /// Optional LRN attributes.
    pub lrn: Option<LrnAttrs>,
    /// Optional GEMM scalar attributes.
    pub gemm: Option<GemmAttrs>,
    /// Optional normalization attributes.
    pub norm: Option<NormAttrs>,
    /// Optional reduction attributes.
    pub reduce: Option<ReduceAttrs>,
    /// Optional gather attributes.
    pub gather: Option<GatherAttrs>,
    /// Optional attention attributes.
    pub attention: Option<AttentionAttrs>,
}
