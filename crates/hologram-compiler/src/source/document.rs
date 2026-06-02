//! Source documents containing one or more graph regions.

use alloc::string::String;
use alloc::vec::Vec;

use crate::error::CompileError;
use crate::source::{SourceDiagnostic, SourceProgram, SourceSpan};

/// Source parsing options shared by all language frontends.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SourceParseOptions {
    graph: Option<String>,
}

impl SourceParseOptions {
    /// Create default parse options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Select a named graph from a multi-graph source document.
    pub fn graph(mut self, graph: impl Into<String>) -> Self {
        self.graph = Some(graph.into());
        self
    }

    /// Return the selected graph name, if one was requested.
    pub fn graph_name(&self) -> Option<&str> {
        self.graph.as_deref()
    }
}

/// One graph region extracted from a source document.
#[derive(Debug, Clone)]
pub struct SourceGraph {
    /// Optional source-visible graph name.
    pub name: Option<String>,
    /// Parsed source IR for this graph.
    pub program: SourceProgram,
    /// Span of the graph region in the original source document.
    pub span: SourceSpan,
}

impl SourceGraph {
    /// Build an anonymous graph region.
    pub fn anonymous(program: SourceProgram) -> Self {
        Self {
            name: None,
            program,
            span: SourceSpan::empty(),
        }
    }

    /// Build a named graph region.
    pub fn named(name: impl Into<String>, program: SourceProgram) -> Self {
        Self {
            name: Some(name.into()),
            program,
            span: SourceSpan::empty(),
        }
    }
}

/// Parsed source document before graph selection.
#[derive(Debug, Clone, Default)]
pub struct SourceDocument {
    graphs: Vec<SourceGraph>,
}

impl SourceDocument {
    /// Create an empty source document.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a source document with one anonymous graph.
    pub fn single(program: SourceProgram) -> Self {
        let mut document = Self::new();
        document.push(SourceGraph::anonymous(program));
        document
    }

    /// Append a graph region.
    pub fn push(&mut self, graph: SourceGraph) {
        self.graphs.push(graph);
    }

    /// Return graph regions in document order.
    pub fn graphs(&self) -> &[SourceGraph] {
        &self.graphs
    }

    /// Select one graph for compilation.
    pub fn select(mut self, options: &SourceParseOptions) -> Result<SourceProgram, CompileError> {
        let index = self
            .selected_index(options)
            .map_err(CompileError::SourceParse)?;
        Ok(self.graphs.remove(index).program)
    }

    /// Select one graph for compilation with a source diagnostic.
    pub fn select_diagnostic(
        mut self,
        options: &SourceParseOptions,
    ) -> Result<SourceProgram, SourceDiagnostic> {
        let index = self
            .selected_index(options)
            .map_err(SourceDiagnostic::global)?;
        Ok(self.graphs.remove(index).program)
    }

    fn selected_index(&self, options: &SourceParseOptions) -> Result<usize, &'static str> {
        match options.graph_name() {
            Some(name) => self.selected_named_index(name),
            None => self.selected_default_index(),
        }
    }

    fn selected_default_index(&self) -> Result<usize, &'static str> {
        match self.graphs.len() {
            0 => Err("source graph missing"),
            1 => Ok(0),
            _ => Err("source graph ambiguous"),
        }
    }

    fn selected_named_index(&self, name: &str) -> Result<usize, &'static str> {
        let mut matched = self
            .graphs
            .iter()
            .enumerate()
            .filter(|(_, graph)| graph.name.as_deref() == Some(name))
            .map(|(index, _)| index);
        let Some(index) = matched.next() else {
            return Err("source graph not found");
        };
        if matched.next().is_some() {
            return Err("source graph ambiguous");
        }
        Ok(index)
    }
}
