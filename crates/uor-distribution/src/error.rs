//! The κ-Distribution error-code taxonomy (spec §6.16).
//!
//! Error responses carry `{"errors":[{"code":<UPPERCASE_UNDERSCORE>,"message":…,"detail":…}]}`; the
//! `code` is one of these tokens. The set is closed here so every conforming registry emits the same
//! codes for the same conditions.

/// A defined κ-Distribution error code (spec §6.16). [`ErrorCode::as_str`] yields the exact wire
/// token (uppercase ASCII letters and underscores only).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ErrorCode {
    /// Blob not found.
    BlobUnknown,
    /// Upload session invalid.
    BlobUploadInvalid,
    /// Upload session not found.
    BlobUploadUnknown,
    /// κ-label does not match content.
    DigestInvalid,
    /// Content fails schema validation.
    SchemaViolation,
    /// Admission filter rejected content.
    FilterRejected,
    /// Admission filter execution failed.
    FilterFailed,
    /// Tag not found.
    TagUnknown,
    /// `tag_set` for absent κ-label.
    TagContentAbsent,
    /// `tag_set_if` expected-value mismatch.
    TagConflict,
    /// Edge not found.
    EdgeUnknown,
    /// Edge source κ-label absent.
    EdgeSourceAbsent,
    /// Composition operands differ in σ-axis.
    AxisMismatch,
    /// Invalid resource path.
    NameInvalid,
    /// Authentication required.
    Unauthorized,
    /// Access denied.
    Denied,
    /// Operation not supported.
    Unsupported,
    /// Rate limit exceeded.
    TooManyRequests,
    /// Unpin blocked by an outstanding finalizer.
    FinalizerOutstanding,
    /// Content length mismatch.
    SizeInvalid,
}

impl ErrorCode {
    /// The exact uppercase-underscore wire token for this code (spec §6.16).
    pub const fn as_str(self) -> &'static str {
        match self {
            ErrorCode::BlobUnknown => "BLOB_UNKNOWN",
            ErrorCode::BlobUploadInvalid => "BLOB_UPLOAD_INVALID",
            ErrorCode::BlobUploadUnknown => "BLOB_UPLOAD_UNKNOWN",
            ErrorCode::DigestInvalid => "DIGEST_INVALID",
            ErrorCode::SchemaViolation => "SCHEMA_VIOLATION",
            ErrorCode::FilterRejected => "FILTER_REJECTED",
            ErrorCode::FilterFailed => "FILTER_FAILED",
            ErrorCode::TagUnknown => "TAG_UNKNOWN",
            ErrorCode::TagContentAbsent => "TAG_CONTENT_ABSENT",
            ErrorCode::TagConflict => "TAG_CONFLICT",
            ErrorCode::EdgeUnknown => "EDGE_UNKNOWN",
            ErrorCode::EdgeSourceAbsent => "EDGE_SOURCE_ABSENT",
            ErrorCode::AxisMismatch => "AXIS_MISMATCH",
            ErrorCode::NameInvalid => "NAME_INVALID",
            ErrorCode::Unauthorized => "UNAUTHORIZED",
            ErrorCode::Denied => "DENIED",
            ErrorCode::Unsupported => "UNSUPPORTED",
            ErrorCode::TooManyRequests => "TOOMANYREQUESTS",
            ErrorCode::FinalizerOutstanding => "FINALIZER_OUTSTANDING",
            ErrorCode::SizeInvalid => "SIZE_INVALID",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_are_spec_wire_tokens() {
        // Spot-check the exact §6.16 tokens, including the underscore-free TOOMANYREQUESTS.
        assert_eq!(ErrorCode::DigestInvalid.as_str(), "DIGEST_INVALID");
        assert_eq!(ErrorCode::TagContentAbsent.as_str(), "TAG_CONTENT_ABSENT");
        assert_eq!(ErrorCode::EdgeSourceAbsent.as_str(), "EDGE_SOURCE_ABSENT");
        assert_eq!(ErrorCode::AxisMismatch.as_str(), "AXIS_MISMATCH");
        assert_eq!(ErrorCode::TooManyRequests.as_str(), "TOOMANYREQUESTS");
        assert_eq!(ErrorCode::FinalizerOutstanding.as_str(), "FINALIZER_OUTSTANDING");
    }

    #[test]
    fn codes_are_uppercase_underscore_only() {
        // Every code MUST be uppercase ASCII letters + underscores (spec §6.16).
        let all = [
            ErrorCode::BlobUnknown,
            ErrorCode::BlobUploadInvalid,
            ErrorCode::BlobUploadUnknown,
            ErrorCode::DigestInvalid,
            ErrorCode::SchemaViolation,
            ErrorCode::FilterRejected,
            ErrorCode::FilterFailed,
            ErrorCode::TagUnknown,
            ErrorCode::TagContentAbsent,
            ErrorCode::TagConflict,
            ErrorCode::EdgeUnknown,
            ErrorCode::EdgeSourceAbsent,
            ErrorCode::AxisMismatch,
            ErrorCode::NameInvalid,
            ErrorCode::Unauthorized,
            ErrorCode::Denied,
            ErrorCode::Unsupported,
            ErrorCode::TooManyRequests,
            ErrorCode::FinalizerOutstanding,
            ErrorCode::SizeInvalid,
        ];
        for code in all {
            let s = code.as_str();
            assert!(!s.is_empty());
            assert!(
                s.bytes().all(|b| b.is_ascii_uppercase() || b == b'_'),
                "{s} must be uppercase/underscore only"
            );
        }
    }
}
