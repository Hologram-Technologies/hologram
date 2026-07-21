//! Level 4 — composition (spec §5.4 / §6.9) + witnesses (§3.6 / §6.8).
//!
//! A composition applies one of the five categorical operations (`g2`/`f4`/`e6`/`e7`/`e8`) to operand
//! κ-labels. The canonical form is the **shared** [`uor_distribution::compose_canonical`]; the composed
//! κ is its hash under the operands' shared σ-axis (§5.4, Appendix B), and the composed blob is the
//! canonical form itself (so it is retrievable + verifiable by the composed κ). A §3.6 witness blob
//! records the derivation; provenance is captured as `composed-of` edges (composed → each operand,
//! op-type in the edge metadata for the Appendix D.6 disambiguation) and a `witness-of` edge.

use std::net::TcpStream;

use hologram_space::{address_bytes_axis, KappaStore};
use uor_distribution::{compose_canonical, witness_blob, ComposeError, ErrorCode, Op};

use crate::http_util::{
    json_string_array, load, parse71, parse_request, read_body, write_error, write_resp,
};

/// `POST /v2/{path}/compose/{op}` — apply a categorical composition to the operand κ-labels in the
/// request body `{"operands":[...]}` (§5.4, §6.9).
pub fn handle_compose(
    stream: &mut TcpStream,
    head: &[u8],
    store: &dyn KappaStore,
) -> std::io::Result<()> {
    let (method, path, _query, content_length, split) = parse_request(head);
    if method != "POST" {
        return write_resp(stream, 405, "Method Not Allowed", &[("Content-Length", "0")], b"");
    }
    let Some(cpos) = path.find("/compose/") else {
        return write_error(stream, 404, "Not Found", ErrorCode::NameInvalid, "not a compose route");
    };
    let prefix = &path[4..cpos];
    let op_token = &path[cpos + "/compose/".len()..];
    let Some(op) = Op::parse(op_token) else {
        return write_error(stream, 400, "Bad Request", ErrorCode::NameInvalid, "unknown operation");
    };

    let body = read_body(stream, head, split, content_length)?;
    let bs = String::from_utf8_lossy(&body);
    let operand_strs = json_string_array(&bs, "operands");
    if operand_strs.is_empty() {
        return write_error(stream, 400, "Bad Request", ErrorCode::NameInvalid, "no operands");
    }
    let mut operands = Vec::new();
    for s in &operand_strs {
        match parse71(s) {
            Some(k) => operands.push(k),
            None => {
                return write_error(
                    stream,
                    400,
                    "Bad Request",
                    ErrorCode::NameInvalid,
                    "malformed operand κ",
                );
            }
        }
    }
    let operand_bytes: Vec<&[u8]> = operands.iter().map(|k| k.as_bytes()).collect();
    let canon = match compose_canonical(op, &operand_bytes) {
        Ok(c) => c,
        // σ-axis homogeneity (§5.4) → 422 AXIS_MISMATCH.
        Err(ComposeError::AxisMismatch) => {
            return write_error(
                stream,
                422,
                "Unprocessable Entity",
                ErrorCode::AxisMismatch,
                "composition operands differ in σ-axis",
            );
        }
        Err(_) => {
            return write_error(
                stream,
                400,
                "Bad Request",
                ErrorCode::NameInvalid,
                "bad operand count for operation",
            );
        }
    };
    let axis = operands[0].sigma_axis().unwrap_or("blake3");

    // Composed blob = the canonical form; composed κ = hash(canon) under the axis (Appendix B).
    let Ok(composed_kappa) = address_bytes_axis(axis, &canon) else {
        return write_error(stream, 400, "Bad Request", ErrorCode::NameInvalid, "σ-axis");
    };
    let _ = store.put_axis(axis, &canon);
    let composed_str = String::from_utf8_lossy(&composed_kappa).to_string();

    // Witness blob (§3.6): replaying its trace re-derives the composed κ. Store it.
    let witness = witness_blob(71, 32, &canon);
    let Ok(witness_kappa) = address_bytes_axis(axis, &witness) else {
        return write_error(stream, 400, "Bad Request", ErrorCode::NameInvalid, "σ-axis");
    };
    let _ = store.put_axis(axis, &witness);
    let witness_str = String::from_utf8_lossy(&witness_kappa).to_string();

    // Provenance edges: composed-of (composed → each operand, op-type in metadata) + witness-of.
    if let (Some(composed_k), Some(witness_k)) = (parse71(&composed_str), parse71(&witness_str)) {
        for operand in &operands {
            let _ = crate::edge::record(
                store,
                prefix,
                &composed_k,
                "composed-of",
                operand,
                op.as_str().as_bytes(),
            );
        }
        let _ = crate::edge::record(store, prefix, &witness_k, "witness-of", &composed_k, b"");
    }

    let operands_json = operand_strs
        .iter()
        .map(|s| format!("\"{s}\""))
        .collect::<Vec<_>>()
        .join(",");
    let out = format!(
        r#"{{"composed":"{composed_str}","witness":"{witness_str}","operands":[{operands_json}],"operation":"{}"}}"#,
        op.as_str()
    );
    write_resp(
        stream,
        200,
        "OK",
        &[
            ("Content-Type", "application/json"),
            ("Content-Length", &out.len().to_string()),
        ],
        out.as_bytes(),
    )
}

/// `GET /v2/{path}/witnesses/{κ}` — the witness blob attesting `κ` (§6.8), resolved via `κ`'s inbound
/// `witness-of` edge.
pub fn handle_witness(
    stream: &mut TcpStream,
    head: &[u8],
    store: &dyn KappaStore,
) -> std::io::Result<()> {
    let (_method, path, _query, _cl, _split) = parse_request(head);
    let Some(wpos) = path.find("/witnesses/") else {
        return write_error(stream, 404, "Not Found", ErrorCode::NameInvalid, "not a witness route");
    };
    let prefix = &path[4..wpos];
    let kappa = &path[wpos + "/witnesses/".len()..];
    let Some(witness_str) = crate::edge::find_witness_source(prefix, kappa) else {
        return write_error(stream, 404, "Not Found", ErrorCode::BlobUnknown, "no witness for κ");
    };
    let Some(wk) = parse71(&witness_str) else {
        return write_error(stream, 404, "Not Found", ErrorCode::BlobUnknown, "bad witness κ");
    };
    match load(store, &wk) {
        Some(b) => write_resp(
            stream,
            200,
            "OK",
            &[
                ("Content-Length", &b.as_ref().len().to_string()),
                ("X-Kappa-Label", &witness_str),
                ("Content-Type", "application/octet-stream"),
            ],
            b.as_ref(),
        ),
        None => write_error(stream, 404, "Not Found", ErrorCode::BlobUnknown, "witness content absent"),
    }
}
