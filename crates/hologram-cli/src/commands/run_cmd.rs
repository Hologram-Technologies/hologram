//! `hologram run` — execute a `.holo` file.

use crate::error::CliError;
use crate::fmt::format_bytes;
use clap::Args;
use hologram_archive::entrypoint::TensorPort;
use hologram_archive::section::model_meta::{ModelMetaSection, SECTION_MODEL_META};
use hologram_archive::section::tokenizer::{MiniBpeEncoder, TokenizerSection, SECTION_TOKENIZER};
use hologram_archive::weight::WeightDType;
use hologram_archive::{HoloLoader, LoadedPlan};
#[allow(deprecated)]
use hologram_exec::{build_schedule, GraphInputs, GraphOutputs, KvExecutor};
use std::io::Write;
use std::path::PathBuf;

/// Arguments for the run subcommand.
#[derive(Args)]
pub struct RunArgs {
    /// Path to the `.holo` file to execute.
    pub file: PathBuf,
    /// Input values as `INDEX:HEX` pairs (e.g. `--input 0:deadbeef`).
    #[arg(long = "input", value_name = "INDEX:HEX")]
    pub inputs: Vec<String>,
    /// Input from file as `SLOT:PATH` pairs (e.g. `--input-file 0:input.bin`).
    #[arg(long = "input-file", value_name = "SLOT:PATH")]
    pub input_files: Vec<String>,
    /// Text prompt for autoregressive generation (requires embedded tokenizer).
    #[arg(long)]
    pub prompt: Option<String>,
    /// Maximum tokens to generate when using --prompt (default: 128).
    #[arg(long, default_value = "128")]
    pub max_tokens: usize,
}

/// Execute the run command.
#[allow(deprecated)]
pub async fn execute(args: RunArgs) -> Result<(), CliError> {
    let loader = HoloLoader::open(&args.file)?;
    let plan = loader.load()?;
    let archive_bytes = loader.as_bytes();

    // Load optional sections.
    let tokenizer = load_section::<TokenizerSection>(archive_bytes, &plan, SECTION_TOKENIZER);
    let model_meta = load_section::<ModelMetaSection>(archive_bytes, &plan, SECTION_MODEL_META);

    print_model_info(&plan, &model_meta);

    if let Some(prompt) = &args.prompt {
        // Guard: check model supports prompt generation.
        if let Some(meta) = &model_meta {
            if !meta.supports_prompt {
                return Err(CliError::InvalidInput(format!(
                    "model kind {:?} does not support --prompt (arch: {})",
                    meta.kind, meta.arch,
                )));
            }
        }
        let tok = tokenizer.as_ref().ok_or_else(|| {
            CliError::InvalidInput(
                "archive has no embedded tokenizer section; \
                 recompile with --tokenizer to use --prompt"
                    .into(),
            )
        })?;
        run_generation(&plan, tok, prompt, args.max_tokens)?;
    } else {
        let mut graph_inputs = parse_inputs(&args.inputs)?;
        load_file_inputs(&args.input_files, &mut graph_inputs)?;

        // Show help if no inputs provided.
        if args.inputs.is_empty() && args.input_files.is_empty() {
            print_input_help(&plan);
        }

        let start = std::time::Instant::now();
        let outputs = hologram_exec::execute_plan(&plan, &graph_inputs)?;
        let elapsed = start.elapsed();

        if let Some(tok) = &tokenizer {
            print_decoded_outputs(&outputs, tok);
        } else {
            print_typed_outputs(&outputs, &plan);
        }
        eprintln!(
            "executed in {:.3}ms (weights {})",
            elapsed.as_secs_f64() * 1000.0,
            format_bytes(plan.weights().len() as u64),
        );
    }
    Ok(())
}

// ── Model info ─────────────────────────────────────────────────────────

/// Print model metadata and entrypoint info to stderr.
fn print_model_info(plan: &LoadedPlan, model_meta: &Option<ModelMetaSection>) {
    if let Some(meta) = model_meta {
        eprintln!(
            "model: {:?} arch={} seq_len={} prompt={}",
            meta.kind, meta.arch, meta.max_seq_len, meta.supports_prompt,
        );
        if !meta.description.is_empty() {
            eprintln!("  {}", meta.description);
        }
    }

    let lh = match plan.layer_header() {
        Some(lh) => lh,
        None => {
            eprintln!("no layer header; using direct graph execution");
            return;
        }
    };
    for layer in &lh.layers {
        let inputs: Vec<String> = layer
            .inputs
            .iter()
            .map(|p| format!("{}:{:?}:{}", p.name, p.shape, p.dtype.name()))
            .collect();
        let outputs: Vec<String> = layer
            .outputs
            .iter()
            .map(|p| format!("{}:{:?}:{}", p.name, p.shape, p.dtype.name()))
            .collect();
        eprintln!(
            "layer {:?}: {:?} [{}] -> [{}]",
            layer.name,
            layer.entrypoint,
            inputs.join(", "),
            outputs.join(", "),
        );
    }
}

/// Print expected input specs to help users understand what the model needs.
fn print_input_help(plan: &LoadedPlan) {
    let lh = match plan.layer_header() {
        Some(lh) => lh,
        None => {
            // Fall back to graph input names.
            eprintln!("inputs (by graph name):");
            for (i, name) in plan.graph().input_names.iter().enumerate() {
                eprintln!("  slot {i}: \"{name}\"");
            }
            return;
        }
    };
    eprintln!("expected inputs:");
    for layer in &lh.layers {
        for (i, port) in layer.inputs.iter().enumerate() {
            let elem_bytes = port.dtype.byte_size();
            let total_elems: u64 = port.shape.iter().product();
            let total_bytes = if elem_bytes > 0 && total_elems > 0 {
                format!("{} bytes", total_elems as usize * elem_bytes)
            } else {
                "dynamic".into()
            };
            eprintln!(
                "  slot {i} '{}': shape {:?} dtype {} ({})",
                port.name,
                port.shape,
                port.dtype.name(),
                total_bytes,
            );
        }
    }
}

// ── Input parsing ──────────────────────────────────────────────────────

/// Parse a list of `INDEX:HEX` strings into `GraphInputs`.
fn parse_inputs(raw: &[String]) -> Result<GraphInputs, CliError> {
    let mut inputs = GraphInputs::new();
    for s in raw {
        let (idx, bytes) = parse_input(s)?;
        inputs.set(idx, bytes);
    }
    Ok(inputs)
}

/// Parse a single `INDEX:HEX` string.
pub fn parse_input(s: &str) -> Result<(u32, Vec<u8>), CliError> {
    let (idx_str, hex_str) = s
        .split_once(':')
        .ok_or_else(|| CliError::InvalidInput(format!("expected INDEX:HEX, got {s:?}")))?;
    let idx: u32 = idx_str
        .parse()
        .map_err(|_| CliError::InvalidInput(format!("invalid index {idx_str:?} in {s:?}")))?;
    let bytes = decode_hex(hex_str).map_err(CliError::InvalidInput)?;
    Ok((idx, bytes))
}

/// Load file-based inputs (`SLOT:PATH` pairs).
fn load_file_inputs(raw: &[String], inputs: &mut GraphInputs) -> Result<(), CliError> {
    for s in raw {
        let (idx_str, path_str) = s
            .split_once(':')
            .ok_or_else(|| CliError::InvalidInput(format!("expected SLOT:PATH, got {s:?}")))?;
        let idx: u32 = idx_str
            .parse()
            .map_err(|_| CliError::InvalidInput(format!("invalid slot {idx_str:?} in {s:?}")))?;
        let bytes = std::fs::read(path_str)
            .map_err(|e| CliError::InvalidInput(format!("reading input file {path_str:?}: {e}")))?;
        eprintln!(
            "loaded slot {idx} from {path_str:?} ({} bytes)",
            bytes.len()
        );
        inputs.set(idx, bytes);
    }
    Ok(())
}

/// Decode a hex string into bytes.
fn decode_hex(s: &str) -> Result<Vec<u8>, String> {
    if !s.len().is_multiple_of(2) {
        return Err(format!("odd-length hex string: {s:?}"));
    }
    (0..s.len())
        .step_by(2)
        .map(|i| {
            u8::from_str_radix(&s[i..i + 2], 16)
                .map_err(|_| format!("invalid hex byte {:?}", &s[i..i + 2]))
        })
        .collect()
}

// ── Section loading ────────────────────────────────────────────────────

/// Generic section loader — try to load and deserialize a section by kind.
fn load_section<T>(archive_bytes: &[u8], plan: &LoadedPlan, kind: u32) -> Option<T>
where
    T: SectionDeserialize,
{
    let entry = plan.sections().find(kind)?;
    let offset = entry.offset as usize;
    let size = entry.size as usize;
    if offset + size > archive_bytes.len() {
        return None;
    }
    T::deserialize_section(&archive_bytes[offset..offset + size]).ok()
}

/// Trait for section types that can be deserialized from bytes.
trait SectionDeserialize: Sized {
    fn deserialize_section(bytes: &[u8]) -> Result<Self, rkyv::rancor::Error>;
}

impl SectionDeserialize for TokenizerSection {
    fn deserialize_section(bytes: &[u8]) -> Result<Self, rkyv::rancor::Error> {
        TokenizerSection::deserialize_from(bytes)
    }
}

impl SectionDeserialize for ModelMetaSection {
    fn deserialize_section(bytes: &[u8]) -> Result<Self, rkyv::rancor::Error> {
        ModelMetaSection::deserialize_from(bytes)
    }
}

// ── Output formatting ──────────────────────────────────────────────────

/// Get output TensorPort specs from the LayerHeader (if available).
fn get_output_ports(plan: &LoadedPlan) -> Vec<TensorPort> {
    plan.layer_header()
        .into_iter()
        .flat_map(|lh| lh.layers.iter())
        .flat_map(|l| l.outputs.iter().cloned())
        .collect()
}

/// Print outputs with dtype-aware formatting.
fn print_typed_outputs(outputs: &GraphOutputs, plan: &LoadedPlan) {
    let output_ports = get_output_ports(plan);
    for i in 0..outputs.len() {
        if let Some((name, data)) = outputs.get(i) {
            let dtype = output_ports.get(i).map(|p| p.dtype);
            match dtype {
                Some(WeightDType::F32) if data.len() >= 4 => {
                    let n = data.len() / 4;
                    let floats: Vec<f32> = (0..n)
                        .map(|j| f32::from_le_bytes(data[j * 4..(j + 1) * 4].try_into().unwrap()))
                        .collect();
                    if floats.len() <= 16 {
                        println!("{name}: {floats:?}");
                    } else {
                        let min = floats.iter().copied().reduce(f32::min).unwrap_or(0.0);
                        let max = floats.iter().copied().reduce(f32::max).unwrap_or(0.0);
                        let mean = floats.iter().sum::<f32>() / floats.len() as f32;
                        println!(
                            "{name}: [{} f32] min={min:.4} max={max:.4} mean={mean:.4}",
                            floats.len(),
                        );
                    }
                }
                Some(WeightDType::F64) if data.len() >= 8 => {
                    let n = data.len() / 8;
                    let floats: Vec<f64> = (0..n)
                        .map(|j| f64::from_le_bytes(data[j * 8..(j + 1) * 8].try_into().unwrap()))
                        .collect();
                    if floats.len() <= 16 {
                        println!("{name}: {floats:?}");
                    } else {
                        println!("{name}: [{} f64 values]", floats.len());
                    }
                }
                Some(WeightDType::I64) if data.len() >= 8 => {
                    let n = data.len() / 8;
                    let ints: Vec<i64> = (0..n)
                        .map(|j| i64::from_le_bytes(data[j * 8..(j + 1) * 8].try_into().unwrap()))
                        .collect();
                    if ints.len() <= 32 {
                        println!("{name}: {ints:?}");
                    } else {
                        println!("{name}: [{} i64 values]", ints.len());
                    }
                }
                Some(WeightDType::I32) if data.len() >= 4 => {
                    let n = data.len() / 4;
                    let ints: Vec<i32> = (0..n)
                        .map(|j| i32::from_le_bytes(data[j * 4..(j + 1) * 4].try_into().unwrap()))
                        .collect();
                    if ints.len() <= 32 {
                        println!("{name}: {ints:?}");
                    } else {
                        println!("{name}: [{} i32 values]", ints.len());
                    }
                }
                _ => {
                    // Fallback: truncated hex with byte count.
                    let hex: String = data.iter().take(64).map(|b| format!("{b:02x}")).collect();
                    let suffix = if data.len() > 64 { "..." } else { "" };
                    println!("{name}: {hex}{suffix} ({} bytes)", data.len());
                }
            }
        }
    }
}

/// Print outputs decoded as text tokens via argmax.
fn print_decoded_outputs(outputs: &GraphOutputs, tok: &TokenizerSection) {
    for i in 0..outputs.len() {
        if let Some((name, data)) = outputs.get(i) {
            if let Some(token_id) = TokenizerSection::argmax_f32(data) {
                let text = tok.id_to_token(token_id).unwrap_or("<unk>");
                println!("{name}: token_id={token_id} \"{text}\"");
            } else {
                let hex: String = data.iter().take(64).map(|b| format!("{b:02x}")).collect();
                let suffix = if data.len() > 64 { "..." } else { "" };
                println!("{name}: {hex}{suffix} ({} bytes)", data.len());
            }
        }
    }
}

// ── Autoregressive generation ──────────────────────────────────────────

/// Autoregressive text generation loop.
#[allow(deprecated)]
fn run_generation(
    plan: &LoadedPlan,
    tok_section: &TokenizerSection,
    prompt: &str,
    max_tokens: usize,
) -> Result<(), CliError> {
    let encoder = MiniBpeEncoder::from_tokenizer_section(tok_section);

    // Determine input dtype for token IDs from TensorPort (default I64).
    let input_dtype = resolve_input_dtype(plan, "input_ids");

    // Encode the prompt.
    let mut token_ids = encoder.encode(prompt);
    eprintln!(
        "prompt: {} tokens (vocab_size={}, input_dtype={})",
        token_ids.len(),
        encoder.vocab_size(),
        input_dtype.name(),
    );

    // Find input slot for "input_ids" (default slot 0).
    let input_slot = plan
        .graph()
        .input_names
        .iter()
        .position(|n| n == "input_ids")
        .unwrap_or(0) as u32;

    // Check if model needs attention_mask.
    let mask_slot = plan
        .graph()
        .input_names
        .iter()
        .position(|n| n == "attention_mask")
        .map(|i| i as u32);
    let mask_dtype = mask_slot.map(|_| resolve_input_dtype(plan, "attention_mask"));

    // Build execution schedule once.
    let schedule = build_schedule(plan.graph())?;

    // Get compiled input sequence length from TensorPort shape.
    // Static-shape ONNX models (no KV-cache) require inputs padded to the
    // compiled seq_len. The model will process the full padded sequence and
    // we extract logits from the last real-token position.
    let compiled_seq_len: Option<usize> = plan
        .layer_header()
        .into_iter()
        .flat_map(|lh| lh.layers.iter())
        .flat_map(|l| l.inputs.iter())
        .find(|p| p.name == "input_ids")
        .and_then(|p| p.shape.get(1).copied())
        .filter(|&s| s > 0)
        .map(|s| s as usize);

    let start = std::time::Instant::now();
    let prompt_len = token_ids.len();

    for step in 0..max_tokens {
        // For static-shape models, pad token IDs to the compiled sequence length.
        // The attention mask distinguishes real tokens (1) from padding (0).
        let effective_tokens = if let Some(max_seq) = compiled_seq_len {
            if token_ids.len() > max_seq {
                // Truncate to max sequence length (take the last max_seq tokens).
                token_ids[token_ids.len() - max_seq..].to_vec()
            } else {
                // Pad with 0s to fill the compiled sequence length.
                let mut padded = token_ids.clone();
                padded.resize(max_seq, 0);
                padded
            }
        } else {
            token_ids.clone()
        };

        let actual_len = token_ids
            .len()
            .min(compiled_seq_len.unwrap_or(token_ids.len()));
        let padded_len = effective_tokens.len();

        // Build inputs: token IDs serialized per input dtype.
        let input_bytes = serialize_token_ids(&effective_tokens, input_dtype);

        let mut inputs = GraphInputs::new();
        inputs.set_with_shape(input_slot, input_bytes, vec![1, padded_len]);

        // Add attention mask: 1 for real tokens, 0 for padding.
        if let Some(slot) = mask_slot {
            let mask_dtype_val = mask_dtype.unwrap_or(WeightDType::I64);
            let mask_bytes = if compiled_seq_len.is_some() {
                serialize_mask(actual_len, padded_len, mask_dtype_val)
            } else {
                serialize_ones(token_ids.len(), mask_dtype_val)
            };
            inputs.set_with_shape(slot, mask_bytes, vec![1, padded_len]);
        }

        let outputs =
            KvExecutor::execute_with_plan(plan.graph(), &schedule, &inputs, plan.weights())?;

        // Argmax over the last real-token position's logits.
        // For padded inputs, logits are [1, padded_len, vocab_size] and we want
        // position (actual_len - 1), not the last padded position.
        let logit_data = match outputs.get(0) {
            Some((_, data)) => data,
            None => {
                return Err(CliError::InvalidInput("model produced no output".into()));
            }
        };

        let vocab_size = encoder.vocab_size();
        let bytes_per_pos = vocab_size * 4; // f32 = 4 bytes
        let target_pos = actual_len.saturating_sub(1);

        // Debug: inspect logits at target position.
        if step == 0 {
            let offset = target_pos * bytes_per_pos;
            if logit_data.len() >= offset + bytes_per_pos {
                let slice = &logit_data[offset..offset + bytes_per_pos];
                let floats: Vec<f32> = slice
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                let nan_count = floats.iter().filter(|f| f.is_nan()).count();
                let inf_count = floats.iter().filter(|f| f.is_infinite()).count();
                let zero_count = floats.iter().filter(|&&f| f == 0.0).count();
                let min = floats.iter().copied().reduce(f32::min).unwrap_or(0.0);
                let max = floats.iter().copied().reduce(f32::max).unwrap_or(0.0);
                let mean = floats.iter().sum::<f32>() / floats.len() as f32;
                eprintln!(
                    "[logit-debug] pos={target_pos} vocab={vocab_size} total_bytes={} nan={nan_count} inf={inf_count} zero={zero_count} min={min:.4} max={max:.4} mean={mean:.6}",
                    logit_data.len()
                );
                // Show top-5 tokens
                let mut indexed: Vec<(usize, f32)> = floats.iter().copied().enumerate().collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                for (i, (tok_id, val)) in indexed.iter().take(5).enumerate() {
                    let tok_str = tok_section.id_to_token(*tok_id as u32).unwrap_or("<unk>");
                    eprintln!(
                        "[logit-debug] top-{}: id={tok_id} val={val:.6} \"{tok_str}\"",
                        i + 1
                    );
                }
            }
        }

        let next_token =
            if compiled_seq_len.is_some() && logit_data.len() >= (target_pos + 1) * bytes_per_pos {
                // Extract logits at the last real-token position.
                let offset = target_pos * bytes_per_pos;
                let logits_slice = &logit_data[offset..offset + bytes_per_pos];
                argmax_with_repetition_penalty(logits_slice, &token_ids)
            } else if logit_data.len() >= bytes_per_pos {
                let last_logits = &logit_data[logit_data.len() - bytes_per_pos..];
                argmax_with_repetition_penalty(last_logits, &token_ids)
            } else {
                argmax_with_repetition_penalty(logit_data, &token_ids)
            };

        let next_token = match next_token {
            Some(id) => id,
            None => {
                eprintln!("\n[no logits in output]");
                break;
            }
        };

        // Check EOS.
        if next_token == encoder.eos_id() {
            break;
        }

        // Decode and print incrementally.
        let text = encoder.decode(&[next_token]);
        print!("{text}");
        std::io::stdout().flush().ok();

        token_ids.push(next_token);

        if step == 0 {
            let prefill_ms = start.elapsed().as_secs_f64() * 1000.0;
            eprintln!("\n[prefill {prefill_ms:.0}ms]");
        }
    }

    let elapsed = start.elapsed();
    let generated = token_ids.len() - prompt_len;
    let tok_per_sec = if elapsed.as_secs_f64() > 0.0 {
        generated as f64 / elapsed.as_secs_f64()
    } else {
        0.0
    };
    eprintln!(
        "\n[{generated} tokens in {:.1}s ({tok_per_sec:.1} tok/s), weights {}]",
        elapsed.as_secs_f64(),
        format_bytes(plan.weights().len() as u64),
    );
    Ok(())
}

/// Argmax over raw f32 logit bytes with repetition penalty.
///
/// Applies a standard repetition penalty (Keskar et al., 2019) to any token
/// that already appears in `generated`. Tokens with positive logits are divided
/// by the penalty; tokens with negative logits are multiplied by it. This
/// discourages the model from looping on previously generated tokens.
///
/// Penalty of 1.0 is equivalent to plain argmax. Recommended: 1.2–1.4.
fn argmax_with_repetition_penalty(logit_bytes: &[u8], generated: &[u32]) -> Option<u32> {
    const PENALTY: f32 = 1.3;
    // How far back to look for repeated tokens.
    const WINDOW: usize = 64;

    if !logit_bytes.len().is_multiple_of(4) {
        return None;
    }
    let mut logits: Vec<f32> = logit_bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().expect("chunk is 4 bytes")))
        .collect();

    let start = generated.len().saturating_sub(WINDOW);
    for &tok in &generated[start..] {
        let idx = tok as usize;
        if idx < logits.len() {
            if logits[idx] > 0.0 {
                logits[idx] /= PENALTY;
            } else {
                logits[idx] *= PENALTY;
            }
        }
    }

    logits
        .iter()
        .enumerate()
        .filter(|(_, v)| v.is_finite())
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
}

/// Resolve the dtype of a named input port from the LayerHeader.
///
/// Falls back to I64 if no port is found or the port has a placeholder
/// dtype (U8 with shape [1] is the compiler's default placeholder).
fn resolve_input_dtype(plan: &LoadedPlan, name: &str) -> WeightDType {
    plan.layer_header()
        .into_iter()
        .flat_map(|lh| lh.layers.iter())
        .flat_map(|l| l.inputs.iter())
        .find(|p| p.name == name)
        .map(|p| {
            // U8 with shape [1] is the compiler's placeholder — treat as unknown.
            if p.dtype == WeightDType::U8 && p.shape == [1] {
                WeightDType::I64
            } else {
                p.dtype
            }
        })
        .unwrap_or(WeightDType::I64)
}

/// Serialize token IDs to bytes in the given dtype.
fn serialize_token_ids(ids: &[u32], dtype: WeightDType) -> Vec<u8> {
    match dtype {
        WeightDType::I32 => ids
            .iter()
            .flat_map(|&id| (id as i32).to_le_bytes())
            .collect(),
        // Default to I64 for all other dtypes.
        _ => ids
            .iter()
            .flat_map(|&id| (id as i64).to_le_bytes())
            .collect(),
    }
}

/// Serialize N ones in the given dtype (for attention masks).
fn serialize_ones(n: usize, dtype: WeightDType) -> Vec<u8> {
    match dtype {
        WeightDType::I32 => (0..n).flat_map(|_| 1i32.to_le_bytes()).collect(),
        WeightDType::F32 => (0..n).flat_map(|_| 1.0f32.to_le_bytes()).collect(),
        _ => (0..n).flat_map(|_| 1i64.to_le_bytes()).collect(),
    }
}

/// Serialize an attention mask: 1 for real tokens, 0 for padding.
fn serialize_mask(real_len: usize, total_len: usize, dtype: WeightDType) -> Vec<u8> {
    match dtype {
        WeightDType::I32 => (0..total_len)
            .flat_map(|i| {
                if i < real_len {
                    1i32.to_le_bytes()
                } else {
                    0i32.to_le_bytes()
                }
            })
            .collect(),
        WeightDType::F32 => (0..total_len)
            .flat_map(|i| {
                if i < real_len {
                    1.0f32.to_le_bytes()
                } else {
                    0.0f32.to_le_bytes()
                }
            })
            .collect(),
        _ => (0..total_len)
            .flat_map(|i| {
                if i < real_len {
                    1i64.to_le_bytes()
                } else {
                    0i64.to_le_bytes()
                }
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_input_valid() {
        let (idx, bytes) = parse_input("0:deadbeef").unwrap();
        assert_eq!(idx, 0);
        assert_eq!(bytes, vec![0xde, 0xad, 0xbe, 0xef]);
    }

    #[test]
    fn parse_input_single_byte() {
        let (idx, bytes) = parse_input("2:ff").unwrap();
        assert_eq!(idx, 2);
        assert_eq!(bytes, vec![0xff]);
    }

    #[test]
    fn parse_input_large_index() {
        let (idx, bytes) = parse_input("255:0102").unwrap();
        assert_eq!(idx, 255);
        assert_eq!(bytes, vec![0x01, 0x02]);
    }

    #[test]
    fn parse_input_missing_colon() {
        assert!(matches!(
            parse_input("0deadbeef"),
            Err(CliError::InvalidInput(_))
        ));
    }

    #[test]
    fn parse_input_invalid_index() {
        assert!(matches!(
            parse_input("abc:deadbeef"),
            Err(CliError::InvalidInput(_))
        ));
    }

    #[test]
    fn parse_input_invalid_hex() {
        assert!(matches!(
            parse_input("0:xyz"),
            Err(CliError::InvalidInput(_))
        ));
    }

    #[test]
    fn parse_input_odd_hex() {
        assert!(matches!(
            parse_input("0:abc"),
            Err(CliError::InvalidInput(_))
        ));
    }

    #[test]
    fn parse_input_empty_hex() {
        let (idx, bytes) = parse_input("0:").unwrap();
        assert_eq!(idx, 0);
        assert_eq!(bytes, Vec::<u8>::new());
    }

    #[test]
    fn parse_inputs_multiple() {
        let raw = vec!["0:ff".to_string(), "1:0102".to_string()];
        let inputs = parse_inputs(&raw).unwrap();
        assert_eq!(inputs.get(0), Some([0xff].as_slice()));
        assert_eq!(inputs.get(1), Some([0x01, 0x02].as_slice()));
    }

    #[test]
    fn decode_hex_empty() {
        assert_eq!(decode_hex("").unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn decode_hex_all_bytes() {
        let hex: String = (0u8..=255).map(|b| format!("{b:02x}")).collect();
        let result = decode_hex(&hex).unwrap();
        let expected: Vec<u8> = (0u8..=255).collect();
        assert_eq!(result, expected);
    }

    #[test]
    fn serialize_token_ids_i64() {
        let ids = vec![1, 2, 3];
        let bytes = serialize_token_ids(&ids, WeightDType::I64);
        assert_eq!(bytes.len(), 24); // 3 * 8
        assert_eq!(&bytes[0..8], &1i64.to_le_bytes());
    }

    #[test]
    fn serialize_token_ids_i32() {
        let ids = vec![1, 2, 3];
        let bytes = serialize_token_ids(&ids, WeightDType::I32);
        assert_eq!(bytes.len(), 12); // 3 * 4
        assert_eq!(&bytes[0..4], &1i32.to_le_bytes());
    }

    #[test]
    fn serialize_ones_i64() {
        let bytes = serialize_ones(2, WeightDType::I64);
        assert_eq!(bytes.len(), 16);
        let val = i64::from_le_bytes(bytes[0..8].try_into().unwrap());
        assert_eq!(val, 1);
    }

    #[test]
    fn serialize_ones_f32() {
        let bytes = serialize_ones(2, WeightDType::F32);
        assert_eq!(bytes.len(), 8);
        let val = f32::from_le_bytes(bytes[0..4].try_into().unwrap());
        assert_eq!(val, 1.0);
    }
}
