//! `hologram run` — execute a `.holo` file.

use crate::error::CliError;
use crate::fmt::format_bytes;
use clap::Args;
use hologram_archive::HoloLoader;
use hologram_exec::{execute_plan, GraphInputs, GraphOutputs};
use std::path::PathBuf;

/// Arguments for the run subcommand.
#[derive(Args)]
pub struct RunArgs {
    /// Path to the `.holo` file to execute.
    pub file: PathBuf,
    /// Input values as `INDEX:HEX` pairs (e.g. `--input 0:deadbeef`).
    #[arg(long = "input", value_name = "INDEX:HEX")]
    pub inputs: Vec<String>,
}

/// Execute the run command.
pub async fn execute(args: RunArgs) -> Result<(), CliError> {
    let loader = HoloLoader::open(&args.file)?;
    let plan = loader.load()?;

    print_entrypoint_info(&plan);

    let graph_inputs = parse_inputs(&args.inputs)?;
    let start = std::time::Instant::now();
    let outputs = execute_plan(&plan, &graph_inputs)?;
    let elapsed = start.elapsed();

    print_outputs(&outputs);
    eprintln!(
        "executed in {:.3}ms (weights {})",
        elapsed.as_secs_f64() * 1000.0,
        format_bytes(plan.weights().len() as u64),
    );
    Ok(())
}

/// Print entrypoint info from the archive's `LayerHeader` to stderr.
fn print_entrypoint_info(plan: &hologram_archive::LoadedPlan) {
    let lh = match plan.layer_header() {
        Some(lh) => lh,
        None => {
            eprintln!("no layer header; using direct graph execution");
            return;
        }
    };
    for layer in &lh.layers {
        let inputs: Vec<&str> = layer.inputs.iter().map(|p| p.name.as_str()).collect();
        let outputs: Vec<&str> = layer.outputs.iter().map(|p| p.name.as_str()).collect();
        eprintln!(
            "layer {:?}: {:?} [{}] -> [{}]",
            layer.name,
            layer.entrypoint,
            inputs.join(", "),
            outputs.join(", "),
        );
    }
}

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

/// Print each named output as `name: <hex>`.
fn print_outputs(outputs: &GraphOutputs) {
    for i in 0..outputs.len() {
        if let Some((name, data)) = outputs.get(i) {
            let hex: String = data.iter().map(|b| format!("{b:02x}")).collect();
            println!("{name}: {hex}");
        }
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
}
