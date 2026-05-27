//! Port identity (name + shape) and open `Extension` sections (FORMAT_VERSION 2).
//!
//! A `.holo` must preserve what ONNX/GGUF carry: each I/O port's semantic name
//! and full shape, so a multi-input model can be driven by name; and arbitrary
//! producer metadata (tokenizer, generation config, …) attached without the
//! format enumerating every consumer.

use hologram_archive::{decode_ports, HoloLoader, HoloWriter, PortDescriptor, SectionKind};

#[test]
fn ports_round_trip_name_and_shape() {
    let inputs = vec![
        PortDescriptor {
            name: "input_ids".into(),
            slot: 0,
            element_count: 7,
            dtype: 5, // i64
            shape: vec![1, 7],
        },
        PortDescriptor {
            name: "attention_mask".into(),
            slot: 1,
            element_count: 7,
            dtype: 5,
            shape: vec![1, 7],
        },
    ];
    let outputs = vec![PortDescriptor {
        name: "logits".into(),
        slot: 9,
        element_count: 7 * 32000,
        dtype: 8, // f32
        shape: vec![1, 7, 32000],
    }];

    let mut w = HoloWriter::new();
    w.set_inputs(inputs.clone());
    w.set_outputs(outputs.clone());
    let bytes = w.finish().unwrap();

    let plan = HoloLoader::from_bytes(&bytes).unwrap().into_plan().unwrap();
    let in_ports = decode_ports(plan.section(SectionKind::Inputs).unwrap()).unwrap();
    let out_ports = decode_ports(plan.section(SectionKind::Outputs).unwrap()).unwrap();

    assert_eq!(in_ports.len(), 2);
    assert_eq!(in_ports[0].name, "input_ids");
    assert_eq!(in_ports[0].shape, vec![1, 7]);
    assert_eq!(in_ports[0].dtype, 5);
    assert_eq!(in_ports[1].name, "attention_mask");
    assert_eq!(out_ports[0].name, "logits");
    assert_eq!(out_ports[0].shape, vec![1, 7, 32000]);
}

#[test]
fn extensions_round_trip_by_key() {
    let mut w = HoloWriter::new();
    w.add_extension("tokenizer.json", b"{\"model\":\"bpe\"}".to_vec());
    w.add_extension("generation_config", vec![1, 2, 3, 4]);
    let bytes = w.finish().unwrap();

    let plan = HoloLoader::from_bytes(&bytes).unwrap().into_plan().unwrap();
    let exts = plan.extensions().unwrap();
    assert_eq!(exts.len(), 2);

    let tok = exts.iter().find(|(k, _)| *k == "tokenizer.json").unwrap().1;
    assert_eq!(tok, b"{\"model\":\"bpe\"}");
    let gen = exts
        .iter()
        .find(|(k, _)| *k == "generation_config")
        .unwrap()
        .1;
    assert_eq!(gen, &[1, 2, 3, 4]);
}

#[test]
fn empty_ports_omit_sections() {
    // A writer with no ports must still produce a loadable archive.
    let bytes = HoloWriter::new().finish().unwrap();
    let plan = HoloLoader::from_bytes(&bytes).unwrap().into_plan().unwrap();
    assert!(plan.section(SectionKind::Inputs).is_err());
    assert!(plan.extensions().unwrap().is_empty());
}
