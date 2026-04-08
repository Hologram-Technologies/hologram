//! Print the `HostMetaSection` block — chat template, sampling defaults,
//! port bindings, and model card — when present in the archive.
//!
//! Called from [`super::summary::print`]. Silently prints nothing if the
//! archive does not contain the section, since host metadata is always
//! optional.

use hologram_archive::section::host_meta::HostMetaSection;
use hologram_archive::LoadedPlan;

/// Print the host metadata block, if the archive contains one. Returns
/// `true` iff anything was printed.
pub fn print(data: &[u8], plan: &LoadedPlan) -> bool {
    let Some(section) = plan.host_meta_from_bytes(data) else {
        return false;
    };
    if section.is_empty() {
        return false;
    }
    println!("Host metadata (v{}):", section.version);
    if let Some(tmpl) = &section.prompt_template {
        println!("  Prompt template: {}", truncate(tmpl, 120));
    }
    if let Some(tmpl) = &section.chat_template {
        println!("  Chat template:   {}", truncate(tmpl, 120));
    }
    if let Some(sampling) = &section.sampling {
        print_sampling(sampling);
    }
    if !section.ports.is_empty() {
        println!("  Ports:");
        for port in &section.ports {
            println!("    {} -> {}", port.logical_name, port.graph_port);
        }
    }
    if let Some(card) = &section.model_card {
        print_model_card(card);
    }
    true
}

fn print_sampling(sampling: &hologram_archive::section::host_meta::SamplingDefaults) {
    println!("  Sampling:");
    if let Some(t) = sampling.temperature {
        println!("    temperature:        {t}");
    }
    if let Some(k) = sampling.top_k {
        println!("    top_k:              {k}");
    }
    if let Some(p) = sampling.top_p {
        println!("    top_p:              {p}");
    }
    if let Some(r) = sampling.repetition_penalty {
        println!("    repetition_penalty: {r}");
    }
    if !sampling.stop.is_empty() {
        println!("    stop:               {:?}", sampling.stop);
    }
}

fn print_model_card(card: &hologram_archive::section::host_meta::ModelCard) {
    println!("  Model card:");
    if let Some(a) = &card.author {
        println!("    author:     {a}");
    }
    if let Some(l) = &card.license {
        println!("    license:    {l}");
    }
    if let Some(u) = &card.source_url {
        println!("    source_url: {u}");
    }
    if !card.tags.is_empty() {
        println!("    tags:       {:?}", card.tags);
    }
}

/// Truncate a string to `max` characters, appending an ellipsis if cut.
/// Chat templates in particular can be kilobytes of jinja — summary output
/// should not dump them in full.
fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let taken: String = s.chars().take(max).collect();
        format!("{taken}… ({} chars total)", s.chars().count())
    }
}

// Suppress the unused-import warning for `HostMetaSection` when the only
// reference is via the `plan.host_meta_from_bytes()` return type. Rust
// still needs the import for docs + intra-crate type inference.
#[allow(dead_code)]
fn _type_anchor(_s: &HostMetaSection) {}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram_archive::section::host_meta::{
        HostMetaSection, ModelCard, PortBinding, SamplingDefaults, HOST_META_VERSION,
    };

    #[test]
    fn truncate_shorter_than_max() {
        assert_eq!(truncate("hello", 10), "hello");
    }

    #[test]
    fn truncate_longer_than_max() {
        let s = "a".repeat(50);
        let out = truncate(&s, 10);
        assert!(out.starts_with(&"a".repeat(10)));
        assert!(out.contains("50 chars total"));
    }

    #[test]
    fn host_meta_section_fixture_builds() {
        // Smoke test: make sure the fixture the other tests imply we could
        // build actually builds without rkyv complaints.
        let section = HostMetaSection {
            version: HOST_META_VERSION,
            prompt_template: Some("x".into()),
            chat_template: None,
            sampling: Some(SamplingDefaults {
                temperature: Some(0.7),
                top_k: Some(40),
                top_p: None,
                repetition_penalty: None,
                stop: vec![],
            }),
            ports: vec![PortBinding {
                logical_name: "logits".into(),
                graph_port: "output_0".into(),
            }],
            model_card: Some(ModelCard {
                author: Some("me".into()),
                license: Some("MIT".into()),
                source_url: None,
                tags: vec!["test".into()],
            }),
        };
        assert!(!section.is_empty());
    }
}
