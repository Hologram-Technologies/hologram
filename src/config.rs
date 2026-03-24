//! Hologram configuration system.
//!
//! Loads settings from (in priority order, highest first):
//! 1. CLI flags / programmatic overrides
//! 2. `.hologram/config.toml` in the current directory (project-local)
//! 3. `~/.hologram/config.toml` (user-global)
//! 4. Built-in defaults
//!
//! # Example `config.toml`
//!
//! ```toml
//! [cache]
//! dir = "~/.hologram/cache"
//!
//! [archive]
//! compress_weights = false
//! compress_graph = false
//!
//! [inference]
//! temperature = 0.7
//! top_k = 40
//! max_tokens = 128
//! ```

use serde::Deserialize;
use std::path::{Path, PathBuf};

/// Cross-platform home directory lookup (replaces `dirs` crate).
fn home_dir() -> Option<PathBuf> {
    #[cfg(windows)]
    {
        std::env::var("USERPROFILE").ok().map(PathBuf::from)
    }
    #[cfg(not(windows))]
    {
        std::env::var("HOME").ok().map(PathBuf::from)
    }
}

/// Top-level hologram configuration.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct HologramConfig {
    /// Cache settings (decompressed archive cache, etc.).
    pub cache: CacheConfig,
    /// Archive format settings (compression, etc.).
    pub archive: ArchiveConfig,
    /// Inference defaults.
    pub inference: InferenceConfig,
}

/// Cache settings.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct CacheConfig {
    /// Directory for decompressed archive caches.
    /// Supports `~` expansion. Default: next to the archive file.
    pub dir: Option<String>,
}

/// Archive format settings.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct ArchiveConfig {
    /// Compress weights in archives. Default: false (zero-copy mmap).
    pub compress_weights: bool,
    /// Compress graph in archives. Default: false (fast loading).
    pub compress_graph: bool,
}

/// Inference defaults (can be overridden by CLI flags).
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct InferenceConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub max_tokens: usize,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 40,
            max_tokens: 128,
        }
    }
}

impl HologramConfig {
    /// Load configuration from the standard locations.
    ///
    /// Merges local `.hologram/config.toml` over global `~/.hologram/config.toml`.
    /// Missing files are silently ignored (defaults used).
    pub fn load() -> Self {
        let global = home_dir().map(|h| h.join(".hologram").join("config.toml"));
        let local = Path::new(".hologram").join("config.toml");

        let mut config = Self::default();

        if let Some(ref path) = global {
            if let Some(c) = Self::load_file(path) {
                config = c;
            }
        }

        if let Some(c) = Self::load_file(&local) {
            config.merge(c);
        }

        config
    }

    /// Load from a specific file path.
    pub fn load_file(path: &Path) -> Option<Self> {
        let contents = std::fs::read_to_string(path).ok()?;
        toml::from_str(&contents).ok()
    }

    /// Merge another config (higher priority) into this one.
    fn merge(&mut self, other: Self) {
        if other.cache.dir.is_some() {
            self.cache.dir = other.cache.dir;
        }
        self.archive = other.archive;
        self.inference = other.inference;
    }

    /// Resolve the cache directory, expanding `~`.
    /// Returns `None` for "cache next to the archive file" (default).
    pub fn cache_dir(&self) -> Option<PathBuf> {
        self.cache.dir.as_ref().map(|d| expand_tilde(d))
    }
}

/// Expand `~` to the user's home directory.
fn expand_tilde(path: &str) -> PathBuf {
    if let Some(rest) = path.strip_prefix('~') {
        if let Some(home) = home_dir() {
            return home.join(rest.trim_start_matches('/'));
        }
    }
    PathBuf::from(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults() {
        let c = HologramConfig::default();
        assert!(c.cache.dir.is_none());
        assert!(!c.archive.compress_weights);
        assert!(!c.archive.compress_graph);
        assert!((c.inference.temperature - 0.7).abs() < 0.01);
        assert_eq!(c.inference.top_k, 40);
        assert_eq!(c.inference.max_tokens, 128);
    }

    #[test]
    fn parse_toml() {
        let toml_str = r#"
[cache]
dir = "~/.hologram/cache"

[archive]
compress_weights = true

[inference]
temperature = 0.0
max_tokens = 256
"#;
        let c: HologramConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(c.cache.dir.as_deref(), Some("~/.hologram/cache"));
        assert!(c.archive.compress_weights);
        assert!(!c.archive.compress_graph);
        assert!((c.inference.temperature).abs() < 0.01);
        assert_eq!(c.inference.max_tokens, 256);
    }

    #[test]
    fn partial_toml() {
        let toml_str = r#"
[cache]
dir = "/tmp/holo-cache"
"#;
        let c: HologramConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(c.cache.dir.as_deref(), Some("/tmp/holo-cache"));
        // Defaults preserved for unspecified sections.
        assert!(!c.archive.compress_weights);
        assert_eq!(c.inference.max_tokens, 128);
    }

    #[test]
    fn expand_tilde_home() {
        let p = expand_tilde("~/foo/bar");
        assert!(!p.to_string_lossy().starts_with('~'));
        assert!(p.to_string_lossy().ends_with("foo/bar"));
    }

    #[test]
    fn expand_tilde_absolute() {
        assert_eq!(expand_tilde("/tmp/cache"), PathBuf::from("/tmp/cache"));
    }

    #[test]
    fn cache_dir_none_by_default() {
        let c = HologramConfig::default();
        assert!(c.cache_dir().is_none());
    }

    #[test]
    fn cache_dir_expands_tilde() {
        let c = HologramConfig {
            cache: CacheConfig {
                dir: Some("~/.hologram/cache".into()),
            },
            ..Default::default()
        };
        let dir = c.cache_dir().unwrap();
        assert!(!dir.to_string_lossy().contains('~'));
        assert!(dir.to_string_lossy().ends_with(".hologram/cache"));
    }
}
