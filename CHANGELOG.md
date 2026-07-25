# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `.holo` **format v4** (specs/refactor/03 §v4): `LayerKind::InferenceModel = 4`
  appended to the closed layer-kind set — an engine-agnostic AI-model layer
  (mandatory engine tag in `aux`, mandatory service name in `entry`, no exit
  code). `Layer::inference_model(content, entry, engine)` constructor; new
  `ManifestError` variants `EmptyLayerEntry` / `MissingEngineTag` /
  `DuplicateLayerEntry`; non-empty layer entry names must now be unique within a
  manifest. Writers emit v4; v2/v3 archives remain loadable.
- `hologram ai` CLI group (download / compile / inspect / infer) behind the new
  `ai` cargo feature on hologram-cli (default off), delegating to the sibling
  `hologram-ai` crate — hologram stays engine-agnostic.
- `hologram_ai_*` C ABI surface behind the new `ai` cargo feature on
  hologram-ffi (default off): compile / download / app load / model listing /
  JSON session invoke, with the `HOLOGRAM_ERROR_AI_*` error band (100–111) and
  `ai-*` FEATURES probes.

## [0.12.1] - 2026-07-20

### Added
- 

### Changed
- 

### Fixed
- 

## [0.12.0] - 2026-07-20

### Added
- 

### Changed
- 

### Fixed
- 

