# Sprint 10: CLI Completeness

**Completed**: 2026-03-06
**Tests**: 684 total workspace tests, zero clippy warnings

## Goal

Make the `hologram` CLI fully functional so the library is ready for `hologram-ai` consumers. The `run` command was a stub; `inspect` didn't exist.

## Deliverables

- [x] `hologram run <file.holo> [--input INDEX:HEX]...` — load archive, parse inputs, execute, print outputs
- [x] `hologram inspect <file.holo>` — print header, node count, input/output names, schedule levels
- [x] `CliError::Exec(ExecError)` variant + `From<ExecError>` impl
- [x] `CliError::Archive(ArchiveError)` variant + `From<ArchiveError>` impl
- [x] Input parser: `--input 0:deadbeef` → `GraphInputs::set(0, vec![0xde, 0xad, 0xbe, 0xef])`
- [x] Output printer: each named output as `name: <hex>` (or `<hex>` if unnamed)
- [x] Inspect: file size, node count, input names, output names, schedule level count
- [x] Sprint 9 archived to `specs/sprints/9-tokio-async.md`
- [x] 15 new tests in `holo-cli`; zero clippy warnings; `just ci` green — **684 total workspace tests**

## Implementation Notes

- `run_cmd.rs`: `parse_input(s)` splits on first `:`, parses left as `u32` index, decodes right as hex bytes. `decode_hex` uses `.is_multiple_of(2)` per clippy. Error messages include the full input string for context.
- `inspect.rs`: loads archive via `holo_archive::load_from_bytes`, builds schedule via `holo_exec::build_schedule` to get level count, prints all metadata.
- `CliError::InvalidInput(String)` added for malformed `--input` flag values.
- `decode_hex("")` returns empty `Vec<u8>` (valid — means no input data for that index).
