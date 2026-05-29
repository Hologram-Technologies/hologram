#!/usr/bin/env bash
# End-to-end bare-metal boot test: build hologram.efi, boot it in QEMU/OVMF with no OS underneath,
# and assert the engine's storage self-check prints PASS on the UEFI console.
#
# Requires: qemu-system-x86_64, OVMF firmware, and the x86_64-unknown-uefi rust target.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EFI_CRATE="$ROOT/substrate/hologram-efi"

if ! command -v qemu-system-x86_64 >/dev/null; then
  echo "SKIP: qemu-system-x86_64 not available"; exit 0
fi

# Locate OVMF firmware. Prefer a combined code+vars image (driven via -bios);
# fall back to the split CODE/VARS pair shipped by modern distros (Ubuntu 24.04's
# `ovmf` package), driven via two pflash drives — VARS must be a writable copy.
OVMF_COMBINED=""
for f in /usr/share/ovmf/OVMF.fd /usr/share/OVMF/OVMF.fd /usr/share/qemu/OVMF.fd; do
  [ -f "$f" ] && OVMF_COMBINED="$f" && break
done
OVMF_CODE=""; OVMF_VARS=""
if [ -z "$OVMF_COMBINED" ]; then
  for c in /usr/share/OVMF/OVMF_CODE_4M.fd /usr/share/OVMF/OVMF_CODE.fd /usr/share/edk2/x64/OVMF_CODE.4m.fd; do
    [ -f "$c" ] && OVMF_CODE="$c" && break
  done
  for v in /usr/share/OVMF/OVMF_VARS_4M.fd /usr/share/OVMF/OVMF_VARS.fd /usr/share/edk2/x64/OVMF_VARS.4m.fd; do
    [ -f "$v" ] && OVMF_VARS="$v" && break
  done
fi
if [ -z "$OVMF_COMBINED" ] && { [ -z "$OVMF_CODE" ] || [ -z "$OVMF_VARS" ]; }; then
  echo "SKIP: OVMF firmware not available"; exit 0
fi

echo "==> building hologram.efi (x86_64-unknown-uefi)"
cargo build --release --target x86_64-unknown-uefi --manifest-path "$EFI_CRATE/Cargo.toml"

ESP="$(mktemp -d)/esp"; mkdir -p "$ESP/EFI/BOOT"
cp "$EFI_CRATE/target/x86_64-unknown-uefi/release/hologram.efi" "$ESP/EFI/BOOT/BOOTX64.EFI"

if [ -n "$OVMF_COMBINED" ]; then
  echo "==> booting in QEMU/OVMF (headless, combined firmware)"
  FW_ARGS=(-bios "$OVMF_COMBINED")
else
  echo "==> booting in QEMU/OVMF (headless, split CODE/VARS firmware)"
  VARS_RW="$(mktemp --suffix=-OVMF_VARS.fd)"; cp "$OVMF_VARS" "$VARS_RW"
  FW_ARGS=(
    -drive "if=pflash,format=raw,unit=0,readonly=on,file=$OVMF_CODE"
    -drive "if=pflash,format=raw,unit=1,file=$VARS_RW"
  )
fi

OUT="$(timeout 120 qemu-system-x86_64 \
  -machine q35 -m 256 \
  "${FW_ARGS[@]}" \
  -drive format=raw,file=fat:rw:"$ESP" \
  -nographic -no-reboot -net none 2>&1 | tr -d '\r' || true)"

echo "$OUT" | grep -aE "HOLOGRAM-BM" || true
if echo "$OUT" | grep -qa "HOLOGRAM-BM: PASS"; then
  echo "==> UEFI boot test: PASS"; exit 0
else
  echo "==> UEFI boot test: FAIL"; exit 1
fi
