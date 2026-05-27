#!/usr/bin/env bash
# End-to-end bare-metal boot test: build hologram.efi, boot it in QEMU/OVMF with no OS underneath,
# and assert the engine's storage self-check prints PASS on the UEFI console.
#
# Requires: qemu-system-x86_64, OVMF firmware, and the x86_64-unknown-uefi rust target.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EFI_CRATE="$ROOT/substrate/hologram-efi"

# Locate OVMF firmware (combined code+vars).
OVMF=""
for f in /usr/share/ovmf/OVMF.fd /usr/share/OVMF/OVMF.fd /usr/share/qemu/OVMF.fd; do
  [ -f "$f" ] && OVMF="$f" && break
done
if [ -z "$OVMF" ] || ! command -v qemu-system-x86_64 >/dev/null; then
  echo "SKIP: qemu-system-x86_64 and/or OVMF not available"; exit 0
fi

echo "==> building hologram.efi (x86_64-unknown-uefi)"
cargo build --release --target x86_64-unknown-uefi --manifest-path "$EFI_CRATE/Cargo.toml"

ESP="$(mktemp -d)/esp"; mkdir -p "$ESP/EFI/BOOT"
cp "$EFI_CRATE/target/x86_64-unknown-uefi/release/hologram.efi" "$ESP/EFI/BOOT/BOOTX64.EFI"

echo "==> booting in QEMU/OVMF (headless)"
OUT="$(timeout 120 qemu-system-x86_64 \
  -machine q35 -m 256 \
  -bios "$OVMF" \
  -drive format=raw,file=fat:rw:"$ESP" \
  -nographic -no-reboot -net none 2>&1 | tr -d '\r' || true)"

echo "$OUT" | grep -aE "HOLOGRAM-BM" || true
if echo "$OUT" | grep -qa "HOLOGRAM-BM: PASS"; then
  echo "==> UEFI boot test: PASS"; exit 0
else
  echo "==> UEFI boot test: FAIL"; exit 1
fi
