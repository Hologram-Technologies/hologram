//! Certificate store — O(1) lookup by `(unit_address, quantum_level)`.
//!
//! Implements the memoization path from the CompileUnit spec: if a
//! `SaturationCertificate` already exists for the pair `(unitAddress, unitQuantumLevel)`,
//! the cascade can skip to Extract without executing stages 1-4.

use uor_foundation::QuantumLevel;

/// Proof that a CompileUnit has been fully resolved through the cascade.
#[derive(Debug, Clone)]
pub struct Certificate {
    /// Content-addressed identifier of the root term graph.
    pub unit_address: [u8; 32],
    /// Quantum level at which the computation was verified.
    pub quantum_level: QuantumLevel,
    /// Total Landauer cost consumed during cascade evaluation.
    pub budget_consumed: f64,
    /// Whether the cascade converged successfully.
    pub converged: bool,
}

/// Composite key: (unit_address, quantum_level) = 33 bytes.
#[derive(Clone, Copy, PartialEq, Eq)]
struct CertKey {
    address: [u8; 32],
    level: QuantumLevel,
}

impl core::hash::Hash for CertKey {
    #[inline]
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        state.write(&self.address);
        state.write_u32(self.level.index());
    }
}

/// Fixed-capacity certificate store using open-addressing with linear probing.
///
/// Designed for O(1) amortized lookup without `std::collections::HashMap`.
/// Power-of-2 sizing ensures modular arithmetic uses bitwise AND instead of division.
///
/// # Performance
///
/// - Lookup: O(1) amortized (linear probing, load factor < 0.75)
/// - Insert: O(1) amortized
/// - Memory: `capacity * size_of::<Slot>()` bytes, pre-allocated
pub struct CertificateStore {
    slots: Vec<Option<(CertKey, Certificate)>>,
    len: usize,
    mask: usize, // capacity - 1 (power of 2)
}

impl CertificateStore {
    /// Create a new store with the given capacity (rounded up to power of 2).
    ///
    /// Pre-allocates all slots to avoid runtime allocation.
    pub fn new(capacity: usize) -> Self {
        let cap = capacity.next_power_of_two().max(16);
        let mut slots = Vec::with_capacity(cap);
        slots.resize_with(cap, || None);
        Self {
            slots,
            len: 0,
            mask: cap - 1,
        }
    }

    /// Create a store pre-sized for the expected number of entries.
    ///
    /// Allocates capacity = `expected * 2` (rounded to power of 2) to keep
    /// load factor well below 0.75, minimizing probe chains.
    pub fn with_expected_load(expected: usize) -> Self {
        Self::new((expected * 2).max(16))
    }

    /// Look up a certificate by `(unit_address, quantum_level)`.
    ///
    /// O(1) amortized — FNV-style hash on the 33-byte key followed by linear probing.
    pub fn get(&self, address: &[u8; 32], level: QuantumLevel) -> Option<&Certificate> {
        let key = CertKey {
            address: *address,
            level,
        };
        let mut idx = self.hash_key(&key);

        for _ in 0..self.slots.len() {
            match &self.slots[idx] {
                Some((k, cert)) if *k == key => return Some(cert),
                None => return None,
                _ => idx = (idx + 1) & self.mask,
            }
        }
        None
    }

    /// Insert or update a certificate.
    ///
    /// O(1) amortized. Resizes (doubles capacity + rehashes) when load factor > 0.75.
    pub fn insert(&mut self, cert: Certificate) {
        let key = CertKey {
            address: cert.unit_address,
            level: cert.quantum_level,
        };

        // Resize if load factor exceeds 0.75.
        if self.len * 4 >= self.slots.len() * 3 {
            self.grow();
        }

        let mut idx = self.hash_key(&key);

        for _ in 0..self.slots.len() {
            match &self.slots[idx] {
                Some((k, _)) if *k == key => {
                    // Update existing entry.
                    self.slots[idx] = Some((key, cert));
                    return;
                }
                None => {
                    self.slots[idx] = Some((key, cert));
                    self.len += 1;
                    return;
                }
                _ => idx = (idx + 1) & self.mask,
            }
        }

        // Should not reach here after grow(), but just in case.
        self.slots[idx] = Some((key, cert));
    }

    /// Double capacity and rehash all entries.
    fn grow(&mut self) {
        let new_cap = (self.slots.len() * 2).max(32);
        let old_slots = core::mem::replace(&mut self.slots, {
            let mut v = Vec::with_capacity(new_cap);
            v.resize_with(new_cap, || None);
            v
        });
        self.mask = new_cap - 1;
        self.len = 0;

        for slot in old_slots.into_iter().flatten() {
            let (key, cert) = slot;
            let mut idx = self.hash_key(&key);
            loop {
                if self.slots[idx].is_none() {
                    self.slots[idx] = Some((key, cert));
                    self.len += 1;
                    break;
                }
                idx = (idx + 1) & self.mask;
            }
        }
    }

    /// Number of stored certificates.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the store is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Capacity of the store (power of 2).
    #[inline]
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// FNV-1a-inspired hash of a CertKey (33 bytes → usize).
    ///
    /// Chosen for speed on small keys: 33 bytes = ~8 iterations of the
    /// 8-byte-at-a-time loop. Total ~5-8ns on modern CPUs.
    #[inline]
    fn hash_key(&self, key: &CertKey) -> usize {
        let mut h: u64 = 0xcbf29ce484222325; // FNV offset basis
        for chunk in key.address.chunks(8) {
            let mut buf = [0u8; 8];
            buf[..chunk.len()].copy_from_slice(chunk);
            h ^= u64::from_le_bytes(buf);
            h = h.wrapping_mul(0x100000001b3); // FNV prime
        }
        h ^= key.level.index() as u64;
        h = h.wrapping_mul(0x100000001b3);
        (h as usize) & self.mask
    }
}

impl CertificateStore {
    /// Save the certificate store to a file.
    ///
    /// Binary format: [u32 count][entries...] where each entry is:
    /// [32 bytes address][1 byte level][8 bytes budget_consumed][1 byte converged]
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        let mut buf = Vec::new();
        let count = self.len as u32;
        buf.extend_from_slice(&count.to_le_bytes());
        for slot in &self.slots {
            if let Some((key, cert)) = slot {
                buf.extend_from_slice(&key.address);
                buf.push(key.level.index() as u8);
                buf.extend_from_slice(&cert.budget_consumed.to_le_bytes());
                buf.push(if cert.converged { 1 } else { 0 });
            }
        }
        std::fs::write(path, &buf)
    }

    /// Load a certificate store from a file.
    pub fn load(path: &std::path::Path) -> std::io::Result<Self> {
        let data = std::fs::read(path)?;
        if data.len() < 4 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "certificate store file too short",
            ));
        }
        let count = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        let mut store = Self::with_expected_load(count);
        let entry_size = 32 + 1 + 8 + 1; // address + level + budget + converged
        let mut offset = 4;
        for _ in 0..count {
            if offset + entry_size > data.len() {
                break;
            }
            let mut address = [0u8; 32];
            address.copy_from_slice(&data[offset..offset + 32]);
            let level = QuantumLevel::new(data[offset + 32] as u32);
            let budget = f64::from_le_bytes([
                data[offset + 33],
                data[offset + 34],
                data[offset + 35],
                data[offset + 36],
                data[offset + 37],
                data[offset + 38],
                data[offset + 39],
                data[offset + 40],
            ]);
            let converged = data[offset + 41] != 0;
            store.insert(Certificate {
                unit_address: address,
                quantum_level: level,
                budget_consumed: budget,
                converged,
            });
            offset += entry_size;
        }
        Ok(store)
    }
}

impl core::fmt::Debug for CertificateStore {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CertificateStore")
            .field("len", &self.len)
            .field("capacity", &self.slots.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cert(addr_byte: u8, level: QuantumLevel) -> Certificate {
        let mut address = [0u8; 32];
        address[0] = addr_byte;
        Certificate {
            unit_address: address,
            quantum_level: level,
            budget_consumed: 5.0,
            converged: true,
        }
    }

    #[test]
    fn insert_and_lookup() {
        let mut store = CertificateStore::new(64);
        let cert = make_cert(42, QuantumLevel::Q0);
        store.insert(cert.clone());

        let result = store.get(&cert.unit_address, QuantumLevel::Q0);
        assert!(result.is_some());
        assert_eq!(result.unwrap().unit_address[0], 42);
        assert!(result.unwrap().converged);
    }

    #[test]
    fn lookup_miss() {
        let store = CertificateStore::new(64);
        let mut addr = [0u8; 32];
        addr[0] = 99;
        assert!(store.get(&addr, QuantumLevel::Q0).is_none());
    }

    #[test]
    fn different_levels_different_entries() {
        let mut store = CertificateStore::new(64);
        let cert_q0 = make_cert(1, QuantumLevel::Q0);
        let cert_q1 = Certificate {
            quantum_level: QuantumLevel::Q1,
            budget_consumed: 10.0,
            ..make_cert(1, QuantumLevel::Q1)
        };

        store.insert(cert_q0.clone());
        store.insert(cert_q1.clone());

        assert_eq!(store.len(), 2);

        let r0 = store.get(&cert_q0.unit_address, QuantumLevel::Q0).unwrap();
        assert_eq!(r0.budget_consumed, 5.0);

        let r1 = store.get(&cert_q1.unit_address, QuantumLevel::Q1).unwrap();
        assert_eq!(r1.budget_consumed, 10.0);
    }

    #[test]
    fn update_existing() {
        let mut store = CertificateStore::new(64);
        let cert1 = make_cert(1, QuantumLevel::Q0);
        store.insert(cert1);

        let cert2 = Certificate {
            budget_consumed: 99.0,
            ..make_cert(1, QuantumLevel::Q0)
        };
        store.insert(cert2);

        assert_eq!(store.len(), 1); // no duplicate
        let r = store
            .get(
                &make_cert(1, QuantumLevel::Q0).unit_address,
                QuantumLevel::Q0,
            )
            .unwrap();
        assert_eq!(r.budget_consumed, 99.0);
    }

    #[test]
    fn many_entries() {
        let mut store = CertificateStore::new(256);
        for i in 0..200u8 {
            store.insert(make_cert(i, QuantumLevel::Q0));
        }
        assert_eq!(store.len(), 200);

        // All should be retrievable.
        for i in 0..200u8 {
            let mut addr = [0u8; 32];
            addr[0] = i;
            assert!(
                store.get(&addr, QuantumLevel::Q0).is_some(),
                "missing entry for byte {}",
                i
            );
        }
    }

    #[test]
    fn empty_store() {
        let store = CertificateStore::new(16);
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
        assert!(store.capacity() >= 16);
    }

    #[test]
    fn power_of_two_capacity() {
        let store = CertificateStore::new(50);
        assert_eq!(store.capacity(), 64); // rounded up
    }

    #[test]
    fn save_load_round_trip() {
        let mut store = CertificateStore::new(64);
        store.insert(make_cert(1, QuantumLevel::Q0));
        store.insert(make_cert(2, QuantumLevel::Q1));
        store.insert(make_cert(3, QuantumLevel::Q0));

        let dir = std::env::temp_dir();
        let path = dir.join("test_cert_store.bin");
        store.save(&path).unwrap();

        let loaded = CertificateStore::load(&path).unwrap();
        assert_eq!(loaded.len(), 3);

        let mut addr1 = [0u8; 32];
        addr1[0] = 1;
        let cert = loaded.get(&addr1, QuantumLevel::Q0).unwrap();
        assert!(cert.converged);
        assert_eq!(cert.budget_consumed, 5.0);

        let mut addr2 = [0u8; 32];
        addr2[0] = 2;
        assert!(loaded.get(&addr2, QuantumLevel::Q1).is_some());

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn certificate_lookup_performance() {
        // Performance contract: 1M lookups < 50ms (< 50ns each)
        let mut store = CertificateStore::new(1024);
        for i in 0..200u8 {
            store.insert(make_cert(i, QuantumLevel::Q0));
        }
        let mut addr = [0u8; 32];
        addr[0] = 100;

        let start = std::time::Instant::now();
        for _ in 0..1_000_000 {
            let _ = store.get(&addr, QuantumLevel::Q0);
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_millis() < 200, // generous CI margin
            "1M cert lookups took {}ms (target < 200ms)",
            elapsed.as_millis()
        );
    }

    #[test]
    fn grow_on_high_load_factor() {
        let mut store = CertificateStore::new(16);
        assert_eq!(store.capacity(), 16);
        for i in 0..13u8 {
            store.insert(make_cert(i, QuantumLevel::Q0));
        }
        assert!(store.capacity() >= 32, "store should have grown");
        assert_eq!(store.len(), 13);
        for i in 0..13u8 {
            let mut addr = [0u8; 32];
            addr[0] = i;
            assert!(
                store.get(&addr, QuantumLevel::Q0).is_some(),
                "missing entry {} after grow",
                i
            );
        }
    }

    #[test]
    fn with_expected_load_pre_sizes() {
        let store = CertificateStore::with_expected_load(100);
        assert!(store.capacity() >= 200);
        assert_eq!(store.len(), 0);
    }
}
