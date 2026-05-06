//! Spec X.1 Certificates section codec tests.

use hologram_archive::certificate_codec::{encode, decode, CertificateRecord};

#[test]
fn empty_round_trip() {
    let bytes = encode(&[]);
    let decoded = decode(&bytes).unwrap();
    assert!(decoded.is_empty());
}

#[test]
fn single_record_round_trip() {
    let r = CertificateRecord {
        witt_bits: 64,
        width_bytes: 32,
        fingerprint: [42u8; 32],
    };
    let bytes = encode(&[r]);
    let decoded = decode(&bytes).unwrap();
    assert_eq!(decoded.len(), 1);
    assert_eq!(decoded[0], r);
}

#[test]
fn many_records_round_trip() {
    let records: Vec<CertificateRecord> = (0..16)
        .map(|i| CertificateRecord {
            witt_bits: 8 + (i as u16) * 8,
            width_bytes: 32,
            fingerprint: {
                let mut a = [0u8; 32];
                a[0] = i as u8;
                a
            },
        })
        .collect();
    let bytes = encode(&records);
    let decoded = decode(&bytes).unwrap();
    assert_eq!(decoded, records);
}

#[test]
fn truncated_input_errors() {
    assert!(decode(&[0u8, 1, 0, 0]).is_err()); // claims 1 record but no body
}
