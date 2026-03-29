//! Encoding: the π-F-λ bridge between continuous and ring domains.
//!
//! Encodings exist only at graph input/output boundaries.
//! Once in the ring, all computation is ring-native.

use crate::level::QuantumLevel;
use crate::word::RingWord;

/// Trait for encoding continuous values into ring elements and back.
pub trait Encoding<W: RingWord> {
    /// Embed a continuous value into the ring (π map).
    fn embed(&self, value: f64) -> W;
    /// Lift a ring element back to continuous space (λ map).
    fn lift(&self, word: W) -> f64;
    /// Name of this encoding.
    fn name(&self) -> &'static str;
}

/// Unsigned encoding: maps [0.0, 1.0] → [0, MAX].
#[derive(Debug, Clone, Copy, Default)]
pub struct UnsignedEncoding<Q: QuantumLevel>(core::marker::PhantomData<Q>);

impl<Q: QuantumLevel> UnsignedEncoding<Q> {
    pub const fn new() -> Self {
        Self(core::marker::PhantomData)
    }
}

impl<Q: QuantumLevel> Encoding<Q::Word> for UnsignedEncoding<Q> {
    #[inline]
    fn embed(&self, value: f64) -> Q::Word {
        let clamped = value.clamp(0.0, 1.0);
        let max_f = Q::Word::MAX.to_u64() as f64;
        Q::Word::from_u64((clamped * max_f + 0.5) as u64)
    }

    #[inline]
    fn lift(&self, word: Q::Word) -> f64 {
        let max_f = Q::Word::MAX.to_u64() as f64;
        word.to_u64() as f64 / max_f
    }

    #[inline]
    fn name(&self) -> &'static str {
        "unsigned"
    }
}

/// Signed encoding: maps [-1.0, 1.0] → [0, MAX], zero at midpoint.
#[derive(Debug, Clone, Copy, Default)]
pub struct SignedEncoding<Q: QuantumLevel>(core::marker::PhantomData<Q>);

impl<Q: QuantumLevel> SignedEncoding<Q> {
    pub const fn new() -> Self {
        Self(core::marker::PhantomData)
    }
}

impl<Q: QuantumLevel> Encoding<Q::Word> for SignedEncoding<Q> {
    #[inline]
    fn embed(&self, value: f64) -> Q::Word {
        let clamped = value.clamp(-1.0, 1.0);
        let max_f = Q::Word::MAX.to_u64() as f64;
        Q::Word::from_u64(((clamped + 1.0) * 0.5 * max_f + 0.5) as u64)
    }

    #[inline]
    fn lift(&self, word: Q::Word) -> f64 {
        let max_f = Q::Word::MAX.to_u64() as f64;
        word.to_u64() as f64 / max_f * 2.0 - 1.0
    }

    #[inline]
    fn name(&self) -> &'static str {
        "signed"
    }
}

/// Angle encoding: maps [0.0, 2π) → [0, MAX].
#[derive(Debug, Clone, Copy, Default)]
pub struct AngleEncoding<Q: QuantumLevel>(core::marker::PhantomData<Q>);

impl<Q: QuantumLevel> AngleEncoding<Q> {
    pub const fn new() -> Self {
        Self(core::marker::PhantomData)
    }
}

impl<Q: QuantumLevel> Encoding<Q::Word> for AngleEncoding<Q> {
    #[inline]
    fn embed(&self, value: f64) -> Q::Word {
        let two_pi = 2.0 * core::f64::consts::PI;
        // Modular: wrap into [0, 2π)
        let wrapped = ((value % two_pi) + two_pi) % two_pi;
        let max_f = Q::Word::MAX.to_u64() as f64;
        Q::Word::from_u64((wrapped / two_pi * (max_f + 1.0)) as u64)
    }

    #[inline]
    fn lift(&self, word: Q::Word) -> f64 {
        let two_pi = 2.0 * core::f64::consts::PI;
        let max_f = Q::Word::MAX.to_u64() as f64;
        word.to_u64() as f64 / (max_f + 1.0) * two_pi
    }

    #[inline]
    fn name(&self) -> &'static str {
        "angle"
    }
}

/// Raw encoding: identity mapping (truncates to word width).
#[derive(Debug, Clone, Copy, Default)]
pub struct RawEncoding<Q: QuantumLevel>(core::marker::PhantomData<Q>);

impl<Q: QuantumLevel> RawEncoding<Q> {
    pub const fn new() -> Self {
        Self(core::marker::PhantomData)
    }
}

impl<Q: QuantumLevel> Encoding<Q::Word> for RawEncoding<Q> {
    #[inline]
    fn embed(&self, value: f64) -> Q::Word {
        let clamped = value.clamp(0.0, Q::Word::MAX.to_u64() as f64);
        Q::Word::from_u64(clamped as u64)
    }

    #[inline]
    fn lift(&self, word: Q::Word) -> f64 {
        word.to_u64() as f64
    }

    #[inline]
    fn name(&self) -> &'static str {
        "raw"
    }
}
