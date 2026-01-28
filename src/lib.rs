//! # qntz
//!
//! Vector quantization primitives for ANN systems.
//!
//! Scope: small, reusable pieces (bit packing, low-bit codes) that higher-level ANN
//! crates can build on. This crate intentionally does **not** implement full indices.
//!
//! ## Contract
//!
//! - **No silent meaning**: packed codes have explicit conventions (e.g. 1-bit codes use
//!   `bit=1 -> +1`, `bit=0 -> -1` where stated).
//! - **No hidden geometry**: this crate provides code operations; it does not define a distance
//!   metric for your original vectors.
//! - **Determinism**: operations are pure and deterministic.
//!
//! ## Example
//!
//! Pack two binary code vectors and compute Hamming distance.
//!
//! ```rust
//! use qntz::simd_ops::{hamming_distance, pack_binary_fast};
//!
//! // Two 8-bit binary vectors (as 0/1 bytes).
//! let a_bits = [1u8, 0, 1, 0, 1, 0, 1, 0];
//! let b_bits = [1u8, 1, 1, 0, 0, 0, 1, 0];
//!
//! let mut a_packed = [0u8; 1];
//! let mut b_packed = [0u8; 1];
//! pack_binary_fast(&a_bits, &mut a_packed);
//! pack_binary_fast(&b_bits, &mut b_packed);
//!
//! let d = hamming_distance(&a_packed, &b_packed);
//! assert_eq!(d, 2);
//! ```

use thiserror::Error;

/// Convenience result type for this crate.
pub type Result<T> = std::result::Result<T, VQuantError>;

/// Errors for quantization code operations.
#[derive(Debug, Error)]
pub enum VQuantError {
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("empty index")]
    EmptyIndex,

    #[error("{0}")]
    Other(String),
}

#[cfg(feature = "rabitq")]
/// RaBitQ-style 1-bit quantization (feature-gated).
pub mod rabitq;

#[cfg(feature = "ternary")]
/// Ternary quantization utilities (feature-gated).
pub mod ternary;

/// Bit packing + popcount-based operations for low-bit codes.
pub mod simd_ops;
