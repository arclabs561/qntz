//! Vector quantization primitives.
//!
//! Scope: small, reusable pieces that higher-level ANN crates can build on.
//! This crate intentionally does **not** implement full ANN indices.

use thiserror::Error;

pub type Result<T> = std::result::Result<T, VQuantError>;

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
pub mod rabitq;

#[cfg(feature = "ternary")]
pub mod ternary;

pub mod simd_ops;

