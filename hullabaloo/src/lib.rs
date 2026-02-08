//! Engine-agnostic primitives and helpers for polyhedral experiments.
//!
//! This crate exposes shared primitives used across solver backends:
//! - numeric traits (`calculo`)
//! - compact bitset types (`types`)
//! - matrix storage/builders (`matrix`)
//! - incidence/adjacency set-families (`set_family`)
//! - fast adjacency builders (`adjacency`)
//! - geometric constructions (`drum`, `matroid`)
//!
//! Solver engines (e.g. DD/LRS/PPL) live in separate crates and build on top of these APIs.

pub use hullabaloo_core::*;
pub use hullabaloo_geom::{drum, geometrizable, matroid};
pub use hullabaloo_geom::{
    CharacteristicPolynomial, Drum, DrumBases, DrumPromotion, DrumSkin, Geometrizable,
    LinearOrientedMatroid, MatroidError, PromotionError,
};
