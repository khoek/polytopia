//! Core data structures and adjacency utilities for polyhedral computations.

pub mod adjacency;
pub mod adjacency_list;
pub mod matrix;
pub mod set_family;
pub mod types;

pub use adjacency::AdjacencyListBuilder;
pub use adjacency_list::AdjacencyList;
