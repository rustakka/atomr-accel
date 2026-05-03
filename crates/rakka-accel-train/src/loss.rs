//! Loss kinds.

#[derive(Debug, Clone, Copy)]
pub enum LossKind {
    Mse,
    CrossEntropy,
    /// Categorical cross-entropy with per-class weights — common in
    /// imbalanced classification.
    WeightedCrossEntropy,
}
