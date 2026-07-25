use std::collections::BTreeMap;

use crate::{BranchId, BranchNode, SchedulerError};

/// Synchronous, pluggable value estimate used for pruning and policy ranking.
pub trait ValueScorer: Send + Sync {
    fn score(&self, branch: &BranchNode) -> f64;
}

/// Uses cumulative branch log-probability as the value estimate.
#[derive(Clone, Copy, Debug, Default)]
pub struct LogprobScorer;

impl ValueScorer for LogprobScorer {
    fn score(&self, branch: &BranchNode) -> f64 {
        branch.cumulative_logprob()
    }
}

/// Deterministic scorer with branch-specific values for policy convergence tests.
#[derive(Clone, Debug)]
pub struct BiasedOracleScorer {
    scores: BTreeMap<BranchId, f64>,
    default_score: f64,
}

impl BiasedOracleScorer {
    pub fn new(
        scores: impl IntoIterator<Item = (BranchId, f64)>,
        default_score: f64,
    ) -> Result<Self, SchedulerError> {
        if !default_score.is_finite() {
            return Err(SchedulerError::InvalidNumber("oracle default score"));
        }
        let mut collected = BTreeMap::new();
        for (branch, score) in scores {
            if !score.is_finite() {
                return Err(SchedulerError::InvalidNumber("oracle branch score"));
            }
            collected.insert(branch, score);
        }
        Ok(Self {
            scores: collected,
            default_score,
        })
    }
}

impl ValueScorer for BiasedOracleScorer {
    fn score(&self, branch: &BranchNode) -> f64 {
        self.scores
            .get(&branch.id())
            .copied()
            .unwrap_or(self.default_score)
    }
}
