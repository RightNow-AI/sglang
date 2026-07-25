use std::fmt;

use crate::BranchId;

/// Errors produced by invalid scheduler inputs or lifecycle transitions.
#[derive(Clone, Debug, PartialEq)]
pub enum SchedulerError {
    UnknownBranch(BranchId),
    BranchNotActive(BranchId),
    BranchHasLiveChildren(BranchId),
    BranchLimitExceeded { limit: u64, requested_total: u64 },
    InvalidWidth(u32),
    InvalidConfig(&'static str),
    InvalidNumber(&'static str),
    BudgetAlreadyExhausted(BranchId),
    ValueScorePending(BranchId),
    UnexpectedValueScore(BranchId),
    PolicyCommandTreeMismatch,
    CounterOverflow(&'static str),
}

impl fmt::Display for SchedulerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownBranch(branch) => write!(formatter, "unknown branch {}", branch.0),
            Self::BranchNotActive(branch) => {
                write!(formatter, "branch {} is not active", branch.0)
            }
            Self::BranchHasLiveChildren(branch) => {
                write!(formatter, "branch {} still has live children", branch.0)
            }
            Self::BranchLimitExceeded {
                limit,
                requested_total,
            } => write!(
                formatter,
                "fork would grow the branch arena to {requested_total} nodes, above limit {limit}"
            ),
            Self::InvalidWidth(width) => write!(formatter, "invalid fork width {width}"),
            Self::InvalidConfig(message) => {
                write!(formatter, "invalid scheduler config: {message}")
            }
            Self::InvalidNumber(field) => write!(formatter, "{field} must be finite"),
            Self::BudgetAlreadyExhausted(branch) => {
                write!(
                    formatter,
                    "branch {} has exhausted its token budget",
                    branch.0
                )
            }
            Self::ValueScorePending(branch) => {
                write!(formatter, "branch {} is awaiting a value score", branch.0)
            }
            Self::UnexpectedValueScore(branch) => write!(
                formatter,
                "branch {} has no pending external value score",
                branch.0
            ),
            Self::PolicyCommandTreeMismatch => {
                write!(
                    formatter,
                    "policy commands do not match policy tree mutations"
                )
            }
            Self::CounterOverflow(field) => write!(formatter, "{field} overflowed"),
        }
    }
}

impl std::error::Error for SchedulerError {}
