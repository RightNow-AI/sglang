//! Deterministic branch-policy engine for AutoTree tree-structured decoding.

mod adaptive;
mod budget;
mod command;
mod error;
mod policies;
mod policy;
#[cfg(feature = "python")]
mod python;
mod scheduler;
mod scorer;
mod types;

pub use adaptive::AdaptiveForkConfig;
pub use budget::BudgetController;
pub use command::{Command, EngineEvent, encode_command_stream};
pub use error::SchedulerError;
pub use policies::{BeamPolicy, BestFirstPolicy, MctsPolicy};
pub use policy::{BeamConfig, BestFirstConfig, MctsConfig, Policy, PolicyConfig, PolicyRng};
#[cfg(feature = "python")]
pub use python::PyScheduler;
pub use scheduler::{DEFAULT_MAX_PENDING_EVENTS, Scheduler, SchedulerConfig};
pub use scorer::{BiasedOracleScorer, LogprobScorer, ValueScorer};
pub use types::{
    BranchId, BranchNode, BranchState, BranchTree, DEFAULT_MAX_TOTAL_BRANCHES, KillReason,
    MAX_BRANCH_WIDTH,
};
