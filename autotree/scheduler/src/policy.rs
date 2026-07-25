use crate::{BranchId, BranchTree, Command, EngineEvent, SchedulerError};

use crate::policies::{BeamPolicy, BestFirstPolicy, MctsPolicy};

/// Portable, explicitly versioned PRNG used by every scheduler policy.
pub type PolicyRng = rand_chacha::ChaCha8Rng;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BeamConfig {
    pub width: u32,
    pub fork_width: u32,
    pub fork_at_tokens: Vec<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BestFirstConfig {
    pub expansion_width: u32,
    pub max_depth: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MctsConfig {
    pub expansion_width: u32,
    pub max_depth: u32,
    pub exploration_weight: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum PolicyConfig {
    Beam(BeamConfig),
    BestFirst(BestFirstConfig),
    Mcts(MctsConfig),
}

impl PolicyConfig {
    pub(crate) fn build(&self) -> Result<Box<dyn Policy>, SchedulerError> {
        match self {
            Self::Beam(config) => Ok(Box::new(BeamPolicy::new(config.clone())?)),
            Self::BestFirst(config) => Ok(Box::new(BestFirstPolicy::new(config.clone())?)),
            Self::Mcts(config) => Ok(Box::new(MctsPolicy::new(config.clone())?)),
        }
    }
}

/// Pluggable deterministic branch policy.
pub trait Policy: Send + Sync {
    fn on_event(
        &mut self,
        event: &EngineEvent,
        tree: &mut BranchTree,
        rng: &mut PolicyRng,
    ) -> Result<Vec<Command>, SchedulerError>;

    /// Selection/pruning path used when an entropy-bearing token delegates all forking to the
    /// scheduler's adaptive controller. The default safely advances the event branch only.
    fn on_event_without_forking(
        &mut self,
        event: &EngineEvent,
        tree: &mut BranchTree,
        _rng: &mut PolicyRng,
    ) -> Result<Vec<Command>, SchedulerError> {
        Ok(tree
            .get(event.branch())
            .is_some_and(|node| node.state() == crate::BranchState::Active)
            .then_some(Command::Continue {
                branch: event.branch(),
            })
            .into_iter()
            .collect())
    }

    /// Selection/pruning hook after the adaptive controller has forked `children`.
    fn on_adaptive_fork(
        &mut self,
        _event: &EngineEvent,
        _tree: &mut BranchTree,
        children: &[BranchId],
        _rng: &mut PolicyRng,
    ) -> Result<Vec<Command>, SchedulerError> {
        Ok(children
            .iter()
            .copied()
            .map(|branch| Command::Continue { branch })
            .collect())
    }
}
