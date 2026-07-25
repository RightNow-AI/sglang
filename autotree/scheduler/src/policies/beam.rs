use std::collections::BTreeSet;

use crate::{
    BeamConfig, BranchId, BranchTree, Command, EngineEvent, KillReason, MAX_BRANCH_WIDTH, Policy,
    PolicyRng, SchedulerError,
};

#[derive(Clone, Debug)]
pub struct BeamPolicy {
    width: usize,
    fork_width: u32,
    fork_at_tokens: BTreeSet<u64>,
}

impl BeamPolicy {
    pub fn new(config: BeamConfig) -> Result<Self, SchedulerError> {
        if config.width == 0 || config.width > MAX_BRANCH_WIDTH {
            return Err(SchedulerError::InvalidConfig(
                "beam width must be between 1 and MAX_BRANCH_WIDTH",
            ));
        }
        if config.fork_width == 0 || config.fork_width > MAX_BRANCH_WIDTH {
            return Err(SchedulerError::InvalidConfig(
                "beam fork_width must be between 1 and MAX_BRANCH_WIDTH",
            ));
        }
        Ok(Self {
            width: usize::try_from(config.width)
                .map_err(|_| SchedulerError::CounterOverflow("beam width"))?,
            fork_width: config.fork_width,
            fork_at_tokens: config.fork_at_tokens.into_iter().collect(),
        })
    }

    fn ranked_frontier(tree: &BranchTree) -> Vec<BranchId> {
        let mut frontier = tree.active_frontier();
        frontier.sort_by(|left, right| {
            let left_node = tree.get(*left).expect("frontier ids come from the arena");
            let right_node = tree.get(*right).expect("frontier ids come from the arena");
            right_node
                .cumulative_logprob()
                .total_cmp(&left_node.cumulative_logprob())
                .then_with(|| left.cmp(right))
        });
        frontier
    }

    fn prune_and_continue(&self, tree: &mut BranchTree) -> Result<Vec<Command>, SchedulerError> {
        let ranked = Self::ranked_frontier(tree);
        let mut commands = Vec::new();
        let mut victims: Vec<_> = ranked.iter().copied().skip(self.width).collect();
        victims.sort_unstable();
        for victim in victims {
            tree.kill(victim, KillReason::BeamPruned)?;
            commands.push(Command::Kill {
                branch: victim,
                reason: KillReason::BeamPruned,
            });
        }

        let ranked = Self::ranked_frontier(tree);
        let minimum_tokens = ranked
            .iter()
            .filter_map(|branch| tree.get(*branch).map(crate::BranchNode::tokens_generated))
            .min();
        commands.extend(ranked.into_iter().filter_map(|branch| {
            (tree.get(branch)?.tokens_generated() == minimum_tokens?)
                .then_some(Command::Continue { branch })
        }));
        Ok(commands)
    }
}

impl Policy for BeamPolicy {
    fn on_event(
        &mut self,
        event: &EngineEvent,
        tree: &mut BranchTree,
        _rng: &mut PolicyRng,
    ) -> Result<Vec<Command>, SchedulerError> {
        let mut commands = Vec::new();
        if event.is_token_sampled() {
            let frontier = tree.active_frontier();
            let common_tokens = frontier.first().and_then(|first| {
                let tokens = tree.get(*first)?.tokens_generated();
                frontier
                    .iter()
                    .all(|branch| {
                        tree.get(*branch)
                            .is_some_and(|node| node.tokens_generated() == tokens)
                    })
                    .then_some(tokens)
            });
            if common_tokens.is_some_and(|tokens| self.fork_at_tokens.contains(&tokens)) {
                for branch in frontier {
                    tree.fork(branch, self.fork_width)?;
                    commands.push(Command::ForkAt {
                        branch,
                        width: self.fork_width,
                    });
                }
            }
        }

        commands.extend(self.prune_and_continue(tree)?);
        Ok(commands)
    }

    fn on_event_without_forking(
        &mut self,
        _event: &EngineEvent,
        tree: &mut BranchTree,
        _rng: &mut PolicyRng,
    ) -> Result<Vec<Command>, SchedulerError> {
        self.prune_and_continue(tree)
    }

    fn on_adaptive_fork(
        &mut self,
        _event: &EngineEvent,
        tree: &mut BranchTree,
        _children: &[BranchId],
        _rng: &mut PolicyRng,
    ) -> Result<Vec<Command>, SchedulerError> {
        self.prune_and_continue(tree)
    }
}
