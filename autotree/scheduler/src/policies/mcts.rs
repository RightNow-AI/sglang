use rand::RngCore;

use crate::{
    BranchId, BranchState, BranchTree, Command, EngineEvent, MAX_BRANCH_WIDTH, MctsConfig, Policy,
    PolicyRng, SchedulerError,
};

const UCT_RELATIVE_TOLERANCE: f64 = 1.0e-12;

fn random_index(rng: &mut PolicyRng, len: usize) -> usize {
    let bound = u64::try_from(len).expect("policy candidate count fits in u64");
    debug_assert!(bound > 0);
    let rejection_threshold = bound.wrapping_neg() % bound;
    loop {
        let sample = rng.next_u64();
        if sample >= rejection_threshold {
            return usize::try_from(sample % bound).expect("sampled index fits in usize");
        }
    }
}

fn uct_scores_nearly_equal(left: f64, right: f64) -> bool {
    let scale = left.abs().max(right.abs()).max(1.0);
    (left - right).abs() <= UCT_RELATIVE_TOLERANCE * scale
}

#[derive(Clone, Debug)]
pub struct MctsPolicy {
    expansion_width: u32,
    max_depth: u32,
    exploration_weight: f64,
}

impl MctsPolicy {
    pub fn new(config: MctsConfig) -> Result<Self, SchedulerError> {
        if config.expansion_width == 0 || config.expansion_width > MAX_BRANCH_WIDTH {
            return Err(SchedulerError::InvalidConfig(
                "MCTS expansion_width must be between 1 and MAX_BRANCH_WIDTH",
            ));
        }
        if config.max_depth == 0 {
            return Err(SchedulerError::InvalidConfig(
                "MCTS max_depth must be greater than zero",
            ));
        }
        if !config.exploration_weight.is_finite() || config.exploration_weight < 0.0 {
            return Err(SchedulerError::InvalidConfig(
                "MCTS exploration_weight must be finite and non-negative",
            ));
        }
        Ok(Self {
            expansion_width: config.expansion_width,
            max_depth: config.max_depth,
            exploration_weight: config.exploration_weight,
        })
    }

    fn choose_child(
        &self,
        tree: &BranchTree,
        parent: BranchId,
        rng: &mut PolicyRng,
    ) -> Option<BranchId> {
        let mut children: Vec<_> = tree
            .children(parent)
            .ok()?
            .iter()
            .copied()
            .filter(|child| tree.get(*child).is_some_and(|node| node.state().is_live()))
            .collect();
        children.sort_unstable();
        if children.is_empty() {
            return None;
        }

        let unvisited: Vec<_> = children
            .iter()
            .copied()
            .filter(|child| tree.get(*child).expect("known child").visits() == 0)
            .collect();
        if !unvisited.is_empty() {
            return Some(unvisited[random_index(rng, unvisited.len())]);
        }

        let parent_visits = tree.get(parent).expect("known parent").visits().max(1) as f64;
        let mut best = None;
        for child in children {
            let node = tree.get(child).expect("known child");
            let visits = node.visits() as f64;
            let exploitation = node.value_sum() / visits;
            let exploration = self.exploration_weight * (parent_visits.ln() / visits).sqrt();
            let score = exploitation + exploration;
            match best {
                None => best = Some((score, child)),
                Some((best_score, _))
                    if !uct_scores_nearly_equal(score, best_score)
                        && score.total_cmp(&best_score).is_gt() =>
                {
                    best = Some((score, child));
                }
                Some((best_score, best_child))
                    if uct_scores_nearly_equal(score, best_score) && child < best_child =>
                {
                    best = Some((score, child));
                }
                Some(_) => {}
            }
        }
        best.map(|(_, branch)| branch)
    }

    fn select(
        &self,
        tree: &mut BranchTree,
        rng: &mut PolicyRng,
    ) -> Result<Vec<Command>, SchedulerError> {
        let mut current = tree.root();
        loop {
            let node = tree
                .get(current)
                .ok_or(SchedulerError::UnknownBranch(current))?;
            match node.state() {
                BranchState::Active if node.depth() < self.max_depth => {
                    let children = tree.fork(current, self.expansion_width)?;
                    let selected = children[random_index(rng, children.len())];
                    return Ok(vec![
                        Command::ForkAt {
                            branch: current,
                            width: self.expansion_width,
                        },
                        Command::Continue { branch: selected },
                    ]);
                }
                BranchState::Active => {
                    return Ok(vec![Command::Continue { branch: current }]);
                }
                BranchState::Expanded => {
                    let Some(child) = self.choose_child(tree, current, rng) else {
                        return Ok(Vec::new());
                    };
                    current = child;
                }
                BranchState::Killed | BranchState::Finalized => return Ok(Vec::new()),
            }
        }
    }

    fn record_event_value(
        event: &EngineEvent,
        tree: &mut BranchTree,
    ) -> Result<(), SchedulerError> {
        if event.is_token_sampled() || matches!(event, EngineEvent::ValueScored { .. }) {
            let branch = event.branch();
            let node = tree
                .get(branch)
                .ok_or(SchedulerError::UnknownBranch(branch))?;
            if node.has_value() {
                tree.backpropagate(branch, node.value_estimate())?;
            }
        }
        Ok(())
    }

    fn select_without_forking(&self, tree: &BranchTree, rng: &mut PolicyRng) -> Option<BranchId> {
        let mut current = tree.root();
        loop {
            let node = tree.get(current)?;
            match node.state() {
                BranchState::Active => return Some(current),
                BranchState::Expanded => current = self.choose_child(tree, current, rng)?,
                BranchState::Killed | BranchState::Finalized => return None,
            }
        }
    }
}

impl Policy for MctsPolicy {
    fn on_event(
        &mut self,
        event: &EngineEvent,
        tree: &mut BranchTree,
        rng: &mut PolicyRng,
    ) -> Result<Vec<Command>, SchedulerError> {
        if (event.is_token_sampled() || matches!(event, EngineEvent::ValueScored { .. }))
            && !tree
                .get(event.branch())
                .ok_or(SchedulerError::UnknownBranch(event.branch()))?
                .has_value()
        {
            return Ok(Vec::new());
        }
        Self::record_event_value(event, tree)?;
        self.select(tree, rng)
    }

    fn on_event_without_forking(
        &mut self,
        event: &EngineEvent,
        tree: &mut BranchTree,
        rng: &mut PolicyRng,
    ) -> Result<Vec<Command>, SchedulerError> {
        Self::record_event_value(event, tree)?;
        Ok(self
            .select_without_forking(tree, rng)
            .map(|branch| vec![Command::Continue { branch }])
            .unwrap_or_default())
    }

    fn on_adaptive_fork(
        &mut self,
        event: &EngineEvent,
        tree: &mut BranchTree,
        children: &[BranchId],
        rng: &mut PolicyRng,
    ) -> Result<Vec<Command>, SchedulerError> {
        Self::record_event_value(event, tree)?;
        if children.is_empty() {
            return Ok(Vec::new());
        }
        Ok(vec![Command::Continue {
            branch: children[random_index(rng, children.len())],
        }])
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;

    use super::*;

    #[test]
    fn near_equal_uct_scores_use_the_stable_branch_id_tie_break() {
        let policy = MctsPolicy::new(MctsConfig {
            expansion_width: 2,
            max_depth: 2,
            exploration_weight: 1.0,
        })
        .unwrap();
        let mut tree = BranchTree::new();
        let root = tree.root();
        let children = tree.fork(root, 2).unwrap();
        tree.backpropagate(children[0], 1.0).unwrap();
        tree.backpropagate(children[1], 1.0 + 1.0e-15).unwrap();

        for seed in 0..32 {
            let mut rng = PolicyRng::seed_from_u64(seed);
            assert_eq!(
                policy.choose_child(&tree, root, &mut rng),
                Some(children[0])
            );
        }
    }
}
