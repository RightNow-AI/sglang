use crate::{
    BestFirstConfig, BranchId, BranchTree, Command, EngineEvent, MAX_BRANCH_WIDTH, Policy,
    PolicyRng, SchedulerError,
};

#[derive(Clone, Debug)]
pub struct BestFirstPolicy {
    expansion_width: u32,
    max_depth: u32,
}

impl BestFirstPolicy {
    pub fn new(config: BestFirstConfig) -> Result<Self, SchedulerError> {
        if config.expansion_width == 0 || config.expansion_width > MAX_BRANCH_WIDTH {
            return Err(SchedulerError::InvalidConfig(
                "best-first expansion_width must be between 1 and MAX_BRANCH_WIDTH",
            ));
        }
        if config.max_depth == 0 {
            return Err(SchedulerError::InvalidConfig(
                "best-first max_depth must be greater than zero",
            ));
        }
        Ok(Self {
            expansion_width: config.expansion_width,
            max_depth: config.max_depth,
        })
    }

    fn ranked_frontier(tree: &BranchTree) -> Vec<BranchId> {
        let mut frontier = tree.active_frontier();
        frontier.sort_by(|left, right| {
            let left_node = tree.get(*left).expect("frontier ids come from the arena");
            let right_node = tree.get(*right).expect("frontier ids come from the arena");
            normalize_signed_zero(right_node.value_estimate())
                .total_cmp(&normalize_signed_zero(left_node.value_estimate()))
                .then_with(|| {
                    right_node
                        .cumulative_logprob()
                        .total_cmp(&left_node.cumulative_logprob())
                })
                .then_with(|| left.cmp(right))
        });
        frontier
    }

    fn select_without_forking(tree: &BranchTree) -> Vec<Command> {
        let ranked = Self::ranked_frontier(tree);
        if ranked.is_empty()
            || ranked
                .iter()
                .any(|branch| !tree.get(*branch).expect("known branch").has_value())
        {
            return Vec::new();
        }
        vec![Command::Continue { branch: ranked[0] }]
    }
}

fn normalize_signed_zero(value: f64) -> f64 {
    if value == 0.0 { 0.0 } else { value }
}

impl Policy for BestFirstPolicy {
    fn on_event(
        &mut self,
        _event: &EngineEvent,
        tree: &mut BranchTree,
        _rng: &mut PolicyRng,
    ) -> Result<Vec<Command>, SchedulerError> {
        // External scores arrive independently. Unscored branches wait for their own score,
        // while already-scored peers remain eligible so one delayed value cannot stall the tree.
        let ranked: Vec<_> = Self::ranked_frontier(tree)
            .into_iter()
            .filter(|branch| tree.get(*branch).expect("known branch").has_value())
            .collect();
        if ranked.is_empty() {
            return Ok(Vec::new());
        }

        let selected = ranked[0];
        let depth = tree.get(selected).expect("known branch").depth();
        if depth >= self.max_depth {
            return Ok(vec![Command::Continue { branch: selected }]);
        }

        let children = tree.fork(selected, self.expansion_width)?;
        let mut commands = vec![Command::ForkAt {
            branch: selected,
            width: self.expansion_width,
        }];
        commands.extend(
            children
                .into_iter()
                .map(|branch| Command::Continue { branch }),
        );
        Ok(commands)
    }

    fn on_event_without_forking(
        &mut self,
        _event: &EngineEvent,
        tree: &mut BranchTree,
        _rng: &mut PolicyRng,
    ) -> Result<Vec<Command>, SchedulerError> {
        Ok(Self::select_without_forking(tree))
    }

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
