use crate::SchedulerError;

/// Defensive ceiling for one expansion. Public serving inputs must stay below this bound.
pub const MAX_BRANCH_WIDTH: u32 = 1_024;

/// Default cap for the monotonic branch arena (root included).
///
/// This permits broad search while bounding scheduler metadata to tens of thousands of nodes
/// instead of allowing one policy event to allocate millions.
pub const DEFAULT_MAX_TOTAL_BRANCHES: u64 = 65_536;

/// Stable arena handle for a branch. The root is always `BranchId(0)`.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct BranchId(pub u64);

/// Lifecycle of a branch in the scheduler-owned tree.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BranchState {
    Active,
    Expanded,
    Killed,
    Finalized,
}

impl BranchState {
    #[must_use]
    pub const fn is_live(self) -> bool {
        matches!(self, Self::Active | Self::Expanded)
    }

    #[must_use]
    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Killed | Self::Finalized)
    }
}

/// Why the scheduler ordered immediate branch reclamation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum KillReason {
    BeamPruned = 0,
    SpeculativeKill = 1,
    BranchBudgetExhausted = 2,
    TreeBudgetExhausted = 3,
    Exhausted = 4,
    Drained = 5,
    AncestorReclaimed = 6,
}

impl KillReason {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BeamPruned => "beam_pruned",
            Self::SpeculativeKill => "speculative_kill",
            Self::BranchBudgetExhausted => "branch_budget_exhausted",
            Self::TreeBudgetExhausted => "tree_budget_exhausted",
            Self::Exhausted => "exhausted",
            Self::Drained => "drained",
            Self::AncestorReclaimed => "ancestor_reclaimed",
        }
    }
}

/// Scheduler metadata for one tree node.
#[derive(Clone, Debug)]
pub struct BranchNode {
    id: BranchId,
    parent: Option<BranchId>,
    children: Vec<BranchId>,
    depth: u32,
    tokens_generated: u64,
    cumulative_logprob: f64,
    value_estimate: f64,
    has_value: bool,
    state: BranchState,
    visits: u64,
    value_sum: f64,
}

impl BranchNode {
    #[must_use]
    pub const fn id(&self) -> BranchId {
        self.id
    }

    #[must_use]
    pub const fn parent(&self) -> Option<BranchId> {
        self.parent
    }

    #[must_use]
    pub fn children(&self) -> &[BranchId] {
        &self.children
    }

    #[must_use]
    pub const fn depth(&self) -> u32 {
        self.depth
    }

    #[must_use]
    pub const fn tokens_generated(&self) -> u64 {
        self.tokens_generated
    }

    #[must_use]
    pub const fn cumulative_logprob(&self) -> f64 {
        self.cumulative_logprob
    }

    #[must_use]
    pub const fn value_estimate(&self) -> f64 {
        self.value_estimate
    }

    #[must_use]
    pub const fn has_value(&self) -> bool {
        self.has_value
    }

    #[must_use]
    pub const fn state(&self) -> BranchState {
        self.state
    }

    #[must_use]
    pub const fn visits(&self) -> u64 {
        self.visits
    }

    #[must_use]
    pub const fn value_sum(&self) -> f64 {
        self.value_sum
    }
}

/// Monotonic arena with O(1) lookup and parent/child navigation.
#[derive(Clone, Debug)]
pub struct BranchTree {
    nodes: Vec<BranchNode>,
    max_total_branches: u64,
}

impl Default for BranchTree {
    fn default() -> Self {
        Self::new()
    }
}

impl BranchTree {
    #[must_use]
    pub fn new() -> Self {
        Self::with_max_total_branches(DEFAULT_MAX_TOTAL_BRANCHES)
            .expect("the default branch cap is valid")
    }

    pub fn with_max_total_branches(max_total_branches: u64) -> Result<Self, SchedulerError> {
        if max_total_branches == 0 {
            return Err(SchedulerError::InvalidConfig(
                "max_total_branches must be greater than zero",
            ));
        }
        Ok(Self {
            nodes: vec![BranchNode {
                id: BranchId(0),
                parent: None,
                children: Vec::new(),
                depth: 0,
                tokens_generated: 0,
                cumulative_logprob: 0.0,
                value_estimate: 0.0,
                has_value: false,
                state: BranchState::Active,
                visits: 0,
                value_sum: 0.0,
            }],
            max_total_branches,
        })
    }

    #[must_use]
    pub const fn root(&self) -> BranchId {
        BranchId(0)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    #[must_use]
    pub const fn max_total_branches(&self) -> u64 {
        self.max_total_branches
    }

    #[must_use]
    pub fn get(&self, branch: BranchId) -> Option<&BranchNode> {
        usize::try_from(branch.0)
            .ok()
            .and_then(|index| self.nodes.get(index))
            .filter(|node| node.id == branch)
    }

    pub fn parent(&self, branch: BranchId) -> Result<Option<BranchId>, SchedulerError> {
        self.get(branch)
            .map(BranchNode::parent)
            .ok_or(SchedulerError::UnknownBranch(branch))
    }

    pub fn children(&self, branch: BranchId) -> Result<&[BranchId], SchedulerError> {
        self.get(branch)
            .map(BranchNode::children)
            .ok_or(SchedulerError::UnknownBranch(branch))
    }

    pub fn iter(&self) -> impl Iterator<Item = &BranchNode> {
        self.nodes.iter()
    }

    pub(crate) fn has_same_structure(&self, other: &Self) -> bool {
        self.nodes.len() == other.nodes.len()
            && self.nodes.iter().zip(&other.nodes).all(|(left, right)| {
                left.id == right.id
                    && left.parent == right.parent
                    && left.children == right.children
                    && left.depth == right.depth
                    && left.state == right.state
            })
    }

    #[must_use]
    pub fn active_frontier(&self) -> Vec<BranchId> {
        self.nodes
            .iter()
            .filter(|node| node.state == BranchState::Active)
            .map(BranchNode::id)
            .collect()
    }

    pub(crate) fn projected_after_token(
        &self,
        branch: BranchId,
        logprob: f64,
    ) -> Result<BranchNode, SchedulerError> {
        if !logprob.is_finite() {
            return Err(SchedulerError::InvalidNumber("logprob"));
        }
        let node = self
            .get(branch)
            .ok_or(SchedulerError::UnknownBranch(branch))?;
        if node.state != BranchState::Active {
            return Err(SchedulerError::BranchNotActive(branch));
        }
        let mut projected = node.clone();
        projected.tokens_generated = projected
            .tokens_generated
            .checked_add(1)
            .ok_or(SchedulerError::CounterOverflow("branch token count"))?;
        projected.cumulative_logprob += logprob;
        if !projected.cumulative_logprob.is_finite() {
            return Err(SchedulerError::InvalidNumber("cumulative logprob"));
        }
        Ok(projected)
    }

    pub fn record_token(&mut self, branch: BranchId, logprob: f64) -> Result<(), SchedulerError> {
        if !logprob.is_finite() {
            return Err(SchedulerError::InvalidNumber("logprob"));
        }
        let node = self.active_mut(branch)?;
        let next_tokens = node
            .tokens_generated
            .checked_add(1)
            .ok_or(SchedulerError::CounterOverflow("branch token count"))?;
        let next_logprob = node.cumulative_logprob + logprob;
        if !next_logprob.is_finite() {
            return Err(SchedulerError::InvalidNumber("cumulative logprob"));
        }
        node.tokens_generated = next_tokens;
        node.cumulative_logprob = next_logprob;
        Ok(())
    }

    pub fn record_value(&mut self, branch: BranchId, score: f64) -> Result<(), SchedulerError> {
        if !score.is_finite() {
            return Err(SchedulerError::InvalidNumber("value score"));
        }
        let node = self.live_mut(branch)?;
        node.value_estimate = score;
        node.has_value = true;
        Ok(())
    }

    pub(crate) fn mark_value_pending(&mut self, branch: BranchId) -> Result<(), SchedulerError> {
        self.active_mut(branch)?.has_value = false;
        Ok(())
    }

    pub fn fork(&mut self, branch: BranchId, width: u32) -> Result<Vec<BranchId>, SchedulerError> {
        if width == 0 {
            return Err(SchedulerError::InvalidWidth(width));
        }
        if width > MAX_BRANCH_WIDTH {
            return Err(SchedulerError::InvalidWidth(width));
        }
        let parent = self
            .get(branch)
            .ok_or(SchedulerError::UnknownBranch(branch))?;
        if parent.state != BranchState::Active {
            return Err(SchedulerError::BranchNotActive(branch));
        }

        let child_depth = parent
            .depth
            .checked_add(1)
            .ok_or(SchedulerError::CounterOverflow("branch depth"))?;
        let parent_tokens = parent.tokens_generated;
        let parent_logprob = parent.cumulative_logprob;
        let parent_value = parent.value_estimate;
        let start = u64::try_from(self.nodes.len())
            .map_err(|_| SchedulerError::CounterOverflow("branch arena"))?;
        let end = start
            .checked_add(u64::from(width))
            .ok_or(SchedulerError::CounterOverflow("branch arena"))?;
        if end > self.max_total_branches {
            return Err(SchedulerError::BranchLimitExceeded {
                limit: self.max_total_branches,
                requested_total: end,
            });
        }
        let children: Vec<_> = (start..end).map(BranchId).collect();

        self.get_mut(branch)?.state = BranchState::Expanded;
        self.get_mut(branch)?.children.clone_from(&children);
        self.nodes
            .extend(children.iter().copied().map(|id| BranchNode {
                id,
                parent: Some(branch),
                children: Vec::new(),
                depth: child_depth,
                tokens_generated: parent_tokens,
                cumulative_logprob: parent_logprob,
                value_estimate: parent_value,
                has_value: false,
                state: BranchState::Active,
                visits: 0,
                value_sum: 0.0,
            }));
        Ok(children)
    }

    pub fn kill(&mut self, branch: BranchId, _reason: KillReason) -> Result<(), SchedulerError> {
        let node = self
            .get(branch)
            .ok_or(SchedulerError::UnknownBranch(branch))?;
        if !node.state.is_live() {
            return Err(SchedulerError::BranchNotActive(branch));
        }
        if node.children.iter().any(|child| {
            self.get(*child)
                .is_some_and(|child_node| child_node.state.is_live())
        }) {
            return Err(SchedulerError::BranchHasLiveChildren(branch));
        }
        self.get_mut(branch)?.state = BranchState::Killed;
        Ok(())
    }

    pub fn finalize(&mut self, branch: BranchId) -> Result<(), SchedulerError> {
        let node = self.active_mut(branch)?;
        if !node.children.is_empty() {
            return Err(SchedulerError::BranchHasLiveChildren(branch));
        }
        node.state = BranchState::Finalized;
        Ok(())
    }

    pub fn backpropagate(&mut self, branch: BranchId, value: f64) -> Result<(), SchedulerError> {
        if !value.is_finite() {
            return Err(SchedulerError::InvalidNumber("MCTS value"));
        }
        if self.get(branch).is_none() {
            return Err(SchedulerError::UnknownBranch(branch));
        }
        let mut updates = Vec::new();
        let mut current = Some(branch);
        while let Some(id) = current {
            let node = self.get(id).ok_or(SchedulerError::UnknownBranch(id))?;
            let visits = node
                .visits
                .checked_add(1)
                .ok_or(SchedulerError::CounterOverflow("MCTS visit count"))?;
            let value_sum = node.value_sum + value;
            if !value_sum.is_finite() {
                return Err(SchedulerError::InvalidNumber("MCTS value sum"));
            }
            updates.push((id, visits, value_sum));
            current = node.parent;
        }
        for (id, visits, value_sum) in updates {
            let node = self.get_mut(id)?;
            node.visits = visits;
            node.value_sum = value_sum;
        }
        Ok(())
    }

    fn get_mut(&mut self, branch: BranchId) -> Result<&mut BranchNode, SchedulerError> {
        usize::try_from(branch.0)
            .ok()
            .and_then(|index| self.nodes.get_mut(index))
            .filter(|node| node.id == branch)
            .ok_or(SchedulerError::UnknownBranch(branch))
    }

    fn active_mut(&mut self, branch: BranchId) -> Result<&mut BranchNode, SchedulerError> {
        let node = self.get_mut(branch)?;
        if node.state != BranchState::Active {
            return Err(SchedulerError::BranchNotActive(branch));
        }
        Ok(node)
    }

    fn live_mut(&mut self, branch: BranchId) -> Result<&mut BranchNode, SchedulerError> {
        let node = self.get_mut(branch)?;
        if !node.state.is_live() {
            return Err(SchedulerError::BranchNotActive(branch));
        }
        Ok(node)
    }
}
