use std::collections::BTreeMap;

use crate::{BranchId, BranchTree, BudgetController, MAX_BRANCH_WIDTH, SchedulerError};

/// Entropy-triggered fork controls. Entropy is measured in natural-log units (nats).
///
/// Adaptive forking is orthogonal to the configured policy: this controller decides whether and
/// how widely to fork, while beam, best-first, or MCTS still selects and prunes the resulting
/// frontier. The width formula is
/// `min(max_fork_width, min_fork_width + floor((entropy - threshold) / entropy_step))`.
#[derive(Clone, Debug, PartialEq)]
pub struct AdaptiveForkConfig {
    /// Minimum predictive entropy that permits a fork.
    pub entropy_threshold_nats: f64,
    /// Required number of generated tokens between an ancestor fork and a descendant fork.
    pub min_tokens_between_forks: u64,
    /// Maximum number of branch nodes allocated in the tree, including terminal nodes.
    pub max_total_branches: u64,
    /// Maximum tree depth at which a branch may fork.
    pub max_depth: u32,
    /// Width used at the threshold.
    pub min_fork_width: u32,
    /// Upper bound on entropy-scaled width.
    pub max_fork_width: u32,
    /// Entropy excess, in nats, required for each branch beyond `min_fork_width`.
    pub entropy_nats_per_extra_branch: f64,
}

#[derive(Clone, Debug)]
pub(crate) struct AdaptiveForkController {
    config: AdaptiveForkConfig,
    last_fork_token_by_branch: BTreeMap<BranchId, u64>,
}

impl AdaptiveForkController {
    pub(crate) fn new(config: AdaptiveForkConfig) -> Result<Self, SchedulerError> {
        if !config.entropy_threshold_nats.is_finite() || config.entropy_threshold_nats < 0.0 {
            return Err(SchedulerError::InvalidConfig(
                "adaptive entropy threshold must be finite and non-negative",
            ));
        }
        if config.min_tokens_between_forks == 0 {
            return Err(SchedulerError::InvalidConfig(
                "adaptive min_tokens_between_forks must be greater than zero",
            ));
        }
        if config.max_total_branches < 2 {
            return Err(SchedulerError::InvalidConfig(
                "adaptive max_total_branches must be at least two",
            ));
        }
        if config.max_depth == 0 {
            return Err(SchedulerError::InvalidConfig(
                "adaptive max_depth must be greater than zero",
            ));
        }
        if config.min_fork_width < 2
            || config.min_fork_width > MAX_BRANCH_WIDTH
            || config.max_fork_width < config.min_fork_width
            || config.max_fork_width > MAX_BRANCH_WIDTH
        {
            return Err(SchedulerError::InvalidConfig(
                "adaptive fork widths must satisfy 2 <= min <= max <= MAX_BRANCH_WIDTH",
            ));
        }
        if !config.entropy_nats_per_extra_branch.is_finite()
            || config.entropy_nats_per_extra_branch <= 0.0
        {
            return Err(SchedulerError::InvalidConfig(
                "adaptive entropy_nats_per_extra_branch must be finite and positive",
            ));
        }
        Ok(Self {
            config,
            last_fork_token_by_branch: BTreeMap::new(),
        })
    }

    pub(crate) fn planned_width(
        &self,
        branch: BranchId,
        entropy: f64,
        tree: &BranchTree,
        budget: &BudgetController,
        reserved_continuations: u64,
    ) -> Result<Option<u32>, SchedulerError> {
        if entropy < self.config.entropy_threshold_nats {
            return Ok(None);
        }
        let node = tree
            .get(branch)
            .ok_or(SchedulerError::UnknownBranch(branch))?;
        if node.depth() >= self.config.max_depth {
            return Ok(None);
        }
        if self
            .last_fork_token_by_branch
            .get(&branch)
            .is_some_and(|last_fork| {
                node.tokens_generated().saturating_sub(*last_fork)
                    < self.config.min_tokens_between_forks
            })
        {
            return Ok(None);
        }

        let width = self.width_for_entropy(entropy);
        let projected_branches = u64::try_from(tree.len())
            .ok()
            .and_then(|count| count.checked_add(u64::from(width)));
        if projected_branches.is_none_or(|count| count > self.config.max_total_branches) {
            return Ok(None);
        }

        // A fork is minimally viable only if every new child can decode one token. Existing
        // Continue commands reserve their next token first, matching enqueue-time enforcement;
        // the inherited root-to-child path must also remain below the per-branch limit.
        let available_total = budget
            .remaining_total()
            .saturating_sub(reserved_continuations);
        if available_total < u64::from(width)
            || node.tokens_generated() >= budget.per_branch_limit()
        {
            return Ok(None);
        }
        Ok(Some(width))
    }

    pub(crate) fn record_fork(&mut self, children: &[BranchId], parent_tokens_generated: u64) {
        for child in children {
            self.last_fork_token_by_branch
                .insert(*child, parent_tokens_generated);
        }
    }

    fn width_for_entropy(&self, entropy: f64) -> u32 {
        let excess = (entropy - self.config.entropy_threshold_nats).max(0.0);
        let extra = (excess / self.config.entropy_nats_per_extra_branch).floor();
        let extra = if extra >= f64::from(u32::MAX) {
            u32::MAX
        } else {
            extra as u32
        };
        self.config
            .min_fork_width
            .saturating_add(extra)
            .min(self.config.max_fork_width)
    }
}
