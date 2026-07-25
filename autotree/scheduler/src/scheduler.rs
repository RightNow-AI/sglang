use std::collections::{BTreeMap, BTreeSet, VecDeque};

use rand::SeedableRng;

use crate::{
    AdaptiveForkConfig, BranchId, BranchState, BranchTree, BudgetController, Command, EngineEvent,
    KillReason, LogprobScorer, Policy, PolicyConfig, PolicyRng, SchedulerError, ValueScorer,
    adaptive::AdaptiveForkController,
};

pub const DEFAULT_MAX_PENDING_EVENTS: u64 = 64;

#[derive(Clone, Debug, PartialEq)]
pub struct SchedulerConfig {
    pub policy: PolicyConfig,
    pub seed: u64,
    pub total_token_budget: u64,
    pub per_branch_token_budget: u64,
    /// Maximum number of nodes in the monotonic branch arena, including the root.
    pub max_total_branches: u64,
    pub speculative_kill_margin: Option<f64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PendingBudgetTerminal {
    Branch,
    Tree,
}

/// Pure event-driven scheduler. It performs no I/O and owns all deterministic state.
pub struct Scheduler {
    tree: BranchTree,
    budget: BudgetController,
    policy: Box<dyn Policy>,
    scorer: Option<Box<dyn ValueScorer>>,
    rng: PolicyRng,
    command_queue: VecDeque<Command>,
    outstanding_continuations: BTreeSet<BranchId>,
    pending_external_values: BTreeMap<BranchId, u64>,
    pending_budget_terminals: BTreeMap<BranchId, PendingBudgetTerminal>,
    accepted_event_count: u64,
    max_pending_events: Option<u64>,
    speculative_kill_margin: Option<f64>,
    adaptive_fork: Option<AdaptiveForkController>,
    adaptive_score_pending: BTreeSet<BranchId>,
    rollback_custom_policy_errors: bool,
}

impl Scheduler {
    pub fn new(config: SchedulerConfig) -> Result<Self, SchedulerError> {
        Self::with_scorer(config, Box::<LogprobScorer>::default())
    }

    pub fn with_scorer(
        config: SchedulerConfig,
        scorer: Box<dyn ValueScorer>,
    ) -> Result<Self, SchedulerError> {
        let policy = config.policy.build()?;
        Self::from_components(config, policy, Some(scorer), None, None, false)
    }

    pub fn new_with_adaptive_forking(
        config: SchedulerConfig,
        adaptive_fork: AdaptiveForkConfig,
    ) -> Result<Self, SchedulerError> {
        Self::with_scorer_and_adaptive_forking(
            config,
            Box::<LogprobScorer>::default(),
            adaptive_fork,
        )
    }

    pub fn with_scorer_and_adaptive_forking(
        config: SchedulerConfig,
        scorer: Box<dyn ValueScorer>,
        adaptive_fork: AdaptiveForkConfig,
    ) -> Result<Self, SchedulerError> {
        let policy = config.policy.build()?;
        Self::from_components(
            config,
            policy,
            Some(scorer),
            None,
            Some(adaptive_fork),
            false,
        )
    }

    /// Builds a scheduler whose engine supplies one `ValueScored` event after each token.
    pub fn with_external_values(config: SchedulerConfig) -> Result<Self, SchedulerError> {
        Self::with_external_values_and_pending_event_budget(config, DEFAULT_MAX_PENDING_EVENTS)
    }

    /// Builds an external-value scheduler that deterministically substitutes the branch's
    /// cumulative logprob after `max_pending_events` subsequently accepted engine events.
    pub fn with_external_values_and_pending_event_budget(
        config: SchedulerConfig,
        max_pending_events: u64,
    ) -> Result<Self, SchedulerError> {
        if max_pending_events == 0 {
            return Err(SchedulerError::InvalidConfig(
                "max_pending_events must be greater than zero",
            ));
        }
        let policy = config.policy.build()?;
        Self::from_components(config, policy, None, Some(max_pending_events), None, false)
    }

    pub fn with_external_values_and_adaptive_forking(
        config: SchedulerConfig,
        adaptive_fork: AdaptiveForkConfig,
    ) -> Result<Self, SchedulerError> {
        Self::with_external_values_and_adaptive_forking_and_pending_event_budget(
            config,
            adaptive_fork,
            DEFAULT_MAX_PENDING_EVENTS,
        )
    }

    pub fn with_external_values_and_adaptive_forking_and_pending_event_budget(
        config: SchedulerConfig,
        adaptive_fork: AdaptiveForkConfig,
        max_pending_events: u64,
    ) -> Result<Self, SchedulerError> {
        if max_pending_events == 0 {
            return Err(SchedulerError::InvalidConfig(
                "max_pending_events must be greater than zero",
            ));
        }
        let policy = config.policy.build()?;
        Self::from_components(
            config,
            policy,
            None,
            Some(max_pending_events),
            Some(adaptive_fork),
            false,
        )
    }

    pub fn with_components(
        config: SchedulerConfig,
        policy: Box<dyn Policy>,
        scorer: Box<dyn ValueScorer>,
    ) -> Result<Self, SchedulerError> {
        Self::from_components(config, policy, Some(scorer), None, None, true)
    }

    fn from_components(
        config: SchedulerConfig,
        policy: Box<dyn Policy>,
        scorer: Option<Box<dyn ValueScorer>>,
        max_pending_events: Option<u64>,
        adaptive_fork: Option<AdaptiveForkConfig>,
        rollback_custom_policy_errors: bool,
    ) -> Result<Self, SchedulerError> {
        if config
            .speculative_kill_margin
            .is_some_and(|margin| !margin.is_finite() || margin < 0.0)
        {
            return Err(SchedulerError::InvalidConfig(
                "speculative_kill_margin must be finite and non-negative",
            ));
        }
        let budget =
            BudgetController::new(config.total_token_budget, config.per_branch_token_budget)?;
        let adaptive_fork = adaptive_fork.map(AdaptiveForkController::new).transpose()?;
        Ok(Self {
            tree: BranchTree::with_max_total_branches(config.max_total_branches)?,
            budget,
            policy,
            scorer,
            rng: PolicyRng::seed_from_u64(config.seed),
            command_queue: VecDeque::new(),
            outstanding_continuations: BTreeSet::new(),
            pending_external_values: BTreeMap::new(),
            pending_budget_terminals: BTreeMap::new(),
            accepted_event_count: 0,
            max_pending_events,
            speculative_kill_margin: config.speculative_kill_margin,
            adaptive_fork,
            adaptive_score_pending: BTreeSet::new(),
            rollback_custom_policy_errors,
        })
    }

    #[must_use]
    pub const fn tree(&self) -> &BranchTree {
        &self.tree
    }

    #[must_use]
    pub const fn budget(&self) -> &BudgetController {
        &self.budget
    }

    pub fn feed_event(&mut self, event: EngineEvent) -> Result<(), SchedulerError> {
        if !self.rollback_custom_policy_errors {
            return self.feed_event_inner(event);
        }
        let tree = self.tree.clone();
        let budget = self.budget.clone();
        let rng = self.rng.clone();
        let command_queue = self.command_queue.clone();
        let outstanding_continuations = self.outstanding_continuations.clone();
        let pending_external_values = self.pending_external_values.clone();
        let pending_budget_terminals = self.pending_budget_terminals.clone();
        let adaptive_fork = self.adaptive_fork.clone();
        let adaptive_score_pending = self.adaptive_score_pending.clone();
        let accepted_event_count = self.accepted_event_count;

        if let Err(error) = self.feed_event_inner(event) {
            self.tree = tree;
            self.budget = budget;
            self.rng = rng;
            self.command_queue = command_queue;
            self.outstanding_continuations = outstanding_continuations;
            self.pending_external_values = pending_external_values;
            self.pending_budget_terminals = pending_budget_terminals;
            self.adaptive_fork = adaptive_fork;
            self.adaptive_score_pending = adaptive_score_pending;
            self.accepted_event_count = accepted_event_count;
            return Err(error);
        }
        Ok(())
    }

    fn feed_event_inner(&mut self, event: EngineEvent) -> Result<(), SchedulerError> {
        let accepted_event_count = self
            .accepted_event_count
            .checked_add(1)
            .ok_or(SchedulerError::CounterOverflow("accepted event count"))?;
        self.process_event(event)?;
        self.accepted_event_count = accepted_event_count;
        self.resolve_expired_pending_scores()
    }

    fn process_event(&mut self, event: EngineEvent) -> Result<(), SchedulerError> {
        if event
            .entropy()
            .is_some_and(|entropy| !entropy.is_finite() || entropy < 0.0)
        {
            return Err(SchedulerError::InvalidNumber("entropy"));
        }
        let branch = event.branch();
        let node = self
            .tree
            .get(branch)
            .ok_or(SchedulerError::UnknownBranch(branch))?;
        let accepts_event = match &event {
            EngineEvent::ValueScored { .. } => node.state().is_live(),
            EngineEvent::TokenSampled { .. }
            | EngineEvent::TokenSampledWithEos { .. }
            | EngineEvent::TokenSampledWithMetadata { .. }
            | EngineEvent::BranchExhausted { .. } => node.state() == BranchState::Active,
        };
        if !accepts_event {
            return Err(SchedulerError::BranchNotActive(branch));
        }
        if event.is_token_sampled()
            && self.scorer.is_none()
            && self.pending_external_values.contains_key(&branch)
        {
            return Err(SchedulerError::ValueScorePending(branch));
        }

        let mut emitted = Vec::new();
        let mut tree_budget_exhausted = false;
        let eos = event.is_eos();
        match &event {
            EngineEvent::TokenSampled {
                branch, logprob, ..
            }
            | EngineEvent::TokenSampledWithEos {
                branch, logprob, ..
            }
            | EngineEvent::TokenSampledWithMetadata {
                branch, logprob, ..
            } => {
                let projected = self.tree.projected_after_token(*branch, *logprob)?;
                let score = self.scorer.as_ref().map(|scorer| scorer.score(&projected));
                if score.is_some_and(|score| !score.is_finite()) {
                    return Err(SchedulerError::InvalidNumber("value score"));
                }
                let branch_tokens = self
                    .tree
                    .get(*branch)
                    .ok_or(SchedulerError::UnknownBranch(*branch))?
                    .tokens_generated();
                let pending_deadline = (self.scorer.is_none() && !eos)
                    .then(|| self.pending_deadline_for_current_event())
                    .transpose()?;
                let outcome = self.budget.consume(*branch, branch_tokens)?;
                self.tree.record_token(*branch, *logprob)?;
                if let Some(score) = score {
                    self.tree.record_value(*branch, score)?;
                } else if let Some(deadline) = pending_deadline {
                    self.tree.mark_value_pending(*branch)?;
                    self.pending_external_values.insert(*branch, deadline);
                }
                self.remove_queued_continue(*branch);
                let external_score_pending = self.scorer.is_none() && !eos;
                if external_score_pending
                    && event.entropy().is_some()
                    && self.adaptive_fork.is_some()
                {
                    self.adaptive_score_pending.insert(*branch);
                }
                tree_budget_exhausted = outcome.tree_exhausted && !external_score_pending;
                if eos {
                    self.tree.finalize(*branch)?;
                    emitted.push(Command::Finalize { branch: *branch });
                    emitted.extend(self.reclaim_completed_ancestors(*branch)?);
                    tree_budget_exhausted = outcome.tree_exhausted;
                } else if outcome.tree_exhausted && external_score_pending {
                    self.pending_budget_terminals
                        .insert(*branch, PendingBudgetTerminal::Tree);
                } else if outcome.branch_exhausted && external_score_pending {
                    self.pending_budget_terminals
                        .insert(*branch, PendingBudgetTerminal::Branch);
                } else if outcome.branch_exhausted && !outcome.tree_exhausted {
                    self.tree.finalize(*branch)?;
                    emitted.push(Command::Finalize { branch: *branch });
                    emitted.extend(self.reclaim_completed_ancestors(*branch)?);
                }
            }
            EngineEvent::BranchExhausted { branch } => {
                self.remove_queued_continue(*branch);
                self.pending_external_values.remove(branch);
                self.pending_budget_terminals.remove(branch);
                self.adaptive_score_pending.remove(branch);
                self.tree.finalize(*branch)?;
                emitted.push(Command::Finalize { branch: *branch });
                emitted.extend(self.reclaim_completed_ancestors(*branch)?);
            }
            EngineEvent::ValueScored { branch, score } => {
                if self.scorer.is_some() || !self.pending_external_values.contains_key(branch) {
                    return Err(SchedulerError::UnexpectedValueScore(*branch));
                }
                self.tree.record_value(*branch, *score)?;
                self.pending_external_values.remove(branch);
                match self.pending_budget_terminals.remove(branch) {
                    Some(PendingBudgetTerminal::Tree) => tree_budget_exhausted = true,
                    Some(PendingBudgetTerminal::Branch) => {
                        self.tree.finalize(*branch)?;
                        emitted.push(Command::Finalize { branch: *branch });
                        emitted.extend(self.reclaim_completed_ancestors(*branch)?);
                    }
                    None => {}
                }
            }
        }

        if tree_budget_exhausted {
            emitted.extend(self.terminate_tree(KillReason::TreeBudgetExhausted)?);
            self.pending_external_values.clear();
            self.pending_budget_terminals.clear();
            self.adaptive_score_pending.clear();
            self.enqueue_commands(emitted);
            return Ok(());
        }

        self.retain_live_pending_scores();
        if eos {
            self.enqueue_commands(emitted);
            return Ok(());
        }

        emitted.extend(self.speculative_prune()?);
        if self.pending_budget_terminals.is_empty() {
            let tree_before_policy = self.tree.clone();
            let adaptive_score_followup = if let EngineEvent::ValueScored { branch, .. } = &event {
                self.adaptive_score_pending.remove(branch)
            } else {
                false
            };
            let policy_commands = self.policy_commands(&event, adaptive_score_followup)?;
            self.validate_policy_commands(&tree_before_policy, &policy_commands)?;
            let terminal_branches: Vec<_> = policy_commands
                .iter()
                .filter_map(|command| match command {
                    Command::Kill { branch, .. } | Command::Finalize { branch } => Some(*branch),
                    Command::ForkAt { .. } | Command::Continue { .. } => None,
                })
                .collect();
            emitted.extend(policy_commands);
            for branch in terminal_branches {
                emitted.extend(self.reclaim_completed_ancestors(branch)?);
            }
        }
        self.retain_live_pending_scores();
        self.enqueue_commands(emitted);
        Ok(())
    }

    fn retain_live_pending_scores(&mut self) {
        self.pending_external_values.retain(|branch, _| {
            self.tree
                .get(*branch)
                .is_some_and(|node| node.state().is_live())
        });
        self.pending_budget_terminals.retain(|branch, _| {
            self.tree
                .get(*branch)
                .is_some_and(|node| node.state().is_live())
        });
        self.adaptive_score_pending.retain(|branch| {
            self.tree
                .get(*branch)
                .is_some_and(|node| node.state().is_live())
        });
    }

    fn policy_commands(
        &mut self,
        event: &EngineEvent,
        adaptive_score_followup: bool,
    ) -> Result<Vec<Command>, SchedulerError> {
        if let (Some(entropy), Some(adaptive)) = (event.entropy(), self.adaptive_fork.as_ref()) {
            let branch = event.branch();
            let can_fork = self
                .tree
                .get(branch)
                .is_some_and(|node| node.state() == BranchState::Active);
            let reserved = u64::try_from(self.outstanding_continuations.len()).unwrap_or(u64::MAX);
            let width = can_fork
                .then(|| {
                    adaptive.planned_width(branch, entropy, &self.tree, &self.budget, reserved)
                })
                .transpose()?
                .flatten();
            if let Some(width) = width {
                let parent_tokens_generated = self
                    .tree
                    .get(branch)
                    .ok_or(SchedulerError::UnknownBranch(branch))?
                    .tokens_generated();
                let children = self.tree.fork(branch, width)?;
                self.adaptive_fork
                    .as_mut()
                    .expect("adaptive controller was present")
                    .record_fork(&children, parent_tokens_generated);
                let mut commands = vec![Command::ForkAt { branch, width }];
                commands.extend(self.policy.on_adaptive_fork(
                    event,
                    &mut self.tree,
                    &children,
                    &mut self.rng,
                )?);
                return Ok(commands);
            }
            return self
                .policy
                .on_event_without_forking(event, &mut self.tree, &mut self.rng);
        }
        if adaptive_score_followup {
            return self
                .policy
                .on_event_without_forking(event, &mut self.tree, &mut self.rng);
        }
        self.policy.on_event(event, &mut self.tree, &mut self.rng)
    }

    fn pending_deadline_for_current_event(&self) -> Result<u64, SchedulerError> {
        self.accepted_event_count
            .checked_add(1)
            .and_then(|count| count.checked_add(self.max_pending_events?))
            .ok_or(SchedulerError::CounterOverflow("pending score deadline"))
    }

    fn resolve_expired_pending_scores(&mut self) -> Result<(), SchedulerError> {
        let expired: Vec<_> = self
            .pending_external_values
            .iter()
            .filter_map(|(branch, deadline)| {
                (*deadline <= self.accepted_event_count).then_some(*branch)
            })
            .collect();
        for branch in expired {
            if !self.pending_external_values.contains_key(&branch) {
                continue;
            }
            let proxy = self
                .tree
                .get(branch)
                .ok_or(SchedulerError::UnknownBranch(branch))?
                .cumulative_logprob();
            self.process_event(EngineEvent::ValueScored {
                branch,
                score: proxy,
            })?;
        }
        Ok(())
    }

    fn validate_policy_commands(
        &self,
        tree_before_policy: &BranchTree,
        commands: &[Command],
    ) -> Result<(), SchedulerError> {
        let mut expected = tree_before_policy.clone();
        for command in commands {
            match *command {
                Command::ForkAt { branch, width } => {
                    let _ = expected.fork(branch, width)?;
                }
                Command::Kill { branch, reason } => expected.kill(branch, reason)?,
                Command::Finalize { branch } => expected.finalize(branch)?,
                Command::Continue { branch } => {
                    let node = expected
                        .get(branch)
                        .ok_or(SchedulerError::UnknownBranch(branch))?;
                    if node.state() != BranchState::Active {
                        return Err(SchedulerError::BranchNotActive(branch));
                    }
                }
            }
        }
        if !expected.has_same_structure(&self.tree) {
            return Err(SchedulerError::PolicyCommandTreeMismatch);
        }
        Ok(())
    }

    #[must_use]
    pub fn poll_commands(&mut self) -> Vec<Command> {
        self.command_queue.drain(..).collect()
    }

    #[cfg(any(feature = "python", test))]
    pub(crate) const fn pending_commands(&self) -> &VecDeque<Command> {
        &self.command_queue
    }

    #[cfg(any(feature = "python", test))]
    pub(crate) fn try_convert_pending_commands<T, E>(
        &mut self,
        convert: impl FnOnce(&VecDeque<Command>) -> Result<T, E>,
    ) -> Result<T, E> {
        let output = convert(self.pending_commands())?;
        self.command_queue.clear();
        Ok(output)
    }

    pub fn drain(&mut self) -> Result<(), SchedulerError> {
        let commands = self.terminate_tree(KillReason::Drained)?;
        self.outstanding_continuations.clear();
        self.pending_external_values.clear();
        self.pending_budget_terminals.clear();
        self.adaptive_score_pending.clear();
        self.enqueue_commands(commands);
        Ok(())
    }

    fn enqueue_commands(&mut self, commands: impl IntoIterator<Item = Command>) {
        for command in commands {
            match command {
                Command::ForkAt { branch, width } => {
                    self.remove_queued_continue(branch);
                    self.command_queue
                        .push_back(Command::ForkAt { branch, width });
                }
                Command::Kill { branch, reason } => {
                    self.remove_queued_continue(branch);
                    self.command_queue
                        .push_back(Command::Kill { branch, reason });
                }
                Command::Finalize { branch } => {
                    self.remove_queued_continue(branch);
                    self.command_queue.push_back(Command::Finalize { branch });
                }
                Command::Continue { branch } => {
                    let branch_can_advance = self.tree.get(branch).is_some_and(|node| {
                        node.state() == BranchState::Active
                            && node.tokens_generated() < self.budget.per_branch_limit()
                    });
                    let reserved =
                        u64::try_from(self.outstanding_continuations.len()).unwrap_or(u64::MAX);
                    if branch_can_advance
                        && !self.pending_external_values.contains_key(&branch)
                        && !self.outstanding_continuations.contains(&branch)
                        && reserved < self.budget.remaining_total()
                    {
                        self.outstanding_continuations.insert(branch);
                        self.command_queue.push_back(Command::Continue { branch });
                    }
                }
            }
        }
    }

    fn remove_queued_continue(&mut self, branch: BranchId) {
        self.outstanding_continuations.remove(&branch);
        self.command_queue.retain(
            |command| !matches!(command, Command::Continue { branch: queued } if *queued == branch),
        );
    }

    fn speculative_prune(&mut self) -> Result<Vec<Command>, SchedulerError> {
        let Some(margin) = self.speculative_kill_margin else {
            return Ok(Vec::new());
        };
        let scored: Vec<_> = self
            .tree
            .active_frontier()
            .into_iter()
            .filter(|branch| {
                self.tree
                    .get(*branch)
                    .is_some_and(crate::BranchNode::has_value)
            })
            .collect();
        let Some(best) = scored.iter().copied().min_by(|left, right| {
            let left_node = self.tree.get(*left).expect("known frontier branch");
            let right_node = self.tree.get(*right).expect("known frontier branch");
            right_node
                .value_estimate()
                .total_cmp(&left_node.value_estimate())
                .then_with(|| left.cmp(right))
        }) else {
            return Ok(Vec::new());
        };
        let best_value = self
            .tree
            .get(best)
            .expect("known frontier branch")
            .value_estimate();
        let mut victims: Vec<_> = scored
            .into_iter()
            .filter(|branch| {
                *branch != best
                    && best_value
                        - self
                            .tree
                            .get(*branch)
                            .expect("known frontier branch")
                            .value_estimate()
                        > margin
            })
            .collect();
        victims.sort_unstable();

        let mut commands = Vec::new();
        for victim in victims {
            self.tree.kill(victim, KillReason::SpeculativeKill)?;
            commands.push(Command::Kill {
                branch: victim,
                reason: KillReason::SpeculativeKill,
            });
            commands.extend(self.reclaim_completed_ancestors(victim)?);
        }
        Ok(commands)
    }

    fn reclaim_completed_ancestors(
        &mut self,
        branch: BranchId,
    ) -> Result<Vec<Command>, SchedulerError> {
        let mut commands = Vec::new();
        let mut current = self.tree.parent(branch)?;
        while let Some(parent) = current {
            let parent_state = self
                .tree
                .get(parent)
                .ok_or(SchedulerError::UnknownBranch(parent))?
                .state();
            if parent_state.is_terminal() {
                current = self.tree.parent(parent)?;
                continue;
            }
            let all_children_terminal = self.tree.children(parent)?.iter().all(|child| {
                self.tree
                    .get(*child)
                    .is_some_and(|node| node.state().is_terminal())
            });
            if !all_children_terminal {
                break;
            }
            self.tree.kill(parent, KillReason::AncestorReclaimed)?;
            commands.push(Command::Kill {
                branch: parent,
                reason: KillReason::AncestorReclaimed,
            });
            current = self.tree.parent(parent)?;
        }
        Ok(commands)
    }

    fn terminate_tree(&mut self, reason: KillReason) -> Result<Vec<Command>, SchedulerError> {
        let mut commands = Vec::new();
        let mut active = self.tree.active_frontier();
        active.sort_by(|left, right| {
            let left_node = self.tree.get(*left).expect("known frontier branch");
            let right_node = self.tree.get(*right).expect("known frontier branch");
            // Values and logprobs are unrelated scales. Rank the scored tier first, then
            // compare only like-for-like numbers within each tier.
            right_node
                .has_value()
                .cmp(&left_node.has_value())
                .then_with(|| {
                    let left_score = if left_node.has_value() {
                        left_node.value_estimate()
                    } else {
                        left_node.cumulative_logprob()
                    };
                    let right_score = if right_node.has_value() {
                        right_node.value_estimate()
                    } else {
                        right_node.cumulative_logprob()
                    };
                    right_score.total_cmp(&left_score)
                })
                .then_with(|| left.cmp(right))
        });

        let best = active.first().copied();
        let mut victims: Vec<_> = active.into_iter().skip(1).collect();
        victims.sort_unstable();
        for victim in victims {
            self.tree.kill(victim, reason)?;
            commands.push(Command::Kill {
                branch: victim,
                reason,
            });
        }
        if let Some(best) = best {
            self.tree.finalize(best)?;
            commands.push(Command::Finalize { branch: best });
        }

        loop {
            let mut reclaimable: Vec<_> = self
                .tree
                .iter()
                .filter(|node| {
                    node.state() == BranchState::Expanded
                        && node.children().iter().all(|child| {
                            self.tree
                                .get(*child)
                                .is_some_and(|child_node| child_node.state().is_terminal())
                        })
                })
                .map(|node| (node.depth(), node.id()))
                .collect();
            if reclaimable.is_empty() {
                break;
            }
            reclaimable.sort_by(|left, right| right.cmp(left));
            for (_, branch) in reclaimable {
                if self
                    .tree
                    .get(branch)
                    .is_some_and(|node| node.state() == BranchState::Expanded)
                {
                    self.tree.kill(branch, reason)?;
                    commands.push(Command::Kill { branch, reason });
                }
            }
        }
        Ok(commands)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BeamConfig;

    #[test]
    fn failed_command_conversion_leaves_the_queue_untouched() {
        let mut scheduler = Scheduler::new(SchedulerConfig {
            policy: PolicyConfig::Beam(BeamConfig {
                width: 1,
                fork_width: 1,
                fork_at_tokens: Vec::new(),
            }),
            seed: 0,
            total_token_budget: 10,
            per_branch_token_budget: 10,
            max_total_branches: crate::DEFAULT_MAX_TOTAL_BRANCHES,
            speculative_kill_margin: None,
        })
        .unwrap();
        scheduler
            .feed_event(EngineEvent::TokenSampled {
                branch: BranchId(0),
                token: 0,
                logprob: 0.0,
            })
            .unwrap();

        let result: Result<(), &str> =
            scheduler.try_convert_pending_commands(|_| Err("injected allocation failure"));

        assert_eq!(result, Err("injected allocation failure"));
        assert_eq!(
            scheduler.poll_commands(),
            vec![Command::Continue {
                branch: BranchId(0),
            }]
        );
    }
}
