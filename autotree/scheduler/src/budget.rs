use crate::{BranchId, SchedulerError};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct BudgetOutcome {
    pub branch_exhausted: bool,
    pub tree_exhausted: bool,
}

/// Exact hard limits for generated tokens in one tree and on one root-to-leaf path.
#[derive(Clone, Debug)]
pub struct BudgetController {
    total_limit: u64,
    per_branch_limit: u64,
    total_consumed: u64,
}

impl BudgetController {
    pub fn new(total_limit: u64, per_branch_limit: u64) -> Result<Self, SchedulerError> {
        if total_limit == 0 {
            return Err(SchedulerError::InvalidConfig(
                "total_token_budget must be greater than zero",
            ));
        }
        if per_branch_limit == 0 {
            return Err(SchedulerError::InvalidConfig(
                "per_branch_token_budget must be greater than zero",
            ));
        }
        Ok(Self {
            total_limit,
            per_branch_limit,
            total_consumed: 0,
        })
    }

    #[must_use]
    pub const fn total_limit(&self) -> u64 {
        self.total_limit
    }

    #[must_use]
    pub const fn per_branch_limit(&self) -> u64 {
        self.per_branch_limit
    }

    #[must_use]
    pub const fn total_consumed(&self) -> u64 {
        self.total_consumed
    }

    #[must_use]
    pub const fn remaining_total(&self) -> u64 {
        self.total_limit.saturating_sub(self.total_consumed)
    }

    pub(crate) fn consume(
        &mut self,
        branch: BranchId,
        branch_tokens_before: u64,
    ) -> Result<BudgetOutcome, SchedulerError> {
        if self.total_consumed >= self.total_limit || branch_tokens_before >= self.per_branch_limit
        {
            return Err(SchedulerError::BudgetAlreadyExhausted(branch));
        }
        self.total_consumed = self
            .total_consumed
            .checked_add(1)
            .ok_or(SchedulerError::CounterOverflow("tree token count"))?;
        let branch_tokens_after = branch_tokens_before
            .checked_add(1)
            .ok_or(SchedulerError::CounterOverflow("branch token count"))?;
        Ok(BudgetOutcome {
            branch_exhausted: branch_tokens_after == self.per_branch_limit,
            tree_exhausted: self.total_consumed == self.total_limit,
        })
    }
}
