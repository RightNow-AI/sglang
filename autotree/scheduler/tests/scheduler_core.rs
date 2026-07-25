use autotree_scheduler::{
    BeamConfig, BestFirstConfig, BranchId, BranchState, BranchTree, Command,
    DEFAULT_MAX_TOTAL_BRANCHES, EngineEvent, KillReason, LogprobScorer, MctsConfig, Policy,
    PolicyConfig, PolicyRng, Scheduler, SchedulerConfig, SchedulerError,
};

struct FailingPolicy;

struct InvalidForkPolicy {
    branch: BranchId,
    width: u32,
}

impl Policy for FailingPolicy {
    fn on_event(
        &mut self,
        event: &EngineEvent,
        tree: &mut BranchTree,
        _rng: &mut PolicyRng,
    ) -> Result<Vec<Command>, SchedulerError> {
        tree.kill(event.branch(), KillReason::Drained)?;
        Err(SchedulerError::InvalidConfig("intentional policy failure"))
    }
}

impl Policy for InvalidForkPolicy {
    fn on_event(
        &mut self,
        _event: &EngineEvent,
        _tree: &mut BranchTree,
        _rng: &mut PolicyRng,
    ) -> Result<Vec<Command>, SchedulerError> {
        Ok(vec![Command::ForkAt {
            branch: self.branch,
            width: self.width,
        }])
    }
}

fn config(
    total_token_budget: u64,
    per_branch_token_budget: u64,
    speculative_kill_margin: Option<f64>,
) -> SchedulerConfig {
    SchedulerConfig {
        policy: PolicyConfig::Beam(BeamConfig {
            width: 2,
            fork_width: 2,
            fork_at_tokens: Vec::new(),
        }),
        seed: 11,
        total_token_budget,
        per_branch_token_budget,
        max_total_branches: DEFAULT_MAX_TOTAL_BRANCHES,
        speculative_kill_margin,
    }
}

#[test]
fn per_branch_budget_terminalizes_on_the_exact_limit() {
    let mut scheduler = Scheduler::new(config(100, 3, None)).unwrap();
    let root = BranchId(0);

    for token in 0..3 {
        scheduler
            .feed_event(EngineEvent::TokenSampled {
                branch: root,
                token,
                logprob: -0.1,
            })
            .unwrap();
    }

    let limit_plus_one = scheduler.feed_event(EngineEvent::TokenSampled {
        branch: root,
        token: 99,
        logprob: -0.1,
    });
    let commands = scheduler.poll_commands();
    assert!(
        limit_plus_one.is_err(),
        "the budget controller accepted token limit+1"
    );
    assert!(commands.contains(&Command::Finalize { branch: root }));
    assert_eq!(scheduler.budget().total_consumed(), 3);
    assert_eq!(scheduler.tree().get(root).unwrap().tokens_generated(), 3);
    assert_eq!(
        scheduler.tree().get(root).unwrap().state(),
        BranchState::Finalized
    );
    assert_eq!(scheduler.budget().total_consumed(), 3);
}

#[test]
fn speculative_kill_is_emitted_immediately_when_frontier_gap_crosses_margin() {
    let mut scheduler = Scheduler::with_external_values(SchedulerConfig {
        policy: PolicyConfig::Beam(BeamConfig {
            width: 2,
            fork_width: 2,
            fork_at_tokens: vec![1],
        }),
        ..config(100, 100, Some(0.5))
    })
    .unwrap();

    scheduler
        .feed_event(EngineEvent::TokenSampled {
            branch: BranchId(0),
            token: 7,
            logprob: -0.1,
        })
        .unwrap();
    let _ = scheduler.poll_commands();

    scheduler
        .feed_event(EngineEvent::ValueScored {
            branch: BranchId(0),
            score: 0.0,
        })
        .unwrap();
    let _ = scheduler.poll_commands();

    scheduler
        .feed_event(EngineEvent::TokenSampled {
            branch: BranchId(1),
            token: 8,
            logprob: -0.1,
        })
        .unwrap();
    let _ = scheduler.poll_commands();
    scheduler
        .feed_event(EngineEvent::ValueScored {
            branch: BranchId(1),
            score: 1.0,
        })
        .unwrap();
    let _ = scheduler.poll_commands();
    scheduler
        .feed_event(EngineEvent::TokenSampled {
            branch: BranchId(2),
            token: 9,
            logprob: -0.1,
        })
        .unwrap();
    let _ = scheduler.poll_commands();
    scheduler
        .feed_event(EngineEvent::ValueScored {
            branch: BranchId(2),
            score: 0.4,
        })
        .unwrap();

    assert_eq!(
        scheduler.poll_commands(),
        vec![Command::Kill {
            branch: BranchId(2),
            reason: KillReason::SpeculativeKill,
        }]
    );
    assert_eq!(
        scheduler.tree().get(BranchId(2)).unwrap().state(),
        BranchState::Killed
    );
}

#[test]
fn invalid_numeric_events_are_rejected_without_spending_budget() {
    let mut scheduler = Scheduler::new(config(10, 10, None)).unwrap();
    assert!(
        scheduler
            .feed_event(EngineEvent::TokenSampled {
                branch: BranchId(0),
                token: 0,
                logprob: f64::NAN,
            })
            .is_err()
    );
    assert_eq!(scheduler.budget().total_consumed(), 0);
}

#[test]
fn batched_poll_drops_stale_and_duplicate_continue_commands() {
    let mut scheduler = Scheduler::with_external_values(SchedulerConfig {
        policy: PolicyConfig::Beam(BeamConfig {
            width: 2,
            fork_width: 2,
            fork_at_tokens: vec![1],
        }),
        ..config(100, 100, Some(0.5))
    })
    .unwrap();

    scheduler
        .feed_event(EngineEvent::TokenSampled {
            branch: BranchId(0),
            token: 0,
            logprob: 0.0,
        })
        .unwrap();
    let _ = scheduler.poll_commands();
    scheduler
        .feed_event(EngineEvent::ValueScored {
            branch: BranchId(0),
            score: 0.0,
        })
        .unwrap();
    let _ = scheduler.poll_commands();

    scheduler
        .feed_event(EngineEvent::TokenSampled {
            branch: BranchId(1),
            token: 1,
            logprob: -0.1,
        })
        .unwrap();
    let _ = scheduler.poll_commands();
    scheduler
        .feed_event(EngineEvent::ValueScored {
            branch: BranchId(1),
            score: 1.0,
        })
        .unwrap();
    scheduler
        .feed_event(EngineEvent::TokenSampled {
            branch: BranchId(2),
            token: 2,
            logprob: -0.1,
        })
        .unwrap();
    scheduler
        .feed_event(EngineEvent::ValueScored {
            branch: BranchId(2),
            score: 0.0,
        })
        .unwrap();

    let commands = scheduler.poll_commands();
    assert!(!commands.contains(&Command::Continue {
        branch: BranchId(2),
    }));
    assert_eq!(
        commands
            .iter()
            .filter(|command| **command
                == Command::Continue {
                    branch: BranchId(1)
                })
            .count(),
        1
    );
    assert!(commands.contains(&Command::Kill {
        branch: BranchId(2),
        reason: KillReason::SpeculativeKill,
    }));
}

#[test]
fn continue_authorizations_never_exceed_remaining_tree_budget() {
    let mut scheduler = Scheduler::new(SchedulerConfig {
        policy: PolicyConfig::Beam(BeamConfig {
            width: 8,
            fork_width: 8,
            fork_at_tokens: vec![1],
        }),
        ..config(2, 10, None)
    })
    .unwrap();

    scheduler
        .feed_event(EngineEvent::TokenSampled {
            branch: BranchId(0),
            token: 0,
            logprob: 0.0,
        })
        .unwrap();
    let commands = scheduler.poll_commands();
    let authorized: Vec<_> = commands
        .iter()
        .filter_map(|command| match command {
            Command::Continue { branch } => Some(*branch),
            _ => None,
        })
        .collect();
    assert_eq!(authorized.len(), 1);

    scheduler
        .feed_event(EngineEvent::TokenSampled {
            branch: authorized[0],
            token: 1,
            logprob: 0.0,
        })
        .expect("every authorized continuation fits within the hard budget");
    assert_eq!(scheduler.budget().total_consumed(), 2);
}

#[test]
fn pathological_branch_widths_are_rejected_before_allocation() {
    let policies = [
        PolicyConfig::Beam(BeamConfig {
            width: 1,
            fork_width: u32::MAX,
            fork_at_tokens: vec![1],
        }),
        PolicyConfig::BestFirst(BestFirstConfig {
            expansion_width: u32::MAX,
            max_depth: 1,
        }),
        PolicyConfig::Mcts(MctsConfig {
            expansion_width: u32::MAX,
            max_depth: 1,
            exploration_weight: 1.0,
        }),
    ];

    for policy in policies {
        assert!(
            Scheduler::new(SchedulerConfig {
                policy,
                ..config(100, 100, None)
            })
            .is_err()
        );
    }
}

#[test]
fn failed_policy_decision_rolls_back_core_state_atomically() {
    let mut scheduler = Scheduler::with_components(
        config(100, 100, None),
        Box::new(FailingPolicy),
        Box::new(LogprobScorer),
    )
    .unwrap();

    assert!(
        scheduler
            .feed_event(EngineEvent::TokenSampled {
                branch: BranchId(0),
                token: 0,
                logprob: -0.1,
            })
            .is_err()
    );
    let root = scheduler.tree().get(BranchId(0)).unwrap();
    assert_eq!(root.state(), BranchState::Active);
    assert_eq!(root.tokens_generated(), 0);
    assert_eq!(scheduler.budget().total_consumed(), 0);
    assert!(scheduler.poll_commands().is_empty());
}

#[test]
fn event_before_poll_consumes_the_queued_continue_reservation() {
    let mut scheduler = Scheduler::new(config(3, 10, None)).unwrap();
    let root = BranchId(0);

    scheduler
        .feed_event(EngineEvent::TokenSampled {
            branch: root,
            token: 0,
            logprob: 0.0,
        })
        .unwrap();
    scheduler
        .feed_event(EngineEvent::TokenSampled {
            branch: root,
            token: 1,
            logprob: 0.0,
        })
        .unwrap();

    let commands = scheduler.poll_commands();
    assert_eq!(
        commands
            .iter()
            .filter(|command| **command == Command::Continue { branch: root })
            .count(),
        1
    );
    assert_eq!(scheduler.budget().total_consumed(), 2);
    assert_eq!(scheduler.budget().remaining_total(), 1);
}

#[test]
fn invalid_custom_policy_forks_are_rejected_without_state_or_queue_changes() {
    let cases = [
        (
            BranchId(999),
            1,
            SchedulerError::UnknownBranch(BranchId(999)),
        ),
        (BranchId(0), 0, SchedulerError::InvalidWidth(0)),
        (
            BranchId(0),
            u32::MAX,
            SchedulerError::InvalidWidth(u32::MAX),
        ),
        (BranchId(0), 1, SchedulerError::PolicyCommandTreeMismatch),
    ];

    for (branch, width, expected) in cases {
        let mut scheduler = Scheduler::with_components(
            config(100, 100, None),
            Box::new(InvalidForkPolicy { branch, width }),
            Box::new(LogprobScorer),
        )
        .unwrap();

        let result = scheduler.feed_event(EngineEvent::TokenSampled {
            branch: BranchId(0),
            token: 0,
            logprob: -0.1,
        });

        assert_eq!(result, Err(expected));
        assert_eq!(scheduler.tree().len(), 1);
        assert_eq!(
            scheduler
                .tree()
                .get(BranchId(0))
                .unwrap()
                .tokens_generated(),
            0
        );
        assert_eq!(scheduler.budget().total_consumed(), 0);
        assert!(scheduler.poll_commands().is_empty());
    }
}

#[test]
fn external_score_for_the_limit_token_arrives_before_terminalization() {
    let mut scheduler = Scheduler::with_external_values(SchedulerConfig {
        policy: PolicyConfig::Beam(BeamConfig {
            width: 1,
            fork_width: 1,
            fork_at_tokens: Vec::new(),
        }),
        ..config(1, 1, None)
    })
    .unwrap();

    scheduler
        .feed_event(EngineEvent::TokenSampled {
            branch: BranchId(0),
            token: 0,
            logprob: -0.1,
        })
        .unwrap();
    assert!(scheduler.poll_commands().is_empty());
    assert_eq!(
        scheduler.tree().get(BranchId(0)).unwrap().state(),
        BranchState::Active
    );
    assert_eq!(scheduler.budget().total_consumed(), 1);

    scheduler
        .feed_event(EngineEvent::ValueScored {
            branch: BranchId(0),
            score: 0.9,
        })
        .unwrap();
    assert_eq!(
        scheduler.poll_commands(),
        vec![Command::Finalize {
            branch: BranchId(0),
        }]
    );
}

#[test]
fn budget_terminal_branch_is_not_forked_by_beam_before_external_score_arrives() {
    let mut scheduler = Scheduler::with_external_values(SchedulerConfig {
        policy: PolicyConfig::Beam(BeamConfig {
            width: 2,
            fork_width: 2,
            fork_at_tokens: vec![1],
        }),
        ..config(100, 1, None)
    })
    .unwrap();

    scheduler
        .feed_event(EngineEvent::TokenSampled {
            branch: BranchId(0),
            token: 0,
            logprob: -0.1,
        })
        .unwrap();

    assert!(scheduler.poll_commands().is_empty());
    assert_eq!(scheduler.tree().len(), 1);
    scheduler
        .feed_event(EngineEvent::ValueScored {
            branch: BranchId(0),
            score: 0.9,
        })
        .expect("the pending terminal must remain a leaf until its score arrives");
    assert_eq!(
        scheduler.poll_commands(),
        vec![Command::Finalize {
            branch: BranchId(0),
        }]
    );
}

#[test]
fn budget_terminal_branch_is_not_forked_by_mcts_from_a_sibling_score() {
    let mut scheduler = Scheduler::with_external_values(SchedulerConfig {
        policy: PolicyConfig::Mcts(MctsConfig {
            expansion_width: 2,
            max_depth: 2,
            exploration_weight: 1.0,
        }),
        ..config(100, 2, None)
    })
    .unwrap();

    scheduler
        .feed_event(EngineEvent::TokenSampled {
            branch: BranchId(0),
            token: 0,
            logprob: 0.0,
        })
        .unwrap();
    scheduler
        .feed_event(EngineEvent::ValueScored {
            branch: BranchId(0),
            score: 0.0,
        })
        .unwrap();
    let _ = scheduler.poll_commands();

    for branch in [BranchId(1), BranchId(2)] {
        scheduler
            .feed_event(EngineEvent::TokenSampled {
                branch,
                token: u32::try_from(branch.0).unwrap(),
                logprob: -0.1,
            })
            .unwrap();
    }

    scheduler
        .feed_event(EngineEvent::ValueScored {
            branch: BranchId(2),
            score: 0.5,
        })
        .unwrap();
    assert_eq!(
        scheduler.tree().get(BranchId(1)).unwrap().state(),
        BranchState::Active
    );
    assert!(
        scheduler
            .tree()
            .get(BranchId(1))
            .unwrap()
            .children()
            .is_empty()
    );

    scheduler
        .feed_event(EngineEvent::ValueScored {
            branch: BranchId(1),
            score: 0.75,
        })
        .expect("the pending terminal must remain a leaf until its score arrives");
}

#[test]
fn tree_termination_prefers_scored_branches_over_unscored_logprob_proxies() {
    let mut scheduler = Scheduler::with_external_values(SchedulerConfig {
        policy: PolicyConfig::Beam(BeamConfig {
            width: 2,
            fork_width: 2,
            fork_at_tokens: vec![1],
        }),
        ..config(100, 100, None)
    })
    .unwrap();

    scheduler
        .feed_event(EngineEvent::TokenSampled {
            branch: BranchId(0),
            token: 0,
            logprob: 0.0,
        })
        .unwrap();
    let _ = scheduler.poll_commands();
    scheduler
        .feed_event(EngineEvent::ValueScored {
            branch: BranchId(0),
            score: 0.0,
        })
        .unwrap();
    let _ = scheduler.poll_commands();

    scheduler
        .feed_event(EngineEvent::TokenSampled {
            branch: BranchId(1),
            token: 1,
            logprob: -0.1,
        })
        .unwrap();
    scheduler
        .feed_event(EngineEvent::TokenSampled {
            branch: BranchId(2),
            token: 2,
            logprob: -10.0,
        })
        .unwrap();
    scheduler
        .feed_event(EngineEvent::ValueScored {
            branch: BranchId(2),
            score: -100.0,
        })
        .unwrap();
    let _ = scheduler.poll_commands();

    scheduler.drain().unwrap();
    let commands = scheduler.poll_commands();
    assert!(commands.contains(&Command::Finalize {
        branch: BranchId(2),
    }));
    assert!(commands.contains(&Command::Kill {
        branch: BranchId(1),
        reason: KillReason::Drained,
    }));
}

#[test]
fn eos_token_finalizes_without_policy_fork_or_continue() {
    let mut scheduler = Scheduler::with_external_values(SchedulerConfig {
        policy: PolicyConfig::Beam(BeamConfig {
            width: 2,
            fork_width: 2,
            fork_at_tokens: vec![1],
        }),
        ..config(100, 100, None)
    })
    .unwrap();

    scheduler
        .feed_event(EngineEvent::token_sampled_with_eos(
            BranchId(0),
            42,
            -0.25,
            true,
        ))
        .unwrap();

    assert_eq!(
        scheduler.poll_commands(),
        vec![Command::Finalize {
            branch: BranchId(0),
        }]
    );
    assert_eq!(
        scheduler.tree().get(BranchId(0)).unwrap().state(),
        BranchState::Finalized
    );
    assert_eq!(scheduler.budget().total_consumed(), 1);
    assert_eq!(
        scheduler.feed_event(EngineEvent::ValueScored {
            branch: BranchId(0),
            score: 1.0,
        }),
        Err(SchedulerError::BranchNotActive(BranchId(0)))
    );
}

#[test]
fn eos_reclamation_drops_an_ancestor_pending_score_before_timeout_processing() {
    let mut scheduler = Scheduler::with_external_values_and_pending_event_budget(
        SchedulerConfig {
            policy: PolicyConfig::Beam(BeamConfig {
                width: 4,
                fork_width: 2,
                fork_at_tokens: vec![1, 2],
            }),
            ..config(100, 100, None)
        },
        4,
    )
    .unwrap();

    scheduler
        .feed_event(EngineEvent::TokenSampled {
            branch: BranchId(0),
            token: 0,
            logprob: 0.0,
        })
        .unwrap();
    scheduler
        .feed_event(EngineEvent::ValueScored {
            branch: BranchId(0),
            score: 0.0,
        })
        .unwrap();
    let _ = scheduler.poll_commands();

    for branch in [BranchId(1), BranchId(2)] {
        scheduler
            .feed_event(EngineEvent::TokenSampled {
                branch,
                token: u32::try_from(branch.0).unwrap(),
                logprob: -0.1,
            })
            .unwrap();
    }
    scheduler
        .feed_event(EngineEvent::ValueScored {
            branch: BranchId(2),
            score: 0.5,
        })
        .unwrap();
    let _ = scheduler.poll_commands();

    scheduler
        .feed_event(EngineEvent::token_sampled_with_eos(
            BranchId(3),
            3,
            -0.1,
            true,
        ))
        .unwrap();
    scheduler
        .feed_event(EngineEvent::token_sampled_with_eos(
            BranchId(4),
            4,
            -0.1,
            true,
        ))
        .expect("EOS reclamation must clear the dead ancestor's pending score");

    assert_eq!(
        scheduler.tree().get(BranchId(1)).unwrap().state(),
        BranchState::Killed
    );
    scheduler
        .feed_event(EngineEvent::TokenSampled {
            branch: BranchId(5),
            token: 5,
            logprob: -0.1,
        })
        .expect("a stale ancestor score must not wedge a live sibling subtree");
}

#[test]
fn total_branch_cap_rejects_a_fork_before_growing_the_arena() {
    let mut scheduler = Scheduler::new(SchedulerConfig {
        policy: PolicyConfig::Beam(BeamConfig {
            width: 4,
            fork_width: 4,
            fork_at_tokens: vec![1, 2],
        }),
        max_total_branches: 8,
        ..config(100, 100, None)
    })
    .unwrap();

    scheduler
        .feed_event(EngineEvent::TokenSampled {
            branch: BranchId(0),
            token: 0,
            logprob: 0.0,
        })
        .unwrap();
    let _ = scheduler.poll_commands();

    for branch in [BranchId(1), BranchId(2), BranchId(3)] {
        scheduler
            .feed_event(EngineEvent::TokenSampled {
                branch,
                token: u32::try_from(branch.0).unwrap(),
                logprob: 0.0,
            })
            .unwrap();
    }
    let error = scheduler
        .feed_event(EngineEvent::TokenSampled {
            branch: BranchId(4),
            token: 4,
            logprob: 0.0,
        })
        .unwrap_err();

    assert_eq!(
        error,
        SchedulerError::BranchLimitExceeded {
            limit: 8,
            requested_total: 9,
        }
    );
    assert_eq!(scheduler.tree().len(), 5);
}

#[test]
fn pending_external_score_uses_logprob_proxy_after_event_budget() {
    let mut scheduler = Scheduler::with_external_values_and_pending_event_budget(
        SchedulerConfig {
            policy: PolicyConfig::Beam(BeamConfig {
                width: 2,
                fork_width: 2,
                fork_at_tokens: vec![1],
            }),
            ..config(100, 100, None)
        },
        2,
    )
    .unwrap();

    scheduler
        .feed_event(EngineEvent::TokenSampled {
            branch: BranchId(0),
            token: 0,
            logprob: 0.0,
        })
        .unwrap();
    let _ = scheduler.poll_commands();
    scheduler
        .feed_event(EngineEvent::ValueScored {
            branch: BranchId(0),
            score: 0.0,
        })
        .unwrap();
    let _ = scheduler.poll_commands();

    scheduler
        .feed_event(EngineEvent::TokenSampled {
            branch: BranchId(1),
            token: 1,
            logprob: -0.25,
        })
        .unwrap();
    scheduler
        .feed_event(EngineEvent::TokenSampled {
            branch: BranchId(2),
            token: 2,
            logprob: -0.5,
        })
        .unwrap();
    scheduler
        .feed_event(EngineEvent::ValueScored {
            branch: BranchId(2),
            score: 0.75,
        })
        .unwrap();

    let timed_out = scheduler.tree().get(BranchId(1)).unwrap();
    assert!(timed_out.has_value());
    assert_eq!(timed_out.value_estimate(), timed_out.cumulative_logprob());
    scheduler
        .feed_event(EngineEvent::TokenSampled {
            branch: BranchId(1),
            token: 3,
            logprob: -0.25,
        })
        .expect("the deterministic fallback must unblock the branch");
}
