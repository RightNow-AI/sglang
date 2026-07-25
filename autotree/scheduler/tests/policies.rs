use autotree_scheduler::{
    BeamConfig, BestFirstConfig, BiasedOracleScorer, BranchId, BranchState, Command,
    DEFAULT_MAX_TOTAL_BRANCHES, EngineEvent, KillReason, MctsConfig, PolicyConfig, Scheduler,
    SchedulerConfig,
};

fn scheduler_config(policy: PolicyConfig) -> SchedulerConfig {
    SchedulerConfig {
        policy,
        seed: 0xA11CE,
        total_token_budget: 10_000,
        per_branch_token_budget: 10_000,
        max_total_branches: DEFAULT_MAX_TOTAL_BRANCHES,
        speculative_kill_margin: None,
    }
}

fn run_mcts_oracle(first_score: f64, second_score: f64, seed: u64) -> (u64, u64) {
    let scorer = BiasedOracleScorer::new(
        [(BranchId(1), first_score), (BranchId(2), second_score)],
        0.5,
    )
    .unwrap();
    let mut config = scheduler_config(PolicyConfig::Mcts(MctsConfig {
        expansion_width: 2,
        max_depth: 1,
        exploration_weight: 0.7,
    }));
    config.seed = seed;
    let mut scheduler = Scheduler::with_scorer(config, Box::new(scorer)).unwrap();

    scheduler
        .feed_event(EngineEvent::TokenSampled {
            branch: BranchId(0),
            token: 0,
            logprob: 0.0,
        })
        .unwrap();
    let mut commands = scheduler.poll_commands();
    for token in 1..=256 {
        let selected = commands
            .iter()
            .rev()
            .find_map(|command| match command {
                Command::Continue { branch } => Some(*branch),
                _ => None,
            })
            .expect("MCTS always selects a live rollout branch");
        scheduler
            .feed_event(EngineEvent::TokenSampled {
                branch: selected,
                token,
                logprob: 0.0,
            })
            .unwrap();
        commands = scheduler.poll_commands();
    }
    (
        scheduler.tree().get(BranchId(1)).unwrap().visits(),
        scheduler.tree().get(BranchId(2)).unwrap().visits(),
    )
}

#[test]
fn beam_prunes_the_k_plus_one_candidate_and_keeps_exactly_top_k() {
    let mut scheduler = Scheduler::new(scheduler_config(PolicyConfig::Beam(BeamConfig {
        width: 2,
        fork_width: 3,
        fork_at_tokens: vec![1],
    })))
    .unwrap();

    scheduler
        .feed_event(EngineEvent::TokenSampled {
            branch: BranchId(0),
            token: 42,
            logprob: -0.25,
        })
        .unwrap();

    let commands = scheduler.poll_commands();
    assert!(commands.contains(&Command::ForkAt {
        branch: BranchId(0),
        width: 3,
    }));
    assert!(commands.contains(&Command::Kill {
        branch: BranchId(3),
        reason: KillReason::BeamPruned,
    }));
    assert_eq!(
        scheduler.tree().active_frontier(),
        vec![BranchId(1), BranchId(2)]
    );
    assert_eq!(
        scheduler.tree().get(BranchId(3)).unwrap().state(),
        BranchState::Killed
    );
}

#[test]
fn beam_pruning_reclaims_an_expanded_parent_after_its_last_child_dies() {
    let mut scheduler = Scheduler::new(scheduler_config(PolicyConfig::Beam(BeamConfig {
        width: 2,
        fork_width: 2,
        fork_at_tokens: vec![1, 2],
    })))
    .unwrap();

    for (branch, token, logprob) in [
        (BranchId(0), 0, 0.0),
        (BranchId(1), 1, -10.0),
        (BranchId(2), 2, 0.0),
    ] {
        scheduler
            .feed_event(EngineEvent::TokenSampled {
                branch,
                token,
                logprob,
            })
            .unwrap();
    }

    let commands = scheduler.poll_commands();
    assert!(commands.contains(&Command::Kill {
        branch: BranchId(1),
        reason: KillReason::AncestorReclaimed,
    }));
    assert_eq!(
        scheduler.tree().get(BranchId(1)).unwrap().state(),
        BranchState::Killed
    );
}

#[test]
fn beam_waits_for_equal_token_depth_then_keeps_highest_logprob_candidates() {
    let mut scheduler = Scheduler::new(scheduler_config(PolicyConfig::Beam(BeamConfig {
        width: 3,
        fork_width: 3,
        fork_at_tokens: vec![1, 2],
    })))
    .unwrap();

    scheduler
        .feed_event(EngineEvent::TokenSampled {
            branch: BranchId(0),
            token: 0,
            logprob: 0.0,
        })
        .unwrap();
    let _ = scheduler.poll_commands();

    for (branch, token, logprob) in [(BranchId(1), 1, -0.1), (BranchId(2), 2, -1.0)] {
        scheduler
            .feed_event(EngineEvent::TokenSampled {
                branch,
                token,
                logprob,
            })
            .unwrap();
        assert!(
            !scheduler
                .poll_commands()
                .iter()
                .any(|command| matches!(command, Command::ForkAt { .. }))
        );
    }

    scheduler
        .feed_event(EngineEvent::TokenSampled {
            branch: BranchId(3),
            token: 3,
            logprob: -0.5,
        })
        .unwrap();
    let commands = scheduler.poll_commands();
    for branch in [BranchId(1), BranchId(2), BranchId(3)] {
        assert!(commands.contains(&Command::ForkAt { branch, width: 3 }));
    }
    assert_eq!(
        scheduler.tree().active_frontier(),
        vec![BranchId(4), BranchId(5), BranchId(6)]
    );
}

#[test]
fn best_first_expands_the_highest_value_frontier_branch() {
    let mut scheduler = Scheduler::with_external_values(scheduler_config(PolicyConfig::BestFirst(
        BestFirstConfig {
            expansion_width: 2,
            max_depth: 2,
        },
    )))
    .unwrap();

    scheduler
        .feed_event(EngineEvent::TokenSampled {
            branch: BranchId(0),
            token: 1,
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
            token: 2,
            logprob: -0.1,
        })
        .unwrap();
    scheduler
        .feed_event(EngineEvent::ValueScored {
            branch: BranchId(1),
            score: 0.1,
        })
        .unwrap();
    assert!(scheduler.poll_commands().contains(&Command::ForkAt {
        branch: BranchId(1),
        width: 2,
    }));
    assert_eq!(scheduler.tree().len(), 5);

    scheduler
        .feed_event(EngineEvent::TokenSampled {
            branch: BranchId(2),
            token: 3,
            logprob: -0.1,
        })
        .unwrap();
    scheduler
        .feed_event(EngineEvent::ValueScored {
            branch: BranchId(2),
            score: 0.9,
        })
        .unwrap();

    let commands = scheduler.poll_commands();
    assert!(commands.contains(&Command::ForkAt {
        branch: BranchId(2),
        width: 2,
    }));
    assert!(!commands.iter().any(|command| matches!(
        command,
        Command::ForkAt {
            branch: BranchId(1),
            ..
        }
    )));
}

#[test]
fn late_value_score_updates_a_still_live_expanded_branch() {
    let mut scheduler =
        Scheduler::with_external_values(scheduler_config(PolicyConfig::Beam(BeamConfig {
            width: 2,
            fork_width: 2,
            fork_at_tokens: vec![1],
        })))
        .unwrap();

    scheduler
        .feed_event(EngineEvent::TokenSampled {
            branch: BranchId(0),
            token: 1,
            logprob: -0.1,
        })
        .unwrap();
    assert_eq!(
        scheduler.tree().get(BranchId(0)).unwrap().state(),
        BranchState::Expanded
    );

    scheduler
        .feed_event(EngineEvent::ValueScored {
            branch: BranchId(0),
            score: 0.8,
        })
        .expect("a live expanded branch can receive its late value score");
    assert_eq!(
        scheduler.tree().get(BranchId(0)).unwrap().value_estimate(),
        0.8
    );
}

#[test]
fn external_values_gate_best_first_expansion_until_the_score_arrives() {
    let mut scheduler = Scheduler::with_external_values(scheduler_config(PolicyConfig::BestFirst(
        BestFirstConfig {
            expansion_width: 2,
            max_depth: 2,
        },
    )))
    .unwrap();

    scheduler
        .feed_event(EngineEvent::TokenSampled {
            branch: BranchId(0),
            token: 1,
            logprob: -0.1,
        })
        .unwrap();
    assert!(scheduler.poll_commands().is_empty());
    assert_eq!(scheduler.tree().len(), 1);
    assert_eq!(scheduler.tree().get(BranchId(0)).unwrap().visits(), 0);

    scheduler
        .feed_event(EngineEvent::ValueScored {
            branch: BranchId(0),
            score: 0.8,
        })
        .unwrap();
    assert_eq!(
        scheduler.poll_commands(),
        vec![
            Command::ForkAt {
                branch: BranchId(0),
                width: 2,
            },
            Command::Continue {
                branch: BranchId(1),
            },
            Command::Continue {
                branch: BranchId(2),
            },
        ]
    );
    assert_eq!(scheduler.budget().total_consumed(), 1);
}

#[test]
fn best_first_skips_unscored_frontier_peers_after_a_score_arrives() {
    let mut scheduler = Scheduler::with_external_values(scheduler_config(PolicyConfig::BestFirst(
        BestFirstConfig {
            expansion_width: 2,
            max_depth: 2,
        },
    )))
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

    scheduler
        .feed_event(EngineEvent::TokenSampled {
            branch: BranchId(1),
            token: 1,
            logprob: -0.1,
        })
        .unwrap();
    scheduler
        .feed_event(EngineEvent::ValueScored {
            branch: BranchId(1),
            score: 0.8,
        })
        .unwrap();

    assert_eq!(
        scheduler.poll_commands(),
        vec![
            Command::ForkAt {
                branch: BranchId(1),
                width: 2,
            },
            Command::Continue {
                branch: BranchId(3),
            },
            Command::Continue {
                branch: BranchId(4),
            },
        ]
    );
}

#[test]
fn best_first_respects_the_exact_max_depth_boundary() {
    let mut scheduler = Scheduler::with_external_values(scheduler_config(PolicyConfig::BestFirst(
        BestFirstConfig {
            expansion_width: 2,
            max_depth: 1,
        },
    )))
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

    for (branch, score) in [(BranchId(1), 0.9), (BranchId(2), 0.1)] {
        scheduler
            .feed_event(EngineEvent::TokenSampled {
                branch,
                token: u32::try_from(branch.0).unwrap(),
                logprob: 0.0,
            })
            .unwrap();
        scheduler
            .feed_event(EngineEvent::ValueScored { branch, score })
            .unwrap();
    }

    assert_eq!(
        scheduler.poll_commands(),
        vec![Command::Continue {
            branch: BranchId(1),
        }]
    );
    assert_eq!(scheduler.tree().len(), 3);
}

#[test]
fn best_first_normalizes_signed_zero_before_logprob_tie_breaking() {
    let mut scheduler = Scheduler::with_external_values(scheduler_config(PolicyConfig::BestFirst(
        BestFirstConfig {
            expansion_width: 2,
            max_depth: 1,
        },
    )))
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

    for (branch, logprob, score) in [(BranchId(1), 0.0, -0.0), (BranchId(2), -1.0, 0.0)] {
        scheduler
            .feed_event(EngineEvent::TokenSampled {
                branch,
                token: u32::try_from(branch.0).unwrap(),
                logprob,
            })
            .unwrap();
        scheduler
            .feed_event(EngineEvent::ValueScored { branch, score })
            .unwrap();
    }

    assert_eq!(
        scheduler.poll_commands(),
        vec![Command::Continue {
            branch: BranchId(1),
        }]
    );
}

#[test]
fn mcts_visit_counts_concentrate_on_the_biased_oracle_subtree() {
    let (better_visits, worse_visits) = run_mcts_oracle(1.0, 0.0, 0xA11CE);
    assert!(worse_visits > 0, "UCT must explore the worse subtree");
    assert!(
        better_visits > worse_visits * 5,
        "biased oracle should concentrate visits: better={better_visits}, worse={worse_visits}"
    );
}

#[test]
fn mcts_concentrates_on_the_better_subtree_when_it_has_the_higher_id() {
    let (worse_visits, better_visits) = run_mcts_oracle(0.0, 1.0, 0xA11CE);
    assert!(worse_visits > 0, "UCT must explore both subtrees");
    assert!(better_visits > worse_visits * 5);
}

#[test]
fn mcts_seed_controls_symmetric_unvisited_tie_breaks() {
    let mut selected = std::collections::BTreeSet::new();
    for seed in 0..32 {
        let scorer = BiasedOracleScorer::new([], 0.5).unwrap();
        let mut config = scheduler_config(PolicyConfig::Mcts(MctsConfig {
            expansion_width: 2,
            max_depth: 1,
            exploration_weight: 1.0,
        }));
        config.seed = seed;
        let mut scheduler = Scheduler::with_scorer(config, Box::new(scorer)).unwrap();
        scheduler
            .feed_event(EngineEvent::TokenSampled {
                branch: BranchId(0),
                token: 0,
                logprob: 0.0,
            })
            .unwrap();
        selected.insert(
            scheduler
                .poll_commands()
                .into_iter()
                .find_map(|command| match command {
                    Command::Continue { branch } => Some(branch),
                    _ => None,
                })
                .unwrap(),
        );
    }
    assert_eq!(
        selected,
        std::collections::BTreeSet::from([BranchId(1), BranchId(2)])
    );
}

#[test]
fn external_mcts_backpropagates_once_when_the_pending_score_arrives() {
    let mut scheduler =
        Scheduler::with_external_values(scheduler_config(PolicyConfig::Mcts(MctsConfig {
            expansion_width: 2,
            max_depth: 1,
            exploration_weight: 1.0,
        })))
        .unwrap();
    scheduler
        .feed_event(EngineEvent::TokenSampled {
            branch: BranchId(0),
            token: 0,
            logprob: 0.0,
        })
        .unwrap();
    assert!(scheduler.poll_commands().is_empty());
    assert_eq!(scheduler.tree().get(BranchId(0)).unwrap().visits(), 0);

    scheduler
        .feed_event(EngineEvent::ValueScored {
            branch: BranchId(0),
            score: 0.75,
        })
        .unwrap();
    assert_eq!(scheduler.tree().get(BranchId(0)).unwrap().visits(), 1);
    assert!(
        scheduler
            .feed_event(EngineEvent::ValueScored {
                branch: BranchId(0),
                score: 0.75,
            })
            .is_err()
    );
    assert_eq!(scheduler.tree().get(BranchId(0)).unwrap().visits(), 1);
}
