use autotree_scheduler::{
    AdaptiveForkConfig, BeamConfig, BiasedOracleScorer, BranchId, Command,
    DEFAULT_MAX_TOTAL_BRANCHES, EngineEvent, KillReason, MctsConfig, PolicyConfig, Scheduler,
    SchedulerConfig, encode_command_stream,
};

fn scripted_mcts(seed: u64) -> Vec<Command> {
    scripted_mcts_steps(seed, 128)
}

fn scripted_mcts_steps(seed: u64, steps: u32) -> Vec<Command> {
    let scorer = BiasedOracleScorer::new([(BranchId(1), 0.9), (BranchId(2), 0.1)], 0.5).unwrap();
    let mut scheduler = Scheduler::with_scorer(
        SchedulerConfig {
            policy: PolicyConfig::Mcts(MctsConfig {
                expansion_width: 2,
                max_depth: 2,
                exploration_weight: 1.0,
            }),
            seed,
            total_token_budget: 1_000,
            per_branch_token_budget: 1_000,
            max_total_branches: DEFAULT_MAX_TOTAL_BRANCHES,
            speculative_kill_margin: None,
        },
        Box::new(scorer),
    )
    .unwrap();

    scheduler
        .feed_event(EngineEvent::TokenSampled {
            branch: BranchId(0),
            token: 0,
            logprob: -0.01,
        })
        .unwrap();
    let mut batch = scheduler.poll_commands();
    let mut stream = batch.clone();

    for token in 1..=steps {
        let selected = batch
            .iter()
            .rev()
            .find_map(|command| match command {
                Command::Continue { branch } => Some(*branch),
                _ => None,
            })
            .expect("script follows the branch selected by MCTS");
        scheduler
            .feed_event(EngineEvent::TokenSampled {
                branch: selected,
                token,
                logprob: -0.01,
            })
            .unwrap();
        batch = scheduler.poll_commands();
        stream.extend(batch.iter().cloned());
    }
    scheduler.drain().unwrap();
    stream.extend(scheduler.poll_commands());
    stream
}

#[test]
fn fixed_seed_mcts_command_stream_matches_the_portable_golden() {
    let expected = vec![
        Command::ForkAt {
            branch: BranchId(0),
            width: 2,
        },
        Command::Continue {
            branch: BranchId(2),
        },
        Command::ForkAt {
            branch: BranchId(1),
            width: 2,
        },
        Command::Continue {
            branch: BranchId(4),
        },
        Command::Continue {
            branch: BranchId(3),
        },
        Command::Continue {
            branch: BranchId(3),
        },
        Command::ForkAt {
            branch: BranchId(2),
            width: 2,
        },
        Command::Continue {
            branch: BranchId(5),
        },
        Command::Continue {
            branch: BranchId(4),
        },
        Command::Continue {
            branch: BranchId(6),
        },
        Command::Continue {
            branch: BranchId(3),
        },
        Command::Continue {
            branch: BranchId(5),
        },
        Command::Continue {
            branch: BranchId(4),
        },
        Command::Continue {
            branch: BranchId(6),
        },
        Command::Continue {
            branch: BranchId(3),
        },
        Command::Continue {
            branch: BranchId(5),
        },
        Command::Kill {
            branch: BranchId(4),
            reason: KillReason::Drained,
        },
        Command::Kill {
            branch: BranchId(5),
            reason: KillReason::Drained,
        },
        Command::Kill {
            branch: BranchId(6),
            reason: KillReason::Drained,
        },
        Command::Finalize {
            branch: BranchId(3),
        },
        Command::Kill {
            branch: BranchId(2),
            reason: KillReason::Drained,
        },
        Command::Kill {
            branch: BranchId(1),
            reason: KillReason::Drained,
        },
        Command::Kill {
            branch: BranchId(0),
            reason: KillReason::Drained,
        },
    ];
    assert_eq!(scripted_mcts_steps(0x5EED, 12), expected);
}

#[test]
fn same_seed_and_events_produce_byte_identical_command_streams() {
    let first = scripted_mcts(0x5EED);
    let second = scripted_mcts(0x5EED);

    assert_eq!(first, second);
    assert_eq!(
        encode_command_stream(&first),
        encode_command_stream(&second)
    );
}

#[test]
fn command_stream_encoding_has_stable_golden_tags_and_little_endian_fields() {
    let reasons = [
        KillReason::BeamPruned,
        KillReason::SpeculativeKill,
        KillReason::BranchBudgetExhausted,
        KillReason::TreeBudgetExhausted,
        KillReason::Exhausted,
        KillReason::Drained,
        KillReason::AncestorReclaimed,
    ];
    let mut commands = vec![Command::ForkAt {
        branch: BranchId(0x0102),
        width: 0x0304_0506,
    }];
    commands.extend(
        reasons
            .into_iter()
            .enumerate()
            .map(|(index, reason)| Command::Kill {
                branch: BranchId(0x10 + u64::try_from(index).unwrap()),
                reason,
            }),
    );
    commands.push(Command::Continue {
        branch: BranchId(0x20),
    });
    commands.push(Command::Finalize {
        branch: BranchId(0x21),
    });

    let expected = vec![
        0x0a, 0, 0, 0, 0, 0, 0, 0, // command count
        0x00, 0x02, 0x01, 0, 0, 0, 0, 0, 0, 0x06, 0x05, 0x04, 0x03, // fork
        0x01, 0x10, 0, 0, 0, 0, 0, 0, 0, 0x00, // beam-pruned kill
        0x01, 0x11, 0, 0, 0, 0, 0, 0, 0, 0x01, // speculative kill
        0x01, 0x12, 0, 0, 0, 0, 0, 0, 0, 0x02, // branch budget kill
        0x01, 0x13, 0, 0, 0, 0, 0, 0, 0, 0x03, // tree budget kill
        0x01, 0x14, 0, 0, 0, 0, 0, 0, 0, 0x04, // exhausted kill
        0x01, 0x15, 0, 0, 0, 0, 0, 0, 0, 0x05, // drained kill
        0x01, 0x16, 0, 0, 0, 0, 0, 0, 0, 0x06, // ancestor kill
        0x02, 0x20, 0, 0, 0, 0, 0, 0, 0, // continue
        0x03, 0x21, 0, 0, 0, 0, 0, 0, 0, // finalize
    ];
    assert_eq!(encode_command_stream(&commands), expected);
    assert_ne!(
        encode_command_stream(&[Command::Continue {
            branch: BranchId(1)
        }]),
        encode_command_stream(&[Command::Finalize {
            branch: BranchId(1)
        }])
    );
}

#[test]
fn entropy_bearing_script_matches_the_adaptive_golden_stream() {
    let mut scheduler = Scheduler::new_with_adaptive_forking(
        SchedulerConfig {
            policy: PolicyConfig::Beam(BeamConfig {
                width: 1,
                fork_width: 2,
                fork_at_tokens: Vec::new(),
            }),
            seed: 0x0E17_E0F1,
            total_token_budget: 100,
            per_branch_token_budget: 100,
            speculative_kill_margin: None,
            max_total_branches: DEFAULT_MAX_TOTAL_BRANCHES,
        },
        AdaptiveForkConfig {
            entropy_threshold_nats: 1.0,
            min_tokens_between_forks: 2,
            max_total_branches: 16,
            max_depth: 4,
            min_fork_width: 2,
            max_fork_width: 2,
            entropy_nats_per_extra_branch: 1.0,
        },
    )
    .unwrap();

    let scripted = [
        (BranchId(0), 0, 0.5),
        (BranchId(0), 1, 1.2),
        (BranchId(1), 2, 2.0),
        (BranchId(1), 3, 2.0),
    ];
    let mut stream = Vec::new();
    for (branch, token, entropy) in scripted {
        scheduler
            .feed_event(EngineEvent::token_sampled_with_metadata(
                branch,
                token,
                -0.1,
                false,
                Some(entropy),
            ))
            .unwrap();
        stream.extend(scheduler.poll_commands());
    }

    let expected = vec![
        Command::Continue {
            branch: BranchId(0),
        },
        Command::ForkAt {
            branch: BranchId(0),
            width: 2,
        },
        Command::Kill {
            branch: BranchId(2),
            reason: KillReason::BeamPruned,
        },
        Command::Continue {
            branch: BranchId(1),
        },
        Command::Continue {
            branch: BranchId(1),
        },
        Command::ForkAt {
            branch: BranchId(1),
            width: 2,
        },
        Command::Kill {
            branch: BranchId(4),
            reason: KillReason::BeamPruned,
        },
        Command::Continue {
            branch: BranchId(3),
        },
    ];
    assert_eq!(stream, expected);
    assert_eq!(
        encode_command_stream(&stream),
        encode_command_stream(&expected)
    );
}
