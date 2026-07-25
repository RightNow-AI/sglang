use autotree_scheduler::{
    AdaptiveForkConfig, BeamConfig, BranchId, Command, EngineEvent, PolicyConfig, Scheduler,
    SchedulerConfig, encode_command_stream,
};

fn scheduler_config(
    total_token_budget: u64,
    beam_width: u32,
    fork_at_tokens: Vec<u64>,
) -> SchedulerConfig {
    SchedulerConfig {
        policy: PolicyConfig::Beam(BeamConfig {
            width: beam_width,
            fork_width: 2,
            fork_at_tokens,
        }),
        seed: 0xA11C_E5EED,
        total_token_budget,
        per_branch_token_budget: total_token_budget,
        speculative_kill_margin: None,
        max_total_branches: autotree_scheduler::DEFAULT_MAX_TOTAL_BRANCHES,
    }
}

fn adaptive_config(min_tokens_between_forks: u64) -> AdaptiveForkConfig {
    AdaptiveForkConfig {
        entropy_threshold_nats: 1.5,
        min_tokens_between_forks,
        max_total_branches: 32,
        max_depth: 8,
        min_fork_width: 2,
        max_fork_width: 4,
        entropy_nats_per_extra_branch: 0.5,
    }
}

fn entropy_event(branch: BranchId, token: u32, entropy: f64) -> EngineEvent {
    EngineEvent::token_sampled_with_metadata(branch, token, -0.1, false, Some(entropy))
}

#[test]
fn entropy_spike_forks_exactly_at_the_spike_instead_of_the_static_position() {
    let mut scheduler = Scheduler::new_with_adaptive_forking(
        scheduler_config(100, 4, vec![99]),
        adaptive_config(2),
    )
    .unwrap();

    scheduler
        .feed_event(entropy_event(BranchId(0), 10, 1.49))
        .unwrap();
    assert_eq!(
        scheduler.poll_commands(),
        vec![Command::Continue {
            branch: BranchId(0)
        }]
    );

    scheduler
        .feed_event(entropy_event(BranchId(0), 11, 2.1))
        .unwrap();
    let commands = scheduler.poll_commands();
    assert_eq!(
        commands.first(),
        Some(&Command::ForkAt {
            branch: BranchId(0),
            width: 3,
        })
    );
}

fn no_entropy_stream(adaptive: bool) -> Vec<Command> {
    let config = scheduler_config(100, 2, vec![2]);
    let mut scheduler = if adaptive {
        Scheduler::new_with_adaptive_forking(config, adaptive_config(2)).unwrap()
    } else {
        Scheduler::new(config).unwrap()
    };
    let mut stream = Vec::new();
    for token in 0..2 {
        scheduler
            .feed_event(EngineEvent::TokenSampled {
                branch: BranchId(0),
                token,
                logprob: -0.1,
            })
            .unwrap();
        stream.extend(scheduler.poll_commands());
    }
    stream
}

#[test]
fn absent_entropy_keeps_the_legacy_golden_stream_byte_identical() {
    let expected = vec![
        Command::Continue {
            branch: BranchId(0),
        },
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
    ];
    let legacy = no_entropy_stream(false);
    let adaptive_without_signal = no_entropy_stream(true);

    assert_eq!(legacy, expected);
    assert_eq!(adaptive_without_signal, expected);
    assert_eq!(
        encode_command_stream(&adaptive_without_signal),
        encode_command_stream(&legacy)
    );
}

#[test]
fn hysteresis_prevents_fork_storms_during_sustained_high_entropy() {
    let mut scheduler = Scheduler::new_with_adaptive_forking(
        scheduler_config(100, 1, Vec::new()),
        adaptive_config(3),
    )
    .unwrap();

    scheduler
        .feed_event(entropy_event(BranchId(0), 0, 2.0))
        .unwrap();
    assert!(matches!(
        scheduler.poll_commands().first(),
        Some(Command::ForkAt {
            branch: BranchId(0),
            ..
        })
    ));

    for token in 1..=2 {
        scheduler
            .feed_event(entropy_event(BranchId(1), token, 2.0))
            .unwrap();
        assert!(
            scheduler
                .poll_commands()
                .iter()
                .all(|command| !matches!(command, Command::ForkAt { .. }))
        );
    }

    scheduler
        .feed_event(entropy_event(BranchId(1), 3, 2.0))
        .unwrap();
    assert!(matches!(
        scheduler.poll_commands().first(),
        Some(Command::ForkAt {
            branch: BranchId(1),
            ..
        })
    ));
}

#[test]
fn budget_starved_adaptive_fork_is_refused() {
    let mut scheduler = Scheduler::new_with_adaptive_forking(
        scheduler_config(2, 2, Vec::new()),
        adaptive_config(1),
    )
    .unwrap();

    scheduler
        .feed_event(entropy_event(BranchId(0), 0, 3.0))
        .unwrap();
    let commands = scheduler.poll_commands();

    assert_eq!(
        commands,
        vec![Command::Continue {
            branch: BranchId(0)
        }]
    );
    assert_eq!(scheduler.tree().len(), 1);
}
