use std::collections::BTreeMap;

use autotree_scheduler::{
    BeamConfig, BranchId, Command, DEFAULT_MAX_TOTAL_BRANCHES, EngineEvent, PolicyConfig,
    Scheduler, SchedulerConfig,
};
use proptest::prelude::*;

fn beam_config(
    beam_width: u32,
    fork_width: u32,
    total_budget: u64,
    branch_budget: u64,
) -> SchedulerConfig {
    SchedulerConfig {
        policy: PolicyConfig::Beam(BeamConfig {
            width: beam_width,
            fork_width,
            fork_at_tokens: vec![1, 2, 3],
        }),
        seed: 17,
        total_token_budget: total_budget,
        per_branch_token_budget: branch_budget,
        max_total_branches: DEFAULT_MAX_TOTAL_BRANCHES,
        speculative_kill_margin: None,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ExternalState {
    Active,
    Expanded,
    Terminal,
}

fn apply_to_external_liveness(
    commands: &[Command],
    states: &mut BTreeMap<BranchId, ExternalState>,
    next_id: &mut u64,
) {
    for command in commands {
        match command {
            Command::ForkAt { branch, width } => {
                assert!(
                    states.get(branch) == Some(&ExternalState::Active),
                    "fork referenced non-active branch {branch:?}"
                );
                states.insert(*branch, ExternalState::Expanded);
                for _ in 0..*width {
                    assert!(
                        states
                            .insert(BranchId(*next_id), ExternalState::Active)
                            .is_none()
                    );
                    *next_id += 1;
                }
            }
            Command::Kill { branch, .. } => {
                let state = states
                    .get_mut(branch)
                    .expect("kill references known branch");
                assert!(
                    *state != ExternalState::Terminal,
                    "kill referenced terminal branch {branch:?}"
                );
                *state = ExternalState::Terminal;
            }
            Command::Finalize { branch } => {
                let state = states
                    .get_mut(branch)
                    .expect("finalize references known branch");
                assert_eq!(*state, ExternalState::Active);
                *state = ExternalState::Terminal;
            }
            Command::Continue { branch } => {
                assert!(
                    states.get(branch) == Some(&ExternalState::Active),
                    "continue referenced non-active branch {branch:?}"
                );
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn commands_never_reference_dead_or_unknown_branches(
        beam_width in 1_u32..5,
        fork_width in 1_u32..5,
        steps in 1_u32..24,
    ) {
        let mut scheduler = Scheduler::new(beam_config(
            beam_width,
            fork_width,
            1_000,
            1_000,
        )).unwrap();
        let mut states = BTreeMap::from([(BranchId(0), ExternalState::Active)]);
        let mut next_id = 1_u64;

        for token in 0..steps {
            let Some(branch) = scheduler.tree().active_frontier().first().copied() else {
                break;
            };
            scheduler.feed_event(EngineEvent::TokenSampled {
                branch,
                token,
                logprob: -f64::from(token + 1) / 100.0,
            }).unwrap();
            let commands = scheduler.poll_commands();
            apply_to_external_liveness(&commands, &mut states, &mut next_id);
        }

        scheduler.drain().unwrap();
        let commands = scheduler.poll_commands();
        apply_to_external_liveness(&commands, &mut states, &mut next_id);
        prop_assert!(states.values().all(|state| *state == ExternalState::Terminal));
    }

    #[test]
    fn hard_budgets_are_never_exceeded(
        total_budget in 1_u64..32,
        branch_budget in 1_u64..16,
        attempts in 1_usize..96,
    ) {
        let mut scheduler = Scheduler::new(beam_config(
            3,
            3,
            total_budget,
            branch_budget,
        )).unwrap();
        for (accepted_before, token) in (0..attempts).enumerate() {
            let frontier = scheduler.tree().active_frontier();
            if frontier.is_empty() {
                break;
            }
            let branch = frontier[token % frontier.len()];
            scheduler.feed_event(EngineEvent::TokenSampled {
                branch,
                token: u32::try_from(token).unwrap(),
                logprob: -0.01,
            }).unwrap();
            let _ = scheduler.poll_commands();

            prop_assert!(scheduler.budget().total_consumed() <= total_budget);
            prop_assert_eq!(
                scheduler.budget().total_consumed(),
                u64::try_from(accepted_before + 1).unwrap()
            );
            for node in scheduler.tree().iter() {
                prop_assert!(node.tokens_generated() <= branch_budget);
            }
        }
    }

    #[test]
    fn every_forked_branch_receives_a_terminal_command_when_drained(
        beam_width in 1_u32..5,
        fork_width in 1_u32..5,
        steps in 1_u32..24,
    ) {
        let mut scheduler = Scheduler::new(beam_config(
            beam_width,
            fork_width,
            1_000,
            1_000,
        )).unwrap();
        let mut terminal_counts = BTreeMap::<BranchId, u32>::new();

        for token in 0..steps {
            let Some(branch) = scheduler.tree().active_frontier().first().copied() else {
                break;
            };
            scheduler.feed_event(EngineEvent::TokenSampled {
                branch,
                token,
                logprob: -0.01,
            }).unwrap();
            for command in scheduler.poll_commands() {
                if let Command::Kill { branch, .. } | Command::Finalize { branch } = command {
                    *terminal_counts.entry(branch).or_default() += 1;
                }
            }
        }

        scheduler.drain().unwrap();
        for command in scheduler.poll_commands() {
            if let Command::Kill { branch, .. } | Command::Finalize { branch } = command {
                *terminal_counts.entry(branch).or_default() += 1;
            }
        }

        prop_assert!(scheduler.tree().len() > 1, "generator must exercise at least one fork");
        for node in scheduler.tree().iter() {
            prop_assert!(node.state().is_terminal());
            prop_assert_eq!(terminal_counts.get(&node.id()), Some(&1));
        }
    }
}
