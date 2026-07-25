use crate::{BranchId, KillReason};

/// Input supplied by the decoding engine or value-head integration.
#[derive(Clone, Debug, PartialEq)]
pub enum EngineEvent {
    TokenSampled {
        branch: BranchId,
        token: u32,
        logprob: f64,
    },
    /// Source-compatible EOS extension for `TokenSampled`.
    ///
    /// Existing Rust callers can keep constructing `TokenSampled`; new callers should use
    /// [`EngineEvent::token_sampled_with_eos`], which maps `eos: false` to the original variant.
    TokenSampledWithEos {
        branch: BranchId,
        token: u32,
        logprob: f64,
        eos: bool,
    },
    /// Source-compatible metadata extension for `TokenSampled`.
    ///
    /// `entropy` is predictive next-token entropy in nats. `None` is normalized back to the
    /// legacy token variants so callers without the signal retain identical behavior.
    TokenSampledWithMetadata {
        branch: BranchId,
        token: u32,
        logprob: f64,
        eos: bool,
        entropy: Option<f64>,
    },
    BranchExhausted {
        branch: BranchId,
    },
    ValueScored {
        branch: BranchId,
        score: f64,
    },
}

impl EngineEvent {
    #[must_use]
    pub const fn token_sampled_with_eos(
        branch: BranchId,
        token: u32,
        logprob: f64,
        eos: bool,
    ) -> Self {
        if eos {
            Self::TokenSampledWithEos {
                branch,
                token,
                logprob,
                eos,
            }
        } else {
            Self::TokenSampled {
                branch,
                token,
                logprob,
            }
        }
    }

    #[must_use]
    pub const fn token_sampled_with_metadata(
        branch: BranchId,
        token: u32,
        logprob: f64,
        eos: bool,
        entropy: Option<f64>,
    ) -> Self {
        if entropy.is_some() {
            Self::TokenSampledWithMetadata {
                branch,
                token,
                logprob,
                eos,
                entropy,
            }
        } else {
            Self::token_sampled_with_eos(branch, token, logprob, eos)
        }
    }

    #[must_use]
    pub const fn branch(&self) -> BranchId {
        match self {
            Self::TokenSampled { branch, .. }
            | Self::TokenSampledWithEos { branch, .. }
            | Self::TokenSampledWithMetadata { branch, .. }
            | Self::BranchExhausted { branch }
            | Self::ValueScored { branch, .. } => *branch,
        }
    }

    #[must_use]
    pub const fn is_token_sampled(&self) -> bool {
        matches!(
            self,
            Self::TokenSampled { .. }
                | Self::TokenSampledWithEos { .. }
                | Self::TokenSampledWithMetadata { .. }
        )
    }

    #[must_use]
    pub const fn is_eos(&self) -> bool {
        matches!(
            self,
            Self::TokenSampledWithEos { eos: true, .. }
                | Self::TokenSampledWithMetadata { eos: true, .. }
        )
    }

    #[must_use]
    pub const fn entropy(&self) -> Option<f64> {
        match self {
            Self::TokenSampledWithMetadata { entropy, .. } => *entropy,
            Self::TokenSampled { .. }
            | Self::TokenSampledWithEos { .. }
            | Self::BranchExhausted { .. }
            | Self::ValueScored { .. } => None,
        }
    }
}

/// Output consumed by the Tree-KV engine and serving layer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Command {
    /// Fork `width` children. Children receive the next `width` globally monotonic
    /// `BranchId`s in ascending order; consumers mirror that allocation starting at id 1.
    ForkAt {
        branch: BranchId,
        width: u32,
    },
    Kill {
        branch: BranchId,
        reason: KillReason,
    },
    Continue {
        branch: BranchId,
    },
    Finalize {
        branch: BranchId,
    },
}

/// Stable binary representation used for byte-for-byte determinism checks.
#[must_use]
pub fn encode_command_stream(commands: &[Command]) -> Vec<u8> {
    let mut encoded = Vec::with_capacity(commands.len().saturating_mul(14));
    encoded.extend_from_slice(
        &u64::try_from(commands.len())
            .unwrap_or(u64::MAX)
            .to_le_bytes(),
    );
    for command in commands {
        match command {
            Command::ForkAt { branch, width } => {
                encoded.push(0);
                encoded.extend_from_slice(&branch.0.to_le_bytes());
                encoded.extend_from_slice(&width.to_le_bytes());
            }
            Command::Kill { branch, reason } => {
                encoded.push(1);
                encoded.extend_from_slice(&branch.0.to_le_bytes());
                encoded.push(*reason as u8);
            }
            Command::Continue { branch } => {
                encoded.push(2);
                encoded.extend_from_slice(&branch.0.to_le_bytes());
            }
            Command::Finalize { branch } => {
                encoded.push(3);
                encoded.extend_from_slice(&branch.0.to_le_bytes());
            }
        }
    }
    encoded
}
