//! `SpeculativeDecoder` — fast draft model proposes K tokens; slow
//! verifier accepts a prefix and resumes from the first rejection.
//!
//! The draft and verifier are user-supplied closures: the draft
//! emits K candidate tokens given a prefix; the verifier returns
//! the longest accepted prefix length. F8 ships the orchestration
//! actor.

use std::sync::Arc;

use async_trait::async_trait;
use rakka_core::actor::{Actor, Context, Props};
use tokio::sync::oneshot;

use rakka_accel_cuda::error::GpuError;

pub trait DraftFn: Send + Sync + 'static {
    fn draft(&self, prefix: &[u32], k: usize) -> Result<Vec<u32>, GpuError>;
}

impl<F> DraftFn for F
where
    F: Fn(&[u32], usize) -> Result<Vec<u32>, GpuError> + Send + Sync + 'static,
{
    fn draft(&self, prefix: &[u32], k: usize) -> Result<Vec<u32>, GpuError> {
        self(prefix, k)
    }
}

pub trait VerifierFn: Send + Sync + 'static {
    /// Return `(accepted_prefix_len, replacement_token_for_first_rejected)`.
    /// `accepted_prefix_len <= candidates.len()`. If equal to
    /// `candidates.len()`, all draft tokens were accepted; the
    /// caller can keep speculating.
    fn verify(
        &self,
        prefix: &[u32],
        candidates: &[u32],
    ) -> Result<(usize, Option<u32>), GpuError>;
}

impl<F> VerifierFn for F
where
    F: Fn(&[u32], &[u32]) -> Result<(usize, Option<u32>), GpuError> + Send + Sync + 'static,
{
    fn verify(
        &self,
        prefix: &[u32],
        candidates: &[u32],
    ) -> Result<(usize, Option<u32>), GpuError> {
        self(prefix, candidates)
    }
}

pub struct SpeculativeConfig {
    pub draft: Arc<dyn DraftFn>,
    pub verifier: Arc<dyn VerifierFn>,
    pub k: usize,
    pub max_total_tokens: usize,
}

impl Clone for SpeculativeConfig {
    fn clone(&self) -> Self {
        Self {
            draft: self.draft.clone(),
            verifier: self.verifier.clone(),
            k: self.k,
            max_total_tokens: self.max_total_tokens,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct DecodeStats {
    pub iterations: u32,
    pub draft_tokens: u32,
    pub accepted_tokens: u32,
    pub final_len: usize,
}

pub enum SpecMsg {
    Decode {
        prefix: Vec<u32>,
        reply: oneshot::Sender<Result<(Vec<u32>, DecodeStats), GpuError>>,
    },
}

pub struct SpeculativeDecoder {
    cfg: SpeculativeConfig,
}

impl SpeculativeDecoder {
    pub fn props(cfg: SpeculativeConfig) -> Props<Self> {
        Props::create(move || SpeculativeDecoder { cfg: cfg.clone() })
    }
}

#[async_trait]
impl Actor for SpeculativeDecoder {
    type Msg = SpecMsg;

    async fn handle(&mut self, _ctx: &mut Context<Self>, msg: SpecMsg) {
        match msg {
            SpecMsg::Decode { prefix, reply } => {
                let cfg = self.cfg.clone();
                tokio::spawn(async move {
                    let mut tokens = prefix;
                    let mut stats = DecodeStats::default();
                    while tokens.len() < cfg.max_total_tokens {
                        let remaining = cfg.max_total_tokens - tokens.len();
                        let k = cfg.k.min(remaining);
                        if k == 0 {
                            break;
                        }
                        let candidates = match cfg.draft.draft(&tokens, k) {
                            Ok(c) => c,
                            Err(e) => {
                                let _ = reply.send(Err(e));
                                return;
                            }
                        };
                        if candidates.is_empty() {
                            break;
                        }
                        // Truncate candidates to remaining budget so
                        // a draft that overshoots (returned more
                        // than `k`) can't blow past the cap.
                        let cand_len = candidates.len().min(remaining);
                        let candidates: Vec<u32> = candidates.into_iter().take(cand_len).collect();
                        stats.iterations += 1;
                        stats.draft_tokens += candidates.len() as u32;
                        let (accepted, replacement) = match cfg.verifier.verify(&tokens, &candidates) {
                            Ok(p) => p,
                            Err(e) => {
                                let _ = reply.send(Err(e));
                                return;
                            }
                        };
                        let acc = accepted.min(candidates.len());
                        tokens.extend_from_slice(&candidates[..acc]);
                        stats.accepted_tokens += acc as u32;
                        if acc < candidates.len() {
                            if let Some(t) = replacement {
                                if tokens.len() < cfg.max_total_tokens {
                                    tokens.push(t);
                                }
                            } else {
                                break;
                            }
                        }
                        if tokens.len() >= cfg.max_total_tokens {
                            break;
                        }
                    }
                    stats.final_len = tokens.len();
                    let _ = reply.send(Ok((tokens, stats)));
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rakka_config::Config;
    use rakka_core::actor::ActorSystem;
    use std::time::Duration;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn draft_accept_loop_terminates() {
        // Draft proposes K consecutive integers from the last token;
        // verifier accepts all.
        let draft: Arc<dyn DraftFn> = Arc::new(|prefix: &[u32], k: usize| {
            let last = prefix.last().copied().unwrap_or(0);
            Ok((1..=k as u32).map(|i| last + i).collect())
        });
        let verifier: Arc<dyn VerifierFn> =
            Arc::new(|_prefix: &[u32], candidates: &[u32]| Ok((candidates.len(), None)));
        let cfg = SpeculativeConfig {
            draft,
            verifier,
            k: 4,
            max_total_tokens: 16,
        };

        let sys = ActorSystem::create("spec-test", Config::empty()).await.unwrap();
        let dec = sys.actor_of(SpeculativeDecoder::props(cfg), "dec").unwrap();

        let (tx, rx) = oneshot::channel();
        dec.tell(SpecMsg::Decode {
            prefix: vec![0],
            reply: tx,
        });
        let (tokens, stats) = tokio::time::timeout(Duration::from_secs(2), rx).await.unwrap().unwrap().unwrap();
        assert!(tokens.len() <= 16);
        assert!(stats.iterations >= 1);

        sys.terminate().await;
    }
}
