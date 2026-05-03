//! `LangGraphGpuNodes` — composable agent graph nodes.
//!
//! Each node implements [`GraphNode<S>`] mapping an input state `S`
//! to an output state. Nodes wired together in a [`NodeGraph<S>`]
//! run in topological order. Designed to mirror LangGraph-style
//! agent flows but with rakka actors and (eventually) GPU-resident
//! intermediate state.
//!
//! F7 ships the synchronous host-side variant: nodes are
//! `Fn(S) -> Result<S>` closures or trait impls. F8+ swaps the state
//! storage to `GpuRef<...>`/`ManagedRef<...>` and adds parallel
//! execution of independent branches.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use async_trait::async_trait;
use rakka_core::actor::{Actor, Context, Props};
use tokio::sync::oneshot;

use rakka_accel_cuda::error::GpuError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub u32);

/// Node-side computation. Receives state, returns updated state.
pub trait GraphNode<S>: Send + Sync + 'static {
    fn run(&self, state: S) -> Result<S, GpuError>;
}

impl<S, F> GraphNode<S> for F
where
    F: Fn(S) -> Result<S, GpuError> + Send + Sync + 'static,
{
    fn run(&self, state: S) -> Result<S, GpuError> {
        self(state)
    }
}

pub struct NodeEntry<S> {
    pub id: NodeId,
    pub node: Arc<dyn GraphNode<S>>,
}

pub struct NodeGraph<S> {
    nodes: HashMap<NodeId, Arc<dyn GraphNode<S>>>,
    /// Edges represented as adjacency: predecessor → successors.
    /// Topological execution requires no cycles.
    edges: HashMap<NodeId, Vec<NodeId>>,
    in_degree: HashMap<NodeId, usize>,
    /// Entry node — the first to run.
    entry: Option<NodeId>,
}

impl<S> Default for NodeGraph<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S> NodeGraph<S> {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: HashMap::new(),
            in_degree: HashMap::new(),
            entry: None,
        }
    }

    pub fn add_node<N: GraphNode<S>>(&mut self, id: NodeId, node: N) {
        self.nodes.insert(id, Arc::new(node));
        self.in_degree.entry(id).or_insert(0);
    }

    pub fn add_edge(&mut self, from: NodeId, to: NodeId) {
        self.edges.entry(from).or_default().push(to);
        *self.in_degree.entry(to).or_insert(0) += 1;
    }

    pub fn set_entry(&mut self, id: NodeId) {
        self.entry = Some(id);
    }

    /// Topological order via Kahn's algorithm. Returns
    /// `Err(Unrecoverable("cycle"))` on cycles.
    fn topo_order(&self) -> Result<Vec<NodeId>, GpuError> {
        let mut in_deg = self.in_degree.clone();
        let mut q: VecDeque<NodeId> = in_deg
            .iter()
            .filter(|(_, &d)| d == 0)
            .map(|(id, _)| *id)
            .collect();
        let mut order = Vec::with_capacity(self.nodes.len());
        let mut seen = HashSet::new();
        while let Some(n) = q.pop_front() {
            if !seen.insert(n) {
                continue;
            }
            order.push(n);
            if let Some(succs) = self.edges.get(&n) {
                for &s in succs {
                    let d = in_deg.entry(s).or_insert(0);
                    *d = d.saturating_sub(1);
                    if *d == 0 {
                        q.push_back(s);
                    }
                }
            }
        }
        if order.len() != self.nodes.len() {
            return Err(GpuError::Unrecoverable("NodeGraph: cycle detected".into()));
        }
        Ok(order)
    }
}

pub enum NodeGraphMsg<S: Send + 'static> {
    Run {
        state: S,
        reply: oneshot::Sender<Result<S, GpuError>>,
    },
}

pub struct LangGraphGpuActor<S: Send + 'static> {
    graph: Arc<NodeGraph<S>>,
}

impl<S: Send + 'static> LangGraphGpuActor<S> {
    pub fn props(graph: NodeGraph<S>) -> Props<Self> {
        let g = Arc::new(graph);
        Props::create(move || LangGraphGpuActor { graph: g.clone() })
    }
}

#[async_trait]
impl<S: Send + 'static> Actor for LangGraphGpuActor<S> {
    type Msg = NodeGraphMsg<S>;

    async fn handle(&mut self, _ctx: &mut Context<Self>, msg: NodeGraphMsg<S>) {
        match msg {
            NodeGraphMsg::Run { state, reply } => {
                let order = match self.graph.topo_order() {
                    Ok(o) => o,
                    Err(e) => {
                        let _ = reply.send(Err(e));
                        return;
                    }
                };
                let mut s = state;
                for id in order {
                    let Some(node) = self.graph.nodes.get(&id) else {
                        let _ = reply.send(Err(GpuError::Unrecoverable(format!(
                            "NodeGraph: node {id:?} missing"
                        ))));
                        return;
                    };
                    match node.run(s) {
                        Ok(next) => s = next,
                        Err(e) => {
                            let _ = reply.send(Err(e));
                            return;
                        }
                    }
                }
                let _ = reply.send(Ok(s));
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

    #[derive(Clone, Debug, PartialEq)]
    struct State {
        x: i32,
        log: Vec<&'static str>,
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn topo_run_executes_in_order() {
        let mut g = NodeGraph::<State>::new();
        g.add_node(NodeId(1), |mut s: State| {
            s.x += 1;
            s.log.push("a");
            Ok(s)
        });
        g.add_node(NodeId(2), |mut s: State| {
            s.x *= 10;
            s.log.push("b");
            Ok(s)
        });
        g.add_node(NodeId(3), |mut s: State| {
            s.x -= 5;
            s.log.push("c");
            Ok(s)
        });
        g.add_edge(NodeId(1), NodeId(2));
        g.add_edge(NodeId(2), NodeId(3));
        g.set_entry(NodeId(1));

        let sys = ActorSystem::create("nodegraph-test", Config::empty()).await.unwrap();
        let actor = sys.actor_of(LangGraphGpuActor::<State>::props(g), "graph").unwrap();

        let (tx, rx) = oneshot::channel();
        actor.tell(NodeGraphMsg::Run {
            state: State { x: 0, log: vec![] },
            reply: tx,
        });
        let s = tokio::time::timeout(Duration::from_secs(2), rx).await.unwrap().unwrap().unwrap();
        // (0 + 1) * 10 - 5 = 5; order = [a, b, c].
        assert_eq!(s.x, 5);
        assert_eq!(s.log, vec!["a", "b", "c"]);

        sys.terminate().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cycle_returns_error() {
        let mut g = NodeGraph::<i32>::new();
        g.add_node(NodeId(1), |s: i32| Ok(s));
        g.add_node(NodeId(2), |s: i32| Ok(s));
        g.add_edge(NodeId(1), NodeId(2));
        g.add_edge(NodeId(2), NodeId(1));

        let sys = ActorSystem::create("cycle-test", Config::empty()).await.unwrap();
        let actor = sys.actor_of(LangGraphGpuActor::<i32>::props(g), "graph").unwrap();

        let (tx, rx) = oneshot::channel();
        actor.tell(NodeGraphMsg::Run { state: 0, reply: tx });
        let r = tokio::time::timeout(Duration::from_secs(2), rx).await.unwrap().unwrap();
        assert!(matches!(r, Err(GpuError::Unrecoverable(_))));

        sys.terminate().await;
    }
}
