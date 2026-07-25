use autotree_scheduler::{BranchId, BranchState, BranchTree, KillReason};

#[test]
fn arena_fork_preserves_parent_state_and_navigation() {
    let mut tree = BranchTree::new();
    let root = tree.root();

    tree.record_token(root, -0.25).expect("root is active");
    tree.record_value(root, 0.75).expect("root is active");
    let children = tree.fork(root, 2).expect("fork succeeds");

    assert_eq!(root, BranchId(0));
    assert_eq!(children, vec![BranchId(1), BranchId(2)]);
    assert_eq!(tree.children(root).unwrap(), children.as_slice());
    assert_eq!(tree.parent(children[0]).unwrap(), Some(root));

    let root_node = tree.get(root).unwrap();
    assert_eq!(root_node.state(), BranchState::Expanded);

    let child = tree.get(children[0]).unwrap();
    assert_eq!(child.parent(), Some(root));
    assert_eq!(child.depth(), 1);
    assert_eq!(child.tokens_generated(), 1);
    assert_eq!(child.cumulative_logprob(), -0.25);
    assert_eq!(child.value_estimate(), 0.75);
    assert_eq!(child.state(), BranchState::Active);
}

#[test]
fn terminal_transitions_are_leaf_first_and_lookup_is_constant_handle_based() {
    let mut tree = BranchTree::new();
    let root = tree.root();
    let children = tree.fork(root, 2).unwrap();

    assert!(tree.kill(root, KillReason::Drained).is_err());
    tree.kill(children[0], KillReason::Drained).unwrap();
    tree.finalize(children[1]).unwrap();
    tree.kill(root, KillReason::AncestorReclaimed).unwrap();

    assert_eq!(tree.get(children[0]).unwrap().state(), BranchState::Killed);
    assert_eq!(
        tree.get(children[1]).unwrap().state(),
        BranchState::Finalized
    );
    assert_eq!(tree.get(root).unwrap().state(), BranchState::Killed);
    assert!(tree.get(BranchId(99)).is_none());
}

#[test]
fn failed_backpropagation_leaves_visit_statistics_unchanged() {
    let mut tree = BranchTree::new();
    let root = tree.root();
    tree.backpropagate(root, f64::MAX).unwrap();

    let before_visits = tree.get(root).unwrap().visits();
    let before_sum = tree.get(root).unwrap().value_sum();
    assert!(tree.backpropagate(root, f64::MAX).is_err());

    assert_eq!(tree.get(root).unwrap().visits(), before_visits);
    assert_eq!(tree.get(root).unwrap().value_sum(), before_sum);
}

#[test]
fn backpropagation_updates_every_ancestor_once() {
    let mut tree = BranchTree::new();
    let root = tree.root();
    let child = tree.fork(root, 2).unwrap()[0];
    let grandchild = tree.fork(child, 2).unwrap()[0];

    tree.backpropagate(grandchild, 0.75).unwrap();

    for branch in [root, child, grandchild] {
        assert_eq!(tree.get(branch).unwrap().visits(), 1);
        assert_eq!(tree.get(branch).unwrap().value_sum(), 0.75);
    }
    assert_eq!(tree.get(BranchId(2)).unwrap().visits(), 0);
}

#[test]
fn nested_forks_use_one_global_contiguous_branch_id_sequence() {
    let mut tree = BranchTree::new();
    let first = tree.fork(tree.root(), 2).unwrap();
    let nested = tree.fork(first[0], 3).unwrap();
    let sibling = tree.fork(first[1], 1).unwrap();

    assert_eq!(first, vec![BranchId(1), BranchId(2)]);
    assert_eq!(nested, vec![BranchId(3), BranchId(4), BranchId(5)]);
    assert_eq!(sibling, vec![BranchId(6)]);
}
