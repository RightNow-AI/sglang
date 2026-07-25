#![cfg(feature = "python")]

use autotree_scheduler::PyScheduler;

fn assert_send<T: Send>() {}

#[test]
fn python_scheduler_is_a_sendable_binding_type() {
    assert_send::<PyScheduler>();
}
