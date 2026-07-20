from sglang.srt.tree.manager import TreeTokenBudget
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-c-test-cpu")


def test_tree_budget_stops_exactly_at_limit_across_branches():
    budget = TreeTokenBudget(5)

    updates = [
        budget.consume(branch_id)
        for branch_id in ("0", "1", "0", "2", "1", "2")
    ]

    assert [update.accepted for update in updates] == [True] * 5 + [False]
    assert [update.spent for update in updates] == [1, 2, 3, 4, 5, 5]
    assert [update.exhausted for update in updates] == [
        False,
        False,
        False,
        False,
        True,
        True,
    ]
    assert budget.spent == 5
    assert budget.spent_per_branch == {"0": 2, "1": 2, "2": 1}
