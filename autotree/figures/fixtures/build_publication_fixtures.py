"""Build compact, explicitly synthetic result documents for renderer tests."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SEEDS = (11, 22, 33)
BUDGETS = (1024, 2048, 4096, 8192)
MODELS = {
    "qwen3-8b": ("Qwen3-8B", 0.22, 0.0040, 980.0),
    "llama31-8b": ("Llama-3.1-8B", 0.18, 0.0044, 930.0),
    "qwen3-32b": ("Qwen3-32B", 0.31, 0.0105, 420.0),
    "r1-distill-70b": ("R1-Distill-70B", 0.37, 0.0210, 230.0),
}
SYSTEMS = {
    "sequential": ("Sequential", -0.015, 1.18, 0.72),
    "vllm": ("vLLM", 0.0, 1.0, 1.0),
    "sglang": ("SGLang", 0.008, 0.93, 1.10),
    "autotree": ("AutoTree", 0.026, 0.64, 1.58),
}


def _provenance(source: str) -> dict[str, str]:
    return {
        "kind": "fixture",
        "source": source,
        "license": "Repository license",
        "notice": "FIXTURE DATA ONLY - NOT A BENCHMARK RESULT.",
    }


def _result_document(
    *,
    model: str,
    system: str,
    accuracy_start: float,
    model_cost: float,
    throughput: float,
) -> dict[str, object]:
    system_label, accuracy_delta, cost_factor, throughput_factor = SYSTEMS[system]
    budget_rows = []
    cells = []
    for budget_index, budget in enumerate(BUDGETS):
        budget_name = f"tokens-{budget}"
        budget_rows.append({"name": budget_name, "max_tokens": budget})
        for seed_index, seed in enumerate(SEEDS):
            jitter = (-0.012, 0.0, 0.012)[seed_index]
            accuracy = min(0.94, accuracy_start + accuracy_delta + 0.095 * budget_index + jitter)
            cost_per_correct = model_cost * cost_factor * (1.0 + 0.72 * budget_index)
            rollout_rate = throughput * throughput_factor * (1.0 + (-0.04, 0.0, 0.04)[seed_index])
            cells.append(
                {
                    "budget_name": budget_name,
                    "protocol_seed": seed,
                    "metrics": {
                        "accuracy_at_k": {"16": round(accuracy, 4)},
                        "input_tokens": 256,
                        "output_tokens": budget,
                        "total_cost_usd": round(cost_per_correct * accuracy * 32, 8),
                        "cost_per_correct_usd": round(cost_per_correct, 8),
                        "rollout_throughput_per_hour": {"mean": round(rollout_rate, 4)},
                    },
                }
            )
    return {
        "schema_version": "thoughtbench.results.v1",
        "artifact_notice": "FIXTURE DATA ONLY - NOT A BENCHMARK RESULT.",
        "benchmark_claims_allowed": False,
        "engine_config": {
            "model": model,
            "mode": "tree" if system == "autotree" else "sequential",
            "budgets": budget_rows,
            "tree": (
                {"branches": 8, "policy": "beam", "scorer": "fixture-score"}
                if system == "autotree"
                else None
            ),
        },
        "task_set": {
            "name": "publication-layout-fixture",
            "provenance": _provenance(
                f"Synthetic {model} / {system_label} publication-layout fixture"
            ),
        },
        "per_seed_metrics": cells,
        "aggregate_metrics": [{"budget_name": row["name"]} for row in budget_rows],
        "samples": [],
    }


def _branch_document(branches: int, center_accuracy: float) -> dict[str, object]:
    cells = []
    for seed_index, seed in enumerate(SEEDS):
        accuracy = center_accuracy + (-0.015, 0.0, 0.015)[seed_index]
        cells.append(
            {
                "budget_name": f"branches-{branches}",
                "protocol_seed": seed,
                "metrics": {
                    "accuracy_at_k": {"16": round(accuracy, 4)},
                    "input_tokens": 256,
                    "output_tokens": 4096,
                    "total_cost_usd": 0.1,
                    "cost_per_correct_usd": 0.01,
                    "rollout_throughput_per_hour": {"mean": 100.0},
                },
            }
        )
    return {
        "schema_version": "thoughtbench.results.v1",
        "artifact_notice": "FIXTURE DATA ONLY - NOT A BENCHMARK RESULT.",
        "benchmark_claims_allowed": False,
        "engine_config": {
            "model": "Qwen3-32B (fixture slot)",
            "mode": "tree",
            "budgets": [{"name": f"branches-{branches}", "max_tokens": 4096}],
            "tree": {"branches": branches, "policy": "beam", "scorer": "fixture-score"},
        },
        "task_set": {
            "name": "branching-layout-fixture",
            "provenance": _provenance(
                f"Synthetic branching-factor k={branches} publication-layout fixture"
            ),
        },
        "per_seed_metrics": cells,
        "aggregate_metrics": [{"budget_name": f"branches-{branches}"}],
        "samples": [],
    }


def main() -> None:
    references: list[dict[str, object]] = []
    for slug, (model, accuracy, cost, throughput) in MODELS.items():
        for system, (system_label, _accuracy_delta, _cost_factor, _throughput_factor) in SYSTEMS.items():
            payload = _result_document(
                model=model,
                system=system,
                accuracy_start=accuracy,
                model_cost=cost,
                throughput=throughput,
            )
            path = ROOT / f"publication-{slug}-{system}.results.json"
            path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
            panels: list[str] = []
            if system in {"sequential", "autotree"}:
                panels.append("scaling")
            if system in {"vllm", "sglang", "autotree"}:
                panels.append("pareto")
            if slug == "qwen3-32b":
                panels.append("throughput")
            references.append(
                {
                    "path": path.name,
                    "label": f"{model} - {system_label}",
                    "model": model,
                    "system": system_label,
                    "panels": panels,
                }
            )

    for branches, accuracy in zip((1, 2, 4, 8, 16, 32), (0.42, 0.50, 0.58, 0.63, 0.64, 0.62), strict=True):
        path = ROOT / f"publication-branch-{branches}.results.json"
        path.write_text(
            json.dumps(_branch_document(branches, accuracy), indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        references.append(
            {
                "path": path.name,
                "label": f"AutoTree k={branches}",
                "model": "Qwen3-32B (fixture slot)",
                "system": "AutoTree",
                "panels": ["branching"],
            }
        )

    bundle_path = ROOT / "all-figures.fixture.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    bundle["thoughtbench_results"] = references
    bundle["supplemental"]["kv_reuse_heatmap"]["values"] = [
        [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
        [1.0, 1.5, 2.1, 2.8, 3.3, 3.7],
        [1.0, 2.0, 3.2, 4.5, 5.5, 6.2],
        [1.0, 2.4, 4.1, 6.3, 8.1, 9.4],
        [1.0, 2.6, 4.6, 7.2, 9.8, 11.6],
    ]
    bundle_path.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
