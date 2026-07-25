"""Validated schemas for figure bundles and supplemental renderer inputs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Provenance(StrictModel):
    kind: str = Field(min_length=1)
    source: str = Field(min_length=1)
    license: str | None = None
    notice: str | None = None


class ResultReference(StrictModel):
    path: str = Field(min_length=1)
    label: str = Field(min_length=1)
    model: str | None = None
    system: str | None = None
    panels: list[Literal["scaling", "pareto", "branching", "throughput"]]


class KVReuseHeatmap(StrictModel):
    schema_version: Literal["autotree.kv-reuse-sweep.v1"]
    provenance: Provenance
    depths: list[int] = Field(min_length=1)
    branching_factors: list[int] = Field(min_length=1)
    values: list[list[float]] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_matrix(self) -> "KVReuseHeatmap":
        if len(self.values) != len(self.depths):
            raise ValueError("kv_reuse_heatmap values must have one row per depth")
        if any(len(row) != len(self.branching_factors) for row in self.values):
            raise ValueError("kv_reuse_heatmap rows must match branching_factors")
        if any(value < 1 for row in self.values for value in row):
            raise ValueError("kv_reuse_heatmap values must be multipliers >= 1")
        return self


class TopologyNode(StrictModel):
    id: str = Field(min_length=1)
    parent: str | None = None
    value: float = Field(ge=0, le=1)
    pruned: bool = False
    label: str | None = None


class TreeTopology(StrictModel):
    schema_version: Literal["autotree.branch-trace.v1"]
    provenance: Provenance
    title: str = Field(min_length=1)
    nodes: list[TopologyNode] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_tree(self) -> "TreeTopology":
        ids = [node.id for node in self.nodes]
        if len(ids) != len(set(ids)):
            raise ValueError("tree_topology node ids must be unique")
        roots = [node for node in self.nodes if node.parent is None]
        if len(roots) != 1:
            raise ValueError("tree_topology must contain exactly one root")
        known = set(ids)
        if any(node.parent not in known for node in self.nodes if node.parent is not None):
            raise ValueError("tree_topology parent must reference a node id")
        children: dict[str, list[str]] = {node_id: [] for node_id in ids}
        for node in self.nodes:
            if node.parent is not None:
                children[node.parent].append(node.id)
        reachable = {roots[0].id}
        queue = [roots[0].id]
        while queue:
            current = queue.pop()
            for child in children[current]:
                if child not in reachable:
                    reachable.add(child)
                    queue.append(child)
        if reachable != known:
            raise ValueError("tree_topology must be connected and acyclic")
        return self


class EffortPoint(StrictModel):
    effort: float = Field(ge=0, le=1)
    seed_values: list[float] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def validate_accuracy(self) -> "EffortPoint":
        if any(value < 0 or value > 1 for value in self.seed_values):
            raise ValueError("thinking_effort seed_values must be in [0, 1]")
        return self


class EffortSeries(StrictModel):
    label: str = Field(min_length=1)
    points: list[EffortPoint] = Field(min_length=1)


class ThinkingEffortSweep(StrictModel):
    schema_version: Literal["autotree.thinking-effort-sweep.v1"]
    provenance: Provenance
    model: str = Field(min_length=1)
    series: list[EffortSeries] = Field(min_length=1)


class SupplementalInputs(StrictModel):
    kv_reuse_heatmap: KVReuseHeatmap
    tree_topology: TreeTopology
    thinking_effort: ThinkingEffortSweep


class FigureBundle(StrictModel):
    schema_version: Literal["autotree.figures.v1"]
    provenance: Provenance
    thoughtbench_results: list[ResultReference] = Field(min_length=1)
    supplemental: SupplementalInputs
