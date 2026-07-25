"""Prometheus metrics derived from engine events."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from .engine import (
    BranchMerged,
    BranchPruned,
    BranchStarted,
    EngineEvent,
    GenerationDone,
)


class ServeMetrics:
    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()
        self.kv_reuse_ratio = Gauge(
            "kv_reuse_ratio",
            "Logical KV tokens divided by physical KV tokens for the latest request.",
            registry=self.registry,
        )
        self.useful_token_ratio = Gauge(
            "useful_token_ratio",
            "Winning-branch tokens divided by all generated branch tokens.",
            registry=self.registry,
        )
        self.active_branches = Gauge(
            "active_branches",
            "Branches started but not yet pruned, merged, or selected as winner.",
            registry=self.registry,
        )
        self.tokens_per_second = Gauge(
            "tokens_per_second",
            "Generated branch tokens per second for the latest request.",
            registry=self.registry,
        )
        self.ttft_seconds = Histogram(
            "ttft_seconds",
            "Time to first generated token in seconds.",
            registry=self.registry,
        )
        self.requests_total = Counter(
            "requests_total",
            "HTTP requests handled by the serving API.",
            ("endpoint", "status"),
            registry=self.registry,
        )
        self.generated_tokens_total = Counter(
            "generated_tokens_total",
            "All tokens generated across branches.",
            registry=self.registry,
        )
        self.branch_events_total = Counter(
            "branch_events_total",
            "Branch lifecycle events emitted by serving engines.",
            ("event",),
            registry=self.registry,
        )
        self.capacity_rejections_total = Counter(
            "capacity_rejections_total",
            "Requests or streams rejected because Tree-KV capacity was exhausted.",
            registry=self.registry,
        )
        self.in_flight_requests = Gauge(
            "in_flight_requests",
            "Generation requests currently executing inside the concurrency limit.",
            registry=self.registry,
        )
        self.queued_requests = Gauge(
            "queued_requests",
            "Generation requests waiting for a concurrency slot.",
            registry=self.registry,
        )
        self.concurrency_rejections_total = Counter(
            "concurrency_rejections_total",
            "Generation requests rejected because the runner stopped admission.",
            registry=self.registry,
        )
        self.quota_rejections_total = Counter(
            "quota_rejections_total",
            "Requests rejected by per-tenant generated-token quotas.",
            registry=self.registry,
        )

    def observe_event(self, event: EngineEvent) -> None:
        if isinstance(event, BranchStarted):
            self.active_branches.inc()
            self.branch_events_total.labels(event="started").inc()
            return
        if isinstance(event, BranchPruned):
            self.active_branches.dec()
            self.branch_events_total.labels(event="pruned").inc()
            return
        if isinstance(event, BranchMerged):
            self.active_branches.dec()
            self.branch_events_total.labels(event="merged").inc()
            return
        if not isinstance(event, GenerationDone):
            return

        self.active_branches.dec()
        self.branch_events_total.labels(event="completed").inc()
        counters = event.counters
        kv_ratio = counters.logical_tokens / max(counters.physical_tokens, 1)
        useful_ratio = counters.useful_tokens / max(event.usage.completion_tokens, 1)
        throughput = event.usage.completion_tokens / max(counters.elapsed_seconds, 1e-9)
        self.kv_reuse_ratio.set(kv_ratio)
        self.useful_token_ratio.set(useful_ratio)
        self.tokens_per_second.set(throughput)
        self.ttft_seconds.observe(counters.ttft_seconds)
        self.generated_tokens_total.inc(event.usage.completion_tokens)
