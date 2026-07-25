"""Tree-KV engine protocol and real scheduler/model implementation."""

from .protocol import (
    BranchMerged,
    BranchPruned,
    BranchStarted,
    EngineCounters,
    EngineEvent,
    EngineProtocol,
    EngineUsage,
    GenerationDone,
    GenerationRequest,
    KVCapacityExceededError,
    Message,
    ModelMetadata,
    TokenGenerated,
    TreeExecution,
    TreeSummary,
)


def __getattr__(name: str):
    if name == "TreeKVEngine":
        from .treekv import TreeKVEngine

        return TreeKVEngine
    raise AttributeError(name)

__all__ = [
    "BranchMerged",
    "BranchPruned",
    "BranchStarted",
    "EngineCounters",
    "EngineEvent",
    "EngineProtocol",
    "EngineUsage",
    "GenerationDone",
    "GenerationRequest",
    "KVCapacityExceededError",
    "Message",
    "ModelMetadata",
    "TokenGenerated",
    "TreeExecution",
    "TreeKVEngine",
    "TreeSummary",
]
