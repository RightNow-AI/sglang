"""Windows collection shims for the narrow CPU-only tree test suite.

SGLang's package initializer imports Triton even when a test only needs the
pure-Python radix tree. On Windows, expose namespace packages and the small
interfaces RadixCache needs so the actual radix_cache.py implementation can be
loaded without importing CUDA-only modules.
"""

import sys
import types
from pathlib import Path


if sys.platform == "win32" and "resource" not in sys.modules:
    resource = types.ModuleType("resource")
    resource.RLIMIT_NOFILE = 0
    resource.RLIMIT_STACK = 1
    resource.getrlimit = lambda _kind: (1024, 1024)
    resource.setrlimit = lambda _kind, _limits: None
    sys.modules["resource"] = resource


if sys.platform == "win32":
    python_root = Path(__file__).resolve().parents[4] / "python"

    def namespace(name, path):
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module

    namespace("sglang", python_root / "sglang")
    namespace("sglang.srt", python_root / "sglang" / "srt")
    namespace("sglang.srt.mem_cache", python_root / "sglang" / "srt" / "mem_cache")
    namespace("sglang.srt.tree", python_root / "sglang" / "srt" / "tree")
    namespace("sglang.test", python_root / "sglang" / "test")
    namespace("sglang.test.ci", python_root / "sglang" / "test" / "ci")

    allocator = types.ModuleType("sglang.srt.mem_cache.allocator")
    allocator.BaseTokenToKVPoolAllocator = type("BaseTokenToKVPoolAllocator", (), {})
    sys.modules[allocator.__name__] = allocator

    memory_pool = types.ModuleType("sglang.srt.mem_cache.memory_pool")
    memory_pool.ReqToTokenPool = type("ReqToTokenPool", (), {})
    sys.modules[memory_pool.__name__] = memory_pool

    metrics = types.ModuleType("sglang.srt.observability.metrics_collector")
    metrics.STAT_LOGGER_ROLE_RADIX_CACHE = "radix_cache"
    metrics.RadixCacheMetricsCollector = type("RadixCacheMetricsCollector", (), {})
    metrics.resolve_collector_class = lambda cls: cls
    sys.modules[metrics.__name__] = metrics

    events = types.ModuleType("sglang.srt.mem_cache.events")

    class KVCacheEventMixin:
        def _record_all_cleared_event(self):
            return None

        def _record_store_event(self, _node):
            return None

        def _record_remove_event(self, _node):
            return None

    events.KVCacheEventMixin = KVCacheEventMixin
    sys.modules[events.__name__] = events

    sessions = types.ModuleType("sglang.srt.mem_cache.session_radix_cache")

    class SessionRadixCacheMixin:
        def _reset_session_radix_state(self):
            return None

        def _discard_session_leaf(self, _node):
            return None

        def _tag_session_leaf(self, _req, _key, node=None):
            return None

    sessions.SessionRadixCacheMixin = SessionRadixCacheMixin
    sys.modules[sessions.__name__] = sessions

    cache_utils = types.ModuleType("sglang.srt.mem_cache.utils")

    class LruStrategy:
        @staticmethod
        def get_priority(node):
            return node.last_access_time

    cache_utils.get_eviction_strategy = lambda _policy: LruStrategy()
    cache_utils.get_hash_str = lambda key, prior_hash=None: str(
        hash((tuple(key), prior_hash))
    )
    cache_utils.split_node_hash_value = lambda values, split_len: (
        values[:split_len],
        values[split_len:],
    )
    sys.modules[cache_utils.__name__] = cache_utils
