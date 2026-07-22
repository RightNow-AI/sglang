"""Numerical parity for the single-read tree-attention decomposition.

The single-read kernel reads a shared prefix's KV ONCE for all B sibling
branches, computes each branch's unique-suffix attention separately, and merges
the two with a log-sum-exp state merge (flashinfer.cascade.merge_state /
_safe_merge_state). This test proves that decomposition is numerically identical
to per-branch full attention - the correctness foundation the kernel rests on.
It needs no GPU: it is the pure attention math, so it runs on CPU and gates the
kernel design before any device wiring.
"""
import math

import pytest

torch = pytest.importorskip("torch")


def _attn_lse(q, k, v, scale):
    """Attention returning (output, log-sum-exp), the online-softmax primitive.

    q: [H, D]   k,v: [T, H, D]  (single query row per head, T keys)
    returns o: [H, D], lse: [H]
    """
    # scores: [H, T]
    scores = torch.einsum("hd,thd->ht", q, k) * scale
    lse = torch.logsumexp(scores, dim=-1)  # [H]
    w = torch.softmax(scores, dim=-1)  # [H, T]
    o = torch.einsum("ht,thd->hd", w, v)  # [H, D]
    return o, lse


def _merge(o_a, lse_a, o_b, lse_b):
    """Merge two partial attention results over disjoint key sets (flash/Hydragen
    online-softmax combine): weight each by its share of the combined denominator."""
    lse = torch.logsumexp(torch.stack([lse_a, lse_b], dim=0), dim=0)  # [H]
    w_a = torch.exp(lse_a - lse).unsqueeze(-1)  # [H,1]
    w_b = torch.exp(lse_b - lse).unsqueeze(-1)
    return o_a * w_a + o_b * w_b


def _full_attn(q, k, v, scale):
    scores = torch.einsum("hd,thd->ht", q, k) * scale
    w = torch.softmax(scores, dim=-1)
    return torch.einsum("ht,thd->hd", w, v)


@pytest.mark.parametrize("dtype,tol", [(torch.float32, 1e-5), (torch.bfloat16, 3e-2)])
@pytest.mark.parametrize("shared_len,suffix_len,H,D", [(128, 7, 8, 64), (1000, 40, 4, 128), (5000, 1, 16, 64)])
def test_shared_read_equals_full_attention(dtype, tol, shared_len, suffix_len, H, D):
    """One branch: full attention over [shared; suffix] == merge(shared, suffix)."""
    torch.manual_seed(0)
    scale = 1.0 / math.sqrt(D)
    q = torch.randn(H, D, dtype=dtype)
    k_shared = torch.randn(shared_len, H, D, dtype=dtype)
    v_shared = torch.randn(shared_len, H, D, dtype=dtype)
    k_suf = torch.randn(suffix_len, H, D, dtype=dtype)
    v_suf = torch.randn(suffix_len, H, D, dtype=dtype)

    k_full = torch.cat([k_shared, k_suf], dim=0)
    v_full = torch.cat([v_shared, v_suf], dim=0)

    ref = _full_attn(q, k_full, v_full, scale).float()

    o_s, lse_s = _attn_lse(q, k_shared, v_shared, scale)
    o_u, lse_u = _attn_lse(q, k_suf, v_suf, scale)
    merged = _merge(o_s, lse_s, o_u, lse_u).float()

    assert torch.allclose(merged, ref, atol=tol, rtol=tol), \
        f"max abs diff {(merged - ref).abs().max().item():.2e} > {tol}"


def test_shared_prefix_read_once_across_branches():
    """The point of the kernel: the shared-prefix attention (o_s, lse_s) is
    computed ONCE and reused for every branch; only each branch's suffix differs.
    Prove that for B branches sharing the prefix, merging the single shared
    result with each branch's suffix equals each branch's full attention."""
    torch.manual_seed(1)
    H, D, shared_len = 8, 64, 512
    B = 8
    scale = 1.0 / math.sqrt(D)

    # ONE shared prefix, read once
    k_shared = torch.randn(shared_len, H, D)
    v_shared = torch.randn(shared_len, H, D)

    for b in range(B):
        q_b = torch.randn(H, D)  # each branch has its own decode query
        suf = 3 + b
        k_suf = torch.randn(suf, H, D)
        v_suf = torch.randn(suf, H, D)

        # shared pass reused (would be batched over all branch queries in the kernel)
        o_s, lse_s = _attn_lse(q_b, k_shared, v_shared, scale)
        o_u, lse_u = _attn_lse(q_b, k_suf, v_suf, scale)
        merged = _merge(o_s, lse_s, o_u, lse_u)

        ref = _full_attn(
            q_b,
            torch.cat([k_shared, k_suf], dim=0),
            torch.cat([v_shared, v_suf], dim=0),
            scale,
        )
        assert torch.allclose(merged, ref, atol=1e-5), \
            f"branch {b}: max diff {(merged - ref).abs().max().item():.2e}"


def test_merge_is_order_independent():
    """merge(a,b) == merge(b,a): the decomposition does not depend on which set
    is 'shared' vs 'suffix', so the kernel is free to batch the shared read."""
    torch.manual_seed(2)
    H, D = 4, 32
    scale = 1.0 / math.sqrt(D)
    q = torch.randn(H, D)
    ka, va = torch.randn(20, H, D), torch.randn(20, H, D)
    kb, vb = torch.randn(5, H, D), torch.randn(5, H, D)
    oa, la = _attn_lse(q, ka, va, scale)
    ob, lb = _attn_lse(q, kb, vb, scale)
    assert torch.allclose(_merge(oa, la, ob, lb), _merge(ob, lb, oa, la), atol=1e-6)
