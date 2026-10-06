"""
Receptive field diagnostics for an attention/attribution block: how far apart
(in timesteps) the token pairs carrying positive mass tend to be. Pure numpy,
no plotting -- see analysis/render/attn_attr_figures.py for the figure that
visualizes this.
"""
import numpy as np


def receptive_field_diagnostics(mat, ts_query, ts_key, exclude_self_loops=False):
    """
    Weighted distribution of |timestep(i) - timestep(j)| for one attribution
    block. Follows the ATTATTR paper's own tree/top-node convention, not
    magnitude-based mass: normalize by abs(the block's raw signed max), then
    clip to positive -- negative attribution is dropped, not folded in by
    magnitude.

    `exclude_self_loops` drops i==j; only meaningful for self-attention
    blocks, where it is a genuine self-loop (leave False for cross-attention,
    where i==j is real signal: same timestep, different token type).

    Returns weighted mean/median/p90 of |timestep(i)-timestep(j)| over the
    kept (positive) cells, plus the flattened (distances, weights) arrays.
    """
    mat = np.asarray(mat, dtype=np.float64)
    ts_query = np.asarray(ts_query)
    ts_key = np.asarray(ts_key)
    assert mat.shape == (len(ts_query), len(ts_key)), (
        f"mat shape {mat.shape} does not match (len(ts_query), len(ts_key)) = "
        f"({len(ts_query)}, {len(ts_key)})"
    )
    if mat.size == 0:  # n_a == 0 on the first step of an episode
        nan = float("nan")
        return {"mean_distance": nan, "median_distance": nan, "p90_distance": nan,
                "distances": np.array([]), "weights": np.array([])}

    peak = mat.max()
    assert peak > 0, (
        "block max is <= 0 -- this block has no positive attribution at all, "
        "so normalizing by it would not isolate positive signal as intended"
    )
    weight = np.clip(mat / peak, 0.0, None)
    if exclude_self_loops:
        assert mat.shape[0] == mat.shape[1], (
            "exclude_self_loops=True needs a square (self-attention) block"
        )
        weight = weight * ~np.eye(mat.shape[0], dtype=bool)

    distance = np.abs(ts_query[:, None] - ts_key[None, :])
    d_flat, w_flat = distance.flatten(), weight.flatten()
    keep = w_flat > 0
    d_flat, w_flat = d_flat[keep], w_flat[keep]

    total = w_flat.sum()
    assert total > 0, "no positive attribution survives normalization/masking in this block"
    w_flat = w_flat / total  # sum to 1 over the kept cells

    order = np.argsort(d_flat)
    d_sorted, w_sorted = d_flat[order], w_flat[order]
    cum = np.cumsum(w_sorted)

    return {
        "mean_distance": float((d_flat * w_flat).sum()),
        "median_distance": float(d_sorted[np.searchsorted(cum, 0.5)]),
        "p90_distance": float(d_sorted[np.searchsorted(cum, 0.9)]),
        "distances": d_flat,
        "weights": w_flat,
    }


def receptive_field_decoder(mat, ts, is_real, include_mask, exclude_self_loops=True):
    """
    Decoder-specific wrapper implementing the include_mask toggle: decides
    which rows/cols of the decoder's [2K, 2K] attribution block get fed into
    `receptive_field_diagnostics`, which is called UNMODIFIED either way --
    this is the only place in the repo where mask-token inclusion/exclusion
    is decided. Fusion/encoder blocks never carry mask tokens (see this
    module's and attn_attr.py's docstrings) and should call
    `receptive_field_diagnostics` directly instead of this wrapper.

    mat: [2K, 2K], one decoder layer x head slice of attr_dec.
    ts: [2K] real timesteps for every decoder position (from
        analysis.attn_attr.decoder_slot_real_timesteps).
    is_real: [2K] bool, True at positions holding a real (non-mask) token
        (from analysis.attn_attr.decoder_real_mask).
    include_mask: True -> full 2K x 2K support (mask positions included in
        the row/col set). False -> restricted to only the `is_real`
        positions (real state/action tokens only).

    Returns the same dict as `receptive_field_diagnostics`.
    """
    if include_mask:
        return receptive_field_diagnostics(mat, ts, ts, exclude_self_loops=exclude_self_loops)
    idx = np.flatnonzero(is_real)
    mat_real = mat[np.ix_(idx, idx)]
    ts_real = ts[idx]
    return receptive_field_diagnostics(mat_real, ts_real, ts_real, exclude_self_loops=exclude_self_loops)
