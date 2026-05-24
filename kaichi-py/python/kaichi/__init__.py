"""kaichi — Perturb-seq guide assignment.

Public API:

    import kaichi
    adata = kaichi.assign("input.h5ad", model="poisson")
    # adata is an in-memory anndata.AnnData with:
    #   .X                       — preserved raw UMI counts
    #   .layers["assigned"]      — binary CSR (1 where assigned)
    #   .obs[...]                — per-cell assignment columns
    #   .uns["kaichi"]           — provenance ({"model", "model_params", "version"})

    # Two-stage score/decide API (v0.2):
    scores = kaichi.score("gRNA_counts.h5ad", model="poisson_gauss", n_jobs=8)
    a1 = kaichi.decide(scores, min_confidence=0.7)  # pyarrow.Table
    a2 = kaichi.decide(scores, min_confidence=0.9)  # reuse cached scores

See docs/design/binding-interop.md for the contract.
"""

from __future__ import annotations

import json
from typing import Any

import anndata
import numpy as np
import pandas as pd
import pyarrow as pa
import scipy.sparse as sp

from ._native import __version__
from . import _native

# Re-export the Rust class under the public name ScoreResult.
ScoreResult = _native.PyScoreResult

__all__ = ["assign", "score", "decide", "to_anndata", "ScoreResult", "__version__"]


def assign(
    h5ad_path: str,
    model: str = "poisson",
    *,
    min_confidence: float | None = None,
    quantile: float | None = None,
    n_jobs: int | None = None,
) -> anndata.AnnData:
    """Run a kaichi guide-assignment model on an h5ad and return an AnnData.

    The h5ad is read by the Rust core (so the binding works on any h5ad the
    Rust reader supports, including categorical batch labels). The returned
    AnnData is constructed in memory — no output file is written.

    Parameters
    ----------
    h5ad_path :
        Path to an h5ad with guide UMI counts in ``X``. Cell barcodes are
        read from ``obs/_index``; guide IDs from ``var/_index``; the optional
        ``obs/batch`` categorical is used by hierarchical models for the γ_b
        per-batch offset.
    model :
        One of ``umi``, ``max``, ``ratio``, ``gauss``, ``poisson_gauss``,
        ``poisson``, ``neg_binomial``, ``binomial``, ``beta2``, ``beta3``,
        ``quantiles``.
    min_confidence :
        Override the model's posterior threshold for assignment. Ignored for
        models that don't use one (``umi``, ``max``, ``ratio``, ``quantiles``).
    quantile :
        Top-fraction threshold for the ``quantiles`` model. Ignored otherwise.
    n_jobs :
        Rayon worker threads for per-guide EM fitting. ``None`` (default) or
        ``0`` uses half of the machine's logical cores — HPC-polite. A positive
        int overrides (e.g. ``n_jobs=os.cpu_count()`` for all cores).

    Returns
    -------
    anndata.AnnData
        ``.X``                  raw UMI counts preserved from input (sparse CSR, uint32)
        ``.layers["assigned"]`` binary sparse CSR (uint8), 1 where (cell, guide) was assigned
        ``.obs``                per-cell assignment columns (guide_id, assignment_confidence,
                                umi_count, is_unassigned, is_multi_infected, n_guides_detected)
        ``.var``                indexed by guide_id
        ``.uns["kaichi"]``      {"model", "model_params", "version"}
    """
    (
        batch_py,
        a_indptr,
        a_indices,
        x_data,
        x_indices,
        x_indptr,
        cell_barcodes,
        guide_ids,
        model_name,
        model_params_json,
    ) = _native._assign_from_h5ad_inmem(
        h5ad_path,
        model,
        min_confidence=min_confidence,
        quantile=quantile,
        n_jobs=n_jobs,
    )

    n_cells = len(cell_barcodes)
    n_guides = len(guide_ids)

    # Preserved counts X.
    X = sp.csr_matrix(
        (x_data, x_indices, x_indptr),
        shape=(n_cells, n_guides),
    )

    # Binary assigned layer. `data` is implicit 1s; we reconstruct here rather
    # than ferry an array of ones across the Rust → Python boundary.
    nnz = int(a_indices.shape[0])
    assigned = sp.csr_matrix(
        (np.ones(nnz, dtype=np.uint8), a_indices, a_indptr),
        shape=(n_cells, n_guides),
    )

    obs = batch_py.to_pandas()
    obs.index = pd.Index(cell_barcodes, name="cell_barcode")

    var = pd.DataFrame(index=pd.Index(guide_ids, name="guide_id"))

    uns: dict[str, Any] = {
        "kaichi": {
            "model": model_name,
            "model_params": json.loads(model_params_json),
            "version": __version__,
        }
    }

    return anndata.AnnData(
        X=X,
        obs=obs,
        var=var,
        layers={"assigned": assigned},
        uns=uns,
    )


def score(
    h5ad_path: str,
    model: str = "poisson_gauss",
    *,
    n_jobs: int | None = None,
) -> ScoreResult:
    """Score guide counts with a two-stage model; return a cached ScoreResult.

    The expensive EM fitting runs once. Call :func:`decide` one or more times
    with different ``min_confidence`` thresholds without re-fitting.

    Parameters
    ----------
    h5ad_path :
        Path to an h5ad with guide UMI counts in ``X``.
    model :
        One of ``poisson_gauss``, ``poisson``, ``neg_binomial``, ``binomial``.
        Single-stage models (``umi``, ``ratio``, ``max``) raise ``ValueError``
        — use :func:`assign` for those.
    n_jobs :
        Rayon worker threads. ``None`` / ``0`` = half of logical cores.

    Returns
    -------
    ScoreResult
        A Rust-owned score matrix. Access via ``.values_csr``, ``.cell_barcodes``,
        ``.guide_ids``, ``.shape``, ``.model``, ``.model_params``.
    """
    return _native._score(h5ad_path, model, n_jobs=n_jobs)


def decide(
    scores: ScoreResult,
    min_confidence: float = 0.9,
) -> pa.Table:
    """Apply a confidence threshold to a :class:`ScoreResult`.

    Parameters
    ----------
    scores :
        The cached output of :func:`score`.
    min_confidence :
        Posterior threshold in [0, 1]. Entries below this threshold are
        left unassigned.

    Returns
    -------
    pyarrow.RecordBatch
        One row per cell with columns: ``cell_barcode``, ``guide_id``,
        ``assignment_confidence``, ``umi_count``, ``is_unassigned``,
        ``is_multi_infected``, ``n_guides_detected``. ``min_confidence`` is
        stamped into the schema metadata so :func:`to_anndata` can recover it
        without the caller passing it through twice.
    """
    batch = _native._decide(scores, min_confidence)
    # pyo3-arrow 0.5 doesn't preserve schema metadata across PyCapsule
    # transit, so stamp it here instead. (The Rust side also stamps it for
    # the CLI path, which uses the RecordBatch in-process.)
    existing = batch.schema.metadata or {}
    new_meta = {**existing, b"kaichi.min_confidence": str(min_confidence).encode("utf-8")}
    return batch.replace_schema_metadata(new_meta)


def to_anndata(
    scores: ScoreResult,
    decisions: pa.RecordBatch | None = None,
) -> anndata.AnnData:
    """Materialize a :class:`ScoreResult` (optionally with decisions) as an AnnData.

    Two modes:

    - ``to_anndata(scores)`` returns a "scored" AnnData with ``X`` = preserved
      UMI counts and ``layers["scores"]`` = float32 posteriors.
    - ``to_anndata(scores, decisions)`` additionally attaches
      ``layers["assigned"]`` (binary CSR) and the per-cell assignment columns
      from ``decisions``. ``min_confidence`` is recovered automatically from
      the RecordBatch schema metadata stamped by :func:`decide`.

    Parameters
    ----------
    scores :
        The output of :func:`score`.
    decisions :
        Optional output of :func:`decide`. When omitted, the returned AnnData
        is in the "scored" stage (no assignment layer or obs columns).

    Returns
    -------
    anndata.AnnData
        Same shape and conventions as :func:`assign`'s output when
        ``decisions`` is supplied.
    """
    n_cells, n_guides = scores.shape

    # Pull the four CSR arrays out of the Rust-owned ScoreResult.
    scores_data = np.asarray(pa.array(scores.scores_data))
    umi_data = np.asarray(pa.array(scores.umi_counts))
    indices = np.asarray(pa.array(scores.indices))
    indptr = np.asarray(pa.array(scores.indptr))

    X = sp.csr_matrix((umi_data, indices, indptr), shape=(n_cells, n_guides))
    scores_layer = sp.csr_matrix(
        (scores_data, indices.copy(), indptr.copy()), shape=(n_cells, n_guides)
    )

    obs = pd.DataFrame(index=pd.Index(pa.array(scores.cell_barcodes).to_pylist(), name="cell_barcode"))
    var = pd.DataFrame(index=pd.Index(pa.array(scores.guide_ids).to_pylist(), name="guide_id"))
    layers: dict[str, Any] = {"scores": scores_layer}
    uns: dict[str, Any] = {
        "kaichi": {
            "model": scores.model,
            "model_params": scores.model_params,
            "stage": "scored",
            "version": __version__,
        }
    }

    if decisions is not None:
        # Build the binary assigned layer from is_unassigned + best guide_id.
        df = decisions.to_pandas()
        guide_to_idx = {g: i for i, g in enumerate(var.index)}
        rows: list[int] = []
        cols: list[int] = []
        for i, (gid, unassigned) in enumerate(zip(df["guide_id"], df["is_unassigned"])):
            if not unassigned and gid is not None and gid in guide_to_idx:
                rows.append(i)
                cols.append(guide_to_idx[gid])
        assigned_data = np.ones(len(rows), dtype=np.uint8)
        assigned = sp.csr_matrix(
            (assigned_data, (rows, cols)), shape=(n_cells, n_guides), dtype=np.uint8
        )
        layers["assigned"] = assigned

        # Attach per-cell assignment columns to obs (drop cell_barcode — it's the index).
        obs_cols = df.drop(columns=["cell_barcode"]).set_index(obs.index)
        obs = obs_cols

        # Recover min_confidence from schema metadata if present.
        meta = decisions.schema.metadata or {}
        mc_raw = meta.get(b"kaichi.min_confidence")
        if mc_raw is not None:
            uns["kaichi"]["min_confidence"] = float(mc_raw.decode("utf-8"))
        uns["kaichi"]["stage"] = "assigned"

    return anndata.AnnData(X=X, obs=obs, var=var, layers=layers, uns=uns)
