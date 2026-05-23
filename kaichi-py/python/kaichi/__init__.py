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

__all__ = ["assign", "score", "decide", "ScoreResult", "__version__"]


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
        ``is_multi_infected``, ``n_guides_detected``.
    """
    return _native._decide(scores, min_confidence)
