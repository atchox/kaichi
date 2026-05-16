"""End-to-end tests for the Python binding.

These exercise the same Schraivogel h5ad fixture the Rust-side equivalence
tests use. The fixture lives under `tests/fixtures/crispat/`; if it's missing
the Rust harness downloads it on first run. Run the Rust suite once before
running these to guarantee the fixture is on disk.
"""

from __future__ import annotations

from pathlib import Path

import anndata
import numpy as np
import pytest
import scipy.sparse as sp

import kaichi

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHRAIVOGEL_H5AD = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "crispat"
    / "example_data"
    / "Schraivogel"
    / "gRNA_counts.h5ad"
)


@pytest.fixture(scope="module")
def h5ad_path() -> str:
    if not SCHRAIVOGEL_H5AD.exists():
        pytest.skip(
            f"Schraivogel fixture not found at {SCHRAIVOGEL_H5AD}. "
            "Run `cargo test -p kaichi-core --test equivalence` once first."
        )
    return str(SCHRAIVOGEL_H5AD)


# ---------------------------------------------------------------------------
# Output shape and schema
# ---------------------------------------------------------------------------


def test_assign_returns_anndata_with_expected_layout(h5ad_path: str) -> None:
    """The contract from binding-interop.md: X preserved, assigned layer, obs
    columns, uns provenance — all on one in-memory AnnData."""
    adata = kaichi.assign(h5ad_path, model="poisson")

    assert isinstance(adata, anndata.AnnData)

    # X preserved as sparse CSR with the same shape as the input counts.
    assert sp.issparse(adata.X)
    assert adata.X.format == "csr"
    assert adata.n_obs > 0 and adata.n_vars > 0

    # Binary assigned layer with matching shape.
    assert "assigned" in adata.layers
    layer = adata.layers["assigned"]
    assert sp.issparse(layer)
    assert layer.shape == adata.X.shape
    # All values are 0 or 1.
    assert np.array_equal(np.unique(layer.data), np.array([1], dtype=layer.dtype))

    # All the obs columns kaichi-core writes.
    for col in [
        "guide_id",
        "umi_count",
        "assignment_confidence",
        "is_unassigned",
        "is_multi_infected",
        "n_guides_detected",
    ]:
        assert col in adata.obs.columns, f"missing obs/{col}"

    # Provenance.
    assert "kaichi" in adata.uns
    assert adata.uns["kaichi"]["model"] == "poisson"
    assert isinstance(adata.uns["kaichi"]["model_params"], dict)
    assert "version" in adata.uns["kaichi"]


def test_assign_respects_min_confidence(h5ad_path: str) -> None:
    """A higher confidence threshold must produce fewer assignments. This is
    the clearest behavioral check that kwargs actually flow through to the
    Rust model."""
    loose = kaichi.assign(h5ad_path, model="poisson", min_confidence=0.5)
    strict = kaichi.assign(h5ad_path, model="poisson", min_confidence=0.95)

    n_loose = int((~loose.obs["is_unassigned"].astype(bool)).sum())
    n_strict = int((~strict.obs["is_unassigned"].astype(bool)).sum())
    assert n_strict <= n_loose, f"strict ({n_strict}) > loose ({n_loose})"
    # And the strict threshold must actually filter SOMETHING out, otherwise
    # we haven't proven the parameter is reaching the model.
    assert n_strict < n_loose


def test_assigned_layer_is_consistent_with_obs(h5ad_path: str) -> None:
    """Every cell's `n_guides_detected` should equal the row sum of the
    binary assigned layer. Cross-checks that the two outputs come from the
    same per-cell assignment loop in Rust."""
    adata = kaichi.assign(h5ad_path, model="poisson")
    row_sums = np.asarray(adata.layers["assigned"].sum(axis=1)).flatten()
    expected = adata.obs["n_guides_detected"].to_numpy()
    assert np.array_equal(row_sums.astype(expected.dtype), expected)


def test_unknown_model_raises(h5ad_path: str) -> None:
    with pytest.raises(ValueError, match="unknown model"):
        kaichi.assign(h5ad_path, model="not_a_real_model")


# ---------------------------------------------------------------------------
# Trivial models (no fitting) should agree with their natural rules
# ---------------------------------------------------------------------------


def test_max_model_assigns_each_cell_to_its_argmax(h5ad_path: str) -> None:
    """The `max` model picks the guide with the highest UMI count per cell
    (or marks the cell unassigned if there's a tie at the top). This is a
    rule, not a fit — easy to verify directly against the input matrix."""
    adata = kaichi.assign(h5ad_path, model="max")

    # Recover the original counts and check assigned cells match argmax.
    X = adata.X.tocsr()
    assigned = adata.layers["assigned"].tocsr()

    # Sample a few assigned cells; full check would scan ~10k cells.
    assigned_cells = np.where(~adata.obs["is_unassigned"].astype(bool))[0][:50]
    for cell in assigned_cells:
        row = X.getrow(cell)
        if row.nnz == 0:
            continue
        top_guide = int(row.indices[np.argmax(row.data)])
        # That guide must be flagged in the assigned layer.
        layer_row = assigned.getrow(cell)
        assert top_guide in layer_row.indices.tolist(), (
            f"max model: cell {cell} expected guide {top_guide} in assigned layer"
        )
