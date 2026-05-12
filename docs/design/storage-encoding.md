# kaichi — Storage and Encoding

This doc pins down how kaichi reads and writes H5AD / H5MU files, which crates are
involved, and where kaichi owns the encoding logic versus delegates it.

---

## Core principle

kaichi-core is **Python-free for all disk I/O**. The CLI is a standalone Rust binary;
the Python and R bindings only do format work for in-memory object construction.
On-disk read/write happens entirely in Rust.

This forces the question: what does kaichi need to understand about the AnnData / MuData
encoding spec? Answer: **as little as possible, with anndata-rs filling in where it can.**

---

## Three layers of the I/O stack

```
┌──────────────────────────────────────────────────────┐
│ Layer 3: H5MU container                              │
│   Owned by kaichi.                                   │
│   - /mod-order attribute                             │
│   - /encoding-type = "MuData", encoding-version      │
│   - /mod/<name>/ groups (each one is an AnnData)     │
└──────────────────────────────────────────────────────┘
                      ▼
┌──────────────────────────────────────────────────────┐
│ Layer 2: Per-modality AnnData                        │
│   Delegated to anndata-rs.                           │
│   - Read existing modalities from input              │
│   - Construct + write the crispr modality kaichi     │
│     produces (X sparse, obs, var, uns)               │
└──────────────────────────────────────────────────────┘
                      ▼
┌──────────────────────────────────────────────────────┐
│ Layer 1: Raw HDF5                                    │
│   Direct access via `hdf5` crate.                    │
│   - H5Ocopy for structural pass-through of unchanged │
│     modalities (no semantic parsing)                 │
│   - File creation, group attributes, fall-back paths │
└──────────────────────────────────────────────────────┘
```

---

## Dependency: anndata-rs

[anndata-rs](https://github.com/kaizhang/anndata-rs) (Kai Zhang, SnapATAC2 author) is
the AnnData spec implementation kaichi depends on for **Layer 2**.

### What we use it for
- Reading per-modality AnnData groups (the crispr modality from a prior run, or a guide-
  only H5AD input).
- Writing the crispr modality kaichi produces.
- Spec-correct encoding of `.X` (sparse CSR), `.obs` and `.var` DataFrames (including
  categorical columns), and `.uns` provenance.

### What we do NOT use it for
- H5MU container structure — anndata-rs does not document MuData support. kaichi
  writes the H5MU wrapper itself (Layer 3).
- Structural pass-through of unmodified modalities — that's a raw HDF5 `H5Ocopy`
  operation (Layer 1) and doesn't need spec interpretation.
- Anything Python-side. The Python binding never imports anndata-rs.

### Dependency strategy

anndata-rs has no published releases on crates.io as of design time. Options ordered
by preference:

1. **Git dependency pinned to a specific commit.** Standard Cargo `git = "..." rev = "..."`.
   Cleanest if upstream is responsive to occasional rev bumps.
2. **Vendor a snapshot into `kaichi-core/vendor/anndata-rs/`.** If upstream proves
   unstable or unresponsive. We own bumping it.
3. **Upstream contribution.** Open an issue / PR asking for a release, possibly
   contributing H5MU support back. Long-term cleanest but slower; treat as parallel
   track, not blocker.

Default: **option 1** at v0. Revisit if it causes pain.

---

## H5MU container — what kaichi owns

The MuData on-disk spec is small. kaichi writes it from scratch using the `hdf5` crate.
The full structure we emit:

```
file.h5mu
├── attrs: encoding-type      = "MuData"
│          encoding-version   = (pinned at v1 release)
│          mod-order          = ["rna", "crispr"]   # ordered list
├── mod/
│   ├── rna/    (group — either H5Ocopy'd from input, or absent if guide-only)
│   └── crispr/ (group — written by anndata-rs)
└── (optional global obs/var if multi-modal joint annotation is provided —
     not used by kaichi v0)
```

The exact attribute names and encoding version we pin to are recorded in code in
`kaichi-core/src/io/h5mu.rs` (constants) and tracked here when bumped.

---

## Structural pass-through (the H5Ocopy trick)

When the user provides an existing H5AD or H5MU and asks kaichi to attach a crispr
modality, kaichi does **not parse the unchanged content**. It copies it byte-for-byte.

### Mechanism
HDF5's `H5Ocopy` (C API) copies any object (group, dataset, with all attributes and
internal references) to another location, potentially in another file. The Rust
`hdf5` crate exposes this.

### When kaichi uses it
| Input | Output crispr from | Pass-through |
|---|---|---|
| RNA H5AD | fresh Cell Ranger counts | Copy whole input file → `/mod/rna/` of output |
| Existing H5MU with rna only | fresh Cell Ranger counts | Copy `/mod/rna/` over |
| Existing H5MU with rna + crispr | recompute from input's crispr counts | Copy `/mod/rna/` over; replace `/mod/crispr/` |
| Existing H5MU with rna + crispr + atac | recompute crispr | Copy `/mod/rna/` and `/mod/atac/`; replace `/mod/crispr/` |

### Why this is safe
- Attributes preserved
- Compression filters preserved (if the destination file has the filter available
  — flag mismatches as errors)
- Internal references preserved (categorical encoding uses these)
- Forward-compatible with AnnData spec additions kaichi has never seen

### What's NOT safe
- External links (objects in other files referenced by symbolic link). AnnData and
  MuData don't use these in practice; verify in a test.
- Exotic compression filters (BLOSC custom plugins) — propagate filter availability
  as a clear error rather than silent failure.

---

## Read path: what kaichi extracts from inputs

### From a Cell Ranger feature-barcode matrix directory
Not HDF5 at all. Plain text MTX format. Parsed directly in Rust (small custom parser
or a crate like `sprs` if it has reasonable MTX support).

### From `molecule_info.h5`
Cell Ranger's own HDF5 schema — not AnnData encoded. Flat datasets at the root:
`barcode_idx`, `feature_idx`, `count`, etc. Parsed directly via `hdf5` crate.
anndata-rs is not involved.

### From an existing H5AD / H5MU
Two cases:

**Case A: input is the RNA side (attach guide modality)**
kaichi only needs cell barcodes for the join.
- Use anndata-rs to open the file lazily.
- Extract `obs_names` only.
- Leave the rest of the file unread; it will be `H5Ocopy`'d into the output.

**Case B: input contains existing guide counts (recompute)**
kaichi needs the count matrix and var metadata.
- Use anndata-rs to read the `crispr` modality (or root, if a guide-only H5AD).
  - Extract: `obs_names`, `var_names`, optional `var` columns (target_gene, sequence,
    etc.), `X`.
- Convert to Arrow `RecordBatch` for the compute layer.

---

## Write path: what kaichi produces

The crispr modality is the only AnnData kaichi constructs fresh. Schema is fully
defined by kaichi (see [data-model.md](data-model.md)).

Procedure:
1. Receive Arrow `RecordBatch` from the compute layer (guide call schema).
2. Construct an anndata-rs `AnnData` with:
   - `X`: sparse CSR UInt32 matrix (cells × guides)
   - `obs`: DataFrame from the Arrow batch
   - `var`: DataFrame from guide IDs plus optional guide library reference metadata
   - `uns/kaichi`: provenance dict
3. Tell anndata-rs to write at the destination HDF5 path — `/mod/crispr/` of the
   output H5MU, or `/` of a standalone H5AD.
4. Write H5MU container attrs (`mod-order`, etc.) via raw `hdf5` crate.

**Confirmed limitation:** anndata-rs only writes AnnData at the root of a new file —
`AnnData::new(filename)`, `AnnData::write(filename)` and `AnnData::open(store)` all
operate on `Backend::Store` (file root), not on a subgroup. Verified by reading
`anndata-rs/anndata/src/anndata.rs`.

**Workaround in kaichi:**
1. anndata-rs writes the crispr modality to a temp `.h5ad` at root.
2. kaichi opens the destination H5MU via the `hdf5` crate.
3. `H5Ocopy` the temp file's root group → destination's `/mod/crispr/`.
4. Delete the temp file.

The internals of anndata-rs use `GroupOp` everywhere, so adding `write_to_group` /
`open_from_group` upstream would be a small change. Worth filing as an issue once
kaichi exists; not worth blocking implementation on.

---

## Encoding decisions pinned at v0

| Decision | Choice | Reason |
|---|---|---|
| Sparse matrix encoding | CSR | AnnData spec default; matches Cell Ranger output |
| Sparse data dtype | `UInt32` for counts | Counts are non-negative integers; saves space |
| Categorical encoding | anndata-rs handles | Spec-compliant by delegation |
| String encoding | UTF-8, variable-length | Modern AnnData default |
| Compression | gzip level 4 on `.X.data`, none on small metadata | Standard for h5ad |
| Chunking | row-major, ~1MB chunks for `.X` | Reasonable for cell-wise access |
| H5MU `encoding-version` | Pin at first release; bump explicitly | Forward compatibility |

---

## In-memory bindings — not this doc's concern

When the user passes an in-memory AnnData (Python) or Seurat object (R), no HDF5 is
involved. The language binding extracts what kaichi needs as Arrow and the Rust core
returns Arrow. The on-disk path described here is bypassed.

See [binding-interop.md](binding-interop.md) for that flow.
