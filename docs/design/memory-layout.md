# kaichi — In-Memory Data Flow

This doc pins down how data is represented in memory inside `kaichi-core`, from
the moment HDF5 hands us a buffer to the moment we hand a `RecordBatch` to a
binding or write it back to disk. It is the in-memory companion to
[storage-encoding.md](storage-encoding.md).

---

## Core principle

**One buffer per piece of data. Multiple typed views over it.**

A "buffer" is a single contiguous allocation produced once at read time. A "view"
is a typed wrapper that interprets that buffer for a specific access pattern
(CSR, CSC, dense array, Arrow primitive array, Arrow record batch column). Views
borrow; they do not copy.

The pipeline is allowed to introduce a new buffer only when one of these is true:

1. **A type promotion is required** that changes the byte width (e.g., `u32` counts
   → `f64` log-counts for the math kernel).
2. **A structural rearrangement is required** that re-orders entries (e.g., CSR →
   CSC transpose).
3. **An output is being constructed** whose values are computed, not copied
   (posteriors, assignment confidences).

Anything else — string slicing, type-tag relabeling, view construction — must be
zero-copy. Any conversion that does not satisfy one of the three rules above is a
bug.

---

## Canonical types

These are the only data structures `kaichi-core` carries across module
boundaries. Anything else is local to a single function.

### `CountMatrix` — sparse counts, cells × guides

```rust
pub struct CountMatrix {
    pub csr: CsrMatrix<u32>,      // nalgebra-sparse view; cells = rows, guides = cols
    pub csc: OnceCell<CscMatrix<u32>>,  // lazily transposed
    pub n_cells: usize,
    pub n_guides: usize,
}
```

Backed by three Arrow `PrimitiveArray`s: `data` (`UInt32`), `indices` (`Int32`),
`indptr` (`Int32`). The Arrow arrays own the buffers; the `CsrMatrix` is
constructed via `try_from_csr_data` over `&[u32]` slices into them. Lifetime
contract: the `CountMatrix` owns the Arrow arrays, so the slices are valid for
its lifetime.

`csc` is built on first access via a single `O(nnz)` pass. After that, per-guide
slicing is `O(1) + O(n_nonzero_in_guide)`.

### `Covariates` — dense per-cell metadata

```rust
pub struct Covariates {
    pub cell_barcodes: StringArray,           // length n_cells
    pub batch: DictionaryArray<Int16Type>,    // length n_cells
    pub total_counts: Float32Array,           // length n_cells
}
```

All three are Arrow primitive/dict arrays. `cell_barcodes` doubles as the index
when constructing the output `RecordBatch` — same allocation, no copy.

### `GuideMetadata` — per-guide names and optional library annotations

```rust
pub struct GuideMetadata {
    pub guide_ids: StringArray,                                // length n_guides
    pub target_gene: Option<DictionaryArray<Int16Type>>,       // optional library column
    pub sequence: Option<StringArray>,                         // optional library column
    // ... other optional columns from the guide library TSV
}
```

### `AssignmentResult` — output record

```rust
pub struct AssignmentResult {
    pub batch: RecordBatch,         // schema in data-model.md
    pub assigned_x: CsrMatrix<u8>,  // binary cells × guides, for layers["assigned"]
}
```

`batch.column("cell_barcode")` is *the same Arrow array* as the input
`Covariates::cell_barcodes`. We never re-allocate cell barcodes after reading.

---

## End-to-end flow

A walk through one `kaichi assign` invocation, naming every buffer and every
view.

### 1. Read

`io::read_h5ad` opens the file via `hdf5-metno` and produces:

```
Arrow Buffer A   ← hdf5: X/data       (uint32, length nnz)
Arrow Buffer B   ← hdf5: X/indices    (int32,  length nnz)
Arrow Buffer C   ← hdf5: X/indptr     (int32,  length n_cells + 1)
Arrow Buffer D   ← hdf5: obs/_index   (string offsets + values)
Arrow Buffer E   ← hdf5: var/_index   (string offsets + values)
```

Each HDF5 read goes into a pre-sized `arrow::buffer::MutableBuffer`, frozen to an
`arrow::buffer::Buffer`, wrapped in the corresponding `PrimitiveArray` /
`StringArray`. **One allocation per dataset. No intermediate `Vec`.**

Outputs returned to the caller:

```rust
struct LoadedInput {
    counts: CountMatrix {
        csr: CsrMatrix<u32> view over (A, B, C),
        csc: OnceCell<CscMatrix<u32>>,
        n_cells, n_guides,
    },
    covariates: Covariates {
        cell_barcodes: StringArray over D,
        batch: dict array (default, single category),
        total_counts: Float32Array computed from CSR row sums,
    },
    guide_metadata: GuideMetadata {
        guide_ids: StringArray over E,
        ..
    },
}
```

**Copies in this stage:** zero, except `total_counts` (computed Float32Array, one
buffer of length `n_cells`).

### 2. Per-guide compute

The model's `assign` method takes `&LoadedInput` and runs per-guide work in
parallel. For each guide `g`:

```rust
let (cell_indices, counts) = input.counts.csc().column(g);
//  cell_indices: &[i32]   ← slice into Buffer B (transposed)
//  counts:       &[u32]   ← slice into Buffer A (transposed)
```

Both slices are views into the transposed CSC buffers. No allocation per guide.

The math kernel runs on a **small dense f64 buffer** of length `n_nonzero_in_guide`:

```rust
let log_counts: Vec<f64> = counts.iter().map(|&c| (c as f64).log2()).collect();
// Allocation: one Vec<f64> of length n_nonzero_in_guide. This is type promotion
// (rule 1) — necessary, justified, named.
```

EM fits `FitParams { pi, lambda_bg, mu_signal, sigma_signal }` over `log_counts`.
The fit is 32 bytes. The responsibilities vector is `Vec<f64>` reused across
iterations.

**Copies in this stage (per guide):**
- `Vec<f64> log_counts`, length `n_nonzero_in_guide` (rule 1: type promotion).
- `Vec<f64> responsibilities`, same length, reused across iterations.
- `FitParams`, 32 bytes.

Total per-guide working memory: `~16 × n_nonzero_in_guide` bytes plus
constant overhead. **Never `n_cells`**.

### 3. Per-cell threshold / assignment

After all guides are fit, we have for each guide a `FitParams` and a closed-form
threshold (the lowest integer count whose posterior signal probability exceeds
`min_confidence`).

To build the output record:

```rust
for cell in 0..n_cells {
    let row = input.counts.csr.row(cell);
    // row.values(): &[u32] of nonzero counts for this cell
    // row.col_indices(): &[i32] of guide indices for this cell
    
    for (guide_idx, count) in row.iter() {
        if count >= thresholds[guide_idx] {
            // record as a passing guide
        }
    }
}
```

Each cell iteration is `O(n_guides_with_nonzero_count_for_this_cell)`. Total
work: `O(nnz)`. No densification.

### 4. Result construction

The output `RecordBatch` is built from arrays we already have plus computed
arrays:

```rust
RecordBatch::try_new(schema, vec![
    Arc::new(covariates.cell_barcodes.clone()),  // Arrow Arc clone — zero copy
    Arc::new(assigned_guide_dict),                // computed: DictionaryArray
    Arc::new(target_gene_dict),                   // computed from guide_metadata
    Arc::new(umi_count_array),                    // computed: UInt32Array
    Arc::new(confidence_array),                   // computed: Float32Array
    Arc::new(model_name_dict),                    // computed: DictionaryArray, single category
    Arc::new(is_unassigned_array),                // computed: BooleanArray
    Arc::new(is_multi_array),                     // computed: BooleanArray
    Arc::new(n_detected_array),                   // computed: UInt8Array
])
```

`Arc::clone` on an Arrow array is a refcount bump. The cell-barcode buffer
allocated at read time is the same buffer that ends up in the output column.

The `assigned_x` CSR (binary layer) is built once from the same passing-cell
loop as in step 3.

### 5. Write

`io::write_h5ad` walks the `AssignmentResult`:

- `X` ← input `CountMatrix.csr` written via `hdf5-metno`. The Arrow buffers A,
  B, C from step 1 are handed directly to HDF5 for write. **Same buffers as
  read.**
- `layers["assigned"]` ← `assigned_x` (built fresh in step 4).
- `obs` ← columns from `AssignmentResult.batch` written via the AnnData spec
  writer.
- `var` ← `GuideMetadata` columns.
- `uns/kaichi` ← run provenance.

**Copies in this stage:** zero for `X`, one per result column (HDF5 needs its
own buffer for the dataset). HDF5 writes are inherently copies; we can't avoid
that boundary.

---

## Buffer ownership table

| Buffer | Allocated by | Owns | Lifetime |
|---|---|---|---|
| `X/data` | HDF5 read | `arrow::Buffer` inside `UInt32Array` | Read → Write |
| `X/indices` | HDF5 read | `arrow::Buffer` inside `Int32Array` | Read → Write |
| `X/indptr` | HDF5 read | `arrow::Buffer` inside `Int32Array` | Read → Write |
| `obs/_index` (cell barcodes) | HDF5 read | `arrow::Buffer` inside `StringArray` | Read → output `RecordBatch` |
| `var/_index` (guide IDs) | HDF5 read | `arrow::Buffer` inside `StringArray` | Read → output `RecordBatch` |
| `total_counts` | computed at read time | `Float32Array` | Read → covariates only |
| CSC transposed buffers | computed on first access | `arrow::Buffer` pair (indices, indptr) | First CSC access → end of compute |
| `log_counts` per guide | per-guide math kernel | `Vec<f64>` | Single guide fit |
| `responsibilities` per guide | per-guide math kernel | `Vec<f64>` | Single guide fit |
| Output `confidence`, `umi_count`, ... | result construction | Arrow primitive arrays | Result → write |
| `assigned_x` (binary CSR) | step 3 passing-cell loop | three Arrow buffers | Result → write |

---

## What we don't do

These are the conversions present in the pre-buffer-views implementation that
this design eliminates:

1. **Long-form Arrow intermediate** (`RecordBatch` of `(cell_barcode, guide_id,
   umi_count)` triples). It was a re-encoding of CSR that threw away the
   structural index — every consumer had to re-bucket it.
2. **`HashMap<String, Vec<u32>>` of length `n_cells` per guide.** This
   re-densified the data, blew memory to `n_cells × n_guides_observed × 4`
   bytes, and re-built the structural index CSR already had.
3. **Re-allocating cell barcodes for the output `RecordBatch`.** Cell barcodes
   are read once and used as both the covariate index and the output index via
   `Arc::clone`.
4. **`Vec<u32>` of zero-padded counts in the model kernel.** Models work on
   nonzero entries (`&[u32]`), not on densified vectors. Cells with no count for
   a guide are excluded by construction, not by filtering after the fact.

---

## Unavoidable copies, named

A list of every place a new buffer is allocated, and why each one satisfies one
of the three rules in the core principle.

| Copy | Size | Rule |
|---|---|---|
| `total_counts` from CSR row sums | `n_cells × 4 bytes` | Rule 3: computed output, not present on disk |
| CSC transpose | `(nnz × 4) + (n_guides × 4)` bytes | Rule 2: structural rearrangement for per-guide slicing |
| `log_counts` per guide | `n_nonzero × 8 bytes` | Rule 1: type promotion `u32 → f64` for log/exp math |
| `responsibilities` per guide | `n_nonzero × 8 bytes` | Rule 1: computed posterior weights, reused across EM iterations |
| Per-guide `FitParams` | 32 bytes | Computed parameters |
| Output `confidence`, `umi_count`, dictionaries | `n_cells × (4 + 4 + ...)` bytes | Rule 3: computed outputs |
| `assigned_x` CSR | `n_assigned × (1 + 4 + 4)` bytes | Rule 3: computed binary layer |
| HDF5 write buffers | per dataset | Output boundary; HDF5 owns its own memory |

Note `n_nonzero` here is per-guide, not the total `nnz`. Peak per-guide working
memory is bounded by the largest single guide, not by the matrix.

---

## Binding integration

The above flow produces an `AssignmentResult` whose `batch: RecordBatch` is the
shared currency with both bindings. See
[binding-interop.md](binding-interop.md) for how this batch crosses the FFI
boundary.

Two notes specific to the buffer-views model:

1. **Zero-copy holds at the FFI boundary.** Arrow's C Data Interface passes
   buffer pointers. Because the `RecordBatch` columns *are* the buffers
   allocated during read and compute, the Python side receives the same memory
   the Rust core operated on. No serialization step.
2. **The CLI path bypasses the binding boundary entirely.** `kaichi assign`
   builds the `RecordBatch`, hands it to the HDF5 writer in the same process,
   and the writer reads off the Arrow buffers directly. The writer does not need
   to know whether the buffers came from a CLI read or a binding-supplied
   input — same code path, same buffer layout.

---

## Implementation cross-references

| Concern | Module |
|---|---|
| HDF5 → Arrow buffers (read) | `kaichi-core/src/io/read.rs` |
| Arrow buffers → HDF5 (write) | `kaichi-core/src/io/write.rs` |
| `CountMatrix`, `Covariates`, `GuideMetadata` definitions | `kaichi-core/src/data.rs` |
| Per-guide compute pattern | `kaichi-core/src/models/*.rs` |
| Output `RecordBatch` schema | `kaichi-core/src/schema.rs` |

This is the contract the implementation should match. Where the current code
diverges (long-form intermediate, HashMap densification — see
[storage-encoding.md](storage-encoding.md) and `models/poisson_gauss.rs::ParsedInput`),
the divergence is a refactor target, not a design choice.
