use crate::data::{BatchLabels, CountMatrix, Covariates, GuideMetadata, LoadedInput};
use crate::score::{ModelKind, ScoreMatrix};
use anyhow::{anyhow, bail, Context, Result};
use arrow::array::{Float32Array, StringArray, StringBuilder, UInt32Array};
use hdf5::{types::VarLenUnicode, File};
use std::path::Path;

/// Read a guide-count H5AD into a `LoadedInput`.
///
/// Expects CSR-encoded `/X` (cells × guides). The index column name is read
/// from the `_index` attribute on `/obs` and `/var`.
pub fn read_h5ad(path: &Path) -> Result<LoadedInput> {
    let file = File::open(path)
        .with_context(|| format!("cannot open H5AD: {}", path.display()))?;

    let cell_barcodes_raw = read_index(&file, "obs")
        .context("reading obs index")?;
    let guide_ids_raw = read_index(&file, "var")
        .context("reading var index")?;

    let n_cells = cell_barcodes_raw.len();
    let n_guides = guide_ids_raw.len();

    let (row_offsets, col_indices, values) = read_csr_x(&file, n_cells)
        .context("reading X matrix")?;

    // total_counts: row sums, computed before moving values into CountMatrix.
    let mut total_counts_vec = vec![0f32; n_cells];
    for (cell, window) in row_offsets.windows(2).enumerate() {
        for &v in &values[window[0]..window[1]] {
            total_counts_vec[cell] += v as f32;
        }
    }

    let counts = CountMatrix::try_from_csr(n_cells, n_guides, row_offsets, col_indices, values)?;

    let mut barcode_builder = StringBuilder::with_capacity(n_cells, n_cells * 24);
    for s in &cell_barcodes_raw {
        barcode_builder.append_value(s.as_str());
    }

    let mut guide_builder = StringBuilder::with_capacity(n_guides, n_guides * 16);
    for s in &guide_ids_raw {
        guide_builder.append_value(s.as_str());
    }

    let batch = read_batch_or_default(&file, n_cells)
        .context("reading obs/batch")?;

    Ok(LoadedInput {
        counts,
        covariates: Covariates {
            cell_barcodes: barcode_builder.finish(),
            total_counts: arrow::array::Float32Array::from(total_counts_vec),
            batch,
        },
        guide_metadata: GuideMetadata {
            guide_ids: guide_builder.finish(),
        },
    })
}

/// Read a "scored" H5AD (produced by `write_scored_h5ad`) back into a
/// `ScoreMatrix`.
///
/// Expects:
/// - `/X`              CSR uint32 (UMI counts)
/// - `/layers/scores`  CSR float32 (posteriors, same indices/indptr as X)
/// - `/uns/kaichi/model`        model name (round-trip identity)
/// - `/uns/kaichi/model_params` JSON blob (fitted params)
///
/// `uns/kaichi/stage` is *not* checked — by user request, `decide` tolerates
/// any H5AD that has a `scores` layer.
pub fn read_scored_h5ad(path: &Path) -> Result<ScoreMatrix> {
    let file = File::open(path)
        .with_context(|| format!("cannot open H5AD: {}", path.display()))?;

    let cell_barcodes_raw = read_index(&file, "obs").context("reading obs index")?;
    let guide_ids_raw = read_index(&file, "var").context("reading var index")?;
    let n_cells = cell_barcodes_raw.len();

    let (x_indptr, x_indices, umi_values) = read_csr_x(&file, n_cells)
        .context("reading X (umi counts)")?;

    // Read float32 scores from /layers/scores.
    let scores_grp = file.group("layers/scores")
        .context("missing /layers/scores — file is not a kaichi-scored h5ad")?;
    let s_indices: Vec<i32> = scores_grp.dataset("indices")
        .context("missing layers/scores/indices")?
        .read_raw::<i32>().context("reading layers/scores/indices")?;
    let s_indptr: Vec<i32> = scores_grp.dataset("indptr")
        .context("missing layers/scores/indptr")?
        .read_raw::<i32>().context("reading layers/scores/indptr")?;
    let s_data: Vec<f32> = scores_grp.dataset("data")
        .context("missing layers/scores/data")?
        .read_raw::<f32>().context("reading layers/scores/data")?;

    if s_indptr.len() != x_indptr.len()
        || s_indices.len() != x_indices.len()
        || s_data.len() != umi_values.len()
    {
        bail!(
            "scores layer shape mismatch: X has nnz={}, indptr len={}; \
             scores has nnz={}, indptr len={}",
            umi_values.len(), x_indptr.len(),
            s_data.len(), s_indptr.len()
        );
    }

    // Pull model name + params from uns/kaichi.
    let kaichi = file.group("uns/kaichi")
        .context("missing /uns/kaichi — file was not written by kaichi")?;
    let model_name: VarLenUnicode = kaichi.dataset("model")
        .context("missing uns/kaichi/model")?
        .read_raw::<VarLenUnicode>()?
        .into_iter().next()
        .ok_or_else(|| anyhow!("uns/kaichi/model is empty"))?;
    let params_str: VarLenUnicode = kaichi.dataset("model_params")
        .context("missing uns/kaichi/model_params")?
        .read_raw::<VarLenUnicode>()?
        .into_iter().next()
        .ok_or_else(|| anyhow!("uns/kaichi/model_params is empty"))?;

    let model = ModelKind::from_name(model_name.as_str())
        .ok_or_else(|| anyhow!("uns/kaichi/model = {:?} is not a known kaichi model", model_name.as_str()))?;
    let model_params: serde_json::Value = serde_json::from_str(params_str.as_str())
        .with_context(|| format!("uns/kaichi/model_params is not valid JSON: {:?}", params_str.as_str()))?;

    let cell_barcodes = string_array_from_vlu(&cell_barcodes_raw);
    let guide_ids = string_array_from_vlu(&guide_ids_raw);

    Ok(ScoreMatrix {
        data: Float32Array::from(s_data),
        umi_counts: UInt32Array::from(umi_values),
        indices: UInt32Array::from(x_indices.into_iter().map(|i| i as u32).collect::<Vec<u32>>()),
        indptr: UInt32Array::from(x_indptr.into_iter().map(|i| i as u32).collect::<Vec<u32>>()),
        cell_barcodes,
        guide_ids,
        model,
        model_params,
    })
}

fn string_array_from_vlu(v: &[VarLenUnicode]) -> StringArray {
    let mut b = StringBuilder::with_capacity(v.len(), v.iter().map(|s| s.len()).sum());
    for s in v { b.append_value(s.as_str()); }
    b.finish()
}

/// Read `obs/batch` if present; otherwise return a single-batch fallback.
///
/// AnnData stores categoricals as a group with `codes` (integer) and
/// `categories` (string) datasets. We also handle the case where `obs/batch`
/// is a plain string dataset (older fixtures sometimes write it that way).
fn read_batch_or_default(file: &File, n_cells: usize) -> Result<BatchLabels> {
    let obs = file.group("obs").context("missing /obs")?;
    if !obs.link_exists("batch") {
        return Ok(BatchLabels::single_batch(n_cells));
    }

    // Modern AnnData categorical: group with codes + categories children.
    if let Ok(grp) = obs.group("batch") {
        if grp.link_exists("codes") && grp.link_exists("categories") {
            let codes_raw: Vec<i64> = read_int_dataset(&grp, "codes")
                .context("reading obs/batch/codes")?;
            let cats: Vec<VarLenUnicode> = grp
                .dataset("categories")?
                .read_raw::<VarLenUnicode>()
                .context("reading obs/batch/categories")?;
            let categories: Vec<String> = cats.into_iter().map(|s| s.to_string()).collect();
            let n_cats = categories.len();
            let mut codes = Vec::with_capacity(codes_raw.len());
            for (i, c) in codes_raw.iter().enumerate() {
                if *c < 0 {
                    bail!(
                        "obs/batch/codes[{i}] = {c}: AnnData uses negative codes for missing values; \
                         kaichi cannot fit hierarchical models without a batch label for every cell"
                    );
                }
                if *c >= u16::MAX as i64 {
                    bail!(
                        "obs/batch/codes[{i}] = {c}: too many batches (kaichi supports up to {} \
                         distinct batches per dataset)", u16::MAX
                    );
                }
                if (*c as usize) >= n_cats {
                    bail!(
                        "obs/batch/codes[{i}] = {c}: code out of range \
                         for {n_cats} declared categories"
                    );
                }
                codes.push(*c as u16);
            }
            return Ok(BatchLabels { codes, categories });
        }
    }

    // Fallback: obs/batch is a plain string dataset.
    if let Ok(ds) = obs.dataset("batch") {
        if let Ok(raw) = ds.read_raw::<VarLenUnicode>() {
            let labels: Vec<String> = raw.into_iter().map(|s| s.to_string()).collect();
            return Ok(build_categorical(labels));
        }
    }

    // Couldn't decode any known form — fall back to single batch rather than fail.
    Ok(BatchLabels::single_batch(n_cells))
}

/// Read an integer dataset, trying common dtypes. Returns i64 so any input
/// dtype can be widened losslessly.
fn read_int_dataset(grp: &hdf5::Group, name: &str) -> Result<Vec<i64>> {
    let ds = grp.dataset(name)?;
    if let Ok(v) = ds.read_raw::<i64>() { return Ok(v); }
    if let Ok(v) = ds.read_raw::<i32>() { return Ok(v.into_iter().map(|x| x as i64).collect()); }
    if let Ok(v) = ds.read_raw::<i16>() { return Ok(v.into_iter().map(|x| x as i64).collect()); }
    if let Ok(v) = ds.read_raw::<i8>()  { return Ok(v.into_iter().map(|x| x as i64).collect()); }
    if let Ok(v) = ds.read_raw::<u32>() { return Ok(v.into_iter().map(|x| x as i64).collect()); }
    if let Ok(v) = ds.read_raw::<u16>() { return Ok(v.into_iter().map(|x| x as i64).collect()); }
    if let Ok(v) = ds.read_raw::<u8>()  { return Ok(v.into_iter().map(|x| x as i64).collect()); }
    bail!("dataset {name}: unsupported integer dtype")
}

/// Build a `BatchLabels` from a per-cell vector of string labels by
/// constructing the category list in first-appearance order.
fn build_categorical(labels: Vec<String>) -> BatchLabels {
    use std::collections::HashMap;
    let mut categories: Vec<String> = Vec::new();
    let mut index: HashMap<String, u16> = HashMap::new();
    let mut codes: Vec<u16> = Vec::with_capacity(labels.len());
    for s in labels {
        let code = match index.get(&s) {
            Some(&k) => k,
            None => {
                let k = categories.len() as u16;
                categories.push(s.clone());
                index.insert(s, k);
                k
            }
        };
        codes.push(code);
    }
    BatchLabels { codes, categories }
}

/// Read the index column from a dataframe group (obs or var).
///
/// The `_index` attribute on the group names which child dataset is the index.
fn read_index(file: &File, group_name: &str) -> Result<Vec<VarLenUnicode>> {
    let group = file
        .group(group_name)
        .with_context(|| format!("missing /{group_name} group"))?;

    let idx_col: VarLenUnicode = group
        .attr("_index")
        .with_context(|| format!("/{group_name} missing _index attribute"))?
        .read_scalar::<VarLenUnicode>()
        .with_context(|| format!("cannot read /{group_name}/_index attribute"))?;

    let ds_path = format!("{group_name}/{}", idx_col.as_str());
    file.dataset(&ds_path)
        .with_context(|| format!("missing index dataset: /{ds_path}"))?
        .read_raw::<VarLenUnicode>()
        .with_context(|| format!("cannot read index dataset: /{ds_path}"))
}

/// Read `/X` as a CSR matrix and return `(row_offsets, col_indices, values)`.
///
/// Values are cast to `u32` regardless of the on-disk integer type.
fn read_csr_x(file: &File, n_cells: usize) -> Result<(Vec<usize>, Vec<usize>, Vec<u32>)> {
    let x_grp = file.group("X").context("missing /X group")?;

    let indices: Vec<i32> = x_grp
        .dataset("indices")
        .context("missing X/indices")?
        .read_raw::<i32>()
        .context("reading X/indices")?;

    let indptr: Vec<i32> = x_grp
        .dataset("indptr")
        .context("missing X/indptr")?
        .read_raw::<i32>()
        .context("reading X/indptr")?;

    if indptr.len() != n_cells + 1 {
        bail!(
            "X/indptr length {} != n_cells+1 ({n_cells}+1)",
            indptr.len()
        );
    }

    let data_ds = x_grp.dataset("data").context("missing X/data")?;
    let values = read_as_u32(&data_ds).context("reading X/data")?;

    let col_indices: Vec<usize> = indices.into_iter().map(|i| i as usize).collect();
    let row_offsets: Vec<usize> = indptr.into_iter().map(|i| i as usize).collect();

    Ok((row_offsets, col_indices, values))
}

/// Read a dataset of any integer type and return `Vec<u32>`.
///
/// HDF5 soft-converts smaller types to i64 transparently. All UMI counts
/// fit in u32 so the cast is always safe for this domain.
fn read_as_u32(ds: &hdf5::Dataset) -> Result<Vec<u32>> {
    // Try u32 first (most common for current anndata files).
    // Fall back to i64 (crispat / older anndata uses int64 for X).
    // Fall back to i32, u16, u8 for other encodings.
    if let Ok(v) = ds.read_raw::<u32>() {
        return Ok(v);
    }
    if let Ok(v) = ds.read_raw::<i64>() {
        return Ok(v.into_iter().map(|x| x as u32).collect());
    }
    if let Ok(v) = ds.read_raw::<i32>() {
        return Ok(v.into_iter().map(|x| x as u32).collect());
    }
    if let Ok(v) = ds.read_raw::<u16>() {
        return Ok(v.into_iter().map(|x| x as u32).collect());
    }
    if let Ok(v) = ds.read_raw::<u8>() {
        return Ok(v.into_iter().map(|x| x as u32).collect());
    }
    bail!("X/data: unsupported dtype (tried u32, i64, i32, u16, u8)")
}

// ---------------------------------------------------------------------------
// Helpers for tests: write a minimal H5AD so we can test the reader in-process
// without depending on Python.
// ---------------------------------------------------------------------------

pub mod testutil {
    use hdf5::{types::VarLenUnicode, File};
    use ndarray::{arr1, Array1};
    use std::str::FromStr;

    fn vlu(s: &str) -> VarLenUnicode {
        VarLenUnicode::from_str(s).unwrap()
    }

    fn write_str_scalar_attr(loc: &hdf5::Location, name: &str, val: &str) {
        let arr = ndarray::arr0(vlu(val));
        loc.new_attr_builder().with_data(&arr).create(name).unwrap();
    }

    fn write_str_array_attr(loc: &hdf5::Location, name: &str, vals: &[&str]) {
        if vals.is_empty() {
            // Write an empty float64 array — matches what anndata Python writes for empty column-order
            let empty: Array1<f64> = Array1::zeros(0);
            loc.new_attr_builder().with_data(&empty).create(name).unwrap();
        } else {
            let arr: Array1<VarLenUnicode> = vals.iter().map(|s| vlu(s)).collect();
            loc.new_attr_builder().with_data(&arr).create(name).unwrap();
        }
    }

    fn write_encoding_attrs(loc: &hdf5::Location, enc_type: &str, enc_ver: &str) {
        write_str_scalar_attr(loc, "encoding-type", enc_type);
        write_str_scalar_attr(loc, "encoding-version", enc_ver);
    }

    fn write_str_dataset(grp: &hdf5::Group, name: &str, vals: &[&str]) {
        let arr: Array1<VarLenUnicode> = vals.iter().map(|s| vlu(s)).collect();
        grp.new_dataset_builder().with_data(&arr).create(name).unwrap();
    }

    /// Write a minimal guide-only H5AD to `path`.
    ///
    /// `x_rows`: (cell_idx, guide_idx, count) triples (sorted).
    pub fn write_test_h5ad(
        path: &std::path::Path,
        cell_barcodes: &[&str],
        guide_ids: &[&str],
        x_rows: &[(usize, usize, u32)],
    ) {
        let n_cells = cell_barcodes.len();
        let n_guides = guide_ids.len();

        // Build CSR from triples (assumed sorted by (cell, guide)).
        let mut data: Vec<i64> = Vec::new();
        let mut indices: Vec<i32> = Vec::new();
        let mut indptr: Vec<i32> = vec![0i32; n_cells + 1];
        for &(ci, gi, v) in x_rows {
            data.push(v as i64);
            indices.push(gi as i32);
            indptr[ci + 1] += 1;
        }
        // prefix-sum indptr
        for i in 1..=n_cells {
            indptr[i] += indptr[i - 1];
        }

        let file = File::create(path).unwrap();

        // root attrs
        write_encoding_attrs(&file, "anndata", "0.1.0");

        // /obs
        let obs = file.create_group("obs").unwrap();
        write_encoding_attrs(&obs, "dataframe", "0.2.0");
        write_str_scalar_attr(&obs, "_index", "_index");
        write_str_array_attr(&obs, "column-order", &[]);
        write_str_dataset(&obs, "_index", cell_barcodes);

        // /var
        let var = file.create_group("var").unwrap();
        write_encoding_attrs(&var, "dataframe", "0.2.0");
        write_str_scalar_attr(&var, "_index", "_index");
        write_str_array_attr(&var, "column-order", &[]);
        write_str_dataset(&var, "_index", guide_ids);

        // /X
        let x_grp = file.create_group("X").unwrap();
        write_encoding_attrs(&x_grp, "csr_matrix", "0.1.0");
        let shape_arr = arr1(&[n_cells as i64, n_guides as i64]);
        x_grp.new_attr_builder().with_data(&shape_arr).create("shape").unwrap();

        let data_arr: Array1<i64> = Array1::from_vec(data);
        let indices_arr: Array1<i32> = Array1::from_vec(indices);
        let indptr_arr: Array1<i32> = Array1::from_vec(indptr);
        x_grp.new_dataset_builder().with_data(&data_arr).create("data").unwrap();
        x_grp.new_dataset_builder().with_data(&indices_arr).create("indices").unwrap();
        x_grp.new_dataset_builder().with_data(&indptr_arr).create("indptr").unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::testutil::write_test_h5ad;
    use super::read_h5ad;
    use tempfile::NamedTempFile;

    fn tmp_h5ad() -> NamedTempFile {
        NamedTempFile::with_suffix(".h5ad").unwrap()
    }

    fn make_test_input() -> tempfile::NamedTempFile {
        let tmp = tmp_h5ad();
        write_test_h5ad(
            tmp.path(),
            &["C1", "C2", "C3"],
            &["gA", "gB"],
            &[(0, 1, 5), (1, 0, 3), (1, 1, 1)],
        );
        tmp
    }

    #[test]
    fn read_h5ad_dimensions() {
        let tmp = make_test_input();
        let loaded = read_h5ad(tmp.path()).unwrap();
        assert_eq!(loaded.counts.n_cells, 3);
        assert_eq!(loaded.counts.n_guides, 2);
    }

    #[test]
    fn read_h5ad_nnz() {
        let tmp = make_test_input();
        let loaded = read_h5ad(tmp.path()).unwrap();
        assert_eq!(loaded.counts.nnz(), 3);
    }

    #[test]
    fn read_h5ad_cell_barcodes_order() {
        let tmp = make_test_input();
        let loaded = read_h5ad(tmp.path()).unwrap();
        let barcodes = &loaded.covariates.cell_barcodes;
        assert_eq!(barcodes.value(0), "C1");
        assert_eq!(barcodes.value(1), "C2");
        assert_eq!(barcodes.value(2), "C3");
    }

    #[test]
    fn read_h5ad_guide_ids_order() {
        let tmp = make_test_input();
        let loaded = read_h5ad(tmp.path()).unwrap();
        let guides = &loaded.guide_metadata.guide_ids;
        assert_eq!(guides.value(0), "gA");
        assert_eq!(guides.value(1), "gB");
    }

    #[test]
    fn read_h5ad_x_values_correct() {
        let tmp = make_test_input();
        let loaded = read_h5ad(tmp.path()).unwrap();
        // cell 0: guide 1 = 5
        let row0 = loaded.counts.csr().get_row(0).unwrap();
        assert_eq!(row0.col_indices(), &[1]);
        assert_eq!(row0.values(), &[5]);
        // cell 1: guide 0 = 3, guide 1 = 1
        let row1 = loaded.counts.csr().get_row(1).unwrap();
        assert_eq!(row1.col_indices(), &[0, 1]);
        assert_eq!(row1.values(), &[3, 1]);
        // cell 2: empty
        assert_eq!(loaded.counts.csr().get_row(2).unwrap().nnz(), 0);
    }

    #[test]
    fn read_h5ad_total_counts_correct() {
        let tmp = make_test_input();
        let loaded = read_h5ad(tmp.path()).unwrap();
        // C1: 5, C2: 3+1=4, C3: 0
        assert_eq!(loaded.covariates.total_counts.value(0), 5.0);
        assert_eq!(loaded.covariates.total_counts.value(1), 4.0);
        assert_eq!(loaded.covariates.total_counts.value(2), 0.0);
    }

    #[test]
    fn read_h5ad_var_index_from_named_column() {
        // Write a file where var _index attr points to a column named "gRNA" (not "_index"),
        // matching the crispat Schraivogel fixture format.
        let tmp = tmp_h5ad();
        use hdf5::{types::VarLenUnicode, File};
        use ndarray::Array1;
        use std::str::FromStr;
        {
            let file = File::create(tmp.path()).unwrap();
            let vlu = |s: &str| VarLenUnicode::from_str(s).unwrap();
            let write_scalar = |loc: &hdf5::Location, name: &str, val: &str| {
                let arr = ndarray::arr0(vlu(val));
                loc.new_attr_builder().with_data(&arr).create(name).unwrap();
            };
            let empty: Array1<f64> = Array1::zeros(0);

            write_scalar(&file, "encoding-type", "anndata");
            write_scalar(&file, "encoding-version", "0.1.0");

            let obs = file.create_group("obs").unwrap();
            write_scalar(&obs, "encoding-type", "dataframe");
            write_scalar(&obs, "encoding-version", "0.2.0");
            write_scalar(&obs, "_index", "_index");
            obs.new_attr_builder().with_data(&empty).create("column-order").unwrap();
            let bcs: Array1<VarLenUnicode> = ["C1", "C2"].iter().map(|s| vlu(s)).collect();
            obs.new_dataset_builder().with_data(&bcs).create("_index").unwrap();

            let var = file.create_group("var").unwrap();
            write_scalar(&var, "encoding-type", "dataframe");
            write_scalar(&var, "encoding-version", "0.2.0");
            // _index attr points to dataset named "gRNA"
            write_scalar(&var, "_index", "gRNA");
            var.new_attr_builder().with_data(&empty).create("column-order").unwrap();
            let guides: Array1<VarLenUnicode> = ["gA", "gB"].iter().map(|s| vlu(s)).collect();
            var.new_dataset_builder().with_data(&guides).create("gRNA").unwrap();

            let x_grp = file.create_group("X").unwrap();
            write_scalar(&x_grp, "encoding-type", "csr_matrix");
            write_scalar(&x_grp, "encoding-version", "0.1.0");
            let shape = ndarray::arr1(&[2i64, 2i64]);
            x_grp.new_attr_builder().with_data(&shape).create("shape").unwrap();
            let data: Array1<i64> = Array1::from_vec(vec![7]);
            let idxs: Array1<i32> = Array1::from_vec(vec![0]);
            let iptr: Array1<i32> = Array1::from_vec(vec![0, 1, 1]);
            x_grp.new_dataset_builder().with_data(&data).create("data").unwrap();
            x_grp.new_dataset_builder().with_data(&idxs).create("indices").unwrap();
            x_grp.new_dataset_builder().with_data(&iptr).create("indptr").unwrap();
        }

        let loaded = read_h5ad(tmp.path()).unwrap();
        assert_eq!(loaded.guide_metadata.guide_ids.value(0), "gA");
        assert_eq!(loaded.guide_metadata.guide_ids.value(1), "gB");
        assert_eq!(loaded.counts.csr().get_row(0).unwrap().values(), &[7]);
    }

    #[test]
    fn read_h5ad_missing_file_returns_err() {
        let result = read_h5ad(std::path::Path::new("/nonexistent/path.h5ad"));
        assert!(result.is_err());
    }

    #[test]
    fn read_h5ad_defaults_to_single_batch_when_obs_batch_absent() {
        // make_test_input doesn't write obs/batch — reader should fall back to single batch.
        let tmp = make_test_input();
        let loaded = read_h5ad(tmp.path()).unwrap();
        let b = &loaded.covariates.batch;
        assert_eq!(b.n_batches(), 1);
        assert_eq!(b.codes, vec![0u16; 3]);
        assert_eq!(b.categories, vec!["0".to_string()]);
    }

    #[test]
    fn read_h5ad_reads_categorical_batch() {
        // Modern AnnData layout: obs/batch is a group with codes + categories children.
        let tmp = tmp_h5ad();
        use hdf5::{types::VarLenUnicode, File};
        use ndarray::Array1;
        use std::str::FromStr;
        {
            let file = File::create(tmp.path()).unwrap();
            let vlu = |s: &str| VarLenUnicode::from_str(s).unwrap();
            let scalar = |loc: &hdf5::Location, name: &str, val: &str| {
                let arr = ndarray::arr0(vlu(val));
                loc.new_attr_builder().with_data(&arr).create(name).unwrap();
            };
            let empty: Array1<f64> = Array1::zeros(0);

            scalar(&file, "encoding-type", "anndata");
            scalar(&file, "encoding-version", "0.1.0");

            let obs = file.create_group("obs").unwrap();
            scalar(&obs, "encoding-type", "dataframe");
            scalar(&obs, "encoding-version", "0.2.0");
            scalar(&obs, "_index", "_index");
            obs.new_attr_builder().with_data(&empty).create("column-order").unwrap();
            let bcs: Array1<VarLenUnicode> = ["C1", "C2", "C3", "C4"].iter().map(|s| vlu(s)).collect();
            obs.new_dataset_builder().with_data(&bcs).create("_index").unwrap();

            // obs/batch as categorical group: codes [0, 1, 0, 1], categories ["A", "B"].
            let batch_grp = obs.create_group("batch").unwrap();
            scalar(&batch_grp, "encoding-type", "categorical");
            scalar(&batch_grp, "encoding-version", "0.2.0");
            let codes_arr: Array1<i32> = Array1::from_vec(vec![0, 1, 0, 1]);
            batch_grp.new_dataset_builder().with_data(&codes_arr).create("codes").unwrap();
            let cats: Array1<VarLenUnicode> = ["A", "B"].iter().map(|s| vlu(s)).collect();
            batch_grp.new_dataset_builder().with_data(&cats).create("categories").unwrap();

            let var = file.create_group("var").unwrap();
            scalar(&var, "encoding-type", "dataframe");
            scalar(&var, "encoding-version", "0.2.0");
            scalar(&var, "_index", "_index");
            var.new_attr_builder().with_data(&empty).create("column-order").unwrap();
            let guides: Array1<VarLenUnicode> = ["gA"].iter().map(|s| vlu(s)).collect();
            var.new_dataset_builder().with_data(&guides).create("_index").unwrap();

            let x_grp = file.create_group("X").unwrap();
            scalar(&x_grp, "encoding-type", "csr_matrix");
            scalar(&x_grp, "encoding-version", "0.1.0");
            let shape = ndarray::arr1(&[4i64, 1i64]);
            x_grp.new_attr_builder().with_data(&shape).create("shape").unwrap();
            let data: Array1<i64> = Array1::from_vec(vec![1, 1, 1, 1]);
            let idxs: Array1<i32> = Array1::from_vec(vec![0, 0, 0, 0]);
            let iptr: Array1<i32> = Array1::from_vec(vec![0, 1, 2, 3, 4]);
            x_grp.new_dataset_builder().with_data(&data).create("data").unwrap();
            x_grp.new_dataset_builder().with_data(&idxs).create("indices").unwrap();
            x_grp.new_dataset_builder().with_data(&iptr).create("indptr").unwrap();
        }

        let loaded = read_h5ad(tmp.path()).unwrap();
        let b = &loaded.covariates.batch;
        assert_eq!(b.codes, vec![0u16, 1, 0, 1]);
        assert_eq!(b.categories, vec!["A".to_string(), "B".to_string()]);
        assert_eq!(b.n_batches(), 2);
    }

    #[test]
    fn read_h5ad_reads_plain_string_batch() {
        // Older fixtures: obs/batch is a plain string dataset.
        let tmp = tmp_h5ad();
        use hdf5::{types::VarLenUnicode, File};
        use ndarray::Array1;
        use std::str::FromStr;
        {
            let file = File::create(tmp.path()).unwrap();
            let vlu = |s: &str| VarLenUnicode::from_str(s).unwrap();
            let scalar = |loc: &hdf5::Location, name: &str, val: &str| {
                let arr = ndarray::arr0(vlu(val));
                loc.new_attr_builder().with_data(&arr).create(name).unwrap();
            };
            let empty: Array1<f64> = Array1::zeros(0);

            scalar(&file, "encoding-type", "anndata");
            scalar(&file, "encoding-version", "0.1.0");

            let obs = file.create_group("obs").unwrap();
            scalar(&obs, "encoding-type", "dataframe");
            scalar(&obs, "encoding-version", "0.2.0");
            scalar(&obs, "_index", "_index");
            obs.new_attr_builder().with_data(&empty).create("column-order").unwrap();
            let bcs: Array1<VarLenUnicode> = ["C1", "C2", "C3"].iter().map(|s| vlu(s)).collect();
            obs.new_dataset_builder().with_data(&bcs).create("_index").unwrap();

            // obs/batch as a plain string dataset: ["X", "Y", "X"].
            let batches: Array1<VarLenUnicode> = ["X", "Y", "X"].iter().map(|s| vlu(s)).collect();
            obs.new_dataset_builder().with_data(&batches).create("batch").unwrap();

            let var = file.create_group("var").unwrap();
            scalar(&var, "encoding-type", "dataframe");
            scalar(&var, "encoding-version", "0.2.0");
            scalar(&var, "_index", "_index");
            var.new_attr_builder().with_data(&empty).create("column-order").unwrap();
            let guides: Array1<VarLenUnicode> = ["gA"].iter().map(|s| vlu(s)).collect();
            var.new_dataset_builder().with_data(&guides).create("_index").unwrap();

            let x_grp = file.create_group("X").unwrap();
            scalar(&x_grp, "encoding-type", "csr_matrix");
            scalar(&x_grp, "encoding-version", "0.1.0");
            let shape = ndarray::arr1(&[3i64, 1i64]);
            x_grp.new_attr_builder().with_data(&shape).create("shape").unwrap();
            let data: Array1<i64> = Array1::from_vec(vec![1, 1, 1]);
            let idxs: Array1<i32> = Array1::from_vec(vec![0, 0, 0]);
            let iptr: Array1<i32> = Array1::from_vec(vec![0, 1, 2, 3]);
            x_grp.new_dataset_builder().with_data(&data).create("data").unwrap();
            x_grp.new_dataset_builder().with_data(&idxs).create("indices").unwrap();
            x_grp.new_dataset_builder().with_data(&iptr).create("indptr").unwrap();
        }

        let loaded = read_h5ad(tmp.path()).unwrap();
        let b = &loaded.covariates.batch;
        // First-appearance order: X gets code 0, Y gets code 1.
        assert_eq!(b.codes, vec![0u16, 1, 0]);
        assert_eq!(b.categories, vec!["X".to_string(), "Y".to_string()]);
    }

    /// Write a minimal h5ad with an obs/batch categorical group built from caller-supplied
    /// codes and categories. Codes can be written as any integer dtype the caller chooses
    /// — pass a fn that builds the codes dataset under the batch group.
    fn write_h5ad_with_batch_codes<F>(
        path: &std::path::Path,
        n_cells: usize,
        categories: &[&str],
        write_codes: F,
    ) where F: FnOnce(&hdf5::Group) {
        use hdf5::{types::VarLenUnicode, File};
        use ndarray::Array1;
        use std::str::FromStr;

        let file = File::create(path).unwrap();
        let vlu = |s: &str| VarLenUnicode::from_str(s).unwrap();
        let scalar = |loc: &hdf5::Location, name: &str, val: &str| {
            let arr = ndarray::arr0(vlu(val));
            loc.new_attr_builder().with_data(&arr).create(name).unwrap();
        };
        let empty: Array1<f64> = Array1::zeros(0);

        scalar(&file, "encoding-type", "anndata");
        scalar(&file, "encoding-version", "0.1.0");

        let obs = file.create_group("obs").unwrap();
        scalar(&obs, "encoding-type", "dataframe");
        scalar(&obs, "encoding-version", "0.2.0");
        scalar(&obs, "_index", "_index");
        obs.new_attr_builder().with_data(&empty).create("column-order").unwrap();
        let bcs: Array1<VarLenUnicode> =
            (0..n_cells).map(|i| vlu(&format!("C{i}"))).collect();
        obs.new_dataset_builder().with_data(&bcs).create("_index").unwrap();

        let batch_grp = obs.create_group("batch").unwrap();
        scalar(&batch_grp, "encoding-type", "categorical");
        scalar(&batch_grp, "encoding-version", "0.2.0");
        let cats: Array1<VarLenUnicode> = categories.iter().map(|s| vlu(s)).collect();
        batch_grp
            .new_dataset_builder()
            .with_data(&cats)
            .create("categories")
            .unwrap();
        write_codes(&batch_grp);

        let var = file.create_group("var").unwrap();
        scalar(&var, "encoding-type", "dataframe");
        scalar(&var, "encoding-version", "0.2.0");
        scalar(&var, "_index", "_index");
        var.new_attr_builder().with_data(&empty).create("column-order").unwrap();
        let guides: Array1<VarLenUnicode> = ["gA"].iter().map(|s| vlu(s)).collect();
        var.new_dataset_builder().with_data(&guides).create("_index").unwrap();

        let x_grp = file.create_group("X").unwrap();
        scalar(&x_grp, "encoding-type", "csr_matrix");
        scalar(&x_grp, "encoding-version", "0.1.0");
        let shape = ndarray::arr1(&[n_cells as i64, 1i64]);
        x_grp.new_attr_builder().with_data(&shape).create("shape").unwrap();
        let data: Array1<i64> = Array1::from_vec(vec![1; n_cells]);
        let idxs: Array1<i32> = Array1::from_vec(vec![0; n_cells]);
        let iptr: Array1<i32> = (0..=n_cells as i32).collect();
        x_grp.new_dataset_builder().with_data(&data).create("data").unwrap();
        x_grp.new_dataset_builder().with_data(&idxs).create("indices").unwrap();
        x_grp.new_dataset_builder().with_data(&iptr).create("indptr").unwrap();
    }

    #[test]
    fn read_h5ad_batch_codes_accept_int8_dtype() {
        // Verifies the i8 branch of `read_int_dataset` — AnnData often stores codes as i8
        // when the number of categories is small.
        let tmp = tmp_h5ad();
        write_h5ad_with_batch_codes(tmp.path(), 3, &["A", "B"], |grp| {
            let codes = ndarray::Array1::from_vec(vec![0i8, 1, 0]);
            grp.new_dataset_builder().with_data(&codes).create("codes").unwrap();
        });
        let loaded = read_h5ad(tmp.path()).unwrap();
        assert_eq!(loaded.covariates.batch.codes, vec![0u16, 1, 0]);
        assert_eq!(loaded.covariates.batch.categories, vec!["A".to_string(), "B".to_string()]);
    }

    #[test]
    fn read_h5ad_rejects_negative_batch_code() {
        // AnnData uses code = −1 for missing/NA. We reject it loudly rather than silently
        // routing those cells into batch 0.
        let tmp = tmp_h5ad();
        write_h5ad_with_batch_codes(tmp.path(), 3, &["A", "B"], |grp| {
            let codes = ndarray::Array1::from_vec(vec![0i32, -1, 1]);
            grp.new_dataset_builder().with_data(&codes).create("codes").unwrap();
        });
        let result = read_h5ad(tmp.path());
        let err = match result {
            Ok(_) => panic!("expected error for negative batch code, got Ok"),
            Err(e) => format!("{e:#}"),
        };
        assert!(err.contains("negative codes") || err.contains("missing"),
                "expected 'negative codes' / 'missing' in error, got: {err}");
    }

    #[test]
    fn read_h5ad_rejects_out_of_range_batch_code() {
        // A code that exceeds n_categories must fail rather than panic later at gamma[code].
        let tmp = tmp_h5ad();
        write_h5ad_with_batch_codes(tmp.path(), 3, &["A", "B"], |grp| {
            let codes = ndarray::Array1::from_vec(vec![0i32, 1, 5]); // 5 ≥ 2 categories
            grp.new_dataset_builder().with_data(&codes).create("codes").unwrap();
        });
        let result = read_h5ad(tmp.path());
        let err = match result {
            Ok(_) => panic!("expected error for out-of-range batch code, got Ok"),
            Err(e) => format!("{e:#}"),
        };
        assert!(err.contains("out of range"), "expected 'out of range', got: {err}");
    }

    #[test]
    fn read_h5ad_large_matrix_dimensions() {
        // 100 cells × 50 guides, 200 nonzeros
        let tmp = tmp_h5ad();
        let cells: Vec<String> = (0..100).map(|i| format!("C{i}")).collect();
        let guides: Vec<String> = (0..50).map(|i| format!("g{i}")).collect();
        let cell_refs: Vec<&str> = cells.iter().map(|s| s.as_str()).collect();
        let guide_refs: Vec<&str> = guides.iter().map(|s| s.as_str()).collect();

        let mut triples: Vec<(usize, usize, u32)> = Vec::new();
        for c in 0..100 {
            let g = c % 50;
            triples.push((c, g, (c + 1) as u32));
            if c + 1 < 100 {
                triples.push((c, (g + 1) % 50, 1));
            }
        }
        triples.sort();
        triples.dedup();

        write_test_h5ad(tmp.path(), &cell_refs, &guide_refs, &triples);
        let loaded = read_h5ad(tmp.path()).unwrap();
        assert_eq!(loaded.counts.n_cells, 100);
        assert_eq!(loaded.counts.n_guides, 50);
        assert!(loaded.counts.nnz() > 0);
    }
}
