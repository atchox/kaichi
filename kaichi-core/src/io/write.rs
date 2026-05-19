use crate::data::{AssignmentResult, LoadedInput};
use anyhow::{Context, Result};
use arrow::array::{Array, BooleanArray, Float32Array, StringArray, UInt8Array};
use arrow::compute::cast;
use arrow::datatypes::DataType;
use hdf5::{types::VarLenUnicode, File, Group, Location};
use nalgebra_sparse::CsrMatrix;
use ndarray::{arr0, arr1, Array1};
use std::path::Path;
use std::str::FromStr;

/// Write a kaichi assignment as a guide-only H5AD.
///
/// Layout:
/// ```text
/// /obs/_index         cell barcodes
/// /obs/guide_identity assigned guide ID or "" if unassigned
/// /obs/n_guides_detected
/// /var/_index         guide IDs
/// /X                  CSR UInt32 (raw UMI counts, preserved from input)
/// /layers/assigned    CSR UInt8  (binary assignment)
/// /uns/kaichi/model   model name
/// /uns/kaichi/model_params  JSON string
/// /uns/kaichi/version kaichi version
/// ```
pub fn write_h5ad(
    input: &LoadedInput,
    result: &AssignmentResult,
    path: &Path,
    model_name: &str,
    model_params_json: &str,
) -> Result<()> {
    let file = File::create(path)
        .with_context(|| format!("cannot create H5AD: {}", path.display()))?;

    write_scalar_str_attr(&file, "encoding-type", "anndata");
    write_scalar_str_attr(&file, "encoding-version", "0.1.0");

    let n_cells = input.counts.n_cells;
    let n_guides = input.counts.n_guides;

    write_obs(&file, input, result, n_cells)?;
    write_var(&file, input, n_guides)?;
    write_csr_group(&file, "X", input.counts.csr(), n_cells, n_guides)?;

    let layers = file.create_group("layers").context("creating /layers")?;
    write_scalar_str_attr(&layers, "encoding-type", "dict");
    write_scalar_str_attr(&layers, "encoding-version", "0.1.0");
    write_csr_group(&layers, "assigned", &result.assigned_x, n_cells, n_guides)?;

    write_uns(&file, model_name, model_params_json)?;

    Ok(())
}

// ---------------------------------------------------------------------------
// obs
// ---------------------------------------------------------------------------

fn write_obs(file: &File, input: &LoadedInput, result: &AssignmentResult, n_cells: usize) -> Result<()> {
    let obs = file.create_group("obs").context("creating /obs")?;
    write_scalar_str_attr(&obs, "encoding-type", "dataframe");
    write_scalar_str_attr(&obs, "encoding-version", "0.2.0");
    write_scalar_str_attr(&obs, "_index", "_index");
    write_str_arr_attr(&obs, "column-order", &[
        "guide_identity", "assignment_confidence", "n_guides_detected", "is_unassigned", "is_multi_infected",
    ]);

    // /obs/_index — cell barcodes
    write_str_dataset(&obs, "_index", &input.covariates.cell_barcodes)?;

    let batch = &result.batch;

    let guide_col_raw = batch.column_by_name("guide_id").context("result missing guide_id")?;
    let guide_col_utf8 = cast(guide_col_raw.as_ref(), &DataType::Utf8)
        .context("casting guide_id to Utf8")?;
    let guide_ids = guide_col_utf8
        .as_any()
        .downcast_ref::<StringArray>()
        .context("guide_id not Utf8 after cast")?;

    let is_unassigned = batch
        .column_by_name("is_unassigned")
        .context("result missing is_unassigned")?
        .as_any()
        .downcast_ref::<BooleanArray>()
        .context("is_unassigned not Boolean")?;

    let is_multi = batch
        .column_by_name("is_multi_infected")
        .context("result missing is_multi_infected")?
        .as_any()
        .downcast_ref::<BooleanArray>()
        .context("is_multi_infected not Boolean")?;

    let n_detected = batch
        .column_by_name("n_guides_detected")
        .context("result missing n_guides_detected")?
        .as_any()
        .downcast_ref::<UInt8Array>()
        .context("n_guides_detected not UInt8")?;

    let confidence = batch
        .column_by_name("assignment_confidence")
        .context("result missing assignment_confidence")?
        .as_any()
        .downcast_ref::<Float32Array>()
        .context("assignment_confidence not Float32")?;

    // guide_identity: assigned guide ID, or "" for unassigned cells
    let guide_identity: Vec<VarLenUnicode> = (0..n_cells)
        .map(|i| {
            if is_unassigned.value(i) || guide_ids.is_null(i) {
                VarLenUnicode::from_str("").unwrap()
            } else {
                VarLenUnicode::from_str(guide_ids.value(i)).unwrap()
            }
        })
        .collect();
    write_array_dataset(&obs, "guide_identity", &Array1::from_vec(guide_identity))?;

    // NaN encodes null (unassigned) — standard anndata convention for missing floats.
    let conf_arr: Array1<f32> = (0..n_cells)
        .map(|i| if confidence.is_null(i) { f32::NAN } else { confidence.value(i) })
        .collect();
    write_array_dataset(&obs, "assignment_confidence", &conf_arr)?;

    let n_det_arr: Array1<i32> = (0..n_cells).map(|i| n_detected.value(i) as i32).collect();
    write_array_dataset(&obs, "n_guides_detected", &n_det_arr)?;

    let is_u_arr: Array1<u8> = (0..n_cells).map(|i| is_unassigned.value(i) as u8).collect();
    write_array_dataset(&obs, "is_unassigned", &is_u_arr)?;

    let is_m_arr: Array1<u8> = (0..n_cells).map(|i| is_multi.value(i) as u8).collect();
    write_array_dataset(&obs, "is_multi_infected", &is_m_arr)?;

    Ok(())
}

// ---------------------------------------------------------------------------
// var
// ---------------------------------------------------------------------------

fn write_var(file: &File, input: &LoadedInput, n_guides: usize) -> Result<()> {
    let var = file.create_group("var").context("creating /var")?;
    write_scalar_str_attr(&var, "encoding-type", "dataframe");
    write_scalar_str_attr(&var, "encoding-version", "0.2.0");
    write_scalar_str_attr(&var, "_index", "_index");
    // no extra columns
    let empty: Array1<f64> = Array1::zeros(0);
    var.new_attr_builder().with_data(&empty).create("column-order")
        .context("writing var column-order attr")?;

    write_str_dataset(&var, "_index", &input.guide_metadata.guide_ids)?;
    let _ = n_guides; // validated implicitly via guide_ids length
    Ok(())
}

// ---------------------------------------------------------------------------
// Sparse matrix groups
// ---------------------------------------------------------------------------

fn write_csr_group<T>(
    parent: &Group,
    name: &str,
    csr: &CsrMatrix<T>,
    n_rows: usize,
    n_cols: usize,
) -> Result<()>
where
    T: hdf5::H5Type + Copy,
    Vec<T>: Into<Vec<T>>,
{
    let grp = parent.create_group(name)
        .with_context(|| format!("creating group {name}"))?;
    write_scalar_str_attr(&grp, "encoding-type", "csr_matrix");
    write_scalar_str_attr(&grp, "encoding-version", "0.1.0");

    let shape = arr1(&[n_rows as i64, n_cols as i64]);
    grp.new_attr_builder().with_data(&shape).create("shape")
        .context("writing shape attr")?;

    // data / indices / indptr written with gzip-3: standard HDF5 filter,
    // readable by h5py without external plugins (unlike blosc).
    let values: Vec<T> = csr.values().to_vec();
    let data_arr = Array1::from_vec(values);
    grp.new_dataset_builder().deflate(3).with_data(&data_arr).create("data")
        .with_context(|| format!("{name}/data"))?;

    let indices: Array1<i32> = csr.col_indices().iter().map(|&c| c as i32).collect();
    grp.new_dataset_builder().deflate(3).with_data(&indices).create("indices")
        .with_context(|| format!("{name}/indices"))?;

    let indptr: Array1<i32> = csr.row_offsets().iter().map(|&o| o as i32).collect();
    grp.new_dataset_builder().deflate(3).with_data(&indptr).create("indptr")
        .with_context(|| format!("{name}/indptr"))?;

    Ok(())
}

// ---------------------------------------------------------------------------
// uns
// ---------------------------------------------------------------------------

fn write_uns(file: &File, model_name: &str, model_params_json: &str) -> Result<()> {
    let uns = file.create_group("uns").context("creating /uns")?;
    write_scalar_str_attr(&uns, "encoding-type", "dict");
    write_scalar_str_attr(&uns, "encoding-version", "0.1.0");

    let kaichi = uns.create_group("kaichi").context("creating /uns/kaichi")?;
    write_scalar_str_attr(&kaichi, "encoding-type", "dict");
    write_scalar_str_attr(&kaichi, "encoding-version", "0.1.0");

    write_scalar_str_dataset(&kaichi, "model", model_name)?;
    write_scalar_str_dataset(&kaichi, "model_params", model_params_json)?;
    write_scalar_str_dataset(&kaichi, "version", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Low-level helpers
// ---------------------------------------------------------------------------

fn vlu(s: &str) -> VarLenUnicode {
    VarLenUnicode::from_str(s).unwrap()
}

fn write_scalar_str_attr(loc: &Location, name: &str, val: &str) {
    let arr = arr0(vlu(val));
    loc.new_attr_builder().with_data(&arr).create(name).unwrap();
}

fn write_str_arr_attr(loc: &Location, name: &str, vals: &[&str]) {
    let arr: Array1<VarLenUnicode> = vals.iter().map(|s| vlu(s)).collect();
    loc.new_attr_builder().with_data(&arr).create(name).unwrap();
}

fn write_str_dataset(grp: &Group, name: &str, arr: &StringArray) -> Result<()> {
    let vlu_vec: Array1<VarLenUnicode> = (0..arr.len())
        .map(|i| VarLenUnicode::from_str(arr.value(i)).unwrap())
        .collect();
    grp.new_dataset_builder()
        .with_data(&vlu_vec)
        .create(name)
        .map(|_| ())
        .with_context(|| format!("writing dataset {name}"))
}

fn write_array_dataset<T: hdf5::H5Type>(grp: &Group, name: &str, arr: &Array1<T>) -> Result<()> {
    grp.new_dataset_builder()
        .with_data(arr)
        .create(name)
        .map(|_| ())
        .with_context(|| format!("writing dataset {name}"))
}

fn write_scalar_str_dataset(grp: &Group, name: &str, val: &str) -> Result<()> {
    let arr: Array1<VarLenUnicode> = Array1::from_vec(vec![vlu(val)]);
    grp.new_dataset_builder()
        .with_data(&arr)
        .create(name)
        .map(|_| ())
        .with_context(|| format!("writing dataset {name}"))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{CountMatrix, Covariates, GuideMetadata};
    use crate::io::read::read_h5ad;
    use arrow::array::{Float32Array, StringBuilder};
    use std::sync::Arc;
    use tempfile::NamedTempFile;

    fn tmp_h5ad() -> NamedTempFile {
        NamedTempFile::with_suffix(".h5ad").unwrap()
    }

    /// Build a small LoadedInput and AssignmentResult for write tests.
    fn make_write_fixture() -> (LoadedInput, AssignmentResult) {
        // 4 cells, 2 guides: C3 and C4 assigned to gA
        //   C1: gA=0, gB=1
        //   C2: gA=1, gB=0
        //   C3: gA=25
        //   C4: gA=30
        let counts = CountMatrix::try_from_csr(
            4, 2,
            vec![0, 1, 2, 3, 4],
            vec![1, 0, 0, 0],
            vec![1, 1, 25, 30],
        ).unwrap();

        let mut bc = StringBuilder::new();
        ["C1", "C2", "C3", "C4"].iter().for_each(|s| bc.append_value(s));
        let covariates = Covariates {
            cell_barcodes: bc.finish(),
            total_counts: Float32Array::from(vec![1.0f32, 1.0, 25.0, 30.0]),
            batch: crate::data::BatchLabels::single_batch(4),
        };

        let mut gd = StringBuilder::new();
        ["gA", "gB"].iter().for_each(|s| gd.append_value(s));
        let guide_metadata = GuideMetadata { guide_ids: gd.finish() };

        let input = LoadedInput { counts, covariates, guide_metadata };

        // assigned_x: cells 2 and 3 assigned to guide 0
        let assigned = CsrMatrix::try_from_csr_data(
            4, 2,
            vec![0, 0, 0, 1, 2],
            vec![0, 0],
            vec![1u8, 1],
        ).unwrap();

        use crate::schema::assignment_schema_ref;
        use arrow::array::{
            BooleanBuilder, DictionaryArray, Float32Builder, StringArray as SA,
            StringDictionaryBuilder, UInt32Builder, UInt8Builder,
        };
        use arrow::datatypes::Int16Type;
        use std::sync::Arc;

        let n = 4;
        let model_val = SA::from(vec!["poisson_gauss"]);
        let model_keys = arrow::array::Int8Array::from(vec![0i8; n]);
        let model_dict = DictionaryArray::try_new(model_keys, Arc::new(model_val)).unwrap();

        let mut guide_dict: StringDictionaryBuilder<Int16Type> = StringDictionaryBuilder::new();
        guide_dict.append_null();
        guide_dict.append_null();
        guide_dict.append_value("gA");
        guide_dict.append_value("gA");

        let mut tg_dict: StringDictionaryBuilder<Int16Type> = StringDictionaryBuilder::new();
        for _ in 0..n { tg_dict.append_null(); }

        let mut umi = UInt32Builder::new();
        umi.append_null(); umi.append_null();
        umi.append_value(25); umi.append_value(30);

        let mut conf = Float32Builder::new();
        conf.append_null(); conf.append_null();
        conf.append_value(0.95); conf.append_value(0.97);

        let mut is_u = BooleanBuilder::new();
        is_u.append_value(true); is_u.append_value(true);
        is_u.append_value(false); is_u.append_value(false);

        let mut is_m = BooleanBuilder::new();
        for _ in 0..n { is_m.append_value(false); }

        let mut n_det = UInt8Builder::new();
        n_det.append_value(0); n_det.append_value(0);
        n_det.append_value(1); n_det.append_value(1);

        let batch = arrow::record_batch::RecordBatch::try_new(
            assignment_schema_ref(),
            vec![
                Arc::new(SA::from(vec!["C1", "C2", "C3", "C4"])),
                Arc::new(guide_dict.finish()),
                Arc::new(tg_dict.finish()),
                Arc::new(umi.finish()),
                Arc::new(conf.finish()),
                Arc::new(model_dict),
                Arc::new(is_u.finish()),
                Arc::new(is_m.finish()),
                Arc::new(n_det.finish()),
            ],
        ).unwrap();

        (input, AssignmentResult { batch, assigned_x: assigned })
    }

    #[test]
    fn write_then_read_obs_names() {
        let (input, result) = make_write_fixture();
        let tmp = tmp_h5ad();
        write_h5ad(&input, &result, tmp.path(), "poisson_gauss", "{}").unwrap();

        let loaded = read_h5ad(tmp.path()).unwrap();
        assert_eq!(loaded.counts.n_cells, 4);
        assert_eq!(loaded.covariates.cell_barcodes.value(0), "C1");
        assert_eq!(loaded.covariates.cell_barcodes.value(3), "C4");
    }

    #[test]
    fn write_then_read_var_names() {
        let (input, result) = make_write_fixture();
        let tmp = tmp_h5ad();
        write_h5ad(&input, &result, tmp.path(), "poisson_gauss", "{}").unwrap();

        let loaded = read_h5ad(tmp.path()).unwrap();
        assert_eq!(loaded.counts.n_guides, 2);
        assert_eq!(loaded.guide_metadata.guide_ids.value(0), "gA");
        assert_eq!(loaded.guide_metadata.guide_ids.value(1), "gB");
    }

    #[test]
    fn write_then_read_x_matrix() {
        let (input, result) = make_write_fixture();
        let tmp = tmp_h5ad();
        write_h5ad(&input, &result, tmp.path(), "poisson_gauss", "{}").unwrap();

        let loaded = read_h5ad(tmp.path()).unwrap();
        // C1: gB=1
        let r0 = loaded.counts.csr().get_row(0).unwrap();
        assert_eq!(r0.col_indices(), &[1]);
        assert_eq!(r0.values(), &[1]);
        // C3: gA=25
        let r2 = loaded.counts.csr().get_row(2).unwrap();
        assert_eq!(r2.col_indices(), &[0]);
        assert_eq!(r2.values(), &[25]);
    }

    #[test]
    fn write_then_read_x_nnz_preserved() {
        let (input, result) = make_write_fixture();
        let tmp = tmp_h5ad();
        write_h5ad(&input, &result, tmp.path(), "poisson_gauss", "{}").unwrap();

        let loaded = read_h5ad(tmp.path()).unwrap();
        assert_eq!(loaded.counts.nnz(), input.counts.nnz());
    }

    #[test]
    fn write_obs_guide_identity() {
        let (input, result) = make_write_fixture();
        let tmp = tmp_h5ad();
        write_h5ad(&input, &result, tmp.path(), "poisson_gauss", "{}").unwrap();

        let file = hdf5::File::open(tmp.path()).unwrap();
        let ds = file.dataset("obs/guide_identity").unwrap();
        let vals: Vec<VarLenUnicode> = ds.read_raw::<VarLenUnicode>().unwrap();
        // C1 and C2 unassigned → ""
        assert_eq!(vals[0].as_str(), "");
        assert_eq!(vals[1].as_str(), "");
        // C3 and C4 assigned to gA
        assert_eq!(vals[2].as_str(), "gA");
        assert_eq!(vals[3].as_str(), "gA");
    }

    #[test]
    fn write_obs_is_unassigned() {
        let (input, result) = make_write_fixture();
        let tmp = tmp_h5ad();
        write_h5ad(&input, &result, tmp.path(), "poisson_gauss", "{}").unwrap();

        let file = hdf5::File::open(tmp.path()).unwrap();
        let ds = file.dataset("obs/is_unassigned").unwrap();
        let vals: Vec<u8> = ds.read_raw::<u8>().unwrap();
        // C1 and C2 unassigned, C3 and C4 assigned
        assert_eq!(vals, vec![1, 1, 0, 0]);
    }

    #[test]
    fn write_obs_is_multi_infected() {
        let (input, result) = make_write_fixture();
        let tmp = tmp_h5ad();
        write_h5ad(&input, &result, tmp.path(), "poisson_gauss", "{}").unwrap();

        let file = hdf5::File::open(tmp.path()).unwrap();
        let ds = file.dataset("obs/is_multi_infected").unwrap();
        let vals: Vec<u8> = ds.read_raw::<u8>().unwrap();
        // fixture has no multi-infected cells
        assert_eq!(vals, vec![0, 0, 0, 0]);
    }

    #[test]
    fn write_obs_n_guides_detected() {
        let (input, result) = make_write_fixture();
        let tmp = tmp_h5ad();
        write_h5ad(&input, &result, tmp.path(), "poisson_gauss", "{}").unwrap();

        let file = hdf5::File::open(tmp.path()).unwrap();
        let ds = file.dataset("obs/n_guides_detected").unwrap();
        let vals: Vec<i32> = ds.read_raw::<i32>().unwrap();
        assert_eq!(vals[0], 0);
        assert_eq!(vals[1], 0);
        assert_eq!(vals[2], 1);
        assert_eq!(vals[3], 1);
    }

    #[test]
    fn write_layers_assigned_group_exists() {
        let (input, result) = make_write_fixture();
        let tmp = tmp_h5ad();
        write_h5ad(&input, &result, tmp.path(), "poisson_gauss", "{}").unwrap();

        let file = hdf5::File::open(tmp.path()).unwrap();
        assert!(file.group("layers/assigned").is_ok(), "layers/assigned group missing");
        assert!(file.dataset("layers/assigned/data").is_ok());
        assert!(file.dataset("layers/assigned/indices").is_ok());
        assert!(file.dataset("layers/assigned/indptr").is_ok());
    }

    #[test]
    fn write_uns_model_name() {
        let (input, result) = make_write_fixture();
        let tmp = tmp_h5ad();
        write_h5ad(&input, &result, tmp.path(), "poisson_gauss", r#"{"min_confidence":0.5}"#).unwrap();

        let file = hdf5::File::open(tmp.path()).unwrap();
        let ds = file.dataset("uns/kaichi/model").unwrap();
        let vals: Vec<VarLenUnicode> = ds.read_raw::<VarLenUnicode>().unwrap();
        assert_eq!(vals[0].as_str(), "poisson_gauss");
    }

    #[test]
    fn write_uns_model_params_json() {
        let (input, result) = make_write_fixture();
        let tmp = tmp_h5ad();
        let params = r#"{"min_confidence":0.5}"#;
        write_h5ad(&input, &result, tmp.path(), "poisson_gauss", params).unwrap();

        let file = hdf5::File::open(tmp.path()).unwrap();
        let ds = file.dataset("uns/kaichi/model_params").unwrap();
        let vals: Vec<VarLenUnicode> = ds.read_raw::<VarLenUnicode>().unwrap();
        assert_eq!(vals[0].as_str(), params);
    }

    #[test]
    fn write_obs_guide_identity_for_multi_infected_cell() {
        // A multi-infected cell should write its best guide name to guide_identity,
        // not blank. The write layer must not treat is_multi as is_unassigned.
        use crate::schema::assignment_schema_ref;
        use arrow::array::{
            BooleanBuilder, DictionaryArray, Float32Builder,
            StringArray as SA, StringDictionaryBuilder, UInt32Builder, UInt8Builder,
        };
        use arrow::datatypes::Int16Type;

        let counts = CountMatrix::try_from_csr(
            1, 2,
            vec![0, 2],
            vec![0, 1],
            vec![25u32, 20],
        ).unwrap();
        let mut bc = StringBuilder::new();
        bc.append_value("CELL1");
        let mut gd = StringBuilder::new();
        ["gA", "gB"].iter().for_each(|s| gd.append_value(s));
        let input = LoadedInput {
            counts,
            covariates: Covariates {
                cell_barcodes: bc.finish(),
                total_counts: Float32Array::from(vec![45.0f32]),
                batch: crate::data::BatchLabels::single_batch(1),
            },
            guide_metadata: GuideMetadata { guide_ids: gd.finish() },
        };

        let model_val = SA::from(vec!["umi"]);
        let model_keys = arrow::array::Int8Array::from(vec![0i8]);
        let model_dict = DictionaryArray::try_new(model_keys, Arc::new(model_val)).unwrap();

        let mut guide_dict: StringDictionaryBuilder<Int16Type> = StringDictionaryBuilder::new();
        guide_dict.append_value("gA"); // best guide

        let mut tg_dict: StringDictionaryBuilder<Int16Type> = StringDictionaryBuilder::new();
        tg_dict.append_null();

        let mut umi = UInt32Builder::new();
        umi.append_value(25);
        let mut conf = Float32Builder::new();
        conf.append_value(1.0);
        let mut is_u = BooleanBuilder::new();
        is_u.append_value(false);
        let mut is_m = BooleanBuilder::new();
        is_m.append_value(true); // multi-infected
        let mut n_det = UInt8Builder::new();
        n_det.append_value(2);

        let batch = arrow::record_batch::RecordBatch::try_new(
            assignment_schema_ref(),
            vec![
                Arc::new(SA::from(vec!["CELL1"])),
                Arc::new(guide_dict.finish()),
                Arc::new(tg_dict.finish()),
                Arc::new(umi.finish()),
                Arc::new(conf.finish()),
                Arc::new(model_dict),
                Arc::new(is_u.finish()),
                Arc::new(is_m.finish()),
                Arc::new(n_det.finish()),
            ],
        ).unwrap();

        let assigned = nalgebra_sparse::CsrMatrix::try_from_csr_data(
            1, 2, vec![0, 2], vec![0, 1], vec![1u8, 1],
        ).unwrap();
        let result = AssignmentResult { batch, assigned_x: assigned };

        let tmp = tmp_h5ad();
        write_h5ad(&input, &result, tmp.path(), "umi", "{}").unwrap();

        let file = hdf5::File::open(tmp.path()).unwrap();
        let ds = file.dataset("obs/guide_identity").unwrap();
        let vals: Vec<VarLenUnicode> = ds.read_raw::<VarLenUnicode>().unwrap();
        assert_eq!(vals[0].as_str(), "gA", "multi-infected cell should write best guide, not blank");
    }

    #[test]
    fn roundtrip_h5ad_preserves_all_cells() {
        // Write via write_h5ad, read back via read_h5ad, confirm cell count and X
        let (input, result) = make_write_fixture();
        let tmp = tmp_h5ad();
        write_h5ad(&input, &result, tmp.path(), "max", "{}").unwrap();

        let loaded = read_h5ad(tmp.path()).unwrap();
        assert_eq!(loaded.counts.n_cells, 4);
        assert_eq!(loaded.counts.n_guides, 2);
        assert_eq!(loaded.counts.nnz(), input.counts.nnz());

        // Total UMI sum preserved
        let orig_sum: u32 = (0..input.counts.n_cells)
            .map(|r| input.counts.csr().get_row(r).unwrap().values().iter().sum::<u32>())
            .sum();
        let loaded_sum: u32 = (0..loaded.counts.n_cells)
            .map(|r| loaded.counts.csr().get_row(r).unwrap().values().iter().sum::<u32>())
            .sum();
        assert_eq!(orig_sum, loaded_sum);
    }
}
