use kaichi_core::{
    io::write_h5ad::write_h5ad,
    models::{AssignmentInput, AssignmentModel, poisson_gauss::PoissonGaussModel},
};
use arrow::{
    array::{Float32Array, StringArray, UInt32Array},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use anndata::{AnnDataOp, AxisArraysOp};
use anndata::backend::Backend;
use anndata_hdf5::H5;
use std::sync::Arc;
use tempfile::NamedTempFile;

fn make_input() -> (AssignmentInput, Vec<String>) {
    // 6 cells, 2 guides: gA is signal for C4/C5/C6, gB is noise
    let counts_schema = Arc::new(Schema::new(vec![
        Field::new("cell_barcode", DataType::Utf8, false),
        Field::new("guide_id", DataType::Utf8, false),
        Field::new("umi_count", DataType::UInt32, false),
    ]));
    let rows: Vec<(&str, &str, u32)> = vec![
        ("C1", "gA", 0), ("C2", "gA", 1), ("C3", "gA", 0),
        ("C4", "gA", 25), ("C5", "gA", 30), ("C6", "gA", 22),
        ("C1", "gB", 1), ("C2", "gB", 0), ("C3", "gB", 2),
        ("C4", "gB", 0), ("C5", "gB", 1), ("C6", "gB", 0),
    ];
    let counts = RecordBatch::try_new(
        counts_schema,
        vec![
            Arc::new(StringArray::from(rows.iter().map(|(c,_,_)| *c).collect::<Vec<_>>())),
            Arc::new(StringArray::from(rows.iter().map(|(_,g,_)| *g).collect::<Vec<_>>())),
            Arc::new(UInt32Array::from(rows.iter().map(|(_,_,v)| *v).collect::<Vec<_>>())),
        ],
    ).unwrap();

    let cov_schema = Arc::new(Schema::new(vec![
        Field::new("cell_barcode", DataType::Utf8, false),
        Field::new("batch", DataType::Utf8, false),
        Field::new("total_counts", DataType::Float32, false),
    ]));
    let cells = vec!["C1", "C2", "C3", "C4", "C5", "C6"];
    let covariates = RecordBatch::try_new(
        cov_schema,
        vec![
            Arc::new(StringArray::from(cells.clone())),
            Arc::new(StringArray::from(vec!["default"; 6])),
            Arc::new(Float32Array::from(vec![0.0f32; 6])),
        ],
    ).unwrap();

    (
        AssignmentInput { counts, covariates },
        vec!["gA".to_string(), "gB".to_string()],
    )
}

#[test]
fn write_h5ad_roundtrip_obs_names() {
    let (input, var_names) = make_input();
    let result = PoissonGaussModel::default().assign(&input).unwrap();

    let tmp = NamedTempFile::with_suffix(".h5ad").unwrap();
    let obs_names: Vec<String> = vec![
        "C1","C2","C3","C4","C5","C6",
    ].into_iter().map(String::from).collect();

    write_h5ad(&input, &result, &obs_names, &var_names, tmp.path(), "poisson_gauss").unwrap();

    // Read back and verify obs/var names
    let store = H5::open(tmp.path()).unwrap();
    let adata = anndata::AnnData::<H5>::open(store).unwrap();

    assert_eq!(adata.n_obs(), 6);
    assert_eq!(adata.n_vars(), 2);

    let written_obs: Vec<String> = adata.obs_names().into_iter().collect();
    assert_eq!(written_obs, obs_names);

    let written_var: Vec<String> = adata.var_names().into_iter().collect();
    assert_eq!(written_var, var_names);
}

#[test]
fn write_h5ad_assigned_layer_correct() {
    let (input, var_names) = make_input();
    let result = PoissonGaussModel::default().assign(&input).unwrap();

    let tmp = NamedTempFile::with_suffix(".h5ad").unwrap();
    let obs_names: Vec<String> = vec!["C1","C2","C3","C4","C5","C6"]
        .into_iter().map(String::from).collect();

    write_h5ad(&input, &result, &obs_names, &var_names, tmp.path(), "poisson_gauss").unwrap();

    let store = H5::open(tmp.path()).unwrap();
    let adata = anndata::AnnData::<H5>::open(store).unwrap();

    // "assigned" layer must exist
    let layer_keys: Vec<String> = adata.layers().keys();
    assert!(layer_keys.contains(&"assigned".to_string()), "missing assigned layer: {layer_keys:?}");
}

#[test]
fn write_h5ad_x_is_raw_umi_counts() {
    use anndata::{ArrayElemOp, data::DynCsrMatrix, ArrayData};

    let (input, var_names) = make_input();
    let result = PoissonGaussModel::default().assign(&input).unwrap();

    let tmp = NamedTempFile::with_suffix(".h5ad").unwrap();
    let obs_names: Vec<String> = vec!["C1","C2","C3","C4","C5","C6"]
        .into_iter().map(String::from).collect();

    write_h5ad(&input, &result, &obs_names, &var_names, tmp.path(), "poisson_gauss").unwrap();

    let store = H5::open(tmp.path()).unwrap();
    let adata = anndata::AnnData::<H5>::open(store).unwrap();

    let x: ArrayData = adata.x().get::<ArrayData>().unwrap().unwrap();
    let csr = match x {
        ArrayData::CsrMatrix(DynCsrMatrix::U32(m)) => m,
        other => panic!("expected u32 CSR for X, got {other:?}"),
    };

    // make_input rows: C4=gA=25, C5=gA=30, C6=gA=22, C3=gB=2, C1=gB=1, C5=gB=1
    let dense = csr_to_dense(&csr);
    assert_eq!(dense[0], vec![0, 1]); // C1 gA=0 gB=1
    assert_eq!(dense[1], vec![1, 0]); // C2 gA=1 gB=0
    assert_eq!(dense[2], vec![0, 2]); // C3 gA=0 gB=2
    assert_eq!(dense[3], vec![25, 0]);
    assert_eq!(dense[4], vec![30, 1]);
    assert_eq!(dense[5], vec![22, 0]);
}

fn csr_to_dense(csr: &nalgebra_sparse::CsrMatrix<u32>) -> Vec<Vec<u32>> {
    let (nrows, ncols) = (csr.nrows(), csr.ncols());
    let mut dense = vec![vec![0u32; ncols]; nrows];
    for (r, row) in csr.row_iter().enumerate() {
        for (&c, &v) in row.col_indices().iter().zip(row.values().iter()) {
            dense[r][c] = v;
        }
    }
    dense
}

#[test]
fn write_h5ad_uns_method_name() {
    let (input, var_names) = make_input();
    let result = PoissonGaussModel::default().assign(&input).unwrap();

    let tmp = NamedTempFile::with_suffix(".h5ad").unwrap();
    let obs_names: Vec<String> = vec!["C1","C2","C3","C4","C5","C6"]
        .into_iter().map(String::from).collect();

    write_h5ad(&input, &result, &obs_names, &var_names, tmp.path(), "poisson_gauss").unwrap();

    let store = H5::open(tmp.path()).unwrap();
    let adata = anndata::AnnData::<H5>::open(store).unwrap();

    use anndata::ElemCollectionOp;
    let method: String = adata.uns()
        .get_item("guide_assignment_method").unwrap().unwrap();
    assert_eq!(method, "kaichi_poisson_gauss");
}
