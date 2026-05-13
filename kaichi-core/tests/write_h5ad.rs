// Roundtrip integration tests: write via write_h5ad, read back via read_h5ad,
// verify the pipeline result is preserved end-to-end.

use kaichi_core::io::read::read_h5ad;
use kaichi_core::io::write::write_h5ad;
use kaichi_core::models::{poisson_gauss::PoissonGaussModel, AssignmentModel};
use kaichi_core::io::read::testutil::write_test_h5ad;
use tempfile::NamedTempFile;

fn tmp_h5ad() -> NamedTempFile {
    NamedTempFile::with_suffix(".h5ad").unwrap()
}

/// Build a 6-cell × 2-guide test fixture: C4/C5/C6 have signal on gA.
fn make_fixture() -> NamedTempFile {
    let tmp = tmp_h5ad();
    write_test_h5ad(
        tmp.path(),
        &["C1", "C2", "C3", "C4", "C5", "C6"],
        &["gA", "gB"],
        &[
            (0, 1, 1),  // C1: gB=1
            (1, 0, 1),  // C2: gA=1
            (2, 1, 2),  // C3: gB=2
            (3, 0, 25), // C4: gA=25  (signal)
            (4, 0, 30), // C5: gA=30  (signal)
            (4, 1, 1),  // C5: gB=1
            (5, 0, 22), // C6: gA=22  (signal)
        ],
    );
    tmp
}

#[test]
fn write_h5ad_roundtrip_obs_names() {
    let fixture = make_fixture();
    let input = read_h5ad(fixture.path()).unwrap();
    let result = PoissonGaussModel::default().assign(&input).unwrap();

    let out = tmp_h5ad();
    write_h5ad(&input, &result, out.path(), "poisson_gauss", "{}").unwrap();

    let loaded = read_h5ad(out.path()).unwrap();
    assert_eq!(loaded.counts.n_cells, 6);

    let bcs = &loaded.covariates.cell_barcodes;
    assert_eq!(bcs.value(0), "C1");
    assert_eq!(bcs.value(5), "C6");
}

#[test]
fn write_h5ad_roundtrip_var_names() {
    let fixture = make_fixture();
    let input = read_h5ad(fixture.path()).unwrap();
    let result = PoissonGaussModel::default().assign(&input).unwrap();

    let out = tmp_h5ad();
    write_h5ad(&input, &result, out.path(), "poisson_gauss", "{}").unwrap();

    let loaded = read_h5ad(out.path()).unwrap();
    assert_eq!(loaded.counts.n_guides, 2);
    assert_eq!(loaded.guide_metadata.guide_ids.value(0), "gA");
    assert_eq!(loaded.guide_metadata.guide_ids.value(1), "gB");
}

#[test]
fn write_h5ad_x_is_raw_umi_counts() {
    let fixture = make_fixture();
    let input = read_h5ad(fixture.path()).unwrap();
    let result = PoissonGaussModel::default().assign(&input).unwrap();

    let out = tmp_h5ad();
    write_h5ad(&input, &result, out.path(), "poisson_gauss", "{}").unwrap();

    let loaded = read_h5ad(out.path()).unwrap();
    // C1: gB=1 only
    let r0 = loaded.counts.csr().get_row(0).unwrap();
    assert_eq!(r0.col_indices(), &[1]);
    assert_eq!(r0.values(), &[1u32]);
    // C4: gA=25
    let r3 = loaded.counts.csr().get_row(3).unwrap();
    assert_eq!(r3.col_indices(), &[0]);
    assert_eq!(r3.values(), &[25u32]);
    // C5: gA=30, gB=1
    let r4 = loaded.counts.csr().get_row(4).unwrap();
    assert_eq!(r4.col_indices(), &[0, 1]);
    assert_eq!(r4.values(), &[30u32, 1u32]);
}

#[test]
fn write_h5ad_assigned_layer_has_correct_cells() {
    let fixture = make_fixture();
    let input = read_h5ad(fixture.path()).unwrap();
    let result = PoissonGaussModel::default().assign(&input).unwrap();

    // Verify assigned_x has nonzeros exactly for the signal cells (C4/C5/C6)
    use arrow::array::BooleanArray;
    let is_u = result.batch.column_by_name("is_unassigned").unwrap()
        .as_any().downcast_ref::<BooleanArray>().unwrap();

    for cell in 0..6 {
        let row = result.assigned_x.get_row(cell).unwrap();
        if !is_u.value(cell) {
            assert!(row.nnz() > 0, "cell {cell} should have nonzero in assigned_x");
        } else {
            assert_eq!(row.nnz(), 0, "cell {cell} should be empty in assigned_x");
        }
    }
}

#[test]
fn write_h5ad_uns_model_name() {
    let fixture = make_fixture();
    let input = read_h5ad(fixture.path()).unwrap();
    let result = PoissonGaussModel::default().assign(&input).unwrap();

    let out = tmp_h5ad();
    write_h5ad(&input, &result, out.path(), "poisson_gauss", "{}").unwrap();

    use hdf5::types::VarLenUnicode;
    let file = hdf5::File::open(out.path()).unwrap();
    let ds = file.dataset("uns/kaichi/model").unwrap();
    let vals: Vec<VarLenUnicode> = ds.read_raw::<VarLenUnicode>().unwrap();
    assert_eq!(vals[0].as_str(), "poisson_gauss");
}

#[test]
fn write_h5ad_uns_model_params() {
    let fixture = make_fixture();
    let input = read_h5ad(fixture.path()).unwrap();
    let result = PoissonGaussModel::default().assign(&input).unwrap();
    let params = PoissonGaussModel::default().params_json().to_string();

    let out = tmp_h5ad();
    write_h5ad(&input, &result, out.path(), "poisson_gauss", &params).unwrap();

    use hdf5::types::VarLenUnicode;
    let file = hdf5::File::open(out.path()).unwrap();
    let ds = file.dataset("uns/kaichi/model_params").unwrap();
    let vals: Vec<VarLenUnicode> = ds.read_raw::<VarLenUnicode>().unwrap();
    assert_eq!(vals[0].as_str(), &params);
}

#[test]
fn write_h5ad_nnz_preserved() {
    let fixture = make_fixture();
    let input = read_h5ad(fixture.path()).unwrap();
    let orig_nnz = input.counts.nnz();
    let result = PoissonGaussModel::default().assign(&input).unwrap();

    let out = tmp_h5ad();
    write_h5ad(&input, &result, out.path(), "poisson_gauss", "{}").unwrap();

    let loaded = read_h5ad(out.path()).unwrap();
    assert_eq!(loaded.counts.nnz(), orig_nnz);
}
