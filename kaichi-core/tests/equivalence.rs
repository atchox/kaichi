mod common;

use kaichi_core::io::read::read_h5ad;
use kaichi_core::models::{
    max::MaxModel,
    poisson_gauss::PoissonGaussModel,
    ratio::RatioModel,
    umi::UmiModel,
    AssignmentModel,
};

use arrow::array::{BooleanArray, StringArray};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn load_reference(model: &str) -> HashMap<String, Vec<String>> {
    let path = common::reference_assignments(model);
    let mut rdr = csv::Reader::from_path(&path)
        .unwrap_or_else(|e| panic!("cannot open reference CSV {}: {e}", path.display()));

    let headers = rdr.headers().expect("missing CSV headers").clone();
    let grna_col = headers.iter().position(|h| h == "gRNA")
        .expect("no 'gRNA' column in reference CSV");

    let mut map: HashMap<String, Vec<String>> = HashMap::new();
    for result in rdr.records() {
        let rec = result.expect("CSV parse error");
        let cell = rec.get(0).expect("missing cell column").to_string();
        let grna = rec.get(grna_col).expect("missing gRNA column").to_string();
        map.entry(cell).or_default().push(grna);
    }
    map
}

/// Of cells both kaichi and the reference assigned, fraction with matching guide.
fn confident_call_agreement(
    result: &kaichi_core::data::AssignmentResult,
    reference: &HashMap<String, Vec<String>>,
) -> f64 {
    use arrow::array::DictionaryArray;
    use arrow::datatypes::Int16Type;

    let batch = &result.batch;
    let barcodes = batch.column_by_name("cell_barcode").unwrap()
        .as_any().downcast_ref::<StringArray>().unwrap();
    let guide_ids = batch.column_by_name("guide_id").unwrap();
    let is_unassigned = batch.column_by_name("is_unassigned").unwrap()
        .as_any().downcast_ref::<BooleanArray>().unwrap();

    let dict = guide_ids.as_any().downcast_ref::<DictionaryArray<Int16Type>>().unwrap();
    let values = dict.values().as_any().downcast_ref::<StringArray>().unwrap();

    let mut agreed = 0usize;
    let mut compared = 0usize;

    for i in 0..batch.num_rows() {
        if is_unassigned.value(i) { continue; }
        let barcode = barcodes.value(i);
        let Some(ref_grnas) = reference.get(barcode) else { continue };

        let key = dict.keys().value(i) as usize;
        let kaichi_grna = values.value(key);

        compared += 1;
        if ref_grnas.iter().any(|g| g == kaichi_grna) {
            agreed += 1;
        }
    }

    if compared == 0 { return 0.0; }
    agreed as f64 / compared as f64
}

/// Of cells the reference assigned, fraction that kaichi also assigned.
fn reference_coverage(
    result: &kaichi_core::data::AssignmentResult,
    reference: &HashMap<String, Vec<String>>,
) -> f64 {
    let batch = &result.batch;
    let barcodes = batch.column_by_name("cell_barcode").unwrap()
        .as_any().downcast_ref::<StringArray>().unwrap();
    let is_unassigned = batch.column_by_name("is_unassigned").unwrap()
        .as_any().downcast_ref::<BooleanArray>().unwrap();

    let kaichi_assigned: HashMap<&str, bool> = (0..batch.num_rows())
        .map(|i| (barcodes.value(i), !is_unassigned.value(i)))
        .collect();

    let covered = reference.keys()
        .filter(|bc| kaichi_assigned.get(bc.as_str()).copied().unwrap_or(false))
        .count();

    if reference.is_empty() { return 1.0; }
    covered as f64 / reference.len() as f64
}

/// Absolute difference between kaichi and reference assignment rates.
fn assignment_rate_diff(
    result: &kaichi_core::data::AssignmentResult,
    reference: &HashMap<String, Vec<String>>,
    n_cells: usize,
) -> f64 {
    let batch = &result.batch;
    let is_unassigned = batch.column_by_name("is_unassigned").unwrap()
        .as_any().downcast_ref::<BooleanArray>().unwrap();

    let kaichi_rate = (0..batch.num_rows())
        .filter(|&i| !is_unassigned.value(i))
        .count() as f64 / n_cells as f64;

    let ref_rate = reference.len() as f64 / n_cells as f64;

    (kaichi_rate - ref_rate).abs()
}

fn load_input() -> kaichi_core::data::LoadedInput {
    let h5ad = common::schraivogel_h5ad();
    read_h5ad(&h5ad)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", h5ad.display()))
}

// ---------------------------------------------------------------------------
// Equivalence tests (≥99% confident-call agreement with crispat reference)
// ---------------------------------------------------------------------------

const MIN_AGREEMENT: f64 = 0.99;
const MIN_COVERAGE: f64 = 0.99;
const MAX_RATE_DIFF: f64 = 0.05;

fn check_equivalence(
    result: &kaichi_core::data::AssignmentResult,
    reference: &HashMap<String, Vec<String>>,
    n_cells: usize,
    model: &str,
) {
    let agreement = confident_call_agreement(result, reference);
    assert!(
        agreement >= MIN_AGREEMENT,
        "{model}: guide agreement {:.4} < {MIN_AGREEMENT}",
        agreement
    );

    let coverage = reference_coverage(result, reference);
    assert!(
        coverage >= MIN_COVERAGE,
        "{model}: reference coverage {:.4} < {MIN_COVERAGE} \
         (kaichi missed calls the reference made)",
        coverage
    );

    let rate_diff = assignment_rate_diff(result, reference, n_cells);
    assert!(
        rate_diff <= MAX_RATE_DIFF,
        "{model}: assignment rate diff {:.4} > {MAX_RATE_DIFF} \
         (kaichi rate differs from reference rate by more than 5%)",
        rate_diff
    );
}

#[test]
fn umi_equivalence() {
    let input = load_input();
    let result = UmiModel::default().assign(&input).unwrap();
    let reference = load_reference("umi");
    check_equivalence(&result, &reference, input.counts.n_cells, "umi");
}

#[test]
fn max_equivalence() {
    let input = load_input();
    let result = MaxModel::default().assign(&input).unwrap();
    let reference = load_reference("max");
    check_equivalence(&result, &reference, input.counts.n_cells, "max");
}

#[test]
fn ratio_equivalence() {
    let input = load_input();
    let result = RatioModel::default().assign(&input).unwrap();
    let reference = load_reference("ratio");
    check_equivalence(&result, &reference, input.counts.n_cells, "ratio");
}

#[test]
fn poisson_gauss_equivalence() {
    let input = load_input();
    let result = PoissonGaussModel::default().assign(&input).unwrap();
    let reference = load_reference("poisson_gauss");
    check_equivalence(&result, &reference, input.counts.n_cells, "poisson_gauss");
}
