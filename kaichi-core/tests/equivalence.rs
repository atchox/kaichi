mod common;

use kaichi_core::io::h5ad::read_h5ad;
use kaichi_core::models::{
    AssignmentInput,
    max::MaxModel,
    ratio::RatioModel,
    umi::UmiModel,
    AssignmentModel,
};

use arrow::array::{BooleanArray, StringArray};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Load a reference CSV into a map of cell_barcode → Vec<gRNA>.
///
/// A cell can appear multiple times (e.g., crispat UMI model lists one row per
/// guide above threshold). Vec captures all positive guides per cell.
/// Column positions vary across models; gRNA is located by header name.
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

/// Compute the fraction of confident-call cells where kaichi and the reference agree.
///
/// Agreement: kaichi's assigned guide is among the guides the reference lists for
/// that cell. Cells absent from the reference are skipped.
fn confident_call_agreement(result: &arrow::record_batch::RecordBatch, reference: &HashMap<String, Vec<String>>) -> f64 {
    let barcodes = result.column_by_name("cell_barcode").unwrap()
        .as_any().downcast_ref::<StringArray>().unwrap();
    let guide_ids = result.column_by_name("guide_id").unwrap();
    let is_unassigned = result.column_by_name("is_unassigned").unwrap()
        .as_any().downcast_ref::<BooleanArray>().unwrap();

    let mut agreed = 0usize;
    let mut compared = 0usize;

    for i in 0..result.num_rows() {
        if is_unassigned.value(i) {
            continue;
        }
        let barcode = barcodes.value(i);
        let Some(ref_grnas) = reference.get(barcode) else { continue };

        // guide_id is a Dictionary array — cast via Any to get the string value
        use arrow::array::DictionaryArray;
        use arrow::datatypes::Int16Type;
        let dict = guide_ids.as_any().downcast_ref::<DictionaryArray<Int16Type>>().unwrap();
        let values = dict.values().as_any().downcast_ref::<StringArray>().unwrap();
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

// ---------------------------------------------------------------------------
// Fixture loading
// ---------------------------------------------------------------------------

fn load_input() -> AssignmentInput {
    let h5ad = common::schraivogel_h5ad();
    read_h5ad(&h5ad)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", h5ad.display()))
}

// ---------------------------------------------------------------------------
// Equivalence tests (≥99% confident-call agreement)
// ---------------------------------------------------------------------------

const MIN_AGREEMENT: f64 = 0.99;

#[test]
fn umi_equivalence() {
    let input = load_input();
    let model = UmiModel::default();
    let result = model.assign(&input).unwrap();
    let reference = load_reference("umi");
    let agreement = confident_call_agreement(&result, &reference);
    assert!(
        agreement >= MIN_AGREEMENT,
        "umi: agreement {:.4} < {MIN_AGREEMENT}",
        agreement
    );
}

#[test]
fn max_equivalence() {
    let input = load_input();
    let model = MaxModel::default();
    let result = model.assign(&input).unwrap();
    let reference = load_reference("max");
    let agreement = confident_call_agreement(&result, &reference);
    assert!(
        agreement >= MIN_AGREEMENT,
        "max: agreement {:.4} < {MIN_AGREEMENT}",
        agreement
    );
}

#[test]
fn ratio_equivalence() {
    let input = load_input();
    let model = RatioModel::default();
    let result = model.assign(&input).unwrap();
    let reference = load_reference("ratio");
    let agreement = confident_call_agreement(&result, &reference);
    assert!(
        agreement >= MIN_AGREEMENT,
        "ratio: agreement {:.4} < {MIN_AGREEMENT}",
        agreement
    );
}
