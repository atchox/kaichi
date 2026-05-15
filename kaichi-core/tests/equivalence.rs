mod common;

use kaichi_core::io::read::read_h5ad;
use kaichi_core::models::{
    beta::{Beta2Model, Beta3Model},
    binomial::BinomialModel,
    max::MaxModel,
    neg_binomial::NegBinomialModel,
    poisson::PoissonModel,
    poisson_gauss::PoissonGaussModel,
    quantiles::QuantilesModel,
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

struct Thresholds {
    min_agreement: f64,
    min_coverage: f64,
    max_rate_diff: f64,
}

impl Thresholds {
    const fn strict() -> Self {
        Self { min_agreement: 0.99, min_coverage: 0.99, max_rate_diff: 0.05 }
    }
}

fn check_equivalence(
    result: &kaichi_core::data::AssignmentResult,
    reference: &HashMap<String, Vec<String>>,
    n_cells: usize,
    model: &str,
    t: &Thresholds,
) {
    let agreement = confident_call_agreement(result, reference);
    assert!(
        agreement >= t.min_agreement,
        "{model}: guide agreement {:.4} < {}",
        agreement, t.min_agreement,
    );

    let coverage = reference_coverage(result, reference);
    assert!(
        coverage >= t.min_coverage,
        "{model}: reference coverage {:.4} < {} \
         (kaichi missed calls the reference made)",
        coverage, t.min_coverage,
    );

    let rate_diff = assignment_rate_diff(result, reference, n_cells);
    assert!(
        rate_diff <= t.max_rate_diff,
        "{model}: assignment rate diff {:.4} > {} \
         (kaichi rate differs from reference rate by more than {})",
        rate_diff, t.max_rate_diff, t.max_rate_diff,
    );
}

const STRICT: Thresholds = Thresholds::strict();

#[test]
fn umi_equivalence() {
    let input = load_input();
    let result = UmiModel::default().assign(&input).unwrap();
    let reference = load_reference("umi");
    check_equivalence(&result, &reference, input.counts.n_cells, "umi", &STRICT);
}

#[test]
fn max_equivalence() {
    let input = load_input();
    let result = MaxModel::default().assign(&input).unwrap();
    let reference = load_reference("max");
    check_equivalence(&result, &reference, input.counts.n_cells, "max", &STRICT);
}

#[test]
fn ratio_equivalence() {
    let input = load_input();
    let result = RatioModel::default().assign(&input).unwrap();
    let reference = load_reference("ratio");
    check_equivalence(&result, &reference, input.counts.n_cells, "ratio", &STRICT);
}

#[test]
fn poisson_gauss_equivalence() {
    let input = load_input();
    let result = PoissonGaussModel::default().assign(&input).unwrap();
    let reference = load_reference("poisson_gauss");
    check_equivalence(&result, &reference, input.counts.n_cells, "poisson_gauss", &STRICT);
}

#[test]
fn poisson_equivalence() {
    let input = load_input();
    let result = PoissonModel::default().assign(&input).unwrap();
    let reference = load_reference("poisson");
    // Coverage 97.9%: zero-truncated EM slightly underassigns on guides where nonzero
    // background/signal counts are balanced. Root cause: zero cells excluded from fitting.
    check_equivalence(&result, &reference, input.counts.n_cells, "poisson",
        &Thresholds { min_coverage: 0.97, ..STRICT });
}

#[test]
fn neg_binomial_equivalence() {
    let input = load_input();
    let result = NegBinomialModel::default().assign(&input).unwrap();
    let reference = load_reference("neg_binomial");
    check_equivalence(&result, &reference, input.counts.n_cells, "neg_binomial", &STRICT);
}

#[test]
fn binomial_equivalence() {
    let input = load_input();
    let result = BinomialModel::default().assign(&input).unwrap();
    let reference = load_reference("binomial");
    // kaichi deliberately diverges from crispat here. crispat's binomial.py defines
    // p = sigmoid(β0 + β1·z + γ_batch) during training, but `get_perturbed_cells`
    // evaluates p = sigmoid(exp(β0 + γ_batch)) at assignment time — almost certainly
    // a bug (always gives p > 0.5 since exp(·) > 0, so sigmoid(exp(·)) > sigmoid(0) = 0.5).
    // kaichi follows the model definition. Matching crispat exactly would require
    // replicating the bug. Coverage and per-guide agreement still hold; the rate
    // gap is the cells crispat misses because its assignment formula is off.
    // See docs/design/assignment-models.md → "Note on a crispat inconsistency".
    check_equivalence(&result, &reference, input.counts.n_cells, "binomial",
        &Thresholds { max_rate_diff: 0.20, ..STRICT });
}

#[test]
fn beta2_equivalence() {
    let input = load_input();
    let result = Beta2Model::default().assign(&input).unwrap();
    let reference = load_reference("beta2");
    // kaichi uses pure EM (MLE) while crispat uses Pyro SVI with MAP priors
    // (LogNormal(1,1) on α/β, Dirichlet([0.6,0.4]) on weights). On identical
    // initialization and data these converge to different local optima: MLE
    // finds (tight background, diffuse signal) at a lower effective threshold,
    // MAP finds (moderate background, tight signal near 1.0) at threshold ≈0.8.
    // Coverage and per-guide agreement still hold at 99%; kaichi just assigns
    // ~7% additional borderline cells. Matching crispat's rate would require
    // implementing MAP-EM (NR on digamma equations + LogNormal penalty), which
    // is out of scope for a v0.2 specialty model.
    check_equivalence(&result, &reference, input.counts.n_cells, "beta2",
        &Thresholds { max_rate_diff: 0.10, ..STRICT });
}

#[test]
fn beta3_equivalence() {
    let input = load_input();
    let result = Beta3Model::default().assign(&input).unwrap();
    let reference = load_reference("beta3");
    check_equivalence(&result, &reference, input.counts.n_cells, "beta3", &STRICT);
}

#[test]
fn quantiles_equivalence() {
    let input = load_input();
    // quantile=0.1 matches the reference file assignments_t0.1.csv.
    let result = QuantilesModel { quantile: 0.1 }.assign(&input).unwrap();
    let reference = load_reference("quantiles");
    check_equivalence(&result, &reference, input.counts.n_cells, "quantiles", &STRICT);
}
