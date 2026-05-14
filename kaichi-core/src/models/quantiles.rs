use super::AssignmentModel;
use super::output::{n_detected_u8, AssignmentOutputBuilder};
use crate::data::{AssignmentResult, LoadedInput};

use anyhow::Result;
use rayon::prelude::*;
use serde_json::{json, Value};

/// Rank-based guide assignment: top `quantile` fraction of cells per guide by guide proportion.
///
/// For each guide, cells are ranked by p_{i,g} = y_{i,g} / total_counts_i.
/// The top ⌊quantile · N_g⌋ cells (N_g = nonzero cells for this guide) are assigned.
/// Confidence is the proportion value. Multi-infection is flagged when a cell is in
/// the top-k for two or more guides.
pub struct QuantilesModel {
    pub quantile: f32,
}

impl Default for QuantilesModel {
    fn default() -> Self {
        Self { quantile: 0.05 }
    }
}

impl AssignmentModel for QuantilesModel {
    fn name(&self) -> &'static str {
        "quantiles"
    }

    fn assign(&self, input: &LoadedInput) -> Result<AssignmentResult> {
        let n_cells = input.counts.n_cells;
        let n_guides = input.counts.n_guides;
        let csc = input.counts.csc();
        let total_counts = &input.covariates.total_counts;

        let total_f64: Vec<f64> = (0..n_cells)
            .map(|i| total_counts.value(i) as f64)
            .collect();

        // Per-guide minimum proportion threshold: proportion at rank k = ⌊q * N_g⌋.
        let prop_thresholds: Vec<Option<f64>> = (0..n_guides)
            .into_par_iter()
            .map(|g| {
                let col = csc.get_col(g).unwrap();
                let mut props: Vec<f64> = col
                    .row_indices()
                    .iter()
                    .zip(col.values())
                    .filter(|(&i, &v)| v > 0 && total_f64[i] > 0.0)
                    .map(|(&i, &v)| v as f64 / total_f64[i])
                    .collect();
                let n_g = props.len();
                let k = ((self.quantile as f64) * n_g as f64).floor() as usize;
                if k == 0 { return None; }
                props.sort_by(f64::total_cmp);
                Some(props[n_g - k]) // k-th largest (0-indexed from end)
            })
            .collect();

        let csr = input.counts.csr();
        let guide_ids_arr = &input.guide_metadata.guide_ids;
        let cell_barcodes = &input.covariates.cell_barcodes;
        let mut out = AssignmentOutputBuilder::new(n_cells, n_guides, self.name());

        for cell in 0..n_cells {
            let total = total_f64[cell];
            let row = csr.get_row(cell).unwrap();
            let mut best: Option<(usize, f32, u32)> = None;
            let mut n_passing: usize = 0;

            if total > 0.0 {
                for (&guide_idx, &count) in row.col_indices().iter().zip(row.values()) {
                    let Some(threshold) = prop_thresholds[guide_idx] else { continue };
                    let prop = count as f64 / total;
                    if prop >= threshold {
                        n_passing += 1;
                        let conf = prop as f32;
                        match best {
                            None => best = Some((guide_idx, conf, count)),
                            Some((_, bp, bc)) => {
                                if conf > bp || (conf == bp && count > bc) {
                                    best = Some((guide_idx, conf, count));
                                }
                            }
                        }
                        out.push_assigned_triple(cell, guide_idx);
                    }
                }
            }

            let n_det = n_detected_u8(n_passing);
            match best {
                None => out.append_unassigned(n_det),
                Some((guide_idx, conf, count)) => {
                    out.append_assigned(
                        guide_ids_arr.value(guide_idx),
                        count,
                        conf,
                        n_passing > 1,
                        n_det,
                    );
                }
            }
        }

        out.finish(cell_barcodes, true)
    }

    fn params_json(&self) -> Value {
        json!({ "quantile": self.quantile })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{CountMatrix, Covariates, GuideMetadata};
    use arrow::array::{BooleanArray, Float32Array, StringBuilder};

    fn make_input(n_cells: usize, n_guides: usize, triples: Vec<(usize, usize, u32)>) -> LoadedInput {
        let mut sorted = triples;
        sorted.sort_unstable_by_key(|&(r, c, _)| (r, c));
        let nnz = sorted.len();
        let mut row_offsets = vec![0usize; n_cells + 1];
        let mut col_indices = Vec::with_capacity(nnz);
        let mut values = Vec::with_capacity(nnz);
        let mut last = 0usize;
        let mut cell_totals = vec![0u32; n_cells];
        for (idx, &(r, c, v)) in sorted.iter().enumerate() {
            while last < r { row_offsets[last + 1] = idx; last += 1; }
            col_indices.push(c);
            values.push(v);
            cell_totals[r] += v;
        }
        for i in (last + 1)..=n_cells { row_offsets[i] = nnz; }
        let counts = CountMatrix::try_from_csr(n_cells, n_guides, row_offsets, col_indices, values).unwrap();
        let mut bc = StringBuilder::new();
        for i in 0..n_cells { bc.append_value(format!("C{i}")); }
        let mut gd = StringBuilder::new();
        for i in 0..n_guides { gd.append_value(format!("g{i}")); }
        LoadedInput {
            counts,
            covariates: Covariates {
                cell_barcodes: bc.finish(),
                total_counts: Float32Array::from(cell_totals.iter().map(|&t| t as f32).collect::<Vec<_>>()),
            },
            guide_metadata: GuideMetadata { guide_ids: gd.finish() },
        }
    }

    fn make_input_with_totals(
        n_cells: usize,
        n_guides: usize,
        triples: Vec<(usize, usize, u32)>,
        totals: Vec<u32>,
    ) -> LoadedInput {
        let mut sorted = triples;
        sorted.sort_unstable_by_key(|&(r, c, _)| (r, c));
        let nnz = sorted.len();
        let mut row_offsets = vec![0usize; n_cells + 1];
        let mut col_indices = Vec::with_capacity(nnz);
        let mut values = Vec::with_capacity(nnz);
        let mut last = 0usize;
        for (idx, &(r, c, v)) in sorted.iter().enumerate() {
            while last < r { row_offsets[last + 1] = idx; last += 1; }
            col_indices.push(c);
            values.push(v);
        }
        for i in (last + 1)..=n_cells { row_offsets[i] = nnz; }
        let counts = CountMatrix::try_from_csr(n_cells, n_guides, row_offsets, col_indices, values).unwrap();
        let mut bc = StringBuilder::new();
        for i in 0..n_cells { bc.append_value(format!("C{i}")); }
        let mut gd = StringBuilder::new();
        for i in 0..n_guides { gd.append_value(format!("g{i}")); }
        LoadedInput {
            counts,
            covariates: Covariates {
                cell_barcodes: bc.finish(),
                total_counts: Float32Array::from(totals.iter().map(|&t| t as f32).collect::<Vec<_>>()),
            },
            guide_metadata: GuideMetadata { guide_ids: gd.finish() },
        }
    }

    fn is_unassigned(r: &AssignmentResult) -> &BooleanArray {
        r.batch.column_by_name("is_unassigned").unwrap()
            .as_any().downcast_ref::<BooleanArray>().unwrap()
    }

    #[test]
    fn top_fraction_assigned() {
        // 5 cells, 1 guide, counts [1,2,3,4,5], totals all = 10 → props [0.1,0.2,0.3,0.4,0.5].
        // quantile=0.4 → k=floor(0.4*5)=2. Sorted ascending: threshold = props[5-2]=props[3]=0.4.
        // Cells with prop >= 0.4: cells 3 (0.4) and 4 (0.5). Cells 0,1,2 unassigned.
        let input = make_input_with_totals(
            5, 1,
            (0..5).map(|c| (c, 0, (c + 1) as u32)).collect(),
            vec![10; 5],
        );
        let model = QuantilesModel { quantile: 0.4 };
        let result = model.assign(&input).unwrap();
        let is_u = is_unassigned(&result);
        assert!(is_u.value(0));
        assert!(is_u.value(1));
        assert!(is_u.value(2));
        assert!(!is_u.value(3));
        assert!(!is_u.value(4));
    }

    #[test]
    fn multi_guide_cells_flagged_multi() {
        // 6 cells × 2 guides. Each guide has 3 cells with high counts.
        // Cell 2 is in top-k for both guides → multi-infected.
        // Guide 0: cells 0,1,2 have count 10 (total=10 so prop=1.0).
        // Guide 1: cells 2,3,4 have count 10 (total=20 so prop=0.5).
        // But totals include both guides. Let me construct carefully.
        // Cell 0: g0=10, g1=0 → total=10, prop_g0=1.0
        // Cell 1: g0=10, g1=0 → total=10, prop_g0=1.0
        // Cell 2: g0=10, g1=10 → total=20, prop_g0=0.5, prop_g1=0.5
        // Cell 3: g0=0, g1=10 → total=10, prop_g1=1.0
        // Cell 4: g0=0, g1=10 → total=10, prop_g1=1.0
        // Cell 5: g0=0, g1=0 → unassigned
        let input = make_input(6, 2, vec![
            (0,0,10), (1,0,10), (2,0,10), (2,1,10), (3,1,10), (4,1,10),
        ]);
        // quantile=0.5 → top 50% per guide.
        // Guide 0 nonzero cells: cells 0,1,2 (N_g=3, k=1). Proportions: 1.0,1.0,0.5.
        //   Sorted desc: [1.0,1.0,0.5]. Threshold = props[3-1] = 1.0. Only cells 0,1 pass (prop=1.0 >= 1.0). Cell 2 has prop=0.5 < 1.0.
        // Actually wait: sorted ascending [0.5,1.0,1.0], n_g-k = 3-1 = 2, props[2] = 1.0. Cells with prop >= 1.0: cells 0 and 1.
        // Guide 1 nonzero cells: cells 2,3,4 (N_g=3, k=1). Proportions: 0.5,1.0,1.0.
        //   Threshold = 1.0. Cells 3,4 pass (prop=1.0). Cell 2 has prop=0.5 < 1.0.
        // So no multi-infection with this threshold. Let me use quantile=0.7.
        // Guide 0: k = floor(0.7*3) = 2. Threshold = props[3-2] = props[1] = 1.0. Same as before... hmm.
        // Actually all nonzero props for guide 0 are {1.0, 1.0, 0.5}. Sorted: [0.5, 1.0, 1.0].
        // k=2 → props[3-2]=props[1]=1.0. Still only cells 0,1 pass.
        // Let me use quantile=1.0 to get all 3 nonzero cells.
        // k = floor(1.0*3) = 3, threshold = props[3-3]=props[0]=0.5. All 3 cells pass.
        let model = QuantilesModel { quantile: 1.0 };
        let result = model.assign(&input).unwrap();
        let is_multi = result.batch.column_by_name("is_multi_infected").unwrap()
            .as_any().downcast_ref::<BooleanArray>().unwrap();
        assert!(is_multi.value(2), "cell 2 is in top-k for both guides");
    }

    #[test]
    fn zero_quantile_assigns_nothing() {
        // k = floor(0 * N_g) = 0 → None for all guides → all unassigned.
        let input = make_input(5, 1, (0..5).map(|c| (c, 0, (c + 1) as u32)).collect());
        let model = QuantilesModel { quantile: 0.0 };
        let result = model.assign(&input).unwrap();
        for i in 0..5 { assert!(is_unassigned(&result).value(i)); }
    }

    #[test]
    fn empty_input_produces_empty_result() {
        let input = make_input(0, 0, vec![]);
        let result = QuantilesModel::default().assign(&input).unwrap();
        assert_eq!(result.batch.num_rows(), 0);
    }

    #[test]
    fn unassigned_columns_are_null() {
        use arrow::array::Array;
        // quantile=0 → all unassigned.
        let input = make_input(3, 1, vec![(1,0,5),(2,0,3)]);
        let model = QuantilesModel { quantile: 0.0 };
        let result = model.assign(&input).unwrap();
        for i in 0..3 {
            assert!(result.batch.column_by_name("guide_id").unwrap().is_null(i));
            assert!(result.batch.column_by_name("umi_count").unwrap().is_null(i));
            assert!(result.batch.column_by_name("assignment_confidence").unwrap().is_null(i));
        }
    }
}
