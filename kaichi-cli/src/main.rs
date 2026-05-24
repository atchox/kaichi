use anyhow::{bail, Result};
use clap::{Parser, Subcommand};
use kaichi_core::models::{em_count_mixture::decide_threshold, TwoStage};
use kaichi_core::score::ScoreMatrix;
use std::path::PathBuf;
use std::time::Instant;

const VERSION: &str = git_version::git_version!(fallback = env!("CARGO_PKG_VERSION"));

#[derive(Parser)]
#[command(name = "kaichi", version = VERSION, about = "CRISPR guide assignment for Perturb-seq")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Assign CRISPR guides to cells (one-shot: score + decide)
    Assign {
        /// Input: guide-count H5AD
        #[arg(long)]
        counts: PathBuf,
        /// Assignment model
        #[arg(long, default_value = "poisson_gauss")]
        model: String,
        /// Output path (.h5ad or .csv)
        #[arg(long)]
        output: Option<PathBuf>,
        /// Number of worker threads. 0 = half of logical cores.
        #[arg(long, default_value_t = 0)]
        threads: usize,
    },
    /// Fit a two-stage model and cache the score matrix to an H5AD.
    ///
    /// Output H5AD layout: X = preserved UMI counts, layers/scores = float32
    /// posteriors, uns/kaichi tracks model name and fitted params. Reject
    /// single-stage models (umi/ratio/max) — use `assign` for those.
    Score {
        /// Input: guide-count H5AD
        #[arg(long)]
        counts: PathBuf,
        /// Two-stage model: poisson_gauss, poisson, neg_binomial, binomial
        #[arg(long, default_value = "poisson_gauss")]
        model: String,
        /// Output: scored H5AD
        #[arg(long)]
        output: PathBuf,
        /// Number of worker threads. 0 = half of logical cores.
        #[arg(long, default_value_t = 0)]
        threads: usize,
    },
    /// Threshold a cached score H5AD into final assignments.
    ///
    /// Model identity and fitted params are pulled from uns/kaichi. Output
    /// extends the input with a `layers/assigned` group plus assignment obs
    /// columns; `--in-place` overwrites the input file.
    Decide {
        /// Input: scored H5AD (from `kaichi score`)
        #[arg(long)]
        scores: PathBuf,
        /// Posterior threshold in [0, 1]
        #[arg(long)]
        min_confidence: f32,
        /// Output H5AD path (mutually exclusive with --in-place)
        #[arg(long)]
        output: Option<PathBuf>,
        /// Overwrite the input scored H5AD with the decided result
        #[arg(long, default_value_t = false, conflicts_with = "output")]
        in_place: bool,
        /// Number of worker threads. 0 = half of logical cores.
        #[arg(long, default_value_t = 0)]
        threads: usize,
    },
}

/// Resolve the requested thread count. `0` means "half of total logical cores"
/// (HPC-friendly default); any positive value is honored as-is.
fn resolve_threads(requested: usize) -> usize {
    if requested > 0 {
        return requested;
    }
    let total = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    (total / 2).max(1)
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Assign { counts, model, output, threads } => {
            init_threads(threads)?;
            eprintln!("reading {}", counts.display());
            let t0 = Instant::now();
            let input = kaichi_core::io::read::read_h5ad(&counts)?;
            eprintln!(
                "loaded {} cells × {} guides ({} nnz) in {:.2}s",
                input.counts.n_cells,
                input.counts.n_guides,
                input.counts.nnz(),
                t0.elapsed().as_secs_f64()
            );

            eprintln!("running model: {model}");
            let t1 = Instant::now();
            let result = run_model(&model, &input)?;
            eprintln!("assigned {} cells in {:.2}s", result.batch.num_rows(), t1.elapsed().as_secs_f64());

            if let Some(out_path) = output {
                match out_path.extension().and_then(|e| e.to_str()) {
                    Some("h5ad") => {
                        let params = run_model_params(&model, &input)?;
                        kaichi_core::io::write::write_h5ad(
                            &input, &result, &out_path, &model, &params,
                        )?;
                    }
                    _ => write_csv(&result, &out_path)?,
                }
                eprintln!("wrote {}", out_path.display());
            }
        }
        Commands::Score { counts, model, output, threads } => {
            init_threads(threads)?;
            eprintln!("reading {}", counts.display());
            let t0 = Instant::now();
            let input = kaichi_core::io::read::read_h5ad(&counts)?;
            eprintln!(
                "loaded {} cells × {} guides ({} nnz) in {:.2}s",
                input.counts.n_cells,
                input.counts.n_guides,
                input.counts.nnz(),
                t0.elapsed().as_secs_f64()
            );

            eprintln!("scoring with model: {model}");
            let t1 = Instant::now();
            let scores = score_model(&model, &input)?;
            eprintln!("scored {} cells in {:.2}s", scores.n_cells(), t1.elapsed().as_secs_f64());

            kaichi_core::io::write::write_scored_h5ad(&scores, &output)?;
            eprintln!("wrote {}", output.display());
        }
        Commands::Decide { scores, min_confidence, output, in_place, threads } => {
            init_threads(threads)?;
            let out_path = match (output, in_place) {
                (Some(p), false) => p,
                (None, true) => scores.clone(),
                (None, false) => bail!("must specify --output PATH or --in-place"),
                (Some(_), true) => unreachable!("clap enforces mutual exclusion"),
            };

            eprintln!("reading {}", scores.display());
            let t0 = Instant::now();
            let score_matrix = kaichi_core::io::read::read_scored_h5ad(&scores)?;
            eprintln!(
                "loaded {} cells × {} guides ({} nnz, model={}) in {:.2}s",
                score_matrix.n_cells(),
                score_matrix.n_guides(),
                score_matrix.nnz(),
                score_matrix.model.name(),
                t0.elapsed().as_secs_f64()
            );

            eprintln!("deciding at min_confidence={min_confidence}");
            let t1 = Instant::now();
            let result = decide_threshold(&score_matrix, min_confidence)?;
            eprintln!("decided {} cells in {:.2}s", result.batch.num_rows(), t1.elapsed().as_secs_f64());

            kaichi_core::io::write::write_assigned_from_scored(
                &score_matrix, &result, &out_path, min_confidence,
            )?;
            eprintln!("wrote {}", out_path.display());
        }
    }
    Ok(())
}

fn init_threads(requested: usize) -> Result<()> {
    let n_threads = resolve_threads(requested);
    rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build_global()
        .map_err(|e| anyhow::anyhow!("rayon thread pool init failed: {e}"))?;
    eprintln!("using {n_threads} thread(s)");
    Ok(())
}

/// Two-stage models only. Single-stage models (umi/ratio/max) fail with a
/// clear pointer to `assign`.
fn score_model(name: &str, input: &kaichi_core::data::LoadedInput) -> Result<ScoreMatrix> {
    use kaichi_core::models::{
        binomial::BinomialModel, neg_binomial::NegBinomialModel,
        poisson::PoissonModel, poisson_gauss::PoissonGaussModel,
    };
    let scores = match name {
        "poisson_gauss" => PoissonGaussModel::default().score(input)?,
        "poisson"       => PoissonModel::default().score(input)?,
        "neg_binomial"  => NegBinomialModel::default().score(input)?,
        "binomial"      => BinomialModel::default().score(input)?,
        "umi" | "ratio" | "max" => bail!(
            "model {name:?} is single-stage and does not support `score`. \
             Use `kaichi assign` instead."
        ),
        other => bail!(
            "unknown model: {other}. Two-stage models: poisson_gauss, poisson, \
             neg_binomial, binomial"
        ),
    };
    Ok(scores)
}

fn run_model(
    name: &str,
    input: &kaichi_core::data::LoadedInput,
) -> Result<kaichi_core::data::AssignmentResult> {
    use kaichi_core::models::AssignmentModel;
    match name {
        "umi"           => kaichi_core::models::umi::UmiModel::default().assign(input),
        "max"           => kaichi_core::models::max::MaxModel::default().assign(input),
        "ratio"         => kaichi_core::models::ratio::RatioModel::default().assign(input),
        "poisson_gauss" => kaichi_core::models::poisson_gauss::PoissonGaussModel::default().assign(input),
        "poisson"       => kaichi_core::models::poisson::PoissonModel::default().assign(input),
        "neg_binomial"  => kaichi_core::models::neg_binomial::NegBinomialModel::default().assign(input),
        "binomial"      => kaichi_core::models::binomial::BinomialModel::default().assign(input),
        "beta2"         => kaichi_core::models::beta::Beta2Model::default().assign(input),
        "beta3"         => kaichi_core::models::beta::Beta3Model::default().assign(input),
        "quantiles"     => kaichi_core::models::quantiles::QuantilesModel::default().assign(input),
        other           => bail!("unknown model: {other}"),
    }
}

fn run_model_params(name: &str, _input: &kaichi_core::data::LoadedInput) -> Result<String> {
    use kaichi_core::models::AssignmentModel;
    let v = match name {
        "umi"           => kaichi_core::models::umi::UmiModel::default().params_json(),
        "max"           => kaichi_core::models::max::MaxModel::default().params_json(),
        "ratio"         => kaichi_core::models::ratio::RatioModel::default().params_json(),
        "poisson_gauss" => kaichi_core::models::poisson_gauss::PoissonGaussModel::default().params_json(),
        "poisson"       => kaichi_core::models::poisson::PoissonModel::default().params_json(),
        "neg_binomial"  => kaichi_core::models::neg_binomial::NegBinomialModel::default().params_json(),
        "binomial"      => kaichi_core::models::binomial::BinomialModel::default().params_json(),
        "beta2"         => kaichi_core::models::beta::Beta2Model::default().params_json(),
        "beta3"         => kaichi_core::models::beta::Beta3Model::default().params_json(),
        "quantiles"     => kaichi_core::models::quantiles::QuantilesModel::default().params_json(),
        other           => bail!("unknown model: {other}"),
    };
    Ok(v.to_string())
}

fn write_csv(result: &kaichi_core::data::AssignmentResult, path: &std::path::Path) -> Result<()> {
    use arrow::array::{Array, BooleanArray, StringArray, UInt32Array};
    use arrow::compute::cast;
    use arrow::datatypes::DataType;
    use std::fs::File;
    use std::io::{BufWriter, Write};

    let batch = &result.batch;
    let barcodes = batch.column_by_name("cell_barcode").unwrap()
        .as_any().downcast_ref::<StringArray>().unwrap();
    let guide_col = cast(batch.column_by_name("guide_id").unwrap().as_ref(), &DataType::Utf8)?;
    let guide_ids = guide_col.as_any().downcast_ref::<StringArray>().unwrap();
    let umi_counts = batch.column_by_name("umi_count").unwrap()
        .as_any().downcast_ref::<UInt32Array>().unwrap();
    let is_unassigned = batch.column_by_name("is_unassigned").unwrap()
        .as_any().downcast_ref::<BooleanArray>().unwrap();

    let mut w = BufWriter::new(File::create(path)?);
    writeln!(w, "cell,gRNA,UMI_counts")?;
    for i in 0..batch.num_rows() {
        if is_unassigned.value(i) { continue; }
        let cell = barcodes.value(i);
        let guide = guide_ids.value(i);
        let umi = if umi_counts.is_null(i) { 0 } else { umi_counts.value(i) };
        writeln!(w, "{cell},{guide},{umi}")?;
    }
    Ok(())
}
