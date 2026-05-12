use anyhow::{bail, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "kaichi", version, about = "CRISPR guide assignment for Perturb-seq")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Assign CRISPR guides to cells
    Assign {
        /// Input H5AD file (guide count matrix)
        #[arg(long)]
        counts: PathBuf,
        /// Assignment model
        #[arg(long, default_value = "poisson_gauss")]
        model: String,
        /// Output CSV file: cell, gRNA, UMI_counts (omit to run without writing)
        #[arg(long)]
        output: Option<PathBuf>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Assign { counts, model, output } => {
            eprintln!("reading {}", counts.display());
            let t0 = Instant::now();
            let input = kaichi_core::io::h5ad::read_h5ad(&counts)?;
            let n_cells = input.covariates.num_rows();
            let n_nnz = input.counts.num_rows();
            eprintln!("loaded {n_cells} cells, {n_nnz} non-zero entries in {:.2}s", t0.elapsed().as_secs_f64());

            eprintln!("running model: {model}");
            let t1 = Instant::now();
            let result = run_model(&model, &input)?;
            eprintln!("assigned {} cells in {:.2}s", result.num_rows(), t1.elapsed().as_secs_f64());

            if let Some(out_path) = output {
                write_csv(&result, &out_path)?;
                eprintln!("wrote {}", out_path.display());
            }
        }
    }
    Ok(())
}

fn run_model(
    name: &str,
    input: &kaichi_core::models::AssignmentInput,
) -> Result<arrow::record_batch::RecordBatch> {
    use kaichi_core::models::AssignmentModel;
    match name {
        "umi"           => kaichi_core::models::umi::UmiModel::default().assign(input),
        "max"           => kaichi_core::models::max::MaxModel::default().assign(input),
        "ratio"         => kaichi_core::models::ratio::RatioModel::default().assign(input),
        "poisson_gauss" => kaichi_core::models::poisson_gauss::PoissonGaussModel::default().assign(input),
        other           => bail!("unknown model: {other}"),
    }
}

fn write_csv(batch: &arrow::record_batch::RecordBatch, path: &std::path::Path) -> Result<()> {
    use arrow::array::{Array, BooleanArray, StringArray, UInt32Array};
    use arrow::compute::cast;
    use arrow::datatypes::DataType;
    use std::fs::File;
    use std::io::{BufWriter, Write};

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
