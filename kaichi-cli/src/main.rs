use anyhow::Result;
use clap::{Parser, Subcommand};

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
        #[arg(long)]
        counts: String,
        #[arg(long)]
        guide_library: String,
        #[arg(long)]
        output: String,
        #[arg(long, default_value = "neg_binomial")]
        model: String,
    },
    /// Compare two sets of guide assignments
    Validate {
        #[arg(long)]
        a: String,
        #[arg(long)]
        b: String,
        #[arg(long, default_value_t = 0.99)]
        tolerance: f64,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Assign { .. } => {
            eprintln!("kaichi assign: not yet implemented");
        }
        Commands::Validate { .. } => {
            eprintln!("kaichi validate: not yet implemented");
        }
    }
    Ok(())
}
