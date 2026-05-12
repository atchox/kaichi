use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

const CRISPAT_URL: &str =
    "https://github.com/velten-group/crispat/archive/refs/heads/main.tar.gz";

static FIXTURES: OnceLock<PathBuf> = OnceLock::new();

/// Returns the path to the extracted crispat fixture directory, downloading it
/// first if it doesn't exist. Safe to call from multiple tests — download runs
/// once per process.
pub fn crispat_dir() -> &'static Path {
    FIXTURES.get_or_init(|| {
        let dest = fixture_root().join("crispat");
        let marker = dest.join(".ready");
        if !marker.exists() {
            download_fixture(&dest);
            std::fs::write(&marker, "").expect("failed to write marker");
        }
        dest
    })
}

pub fn schraivogel_h5ad() -> PathBuf {
    crispat_dir().join("example_data/Schraivogel/gRNA_counts.h5ad")
}

pub fn reference_assignments(model: &str) -> PathBuf {
    let (dir, file) = match model {
        "umi"           => ("UMI",               "assignments_t5.csv"),
        "max"           => ("maximum",            "assignments.csv"),
        "ratio"         => ("ratios",             "assignments_t0.3.csv"),
        "neg_binomial"  => ("negative_binomial",  "assignments.csv"),
        "poisson_gauss" => ("poisson_gauss",      "assignments.csv"),
        other => panic!("no reference fixture registered for model '{}'", other),
    };
    crispat_dir()
        .join("example_data/guide_assignments")
        .join(dir)
        .join(file)
}

fn fixture_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("kaichi-core has no parent dir")
        .join("tests/fixtures")
}

fn download_fixture(dest: &Path) {
    if dest.exists() {
        std::fs::remove_dir_all(dest).expect("failed to remove stale fixture dir");
    }
    std::fs::create_dir_all(dest).expect("failed to create fixture dir");

    let tarball = dest.with_extension("tar.gz");

    run("curl", &[
        "-fsSL",
        CRISPAT_URL,
        "-o", tarball.to_str().expect("fixture path is not valid UTF-8"),
    ]);

    run("tar", &[
        "-xzf", tarball.to_str().expect("fixture path is not valid UTF-8"),
        "-C", dest.to_str().expect("fixture path is not valid UTF-8"),
        "--strip-components=1",
    ]);

    std::fs::remove_file(&tarball).ok();
}

fn run(program: &str, args: &[&str]) {
    let status = Command::new(program)
        .args(args)
        .status()
        .unwrap_or_else(|e| panic!("failed to run `{program}`: {e}"));
    assert!(status.success(), "`{program} {}` failed", args.join(" "));
}
