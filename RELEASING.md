# Releasing kaichi

## One-time setup

### 1. Install cargo-release
```bash
cargo install cargo-release
```

### 2. Configure PyPI trusted publishing
On PyPI, go to the `kaichi` project → *Publishing* → *Add a new publisher* and
set:
- Owner: `<your-github-org>`
- Repository: `kaichi`
- Workflow: `release.yml`
- Environment: `pypi`

This lets GitHub Actions publish to PyPI without storing an API token.

### 3. Create the GitHub environment
In the repo settings → *Environments*, create an environment named `pypi`.
Optionally add a required reviewer so every publish needs a manual approval.

---

## Cutting a release

```bash
# Patch: 0.1.0 → 0.1.1
cargo release patch

# Minor: 0.1.0 → 0.2.0
cargo release minor

# Major: 0.1.0 → 1.0.0
cargo release major
```

`cargo-release` will:
1. Run `cargo test -p kaichi-core -p kaichi-cli`
2. Bump the version in the workspace `Cargo.toml`
3. Commit the change
4. Tag the commit `v<version>`
5. Push the commit and tag to GitHub

GitHub Actions then triggers automatically on the tag and:
1. Builds wheels for Linux x86_64, macOS arm64, and macOS x86_64
2. Builds a source distribution
3. Publishes everything to PyPI

---

## Version strings

| Context | Value at tag `v0.2.0` | Value between tags |
|---|---|---|
| `kaichi.__version__` | `v0.2.0` | `v0.1.0-3-gabcdef1` |
| PyPI / wheel metadata | `0.2.0` | *(not published)* |

Both come from the same source: `Cargo.toml` for the wheel version, and
`git describe` (via `git_version!()`) for the runtime string. They agree
exactly at a tag.

---

## Platforms

| Platform | Runner | Notes |
|---|---|---|
| Linux x86_64 | `ubuntu-latest` (manylinux) | HDF5 from yum |
| macOS arm64 | `macos-latest` | HDF5 from Homebrew, bundled by delocate |
| macOS x86_64 | `macos-13` | HDF5 from Homebrew, bundled by delocate |

Windows is not currently supported.
