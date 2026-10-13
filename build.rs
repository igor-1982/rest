extern crate dunce;
use std::env;
use std::path::PathBuf;
use std::process::Command;

/// Run a `git` subcommand inside the package root and return its trimmed stdout.
/// Returns `None` whenever git is unavailable, the directory is not a repository,
/// or the command fails, so that a build never breaks just because the revision
/// cannot be determined.
fn run_git(args: &[&str]) -> Option<String> {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").ok()?;
    let output = Command::new("git")
        .args(args)
        .current_dir(&manifest_dir)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// Embed the git revision of the source tree into the binary at compile time.
///
/// `REST_GIT_COMMIT` holds the full `HEAD` commit id, `REST_GIT_COMMIT_DATE` the
/// committer date of that commit (the moment the commit object was created, kept
/// in the commit's own timezone), and `REST_GIT_DIRTY` is `"1"` when the working
/// tree (or the index) differs from the commit. Untracked files are deliberately
/// ignored, matching the usual `git describe --dirty` semantics; otherwise
/// generated files such as `target/` would mark every build as local. When the
/// build happens outside a git checkout all values fall back to safe defaults
/// instead of failing the build.
fn embed_git_revision() {
    let commit = run_git(&["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".to_string());
    let commit_date =
        run_git(&["log", "-1", "--format=%cd", "--date=format:%Y-%m-%d %H:%M:%S %z"])
            .unwrap_or_else(|| "unknown".to_string());
    let dirty = run_git(&["status", "--porcelain", "--untracked-files=no"])
        .map(|status| !status.is_empty())
        .unwrap_or(false);
    println!("cargo:rustc-env=REST_GIT_COMMIT={}", commit);
    println!("cargo:rustc-env=REST_GIT_COMMIT_DATE={}", commit_date);
    println!("cargo:rustc-env=REST_GIT_DIRTY={}", if dirty { "1" } else { "0" });

    // Emitting any `rerun-if-changed` disables Cargo's default "rerun on any package
    // change" behaviour, so list every input this script actually depends on. The git
    // metadata is watched as well, so that a commit or a `git add` refreshes the stamp
    // even when no source file was touched.
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=Cargo.toml");
    println!("cargo:rerun-if-changed=src");
    if let Some(git_dir) = run_git(&["rev-parse", "--git-dir"]) {
        let git_dir = PathBuf::from(git_dir);
        println!("cargo:rerun-if-changed={}", git_dir.join("HEAD").display());
        println!("cargo:rerun-if-changed={}", git_dir.join("index").display());
    }
}

fn main() {

    // conditionally link to the libraries based on the features
    #[cfg(feature = "intel-mkl")] {
        println!("cargo:rustc-link-lib=mkl_rt");
        let blas_dir = if let Ok(blas_dir) = env::var("MKLROOT") {
            PathBuf::from(blas_dir)
        } else {panic!("MKLROOT not set for feature intel-mkl")};
        println!("cargo:rustc-link-search={}/lib",&blas_dir.display());
    }

    // AJZ34: After dftd3/4 v0.2, libxc v0.1, by default we support dynamic loading.
    // libs-dftd3.so, libdftd4.so, libxc.so are not necessarily linked at compile time.
    // User should specify those libraries if dftd3/4/libxc computation requested (by conda or LD_LIBRARY_PATH).
    //
    // #[cfg(feature = "dftd3")]
    // println!("cargo:rustc-link-lib=s-dftd3");
    // #[cfg(feature = "dftd4")]
    // println!("cargo:rustc-link-lib=dftd4");

    // `rest2fch` (the MOKIT-derived Fortran library writing fchk MO coefficients)
    // is only linked when the `librest2fch` feature is enabled; without it the
    // MO sections are written natively in Rust (SCF::fchk_write_mo_rust).
    let mut library_names: Vec<&str> = vec!["cint","restmatr","hdf5","openblas","gomp"];
    if cfg!(feature = "librest2fch") {
        library_names.push("rest2fch");
    }
    library_names.iter().for_each(|name| {
        println!("cargo:rustc-link-lib={}",*name);
    });

    let external_dir = if let Ok(external_dir) = env::var("REST_EXT_DIR") {
        PathBuf::from(external_dir)
    } else {panic!("REST_EXT_DIR not set, which,however, is necessary for the build script to run.")};


    let library_path = [
        dunce::canonicalize(&external_dir).unwrap(),
    ];
    library_path.iter().for_each(|path| {
        println!("cargo:rustc-link-search={}",env::join_paths(&[path]).unwrap().to_str().unwrap())
    });
    //println!("cargo:rustc-link-arg=-Wl,--no-as-needed,-lgomp");

    embed_git_revision();
}
