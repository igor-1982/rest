extern crate dunce;
use std::env;
//use std::path::PathBuf;

fn main() {
    //let root = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());

    let external_dir = if let Ok(external_dir) = env::var("REST_EXT_DIR") {
        external_dir
    } else {"".to_string()};
    let blas_dir = if let Ok(blas_dir) = env::var("REST_BLAS_DIR") {
        blas_dir
    } else {"".to_string()};
    let cint_dir = if let Ok(cint_dir) = env::var("REST_CINT_DIR") {
        cint_dir
    } else {"".to_string()};
    let xc_dir = if let Ok(xc_dir) = env::var("REST_XC_DIR") {
        xc_dir
    } else {"".to_string()};
    let hdf5_dir = if let Ok(hdf5_dir) = env::var("REST_HDF5_DIR") {
        hdf5_dir
    } else {"".to_string()};

    let library_names = ["restmatr","openblas","xc","hdf5","rest2fch"];
    library_names.iter().for_each(|name| {
        println!("cargo:rustc-link-lib={}",*name);
    });
    let library_path = [
        //dunce::canonicalize("/home/igor/Documents/Package-Pool/rest_workspace/rest/src/dft/libxc").unwrap(),
        dunce::canonicalize(external_dir).unwrap(),
        dunce::canonicalize(blas_dir).unwrap(),
        dunce::canonicalize(cint_dir).unwrap(),
        dunce::canonicalize(hdf5_dir).unwrap(),
        dunce::canonicalize(xc_dir).unwrap(),
    ];
    library_path.iter().for_each(|path| {
        println!("cargo:rustc-link-search=native={}",env::join_paths(&[path]).unwrap().to_str().unwrap())
    });
}