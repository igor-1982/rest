extern crate dunce;
use std::{env, process::Command};
//use std::path::PathBuf;

fn main() -> miette::Result<()> {

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

    //let rest_dir = if let Ok(rest_dir) = env::var("REST_HOME") {
    //    rest_dir
    //} else {"".to_string()};

    build_dftd3and4();

    let library_names = ["restmatr","openblas","xc","hdf5","rest2fch","cgto"];
    library_names.iter().for_each(|name| {
        println!("cargo:rustc-link-lib={}",*name);
    });
    let library_path = [
        dunce::canonicalize(&external_dir).unwrap(),
        dunce::canonicalize(&blas_dir).unwrap(),
        dunce::canonicalize(&cint_dir).unwrap(),
        dunce::canonicalize(&hdf5_dir).unwrap(),
        dunce::canonicalize(&xc_dir).unwrap(),
    ];
    library_path.iter().for_each(|path| {
        println!("cargo:rustc-link-search=native={}",env::join_paths(&[path]).unwrap().to_str().unwrap())
    });


    Ok(())

}


fn build_dftd3and4() {
    let rest_dir = if let Ok(rest_dir) = env::var("REST_HOME") {
        rest_dir
    } else {"".to_string()};
    let external_dir = if let Ok(external_dir) = env::var("REST_EXT_DIR") {
        external_dir
    } else {"".to_string()};
    let external_inc = if let Ok(external_inc) = env::var("REST_EXT_INC") {
        external_inc
    } else {"".to_string()};

    let fortran_compiler = if let Ok(fortran_compiler) = env::var("REST_FORTRAN_COMPILER") {
        fortran_compiler
    } else {"gfortran".to_string()};
    //let mut binding = Command::new(&fortran_compiler);

    // compile dftd3_rest
    let dftd3_rest_file = format!("{}/rest/src/external_libs/dftd3_rest.f90", &rest_dir);
    let dftd3_rest_libr = format!("{}/libdftd3_rest.so",&external_dir);
    let dftd3_rest_link = format!("-L{}",&external_dir);
    let dftd3_rest_include = format!("-I{}/{}",&external_inc,"dftd3");
    
    Command::new(&fortran_compiler).arg("-shared").arg("-fPIC").arg("-O2")
        .arg(&dftd3_rest_file)
        .arg("-o").arg(&dftd3_rest_libr)
        .arg(&dftd3_rest_link).arg("-ls-dftd3")
        .arg(&dftd3_rest_include).status().unwrap();


    // compile dftd4_rest
    let dftd4_rest_file = format!("{}/rest/src/external_libs/dftd4_rest.f90", &rest_dir.to_string());
    let dftd4_rest_libr = format!("{}/libdftd4_rest.so",&external_dir.to_string());
    let dftd4_rest_link = format!("-L{}",&external_dir.to_string());
    let dftd4_rest_include = format!("-I{}/{}",&external_inc.to_string(),"dftd4");

    Command::new(&fortran_compiler).arg("-shared").arg("-fPIC").arg("-O2")
        .arg(&dftd4_rest_file)
        .arg("-o").arg(&dftd4_rest_libr)
        .arg(&dftd4_rest_link).arg("-ldftd4")
        .arg(&dftd4_rest_include).status().unwrap();

    println!("cargo:rerun-if-changed={}", &dftd3_rest_libr);
    println!("cargo:rerun-if-changed={}", &dftd4_rest_libr);
    println!("cargo:rerun-if-changed={}", &dftd3_rest_file);
    println!("cargo:rerun-if-changed={}", &dftd4_rest_file);

    let library_names = ["s-dftd3","dftd4","dftd3_rest","dftd4_rest"];
    library_names.iter().for_each(|name| {
        println!("cargo:rustc-link-lib={}",*name);
    });


}
