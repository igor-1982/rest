extern crate dunce;
use std::{env, process::Command};
//use std::path::PathBuf;

fn main() -> miette::Result<()> {

    let external_dir = if let Ok(external_dir) = env::var("REST_EXT_DIR") {
        external_dir
    } else {"".to_string()};
    let external_inc = if let Ok(external_inc) = env::var("REST_EXT_INC") {
        external_inc
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

    let library_names = ["restmatr","openblas","xc","hdf5","rest2fch","cgto","s-dftd3","dftd4","dftd3_rest","dftd4_rest"];
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

    let fortran_compiler = if let Ok(fortran_compiler) = env::var("REST_FORTRAN_COMPILER") {
        fortran_compiler
    } else {"gfortran".to_string()};

    let openblas_libr = format!("-L{} -lopenblas", &blas_dir);

    let dftd3_rest_file = format!("src/external_libs/dftd3_rest.f90");
    let dftd3_rest_libr = format!("{}/libdftd3_rest.so",&external_dir.to_string());
    let dftd3_rest_link = format!("-L{} -ls-dftd3",&external_dir.to_string());
    let dftd3_rest_include = format!("-I{}/{}",&external_inc.to_string(),"dftd3");

    Command::new(fortran_compiler.clone())
        .args(&["-shared", "-fPIC", "-O2",&dftd3_rest_file,"-o",&dftd3_rest_libr,&dftd3_rest_link,&openblas_libr,&dftd3_rest_include])
        .status().unwrap();


    println!("cargo:rerun-if-changed=src/externalernal_libs/dftd3_rest.f90");
    println!("cargo:rerun-if-changed={}/libdftd3_rest.so", &external_dir.to_string());

    let dftd4_rest_file = format!("src/external_libs/dftd4_rest.f90");
    let dftd4_rest_libr = format!("{}/libdftd4_rest.so",&external_dir.to_string());
    let dftd4_rest_link = format!("-L{} -ldftd4",&external_dir.to_string());
    let dftd4_rest_include = format!("-I{}/{}",&external_inc.to_string(),"dftd4");

    Command::new(fortran_compiler.clone())
        .args(&["-shared", "-fPIC", "-O2",&dftd4_rest_file,"-o",&dftd4_rest_libr,&dftd4_rest_link,&openblas_libr,&dftd4_rest_include])
        .status().unwrap();

    println!("cargo:rerun-if-changed=src/externalernal_libs/dftd4_rest.f90");
    println!("cargo:rerun-if-changed={}/libdftd4_rest.so", &external_dir.to_string());

    Ok(())

}
