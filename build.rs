extern crate dunce;
use std::{env, process::Command};
//use std::env;
//use std::path::PathBuf;

fn main() -> miette::Result<()> {
    //let root = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());

    //// for libecpint ffi
    //let cpp = std::path::PathBuf::from("/usr/include/c++/11");
    //let cpp1 = std::path::PathBuf::from("/usr/include/c++/11/bits");
    //let libecpint_dir = std::path::PathBuf::from("/usr/include/libecpint");
    //let rest_main_dir = std::path::PathBuf::from("src");
    //let rest_ecpint_dir = std::path::PathBuf::from("src/basis_io/ecp");
    //let mut b = autocxx_build::Builder::new("src/basis_io/ecp/mod.rs", &[&cpp, &cpp1, &libecpint_dir, &rest_main_dir, &rest_ecpint_dir]).build()?;
    //b.flag_if_supported("-std=c++14").file("src/basis_io/ecp/run.cpp").compile("run");
    ////b.flag_if_supported("-std=c++14").file("src/run.cpp").compile("run");

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
        //dunce::canonicalize("/home/igor/Documents/Package-Pool/rest_workspace/rest/src/dft/libxc").unwrap(),
        dunce::canonicalize(&external_dir).unwrap(),
        dunce::canonicalize(&blas_dir).unwrap(),
        dunce::canonicalize(&cint_dir).unwrap(),
        dunce::canonicalize(&hdf5_dir).unwrap(),
        dunce::canonicalize(&xc_dir).unwrap(),
    ];
    library_path.iter().for_each(|path| {
        println!("cargo:rustc-link-search=native={}",env::join_paths(&[path]).unwrap().to_str().unwrap())
    });

    //println!("cargo:rustc-link-search=native=/home/igor/Documents/Package-Pool/rest_workspace/target/debug/build/cxx-d0f08da38178fbbe/out");
    //target/debug/build/cxx-fb034afd48f65373/out
    //

    //// rebuild source if changed
    //println!("cargo:rerun-if-changed=src/basis_io/ecp/mod.rs");
    //println!("cargo:rerun-if-changed=build.rs");
    //println!("cargo:rerun-if-changed=src/basis_io/ecp/run.cpp");
    //println!("cargo:rerun-if-changed=src/basis_io/ecp/run.h");

    //// libecpint required link libraries                                                                                                                                     
    //println!("cargo:rustc-link-lib=ecpint");
    //println!("cargo:rustc-link-lib=pugixml");
    //println!("cargo:rustc-link-lib=Faddeeva");
    let fortran_compiler = if let Ok(fortran_compiler) = env::var("REST_FORTRAN_COMPILER") {
        fortran_compiler
    } else {"gfortran".to_string()};

    let openblas_libr = format!("-L{} -lopenblas",&blas_dir);

    let dftd3_rest_file = format!("src/external_libs/dftd3_rest.f90");
    let dftd3_rest_libr = format!("{}/libdftd3_rest.so",&external_dir.to_string());
    let dftd3_rest_link = format!("-L{} -ls-dftd3",&external_dir.to_string());
    let dftd3_rest_include = format!("-I{}/{}",&external_inc.to_string(),"dftd3");

    Command::new(fortran_compiler.clone())
        .args(&["-shared", "-fPIC", "-O2",&dftd3_rest_file,"-o",&dftd3_rest_libr,&dftd3_rest_link,&openblas_libr,&dftd3_rest_include])
        .status().unwrap();


    println!("cargo:rustc-link-lib=dftd3_rest");
    println!("cargo:rustc-link-search=native={}",&external_dir.to_string());


    println!("cargo:rerun-if-changed=src/externalernal_libs/dftd3_rest.f90");
    println!("cargo:rerun-if-changed={}/libdftd3_rest.so", &external_dir.to_string());

    let dftd4_rest_file = format!("src/external_libs/dftd4_rest.f90");
    let dftd4_rest_libr = format!("{}/libdftd4_rest.so",&external_dir.to_string());
    let dftd4_rest_link = format!("-L{} -ldftd4",&external_dir.to_string());
    let dftd4_rest_include = format!("-I{}/{}",&external_inc.to_string(),"dftd4");

    Command::new(fortran_compiler.clone())
        .args(&["-shared", "-fPIC", "-O2",&dftd4_rest_file,"-o",&dftd4_rest_libr,&dftd4_rest_link,&openblas_libr,&dftd4_rest_include])
        .status().unwrap();


    println!("cargo:rustc-link-lib=dftd4_rest");
    println!("cargo:rustc-link-search=native={}",&external_dir.to_string());


    println!("cargo:rerun-if-changed=src/externalernal_libs/dftd4_rest.f90");
    println!("cargo:rerun-if-changed={}/libdftd4_rest.so", &external_dir.to_string());


    Ok(())

}
