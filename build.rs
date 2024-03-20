extern crate dunce;
use std::env;
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

    let library_names = ["restmatr","openblas","xc","hdf5","rest2fch","cgto"];
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


    Ok(())

}
