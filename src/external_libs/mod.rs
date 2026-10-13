#![warn(unused)]

// The MOKIT-derived Fortran library `librest2fch` (and its FFI binding) is only
// compiled and linked when the `librest2fch` feature is enabled. Without the
// feature, the fchk MO sections are written natively in Rust (see
// `SCF::fchk_write_mo_rust` in scf_io/fchk.rs).
#[cfg(feature = "librest2fch")]
mod ffi_mokit;

#[cfg(feature = "librest2fch")]
use std::ffi::{c_char};
#[cfg(feature = "librest2fch")]
use crate::external_libs::ffi_mokit::*;

#[cfg(feature = "librest2fch")]
pub fn py2fch(
    fchname: String,
    nbf: usize,
    nif: usize,
    eigenvector:&[f64],
    ab: char,
    eigenvalues:&[f64],
    natorb: usize,
    gen_density: usize)
{
    //let fchname_cstring = CString::new(fchname).expect("CString::new failed");
    let fchname_chars:&Vec<c_char> = &fchname.chars().map(|c| c as c_char).collect();
    unsafe{rest2fch_(
        //fchname_cstring.as_ptr(),
        fchname_chars.as_ptr(),
        &(fchname.len() as i32),
        &(nbf as i32),
        &(nif as i32),
        eigenvector.as_ptr(),
        &(ab as c_char),
        eigenvalues.as_ptr(),
        &(natorb as i32),
        &(gen_density as i32)
    )
    }
}
