mod ffi_mokit;
use crate::external_libs::ffi_mokit::*;
use std::ffi::{c_double, c_int, c_char, CStr, CString};


pub fn py2fch(
    fchname: String,
    nbf: usize,
    nif: usize,
    eigenvector:&[f64], 
    ab: char,
    eigenvalues:&[f64],
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
        &(gen_density as i32))
    }
}