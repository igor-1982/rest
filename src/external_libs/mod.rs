mod ffi_mokit;
mod ffi_dftd;
use crate::external_libs::ffi_mokit::*;
use crate::external_libs::ffi_dftd::*;
use crate::geom_io::get_charge;
use std::ffi::{c_double, c_int, c_char, CStr, CString};
use crate::scf_io::SCF;

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

pub fn dftd(scf_data: &SCF) -> (f64, Option<Vec<f64>>, Option<Vec<f64>>) {
    //create a list contains dftd supported functional, if scf_data.mol.ctrl.xc is in the list, then do dftd3_atm; else print invalid input, and return 0.0
    let dftd_supported = vec!["am05", "b-lyp", "blyp", "b-p", "bp86", "bp", "b-p86", "b1b95", "b1lyp", 
                                            "b1-lyp", "b1p", "b1-p", "b1p86", "b1pw", "b1-pw", "b1pw91", "b2gpplyp", "b2gp-plyp", 
                                            "b2plyp", "b2-plyp", "b3-lyp", "b3lyp", "b3p", "b3-p", "b3p86", "b3pw", "b3-pw", "b3pw91", 
                                            "b97", "b97d", "b97m", "bh-lyp", "bhlyp", "bpbe", "bpw", "b-pw", "cam-b3lyp", "camb3lyp", 
                                            "cam-qtp01", "camqtp01", "cam-qtp(01)", "dftb(mio)", "dftb3", "dodblyp", "dod-blyp", "dodpbe", 
                                            "dod-pbe", "dodpbeb95", "dod-pbeb95", "dodpbep86", "dod-pbep86", "dodsvwn", "dod-svwn", "dsdblyp", 
                                            "dsd-blyp", "dsdpbe", "dsd-pbe", "dsdpbeb95", "dsd-pbeb95", "dsdpbep86", "dsd-pbep86", "dsdsvwn", 
                                            "dsd-svwn", "glyp", "g-lyp", "hf", "hse03", "hse06", "hse12", "hse12s", "hsesol", "kpr2scan50", 
                                            "kpr2scan50", "lb94", "lc-blyp", "lcblyp", "lc-dftb", "lc-wpbe", "lcwpbe", "lc-ωpbe", "lcωpbe", 
                                            "lc-omegapbe", "lcomegapbe", "lc-wpbeh", "lcwpbeh", "lc-ωpbeh", "lcωpbeh", "lc-omegapbeh", 
                                            "lcomegapbeh", "lh07ssvwn", "lh07s-svwn", "lh07tsvwn", "lh07t-svwn", "lh12ctssifpw92", 
                                            "lh12ct-ssifpw92", "lh12ctssirpw92", "lh12ct-ssirpw92", "lh14tcalpbe", "lh14t-calpbe", 
                                            "lh20t", "m06", "m06l", "mn12sx", "mn12-sx", "mpw1b95", "mpw1lyp", "mpw1-lyp", "mpw1pw", 
                                            "mpw1-pw", "mpw1pw91", "mpw2plyp", "mpwb1k", "mpwlyp", "mpw-lyp", "mpwpw", "mpw-pw", "mpwpw91", 
                                            "o-lyp", "olyp", "o3-lyp", "o3lyp", "opbe", "pbe", "pbe0", "pbe02", "pbe0-2", "pbe0dh", 
                                            "pbe0-dh", "pbesol", "pr2scan50", "pr2scan50", "pr2scan69", "pr2scan69", "pw1pw", "pw1-pw", 
                                            "pw6b95", "pw86pbe", "pw91", "pwp", "pw-p", "pw91p86", "pwp1", "pwpb95", "r2scan", "r2scan-3c", 
                                            "r2scan_3c", "r2scan3c", "r2scan-cidh", "r2scancidh", "r2scan-qidh", "r2scanqidh", "r2scan0", 
                                            "r2scan0-2", "r2scan02", "r2scan0-dh", "r2scan0dh", "r2scan50", "r2scanh", "revdod-pbep86", 
                                            "revdodpbep86", "revdsd-blyp", "revdsdblyp", "revdsd-pbe", "revdsd-pbepbe", "revdsdpbe", 
                                            "revdsdpbepbe", "revdsd-pbep86", "revdsdpbep86", "revpbe", "revpbe0", "revpbe0dh", "revpbe0-dh", 
                                            "revpbe38", "revtpss", "revtpss0", "revtpssh", "rpbe", "rpw86pbe", "rscan", "scan", "tpss", 
                                            "tpss0", "tpssh", "wb97", "ωb97", "omegab97", "wb97m", "ωb97m", "omegab97m", "wb97m-rev", 
                                            "ωb97m-rev", "omegab97m-rev", "wb97m_rev", "ωb97m_rev", "omegab97m_rev", "wb97x", "ωb97x", 
                                            "omegab97x", "wb97x-3c", "ωb97x-3c", "omegab97x-3c", "wb97x_3c", "ωb97x_3c", "omegab97x_3c", 
                                            "wb97x-rev", "ωb97x-rev", "omegab97x-rev", "wb97x_rev", "ωb97x_rev", "omegab97x_rev", "wpr2scan50", 
                                            "wpr2scan50", "wr2scan", "x-lyp", "xlyp", "x3-lyp", "x3lyp"];
    if dftd_supported.contains(&scf_data.mol.ctrl.xc.as_str()) {
        //if scf_data.mol.ctrl.empirical_dispersion.clone() is "d3" or "d3bj", then use dftd3_atm; "d4" use dftd4, else print invalid input, and return 0.0
        if let Some(tmp_emprical) = &scf_data.mol.ctrl.empirical_dispersion {
            if tmp_emprical == "d3" || tmp_emprical == "d3bj" {
                return dftd3_atm(scf_data);
            } else if tmp_emprical == "d4" {
                return dftd4_atm(scf_data); 
            } else {
                println!("Invalid input for empirical_dispersion: {}.\nDo not invoke the empirical dispersion evaluation!", tmp_emprical);
                return (0.0, None, None);
            }
        } else {
            println!("No empirical_dispersion.");
            return (0.0, None, None);
        }
    } else {
        println!("Invalid input for xc: {}.\nDo not invoke the empirical dispersion evaluation!", scf_data.mol.ctrl.xc);
        return (0.0, None, None);
    }
}



pub fn dftd3_atm(scf_data: &SCF) -> (f64, Option<Vec<f64>>, Option<Vec<f64>>) {

    let num = get_charge(&scf_data.mol.geom.elem).iter().map(|x| *x as i32).collect::<Vec<i32>>();
    let mut energy = 0.0;
    let mut gradient = vec![0.0; num.len()*3];
    let mut sigma = vec![0.0; 20];

    println!("Calculating DFTD3:");
    unsafe{calc_dftd3_atm_rest_(
        num.as_ptr(),
        &(num.len() as c_int),
        scf_data.mol.geom.position.data.as_ptr(),
        &scf_data.mol.ctrl.charge as *const f64,
        &(scf_data.mol.ctrl.spin as i32 -1),
        scf_data.mol.ctrl.xc.as_ptr() as *const c_char,
        &(scf_data.mol.ctrl.xc.len() as i32),
        &mut energy,
        gradient.as_mut_ptr(),
        sigma.as_mut_ptr(),
        scf_data.mol.ctrl.empirical_dispersion.clone().unwrap().as_ptr() as *const c_char,
        &(scf_data.mol.ctrl.empirical_dispersion.clone().unwrap().len() as i32),
        )
    }
    (energy, Some(gradient), Some(sigma))
}

pub fn dftd4_atm(scf_data: &SCF) -> (f64, Option<Vec<f64>>, Option<Vec<f64>>) {

    let num = get_charge(&scf_data.mol.geom.elem).iter().map(|x| *x as i32).collect::<Vec<i32>>();
    let mut energy = 0.0;
    let mut gradient = vec![0.0; num.len()*3];
    let mut sigma = vec![0.0; 20];


    println!("Calculating DFTD4:");
    unsafe{calc_dftd4_rest_(
        num.as_ptr(),
        &(num.len() as c_int),
        scf_data.mol.geom.position.data.as_ptr(),
        &scf_data.mol.ctrl.charge as *const f64,
        &(scf_data.mol.ctrl.spin as i32 -1),
        scf_data.mol.ctrl.xc.as_ptr() as *const c_char,
        &(scf_data.mol.ctrl.xc.len() as i32),
        &mut energy,
        gradient.as_mut_ptr(),
        sigma.as_mut_ptr(),
    )
    }
    (energy, Some(gradient), Some(sigma))
}
