mod libxc;
pub mod gen_grids;

use rest_tensors::{MatrixFull, MatrixFullSliceMut, TensorSliceMut, RIFull, MatrixFullSlice};
use rest_tensors::matrix_blas_lapack::{_dgemm_nn,_dgemm_tn, _einsum_01_serial, _einsum_02_serial, _einsum_01_rayon, _einsum_02_rayon};
use itertools::{Itertools, izip};
use libc::access;
use tensors::{BasicMatrix, MathMatrix, ParMathMatrix};
use tensors::external_libs::{general_dgemm_f, matr_copy};
use tensors::matrix_blas_lapack::{_dgemm, _dgemm_full, contract_vxc_0_serial};
//use numgrid::{self, radial_grid_lmg_bse};
use self::gen_grids::radial_grid_lmg_bse;
use rayon::iter::{IntoParallelRefIterator, IndexedParallelIterator, ParallelIterator, IntoParallelRefMutIterator};
use regex::Regex;
use crate::basis_io::{Basis4Elem, cartesian_gto_cint, cartesian_gto_std, gto_value, BasCell, gto_value_debug, cint_norm_factor, gto_1st_value, spheric_gto_value_matrixfull, spheric_gto_1st_value_batch, spheric_gto_value_matrixfull_serial, spheric_gto_1st_value_batch_serial, spheric_gto_value_serial, spheric_gto_1st_value_serial};
use crate::molecule_io::Molecule;
use crate::geom_io::get_mass_charge;
use crate::scf_io::SCF;
use crate::utilities::{self, balancing};
use core::num;
use std::collections::HashMap;
use std::io::Read;
use std::iter::Zip;
use std::ops::Range;
use std::option::IntoIter;
use std::os::raw::c_int;
use std::path::Iter;
use std::sync::mpsc::channel;
use serde::{Deserialize, Serialize};

//extern crate rest_libxc  as libxc;
use libxc::{XcFuncType, LibXCFamily};
//use std::intrinsics::expf64;

#[derive(Clone,Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DFAFamily {
    LDA,
    GGA,
    MGGA,
    HybridGGA,
    HybridMGGA,
    PT2,
    SBGE2,
    RPA,
    SCSRPA,
    Unknown
}

#[derive(Clone)]
pub struct DFA4REST {
    pub spin_channel: usize,
    pub dfa_compnt_scf: Vec<usize>,
    pub dfa_paramr_scf: Vec<f64>,
    pub dfa_hybrid_scf: f64,
    pub dfa_family_pos: Option<DFAFamily>,
    pub dfa_compnt_pos: Option<Vec<usize>>,
    pub dfa_paramr_pos: Option<Vec<f64>>,
    pub dfa_hybrid_pos: Option<f64>,
    pub dfa_paramr_adv: Option<Vec<f64>>,
}

impl DFAFamily {
    pub fn to_libxc_family(&self) -> libxc::LibXCFamily {
        match self {
            DFAFamily::LDA => libxc::LibXCFamily::LDA,
            DFAFamily::GGA => libxc::LibXCFamily::GGA,
            DFAFamily::MGGA => libxc::LibXCFamily::MGGA,
            DFAFamily::HybridGGA => libxc::LibXCFamily::HybridGGA,
            DFAFamily::HybridMGGA => libxc::LibXCFamily::HybridMGGA,
            DFAFamily::PT2 => libxc::LibXCFamily::HybridGGA,
            DFAFamily::SBGE2 => libxc::LibXCFamily::HybridGGA,
            DFAFamily::SCSRPA => libxc::LibXCFamily::HybridGGA,
            DFAFamily::RPA => libxc::LibXCFamily::GGA,
            _ => libxc::LibXCFamily::Unknown,
        }
    }
    pub fn from_libxc_family(family: &libxc::LibXCFamily) -> DFAFamily {
        match family {
            libxc::LibXCFamily::LDA => DFAFamily::LDA,
            libxc::LibXCFamily::GGA => DFAFamily::GGA,
            libxc::LibXCFamily::MGGA => DFAFamily::MGGA,
            libxc::LibXCFamily::HybridGGA => DFAFamily::HybridGGA,
            libxc::LibXCFamily::HybridMGGA => DFAFamily::HybridMGGA,
            _ => DFAFamily::Unknown,
        }
    }
    pub fn to_name(&self) -> String {
        match self {
            DFAFamily::LDA => {"LDA".to_string()},
            DFAFamily::GGA => {"GGA".to_string()},
            DFAFamily::MGGA => {"Meta-GGA".to_string()},
            DFAFamily::HybridGGA => {"Hybrid-GGA".to_string()},
            DFAFamily::HybridMGGA => {"Hybrid-meta-GGA".to_string()},
            DFAFamily::PT2 => {"PT2".to_string()},
            DFAFamily::SBGE2 => {"SBGE2".to_string()},
            DFAFamily::RPA => {"RPA".to_string()},
            DFAFamily::SCSRPA => {"SCSRPA".to_string()},
            DFAFamily::Unknown => {"Unknown".to_string()},
        }

    }
}

impl DFA4REST {

    pub fn xc_version(&self) {
        let mut vmajor:c_int = 0;
        let mut vminor:c_int = 0;
        let mut vmicro:c_int = 0;
        unsafe{libxc::ffi_xc::xc_version(&mut  vmajor, &mut vminor, &mut vmicro)};
        println!("Libxc version used in REST: {}.{}.{}", vmajor, vminor, vmicro);
    }

    pub fn new_xc(spin_channel: usize, print_level: usize) -> DFA4REST {
        DFA4REST { 
            spin_channel, 
            dfa_compnt_scf: vec![], 
            dfa_paramr_scf: vec![], 
            dfa_hybrid_scf: 0.0, 
            dfa_family_pos: None, 
            dfa_compnt_pos: None, 
            dfa_paramr_pos: None, 
            dfa_hybrid_pos: None, 
            dfa_paramr_adv: None }
    }

    pub fn new(name: &str, spin_channel: usize, print_level: usize) -> DFA4REST {
        let tmp_name = name.to_lowercase();
        let post_dfa = DFA4REST::parse_postscf(&tmp_name, spin_channel);
        match post_dfa {
            Some(dfa) => {
                if print_level> 0 {
                    println!("the scf functional for '{}' contains", &name);
                    &dfa.dfa_compnt_scf.iter().for_each(|xc_func| {
                        dfa.init_libxc(xc_func).xc_func_info_printout()
                    });
                    if let (Some(dfatype),Some(dfacomp)) = 
                        (&dfa.dfa_family_pos, &dfa.dfa_compnt_pos) {
                        //match dfatype {
                        //    DFAFamily::PT2 => println!("XYG3-type functional '{}' is employed", &name),
                        //    DFAFamily::RPA => println!("RPA-type functional '{}' is employed", &name),
                        //    _ => println!("Standard DFA '{}' is employed", &name),
                        //}
                        println!("the post-scf functional '{}' is employed, which contains", &name);
                        dfacomp.into_iter().for_each(|xc_func| {
                            dfa.init_libxc(xc_func).xc_func_info_printout()
                        })
                    };
                }
                dfa
            },
            None => {
                let dfa = DFA4REST::parse_scf(&tmp_name, spin_channel);
                if print_level> 0 {
                    println!("the functional of '{}' contains", &name);
                    dfa.dfa_compnt_scf.iter().for_each(|xc_func| {
                        let tmp_dfa = dfa.init_libxc(xc_func);
                        tmp_dfa.xc_func_info_printout();
                    });
                };
                dfa
            },
        }
    }

    pub fn libxc_code_fdqc(name: &str) -> [usize;3] {
        let lower_name = name.to_lowercase();
        //println!("Debug: {:?}", &lower_name);
        // for a list of exchange-correlation functionals
        if lower_name.eq(&"hf".to_string()) {
            [0,0,0]
        } else if lower_name.eq(&"svwn".to_string()) {
            [0,1,7]
        } else if lower_name.eq(&"svwn-rpa".to_string()) {
            [0,1,8]
        } else if lower_name.eq(&"pz-lda".to_string()) {
            [0,1,9]
        } else if lower_name.eq(&"pw-lda".to_string()) {
            [0,1,12]
        } else if lower_name.eq(&"blyp".to_string()) {
            [0,106,131]
        } else if lower_name.eq(&"xlyp".to_string()) {
            [166,0,0]
        } else if lower_name.eq(&"pbe".to_string()) {
            [0,101,130]
        } else if lower_name.eq(&"xpbe".to_string()) {
            [0,123,136]
        } else if lower_name.eq(&"b3lyp".to_string()) {
            [402,0,0]
        } else if lower_name.eq(&"x3lyp".to_string()) {
            [411,0,0]
        } else if lower_name.eq(&"pbe0".to_string()) {
            [406,0,0]
        // for a list of exchange functionals
        } else if lower_name.eq(&"lda_x_slater".to_string()) {
            [0,1,0]
        } else if lower_name.eq(&"gga_x_b88".to_string()) {
            [0,106,0]
        } else if lower_name.eq(&"gga_x_pbe".to_string()) {
            [0,101,0]
        } else if lower_name.eq(&"gga_x_xpbe".to_string()) {
            [0,123,0]
        // for a list of correlation functionals
        } else if lower_name.eq(&"lda_c_vwn".to_string()) {
            [0,0,7]
        } else if lower_name.eq(&"lda_c_vwn_rpa".to_string()) {
            [0,0,8]
        } else if lower_name.eq(&"gga_c_lyp".to_string()) {
            [0,0,131]
        } else if lower_name.eq(&"gga_c_pbe".to_string()) {
            [0,0,130]
        } else if lower_name.eq(&"gga_c_xpbe".to_string()) {
            [0,0,136]
        } else {
            println!("Unknown XC method is specified: {}. The standard Hartree-Fock approximation is involked", &name);
            [0,0,0]
        }
    }

    //pub fn xc_func_init_fdqc(name: &str, spin_channel: usize) -> Vec<XcFuncType> {
    //    let lower_name = name.to_lowercase();
    //    let xc_code = DFA4REST::libxc_code_fdqc(name);
    //    let mut xc_list: Vec<XcFuncType> = vec![];
    //    xc_code.iter().for_each(|x| {
    //        if *x!=0 {
    //            xc_list.push(XcFuncType::xc_func_init(*x, spin_channel));
    //        }
    //    });
    //    xc_list
    //}

    pub fn xc_func_init_fdqc(name: &str, spin_channel: usize) -> Vec<usize> {
        let xc_code = DFA4REST::libxc_code_fdqc(name);
        xc_code.iter().filter(|x| **x!=0).map(|x| *x).collect::<Vec<usize>>()
    }

    pub fn init_libxc(&self, xc_code: &usize) -> XcFuncType {
        XcFuncType::xc_func_init(*xc_code, self.spin_channel)
    }

    pub fn get_hybrid_libxc(dfa_compnt_scf: &Vec<usize>,spin_channel:usize) -> f64 {
        let hybrid_list = dfa_compnt_scf
            .iter()
            .filter(|xc_func| {XcFuncType::xc_func_init(**xc_func,spin_channel).use_exact_exchange()})
            .map(|xc_func| {XcFuncType::xc_func_init(*xc_func,spin_channel).xc_hyb_exx_coeff()}).collect_vec();
        //let count = hybrid_list.iter().fold(0,|acc, x| {if ! x.eq(&0.0) acc + 1});
        let hybrid_coeff = if hybrid_list.len() == 1 {
            hybrid_list[0]
        } else {
            0.0
        };
        hybrid_coeff
    }

    pub fn parse_scf(name: &str, spin_channel: usize) -> DFA4REST {
        let tmp_name = name.to_lowercase();
        //let dfa_compnt_scf = vec![libxc::XcFuncType::xc_func_init_fdqc(&tmp_name, spin_channel)];
        let dfa_compnt_scf = DFA4REST::xc_func_init_fdqc(&tmp_name, spin_channel);
        let dfa_hybrid_scf = DFA4REST::get_hybrid_libxc(&dfa_compnt_scf,spin_channel);
        let dfa_paramr_scf =  vec![1.0;dfa_compnt_scf.len()];

        DFA4REST {
            spin_channel,
            dfa_family_pos: None,
            dfa_compnt_pos: None,
            dfa_paramr_pos: None,
            dfa_hybrid_pos: None,
            dfa_paramr_adv: None,
            dfa_compnt_scf,
            dfa_paramr_scf,
            dfa_hybrid_scf,
        }
    }
    pub fn parse_scf_nonstd(codelist:Vec<usize>, paramlist:Vec<f64>, spin_channel: usize) -> DFA4REST {
        if codelist.len()!=paramlist.len() {panic!("codelist (len: {}) does not match paramlist (len: {})", codelist.len(), paramlist.len())}
        let dfa_hybrid_scf = DFA4REST::get_hybrid_libxc(&codelist,spin_channel);
        DFA4REST {
            spin_channel,
            dfa_family_pos: None,
            dfa_compnt_pos: None,
            dfa_paramr_pos: None,
            dfa_hybrid_pos: None,
            dfa_paramr_adv: None,
            dfa_compnt_scf: codelist,
            dfa_paramr_scf: paramlist,
            dfa_hybrid_scf,
        }
    }

    pub fn parse_postscf(name: &str,spin_channel: usize) -> Option<DFA4REST> {
        let tmp_name = name.to_lowercase();
        if tmp_name.eq("xyg3") {
            let dfa_family_pos = Some(DFAFamily::PT2);
            let pos_dfa = ["lda_x_slater", "gga_x_b88","lda_c_vwn_rpa","gga_c_lyp"];
            let dfa_compnt_pos: Option<Vec<usize>> = Some(pos_dfa.iter().map(|xc| {
                DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
                .flatten()
                .collect());
            let dfa_paramr_pos = Some(vec![-0.0140,0.2107,0.00,0.6789]);
            let dfa_hybrid_pos = Some(0.8033);
            let dfa_paramr_adv = Some(vec![0.3211,0.3211]);

            let scf_dfa = ["b3lyp"];
            let dfa_compnt_scf: Vec<usize> = scf_dfa.iter().map(|xc| {
                DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
                .flatten().collect();
            let dfa_paramr_scf = vec![1.0;dfa_compnt_scf.len()];
            let dfa_hybrid_scf = DFA4REST::get_hybrid_libxc(&dfa_compnt_scf,spin_channel);
            Some(DFA4REST{
                spin_channel,
                dfa_compnt_scf,
                dfa_paramr_scf,
                dfa_hybrid_scf,
                dfa_paramr_adv,
                dfa_family_pos,
                dfa_compnt_pos,
                dfa_paramr_pos,
                dfa_hybrid_pos
            })
        } else if tmp_name.eq("xygjos") {
            let dfa_family_pos = Some(DFAFamily::PT2);
            let pos_dfa = ["lda_x_slater", "gga_x_b88","lda_c_vwn_rpa","gga_c_lyp"];
            //let dfa_compnt_pos: Option<Vec<XcFuncType>> = Some(pos_dfa.iter().map(|xc| {
            //    DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
            //    .flatten().collect());
            let dfa_compnt_pos: Option<Vec<usize>> = Some(pos_dfa.iter().map(|xc| {
                DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
                .flatten().collect());
            let dfa_paramr_pos = Some(vec![0.2269,0.000,0.2309,0.2754]);
            let dfa_hybrid_pos = Some(0.7731);
            let dfa_paramr_adv = Some(vec![0.4364,0.0000]);

            let dfa_family_scf = DFAFamily::HybridGGA;
            let scf_dfa = ["b3lyp"];
            //let dfa_compnt_scf: Vec<XcFuncType> = scf_dfa.iter().map(|xc| {
            //    DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
            //    .flatten().collect();
            let dfa_compnt_scf: Vec<usize> = scf_dfa.iter().map(|xc| {
                DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
                .flatten().collect();
            let dfa_paramr_scf = vec![1.0;dfa_compnt_scf.len()];
            let dfa_hybrid_scf = DFA4REST::get_hybrid_libxc(&dfa_compnt_scf,spin_channel);
            Some(DFA4REST{
                spin_channel,
                dfa_compnt_scf,
                dfa_paramr_scf,
                dfa_hybrid_scf,
                dfa_paramr_adv,
                dfa_family_pos,
                dfa_compnt_pos,
                dfa_paramr_pos,
                dfa_hybrid_pos
            })
        } else if tmp_name.eq("xyg7") {
            let dfa_family_pos = Some(DFAFamily::PT2);
            let pos_dfa = ["lda_x_slater", "gga_x_b88","lda_c_vwn_rpa","gga_c_lyp"];
            //let dfa_compnt_pos: Option<Vec<XcFuncType>> = Some(pos_dfa.iter().map(|xc| {
            //    DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
            //    .flatten().collect());
            let dfa_compnt_pos: Option<Vec<usize>> = Some(pos_dfa.iter().map(|xc| {
                DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
                .flatten().collect());
            let dfa_paramr_pos = Some(vec![0.2055,-0.1408,0.4056,0.1159]);
            let dfa_hybrid_pos = Some(0.8971);
            let dfa_paramr_adv = Some(vec![0.4052,0.2589]);

            let dfa_family_scf = DFAFamily::HybridGGA;
            let scf_dfa = ["b3lyp"];
            //let dfa_compnt_scf: Vec<XcFuncType> = scf_dfa.iter().map(|xc| {
            //    DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
            //    .flatten().collect();
            let dfa_compnt_scf: Vec<usize> = scf_dfa.iter().map(|xc| {
                DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
                .flatten().collect();
            let dfa_paramr_scf = vec![1.0;dfa_compnt_scf.len()];
            let dfa_hybrid_scf = DFA4REST::get_hybrid_libxc(&dfa_compnt_scf,spin_channel);
            Some(DFA4REST{
                spin_channel,
                dfa_compnt_scf,
                dfa_paramr_scf,
                dfa_hybrid_scf,
                dfa_paramr_adv,
                dfa_family_pos,
                dfa_compnt_pos,
                dfa_paramr_pos,
                dfa_hybrid_pos
            })
        } else if tmp_name.eq("zrps") {
            let dfa_family_pos = Some(DFAFamily::SBGE2);
            let pos_dfa = ["gga_x_pbe","gga_c_pbe"];
            let scf_dfa = ["pbe0"];
            let dfa_paramr_pos = Some(vec![0.5,0.75]);
            let dfa_hybrid_pos = Some(0.5);
            let dfa_paramr_adv = Some(vec![0.25,0.00]);

            // Now initialize ZRPS
            let dfa_compnt_pos: Option<Vec<usize>> = Some(pos_dfa.iter().map(|xc| {
                DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
                .flatten().collect());
            let dfa_family_scf = DFAFamily::HybridGGA;
            let dfa_compnt_scf: Vec<usize> = scf_dfa.iter().map(|xc| {
                DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
                .flatten().collect();
            let dfa_paramr_scf = vec![1.0;dfa_compnt_scf.len()];
            let dfa_hybrid_scf = DFA4REST::get_hybrid_libxc(&dfa_compnt_scf,spin_channel);
            Some(DFA4REST{
                spin_channel,
                dfa_compnt_scf,
                dfa_paramr_scf,
                dfa_hybrid_scf,
                dfa_paramr_adv,
                dfa_family_pos,
                dfa_compnt_pos,
                dfa_paramr_pos,
                dfa_hybrid_pos
            })
        } else if tmp_name.eq("rpa@b3lyp") {
            let dfa_family_pos = Some(DFAFamily::RPA);
            let dfa_compnt_pos: Option<Vec<usize>> = Some(vec![]);
            let dfa_paramr_pos = Some(vec![]);
            let dfa_hybrid_pos = Some(1.0);
            let dfa_paramr_adv = Some(vec![1.0]);

            let dfa_family_scf = DFAFamily::HybridGGA;
            let scf_dfa = ["b3lyp"];
            let dfa_compnt_scf: Vec<usize> = scf_dfa.iter().map(|xc| {
                DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
                .flatten().collect();
            let dfa_paramr_scf = vec![1.0;dfa_compnt_scf.len()];
            let dfa_hybrid_scf = DFA4REST::get_hybrid_libxc(&dfa_compnt_scf,spin_channel);
            Some(DFA4REST{
                spin_channel,
                dfa_compnt_scf,
                dfa_paramr_scf,
                dfa_hybrid_scf,
                dfa_paramr_adv,
                dfa_family_pos,
                dfa_compnt_pos,
                dfa_paramr_pos,
                dfa_hybrid_pos
            })
        } else if tmp_name.eq("rpa@pbe") {
            let dfa_family_pos = Some(DFAFamily::RPA);
            let dfa_compnt_pos: Option<Vec<usize>> = Some(vec![]);
            let dfa_paramr_pos = Some(vec![]);
            let dfa_hybrid_pos = Some(1.0);
            let dfa_paramr_adv = Some(vec![1.0]);

            let dfa_family_scf = DFAFamily::GGA;
            let scf_dfa = ["pbe"];
            let dfa_compnt_scf: Vec<usize> = scf_dfa.iter().map(|xc| {
                DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
                .flatten().collect();
            let dfa_paramr_scf = vec![1.0;dfa_compnt_scf.len()];
            let dfa_hybrid_scf = 0.0;
            Some(DFA4REST{
                spin_channel,
                dfa_compnt_scf,
                dfa_paramr_scf,
                dfa_hybrid_scf,
                dfa_paramr_adv,
                dfa_family_pos,
                dfa_compnt_pos,
                dfa_paramr_pos,
                dfa_hybrid_pos
            })
        } else if tmp_name.eq("scsrpa") {
            let dfa_family_pos = Some(DFAFamily::SCSRPA);
            let dfa_compnt_pos: Option<Vec<usize>> = Some(vec![]);
            let dfa_paramr_pos = Some(vec![]);
            let dfa_hybrid_pos = Some(1.0);
            let dfa_paramr_adv = Some(vec![1.2,0.75]);

            let dfa_family_scf = DFAFamily::HybridGGA;
            let scf_dfa = ["pbe0"];
            let dfa_compnt_scf: Vec<usize> = scf_dfa.iter().map(|xc| {
                DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
                .flatten().collect();
            let dfa_paramr_scf = vec![1.0;dfa_compnt_scf.len()];
            let dfa_hybrid_scf = DFA4REST::get_hybrid_libxc(&dfa_compnt_scf,spin_channel);
            Some(DFA4REST{
                spin_channel,
                dfa_compnt_scf,
                dfa_paramr_scf,
                dfa_hybrid_scf,
                dfa_paramr_adv,
                dfa_family_pos,
                dfa_compnt_pos,
                dfa_paramr_pos,
                dfa_hybrid_pos
            })
        } else if tmp_name.eq("r-xdh7") {
            let dfa_family_pos = Some(DFAFamily::SCSRPA);
            let pos_dfa = ["lda_x_slater", "gga_x_b88","lda_c_vwn_rpa","gga_c_lyp"];
            let dfa_compnt_pos: Option<Vec<usize>> = Some(pos_dfa.iter().map(|xc| {
                DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
                .flatten().collect());
            let dfa_paramr_pos = Some(vec![0.3600,-0.2917,0.4937,-0.4301]);
            let dfa_hybrid_pos = Some(0.9081);
            let dfa_paramr_adv = Some(vec![0.8624,0.2359]);

            let dfa_family_scf = DFAFamily::HybridGGA;
            let scf_dfa = ["b3lyp"];
            let dfa_compnt_scf: Vec<usize> = scf_dfa.iter().map(|xc| {
                DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
                .flatten().collect();
            let dfa_paramr_scf = vec![1.0;dfa_compnt_scf.len()];
            let dfa_hybrid_scf = DFA4REST::get_hybrid_libxc(&dfa_compnt_scf,spin_channel);
            Some(DFA4REST{
                spin_channel,
                dfa_compnt_scf,
                dfa_paramr_scf,
                dfa_hybrid_scf,
                dfa_paramr_adv,
                dfa_family_pos,
                dfa_compnt_pos,
                dfa_paramr_pos,
                dfa_hybrid_pos
            })
        } else if tmp_name.eq("mp2") {
            let dfa_family_pos = Some(DFAFamily::PT2);
            let dfa_compnt_pos: Option<Vec<usize>> = Some(vec![]);
            let dfa_paramr_pos = Some(vec![]);
            let dfa_hybrid_pos = Some(1.0);
            let dfa_paramr_adv = Some(vec![1.0,1.0]);

            let scf_dfa = [];
            let dfa_compnt_scf: Vec<usize> = scf_dfa.iter().map(|xc| {
                DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
                .flatten().collect();
            let dfa_paramr_scf = vec![1.0;dfa_compnt_scf.len()];
            let dfa_hybrid_scf = 1.0;
            Some(DFA4REST{
                spin_channel,
                dfa_compnt_scf,
                dfa_paramr_scf,
                dfa_hybrid_scf,
                dfa_paramr_adv,
                dfa_family_pos,
                dfa_compnt_pos,
                dfa_paramr_pos,
                dfa_hybrid_pos
            })
        } else if tmp_name.eq("scs-mp2") {
            let dfa_family_pos = Some(DFAFamily::PT2);
            let dfa_compnt_pos: Option<Vec<usize>> = Some(vec![]);
            let dfa_paramr_pos = Some(vec![]);
            let dfa_hybrid_pos = Some(1.0);
            let dfa_paramr_adv = Some(vec![1.2,0.333333333]);

            let scf_dfa = [];
            //let dfa_compnt_scf: Vec<XcFuncType> = scf_dfa.iter().map(|xc| {
            //    DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
            //    .flatten().collect();
            let dfa_compnt_scf: Vec<usize> = scf_dfa.iter().map(|xc| {
                DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
                .flatten().collect();
            let dfa_paramr_scf = vec![1.0;dfa_compnt_scf.len()];
            let dfa_hybrid_scf = 1.0;
            Some(DFA4REST{
                spin_channel,
                dfa_compnt_scf,
                dfa_paramr_scf,
                dfa_hybrid_scf,
                dfa_paramr_adv,
                dfa_family_pos,
                dfa_compnt_pos,
                dfa_paramr_pos,
                dfa_hybrid_pos
            })
        } else {
            None
        }
    }

    pub fn is_dfa_scf(&self) -> bool {
        self.dfa_compnt_scf.len() !=0
    }

    pub fn is_hybrid(&self) -> bool {
        self.dfa_hybrid_scf.abs() >= 1.0e-6
    }

    pub fn is_fifth_dfa(&self) -> bool {
        match self.dfa_family_pos {
            None => false,
            _ => true
        }
    }

    pub fn use_eri(&self) -> bool {
        self.is_hybrid() || self.is_fifth_dfa()
    }


    pub fn use_density_gradient(&self) -> bool {
        let mut is_flag = self.dfa_compnt_scf.iter().fold(false, |acc, xc_func| {
            acc || self.init_libxc(xc_func).use_density_gradient()
        });
        if let Some(dfa_compnt_pos) = &self.dfa_compnt_pos {
            is_flag = is_flag || dfa_compnt_pos.iter().fold(false, |acc, xc_func| {
                acc || self.init_libxc(xc_func).use_density_gradient()
            });
        }
        is_flag
    }

    pub fn use_kinetic_density(&self) -> bool {
        let mut is_flag = self.dfa_compnt_scf.iter().fold(false, |acc, xc_func| {
            acc || self.init_libxc(xc_func).use_kinetic_density()
        });
        if let Some(dfa_compnt_pos) = &self.dfa_compnt_pos {
            is_flag = is_flag || dfa_compnt_pos.iter().fold(false, |acc, xc_func| {
                acc || self.init_libxc(xc_func).use_kinetic_density()
            });
        }
        is_flag
    }

    pub fn xc_exc_vxc(&self, grids: &Grids, spin_channel: usize, dm: &Vec<MatrixFull<f64>>, mo: &[MatrixFull<f64>;2], occ: &[Vec<f64>;2], print_level:usize) -> (Vec<f64>, Vec<MatrixFull<f64>>) {
        let num_grids = grids.coordinates.len();
        let num_basis = dm[0].size[0];
        let mut exc = MatrixFull::new([num_grids,1],0.0);
        let mut exc_total = vec![0.0;spin_channel];
        let mut vxc_ao = vec![MatrixFull::new([num_basis,num_grids],0.0);spin_channel];
        let dt0 = utilities::init_timing();

        let (rho,rhop) = grids.prepare_tabulated_density_2(mo, occ, spin_channel);
        let dt2 = utilities::timing(&dt0, Some("evaluate rho and rhop"));
        let sigma = if self.use_density_gradient() {
            prepare_tabulated_sigma_rayon(&rhop, spin_channel)
        } else {
            MatrixFull::empty()
        };

        let mut vrho = MatrixFull::new([num_grids,spin_channel],0.0);
        let mut vsigma=if self.use_density_gradient() && spin_channel==1 {
            MatrixFull::new([num_grids,1],0.0)
        } else if self.use_density_gradient() && spin_channel==2 {
            MatrixFull::new([num_grids,3],0.0)
        } else {
            MatrixFull::empty()
        };
        let dt3 = utilities::timing(&dt2, Some("evaluate sigma"));

        self.dfa_compnt_scf.iter().zip(self.dfa_paramr_scf.iter()).for_each(|(xc_func,xc_para)| {
            let xc_func = self.init_libxc(xc_func);
            match xc_func.xc_func_family {
                libxc::LibXCFamily::LDA => {
                    if spin_channel==1 {
                        let (tmp_exc,tmp_vrho) = xc_func.lda_exc_vxc(rho.data_ref().unwrap());
                        let tmp_exc = MatrixFull::from_vec([num_grids,1],tmp_exc).unwrap();
                        let tmp_vrho = MatrixFull::from_vec([num_grids,1],tmp_vrho).unwrap();
                        exc.par_self_scaled_add(&tmp_exc,*xc_para);
                        vrho.par_self_scaled_add(&tmp_vrho,*xc_para);

                    } else {
                        let (tmp_exc,tmp_vrho) = xc_func.lda_exc_vxc(rho.transpose().data_ref().unwrap());
                        let tmp_exc = MatrixFull::from_vec([num_grids,1],tmp_exc).unwrap();
                        //let tmp_vrho = MatrixFull::from_vec([num_grids,spin_channel],tmp_vrho).unwrap();
                        let tmp_vrho = MatrixFull::from_vec([2,num_grids],tmp_vrho).unwrap();
                        exc.par_self_scaled_add(&tmp_exc,*xc_para);
                        vrho.par_self_scaled_add(&tmp_vrho.transpose_and_drop(),*xc_para);
                    }
                },
                libxc::LibXCFamily::GGA | libxc::LibXCFamily::HybridGGA => {
                    if spin_channel==1 {
                        let (tmp_exc,tmp_vrho, tmp_vsigma) = xc_func.gga_exc_vxc(rho.data_ref().unwrap(),sigma.data_ref().unwrap());
                        let tmp_exc = MatrixFull::from_vec([num_grids,1],tmp_exc).unwrap();
                        let tmp_vrho = MatrixFull::from_vec([num_grids,1],tmp_vrho).unwrap();
                        let tmp_vsigma= MatrixFull::from_vec([num_grids,1],tmp_vsigma).unwrap();
                        exc.par_self_scaled_add(&tmp_exc,*xc_para);
                        vrho.par_self_scaled_add(&tmp_vrho,*xc_para);
                        vsigma.par_self_scaled_add(&tmp_vsigma, *xc_para);
                    } else {
                        let (tmp_exc,tmp_vrho, tmp_vsigma) = xc_func.gga_exc_vxc(rho.transpose().data_ref().unwrap(),sigma.transpose().data_ref().unwrap());
                        let tmp_exc = MatrixFull::from_vec([num_grids,1],tmp_exc).unwrap();
                        let tmp_vrho = MatrixFull::from_vec([2,num_grids],tmp_vrho).unwrap();
                        let tmp_vsigma= MatrixFull::from_vec([3,num_grids],tmp_vsigma).unwrap();
                        exc.par_self_scaled_add(&tmp_exc,*xc_para);
                        vrho.par_self_scaled_add(&tmp_vrho.transpose_and_drop(),*xc_para);
                        vsigma.par_self_scaled_add(&tmp_vsigma.transpose_and_drop(), *xc_para);
                    }
                },
                _ => {println!("{} is not yet implemented", xc_func.get_family_name())}
            }
        });

        let dt4 = utilities::timing(&dt3, Some("evaluate vrho and vsigma"));
        
        if let Some(ao) = &grids.ao {
            let ao_ref = ao.to_matrixfullslice();
            // for vrho
            for i_spin in  0..spin_channel {
                let mut vxc_ao_s = &mut vxc_ao[i_spin];
                let vrho_s = vrho.slice_column(i_spin);
                let ao_ref = ao.to_matrixfullslice();
                // generate vxc grid by grid
                contract_vxc_0(vxc_ao_s, &ao_ref, vrho_s, None);
            }
            // for vsigma
            if self.use_density_gradient() {
                if let Some(aop) = &grids.aop {
                    if spin_channel==1 {
                        // vxc_ao_s: the shape of [num_basis, num_grids]
                        let mut vxc_ao_s = &mut vxc_ao[0];
                        // vsigma_s: a slice with the length of [num_grids]
                        let vsigma_s = vsigma.slice_column(0);
                        // rhop_s:  the shape of [num_grids, 3]
                        let rhop_s = rhop.get_reducing_matrix(0).unwrap();
                        
                        // (nabla rho)[num_grids, 3] dot (nabla ao)[num_basis, num_grids, 3] -> [num_basis, num_grids]
                        //               p,       n                    i,        p,       n  ->     i,       p
                        //   einsum(pn, ipn -> ip)
                        let mut wao = MatrixFull::new([num_basis, num_grids],0.0);
                        for x in 0usize..3usize {
                            // aop_x: the shape of [num_basis, num_grids]
                            let aop_x = aop.get_reducing_matrix(x).unwrap();
                            // rhop_s_x: a slice with the length of [num_grids]
                            let rhop_s_x = rhop_s.get_slice_x(x);
                            contract_vxc_0(&mut wao, &aop_x, rhop_s_x, None);
                        }

                        contract_vxc_0(vxc_ao_s, &wao.to_matrixfullslice(), vsigma_s,Some(4.0));

                        //println!("debug awo:");
                        //(0..100).for_each(|i| {
                        //    println!("{:16.8},{:16.8}",vxc_ao_s[[0,i]],vxc_ao_s[[1,i]]);
                        //});

                    } else {
                        // ==================================
                        // at first i_spin == 0
                        // ==================================
                        {
                            let mut vxc_ao_a = &mut vxc_ao[0];
                            let rhop_a = rhop.get_reducing_matrix(0).unwrap();
                            let vsigma_uu = vsigma.slice_column(0);
                            let mut dao = MatrixFull::new([num_basis, num_grids],0.0);
                            for x in 0usize..3usize {
                                // aop_x: the shape of [num_basis, num_grids]
                                let aop_x = aop.get_reducing_matrix(x).unwrap();
                                // rhop_s_x: a slice with the length of [num_grids]
                                let rhop_s_x = rhop_a.get_slice_x(x);
                                contract_vxc_0(&mut dao, &aop_x, rhop_s_x,None);
                            }
                            contract_vxc_0(vxc_ao_a, &dao.to_matrixfullslice(), &vsigma_uu,Some(4.0));

                            let rhop_b = rhop.get_reducing_matrix(1).unwrap();
                            let vsigma_ud = vsigma.slice_column(1);
                            dao.data.iter_mut().for_each(|d| {*d=0.0});
                            for x in 0usize..3usize {
                                // aop_x: the shape of [num_basis, num_grids]
                                let aop_x = aop.get_reducing_matrix(x).unwrap();
                                // rhop_s_x: a slice with the length of [num_grids]
                                let rhop_s_x = rhop_b.get_slice_x(x);
                                contract_vxc_0(&mut dao, &aop_x, rhop_s_x,None);
                            }
                            contract_vxc_0(vxc_ao_a, &dao.to_matrixfullslice(), &vsigma_ud,Some(2.0));
                        }
                        // ==================================
                        // them i_spin == 1
                        // ==================================
                        {
                            let mut vxc_ao_b = &mut vxc_ao[1];
                            let rhop_b = rhop.get_reducing_matrix(1).unwrap();
                            let vsigma_dd = vsigma.slice_column(2);
                            let mut dao = MatrixFull::new([num_basis, num_grids],0.0);
                            for x in 0usize..3usize {
                                // aop_x: the shape of [num_basis, num_grids]
                                let aop_x = aop.get_reducing_matrix(x).unwrap();
                                // rhop_s_x: a slice with the length of [num_grids]
                                let rhop_s_x = rhop_b.get_slice_x(x);
                                contract_vxc_0(&mut dao, &aop_x, rhop_s_x,None);
                            }
                            contract_vxc_0(vxc_ao_b, &dao.to_matrixfullslice(), &vsigma_dd,Some(4.0));

                            let rhop_a = rhop.get_reducing_matrix(0).unwrap();
                            let vsigma_ud = vsigma.slice_column(1);
                            dao.data.iter_mut().for_each(|d| {*d=0.0});
                            for x in 0usize..3usize {
                                // aop_x: the shape of [num_basis, num_grids]
                                let aop_x = aop.get_reducing_matrix(x).unwrap();
                                // rhop_s_x: a slice with the length of [num_grids]
                                let rhop_s_x = rhop_a.get_slice_x(x);
                                contract_vxc_0(&mut dao, &aop_x, rhop_s_x,None);
                            }
                            contract_vxc_0(vxc_ao_b, &dao.to_matrixfullslice(), &vsigma_ud,Some(2.0));
                        }
                        // ==================================


                    }
                }
            }
        }
        //println!("debug ");
        //(0..100).for_each(|i| {
        //    println!("{:16.8},{:16.8},{:16.8}", vsigma[[i,0]],vsigma[[i,1]],vsigma[[i,2]]);
        //});

        let dt5 = utilities::timing(&dt4, Some("from vrho -> vxc_ao"));

        let mut total_elec = [0.0;2];
        for i_spin in 0..spin_channel {
            let mut total_elec_s = total_elec.get_mut(i_spin).unwrap();
            exc_total[i_spin] = izip!(exc.data.iter(),rho.iter_column(i_spin),grids.weights.iter())
                .fold(0.0,|acc,(exc,rho,weight)| {
                    *total_elec_s += rho*weight;
                    acc + exc * rho * weight
                });
            //exc.data.iter_mut().zip(rho.iter_j(i_spin)).for_each(|(exc,rho)| {
            //    *exc  = *exc* rho
        }
        if print_level > 0 {
            if spin_channel==1 {
                println!("total electron number: {:16.8}", total_elec[0])
            } else {
                println!("electron number in alpha-channel: {:12.8}", total_elec[0]);
                println!("electron number in beta-channel:  {:12.8}", total_elec[1]);
            }
        }
        let dt6 = utilities::timing(&dt5, Some("evaluate exc and en"));

        for i_spin in 0..spin_channel {
            let vxc_ao_s = vxc_ao.get_mut(i_spin).unwrap();
            vxc_ao_s.iter_columns_full_mut().zip(grids.weights.iter()).for_each(|(vxc_ao_s,w)| {
                vxc_ao_s.iter_mut().for_each(|f| {*f *= *w})
            });
        }

        let dt7 = utilities::timing(&dt6, Some("weight vxc_ao"));

        (exc_total,vxc_ao)
    }
    pub fn xc_exc_vxc_slots_dm_only(&self, 
        range_grids: Range<usize>, 
        grids: &Grids, 
        spin_channel: usize, 
        dm: &Vec<MatrixFull<f64>>, 
        mo: &[MatrixFull<f64>;2], 
        occ: &[Vec<f64>;2]) -> (Vec<f64>, Vec<MatrixFull<f64>>,[f64;2]) 
        {
        //let num_grids = grids.coordinates.len();
        let num_grids = range_grids.len();
        let num_basis = dm[0].size[0];


        let loc_coordinates = &grids.coordinates[range_grids.clone()];
        let loc_weights = &grids.weights[range_grids.clone()];

        //println!("thread_id: {:?}, rayon_threads_number: {:?}, omp_threads_number: {:?}",
        //    rayon::current_thread_index().unwrap(), rayon::current_num_threads(), utilities::omp_get_num_threads_wrapper());

        let mut loc_exc = MatrixFull::new([num_grids,1],0.0);
        let mut loc_exc_total = vec![0.0;spin_channel];
        let mut loc_vxc_ao = vec![MatrixFull::new([num_basis,num_grids],0.0);spin_channel];
        //println!("debug loc_vxc_ao size: {:?}", loc_vxc_ao[0].size());
        let dt0 = utilities::init_timing();

        /// rho and rhop have been localized.
        //let (loc_rho,loc_rhop) = grids.prepare_tabulated_density_slots(mo, occ, spin_channel,range_grids.clone());
        let (loc_rho,loc_rhop) = grids.prepare_tabulated_density_slots_dm_only(dm, spin_channel,range_grids.clone());
        let loc_sigma = if self.use_density_gradient() {
            prepare_tabulated_sigma(&loc_rhop, spin_channel)
        } else {
            MatrixFull::empty()
        };

        let mut loc_vrho = MatrixFull::new([num_grids,spin_channel],0.0);
        let mut loc_vsigma=if self.use_density_gradient() && spin_channel==1 {
            MatrixFull::new([num_grids,1],0.0)
        } else if self.use_density_gradient() && spin_channel==2 {
            MatrixFull::new([num_grids,3],0.0)
        } else {
            MatrixFull::empty()
        };

        self.dfa_compnt_scf.iter().zip(self.dfa_paramr_scf.iter()).for_each(|(xc_func,xc_para)| {
            let xc_func = self.init_libxc(xc_func);
            match xc_func.xc_func_family {
                libxc::LibXCFamily::LDA => {
                    if spin_channel==1 {
                        let (tmp_exc,tmp_vrho) = xc_func.lda_exc_vxc(loc_rho.data_ref().unwrap());
                        let tmp_exc = MatrixFull::from_vec([num_grids,1],tmp_exc).unwrap();
                        let tmp_vrho = MatrixFull::from_vec([num_grids,1],tmp_vrho).unwrap();
                        loc_exc.self_scaled_add(&tmp_exc,*xc_para);
                        loc_vrho.self_scaled_add(&tmp_vrho,*xc_para);

                    } else {
                        let (tmp_exc,tmp_vrho) = xc_func.lda_exc_vxc(loc_rho.transpose().data_ref().unwrap());
                        let tmp_exc = MatrixFull::from_vec([num_grids,1],tmp_exc).unwrap();
                        //let tmp_vrho = MatrixFull::from_vec([num_grids,spin_channel],tmp_vrho).unwrap();
                        let tmp_vrho = MatrixFull::from_vec([2,num_grids],tmp_vrho).unwrap();
                        loc_exc.self_scaled_add(&tmp_exc,*xc_para);
                        loc_vrho.self_scaled_add(&tmp_vrho.transpose_and_drop(),*xc_para);
                    }
                },
                libxc::LibXCFamily::GGA | libxc::LibXCFamily::HybridGGA => {
                    if spin_channel==1 {
                        let (tmp_exc,tmp_vrho, tmp_vsigma) = xc_func.gga_exc_vxc(loc_rho.data_ref().unwrap(),loc_sigma.data_ref().unwrap());
                        let tmp_exc = MatrixFull::from_vec([num_grids,1],tmp_exc).unwrap();
                        let tmp_vrho = MatrixFull::from_vec([num_grids,1],tmp_vrho).unwrap();
                        let tmp_vsigma= MatrixFull::from_vec([num_grids,1],tmp_vsigma).unwrap();
                        loc_exc.self_scaled_add(&tmp_exc,*xc_para);
                        loc_vrho.self_scaled_add(&tmp_vrho,*xc_para);
                        loc_vsigma.self_scaled_add(&tmp_vsigma, *xc_para);
                    } else {
                        let (tmp_exc,tmp_vrho, tmp_vsigma) = xc_func.gga_exc_vxc(loc_rho.transpose().data_ref().unwrap(),loc_sigma.transpose().data_ref().unwrap());
                        let tmp_exc = MatrixFull::from_vec([num_grids,1],tmp_exc).unwrap();
                        let tmp_vrho = MatrixFull::from_vec([2,num_grids],tmp_vrho).unwrap();
                        let tmp_vsigma= MatrixFull::from_vec([3,num_grids],tmp_vsigma).unwrap();
                        loc_exc.self_scaled_add(&tmp_exc,*xc_para);
                        loc_vrho.self_scaled_add(&tmp_vrho.transpose_and_drop(),*xc_para);
                        loc_vsigma.self_scaled_add(&tmp_vsigma.transpose_and_drop(), *xc_para);
                    }
                },
                _ => {println!("{} is not yet implemented", xc_func.get_family_name())}
            }
        });

        if let Some(ao) = &grids.ao {
            // for vrho
            for i_spin in  0..spin_channel {
                let mut loc_vxc_ao_s = &mut loc_vxc_ao[i_spin];
                let loc_vrho_s = loc_vrho.slice_column(i_spin);
                let loc_ao_ref = ao.to_matrixfullslice_columns(range_grids.clone());
                // generate vxc grid by grid
                contract_vxc_0_serial(loc_vxc_ao_s, &loc_ao_ref, loc_vrho_s, None);
            }
            // for vsigma
            if self.use_density_gradient() {
                if let Some(aop) = &grids.aop {
                    if spin_channel==1 {
                        // vxc_ao_s: the shape of [num_basis, num_grids]
                        let mut loc_vxc_ao_s = &mut loc_vxc_ao[0];
                        // vsigma_s: a slice with the length of [num_grids]
                        let loc_vsigma_s = loc_vsigma.slice_column(0);
                        // rhop_s:  the shape of [num_grids, 3]
                        let loc_rhop_s = loc_rhop.get_reducing_matrix(0).unwrap();
                        
                        // (nabla rho)[num_grids, 3] dot (nabla ao)[num_basis, num_grids, 3] -> [num_basis, num_grids]
                        //               p,       n                    i,        p,       n  ->     i,       p
                        //   einsum(pn, ipn -> ip)
                        let mut loc_wao = MatrixFull::new([num_basis, num_grids],0.0);
                        for x in 0usize..3usize {
                            // aop_x: the shape of [num_basis, num_grids]
                            let loc_aop_x = aop.get_reducing_matrix_columns(range_grids.clone(),x).unwrap();
                            // rhop_s_x: a slice with the length of [num_grids]
                            let loc_rhop_s_x = loc_rhop_s.get_slice_x(x);
                            contract_vxc_0_serial(&mut loc_wao, &loc_aop_x, loc_rhop_s_x, None);
                        }

                        contract_vxc_0_serial(loc_vxc_ao_s, &loc_wao.to_matrixfullslice(), loc_vsigma_s,Some(4.0));

                        //println!("debug awo:");
                        //(0..100).for_each(|i| {
                        //    println!("{:16.8},{:16.8}",vxc_ao_s[[0,i]],vxc_ao_s[[1,i]]);
                        //});

                    } else {
                        // ==================================
                        // at first i_spin == 0
                        // ==================================
                        {
                            let mut loc_vxc_ao_a = &mut loc_vxc_ao[0];
                            let loc_rhop_a = loc_rhop.get_reducing_matrix(0).unwrap();
                            let loc_vsigma_uu = loc_vsigma.slice_column(0);
                            let mut loc_dao = MatrixFull::new([num_basis, num_grids],0.0);
                            for x in 0usize..3usize {
                                // aop_x: the shape of [num_basis, num_grids]
                                let loc_aop_x = aop.get_reducing_matrix_columns(range_grids.clone(),x).unwrap();
                                // rhop_s_x: a slice with the length of [num_grids]
                                let loc_rhop_s_x = loc_rhop_a.get_slice_x(x);
                                contract_vxc_0_serial(&mut loc_dao, &loc_aop_x, loc_rhop_s_x,None);
                            }
                            contract_vxc_0_serial(loc_vxc_ao_a, &loc_dao.to_matrixfullslice(), &loc_vsigma_uu,Some(4.0));

                            let loc_rhop_b = loc_rhop.get_reducing_matrix(1).unwrap();
                            let loc_vsigma_ud = loc_vsigma.slice_column(1);
                            loc_dao.data.iter_mut().for_each(|d| {*d=0.0});
                            for x in 0usize..3usize {
                                // aop_x: the shape of [num_basis, num_grids]
                                let loc_aop_x = aop.get_reducing_matrix_columns(range_grids.clone(),x).unwrap();
                                // rhop_s_x: a slice with the length of [num_grids]
                                let loc_rhop_s_x = loc_rhop_b.get_slice_x(x);
                                contract_vxc_0_serial(&mut loc_dao, &loc_aop_x, loc_rhop_s_x,None);
                            }
                            contract_vxc_0_serial(loc_vxc_ao_a, &loc_dao.to_matrixfullslice(), &loc_vsigma_ud,Some(2.0));
                        }
                        // ==================================
                        // them i_spin == 1
                        // ==================================
                        {
                            let mut loc_vxc_ao_b = &mut loc_vxc_ao[1];
                            let loc_rhop_b = loc_rhop.get_reducing_matrix(1).unwrap();
                            let loc_vsigma_dd = loc_vsigma.slice_column(2);
                            let mut loc_dao = MatrixFull::new([num_basis, num_grids],0.0);
                            for x in 0usize..3usize {
                                // aop_x: the shape of [num_basis, num_grids]
                                let loc_aop_x = aop.get_reducing_matrix_columns(range_grids.clone(),x).unwrap();
                                // rhop_s_x: a slice with the length of [num_grids]
                                let loc_rhop_s_x = loc_rhop_b.get_slice_x(x);
                                contract_vxc_0_serial(&mut loc_dao, &loc_aop_x, loc_rhop_s_x,None);
                            }
                            contract_vxc_0_serial(loc_vxc_ao_b, &loc_dao.to_matrixfullslice(), &loc_vsigma_dd,Some(4.0));

                            let loc_rhop_a = loc_rhop.get_reducing_matrix(0).unwrap();
                            let loc_vsigma_ud = loc_vsigma.slice_column(1);
                            loc_dao.data.iter_mut().for_each(|d| {*d=0.0});
                            for x in 0usize..3usize {
                                // aop_x: the shape of [num_basis, num_grids]
                                let loc_aop_x = aop.get_reducing_matrix_columns(range_grids.clone(),x).unwrap();
                                // rhop_s_x: a slice with the length of [num_grids]
                                let loc_rhop_s_x = loc_rhop_a.get_slice_x(x);
                                contract_vxc_0_serial(&mut loc_dao, &loc_aop_x, loc_rhop_s_x,None);
                            }
                            contract_vxc_0_serial(loc_vxc_ao_b, &loc_dao.to_matrixfullslice(), &loc_vsigma_ud,Some(2.0));
                        }
                        // ==================================


                    }
                }
            }
        }
        //println!("debug ");
        //(0..100).for_each(|i| {
        //    println!("{:16.8},{:16.8},{:16.8}", vsigma[[i,0]],vsigma[[i,1]],vsigma[[i,2]]);
        //});

        let mut loc_total_elec = [0.0;2];
        for i_spin in 0..spin_channel {
            let mut loc_total_elec_s = loc_total_elec.get_mut(i_spin).unwrap();
            loc_exc_total[i_spin] = izip!(loc_exc.data.iter(),loc_rho.iter_column(i_spin),loc_weights.iter())
                .fold(0.0,|acc,(exc,rho,weight)| {
                    *loc_total_elec_s += rho*weight;
                    acc + exc * rho * weight
                });
            //exc.data.iter_mut().zip(rho.iter_j(i_spin)).for_each(|(exc,rho)| {
            //    *exc  = *exc* rho
        }
        //if let Some(id) = rayon::current_thread_index() {

        //}

        for i_spin in 0..spin_channel {
            let loc_vxc_ao_s = loc_vxc_ao.get_mut(i_spin).unwrap();
            loc_vxc_ao_s.iter_columns_full_mut().zip(loc_weights.iter()).for_each(|(vxc_ao_s,w)| {
                vxc_ao_s.iter_mut().for_each(|f| {*f *= *w})
            });
        }

        (loc_exc_total,loc_vxc_ao,loc_total_elec)
    }

    pub fn xc_exc_vxc_slots(&self, 
        range_grids: Range<usize>, 
        grids: &Grids, 
        spin_channel: usize, 
        dm: &Vec<MatrixFull<f64>>, 
        mo: &[MatrixFull<f64>;2], 
        occ: &[Vec<f64>;2]) -> (Vec<f64>, Vec<MatrixFull<f64>>,[f64;2]) 
        {
        //let num_grids = grids.coordinates.len();
        let num_grids = range_grids.len();
        let num_basis = dm[0].size[0];


        let loc_coordinates = &grids.coordinates[range_grids.clone()];
        let loc_weights = &grids.weights[range_grids.clone()];

        //println!("thread_id: {:?}, rayon_threads_number: {:?}, omp_threads_number: {:?}",
        //    rayon::current_thread_index().unwrap(), rayon::current_num_threads(), utilities::omp_get_num_threads_wrapper());

        let mut loc_exc = MatrixFull::new([num_grids,1],0.0);
        let mut loc_exc_total = vec![0.0;spin_channel];
        let mut loc_vxc_ao = vec![MatrixFull::new([num_basis,num_grids],0.0);spin_channel];
        let dt0 = utilities::init_timing();

        /// rho and rhop have been localized.
        let (loc_rho,loc_rhop) = grids.prepare_tabulated_density_slots(mo, occ, spin_channel,range_grids.clone());
        let loc_sigma = if self.use_density_gradient() {
            prepare_tabulated_sigma(&loc_rhop, spin_channel)
        } else {
            MatrixFull::empty()
        };

        let mut loc_vrho = MatrixFull::new([num_grids,spin_channel],0.0);
        let mut loc_vsigma=if self.use_density_gradient() && spin_channel==1 {
            MatrixFull::new([num_grids,1],0.0)
        } else if self.use_density_gradient() && spin_channel==2 {
            MatrixFull::new([num_grids,3],0.0)
        } else {
            MatrixFull::empty()
        };

        self.dfa_compnt_scf.iter().zip(self.dfa_paramr_scf.iter()).for_each(|(xc_func,xc_para)| {
            let xc_func = self.init_libxc(xc_func);
            match xc_func.xc_func_family {
                libxc::LibXCFamily::LDA => {
                    if spin_channel==1 {
                        let (tmp_exc,tmp_vrho) = xc_func.lda_exc_vxc(loc_rho.data_ref().unwrap());
                        let tmp_exc = MatrixFull::from_vec([num_grids,1],tmp_exc).unwrap();
                        let tmp_vrho = MatrixFull::from_vec([num_grids,1],tmp_vrho).unwrap();
                        loc_exc.self_scaled_add(&tmp_exc,*xc_para);
                        loc_vrho.self_scaled_add(&tmp_vrho,*xc_para);

                    } else {
                        let (tmp_exc,tmp_vrho) = xc_func.lda_exc_vxc(loc_rho.transpose().data_ref().unwrap());
                        let tmp_exc = MatrixFull::from_vec([num_grids,1],tmp_exc).unwrap();
                        //let tmp_vrho = MatrixFull::from_vec([num_grids,spin_channel],tmp_vrho).unwrap();
                        let tmp_vrho = MatrixFull::from_vec([2,num_grids],tmp_vrho).unwrap();
                        loc_exc.self_scaled_add(&tmp_exc,*xc_para);
                        loc_vrho.self_scaled_add(&tmp_vrho.transpose_and_drop(),*xc_para);
                    }
                },
                libxc::LibXCFamily::GGA | libxc::LibXCFamily::HybridGGA => {
                    if spin_channel==1 {
                        let (tmp_exc,tmp_vrho, tmp_vsigma) = xc_func.gga_exc_vxc(loc_rho.data_ref().unwrap(),loc_sigma.data_ref().unwrap());
                        let tmp_exc = MatrixFull::from_vec([num_grids,1],tmp_exc).unwrap();
                        let tmp_vrho = MatrixFull::from_vec([num_grids,1],tmp_vrho).unwrap();
                        let tmp_vsigma= MatrixFull::from_vec([num_grids,1],tmp_vsigma).unwrap();
                        loc_exc.self_scaled_add(&tmp_exc,*xc_para);
                        loc_vrho.self_scaled_add(&tmp_vrho,*xc_para);
                        loc_vsigma.self_scaled_add(&tmp_vsigma, *xc_para);
                    } else {
                        let (tmp_exc,tmp_vrho, tmp_vsigma) = xc_func.gga_exc_vxc(loc_rho.transpose().data_ref().unwrap(),loc_sigma.transpose().data_ref().unwrap());
                        let tmp_exc = MatrixFull::from_vec([num_grids,1],tmp_exc).unwrap();
                        let tmp_vrho = MatrixFull::from_vec([2,num_grids],tmp_vrho).unwrap();
                        let tmp_vsigma= MatrixFull::from_vec([3,num_grids],tmp_vsigma).unwrap();
                        loc_exc.self_scaled_add(&tmp_exc,*xc_para);
                        loc_vrho.self_scaled_add(&tmp_vrho.transpose_and_drop(),*xc_para);
                        loc_vsigma.self_scaled_add(&tmp_vsigma.transpose_and_drop(), *xc_para);
                    }
                },
                _ => {println!("{} is not yet implemented", xc_func.get_family_name())}
            }
        });

        if let Some(ao) = &grids.ao {
            // for vrho
            for i_spin in  0..spin_channel {
                let mut loc_vxc_ao_s = &mut loc_vxc_ao[i_spin];
                let loc_vrho_s = loc_vrho.slice_column(i_spin);
                let loc_ao_ref = ao.to_matrixfullslice_columns(range_grids.clone());
                // generate vxc grid by grid
                contract_vxc_0_serial(loc_vxc_ao_s, &loc_ao_ref, loc_vrho_s, None);
            }
            // for vsigma
            if self.use_density_gradient() {
                if let Some(aop) = &grids.aop {
                    if spin_channel==1 {
                        // vxc_ao_s: the shape of [num_basis, num_grids]
                        let mut loc_vxc_ao_s = &mut loc_vxc_ao[0];
                        // vsigma_s: a slice with the length of [num_grids]
                        let loc_vsigma_s = loc_vsigma.slice_column(0);
                        // rhop_s:  the shape of [num_grids, 3]
                        let loc_rhop_s = loc_rhop.get_reducing_matrix(0).unwrap();
                        
                        // (nabla rho)[num_grids, 3] dot (nabla ao)[num_basis, num_grids, 3] -> [num_basis, num_grids]
                        //               p,       n                    i,        p,       n  ->     i,       p
                        //   einsum(pn, ipn -> ip)
                        let mut loc_wao = MatrixFull::new([num_basis, num_grids],0.0);
                        for x in 0usize..3usize {
                            // aop_x: the shape of [num_basis, num_grids]
                            let loc_aop_x = aop.get_reducing_matrix_columns(range_grids.clone(),x).unwrap();
                            // rhop_s_x: a slice with the length of [num_grids]
                            let loc_rhop_s_x = loc_rhop_s.get_slice_x(x);
                            contract_vxc_0_serial(&mut loc_wao, &loc_aop_x, loc_rhop_s_x, None);
                        }

                        contract_vxc_0_serial(loc_vxc_ao_s, &loc_wao.to_matrixfullslice(), loc_vsigma_s,Some(4.0));

                        //println!("debug awo:");
                        //(0..100).for_each(|i| {
                        //    println!("{:16.8},{:16.8}",vxc_ao_s[[0,i]],vxc_ao_s[[1,i]]);
                        //});

                    } else {
                        // ==================================
                        // at first i_spin == 0
                        // ==================================
                        {
                            let mut loc_vxc_ao_a = &mut loc_vxc_ao[0];
                            let loc_rhop_a = loc_rhop.get_reducing_matrix(0).unwrap();
                            let loc_vsigma_uu = loc_vsigma.slice_column(0);
                            let mut loc_dao = MatrixFull::new([num_basis, num_grids],0.0);
                            for x in 0usize..3usize {
                                // aop_x: the shape of [num_basis, num_grids]
                                let loc_aop_x = aop.get_reducing_matrix_columns(range_grids.clone(),x).unwrap();
                                // rhop_s_x: a slice with the length of [num_grids]
                                let loc_rhop_s_x = loc_rhop_a.get_slice_x(x);
                                contract_vxc_0_serial(&mut loc_dao, &loc_aop_x, loc_rhop_s_x,None);
                            }
                            contract_vxc_0_serial(loc_vxc_ao_a, &loc_dao.to_matrixfullslice(), &loc_vsigma_uu,Some(4.0));

                            let loc_rhop_b = loc_rhop.get_reducing_matrix(1).unwrap();
                            let loc_vsigma_ud = loc_vsigma.slice_column(1);
                            loc_dao.data.iter_mut().for_each(|d| {*d=0.0});
                            for x in 0usize..3usize {
                                // aop_x: the shape of [num_basis, num_grids]
                                let loc_aop_x = aop.get_reducing_matrix_columns(range_grids.clone(),x).unwrap();
                                // rhop_s_x: a slice with the length of [num_grids]
                                let loc_rhop_s_x = loc_rhop_b.get_slice_x(x);
                                contract_vxc_0_serial(&mut loc_dao, &loc_aop_x, loc_rhop_s_x,None);
                            }
                            contract_vxc_0_serial(loc_vxc_ao_a, &loc_dao.to_matrixfullslice(), &loc_vsigma_ud,Some(2.0));
                        }
                        // ==================================
                        // them i_spin == 1
                        // ==================================
                        {
                            let mut loc_vxc_ao_b = &mut loc_vxc_ao[1];
                            let loc_rhop_b = loc_rhop.get_reducing_matrix(1).unwrap();
                            let loc_vsigma_dd = loc_vsigma.slice_column(2);
                            let mut loc_dao = MatrixFull::new([num_basis, num_grids],0.0);
                            for x in 0usize..3usize {
                                // aop_x: the shape of [num_basis, num_grids]
                                let loc_aop_x = aop.get_reducing_matrix_columns(range_grids.clone(),x).unwrap();
                                // rhop_s_x: a slice with the length of [num_grids]
                                let loc_rhop_s_x = loc_rhop_b.get_slice_x(x);
                                contract_vxc_0_serial(&mut loc_dao, &loc_aop_x, loc_rhop_s_x,None);
                            }
                            contract_vxc_0_serial(loc_vxc_ao_b, &loc_dao.to_matrixfullslice(), &loc_vsigma_dd,Some(4.0));

                            let loc_rhop_a = loc_rhop.get_reducing_matrix(0).unwrap();
                            let loc_vsigma_ud = loc_vsigma.slice_column(1);
                            loc_dao.data.iter_mut().for_each(|d| {*d=0.0});
                            for x in 0usize..3usize {
                                // aop_x: the shape of [num_basis, num_grids]
                                let loc_aop_x = aop.get_reducing_matrix_columns(range_grids.clone(),x).unwrap();
                                // rhop_s_x: a slice with the length of [num_grids]
                                let loc_rhop_s_x = loc_rhop_a.get_slice_x(x);
                                contract_vxc_0_serial(&mut loc_dao, &loc_aop_x, loc_rhop_s_x,None);
                            }
                            contract_vxc_0_serial(loc_vxc_ao_b, &loc_dao.to_matrixfullslice(), &loc_vsigma_ud,Some(2.0));
                        }
                        // ==================================


                    }
                }
            }
        }
        //println!("debug ");
        //(0..100).for_each(|i| {
        //    println!("{:16.8},{:16.8},{:16.8}", vsigma[[i,0]],vsigma[[i,1]],vsigma[[i,2]]);
        //});

        let mut loc_total_elec = [0.0;2];
        for i_spin in 0..spin_channel {
            let mut loc_total_elec_s = loc_total_elec.get_mut(i_spin).unwrap();
            loc_exc_total[i_spin] = izip!(loc_exc.data.iter(),loc_rho.iter_column(i_spin),loc_weights.iter())
                .fold(0.0,|acc,(exc,rho,weight)| {
                    *loc_total_elec_s += rho*weight;
                    acc + exc * rho * weight
                });
            //exc.data.iter_mut().zip(rho.iter_j(i_spin)).for_each(|(exc,rho)| {
            //    *exc  = *exc* rho
        }
        //if let Some(id) = rayon::current_thread_index() {

        //}

        for i_spin in 0..spin_channel {
            let loc_vxc_ao_s = loc_vxc_ao.get_mut(i_spin).unwrap();
            loc_vxc_ao_s.iter_columns_full_mut().zip(loc_weights.iter()).for_each(|(vxc_ao_s,w)| {
                vxc_ao_s.iter_mut().for_each(|f| {*f *= *w})
            });
        }

        (loc_exc_total,loc_vxc_ao,loc_total_elec)
    }

    pub fn post_xc_exc(&self, post_xc: &Vec<String>, grids: &crate::dft::Grids, dm: &Vec<MatrixFull<f64>>, mo: &[MatrixFull<f64>;2], occ: &[Vec<f64>;2]) 
    -> Vec<[f64;2]> {
        let mut post_xc_energy:Vec<[f64;2]>=vec![];
        let spin_channel = self.spin_channel;
        let num_grids = grids.coordinates.len();
        let num_basis = dm[0].size[0];
        let dt0 = utilities::init_timing();
        let (rho,rhop) = grids.prepare_tabulated_density_2(mo, occ, spin_channel);
        let use_density_gradient = post_xc.iter().fold(false,|flag, x| {
            let code = DFA4REST::xc_func_init_fdqc(x,spin_channel);
            let x_flag = code.iter().fold(false, |flag, xc_code| {
                let xc_func = self.init_libxc(xc_code);
                flag || xc_func.is_gga()|| xc_func.is_hybrid_gga()
            });
            flag || x_flag
        });
        let sigma = if use_density_gradient {
            prepare_tabulated_sigma_rayon(&rhop, spin_channel)
        } else {
            MatrixFull::empty()
        };
        post_xc.iter().for_each(|x| {
            let mut exc = MatrixFull::new([num_grids,1],0.0);
            let mut exc_total =[0.0,0.0];
            let code = DFA4REST::xc_func_init_fdqc(x,spin_channel);
            //println!("debug xc_code: {:?}", &code);
            code.iter().for_each(|xc_code| {
                exc.par_self_scaled_add(&self.xc_exc_code(xc_code, &rho, &sigma, spin_channel),1.0);
            });

            for i_spin in 0..spin_channel {
                exc_total[i_spin] = izip!(exc.data.iter(),rho.iter_column(i_spin),grids.weights.iter())
                    .fold(0.0,|acc,(exc,rho,weight)| {
                        acc + exc * rho * weight
                    });
            };
            //println!("exc_total: {:?}", &exc_total);

            post_xc_energy.push(exc_total);
        });

        post_xc_energy

    }

    pub fn xc_exc(&self, grids: &mut Grids, spin_channel: usize, dm: &mut Vec<MatrixFull<f64>>, mo: &mut [MatrixFull<f64>;2], occ: &mut [Vec<f64>;2],iop: usize) -> Vec<f64> {
        let num_grids = grids.coordinates.len();
        let num_basis = dm[0].size[0];
        let mut exc = MatrixFull::new([num_grids,1],0.0);
        let mut exc_total = vec![0.0;spin_channel];
        let dt0 = utilities::init_timing();
        let (rho,rhop) = grids.prepare_tabulated_density_2(mo, occ, spin_channel);
        let dt2 = utilities::timing(&dt0, Some("evaluate rho and rhop"));
        let sigma = if self.use_density_gradient() {
            prepare_tabulated_sigma_rayon(&rhop, spin_channel)
        } else {
            MatrixFull::empty()
        };

        if iop==0 {  // for the SCF energy
            //let rho_trans = if spin_channel== 1 {
            //    None
            //} else {
            //    Some(rho.transpose())
            //};
            self.dfa_compnt_scf.iter().zip(self.dfa_paramr_scf.iter()).for_each(|(xc_func,xc_para)| {
                let xc_func = self.init_libxc(xc_func);
                match xc_func.xc_func_family {
                    libxc::LibXCFamily::LDA => {
                        let tmp_exc = MatrixFull::from_vec([num_grids,1],
                            if spin_channel==1 {
                                xc_func.lda_exc(rho.data_ref().unwrap())
                            } else {
                                xc_func.lda_exc(rho.transpose().data_ref().unwrap())
                            }
                        ).unwrap();
                        exc.par_self_scaled_add(&tmp_exc,*xc_para);
                    },
                    libxc::LibXCFamily::GGA | libxc::LibXCFamily::HybridGGA => {
                        let tmp_exc = MatrixFull::from_vec([num_grids,1],
                            if spin_channel==1 {
                                xc_func.gga_exc(rho.data_ref().unwrap(),sigma.data_ref().unwrap())
                            } else {
                                xc_func.gga_exc(rho.transpose().data_ref().unwrap(),sigma.transpose().data_ref().unwrap())
                            }
                        ).unwrap();
                        exc.par_self_scaled_add(&tmp_exc,*xc_para);
                    },
                    _ => {println!("{} is not yet implemented", xc_func.get_family_name())}
                }
            });
        } else if iop==1 { // for the post-SCF energy calculation
            if let (Some(dfa_paramr),Some(dfa_compnt)) = (&self.dfa_paramr_pos, &self.dfa_compnt_pos) {
                dfa_compnt.iter().zip(dfa_paramr.iter()).for_each(|(xc_func,xc_para)| {
                    let xc_func = self.init_libxc(xc_func);
                    match xc_func.xc_func_family {
                        libxc::LibXCFamily::LDA => {
                            let tmp_exc = MatrixFull::from_vec([num_grids,1],
                                if spin_channel==1 {
                                    xc_func.lda_exc(rho.data_ref().unwrap())
                                } else {
                                    xc_func.lda_exc(rho.transpose().data_ref().unwrap())
                                }
                            ).unwrap();
                            exc.par_self_scaled_add(&tmp_exc,*xc_para);
                        },
                        libxc::LibXCFamily::GGA | libxc::LibXCFamily::HybridGGA => {
                            let tmp_exc = MatrixFull::from_vec([num_grids,1],
                                if spin_channel==1 {
                                    xc_func.gga_exc(rho.data_ref().unwrap(),sigma.data_ref().unwrap())
                                } else {
                                    xc_func.gga_exc(rho.transpose().data_ref().unwrap(),sigma.transpose().data_ref().unwrap())
                                }
                            ).unwrap();
                            exc.par_self_scaled_add(&tmp_exc,*xc_para);
                        },
                        _ => {println!("{} is not yet implemented", xc_func.get_family_name())}
                    }
                });
            }
        }
        let mut total_elec = [0.0;2];
        for i_spin in 0..spin_channel {
            let mut total_elec_s = total_elec.get_mut(i_spin).unwrap();
            exc_total[i_spin] = izip!(exc.data.iter(),rho.iter_column(i_spin),grids.weights.iter())
                .fold(0.0,|acc,(exc,rho,weight)| {
                    *total_elec_s += rho*weight;
                    acc + exc * rho * weight
                });
            //exc.data.iter_mut().zip(rho.iter_j(i_spin)).for_each(|(exc,rho)| {
            //    *exc  = *exc* rho
        }
        //if spin_channel==1 {
        //    println!("total electron number: {:16.8}", total_elec[0])
        //} else {
        //    println!("electron number in alpha-channel: {:12.8}", total_elec[0]);
        //    println!("electron number in beta-channel:  {:12.8}", total_elec[1]);
        //}
        exc_total
    }
    pub fn xc_exc_code(&self, xc_code: &usize, rho: &MatrixFull<f64>, sigma:&MatrixFull<f64>, spin_channel: usize) -> MatrixFull<f64> {
        let xc_func = self.init_libxc(xc_code);
        let num_grids = rho.size()[0];
        let tmp_exc = match xc_func.xc_func_family {
            libxc::LibXCFamily::LDA => {
                MatrixFull::from_vec([num_grids,1],
                    if spin_channel==1 {
                        xc_func.lda_exc(rho.data_ref().unwrap())
                    } else {
                        xc_func.lda_exc(rho.transpose().data_ref().unwrap())
                    }
                ).unwrap()
                //exc.par_self_scaled_add(&tmp_exc,*xc_para);
            },
            libxc::LibXCFamily::GGA | libxc::LibXCFamily::HybridGGA => {
                //println!("debug, {:?}, {:?}, {:?}", xc_code, xc_func, sigma.data.len());
                MatrixFull::from_vec([num_grids,1],
                    if spin_channel==1 {
                        xc_func.gga_exc(rho.data_ref().unwrap(),sigma.data_ref().unwrap())
                    } else {
                        xc_func.gga_exc(rho.transpose().data_ref().unwrap(),sigma.transpose().data_ref().unwrap())
                    }
                ).unwrap()
                //exc.par_self_scaled_add(&tmp_exc,*xc_para);
            },
            _ => {panic!("{} is not yet implemented", xc_func.get_family_name())}
        };
        tmp_exc
    }
}

pub fn contract_vxc_0(mat_a: &mut MatrixFull<f64>, mat_b: &MatrixFullSlice<f64>, slice_c: &[f64], scaling_factor: Option<f64>) {
    match scaling_factor {
        None =>  {
            mat_a.par_iter_columns_full_mut().zip(mat_b.par_iter_columns_full()).map(|(mat_a,mat_b)| (mat_a,mat_b))
            .zip(slice_c.par_iter())
            .for_each(|((mat_a,mat_b), slice_c)| {
                    mat_a.iter_mut().zip(mat_b.iter()).for_each(|(mat_a, mat_b)| {
                        *mat_a += mat_b*slice_c
                });
            });
            //mat_a.iter_mut_columns_full().zip(mat_b.iter_columns_full()).map(|(mat_a,mat_b)| (mat_a,mat_b))
            //.zip(slice_c.iter())
            //.for_each(|((mat_a,mat_b), slice_c)| {
            //        mat_a.iter_mut().zip(mat_b.iter()).for_each(|(mat_a, mat_b)| {
            //            *mat_a += mat_b*slice_c
            //    });
            //});
        },
        Some(s) => {
            mat_a.par_iter_columns_full_mut().zip(mat_b.par_iter_columns_full()).map(|(mat_a,mat_b)| (mat_a,mat_b))
            .zip(slice_c.par_iter())
            .for_each(|((mat_a,mat_b), slice_c)| {
                    mat_a.iter_mut().zip(mat_b.iter()).for_each(|(mat_a, mat_b)| {
                        *mat_a += mat_b*slice_c*s
                });
            });
            //izip!(mat_a.iter_mut_columns_full(),mat_b.iter_columns_full(), slice_c.iter())
            //    .for_each(|(mat_a,mat_b, slice_c)| {
            //        mat_a.iter_mut().zip(mat_b.iter()).for_each(|(mat_a, mat_b)| {
            //            *mat_a += mat_b*slice_c*s
            //    });
            //});

        }
    }
}

/// Prepare `sigma[0] = rhop_u dot rhop_u => sigma_uu`
///         `sigma[1] = rhop_u dot rhop_d => sigma_ud`
///         `sigma[2] = rhop_d dot rhop_d => sigma_dd`
fn prepare_tabulated_sigma(rhop: &RIFull<f64>, spin_channel: usize) -> MatrixFull<f64> {
    let grids_len = rhop.size[0];
    if spin_channel==1 {
            let mut sigma = MatrixFull::new([grids_len,1],0.0);
            let rhop_x = rhop.iter_slices_x(0, 0);
            let rhop_y = rhop.iter_slices_x(1, 0);
            let rhop_z = rhop.iter_slices_x(2, 0);
            izip!(sigma.iter_column_mut(0), rhop_x,rhop_y,rhop_z).for_each(|(sigma, dx,dy,dz)| {
                *sigma = dx.powf(2.0) + dy.powf(2.0) + dz.powf(2.0);
            });
            return sigma
        } else {
            let mut sigma = MatrixFull::new([grids_len,3],0.0);
            let rhop_xu = rhop.iter_slices_x(0, 0);
            let rhop_yu = rhop.iter_slices_x(1, 0);
            let rhop_zu = rhop.iter_slices_x(2, 0);
            izip!(sigma.iter_column_mut(0), rhop_xu,rhop_yu,rhop_zu).for_each(|(sigma, dx,dy,dz)| {
                *sigma = dx.powf(2.0) + dy.powf(2.0) + dz.powf(2.0);
            });
            let rhop_xu = rhop.iter_slices_x(0,0);
            let rhop_yu = rhop.iter_slices_x(1,0);
            let rhop_zu = rhop.iter_slices_x(2,0);
            let rhop_xd = rhop.iter_slices_x(0,1);
            let rhop_yd = rhop.iter_slices_x(1,1);
            let rhop_zd = rhop.iter_slices_x(2,1);
            izip!(sigma.iter_column_mut(1), rhop_xu,rhop_yu,rhop_zu, rhop_xd,rhop_yd,rhop_zd)
                .for_each(|(sigma, dxu,dyu,dzu, dxd, dyd, dzd)| {
                *sigma = dxu*dxd+dyu*dyd+dzu*dzd;
            });
            let rhop_xd = rhop.iter_slices_x(0,1);
            let rhop_yd = rhop.iter_slices_x(1,1);
            let rhop_zd = rhop.iter_slices_x(2,1);
            izip!(sigma.iter_column_mut(2), rhop_xd,rhop_yd,rhop_zd).for_each(|(sigma, dx,dy,dz)| {
                *sigma = dx.powf(2.0) + dy.powf(2.0) + dz.powf(2.0);
            });
            return sigma
        }
}

/// Rayon parallel version to prepare 
///         `sigma[0] = rhop_u dot rhop_u => sigma_uu`
///         `sigma[1] = rhop_u dot rhop_d => sigma_ud`
///         `sigma[2] = rhop_d dot rhop_d => sigma_dd`
fn prepare_tabulated_sigma_rayon(rhop: &RIFull<f64>, spin_channel: usize) -> MatrixFull<f64> {
    let grids_len = rhop.size[0];
    if spin_channel==1 {
            let mut sigma = MatrixFull::new([grids_len,1],0.0);
            let rhop_x = rhop.par_iter_slices_x(0, 0);
            let rhop_y = rhop.par_iter_slices_x(1, 0);
            let rhop_z = rhop.par_iter_slices_x(2, 0);
            //izip!(sigma.par_iter_column_mut(0), rhop_x,rhop_y,rhop_z).for_each(|(sigma, dx,dy,dz)| {
            //    *sigma = dx.powf(2.0) + dy.powf(2.0) + dz.powf(2.0);
            //});
            sigma.par_iter_column_mut(0).zip(rhop_x).zip(rhop_y).zip(rhop_z)
               .for_each(|(((sigma,dx),dy),dz)| {
                *sigma = dx.powf(2.0) + dy.powf(2.0) + dz.powf(2.0);
            });
            return sigma
        } else {
            let mut sigma = MatrixFull::new([grids_len,3],0.0);
            let rhop_xu = rhop.par_iter_slices_x(0, 0);
            let rhop_yu = rhop.par_iter_slices_x(1, 0);
            let rhop_zu = rhop.par_iter_slices_x(2, 0);
            //izip!(sigma.par_iter_column_mut(0), rhop_xu,rhop_yu,rhop_zu).for_each(|(sigma, dx,dy,dz)| {
            sigma.par_iter_column_mut(0).zip(rhop_xu).zip(rhop_yu).zip(rhop_zu)
               .for_each(|(((sigma,dx),dy),dz)| {
                *sigma = dx.powf(2.0) + dy.powf(2.0) + dz.powf(2.0);
            });
            let rhop_xu = rhop.par_iter_slices_x(0,0);
            let rhop_yu = rhop.par_iter_slices_x(1,0);
            let rhop_zu = rhop.par_iter_slices_x(2,0);
            let rhop_xd = rhop.par_iter_slices_x(0,1);
            let rhop_yd = rhop.par_iter_slices_x(1,1);
            let rhop_zd = rhop.par_iter_slices_x(2,1);
            //izip!(sigma.par_iter_column_mut(1), rhop_xu,rhop_yu,rhop_zu, rhop_xd,rhop_yd,rhop_zd)
            sigma.par_iter_column_mut(1).zip(rhop_xu).zip(rhop_yu).zip(rhop_zu).zip(rhop_xd).zip(rhop_yd).zip(rhop_zd)
                .for_each(|((((((sigma, dxu),dyu),dzu), dxd), dyd), dzd)| {
                *sigma = dxu*dxd+dyu*dyd+dzu*dzd;
            });
            let rhop_xd = rhop.par_iter_slices_x(0,1);
            let rhop_yd = rhop.par_iter_slices_x(1,1);
            let rhop_zd = rhop.par_iter_slices_x(2,1);
            //izip!(sigma.par_iter_column_mut(2), rhop_xd,rhop_yd,rhop_zd).for_each(|(sigma, dx,dy,dz)| {
            sigma.par_iter_column_mut(2).zip(rhop_xd).zip(rhop_yd).zip(rhop_zd)
               .for_each(|(((sigma,dx),dy),dz)| {
                *sigma = dx.powf(2.0) + dy.powf(2.0) + dz.powf(2.0);
            });
            return sigma
        }
}


#[derive(Clone)]
pub struct Grids {
    pub ao: Option<MatrixFull<f64>>,
    pub aop: Option<RIFull<f64>>,
    pub weights: Vec<f64>,
    pub coordinates: Vec<[f64;3]>,
    pub parallel_balancing: Vec<Range<usize>>,
}

impl Grids {
    pub fn build(mol: &Molecule) -> Grids {
        // set some system-independent parameters
        //let radial_precision = 1.0e-12;
        //let min_num_angular_points: usize = 590;
        //let max_num_angular_points: usize = 590;
        //let hardness: usize = 3;

        //let mut time_record = utilities::TimeRecords::new();
        //time_record.new_item("Overall", "the overall grid generation");

        if ! &mol.ctrl.external_grids.to_lowercase().eq("none") &&
            std::path::Path::new(&mol.ctrl.external_grids).is_file() {

            let dt0 = utilities::init_timing();


            let mut weights:Vec<f64> = Vec::new();
            let mut coordinates: Vec<[f64;3]> = Vec::new();

            let mut grids_file = std::fs::File::open(&mol.ctrl.external_grids).unwrap();
            let mut content = String::new();
            grids_file.read_to_string(&mut content);
            //println!("{}",&content);
            let re1 = Regex::new(r"(?x)\s*
                (?P<x>[\+-]?\d+.\d+[eE][\+-]?\d+)\s*,# the 'x' position
                \s*
                (?P<y>[\+-]?\d+.\d+[eE][\+-]?\d+)\s*,# the 'y' position
                \s*
                (?P<z>[\+-]?\d+.\d+[eE][\+-]?\d+)\s*,# the 'z' position
                \s*
                (?P<w>[\+-]?\d+.\d+[eE][\+-]?\d+)\s*# the 'w' weight
                \s*\n").unwrap();
            //if let Some(cap)  = re1.captures(&content) {
            //    println!("{:?}", &cap)
            //}
            for cap in re1.captures_iter(&content) {
                let x:f64 = cap[1].parse().unwrap();
                let y:f64 = cap[2].parse().unwrap();
                let z:f64 = cap[3].parse().unwrap();
                let w:f64 = cap[4].parse().unwrap();
                coordinates.push([x,y,z]);
                weights.push(w);
                //println!("{:16.8} {:16.8} {:16.8} {:16.8}", x,y,z,w);
            }

            //println!("Size of imported grids: {}",weights.len());

            utilities::timing(&dt0, Some("Importing the grids"));

            let parallel_balancing = balancing(coordinates.len(), rayon::current_num_threads());

            return Grids {
                weights,
                coordinates,
                ao: None,
                aop: None, 
                parallel_balancing,
            }
        }

        let dt0 = utilities::init_timing();

        let radial_precision = mol.ctrl.radial_precision;
        let min_num_angular_points: usize = mol.ctrl.min_num_angular_points;
        let max_num_angular_points: usize = mol.ctrl.max_num_angular_points;
        let hardness: usize = mol.ctrl.hardness;
        let pruning: String = mol.ctrl.pruning.clone();
        let grid_gen_level: usize = mol.ctrl.grid_gen_level;
        let rad_grid_method: String = mol.ctrl.rad_grid_method.clone();

        // obtain system-dependent parameters
        let mass_charge = get_mass_charge(&mol.geom.elem);
        let proton_charges: Vec<i32> = mass_charge.iter().map(|value| value.1 as i32).collect();
        let center_coordinates_bohr = mol.geom.to_numgrid_io();
        //mol.fdqc_bas[0].
        let mut alpha_max: Vec<f64> = vec![];
        let mut alpha_min: Vec<HashMap<usize,f64>> = vec![];
        mol.basis4elem.iter().for_each(|value| {
            let (tmp_alpha_min, tmp_alpha_max) = value.to_numgrid_io();
            alpha_max.push(tmp_alpha_max);
            alpha_min.push(tmp_alpha_min);
        });
        //println!("{:?}, {:?}",&alpha_min, &alpha_max);

        let mut num_points: usize = 0;
        let mut coordinates: Vec<[f64;3]> =vec![];
        let mut weights:Vec<f64> = vec![];

        alpha_min.iter().zip(alpha_max.iter()).enumerate().for_each(|(center_index,value)| {
            let (rs_atom, ws_atom) = gen_grids::atom_grid(
                value.0.clone(), 
                value.1.clone(), 
                radial_precision, 
                min_num_angular_points, 
                max_num_angular_points, 
                proton_charges.clone(), 
                center_index, 
                center_coordinates_bohr.clone(), 
                hardness,
                pruning.clone(),
                rad_grid_method.clone(),
                grid_gen_level,
            );
            //println!("alpha_min: {:?}, alpha_max: {:6.3}",&value.0, &value.1);
            //println!("rs_atom: {:?}, ws_atom: {:?}",&rs_atom, &ws_atom);
            num_points += rs_atom.len();
            coordinates.extend(rs_atom.iter().map(|value| [value.0,value.1,value.2]));
            weights.extend(ws_atom);
        });

       /*  let ngrids = weights.len();
        let mut x = vec![0.0; ngrids];
        let mut y = vec![0.0; ngrids];
        let mut z = vec![0.0; ngrids];
        x.iter_mut().zip(y.iter_mut()).zip(z.iter_mut()).zip(coordinates.iter()).for_each(|(((x,y),z), p)|{
            *x = p[0];
            *y = p[1];
            *z = p[2];
        });
        println!("x_is: {:?}", &x);
        println!("y_is: {:?}", &y);
        println!("z_is: {:?}", &z);
        println!("w_is: {:?}", &weights); */


        utilities::timing(&dt0, Some("Generating the grids"));
        let parallel_balancing = balancing(coordinates.len(), rayon::current_num_threads());
        Grids {
            weights,
            coordinates,
            ao: None,
            aop: None, 
            parallel_balancing
        }

    }
    pub fn build_nonstd(center_coordinates_bohr:Vec<(f64,f64,f64)>, proton_charges:Vec<i32>, alpha_min: Vec<HashMap<usize,f64>>, alpha_max:Vec<f64>) -> Grids {
        let radial_precision = 1.0e-12;
        let min_num_angular_points: usize = 50;
        let max_num_angular_points: usize = 50;
        let hardness: usize = 3;
        let pruning: String = String::from("sg1");
        let rad_grid_method: String = String::from("treutler");
        let grid_gen_level: usize = 3;

        let mut coordinates: Vec<[f64;3]> =vec![];
        let mut weights:Vec<f64> = vec![];
        let mut num_points:usize = 0;
        //println!("{:?}, {:?}",&alpha_min, &alpha_max);

        alpha_min.iter().zip(alpha_max.iter()).enumerate().for_each(|(center_index,value)| {
            let (rs_atom, ws_atom) = gen_grids::atom_grid(
                value.0.clone(), 
                value.1.clone(), 
                radial_precision, 
                min_num_angular_points, 
                max_num_angular_points, 
                proton_charges.clone(), 
                center_index, 
                center_coordinates_bohr.clone(), 
                hardness,
                pruning.clone(),
                rad_grid_method.clone(),
                grid_gen_level,

            );
            //println!("alpha_min: {:?}, alpha_max: {:6.3}",&value.0, &value.1);
            //println!("rs_atom: {:?}, ws_atom: {:?}",&rs_atom, &ws_atom);
            num_points += rs_atom.len();
            coordinates.extend(rs_atom.iter().map(|value| [value.0,value.1,value.2]));
            weights.extend(ws_atom);
        });


        Grids {
            weights,
            coordinates,
            ao: None,
            aop: None,
            parallel_balancing: vec![],
        }
    }

    pub fn formated_output(&self) {
        self.coordinates.iter().zip(self.weights.iter()).for_each(|value| {
            println!("r: ({:6.3},{:6.3},{:6.3}), w: {:16.8}",value.0[0],value.0[1],value.0[2],value.1);
        })
    }


    pub fn prepare_tabulated_ao(&mut self, mol: &Molecule) {
        self.prepare_tabulated_ao_rayon_v02(mol)
    }

    pub fn prepare_tabulated_ao_rayon_v02(&mut self, mol: &Molecule) {
        //In this subroutine, we call the lapack dgemm in a rayon parallel environment.
        //In order to ensure the efficiency, we disable the openmp ability and re-open it in the end of subroutien
        //let default_omp_num_threads = unsafe {utilities::openblas_get_num_threads()};
        let default_omp_num_threads = utilities::omp_get_num_threads_wrapper();
        utilities::omp_set_num_threads_wrapper(1);

        let num_grids = self.coordinates.len();
        let num_basis = mol.num_basis;

        let mut ao = MatrixFull::new([num_basis,num_grids],0.0);
        let mut aop =  if mol.xc_data.use_density_gradient() {
            Some(RIFull::new([num_basis,num_grids,3],0.0))
        } else {
            None
        };

        let par_tasks = utilities::balancing(num_grids, rayon::current_num_threads());
        let (sender, receiver) = channel();
        par_tasks.par_iter().for_each_with(sender, |s, range_grids| {


            let loc_num_grids = range_grids.len();

            let mut loc_ao = MatrixFull::new([num_basis, loc_num_grids],0.0);
            let mut loc_aop =  if mol.xc_data.use_density_gradient() {
                RIFull::new([num_basis,loc_num_grids,3],0.0)
            } else {
                RIFull::empty()
            };
            mol.basis4elem.iter().zip(mol.geom.position.iter_columns_full()).for_each(|(elem, geom)| {
                let ind_glb_bas = elem.global_index.0;
                let loc_num_bas = elem.global_index.1;
                let start = ind_glb_bas;
                let end = start + loc_num_bas;
                let mut tmp_geom = [0.0;3];
                tmp_geom.iter_mut().zip(geom.iter()).for_each(|value| {*value.0 = *value.1});
                let tab_den = spheric_gto_value_serial(&self.coordinates[range_grids.clone()], &tmp_geom, elem);

                loc_ao.copy_from_matr(start..end, 0..loc_num_grids, &tab_den, 0..loc_num_bas, 0..loc_num_grids);

                if mol.xc_data.use_density_gradient() {
                    //println!("debug 01");
                    let tab_dev = spheric_gto_1st_value_serial(&self.coordinates[range_grids.clone()], &tmp_geom, elem);
                    //println!("debug 02");
                    for x in 0..3 {
                        let gto_1st_x = &tab_dev[x];
                        loc_aop.copy_from_matr(start..end, 0..loc_num_grids, x, 0, 
                            gto_1st_x, 0..loc_num_bas, 0..loc_num_grids);
                    }
                    //println!("debug 03");
                    //Some(RIFull::new([num_loc_bas,num_grids,3],0.0))
                };
            });
            s.send((loc_ao, loc_aop, range_grids)).unwrap()
        });
        receiver.into_iter().for_each(|(loc_ao, loc_aop, range_grids)| {
            let loc_num_grids = range_grids.len();
            ao.copy_from_matr(0..num_basis, range_grids.clone(), &loc_ao, 0..num_basis, 0..loc_num_grids);
            if let Some(aop) = &mut aop {
                aop.copy_from_ri(0..num_basis, range_grids.clone(),0..3,
                    &loc_aop,0..num_basis, 0..loc_num_grids, 0..3);
            }
        });

        self.ao = Some(ao);
        self.aop = aop;

        utilities::omp_set_num_threads_wrapper(default_omp_num_threads);
    }

    pub fn prepare_tabulated_ao_rayon(&mut self, mol: &Molecule) {
        //In this subroutine, we call the lapack dgemm in a rayon parallel environment.
        //In order to ensure the efficiency, we disable the openmp ability and re-open it in the end of subroutien
        //let default_omp_num_threads = unsafe {utilities::openblas_get_num_threads()};
        let default_omp_num_threads = utilities::omp_get_num_threads_wrapper();
        //println!("debug: default_omp_num_threads: {}", default_omp_num_threads);
        utilities::omp_set_num_threads_wrapper(1);

        let num_grids = self.coordinates.len();

        let mut ao = MatrixFull::new([num_grids,mol.num_basis],0.0);
        let mut aop =  if mol.xc_data.use_density_gradient() {
            Some(RIFull::new([mol.num_basis,num_grids,3],0.0))
        } else {
            None
        };

        let (sender, receiver) = channel();
        mol.basis4elem.par_iter().zip(mol.geom.position.par_iter_columns_full()).for_each_with(sender, |s, (elem,geom)| {
            let ind_glb_bas = elem.global_index.0;
            let num_loc_bas = elem.global_index.1;
            //let mut tab_den = MatrixFull::new([num_grids, num_loc_bas],0.0);

            let mut tmp_geom = [0.0;3];
            tmp_geom.iter_mut().zip(geom.iter()).for_each(|value| {*value.0 = *value.1});
            let tab_den = spheric_gto_value_matrixfull_serial(&self.coordinates, &tmp_geom, elem);

            let tab_dev = if mol.xc_data.use_density_gradient() {
                Some(spheric_gto_1st_value_batch_serial(&self.coordinates, &tmp_geom, elem))
                //Some(RIFull::new([num_loc_bas,num_grids,3],0.0))
            } else {
                None
            };

            s.send((ind_glb_bas,num_loc_bas,tab_den,tab_dev)).unwrap()
        });
        receiver.into_iter().for_each(|(ind_glb_bas,num_loc_bas,tab_den,tab_dev)| {
            let start = ind_glb_bas;
            let end = ind_glb_bas + num_loc_bas;
            ao.iter_columns_mut(start..end).zip(tab_den.iter_columns_full())
            .for_each(|(to,from)| {
                to.iter_mut().zip(from.iter()).for_each(|(to,from)| {*to = *from});
            });
            if let (Some(aop), Some(tab_dev)) = (&mut aop, tab_dev) {
                for x in 0..3 {
                    let gto_1st_x = tab_dev.get(x).unwrap().transpose();

                    aop.copy_from_matr(start..end, 0..num_grids, x, 0, 
                        &gto_1st_x, 0..num_loc_bas, 0..num_grids);

                    //let mut rhop_x = aop.get_reducing_matrix_mut(x).unwrap();
                    //rhop_x.iter_submatrix_mut(start..end,0..num_grids)
                    //.zip(gto_1st_x.data.iter()).for_each(|(to,from)| {*to = *from});
                }
            }
        });

        self.ao = Some(ao.transpose_and_drop());
        self.aop = aop;

        utilities::omp_set_num_threads_wrapper(default_omp_num_threads);

    }

    pub fn prepare_tabulated_ao_old(&mut self, mol: &Molecule) {
        // In this subroutine, we call the lapack dgemm in a rayon parallel environment.
        // In order to ensure the efficiency, we disable the openmp ability and re-open it in the end of subroutien
        // let default_omp_num_threads = unsafe {utilities::openblas_get_num_threads()};
        //let dt_1 = time::Local::now();
        let num_grids = self.coordinates.len();
        // first for density
        //let dt0 = utilities::init_timing();
        //mol.basis4elem.iter().for_each(|elem| {

        //});

        let mut time_records = utilities::TimeRecords::new();
        time_records.new_item("TabAO", "the generation of tabulated AO and its derivatives");
        time_records.count_start("TabAO");

        time_records.new_item("1", "spheric_gto_value_matrixfull");

        let mut tab_den = MatrixFull::new([num_grids,mol.num_basis], 0.0);
        let mut start:usize = 0;
        mol.basis4elem.iter().zip(mol.geom.position.iter_columns_full()).for_each(|(elem,geom)| {
            let mut tmp_geom = [0.0;3];
            tmp_geom.iter_mut().zip(geom.iter()).for_each(|value| {*value.0 = *value.1});
            time_records.count_start("1");
            let tmp_spheric = spheric_gto_value_matrixfull(&self.coordinates, &tmp_geom, elem);
            time_records.count("1");
            let s_len = tmp_spheric.size[1];
            tab_den.iter_columns_mut(start..start+s_len).zip(tmp_spheric.iter_columns_full())
            .for_each(|(to,from)| {
                to.par_iter_mut().zip(from.par_iter()).for_each(|(to,from)| {*to = *from});
            });
            start += s_len;
        });
        self.ao = Some(tab_den.transpose_and_drop());

        // then for density gradient
        if mol.xc_data.use_density_gradient() {

            time_records.new_item("2", "spheric_gto_1st_value_batch");
            time_records.new_item("3", "copy");
            time_records.new_item("4", "transpose");


            let mut tab_dev = RIFull::new([mol.num_basis,num_grids,3],0.0);
            let mut start: usize = 0;
            mol.basis4elem.iter().zip(mol.geom.position.iter_columns_full()).for_each(|(elem,geom)| {
                let mut tmp_geom = [0.0;3];
                tmp_geom.iter_mut().zip(geom.iter()).for_each(|value| {*value.0 = *value.1});
                time_records.count_start("2");
                let gto_1st = spheric_gto_1st_value_batch(&self.coordinates, &tmp_geom, elem);
                time_records.count("2");
                let len = gto_1st[0].size[1];
                for x in 0..3 {
                    time_records.count_start("4");
                    let gto_1st_x = gto_1st.get(x).unwrap().transpose();
                    time_records.count("4");

                    time_records.count_start("3");
                    let mut rhop_x = tab_dev.get_reducing_matrix_mut(x).unwrap();
                    rhop_x.iter_submatrix_mut(start..start+len,0..num_grids)
                    .zip(gto_1st_x.data.iter()).for_each(|(to,from)| {*to = *from});
                    //rhop_x.get2d_slice_mut([start,y], len).unwrap().iter_mut()
                    //    .zip(gto_1st_x.iter()).for_each(|(to,from)| {*to = *from});
                    time_records.count("3");
                }
                start = start + len;
            });

            //for i in 0..100 {
            //    println!("{:16.8},{:16.8},{:16.8}",tab_dev[[0,i,0]],tab_dev[[0,i,1]],tab_dev[[0,i,2]]);
            //}

            self.aop = Some(tab_dev);
            //let dt2 = utilities::timing(&dt0, Some("tabulated aop"));

        }


        time_records.count("TabAO");
        time_records.report_all();

        // to implement kinetic density
    }
    pub fn prepare_tabulated_density_prev(&self, dm: &mut Vec<MatrixFull<f64>>, spin_channel: usize) -> MatrixFull<f64> {
        let mut cur_rho = MatrixFull::new([self.coordinates.len(),spin_channel],0.0);
        if let Some(ao) = &self.ao {
            for i_spin in 0..spin_channel {
                let dm_s = &mut dm[i_spin];
                let num_basis = ao.size[0];
                //println!("debug print rho");
                //cur_rho.iter_j(i_spin).for_each(|f| {println!("debug {}: {}",i, f); i+=1});
                ao.iter_columns_full().zip(cur_rho.iter_column_mut(i_spin))
                .for_each(|(ao_r,cur_rho_spin)| {
                    let ao_rv = ao_r.to_vec();
                    let mut ao_rr = MatrixFull::from_vec([num_basis,1], ao_rv).unwrap();
                    let mut tmp_mat = MatrixFull::new([num_basis,1],0.0);
                    tmp_mat.lapack_dgemm(&mut ao_rr, dm_s, 'T', 'N', 1.0, 0.0);
                    *cur_rho_spin = tmp_mat.data.iter().zip(ao_rr.data.iter()).fold(0.0, |acc,(a,b)| {acc + a*b});
                })
            };
        }
        cur_rho
    }
    pub fn prepare_tabulated_density(&self, dm: &Vec<MatrixFull<f64>>, spin_channel: usize) -> MatrixFull<f64> {
        let default_omp_num_threads = utilities::omp_get_num_threads_wrapper();
        utilities::omp_set_num_threads_wrapper(1);
        let num_grids = self.coordinates.len();
        let mut cur_rho = MatrixFull::new([num_grids,spin_channel],0.0);
        for i_spin in 0..spin_channel {
            if let Some(ao) = &self.ao {
                let dt0 = utilities::init_timing();
                let dm_s = dm.get(i_spin).unwrap();
                let mut wao = MatrixFull::new(ao.size.clone(),0.0);
                _dgemm_full(dm_s,'N',ao,'N',&mut wao, 1.0, 0.0);
                //wao.lapack_dgemm(dm_s, ao, 'N', 'N', 1.0, 0.0);
                //let wao = _degemm_nn_(&dm_s.to_matrixfullslice(), &ao.to_matrixfullslice());
                let dt1 = utilities::timing(&dt0, Some("Evalute weighted ao (wao)"));
                ao.par_iter_columns_full().zip(wao.par_iter_columns_full()).map(|(ao_r,wao_r)| (ao_r,wao_r))
                .zip(cur_rho.par_iter_column_mut(i_spin))
                .for_each(|((ao_r,wao_r),cur_rho_s)| {
                    *cur_rho_s = wao_r.iter().zip(ao_r.iter()).fold(0.0, |acc, (a,b)| {
                        acc + a*b
                    })
                });
                //println!("{:?}", &cur_rho.data[num_grids*i_spin..num_grids*i_spin+100]);
                let dt2 = utilities::timing(&dt1, Some("Contracting ao*wao"));
            };
        };
        utilities::omp_set_num_threads_wrapper(default_omp_num_threads);
        cur_rho
    }

    pub fn prepare_tabulated_density_2(&self, mo: &[MatrixFull<f64>;2], occ: &[Vec<f64>;2], spin_channel: usize) -> (MatrixFull<f64>,RIFull<f64>) {
        let mut cur_rho = MatrixFull::new([self.coordinates.len(),spin_channel],0.0);
        let num_grids = self.coordinates.len();
        let num_basis = mo[0].size.get(0).unwrap();
        let num_state = mo[0].size.get(1).unwrap();
        if let (Some(ao), Some(aop)) = (&self.ao, &self.aop) {
            let mut cur_rhop = RIFull::new([num_grids,3,spin_channel],0.0);
            for i_spin in 0..spin_channel {
                let mo_s = mo.get(i_spin).unwrap();
                // assume that the molecular obitals have been orderd: occupation first, then virtual.
                //let mut occ_s = occ.get(i_spin).unwrap()
                //    .iter().filter(|occ| **occ>0.0).map(|occ| occ.sqrt()).collect_vec();
                //==================================
                // now locate the highest obital that has electron with occupation largger than 1.0e-4
                //let homo_s = occ[i_spin].iter().enumerate().fold(0_usize,|x, (ob, occ)| {if *occ>1.0e-4 {ob} else {x}});
                let homo_s  = occ[i_spin].iter().enumerate()
                    .filter(|(i,occ)| **occ >=1.0e-6)
                    .map(|(i,occ)| i).max();
                let mut occ_s = if let Some(homo_s) = homo_s {
                    occ.get(i_spin).unwrap()[0..homo_s+1].iter().map(|occ| occ.sqrt()).collect::<Vec<f64>>()
                } else {
                    vec![]
                };
                //==================================
                let num_occ = occ_s.len();
                // wmo = weigthed mo ('ij,j->ij'): mo_s(ij), occ_s(j) -> wmo(ij)
                let mut wmo = _einsum_01_rayon(&mo_s.to_matrixfullslice(),&occ_s);

                let mut tmo = MatrixFull::new([num_occ,num_grids],0.0);
                tmo.to_matrixfullslicemut().lapack_dgemm(&wmo.to_matrixfullslice(), &ao.to_matrixfullslice(), 'T', 'N', 1.0, 0.0);
                let rho_s = _einsum_02_rayon(&tmo.to_matrixfullslice(), &tmo.to_matrixfullslice());
                cur_rho.par_iter_column_mut(i_spin).zip(rho_s.par_iter()).for_each(|(to, from)| {*to = *from});
                
                for i in (0..3) {
                    let mut tmop = MatrixFull::new([num_occ,num_grids],0.0);
                    tmop.to_matrixfullslicemut()
                        .lapack_dgemm(&wmo.to_matrixfullslice(), &aop.get_reducing_matrix(i).unwrap(), 'T','N',1.0,0.0);
                    let rhopi_s = _einsum_02_rayon(&tmop.to_matrixfullslice(), &tmo.to_matrixfullslice());
                    cur_rhop.get_reducing_matrix_mut(i_spin).unwrap().par_iter_mut_j(i)
                    .zip(rhopi_s.par_iter()).for_each(|(to, from)| {*to = *from*2.0});
                }
            };
            return (cur_rho, cur_rhop)
        };
        if let Some(ao) = &self.ao {
            for i_spin in 0..spin_channel {
                let mo_s = mo.get(i_spin).unwrap();
                // assume that the molecular obitals have been orderd: occupation first, then virtual.
                let mut occ_s = occ.get(i_spin).unwrap()
                    .iter().filter(|occ| **occ>0.0).map(|occ| occ.sqrt()).collect_vec();
                let num_occu = occ_s.len();
                // wmo = weigthed mo ('ij,j->ij'): mo_s(ij), occ_s(j) -> wmo(ij)
                let mut wmo = _einsum_01_rayon(&mo_s.to_matrixfullslice(),&occ_s);

                let mut tmo = MatrixFull::new([wmo.size[1],ao.size[1]],0.0);
                tmo.to_matrixfullslicemut().lapack_dgemm(&wmo.to_matrixfullslice(), &ao.to_matrixfullslice(), 'T', 'N', 1.0, 0.0);
                let rho_s = _einsum_02_rayon(&tmo.to_matrixfullslice(), &tmo.to_matrixfullslice());
                cur_rho.par_iter_column_mut(i_spin).zip(rho_s.par_iter()).for_each(|(to, from)| {*to = *from});

            };
            let mut cur_rhop = RIFull::empty();
            return (cur_rho, cur_rhop)
        }
        let cur_rhop = RIFull::empty();
        (cur_rho, cur_rhop)
    }

    pub fn prepare_tabulated_density_slots_dm_only(&self, dm:& Vec<MatrixFull<f64>>, spin_channel: usize, range_grids: Range<usize>) -> (MatrixFull<f64>,RIFull<f64>) {
        let num_grids = range_grids.len();
        let num_basis = dm[0].size[0];
        //let mut cur_rho = MatrixFull::new([num_grids,spin_channel],0.0);
        //let mut cur_rhop = RIFull::empty();
        let cur_rho = if let Some(ao) = &self.ao {
            let mut cur_rho = MatrixFull::new([num_grids,spin_channel],0.0);
            //let mut cur_rhop = RIFull::new([num_grids,3,spin_channel],0.0);
            for i_spin in 0..spin_channel {
                let dm_s = &dm[i_spin];
                let mut wao = MatrixFull::new([num_basis,num_grids],0.0);

                _dgemm(
                    dm_s,(0..num_basis,0..num_basis),'N',
                    ao,(0..num_basis,range_grids.clone()),'N',
                    &mut wao, (0..num_basis,0..num_grids),1.0,0.0);

                ao.iter_columns(range_grids.clone()).zip(wao.iter_columns_full()).map(|(ao_r,wao_r)| (ao_r,wao_r))
                .zip(cur_rho.iter_column_mut(i_spin))
                .for_each(|((ao_r,wao_r),cur_rho_s)| {
                    *cur_rho_s = wao_r.iter().zip(ao_r.iter()).fold(0.0, |acc, (a,b)| {
                        acc + a*b
                    })
                });
            }
            cur_rho
        } else {
            MatrixFull::empty()
        };

        let mut cur_rhop = if let (Some(ao), Some(aop)) = (&self.ao, &self.aop) {
            let mut cur_rhop = RIFull::new([num_grids,3,spin_channel],0.0);
            for i_spin in 0..spin_channel {
                let dm_s = &dm[i_spin];
                let mut rhop_s = cur_rhop.get_reducing_matrix_mut(i_spin).unwrap();
                for i in (0..3) {
                    let mut aop_i = aop.get_reducing_matrix(i).unwrap();
                    let mut wao = MatrixFull::new([num_basis,num_grids],0.0);
                    _dgemm(
                        dm_s, (0..num_basis,0..num_basis),'N',
                        &aop_i, (0..num_basis, range_grids.clone()),'N',
                        &mut wao, (0..num_basis, 0..num_grids), 1.0, 0.0);
                    //wao.to_matrixfullslicemut().lapack_dgemm(&dm.to_matrixfullslice(),&aop_i, 'N','N', 1.0, 0.0);
                    ao.iter_columns(range_grids.clone()).zip(wao.iter_columns_full()).map(|(ao_r,wao_r)| (ao_r,wao_r))
                    .zip(rhop_s.iter_mut_j(i))
                    .for_each(|((ao_r,wao_r), cur_rhop_r)| {
                        *cur_rhop_r = 2.0*wao_r.iter().zip(ao_r.iter()).fold(0.0, |acc,(wao,ao)| {acc + wao*ao})
                    });
                }
            }
            cur_rhop
        } else {
            RIFull::empty()
        };
        (cur_rho, cur_rhop)
    }

    //pub fn prepare_tabulated_density_slots_cudarc(&self, mo: &[MatrixFull<f64>;2], occ: &[Vec<f64>;2], spin_channel: usize,range_grids: Range<usize>) {
    //    use cudarc::driver::{CudaDevice,LaunchAsync,LaunchConfig};
    //    let num_grids = range_grids.len();
    //    //let dev: Option<Arc<CudaDevice>>> = CudaDevice::new(0).unwrap_or(None);
    //}

    pub fn prepare_tabulated_density_slots(&self, mo: &[MatrixFull<f64>;2], occ: &[Vec<f64>;2], spin_channel: usize, range_grids: Range<usize>) -> (MatrixFull<f64>,RIFull<f64>) {
        let num_grids = range_grids.len();
        let num_basis = mo[0].size.get(0).unwrap();
        let num_state = mo[0].size.get(1).unwrap();
        let mut cur_rho = MatrixFull::new([num_grids,spin_channel],0.0);
        //let num_grids = self.coordinates.len();
        if let (Some(ao), Some(aop)) = (&self.ao, &self.aop) {
            let mut cur_rhop = RIFull::new([num_grids,3,spin_channel],0.0);
            for i_spin in 0..spin_channel {
                let mo_s = mo.get(i_spin).unwrap();
                //==================================
                // WRONG:: assume that the molecular obitals have been orderd: occupation first, then virtual,
                //         which is wrong for the dSCF calculation with forced occupation.
                //let mut occ_s = occ.get(i_spin).unwrap()
                //    .iter().filter(|occ| **occ>0.0).map(|occ| occ.sqrt()).collect_vec();
                //let homo_s = occ[i_spin].iter().enumerate().fold(0_usize,|x, (ob, occ)| {if *occ>1.0e-4 {ob} else {x}});
                //==================================
                // now locate the highest orbital that has electron with occupation larger than 1.0e-4
                let homo_s  = occ[i_spin].iter().enumerate()
                    .filter(|(i,occ)| **occ >=1.0e-6)
                    .map(|(i,occ)| i).max();
                let mut occ_s = if let Some(homo_s) = homo_s {
                    occ.get(i_spin).unwrap()[0..homo_s+1].iter().map(|occ| occ.sqrt()).collect::<Vec<f64>>()
                } else {
                    vec![]
                };
                //==================================
                let num_occ = occ_s.len();
                let mut wmo = _einsum_01_serial(&mo_s.to_matrixfullslice(),&occ_s);

                let mut tmo = MatrixFull::new([num_occ,num_grids],0.0);
                _dgemm(&wmo, (0..wmo.size[0], 0..wmo.size[1]), 'T',
                       ao, (0..ao.size[0],range_grids.clone()), 'N',
                       &mut tmo, (0..wmo.size[1],0..num_grids),
                       1.0,0.0
                );
                //tmo.to_matrixfullslicemut().lapack_dgemm(&wmo.to_matrixfullslice(), &ao.to_matrixfullslice(), 'T', 'N', 1.0, 0.0);
                let rho_s = _einsum_02_serial(&tmo.to_matrixfullslice(), &tmo.to_matrixfullslice());
                cur_rho.iter_column_mut(i_spin).zip(rho_s.iter()).for_each(|(to, from)| {*to = *from});
                
                for i in (0..3) {
                    let mut tmop = MatrixFull::new([num_occ,num_grids],0.0);
                    let aop_i = aop.get_reducing_matrix(i).unwrap();
                    _dgemm(&wmo, (0..wmo.size[0], 0..wmo.size[1]), 'T',
                           &aop_i, (0..aop_i.size[0],range_grids.clone()), 'N',
                           &mut tmop, (0..wmo.size[1],0..num_grids),
                           1.0,0.0
                    );
                    //tmop.to_matrixfullslicemut()
                    //    .lapack_dgemm(&wmo.to_matrixfullslice(), &aop.get_reducing_matrix(i).unwrap(), 'T','N',1.0,0.0);
                    let rhopi_s = _einsum_02_serial(&tmop.to_matrixfullslice(), &tmo.to_matrixfullslice());
                    cur_rhop.get_reducing_matrix_mut(i_spin).unwrap().iter_mut_j(i)
                    .zip(rhopi_s.iter()).for_each(|(to, from)| {*to = *from*2.0});
                }
            };
            return (cur_rho, cur_rhop)
        };
        if let Some(ao) = &self.ao {
            for i_spin in 0..spin_channel {
                let mo_s = mo.get(i_spin).unwrap();
                // assume that the molecular obitals have been orderd: occupation first, then virtual.
                let mut occ_s = occ.get(i_spin).unwrap()
                    .iter().filter(|occ| **occ>0.0).map(|occ| occ.sqrt()).collect_vec();
                let num_occu = occ_s.len();
                // wmo = weigthed mo ('ij,j->ij'): mo_s(ij), occ_s(j) -> wmo(ij)
                let mut wmo = _einsum_01_serial(&mo_s.to_matrixfullslice(),&occ_s);

                let mut tmo = MatrixFull::new([wmo.size[1],num_grids],0.0);
                _dgemm(&wmo, (0..wmo.size[0], 0..wmo.size[1]), 'T',
                       ao, (0..ao.size[0],range_grids.clone()), 'N',
                       &mut tmo, (0..wmo.size[1],0..num_grids),
                       1.0,0.0
                );
                //tmo.to_matrixfullslicemut().lapack_dgemm(&wmo.to_matrixfullslice(), &ao.to_matrixfullslice(), 'T', 'N', 1.0, 0.0);
                let rho_s = _einsum_02_serial(&tmo.to_matrixfullslice(), &tmo.to_matrixfullslice());
                cur_rho.iter_column_mut(i_spin).zip(rho_s.iter()).for_each(|(to, from)| {*to = *from});

            };
            let mut cur_rhop = RIFull::empty();
            return (cur_rho, cur_rhop)
        }
        let cur_rhop = RIFull::empty();
        (cur_rho, cur_rhop)
    }

    pub fn prepare_tabulated_rhop(&self, dm: &mut Vec<MatrixFull<f64>>, spin_channel: usize) -> RIFull<f64> {
        let num_basis = dm.get(0).unwrap().size.get(0).unwrap().clone();
        let num_grids = self.coordinates.len();
        let mut cur_rhop = RIFull::new([num_grids,3,spin_channel],0.0);
        for i_spin in 0..spin_channel {
            let dm = &mut dm[i_spin];
            let mut rhop_s = cur_rhop.get_reducing_matrix_mut(i_spin).unwrap();
            if let (Some(ao), Some(aop)) = (&self.ao, &self.aop) {
                // for the ao gradient along the direction i = (x,y,z)
                for i in (0..3) {
                    let mut aop_i = aop.get_reducing_matrix(i).unwrap();
                    let mut wao = MatrixFull::new([num_basis,num_grids],0.0);
                    wao.to_matrixfullslicemut().lapack_dgemm(&dm.to_matrixfullslice(),&aop_i, 'N','N', 1.0, 0.0);

                    // ====== native dgemm coded by rust
                    //let mut aop_i = aop.get_reducing_matrix(i).unwrap();
                    //let wao=_degemm_nn_(&dm.to_matrixfullslice(), &aop_i);
                    //==================================
                    ao.par_iter_columns_full().zip(wao.par_iter_columns_full()).map(|(ao_r,wao_r)| (ao_r,wao_r))
                    .zip(rhop_s.par_iter_mut_j(i))
                    .for_each(|((ao_r,wao_r), cur_rhop_r)| {
                        *cur_rhop_r = 2.0*wao_r.iter().zip(ao_r.iter()).fold(0.0, |acc,(wao,ao)| {acc + wao*ao})
                    });
                    //ao.iter_columns_full().zip(wao.iter_columns_full()).map(|(ao_r,wao_r)| (ao_r,wao_r))
                    //.zip(rhop_s.iter_mut_j(i))
                    //.for_each(|((ao_r,wao_r), cur_rhop_r)| {
                    //    *cur_rhop_r = 2.0*wao_r.iter().zip(ao_r.iter()).fold(0.0, |acc,(wao,ao)| {acc + wao*ao})
                    //});
                };
            }
        };
        cur_rhop
    }
    pub fn evaluate_density(&self, dm: &mut Vec<MatrixFull<f64>>) -> [f64;2] {
        let mut total_density = [0.0f64;2];
        if let Some(ao) = &self.ao {
            ao.iter_columns(0..self.weights.len())
                .zip(self.weights.iter()).for_each(|(ao_r, w)| {
                let mut density_r_sum = [0.0;2];
                let ao_rv = ao_r.to_vec();
                let tmp_len = ao_rv.len();
                let mut ao_rr = MatrixFull::from_vec([tmp_len,1], ao_rv).unwrap();
                dm.iter_mut().zip(density_r_sum.iter_mut()).for_each(|(dm_s, density_r_sum)| {
                    let mut tmp_mat = MatrixFull::new([tmp_len,1],0.0);
                    tmp_mat.lapack_dgemm(&mut ao_rr, dm_s, 'T', 'N', 1.0, 0.0);
                    *density_r_sum += tmp_mat.data.iter().zip(ao_rr.data.iter()).fold(0.0, |acc,(a,b)| {acc + a*b});
                });
                total_density.iter_mut().zip(density_r_sum.iter()).for_each(|(to,from)| *to += from*w);
            });
        }
        total_density
    }
    //pub fn evalute_xc(&self, dm: &mut [MatrixFull<f64>;2], xc: XcFuncType) -> MatrixFull<>{

    //}
}


pub fn numerical_density(grid: &Grids, mol: &Molecule, dm: &mut [MatrixFull<f64>;2]) -> [f64;2] {
    let mut total_density = [0.0f64;2];
    //let mut count:usize = 0;
    grid.coordinates.iter().zip(grid.weights.iter()).for_each(|(r,w)| {
        let mut density_r_sum = [0.0;2];
        let mut density_r:Vec<f64> = vec![];
        mol.basis4elem.iter().zip(mol.geom.position.iter_columns_full()).for_each(|(elem,geom)| {
            let mut tmp_geom = [0.0;3];
            tmp_geom.iter_mut().zip(geom.iter()).for_each(|value| {*value.0 = *value.1});
            density_r.extend(gto_value(r, &tmp_geom, elem, &mol.ctrl.basis_type));
        });
        let mut density_rr = MatrixFull::from_vec([mol.num_basis,1],density_r).unwrap();
        dm.iter_mut().zip(density_r_sum.iter_mut()).for_each(|(dm_s, density_r_sum)| {
            let mut tmp_mat = MatrixFull::new([mol.num_basis,1],0.0);
            tmp_mat.lapack_dgemm(&mut density_rr, dm_s, 'T', 'N', 1.0, 0.0);
            *density_r_sum += tmp_mat.data.iter().zip(density_rr.data.iter()).fold(0.0, |acc,(a,b)| {acc + a*b});
        });
        total_density.iter_mut().zip(density_r_sum.iter()).for_each(|(to,from)| *to += from*w);
    });
    
    total_density
}

pub fn par_numerical_density(grid: &Grids, mol: &Molecule, dm: &mut [MatrixFull<f64>;2]) -> [f64;2] {
    let mut total_density = [0.0f64;2];
    //let mut count:usize = 0;
    // In this subroutine, we call the lapack dgemm in a rayon parallel environment.
    // In order to ensure the efficiency, we disable the openmp ability and re-open it in the end of subroutien
    let default_omp_num_threads = utilities::omp_get_num_threads_wrapper();
    //println!("debug: default omp_num_threads: {}", default_omp_num_threads);
    utilities::omp_set_num_threads_wrapper(1);

    let local_basis4elem = mol.basis4elem.clone();
    let local_position = mol.geom.position.clone();
    let num_basis = mol.num_basis;
    let basis_type = mol.ctrl.basis_type.clone();
    let (sender,receiver) = channel();
    grid.coordinates.par_iter().zip(grid.weights.par_iter()).for_each_with(sender, |s,(r,w)| {
        //let r = &grid.coordinates[0];
        //let w = &grid.weights[0];
        let mut local_total_density = [0.0f64;2];
        let mut density_r_sum = [0.0;2];
        let mut density_r:Vec<f64> = vec![];
        let mut local_dm = dm.clone();
        local_basis4elem.iter().zip(local_position.iter_columns_full()).for_each(|(elem,geom)| {
            let mut tmp_geom = [0.0;3];
            tmp_geom.iter_mut().zip(geom.iter()).for_each(|value| {*value.0 = *value.1});
            density_r.extend(gto_value(r, &tmp_geom, elem, &basis_type));
        });
        //if count<=10 {println!("{:?}", density_r)};
        //println!{"debug 1"};
        let mut density_rr = MatrixFull::from_vec([num_basis,1],density_r).unwrap();
        //println!{"debug 2"};
        local_dm.iter_mut().zip(density_r_sum.iter_mut()).for_each(|(dm_s, density_r_sum)| {
            let mut tmp_mat = MatrixFull::new([num_basis,1],0.0);
            tmp_mat.lapack_dgemm(&mut density_rr, dm_s, 'T', 'N', 1.0, 0.0);
            *density_r_sum += tmp_mat.data.iter().zip(density_rr.data.iter()).fold(0.0, |acc,(a,b)| {acc + a*b});
        });
        //println!{"debug 3"};
        //if count<=10 {println!("{:?},{},{}", r,w,density_r_sum)};
        //count += 1;
        local_total_density.iter_mut().zip(density_r_sum.iter()).for_each(|(to,from)| *to += from*w);
        s.send(local_total_density).unwrap();
    });

    receiver.iter().for_each(|value| {
        total_density.iter_mut().zip(value.iter()).for_each(|(to,from)| *to += from);

    });

    // reuse the default omp_num_threads setting
    utilities::omp_set_num_threads_wrapper(default_omp_num_threads);
    
    total_density
}



//#[test]
//fn test_numgrid_angular() {
//    let (coordinates, weights) = gen_grids::angular_grid(50);
//    println!("{:?}",coordinates);
//    println!("{:?}",weights);
//}
//#[test]
//fn test_numgrid_radii() {
//    let (coordinates, weights) = gen_grids::radial_grid_lmg_bse("sto-3g",1.0e-12,8);
//    println!("{:?}",coordinates);
//    println!("{:?}",weights);
//}
#[test]
fn debug_num_density_for_atom() {
    let angular = 2;
    let num_basis = 2*angular+1;
    let mut dm = [
        //MatrixFull::from_vec([1,1], vec![2.0]).unwrap(),
        //MatrixFull::from_vec([num_basis,num_basis], vec![
        //    1.0,0.0,0.0,
        //    0.0,1.0,0.0,
        //    0.0,0.0,0.0]).unwrap(), 
        MatrixFull::from_vec([5,5], vec![
            2.0,0.0,0.0,0.0,0.0,
            0.0,0.0,0.0,0.0,0.0,
            0.0,0.0,0.0,0.0,0.0,
            0.0,0.0,0.0,0.0,0.0,
            0.0,0.0,0.0,0.0,0.0]).unwrap(), 
        MatrixFull::empty()];
    let mut alpha_min_h: HashMap<usize, f64> = HashMap::new();
    alpha_min_h.insert(angular,0.122);
    let mut alpha_max_h: f64 = 0.122;
    let mut basis4elem = vec![Basis4Elem {
        electron_shells: vec![
            BasCell {
                function_type: None,
                region: None,
                angular_momentum: vec![angular as i32],
                exponents: vec![0.122],
                coefficients: vec![vec![1.0/cint_norm_factor(angular as i32, 0.122)]],
                native_coefficients: vec![vec![1.0/cint_norm_factor(angular as i32, 0.122)]]
            }
        ],
        references: None,
        ecp_electrons: None,
        ecp_potentials: None,
        global_index: (0,0)
    }];
    let mut center_coordinates_bohr = vec![(0.0,0.0,0.0)];
    let mut proton_charges = vec![1];
    let grids = Grids::build_nonstd(
        center_coordinates_bohr.clone(), 
        proton_charges.clone(), 
        vec![alpha_min_h], 
        vec![alpha_max_h]);
    let mut total_density = 0.0;
    let mut count:usize =0;
    grids.coordinates.iter().zip(grids.weights.iter()).for_each(|(r,w)| {
        let mut density_r_sum = 0.0;
        let mut density_r:Vec<f64> = vec![];
        basis4elem.iter().zip(center_coordinates_bohr.iter()).for_each(|(elem,geom_nonstd)| {
            let geom = [geom_nonstd.0,geom_nonstd.1,geom_nonstd.2];
            let mut tmp_geom = [0.0;3];
            tmp_geom.iter_mut().zip(geom.iter()).for_each(|value| {*value.0 = *value.1});
            //density_r.extend(gto_value(r, &tmp_geom, elem, &mol.ctrl.basis_type));
            //let tmp_vec = gto_value(r, &tmp_geom, elem, &"spheric".to_string());
            let tmp_vec = gto_value(r, &tmp_geom, elem, &"spheric".to_string());
            //println!("debug 0: len {}", &tmp_vec.len());
            density_r.extend(tmp_vec);
            //if count<=10 {println!("{:?},{:?},{:?}", density_r,elem.electron_shells[0].exponents, elem.electron_shells[0].coefficients[0])};
        });
        //println!{"debug 1"};
        let mut density_rr = MatrixFull::from_vec([num_basis,1],density_r).unwrap();
        //println!{"debug 2"};
        dm.iter_mut().for_each(|dm_s| {
            let mut tmp_mat = MatrixFull::new([num_basis,1],0.0);
            tmp_mat.lapack_dgemm(&mut density_rr, dm_s, 'T', 'N', 1.0, 0.0);
            if count<=10 {println!("count: {},{:?}",count, &tmp_mat.data)};
            density_r_sum += tmp_mat.data.iter().zip(density_rr.data.iter()).fold(0.0, |acc,(a,b)| {acc + a*b});
        });
        if count<=10 {println!("{:?},{},{}", r,w,density_r_sum)};
        count += 1;
        //println!{"debug 3"};
        total_density += density_r_sum * w;
    });

    println!("Total density: {}", total_density);
    
}

#[test]
fn test_libxc() {
    let mut rho:Vec<f64> = vec![0.1,0.2,0.3,0.4,0.5,0.6,0.8];
    let sigma:Vec<f64> = vec![0.2,0.3,0.4,0.5,0.6,0.7];
    //&rho.par_iter().for_each(|c| {println!("{:16.8}",c)});
    //let mut exc:Vec<c_double> = vec![0.0,0.0,0.0,0.0,0.0];
    //let mut vrho:Vec<c_double> = vec![0.0,0.0,0.0,0.0,0.0];
    //let func_id: usize = ffi_xc::XC_GGA_X_XPBE as usize;
    let spin_channel: usize = 1;

    let mut my_xc = DFA4REST::parse_scf("lda_x_slater", spin_channel); 
    //let mut my_xc = XcFuncType::xc_func_init_fdqc(&"pw-lda",spin_channel); 



    let mut exc = MatrixFull::new([rho.len()/spin_channel,1],0.0);
    let mut vrho = MatrixFull::new([rho.len()/spin_channel,spin_channel],0.0);
    my_xc.dfa_compnt_scf.iter().zip(my_xc.dfa_paramr_scf.iter()).for_each(|(xc_func, xc_para)| {
        //let mut new_vec = vec![-0.34280861230056237, -0.43191178672272906, -0.494415573788165, -0.5441747517896713, -0.586194481347579, -0.622924588811561, -0.6856172246011247];
        //let tmp_c = (new_vec.as_mut_ptr(), new_vec.len(),new_vec.capacity());
        //let new_vec = unsafe{Vec::from_raw_parts(tmp_c.0, tmp_c.1, tmp_c.2)};
        //new_vec.par_iter().for_each(|c| {println!("{:16.8e}",c)});
        let xc_func = my_xc.init_libxc(xc_func);

        let (tmp_exc, tmp_vrho) = xc_func.lda_exc_vxc(&rho);
        //let tmp_exc_2 = tmp_exc.clone();
        //println!("WARNNING:: unsolved rayon par_iter problem. It should be relevant to be the fact that tmp_exc is prepared by libxc via ffi");
        //println!("tmp_vec_2 copied from tmp_exc: {:?},{},{}", &tmp_exc_2, tmp_exc_2.len(),tmp_exc_2.capacity());
        //println!("tmp_vec");
        //&tmp_exc.iter().for_each(|c| {
        //    println!("{:16.8e}",c);
        //});
        //println!("tmp_vec_2");
        //&tmp_exc_2.par_iter().for_each(|c| {
        //    println!("{:16.8e}",c);
        //});
        //println!("tmp_vec_2");
        //&tmp_exc.iter().for_each(|c| {
        //    println!("{:16.8e}",c);
        //});
        let mut tmp_exc = MatrixFull::from_vec([rho.len()/spin_channel,1],tmp_exc).unwrap();
        let mut tmp_vrho = MatrixFull::from_vec([rho.len()/spin_channel,spin_channel],tmp_vrho).unwrap();
        //println!("{:?}", &tmp_exc.data);
        //println!("{:?}", &tmp_vrho.data);
        exc.par_self_scaled_add(&tmp_exc,*xc_para);
        vrho.par_self_scaled_add(&tmp_vrho,*xc_para);
        //exc.data.par_iter_mut().zip(tmp_exc.data.par_iter()).for_each(|(c,p)| {
        //    println!("{:16.8e},{:16.8e}",c,p);
        //});
        //println!("{:?}", &exc.data);
        //println!("{:?}", &vrho.data);
    });
    println!("{:?}", exc.data);
    println!("{:?}", vrho.data);
}
#[test]
fn test_zip() {
    let dd = vec![1,2,3,4,5,6];
    let ff = vec![1,3,5];
    let gg = vec![2,4,6];
    izip!(dd.chunks_exact(2),ff.iter(),gg.iter()).for_each(|(dd,ff,gg)| {
        println!("dd {:?}, ff {}, gg {}",dd,ff,gg)
    });
}
#[test]
fn read_grid() {
    let mut grids_file = std::fs::File::open("/home/igor/Documents/Package-Pool/Rust/rest/grids").unwrap();
    let mut content = String::new();
    grids_file.read_to_string(&mut content);
    //println!("{}",&content);
    let re1 = Regex::new(r"(?x)\s*
        (?P<x>[\+-]?\d+.\d+[eE][\+-]?\d+)\s*,# the 'x' position
        \s+
        (?P<y>[\+-]?\d+.\d+[eE][\+-]?\d+)\s*,# the 'y' position
        \s+
        (?P<z>[\+-]?\d+.\d+[eE][\+-]?\d+)\s*,# the 'z' position
        \s+
        (?P<w>[\+-]?\d+.\d+[eE][\+-]?\d+)\s*# the 'w' weight
        \s*\n").unwrap();
    //if let Some(cap)  = re1.captures(&content) {
    //    println!("{:?}", &cap)
    //}
    for cap in re1.captures_iter(&content) {
        let x:f64 = cap[1].parse().unwrap();
        let y:f64 = cap[2].parse().unwrap();
        let z:f64 = cap[3].parse().unwrap();
        let w:f64 = cap[4].parse().unwrap();
        println!("{:16.8} {:16.8} {:16.8} {:16.8}", x,y,z,w);
    }
}
#[test]
fn debug_transpose() {
    let len_a = 111_usize;
    let len_b = 40000_usize;
    let orig_a:Vec<f64> = (0..len_a*len_b).map(|i| {i as f64}).collect();
    let a_mat = MatrixFull::from_vec([len_a,len_b],orig_a).unwrap();
    let dt0 = utilities::init_timing();
    let b_mat = a_mat.transpose_and_drop();
    //b_mat.formated_output(10, "full");
    let dt1 = utilities::timing(&dt0, Some("old transpose"));
    let orig_a:Vec<f64> = (0..len_a*len_b).map(|i| {i as f64}).collect();
    let a_mat = MatrixFull::from_vec([len_a,len_b],orig_a).unwrap();
    let dt0 = utilities::init_timing();
    let c_mat = a_mat.transpose_and_drop();
    //b_mat.formated_output(10, "full");
    let dt1 = utilities::timing(&dt0, Some("new transpose"));
    b_mat.data.iter().zip(c_mat.data.iter()).for_each(|(b,c)| {
        assert!(*b==*c);
    });
}



#[test]
fn test_balancing() {
    let dd = balancing(550, 23);
    println!("{:?}",dd);
}