#![warn(unused_imports)]
pub mod libxc_helper;
pub mod gen_grids;
pub mod deep_learning;
pub mod libxc_itrf;
pub mod xc_deriv;
pub mod num_int;
pub mod nlc;
pub mod parse_xc;
pub mod response;
pub mod numint_matmul;
pub mod xceff;

#[cfg(feature = "mpi")]
use mpi::collective::SystemOperation;
// use mpi::ffi::MPI_T_SCOPE_GROUP_EQ;
use rest_tensors::{MatrixFull, RIFull, MatrixFullSlice};
use rest_tensors::matrix_blas_lapack::{_einsum_01_serial, _einsum_02_serial, _einsum_01_rayon, _einsum_02_rayon};
use itertools::{Itertools, izip};
use tensors::{BasicMatrix, MathMatrix, ParMathMatrix};
// use tensors::external_libs::{general_dgemm_f, matr_copy};
use tensors::matrix_blas_lapack::{_dgemm, _dgemm_full, contract_vxc_0_serial};
use rest_tensors::matrix_blas_lapack::{omp_get_num_threads_wrapper, omp_set_num_threads_wrapper};
//use numgrid::{self, radial_grid_lmg_bse};
// use self::gen_grids::radial_grid_lmg_bse;
use rayon::iter::{IntoParallelRefIterator, IndexedParallelIterator, ParallelIterator, IntoParallelRefMutIterator};
use regex::Regex;
use crate::basis_io::{gto_1st_value_batch_serial, gto_1st_value_serial, gto_value, gto_value_matrixfull_serial, gto_value_serial, spheric_gto_1st_value_batch, 
    spheric_gto_value_matrixfull};
use crate::molecule_io::Molecule;
use crate::geom_io::get_mass_charge;
use crate::mpi_io::{MPIData, MPIOperator};
#[cfg(feature = "mpi")]
use crate::mpi_io::{mpi_broadcast, mpi_reduce};
use crate::utilities::{self, balancing};
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::ops::Range;
use std::sync::mpsc::channel;
use serde::{Deserialize, Serialize};

use libxc::prelude::*;
use crate::dft::libxc_helper::{xc_code_fdqc, xc_func_init, lda_exc_vxc, gga_exc_vxc, mgga_exc_vxc, lda_exc, gga_exc, mgga_exc};


#[derive(Clone,Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DFTType {
    Standard,
    NonStandard,
    DeepLearning
}

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
    // (omega, alpha, beta) in libxc convention; note beta is usually not used in computation.
    pub dfa_rsh_scf: Option<(f64, f64, f64)>,
    pub dfa_family_pos: Option<DFAFamily>,
    pub dfa_compnt_pos: Option<Vec<usize>>,
    pub dfa_paramr_pos: Option<Vec<f64>>,
    pub dfa_hybrid_pos: Option<f64>,
    pub dfa_paramr_adv: Option<Vec<f64>>,
    pub nlc_vv10: Option<(f64, f64)>,
}

impl DFAFamily {
    pub fn to_libxc_family(&self) -> LibXCFamily {
        match self {
            DFAFamily::LDA => LibXCFamily::LDA,
            DFAFamily::GGA => LibXCFamily::GGA,
            DFAFamily::MGGA => LibXCFamily::MGGA,
            DFAFamily::HybridGGA => LibXCFamily::HybGGA,
            DFAFamily::HybridMGGA => LibXCFamily::HybMGGA,
            DFAFamily::PT2 => LibXCFamily::HybGGA,
            DFAFamily::SBGE2 => LibXCFamily::HybGGA,
            DFAFamily::SCSRPA => LibXCFamily::HybGGA,
            DFAFamily::RPA => LibXCFamily::GGA,
            DFAFamily::Unknown => panic!("Unknown DFA family cannot be converted to a specific libxc family"),
        }
    }
    pub fn from_libxc_family(family: &LibXCFamily) -> DFAFamily {
        match family {
            LibXCFamily::LDA => DFAFamily::LDA,
            LibXCFamily::GGA => DFAFamily::GGA,
            LibXCFamily::MGGA => DFAFamily::MGGA,
            LibXCFamily::HybGGA => DFAFamily::HybridGGA,
            LibXCFamily::HybMGGA => DFAFamily::HybridMGGA,
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
        let (major, minor, micro) = libxc::util::libxc_version();
        println!("Libxc version used in REST: {}.{}.{}", major, minor, micro);
    }


    pub fn new_xc(spin_channel: usize, print_level: usize) -> DFA4REST {
        DFA4REST { 
            spin_channel, 
            dfa_compnt_scf: vec![], 
            dfa_paramr_scf: vec![], 
            dfa_hybrid_scf: 0.0,
            dfa_rsh_scf: None,
            dfa_family_pos: None, 
            dfa_compnt_pos: None, 
            dfa_paramr_pos: None, 
            dfa_hybrid_pos: None, 
            dfa_paramr_adv: None,
            nlc_vv10: None,
        }
    }
    
    pub fn summary(&self, print_level: usize) {
        if print_level > 0 {
            println!("==== DFA Summary (legacy) ====");
            println!("Spin channel: {}", self.spin_channel);
            println!("SCF DFA components: {:?}", self.dfa_compnt_scf);
            println!("SCF DFA parameters: {:?}", self.dfa_paramr_scf);
            println!("SCF DFA hybrid coeff: {:16.8}", self.dfa_hybrid_scf);
            if self.is_rsh() {
                println!("Range-Separated Hybrid parameters:");
                println!("  omega: {:16.8}", self.omega().unwrap());
                println!("  alpha (LR-HF coeff): {:16.8}", self.rsh_alpha().unwrap());
            }
            if let Some(dfatype) = &self.dfa_family_pos {
                println!("Post-SCF DFA family: {}", dfatype.to_name());
                if let Some(dfacomp) = &self.dfa_compnt_pos {
                    println!("Post-SCF DFA components: {:?}", dfacomp);
                }
                if let Some(dfaparam) = &self.dfa_paramr_pos {
                    println!("Post-SCF DFA parameters: {:?}", dfaparam);
                }
                if let Some(dfahybrid) = &self.dfa_hybrid_pos {
                    println!("Post-SCF DFA hybrid coeff: {:16.8}", dfahybrid);
                }
            } else {
                println!("Post-SCF DFA: None");
            }
            if let Some(dfaparam_adv) = &self.dfa_paramr_adv {
                println!("Advanced DFA parameters: {:?}", dfaparam_adv);
            }
        }
        if print_level > 1 {
            self.describe();
        }
        println!("==== End of Summary ====");
    }

    pub fn describe(&self) {
        println!("==== detailed info of the scf functional ====");
        &self.dfa_compnt_scf.iter().for_each(|xc_func| {
            println!("{}", self.init_libxc(xc_func).describe())
        });
        if let (Some(dfatype),Some(dfacomp)) = 
            (&self.dfa_family_pos, &self.dfa_compnt_pos) {
            //match dfatype {
            //    DFAFamily::PT2 => println!("XYG3-type functional '{}' is employed", &name),
            //    DFAFamily::RPA => println!("RPA-type functional '{}' is employed", &name),
            //    _ => println!("Standard DFA '{}' is employed", &name),
            //}
            println!("==== detailed info of the post-scf functional ====");
            dfacomp.into_iter().for_each(|xc_func| {
                println!("{}", self.init_libxc(xc_func).describe())
            })
        };
    }

    pub fn new_nonstandard(
        spin_channel: usize, 
        print_level:  usize, 
        xc_namelist:  &Option<Vec<String>>, 
        xc_paramlist: &Option<Vec<f64>>, 
        dfa_hybrid_scf: &Option<f64>,
    ) -> DFA4REST {
        let mut dfa = if let (Some(codelist), Some(paramlist), Some(dfa_hybrid_scf)) = (&xc_namelist, &xc_paramlist, &dfa_hybrid_scf) {
            DFA4REST::parse_scf_nonstd(codelist, paramlist, dfa_hybrid_scf, spin_channel)
        } else {
            panic!("xc_namelist, xc_paramlist and dfa_hybrid_scf should be provided for a non-standard setting of DFA")
        };
        dfa
    }

    pub fn new_deep_learning(spin_channel: usize, print_level: usize, xc_model:&Option<String>) -> DFA4REST {
        let mut dfa = if let Some(xc_model) = xc_model {
            DFA4REST::parse_scf_dldft(xc_model, spin_channel)
        } else {
            panic!("xc_model should be provided for deep-learning DFA model")
        };
        dfa
    }

    pub fn update_pt2_params(&mut self, os_factor:Option<f64>, ss_factor:Option<f64>) {
        match self.dfa_family_pos {
            Some(DFAFamily::PT2) => {
                let mut params_adv = self.dfa_paramr_adv.take().unwrap_or(vec![1.0, 1.0]);
                if let Some(osf) = os_factor {
                    params_adv[0] = osf;
                }
                if let Some(ssf) = ss_factor {
                    params_adv[1] = ssf;
                }
                self.dfa_paramr_adv = Some(params_adv);
            },
            _ => { }
        }
    }

    pub fn parse_scf_dldft(xc_model:&String, spin_channel: usize) -> DFA4REST {

        let mut codelist: Vec<&str> = vec![];
        let mut paralist: Vec<f64> = vec![];
        let mut dfa_hybrid_scf: f64 = 0.0;
        if xc_model.eq("dl_dfa_scf") {
            // 毕升的机器学习杂化泛函，fake成b3lyp，但是初始为BLYP
            codelist = vec!["lda_x_slater", "gga_x_b88", "lda_c_vwn_rpa", "gga_c_lyp"];
            paralist = vec![0.00, 1.00, 0.00, 1.00];
            dfa_hybrid_scf = 0.00001;
        };

        // Parse the xc functionals
        let dfa_compnt_scf = codelist.iter().map(|xc| {
            let xc_code = DFA4REST::libxc_code_fdqc(xc);
            xc_code.iter().filter(|x| **x!=0).map(|x| *x).collect::<Vec<usize>>()
        }).flatten().collect::<Vec<usize>>();
        // Parse the xc parameters
        let dfa_paramr_scf = codelist.iter().zip(paralist.iter()).map(|(xc, param)| {
            let xc_code = DFA4REST::libxc_code_fdqc(xc);
            xc_code.iter().filter(|x| **x!=0).map(|x| *param).collect::<Vec<f64>>()
        }).flatten().collect::<Vec<f64>>();
        let dfa_rsh_scf = DFA4REST::get_rsh_libxc(&dfa_compnt_scf, spin_channel);

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
            dfa_rsh_scf,
            nlc_vv10: None,
        }
    }

    pub fn new(name: &str, spin_channel: usize, print_level: usize) -> DFA4REST {
        
        let tmp_name = name.to_lowercase();
        let post_dfa = DFA4REST::parse_postscf(&tmp_name, spin_channel);
        match post_dfa {
            Some(dfa) => {
                dfa
            },
            None => {
                let dfa = DFA4REST::parse_scf(&tmp_name, spin_channel);
                dfa
            },
        }
    }

    pub fn libxc_code_fdqc(name: &str) -> [usize;3] {
        xc_code_fdqc(name)
    }

    pub fn xc_func_init_fdqc(name: &str, spin_channel: usize) -> Vec<usize> {
        let xc_code = DFA4REST::libxc_code_fdqc(name);
        xc_code.iter().filter(|x| **x!=0).map(|x| *x).collect::<Vec<usize>>()
    }

    pub fn init_libxc(&self, xc_code: &usize) -> LibXCFunctional {
        xc_func_init(*xc_code, self.spin_channel)
    }

    pub fn init_libxc_and_set_param(&self, xc_code: &usize) -> LibXCFunctional {
        let mut func = self.init_libxc(xc_code);
        if let Some(omega) = self.omega() {
            if func.ext_param_names().contains(&"_omega".to_string()) {
                func.set_ext_param_by_name("_omega", omega);
            }
        }
        func
    }

    pub fn get_hybrid_libxc(dfa_compnt_scf: &Vec<usize>,spin_channel:usize) -> f64 {
        let mut hybrid_coeff = None;
        for xc_func in dfa_compnt_scf {
            let func = xc_func_init(*xc_func, spin_channel);
            let hyb_exx_coeff = if let Some((_omega, alpha, beta)) = func.cam_coef() {
                alpha + beta // for RSH, return alpha + beta (matches pyscf)
            } else {
                // for non-RSH, return hybrid coefficient; if not available, return 0.0
                func.hyb_exx_coef().unwrap_or(0.0)
            };
            if hyb_exx_coeff.abs() > 1e-10 {
                if hybrid_coeff.is_some() {
                    panic!("Multiple hybrid functionals are specified in the DFA components for SCF. Currently this is not supported.");
                }
                hybrid_coeff = Some(hyb_exx_coeff);
            }
        }
        hybrid_coeff.unwrap_or(0.0)
    }

    pub fn get_rsh_libxc(dfa_compnt_scf: &Vec<usize>, spin_channel: usize) -> Option<(f64, f64, f64)> {
        let mut result = None;
        for xc_func in dfa_compnt_scf {
            let func = xc_func_init(*xc_func, spin_channel);
            if func.is_hyb_cam() {
                let (omega, alpha, beta) = func.cam_coef().unwrap_or((0.0, 0.0, 0.0));
                if alpha.abs() < 1e-10 && beta.abs() < 1e-10 {
                    continue;
                }
                if result.is_some() {
                    panic!("Multiple RSH functionals are specified in the DFA components for SCF. Currently this is not supported.");
                }
                result = Some((omega, alpha, alpha + beta));
            }
        }
        result
    }

    pub fn is_rsh(&self) -> bool {
        self.omega().is_some()
    }

    pub fn omega(&self) -> Option<f64> {
        self.rsh_params().map(|(omega, _, _)| omega)
    }

    pub fn rsh_alpha(&self) -> Option<f64> {
        self.rsh_params().map(|(_, alpha, _)| alpha)
    }

    pub fn rsh_params(&self) -> Option<(f64, f64, f64)> {
        self.dfa_rsh_scf
    }

    pub fn parse_scf(name: &str, spin_channel: usize) -> DFA4REST {
        let tmp_name = name.to_lowercase();
        let legacy_nlc_ids: Option<Vec<usize>> = match tmp_name.as_str() {
            "wb97x-v" => Some(vec![466]),
            "wb97m-v" => Some(vec![531]),
            _ => None,
        };
        let dfa_compnt_scf = match &legacy_nlc_ids {
            Some(ids) => ids.clone(),
            None => DFA4REST::xc_func_init_fdqc(&tmp_name, spin_channel),
        };
        let mut dfa_hybrid_scf = DFA4REST::get_hybrid_libxc(&dfa_compnt_scf,spin_channel);
        // The HF reference carries the full exact exchange:
        // dfa_hybrid_scf = 1.0 so that every response consumer (TDDFT/
        // stability matvecs, gradients, hessians, energy components) sees the
        // correct exchange scaling. Keyed on the requested name ("hf"), not on
        // the component list. The SCF's own Fock build dispatches on the empty
        // component list into the dedicated `generate_hf_hamiltonian_*` paths,
        // which apply the exchange at unit strength directly and do not read
        // this field.
        if tmp_name == "hf" {
            dfa_hybrid_scf = 1.0;
        }
        let dfa_paramr_scf =  vec![1.0;dfa_compnt_scf.len()];
        let dfa_rsh_scf = DFA4REST::get_rsh_libxc(&dfa_compnt_scf, spin_channel);

        let mut dfa = DFA4REST {
            spin_channel,
            dfa_family_pos: None,
            dfa_compnt_pos: None,
            dfa_paramr_pos: None,
            dfa_hybrid_pos: None,
            dfa_paramr_adv: None,
            dfa_compnt_scf,
            dfa_paramr_scf,
            dfa_hybrid_scf,
            dfa_rsh_scf,
            nlc_vv10: None,
        };
        if legacy_nlc_ids.is_some() {
            dfa.nlc_vv10 = dfa
                .dfa_compnt_scf
                .iter()
                .map(|id| dfa.init_libxc(id))
                .find_map(|func| func.vv10_coef());
        }
        dfa
    }

    pub fn parse_scf_nonstd(codelist: &Vec<String>, paramlist: &Vec<f64>, dfa_hybrid_scf: &f64, spin_channel: usize) -> DFA4REST {
        if codelist.len()!=paramlist.len() {panic!("codelist (len: {}) does not match paramlist (len: {})", codelist.len(), paramlist.len())}
        // Parse the xc functionals
        let dfa_compnt_scf = codelist.iter().map(|xc| {
            let xc_code = DFA4REST::libxc_code_fdqc(xc);
            xc_code.iter().filter(|x| **x!=0).map(|x| *x).collect::<Vec<usize>>()
        }).flatten().collect::<Vec<usize>>();
        // Parse the xc parameters
        let dfa_paramr_scf = codelist.iter().zip(paramlist.iter()).map(|(xc, param)| {
            let xc_code = DFA4REST::libxc_code_fdqc(xc);
            xc_code.iter().filter(|x| **x!=0).map(|x| *param).collect::<Vec<f64>>()
        }).flatten().collect::<Vec<f64>>();
        let dfa_rsh_scf = DFA4REST::get_rsh_libxc(&dfa_compnt_scf, spin_channel);

        println!("==== IGOR debug for nonstd DFT parse ====");
        println!("codelist: {:?}, xc_hybrid: {:16.8}", codelist, dfa_hybrid_scf);
        println!("dfa_compnt_scf: {:?}", &dfa_compnt_scf);
        println!("dfa_paramr_scf: {:?}", &dfa_paramr_scf);
        println!("==== IGOR debug for nonstd DFT parse ====");
        DFA4REST {
            spin_channel,
            dfa_family_pos: None,
            dfa_compnt_pos: None,
            dfa_paramr_pos: None,
            dfa_hybrid_pos: None,
            dfa_paramr_adv: None,
            dfa_compnt_scf,
            dfa_paramr_scf,
            dfa_hybrid_scf: *dfa_hybrid_scf,
            dfa_rsh_scf,
            nlc_vv10: None,
        }
    }

    pub fn parse_postscf(name: &str, spin_channel: usize) -> Option<DFA4REST> {
        let tmp_name = name.to_lowercase();
        if tmp_name.starts_with("dsd") {
            panic!(
                "DSD-type functional '{}' is no longer supported by the legacy xc parser: \
                the DSD* definitions in legacy parser are confusing. Please set xc_parser = \"parse_xc\" \
                in the ctrl input instead, e.g.\n \
                xc_parser = \"parse_xc\"\n \
                xc = \"dsdpbep86-d3bj\"\n \
                or \n \
                xc_parser = \"parse_xc\"\n \
                xc = \"dsdpbep86\"\n \
                (no dispersion version)",
                name
            );
        }
        if tmp_name.eq("xyg3") {
            // XYG3 functional
            // Proc. Natl. Acad. Sci. U.S.A. 106, 13, 4963-4968 (2009); https://pnas.org/doi/full/10.1073/pnas.0901093106
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
            let dfa_hybrid_scf = DFA4REST::get_hybrid_libxc(&dfa_compnt_scf, spin_channel);
            Some(DFA4REST{
                spin_channel,
                dfa_compnt_scf,
                dfa_paramr_scf,
                dfa_hybrid_scf,
                dfa_paramr_adv,
                dfa_family_pos,
                dfa_compnt_pos,
                dfa_paramr_pos,
                dfa_hybrid_pos,
                dfa_rsh_scf: None,
            nlc_vv10: None,
        })
        } else if tmp_name.eq("xygjos") {
            // XYGJ-OS functional
            // Proc. Natl. Acad. Sci. U.S.A. 108, 50, 19896-19900 (2011); https://pnas.org/doi/full/10.1073/pnas.1115123108
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
                dfa_hybrid_pos,
                dfa_rsh_scf: None,
            nlc_vv10: None,
        })
        } else if tmp_name.eq("xyg7") {
            // XYG7 functional 
            // J. Phys. Chem. Lett. 12, 10, 2638-2644 (2021); https://doi.org/10.1021/acs.jpclett.1c00360
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
                dfa_hybrid_pos,
                dfa_rsh_scf: None,
            nlc_vv10: None,
        })
        } else if tmp_name.eq("xyg2") {
            // XYG2 functional
            // Yan, W., PhD thesis, Fudan University, Shanghai, China (2022).
            // Precision Chemistry (2026); https://doi.org/10.1021/prechem.5c00432
            let dfa_family_pos = Some(DFAFamily::PT2);
            let pos_dfa = ["lda_x_slater", "gga_x_b88","lda_c_vwn_rpa","gga_c_lyp"];
            let dfa_compnt_pos: Option<Vec<usize>> = Some(pos_dfa.iter().map(|xc| {
                DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
                .flatten()
                .collect());
            let dfa_paramr_pos = Some(vec![0.00,0.1984,0.00,0.6613]);
            let dfa_hybrid_pos = Some(0.8016);
            let dfa_paramr_adv = Some(vec![0.3387,0.3387]);

            let scf_dfa = ["b3lyp"];
            let dfa_compnt_scf: Vec<usize> = scf_dfa.iter().map(|xc| {
                DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
                .flatten().collect();
            let dfa_paramr_scf = vec![1.0;dfa_compnt_scf.len()];
            let dfa_hybrid_scf = DFA4REST::get_hybrid_libxc(&dfa_compnt_scf, spin_channel);
            Some(DFA4REST{
                spin_channel,
                dfa_compnt_scf,
                dfa_paramr_scf,
                dfa_hybrid_scf,
                dfa_paramr_adv,
                dfa_family_pos,
                dfa_compnt_pos,
                dfa_paramr_pos,
                dfa_hybrid_pos,
                dfa_rsh_scf: None,
            nlc_vv10: None,
        })
        } else if tmp_name.eq("xdh-pbe0") {
            // xDH-PBE0 functional
            // J. Chem. Phys. 136, 174103 (2012); https://doi.org/10.1063/1.3703893
            let dfa_family_pos = Some(DFAFamily::PT2);
            let pos_dfa = ["gga_x_pbe", "gga_c_pbe"];
            let dfa_compnt_pos: Option<Vec<usize>> = Some(pos_dfa.iter().map(|xc| {
                DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
                .flatten().collect());
            let dfa_paramr_pos = Some(vec![0.1665, 0.5292]);
            let dfa_hybrid_pos = Some(0.8335);
            let dfa_paramr_adv = Some(vec![0.5428, 0.0000]);

            let dfa_family_scf = DFAFamily::HybridGGA;
            let scf_dfa = ["pbe0"];
            //let dfa_compnt_scf: Vec<XcFuncType> = scf_dfa.iter().map(|xc| {
            //    DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
            //    .flatten().collect();
            let dfa_compnt_scf: Vec<usize> = scf_dfa.iter().map(|xc| {
                DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
                .flatten().collect();
            let dfa_paramr_scf = vec![1.0; dfa_compnt_scf.len()];
            let dfa_hybrid_scf = DFA4REST::get_hybrid_libxc(&dfa_compnt_scf, spin_channel);
            Some(DFA4REST{
                spin_channel,
                dfa_compnt_scf,
                dfa_paramr_scf,
                dfa_hybrid_scf,
                dfa_paramr_adv,
                dfa_family_pos,
                dfa_compnt_pos,
                dfa_paramr_pos,
                dfa_hybrid_pos,
                dfa_rsh_scf: None,
            nlc_vv10: None,
        })
        } else if tmp_name.eq("zrps") {
            // ZRPS
            // Phys. Rev. Lett. 117, 133002 (2016); https://doi.org/10.1103/PhysRevLett.117.133002
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
                dfa_hybrid_pos,
                dfa_rsh_scf: None,
            nlc_vv10: None,
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
                dfa_hybrid_pos,
                dfa_rsh_scf: None,
            nlc_vv10: None,
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
                dfa_hybrid_pos,
                dfa_rsh_scf: None,
            nlc_vv10: None,
        })
        } else if tmp_name.eq("scsrpa") {
            // scsRPA
            // J. Phys. Chem. Lett. 10, 10, 2617-2623 (2019); https://doi.org/10.1021/acs.jpclett.9b00946
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
                dfa_hybrid_pos,
                dfa_rsh_scf: None,
            nlc_vv10: None,
        })
        } else if tmp_name.eq("r-xdh7") {
            // R-xDH7
            // JACS Au 4, 8, 3205-3216 (2024); https://doi.org/10.1021/jacsau.4c00488
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
                dfa_hybrid_pos,
                dfa_rsh_scf: None,
            nlc_vv10: None,
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
                dfa_hybrid_pos,
                dfa_rsh_scf: None,
            nlc_vv10: None,
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
                dfa_hybrid_pos,
                dfa_rsh_scf: None,
            nlc_vv10: None,
        })
        } else if tmp_name.eq("b2plyp") {
            // below are some popular B2PLYP-type DH functionals
            // B2PLYP
            // J. Chem. Phys. 2006, 124, 034108.
            // KS part
            let dfa_family_scf = DFAFamily::HybridGGA;
            let scf_dfa = ["gga_x_b88", "gga_c_lyp"];
            let dfa_compnt_scf: Vec<usize> = scf_dfa.iter().map(|xc| {
                DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
                .flatten().collect();
            let dfa_paramr_scf = vec![0.47, 0.73];
            let dfa_hybrid_scf = 0.53;
            // PT2 part
            let dfa_family_pos = Some(DFAFamily::PT2);
            let dfa_paramr_adv = Some(vec![0.27, 0.27]);
            let dfa_compnt_pos: Option<Vec<usize>> = Some(dfa_compnt_scf.clone());
            let dfa_paramr_pos = Some(dfa_paramr_scf.clone());
            let dfa_hybrid_pos = Some(dfa_hybrid_scf);
            Some(DFA4REST{
                spin_channel,
                dfa_compnt_scf,
                dfa_paramr_scf,
                dfa_hybrid_scf,
                dfa_paramr_adv,
                dfa_family_pos,
                dfa_compnt_pos,
                dfa_paramr_pos,
                dfa_hybrid_pos,
                dfa_rsh_scf: None,
            nlc_vv10: None,
        })
        } else if tmp_name.eq("b2gpplyp") {
            // B2GP-PLYP
            // J. Phys. Chem. A 2008, 112, 12868–12886.
            // KS part
            let dfa_family_scf = DFAFamily::HybridGGA;
            let scf_dfa = ["gga_x_b88", "gga_c_lyp"];
            let dfa_compnt_scf: Vec<usize> = scf_dfa.iter().map(|xc| {
                DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
                .flatten().collect();
            let dfa_paramr_scf = vec![0.35, 0.64];
            let dfa_hybrid_scf = 0.65;
            // PT2 part
            let dfa_family_pos = Some(DFAFamily::PT2);
            let dfa_paramr_adv = Some(vec![0.36, 0.36]);
            let dfa_compnt_pos: Option<Vec<usize>> = Some(dfa_compnt_scf.clone());
            let dfa_paramr_pos = Some(dfa_paramr_scf.clone());
            let dfa_hybrid_pos = Some(dfa_hybrid_scf);
            Some(DFA4REST{
                spin_channel,
                dfa_compnt_scf,
                dfa_paramr_scf,
                dfa_hybrid_scf,
                dfa_paramr_adv,
                dfa_family_pos,
                dfa_compnt_pos,
                dfa_paramr_pos,
                dfa_hybrid_pos,
                dfa_rsh_scf: None,
            nlc_vv10: None,
        })
        } else if tmp_name.eq("pbe-qidh") {
            // PBE-QIDH
            // J. Chem. Phys. 2014, 141, 031101.
            let dfa_family_scf = DFAFamily::HybridGGA;
            let scf_dfa = ["gga_x_pbe", "gga_c_pbe"];
            let dfa_compnt_scf: Vec<usize> = scf_dfa.iter().map(|xc| {
                DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
                .flatten().collect();
            let dfa_paramr_scf = vec![0.306639, 0.666667];
            let dfa_hybrid_scf = 0.693361;
            // PT2 part
            let dfa_family_pos = Some(DFAFamily::PT2);
            let dfa_paramr_adv = Some(vec![0.333333, 0.333333]);
            let dfa_compnt_pos: Option<Vec<usize>> = Some(dfa_compnt_scf.clone());
            let dfa_paramr_pos = Some(dfa_paramr_scf.clone());
            let dfa_hybrid_pos = Some(dfa_hybrid_scf);
            Some(DFA4REST{
                spin_channel,
                dfa_compnt_scf,
                dfa_paramr_scf,
                dfa_hybrid_scf,
                dfa_paramr_adv,
                dfa_family_pos,
                dfa_compnt_pos,
                dfa_paramr_pos,
                dfa_hybrid_pos,
                dfa_rsh_scf: None,
            nlc_vv10: None,
        })
        } else if tmp_name.eq("pbe0dh") {
            // PBE0-DH
            // J. Chem. Phys. 2011, 135, 024106.
            let dfa_family_scf = DFAFamily::HybridGGA;
            let scf_dfa = ["gga_x_pbe", "gga_c_pbe"];
            let dfa_compnt_scf: Vec<usize> = scf_dfa.iter().map(|xc| {
                DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
                .flatten().collect();
            let dfa_paramr_scf = vec![0.50, 0.875];
            let dfa_hybrid_scf = 0.50;
            // PT2 part
            let dfa_family_pos = Some(DFAFamily::PT2);
            let dfa_paramr_adv = Some(vec![0.125, 0.125]);
            let dfa_compnt_pos: Option<Vec<usize>> = Some(dfa_compnt_scf.clone());
            let dfa_paramr_pos = Some(dfa_paramr_scf.clone());
            let dfa_hybrid_pos = Some(dfa_hybrid_scf);
            Some(DFA4REST{
                spin_channel,
                dfa_compnt_scf,
                dfa_paramr_scf,
                dfa_hybrid_scf,
                dfa_paramr_adv,
                dfa_family_pos,
                dfa_compnt_pos,
                dfa_paramr_pos,
                dfa_hybrid_pos,
                dfa_rsh_scf: None,
            nlc_vv10: None,
        })
        } else if tmp_name.eq("r-xyg3") {
            // Renormalized XYG3 functional (experimental)
            // Replaces PT2 correlation with sBGE2 in the post-SCF part
            // No publication yet - experimental test of sBGE2 in XYG3 framework
            let dfa_family_pos = Some(DFAFamily::SBGE2);
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
            let dfa_hybrid_scf = DFA4REST::get_hybrid_libxc(&dfa_compnt_scf, spin_channel);
            Some(DFA4REST{
                spin_channel,
                dfa_compnt_scf,
                dfa_paramr_scf,
                dfa_hybrid_scf,
                dfa_paramr_adv,
                dfa_family_pos,
                dfa_compnt_pos,
                dfa_paramr_pos,
                dfa_hybrid_pos,
                dfa_rsh_scf: None,
            nlc_vv10: None,
        })
        } else if tmp_name.eq("r-xygjos") {
            // Renormalized XYGJOS functional (experimental)
            // Replaces PT2 correlation with sBGE2 in the post-SCF part
            // No publication yet - experimental test of sBGE2 in XYGJOS framework
            let dfa_family_pos = Some(DFAFamily::SBGE2);
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
                dfa_hybrid_pos,
                dfa_rsh_scf: None,
            nlc_vv10: None,
        })
        } else if tmp_name.eq("r-xyg7") {
            // Renormalized XYG7 functional (experimental)
            // Replaces PT2 correlation with sBGE2 in the post-SCF part
            // No publication yet - experimental test of sBGE2 in XYG7 framework
            let dfa_family_pos = Some(DFAFamily::SBGE2);
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
                dfa_hybrid_pos,
                dfa_rsh_scf: None,
            nlc_vv10: None,
        })
        } else if tmp_name.eq("r-xyg2") {
            // Renormalized XYG2 functional (experimental)
            let dfa_family_pos = Some(DFAFamily::SBGE2);
            let pos_dfa = ["lda_x_slater", "gga_x_b88","lda_c_vwn_rpa","gga_c_lyp"];
            let dfa_compnt_pos: Option<Vec<usize>> = Some(pos_dfa.iter().map(|xc| {
                DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
                .flatten()
                .collect());
            let dfa_paramr_pos = Some(vec![0.00,0.1984,0.00,0.6613]);
            let dfa_hybrid_pos = Some(0.8016);
            let dfa_paramr_adv = Some(vec![0.3387,0.3387]);

            let scf_dfa = ["b3lyp"];
            let dfa_compnt_scf: Vec<usize> = scf_dfa.iter().map(|xc| {
                DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
                .flatten().collect();
            let dfa_paramr_scf = vec![1.0;dfa_compnt_scf.len()];
            let dfa_hybrid_scf = DFA4REST::get_hybrid_libxc(&dfa_compnt_scf, spin_channel);
            Some(DFA4REST{
                spin_channel,
                dfa_compnt_scf,
                dfa_paramr_scf,
                dfa_hybrid_scf,
                dfa_paramr_adv,
                dfa_family_pos,
                dfa_compnt_pos,
                dfa_paramr_pos,
                dfa_hybrid_pos,
                dfa_rsh_scf: None,
            nlc_vv10: None,
        })
        } else {
            None
        }
    }

    pub fn is_dfa_scf(&self) -> bool {
        self.dfa_compnt_scf.len() !=0
    }

    pub fn is_hybrid(&self) -> bool {
        self.dfa_hybrid_scf.abs() >= 1.0e-6 || self.is_rsh()
    }

    pub fn is_fifth_dfa(&self) -> bool {
        match self.dfa_family_pos {
            None => false,
            _ => true
        }
    }

    pub fn is_rpa(&self) -> bool {
        match self.dfa_family_pos {
            Some(DFAFamily::RPA) => true,
            _ => false
        }
    }

    pub fn use_eri(&self) -> bool {
        self.is_hybrid() || self.is_fifth_dfa()
    }


    pub fn use_density_gradient(&self) -> bool {
        self
            .dfa_compnt_pos
            .iter()
            .flatten()
            .chain(self.dfa_compnt_scf.iter())
            .any(|xc_func_id| {
                // Only LDA and HybLDA do not need density gradient
                !matches!(self.init_libxc(xc_func_id).family(), libxc::enums::LibXCFamily::LDA | libxc::enums::LibXCFamily::HybLDA)
            })
    }

    fn prepare_dft_quantities(
        &self,
        grids: &Grids,
        spin_channel: usize,
        mo: &[MatrixFull<f64>; 2],
        occ: &[Vec<f64>; 2],
        use_density_gradient: bool,
    ) -> (MatrixFull<f64>, RIFull<f64>, MatrixFull<f64>, MatrixFull<f64>, MatrixFull<f64>) {
        let num_grids = grids.coordinates.len();
        let all_grids = 0..num_grids;
        let mut rho: MatrixFull<f64> = MatrixFull::empty();
        let mut rhop: RIFull<f64> = RIFull::empty();
        let mut lapl: MatrixFull<f64> = MatrixFull::empty();
        let mut tau: MatrixFull<f64> = MatrixFull::empty();
       
        if self.use_kinetic_density() {
            let rho_ensemble = if grids.ao_compressed.is_some() {
                let mo_use: [MatrixFull<f64>; 2] = if !mo[1].data.is_empty() || spin_channel == 1 {
                    [mo[0].clone(), mo[1].clone()]
                } else {
                    [mo[0].clone(), mo[0].clone()]
                };
                grids.prepare_tabulated_density_ensemble_slots_compressed(
                    &self,
                    &mo_use,
                    occ,
                    spin_channel,
                    all_grids.clone(),
                )
            } else {
                grids.prepare_tabulated_density_ensemble_slots(&self, mo, occ, spin_channel, all_grids.clone())
            };
            let rho_vec = rho_ensemble.get_reducing_matrix(0).unwrap().iter().copied().collect_vec();
            rho = MatrixFull::from_vec([num_grids, spin_channel], rho_vec).unwrap();
            let rhop_vec:Vec<f64> = rho_ensemble.get_slices(0..num_grids, 0..spin_channel, 1..4).copied().collect();
            rhop = RIFull::from_vec([num_grids, spin_channel, 3],rhop_vec).unwrap();
            rhop = rhop.transpose_ikj();
            if self.use_kinetic_density() {
                // let lapl_vec = rho_ensemble.get_reducing_matrix(4).unwrap().iter().copied().collect_vec();
                // lapl = MatrixFull::from_vec([num_grids, spin_channel], lapl_vec).unwrap();
                lapl = MatrixFull::new([num_grids, spin_channel], 0.0);
                let tau_vec = rho_ensemble.get_reducing_matrix(5).unwrap().iter().copied().collect_vec();
                tau = MatrixFull::from_vec([num_grids, spin_channel], tau_vec).unwrap();
            }
            // mGGA: tau needed — use dense path (compressed mGGA not yet implemented for this path)
            // (rho, rhop, tau) = grids.prepare_tabulated_density_3(mo, occ, spin_channel);
            // change to the compressed version
            // (rho, rhop, tau) = grids.prepare_tabulated_density_
            // lapl = MatrixFull::new([num_grids, spin_channel], 0.0);
        } else if grids.ao_compressed.is_some() {
            // Use compressed density computation (avoids decompressing dense AO/AOP)
            let mo_use: [MatrixFull<f64>; 2] = if !mo[1].data.is_empty() || spin_channel == 1 {
                [mo[0].clone(), mo[1].clone()]
            } else {
                [mo[0].clone(), mo[0].clone()]
            };
            (rho, rhop) = grids.prepare_tabulated_density_slots_compressed(
                &mo_use, occ, spin_channel, all_grids,
            );
        } else {
            (rho, rhop) = if !mo[1].data.is_empty() || spin_channel == 1 {
                grids.prepare_tabulated_density_2(mo, occ, spin_channel)
            } else {
                let mut mo_temp = mo.clone();
                mo_temp[1] = mo_temp[0].clone();
                grids.prepare_tabulated_density_2(&mo_temp, occ, spin_channel)
            };
        }

        let sigma = if use_density_gradient {
            prepare_tabulated_sigma_rayon(&rhop, spin_channel)
        } else {
            MatrixFull::empty()
        };

        (rho, rhop, sigma, lapl, tau)
    }

    fn compute_exc_by_family(&self, xc_func: &LibXCFunctional, spin_channel: usize, rho: &MatrixFull<f64>, sigma: &MatrixFull<f64>, lapl: &MatrixFull<f64>, tau: &MatrixFull<f64>) -> MatrixFull<f64> {
        let num_grids = rho.size()[0];
        match xc_func.family() {
            LibXCFamily::LDA => MatrixFull::from_vec([num_grids, 1], if spin_channel == 1 {
                lda_exc(xc_func, rho.data_ref().unwrap())
            } else {
                lda_exc(xc_func, rho.transpose().data_ref().unwrap())
            }).unwrap(),
            LibXCFamily::GGA | LibXCFamily::HybGGA => MatrixFull::from_vec([num_grids, 1], if spin_channel == 1 {
                gga_exc(xc_func, rho.data_ref().unwrap(), sigma.data_ref().unwrap())
            } else {
                gga_exc(xc_func, rho.transpose().data_ref().unwrap(), sigma.transpose().data_ref().unwrap())
            }).unwrap(),
            LibXCFamily::MGGA | LibXCFamily::HybMGGA => MatrixFull::from_vec([num_grids, 1], if spin_channel == 1 {
                mgga_exc(xc_func, rho.data_ref().unwrap(), sigma.data_ref().unwrap(), lapl.data_ref().unwrap(), tau.data_ref().unwrap())
            } else {
                mgga_exc(xc_func, rho.transpose().data_ref().unwrap(), sigma.transpose().data_ref().unwrap(), lapl.transpose().data_ref().unwrap(), tau.transpose().data_ref().unwrap())
            }).unwrap(),
            xc_family => panic!("{xc_family:?} is not yet implemented"),
        }
    }

    fn integrate_exc(&self, exc: &MatrixFull<f64>, rho: &MatrixFull<f64>, weights: &[f64], spin_channel: usize) -> (Vec<f64>, [f64; 2]) {
        let mut exc_total = vec![0.0; spin_channel];
        let mut total_elec = [0.0; 2];
        for i_spin in 0..spin_channel {
            let total_elec_s = total_elec.get_mut(i_spin).unwrap();
            exc_total[i_spin] = izip!(exc.data.iter(), rho.iter_column(i_spin), weights.iter())
                .fold(0.0, |acc, (exc, rho, weight)| {
                    *total_elec_s += rho * weight;
                    acc + exc * rho * weight
                });
        }
        (exc_total, total_elec)
    }

    pub fn use_kinetic_density(&self) -> bool {
        self
            .dfa_compnt_pos
            .iter()
            .flatten()
            .chain(self.dfa_compnt_scf.iter())
            .any(|xc_func_id| self.init_libxc(xc_func_id).needs_tau())
    }

    pub fn xc_exc_vxc(&self, grids: &Grids, spin_channel: usize, dm: &Vec<MatrixFull<f64>>, mo: &[MatrixFull<f64>;2], occ: &[Vec<f64>;2], print_level:usize) -> (Vec<f64>, Vec<MatrixFull<f64>>) {
        let num_grids = grids.coordinates.len();
        let num_basis = dm[0].size[0];
        let mut exc = MatrixFull::new([num_grids,1],0.0);
        let mut exc_total = vec![0.0;spin_channel];
        let mut vxc_ao = vec![MatrixFull::new([num_basis,num_grids],0.0);spin_channel];
        let dt0 = utilities::init_timing();

        let (rho, rhop, sigma, _lapl, _tau) = self.prepare_dft_quantities(
            grids,
            spin_channel,
            mo,
            occ,
            self.use_density_gradient(),
        );
        
        let dt2 = utilities::timing(&dt0, Some("evaluate rho/rhop/sigma"));
        let mut vrho = MatrixFull::new([num_grids,spin_channel],0.0);
        let mut vsigma=if self.use_density_gradient() && spin_channel==1 {
            MatrixFull::new([num_grids,1],0.0)
        } else if self.use_density_gradient() && spin_channel==2 {
            MatrixFull::new([num_grids,3],0.0)
        } else {
            MatrixFull::empty()
        };
        let dt3 = utilities::timing(&dt2, Some("init vrho/vsigma"));

        self.dfa_compnt_scf.iter().zip(self.dfa_paramr_scf.iter()).for_each(|(xc_func,xc_para)| {
            let xc_func = self.init_libxc_and_set_param(xc_func);
            match xc_func.family() {
                LibXCFamily::LDA => {
                    if spin_channel==1 {
                        let (tmp_exc,tmp_vrho) = lda_exc_vxc(&xc_func,rho.data_ref().unwrap());
                        let tmp_exc = MatrixFull::from_vec([num_grids,1],tmp_exc).unwrap();
                        let tmp_vrho = MatrixFull::from_vec([num_grids,1],tmp_vrho).unwrap();
                        exc.par_self_scaled_add(&tmp_exc,*xc_para);
                        vrho.par_self_scaled_add(&tmp_vrho,*xc_para);

                    } else {
                        let (tmp_exc,tmp_vrho) = lda_exc_vxc(&xc_func,rho.transpose().data_ref().unwrap());
                        let tmp_exc = MatrixFull::from_vec([num_grids,1],tmp_exc).unwrap();
                        //let tmp_vrho = MatrixFull::from_vec([num_grids,spin_channel],tmp_vrho).unwrap();
                        let tmp_vrho = MatrixFull::from_vec([2,num_grids],tmp_vrho).unwrap();
                        exc.par_self_scaled_add(&tmp_exc,*xc_para);
                        vrho.par_self_scaled_add(&tmp_vrho.transpose_and_drop(),*xc_para);
                    }
                },
                LibXCFamily::GGA | LibXCFamily::HybGGA => {
                    if spin_channel==1 {
                        let (tmp_exc,tmp_vrho, tmp_vsigma) = gga_exc_vxc(&xc_func,rho.data_ref().unwrap(),sigma.data_ref().unwrap());
                        let tmp_exc = MatrixFull::from_vec([num_grids,1],tmp_exc).unwrap();
                        let tmp_vrho = MatrixFull::from_vec([num_grids,1],tmp_vrho).unwrap();
                        let tmp_vsigma= MatrixFull::from_vec([num_grids,1],tmp_vsigma).unwrap();
                        exc.par_self_scaled_add(&tmp_exc,*xc_para);
                        vrho.par_self_scaled_add(&tmp_vrho,*xc_para);
                        vsigma.par_self_scaled_add(&tmp_vsigma, *xc_para);
                    } else {
                        let (tmp_exc,tmp_vrho, tmp_vsigma) = gga_exc_vxc(&xc_func,rho.transpose().data_ref().unwrap(),sigma.transpose().data_ref().unwrap());
                        let tmp_exc = MatrixFull::from_vec([num_grids,1],tmp_exc).unwrap();
                        let tmp_vrho = MatrixFull::from_vec([2,num_grids],tmp_vrho).unwrap();
                        let tmp_vsigma= MatrixFull::from_vec([3,num_grids],tmp_vsigma).unwrap();
                        exc.par_self_scaled_add(&tmp_exc,*xc_para);
                        vrho.par_self_scaled_add(&tmp_vrho.transpose_and_drop(),*xc_para);
                        vsigma.par_self_scaled_add(&tmp_vsigma.transpose_and_drop(), *xc_para);
                    }
                },
                xc_family => panic!("{xc_family:?} is not yet implemented")
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

        let (exc_total, total_elec) = self.integrate_exc(&exc, &rho, &grids.weights, spin_channel);
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

    fn build_active_grid_mask(rho: &MatrixFull<f64>, num_grids: usize, spin_channel: usize, threshold: f64) -> Vec<bool> {
        (0..num_grids)
            .map(|p| {
                let rho_sum: f64 = (0..spin_channel).map(|s| rho[[p, s]].abs()).sum();
                rho_sum > threshold
            })
            .collect()
    }

    fn compact_rho(rho: &MatrixFull<f64>, mask: &[bool], num_active: usize, spin_channel: usize) -> MatrixFull<f64> {
        let mut compact = MatrixFull::new([num_active, spin_channel], 0.0);
        let mut idx = 0usize;
        for p in 0..rho.size[0] {
            if mask[p] {
                for s in 0..spin_channel {
                    compact[[idx, s]] = rho[[p, s]];
                }
                idx += 1;
            }
        }
        compact
    }

    fn compact_exc(exc: &MatrixFull<f64>, mask: &[bool], num_active: usize) -> MatrixFull<f64> {
        let mut compact = MatrixFull::new([num_active, 1], 0.0);
        let mut idx = 0usize;
        for p in 0..exc.size[0] {
            if mask[p] {
                compact[[idx, 0]] = exc[[p, 0]];
                idx += 1;
            }
        }
        compact
    }

    fn compact_vrho(vrho: &MatrixFull<f64>, mask: &[bool], num_active: usize, spin_channel: usize) -> MatrixFull<f64> {
        let mut compact = MatrixFull::new([num_active, spin_channel], 0.0);
        let mut idx = 0usize;
        for p in 0..vrho.size[0] {
            if mask[p] {
                for s in 0..spin_channel {
                    compact[[idx, s]] = vrho[[p, s]];
                }
                idx += 1;
            }
        }
        compact
    }

    fn compact_vsigma(vsigma: &MatrixFull<f64>, mask: &[bool], num_active: usize, ncol: usize) -> MatrixFull<f64> {
        let mut compact = MatrixFull::new([num_active, ncol], 0.0);
        let mut idx = 0usize;
        for p in 0..vsigma.size[0] {
            if mask[p] {
                for c in 0..ncol {
                    compact[[idx, c]] = vsigma[[p, c]];
                }
                idx += 1;
            }
        }
        compact
    }

    fn compact_sigma(sigma: &MatrixFull<f64>, mask: &[bool], num_active: usize, ncol: usize) -> MatrixFull<f64> {
        Self::compact_vsigma(sigma, mask, num_active, ncol)
    }

    fn compact_vtau(vtau: &MatrixFull<f64>, mask: &[bool], num_active: usize, spin_channel: usize) -> MatrixFull<f64> {
        Self::compact_vrho(vtau, mask, num_active, spin_channel)
    }

    fn compact_tau(tau: &MatrixFull<f64>, mask: &[bool], num_active: usize, spin_channel: usize) -> MatrixFull<f64> {
        Self::compact_vrho(tau, mask, num_active, spin_channel)
    }

    fn compact_rhop(rhop: &RIFull<f64>, mask: &[bool], num_active: usize, spin_channel: usize) -> RIFull<f64> {
        let num_total = rhop.size[0];
        let mut data = vec![0.0f64; num_active * 3 * spin_channel];
        let mut idx = 0usize;
        for p in 0..num_total {
            if mask[p] {
                for s in 0..spin_channel {
                    let rhop_s = rhop.get_reducing_matrix(s).unwrap();
                    for x in 0usize..3usize {
                        let rhop_s_x = rhop_s.get_slice_x(x);
                        data[idx + x * num_active + s * num_active * 3] = rhop_s_x[p];
                    }
                }
                idx += 1;
            }
        }
        RIFull::from_vec([num_active, 3, spin_channel], data).unwrap()
    }

    fn compact_weights(weights: &[f64], mask: &[bool], num_active: usize) -> Vec<f64> {
        weights.iter().enumerate()
            .filter(|(p, _)| mask[*p])
            .map(|(_, w)| *w)
            .collect()
    }

    fn extract_active_ao_columns(
        ao: &MatrixFull<f64>,
        range_grids: &std::ops::Range<usize>,
        mask: &[bool],
        num_active: usize,
        num_basis: usize,
    ) -> MatrixFull<f64> {
        let mut ao_active = MatrixFull::new([num_basis, num_active], 0.0);
        let mut idx = 0usize;
        let offset = range_grids.start;
        for p_local in 0..mask.len() {
            if mask[p_local] {
                let p_global = offset + p_local;
                for mu in 0..num_basis {
                    ao_active[[mu, idx]] = ao[[mu, p_global]];
                }
                idx += 1;
            }
        }
        ao_active
    }

    fn extract_active_aop_columns(
        aop: &RIFull<f64>,
        range_grids: &std::ops::Range<usize>,
        mask: &[bool],
        num_active: usize,
        num_basis: usize,
    ) -> RIFull<f64> {
        let ngrids_full = aop.size[1];
        let mut data = vec![0.0f64; num_basis * num_active * 3];
        let mut idx = 0usize;
        let offset = range_grids.start;
        for p_local in 0..mask.len() {
            if mask[p_local] {
                let p_global = offset + p_local;
                for x in 0usize..3usize {
                    let aop_x = aop.get_reducing_matrix(x).unwrap();
                    for mu in 0..num_basis {
                        let flat_src = p_global * num_basis + mu;
                        let flat_dst = mu + idx * num_basis + x * num_basis * num_active;
                        data[flat_dst] = aop_x.data[flat_src];
                    }
                }
                idx += 1;
            }
        }
        RIFull::from_vec([num_basis, num_active, 3], data).unwrap()
    }

    pub fn xc_exc_vxc_slots_dm_only(
        &self, 
        range_grids: Range<usize>, 
        grids: &Grids, 
        spin_channel: usize, 
        dm: &Vec<MatrixFull<f64>>, 
        mo: &[MatrixFull<f64>;2], 
        occ: &[Vec<f64>;2],
        print_level: usize,
        vxc_screen_threshold: f64,
    ) -> (Vec<f64>, Vec<MatrixFull<f64>>, [f64;2]) 
    {
        let num_grids = range_grids.len();
        let num_basis = dm[0].size[0];

        let mut loc_rho = MatrixFull::empty();
        let mut loc_rhop = RIFull::empty();
        let mut loc_lapl = MatrixFull::empty();
        let mut loc_tau = MatrixFull::empty();
        if self.use_kinetic_density() {
            let order = 2; 
            (loc_rho, loc_rhop, loc_tau) = if grids.ao_compressed.is_some() {
                grids.prepare_tabulated_density_2_slots_dm_only_compressed(dm, spin_channel, order, range_grids.clone())
            } else {
                grids.prepare_tabulated_density_2_slots_dm_only(dm, spin_channel, order, range_grids.clone())
            };
            loc_lapl = MatrixFull::new([num_grids, spin_channel], 0.0);
        }
        else {
            (loc_rho,loc_rhop) = if grids.ao_compressed.is_some() {
                if print_level >= 2 && grids.ao.is_some() {
                    // DEBUG: compare dense vs compressed density
                    let (r_dense, rp_dense) = grids.prepare_tabulated_density_slots_dm_only(dm, spin_channel, range_grids.clone());
                    let (r_comp, rp_comp): (MatrixFull<f64>, RIFull<f64>) = grids.prepare_tabulated_density_slots_dm_only_compressed(dm, spin_channel, range_grids.clone());
                    
                    // Compare rho
                    let mut max_d = 0f64; let mut first_g = None;
                    for g in 0..range_grids.len() {
                        let d = (r_dense[[g, 0]] - r_comp[[g, 0]]).abs();
                        if d > max_d { max_d = d; }
                        if d > 1e-8 && first_g.is_none() { first_g = Some((g, r_dense[[g,0]], r_comp[[g,0]])); }
                    }
                    println!(" [DEBUG-non0tab] rho: max|Δ|={:.2e} ngrids={}", max_d, range_grids.len());
                    if let Some((g, dval, cval)) = first_g {
                        println!(" [DEBUG-non0tab] rho first diff: g={} dense={:.6e} comp={:.6e}", g, dval, cval);
                    }
                    
                    // Compare rhop
                    if !rp_dense.size.is_empty() && !rp_comp.size.is_empty() {
                        let mut max_rp = 0f64; let mut first_rp = None;
                        let rd0 = rp_dense.get_reducing_matrix(0).unwrap();
                        let rc0 = rp_comp.get_reducing_matrix(0).unwrap();
                        for x in 0usize..3usize {
                            for g in 0..range_grids.len() {
                                let d = (rd0.get_slice_x(x)[g] - rc0.get_slice_x(x)[g]).abs();
                                if d > max_rp { max_rp = d; }
                                if d > 1e-4 && first_rp.is_none() { first_rp = Some((g, x)); }
                            }
                        }
                        println!(" [DEBUG-non0tab] rhop: max|Δ|={:.2e}", max_rp);
                        if let Some((g, x)) = first_rp {
                            println!(" [DEBUG-non0tab] rhop first diff: g={} x={}", g, x);
                        }
                    }
                    (r_comp, rp_comp)
                } else {
                    grids.prepare_tabulated_density_slots_dm_only_compressed(dm, spin_channel, range_grids.clone())
                }
            } else {
                grids.prepare_tabulated_density_slots_dm_only(dm, spin_channel,range_grids.clone())
            };
        }
        let loc_sigma = if self.use_density_gradient() {
            prepare_tabulated_sigma(&loc_rhop, spin_channel)
        } else {
            MatrixFull::empty()
        };

        // Density screening
        // The screening decision is per-batch but entirely local: compact only
        // when at least one grid point in this batch falls below the density
        // threshold. This avoids the thread-count dependency of a percentage-based
        // heuristic when grids are spatially (contiguously) partitioned.
        // `vxc_screen_threshold <= 0.0` disables the screen outright: no mask is
        // built and the grids are not scanned again.
        let mut active_mask: Vec<bool> = Vec::new();
        let mut num_active = num_grids;
        let mut use_screening = false;
        if vxc_screen_threshold > 0.0 {
            active_mask = Self::build_active_grid_mask(&loc_rho, num_grids, spin_channel, vxc_screen_threshold);
            num_active = active_mask.iter().filter(|&&x| x).count();
            use_screening = num_active > 0 && num_active < num_grids * 3 / 4;
        }

        if print_level >= 1 && use_screening {
            let ratio = num_active as f64 / num_grids as f64 * 100.0;
            println!(" [VXC-screen(dm_only)] active grids: {}/{} ({:.1}%), saved ~{:.0}%",
                num_active, num_grids, ratio, 100.0 - ratio);
        }

        if use_screening {
            // === Screened path: compact arrays, smaller vxc_ao ===
            let n_g = num_active;
            let loc_rho_c = Self::compact_rho(&loc_rho, &active_mask, num_active, spin_channel);
            let loc_rhop_c = Self::compact_rhop(&loc_rhop, &active_mask, num_active, spin_channel);
            let loc_sigma_c = if self.use_density_gradient() {
                Self::compact_sigma(&loc_sigma, &active_mask, num_active,
                    if spin_channel==1 {1} else {3})
            } else { MatrixFull::empty() };
            let loc_lapl_c = if self.use_kinetic_density() {
                Self::compact_vrho(&loc_lapl, &active_mask, num_active, spin_channel)
            } else { MatrixFull::empty() };
            let loc_tau_c = if self.use_kinetic_density() {
                Self::compact_tau(&loc_tau, &active_mask, num_active, spin_channel)
            } else { MatrixFull::empty() };
            let mut loc_exc_c = MatrixFull::new([num_active, 1], 0.0);

            let mut loc_vrho_c = MatrixFull::new([n_g, spin_channel], 0.0);
            let mut loc_vsigma_c = if self.use_density_gradient() && spin_channel==1 {
                MatrixFull::new([n_g, 1], 0.0)
            } else if self.use_density_gradient() && spin_channel==2 {
                MatrixFull::new([n_g, 3], 0.0)
            } else { MatrixFull::empty() };
            let mut loc_vtau_c = if self.use_kinetic_density() {
                MatrixFull::new([n_g, spin_channel], 0.0)
            } else { MatrixFull::empty() };

            // libxc on compact arrays
            self.dfa_compnt_scf.iter().zip(self.dfa_paramr_scf.iter()).for_each(|(xc_func,xc_para)| {
                let xc_func = self.init_libxc_and_set_param(xc_func);
                match xc_func.family() {
                    LibXCFamily::LDA => {
                        if spin_channel==1 {
                            let (tmp_exc,tmp_vrho) = lda_exc_vxc(&xc_func,loc_rho_c.data_ref().unwrap());
                            let tmp_exc = MatrixFull::from_vec([n_g,1],tmp_exc).unwrap();
                            let tmp_vrho = MatrixFull::from_vec([n_g,1],tmp_vrho).unwrap();
                            loc_exc_c.self_scaled_add(&tmp_exc,*xc_para);
                            loc_vrho_c.self_scaled_add(&tmp_vrho,*xc_para);
                        } else {
                            let (tmp_exc,tmp_vrho) = lda_exc_vxc(&xc_func,loc_rho_c.transpose().data_ref().unwrap());
                            let tmp_exc = MatrixFull::from_vec([n_g,1],tmp_exc).unwrap();
                            let tmp_vrho = MatrixFull::from_vec([2,n_g],tmp_vrho).unwrap();
                            loc_exc_c.self_scaled_add(&tmp_exc,*xc_para);
                            loc_vrho_c.self_scaled_add(&tmp_vrho.transpose_and_drop(),*xc_para);
                        }
                    },
                    LibXCFamily::GGA | LibXCFamily::HybGGA => {
                        if spin_channel==1 {
                            let (tmp_exc,tmp_vrho, tmp_vsigma) = gga_exc_vxc(&xc_func,loc_rho_c.data_ref().unwrap(),loc_sigma_c.data_ref().unwrap());
                            let tmp_exc = MatrixFull::from_vec([n_g,1],tmp_exc).unwrap();
                            let tmp_vrho = MatrixFull::from_vec([n_g,1],tmp_vrho).unwrap();
                            let tmp_vsigma= MatrixFull::from_vec([n_g,1],tmp_vsigma).unwrap();
                            loc_exc_c.self_scaled_add(&tmp_exc,*xc_para);
                            loc_vrho_c.self_scaled_add(&tmp_vrho,*xc_para);
                            loc_vsigma_c.self_scaled_add(&tmp_vsigma, *xc_para);
                        } else {
                            let (tmp_exc,tmp_vrho, tmp_vsigma) = gga_exc_vxc(&xc_func,loc_rho_c.transpose().data_ref().unwrap(),loc_sigma_c.transpose().data_ref().unwrap());
                            let tmp_exc = MatrixFull::from_vec([n_g,1],tmp_exc).unwrap();
                            let tmp_vrho = MatrixFull::from_vec([2,n_g],tmp_vrho).unwrap();
                            let tmp_vsigma= MatrixFull::from_vec([3,n_g],tmp_vsigma).unwrap();
                            loc_exc_c.self_scaled_add(&tmp_exc,*xc_para);
                            loc_vrho_c.self_scaled_add(&tmp_vrho.transpose_and_drop(),*xc_para);
                            loc_vsigma_c.self_scaled_add(&tmp_vsigma.transpose_and_drop(), *xc_para);
                        }
                    },
                    LibXCFamily::MGGA | LibXCFamily::HybMGGA => {
                        if spin_channel==1 {
                            let (tmp_exc,tmp_vrho,tmp_vsigma,tmp_valpl,tmp_vtau)
                                = mgga_exc_vxc(&xc_func, loc_rho_c.data_ref().unwrap(), loc_sigma_c.data_ref().unwrap(),
                                    loc_lapl_c.data_ref().unwrap(), loc_tau_c.data_ref().unwrap());
                            let tmp_exc = MatrixFull::from_vec([n_g,1],tmp_exc).unwrap();
                            let tmp_vrho = MatrixFull::from_vec([n_g,1],tmp_vrho).unwrap();
                            let tmp_vsigma= MatrixFull::from_vec([n_g,1],tmp_vsigma).unwrap();
                            let tmp_vtau = MatrixFull::from_vec([n_g,1],tmp_vtau).unwrap();
                            loc_exc_c.self_scaled_add(&tmp_exc,*xc_para);
                            loc_vrho_c.self_scaled_add(&tmp_vrho,*xc_para);
                            loc_vsigma_c.self_scaled_add(&tmp_vsigma, *xc_para);
                            loc_vtau_c.self_scaled_add(&tmp_vtau, *xc_para);
                        } else {
                            let (tmp_exc,tmp_vrho,tmp_vsigma,tmp_valpl,tmp_vtau)
                                = mgga_exc_vxc(&xc_func, loc_rho_c.transpose().data_ref().unwrap(), loc_sigma_c.transpose().data_ref().unwrap(),
                                    loc_lapl_c.transpose().data_ref().unwrap(), loc_tau_c.transpose().data_ref().unwrap());
                            let tmp_exc = MatrixFull::from_vec([n_g,1],tmp_exc).unwrap();
                            let tmp_vrho = MatrixFull::from_vec([2,n_g],tmp_vrho).unwrap();
                            let tmp_vsigma= MatrixFull::from_vec([3,n_g],tmp_vsigma).unwrap();
                            let tmp_vtau = MatrixFull::from_vec([2,n_g],tmp_vtau).unwrap();
                            loc_exc_c.self_scaled_add(&tmp_exc,*xc_para);
                            loc_vrho_c.self_scaled_add(&tmp_vrho.transpose_and_drop(),*xc_para);
                            loc_vsigma_c.self_scaled_add(&tmp_vsigma.transpose_and_drop(), *xc_para);
                            loc_vtau_c.self_scaled_add(&tmp_vtau.transpose_and_drop(), *xc_para);
                        }
                    },
                    xc_family => panic!("{xc_family:?} is not yet implemented"),
                }
            });

            // Build compact ao/aop with only active grid columns
            let ao_dense_owned: Option<MatrixFull<f64>>;
            let ao_ref: &MatrixFull<f64> = match &grids.ao {
                Some(a) => { ao_dense_owned = None; a }
                None => {
                    let c = grids.ao_compressed.as_ref().unwrap();
                    ao_dense_owned = Some(Grids::decompress_ao(c));
                    ao_dense_owned.as_ref().unwrap()
                }
            };
            let ao_active = Self::extract_active_ao_columns(
                ao_ref, &range_grids, &active_mask, num_active, num_basis);

            let aop_dense_owned: Option<RIFull<f64>>;
            let aop_ref: Option<&RIFull<f64>> = match &grids.aop {
                Some(a) => { aop_dense_owned = None; Some(a) }
                None => match &grids.aop_compressed {
                    Some(c) => {
                        aop_dense_owned = Some(Grids::decompress_aop(c));
                        Some(aop_dense_owned.as_ref().unwrap())
                    }
                    None => { aop_dense_owned = None; None }
                }
            };
            let aop_active: Option<RIFull<f64>> = aop_ref.map(|aop| {
                Self::extract_active_aop_columns(aop, &range_grids, &active_mask, num_active, num_basis)
            });
            let weights_active = Self::compact_weights(&grids.weights[range_grids.clone()], &active_mask, num_active);

            // vxc_ao arrays
            let mut loc_vxc_ao_0 = vec![MatrixFull::new([num_basis, n_g], 0.0); spin_channel];
            let mut loc_vxc_ao_1 = if self.use_kinetic_density() {
                vec![MatrixFull::new([num_basis, n_g], 0.0); spin_channel]
            } else { vec![] };
            let mut loc_vxc_mat = vec![MatrixFull::new([num_basis, num_basis], 0.0); spin_channel];

            // Build vxc_ao from vrho (compact)
            for i_spin in 0..spin_channel {
                let mut loc_vxc_ao_s = &mut loc_vxc_ao_0[i_spin];
                let loc_vrho_s = loc_vrho_c.slice_column(i_spin);
                let loc_ao_ref = ao_active.to_matrixfullslice_columns(0..n_g);
                contract_vxc_0_serial(loc_vxc_ao_s, &loc_ao_ref, loc_vrho_s, None);
            }
            // GGA sigma
            if self.use_density_gradient() {
                if let Some(ref aop_active) = aop_active {
                    if spin_channel == 1 {
                        let mut loc_vxc_ao_s = &mut loc_vxc_ao_0[0];
                        let loc_vsigma_s = loc_vsigma_c.slice_column(0);
                        let loc_rhop_s = loc_rhop_c.get_reducing_matrix(0).unwrap();
                        let mut loc_wao = MatrixFull::new([num_basis, n_g], 0.0);
                        for x in 0usize..3usize {
                            let loc_aop_x = aop_active.get_reducing_matrix_columns(0..n_g, x).unwrap();
                            let loc_rhop_s_x = loc_rhop_s.get_slice_x(x);
                            contract_vxc_0_serial(&mut loc_wao, &loc_aop_x, loc_rhop_s_x, None);
                        }
                        contract_vxc_0_serial(loc_vxc_ao_s, &loc_wao.to_matrixfullslice(), loc_vsigma_s, Some(4.0));
                    } else {
                        {
                            let mut loc_vxc_ao_a = &mut loc_vxc_ao_0[0];
                            let loc_rhop_a = loc_rhop_c.get_reducing_matrix(0).unwrap();
                            let loc_vsigma_uu = loc_vsigma_c.slice_column(0);
                            let mut loc_dao = MatrixFull::new([num_basis, n_g], 0.0);
                            for x in 0usize..3usize {
                                let loc_aop_x = aop_active.get_reducing_matrix_columns(0..n_g, x).unwrap();
                                let loc_rhop_s_x = loc_rhop_a.get_slice_x(x);
                                contract_vxc_0_serial(&mut loc_dao, &loc_aop_x, loc_rhop_s_x, None);
                            }
                            contract_vxc_0_serial(loc_vxc_ao_a, &loc_dao.to_matrixfullslice(), &loc_vsigma_uu, Some(4.0));
                            let loc_rhop_b = loc_rhop_c.get_reducing_matrix(1).unwrap();
                            let loc_vsigma_ud = loc_vsigma_c.slice_column(1);
                            loc_dao.data.iter_mut().for_each(|d| {*d=0.0});
                            for x in 0usize..3usize {
                                let loc_aop_x = aop_active.get_reducing_matrix_columns(0..n_g, x).unwrap();
                                let loc_rhop_s_x = loc_rhop_b.get_slice_x(x);
                                contract_vxc_0_serial(&mut loc_dao, &loc_aop_x, loc_rhop_s_x, None);
                            }
                            contract_vxc_0_serial(loc_vxc_ao_a, &loc_dao.to_matrixfullslice(), &loc_vsigma_ud, Some(2.0));
                        }
                        {
                            let mut loc_vxc_ao_b = &mut loc_vxc_ao_0[1];
                            let loc_rhop_b = loc_rhop_c.get_reducing_matrix(1).unwrap();
                            let loc_vsigma_dd = loc_vsigma_c.slice_column(2);
                            let mut loc_dao = MatrixFull::new([num_basis, n_g], 0.0);
                            for x in 0usize..3usize {
                                let loc_aop_x = aop_active.get_reducing_matrix_columns(0..n_g, x).unwrap();
                                let loc_rhop_s_x = loc_rhop_b.get_slice_x(x);
                                contract_vxc_0_serial(&mut loc_dao, &loc_aop_x, loc_rhop_s_x, None);
                            }
                            contract_vxc_0_serial(loc_vxc_ao_b, &loc_dao.to_matrixfullslice(), &loc_vsigma_dd, Some(4.0));
                            let loc_rhop_a = loc_rhop_c.get_reducing_matrix(0).unwrap();
                            let loc_vsigma_ud = loc_vsigma_c.slice_column(1);
                            loc_dao.data.iter_mut().for_each(|d| {*d=0.0});
                            for x in 0usize..3usize {
                                let loc_aop_x = aop_active.get_reducing_matrix_columns(0..n_g, x).unwrap();
                                let loc_rhop_s_x = loc_rhop_a.get_slice_x(x);
                                contract_vxc_0_serial(&mut loc_dao, &loc_aop_x, loc_rhop_s_x, None);
                            }
                            contract_vxc_0_serial(loc_vxc_ao_b, &loc_dao.to_matrixfullslice(), &loc_vsigma_ud, Some(2.0));
                        }
                    }
                }
            }

            // GEMM contraction (compact)
            for i_spin in 0..spin_channel {
                let mut loc_vxc_mat_s = loc_vxc_mat.get_mut(i_spin).unwrap();
                let mut loc_vxc_ao_s = loc_vxc_ao_0.get_mut(i_spin).unwrap();
                loc_vxc_ao_s.iter_columns_full_mut().zip(weights_active.iter()).for_each(|(vxc_ao_s,w)| {
                    vxc_ao_s.iter_mut().for_each(|f| {*f *= *w})
                });
                _dgemm(
                    &ao_active, (0..num_basis, 0..n_g), 'N',
                    loc_vxc_ao_s, (0..num_basis, 0..n_g), 'T',
                    loc_vxc_mat_s, (0..num_basis, 0..num_basis),
                    1.0, 0.0
                );
            }
            // MGGA tau
            if self.use_kinetic_density() {
                if let Some(ref aop_active) = aop_active {
                    for i_spin in 0..spin_channel {
                        let loc_vxc_mat_s = loc_vxc_mat.get_mut(i_spin).unwrap();
                        let mut loc_vtau_s = loc_vtau_c.slice_column_mut(i_spin);
                        let mut loc_vxc_ao_1_s = &mut loc_vxc_ao_1[i_spin];
                        loc_vtau_s.iter_mut().zip(weights_active.iter()).for_each(
                            |(vtau_s, w)| {*vtau_s *= *w}
                        );
                        for ic in 0usize..3usize {
                            let loc_aop_ic = aop_active.get_reducing_matrix_columns(0..n_g, ic).unwrap();
                            contract_vxc_0_serial(loc_vxc_ao_1_s, &loc_aop_ic, loc_vtau_s, Some(0.5));
                            _dgemm(
                                &loc_aop_ic, (0..num_basis, 0..n_g), 'N',
                                loc_vxc_ao_1_s, (0..num_basis, 0..n_g), 'T',
                                loc_vxc_mat_s, (0..num_basis, 0..num_basis), 1.0, 1.0
                            );
                            loc_vxc_ao_1_s.data.iter_mut().for_each(|t| {*t=0.0});
                        }
                    }
                }
            }

            let (loc_exc_total, loc_total_elec) = self.integrate_exc(&loc_exc_c, &loc_rho_c, &weights_active, spin_channel);
            (loc_exc_total, loc_vxc_mat, loc_total_elec)
        } else {
            // === Original path: no screening, use full grids ===
            let loc_weights = &grids.weights[range_grids.clone()];
            let mut loc_exc = MatrixFull::new([num_grids, 1], 0.0);
            let mut loc_exc_total = vec![0.0; spin_channel];
            let mut loc_vxc_ao_0 = vec![MatrixFull::new([num_basis, num_grids], 0.0); spin_channel];
            let mut loc_vxc_ao_1 = if self.use_kinetic_density() {
                vec![MatrixFull::new([num_basis, num_grids], 0.0); spin_channel]
            } else { vec![] };
            let mut loc_vxc_mat = vec![MatrixFull::new([num_basis, num_basis], 0.0); spin_channel];

            let mut loc_vrho = MatrixFull::new([num_grids, spin_channel], 0.0);
            let mut loc_vsigma = if self.use_density_gradient() && spin_channel==1 {
                MatrixFull::new([num_grids, 1], 0.0)
            } else if self.use_density_gradient() && spin_channel==2 {
                MatrixFull::new([num_grids, 3], 0.0)
            } else { MatrixFull::empty() };
            let mut loc_vtau = if self.use_kinetic_density() {
                MatrixFull::new([num_grids, spin_channel], 0.0)
            } else { MatrixFull::empty() };

            self.dfa_compnt_scf.iter().zip(self.dfa_paramr_scf.iter()).for_each(|(xc_func,xc_para)| {
                let xc_func = self.init_libxc_and_set_param(xc_func);
                match xc_func.family() {
                    LibXCFamily::LDA => {
                        if spin_channel==1 {
                            let (tmp_exc,tmp_vrho) = lda_exc_vxc(&xc_func,loc_rho.data_ref().unwrap());
                            let tmp_exc = MatrixFull::from_vec([num_grids,1],tmp_exc).unwrap();
                            let tmp_vrho = MatrixFull::from_vec([num_grids,1],tmp_vrho).unwrap();
                            loc_exc.self_scaled_add(&tmp_exc,*xc_para);
                            loc_vrho.self_scaled_add(&tmp_vrho,*xc_para);
                        } else {
                            let (tmp_exc,tmp_vrho) = lda_exc_vxc(&xc_func,loc_rho.transpose().data_ref().unwrap());
                            let tmp_exc = MatrixFull::from_vec([num_grids,1],tmp_exc).unwrap();
                            let tmp_vrho = MatrixFull::from_vec([2,num_grids],tmp_vrho).unwrap();
                            loc_exc.self_scaled_add(&tmp_exc,*xc_para);
                            loc_vrho.self_scaled_add(&tmp_vrho.transpose_and_drop(),*xc_para);
                        }
                    },
                    LibXCFamily::GGA | LibXCFamily::HybGGA => {
                        if spin_channel==1 {
                            let (tmp_exc,tmp_vrho, tmp_vsigma) = gga_exc_vxc(&xc_func,loc_rho.data_ref().unwrap(),loc_sigma.data_ref().unwrap());
                            let tmp_exc = MatrixFull::from_vec([num_grids,1],tmp_exc).unwrap();
                            let tmp_vrho = MatrixFull::from_vec([num_grids,1],tmp_vrho).unwrap();
                            let tmp_vsigma= MatrixFull::from_vec([num_grids,1],tmp_vsigma).unwrap();
                            loc_exc.self_scaled_add(&tmp_exc,*xc_para);
                            loc_vrho.self_scaled_add(&tmp_vrho,*xc_para);
                            loc_vsigma.self_scaled_add(&tmp_vsigma, *xc_para);
                        } else {
                            let (tmp_exc,tmp_vrho, tmp_vsigma) = gga_exc_vxc(&xc_func,loc_rho.transpose().data_ref().unwrap(),loc_sigma.transpose().data_ref().unwrap());
                            let tmp_exc = MatrixFull::from_vec([num_grids,1],tmp_exc).unwrap();
                            let tmp_vrho = MatrixFull::from_vec([2,num_grids],tmp_vrho).unwrap();
                            let tmp_vsigma= MatrixFull::from_vec([3,num_grids],tmp_vsigma).unwrap();
                            loc_exc.self_scaled_add(&tmp_exc,*xc_para);
                            loc_vrho.self_scaled_add(&tmp_vrho.transpose_and_drop(),*xc_para);
                            loc_vsigma.self_scaled_add(&tmp_vsigma.transpose_and_drop(), *xc_para);
                        }
                    },
                    LibXCFamily::MGGA | LibXCFamily::HybMGGA => {
                        if spin_channel==1 {
                            let (tmp_exc,tmp_vrho,tmp_vsigma,tmp_valpl,tmp_vtau)
                                = mgga_exc_vxc(&xc_func, loc_rho.data_ref().unwrap(), loc_sigma.data_ref().unwrap(),
                                    loc_lapl.data_ref().unwrap(), loc_tau.data_ref().unwrap());
                            let tmp_exc = MatrixFull::from_vec([num_grids,1],tmp_exc).unwrap();
                            let tmp_vrho = MatrixFull::from_vec([num_grids,1],tmp_vrho).unwrap();
                            let tmp_vsigma= MatrixFull::from_vec([num_grids,1],tmp_vsigma).unwrap();
                            let tmp_vtau = MatrixFull::from_vec([num_grids,1],tmp_vtau).unwrap();
                            loc_exc.self_scaled_add(&tmp_exc,*xc_para);
                            loc_vrho.self_scaled_add(&tmp_vrho,*xc_para);
                            loc_vsigma.self_scaled_add(&tmp_vsigma, *xc_para);
                            loc_vtau.self_scaled_add(&tmp_vtau, *xc_para);
                        } else {
                            let (tmp_exc,tmp_vrho,tmp_vsigma,tmp_valpl,tmp_vtau)
                                = mgga_exc_vxc(&xc_func, loc_rho.transpose().data_ref().unwrap(), loc_sigma.transpose().data_ref().unwrap(),
                                    loc_lapl.transpose().data_ref().unwrap(), loc_tau.transpose().data_ref().unwrap());
                            let tmp_exc = MatrixFull::from_vec([num_grids,1],tmp_exc).unwrap();
                            let tmp_vrho = MatrixFull::from_vec([2,num_grids],tmp_vrho).unwrap();
                            let tmp_vsigma= MatrixFull::from_vec([3,num_grids],tmp_vsigma).unwrap();
                            let tmp_vtau = MatrixFull::from_vec([2,num_grids],tmp_vtau).unwrap();
                            loc_exc.self_scaled_add(&tmp_exc,*xc_para);
                            loc_vrho.self_scaled_add(&tmp_vrho.transpose_and_drop(),*xc_para);
                            loc_vsigma.self_scaled_add(&tmp_vsigma.transpose_and_drop(), *xc_para);
                            loc_vtau.self_scaled_add(&tmp_vtau.transpose_and_drop(), *xc_para);
                        }
                    },
                    xc_family => panic!("{xc_family:?} is not yet implemented"),
                }
            });

            if let Some(_ao_c) = &grids.ao_compressed {
                // Compressed production path (always active)
                let mut vxc_mat_comp: Vec<MatrixFull<f64>> = vec![MatrixFull::new([num_basis, num_basis], 0.0); spin_channel];
                grids.contract_response_compressed(
                    &range_grids, &loc_vrho, &loc_vsigma, &loc_vtau,
                    loc_weights, &mut vxc_mat_comp, spin_channel,
                    self.use_density_gradient(), self.use_kinetic_density(),
                    &loc_rhop, num_basis,
                );

                // DEBUG compare: only at print_level >= 2 (and dense AO must exist)
                if print_level >= 2 {
                    if let Some(ao) = &grids.ao {
                        let mut vxc_mat_dense: Vec<MatrixFull<f64>> = vec![MatrixFull::new([num_basis, num_basis], 0.0); spin_channel];
                        for i_spin in 0..spin_channel {
                            let loc_vrho_s = loc_vrho.slice_column(i_spin);
                            let loc_ao_ref = ao.to_matrixfullslice_columns(range_grids.clone());
                            let mut vxc_ao_d = MatrixFull::new([num_basis, range_grids.len()], 0.0);
                            contract_vxc_0_serial(&mut vxc_ao_d, &loc_ao_ref, loc_vrho_s, None);
                            vxc_ao_d.iter_columns_full_mut().zip(loc_weights.iter()).for_each(|(col, w)| {
                                col.iter_mut().for_each(|v| *v *= *w);
                            });
                            _dgemm(ao, (0..num_basis, range_grids.clone()), 'N',
                                   &vxc_ao_d, (0..num_basis, 0..range_grids.len()), 'T',
                                   &mut vxc_mat_dense[i_spin], (0..num_basis, 0..num_basis), 1.0, 0.0);
                        }
                        let mut max_vxc = 0f64;
                        let mut first_diff = None;
                        for s in 0..spin_channel {
                            for mu in 0..num_basis {
                                for nu in 0..num_basis {
                                    let d = (vxc_mat_dense[s][[mu, nu]] - vxc_mat_comp[s][[mu, nu]]).abs();
                                    if d > max_vxc { max_vxc = d; }
                                    if d > 1e-8 && first_diff.is_none() { first_diff = Some((s, mu, nu, vxc_mat_dense[s][[mu, nu]], vxc_mat_comp[s][[mu, nu]])); }
                                }
                            }
                        }
                        println!(" [DEBUG-non0tab] vxc_mat: max|Δ|={:.2e} nao={}", max_vxc, num_basis);
                        if let Some((s, mu, nu, dval, cval)) = first_diff {
                            println!(" [DEBUG-non0tab] vxc_mat first diff: spin={} mu={} nu={} dense={:.6e} comp={:.6e}", s, mu, nu, dval, cval);
                        }
                    }
                }

                // Use compressed result
                for s in 0..spin_channel {
                    loc_vxc_mat[s] = vxc_mat_comp[s].clone();
                }
            } else if let Some(ao) = &grids.ao {
                // LDA vrho * AO
                for i_spin in 0..spin_channel {
                    let mut loc_vxc_ao_s = &mut loc_vxc_ao_0[i_spin];
                    let loc_vrho_s = loc_vrho.slice_column(i_spin);
                    let loc_ao_ref = ao.to_matrixfullslice_columns(range_grids.clone());
                    contract_vxc_0_serial(loc_vxc_ao_s, &loc_ao_ref, loc_vrho_s, None);
                }
                // GGA vsigma * rho_p \dot AO_P
                if self.use_density_gradient() {
                    if let Some(aop) = &grids.aop {
                        if spin_channel == 1 {
                            let mut loc_vxc_ao_s = &mut loc_vxc_ao_0[0];
                            let loc_vsigma_s = loc_vsigma.slice_column(0);
                            let loc_rhop_s = loc_rhop.get_reducing_matrix(0).unwrap();
                            let mut loc_wao = MatrixFull::new([num_basis, num_grids], 0.0);
                            for x in 0usize..3usize {
                                let loc_aop_x = aop.get_reducing_matrix_columns(range_grids.clone(), x).unwrap();
                                let loc_rhop_s_x = loc_rhop_s.get_slice_x(x);
                                contract_vxc_0_serial(&mut loc_wao, &loc_aop_x, loc_rhop_s_x, None);
                            }
                            contract_vxc_0_serial(loc_vxc_ao_s, &loc_wao.to_matrixfullslice(), loc_vsigma_s, Some(4.0));
                        } else {
                            {
                                let mut loc_vxc_ao_a = &mut loc_vxc_ao_0[0];
                                let loc_rhop_a = loc_rhop.get_reducing_matrix(0).unwrap();
                                let loc_vsigma_uu = loc_vsigma.slice_column(0);
                                let mut loc_dao = MatrixFull::new([num_basis, num_grids], 0.0);
                                for x in 0usize..3usize {
                                    let loc_aop_x = aop.get_reducing_matrix_columns(range_grids.clone(), x).unwrap();
                                    let loc_rhop_s_x = loc_rhop_a.get_slice_x(x);
                                    contract_vxc_0_serial(&mut loc_dao, &loc_aop_x, loc_rhop_s_x, None);
                                }
                                contract_vxc_0_serial(loc_vxc_ao_a, &loc_dao.to_matrixfullslice(), &loc_vsigma_uu, Some(4.0));
                                let loc_rhop_b = loc_rhop.get_reducing_matrix(1).unwrap();
                                let loc_vsigma_ud = loc_vsigma.slice_column(1);
                                loc_dao.data.iter_mut().for_each(|d| {*d=0.0});
                                for x in 0usize..3usize {
                                    let loc_aop_x = aop.get_reducing_matrix_columns(range_grids.clone(), x).unwrap();
                                    let loc_rhop_s_x = loc_rhop_b.get_slice_x(x);
                                    contract_vxc_0_serial(&mut loc_dao, &loc_aop_x, loc_rhop_s_x, None);
                                }
                                contract_vxc_0_serial(loc_vxc_ao_a, &loc_dao.to_matrixfullslice(), &loc_vsigma_ud, Some(2.0));
                            }
                            {
                                let mut loc_vxc_ao_b = &mut loc_vxc_ao_0[1];
                                let loc_rhop_b = loc_rhop.get_reducing_matrix(1).unwrap();
                                let loc_vsigma_dd = loc_vsigma.slice_column(2);
                                let mut loc_dao = MatrixFull::new([num_basis, num_grids], 0.0);
                                for x in 0usize..3usize {
                                    let loc_aop_x = aop.get_reducing_matrix_columns(range_grids.clone(), x).unwrap();
                                    let loc_rhop_s_x = loc_rhop_b.get_slice_x(x);
                                    contract_vxc_0_serial(&mut loc_dao, &loc_aop_x, loc_rhop_s_x, None);
                                }
                                contract_vxc_0_serial(loc_vxc_ao_b, &loc_dao.to_matrixfullslice(), &loc_vsigma_dd, Some(4.0));
                                let loc_rhop_a = loc_rhop.get_reducing_matrix(0).unwrap();
                                let loc_vsigma_ud = loc_vsigma.slice_column(1);
                                loc_dao.data.iter_mut().for_each(|d| {*d=0.0});
                                for x in 0usize..3usize {
                                    let loc_aop_x = aop.get_reducing_matrix_columns(range_grids.clone(), x).unwrap();
                                    let loc_rhop_s_x = loc_rhop_a.get_slice_x(x);
                                    contract_vxc_0_serial(&mut loc_dao, &loc_aop_x, loc_rhop_s_x, None);
                                }
                                contract_vxc_0_serial(loc_vxc_ao_b, &loc_dao.to_matrixfullslice(), &loc_vsigma_ud, Some(2.0));
                            }
                        }
                    }
                }
                // GEMM contraction for LDA and GGA (compact) AO^T @ vxc_ao
                for i_spin in 0..spin_channel {
                    let mut loc_vxc_mat_s = loc_vxc_mat.get_mut(i_spin).unwrap();
                        let mut loc_vxc_ao_s = loc_vxc_ao_0.get_mut(i_spin).unwrap();
                        loc_vxc_ao_s.iter_columns_full_mut().zip(loc_weights.iter()).for_each(|(vxc_ao_s,w)| {
                            vxc_ao_s.iter_mut().for_each(|f| {*f *= *w})
                        });
                        _dgemm(
                            ao, (0..num_basis, range_grids.clone()), 'N',
                            loc_vxc_ao_s, (0..num_basis, 0..range_grids.len()), 'T',
                            loc_vxc_mat_s, (0..num_basis, 0..num_basis),
                            1.0, 0.0
                        );
                }
                // MGGA part 
                if self.use_kinetic_density() {
                    let Some(aop) = &grids.aop else {
                        panic!("aop is not available in xc_exc_vxc_slots_dm_only");
                    };
                    for i_spin in 0..spin_channel {
                        let loc_vxc_mat_s = loc_vxc_mat.get_mut(i_spin).unwrap();
                        let mut loc_vtau_s = loc_vtau.slice_column_mut(i_spin);
                        let mut loc_vxc_ao_1_s = &mut loc_vxc_ao_1[i_spin];
                        loc_vtau_s.iter_mut().zip(loc_weights.iter()).for_each(
                            |(vtau_s, w)| {*vtau_s *= *w}
                        );
                        for ic in 0usize..3usize {
                            let loc_aop_ic = aop.get_reducing_matrix_columns(range_grids.clone(), ic).unwrap();
                            contract_vxc_0_serial(loc_vxc_ao_1_s, &loc_aop_ic, loc_vtau_s, Some(0.5));
                            _dgemm(
                                &loc_aop_ic, (0..num_basis, 0..range_grids.len()), 'N',
                                loc_vxc_ao_1_s, (0..num_basis, 0..range_grids.len()), 'T',
                                loc_vxc_mat_s, (0..num_basis, 0..num_basis), 1.0, 1.0
                            );
                            loc_vxc_ao_1_s.data.iter_mut().for_each(|t| {*t=0.0});
                        }
                    }
                }
            }
            let (loc_exc_total, loc_total_elec) = self.integrate_exc(&loc_exc, &loc_rho, loc_weights, spin_channel);
            (loc_exc_total, loc_vxc_mat, loc_total_elec)
        }
    }

    pub fn xc_exc_vxc_slots(
        &self, 
        range_grids: Range<usize>, 
        grids: &Grids, 
        spin_channel: usize, 
        dm: &Vec<MatrixFull<f64>>, 
        mo: &[MatrixFull<f64>;2], 
        occ: &[Vec<f64>;2],
        print_level: usize,
        vxc_screen_threshold: f64,
    ) -> (Vec<f64>, Vec<MatrixFull<f64>>, [f64;2]) 
    {
        let num_grids = range_grids.len();
        let num_basis = dm[0].size[0];

        let mut loc_rho: MatrixFull<f64> = MatrixFull::empty();
        let mut loc_rhop: RIFull<f64> = RIFull::empty();
        let mut loc_lapl: MatrixFull<f64> = MatrixFull::empty();
        let mut loc_tau: MatrixFull<f64> = MatrixFull::empty();
        if self.use_kinetic_density() {
            // for meta-GGA
            let mo_use: [MatrixFull<f64>; 2] = if !mo[1].data.is_empty() || spin_channel == 1 {
                [mo[0].clone(), mo[1].clone()]
            } else {
                [mo[0].clone(), mo[0].clone()]
            };

            let loc_rho_ensemble = if grids.ao_compressed.is_some() {
                grids.prepare_tabulated_density_ensemble_slots_compressed(
                    &self,
                    &mo_use,
                    occ,
                    spin_channel,
                    range_grids.clone(),
                )
            } else {
                grids.prepare_tabulated_density_ensemble_slots(&self, &mo_use, occ, spin_channel, range_grids.clone())
            };
            let loc_rho_vec = loc_rho_ensemble.get_reducing_matrix(0).unwrap().iter().copied().collect_vec();
            loc_rho = MatrixFull::from_vec([num_grids, spin_channel], loc_rho_vec).unwrap();
            let loc_rhop_vec:Vec<f64> = loc_rho_ensemble.get_slices(0..num_grids, 0..spin_channel, 1..4).copied().collect();
            loc_rhop = RIFull::from_vec([num_grids, spin_channel, 3],loc_rhop_vec).unwrap();
            loc_rhop = loc_rhop.transpose_ikj();
            let loc_lapl_vec = loc_rho_ensemble.get_reducing_matrix(4).unwrap().iter().copied().collect_vec();
            loc_lapl = MatrixFull::from_vec([num_grids, spin_channel], loc_lapl_vec).unwrap();
            let loc_tau_vec = loc_rho_ensemble.get_reducing_matrix(5).unwrap().iter().copied().collect_vec();
            loc_tau = MatrixFull::from_vec([num_grids, spin_channel], loc_tau_vec).unwrap();
        } else {
            (loc_rho,loc_rhop) = if grids.ao_compressed.is_some() {
                let mo_use: [MatrixFull<f64>; 2] = if ! mo[1].data.is_empty() || spin_channel == 1 {
                    [mo[0].clone(), mo[1].clone()]
                } else {
                    [mo[0].clone(), mo[0].clone()]
                };
                grids.prepare_tabulated_density_slots_compressed(&mo_use, occ, spin_channel, range_grids.clone())
            } else if ! mo[1].data.is_empty() || spin_channel == 1 {
                grids.prepare_tabulated_density_slots(mo, occ, spin_channel,range_grids.clone())
            } else {
                let mut mo_temp = mo.clone();
                mo_temp[1] = mo_temp[0].clone();
                grids.prepare_tabulated_density_slots(&mo_temp, occ, spin_channel,range_grids.clone())
            }
        };
        let loc_sigma = if self.use_density_gradient() {
            prepare_tabulated_sigma(&loc_rhop, spin_channel)
        } else {
            MatrixFull::empty()
        };

        // Density screening
        // The screening decision is per-batch but entirely local: compact only
        // when at least one grid point in this batch falls below the density
        // threshold. This avoids the thread-count dependency of a percentage-based
        // heuristic when grids are spatially (contiguously) partitioned.
        // `vxc_screen_threshold <= 0.0` disables the screen outright: no mask is
        // built and the grids are not scanned again.
        let mut active_mask: Vec<bool> = Vec::new();
        let mut num_active = num_grids;
        let mut use_screening = false;
        if vxc_screen_threshold > 0.0 {
            active_mask = Self::build_active_grid_mask(&loc_rho, num_grids, spin_channel, vxc_screen_threshold);
            num_active = active_mask.iter().filter(|&&x| x).count();
            use_screening = num_active > 0 && num_active < num_grids * 3 / 4;
        }

        if print_level >= 1 && use_screening {
            let ratio = num_active as f64 / num_grids as f64 * 100.0;
            println!(" [VXC-screen(coeff)]  active grids: {}/{} ({:.1}%), saved ~{:.0}%",
                num_active, num_grids, ratio, 100.0 - ratio);
        }

        if use_screening {
            // === Screened path ===
            let n_g = num_active;
            let loc_rho_c = Self::compact_rho(&loc_rho, &active_mask, num_active, spin_channel);
            let loc_rhop_c = Self::compact_rhop(&loc_rhop, &active_mask, num_active, spin_channel);
            let loc_sigma_c = if self.use_density_gradient() {
                Self::compact_sigma(&loc_sigma, &active_mask, num_active,
                    if spin_channel==1 {1} else {3})
            } else { MatrixFull::empty() };
            let loc_lapl_c = if self.use_kinetic_density() {
                Self::compact_vrho(&loc_lapl, &active_mask, num_active, spin_channel)
            } else { MatrixFull::empty() };
            let loc_tau_c = if self.use_kinetic_density() {
                Self::compact_tau(&loc_tau, &active_mask, num_active, spin_channel)
            } else { MatrixFull::empty() };
            let mut loc_exc_c = MatrixFull::new([num_active, 1], 0.0);

            let mut loc_vrho_c = MatrixFull::new([n_g, spin_channel], 0.0);
            let mut loc_vsigma_c = if self.use_density_gradient() && spin_channel==1 {
                MatrixFull::new([n_g, 1], 0.0)
            } else if self.use_density_gradient() && spin_channel==2 {
                MatrixFull::new([n_g, 3], 0.0)
            } else { MatrixFull::empty() };
            let mut loc_vtau_c = if self.use_kinetic_density() {
                MatrixFull::new([n_g, spin_channel], 0.0)
            } else { MatrixFull::empty() };

            // libxc on compact arrays
            self.dfa_compnt_scf.iter().zip(self.dfa_paramr_scf.iter()).for_each(|(xc_func,xc_para)| {
                let xc_func = self.init_libxc_and_set_param(xc_func);
                match xc_func.family() {
                    LibXCFamily::LDA => {
                        if spin_channel==1 {
                            let (tmp_exc,tmp_vrho) = lda_exc_vxc(&xc_func,loc_rho_c.data_ref().unwrap());
                            let tmp_exc = MatrixFull::from_vec([n_g,1],tmp_exc).unwrap();
                            let tmp_vrho = MatrixFull::from_vec([n_g,1],tmp_vrho).unwrap();
                            loc_exc_c.self_scaled_add(&tmp_exc,*xc_para);
                            loc_vrho_c.self_scaled_add(&tmp_vrho,*xc_para);
                        } else {
                            let (tmp_exc,tmp_vrho) = lda_exc_vxc(&xc_func,loc_rho_c.transpose().data_ref().unwrap());
                            let tmp_exc = MatrixFull::from_vec([n_g,1],tmp_exc).unwrap();
                            let tmp_vrho = MatrixFull::from_vec([2,n_g],tmp_vrho).unwrap();
                            loc_exc_c.self_scaled_add(&tmp_exc,*xc_para);
                            loc_vrho_c.self_scaled_add(&tmp_vrho.transpose_and_drop(),*xc_para);
                        }
                    },
                    LibXCFamily::GGA | LibXCFamily::HybGGA => {
                        if spin_channel==1 {
                            let (tmp_exc,tmp_vrho, tmp_vsigma) = gga_exc_vxc(&xc_func,loc_rho_c.data_ref().unwrap(),loc_sigma_c.data_ref().unwrap());
                            let tmp_exc = MatrixFull::from_vec([n_g,1],tmp_exc).unwrap();
                            let tmp_vrho = MatrixFull::from_vec([n_g,1],tmp_vrho).unwrap();
                            let tmp_vsigma= MatrixFull::from_vec([n_g,1],tmp_vsigma).unwrap();
                            loc_exc_c.self_scaled_add(&tmp_exc,*xc_para);
                            loc_vrho_c.self_scaled_add(&tmp_vrho,*xc_para);
                            loc_vsigma_c.self_scaled_add(&tmp_vsigma, *xc_para);
                        } else {
                            let (tmp_exc,tmp_vrho, tmp_vsigma) = gga_exc_vxc(&xc_func,loc_rho_c.transpose().data_ref().unwrap(),loc_sigma_c.transpose().data_ref().unwrap());
                            let tmp_exc = MatrixFull::from_vec([n_g,1],tmp_exc).unwrap();
                            let tmp_vrho = MatrixFull::from_vec([2,n_g],tmp_vrho).unwrap();
                            let tmp_vsigma= MatrixFull::from_vec([3,n_g],tmp_vsigma).unwrap();
                            loc_exc_c.self_scaled_add(&tmp_exc,*xc_para);
                            loc_vrho_c.self_scaled_add(&tmp_vrho.transpose_and_drop(),*xc_para);
                            loc_vsigma_c.self_scaled_add(&tmp_vsigma.transpose_and_drop(), *xc_para);
                        }
                    },
                    LibXCFamily::MGGA | LibXCFamily::HybMGGA => {
                        if spin_channel==1 {
                            let (tmp_exc,tmp_vrho,tmp_vsigma,tmp_valpl,tmp_vtau)
                                = mgga_exc_vxc(&xc_func, loc_rho_c.data_ref().unwrap(), loc_sigma_c.data_ref().unwrap(),
                                    loc_lapl_c.data_ref().unwrap(), loc_tau_c.data_ref().unwrap());
                            let tmp_exc = MatrixFull::from_vec([n_g,1],tmp_exc).unwrap();
                            let tmp_vrho = MatrixFull::from_vec([n_g,1],tmp_vrho).unwrap();
                            let tmp_vsigma= MatrixFull::from_vec([n_g,1],tmp_vsigma).unwrap();
                            let tmp_vtau = MatrixFull::from_vec([n_g,1],tmp_vtau).unwrap();
                            loc_exc_c.self_scaled_add(&tmp_exc,*xc_para);
                            loc_vrho_c.self_scaled_add(&tmp_vrho,*xc_para);
                            loc_vsigma_c.self_scaled_add(&tmp_vsigma, *xc_para);
                            loc_vtau_c.self_scaled_add(&tmp_vtau, *xc_para);
                        } else {
                            let (tmp_exc,tmp_vrho,tmp_vsigma,tmp_valpl,tmp_vtau)
                                = mgga_exc_vxc(&xc_func, loc_rho_c.transpose().data_ref().unwrap(), loc_sigma_c.transpose().data_ref().unwrap(),
                                    loc_lapl_c.transpose().data_ref().unwrap(), loc_tau_c.transpose().data_ref().unwrap());
                            let tmp_exc = MatrixFull::from_vec([n_g,1],tmp_exc).unwrap();
                            let tmp_vrho = MatrixFull::from_vec([2,n_g],tmp_vrho).unwrap();
                            let tmp_vsigma= MatrixFull::from_vec([3,n_g],tmp_vsigma).unwrap();
                            let tmp_vtau = MatrixFull::from_vec([2,n_g],tmp_vtau).unwrap();
                            loc_exc_c.self_scaled_add(&tmp_exc,*xc_para);
                            loc_vrho_c.self_scaled_add(&tmp_vrho.transpose_and_drop(),*xc_para);
                            loc_vsigma_c.self_scaled_add(&tmp_vsigma.transpose_and_drop(), *xc_para);
                            loc_vtau_c.self_scaled_add(&tmp_vtau.transpose_and_drop(), *xc_para);
                        }
                    },
                    xc_family => panic!("{xc_family:?} is not yet implemented"),
                }
            });

            let ao_dense_owned2: Option<MatrixFull<f64>>;
            let ao_ref2: &MatrixFull<f64> = match &grids.ao {
                Some(a) => { ao_dense_owned2 = None; a }
                None => {
                    let c = grids.ao_compressed.as_ref().unwrap();
                    ao_dense_owned2 = Some(Grids::decompress_ao(c));
                    ao_dense_owned2.as_ref().unwrap()
                }
            };
            let ao_active = Self::extract_active_ao_columns(
                ao_ref2, &range_grids, &active_mask, num_active, num_basis);

            let aop_dense_owned2: Option<RIFull<f64>>;
            let aop_ref2: Option<&RIFull<f64>> = match &grids.aop {
                Some(a) => { aop_dense_owned2 = None; Some(a) }
                None => match &grids.aop_compressed {
                    Some(c) => {
                        aop_dense_owned2 = Some(Grids::decompress_aop(c));
                        Some(aop_dense_owned2.as_ref().unwrap())
                    }
                    None => { aop_dense_owned2 = None; None }
                }
            };
            let aop_active: Option<RIFull<f64>> = aop_ref2.map(|aop| {
                Self::extract_active_aop_columns(aop, &range_grids, &active_mask, num_active, num_basis)
            });
            let weights_active = Self::compact_weights(&grids.weights[range_grids.clone()], &active_mask, num_active);

            let mut loc_vxc_ao = vec![MatrixFull::new([num_basis, n_g], 0.0); spin_channel];
            let mut loc_vxc_ao_1 = if self.use_kinetic_density() {
                vec![MatrixFull::new([num_basis, n_g], 0.0); spin_channel]
            } else { vec![] };
            let mut loc_vxc_mat = vec![MatrixFull::new([num_basis, num_basis], 0.0); spin_channel];

            for i_spin in 0..spin_channel {
                let mut loc_vxc_ao_s = &mut loc_vxc_ao[i_spin];
                let loc_vrho_s = loc_vrho_c.slice_column(i_spin);
                let loc_ao_ref = ao_active.to_matrixfullslice_columns(0..n_g);
                contract_vxc_0_serial(loc_vxc_ao_s, &loc_ao_ref, loc_vrho_s, None);
            }
            if self.use_density_gradient() {
                if let Some(ref aop_active) = aop_active {
                    if spin_channel == 1 {
                        let mut loc_vxc_ao_s = &mut loc_vxc_ao[0];
                        let loc_vsigma_s = loc_vsigma_c.slice_column(0);
                        let loc_rhop_s = loc_rhop_c.get_reducing_matrix(0).unwrap();
                        let mut loc_wao = MatrixFull::new([num_basis, n_g], 0.0);
                        for x in 0usize..3usize {
                            let loc_aop_x = aop_active.get_reducing_matrix_columns(0..n_g, x).unwrap();
                            let loc_rhop_s_x = loc_rhop_s.get_slice_x(x);
                            contract_vxc_0_serial(&mut loc_wao, &loc_aop_x, loc_rhop_s_x, None);
                        }
                        contract_vxc_0_serial(loc_vxc_ao_s, &loc_wao.to_matrixfullslice(), loc_vsigma_s, Some(4.0));
                    } else {
                        {
                            let mut loc_vxc_ao_a = &mut loc_vxc_ao[0];
                            let loc_rhop_a = loc_rhop_c.get_reducing_matrix(0).unwrap();
                            let loc_vsigma_uu = loc_vsigma_c.slice_column(0);
                            let mut loc_dao = MatrixFull::new([num_basis, n_g], 0.0);
                            for x in 0usize..3usize {
                                let loc_aop_x = aop_active.get_reducing_matrix_columns(0..n_g, x).unwrap();
                                let loc_rhop_s_x = loc_rhop_a.get_slice_x(x);
                                contract_vxc_0_serial(&mut loc_dao, &loc_aop_x, loc_rhop_s_x, None);
                            }
                            contract_vxc_0_serial(loc_vxc_ao_a, &loc_dao.to_matrixfullslice(), &loc_vsigma_uu, Some(4.0));
                            let loc_rhop_b = loc_rhop_c.get_reducing_matrix(1).unwrap();
                            let loc_vsigma_ud = loc_vsigma_c.slice_column(1);
                            loc_dao.data.iter_mut().for_each(|d| {*d=0.0});
                            for x in 0usize..3usize {
                                let loc_aop_x = aop_active.get_reducing_matrix_columns(0..n_g, x).unwrap();
                                let loc_rhop_s_x = loc_rhop_b.get_slice_x(x);
                                contract_vxc_0_serial(&mut loc_dao, &loc_aop_x, loc_rhop_s_x, None);
                            }
                            contract_vxc_0_serial(loc_vxc_ao_a, &loc_dao.to_matrixfullslice(), &loc_vsigma_ud, Some(2.0));
                        }
                        {
                            let mut loc_vxc_ao_b = &mut loc_vxc_ao[1];
                            let loc_rhop_b = loc_rhop_c.get_reducing_matrix(1).unwrap();
                            let loc_vsigma_dd = loc_vsigma_c.slice_column(2);
                            let mut loc_dao = MatrixFull::new([num_basis, n_g], 0.0);
                            for x in 0usize..3usize {
                                let loc_aop_x = aop_active.get_reducing_matrix_columns(0..n_g, x).unwrap();
                                let loc_rhop_s_x = loc_rhop_b.get_slice_x(x);
                                contract_vxc_0_serial(&mut loc_dao, &loc_aop_x, loc_rhop_s_x, None);
                            }
                            contract_vxc_0_serial(loc_vxc_ao_b, &loc_dao.to_matrixfullslice(), &loc_vsigma_dd, Some(4.0));
                            let loc_rhop_a = loc_rhop_c.get_reducing_matrix(0).unwrap();
                            let loc_vsigma_ud = loc_vsigma_c.slice_column(1);
                            loc_dao.data.iter_mut().for_each(|d| {*d=0.0});
                            for x in 0usize..3usize {
                                let loc_aop_x = aop_active.get_reducing_matrix_columns(0..n_g, x).unwrap();
                                let loc_rhop_s_x = loc_rhop_a.get_slice_x(x);
                                contract_vxc_0_serial(&mut loc_dao, &loc_aop_x, loc_rhop_s_x, None);
                            }
                            contract_vxc_0_serial(loc_vxc_ao_b, &loc_dao.to_matrixfullslice(), &loc_vsigma_ud, Some(2.0));
                        }
                    }
                }
            }
            for i_spin in 0..spin_channel {
                let mut loc_vxc_mat_s = loc_vxc_mat.get_mut(i_spin).unwrap();
                let mut loc_vxc_ao_s = loc_vxc_ao.get_mut(i_spin).unwrap();
                loc_vxc_ao_s.iter_columns_full_mut().zip(weights_active.iter()).for_each(|(vxc_ao_s,w)| {
                    vxc_ao_s.iter_mut().for_each(|f| {*f *= *w})
                });
                _dgemm(
                    &ao_active, (0..num_basis, 0..n_g), 'N',
                    loc_vxc_ao_s, (0..num_basis, 0..n_g), 'T',
                    loc_vxc_mat_s, (0..num_basis, 0..num_basis),
                    1.0, 0.0
                );
            }
            if self.use_kinetic_density() {
                if let Some(ref aop_active) = aop_active {
                    for i_spin in 0..spin_channel {
                        let loc_vxc_mat_s = loc_vxc_mat.get_mut(i_spin).unwrap();
                        let mut loc_vtau_s = loc_vtau_c.slice_column_mut(i_spin);
                        let mut loc_vxc_ao_1_s = &mut loc_vxc_ao_1[i_spin];
                        loc_vtau_s.iter_mut().zip(weights_active.iter()).for_each(
                            |(vtau_s, w)| {*vtau_s *= *w}
                        );
                        for ic in 0usize..3usize {
                            let loc_aop_ic = aop_active.get_reducing_matrix_columns(0..n_g, ic).unwrap();
                            contract_vxc_0_serial(loc_vxc_ao_1_s, &loc_aop_ic, loc_vtau_s, Some(0.5));
                            _dgemm(
                                &loc_aop_ic, (0..num_basis, 0..n_g), 'N',
                                loc_vxc_ao_1_s, (0..num_basis, 0..n_g), 'T',
                                loc_vxc_mat_s, (0..num_basis, 0..num_basis), 1.0, 1.0
                            );
                            loc_vxc_ao_1_s.data.iter_mut().for_each(|t| {*t=0.0});
                        }
                    }
                }
            }
            let (loc_exc_total, loc_total_elec) = self.integrate_exc(&loc_exc_c, &loc_rho_c, &weights_active, spin_channel);
            (loc_exc_total, loc_vxc_mat, loc_total_elec)
        } else {
            // === Original path (no screening) ===
            let loc_weights = &grids.weights[range_grids.clone()];

            let mut loc_exc = MatrixFull::new([num_grids, 1], 0.0);
            let mut loc_exc_total = vec![0.0; spin_channel];
            let mut loc_vxc_mat = vec![MatrixFull::new([num_basis, num_basis], 0.0); spin_channel];
            let mut loc_vxc_ao = vec![MatrixFull::new([num_basis, num_grids], 0.0); spin_channel];
            let mut loc_vxc_ao_1 = if self.use_kinetic_density() {
                vec![MatrixFull::new([num_basis, num_grids], 0.0); spin_channel]
            } else { vec![] };

            let mut loc_vrho = MatrixFull::new([num_grids, spin_channel], 0.0);
            let mut loc_vsigma = if self.use_density_gradient() && spin_channel==1 {
                MatrixFull::new([num_grids, 1], 0.0)
            } else if self.use_density_gradient() && spin_channel==2 {
                MatrixFull::new([num_grids, 3], 0.0)
            } else { MatrixFull::empty() };
            let mut loc_vtau = if self.use_kinetic_density() {
                MatrixFull::new([num_grids, spin_channel], 0.0)
            } else { MatrixFull::empty() };

            self.dfa_compnt_scf.iter().zip(self.dfa_paramr_scf.iter()).for_each(|(xc_func,xc_para)| {
                let xc_func = self.init_libxc_and_set_param(xc_func);
                match xc_func.family() {
                    LibXCFamily::LDA => {
                        if spin_channel==1 {
                            let (tmp_exc,tmp_vrho) = lda_exc_vxc(&xc_func,loc_rho.data_ref().unwrap());
                            let tmp_exc = MatrixFull::from_vec([num_grids,1],tmp_exc).unwrap();
                            let tmp_vrho = MatrixFull::from_vec([num_grids,1],tmp_vrho).unwrap();
                            loc_exc.self_scaled_add(&tmp_exc,*xc_para);
                            loc_vrho.self_scaled_add(&tmp_vrho,*xc_para);
                        } else {
                            let (tmp_exc,tmp_vrho) = lda_exc_vxc(&xc_func,loc_rho.transpose().data_ref().unwrap());
                            let tmp_exc = MatrixFull::from_vec([num_grids,1],tmp_exc).unwrap();
                            let tmp_vrho = MatrixFull::from_vec([2,num_grids],tmp_vrho).unwrap();
                            loc_exc.self_scaled_add(&tmp_exc,*xc_para);
                            loc_vrho.self_scaled_add(&tmp_vrho.transpose_and_drop(),*xc_para);
                        }
                    },
                    LibXCFamily::GGA | LibXCFamily::HybGGA => {
                        if spin_channel==1 {
                            let (tmp_exc,tmp_vrho, tmp_vsigma) = gga_exc_vxc(&xc_func,loc_rho.data_ref().unwrap(),loc_sigma.data_ref().unwrap());
                            let tmp_exc = MatrixFull::from_vec([num_grids,1],tmp_exc).unwrap();
                            let tmp_vrho = MatrixFull::from_vec([num_grids,1],tmp_vrho).unwrap();
                            let tmp_vsigma= MatrixFull::from_vec([num_grids,1],tmp_vsigma).unwrap();
                            loc_exc.self_scaled_add(&tmp_exc,*xc_para);
                            loc_vrho.self_scaled_add(&tmp_vrho,*xc_para);
                            loc_vsigma.self_scaled_add(&tmp_vsigma, *xc_para);
                        } else {
                            let (tmp_exc,tmp_vrho, tmp_vsigma) = gga_exc_vxc(&xc_func,loc_rho.transpose().data_ref().unwrap(),loc_sigma.transpose().data_ref().unwrap());
                            let tmp_exc = MatrixFull::from_vec([num_grids,1],tmp_exc).unwrap();
                            let tmp_vrho = MatrixFull::from_vec([2,num_grids],tmp_vrho).unwrap();
                            let tmp_vsigma= MatrixFull::from_vec([3,num_grids],tmp_vsigma).unwrap();
                            loc_exc.self_scaled_add(&tmp_exc,*xc_para);
                            loc_vrho.self_scaled_add(&tmp_vrho.transpose_and_drop(),*xc_para);
                            loc_vsigma.self_scaled_add(&tmp_vsigma.transpose_and_drop(), *xc_para);
                        }
                    },
                    LibXCFamily::MGGA | LibXCFamily::HybMGGA => {
                        if spin_channel==1 {
                            let (tmp_exc,tmp_vrho,tmp_vsigma,tmp_valpl,tmp_vtau)
                                = mgga_exc_vxc(&xc_func, loc_rho.data_ref().unwrap(), loc_sigma.data_ref().unwrap(),
                                    loc_lapl.data_ref().unwrap(), loc_tau.data_ref().unwrap());
                            let tmp_exc = MatrixFull::from_vec([num_grids,1],tmp_exc).unwrap();
                            let tmp_vrho = MatrixFull::from_vec([num_grids,1],tmp_vrho).unwrap();
                            let tmp_vsigma= MatrixFull::from_vec([num_grids,1],tmp_vsigma).unwrap();
                            let tmp_vtau = MatrixFull::from_vec([num_grids,1],tmp_vtau).unwrap();
                            loc_exc.self_scaled_add(&tmp_exc,*xc_para);
                            loc_vrho.self_scaled_add(&tmp_vrho,*xc_para);
                            loc_vsigma.self_scaled_add(&tmp_vsigma, *xc_para);
                            loc_vtau.self_scaled_add(&tmp_vtau, *xc_para);
                        } else {
                            let (tmp_exc,tmp_vrho,tmp_vsigma,tmp_valpl,tmp_vtau)
                                = mgga_exc_vxc(&xc_func, loc_rho.transpose().data_ref().unwrap(), loc_sigma.transpose().data_ref().unwrap(),
                                    loc_lapl.transpose().data_ref().unwrap(), loc_tau.transpose().data_ref().unwrap());
                            let tmp_exc = MatrixFull::from_vec([num_grids,1],tmp_exc).unwrap();
                            let tmp_vrho = MatrixFull::from_vec([2,num_grids],tmp_vrho).unwrap();
                            let tmp_vsigma= MatrixFull::from_vec([3,num_grids],tmp_vsigma).unwrap();
                            let tmp_vtau = MatrixFull::from_vec([2,num_grids],tmp_vtau).unwrap();
                            loc_exc.self_scaled_add(&tmp_exc,*xc_para);
                            loc_vrho.self_scaled_add(&tmp_vrho.transpose_and_drop(),*xc_para);
                            loc_vsigma.self_scaled_add(&tmp_vsigma.transpose_and_drop(), *xc_para);
                            loc_vtau.self_scaled_add(&tmp_vtau.transpose_and_drop(), *xc_para);
                        }
                    },
                    xc_family => panic!("{xc_family:?} is not yet implemented"),
                }
            });

            if let Some(_ao_c) = &grids.ao_compressed {
                grids.contract_response_compressed(
                    &range_grids, &loc_vrho, &loc_vsigma, &loc_vtau,
                    loc_weights, &mut loc_vxc_mat, spin_channel,
                    self.use_density_gradient(), self.use_kinetic_density(),
                    &loc_rhop, num_basis,
                );
            } else if let Some(ao) = &grids.ao {
                for i_spin in 0..spin_channel {
                    let mut loc_vxc_ao_s = &mut loc_vxc_ao[i_spin];
                    let loc_vrho_s = loc_vrho.slice_column(i_spin);
                    let loc_ao_ref = ao.to_matrixfullslice_columns(range_grids.clone());
                    contract_vxc_0_serial(loc_vxc_ao_s, &loc_ao_ref, loc_vrho_s, None);
                }
                if self.use_density_gradient() {
                    if let Some(aop) = &grids.aop {
                        if spin_channel == 1 {
                            let mut loc_vxc_ao_s = &mut loc_vxc_ao[0];
                            let loc_vsigma_s = loc_vsigma.slice_column(0);
                            let loc_rhop_s = loc_rhop.get_reducing_matrix(0).unwrap();
                            let mut loc_wao = MatrixFull::new([num_basis, num_grids], 0.0);
                            for x in 0usize..3usize {
                                let loc_aop_x = aop.get_reducing_matrix_columns(range_grids.clone(), x).unwrap();
                                let loc_rhop_s_x = loc_rhop_s.get_slice_x(x);
                                contract_vxc_0_serial(&mut loc_wao, &loc_aop_x, loc_rhop_s_x, None);
                            }
                            contract_vxc_0_serial(loc_vxc_ao_s, &loc_wao.to_matrixfullslice(), loc_vsigma_s, Some(4.0));
                        } else {
                            {
                                let mut loc_vxc_ao_a = &mut loc_vxc_ao[0];
                                let loc_rhop_a = loc_rhop.get_reducing_matrix(0).unwrap();
                                let loc_vsigma_uu = loc_vsigma.slice_column(0);
                                let mut loc_dao = MatrixFull::new([num_basis, num_grids], 0.0);
                                for x in 0usize..3usize {
                                    let loc_aop_x = aop.get_reducing_matrix_columns(range_grids.clone(), x).unwrap();
                                    let loc_rhop_s_x = loc_rhop_a.get_slice_x(x);
                                    contract_vxc_0_serial(&mut loc_dao, &loc_aop_x, loc_rhop_s_x, None);
                                }
                                contract_vxc_0_serial(loc_vxc_ao_a, &loc_dao.to_matrixfullslice(), &loc_vsigma_uu, Some(4.0));
                                let loc_rhop_b = loc_rhop.get_reducing_matrix(1).unwrap();
                                let loc_vsigma_ud = loc_vsigma.slice_column(1);
                                loc_dao.data.iter_mut().for_each(|d| {*d=0.0});
                                for x in 0usize..3usize {
                                    let loc_aop_x = aop.get_reducing_matrix_columns(range_grids.clone(), x).unwrap();
                                    let loc_rhop_s_x = loc_rhop_b.get_slice_x(x);
                                    contract_vxc_0_serial(&mut loc_dao, &loc_aop_x, loc_rhop_s_x, None);
                                }
                                contract_vxc_0_serial(loc_vxc_ao_a, &loc_dao.to_matrixfullslice(), &loc_vsigma_ud, Some(2.0));
                            }
                            {
                                let mut loc_vxc_ao_b = &mut loc_vxc_ao[1];
                                let loc_rhop_b = loc_rhop.get_reducing_matrix(1).unwrap();
                                let loc_vsigma_dd = loc_vsigma.slice_column(2);
                                let mut loc_dao = MatrixFull::new([num_basis, num_grids], 0.0);
                                for x in 0usize..3usize {
                                    let loc_aop_x = aop.get_reducing_matrix_columns(range_grids.clone(), x).unwrap();
                                    let loc_rhop_s_x = loc_rhop_b.get_slice_x(x);
                                    contract_vxc_0_serial(&mut loc_dao, &loc_aop_x, loc_rhop_s_x, None);
                                }
                                contract_vxc_0_serial(loc_vxc_ao_b, &loc_dao.to_matrixfullslice(), &loc_vsigma_dd, Some(4.0));
                                let loc_rhop_a = loc_rhop.get_reducing_matrix(0).unwrap();
                                let loc_vsigma_ud = loc_vsigma.slice_column(1);
                                loc_dao.data.iter_mut().for_each(|d| {*d=0.0});
                                for x in 0usize..3usize {
                                    let loc_aop_x = aop.get_reducing_matrix_columns(range_grids.clone(), x).unwrap();
                                    let loc_rhop_s_x = loc_rhop_a.get_slice_x(x);
                                    contract_vxc_0_serial(&mut loc_dao, &loc_aop_x, loc_rhop_s_x, None);
                                }
                                contract_vxc_0_serial(loc_vxc_ao_b, &loc_dao.to_matrixfullslice(), &loc_vsigma_ud, Some(2.0));
                            }
                        }
                    }
                }
                for i_spin in 0..spin_channel {
                    let mut loc_vxc_mat_s = loc_vxc_mat.get_mut(i_spin).unwrap();
                    let mut loc_vxc_ao_s = loc_vxc_ao.get_mut(i_spin).unwrap();
                    loc_vxc_ao_s.iter_columns_full_mut().zip(loc_weights.iter()).for_each(|(vxc_ao_s,w)| {
                        vxc_ao_s.iter_mut().for_each(|f| {*f *= *w})
                    });
                    _dgemm(
                        ao, (0..num_basis, range_grids.clone()), 'N',
                        loc_vxc_ao_s, (0..num_basis, 0..range_grids.len()), 'T',
                        loc_vxc_mat_s, (0..num_basis, 0..num_basis),
                        1.0, 0.0
                    );
                }
                if self.use_kinetic_density() {
                    let Some(aop) = &grids.aop else {
                        panic!("aop is not available");
                    };
                    for i_spin in 0..spin_channel {
                        let loc_vxc_mat_s = loc_vxc_mat.get_mut(i_spin).unwrap();
                        let mut loc_vtau_s = loc_vtau.slice_column_mut(i_spin);
                        let mut loc_vxc_ao_1_s = &mut loc_vxc_ao_1[i_spin];
                        loc_vtau_s.iter_mut().zip(loc_weights.iter()).for_each(
                            |(vtau_s, w)| {*vtau_s *= *w}
                        );
                        for ic in 0usize..3usize {
                            let loc_aop_ic = aop.get_reducing_matrix_columns(range_grids.clone(), ic).unwrap();
                            contract_vxc_0_serial(loc_vxc_ao_1_s, &loc_aop_ic, loc_vtau_s, Some(0.5));
                            _dgemm(
                                &loc_aop_ic, (0..num_basis, 0..range_grids.len()), 'N',
                                loc_vxc_ao_1_s, (0..num_basis, 0..range_grids.len()), 'T',
                                loc_vxc_mat_s, (0..num_basis, 0..num_basis), 1.0, 1.0
                            );
                            loc_vxc_ao_1_s.data.iter_mut().for_each(|t| {*t=0.0});
                        }
                    }
                }
            }
            let (loc_exc_total, loc_total_elec) = self.integrate_exc(&loc_exc, &loc_rho, loc_weights, spin_channel);
            (loc_exc_total, loc_vxc_mat, loc_total_elec)
        }
    }

    pub fn post_xc_exc(&self, post_xc: &Vec<String>, grids: &crate::dft::Grids, dm: &Vec<MatrixFull<f64>>, mo: &[MatrixFull<f64>;2], occ: &[Vec<f64>;2]) 
    -> Vec<[f64;2]> {
        let mut post_xc_energy:Vec<[f64;2]>=vec![];
        let spin_channel = self.spin_channel;
        let num_grids = grids.coordinates.len();
        let num_basis = dm[0].size[0];
        let dt0 = utilities::init_timing();
        let use_density_gradient = post_xc.iter().fold(false,|flag, x| {
            let code = DFA4REST::xc_func_init_fdqc(x,spin_channel);
            let x_flag = code.iter().fold(false, |flag, xc_code| {
                let xc_func = self.init_libxc(xc_code);
                flag || !matches!(xc_func.family(), LibXCFamily::LDA | LibXCFamily::HybLDA)
            });
            flag || x_flag
        });
        let (rho, rhop, sigma, _lapl, _tau) = self.prepare_dft_quantities(
            grids,
            spin_channel,
            mo,
            occ,
            use_density_gradient,
        );
        let _dt2 = utilities::timing(&dt0, Some("evaluate rho/rhop/sigma"));
        //let (rho,rhop) = grids.prepare_tabulated_density(dm, spin_channel);
        post_xc.iter().for_each(|x| {
            let mut exc = MatrixFull::new([num_grids,1],0.0);
            let mut exc_total = [0.0, 0.0];
            let code = DFA4REST::xc_func_init_fdqc(x,spin_channel);
            //println!("debug xc_code: {:?}", &code);
            code.iter().for_each(|xc_code| {
                exc.par_self_scaled_add(&self.xc_exc_code(xc_code, &rho, &sigma, spin_channel),1.0);
            });

            let (exc_total_vec, _) = self.integrate_exc(&exc, &rho, &grids.weights, spin_channel);
            exc_total[0] = exc_total_vec[0];
            if spin_channel == 2 {
                exc_total[1] = exc_total_vec[1];
            }
            //println!("exc_total: {:?}", &exc_total);

            post_xc_energy.push(exc_total);
        });

        post_xc_energy

    }

    pub fn post_tabulated_exc(&self, 
        grids: &crate::dft::Grids, 
        dm: &Vec<MatrixFull<f64>>, 
        mo: &[MatrixFull<f64>;2], 
        occ: &[Vec<f64>;2]) 
    {
        use std::io::Write;
        let dt0 = utilities::init_timing();
        let spin_channel = self.spin_channel;
        let num_grids = grids.coordinates.len();
        let num_basis = dm[0].size[0];
        let use_density_gradient = self.use_density_gradient();

        //// 获取rayon的当前线程的ID
        //let id = rayon::current_thread_index().unwrap_or_default();
        //// 创建一个文件，包含当前线程的ID的信息
        //let mut file = File::create(format!("debug_tabulated_exc_{}.txt", id)).unwrap();


        let mut str_lines: Vec<String> = vec![String::new();num_grids+1];
        //let mut str_lines_beta: Vec<String> = if spin_channel == 2 {
        //    vec![String::new();num_grids]
        //} else {
        //    vec![]
        //};

        let mut post_xc_energy:Vec<[f64;2]>=vec![];

        if spin_channel == 1 { 
            str_lines[0].push_str(&format!("{:>20} " , "rho"));
        } else {
            str_lines[0].push_str(&format!("{:>20}" , "rho_alpha"));
            str_lines[0].push_str(&format!("{:>20} ", "rho_beta"));
        };

        if use_density_gradient {
            if spin_channel == 1 {
                str_lines[0].push_str(&format!("{:>20}", "rhop_x"));
                str_lines[0].push_str(&format!("{:>20}", "rhop_y"));
                str_lines[0].push_str(&format!("{:>20}", "rhop_z"));
                str_lines[0].push_str(&format!("{:>20}", "sigma"));
            } else {
                str_lines[0].push_str(&format!("{:>20}","rhop_alpha_x"));
                str_lines[0].push_str(&format!("{:>20}","rhop_alpha_y"));
                str_lines[0].push_str(&format!("{:>20}","rhop_alpha_z"));
                str_lines[0].push_str(&format!("{:>20}","rhop_beta_x"));
                str_lines[0].push_str(&format!("{:>20}","rhop_beta_y"));
                str_lines[0].push_str(&format!("{:>20}","rhop_beta_z"));
                str_lines[0].push_str(&format!("{:>20}","sigma_aa"));
                str_lines[0].push_str(&format!("{:>20}","sigma_ab"));
                str_lines[0].push_str(&format!("{:>20}","sigma_bb"));
            }
        }

        self.dfa_compnt_scf.iter().enumerate().for_each(|(i_xc, xc)| { 
            //if spin_channel == 1 {
                str_lines[0].push_str(&format!("{:>20} ", libxc::util::libxc_functional_get_name(*xc as i32).unwrap_or_else(|| "Unknown_XC".to_string())));
            //} else {
            //    str_lines[0].push_str(&format!("{:>20}_alpha ", &code_to_name(*xc)));
            //    str_lines[0].push_str(&format!("{:>20}_beta ", &code_to_name(*xc)));
            //}
        });

        let (rho, rhop, sigma, _lapl, _tau) = self.prepare_dft_quantities(
            grids,
            spin_channel,
            mo,
            occ,
            use_density_gradient,
        );

        str_lines[1..].par_iter_mut().zip(rho.par_iter_column(0)).for_each(|(line,rho)| {
            line.push_str(&format!("{:20.10} ", rho));
        });
        if spin_channel == 2 {
            str_lines[1..].par_iter_mut().zip(rho.par_iter_column(1)).for_each(|(line,rho)| {
                line.push_str(&format!("{:20.10} ", rho));
            });
        }

        if use_density_gradient { 
            str_lines[1..].par_iter_mut().zip(rhop.par_iter_slices_x(0, 0)).for_each(|(line,rhop_i)| {
                line.push_str(&format!("{:20.10} ", rhop_i));
            });
            str_lines[1..].par_iter_mut().zip(rhop.par_iter_slices_x(1, 0)).for_each(|(line,rhop_i)| {
                line.push_str(&format!("{:20.10} ", rhop_i));
            });
            str_lines[1..].par_iter_mut().zip(rhop.par_iter_slices_x(2, 0)).for_each(|(line,rhop_i)| {
                line.push_str(&format!("{:20.10} ", rhop_i));
            });
            if spin_channel == 2 {
                str_lines[1..].par_iter_mut().zip(rhop.par_iter_slices_x(0, 1)).for_each(|(line,rhop_i)| {
                    line.push_str(&format!("{:20.10} ", rhop_i));
                });
                str_lines[1..].par_iter_mut().zip(rhop.par_iter_slices_x(1, 1)).for_each(|(line,rhop_i)| {
                    line.push_str(&format!("{:20.10} ", rhop_i));
                });
                str_lines[1..].par_iter_mut().zip(rhop.par_iter_slices_x(2, 1)).for_each(|(line,rhop_i)| {
                    line.push_str(&format!("{:20.10} ", rhop_i));
                });
            }
            if spin_channel == 1 {
                str_lines[1..].par_iter_mut().zip(sigma.par_iter_column(0)).for_each(|(line,sigma_i)| {
                    line.push_str(&format!("{:20.10} ", sigma_i));
                });
            } else if spin_channel == 2 {
                str_lines[1..].par_iter_mut().zip(sigma.par_iter_column(0)).for_each(|(line,sigma_i)| {
                    line.push_str(&format!("{:20.10} ", sigma_i));
                });
                str_lines[1..].par_iter_mut().zip(sigma.par_iter_column(1)).for_each(|(line,sigma_i)| {
                    line.push_str(&format!("{:20.10} ", sigma_i));
                });
                str_lines[1..].par_iter_mut().zip(sigma.par_iter_column(2)).for_each(|(line,sigma_i)| {
                    line.push_str(&format!("{:20.10} ", sigma_i));
                });
            }
        }
        self.dfa_compnt_scf.iter().enumerate().for_each(|(i_xc, xc)| { 
            //let mut exc = MatrixFull::new([num_grids,1],0.0);
            let exc = self.xc_exc_code(xc, &rho, &sigma, spin_channel);
            str_lines[1..].par_iter_mut().zip(exc.par_iter_column(0)).for_each(|(line,exc_i)| {
                line.push_str(&format!("{:20.10} ", exc_i));
            });
            if spin_channel == 2 {
                str_lines[1..].par_iter_mut().zip(exc.par_iter_column(1)).for_each(|(line,exc_i)| {
                    line.push_str(&format!("{:20.10} ", exc_i));
                });

            }
        });
        //将str_lines：Vec<String>，按照每一个element是一行的方式写入文件
        let mut file = File::create(format!("debug_tabulated_exc.txt")).unwrap();
        for line in str_lines.iter() {
            writeln!(file, "{}", line).expect("Unable to write file");
        }
        //file.close();

        //post_xc.iter().for_each(|x| {
        //    let mut exc = MatrixFull::new([num_grids,1],0.0);
        //    let mut exc_total =[0.0,0.0];
        //    let code = DFA4REST::xc_func_init_fdqc(x,spin_channel);
        //    //println!("debug xc_code: {:?}", &code);
        //    code.iter().for_each(|xc_code| {
        //        exc.par_self_scaled_add(&self.xc_exc_code(xc_code, &rho, &sigma, spin_channel),1.0);
        //    });

        //    for i_spin in 0..spin_channel {
        //        exc_total[i_spin] = izip!(exc.data.iter(),rho.iter_column(i_spin),grids.weights.iter())
        //            .fold(0.0,|acc,(exc,rho,weight)| {
        //                acc + exc * rho * weight
        //            });
        //    };
        //    //println!("exc_total: {:?}", &exc_total);

        //    post_xc_energy.push(exc_total);
        //});
    }

    pub fn xc_exc_list(&self, xc_code_list: &Vec<usize>, grids: &crate::dft::Grids, dm: &Vec<MatrixFull<f64>>, mo: &[MatrixFull<f64>;2], occ: &[Vec<f64>;2]) 
    -> Vec<[f64;2]> {
        let mut xc_energy:Vec<[f64;2]>=vec![];
        let spin_channel = self.spin_channel;
        let num_grids = grids.coordinates.len();
        let num_basis = dm[0].size[0];
        let dt0 = utilities::init_timing();
        let use_density_gradient = xc_code_list.iter().fold(false,|flag, xc_code| {
            let xc_func = self.init_libxc(xc_code);
            flag || !matches!(xc_func.family(), LibXCFamily::LDA | LibXCFamily::HybLDA)
        });
        let (rho, rhop, sigma, _lapl, _tau) = self.prepare_dft_quantities(
            grids,
            spin_channel,
            mo,
            occ,
            use_density_gradient,
        );
        let _dt2 = utilities::timing(&dt0, Some("evaluate rho/rhop/sigma"));
        //let (rho,rhop) = grids.prepare_tabulated_density(dm, spin_channel);
        xc_code_list.iter().for_each(|xc_code| {
            //let mut exc = MatrixFull::new([num_grids,1],0.0);
            let mut exc_total = [0.0, 0.0];
            let exc = self.xc_exc_code(xc_code, &rho, &sigma, spin_channel);
            //});

            let (exc_total_vec, _) = self.integrate_exc(&exc, &rho, &grids.weights, spin_channel);
            exc_total[0] = exc_total_vec[0];
            if spin_channel == 2 {
                exc_total[1] = exc_total_vec[1];
            }
            //println!("exc_total: {:?}", &exc_total);

            xc_energy.push(exc_total);
        });

        xc_energy

    }

    pub fn xc_exc(&self, grids: &mut Grids, spin_channel: usize, dm: &mut Vec<MatrixFull<f64>>, mo: &mut [MatrixFull<f64>;2], occ: &mut [Vec<f64>;2],iop: usize, mpi_operator: &Option<MPIOperator>) -> Vec<f64> {
        let num_grids = grids.coordinates.len();
        let num_basis = dm[0].size[0];
        let mut exc = MatrixFull::new([num_grids,1],0.0);
        let mut exc_total = vec![0.0;spin_channel];
        let dt0 = utilities::init_timing();
        let (rho, rhop, sigma, lapl, tau) = self.prepare_dft_quantities(
            grids,
            spin_channel,
            mo,
            occ,
            self.use_density_gradient(),
        );
        let dt2 = utilities::timing(&dt0, Some("evaluate rho/rhop/sigma"));

        if iop==0 {  // for the SCF energy
            //let rho_trans = if spin_channel== 1 {
            //    None
            //} else {
            //    Some(rho.transpose())
            //};
            self.dfa_compnt_scf.iter().zip(self.dfa_paramr_scf.iter()).for_each(|(xc_func,xc_para)| {
                let xc_func = self.init_libxc_and_set_param(xc_func);
                let tmp_exc = self.compute_exc_by_family(&xc_func, spin_channel, &rho, &sigma, &lapl, &tau);
                exc.par_self_scaled_add(&tmp_exc, *xc_para);
            });
        } else if iop==1 { // for the post-SCF energy calculation
            if let (Some(dfa_paramr),Some(dfa_compnt)) = (&self.dfa_paramr_pos, &self.dfa_compnt_pos) {
                dfa_compnt.iter().zip(dfa_paramr.iter()).for_each(|(xc_func,xc_para)| {
                    let xc_func = self.init_libxc_and_set_param(xc_func);
                    let tmp_exc = self.compute_exc_by_family(&xc_func, spin_channel, &rho, &sigma, &lapl, &tau);
                    exc.par_self_scaled_add(&tmp_exc, *xc_para);
                });
            }
        }
        let (exc_total, _total_elec) = self.integrate_exc(&exc, &rho, &grids.weights, spin_channel);
        #[cfg(feature = "mpi")]
        let global_exc_total = if let Some(mpi_op) = &mpi_operator {
            let my_rank = mpi_op.rank;
            let mut global_exc_total = mpi_reduce(&mpi_op.world, &exc_total , 0, &SystemOperation::sum());
            mpi_broadcast(&mpi_op.world, &mut global_exc_total, 0);
            
            global_exc_total
        } else {
            exc_total
        };
        #[cfg(not(feature = "mpi"))]
        let global_exc_total = exc_total;

        global_exc_total

    }

    pub fn xc_exc_code(&self, xc_code: &usize, rho: &MatrixFull<f64>, sigma:&MatrixFull<f64>, spin_channel: usize) -> MatrixFull<f64> {
        let xc_func = self.init_libxc_and_set_param(xc_code);
        let num_grids = rho.size()[0];
        self.compute_exc_by_family(&xc_func, spin_channel, rho, sigma, &MatrixFull::empty(), &MatrixFull::empty())
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
/// IGOR MARK HERE for unefficient use of  powf
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
/// IGOR MARK HERE for unefficient use of  powf
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

/// non0tab sparse mask table: for each grid batch, records which AO indices
/// have values above the cutoff threshold.
/// Compressed AO grid storage: per-batch dense sub-matrices with only
/// non-zero AO rows, plus index maps to global AO space.
#[derive(Debug, Clone)]
pub struct CompressedGridAO {
    /// Per-batch compressed AO values: [n_active_ao, n_batch_grids]
    pub batches: Vec<MatrixFull<f64>>,
    /// Per-batch: batch-local index → global AO index
    pub batch_ao_map: Vec<Vec<usize>>,
    /// Per-batch grid range in global grid space
    pub batch_grid_ranges: Vec<std::ops::Range<usize>>,
    pub blksize: usize,
    pub nao_total: usize,
    pub ngrids: usize,
}

/// Compressed AOP grid storage: same as CompressedGridAO but for 3 gradient
/// components (x, y, z). Stored as [x_batch, y_batch, z_batch] per batch.
#[derive(Debug, Clone)]
pub struct CompressedGridAOP {
    /// Per-batch compressed AOP values: 3 components × [n_active_ao, n_batch_grids]
    pub batches: Vec<[MatrixFull<f64>; 3]>,
    /// Per-batch: batch-local index → global AO index
    pub batch_aop_map: Vec<Vec<usize>>,
    /// Per-batch grid range in global grid space
    pub batch_grid_ranges: Vec<std::ops::Range<usize>>,
    pub blksize: usize,
    pub nao_total: usize,
    pub ngrids: usize,
}

#[derive(Debug, Clone)]
pub struct Non0Tab {
    /// Per-batch list of non-zero AO global indices
    pub batch_ao_indices: Vec<Vec<usize>>,
    /// Per-batch list of non-zero AO global indices for gradient (aop)
    pub batch_aop_indices: Vec<Vec<usize>>,
    /// Grid batch size (BLKSIZE)
    pub blksize: usize,
    /// Total number of grid points
    pub ngrids: usize,
    /// Total number of AO basis functions
    pub nao: usize,
    /// AO values below |ao_cutoff| are treated as zero
    pub ao_cutoff: f64,
    /// Average fraction of non-zero AOs per grid batch
    pub sparsity_ratio: f64,
    /// Total non-zero ao entries (across all batches)
    pub total_nonzero_ao: usize,
    /// Kept for compatibility: the AOP block is row-aligned with the AO block, so
    /// there is no separate gradient mask and this equals `total_stored_aop`.
    pub total_nonzero_aop: usize,
    /// AOP entries actually written into `CompressedGridAOP`: the block is stored row-aligned
    /// with the AO block, so it holds all three gradient components of every stored AO row, i.e.
    /// `3 * total_nonzero_ao`. This is the number that matches the memory footprint, and it is
    /// larger than `total_nonzero_aop` whenever the gradient mask is sparser in rows than the AO
    /// mask, which is the usual case for diffuse basis functions.
    pub total_stored_aop: usize,
    /// Total elements in dense ao (ngrids * nao)
    pub total_elements: usize,
}

#[derive(Clone)]
pub struct Grids {
    pub ao: Option<MatrixFull<f64>>,
    pub aop: Option<RIFull<f64>>,
    pub weights: Vec<f64>,
    pub coordinates: Vec<[f64;3]>,
    pub parallel_balancing: Vec<Range<usize>>,
    /// non0tab sparse mask (generated when ao_cutoff > 0.0)
    pub non0tab: Option<Non0Tab>,
    /// AO cutoff threshold for non0tab generation; 0.0 disables
    pub ao_cutoff: f64,
    /// Compressed AO storage (populated from non0tab when ao_cutoff > 0.0)
    pub ao_compressed: Option<CompressedGridAO>,
    /// Compressed AOP storage
    pub aop_compressed: Option<CompressedGridAOP>,
    /// Atomic index of the center each grid point was generated on;
    /// `usize::MAX` if the point is not associated with any atom (e.g. external grids)
    pub atm_idx: Vec<usize>,
    /// Quadrature weight (radial * angular) of each grid point, before Becke partitioning
    pub quadrature_weights: Vec<f64>,
}

impl Grids {
    pub fn build(mol: &mut Molecule) -> Grids {

        let mut global_grid = Grids {
            coordinates: Vec::new(),
            weights: Vec::new(),
            ao: None,
            aop: None,
            parallel_balancing: Vec::new(),
            non0tab: None,
            ao_cutoff: 0.0,
            ao_compressed: None,
            aop_compressed: None,
            atm_idx: Vec::new(),
            quadrature_weights: Vec::new(),
        };

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

            // external grids carry no per-atom information, and their weights are used
            // as they are (no Becke partitioning is applied)
            let mut atm_idx = vec![usize::MAX; weights.len()];
            let mut quadrature_weights = weights.clone();

            let num_threads = rayon::current_num_threads();
            utilities::apply_round_robin_permutation(&mut coordinates, &mut weights, &mut atm_idx, &mut quadrature_weights);
            let parallel_balancing = balancing(coordinates.len(), num_threads);

            global_grid = Grids {
                weights,
                coordinates,
                ao: None,
                aop: None,
                parallel_balancing,
                non0tab: None,
                ao_cutoff: 0.0,
                ao_compressed: None,
                aop_compressed: None,
                atm_idx,
                quadrature_weights,
            };
            return global_grid;


        }

        let dt0 = utilities::init_timing();

        let radial_precision = mol.ctrl.radial_precision;
        let min_num_angular_points: usize = mol.ctrl.min_num_angular_points;
        let max_num_angular_points: usize = mol.ctrl.max_num_angular_points;
        let hardness: usize = mol.ctrl.hardness;
        let pruning: String = mol.ctrl.pruning.clone();
        let grid_gen_level: usize = mol.ctrl.grid_gen_level;
        let rad_grid_method: String = mol.ctrl.rad_grid_method.clone();
        let radii_adjust = gen_grids::RadiiAdjust::from_str(&mol.ctrl.radii_adjust);

        // obtain system-dependent parameters
        //let mass_charge = get_mass_charge(&mol.geom.elem);
        //let mut proton_charges: Vec<i32> = mass_charge.iter().map(|value| value.1 as i32).collect();
        //if mol.geom.ghost_bs_elem.len() > 0 {
        //    proton_charges.append(&mut vec![1;mol.geom.ghost_bs_elem.len()])
        //}
        let mass_charge = get_mass_charge(&mol.geom.rg_elem);
        let proton_charges: Vec<i32> = mass_charge.iter().map(|value| value.1 as i32).collect();
        let center_coordinates_bohr = mol.geom.to_numgrid_io();
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
        let mut atm_idx: Vec<usize> = vec![];
        let mut quadrature_weights: Vec<f64> = vec![];

        alpha_min.iter().zip(alpha_max.iter()).enumerate().for_each(|(center_index,value)| {
            let (rs_atom, ws_atom, ws_quad_atom) = gen_grids::atom_grid_with_isdf(
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
                radii_adjust,
                mol.ctrl.use_isdf,
            );
            //println!("alpha_min: {:?}, alpha_max: {:6.3}",&value.0, &value.1);
            //println!("rs_atom: {:?}, ws_atom: {:?}",&rs_atom, &ws_atom);
            num_points += rs_atom.len();
            coordinates.extend(rs_atom.iter().map(|value| [value.0,value.1,value.2]));
            weights.extend(ws_atom);
            atm_idx.extend(vec![center_index; rs_atom.len()]);
            quadrature_weights.extend(ws_quad_atom);
        });

        utilities::timing(&dt0, Some("Generating the grids"));
        let num_threads = rayon::current_num_threads();
        utilities::apply_round_robin_permutation(&mut coordinates, &mut weights, &mut atm_idx, &mut quadrature_weights);
        let parallel_balancing = balancing(coordinates.len(), num_threads);
        global_grid = Grids {
            weights,
            coordinates,
            ao: None,
            aop: None,
            parallel_balancing,
            non0tab: None,
            ao_cutoff: 0.0,
            ao_compressed: None,
            aop_compressed: None,
            atm_idx,
            quadrature_weights,
        };

        if let Some(mpi_data) = &mut mol.mpi_data {
            return mpi_data.distribute_grids_tasks(&global_grid);

        } else {
            return global_grid
        }

    }

    pub fn build_nonstd(center_coordinates_bohr:Vec<(f64,f64,f64)>, proton_charges:Vec<i32>, alpha_min: Vec<HashMap<usize,f64>>, alpha_max:Vec<f64>, mpi_data: &mut Option<MPIData>) -> Grids {
        let radial_precision = 1.0e-12;
        let min_num_angular_points: usize = 50;
        let max_num_angular_points: usize = 50;
        let hardness: usize = 3;
        let pruning: String = String::from("sg1");
        let rad_grid_method: String = String::from("treutler");
        let grid_gen_level: usize = 3;

        let mut coordinates: Vec<[f64;3]> =vec![];
        let mut weights:Vec<f64> = vec![];
        let mut atm_idx: Vec<usize> = vec![];
        let mut quadrature_weights: Vec<f64> = vec![];
        let mut num_points:usize = 0;
        //println!("{:?}, {:?}",&alpha_min, &alpha_max);

        alpha_min.iter().zip(alpha_max.iter()).enumerate().for_each(|(center_index,value)| {
            let (rs_atom, ws_atom, ws_quad_atom) = gen_grids::atom_grid(
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
                gen_grids::RadiiAdjust::Becke,
            );
            //println!("alpha_min: {:?}, alpha_max: {:6.3}",&value.0, &value.1);
            //println!("rs_atom: {:?}, ws_atom: {:?}",&rs_atom, &ws_atom);
            num_points += rs_atom.len();
            coordinates.extend(rs_atom.iter().map(|value| [value.0,value.1,value.2]));
            weights.extend(ws_atom);
            atm_idx.extend(vec![center_index; rs_atom.len()]);
            quadrature_weights.extend(ws_quad_atom);
        });


        let global_grid = Grids {
            weights,
            coordinates,
            ao: None,
            aop: None,
            parallel_balancing: vec![],
            non0tab: None,
            ao_cutoff: 0.0,
            ao_compressed: None,
            aop_compressed: None,
            atm_idx,
            quadrature_weights,
        };
        if let Some(local_mpi_data) = mpi_data {
            return local_mpi_data.distribute_grids_tasks(&global_grid);

        } else {
            return global_grid
        }
    }

    /// Build a grid using the same atom-grid machinery as [`Grids::build`], but with an explicit
    /// `level` override (instead of `mol.ctrl.grid_gen_level`).
    ///
    /// This is used by the `analdrv` module to generate skeleton / cphf grids at levels
    /// that may differ from the SCF DFT grid. Unlike [`Grids::build`], this skips the
    /// `external_grids` path and MPI distribution, and does not apply the round-robin permutation.
    pub fn build_with_level(mol: &Molecule, level: usize) -> Grids {
        let radial_precision = mol.ctrl.radial_precision;
        let min_num_angular_points: usize = mol.ctrl.min_num_angular_points;
        let max_num_angular_points: usize = mol.ctrl.max_num_angular_points;
        let hardness: usize = mol.ctrl.hardness;
        let pruning: String = mol.ctrl.pruning.clone();
        let rad_grid_method: String = mol.ctrl.rad_grid_method.clone();
        let radii_adjust = gen_grids::RadiiAdjust::from_str(&mol.ctrl.radii_adjust);

        let mass_charge = get_mass_charge(&mol.geom.rg_elem);
        let proton_charges: Vec<i32> = mass_charge.iter().map(|value| value.1 as i32).collect();
        let center_coordinates_bohr = mol.geom.to_numgrid_io();
        let mut alpha_max: Vec<f64> = vec![];
        let mut alpha_min: Vec<HashMap<usize, f64>> = vec![];
        mol.basis4elem.iter().for_each(|value| {
            let (tmp_alpha_min, tmp_alpha_max) = value.to_numgrid_io();
            alpha_max.push(tmp_alpha_max);
            alpha_min.push(tmp_alpha_min);
        });

        let mut coordinates: Vec<[f64; 3]> = vec![];
        let mut weights: Vec<f64> = vec![];
        let mut atm_idx: Vec<usize> = vec![];
        let mut quadrature_weights: Vec<f64> = vec![];
        alpha_min.iter().zip(alpha_max.iter()).enumerate().for_each(|(center_index, value)| {
            let (rs_atom, ws_atom, ws_quad_atom) = gen_grids::atom_grid_with_isdf(
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
                level,
                radii_adjust,
                mol.ctrl.use_isdf,
            );
            coordinates.extend(rs_atom.iter().map(|value| [value.0, value.1, value.2]));
            weights.extend(ws_atom);
            atm_idx.extend(vec![center_index; rs_atom.len()]);
            quadrature_weights.extend(ws_quad_atom);
        });

        Grids {
            weights,
            coordinates,
            ao: None,
            aop: None,
            parallel_balancing: Vec::new(),
            non0tab: None,
            ao_cutoff: 0.0,
            ao_compressed: None,
            aop_compressed: None,
            atm_idx,
            quadrature_weights,
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

    /// Auto-select non0tab batch size based on nao to keep per-batch
    /// working set (ao_comp + vxc_ao) within reasonable cache bounds.
    pub fn auto_non0tab_blksize(nao: usize) -> usize {
        // Target: [n_active, blksize] × f64 × 2 fits in ~512 KB
        // Worst-case n_active = nao, so blksize = 512KB / (nao * 16)
        // Clamp to [32, 256]
        let target_bytes = 512 * 1024;
        let blk = target_bytes / (nao.max(1) * 16);
        blk.max(32).min(256)
    }
    /// Must be called after `prepare_tabulated_ao`.
    /// If `ao_cutoff <= 0.0`, this is a no-op.
    /// BLKSIZE auto-selection: if `ctrl.non0tab_blksize == 0`, picks based on nao.
    pub fn build_non0tab(&mut self, mol: &Molecule) {
        if self.ao_cutoff <= 0.0 {
            self.non0tab = None;
            return;
        }

        let ao = match &self.ao {
            Some(ao) => ao,
            None => { self.non0tab = None; return; }
        };

        let nao = mol.num_basis;
        let ngrids = self.coordinates.len();
        let cutoff = self.ao_cutoff;

        // Auto-select BLKSIZE when user sets non0tab_blksize = 0
        let blksize = if mol.ctrl.non0tab_blksize == 0 {
            Self::auto_non0tab_blksize(nao)
        } else {
            mol.ctrl.non0tab_blksize
        };

        let nbatches = (ngrids + blksize - 1) / blksize;

        let mut batch_ao_indices = Vec::with_capacity(nbatches);
        let mut batch_aop_indices = Vec::with_capacity(nbatches);
        let mut total_nonzero_ao = 0usize;
        let mut total_nonzero_aop = 0usize;

        for ibatch in 0..nbatches {
            let g_start = ibatch * blksize;
            let g_end = (g_start + blksize).min(ngrids);
            let nbatch = g_end - g_start;

            // For ao: check which AOs have |value| > cutoff for any grid point in batch
            let mut ao_mask = vec![false; nao];
            for mu in 0..nao {
                for g in g_start..g_end {
                    if ao[[mu, g]].abs() > cutoff {
                        ao_mask[mu] = true;
                        break;
                    }
                }
            }
            // wjyan: put generation of ao indices afterwards
            // let indices: Vec<usize> = (0..nao).filter(|&mu| ao_mask[mu]).collect();
            // total_nonzero_ao += indices.len() * nbatch;
            // batch_ao_indices.push(indices);

            // For aop: same check per gradient component
            if let Some(ref aop) = self.aop {
                let mut aop_mask = vec![false; nao];
                for mu in 0..nao {
                    for g in g_start..g_end {
                        for x in 0usize..3usize {
                            let aop_x = aop.get_reducing_matrix(x).unwrap();
                            let flat_idx = g * nao + mu;
                            if aop_x.data[flat_idx].abs() > cutoff {
                                aop_mask[mu] = true;
                                break;
                            }
                        }
                        if aop_mask[mu] { break; }
                    }
                }
                let aop_indices: Vec<usize> = (0..nao).filter(|&mu| aop_mask[mu]).collect();
                total_nonzero_aop += aop_indices.len() * nbatch;
                batch_aop_indices.push(aop_indices);
                // update ao_mask such that any non-zero aop value is not ommitted in ao_mask, to ensure consistent indexing for compressed storage
                for mu in 0..nao {
                    if !ao_mask[mu] && aop_mask[mu] {
                        ao_mask[mu] = true;
                    }
                }
            // Now generate the ao indices after checking both ao and aop masks, to ensure consistent indexing for compressed storage
            let indices: Vec<usize> = (0..nao).filter(|&mu| ao_mask[mu]).collect();
            total_nonzero_ao += indices.len() * nbatch;
            batch_ao_indices.push(indices);
            }
        }

        let total_elements = ngrids * nao;
        let sparsity_ratio = if total_nonzero_ao > 0 {
            total_nonzero_ao as f64 / total_elements as f64
        } else { 1.0 };

        // Skip compression if AO is too dense: overhead > benefit
        // Also skip if NG (no gain) — all AOs active everywhere
        let skip_compression = sparsity_ratio > 0.90 || total_nonzero_ao >= total_elements;

        let auto_note = if mol.ctrl.non0tab_blksize == 0 { " (auto)" } else { "" };

        if mol.ctrl.print_level >= 1 {
            let aop_sparsity = if total_nonzero_aop > 0 {
                total_nonzero_aop as f64 / (total_elements * 3) as f64 * 100.0
            } else { 0.0 };
            if skip_compression {
                println!(" [non0tab] cutoff={:.1e}, blksize={}{}, ao-sparsity={:.1}% → skipping (AO too dense, no benefit)",
                    cutoff, blksize, auto_note, sparsity_ratio * 100.0);
            } else {
                println!(" [non0tab] cutoff={:.1e}, blksize={}{}, ao-sparsity={:.1}% ({}/{}), aop-sparsity={:.1}%",
                    cutoff, blksize, auto_note,
                    sparsity_ratio * 100.0, total_nonzero_ao, total_elements,
                    aop_sparsity);
            }
        }

        if skip_compression {
            self.non0tab = None;
            return;
        }

        self.non0tab = Some(Non0Tab {
            batch_ao_indices,
            batch_aop_indices,
            blksize,
            ngrids,
            nao,
            ao_cutoff: cutoff,
            sparsity_ratio,
            total_nonzero_ao,
            total_nonzero_aop,
            // row-aligned with the AO block: three components per stored AO row
            total_stored_aop: 3 * total_nonzero_ao,
            total_elements,
        });

        if mol.ctrl.print_level >= 1 {
            let aop_sparsity = if total_nonzero_aop > 0 {
                total_nonzero_aop as f64 / (total_elements * 3) as f64 * 100.0
            } else { 0.0 };
            println!(" [non0tab] cutoff={:.1e}, blksize={}, ao-sparsity={:.1}% ({}/{}), aop-sparsity={:.1}%",
                cutoff, blksize,
                sparsity_ratio * 100.0, total_nonzero_ao, total_elements,
                aop_sparsity);
        }
    }

    /// Build compressed AO/AOP storage from the non0tab mask and dense AO/AOP.
    /// Must be called after `build_non0tab`.
    /// If `ao_cutoff <= 0.0` or no non0tab, this is a no-op.
    pub fn build_compressed_storage(&mut self) {
        let non0tab = match &self.non0tab {
            Some(nt) => nt,
            None => return,
        };

        // -- Compress AO --
        if let Some(ref ao) = self.ao {
            let nbatches = non0tab.batch_ao_indices.len();
            let mut batches = Vec::with_capacity(nbatches);
            let mut batch_grid_ranges = Vec::with_capacity(nbatches);

            for ibatch in 0..nbatches {
                let g_start = ibatch * non0tab.blksize;
                let g_end = (g_start + non0tab.blksize).min(non0tab.ngrids);
                let nbatch = g_end - g_start;
                let indices = &non0tab.batch_ao_indices[ibatch];
                let n_active = indices.len();

                let mut batch_ao = MatrixFull::new([n_active, nbatch], 0.0);
                for (i_local, &mu_global) in indices.iter().enumerate() {
                    for g in g_start..g_end {
                        batch_ao[[i_local, g - g_start]] = ao[[mu_global, g]];
                    }
                }
                batches.push(batch_ao);
                batch_grid_ranges.push(g_start..g_end);
            }

            self.ao_compressed = Some(CompressedGridAO {
                batches,
                batch_ao_map: non0tab.batch_ao_indices.clone(),
                batch_grid_ranges,
                blksize: non0tab.blksize,
                nao_total: non0tab.nao,
                ngrids: non0tab.ngrids,
            });
        }

        // -- Compress AOP (use same AO-based index set so that
        //    aop_batch rows match batch_ao rows in contract_response_compressed) --
        if let (Some(ref aop), true) = (&self.aop, !non0tab.batch_ao_indices.is_empty()) {
            let nbatches = non0tab.batch_ao_indices.len();
            let mut batches = Vec::with_capacity(nbatches);
            let mut batch_grid_ranges = Vec::with_capacity(nbatches);

            for ibatch in 0..nbatches {
                let g_start = ibatch * non0tab.blksize;
                let g_end = (g_start + non0tab.blksize).min(non0tab.ngrids);
                let nbatch = g_end - g_start;
                let indices = &non0tab.batch_ao_indices[ibatch];
                let n_active = indices.len();

                let mut batch_aop: [MatrixFull<f64>; 3] = [
                    MatrixFull::new([n_active, nbatch], 0.0),
                    MatrixFull::new([n_active, nbatch], 0.0),
                    MatrixFull::new([n_active, nbatch], 0.0),
                ];
                for x in 0usize..3usize {
                    let aop_x = aop.get_reducing_matrix(x).unwrap();
                    for (i_local, &mu_global) in indices.iter().enumerate() {
                        for g in g_start..g_end {
                            let flat_src = g * non0tab.nao + mu_global;
                            batch_aop[x][[i_local, g - g_start]] = aop_x.data[flat_src];
                        }
                    }
                }
                batches.push(batch_aop);
                batch_grid_ranges.push(g_start..g_end);
            }

            self.aop_compressed = Some(CompressedGridAOP {
                batches,
                batch_aop_map: non0tab.batch_ao_indices.clone(),
                batch_grid_ranges,
                blksize: non0tab.blksize,
                nao_total: non0tab.nao,
                ngrids: non0tab.ngrids,
            });
        }
    }

    /// Sparse two-pass AO generation: scan batch-by-batch, then compress.
    ///
    /// Avoids allocating the full dense AO/AOP matrices.  Instead:
    ///   Pass 1 — Compute AO per batch, scan for non-zero entries, build masks.
    ///   Pass 2 — Recompute AO per batch, extract active rows into compressed storage.
    ///
    /// Peak memory: ~nao × blksize × 4 × 8 bytes (a few MB) instead of
    ///              ~nao × ngrids × 4 × 8 bytes (potentially GB).
    ///
    /// If the overall AO sparsity exceeds 90 %, falls back to the standard
    /// dense path (`prepare_tabulated_ao_rayon_v02`).
    pub fn prepare_tabulated_ao_sparse(&mut self, mol: &Molecule) {
        let nao = mol.num_basis;
        let ngrids = self.coordinates.len();
        let cutoff = self.ao_cutoff;
        let do_gradient = mol.xc_data.use_density_gradient();

        let default_omp_num_threads = mol.ctrl.num_threads.unwrap();

        let blksize = if mol.ctrl.non0tab_blksize == 0 {
            Self::auto_non0tab_blksize(nao)
        } else {
            mol.ctrl.non0tab_blksize
        };
        let nbatches = (ngrids + blksize - 1) / blksize;

        let auto_note = if mol.ctrl.non0tab_blksize == 0 { " (auto)" } else { "" };

        // pre-compute grid ranges per batch
        let batch_ranges: Vec<std::ops::Range<usize>> = (0..nbatches)
            .map(|ib| {
                let s = ib * blksize;
                let e = (s + blksize).min(ngrids);
                s..e
            })
            .collect();

        // ── Single pass: evaluate each batch once, decide the active AO rows from
        //    the values, and gather those rows straight into the compressed blocks.
        //    The AOP block is row-aligned with the AO block, so it reuses the AO
        //    index list and needs no mask of its own. ──
        type BatchOut = (Vec<usize>, MatrixFull<f64>, Option<[MatrixFull<f64>; 3]>);
        let batch_indices: Vec<usize> = (0..nbatches).collect();
        let batch_outputs: Vec<BatchOut> = batch_indices
            .par_iter()
            .map(|&ibatch| -> BatchOut {
                omp_set_num_threads_wrapper(1);
                let g_range = &batch_ranges[ibatch];
                let nbatch = g_range.len();

                let mut ao_indices: Vec<usize> = Vec::new();
                let mut ao_vals: Vec<f64> = Vec::new();
                let mut aop_vals: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];

                mol.basis4elem
                    .iter()
                    .zip(mol.geom.rg_position.iter_columns_full())
                    .for_each(|(elem, geom)| {
                        let start = elem.global_index.0;
                        let nbas = elem.global_index.1;
                        let tmp_geom: [f64; 3] = geom.try_into().unwrap();
                        let tab = gto_value_serial(
                            &self.coordinates[g_range.clone()],
                            &tmp_geom,
                            elem,
                            &mol.ctrl.basis_type,
                        );
                        let tab_dev = if do_gradient {
                            Some(gto_1st_value_serial(
                                &self.coordinates[g_range.clone()],
                                &tmp_geom,
                                elem,
                                &mol.ctrl.basis_type,
                            ))
                        } else {
                            None
                        };
                        // Every basis function belongs to exactly one element, so the
                        // per-row decision can be made here: keep a row when any grid
                        // point of this batch has |value| above the cutoff. Elements are
                        // visited in global AO order, so the gathered rows stay ordered.
                        for r in 0..nbas {
                            if !(0..nbatch).any(|g| tab[[r, g]].abs() > cutoff) {
                                continue;
                            }
                            ao_indices.push(start + r);
                            for g in 0..nbatch {
                                ao_vals.push(tab[[r, g]]);
                            }
                            if let Some(dev) = &tab_dev {
                                for x in 0usize..3usize {
                                    for g in 0..nbatch {
                                        aop_vals[x].push(dev[x][[r, g]]);
                                    }
                                }
                            }
                        }
                    });

                let n_active = ao_indices.len();
                if n_active == 0 {
                    return (ao_indices, MatrixFull::<f64>::empty(), None);
                }
                // The staging above is row-major, but `MatrixFull` is column-major
                // (`indicing = [1, size[0]]`), so reorder before wrapping the buffer.
                let transpose = |src: Vec<f64>| -> MatrixFull<f64> {
                    let mut flat = vec![0.0f64; n_active * nbatch];
                    for i in 0..n_active {
                        for g in 0..nbatch {
                            flat[i + g * n_active] = src[i * nbatch + g];
                        }
                    }
                    MatrixFull::<f64>::from_vec([n_active, nbatch], flat).unwrap()
                };
                let ao_block = transpose(ao_vals);
                let [aop_x, aop_y, aop_z] = aop_vals;
                let aop_block = if do_gradient {
                    Some([transpose(aop_x), transpose(aop_y), transpose(aop_z)])
                } else {
                    None
                };
                (ao_indices, ao_block, aop_block)
            })
            .collect();

        // split the batch outputs into the index lists and the two compressed blocks
        let mut batch_ao_indices: Vec<Vec<usize>> = Vec::with_capacity(nbatches);
        let mut ao_comp_batches: Vec<MatrixFull<f64>> = Vec::with_capacity(nbatches);
        let mut aop_comp_batches: Vec<[MatrixFull<f64>; 3]> = Vec::with_capacity(nbatches);
        let mut total_nonzero_ao: usize = 0;
        for (indices, ao, aop) in batch_outputs {
            total_nonzero_ao += indices.len() * blksize; // upper bound (last batch may be shorter)
            batch_ao_indices.push(indices);
            ao_comp_batches.push(ao);
            aop_comp_batches.push(aop.unwrap_or_else(|| [
                MatrixFull::<f64>::empty(),
                MatrixFull::<f64>::empty(),
                MatrixFull::<f64>::empty(),
            ]));
        }
        // The AOP block is stored row-aligned with the AO block (see
        // `contract_response_compressed`), so it holds all three gradient components of
        // every stored AO row. `total_stored_aop` is therefore the number that matches
        // the memory footprint, and it is what the log reports.
        let total_stored_aop: usize = if do_gradient { 3 * total_nonzero_ao } else { 0 };
        let ao_rows_per_batch =
            batch_ao_indices.iter().map(|v| v.len()).sum::<usize>() as f64 / nbatches.max(1) as f64;
        let total_elements = ngrids * nao;
        let sparsity_ratio = if total_nonzero_ao > 0 {
            total_nonzero_ao as f64 / total_elements as f64
        } else {
            1.0
        };

        // skip if too dense
        let skip = sparsity_ratio > 0.90 || total_nonzero_ao >= total_elements;
        if mol.ctrl.print_level >= 1 {
            if skip {
                println!(
                    " [non0tab] cutoff={:.1e}, blksize={}{}, ao-sparsity={:.1}% → skipping (AO too dense), falling back to dense",
                    cutoff, blksize, auto_note, sparsity_ratio * 100.0
                );
            } else {
                let stored_aop_sparsity = if total_stored_aop > 0 {
                    total_stored_aop as f64 / (total_elements * 3) as f64 * 100.0
                } else {
                    0.0
                };
                let gib = |entries: usize| entries as f64 * 8.0 / (1024.0 * 1024.0 * 1024.0);
                println!(
                    " [non0tab] cutoff={:.1e}, blksize={}{}, ao stored={:.1}% ({} entries, {:.2} GiB), aop stored={:.1}% ({} entries, {:.2} GiB), rows/batch: ao {:.0}/{}",
                    cutoff, blksize, auto_note,
                    sparsity_ratio * 100.0, total_nonzero_ao, gib(total_nonzero_ao),
                    stored_aop_sparsity, total_stored_aop, gib(total_stored_aop),
                    ao_rows_per_batch, nao,
                );
            }
        }

        if skip {
            omp_set_num_threads_wrapper(default_omp_num_threads);
            // fall back to dense path
            self.prepare_tabulated_ao_rayon_v02(mol);
            return;
        }

        // The compressed blocks and the index lists were produced by the single pass
        // above; there is no second evaluation and no staging array to unzip.

        // ── Store results ──
        self.non0tab = Some(Non0Tab {
            // The AOP block is row-aligned with the AO block, so it shares the AO
            // index list; the separate gradient mask is no longer computed.
            batch_ao_indices: batch_ao_indices.clone(),
            batch_aop_indices: batch_ao_indices.clone(),
            blksize,
            ngrids,
            nao,
            ao_cutoff: cutoff,
            sparsity_ratio,
            total_nonzero_ao,
            total_nonzero_aop: total_stored_aop,
            total_stored_aop,
            total_elements,
        });

        let batch_grid_ranges = batch_ranges.clone();
        self.ao_compressed = Some(CompressedGridAO {
            batches: ao_comp_batches,
            batch_ao_map: batch_ao_indices.clone(),
            batch_grid_ranges: batch_grid_ranges.clone(),
            blksize,
            nao_total: nao,
            ngrids,
        });

        if do_gradient && !batch_ao_indices.is_empty() {
            self.aop_compressed = Some(CompressedGridAOP {
                batches: aop_comp_batches,
                // row-aligned with the AO block, see `contract_response_compressed`
                batch_aop_map: batch_ao_indices.clone(),
                batch_grid_ranges,
                blksize,
                nao_total: nao,
                ngrids,
            });
        }

        omp_set_num_threads_wrapper(default_omp_num_threads);
    }

    /// Returns a reference to the dense AO matrix, decompressing from
    /// compressed storage if necessary (for post-SCF modules that need dense).
    pub fn ensure_dense_ao(&mut self) -> &MatrixFull<f64> {
        if self.ao.is_none() {
            if let Some(ref c) = self.ao_compressed {
                let ao_dense = Self::decompress_ao(c);
                self.ao = Some(ao_dense);
            } else {
                panic!("dense AO is not available and no compressed storage to decompress");
            }
        }
        self.ao.as_ref().unwrap()
    }

    /// Returns a reference to the dense AOP matrix, decompressing from
    /// compressed storage if necessary.
    pub fn ensure_dense_aop(&mut self) -> &RIFull<f64> {
        if self.aop.is_none() {
            if let Some(ref c) = self.aop_compressed {
                let mut aop_dense = RIFull::new([c.nao_total, c.ngrids, 3], 0.0);
                for ibatch in 0..c.batches.len() {
                    let indices = &c.batch_aop_map[ibatch];
                    let g_range = &c.batch_grid_ranges[ibatch];
                    let batch_aop = &c.batches[ibatch];
                    for x in 0usize..3usize {
                        let aop_x = aop_dense.get_reducing_matrix_mut(x).unwrap();
                        for (i_local, &mu_global) in indices.iter().enumerate() {
                            for g in g_range.clone() {
                                let flat_dst = g * c.nao_total + mu_global;
                                aop_x.data[flat_dst] = batch_aop[x][[i_local, g - g_range.start]];
                            }
                        }
                    }
                }
                self.aop = Some(aop_dense);
            } else {
                panic!("dense AOP is not available and no compressed storage to decompress");
            }
        }
        self.aop.as_ref().unwrap()
    }

    /// Decompress AO from compressed storage back to dense format, for verification.
    pub fn decompress_ao(compressed: &CompressedGridAO) -> MatrixFull<f64> {
        let nao = compressed.nao_total;
        let ngrids = compressed.ngrids;
        let mut ao_dense = MatrixFull::new([nao, ngrids], 0.0);
        for ibatch in 0..compressed.batches.len() {
            let batch_ao = &compressed.batches[ibatch];
            let indices = &compressed.batch_ao_map[ibatch];
            let g_range = &compressed.batch_grid_ranges[ibatch];
            for (i_local, &mu_global) in indices.iter().enumerate() {
                for g in g_range.clone() {
                    ao_dense[[mu_global, g]] = batch_ao[[i_local, g - g_range.start]];
                }
            }
        }
        ao_dense
    }

    /// Decompress AOP from compressed storage back to dense format.
    pub fn decompress_aop(compressed: &CompressedGridAOP) -> RIFull<f64> {
        let mut aop_dense = RIFull::new([compressed.nao_total, compressed.ngrids, 3], 0.0);
        for ibatch in 0..compressed.batches.len() {
            let indices = &compressed.batch_aop_map[ibatch];
            let g_range = &compressed.batch_grid_ranges[ibatch];
            let batch_aop = &compressed.batches[ibatch];
            for x in 0usize..3usize {
                let aop_x = aop_dense.get_reducing_matrix_mut(x).unwrap();
                for (i_local, &mu_global) in indices.iter().enumerate() {
                    for g in g_range.clone() {
                        let flat_dst = g * compressed.nao_total + mu_global;
                        aop_x.data[flat_dst] = batch_aop[x][[i_local, g - g_range.start]];
                    }
                }
            }
        }
        aop_dense
    }

    /// Memory footprint estimate for dense + compressed storage (bytes).
    pub fn memory_footprint(&self) -> (usize, usize, usize) {
        let dense_ao_bytes = self.ao.as_ref().map(|m| m.data.len() * 8).unwrap_or(0);
        let dense_aop_bytes = self.aop.as_ref().map(|m| m.data.len() * 8).unwrap_or(0);
        let mut comp_bytes = 0usize;
        if let Some(ref c) = self.ao_compressed {
            for b in &c.batches { comp_bytes += b.data.len() * 8; }
        }
        if let Some(ref c) = self.aop_compressed {
            for b in &c.batches {
                for x in 0usize..3usize { comp_bytes += b[x].data.len() * 8; }
            }
        }
        (dense_ao_bytes + dense_aop_bytes, comp_bytes, dense_ao_bytes + dense_aop_bytes + comp_bytes)
    }

    /// Build the XC response matrix (loc_vxc_mat) using compressed AO/AOP storage.
    /// Processes batches that overlap with `range_grids`. Accumulates into `loc_vxc_mat`.
    pub fn contract_response_compressed(
        &self,
        range_grids: &std::ops::Range<usize>,
        vrho: &MatrixFull<f64>,
        vsigma: &MatrixFull<f64>,
        vtau: &MatrixFull<f64>,
        weights: &[f64],
        loc_vxc_mat: &mut [MatrixFull<f64>],
        spin_channel: usize,
        use_density_gradient: bool,
        use_kinetic_density: bool,
        loc_rhop: &RIFull<f64>,
        num_basis: usize,
    ) {
        let ao_c = match &self.ao_compressed { Some(c) => c, None => return };
        let aop_c = self.aop_compressed.as_ref();

        let ibatch_start = range_grids.start / ao_c.blksize;
        let ibatch_end = ((range_grids.end + ao_c.blksize - 1) / ao_c.blksize).min(ao_c.batches.len());

        for ibatch in ibatch_start..ibatch_end {
            let batch_ao = &ao_c.batches[ibatch];
            let indices = &ao_c.batch_ao_map[ibatch];
            let n_active = indices.len();
            if n_active == 0 { continue; }

            let g_range = &ao_c.batch_grid_ranges[ibatch];

            let g_local_start = (range_grids.start.saturating_sub(g_range.start));
            let g_local_end = (range_grids.end.min(g_range.end) - g_range.start).min(batch_ao.size[1]);
            if g_local_end <= g_local_start { continue; }
            let n_batch = g_local_end - g_local_start;
            // offset of this batch's first grid within the vrho/vsigma/vtau/rhop arrays
            let pot_offset = g_range.start + g_local_start - range_grids.start;

            // -- LDA: build vxc_ao_comp[n_active, n_batch] from vrho --
            let mut vxc_ao_comp = vec![MatrixFull::new([n_active, n_batch], 0.0); spin_channel];
            for i_spin in 0..spin_channel {
                let loc_vrho_s = vrho.slice_column(i_spin);
                let batch_ao_slice = batch_ao.to_matrixfullslice_columns(g_local_start..g_local_end);
                contract_vxc_0_serial(&mut vxc_ao_comp[i_spin], &batch_ao_slice, &loc_vrho_s[pot_offset..pot_offset + n_batch], None);
            }

            // -- GGA: vsigma contribution --
            if use_density_gradient {
                if let Some(aop_c) = aop_c {
                    let aop_batch = &aop_c.batches[ibatch];
                    if spin_channel == 1 {
                        let loc_vsigma_s = vsigma.slice_column(0);
                        let loc_rhop_s = loc_rhop.get_reducing_matrix(0).unwrap();
                        let mut loc_wao = MatrixFull::new([n_active, n_batch], 0.0);
                        for x in 0usize..3usize {
                            let loc_aop_x = aop_batch[x].to_matrixfullslice_columns(g_local_start..g_local_end);
                            let loc_rhop_s_x = loc_rhop_s.get_slice_x(x);
                            contract_vxc_0_serial(&mut loc_wao, &loc_aop_x, &loc_rhop_s_x[pot_offset..pot_offset + n_batch], None);
                        }
                        contract_vxc_0_serial(&mut vxc_ao_comp[0], &loc_wao.to_matrixfullslice(), &loc_vsigma_s[pot_offset..pot_offset + n_batch], Some(4.0));
                    } else {
                        {
                            let loc_rhop_a = loc_rhop.get_reducing_matrix(0).unwrap();
                            let loc_vsigma_uu = vsigma.slice_column(0);
                            let mut loc_dao = MatrixFull::new([n_active, n_batch], 0.0);
                            for x in 0usize..3usize {
                                let loc_aop_x = aop_batch[x].to_matrixfullslice_columns(g_local_start..g_local_end);
                                let loc_rhop_s_x = loc_rhop_a.get_slice_x(x);
                                contract_vxc_0_serial(&mut loc_dao, &loc_aop_x, &loc_rhop_s_x[pot_offset..pot_offset + n_batch], None);
                            }
                            contract_vxc_0_serial(&mut vxc_ao_comp[0], &loc_dao.to_matrixfullslice(), &loc_vsigma_uu[pot_offset..pot_offset + n_batch], Some(4.0));
                            let loc_rhop_b = loc_rhop.get_reducing_matrix(1).unwrap();
                            let loc_vsigma_ud = vsigma.slice_column(1);
                            loc_dao.data.iter_mut().for_each(|d| *d = 0.0);
                            for x in 0usize..3usize {
                                let loc_aop_x = aop_batch[x].to_matrixfullslice_columns(g_local_start..g_local_end);
                                let loc_rhop_s_x = loc_rhop_b.get_slice_x(x);
                                contract_vxc_0_serial(&mut loc_dao, &loc_aop_x, &loc_rhop_s_x[pot_offset..pot_offset + n_batch], None);
                            }
                            contract_vxc_0_serial(&mut vxc_ao_comp[0], &loc_dao.to_matrixfullslice(), &loc_vsigma_ud[pot_offset..pot_offset + n_batch], Some(2.0));
                        }
                        {
                            let loc_rhop_b = loc_rhop.get_reducing_matrix(1).unwrap();
                            let loc_vsigma_dd = vsigma.slice_column(2);
                            let mut loc_dao = MatrixFull::new([n_active, n_batch], 0.0);
                            for x in 0usize..3usize {
                                let loc_aop_x = aop_batch[x].to_matrixfullslice_columns(g_local_start..g_local_end);
                                let loc_rhop_s_x = loc_rhop_b.get_slice_x(x);
                                contract_vxc_0_serial(&mut loc_dao, &loc_aop_x, &loc_rhop_s_x[pot_offset..pot_offset + n_batch], None);
                            }
                            contract_vxc_0_serial(&mut vxc_ao_comp[1], &loc_dao.to_matrixfullslice(), &loc_vsigma_dd[pot_offset..pot_offset + n_batch], Some(4.0));
                            let loc_rhop_a = loc_rhop.get_reducing_matrix(0).unwrap();
                            let loc_vsigma_ud = vsigma.slice_column(1);
                            loc_dao.data.iter_mut().for_each(|d| *d = 0.0);
                            for x in 0usize..3usize {
                                let loc_aop_x = aop_batch[x].to_matrixfullslice_columns(g_local_start..g_local_end);
                                let loc_rhop_s_x = loc_rhop_a.get_slice_x(x);
                                contract_vxc_0_serial(&mut loc_dao, &loc_aop_x, &loc_rhop_s_x[pot_offset..pot_offset + n_batch], None);
                            }
                            contract_vxc_0_serial(&mut vxc_ao_comp[1], &loc_dao.to_matrixfullslice(), &loc_vsigma_ud[pot_offset..pot_offset + n_batch], Some(2.0));
                        }
                    }
                }
            }

            // -- Apply weights --
            for i_spin in 0..spin_channel {
                vxc_ao_comp[i_spin].iter_columns_full_mut().enumerate()
                    .for_each(|(g, col)| {
                        let w = weights[pot_offset + g];
                        col.iter_mut().for_each(|v| *v *= w);
                    });
            }

            // -- GEMM + scatter: ao_comp[n_active, n_batch] × vxc_ao_comp^T → loc_vxc_mat --
            for i_spin in 0..spin_channel {
                let mut loc_mat_batch = MatrixFull::new([n_active, n_active], 0.0);
                _dgemm(
                    batch_ao, (0..n_active, g_local_start..g_local_end), 'N',
                    &vxc_ao_comp[i_spin], (0..n_active, 0..n_batch), 'T',
                    &mut loc_mat_batch, (0..n_active, 0..n_active), 1.0, 0.0,
                );
                // Scatter to the global vxc matrix. `loc_mat_batch` is symmetric in the
                // batch-local indices and both matrices are column-major, so with `j_l`
                // outermost the destination column `nu` is fixed and the inner loop walks
                // `mu`: source and destination are then both sequential. Measured on
                // (H2O)10 / PBE0: scatter phase −24%. A per-batch pair gate was tried and
                // rejected (only ~6-9% of the pairs were skippable while building the
                // per-AO maxima cost ~3x the scatter phase).
                let vxc_mat_s = &mut loc_vxc_mat[i_spin];
                for j_l in 0..n_active {
                    let nu = indices[j_l];
                    for i_l in 0..n_active {
                        let mu = indices[i_l];
                        vxc_mat_s[[mu, nu]] += loc_mat_batch[[i_l, j_l]];
                    }
                }
            }

            // -- mGGA: tau contribution --
            if use_kinetic_density {
                if let Some(aop_c) = aop_c {
                    let aop_batch = &aop_c.batches[ibatch];
                    let mut vxc_ao_tau = vec![MatrixFull::new([n_active, n_batch], 0.0); spin_channel];
                    for i_spin in 0..spin_channel {
                        let loc_vtau_s = vtau.slice_column(i_spin);
                        let vtau_weighted: Vec<f64> = (0..n_batch)
                            .map(|g| loc_vtau_s[pot_offset + g] * weights[pot_offset + g])
                            .collect();
                        for ic in 0usize..3usize {
                            let loc_aop_ic = aop_batch[ic].to_matrixfullslice_columns(g_local_start..g_local_end);
                            contract_vxc_0_serial(&mut vxc_ao_tau[i_spin], &loc_aop_ic, &vtau_weighted, Some(0.5));
                            let mut loc_mat_batch = MatrixFull::new([n_active, n_active], 0.0);
                            _dgemm(
                                &aop_batch[ic], (0..n_active, g_local_start..g_local_end), 'N',
                                &vxc_ao_tau[i_spin], (0..n_active, 0..n_batch), 'T',
                                &mut loc_mat_batch, (0..n_active, 0..n_active), 1.0, 1.0,
                            );
                            let vxc_mat_s = &mut loc_vxc_mat[i_spin];
                            for j_l in 0..n_active {
                                let nu = indices[j_l];
                                for i_l in 0..n_active {
                                    let mu = indices[i_l];
                                    vxc_mat_s[[mu, nu]] += loc_mat_batch[[i_l, j_l]];
                                }
                            }
                            vxc_ao_tau[i_spin].data.iter_mut().for_each(|t| *t = 0.0);
                        }
                    }
                }
            }
        }
    }

    /// Drop the grid points whose initial-guess density contribution is negligible
    /// and rebuild the tabulated AO/AOP data, once, before the SCF iterations.
    ///
    /// The criterion is `sum_spin |rho_spin| * |w_p| < threshold` (borrowed from
    /// PyFock's `DFT.grid_pruning_mask`, which drops `|rho * w| < 1e-11`). The
    /// weight is part of the criterion on purpose: a bare `|rho|` threshold also
    /// drops far-tail points whose quadrature weight is large.
    ///
    /// Returns the number of dropped points; 0 when the feature is off, when fewer
    /// than 2% of the points would go (the rebuild would not pay off) or when no
    /// tabulated density is available.
    pub fn prune_by_density(
        &mut self,
        mol: &Molecule,
        dm: &Vec<MatrixFull<f64>>,
        spin_channel: usize,
        threshold: f64,
    ) -> usize {
        if threshold <= 0.0 { return 0; }
        let ngrids = self.coordinates.len();
        if ngrids == 0 { return 0; }
        // One grid table per point; a short table would make the survivors of the
        // compaction below silently mis-aligned (`atm_idx` / quadrature weights are
        // indexed by the surviving points).
        assert!(
            self.atm_idx.len() == ngrids && self.quadrature_weights.len() == ngrids,
            "prune_by_density: grid tables are not aligned ({} points, {} atom indices, {} quadrature weights)",
            ngrids, self.atm_idx.len(), self.quadrature_weights.len()
        );

        let rho: MatrixFull<f64> = if self.ao_compressed.is_some() {
            self.prepare_tabulated_density_slots_dm_only_compressed(dm, spin_channel, 0..ngrids).0
        } else if self.ao.is_some() {
            self.prepare_tabulated_density_slots_dm_only(dm, spin_channel, 0..ngrids).0
        } else {
            if mol.ctrl.print_level >= 1 {
                println!(" [grid-prune] threshold {:.1e}: skipped, the grid holds no tabulated AO values", threshold);
            }
            return 0;
        };

        // Per-point density contribution `Σ_s |ρ_s| · |w_p|`, and the two sums that make
        // the truncation auditable: what the dropped points carried, and what the whole
        // grid carries. The XC integrand is `ρ·w·e_xc`, so the dropped sum bounds the
        // missed energy up to `max|e_xc|` (order 1 Ha for GGA); its *share* of the total
        // is the scale-free number, and it is the one that decides whether an absolute
        // threshold is loose enough for the system at hand.
        let mut contributions = Vec::with_capacity(ngrids);
        let mut dropped_sum = 0.0_f64;
        let mut dropped_max = 0.0_f64;
        let mut total_sum = 0.0_f64;
        for p in 0..ngrids {
            let mut rho_sum = 0.0f64;
            for s in 0..spin_channel { rho_sum += rho[[p, s]].abs(); }
            let contribution = rho_sum * self.weights[p].abs();
            total_sum += contribution;
            if contribution < threshold {
                dropped_sum += contribution;
                if contribution > dropped_max { dropped_max = contribution; }
            }
            contributions.push(contribution);
        }
        let keep: Vec<bool> = contributions.iter().map(|&c| c >= threshold).collect();
        let n_keep = keep.iter().filter(|&&k| k).count();
        let n_drop = ngrids - n_keep;
        if n_drop * 50 < ngrids {
            if mol.ctrl.print_level >= 1 {
                println!(" [grid-prune] threshold {:.1e}: only {} of {} points would go (<2%), keeping the full grid (they carry {:.3e} of the {:.3e} grid total)",
                    threshold, n_drop, ngrids, dropped_sum, total_sum);
            }
            return 0;
        }

        let mut coords = Vec::with_capacity(n_keep);
        let mut weights = Vec::with_capacity(n_keep);
        let mut atm_idx = Vec::with_capacity(n_keep);
        let mut quadrature_weights = Vec::with_capacity(n_keep);
        for p in 0..ngrids {
            if keep[p] {
                coords.push(self.coordinates[p]);
                weights.push(self.weights[p]);
                atm_idx.push(self.atm_idx.get(p).copied().unwrap_or(usize::MAX));
                quadrature_weights.push(self.quadrature_weights.get(p).copied().unwrap_or(0.0));
            }
        }
        self.coordinates = coords;
        self.weights = weights;
        self.atm_idx = atm_idx;
        self.quadrature_weights = quadrature_weights;
        self.parallel_balancing = crate::utilities::balancing(n_keep, rayon::current_num_threads());

        // The kept points are a subset of the points the tables already cover, so the
        // tabulated AO/AOP values can be copied into the smaller grid instead of being
        // evaluated again (`compact_tabulated_ao`). That costs about a fifth of a
        // re-tabulation but needs the old and the new table at once, so
        // `grid_prune_compact = false` (or a table that cannot be compacted) selects the
        // re-tabulation, which keeps the peak at one table.
        let compacted = mol.ctrl.grid_prune_compact && self.compact_tabulated_ao(&keep);
        if !compacted {
            // Release the stale tables *before* rebuilding: the old AOP table can be
            // several GB, and the rebuild below allocates a fresh one.
            self.ao = None;
            self.aop = None;
            self.non0tab = None;
            self.ao_compressed = None;
            self.aop_compressed = None;
            if self.ao_cutoff > 0.0 {
                self.prepare_tabulated_ao_sparse(mol);
            } else {
                self.prepare_tabulated_ao(mol);
                self.build_non0tab(mol);
                self.build_compressed_storage();
            }
        }
        if mol.ctrl.print_level >= 1 {
            println!(" [grid-prune] threshold {:.1e}: dropped {} of {} grid points ({:.1}%), they carried {:.3e} of the {:.3e} total density contribution ({:.2e}), largest dropped term {:.3e}; tabulated AO/AOP {}",
                threshold, n_drop, ngrids, n_drop as f64 / ngrids as f64 * 100.0,
                dropped_sum, total_sum,
                if total_sum > 0.0 { dropped_sum / total_sum } else { 0.0 },
                dropped_max,
                if compacted {
                    "compacted in place"
                } else if !mol.ctrl.grid_prune_compact {
                    "rebuilt (grid_prune_compact = false)"
                } else {
                    "rebuilt"
                });
        }
        n_drop
    }

    /// Copy the tabulated AO/AOP values of the surviving grid points into a fresh,
    /// re-chunked batch structure, instead of evaluating every basis function again.
    ///
    /// `keep` masks the points the tables were built on (`keep.len()` must equal the number
    /// of points they cover), and the point arrays must already hold the surviving points in
    /// ascending order. The compaction walks the kept points in that same order, so the new
    /// batches are again the contiguous ranges `ib*blksize..(ib+1)*blksize` of the new grid,
    /// which is the layout the consumers derive from `blksize`
    /// (see `prepare_tabulated_density_slots_dm_only_compressed`).
    ///
    /// Every stored value is copied verbatim, so the compacted table is the one the
    /// re-tabulation would have produced, up to the rows a contributing old batch had
    /// dropped: those are the entries whose values are all below `ao_cutoff` in that batch,
    /// which the compressed storage treats as zero by construction, and they enter the
    /// merged batch as zeros. Old blocks are released as soon as no later batch can need
    /// them, so the peak stays at roughly the old table's size rather than old + new.
    ///
    /// Returns `false`, leaving the object untouched, when there is nothing to compact:
    /// no tabulated values, inconsistent bookkeeping, or an empty surviving set.
    pub fn compact_tabulated_ao(&mut self, keep: &[bool]) -> bool {
        let old_ngrids = keep.len();
        let n_keep = keep.iter().filter(|&&k| k).count();
        if n_keep == 0 {
            // an empty surviving set is refused here; the caller guards it (see
            // the `n_drop`/`n_keep` checks in `prune_by_density`)
            return false;
        }
        if n_keep == old_ngrids {
            // nothing was dropped, so the tables already match the grid
            return true;
        }
        let mut new2old: Vec<usize> = Vec::with_capacity(n_keep);
        for (p, &k) in keep.iter().enumerate() {
            if k { new2old.push(p); }
        }

        // ── compressed path (the default, `ao_cutoff > 0`) ──
        if let Some(ao_c) = self.ao_compressed.as_ref() {
            let aop_c = self.aop_compressed.as_ref();
            let blksize = ao_c.blksize.max(1);
            let nao = ao_c.nao_total;
            if ao_c.ngrids != old_ngrids
                || ao_c.batches.len() != ao_c.batch_grid_ranges.len()
                || ao_c.batches.len() != ao_c.batch_ao_map.len()
                || ao_c.batches.is_empty()
                || aop_c.map_or(false, |c| {
                    c.ngrids != old_ngrids
                        || c.batches.len() != ao_c.batches.len()
                        || c.batch_aop_map.len() != ao_c.batches.len()
                })
            {
                return false;
            }
            let ao_cutoff = self.non0tab.as_ref().map_or(0.0, |n| n.ao_cutoff);

            // take ownership of the blocks so each old batch can be freed as it is consumed
            let CompressedGridAO { batches: mut old_ao, batch_ao_map: old_map, blksize: _, .. } =
                self.ao_compressed.take().unwrap();
            let mut old_aop = self.aop_compressed.take().map(|c| c.batches);

            let nbatches_new = (n_keep + blksize - 1) / blksize;
            let mut new_batches: Vec<MatrixFull<f64>> = Vec::with_capacity(nbatches_new);
            let mut new_aop_batches: Vec<[MatrixFull<f64>; 3]> = Vec::with_capacity(nbatches_new);
            let mut new_maps: Vec<Vec<usize>> = Vec::with_capacity(nbatches_new);
            let mut new_ranges: Vec<std::ops::Range<usize>> = Vec::with_capacity(nbatches_new);
            // global AO index -> row of the batch being built (usize::MAX = absent)
            let mut row_of = vec![usize::MAX; nao];
            let mut freed_upto = 0usize;

            for ib in 0..nbatches_new {
                let gs = ib * blksize;
                let ge = (gs + blksize).min(n_keep);
                let nb = ge - gs;

                // the kept points of this batch, grouped by the old batch they came from
                // (ascending, because `new2old` is ascending)
                let mut groups: Vec<(usize, usize, usize)> = Vec::new(); // (old batch, first col, end col)
                for g in gs..ge {
                    let p = new2old[g];
                    let ob = p / blksize;
                    debug_assert!(ob < old_ao.len());
                    match groups.last_mut() {
                        Some((last, _, end)) if *last == ob => *end = p - ob * blksize + 1,
                        _ => groups.push((ob, p - ob * blksize, p - ob * blksize + 1)),
                    }
                }

                // the merged active-row set of this batch, in the old tables' ascending order
                let mut map: Vec<usize> = Vec::new();
                for &(ob, _, _) in &groups {
                    for &mu in old_map[ob].iter() {
                        if row_of[mu] == usize::MAX {
                            row_of[mu] = map.len();
                            map.push(mu);
                        }
                    }
                }
                map.sort_unstable();
                for (r, &mu) in map.iter().enumerate() {
                    row_of[mu] = r;
                }

                // An entirely inactive batch is stored as an empty block, exactly as the
                // re-tabulation stores it (`n_active == 0`).
                let empty_block = || MatrixFull::<f64>::empty();
                let mut ao_block = if map.is_empty() {
                    empty_block()
                } else {
                    MatrixFull::new([map.len(), nb], 0.0)
                };
                let want_aop = old_aop.is_some();
                let mut aop_block = [
                    MatrixFull::<f64>::empty(),
                    MatrixFull::<f64>::empty(),
                    MatrixFull::<f64>::empty(),
                ];
                if want_aop && !map.is_empty() {
                    for x in 0..3 {
                        aop_block[x] = MatrixFull::new([map.len(), nb], 0.0);
                    }
                }
                if !map.is_empty() {
                    for g in gs..ge {
                        let p = new2old[g];
                        let ob = p / blksize;
                        // a contributing old batch whose own row set is empty stores no
                        // values at all (its points were below the cutoff for every AO)
                        if old_map[ob].is_empty() {
                            continue;
                        }
                        let col = p - ob * blksize;
                        let gcol = g - gs;
                        let src = old_ao[ob].slice_column(col);
                        let dst = ao_block.slice_column_mut(gcol);
                        for (r_old, &mu) in old_map[ob].iter().enumerate() {
                            dst[row_of[mu]] = src[r_old];
                        }
                        if want_aop {
                            let src_aop = &old_aop.as_ref().unwrap()[ob];
                            for x in 0..3 {
                                let s = src_aop[x].slice_column(col);
                                let d = aop_block[x].slice_column_mut(gcol);
                                for (r_old, &mu) in old_map[ob].iter().enumerate() {
                                    d[row_of[mu]] = s[r_old];
                                }
                            }
                        }
                    }
                }
                for &mu in map.iter() {
                    row_of[mu] = usize::MAX;
                }

                new_batches.push(ao_block);
                if want_aop {
                    new_aop_batches.push(aop_block);
                }
                new_maps.push(map);
                new_ranges.push(gs..ge);

                // Release the old blocks that no later batch can reach: the kept points are
                // ascending, so a new batch only ever draws from the old batches of its own
                // range, and nothing below the next batch's first old batch can come back.
                let next_first = if ge < n_keep {
                    (new2old[ge] / blksize).min(old_ao.len())
                } else {
                    old_ao.len()
                };
                while freed_upto < next_first {
                    old_ao[freed_upto] = MatrixFull::empty();
                    if let Some(v) = old_aop.as_mut() {
                        v[freed_upto] = [
                            MatrixFull::empty(),
                            MatrixFull::empty(),
                            MatrixFull::empty(),
                        ];
                    }
                    freed_upto += 1;
                }
            }
            debug_assert!(freed_upto <= old_ao.len());

            // bookkeeping: same statistics the re-tabulation reports
            let total_stored_ao: usize = new_maps
                .iter()
                .zip(new_ranges.iter())
                .map(|(m, r)| m.len() * r.len())
                .sum();
            let total_elements = n_keep * nao;
            let sparsity_ratio = if total_elements > 0 {
                total_stored_ao as f64 / total_elements as f64
            } else {
                1.0
            };
            let do_gradient = !new_aop_batches.is_empty();
            let total_stored_aop = if do_gradient { 3 * total_stored_ao } else { 0 };

            self.ao_compressed = Some(CompressedGridAO {
                batches: new_batches,
                batch_ao_map: new_maps.clone(),
                batch_grid_ranges: new_ranges.clone(),
                blksize,
                nao_total: nao,
                ngrids: n_keep,
            });
            self.aop_compressed = if do_gradient {
                Some(CompressedGridAOP {
                    batches: new_aop_batches,
                    batch_aop_map: new_maps.clone(),
                    batch_grid_ranges: new_ranges,
                    blksize,
                    nao_total: nao,
                    ngrids: n_keep,
                })
            } else {
                None
            };
            self.non0tab = Some(Non0Tab {
                batch_ao_indices: new_maps.clone(),
                batch_aop_indices: new_maps,
                blksize,
                ngrids: n_keep,
                nao,
                ao_cutoff,
                sparsity_ratio,
                total_nonzero_ao: total_stored_ao,
                total_nonzero_aop: total_stored_aop,
                total_stored_aop,
                total_elements,
            });
            // A dense copy that was decompressed from the old storage no longer matches
            // the grid; drop it so `ensure_dense_ao`/`ensure_dense_aop` rebuild it from the
            // compacted storage on demand (this is what the re-tabulation path did too).
            self.ao = None;
            self.aop = None;
            return true;
        }

        // ── dense path (`ao_cutoff = 0`): keep the surviving columns ──
        if self.ao.is_some() {
            let nao = self.ao.as_ref().unwrap().size[0];
            if self.ao.as_ref().unwrap().size[1] != old_ngrids {
                return false;
            }
            let old_ao = self.ao.take().unwrap();
            let mut ao = MatrixFull::new([nao, n_keep], 0.0);
            for (g_new, &g_old) in new2old.iter().enumerate() {
                ao.slice_column_mut(g_new)
                    .copy_from_slice(old_ao.slice_column(g_old));
            }
            self.ao = Some(ao);
            if let Some(old_aop) = self.aop.take() {
                let mut aop = RIFull::<f64>::new([nao, n_keep, 3], 0.0);
                for x in 0..3 {
                    let src = old_aop.get_reducing_matrix(x).unwrap();
                    let mut dst = aop.get_reducing_matrix_mut(x).unwrap();
                    // column-major slab [nao, ngrids]: column g is the contiguous run
                    // `g*nao .. (g+1)*nao` (same layout `decompress_aop` writes)
                    for (g_new, &g_old) in new2old.iter().enumerate() {
                        dst.data[g_new * nao..(g_new + 1) * nao]
                            .copy_from_slice(&src.data[g_old * nao..(g_old + 1) * nao]);
                    }
                }
                self.aop = Some(aop);
            }
            return true;
        }

        false
    }
    pub fn prepare_tabulated_ao_rayon_v02(&mut self, mol: &Molecule) {
        //In this subroutine, we call the lapack dgemm in a rayon parallel environment.
        //In order to ensure the efficiency, we disable the openmp ability and re-open it in the end of subroutien
        //let default_omp_num_threads = unsafe {utilities::openblas_get_num_threads()};
        //let default_omp_num_threads = utilities::omp_get_num_threads_wrapper();
        let default_omp_num_threads = mol.ctrl.num_threads.unwrap();

        let num_grids = self.coordinates.len();
        let num_basis = mol.num_basis;

        // handle memory exceed
        // AJZ: here tabulated grids will cost (num_basis * num_grids * 1 or 4) memory, depending on whether gradients are needed.
        let ncomp = if mol.xc_data.use_density_gradient() { 4 } else { 1 };
        let estimated_mem = (num_basis * num_grids * ncomp) as f64 * 8.0 / 1024.0 / 1024.0; // in MB
        let mem_avail = mol.ctrl.max_memory.map(|m| m - crate::utilities::memory_batch::detect_used_memory_mb("proc"));
        crate::utilities::memory_batch::handle_memory_exceed(estimated_mem, mem_avail, mol.ctrl.abort_on_mem_exceed);

        let mut ao = MatrixFull::new([num_basis,num_grids],0.0);
        let mut aop =  if mol.xc_data.use_density_gradient() {
            Some(RIFull::new([num_basis,num_grids,3],0.0))
        } else {
            None
        };

        let par_tasks = utilities::balancing(num_grids, rayon::current_num_threads());
        let (sender, receiver) = channel();
        par_tasks.par_iter().for_each_with(sender, |s, range_grids| {
            omp_set_num_threads_wrapper(1);

            let loc_num_grids = range_grids.len();

            let mut loc_ao = MatrixFull::new([num_basis, loc_num_grids],0.0);
            let mut loc_aop =  if mol.xc_data.use_density_gradient() {
                RIFull::new([num_basis,loc_num_grids,3],0.0)
            } else {
                RIFull::empty()
            };
            //mol.basis4elem.iter().zip(mol.geom.position.iter_columns_full()).for_each(|(elem, geom)| {
            mol.basis4elem.iter().zip(mol.geom.rg_position.iter_columns_full()).for_each(|(elem, geom)| {
                let ind_glb_bas = elem.global_index.0;
                let loc_num_bas = elem.global_index.1;
                let start = ind_glb_bas;
                let end = start + loc_num_bas;
                //let mut tmp_geom = [0.0;3];
                //tmp_geom.iter_mut().zip(geom.iter()).for_each(|value| {*value.0 = *value.1});
                let tmp_geom:[f64;3] = geom.try_into().unwrap();
                let tab_den = gto_value_serial(&self.coordinates[range_grids.clone()], &tmp_geom, elem, &mol.ctrl.basis_type);
                //println!("debug info: start: {}, end: {}, loc_num_grids: {}, loc_num_bas: {}", start, end, loc_num_grids, loc_num_bas);

                loc_ao.copy_from_matr(start..end, 0..loc_num_grids, &tab_den, 0..loc_num_bas, 0..loc_num_grids);

                if mol.xc_data.use_density_gradient() {
                    //println!("debug 01");
                    let tab_dev = gto_1st_value_serial(&self.coordinates[range_grids.clone()], &tmp_geom, elem, &mol.ctrl.basis_type);
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

        omp_set_num_threads_wrapper(default_omp_num_threads);
    }

    pub fn prepare_tabulated_ao_rayon(&mut self, mol: &Molecule) {
        //In this subroutine, we call the lapack dgemm in a rayon parallel environment.
        //In order to ensure the efficiency, we disable the openmp ability and re-open it in the end of subroutien
        let default_omp_num_threads = mol.ctrl.num_threads.unwrap();

        let num_grids = self.coordinates.len();

        let mut ao = MatrixFull::new([num_grids,mol.num_basis],0.0);
        let mut aop =  if mol.xc_data.use_density_gradient() {
            Some(RIFull::new([mol.num_basis,num_grids,3],0.0))
        } else {
            None
        };

        let (sender, receiver) = channel();
        mol.basis4elem.par_iter().zip(mol.geom.rg_position.par_iter_columns_full()).for_each_with(sender, |s, (elem,geom)| {
            omp_set_num_threads_wrapper(1);
            let ind_glb_bas = elem.global_index.0;
            let num_loc_bas = elem.global_index.1;
            //let mut tab_den = MatrixFull::new([num_grids, num_loc_bas],0.0);

            let mut tmp_geom = [0.0;3];
            tmp_geom.iter_mut().zip(geom.iter()).for_each(|value| {*value.0 = *value.1});
            let tab_den = gto_value_matrixfull_serial(&self.coordinates, &tmp_geom, elem, &mol.ctrl.basis_type);

            let tab_dev = if mol.xc_data.use_density_gradient() {
                Some(gto_1st_value_batch_serial(&self.coordinates, &tmp_geom, elem, &mol.ctrl.basis_type))
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

        omp_set_num_threads_wrapper(default_omp_num_threads);

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
        mol.basis4elem.iter().zip(mol.geom.rg_position.iter_columns_full()).for_each(|(elem,geom)| {
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
            mol.basis4elem.iter().zip(mol.geom.rg_position.iter_columns_full()).for_each(|(elem,geom)| {
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
        //let default_omp_num_threads = omp_get_num_threads_wrapper();
        //omp_set_num_threads_wrapper(1);
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
        //utilities::omp_set_num_threads_wrapper(default_omp_num_threads);
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
                //NOTE: here assume that the molecular obitals have been orderd: occupation first, then virtual.
                //        which, however, is wrong for the dSCF calculation with forced occupation.
                //let mut occ_s = occ.get(i_spin).unwrap()
                //    .iter().filter(|occ| **occ>0.0).map(|occ| occ.sqrt()).collect_vec();
                //==================================
                //NOTE: now locate the highest obital that has electron with occupation largger than 1.0e-4
                //let homo_s = occ[i_spin].iter().enumerate().fold(0_usize,|x, (ob, occ)| {if *occ>1.0e-4 {ob} else {x}});
                let homo_s  = occ[i_spin].iter().enumerate()
                    .filter(|(i,occ)| **occ >=1.0e-6)
                    .map(|(i,occ)| i).max();
                let mut occ_s = if let Some(homo_s) = homo_s {
                    occ.get(i_spin).unwrap()[0..homo_s+1].iter().map(|occ| occ.sqrt()).collect::<Vec<f64>>()
                } else {
                    // In this case, no electrons in the i_spin channel, for which homo_s = None
                    vec![]
                };
                //==================================
                let num_occ = occ_s.len();
                // wmo = weighted mo ('ij,j->ij'): mo_s(ij), occ_s(j) -> wmo(ij)
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
                // wmo = weighted mo ('ij,j->ij'): mo_s(ij), occ_s(j) -> wmo(ij)
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

    pub fn prepare_tabulated_density_3(&self, mo: &[MatrixFull<f64>;2], occ: &[Vec<f64>;2], spin_channel: usize) -> (MatrixFull<f64>, RIFull<f64>, MatrixFull<f64>) {
        let mut cur_rho = MatrixFull::new([self.coordinates.len(),spin_channel],0.0);
        let num_grids = self.coordinates.len();
        let num_basis = mo[0].size.get(0).unwrap();
        let num_state = mo[0].size.get(1).unwrap();
        if let (Some(ao), Some(aop)) = (&self.ao, &self.aop) {
            let mut cur_rhop = RIFull::new([num_grids, 3, spin_channel],0.0);
            let mut cur_tau = MatrixFull::new([num_grids, spin_channel],0.0);
            for i_spin in 0..spin_channel {
                let mo_s = mo.get(i_spin).unwrap();
                let homo_s  = occ[i_spin].iter().enumerate()
                    .filter(|(i,occ)| **occ >=1.0e-6)
                    .map(|(i,occ)| i).max().unwrap();
                let mut occ_s = occ.get(i_spin).unwrap()[0..homo_s+1].iter().map(|occ| occ.sqrt()).collect::<Vec<f64>>();
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
                    // nabla rho
                    let rhopi_s = _einsum_02_rayon(&tmop.to_matrixfullslice(), &tmo.to_matrixfullslice());
                    cur_rhop.get_reducing_matrix_mut(i_spin).unwrap() // spin
                    .par_iter_mut_j(i) // x, y, z
                    .zip(rhopi_s.par_iter()) // nabla_rho = phi_i * nabla_phi_i + nabla_phi_i * phi_i
                    .for_each(
                        |(to, from)| {*to = *from*2.0}
                    );
                    // tau = 1/2 * |nabla phi|^2 over x, y, z 
                    let tau_s = _einsum_02_rayon(&tmop.to_matrixfullslice(), &tmop.to_matrixfullslice());
                    cur_tau.par_iter_column_mut(i_spin)
                    .zip(tau_s.par_iter())
                    .for_each(
                        |(to, from)| {*to += *from*0.5}
                    );
                }
            };
            return (cur_rho, cur_rhop, cur_tau)
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
            let mut cur_tau = MatrixFull::empty();
            return (cur_rho, cur_rhop, cur_tau)
        }
        let cur_rhop = RIFull::empty();
        let cur_tau = MatrixFull::empty();
        (cur_rho, cur_rhop, cur_tau)
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

    /// Compressed version: density from DM using compressed AO/AOP storage.
    pub fn prepare_tabulated_density_slots_dm_only_compressed(
        &self,
        dm: &Vec<MatrixFull<f64>>,
        spin_channel: usize,
        range_grids: Range<usize>,
    ) -> (MatrixFull<f64>, RIFull<f64>) {
        let ao_c = self.ao_compressed.as_ref()
            .expect("compressed AO must be built before calling compressed density");
        let aop_c = self.aop_compressed.as_ref();
        let nao = dm[0].size[0];
        let n_grids_total = range_grids.len();

        let ibatch_start = range_grids.start / ao_c.blksize;
        let ibatch_end = ((range_grids.end + ao_c.blksize - 1) / ao_c.blksize).min(ao_c.batches.len());

        // ---- rho ----
        let mut cur_rho = MatrixFull::new([n_grids_total, spin_channel], 0.0);
        for ibatch in ibatch_start..ibatch_end {
            let batch_ao = &ao_c.batches[ibatch];
            let indices = &ao_c.batch_ao_map[ibatch];
            let n_active = indices.len();
            if n_active == 0 { continue; }
            let g_range = &ao_c.batch_grid_ranges[ibatch];
            let g_local_start = (range_grids.start.saturating_sub(g_range.start));
            let g_local_end = (range_grids.end.min(g_range.end) - g_range.start).min(batch_ao.size[1]);
            if g_local_end <= g_local_start { continue; }
            let n_batch = g_local_end - g_local_start;
            let g_out_start = g_range.start + g_local_start - range_grids.start;

            // The contraction runs over the batch's active AO set only, so both
            // indices of `dm_sub` are restricted to it: the gather and the GEMM
            // shrink by `nao / n_active` with no change to the summed expression
            // (the rows outside the active set were computed but never read).
            let mut dm_sub = vec![MatrixFull::new([n_active, n_active], 0.0); spin_channel];
            for i_spin in 0..spin_channel {
                let dm_s = &dm[i_spin];
                let dm_sub_s = &mut dm_sub[i_spin];
                for (j_local, &nu_global) in indices.iter().enumerate() {
                    for (i_local, &mu_global) in indices.iter().enumerate() {
                        dm_sub_s[[i_local, j_local]] = dm_s[[mu_global, nu_global]];
                    }
                }
            }

            for i_spin in 0..spin_channel {
                let mut wao = MatrixFull::new([n_active, n_batch], 0.0);
                _dgemm(
                    &dm_sub[i_spin], (0..n_active, 0..n_active), 'N',
                    batch_ao, (0..n_active, g_local_start..g_local_end), 'N',
                    &mut wao, (0..n_active, 0..n_batch), 1.0, 0.0,
                );

                for g in 0..n_batch {
                    let mut rho_val = 0.0;
                    for i_local in 0..n_active {
                        rho_val += batch_ao[[i_local, g_local_start + g]] * wao[[i_local, g]];
                    }
                    cur_rho[[g_out_start + g, i_spin]] = rho_val;
                }
            }
        }

        // ---- rhop (gradient) ----
        let need_rhop = aop_c.is_some() && spin_channel > 0;
        let mut cur_rhop = if need_rhop {
            RIFull::new([n_grids_total, 3, spin_channel], 0.0)
        } else {
            RIFull::empty()
        };

        if let Some(aop_c) = aop_c {
            if need_rhop && !aop_c.batch_aop_map.is_empty() {
                for ibatch in ibatch_start..ibatch_end {
                    let aop_indices = &aop_c.batch_aop_map[ibatch];
                    let n_active_aop = aop_indices.len();
                    if n_active_aop == 0 { continue; }
                    let g_range = &ao_c.batch_grid_ranges[ibatch];
                    let g_local_start = (range_grids.start.saturating_sub(g_range.start));
                    let g_local_end = (range_grids.end.min(g_range.end) - g_range.start)
                        .min(aop_c.batches.get(ibatch).map(|b| b[0].size[1]).unwrap_or(0));
                    if g_local_end <= g_local_start { continue; }
                    let n_batch = g_local_end - g_local_start;
                    let g_out_start = g_range.start + g_local_start - range_grids.start;

                    let ao_batch = &ao_c.batches[ibatch];
                    let ao_indices = &ao_c.batch_ao_map[ibatch];
                    let n_active_ao = ao_indices.len();

                    // same row/column restriction as the ρ part: rows are the set the
                    // final contraction sums over (`ao_indices`), columns are the AOP
                    // block's active set (`aop_indices`, equal to `ao_indices` by the
                    // row-aligned storage).
                    let mut dm_sub = vec![MatrixFull::new([n_active_ao, n_active_aop], 0.0); spin_channel];
                    for i_spin in 0..spin_channel {
                        let dm_s = &dm[i_spin];
                        for (j_local, &nu_global) in aop_indices.iter().enumerate() {
                            for (i_local, &mu_global) in ao_indices.iter().enumerate() {
                                dm_sub[i_spin][[i_local, j_local]] = dm_s[[mu_global, nu_global]];
                            }
                        }
                    }

                    for i_spin in 0..spin_channel {
                        for x in 0usize..3usize {
                            let aop_x_batch = &aop_c.batches[ibatch][x];
                            let mut wao = MatrixFull::new([n_active_ao, n_batch], 0.0);
                            _dgemm(
                                &dm_sub[i_spin], (0..n_active_ao, 0..n_active_aop), 'N',
                                aop_x_batch, (0..n_active_aop, g_local_start..g_local_end), 'N',
                                &mut wao, (0..n_active_ao, 0..n_batch), 1.0, 0.0,
                            );

                            let mut rhop_s = cur_rhop.get_reducing_matrix_mut(i_spin).unwrap();
                            for g in 0..n_batch {
                                let mut rhop_val = 0.0;
                                for i_local in 0..n_active_ao {
                                    rhop_val += ao_batch[[i_local, g_local_start + g]] * wao[[i_local, g]];
                                }
                                let col = rhop_s.get_column_mut(x);
                                col[g_out_start + g] = 2.0 * rhop_val;
                            }
                        }
                    }
                }
            }
        }

        (cur_rho, cur_rhop)
    }

    pub fn prepare_tabulated_density_2_slots_dm_only(
        &self, 
        dm:&Vec<MatrixFull<f64>>, 
        spin_channel: usize, 
        order: usize, 
        range_grids: Range<usize>
    ) -> (MatrixFull<f64>, RIFull<f64>, MatrixFull<f64>) 
    {
        /*
        Args:
            dm: density matrix <[num_basis, num_basis]; spin_channel>
            spin_channel: spin
            order: order of the density
                0 for LDA
                1 for GGA or Hybrid-GGA
                2 for MGGA or Hybrid-MGGA
            range_grids: range of grids
        Return:
            rho [num_grids, spin_channel]
            rhop [num_grids, 3, spin_channel]
            tau [num_grids, spin_channel]
        Note:
            Laplacian is not supported here. 
        */
        let num_grids = range_grids.len();
        let num_basis = dm[0].size[0];
        
        let ao = self.ao.as_ref().unwrap();
        let aop = if order >=1 {
            self.aop.as_ref().unwrap()
        } else {
            &RIFull::empty()
        };

        let mut cur_rho = MatrixFull::empty();
        let mut cur_rhop = RIFull::empty();
        let mut cur_tau = MatrixFull::empty();
        if order == 2 {
            cur_rho = MatrixFull::new([num_grids, spin_channel],0.0);
            cur_rhop = RIFull::new([num_grids,3,spin_channel],0.0);
            cur_tau = MatrixFull::new([num_grids, spin_channel],0.0);
        } else if order == 1 {
            cur_rho = MatrixFull::new([num_grids, spin_channel],0.0);
            cur_rhop = RIFull::new([num_grids,3,spin_channel],0.0);
        } else if order == 0 {
            cur_rho = MatrixFull::new([num_grids, spin_channel],0.0);
        } else {
            panic!("order {} is not supported in prepare_tabulated_density_2", order);
        }
        
        for i_spin in 0..spin_channel {
            // rho 
            let dm_s = &dm[i_spin];
            let mut wao = MatrixFull::new([num_basis, num_grids],0.0);
            _dgemm(
                dm_s, (0..num_basis, 0..num_basis), 'N',
                ao, (0..num_basis, range_grids.clone()), 'N',
                &mut wao,  (0..num_basis, 0..num_grids), 
                1.0, 0.0
            );
            ao.iter_columns(range_grids.clone())
            .zip(wao.iter_columns_full())
            .map(|(ao_r,wao_r)| (ao_r,wao_r))
            .zip(cur_rho.iter_column_mut(i_spin))
            .for_each(
                |((ao_r, wao_r), cur_rho_s)| 
                {
                    *cur_rho_s = wao_r.iter().zip(ao_r.iter()).fold(0.0, |acc, (a,b)| {acc + a*b})
                }
            );
            if order >= 1 {
                let mut rhop_s = cur_rhop.get_reducing_matrix_mut(i_spin).unwrap();
                for ic in 0..3usize {
                    let mut aop_i = aop.get_reducing_matrix(ic).unwrap();
                    let mut wao = MatrixFull::new([num_basis, num_grids],0.0);
                    _dgemm(
                        dm_s, (0..num_basis, 0..num_basis), 'N',
                        &aop_i, (0..num_basis, range_grids.clone()), 'N',
                        &mut wao, (0..num_basis, 0..num_grids), 1.0, 0.0
                    );
                    //wao.to_matrixfullslicemut().lapack_dgemm(&dm.to_matrixfullslice(),&aop_i, 'N','N', 1.0, 0.0);
                    ao.iter_columns(range_grids.clone())
                    .zip(wao.iter_columns_full())
                    .map(|(ao_r, wao_r)| (ao_r, wao_r))
                    .zip(rhop_s.iter_mut_j(ic))
                    .for_each(
                        |((ao_r,wao_r), cur_rhop_r)| 
                        {
                            *cur_rhop_r = 2.0 * wao_r.iter().zip(ao_r.iter()).fold(0.0, |acc,(wao,ao)| {acc + wao*ao})
                        }
                    );
                    if order == 2 {
                        aop_i.iter_columns(range_grids.clone()).unwrap()
                        .zip(wao.iter_columns_full())
                        .map(|(aop_r, wao_r)|(aop_r, wao_r))
                        .zip(cur_tau.iter_column_mut(i_spin))
                        .for_each(
                            |((aop_r, wao_r), cur_tau_s)|
                            {
                                *cur_tau_s += 0.5 * wao_r.iter().zip(aop_r.iter()).fold(0.0, |acc, (a,b)| {acc + a*b})
                            }
                        );
                    }
                }
            }
        } // end spin case 
        (cur_rho, cur_rhop, cur_tau)
    }

    /// Compressed version of `prepare_tabulated_density_2_slots_dm_only`.
    pub fn prepare_tabulated_density_2_slots_dm_only_compressed(
        &self,
        dm: &Vec<MatrixFull<f64>>,
        spin_channel: usize,
        order: usize,
        range_grids: Range<usize>,
    ) -> (MatrixFull<f64>, RIFull<f64>, MatrixFull<f64>) {
        if order > 2 {
            panic!("order {} is not supported in prepare_tabulated_density_2_slots_dm_only_compressed", order);
        }

        let ao_c = self
            .ao_compressed
            .as_ref()
            .expect("compressed AO must be built before calling prepare_tabulated_density_2_slots_dm_only_compressed");
        let aop_c = if order >= 1 {
            Some(
                self.aop_compressed
                    .as_ref()
                    .expect("compressed AOP must be built before calling order>=1 compressed density"),
            )
        } else {
            None
        };

        let nao = dm[0].size[0];
        let n_grids_total = range_grids.len();

        let mut cur_rho = MatrixFull::new([n_grids_total, spin_channel], 0.0);
        let mut cur_rhop = if order >= 1 {
            RIFull::new([n_grids_total, 3, spin_channel], 0.0)
        } else {
            RIFull::empty()
        };
        let mut cur_tau = if order == 2 {
            MatrixFull::new([n_grids_total, spin_channel], 0.0)
        } else {
            MatrixFull::empty()
        };

        let ibatch_start = range_grids.start / ao_c.blksize;
        let ibatch_end = ((range_grids.end + ao_c.blksize - 1) / ao_c.blksize).min(ao_c.batches.len());

        for ibatch in ibatch_start..ibatch_end {
            let ao_batch = &ao_c.batches[ibatch];
            let ao_indices = &ao_c.batch_ao_map[ibatch];
            let n_active_ao = ao_indices.len();
            if n_active_ao == 0 {
                continue;
            }

            let g_range = &ao_c.batch_grid_ranges[ibatch];
            let g_local_start = range_grids.start.saturating_sub(g_range.start);
            let mut g_local_end = (range_grids.end.min(g_range.end) - g_range.start).min(ao_batch.size[1]);

            let mut n_active_aop = 0usize;
            let mut aop_indices_opt: Option<&Vec<usize>> = None;
            let mut aop_batches_opt: Option<&[MatrixFull<f64>; 3]> = None;

            if let Some(aop_c) = aop_c {
                if ibatch < aop_c.batches.len() && ibatch < aop_c.batch_aop_map.len() {
                    aop_indices_opt = Some(&aop_c.batch_aop_map[ibatch]);
                    aop_batches_opt = Some(&aop_c.batches[ibatch]);
                    if let Some(aop_batches) = aop_batches_opt {
                        g_local_end = g_local_end.min(aop_batches[0].size[1]);
                    }
                    n_active_aop = aop_indices_opt.map(|x| x.len()).unwrap_or(0);
                }
            }

            if g_local_end <= g_local_start {
                continue;
            }

            let n_batch = g_local_end - g_local_start;
            let g_out_start = g_range.start + g_local_start - range_grids.start;

            // Both indices are restricted to the batch's active set (see the note in
            // `prepare_tabulated_density_slots_dm_only_compressed`); the AOP block is
            // row-aligned with the AO block, so its rows use `ao_indices` as well.
            let mut dm_sub_ao = vec![MatrixFull::new([n_active_ao, n_active_ao], 0.0); spin_channel];
            for i_spin in 0..spin_channel {
                let dm_s = &dm[i_spin];
                for (j_local, &nu_global) in ao_indices.iter().enumerate() {
                    for (i_local, &mu_global) in ao_indices.iter().enumerate() {
                        dm_sub_ao[i_spin][[i_local, j_local]] = dm_s[[mu_global, nu_global]];
                    }
                }
            }

            let mut dm_sub_aop: Option<Vec<MatrixFull<f64>>> = None;
            if order >= 1 {
                if let Some(aop_indices) = aop_indices_opt {
                    if n_active_aop > 0 {
                        debug_assert_eq!(n_active_aop, n_active_ao);
                        let mut tmp = vec![MatrixFull::new([n_active_ao, n_active_aop], 0.0); spin_channel];
                        for i_spin in 0..spin_channel {
                            let dm_s = &dm[i_spin];
                            for (j_local, &nu_global) in aop_indices.iter().enumerate() {
                                for (i_local, &mu_global) in ao_indices.iter().enumerate() {
                                    tmp[i_spin][[i_local, j_local]] = dm_s[[mu_global, nu_global]];
                                }
                            }
                        }
                        dm_sub_aop = Some(tmp);
                    }
                }
            }

            for i_spin in 0..spin_channel {
                // rho
                let mut wao = MatrixFull::new([n_active_ao, n_batch], 0.0);
                _dgemm(
                    &dm_sub_ao[i_spin],
                    (0..n_active_ao, 0..n_active_ao),
                    'N',
                    ao_batch,
                    (0..n_active_ao, g_local_start..g_local_end),
                    'N',
                    &mut wao,
                    (0..n_active_ao, 0..n_batch),
                    1.0,
                    0.0,
                );

                for g in 0..n_batch {
                    let mut rho_val = 0.0;
                    for i_local in 0..n_active_ao {
                        rho_val += ao_batch[[i_local, g_local_start + g]] * wao[[i_local, g]];
                    }
                    cur_rho[[g_out_start + g, i_spin]] = rho_val;
                }

                if order >= 1 {
                    if let (Some(aop_batches), Some(aop_indices), Some(dm_sub_aop)) =
                        (aop_batches_opt, aop_indices_opt, dm_sub_aop.as_ref())
                    {
                        for x in 0usize..3usize {
                            let aop_x_batch = &aop_batches[x];
                            let mut wao_p = MatrixFull::new([n_active_ao, n_batch], 0.0);
                            _dgemm(
                                &dm_sub_aop[i_spin],
                                (0..n_active_ao, 0..n_active_aop),
                                'N',
                                aop_x_batch,
                                (0..n_active_aop, g_local_start..g_local_end),
                                'N',
                                &mut wao_p,
                                (0..n_active_ao, 0..n_batch),
                                1.0,
                                0.0,
                            );

                            {
                                let mut rhop_s = cur_rhop.get_reducing_matrix_mut(i_spin).unwrap();
                                let col = rhop_s.get_column_mut(x);
                                for g in 0..n_batch {
                                    let mut rhop_val = 0.0;
                                    for i_local in 0..n_active_ao {
                                        rhop_val += ao_batch[[i_local, g_local_start + g]] * wao_p[[i_local, g]];
                                    }
                                    col[g_out_start + g] = 2.0 * rhop_val;
                                }
                            }

                            if order == 2 {
                                for g in 0..n_batch {
                                    let mut tau_val = 0.0;
                                    for i_local in 0..n_active_aop {
                                        tau_val += aop_x_batch[[i_local, g_local_start + g]] * wao_p[[i_local, g]];
                                    }
                                    cur_tau[[g_out_start + g, i_spin]] += 0.5 * tau_val;
                                }
                            }
                        }
                    }
                }
            }
        }

        (cur_rho, cur_rhop, cur_tau)
    }

    pub fn prepare_tabulated_density_ensemble_slots(
        &self, 
        xc_method: &DFA4REST, 
        mo: &[MatrixFull<f64>; 2], 
        occ: &[Vec<f64>; 2], 
        spin_channel: usize, 
        range_grids: Range<usize>
    ) -> RIFull<f64> 
    {
        /*
            Args:
                mo: orbital coeffients [num_basis, num_state]
                occ: occupation number [num_state]
                spin_channel: spin
                range_grids: range of grids
            Return:
                rho [num_grids, spin_channel, nvar]
                where nvar = 1 for LDA, 4 for GGA, 6 for MGGA 
                tabulated as rho, rho_x, rho_y, rho_z, laplacian, tau
            Note:
                currently, laplacian is not implemented and is set to zero
        */
        let num_grids = range_grids.len();
        let num_basis = mo[0].size.get(0).unwrap();
        let num_state = mo[0].size.get(1).unwrap();

        // let nvar = match self.get_family_name() {
        //     "LDA".to_string() => 1,
        //     "GGA".to_string() => 4,
        //     "MGGA".to_string() => 6,
        //     "HybridGGA".to_string() => 4,
        //     "HybridMGGA".to_string() => 6,
        //     _ => 1,
        // }
        let mut nvar = 1;
        let do_mgga = xc_method.use_kinetic_density();
        let do_gga = xc_method.use_density_gradient();
        if do_mgga {
            nvar = 6;
        } else if do_gga {
            nvar = 4;
        } else {
            nvar = 1;
        }
        let mut cur_rho = RIFull::new([num_grids, spin_channel, nvar], 0.0);
        let ao = self.ao.as_ref().unwrap();
        let aop = if xc_method.use_density_gradient() {
            self.aop.as_ref().unwrap()
        } else {
            &RIFull::empty()
        };
        for i_spin in 0..spin_channel {
            let mo_s = mo.get(i_spin).unwrap();
            let homo_s = occ[i_spin].iter()
                    .enumerate()
                    .filter(|(i,occ)| **occ >=1.0e-6)
                    .map(|(i,occ)| i).max();
            let mut occ_s = if let Some(homo_s) = homo_s {
                occ.get(i_spin).unwrap()[0..homo_s+1].iter().map(|occ| occ.sqrt()).collect::<Vec<f64>>()
            } else {
                // In this case, no electrons in the i_spin channel, for which homo_s = None
                vec![]
            };
            let num_occ = occ_s.len();
            let mut wmo = _einsum_01_serial(&mo_s.to_matrixfullslice(), &occ_s);
            let mut tmo = MatrixFull::new([num_occ, num_grids], 0.0);
            // tmo = C.T matmul ao (half transform)
            _dgemm(
                &wmo, (0..wmo.size[0], 0..wmo.size[1]), 'T', 
                ao, (0..ao.size[0], range_grids.clone()), 'N',
                &mut tmo, (0..wmo.size[1], 0..num_grids),
                1.0, 0.0
            );
            // spin case: rho_s = tmo * tmo (C_ug * C_vg * phi_ug * phi_vg => rho_g) 
            let rho_s = _einsum_02_serial(&tmo.to_matrixfullslice(), &tmo.to_matrixfullslice());
            // cur_rho[0][ispin] = rho_s
            cur_rho.get_reducing_matrix_mut(0).unwrap()
            .iter_mut_j(i_spin)
            .zip(rho_s.iter())
            .for_each(
                |(to, from)| {*to = *from}
            );
            if do_gga {
                for ic in 0..3 { // x, y, z
                    let mut tmop = MatrixFull::new([num_occ, num_grids], 0.0);
                    let aop_ic = aop.get_reducing_matrix(ic).unwrap();
                    _dgemm(
                        &wmo, (0..wmo.size[0], 0..wmo.size[1]), 'T',
                        &aop_ic, (0..aop_ic.size[0], range_grids.clone()), 'N',
                        &mut tmop, (0..wmo.size[1], 0..num_grids),
                        1.0, 0.0
                    );
                    let rhop_ic_s = _einsum_02_serial(&tmop.to_matrixfullslice(), &tmo.to_matrixfullslice());
                    // cur_rho[ic+1][ispin] = rhop_ic_s
                    cur_rho.get_reducing_matrix_mut(ic + 1).unwrap()
                    .iter_mut_j(i_spin)
                    .zip(rhop_ic_s.iter())
                    .for_each(
                        |(to, from)| {*to = *from * 2.0}
                    );
                    if do_mgga {
                        // tau = 1/2 * |grad phi|^2 sum over x,y,z
                        let tau_s = _einsum_02_serial(&tmop.to_matrixfullslice(), &tmop.to_matrixfullslice());
                        // cur_rho[5][ispin] = tau_s
                        cur_rho.get_reducing_matrix_mut(5).unwrap()
                        .iter_mut_j(i_spin)
                        .zip(tau_s.iter())
                        .for_each(
                            |(to, from)| {*to += *from*0.5}
                        );
                    }
                }
            }
        }
        cur_rho 
    }

    /// Compressed version of `prepare_tabulated_density_ensemble_slots`.
    pub fn prepare_tabulated_density_ensemble_slots_compressed(
        &self,
        xc_method: &DFA4REST,
        mo: &[MatrixFull<f64>; 2],
        occ: &[Vec<f64>; 2],
        spin_channel: usize,
        range_grids: Range<usize>,
    ) -> RIFull<f64> {
        let num_grids = range_grids.len();

        let do_mgga = xc_method.use_kinetic_density();
        let do_gga = xc_method.use_density_gradient();
        let nvar = if do_mgga {
            6
        } else if do_gga {
            4
        } else {
            1
        };

        let ao_c = self.ao_compressed
            .as_ref()
            .expect("compressed AO must be built before calling prepare_tabulated_density_ensemble_slots_compressed");
        let aop_c = if do_gga || do_mgga {
            Some(
                self.aop_compressed
                    .as_ref()
                    .expect("compressed AOP must be built before calling GGA/MGGA compressed ensemble density"),
            )
        } else {
            None
        };

        let mut cur_rho = RIFull::new([num_grids, spin_channel, nvar], 0.0);

        // The occupation weights and the scaled MO coefficients do not depend on the
        // grid batch, so build them once per spin instead of in every batch.
        let nao = mo[0].size[0];
        let wmo_scaled: Vec<(Vec<f64>, MatrixFull<f64>)> = (0..spin_channel)
            .map(|i_spin| {
                let mo_s = &mo[i_spin];
                let homo_s = occ[i_spin]
                    .iter()
                    .enumerate()
                    .filter(|(_, occ)| **occ >= 1.0e-6)
                    .map(|(i, _)| i)
                    .max();
                let occ_s: Vec<f64> = if let Some(homo_s) = homo_s {
                    occ[i_spin][0..homo_s + 1].iter().map(|occ| occ.sqrt()).collect()
                } else {
                    vec![]
                };
                let nocc = occ_s.len();
                let mut mo_scaled = MatrixFull::new([nao, nocc], 0.0);
                for mu in 0..nao {
                    for j in 0..nocc {
                        mo_scaled[[mu, j]] = mo_s[[mu, j]] * occ_s[j];
                    }
                }
                (occ_s, mo_scaled)
            })
            .collect();

        let ibatch_start = range_grids.start / ao_c.blksize;
        let ibatch_end = ((range_grids.end + ao_c.blksize - 1) / ao_c.blksize).min(ao_c.batches.len());

        for ibatch in ibatch_start..ibatch_end {
            let ao_batch = &ao_c.batches[ibatch];
            let ao_indices = &ao_c.batch_ao_map[ibatch];
            let n_active_ao = ao_indices.len();
            if n_active_ao == 0 {
                continue;
            }

            let g_range = &ao_c.batch_grid_ranges[ibatch];
            let g_local_start = range_grids.start.saturating_sub(g_range.start);
            let mut g_local_end = (range_grids.end.min(g_range.end) - g_range.start).min(ao_batch.size[1]);

            let mut n_active_aop = 0usize;
            let mut aop_indices_opt: Option<&Vec<usize>> = None;
            let mut aop_batches_opt: Option<&[MatrixFull<f64>; 3]> = None;

            if let Some(aop_c) = aop_c {
                if ibatch < aop_c.batches.len() && ibatch < aop_c.batch_aop_map.len() {
                    aop_indices_opt = Some(&aop_c.batch_aop_map[ibatch]);
                    aop_batches_opt = Some(&aop_c.batches[ibatch]);
                    if let Some(aop_batches) = aop_batches_opt {
                        g_local_end = g_local_end.min(aop_batches[0].size[1]);
                    }
                    n_active_aop = aop_indices_opt.map(|x| x.len()).unwrap_or(0);
                }
            }

            if g_local_end <= g_local_start {
                continue;
            }

            let n_batch = g_local_end - g_local_start;
            let g_out_start = g_range.start + g_local_start - range_grids.start;

            for i_spin in 0..spin_channel {
                let (occ_s, mo_scaled) = &wmo_scaled[i_spin];
                let nocc = occ_s.len();
                if nocc == 0 {
                    continue;
                }

                // wmo_ao[n_active_ao, nocc] = mo_sub × sqrt(occ), pre-scaled per spin
                let mut wmo_ao = MatrixFull::new([n_active_ao, nocc], 0.0);
                for (i_local, &mu_global) in ao_indices.iter().enumerate() {
                    for j in 0..nocc {
                        wmo_ao[[i_local, j]] = mo_scaled[[mu_global, j]];
                    }
                }

                // tmo[nocc, n_batch] = wmo_ao^T × ao_batch
                let mut tmo = MatrixFull::new([nocc, n_batch], 0.0);
                _dgemm(
                    &wmo_ao,
                    (0..n_active_ao, 0..nocc),
                    'T',
                    ao_batch,
                    (0..n_active_ao, g_local_start..g_local_end),
                    'N',
                    &mut tmo,
                    (0..nocc, 0..n_batch),
                    1.0,
                    0.0,
                );

                {
                    let mut rho_mat = cur_rho.get_reducing_matrix_mut(0).unwrap();
                    let rho_col = rho_mat.get_column_mut(i_spin);
                    for g in 0..n_batch {
                        let rho_val: f64 = (0..nocc).map(|i| tmo[[i, g]] * tmo[[i, g]]).sum();
                        rho_col[g_out_start + g] = rho_val;
                    }
                }

                if do_gga {
                    if let (Some(aop_batches), Some(aop_indices)) = (aop_batches_opt, aop_indices_opt) {
                        if n_active_aop == 0 {
                            continue;
                        }

                        // The AOP block is row-aligned with the AO block, so it shares
                        // the same scaled MO matrix; `wmo_ao` is reused rather than
                        // rebuilding a duplicate here.
                        debug_assert_eq!(aop_indices.len(), ao_indices.len());

                        for ic in 0usize..3usize {
                            let aop_ic_batch = &aop_batches[ic];
                            let mut tmop = MatrixFull::new([nocc, n_batch], 0.0);
                            _dgemm(
                                &wmo_ao,
                                (0..n_active_ao, 0..nocc),
                                'T',
                                aop_ic_batch,
                                (0..n_active_aop, g_local_start..g_local_end),
                                'N',
                                &mut tmop,
                                (0..nocc, 0..n_batch),
                                1.0,
                                0.0,
                            );

                            {
                                let mut rhop_mat = cur_rho.get_reducing_matrix_mut(ic + 1).unwrap();
                                let rhop_col = rhop_mat.get_column_mut(i_spin);
                                for g in 0..n_batch {
                                    let rhop_val: f64 = (0..nocc)
                                        .map(|i| tmop[[i, g]] * tmo[[i, g]])
                                        .sum();
                                    rhop_col[g_out_start + g] = 2.0 * rhop_val;
                                }
                            }

                            if do_mgga {
                                let mut tau_mat = cur_rho.get_reducing_matrix_mut(5).unwrap();
                                let tau_col = tau_mat.get_column_mut(i_spin);
                                for g in 0..n_batch {
                                    let tau_val: f64 = (0..nocc)
                                        .map(|i| tmop[[i, g]] * tmop[[i, g]])
                                        .sum();
                                    tau_col[g_out_start + g] += 0.5 * tau_val;
                                }
                            }
                        }
                    }
                }
            }
        }

        cur_rho
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
                // NOTE:: here assume that the molecular obitals have been orderd: occupation first, then virtual,
                //        which, however, is wrong for the dSCF calculation with forced occupation.
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
                    // In this case, no electrons in the i_spin channel, for which homo_s = None
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

    /// Compressed version: density from MO coefficients using compressed AO/AOP storage.
    /// Coefficient path (REST default): wmo = C × √occ, tmo = wmo^T × ao, ρ = Σ|tmo|²
    pub fn prepare_tabulated_density_slots_compressed(
        &self,
        mo: &[MatrixFull<f64>; 2],
        occ: &[Vec<f64>; 2],
        spin_channel: usize,
        range_grids: Range<usize>,
    ) -> (MatrixFull<f64>, RIFull<f64>) {
        let ao_c = self.ao_compressed.as_ref()
            .expect("compressed AO must be built first");
        let aop_c = self.aop_compressed.as_ref();
        let nao = mo[0].size[0];
        let n_grids_total = range_grids.len();

        let ibatch_start = range_grids.start / ao_c.blksize;
        let ibatch_end = ((range_grids.end + ao_c.blksize - 1) / ao_c.blksize).min(ao_c.batches.len());

        // The occupation weights and the scaled MO coefficients do not depend on the
        // grid batch, so build them once per spin instead of in every batch.
        let wmo_scaled: Vec<(Vec<f64>, MatrixFull<f64>)> = (0..spin_channel)
            .map(|i_spin| {
                let mo_s = &mo[i_spin];
                let homo_s = occ[i_spin].iter().enumerate()
                    .filter(|(_, o)| **o >= 1.0e-6).map(|(i, _)| i).max();
                let occ_s: Vec<f64> = if let Some(h) = homo_s {
                    occ[i_spin][0..=h].iter().map(|o| o.sqrt()).collect()
                } else { vec![] };
                let nocc = occ_s.len();
                let mut mo_scaled = MatrixFull::new([nao, nocc], 0.0);
                for mu in 0..nao {
                    for j in 0..nocc {
                        mo_scaled[[mu, j]] = mo_s[[mu, j]] * occ_s[j];
                    }
                }
                (occ_s, mo_scaled)
            })
            .collect();

        // ρ and ∇ρ are assembled in a single pass over the batches: the scaled-MO
        // matrix `wmo` and the intermediate `tmo = wmoᵀ·ao` are shared by both, and
        // the AOP block is row-aligned with the AO block so its index list and its
        // scaled MO matrix are the same. The pre-merge code ran a second loop over
        // the batches for ∇ρ and rebuilt both intermediates there (measured: −26% of
        // the density/∇ρ assembly on (H2O)30 / B3LYP / def2-SVP / 4 threads).
        let need_rhop = aop_c.is_some() && spin_channel > 0;
        let mut cur_rho = MatrixFull::new([n_grids_total, spin_channel], 0.0);
        let mut cur_rhop = if need_rhop {
            RIFull::new([n_grids_total, 3, spin_channel], 0.0)
        } else { RIFull::empty() };

        for ibatch in ibatch_start..ibatch_end {
            let batch_ao = &ao_c.batches[ibatch];
            let indices = &ao_c.batch_ao_map[ibatch];
            let n_active = indices.len();
            if n_active == 0 { continue; }
            let g_range = &ao_c.batch_grid_ranges[ibatch];
            let g_local_start = (range_grids.start.saturating_sub(g_range.start));
            let g_local_end = (range_grids.end.min(g_range.end) - g_range.start).min(batch_ao.size[1]);
            if g_local_end <= g_local_start { continue; }
            let n_batch = g_local_end - g_local_start;
            let g_out_start = g_range.start + g_local_start - range_grids.start;

            for i_spin in 0..spin_channel {
                let (occ_s, mo_scaled) = &wmo_scaled[i_spin];
                let nocc = occ_s.len();
                if nocc == 0 { continue; }

                // wmo[n_active, nocc] = mo_sub × √occ (pre-scaled per spin)
                // Both matrices are column-major: running the inner index over
                // `indices` keeps the source and the destination sequential.
                let mut wmo = MatrixFull::new([n_active, nocc], 0.0);
                for j in 0..nocc {
                    for (i_local, &mu_global) in indices.iter().enumerate() {
                        wmo[[i_local, j]] = mo_scaled[[mu_global, j]];
                    }
                }

                // tmo[nocc, n_batch] = wmo^T × batch_ao
                let mut tmo = MatrixFull::new([nocc, n_batch], 0.0);
                _dgemm(&wmo, (0..n_active, 0..nocc), 'T',
                       batch_ao, (0..n_active, g_local_start..g_local_end), 'N',
                       &mut tmo, (0..nocc, 0..n_batch), 1.0, 0.0);

                for g in 0..n_batch {
                    let rho_val: f64 = (0..nocc).map(|i| tmo[[i, g]] * tmo[[i, g]]).sum();
                    cur_rho[[g_out_start + g, i_spin]] = rho_val;
                }

                // ---- ∇ρ from the same wmo/tmo ----
                if need_rhop {
                    let aop_c = aop_c.unwrap();
                    let n_active_aop = aop_c.batch_aop_map[ibatch].len();
                    if n_active_aop == 0 { continue; }
                    let aop_batch = &aop_c.batches[ibatch];
                    debug_assert_eq!(n_active_aop, n_active);
                    debug_assert_eq!(aop_batch[0].size[1], batch_ao.size[1]);

                    for x in 0usize..3usize {
                        let mut tmop = MatrixFull::new([nocc, n_batch], 0.0);
                        _dgemm(&wmo, (0..n_active_aop, 0..nocc), 'T',
                               &aop_batch[x], (0..n_active_aop, g_local_start..g_local_end), 'N',
                               &mut tmop, (0..nocc, 0..n_batch), 1.0, 0.0);

                        let mut rhop_s = cur_rhop.get_reducing_matrix_mut(i_spin).unwrap();
                        for g in 0..n_batch {
                            let rhop_val: f64 = (0..nocc)
                                .map(|i| tmop[[i, g]] * tmo[[i, g]]).sum();
                            let col = rhop_s.get_column_mut(x);
                            col[g_out_start + g] = 2.0 * rhop_val;
                        }
                    }
                }
            }
        }

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

    //pub fn evaluate_density(&self, dm: &Vec<MatrixFull<f64>>, mpi_operator: &Option<MPIOperator>) -> [f64;2] {
    //    if let Some(mpi_op) = mpi_operator {
    //        let mut total_density = [0.0f64;2];
    //        let mut tmp_density = self.evaluate_density_rayon(dm);
    //        //println!("debug rank {} with tmp_density: {:?} before reduce", mpi_op.rank, &tmp_density);
    //        let mut tmp_density = mpi_reduce(&mpi_op.world, &tmp_density, 0, &SystemOperation::sum());
    //        //println!("debug rank {} with tmp_density: {:?} after reduce", mpi_op.rank, &tmp_density);
    //        mpi_broadcast_vector(&mpi_op.world, &mut tmp_density, 0);
    //        total_density.iter_mut().zip(tmp_density.iter()).for_each(|(to, from)| *to += *from);
    //        total_density
    //    } else {
    //        self.evaluate_density_rayon(dm)
    //    }

    //}
    
    //pub fn evaluate_density_rayon(&self, dm: &Vec<MatrixFull<f64>>) -> [f64;2] {
    //    let mut total_density = [0.0f64;2];
    //    if let Some(ao) = &self.ao {
    //        ao.iter_columns(0..self.weights.len())
    //            .zip(self.weights.iter()).for_each(|(ao_r, w)| {
    //            let mut density_r_sum = [0.0;2];
    //            //let ao_rv = ao_r.to_vec();
    //            //let tmp_len = ao_rv.len();
    //            //let ao_rr = MatrixFull::from_vec([tmp_len,1], ao_rv).unwrap();
    //            let tmp_len = ao_r.len();
    //            let ao_rr = MatrixFullSlice {
    //                size: &[1,tmp_len], 
    //                indicing: &[tmp_len,1],
    //                data: ao_r
    //            };
    //            dm.iter().zip(density_r_sum.iter_mut()).for_each(|(dm_s, density_r_sum)| {
    //                if dm_s.size().iter().fold(0, |acc, x| acc * x) != 0 {
    //                    let mut tmp_mat = MatrixFull::new([1,tmp_len],0.0);
    //                    _dgemm_full(&ao_rr, 'N', dm_s, 'N', &mut tmp_mat, 1.0, 0.0);
    //                    //tmp_mat.lapack_dgemm(&mut ao_rr, dm_s, 'T', 'N', 1.0, 0.0);
    //                    *density_r_sum += tmp_mat.data.iter().zip(ao_rr.data.iter()).fold(0.0, |acc,(a,b)| {acc + a*b});
    //                }
    //            });
    //            total_density.iter_mut().zip(density_r_sum.iter()).for_each(|(to,from)| *to += from*w);
    //        });
    //    }
    //    total_density
    //}
    //pub fn evalute_xc(&self, dm: &mut [MatrixFull<f64>;2], xc: XcFuncType) -> MatrixFull<>{

    //}
}



pub fn numerical_density_v01(grid: &Grids, mol: &Molecule, dm: &mut [MatrixFull<f64>;2]) -> [f64;2] {
    let mut total_density = [0.0f64;2];
    //let mut count:usize = 0;
    grid.coordinates.iter().zip(grid.weights.iter()).for_each(|(r,w)| {
        let mut density_r_sum = [0.0;2];
        let mut density_r:Vec<f64> = vec![];
        mol.basis4elem.iter().zip(mol.geom.rg_position.iter_columns_full()).for_each(|(elem,geom)| {
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

pub fn numerical_density(grid: &Grids, mol: &Molecule, dm: &Vec<MatrixFull<f64>>, mpi_operator: &Option<MPIOperator>) -> [f64;2] {
    #[cfg(feature = "mpi")]
    if let Some(mpi_op) = mpi_operator {
        let mut total_density = [0.0f64;2];
        let mut tmp_density = numerical_density_rayon(grid, mol, dm);
        let mut tmp_density = mpi_reduce(&mpi_op.world, &tmp_density, 0, &SystemOperation::sum());
        mpi_broadcast(&mpi_op.world, &mut tmp_density, 0);
        total_density.iter_mut().zip(tmp_density.iter()).for_each(|(to, from)| *to += *from);
        total_density
    } else
    {
        numerical_density_rayon(grid, mol, dm)
    }
    #[cfg(not(feature = "mpi"))]
    { numerical_density_rayon(grid, mol, dm) }
}

pub fn numerical_orbital_population(grid: &Grids, mol: &Molecule) -> Vec<f64> {
    let mut orbital_densities = vec![0.0;mol.num_basis];
    let default_omp_num_threads = omp_get_num_threads_wrapper();
    let local_basis4elem = mol.basis4elem.clone();
    let local_position = mol.geom.rg_position.clone();
    let num_basis = mol.num_basis;
    let basis_type = mol.ctrl.basis_type.clone();
    let num_grids = grid.coordinates.len();
    //let (sender,receiver) = channel();

    // reuse the default omp_num_threads setting
    omp_set_num_threads_wrapper(default_omp_num_threads);

    if let Some(tabulated_ao) = &grid.ao { 
        orbital_densities.iter_mut().enumerate().for_each(|(i,to)| {
            grid.weights.iter().zip(tabulated_ao.iter_row(i)).for_each(|(w,ao_r_r)| {
                *to += ao_r_r*ao_r_r * w
            })
        })
    } else {
        panic!("tabulated ao is not available")
    }

    //orbital_densities.iter_mut().enumerate().for_each(|(i,to));

    
    orbital_densities

}

pub fn numerical_density_rayon(grid: &Grids, mol: &Molecule, dm: &Vec<MatrixFull<f64>>) -> [f64;2] {
    let mut total_density = [0.0f64;2];
    let mut given_orbital_densities = 0.0;
    let orb_index = 13_usize;

    //let mut fack_dm = MatrixFull::new([mol.num_basis,mol.num_basis],0.0);
    ////fack_dm.iter_diagonal_mut().unwrap().for_each(|x| {
    ////    *x = 1.0;
    ////});
    //fack_dm[(12,12)] = 1.0;

    //let mut count:usize = 0;
    // In this subroutine, we call the lapack dgemm in a rayon parallel environment.
    let default_omp_num_threads = omp_get_num_threads_wrapper();

    let local_basis4elem = mol.basis4elem.clone();
    let local_position = mol.geom.rg_position.clone();
    let num_basis = mol.num_basis;
    let basis_type = mol.ctrl.basis_type.clone();
    let (sender,receiver) = channel();
    grid.coordinates.par_iter().zip(grid.weights.par_iter()).for_each_with(sender, |s,(r,w)| {
        omp_set_num_threads_wrapper(1);

        //let mut fack_dm = MatrixFull::new([mol.num_basis,mol.num_basis],0.0);
        ////fack_dm.iter_diagonal_mut().unwrap().for_each(|x| {
        ////    *x = 1.0;
        ////});
        //fack_dm[(orb_index,orb_index)] = 1.0;

        let mut local_total_density = [0.0f64;2];
        let mut local_orbital_densities = 0.0;
        let mut density_r_sum = [0.0;2];
        let mut density_r:Vec<f64> = vec![];
        let mut local_dm = dm.clone();
        local_basis4elem.iter().zip(local_position.iter_columns_full()).for_each(|(elem,geom)| {
            let mut tmp_geom = [0.0;3];
            tmp_geom.iter_mut().zip(geom.iter()).for_each(|value| {*value.0 = *value.1});
            //density_r.extend(gto_value(r, &tmp_geom, elem, &basis_type));
            let tmp_coordinate = [r.clone()];
            let tab_den = gto_value_serial(&tmp_coordinate, &tmp_geom, elem, &basis_type);
            density_r.extend(tab_den.data.clone());
        });

        let mut local_orbital_densities = density_r[orb_index];
        local_orbital_densities *= local_orbital_densities*w;

        let mut density_rr = MatrixFull::from_vec([num_basis,1],density_r).unwrap();
        local_dm.iter_mut().zip(density_r_sum.iter_mut()).for_each(|(dm_s, density_r_sum)| {
            let mut tmp_mat = MatrixFull::new([num_basis,1],0.0);
            tmp_mat.lapack_dgemm(&mut density_rr, dm_s, 'T', 'N', 1.0, 0.0);
            //tmp_mat.lapack_dgemm(&mut density_rr, &mut fack_dm, 'T', 'N', 1.0, 0.0);
            *density_r_sum += tmp_mat.data.iter().zip(density_rr.data.iter()).fold(0.0, |acc,(a,b)| {acc + a*b});
        });
        local_total_density.iter_mut().zip(density_r_sum.iter()).for_each(|(to,from)| *to += from*w);
        s.send((local_total_density,local_orbital_densities)).unwrap();
    });

    receiver.iter().for_each(|(value,orb_value)| {
        total_density.iter_mut().zip(value.iter()).for_each(|(to,from)| *to += from);
        given_orbital_densities += orb_value;
    });

    // reuse the default omp_num_threads setting
    omp_set_num_threads_wrapper(default_omp_num_threads);
    println!("debug: given orbital density: {:?}", given_orbital_densities);
    
    total_density
}


#[test]
fn test_non0tab_build() {
    // Test that Non0Tab struct can be constructed and fields are accessible
    let nt = Non0Tab {
        batch_ao_indices: vec![vec![0, 1, 2], vec![0, 2]],
        batch_aop_indices: vec![vec![0, 1, 2], vec![0, 2]],
        blksize: 128,
        ngrids: 256,
        nao: 10,
        ao_cutoff: 1e-10,
        sparsity_ratio: 0.25,
        total_nonzero_ao: 640,
        total_nonzero_aop: 640,
        total_stored_aop: 640,
        total_elements: 2560,
    };
    assert_eq!(nt.batch_ao_indices.len(), 2);
    assert_eq!(nt.blksize, 128);
    assert!((nt.sparsity_ratio - 0.25).abs() < 1e-10);
    assert_eq!(nt.total_nonzero_ao, 640);
    assert_eq!(nt.total_elements, 2560);

    // Clone
    let nt2 = nt.clone();
    assert_eq!(nt2.batch_ao_indices, nt.batch_ao_indices);

    // Grids with non0tab
    let grids = Grids {
        coordinates: vec![[0.0; 3]; 100],
        weights: vec![1.0; 100],
        ao: None,
        aop: None,
        parallel_balancing: vec![],
        non0tab: Some(nt),
        ao_cutoff: 1e-10,
        ao_compressed: None,
        aop_compressed: None,
        atm_idx: vec![0; 100],
        quadrature_weights: vec![1.0; 100],
    };
    assert!(grids.non0tab.is_some());
    assert_eq!(grids.ao_cutoff, 1e-10);

    // ao_cutoff = 0.0 → non0tab should be None
    let grids2 = Grids {
        coordinates: vec![[0.0; 3]; 100],
        weights: vec![1.0; 100],
        ao: None,
        aop: None,
        parallel_balancing: vec![],
        non0tab: None,
        ao_cutoff: 0.0,
        ao_compressed: None,
        aop_compressed: None,
        atm_idx: vec![0; 100],
        quadrature_weights: vec![1.0; 100],
    };
    assert!(grids2.non0tab.is_none());
    assert_eq!(grids2.ao_cutoff, 0.0);
}

#[test]
fn test_non0tab_clone() {
    let nt = Non0Tab {
        batch_ao_indices: vec![vec![0, 1], vec![2, 3]],
        batch_aop_indices: vec![vec![0, 1], vec![2, 3]],
        blksize: 64,
        ngrids: 128,
        nao: 4,
        ao_cutoff: 1e-10,
        sparsity_ratio: 1.0,
        total_nonzero_ao: 512,
        total_nonzero_aop: 512,
        total_stored_aop: 512,
        total_elements: 512,
    };
    let nt2 = nt.clone();
    assert_eq!(nt2.blksize, nt.blksize);
    assert_eq!(nt2.total_elements, nt.total_elements);
}

#[test]
fn test_non0tab_compress_roundtrip() {
    // Create a small synthetic AO matrix with known sparsity
    let nao = 4usize;
    let ngrids = 16usize;
    let blksize = 8usize;

    // AO: only mu=0,2 are non-zero in first batch; mu=1,3 in second batch
    let mut ao = MatrixFull::new([nao, ngrids], 0.0);
    for g in 0..8usize {
        ao[[0, g]] = 1.0 + g as f64 * 0.1;   // mu=0 non-zero in batch 0
        ao[[2, g]] = 2.0 + g as f64 * 0.1;   // mu=2 non-zero in batch 0
    }
    for g in 8..16usize {
        ao[[1, g]] = 3.0 + g as f64 * 0.1;   // mu=1 non-zero in batch 1
        ao[[3, g]] = 4.0 + g as f64 * 0.1;   // mu=3 non-zero in batch 1
    }

    // Build Non0Tab reflecting this sparsity
    let nt = Non0Tab {
        batch_ao_indices: vec![vec![0, 2], vec![1, 3]],
        batch_aop_indices: vec![],
        blksize,
        ngrids,
        nao,
        ao_cutoff: 1e-10,
        sparsity_ratio: 0.5,
        total_nonzero_ao: nao * ngrids / 2,
        total_nonzero_aop: 0,
        total_stored_aop: 0,
        total_elements: nao * ngrids,
    };

    // Compress
    let nbatches = nt.batch_ao_indices.len();
    let mut batches = Vec::with_capacity(nbatches);
    let mut batch_grid_ranges = Vec::with_capacity(nbatches);
    for ibatch in 0..nbatches {
        let g_start = ibatch * nt.blksize;
        let g_end = (g_start + nt.blksize).min(nt.ngrids);
        let nbatch = g_end - g_start;
        let indices = &nt.batch_ao_indices[ibatch];
        let n_active = indices.len();
        let mut batch_ao = MatrixFull::new([n_active, nbatch], 0.0);
        for (i_local, &mu_global) in indices.iter().enumerate() {
            for g in g_start..g_end {
                batch_ao[[i_local, g - g_start]] = ao[[mu_global, g]];
            }
        }
        batches.push(batch_ao);
        batch_grid_ranges.push(g_start..g_end);
    }
    let compressed = CompressedGridAO {
        batches,
        batch_ao_map: nt.batch_ao_indices.clone(),
        batch_grid_ranges,
        blksize: nt.blksize,
        nao_total: nt.nao,
        ngrids: nt.ngrids,
    };

    // Decompress
    let ao_roundtrip = Grids::decompress_ao(&compressed);

    // Verify roundtrip
    for mu in 0..nao {
        for g in 0..ngrids {
            let diff = (ao[[mu, g]] - ao_roundtrip[[mu, g]]).abs();
            assert!(diff < 1e-14,
                "roundtrip error at mu={mu}, g={g}: orig={}, rt={}",
                ao[[mu, g]], ao_roundtrip[[mu, g]]);
        }
    }

    // Verify sparsity: zero entries should remain zero
    for g in 0..8usize {
        assert_eq!(ao_roundtrip[[1, g]], 0.0);
        assert_eq!(ao_roundtrip[[3, g]], 0.0);
    }
    for g in 8..16usize {
        assert_eq!(ao_roundtrip[[0, g]], 0.0);
        assert_eq!(ao_roundtrip[[2, g]], 0.0);
    }

    // Verify memory: compressed < dense
    let dense_bytes = ao.data.len() * 8;
    let mut comp_bytes = 0usize;
    for b in &compressed.batches { comp_bytes += b.data.len() * 8; }
    assert!(comp_bytes < dense_bytes,
        "compressed {comp_bytes} should be < dense {dense_bytes}");
}

#[test]
fn test_auto_non0tab_blksize() {
    // Small system: large blksize
    let blk = Grids::auto_non0tab_blksize(100);
    assert!(blk >= 128, "small nao should get large blksize, got {blk}");
    assert!(blk <= 256, "blksize should be clamped, got {blk}");

    // Medium system
    let blk = Grids::auto_non0tab_blksize(730);
    assert!(blk >= 32 && blk <= 128, "medium nao blksize range, got {blk}");

    // Large system: small blksize
    let blk = Grids::auto_non0tab_blksize(4600);
    assert!(blk >= 32, "large nao should get small blksize, got {blk}");
    assert!(blk <= 64, "large nao blksize should be small, got {blk}");

    // Very large: clamped to minimum
    let blk = Grids::auto_non0tab_blksize(20000);
    assert_eq!(blk, 32, "very large nao should hit minimum");

    // Tiny system: clamped to maximum
    let blk = Grids::auto_non0tab_blksize(10);
    assert_eq!(blk, 256, "tiny nao should hit maximum");
}

// ── contract_response_compressed correctness with multi-batch grid offsets ──
//
// This test validates the fix for the vrho/vsigma/vtau/rhop offset bug in
// `contract_response_compressed`. The bug caused the function to always index
// potential arrays from element 0, ignoring the fact that each batch may start
// at a different position within `range_grids`. This test covers:
//   1. full-range [0, N) – exercises batch 1+ with non-zero offset
//   2. sub-range [a, b) – simulates a rayon-thread chunk not starting at 0
#[test]
fn test_non0tab_contract_response_offset() {
    use std::ops::Range;

    let nao: usize = 4;
    let ngrids: usize = 16;
    let blksize: usize = 8;

    // ── Build dense AO with known sparsity per batch ──
    let mut ao_full = MatrixFull::<f64>::new([nao, ngrids], 0.0);
    for g in 0usize..8usize {
        ao_full[[0, g]] = 1.0;
        ao_full[[1, g]] = 1.0;
    }
    for g in 8usize..16usize {
        ao_full[[2, g]] = 1.0;
        ao_full[[3, g]] = 1.0;
    }

    // ── Build Non0Tab + CompressedGridAO ──
    let nbatches = 2;
    let mut batch_ao_indices: Vec<Vec<usize>> = Vec::with_capacity(nbatches);
    let mut batch_aop_indices: Vec<Vec<usize>> = Vec::with_capacity(nbatches);
    let mut compressed_batches: Vec<MatrixFull<f64>> = Vec::with_capacity(nbatches);
    let mut batch_grid_ranges: Vec<Range<usize>> = Vec::with_capacity(nbatches);
    for ibatch in 0..nbatches {
        let g_start = ibatch * blksize;
        let g_end = (g_start + blksize).min(ngrids);
        let nbatch = g_end - g_start;
        let indices: Vec<usize> = (0..nao).filter(|&mu| (ao_full[[mu, g_start]] as f64).abs() > 1e-10f64).collect();
        let n_active = indices.len();
        let mut batch_ao = MatrixFull::new([n_active, nbatch], 0.0);
        for (i_local, &mu_global) in indices.iter().enumerate() {
            for g in g_start..g_end {
                batch_ao[[i_local, g - g_start]] = ao_full[[mu_global, g]];
            }
        }
        batch_ao_indices.push(indices.clone());
        batch_aop_indices.push(indices.clone()); // same mask for simplicity
        compressed_batches.push(batch_ao);
        batch_grid_ranges.push(g_start..g_end);
    }

    let compressed = CompressedGridAO {
        batches: compressed_batches.clone(),
        batch_ao_map: batch_ao_indices.clone(),
        batch_grid_ranges: batch_grid_ranges.clone(),
        blksize,
        nao_total: nao,
        ngrids,
    };

    // Dummy compressed AOP (won't be exercised since GGA=mGGA=false)
    let aop_compressed = CompressedGridAOP {
        batches: vec![],
        batch_aop_map: vec![],
        batch_grid_ranges: vec![],
        blksize,
        nao_total: nao,
        ngrids,
    };

    let grids = Grids {
        coordinates: vec![[0.0; 3]; ngrids],
        weights: vec![1.0; ngrids],
        ao: Some(ao_full.clone()),
        aop: None,
        parallel_balancing: vec![],
        non0tab: None,
        ao_cutoff: 1e-10,
        ao_compressed: Some(compressed),
        aop_compressed: Some(aop_compressed),
        atm_idx: vec![0; ngrids],
        quadrature_weights: vec![1.0; ngrids],
    };

    // Build global vrho: 1.0 for grids 0..8, 2.0 for grids 8..16
    let vrho_global = {
        let mut m = MatrixFull::<f64>::new([ngrids, 1], 0.0);
        for g in 0usize..8usize { m[[g, 0]] = 1.0; }
        for g in 8usize..16usize { m[[g, 0]] = 2.0; }
        m
    };
    let weights_global: Vec<f64> = vec![1.0; ngrids];
    let vsigma = MatrixFull::<f64>::empty();
    let vtau = MatrixFull::<f64>::empty();
    let rhop = RIFull::<f64>::empty();

    // Dense reference: vxc_mat_ref[mu,nu] = Σ_g ao[mu,g] * vrho[g] * w[g] * ao[nu,g]
    let compute_dense_ref = |r: &Range<usize>| -> MatrixFull<f64> {
        let mut ref_mat = MatrixFull::<f64>::new([nao, nao], 0.0);
        for mu in 0..nao {
            for nu in 0..nao {
                let mut acc = 0.0;
                for g in r.clone() {
                    acc += ao_full[[mu, g]] * vrho_global[[g, 0]] * weights_global[g] * ao_full[[nu, g]];
                }
                ref_mat[[mu, nu]] = acc;
            }
        }
        ref_mat
    };

    // Helper: build range_grids-relative vrho and call contract_response_compressed
    // The real code passes vrho indexed relative to range_grids, so vrho[0] ↔ grid range_grids.start
    let run_compressed = |grids: &Grids, r: &Range<usize>|
        -> Vec<MatrixFull<f64>>
    {
        let spin = 1usize;
        // Build vrho (and weights) relative to the current range
        let vrho_local = {
            let mut m = MatrixFull::<f64>::new([r.len(), 1], 0.0);
            for (i, g) in r.clone().enumerate() {
                m[[i, 0]] = vrho_global[[g, 0]];
            }
            m
        };
        let weights_local: Vec<f64> = r.clone().map(|g| weights_global[g]).collect();
        let mut result = vec![MatrixFull::<f64>::new([nao, nao], 0.0); spin];
        grids.contract_response_compressed(
            r, &vrho_local, &vsigma, &vtau,
            &weights_local, &mut result, spin,
            false, false,
            &rhop, nao,
        );
        result
    };

    // ── Test 1: full range [0, 16) ──
    {
        let r: Range<usize> = 0..ngrids;
        let ref_mat = compute_dense_ref(&r);
        let comp_mat = &run_compressed(&grids, &r)[0];

        for mu in 0..nao {
            for nu in 0..nao {
                let diff = (ref_mat[[mu, nu]] - comp_mat[[mu, nu]]).abs();
                assert!(diff < 1e-12,
                    "[full] vxc_mat[{mu},{nu}]: ref={:.6e} comp={:.6e} diff={:.2e}",
                    ref_mat[[mu, nu]], comp_mat[[mu, nu]], diff);
            }
        }
        assert!((ref_mat[[0, 0]] - 8.0).abs() < 1e-12, "batch0 ref[0,0] should be 8");
        assert!((ref_mat[[2, 2]] - 16.0).abs() < 1e-12, "batch1 ref[2,2] should be 16");
    }

    // ── Test 2: sub-range [4, 12) — simulates rayon thread chunk ──
    {
        let r: Range<usize> = 4..12;
        let ref_mat = compute_dense_ref(&r);
        let comp_mat = &run_compressed(&grids, &r)[0];

        for mu in 0..nao {
            for nu in 0..nao {
                let diff = (ref_mat[[mu, nu]] - comp_mat[[mu, nu]]).abs();
                assert!(diff < 1e-12,
                    "[sub] vxc_mat[{mu},{nu}]: ref={:.6e} comp={:.6e} diff={:.2e}",
                    ref_mat[[mu, nu]], comp_mat[[mu, nu]], diff);
            }
        }
        // batch0: 4 grids × 1.0 = 4 for {0,1} pairs
        assert!((ref_mat[[0, 0]] - 4.0).abs() < 1e-12, "sub ref[0,0] should be 4");
        // batch1: 4 grids × 2.0 = 8 for {2,3} pairs
        assert!((ref_mat[[2, 2]] - 8.0).abs() < 1e-12, "sub ref[2,2] should be 8");
    }

    // ── Test 3: sub-range [8, 16) — starts exactly at batch boundary ──
    {
        let r: Range<usize> = 8..16;
        let ref_mat = compute_dense_ref(&r);
        let comp_mat = &run_compressed(&grids, &r)[0];

        for mu in 0..nao {
            for nu in 0..nao {
                let diff = (ref_mat[[mu, nu]] - comp_mat[[mu, nu]]).abs();
                assert!(diff < 1e-12,
                    "[boundary] vxc_mat[{mu},{nu}]: ref={:.6e} comp={:.6e} diff={:.2e}",
                    ref_mat[[mu, nu]], comp_mat[[mu, nu]], diff);
            }
        }
        // batch1 only: 8 grids × 2.0 = 16 for {2,3} pairs
        assert!((ref_mat[[2, 2]] - 16.0).abs() < 1e-12, "boundary ref[2,2] should be 16");
        assert!((ref_mat[[0, 0]] - 0.0).abs() < 1e-12, "boundary ref[0,0] should be 0");
    }
}
