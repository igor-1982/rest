#![warn(unused_imports)]

mod pyrest_molecule_io;
pub mod with_clause;
mod int_cross;
mod frozen;
pub mod basis;
pub use frozen::count_frozen_core_states;

use array_tool::vec::Intersect;
use pyo3::{pyclass};
use rayon::prelude::{IntoParallelRefIterator, IndexedParallelIterator, ParallelIterator};
use rest_libcint::prelude::*;
use rest_libcint::{CINTR2CDATA, CintType};
use rest_tensors::{ERIFull,RIFull,ERIFold4,TensorSlice,TensorSliceMut,TensorOpt, MatrixUpper, MatrixFull};
use tensors::{BasicMatrix, SubMatrixUpper};
use tensors::external_libs::{matr_copy_from_ri};
use tensors::matrix_blas_lapack::{_dgemm, _dgemm_full, _dtrtrs, _power_rayon_for_symmetric_matrix};
use tensors::matrix_blas_lapack::{omp_set_num_threads_wrapper};
use std::collections::HashMap;
use std::ops::Range;
use std::sync::mpsc::channel;
use regex::Regex;
use crate::basis_io::etb::{get_etb_elem, etb_gen_for_atom_list, InfoV2};
use crate::constants::{ATM_NUC, ATM_NUC_MOD_OF, AUXBAS_THRESHOLD, ENV_PRT_START, NUC_ECP, NUC_STAD_CHARGE};
use crate::dft::{DFTType, DFA4REST, parse_xc};
use crate::geom_io::{GeomCell, get_mass_charge};
use crate::basis_io::{BasInfo, Basis4Elem};
use crate::ctrl_io::{overall_parse_and_report_on_ctrl_geom, InputKeywords, parse_ctl};
#[cfg(feature = "mpi")]
use crate::mpi_io::mpi_isend_irecv_wrt_distribution_v03;
use crate::mpi_io::{MPIData, MPIOperator};
use crate::utilities;
use crate::basis_io::bse_downloader::{self, ctrl_element_checker, local_element_checker};
use crate::basis_io::basis_list::{basis_fuzzy_matcher, check_basis_name};
use crate::ri_jk;


pub use basis::get_basis_name;

#[derive(Clone)]
#[pyclass]
/// # Molecule
/// `Molecule` contains all basic information of the molecule in the calculation task
/// self.ctrl:         include the input keywords and many derived keywors to control the calculation task
/// self.bas:          the basis set shell by shell organized for libcint
/// self.geom:         the molecular structure information
/// self.xc_data:      the information and operations for DFT: [`DFA4REST`](DFA4REST).
/// 
/// self.spin_channel: 1: spin unpolarized or 2: spin polarized
/// self.num_state:    the number of molecular orbitals
/// self.num_basis:    the number of basis functions
/// self.start_mo:     the first molecular orbital in the frozen-core approximation for pt2, rpa and so forth
/// 
/// self.fdqc_bas:     the basis set information [`BasInfo`](BasInfo) for each shell
/// self.cint_*:       all these fields are the interface to ```libcint```
/// self.cint_type:    CintType::Spheric or CintType::Cartesian
pub struct Molecule {
    #[pyo3(get, set)]
    pub ctrl: InputKeywords,
    pub mpi_data: Option<MPIData>, 
    #[pyo3(get, set)]
    pub geom : GeomCell,
    #[pyo3(get, set)]
    pub spin_channel: usize,
    // exchange-correlation functionals
    pub dfadef: Option<parse_xc::DFAdef>,
    pub xc_data: DFA4REST,
    pub use_eri: bool,
    #[pyo3(get, set)]
    pub num_state: usize,
    #[pyo3(get, set)]
    pub num_basis: usize,
    #[pyo3(get, set)]
    pub num_auxbas: usize,
    #[pyo3(get, set)]
    pub num_elec : [f64;3],
    // for frozen-core pt2, rpa and so forth
    #[pyo3(get, set)]
    pub start_mo : usize,
    // for effective core potential in SCF
    pub ecp_electrons : usize,

    pub auxbas4elem: Vec<Basis4Elem>,

    pub basis4elem: Vec<Basis4Elem>,
    // natm_real: number of real atoms (geom.elem.len())
    pub natm_real: usize,
    // natm_all: total number of atoms with basis functions (real + ghost_bs)
    pub natm_all: usize,
    // fdqc_bas: store information of each basis functions
    pub fdqc_bas : Vec<BasInfo>,
    //  cint_fdqc: vec![[start of a basis shell, num of basis funciton in this shell]; num of shells]
    pub cint_fdqc: Vec<Vec<usize>>,
    //  cint_bas : vec![data for each basis shell; num of shells]
    //             data for each basis shell contains 8 slots, which are organized in the order of "bas" required by libcint
    pub cint_bas : Vec<Vec<i32>>,
    //  vec![data for each atom; num of atoms]
    //  data for each atom contains 6 slots, which are organized in the order of "atm" required by libcint
    pub cint_atm : Vec<Vec<i32>>,
    //  save the value of coordinates, exponents, contraction coefficients in the order of "env" required by libcint
    pub cint_env : Vec<f64>,
    //  cint_ecpbas : vec![data for each ecp basis shell; num of shells]
    //             data for each basis shell contains 8 slots, which are organized in the order of "bas" required by libcint
    pub cint_ecpbas : Option<Vec<Vec<i32>>>,
    pub fdqc_aux_bas : Vec<BasInfo>,
    pub cint_aux_fdqc: Vec<Vec<usize>>,
    pub cint_aux_bas : Vec<Vec<i32>>,
    pub cint_aux_atm : Vec<Vec<i32>>,
    pub cint_aux_env : Vec<f64>,
    pub cint_type: CintType,
}

impl Molecule {
    pub fn init_mol() -> Molecule {
        Molecule {
            ctrl:InputKeywords::init_ctrl(),
            mpi_data: None,
            dfadef: None, 
            xc_data: DFA4REST::new("hf",1, 0),
            use_eri: false,
            geom: GeomCell::init_geom(),
            num_elec: [0.0;3],
            num_state: 0,
            num_basis: 0,
            num_auxbas: 0,
            start_mo: 0,   // all-electron pt2, rpa and so forth
            ecp_electrons: 0,
            spin_channel: 1,
            auxbas4elem: vec![],
            natm_real: 0,
            natm_all: 0,
            basis4elem: vec![],
            fdqc_bas: vec![],
            cint_bas: vec![],
            cint_ecpbas: None,
            cint_fdqc: vec![],
            cint_atm: vec![],
            cint_env: vec![],
            fdqc_aux_bas: vec![],
            cint_aux_bas: vec![],
            cint_aux_fdqc: vec![],
            cint_aux_atm: vec![],
            cint_aux_env: vec![],
            cint_type: CintType::Spheric,
            //cint_data: CINTR2CDATA::new()
        }
    }

    pub fn print_auxbas(&self) {
        println!("cint_aux_bas: {:?}", self.cint_aux_bas);
        println!("cint_aux_fdqc: {:?}", self.cint_aux_fdqc);
        println!("cint_aux_atm: {:?}", self.cint_aux_atm);
        println!("cint_aux_env: {:?}", self.cint_aux_env);
        println!("cint_type: {:}", match self.cint_type {
            CintType::Spheric => "Spheric",
            CintType::Cartesian => "Cartesian",
            CintType::Spinor => "Spinor",
        })
    }

    pub fn build(ctrl_file: String, mpi_data: Option<MPIData>) -> anyhow::Result<Molecule> {
        //let mut mol = Molecule::new();
        //let (mut ctrl, mut geom) = RawCtrl::parse_ctl_from_jsonfile_v02(ctrl_file)?;
        let (mut ctrl, mut geom) = parse_ctl(ctrl_file)?;
        if let Some(local_mpi_data) = &mpi_data {
            if local_mpi_data.rank != 0 {ctrl.print_level = 0};
        };
        overall_parse_and_report_on_ctrl_geom(&mut ctrl, &mut geom);
        Molecule::build_native(ctrl, geom, mpi_data)
    }

    pub fn build_native(mut ctrl: InputKeywords, mut geom: GeomCell, mpi_data: Option<MPIData>) -> anyhow::Result<Molecule> {

        let mut mol = Molecule::init_mol();
        mol.ctrl = ctrl.clone();
        mol.geom = geom.clone();
        mol.mpi_data = mpi_data;

        mol.spin_channel = ctrl.spin_channel;
        let cint_type = if ctrl.basis_type.to_lowercase()==String::from("spheric") {
            CintType::Spheric
        } else if ctrl.basis_type.to_lowercase()==String::from("cartesian") {
            CintType::Cartesian
        } else {
            panic!("Error:: Unknown basis type '{}'. Please use either 'spheric' or 'Cartesian'", 
                   ctrl.basis_type);
        };
        mol.cint_type = cint_type;

        let chkbasis = ctrl.basis_path == "chkfile";
        if !chkbasis {
            let (mut basis4elem,mut cint_atm,mut cint_bas,cint_env,
                fdqc_bas,cint_fdqc,num_basis,num_state, cint_ecpbas) 
                = Molecule::collect_basis(&mut ctrl, &mut geom);

            let bas = &basis4elem;
            let ecp_electrons = bas.iter().fold(0, |acc, i| {
                let ecp_electrons = if let Some(num_ecp) = i.ecp_electrons {num_ecp} else {0};
                acc + ecp_electrons
            });
            mol.basis4elem = basis4elem;
            mol.cint_atm = cint_atm;
            mol.cint_bas = cint_bas;
            mol.cint_env = cint_env;
            mol.fdqc_bas = fdqc_bas;
            mol.cint_fdqc = cint_fdqc;
            mol.update_num_elec();
            mol.num_basis = num_basis;
            mol.num_state = num_state;
            mol.cint_ecpbas = cint_ecpbas;
            mol.ecp_electrons = ecp_electrons;
        }
        mol.natm_real = geom.elem.len();
        mol.natm_all = mol.cint_atm.len();

        let (mut auxbas , mut cint_aux_atm,mut cint_aux_bas,cint_aux_env,
                mut fdqc_aux_bas,mut cint_aux_fdqc,num_auxbas) 
            =(Vec::new(),Vec::new(),Vec::new(),Vec::new(),Vec::new(),Vec::new(),0);
        mol.auxbas4elem = auxbas;
        mol.cint_aux_atm = cint_aux_atm;
        mol.cint_aux_bas = cint_aux_bas;
        mol.cint_aux_env = cint_aux_env;
        mol.fdqc_aux_bas = fdqc_aux_bas;
        mol.cint_aux_fdqc = cint_aux_fdqc;
        mol.num_auxbas = num_auxbas;

        //let mut cint_data = CINTR2CDATA::new();
        //cint_data.set_cint_type(&cint_type);
        //cint_data.initial_r2c(&cint_atm, natm, &cint_bas, nbas, &env);

        //fdqc_aux_bas.iter().for_each(|i| {
        //    println!("{}", i.formated_name());
        //});

        //let basis4elem = bas;
        let (dfadef, xc_data) = match &ctrl.xc_type {
            DFTType::Standard => {
                println!("Using xc_parser {}", ctrl.xc_parser);
                match ctrl.xc_parser.as_str() {
                    "legacy" => {
                    let mut cur_xc_data = DFA4REST::new(&ctrl.xc, ctrl.spin_channel, ctrl.print_level);
                    cur_xc_data.update_pt2_params(ctrl.ri_pt2.os_factor, ctrl.ri_pt2.ss_factor);
                    (None, cur_xc_data)
                    },
                    "parse_xc" => {
                        let mut dfadef = parse_xc::parse_and_derive(&ctrl.xc, ctrl.spin_channel, ctrl.print_level);
                        let (san, err_strings) = dfadef.check_sanity();
                        if !san {
                            panic!("{}", err_strings.join("\n"));
                        }
                        let mut xc_data = dfadef.to_dfa4rest();
                        xc_data.update_pt2_params(ctrl.ri_pt2.os_factor, ctrl.ri_pt2.ss_factor);
                        (Some(dfadef.clone()), xc_data)
                    },
                    _ => {
                        panic!("Error:: Unknown xc_parser '{}'. Please use either 'legacy' or 'parse_xc'", 
                               ctrl.xc_parser);
                    }
                }
            },
            DFTType::NonStandard => {
                println!("Warning: DFTType::NonStandard is about to be deprecated. Please use xc_parser = 'parse_xc' instead.");
                (None, DFA4REST::new_nonstandard(ctrl.spin_channel, ctrl.print_level, &ctrl.xc_namelist, &ctrl.xc_paralist, &ctrl.dfa_hybrid_scf))
            },
            DFTType::DeepLearning => {(None, DFA4REST::new_deep_learning(ctrl.spin_channel, ctrl.print_level, &ctrl.xc_model))}
        };

        xc_data.summary(ctrl.print_level);
        if let Some(stop_at) = &ctrl.stop_at {
            if stop_at == "parse_xc" {
                std::process::exit(0);
            }
        }
        mol.dfadef = dfadef;
        mol.xc_data = xc_data;

        mol.use_eri = mol.xc_data.use_eri();
        

        mol.start_mo = mol.generate_start_mo(mol.ecp_electrons);

        if ctrl.print_level>0 {
            mol.xc_data.xc_version();
            println!("nbas: {}, natm: {} for standard basis sets", mol.cint_bas.len(), mol.cint_atm.len());
            println!("First valence state for the frozen-core algorithm: {:5}", mol.start_mo);
        };

        // check and prepare the auxiliary basis sets
        if mol.ctrl.use_auxbas {mol.initialize_auxbas()};

        //println!("Debug for AuxBasis4Eelem");
        //mol.auxbas4elem.iter().enumerate().for_each(|(i,x)| {
        //    println!("Basis functions for Atom {}: ({},{})", i,x.global_index.0,x.global_index.1);
        //});

        if mol.ctrl.print_level>0 {
            println!("numbasis: {:2}, num_auxbas: {:2}", mol.num_basis,mol.num_auxbas)
        };
        if mol.ctrl.print_level>4 {mol.print_auxbas()};

        Ok(mol)

    }

    pub fn has_ecp(&self) -> bool {
        self.cint_ecpbas.is_some()
    }

    pub fn set_cint_data(
        &mut self,
        atm: Vec<Vec<i32>>,
        bas: Vec<Vec<i32>>,
        env: Vec<f64>,
        ecpbas: Option<Vec<Vec<i32>>>,
        cint_type: Option<CintType>,
        basis4elem: Option<Vec<Basis4Elem>>,
        fdqc_bas: Option<Vec<BasInfo>>,
        cint_fdqc: Option<Vec<Vec<usize>>>,
    ) {
        self.cint_atm = atm;
        self.cint_bas = bas;
        self.cint_env = env;
        self.cint_ecpbas = ecpbas;
        if let Some(ct) = cint_type {
            self.cint_type = ct;
        }
        if let Some(b) = basis4elem {
            self.basis4elem = b;
            self.ecp_electrons = self.basis4elem.iter().fold(0, |acc, i| {
                acc + i.ecp_electrons.unwrap_or(0)
            });
        }
        if let Some(fdqc) = fdqc_bas {
            self.fdqc_bas = fdqc;
            self.cint_fdqc = cint_fdqc.unwrap_or(vec![]);
            self.num_basis = self.fdqc_bas.len();
        } else {
            let (fdqc_bas, cint_fdqc) = basis::build_fdqc(&self.cint_bas, &self.cint_type);
            self.fdqc_bas = fdqc_bas;
            self.cint_fdqc = cint_fdqc;
            self.num_basis = self.fdqc_bas.len();
        }
        self.natm_real = self.geom.elem.len();
        self.natm_all = self.cint_atm.len();
        self.num_state = self.num_basis;
    }

    pub fn update_num_elec(&mut self) {
        let mut num_elec = [0.0; 3];
        self.cint_atm.iter().for_each(|atm| {
            num_elec[0] += atm[ATM_NUC] as f64;
        });
        num_elec[0] -= self.ctrl.charge;
        if self.ctrl.use_int_nelec {
            sanity_check_nelec(num_elec[0], self.ctrl.spin);
        }
        let unpair_elec = self.ctrl.spin - 1.0_f64;
        num_elec[1] = (num_elec[0] - unpair_elec) / 2.0 + unpair_elec;
        num_elec[2] = (num_elec[0] - unpair_elec) / 2.0;
        self.num_elec = num_elec;
    }

    pub fn initialize_auxbas(&mut self) {
        let cint_type = if self.ctrl.basis_type.to_lowercase()==String::from("spheric") {
            CintType::Spheric
        } else if self.ctrl.basis_type.to_lowercase()==String::from("cartesian") {
            CintType::Cartesian
        } else {
            panic!("Error:: Unknown basis type '{}'. Please use either 'spheric' or 'Cartesian'", 
                   self.ctrl.basis_type);
        };


        let etb = if (self.ctrl.even_tempered_basis == true) {
            let etb_elem = get_etb_elem(&self.geom, &self.ctrl.etb_start_atom_number);
            let etb_basis = etb_gen_for_atom_list(&self, &self.ctrl.etb_beta, &etb_elem);
            Some(etb_basis)
        }
        else{
            None
        };


        // at first, deallocate the cint data for standard basis set
        //self.cint_data.final_c2r();
        // loading the auxiliary basis sets
        let (mut auxbas,mut cint_aux_atm,mut cint_aux_bas,cint_aux_env,
                mut fdqc_aux_bas,mut cint_aux_fdqc,num_auxbas) 
            = if self.ctrl.use_auxbas {
                Molecule::collect_auxbas(&mut self.ctrl, &mut self.geom, etb)
        } else {
            (Vec::new(),Vec::new(),Vec::new(),Vec::new(),Vec::new(),Vec::new(),0)
        };

        self.cint_aux_atm = cint_aux_atm;
        self.cint_aux_bas = cint_aux_bas;
        self.cint_aux_env = cint_aux_env;
        self.fdqc_aux_bas = fdqc_aux_bas;
        self.cint_aux_fdqc = cint_aux_fdqc;
        self.num_auxbas = num_auxbas;
        self.auxbas4elem = auxbas;

        //let mut env = self.cint_env.clone();

        let off = self.cint_env.len() as i32;
        let natm_off = self.cint_atm.len() as i32;
        let nbas_off = self.cint_bas.len() as i32;
        // append auxliary basis set list into the standard basis set list for libcint
        self.cint_aux_atm.iter_mut().for_each(|i| {
            // offset the position that store coordinates in the env list
            if let Some(j) = i.get_mut(1) {*j += off};
            // offset the position that store nuclear charge distribution parameters in the env list
            if let Some(j) = i.get_mut(3) {*j += off};
        });
        self.cint_aux_bas.iter_mut().for_each(|i| {
            // =========================================================
            // offset the atom index in the atm list
            // WARNNING:: do not offset the atom index in the atm list, such that the auxiliary basis sets
            //            use the same cint_atm as the ordinary basis sets.
            //if let Some(j) = i.get_mut(0) {*j += natm_off};
            // =========================================================
            // offset the position that stores basis exponents in the env list
            if let Some(j) = i.get_mut(5) {*j += off};
            // offset the position that stores basis coefficients in the env list
            if let Some(j) = i.get_mut(6) {*j += off};
        });

        self.fdqc_aux_bas.iter_mut().for_each(|i| {i.cint_index0 += nbas_off as usize});

        //println!("final nbas: {},final natm: {}", nbas, natm);
        if self.ctrl.print_level>0 {
            println!("final nbas: {},final natm: {}", self.cint_bas.len()+self.cint_aux_bas.len(), self.cint_atm.len()+self.cint_aux_atm.len());
        }

    }

    pub fn reload_auxbas(&mut self, auxbas_path: String) {
        // Save the new auxiliary basis path
        let original_auxbas_path = self.ctrl.auxbas_path.clone();
        self.ctrl.auxbas_path = auxbas_path;

        // Reload auxiliary basis information
        let etb = if self.ctrl.even_tempered_basis {
            let etb_elem = get_etb_elem(&self.geom, &self.ctrl.etb_start_atom_number);
            let etb_basis = etb_gen_for_atom_list(&self, &self.ctrl.etb_beta, &etb_elem);
            Some(etb_basis)
        } else {
            None
        };

        let (auxbas, cint_aux_atm, cint_aux_bas, cint_aux_env, fdqc_aux_bas, cint_aux_fdqc, num_auxbas) =
            Molecule::collect_auxbas(&mut self.ctrl, &mut self.geom, etb);

        // Update auxiliary basis fields
        self.cint_aux_atm = cint_aux_atm;
        self.cint_aux_bas = cint_aux_bas;
        self.cint_aux_env = cint_aux_env;
        self.fdqc_aux_bas = fdqc_aux_bas;
        self.cint_aux_fdqc = cint_aux_fdqc;
        self.num_auxbas = num_auxbas;
        self.auxbas4elem = auxbas;

        // Apply offsets as in initialize_auxbas
        let off = self.cint_env.len() as i32;
        let nbas_off = self.cint_bas.len() as i32;

        self.cint_aux_atm.iter_mut().for_each(|i| {
            if let Some(j) = i.get_mut(1) {*j += off};
            if let Some(j) = i.get_mut(3) {*j += off};
        });
        self.cint_aux_bas.iter_mut().for_each(|i| {
            if let Some(j) = i.get_mut(5) {*j += off};
            if let Some(j) = i.get_mut(6) {*j += off};
        });
        self.fdqc_aux_bas.iter_mut().for_each(|i| {i.cint_index0 += nbas_off as usize});

        if self.ctrl.print_level > 0 {
            println!("Reloaded auxiliary basis: {}", self.ctrl.auxbas_path);
            println!("New auxiliary basis size: {}", self.num_auxbas);
        }
    }

    pub fn initialize_cint(&self, for_ri: bool) -> CINTR2CDATA {

        if for_ri {
            let mut final_cint_atm = self.cint_atm.clone();
            let mut final_cint_bas = self.cint_bas.clone();
            let mut final_cint_env = self.cint_env.clone();
            final_cint_atm.extend(self.cint_aux_atm.clone());
            final_cint_bas.extend(self.cint_aux_bas.clone());
            final_cint_env.extend(self.cint_aux_env.clone());
            let natm = final_cint_atm.len() as i32;
            let nbas = final_cint_bas.len() as i32;
            let mut cint_data = CINTR2CDATA::new();
            cint_data.set_cint_type(self.cint_type);
            if let Some(final_cint_ecp) = &self.cint_ecpbas {
                let necp = final_cint_ecp.len() as i32;
                cint_data.initial_r2c_with_ecp(&final_cint_atm, natm, &final_cint_bas, nbas, &final_cint_ecp, necp, &final_cint_env);
            } else {
                cint_data.initial_r2c(&final_cint_atm, natm, &final_cint_bas, nbas, &final_cint_env);
            }
            cint_data
        } else {
            let final_cint_atm = &self.cint_atm.clone();
            let final_cint_bas = &self.cint_bas.clone();
            let final_cint_env = &self.cint_env.clone();
            let natm = final_cint_atm.len() as i32;
            let nbas = final_cint_bas.len() as i32;
            let mut cint_data = CINTR2CDATA::new();
            cint_data.set_cint_type(self.cint_type);
            if let Some(final_cint_ecp) = &self.cint_ecpbas {
                let necp = final_cint_ecp.len() as i32;
                cint_data.initial_r2c_with_ecp(final_cint_atm, natm, final_cint_bas, nbas, final_cint_ecp, necp, final_cint_env);
            } else {
                cint_data.initial_r2c(final_cint_atm, natm, final_cint_bas, nbas, final_cint_env);
            }
            cint_data
        }
    }

    pub fn update_geom_poisition_in_cint_env(&self, position: &MatrixFull<f64>) -> Vec<f64> {
        let mut cint_env = self.cint_env.clone();
        self.cint_atm.iter().zip(position.iter_columns_full()).for_each(|(atm, position)| {
            let pos_env_start = atm[1] as usize;
            cint_env[pos_env_start..pos_env_start+3].iter_mut().zip(position.iter()).for_each(|(x, y)| {*x = *y})
            //cint_env[pos_env_start] = position[0];
            //cint_env[pos_env_start+1] = position[1];
            //cint_env[pos_env_start+2] = position[2];
        });
        cint_env
    }

    pub fn collect_auxbas(ctrl: &InputKeywords,geom: &GeomCell, etb: Option<InfoV2>) -> 
            (Vec<Basis4Elem>, Vec<Vec<i32>>, Vec<Vec<i32>>, Vec<f64>, Vec<BasInfo>, Vec<Vec<usize>>, usize) {

        let mut aux_atm: Vec<Vec<i32>> = vec![];
        let mut aux_env: Vec<f64> = vec![0.0; ENV_PRT_START as usize];
        let mut geom_start: i32 = ENV_PRT_START as i32;
        let cint_type = if ctrl.basis_type.to_lowercase()==String::from("spheric") {
            CintType::Spheric
        } else if ctrl.basis_type.to_lowercase()==String::from("cartesian") {
            CintType::Cartesian
        } else {
            panic!("Error:: Unknown basis type '{}'. Please use either 'spheric' or 'Cartesian'", 
                   ctrl.basis_type);
        };


        // for standard atoms
        let mass_charge = get_mass_charge(&geom.elem);
        geom.elem.iter().enumerate().zip(mass_charge.iter())
            .for_each(|((atm_index,atm_elem),(tmp_mass,tmp_charge))| {
            aux_atm.push(vec![*tmp_charge as i32,geom_start,NUC_STAD_CHARGE,geom_start+3,0,0]);
            (0..3).into_iter().for_each(|i| {
                if let Some(tmp_value) = geom.position.get(&[i,atm_index]) {
                    //for the coordinates
                    aux_env.push(*tmp_value);
                }
            });
            //for the nuclear charge distribution parameter
            aux_env.push(0.0);
            geom_start += 4;
        });
        // for ghost atoms with basis sets
        if geom.ghost_bs_elem.len() > 0 {
            let ghost_mass_charge = get_mass_charge(&geom.ghost_bs_elem);
            geom.ghost_bs_elem.iter().enumerate().zip(ghost_mass_charge.iter())
                .for_each(|((atm_index, atm_elem), (tmp_mass, tmp_charge))| {
                aux_atm.push(vec![0, geom_start, NUC_STAD_CHARGE, geom_start+3, 0, 0]);
                (0..3).into_iter().for_each(|i| {
                    if let Some(tmp_value) = geom.ghost_bs_pos.get(&[i, atm_index]) {
                        //for the coordinates
                        aux_env.push(*tmp_value);
                    }
                });
                //for the nuclear charge distribution parameter
                aux_env.push(0.0);
                geom_start += 4;
            });
        }

        // Now for bas inf.
        let mut auxbas_total: Vec<Basis4Elem> = vec![];
        let mut aux_bas: Vec<Vec<i32>> = vec![];
        let mut basis_start = geom_start;
        let mut auxbas_info: Vec<BasInfo> = vec![];
        let mut aux_cint_fdqc: Vec<Vec<usize>> = vec![];

        let ctrl_elem = ctrl_element_checker(geom);
        let local_elem = local_element_checker(&ctrl.auxbas_path);
        //if ctrl.print_level>0 {
        //    println!("Elements involved are {:?}", ctrl_elem);
        //    println!("Elements with local basis set available are {:?}", local_elem);
        //    println!("Start to gather the auxiliary basis sets of the rest elements online")
        //}
        let etb_elem = get_etb_elem(geom, &ctrl.etb_start_atom_number);

        let elem_intersection = ctrl_elem.intersect(local_elem.clone());
        let mut required_elem = vec![];
        for ctrl_item in ctrl_elem {
            if !elem_intersection.contains(&ctrl_item) {
                if ! etb_elem.contains(&ctrl_item) {
                    required_elem.push(ctrl_item)
                }
            }
        }
    
        if required_elem.len() != 0 {
            //bse_auxbas_getter insert here
            let re = Regex::new(r"/?(?P<basis>[^/]*)/?$").unwrap();
            let cap = re.captures(&ctrl.auxbas_path).unwrap();
            let auxbas_name = cap.name("basis").unwrap().as_str().to_string();
            if ctrl.print_level > 0 {
                println!("auxbas_name = {} from {}", &auxbas_name, &ctrl.auxbas_path)
            };
            if check_basis_name(&auxbas_name) {
                bse_downloader::bse_auxbas_getter_v2(&auxbas_name,&geom, &ctrl.auxbas_path, &required_elem, ctrl.print_level);
            }
            else {
                if ctrl.print_level > 0 {
                    println!("Error: Missing local aux basis sets for {:?}", &required_elem);
                }
                let matched = basis_fuzzy_matcher(&auxbas_name);
                match matched {
                    Some(_) => panic!("{} may not be a valid basis set name, similar name is {}", auxbas_name, matched.unwrap()),
                    None => panic!("{} may not be a valid basis set name, please check.", auxbas_name),
            }
        }

        };

        //import the basis set of atoms and ghost atoms one by one
        let elem_tot = geom.elem.clone().into_iter().chain(geom.ghost_bs_elem.clone().into_iter()).collect::<Vec<_>>();
        for (atm_index, atm_elem) in elem_tot.iter().enumerate() {
            let tmp_path = format!("{}/{}.json",&ctrl.auxbas_path, &atm_elem);
            let mut tmp_basis = match &etb {
                None => Basis4Elem::parse_json_from_file(tmp_path,&cint_type).unwrap(),
                Some(x) => { match x.elements.get(atm_elem) {
                    Some(y) => y.clone(),
                    None => Basis4Elem::parse_json_from_file(tmp_path,&cint_type).unwrap(),
                }},
            };
           
            //println!("tmp_auxbasis = {:?}", tmp_basis);

            let mut num_basis_per_atm = 0_usize;
            for tmp_bascell in &tmp_basis.electron_shells {
                let mut num_primitive: i32 = tmp_bascell.exponents.len() as i32;
                let mut num_contracted: i32 = tmp_bascell.coefficients.len() as i32;
                let mut angular_mom: i32 = tmp_bascell.angular_momentum[0];
                let tmp_bas_info = BasInfo::new();
                aux_env.extend(tmp_bascell.exponents.clone());
                for (index, coe_vec) in tmp_bascell.coefficients.iter().enumerate() {
                    aux_env.extend(coe_vec.clone());
                };
                let mut tmp_bas_vec: Vec<i32> = vec![atm_index as i32, 
                            angular_mom,
                            num_primitive,
                            num_contracted,
                            0,
                            basis_start,
                            basis_start+num_primitive,
                            0];
                let (ang,tmp_bas_num) = match &cint_type {
                    CintType::Cartesian => {let ang = tmp_bas_vec[1] as usize; (ang,(ang+1)*(ang+2)/2)},
                    CintType::Spheric => {let ang = tmp_bas_vec[1] as usize; (ang, ang*2+1)},
                    // NOTE:: Spinor is not implemented
                    CintType::Spinor => {let ang = tmp_bas_vec[1] as usize; (ang, ang*2+1)},
                };
                let mut tmp_len = 0;
                let tmp_start = if aux_cint_fdqc.len()==0 {0} 
                                    else {aux_cint_fdqc[aux_cint_fdqc.len()-1][0]+aux_cint_fdqc[aux_cint_fdqc.len()-1][1]};
                //Now for bas info. of each basis function and their link to the libcint data structure
                (0..num_contracted as usize).into_iter().for_each(|index0| {
                    (0..tmp_bas_num).into_iter().for_each(|index1| {
                        let bas_type = if num_primitive == 1 {
                            String::from("Primitive")
                        } else {
                            String::from("Contracted")
                        };
                        tmp_len += 1;
                        auxbas_info.push(BasInfo {
                            bas_name: get_basis_name(ang, &cint_type, index1),
                            bas_type,
                            elem_index0: atm_index,
                            cint_index0: aux_bas.len(),
                            cint_index1: index0*tmp_bas_num+index1,
                        });
                        //println!("auxbas_info = {:?}", auxbas_info);
                    });
                });
                aux_cint_fdqc.push(vec![tmp_start,tmp_len]);
                aux_bas.push(tmp_bas_vec);
                basis_start += num_primitive + num_primitive*num_contracted;
                num_basis_per_atm += tmp_len 
            }
            auxbas_total.push(tmp_basis);
            if atm_index !=0 {
                auxbas_total[atm_index].global_index.0 = auxbas_total[atm_index-1].global_index.0 + auxbas_total[atm_index-1].global_index.1; 
                auxbas_total[atm_index].global_index.1 = num_basis_per_atm;
            } else {
                auxbas_total[atm_index].global_index.0 = 0;
                auxbas_total[atm_index].global_index.1 = num_basis_per_atm;
            }
        };
        // At current stage, we skip the linear-dependence check of the basis sets
        let num_auxbas = auxbas_info.len();

        //println!("auxbas_total = {:?}", auxbas_total);
        (auxbas_total, aux_atm, aux_bas, aux_env,auxbas_info,aux_cint_fdqc,num_auxbas)
    }
}

pub fn build_cint(
    basis_per_atom: &[Basis4Elem],
    geom: &GeomCell,
    cint_type: &CintType,
) -> (Vec<Vec<i32>>, Vec<Vec<i32>>, Vec<f64>, Vec<BasInfo>, Vec<Vec<usize>>, usize, Option<Vec<Vec<i32>>>) {
    // Prepare atm info.
    let mut atm: Vec<Vec<i32>> = vec![];
    let mut env: Vec<f64> = vec![0.0;ENV_PRT_START];
    let mut geom_start: i32 = ENV_PRT_START as i32;

    // for standard atoms
    let mass_charge = get_mass_charge(&geom.elem);
    geom.elem.iter().enumerate().zip(mass_charge.iter())
        .for_each(|((atm_index,atm_elem),(tmp_mass,tmp_charge))| {
        atm.push(vec![*tmp_charge as i32,geom_start,NUC_STAD_CHARGE,geom_start+3,0,0]);
        (0..3).into_iter().for_each(|i| {
            if let Some(tmp_value) = geom.position.get(&[i,atm_index]) {
                env.push(*tmp_value);
            }
        });
        env.push(0.0);
        geom_start += 4;
    });
    // for ghost atoms with basis sets
    if geom.ghost_bs_elem.len() > 0 {
        let ghost_mass_charge = get_mass_charge(&geom.ghost_bs_elem);
        geom.ghost_bs_elem.iter().enumerate().zip(ghost_mass_charge.iter())
            .for_each(|((atm_index, atm_elem), (tmp_mass, tmp_charge))| {
            atm.push(vec![0, geom_start, NUC_STAD_CHARGE, geom_start+3, 0, 0]);
            (0..3).into_iter().for_each(|i| {
                if let Some(tmp_value) = geom.ghost_bs_pos.get(&[i, atm_index]) {
                    env.push(*tmp_value);
                }
            });
            env.push(0.0);
            geom_start += 4;
        });
    }

    // Now for bas inf.
    let mut bas: Vec<Vec<i32>> = vec![];
    let mut basis_start = geom_start;

    for (atm_index, tmp_basis) in basis_per_atom.iter().enumerate() {
        for tmp_bascell in &tmp_basis.electron_shells {
            let num_primitive: i32 = tmp_bascell.exponents.len() as i32;
            let num_contracted: i32 = tmp_bascell.coefficients.len() as i32;
            let angular_mom: i32 = tmp_bascell.angular_momentum[0];
            tmp_bascell.exponents.iter().for_each(|x| {
                env.push(*x);
            });
            for coe_vec in &tmp_bascell.coefficients {
                coe_vec.iter().for_each(|x| {
                    env.push(*x);
                });
            };
            bas.push(vec![atm_index as i32,
                        angular_mom,
                        num_primitive,
                        num_contracted,
                        0,
                        basis_start,
                        basis_start+num_primitive,
                        0]);
            basis_start += num_primitive + num_primitive*num_contracted;
        }
    }

    let (bas_info, cint_fdqc) = basis::build_fdqc(&bas, cint_type);
    let num_basis = bas_info.len();

    // now import ecp basis infom.
    let mut ecpbas: Vec<Vec<i32>> = vec![];
    let mut ecp_start = env.len() as i32;
    let ghost_atm_index = geom.get_start_index_of_ghost_atoms();
    basis_per_atom.iter().zip(atm.iter_mut().enumerate()).for_each(|(bas, (atm_index,cur_atm))| {
        if atm_index < ghost_atm_index {
            if let (Some(ecp), Some(necp))  = (&bas.ecp_potentials, &bas.ecp_electrons) {
                cur_atm[ATM_NUC] -= *necp as i32;
                cur_atm[ATM_NUC_MOD_OF] = NUC_ECP;

                let ecp_ang_start = (ecp.len()-1) as i32;
                for ecpcell in ecp.iter() {
                    let angl = ecpcell.angular_momentum[0];
                    let coeffs = &ecpcell.coefficients;
                    let gaussian_exponents = &ecpcell.gaussian_exponents;
                    let num_exp = gaussian_exponents.len() as i32;
                    let num_coeffs = coeffs.len() as i32;

                    let mut r_exponents_group: HashMap<i32, Vec<usize>> = HashMap::new();

                    ecpcell.r_exponents.iter().enumerate().for_each(|(index, r_exponents)| {
                        if r_exponents_group.contains_key(r_exponents) {
                            r_exponents_group.get_mut(r_exponents).unwrap().push(index);
                        } else {
                            r_exponents_group.insert(*r_exponents, vec![index]);
                        }
                    });

                    r_exponents_group.iter().for_each(|(r_exponent, index_vec)| {
                        let mut ecp_exp_start = env.len() as i32;
                        let num_index_vec = index_vec.len() as i32;
                        env.extend(gaussian_exponents.iter().enumerate()
                            .filter(|(index, x)| index_vec.contains(index) )
                            .map(|(_, x)| x)
                        );
                        if num_index_vec > num_exp {
                            panic!("bad ecp basis for the elem of {}", &geom.elem[atm_index]);
                        }
                        coeffs.iter().for_each(|each_coeffs| {
                            let len_coeffs = each_coeffs.len() as i32;
                            if len_coeffs != num_exp {
                                panic!("bad ecp basis for the elem of {}", &geom.elem[atm_index]);
                            }
                            let mut ecp_coeff_start = env.len() as i32;
                            let mut tmp_ecpbas_vec: Vec<i32> = vec![atm_index as i32,
                                        if angl==ecp_ang_start {-1} else {angl},
                                        num_index_vec,
                                        *r_exponent,
                                        0,
                                        ecp_exp_start,
                                        ecp_coeff_start,
                                        0];
                            ecpbas.push(tmp_ecpbas_vec);

                            env.extend(each_coeffs.iter().enumerate()
                                .filter(|(index, x)| index_vec.contains(index))
                                .map(|(_, x)| x)
                            );
                        });
                    });
                }
            };
        }
    });

    let final_ecpbas = if ecpbas.len() == 0 {None} else {Some(ecpbas)};

    (atm, bas, env, bas_info, cint_fdqc, num_basis, final_ecpbas)
}

impl Molecule {
    pub fn collect_basis(ctrl: &InputKeywords,geom: &GeomCell) -> 
            (Vec<Basis4Elem>, Vec<Vec<i32>>, Vec<Vec<i32>>, Vec<f64>, Vec<BasInfo>, Vec<Vec<usize>>, usize, usize, Option<Vec<Vec<i32>>>) {

        let cint_type = if ctrl.basis_type.to_lowercase()==String::from("spheric") {
            CintType::Spheric
        } else if ctrl.basis_type.to_lowercase()==String::from("cartesian") {
            CintType::Cartesian
        } else {
            panic!("Error:: Unknown basis type '{}'. Please use either 'spheric' or 'Cartesian'", 
                   ctrl.basis_type);
        };

        let ctrl_elem = ctrl_element_checker(geom);
        let local_elem = local_element_checker(&ctrl.basis_path);
        let elem_intersection = ctrl_elem.intersect(local_elem.clone());
        let mut required_elem = vec![];
        for ctrl_item in ctrl_elem {
            if !elem_intersection.contains(&ctrl_item) {
                required_elem.push(ctrl_item)
            }
        }

        if required_elem.len() != 0 {
            let re = Regex::new(r"/?(?P<basis>[^/]*)/?$").unwrap();
            let cap = re.captures(&ctrl.basis_path).unwrap();
            let basis_name = cap.name("basis").unwrap().as_str().to_string();
            if check_basis_name(&basis_name) {
                bse_downloader::bse_basis_getter_v2(&basis_name,&geom, &ctrl.basis_path, &required_elem);
            }  else {
                if ctrl.print_level > 0 {
                    println!("Error: Missing local basis sets for {:?}", &required_elem);
                }
                let matched = basis_fuzzy_matcher(&basis_name);
                match matched {
                    Some(_) => panic!("{} may not be a valid basis set name, similar name is {}", basis_name, matched.unwrap()),
                    None => panic!("{} may not be a valid basis set name, please check.", basis_name),
                }
            }
        };

        let basis_total = basis::read_basis_per_atom(ctrl, geom, &cint_type);

        let (atm, bas, env, bas_info, cint_fdqc, num_basis, ecpbas) =
            build_cint(&basis_total, geom, &cint_type);

        let num_state = num_basis;

        (basis_total, atm, bas, env, bas_info, cint_fdqc, num_basis, num_state, ecpbas)
    }

    pub fn int_ij_matrixuppers(&self,op_name: String, comp: usize) -> Vec<MatrixUpper<f64>> {
        let mut cur_op = op_name.clone();
        let mut cint_data = self.initialize_cint(false);

        //if op_name == String::from("dipole") {
        //    let comp = 3;
        //}

        let nbas = self.fdqc_bas.len();
        let tmp_size:usize = nbas*(nbas+1)/2;
        let mut mat_global = vec![MatrixUpper::new(tmp_size,0.0);comp];
        let nbas_shell = self.cint_bas.len();
        for j in 0..nbas_shell {
            let bas_start_j = self.cint_fdqc[j][0];
            let bas_len_j = self.cint_fdqc[j][1];
            // for i < j
            for i in 0..j {
                let bas_start_i = self.cint_fdqc[i][0];
                let bas_len_i = self.cint_fdqc[i][1];
                let tmp_size = [bas_len_i,bas_len_j,comp];
                let mat_local = RIFull::from_vec(tmp_size,
                    cint_data.cint_ij(i as i32, j as i32, &cur_op)).unwrap();
                (0..comp).into_iter().for_each(|tmp_comp| {
                    let mat_local_full = mat_local.get_reducing_matrix(tmp_comp).unwrap();
                    let mut mat_global_full = mat_global.get_mut(tmp_comp).unwrap();
                    (0..bas_len_j).into_iter().for_each(|tmp_j| {
                        let gj = tmp_j + bas_start_j;
                        let global_ij_start = gj*(gj+1)/2+bas_start_i;
                        let local_ij_start = tmp_j*bas_len_i;
                        //let length = if bas_start_i+bas_len_i <= gj+1 {bas_len_i} else {gj+1-bas_start_i};
                        let length = bas_len_i;
                        let mat_global_j = mat_global_full.get1d_slice_mut(global_ij_start,length).unwrap();
                        let mat_local_j = mat_local_full.get1d_slice(local_ij_start,length).unwrap();
                        mat_global_j.iter_mut().zip(mat_local_j.iter()).for_each(|(gij,lij)| {
                            *gij = *lij
                        });
                    });
                });
            };
            // for i = j 
            let tmp_size = [bas_len_j,bas_len_j,comp];
            let vec_local = cint_data.cint_ij(j as i32, j as i32, &cur_op);
            //println!("for {}, i=j={}: {:?}", &cur_op, j, &vec_local);
            let mat_local = RIFull::from_vec(tmp_size,vec_local).unwrap();
            //if j==0 {
            //}
            (0..comp).into_iter().for_each(|tmp_comp| {
                let mat_local_full = mat_local.get_reducing_matrix(tmp_comp).unwrap();
                let mut mat_global_full = mat_global.get_mut(tmp_comp).unwrap();
                (0..bas_len_j).into_iter().for_each(|tmp_j| {
                    let gj = bas_start_j + tmp_j;
                    let global_ij_start = gj*(gj+1)/2+bas_start_j;
                    let local_ij_start = tmp_j*bas_len_j;
                    let length = tmp_j + 1;
                    let mat_global_j = mat_global_full.get1d_slice_mut(global_ij_start,length).unwrap();
                    let mat_local_j = mat_local_full.get1d_slice(local_ij_start,length).unwrap();
                    mat_global_j.iter_mut().zip(mat_local_j.iter()).for_each(|(gij,lij)| {
                        *gij = *lij
                    });
                })
            });
        }

        cint_data.final_c2r();
        //vec_2d
        mat_global


    }

    pub fn int_ij_matrixupper(&self, op_name: String) -> MatrixUpper<f64> {
        self.int_ij_matrixupper_v02(op_name)
    }

    #[inline]
    pub fn int_ij_matrixupper_v01(&self,op_name: String) -> MatrixUpper<f64> {
        let mut cint_data = self.initialize_cint(false);

        if op_name.eq("dipole") {
            panic!("Error:: for dipole moment calculation: {}, please use int_ij_matrxuppers",&op_name);
        }

        let mut cur_op = op_name.clone();
        if op_name == String::from("hcore") {
            cur_op = String::from("kinetic");
        };

        let nbas = self.fdqc_bas.len();
        let tmp_size:usize = nbas*(nbas+1)/2;
        let mut mat_global = MatrixUpper::new(tmp_size,0.0);
        let nbas_shell = self.cint_bas.len();
        for j in 0..nbas_shell {
            let bas_start_j = self.cint_fdqc[j][0];
            let bas_len_j = self.cint_fdqc[j][1];
            // for i < j
            for i in 0..j {
                let bas_start_i = self.cint_fdqc[i][0];
                let bas_len_i = self.cint_fdqc[i][1];
                let tmp_size = [bas_len_i,bas_len_j];
                let mat_local = MatrixFull::from_vec(tmp_size,
                    cint_data.cint_ij(i as i32, j as i32, &cur_op)).unwrap();
                (0..bas_len_j).into_iter().for_each(|tmp_j| {
                    let gj = tmp_j + bas_start_j;
                    let global_ij_start = gj*(gj+1)/2+bas_start_i;
                    let local_ij_start = tmp_j*bas_len_i;
                    //let length = if bas_start_i+bas_len_i <= gj+1 {bas_len_i} else {gj+1-bas_start_i};
                    let length = bas_len_i;
                    let mat_global_j = mat_global.get1d_slice_mut(global_ij_start,length).unwrap();
                    let mat_local_j = mat_local.get1d_slice(local_ij_start,length).unwrap();
                    mat_global_j.iter_mut().zip(mat_local_j.iter()).for_each(|(gij,lij)| {
                        *gij = *lij
                    });
                });
            };
            // for i = j 
            let tmp_size = [bas_len_j,bas_len_j];
            let mat_local = MatrixFull::from_vec(tmp_size,
                cint_data.cint_ij(j as i32, j as i32, &cur_op)).unwrap();
            (0..bas_len_j).into_iter().for_each(|tmp_j| {
                let gj = bas_start_j + tmp_j;
                let global_ij_start = gj*(gj+1)/2+bas_start_j;
                let local_ij_start = tmp_j*bas_len_j;
                let length = tmp_j + 1;
                let mat_global_j = mat_global.get1d_slice_mut(global_ij_start,length).unwrap();
                let mat_local_j = mat_local.get1d_slice(local_ij_start,length).unwrap();
                mat_global_j.iter_mut().zip(mat_local_j.iter()).for_each(|(gij,lij)| {
                    *gij = *lij
                });
            });
        }
        if op_name.eq("hcore") {
            //println!("Debug: The Kinetic matrix:");
            //mat_global.formated_output(5, "lower".to_string());
            cint_data.cint_del_optimizer_rust();
            cint_data.cint1e_nuc_optimizer_rust();
            let cur_op = String::from("nuclear");
            //if let Some(ecpbas) = &self.cint_ecpbas {
            //    cur_op_list.push(String::from("ecp"));
            //}
            for j in 0..nbas_shell {
                let bas_start_j = self.cint_fdqc[j][0];
                let bas_len_j = self.cint_fdqc[j][1];
                // for i < j
                for i in 0..j {
                    let bas_start_i = self.cint_fdqc[i][0];
                    let bas_len_i = self.cint_fdqc[i][1];
                    let tmp_size = [bas_len_i,bas_len_j];
                    let mat_local = MatrixFull::from_vec(tmp_size,
                        cint_data.cint_ij(i as i32, j as i32, &cur_op)).unwrap();
                    (0..bas_len_j).into_iter().for_each(|tmp_j| {
                        let gj = tmp_j + bas_start_j;
                        let global_ij_start = gj*(gj+1)/2+bas_start_i;
                        let local_ij_start = tmp_j*bas_len_i;
                        //let length = if bas_start_i+bas_len_i <= gj+1 {bas_len_i} else {gj+1-bas_start_i};
                        let length = bas_len_i;
                        //println!("debug: global_ij_start:{}, length: {}",global_ij_start, length);
                        let mat_global_j = mat_global.get1d_slice_mut(global_ij_start,length).unwrap();
                        let mat_local_j = mat_local.get1d_slice(local_ij_start,length).unwrap();
                        mat_global_j.iter_mut().zip(mat_local_j.iter()).for_each(|(gij,lij)| {
                            *gij += *lij
                        });
                    });
                };
                // for i = j 
                let tmp_size = [bas_len_j,bas_len_j];
                let mat_local = MatrixFull::from_vec(tmp_size,
                    cint_data.cint_ij(j as i32, j as i32, &cur_op)).unwrap();
                //if j==0 {println!("debug: I:{}, bas_len_j:{} cur_op: {}, {:?}", j,bas_start_j, &cur_op, &mat_local.data)};
                (0..bas_len_j).into_iter().for_each(|tmp_j| {
                    let gj = bas_start_j + tmp_j;
                    let global_ij_start = gj*(gj+1)/2+bas_start_j;
                    let local_ij_start = tmp_j*bas_len_j;
                    let length = tmp_j + 1;
                    //println!("debug: global_ij_start:{}, length: {}",global_ij_start, length);
                    let mat_global_j = mat_global.get1d_slice_mut(global_ij_start,length).unwrap();
                    let mat_local_j = mat_local.get1d_slice(local_ij_start,length).unwrap();
                    mat_global_j.iter_mut().zip(mat_local_j.iter()).for_each(|(gij,lij)| {
                        *gij += *lij
                    });
                });
            }
            if let Some(ecpbas) = &self.cint_ecpbas {
                cint_data.cint1e_ecp_optimizer_rust();
                let cur_op = String::from("ecp");
                 for j in 0..nbas_shell {
                     let bas_start_j = self.cint_fdqc[j][0];
                     let bas_len_j = self.cint_fdqc[j][1];
                     // for i < j
                     for i in 0..j {
                         let bas_start_i = self.cint_fdqc[i][0];
                         let bas_len_i = self.cint_fdqc[i][1];
                         let tmp_size = [bas_len_i,bas_len_j];
                         let mat_local = MatrixFull::from_vec(tmp_size,
                             cint_data.cint_ij(i as i32, j as i32, &cur_op)).unwrap();
                         (0..bas_len_j).into_iter().for_each(|tmp_j| {
                             let gj = tmp_j + bas_start_j;
                             let global_ij_start = gj*(gj+1)/2+bas_start_i;
                             let local_ij_start = tmp_j*bas_len_i;
                             //let length = if bas_start_i+bas_len_i <= gj+1 {bas_len_i} else {gj+1-bas_start_i};
                             let length = bas_len_i;
                             //println!("debug: global_ij_start:{}, length: {}",global_ij_start, length);
                             let mat_global_j = mat_global.get1d_slice_mut(global_ij_start,length).unwrap();
                             let mat_local_j = mat_local.get1d_slice(local_ij_start,length).unwrap();
                             mat_global_j.iter_mut().zip(mat_local_j.iter()).for_each(|(gij,lij)| {
                                 *gij += *lij
                             });
                         });
                     };
                     // for i = j 
                     let tmp_size = [bas_len_j,bas_len_j];
                     let mat_local = MatrixFull::from_vec(tmp_size,
                         cint_data.cint_ij(j as i32, j as i32, &cur_op)).unwrap();
                     //if j==0 {println!("debug: I:{}, bas_len_j:{} cur_op: {}, {:?}", j,bas_start_j, &cur_op, &mat_local.data)};
                     (0..bas_len_j).into_iter().for_each(|tmp_j| {
                         let gj = bas_start_j + tmp_j;
                         let global_ij_start = gj*(gj+1)/2+bas_start_j;
                         let local_ij_start = tmp_j*bas_len_j;
                         let length = tmp_j + 1;
                         //println!("debug: global_ij_start:{}, length: {}",global_ij_start, length);
                         let mat_global_j = mat_global.get1d_slice_mut(global_ij_start,length).unwrap();
                         let mat_local_j = mat_local.get1d_slice(local_ij_start,length).unwrap();
                         mat_global_j.iter_mut().zip(mat_local_j.iter()).for_each(|(gij,lij)| {
                             *gij += *lij
                         });
                     });
                 }
            }
            //println!("The h-core matrix:");
            //mat_global.formated_output(5, "lower".to_string());
        }
        cint_data.final_c2r();
        //vec_2d
        mat_global
    }

    pub fn int_ij_matrixupper_v02(&self, op_name: String) -> MatrixUpper<f64> {
        use rest_libcint_wrapper::*;
        let mut cint_data = self.initialize_cint(false);
        //let mut cur_op = op_name.to_string();
        let mut out = vec![]; 
        let mut out_shape = vec![];
        if op_name.eq("ovlp") {
            (out,out_shape) = cint_data.integral_s2ij::<int1e_ovlp>(None);
        } else if op_name.eq("kinetic") {
            (out, out_shape) = cint_data.integral_s2ij::<int1e_kin>(None);
        } else if op_name.eq("nuclear") {
            (out, out_shape) = cint_data.integral_s2ij::<int1e_nuc>(None);
        } else if op_name.eq("ecp") {
            let mut tmp_out = vec![];
            let mut tmp_out_shape = vec![];
            //(out, out_shape) = cint_data.integral_ecp_s1::<ECPscalar>(None);
            (tmp_out, tmp_out_shape) = cint_data.integral_ecp_s1::<ECPscalar>(None);
            let tmp_ecp = MatrixFull::from_vec(tmp_out_shape.try_into().unwrap(), tmp_out).unwrap();

            out = tmp_ecp.iter_matrixupper().unwrap().map(|x| *x).collect::<Vec<f64>>();
            //.iter_matrixupper().unwrap()).for_each(|(o,f)| {
            //    *o = *f
            //});
        } else if op_name.eq("point charge") {
            // for the ghost point charge term
            // <a|-Za/|Ra-r||b> -> -Za*<a|1/|r||b> by setting PTR_RINV_ORIG as Ra
            let orig_orig = cint_data.get_rinv_origin();
            self.geom.ghost_pc_chrg.iter().zip(self.geom.ghost_pc_pos.iter_columns_full()).for_each(|(charge, pos)| {
                let mut tmp_out = vec![];
                let mut tmp_out_shape = vec![];
                cint_data.set_rinv_origin([pos[0], pos[1], pos[2]]);
                (tmp_out, tmp_out_shape) = cint_data.integral_s2ij::<int1e_rinv>(None);
                //println!("debug pos: {:?}, charge: {}", pos, charge);
                if out.len() == 0 {
                    out = vec![0.0;tmp_out.len()];
                }
                out.iter_mut().zip(tmp_out.iter()).for_each(|(o, t)| {
                    *o -= *t * charge
                });
            });

            cint_data.set_rinv_origin(orig_orig);
        } else if op_name.eq("hcore") {
            // for the kinetic term
            (out, out_shape) = cint_data.integral_s2ij::<int1e_kin>(None);
            let mut tmp_out = vec![];
            let mut tmp_out_shape = vec![];
            // for the standard nuclear term
            (tmp_out, tmp_out_shape) = cint_data.integral_s2ij::<int1e_nuc>(None);
            out.iter_mut().zip(tmp_out.iter()).for_each(|(o, t)| {
                *o += *t
            });
            // for the standard ECP term
            if let Some(ecpbas) = &self.cint_ecpbas {
                (tmp_out, tmp_out_shape) = cint_data.integral_ecp_s1::<ECPscalar>(None);
                let tmp_ecp = MatrixFull::from_vec([self.num_basis,self.num_basis], tmp_out).unwrap();

                out.iter_mut().zip(tmp_ecp.iter_matrixupper().unwrap()).for_each(|(o,f)| {
                    *o += *f
                });
            }

        } else {
            panic!("Error:: op_name: {}", op_name);
        }

        MatrixUpper::from_vec(out.len(), out).unwrap()

    }

    #[inline]
    pub fn int_ijkl_given_kl(&self, k: usize, l: usize) -> Vec<MatrixFull<f64>> {
        let bas_start_l = self.cint_fdqc[l][0];
        let bas_len_l = self.cint_fdqc[l][1];
        let bas_start_k = self.cint_fdqc[k][0];
        let bas_len_k = self.cint_fdqc[k][1];
        let nbas = self.num_basis;
        let mut mat_vec = vec![MatrixFull::new([nbas,nbas],0.0);bas_len_k*bas_len_l];

        let mut cint_data = self.initialize_cint(false);
        let nbas_shell = self.cint_bas.len();
        cint_data.cint2e_optimizer_rust();
        for j in 0..nbas_shell {
            let bas_start_j = self.cint_fdqc[j][0];
            let bas_len_j = self.cint_fdqc[j][1];
            //let (i_start, i_end) = (0,j+1);
            for i in 0..j+1 {
                let bas_start_i = self.cint_fdqc[i][0];
                let bas_len_i = self.cint_fdqc[i][1];
                let buf = cint_data.cint_ijkl_by_shell(i as i32, j as i32, k as i32, l as i32);
                let tmp_eri = ERIFull::from_vec([bas_len_i,bas_len_j,bas_len_k,bas_len_l],buf).unwrap();

                for loc_l in 0..bas_len_l {
                    for loc_k in 0..bas_len_k {
                        let tmp_index = loc_l*bas_len_k + loc_k;
                        let mut tmp_mat = &mut mat_vec[tmp_index];

                        let tmp_loc = tmp_eri.get_reducing_matrix(&[loc_k,loc_l]).data.iter();

                        tmp_mat.iter_submatrix_mut(bas_start_i..bas_start_i+bas_len_i, bas_start_j..bas_start_j+bas_len_j)
                        .zip(tmp_loc).for_each(|(gij,lij)| {*gij = *lij});
                    }
                }
            };
        };
        //mat_full
        mat_vec
    }

    #[inline]
    pub fn int_ijkl_given_kl_v02(&self, k: usize, l: usize) -> MatrixFull<MatrixFull<f64>> {
        let bas_start_l = self.cint_fdqc[l][0];
        let bas_len_l = self.cint_fdqc[l][1];
        let bas_start_k = self.cint_fdqc[k][0];
        let bas_len_k = self.cint_fdqc[k][1];
        let nbas = self.num_basis;
        let mut mat_full = 
            MatrixFull::new([bas_len_k,bas_len_l],MatrixFull::new([nbas,nbas],0.0));
        //let mut mat_vec = vec![MatrixFull::new([nbas,nbas],0.0);bas_len_k*bas_len_l];

        let mut cint_data = self.initialize_cint(false);
        let nbas_shell = self.cint_bas.len();
        cint_data.cint2e_optimizer_rust();
        for j in 0..nbas_shell {
            let bas_start_j = self.cint_fdqc[j][0];
            let bas_len_j = self.cint_fdqc[j][1];
            //let (i_start, i_end) = (0,j+1);
            for i in 0..j+1 {
                let bas_start_i = self.cint_fdqc[i][0];
                let bas_len_i = self.cint_fdqc[i][1];
                let buf = cint_data.cint_ijkl_by_shell(i as i32, j as i32, k as i32, l as i32);
                let tmp_eri = ERIFull::from_vec([bas_len_i,bas_len_j,bas_len_k,bas_len_l],buf).unwrap();

                for loc_l in 0..bas_len_l {
                    for loc_k in 0..bas_len_k {
                        //let tmp_index = loc_l*bas_len_k + loc_k;
                        //let mut tmp_mat = &mut mat_vec[tmp_index];
                        let mut tmp_mat = &mut mat_full[[loc_k,loc_l]];

                        let tmp_loc = tmp_eri.get_reducing_matrix(&[loc_k,loc_l]);

                        tmp_mat.iter_submatrix_mut(bas_start_i..bas_start_i+bas_len_i, bas_start_j..bas_start_j+bas_len_j)
                        .zip(tmp_loc.data.iter()).for_each(|(gij,lij)| {*gij = *lij});
                    }
                }
            };
        };
        mat_full
        //mat_vec
    }
    #[inline]
    pub fn int_ijkl_given_kl_v03(&self, k: usize, l: usize, matrixupper_index: &MatrixUpper<[usize;2]>, cint_data: &mut CINTR2CDATA) -> MatrixFull<MatrixUpper<f64>> {

        let bas_start_l = self.cint_fdqc[l][0];
        let bas_len_l = self.cint_fdqc[l][1];
        let bas_start_k = self.cint_fdqc[k][0];
        let bas_len_k = self.cint_fdqc[k][1];
        let nbas = self.num_basis;

        //println!("{}-{},{}-{}, size:{}",bas_start_k, bas_len_k, bas_start_l,bas_len_l, matrixupper_index.size);
        let mut mat_full = 
            //MatrixFull::new([bas_len_k,bas_len_l],MatrixFull::new([nbas,nbas],0.0));
            MatrixFull::new([bas_len_k,bas_len_l],MatrixUpper::new(matrixupper_index.size,0.0));
        //let mut mat_vec = vec![MatrixFull::new([nbas,nbas],0.0);bas_len_k*bas_len_l];

        //let mut cint_data = self.initialize_cint(false);
        let nbas_shell = self.cint_bas.len();
        //cint_data.cint2e_optimizer_rust();
        for j in 0..nbas_shell {
            let bas_start_j = self.cint_fdqc[j][0];
            let bas_len_j = self.cint_fdqc[j][1];
            //let (i_start, i_end) = (0,j+1);
            for i in 0..j+1 {
                let bas_start_i = self.cint_fdqc[i][0];
                let bas_len_i = self.cint_fdqc[i][1];
                let buf = cint_data.cint_ijkl_by_shell(i as i32, j as i32, k as i32, l as i32);
                let tmp_eri = ERIFull::from_vec([bas_len_i,bas_len_j,bas_len_k,bas_len_l],buf).unwrap();

                for loc_l in 0..bas_len_l {
                    for loc_k in 0..bas_len_k {
                        //let tmp_index = loc_l*bas_len_k + loc_k;
                        //let mut tmp_mat = &mut mat_vec[tmp_index];
                        let mut tmp_mat = &mut mat_full[[loc_k,loc_l]];
                        let tmp_loc = tmp_eri.get_reducing_matrix(&[loc_k,loc_l]);

                        if i!=j {
                            tmp_mat.iter_submatrix_mut(bas_start_i..bas_start_i+bas_len_i, bas_start_j..bas_start_j+bas_len_j, matrixupper_index)
                                .zip(tmp_loc.iter()).for_each(|(gij,lij)| {*gij = *lij});
                        } else {
                            //println!("i=j={}:({},{})",i, bas_start_i,bas_len_i);
                            tmp_mat.iter_submatrix_mut(bas_start_i..bas_start_i+bas_len_i, bas_start_j..bas_start_j+bas_len_j, matrixupper_index)
                                .zip(tmp_loc.iter_matrixupper().unwrap()).for_each(|(gij,lij)| {*gij = *lij});

                        }
                    }
                }
            };
        };
        //cint_data.final_c2r();
        mat_full
        //mat_vec
    }
    #[inline]
    /// to save the memory required for vj_on_the_fly, evaluate the ERI elements batch by batch (batch: Range<usize>)
    pub fn int_ijkl_given_kl_batch(&self, k: usize, l: usize, batch: Range<usize>, matrixupper_index: &MatrixUpper<[usize;2]>) -> MatrixFull<SubMatrixUpper<f64>> {

        let bas_start_l = self.cint_fdqc[l][0];
        let bas_len_l = self.cint_fdqc[l][1];
        let bas_start_k = self.cint_fdqc[k][0];
        let bas_len_k = self.cint_fdqc[k][1];
        let nbas = self.num_basis;

        let global_start = batch.start;
        let global_length = batch.end;
        let [global_start_row, global_start_col] = matrixupper_index[global_start];
        let [global_end_row, global_end_col] = matrixupper_index[global_length-1];

        let mut mat_full = 
            MatrixFull::new([bas_len_k,bas_len_l],SubMatrixUpper::new(batch.clone(),matrixupper_index.size, 0.0));

        let mut cint_data = self.initialize_cint(false);
        let nbas_shell = self.cint_bas.len();
        cint_data.cint2e_optimizer_rust();
        for j in 0..nbas_shell {
            let bas_start_j = self.cint_fdqc[j][0];
            let bas_len_j = self.cint_fdqc[j][1];
            let rough_start = bas_start_j*(bas_start_j+1)/2;
            let rough_end = (bas_start_j+bas_len_j)*(bas_start_j+bas_len_j-1)/2;
            if rough_start < global_length {
                for i in 0..j+1 {
                    let bas_start_i = self.cint_fdqc[i][0];
                    let bas_len_i = self.cint_fdqc[i][1];
                    let curr_start = rough_start + bas_start_i;
                    let curr_end = rough_end + bas_start_i+bas_len_i-1;
                    ////debug 514
                    //if k==0 && l==0 && (batch.start == 60|| batch.start==75){
                    //    println!("batch_start: {}, batch_end: {}, bas_start_i: {}, bas_len_i: {}, bas_start_j: {}, bas_len_j: {}", batch.start, batch.end, bas_start_i, bas_len_i, bas_start_j, bas_len_j);
                    //    println!("global_s and e: ({},{}), ({},{}), curr_start: {}, curr_end: {}", 
                    //        global_start_row, global_start_col, 
                    //        global_end_row, global_end_col, 
                    //        curr_start, curr_end);
                    //}
                    if (curr_start >= global_start && curr_start < global_length) ||
                       (curr_end >= global_start && curr_end < global_length) || 
                       (curr_start < global_start && curr_end >= global_length) {

                        //if k==0 && l==0 && (batch.start == 60|| batch.start==75){
                        //    println!("debug: pass");
                        //}

                        let start_point = if curr_start >= global_start {
                            0_usize
                        } else {
                            if global_start_row < bas_start_i {
                                //(global_start_col-bas_start_j)*bas_len_i+bas_start_i
                                (global_start_col-bas_start_j)*bas_len_i
                            } else if global_start_row >= bas_start_i+bas_len_i {
                                (global_start_col-bas_start_j+1)*bas_len_i
                            } else {
                                (global_start_col-bas_start_j)*bas_len_i+(global_start_row-bas_start_i)
                            }
                        };
                        //let end_point = if curr_end <global_length {
                        //    bas_len_i*bas_len_j - 1
                        //} else {
                        //    global_end_col*bas_len_i + global_end_row - bas_start_j*bas_len_i  - bas_start_i
                        //};

                        let buf = cint_data.cint_ijkl_by_shell(i as i32, j as i32, k as i32, l as i32);
                        let tmp_eri = ERIFull::from_vec([bas_len_i,bas_len_j,bas_len_k,bas_len_l],buf).unwrap();

                        for loc_l in 0..bas_len_l {
                            for loc_k in 0..bas_len_k {

                                let mut tmp_mat = &mut mat_full[[loc_k,loc_l]];
                                let tmp_loc = tmp_eri.get_reducing_matrix(&[loc_k,loc_l]);

                                if i!=j {
                                    ////debug 514
                                    //if k==0 && l==0 && (batch.start == 60|| batch.start==75){
                                    //    println!("batch_start: {}, batch_end: {}, bas_start_i: {}, bas_len_i: {}, bas_start_j: {}, bas_len_j: {}", batch.start, batch.end, bas_start_i, bas_len_i, bas_start_j, bas_len_j);
                                    //    println!("global_s and e: ({},{}), ({},{}), start_point: {}, curr_start: {}, curr_end: {}", 
                                    //        global_start_row, global_start_col, 
                                    //        global_end_row, global_end_col, 
                                    //        start_point, curr_start, curr_end);
                                    //}
                                    tmp_mat.iter_submatrix_mut(bas_start_i..bas_start_i+bas_len_i, bas_start_j..bas_start_j+bas_len_j, matrixupper_index)
                                        .zip(tmp_loc.data[start_point..].iter()).for_each(|(gij,lij)| {*gij = *lij});
                                } else {
                                    ////debug 514
                                    //if k==0 && l==0 && bas_start_i == 9 {
                                    //    println!("batch_start: {}, batch_end: {}, bas_start_i: {}, bas_len_i: {}, bas_start_j: {}, bas_len_j: {}", batch.start, batch.end, bas_start_i, bas_len_i, bas_start_j, bas_len_j);
                                    //    println!("global_s and e: ({},{}), ({},{}), start_point: {}, curr_start: {}, curr_end: {}", 
                                    //        global_start_row, global_start_col, 
                                    //        global_end_row, global_end_col, 
                                    //        start_point, curr_start, curr_end);
                                    //    //println!("{:?}", &tmp_loc.data);
                                    //}
                                    //println!("i=j={}:({},{})",i, bas_start_i,bas_len_i);
                                    tmp_mat.iter_submatrix_mut(bas_start_i..bas_start_i+bas_len_i, bas_start_j..bas_start_j+bas_len_j, matrixupper_index)
                                        .zip(tmp_loc.iter_matrixupper_shift(start_point).unwrap()).for_each(|(gij,lij)| {*gij = *lij});
                                }
                            }
                        }
                    }
                };
            }
        };
        mat_full
        //mat_vec
    }
    #[inline]
    pub fn int_ijkl_erifull(&self) -> ERIFull<f64> {
        //let mut dt_cint = 0.0_f64;
        //let dt1 = time::Local::now();
        let mut cint_data = self.initialize_cint(false);
        let nbas = self.num_basis;
        let mut mat_full = 
            ERIFull::new([nbas,nbas,nbas,nbas],0.0);
        let nbas_shell = self.cint_bas.len();
        cint_data.cint2e_optimizer_rust();
        for l in 0..nbas_shell {
            let bas_start_l = self.cint_fdqc[l][0];
            let bas_len_l = self.cint_fdqc[l][1];
            for k in 0..(l+1) {
                let bas_start_k = self.cint_fdqc[k][0];
                let bas_len_k = self.cint_fdqc[k][1];
                for j in 0..nbas_shell {
                    let bas_start_j = self.cint_fdqc[j][0];
                    let bas_len_j = self.cint_fdqc[j][1];
                    //let (i_start, i_end) = (0,j+1);
                    for i in 0..j+1 {
                        let bas_start_i = self.cint_fdqc[i][0];
                        let bas_len_i = self.cint_fdqc[i][1];
                        let buf = cint_data.cint_ijkl_by_shell(i as i32, j as i32, k as i32, l as i32);
                        //let dt_cint_0 = time::Local::now();
                        //let dt_cint_1 = time::Local::now();
                        mat_full.chrunk_copy([bas_start_i..bas_start_i+bas_len_i,
                                              bas_start_j..bas_start_j+bas_len_j,
                                              bas_start_k..bas_start_k+bas_len_k,
                                              bas_start_l..bas_start_l+bas_len_l,
                                              ], buf.clone());
                        // copy the "upper" part to the lower part
                        if i<j {
                            mat_full.chrunk_copy_transpose_ij([
                                bas_start_i..bas_start_i+bas_len_i,
                                bas_start_j..bas_start_j+bas_len_j,
                                bas_start_k..bas_start_k+bas_len_k,
                                bas_start_l..bas_start_l+bas_len_l,
                                ], buf);
                        }
                    };
                }
            }
        }
        cint_data.final_c2r();
        // to copy the upper part of the (k,l) pair to the lower block
        for k in 0..nbas {
            for l in 0..k {
                let from_slice =  mat_full.get4d_slice([0,0,l,k], mat_full.indicing[2]).unwrap().to_vec();
                let mut to_slice = mat_full.get4d_slice_mut([0,0,k,l], mat_full.indicing[2]).unwrap();
                //unsafe{
                //    dcopy(to_slice.len() as i32, &from_slice, 1, to_slice, 1);
                //}
                to_slice.iter_mut().zip(from_slice.iter()).for_each(|(t,f)|*t = *f);
            }
        }
        mat_full
    }
    #[inline]
    pub fn int_ijkl_erifold4(&self) -> ERIFold4<f64> {
        let mut cint_data = self.initialize_cint(false);
        //let mut dt_cint = 0.0_f64;
        //let dt1 = time::Local::now();
        let nbas = self.num_basis;
        let npair = nbas*(nbas+1)/2;
        let mut mat_full = ERIFold4::new([npair,npair],0.0);
        let nbas_shell = self.cint_bas.len();
        cint_data.cint2e_optimizer_rust();
        for l in 0..nbas_shell {
            let bas_start_l = self.cint_fdqc[l][0];
            let bas_len_l = self.cint_fdqc[l][1];
            for k in 0..(l+1) {
                let bas_start_k = self.cint_fdqc[k][0];
                let bas_len_k = self.cint_fdqc[k][1];
                for j in 0..nbas_shell {
                    let bas_start_j = self.cint_fdqc[j][0];
                    let bas_len_j = self.cint_fdqc[j][1];
                    let (i_start, i_end) = (0,j+1);
                    for i in 0..j+1 {
                        let bas_start_i = self.cint_fdqc[i][0];
                        let bas_len_i = self.cint_fdqc[i][1];
                        let buf = cint_data.cint_ijkl_by_shell(i as i32, j as i32, k as i32, l as i32);
                        mat_full.chunk_copy_from_local_erifull(nbas, 
                            bas_start_i..bas_start_i+bas_len_i,
                            bas_start_j..bas_start_j+bas_len_j,
                            bas_start_k..bas_start_k+bas_len_k,
                            bas_start_l..bas_start_l+bas_len_l,
                            buf);
                    }
                }
            }
        }
        cint_data.final_c2r();
        mat_full
    }
    pub fn int_ijkl_from_r3fn(&self, r3fn: &RIFull<f64>) -> ERIFold4<f64> {
        let nbas = self.num_basis;
        let npair = nbas*(nbas+1)/2;

        let mut mat_full = MatrixFull::new([npair,npair],0.0);
        let mut tmp_mat = MatrixFull::new([npair,1],0.0);
        &r3fn.data.chunks(self.num_basis*self.num_basis).for_each(|value| {
           value.iter().enumerate().filter(|value| value.0%nbas<=value.0/nbas)
           .map(|value| value.1)
           .zip(tmp_mat.data.iter_mut())
           .for_each(|value| {*value.1 = *value.0});
           let mut tmp_mat_b = MatrixFull::from_vec([npair,1],tmp_mat.data.clone()).unwrap();
           mat_full.lapack_dgemm(&mut tmp_mat, &mut tmp_mat_b, 'N', 'T', 1.0, 1.0);
        });
        ERIFold4::from_vec([npair,npair],mat_full.data).unwrap()

    }

    pub fn int_ijk_rifull(&self) -> RIFull<f64> {
        let mut cint_data = self.initialize_cint(true);
        // It is O_V = (ij|\nu)
        let n_basis = self.num_basis;
        let n_auxbas = self.num_auxbas;
        let mut ri3fn = RIFull::new([n_basis,n_basis,n_auxbas],0.0);
        let n_basis_shell = self.cint_bas.len();
        let n_auxbas_shell = self.cint_aux_bas.len();
        cint_data.cint3c2e_optimizer_rust();
        for k in 0..n_auxbas_shell {
            let basis_start_k = self.cint_aux_fdqc[k][0];
            let basis_len_k = self.cint_aux_fdqc[k][1];
            let gk  = k + n_basis_shell;
            for j in 0..n_basis_shell {
                let basis_start_j = self.cint_fdqc[j][0];
                let basis_len_j = self.cint_fdqc[j][1];
                // can be optimized with "for i in 0..(j+1)"
                for i in 0..n_basis_shell {
                    let basis_start_i = self.cint_fdqc[i][0];
                    let basis_len_i = self.cint_fdqc[i][1];
                    let buf = RIFull::from_vec([basis_len_i, basis_len_j,basis_len_k], 
                        cint_data.cint_3c2e(i as i32, j as i32, gk as i32)).unwrap();
                    ri3fn.copy_from_ri(
                        basis_start_i..basis_start_i+basis_len_i,
                        basis_start_j..basis_start_j+basis_len_j,
                        basis_start_k..basis_start_k+basis_len_k,
                        & buf, 
                        0..basis_len_i, 
                        0..basis_len_j, 
                        0..basis_len_k);
                    //let mut tmp_slices = ri3fn.get_slices_mut(
                    //    basis_start_i..basis_start_i+basis_len_i,
                    //    basis_start_j..basis_start_j+basis_len_j,
                    //    basis_start_k..basis_start_k+basis_len_k);
                    //tmp_slices.zip(buf.iter()).for_each(|value| {*value.0 = *value.1});
                }
            }
        }
        cint_data.final_c2r();
        ri3fn
    }


    pub fn int_ijk_rifull_rayon(&self) -> RIFull<f64> {
        // It is O_V = (ij|\nu)
        let n_basis = self.num_basis;
        let n_auxbas = self.num_auxbas;
        let mut ri3fn = RIFull::new([n_basis,n_basis,n_auxbas],0.0);
        let n_basis_shell = self.cint_bas.len();
        let n_auxbas_shell = self.cint_aux_bas.len();
        let cint_type = if self.ctrl.basis_type.to_lowercase()==String::from("spheric") {
            CintType::Spheric
        } else if self.ctrl.basis_type.to_lowercase()==String::from("cartesian") {
            CintType::Cartesian
        } else {
            panic!("Error:: Unknown basis type '{}'. Please use either 'spheric' or 'Cartesian'", 
                   self.ctrl.basis_type);
        };

        let (sender, receiver) = channel();
        self.cint_aux_fdqc.par_iter().enumerate().for_each_with(sender,|s,(k,bas_info)| {

            // first, initialize rust_cint for each rayon threads
            let mut cint_data = self.initialize_cint(true);

            let basis_start_k = bas_info[0];
            let basis_len_k = bas_info[1];
            let mut ri_rayon = RIFull::new([n_basis,n_basis,basis_len_k],1.0);
            let gk  = k + n_basis_shell;

            for j in 0..n_basis_shell {
                //let basis_start_j = cint_fdqc[j][0];
                //let basis_len_j = cint_fdqc[j][1];
                let basis_start_j = self.cint_fdqc[j][0];
                let basis_len_j = self.cint_fdqc[j][1];
                // can be optimized with "for i in 0..(j+1)"
                for i in 0..n_basis_shell {
                    //let basis_start_i = cint_fdqc[i][0];
                    //let basis_len_i = cint_fdqc[i][1];
                    let basis_start_i = self.cint_fdqc[i][0];
                    let basis_len_i = self.cint_fdqc[i][1];
                    let buf = RIFull::from_vec([basis_len_i, basis_len_j,basis_len_k], 
                        cint_data.cint_3c2e(i as i32, j as i32, gk as i32)).unwrap();
                    ri_rayon.copy_from_ri(
                        basis_start_i..basis_start_i+basis_len_i,
                        basis_start_j..basis_start_j+basis_len_j,
                        0..basis_len_k,
                        //basis_start_k..basis_start_k+basis_len_k,
                        & buf, 
                        0..basis_len_i, 
                        0..basis_len_j, 
                        0..basis_len_k);
                }
            }
            cint_data.final_c2r();


            s.send((ri_rayon,basis_start_k,basis_len_k)).unwrap()

        });

        receiver.into_iter().for_each(|(ri_rayon, basis_start_k,basis_len_k)| {
            ri3fn.copy_from_ri(
                0..n_basis,0..n_basis,basis_start_k..basis_start_k+basis_len_k,
                &ri_rayon,
                0..n_basis,0..n_basis,0..basis_len_k,
            );
        });

        ri3fn
    }

    pub fn int_ij_aux_columb(&self) -> MatrixFull<f64> {
        self.int_ij_aux_columb_new()
    }

    pub fn int_ij_aux_columb_serial(&self) -> MatrixFull<f64> {
        let mut cint_data = self.initialize_cint(true);
        let n_auxbas = self.num_auxbas;
        let n_basis_shell = self.cint_bas.len();
        let n_auxbas_shell = self.cint_aux_bas.len();
        cint_data.cint2c2e_optimizer_rust();
        let mut aux_v = MatrixFull::new([n_auxbas,n_auxbas],0.0);
        for l in 0..n_auxbas_shell {
            let basis_start_l = self.cint_aux_fdqc[l][0];
            let basis_len_l = self.cint_aux_fdqc[l][1];
            let gl  = l + n_basis_shell;
            for k in 0..n_auxbas_shell {
                let basis_start_k = self.cint_aux_fdqc[k][0];
                let basis_len_k = self.cint_aux_fdqc[k][1];
                let gk  = k + n_basis_shell;
                let buf = cint_data.cint_2c2e(gk as i32, gl as i32);
                //println!("debug 1 start with k: ({},{},{}), l: ({},{},{})", k,basis_start_k,basis_len_k, l,basis_start_l,basis_len_l);
                let mut tmp_slices = aux_v.iter_submatrix_mut(
                    basis_start_k..basis_start_k+basis_len_k,
                    basis_start_l..basis_start_l+basis_len_l);
                tmp_slices.zip(buf.iter()).for_each(|value| {*value.0 = *value.1});
                //println!("debug 1 end with k: {}, l: {}", l, k);

            }
        }
        cint_data.final_c2r();
        //aux_v.formated_output_e(4, "full");
        aux_v
    }
    
    pub fn int_ij_aux_columb_rayon(&self) -> MatrixFull<f64> {
        let n_auxbas = self.num_auxbas;
        let n_basis_shell = self.cint_bas.len();
        let n_auxbas_shell = self.cint_aux_bas.len();
        let mut aux_v = MatrixFull::new([n_auxbas,n_auxbas],0.0);
        let (sender, receiver) = channel();
        self.cint_aux_fdqc.par_iter().enumerate().for_each_with(sender,|s,(l,fdqc)| {
            let mut cint_data = self.initialize_cint(true);
            cint_data.cint2c2e_optimizer_rust();
            let basis_start_l = fdqc[0];
            let basis_len_l = fdqc[1];
            let gl  = l + n_basis_shell;
            let mut loc_aux_v = MatrixFull::new([n_auxbas,basis_len_l],0.0);
            for k in 0..n_auxbas_shell {
                let basis_start_k = self.cint_aux_fdqc[k][0];
                let basis_len_k = self.cint_aux_fdqc[k][1];
                let gk  = k + n_basis_shell;
                let buf = cint_data.cint_2c2e(gk as i32, gl as i32);
                let mut tmp_slices = loc_aux_v.iter_submatrix_mut(
                    basis_start_k..basis_start_k+basis_len_k,
                    0..basis_len_l);
                tmp_slices.zip(buf.iter()).for_each(|value| {*value.0 = *value.1});
            }
            cint_data.final_c2r();
            s.send((loc_aux_v, basis_start_l, basis_len_l)).unwrap()
        });
        receiver.into_iter().for_each(|(loc_aux_v, basis_start_l, basis_len_l)| {
            aux_v.copy_from_matr(0..n_auxbas, basis_start_l..basis_start_l+basis_len_l, &loc_aux_v, 0..n_auxbas, 0..basis_len_l);
        });
        //aux_v.formated_output_e(4, "full");
        aux_v
    }

    pub fn int_ij_aux_columb_new(&self) -> MatrixFull<f64> {
        use rest_libcint_wrapper::*;
        omp_set_num_threads_wrapper(self.ctrl.num_threads.unwrap());
        let n_auxbas = self.num_auxbas;
        let mut cint_data = self.initialize_cint(true);
        let n_basis_shell = self.cint_bas.len() as i32;
        let n_auxbas_shell = self.cint_aux_bas.len() as i32;
        let shl_slices = vec![[n_basis_shell, n_basis_shell + n_auxbas_shell], [n_basis_shell, n_basis_shell + n_auxbas_shell]];
        let mut out = vec![];
        //let mut out_shape = vec![];
        (out, _) = cint_data.integral_s1::<int2c2e>(Some(&shl_slices));

        MatrixFull::from_vec([n_auxbas, n_auxbas], out).unwrap()

    }

    pub fn prepare_ri3fn_for_ri_v(&self) -> RIFull<f64> {
        // prepare O_V * V^{-1/2}

        let mut time_records = utilities::TimeRecords::new();
        time_records.new_item("all ri", "for the total evaluations of ri3fn");
        time_records.count_start("all ri");

        time_records.new_item("prim ri", "for the pure three-center integrals");
        time_records.new_item("aux_ij", "for the coulomb matrix of auxiliary basis");
        time_records.new_item("sqrt_matr", "for inverse sqrt of the coulomb matrix of auxiliary basis");
        time_records.new_item("trans", "for O_V*V^{-1/2}");


        let n_basis = self.num_basis;
        let n_auxbas = self.num_auxbas;


        // at frist, prepare the 3-center integrals: O_V = (ij|\nu)
        time_records.count_start("prim ri");
        let mut ri3fn = self.int_ijk_rifull_rayon();
        time_records.count("prim ri");
        // then the auxiliary 2-center coulumb matrix: V=(\nu|\mu)
        time_records.count_start("aux_ij");
        let mut aux_v = self.int_ij_aux_columb();
        time_records.count("aux_ij");

        // we calculate the square roote of the inversion matrix: V^{-1/2}
        time_records.count_start("sqrt_matr");
        aux_v = aux_v.lapack_power(-0.5, AUXBAS_THRESHOLD).unwrap();
        time_records.count("sqrt_matr");

        // println!("Print out {} 2-center auxiliary coulomb integral elements", tmp_num);

        // perform O_V*V^{-1/2}
        time_records.count_start("trans");
        let mut tmp_ovlp_matr = MatrixFull::new([n_basis,n_auxbas],0.0);
        let mut aux_ovlp_matr = MatrixFull::new([n_basis,n_auxbas],0.0);
        for j in 0..n_basis {
            tmp_ovlp_matr.data.iter_mut().zip(ri3fn.get_slices(0..n_basis,j..j+1,0..n_auxbas))
                .for_each(|value| {
                    //println!("debug: {:?}", value);
                    *value.0 = *value.1;
                });
            aux_ovlp_matr.lapack_dgemm(&mut tmp_ovlp_matr, &mut aux_v, 'N', 'N', 1.0,0.0);
            ri3fn.get_slices_mut(0..n_basis,j..j+1,0..n_auxbas).zip(aux_ovlp_matr.data.iter())
                .for_each(|value| {*value.0 = *value.1});
            //ri3fn.copy_from_matr(0..n_basis, 0..n_auxbas, j, 1, 
            //    &aux_ovlp_matr, 0..n_basis, 0..n_auxbas)
        }
        time_records.count("trans");

        time_records.count("all ri");

        if self.ctrl.print_level>=2 {
            time_records.report_all();
        }

        ri3fn
    }

    pub fn prepare_ri3fn_for_ri_v_rayon(&mut self) -> RIFull<f64> {
        // prepare O_V * V^{-1/2}

        let mut time_records = utilities::TimeRecords::new();
        time_records.new_item("all ri", "for the total evaluations of ri3fn");
        time_records.count_start("all ri");

        time_records.new_item("prim ri", "for the pure three-center integrals");
        time_records.new_item("aux_ij", "for the coulomb matrix of auxiliary basis");
        time_records.new_item("sqrt_matr", "for inverse sqrt of the coulomb matrix of auxiliary basis");
        time_records.new_item("trans", "for O_V*V^{-1/2}");


        let n_basis = self.num_basis;
        let n_auxbas = self.num_auxbas;


        // at frist, prepare the 3-center integrals: O_V = (ij|\nu)
        time_records.count_start("prim ri");
        let mut ri3fn = self.int_ijk_rifull_rayon();
        time_records.count("prim ri");
        // then the auxiliary 2-center coulumb matrix: V=(\nu|\mu)
        time_records.count_start("aux_ij");
        let mut aux_v = self.int_ij_aux_columb();  //issue occurred here
        time_records.count("aux_ij");

        time_records.count_start("sqrt_matr");
        //// we calculate the square roote of the inversion matrix: V^{-1/2}
        aux_v = aux_v.lapack_power(-0.5, AUXBAS_THRESHOLD).unwrap();
        // we calculate the Cholesky decomposition L of the inversion matrix: L*L^{T}=V^{-1}
        //aux_v = aux_v.to_matrixfullslicemut().cholesky_decompose_inverse('L').unwrap();
        time_records.count("sqrt_matr");

        // println!("Print out {} 2-center auxiliary coulomb integral elements", tmp_num);

        // perform O_V*V^{-1/2}
        time_records.count_start("trans");

        // 1) Version of Fortran wrapper
        //let ri3fn_size = ri3fn.size.clone();
        //rest_tensors::external_libs::special_dgemm_f_01(
        //    &mut ri3fn.data[..], &ri3fn_size, 0..n_basis, 0, 0..n_auxbas,
        //    aux_v.data_ref().unwrap(), aux_v.size(), 0..n_auxbas, 0..n_auxbas,
        //    1.0, 0.0
        //);

        let mut tmp_ovlp_matr = MatrixFull::new([n_basis,n_auxbas],0.0);
        let mut aux_ovlp_matr = MatrixFull::new([n_basis,n_auxbas],0.0);
        let size = [n_basis,n_auxbas];
        for j in 0..n_basis {
            // 2) Rayon Version
            //utilities::omp_set_num_threads(1);
            //let task_distribution = utilities::balancing(n_auxbas,rayon::current_num_threads()-1);
            ////let mut aux_ovlp_matr = MatrixFull::new([n_basis,n_auxbas],0.0);
            //let (sender, receiver) = channel();
            //task_distribution.par_iter().for_each_with(sender,|s,range_auxbas| {
            //    let mut loc_ovlp_matr = MatrixFull::new([n_basis,range_auxbas.len()],0.0);
            //    loc_ovlp_matr.data.iter_mut().zip(ri3fn.get_slices(0..n_basis,j..j+1,range_auxbas.clone()))
            //        .for_each(|value| {
            //            *value.0 = *value.1;
            //        });
            //    let mut aux_ovlp_matr = MatrixFull::new([n_basis,n_auxbas],0.0);
            //    _dgemm(&loc_ovlp_matr, (0..n_basis,0..range_auxbas.len()), 'N', 
            //        &aux_v, (range_auxbas.clone(), 0..n_auxbas), 'N', 
            //        &mut aux_ovlp_matr, (0..n_basis, 0..n_auxbas), 1.0, 0.0);
            //    s.send((aux_ovlp_matr,range_auxbas)).unwrap()
            //});
            //receiver.into_iter().enumerate().for_each(|(index, (loc_aux_ovlp_matr,range_auxbas))| {
            //    //aux_ovlp_matr += loc_aux_ovlp_matr;
            //    if index != 0 {
            //    ri3fn.get_slices_mut(0..n_basis,j..j+1,0..n_auxbas).zip(loc_aux_ovlp_matr.data.iter())
            //        .for_each(|value| {*value.0 += *value.1});
            //    } else {
            //    ri3fn.get_slices_mut(0..n_basis,j..j+1,0..n_auxbas).zip(loc_aux_ovlp_matr.data.iter())
            //        .for_each(|value| {*value.0 = *value.1});
            //    }
            //});
            //utilities::omp_set_num_threads(self.ctrl.num_threads.unwrap());

            // 3) the old version

            //tmp_ovlp_matr.data.iter_mut().zip(ri3fn.get_slices(0..n_basis,j..j+1,0..n_auxbas))
            //    .for_each(|value| {
            //        *value.0 = *value.1;
            //    });

            matr_copy_from_ri(&ri3fn.data, &ri3fn.size,0..n_basis, 0..n_auxbas, j, 1,
                &mut tmp_ovlp_matr.data, &size, 0..n_basis, 0..n_auxbas);

            aux_ovlp_matr.to_matrixfullslicemut().lapack_dgemm(
                &tmp_ovlp_matr.to_matrixfullslice(), &aux_v.to_matrixfullslice(), 
                'N', 'N', 1.0,0.0);

            //ri3fn.get_slices_mut(0..n_basis,j..j+1,0..n_auxbas).zip(aux_ovlp_matr.data.iter())
            //    .for_each(|value| {*value.0 = *value.1});
            ri3fn.copy_from_matr(0..n_basis, 0..n_auxbas, j, 1, 
                &aux_ovlp_matr, 0..n_basis, 0..n_auxbas)
        }
        
        time_records.count("trans");

        time_records.count("all ri");

        if self.ctrl.print_level>=2 {
            time_records.report_all();
        }

        ri3fn
    }

    pub fn prepare_ri3fn_for_ri_v_full_rayon(&self) -> RIFull<f64> {
        // prepare O_V * V^{-1/2}

        let mut time_records = utilities::TimeRecords::new();
        time_records.new_item("all ri", "for the total evaluations of ri3fn");
        time_records.count_start("all ri");

        time_records.new_item("prim ri", "for the pure three-center integrals");
        time_records.new_item("aux_ij", "for the coulomb matrix of auxiliary basis");
        //time_records.new_item("sqrt_matr", "for inverse sqrt of the coulomb matrix of auxiliary basis");
        //time_records.new_item("trans", "for O_V*V^{-1/2}");


        let n_basis = self.num_basis;
        let n_auxbas = self.num_auxbas;


        // First the Cholesky decomposition `L` of the inverse of the auxiliary 2-center coulumb matrix: V=(\nu|\mu)
        time_records.count_start("aux_ij");
        let mut aux_v = self.int_ij_aux_columb();
        aux_v = aux_v.lapack_power(-0.5, AUXBAS_THRESHOLD).unwrap();
        //aux_v = aux_v.to_matrixfullslicemut().cholesky_decompose_inverse('L').unwrap();
        time_records.count("aux_ij");

        // Then, prepare the 3-center integrals: O_V = (ij|\nu), and multiple with `L`
        time_records.count_start("prim ri");
        let mut ri3fn = RIFull::new([n_basis,n_basis,n_auxbas],0.0);
        let n_basis_shell = self.cint_bas.len();
        let n_auxbas_shell = self.cint_aux_bas.len();
        let cint_type = if self.ctrl.basis_type.to_lowercase()==String::from("spheric") {
            CintType::Spheric
        } else if self.ctrl.basis_type.to_lowercase()==String::from("cartesian") {
            CintType::Cartesian
        } else {
            panic!("Error:: Unknown basis type '{}'. Please use either 'spheric' or 'Cartesian'", 
                   self.ctrl.basis_type);
        };

        let (sender, receiver) = channel();
        //self.cint_aux_fdqc.par_iter().enumerate().for_each_with(sender,|s,(k,bas_info)| {
        self.cint_fdqc.par_iter().enumerate().for_each_with(sender,|s, (j,bas_info_j)| {
            omp_set_num_threads_wrapper(1);
            let basis_start_j = bas_info_j[0];
            let basis_len_j = bas_info_j[1];

            // first, initialize rust_cint for each rayon threads
            let mut cint_data = self.initialize_cint(true);
            let mut ri_rayon = RIFull::new([n_basis,basis_len_j,n_auxbas],0.0);

            self.cint_aux_fdqc.iter().enumerate().for_each(|(k,bas_info_k)| {
                let basis_start_k = bas_info_k[0];
                let basis_len_k = bas_info_k[1];
                let gk  = k + n_basis_shell;
                self.cint_fdqc.iter().enumerate().for_each(|(i,bas_info_i)| {
                    let basis_start_i = bas_info_i[0];
                    let basis_len_i = bas_info_i[1];

                    let buf = RIFull::from_vec([basis_len_i, basis_len_j,basis_len_k], 
                        cint_data.cint_3c2e(i as i32, j as i32, gk as i32)).unwrap();

                    ri_rayon.copy_from_ri(
                        basis_start_i..basis_start_i+basis_len_i,
                        0..basis_len_j,
                        basis_start_k..basis_start_k+basis_len_k,
                        //basis_start_k..basis_start_k+basis_len_k,
                        & buf, 
                        0..basis_len_i, 
                        0..basis_len_j, 
                        0..basis_len_k);
                });
            });


            cint_data.final_c2r();

            let mut tmp_ovlp_matr = MatrixFull::new([n_basis,n_auxbas],0.0);
            let mut aux_ovlp_matr = MatrixFull::new([n_basis,n_auxbas],0.0);
            let size = [n_basis,n_auxbas];
            for j in 0..basis_len_j {

                matr_copy_from_ri(&ri_rayon.data, &ri_rayon.size,0..n_basis, 0..n_auxbas, j, 1,
                    &mut tmp_ovlp_matr.data, &size, 0..n_basis, 0..n_auxbas);

                //aux_ovlp_matr.to_matrixfullslicemut().lapack_dgemm(
                //    &tmp_ovlp_matr.to_matrixfullslice(), &aux_v.to_matrixfullslice(), 
                //    'N', 'N', 1.0,0.0);
                _dgemm(
                    &tmp_ovlp_matr, (0..n_basis,0..n_auxbas), 'N', 
                    &aux_v, (0..n_auxbas, 0..n_auxbas), 'N', 
                    &mut aux_ovlp_matr, (0..n_basis,0..n_auxbas), 1.0, 0.0);

                ri_rayon.copy_from_matr(0..n_basis, 0..n_auxbas, j, 1, 
                    &aux_ovlp_matr, 0..n_basis, 0..n_auxbas)
            }


            s.send((ri_rayon,basis_start_j,basis_len_j)).unwrap()

        });

        receiver.into_iter().for_each(|(ri_rayon, basis_start_j,basis_len_j)| {
            ri3fn.copy_from_ri(
                0..n_basis,basis_start_j..basis_start_j+basis_len_j,0..n_auxbas,
                &ri_rayon,
                0..n_basis,0..basis_len_j,0..n_auxbas,
            );
        });

        omp_set_num_threads_wrapper(self.ctrl.num_threads.unwrap());

        time_records.count("prim ri");
        time_records.count("all ri");

        if self.ctrl.print_level>=2 {
            time_records.report_all();
        }


        ri3fn
    }

    // generate the 2-center auxiliary Coulomb matrix with range-separated kernel
    pub fn int_ij_aux_columb_with_omega(&self, omega: f64) -> MatrixFull<f64> {
        let n_auxbas = self.num_auxbas;
        let n_basis_shell = self.cint_bas.len();
        let n_auxbas_shell = self.cint_aux_bas.len();
        let mut aux_v = MatrixFull::new([n_auxbas,n_auxbas],0.0);
        let (sender, receiver) = channel();
        // For SR integrals: set omega < 0 (PTR_RANGE_OMEGA=8, negative → erfc(ωr)/r)
        let range_omega = -omega;
        self.cint_aux_fdqc.par_iter().enumerate().for_each_with(sender,|s,(l,fdqc)| {
            let mut cint_data = self.initialize_cint(true);
            cint_data.set_omega(range_omega);
            cint_data.cint2c2e_optimizer_rust();
            let basis_start_l = fdqc[0];
            let basis_len_l = fdqc[1];
            let gl  = l + n_basis_shell;
            let mut loc_aux_v = MatrixFull::new([n_auxbas,basis_len_l],0.0);
            for k in 0..n_auxbas_shell {
                let basis_start_k = self.cint_aux_fdqc[k][0];
                let basis_len_k = self.cint_aux_fdqc[k][1];
                let gk  = k + n_basis_shell;
                let buf = cint_data.cint_2c2e(gk as i32, gl as i32);
                let mut tmp_slices = loc_aux_v.iter_submatrix_mut(
                    basis_start_k..basis_start_k+basis_len_k,
                    0..basis_len_l);
                tmp_slices.zip(buf.iter()).for_each(|value| {*value.0 = *value.1});
            }
            cint_data.final_c2r();
            s.send((loc_aux_v, basis_start_l, basis_len_l)).unwrap()
        });
        receiver.into_iter().for_each(|(loc_aux_v, basis_start_l, basis_len_l)| {
            aux_v.copy_from_matr(0..n_auxbas, basis_start_l..basis_start_l+basis_len_l, &loc_aux_v, 0..n_auxbas, 0..basis_len_l);
        });
        aux_v
    }

    // generate the short-range 3-center RI integrals: (mu nu | erfc(omega*r12)/r12 | P)
    pub fn prepare_ri3fn_sr_rayon(&self, omega: f64) -> RIFull<f64> {
        let n_basis = self.num_basis;
        let n_auxbas = self.num_auxbas;

        // For SR integrals: set omega < 0 (PTR_RANGE_OMEGA=8, negative → erfc(ωr)/r)
        let range_omega = -omega;

        // First, the Cholesky decomposition of the SR 2-center Coulomb matrix
        let mut aux_v = self.int_ij_aux_columb_with_omega(omega);
        aux_v = aux_v.lapack_power(-0.5, AUXBAS_THRESHOLD).unwrap();

        // Then, prepare the 3-center SR integrals
        let mut ri3fn = RIFull::new([n_basis,n_basis,n_auxbas],0.0);
        let n_basis_shell = self.cint_bas.len();
        let n_auxbas_shell = self.cint_aux_bas.len();
        let cint_type = if self.ctrl.basis_type.to_lowercase()==String::from("spheric") {
            CintType::Spheric
        } else if self.ctrl.basis_type.to_lowercase()==String::from("cartesian") {
            CintType::Cartesian
        } else {
            panic!("Error:: Unknown basis type '{}'", self.ctrl.basis_type);
        };

        let (sender, receiver) = channel();
        self.cint_fdqc.par_iter().enumerate().for_each_with(sender,|s, (j,bas_info_j)| {
            omp_set_num_threads_wrapper(1);
            let basis_start_j = bas_info_j[0];
            let basis_len_j = bas_info_j[1];

            let mut cint_data = self.initialize_cint(true);
            cint_data.set_omega(range_omega);
            cint_data.cint3c2e_optimizer_rust();
            let mut ri_rayon = RIFull::new([n_basis,basis_len_j,n_auxbas],0.0);

            self.cint_aux_fdqc.iter().enumerate().for_each(|(k,bas_info_k)| {
                let basis_start_k = bas_info_k[0];
                let basis_len_k = bas_info_k[1];
                let gk  = k + n_basis_shell;
                self.cint_fdqc.iter().enumerate().for_each(|(i,bas_info_i)| {
                    let basis_start_i = bas_info_i[0];
                    let basis_len_i = bas_info_i[1];
                    let buf = RIFull::from_vec([basis_len_i, basis_len_j,basis_len_k], 
                        cint_data.cint_3c2e(i as i32, j as i32, gk as i32)).unwrap();
                    ri_rayon.copy_from_ri(
                        basis_start_i..basis_start_i+basis_len_i,
                        0..basis_len_j,
                        basis_start_k..basis_start_k+basis_len_k,
                        & buf, 
                        0..basis_len_i, 
                        0..basis_len_j, 
                        0..basis_len_k);
                });
            });

            cint_data.final_c2r();

            let mut tmp_ovlp_matr = MatrixFull::new([n_basis,n_auxbas],0.0);
            let mut aux_ovlp_matr = MatrixFull::new([n_basis,n_auxbas],0.0);
            let size = [n_basis,n_auxbas];
            for j in 0..basis_len_j {
                matr_copy_from_ri(&ri_rayon.data, &ri_rayon.size,0..n_basis, 0..n_auxbas, j, 1,
                    &mut tmp_ovlp_matr.data, &size, 0..n_basis, 0..n_auxbas);
                _dgemm(
                    &tmp_ovlp_matr, (0..n_basis,0..n_auxbas), 'N', 
                    &aux_v, (0..n_auxbas, 0..n_auxbas), 'N', 
                    &mut aux_ovlp_matr, (0..n_basis,0..n_auxbas), 1.0, 0.0);
                ri_rayon.copy_from_matr(0..n_basis, 0..n_auxbas, j, 1, 
                    &aux_ovlp_matr, 0..n_basis, 0..n_auxbas)
            }

            s.send((ri_rayon,basis_start_j,basis_len_j)).unwrap()
        });

        receiver.into_iter().for_each(|(ri_rayon, basis_start_j,basis_len_j)| {
            ri3fn.copy_from_ri(
                0..n_basis,basis_start_j..basis_start_j+basis_len_j,0..n_auxbas,
                &ri_rayon,
                0..n_basis,0..basis_len_j,0..n_auxbas,
            );
        });

        omp_set_num_threads_wrapper(self.ctrl.num_threads.unwrap());
        ri3fn
    }

    // generate the 3-center RI integrals and the basis pair symmetry is used to save the memory
    pub fn prepare_rimatr_for_ri_v_rayon_v01(&self) -> (MatrixFull<f64>,MatrixFull<usize>,Vec<[usize;2]>) {

        let mut time_records = utilities::TimeRecords::new();
        time_records.new_item("all ri", "for the total evaluations of ri3fn");
        time_records.count_start("all ri");

        time_records.new_item("prim ri", "for the pure three-center integrals");
        time_records.new_item("aux_ij", "for the coulomb matrix of auxiliary basis");

        let n_basis = self.num_basis;
        let n_auxbas = self.num_auxbas;
        let n_baspar = (self.num_basis+1)*self.num_basis/2;

        // First the Cholesky decomposition `L` of the inverse of the auxiliary 2-center coulumb matrix: V=(\nu|\mu)
        time_records.count_start("aux_ij");
        let mut aux_v = self.int_ij_aux_columb();
        aux_v = aux_v.lapack_power(-0.5, AUXBAS_THRESHOLD).unwrap();
        //aux_v = aux_v.to_matrixfullslicemut().cholesky_decompose_inverse('L').unwrap();
        time_records.count("aux_ij");

        // Then, prepare the 3-center integrals: O_V = (ij|\nu), and multiple with `L`
        time_records.count_start("prim ri");
        let mut ri3fn = MatrixFull::new([n_baspar,n_auxbas],0.0);
        let n_basis_shell = self.cint_bas.len();
        let n_auxbas_shell = self.cint_aux_bas.len();
        let cint_type = if self.ctrl.basis_type.to_lowercase()==String::from("spheric") {
            CintType::Spheric
        } else if self.ctrl.basis_type.to_lowercase()==String::from("cartesian") {
            CintType::Cartesian
        } else {
            panic!("Error:: Unknown basis type '{}'. Please use either 'spheric' or 'Cartesian'", 
                   self.ctrl.basis_type);
        };

        // gather all shellpair together for parallel implemention
        let mut par_shellpair: Vec<[usize;2]> = vec![];
        for j in 0..n_basis_shell {
            for i in 0..j+1 {
                par_shellpair.push([i,j]);
            }
        };

        let (sender, receiver) = channel();
        par_shellpair.par_iter().for_each_with(sender, |s, shell_index| {

            omp_set_num_threads_wrapper(1);
            // first, initialize rust_cint for each rayon threads
            let mut cint_data = self.initialize_cint(true);

            let i = shell_index[0];
            let j = shell_index[1];
            let basis_start_j = self.cint_fdqc[j][0];
            let basis_len_j = self.cint_fdqc[j][1];
            let basis_start_i = self.cint_fdqc[i][0];
            let basis_len_i = self.cint_fdqc[i][1];
            //let mut ri_rayon = if i != j {
            //    MatrixFull::new([basis_len_i*basis_len_j,n_auxbas],0.0)
            //} else {
            //    MatrixFull::new([(basis_len_i+1)*basis_len_i/2,n_auxbas],0.0)
            //};

            let mut ri_rayon =  RIFull::new([basis_len_i,basis_len_j,n_auxbas],0.0);

            self.cint_aux_fdqc.iter().enumerate().for_each(|(k,bas_info)| {
                let basis_start_k = bas_info[0];
                let basis_len_k = bas_info[1];
                let gk  = k + n_basis_shell;
                let buf = RIFull::from_vec([basis_len_i, basis_len_j,basis_len_k], 
                    cint_data.cint_3c2e(i as i32, j as i32, gk as i32)).unwrap();
                ri_rayon.copy_from_ri(
                    0..basis_len_i,
                    0..basis_len_j,
                    basis_start_k..basis_start_k+basis_len_k,
                    //basis_start_k..basis_start_k+basis_len_k,
                    & buf, 
                    0..basis_len_i, 
                    0..basis_len_j, 
                    0..basis_len_k);
            });

            cint_data.final_c2r();

            let mut tmp_ovlp_matr = MatrixFull::new([basis_len_i,n_auxbas],0.0);
            let mut aux_ovlp_matr = MatrixFull::new([basis_len_i,n_auxbas],0.0);
            let size = [basis_len_i,n_auxbas];
            for j in 0..basis_len_j {

                matr_copy_from_ri(&ri_rayon.data, &ri_rayon.size,0..basis_len_i, 0..n_auxbas, j, 1,
                    &mut tmp_ovlp_matr.data, &size, 0..basis_len_i, 0..n_auxbas);

                _dgemm(
                    &tmp_ovlp_matr, (0..basis_len_i,0..n_auxbas), 'N', 
                    &aux_v, (0..n_auxbas, 0..n_auxbas), 'N', 
                    &mut aux_ovlp_matr, (0..basis_len_i,0..n_auxbas), 1.0, 0.0);

                ri_rayon.copy_from_matr(0..basis_len_i, 0..n_auxbas, j, 1, 
                    &aux_ovlp_matr, 0..basis_len_i, 0..n_auxbas);
            }

            s.send((ri_rayon,basis_start_i,basis_len_i,basis_start_j,basis_len_j)).unwrap();
        });

        //time_records.new_item("prim ri receive", "");
        //time_records.count_start("prim ri receive");

        receiver.into_iter().for_each(|(ri_rayon, basis_start_i,basis_len_i, basis_start_j,basis_len_j)| {
            if basis_start_i != basis_start_j {
                ri_rayon.iter_auxbas(0..n_auxbas).unwrap().enumerate().for_each(|(k,i_matr)| {
                    let mut loc_ind = 0;
                    for jj in basis_start_j..basis_start_j+basis_len_j {
                        let ri_ind = (jj+1)*jj/2;
                        for ii in basis_start_i..basis_start_i+basis_len_i {
                            ri3fn[(ri_ind+ii,k)] = i_matr[loc_ind];
                            loc_ind += 1;
                        }
                    }
                });
            } else {
                ri_rayon.iter_auxbas(0..n_auxbas).unwrap().enumerate().for_each(|(k,i_matr)| {
                    let mut loc_ind = 0;
                    for jj in basis_start_j..basis_start_j+basis_len_j {
                        let ri_ind = (jj+1)*jj/2;
                        for ii in basis_start_j..basis_start_j+basis_len_j {
                            if ii <= jj {
                                //println!("{:2},{:2},{:2},{:?},{:?}",ii,jj,ri_ind+ii,loc_ind,basis_len_j);
                                ri3fn[(ri_ind+ii,k)] = i_matr[loc_ind];
                                loc_ind += 1;
                            } else {
                                loc_ind += 1;
                            }
                        }
                    }
                });
            }
        });

        //time_records.count("prim ri receive");

        let mut basbas2baspar = MatrixFull::new([n_basis,n_basis],0_usize);
        let mut baspar2basbas = vec![[0_usize;2];n_baspar];
        basbas2baspar.iter_columns_full_mut().enumerate().for_each(|(j,slice_x)| {
            &slice_x.iter_mut().enumerate().for_each(|(i,map_v)| {
                let baspar_ind = if i< j {(j+1)*j/2+i} else {(i+1)*i/2+j};
                *map_v = baspar_ind;
                baspar2basbas[baspar_ind] = [i,j];
            });
        });

        omp_set_num_threads_wrapper(self.ctrl.num_threads.unwrap());

        time_records.count("prim ri");

        time_records.count("all ri");


        if self.ctrl.print_level>=2 {
            time_records.report_all();
        }

        (ri3fn,basbas2baspar,baspar2basbas)

    }

    // generate the 3-center RI integrals and the basis pair symmetry is used to save the memory
    pub fn prepare_rimatr_for_ri_v_rayon_v02(&self) -> (MatrixFull<f64>,MatrixFull<usize>,Vec<[usize;2]>) {

        let mut time_records = utilities::TimeRecords::new();
        time_records.new_item("all ri", "for the total evaluations of ri3fn");
        time_records.count_start("all ri");

        time_records.new_item("prim ri", "for the pure three-center integrals");
        time_records.new_item("aux_ij", "for the coulomb matrix of auxiliary basis");
        time_records.new_item("inv_sqrt", "for the sqrt inverse of aux_ij");

        //time_records.new_item("ri: cint_3c2e", "debug");
        //time_records.new_item("ri: copy_from_ri","debug");
        //time_records.new_item("ri: ri_dot_sqrtv","debug");
        //time_records.new_item("ri: dgemm","debug");

        let n_basis = self.num_basis;
        let n_auxbas = self.num_auxbas;
        let n_baspar = (self.num_basis+1)*self.num_basis/2;

        // First the Cholesky decomposition `L` of the inverse of the auxiliary 2-center coulumb matrix: V=(\nu|\mu)
        time_records.count_start("aux_ij");
        let mut aux_v = self.int_ij_aux_columb();
        time_records.count("aux_ij");
        time_records.count_start("inv_sqrt");
        aux_v = _power_rayon_for_symmetric_matrix(&aux_v, -0.5, AUXBAS_THRESHOLD).unwrap();
        //aux_v = aux_v.lapack_power(-0.5, AUXBAS_THRESHOLD).unwrap();
        //aux_v = aux_v.to_matrixfullslicemut().cholesky_decompose_inverse('L').unwrap();
        time_records.count("inv_sqrt");

        // Then, prepare the 3-center integrals: O_V = (ij|\nu), and multiple with `L`
        time_records.count_start("prim ri");
        let mut ri3fn = MatrixFull::new([n_baspar,n_auxbas],0.0);
        let n_basis_shell = self.cint_bas.len();
        let n_auxbas_shell = self.cint_aux_bas.len();
        let cint_type = if self.ctrl.basis_type.to_lowercase()==String::from("spheric") {
            CintType::Spheric
        } else if self.ctrl.basis_type.to_lowercase()==String::from("cartesian") {
            CintType::Cartesian
        } else {
            panic!("Error:: Unknown basis type '{}'. Please use either 'spheric' or 'Cartesian'", 
                   self.ctrl.basis_type);
        };

        let (sender, receiver) = channel();
        //par_shellpair.par_iter().for_each_with(sender, |s, shell_index| {
        (0..n_basis_shell).rev().collect::<Vec<usize>>().par_iter().for_each_with(sender, |s, gj| {
            omp_set_num_threads_wrapper(1);
            //// ========== for efficiency debug ==============
            //let mut sub_time_records = utilities::TimeRecords::new();
            //sub_time_records.new_item("ri: cint_3c2e", "debug");
            //sub_time_records.new_item("ri: copy_from_ri","debug");
            //sub_time_records.new_item("ri: ri_dot_sqrtv","debug");
            //sub_time_records.new_item("ri: dgemm","debug");
            ////================================================
            let bas_j = *gj;
            // first, initialize rust_cint for each rayon threads
            let mut cint_data = self.initialize_cint(true);

            //let i = shell_index[0];
            //let j = shell_index[1];
            let basis_start_j = self.cint_fdqc[bas_j][0];
            let basis_len_j = self.cint_fdqc[bas_j][1];
            let global_start = (basis_start_j+1)*basis_start_j/2;

            let mut i_length:usize = 0;
            let mut pair_length = 0; 
            for bas_i in 0..bas_j+1 {
                let basis_start_i = self.cint_fdqc[bas_i][0];
                let basis_len_i = self.cint_fdqc[bas_i][1];
                i_length += basis_len_i;

                if bas_i!=bas_j {
                    pair_length += basis_len_i*basis_len_j
                } else {
                    pair_length += (basis_len_i+1)*basis_len_i/2
                };
            }


            //if rayon::current_thread_index().unwrap()==0 {
            //    let bas_i = 0;
            //    let bas_k = n_basis_shell-1;
            //    let gk = n_basis_shell-1;

            //    let basis_len_i = self.cint_fdqc[bas_i][1];
            //    let basis_len_k = self.cint_fdqc[bas_k][1];
            //    //let basis_len_k = self.cint_aux_fdqc[bas_k][1];

            //    let buf_1 = RIFull::from_vec([basis_len_i, basis_len_j,basis_len_k], 
            //        cint_data.cint_3c2e(bas_i as i32, bas_j as i32, gk as i32)).unwrap();

            //    let buf_2 = RIFull::from_vec([basis_len_k,basis_len_i, basis_len_j], 
            //        cint_data.cint_3c2e(gk as i32,bas_i as i32, bas_j as i32)).unwrap();

            //    let buf_3 = RIFull::from_vec([basis_len_j, basis_len_i,basis_len_k], 
            //        cint_data.cint_3c2e(bas_i as i32, bas_j as i32, gk as i32)).unwrap();

            //    let buf_4 = RIFull::from_vec([basis_len_j, basis_len_i,basis_len_k], 
            //        cint_data.cint_3c2e(bas_i as i32, gk as i32, bas_j as i32)).unwrap();

            //    println!("buf_1: (i:{:2},j:{:2},k:{:2}) {:?}", bas_i, bas_j, gk, &buf_1.data);
            //    println!("buf_2: (i:{:2},j:{:2},k:{:2}) {:?}", bas_i, bas_j, gk, &buf_2.data);
            //    println!("buf_3: (i:{:2},j:{:2},k:{:2}) {:?}", bas_i, bas_j, gk, &buf_3.data);
            //    println!("buf_4: (i:{:2},j:{:2},k:{:2}) {:?}", bas_i, bas_j, gk, &buf_4.data);
            //}




            let mut ri_rayon =  RIFull::new([i_length,basis_len_j,n_auxbas],0.0);
            //let mut ri_rayon =  RIFull::new([n_auxbas,i_length,basis_len_j],0.0);
            let mut loc_rimatr = MatrixFull::new([pair_length,n_auxbas],0.0);
            for bas_i in 0..bas_j+1 {
                let basis_start_i = self.cint_fdqc[bas_i][0];
                let basis_len_i = self.cint_fdqc[bas_i][1];
                self.cint_aux_fdqc.iter().enumerate().for_each(|(k,bas_info)| {
                    let basis_start_k = bas_info[0];
                    let basis_len_k = bas_info[1];
                    let gk  = k + n_basis_shell;
                    //sub_time_records.count_start("ri: cint_3c2e");
                    let buf = RIFull::from_vec([basis_len_i, basis_len_j,basis_len_k], 
                        cint_data.cint_3c2e(bas_i as i32, bas_j as i32, gk as i32)).unwrap();
                    //let buf = RIFull::from_vec([basis_len_k,basis_len_i, basis_len_j], 
                    //    cint_data.cint_3c2e(gk as i32,bas_i as i32, bas_j as i32)).unwrap();
                    //sub_time_records.count("ri: cint_3c2e");
                    //sub_time_records.count_start("ri: copy_from_ri");
                    ri_rayon.copy_from_ri(
                        basis_start_i..basis_start_i+basis_len_i,
                        0..basis_len_j,
                        basis_start_k..basis_start_k+basis_len_k,
                        & buf, 
                        0..basis_len_i, 
                        0..basis_len_j, 
                        0..basis_len_k);
                    //sub_time_records.count("ri: copy_from_ri");
                });
            }

            cint_data.final_c2r();

            //sub_time_records.count_start("ri: ri_dot_sqrtv");
            let mut tmp_ovlp_matr = MatrixFull::new([i_length,n_auxbas],0.0);
            let mut aux_ovlp_matr = MatrixFull::new([i_length,n_auxbas],0.0);
            //let mut tmp_ovlp_matr = MatrixFull::new([n_auxbas,i_length],0.0);
            //let mut aux_ovlp_matr = MatrixFull::new([n_auxbas,i_length],0.0);
            let size = [i_length,n_auxbas];
            for loc_j in 0..basis_len_j {
                let gj = loc_j + basis_start_j;

                matr_copy_from_ri(&ri_rayon.data, &ri_rayon.size,0..i_length, 0..n_auxbas, loc_j, 1,
                    &mut tmp_ovlp_matr.data, &size, 0..i_length, 0..n_auxbas);

                _dgemm(
                    &tmp_ovlp_matr, (0..i_length,0..n_auxbas), 'N', 
                    &aux_v, (0..n_auxbas, 0..n_auxbas), 'N', 
                    &mut aux_ovlp_matr, (0..i_length,0..n_auxbas), 1.0, 0.0);

                let loc_x_start = (gj+1)*gj/2 - global_start;
                //loc_rimatr.copy_from_matr((loc_x_start..loc_x_start+gj+1), (0..n_auxbas), &aux_ovlp_matr, (0..gj+1), (0..n_auxbas));
                loc_rimatr.iter_columns_full_mut().zip(aux_ovlp_matr.iter_columns_full()).for_each(|(to, from)| {
                    to[loc_x_start..loc_x_start+gj+1].copy_from_slice(&from[0..gj+1])

                });
            }

            s.send((loc_rimatr,global_start,pair_length)).unwrap();
            //s.send((loc_rimatr,global_start,pair_length, sub_time_records)).unwrap();
        });

        time_records.new_item("prim ri receive", "");
        time_records.count_start("prim ri receive");

        receiver.into_iter().for_each(|(loc_rimatr, global_start, pair_length)| {
            //ri3fn.copy_from_matr(global_start..global_start+pair_length, 0..n_auxbas, &loc_rimatr, 0..pair_length, 0..n_auxbas);
            ri3fn.par_iter_columns_full_mut().zip(loc_rimatr.par_iter_columns_full()).for_each(|(to, from)| {
                to[global_start..global_start+pair_length].copy_from_slice(from);
            });
            //time_records.max(&sub_timerecords);
        });
        time_records.count("prim ri receive");

        let mut basbas2baspar = MatrixFull::new([n_basis,n_basis],0_usize);
        let mut baspar2basbas = vec![[0_usize;2];n_baspar];
        basbas2baspar.iter_columns_full_mut().enumerate().for_each(|(j,slice_x)| {
            &slice_x.iter_mut().enumerate().for_each(|(i,map_v)| {
                let baspar_ind = if i< j {(j+1)*j/2+i} else {(i+1)*i/2+j};
                *map_v = baspar_ind;
                baspar2basbas[baspar_ind] = [i,j];
            });
        });

        omp_set_num_threads_wrapper(self.ctrl.num_threads.unwrap());

        time_records.count("prim ri");

        time_records.count("all ri");


        if self.ctrl.print_level>=2 {
            time_records.report_all();
        }

        (ri3fn,basbas2baspar,baspar2basbas)

    }

    // generate the 3-center RI integrals and the basis pair symmetry is used to save the memory
    pub fn prepare_rimatr_for_ri_v_rayon_v03(&self) -> (MatrixFull<f64>,MatrixFull<usize>,Vec<[usize;2]>) {

        let mut time_records = utilities::TimeRecords::new();
        time_records.new_item("all ri", "for the total evaluations of ri3fn");
        time_records.count_start("all ri");

        time_records.new_item("prim ri", "for the pure three-center integrals");
        time_records.new_item("aux_ij", "for the coulomb matrix of auxiliary basis");
        time_records.new_item("inv_sqrt", "for the sqrt inverse of aux_ij");

        let n_basis = self.num_basis;
        let n_auxbas = self.num_auxbas;
        let n_baspar = (self.num_basis+1)*self.num_basis/2;

        // First the Cholesky decomposition `L` of the inverse of the auxiliary 2-center coulumb matrix: V=(\nu|\mu)
        time_records.count_start("aux_ij");
        let mut aux_v = self.int_ij_aux_columb();
        time_records.count("aux_ij");
        time_records.count_start("inv_sqrt");
        //aux_v = aux_v.lapack_power(-0.5, AUXBAS_THRESHOLD).unwrap();
        aux_v = _power_rayon_for_symmetric_matrix(&aux_v, -0.5, AUXBAS_THRESHOLD).unwrap();
        //aux_v = aux_v.to_matrixfullslicemut().cholesky_decompose_inverse('L').unwrap();
        time_records.count("inv_sqrt");

        // Then, prepare the 3-center integrals: O_V = (ij|\nu), and multiple with `L`
        time_records.count_start("prim ri");
        let mut tmp_ri3fn = MatrixFull::new([n_baspar,n_auxbas],0.0);
        let n_basis_shell = self.cint_bas.len();
        let n_auxbas_shell = self.cint_aux_bas.len();
        let cint_type = if self.ctrl.basis_type.to_lowercase()==String::from("spheric") {
            CintType::Spheric
        } else if self.ctrl.basis_type.to_lowercase()==String::from("cartesian") {
            CintType::Cartesian
        } else {
            panic!("Error:: Unknown basis type '{}'. Please use either 'spheric' or 'Cartesian'", 
                   self.ctrl.basis_type);
        };

        let (sender, receiver) = channel();
        //par_shellpair.par_iter().for_each_with(sender, |s, shell_index| {
        (0..n_basis_shell).rev().collect::<Vec<usize>>().par_iter().for_each_with(sender, |s, gj| {
            omp_set_num_threads_wrapper(1);
            //// ========== for efficiency debug ==============
            //let mut sub_time_records = utilities::TimeRecords::new();
            //sub_time_records.new_item("ri: cint_3c2e", "debug");
            //sub_time_records.new_item("ri: copy_from_ri","debug");
            //sub_time_records.new_item("ri: ri_dot_sqrtv","debug");
            //sub_time_records.new_item("ri: dgemm","debug");
            ////================================================
            let bas_j = *gj;
            // first, initialize rust_cint for each rayon threads
            let mut cint_data = self.initialize_cint(true);

            //let i = shell_index[0];
            //let j = shell_index[1];
            let basis_start_j = self.cint_fdqc[bas_j][0];
            let basis_len_j = self.cint_fdqc[bas_j][1];
            let global_start = (basis_start_j+1)*basis_start_j/2;

            let mut i_length:usize = 0;
            let mut pair_length = 0; 
            for bas_i in 0..bas_j+1 {
                let basis_start_i = self.cint_fdqc[bas_i][0];
                let basis_len_i = self.cint_fdqc[bas_i][1];
                i_length += basis_len_i;

                if bas_i!=bas_j {
                    pair_length += basis_len_i*basis_len_j
                } else {
                    pair_length += (basis_len_i+1)*basis_len_i/2
                };
            }


            let mut ri_rayon =  RIFull::new([i_length,basis_len_j,n_auxbas],0.0);
            let mut loc_rimatr = MatrixFull::new([pair_length,n_auxbas],0.0);
            for bas_i in 0..bas_j+1 {
                let basis_start_i = self.cint_fdqc[bas_i][0];
                let basis_len_i = self.cint_fdqc[bas_i][1];
                self.cint_aux_fdqc.iter().enumerate().for_each(|(k,bas_info)| {
                    let basis_start_k = bas_info[0];
                    let basis_len_k = bas_info[1];
                    let gk  = k + n_basis_shell;
                    //sub_time_records.count_start("ri: cint_3c2e");
                    let buf = RIFull::from_vec([basis_len_i, basis_len_j,basis_len_k], 
                        cint_data.cint_3c2e(bas_i as i32, bas_j as i32, gk as i32)).unwrap();
                    //sub_time_records.count("ri: cint_3c2e");
                    //sub_time_records.count_start("ri: copy_from_ri");
                    ri_rayon.copy_from_ri(
                        basis_start_i..basis_start_i+basis_len_i,
                        0..basis_len_j,
                        basis_start_k..basis_start_k+basis_len_k,
                        & buf, 
                        0..basis_len_i, 
                        0..basis_len_j, 
                        0..basis_len_k);
                    //sub_time_records.count("ri: copy_from_ri");
                });
            }

            cint_data.final_c2r();

            //sub_time_records.count_start("ri: ri_dot_sqrtv");
            let mut tmp_ovlp_matr = MatrixFull::new([i_length,n_auxbas],0.0);
            //let mut aux_ovlp_matr = MatrixFull::new([i_length,n_auxbas],0.0);
            let size = [i_length,n_auxbas];
            for loc_j in 0..basis_len_j {
                let gj = loc_j + basis_start_j;
                let loc_x_start = (gj+1)*gj/2 - global_start;

                matr_copy_from_ri(&ri_rayon.data, &ri_rayon.size,0..i_length, 0..n_auxbas, loc_j, 1,
                    &mut tmp_ovlp_matr.data, &size, 0..i_length, 0..n_auxbas);

                loc_rimatr.copy_from_matr((loc_x_start..loc_x_start+gj+1), (0..n_auxbas), &tmp_ovlp_matr, (0..gj+1), (0..n_auxbas));
            }


            s.send((loc_rimatr,global_start,pair_length)).unwrap();
            //s.send((loc_rimatr,global_start,pair_length, sub_time_records)).unwrap();
        });

        time_records.new_item("rec_ri", "the collection of primitry RI tensor after multi-threads parallization");
        time_records.count_start("rec_ri");

        receiver.into_iter().for_each(|(loc_rimatr, global_start, pair_length)| {
            tmp_ri3fn.par_iter_columns_full_mut().zip(loc_rimatr.par_iter_columns_full()).for_each(|(to, from)| {
                to[global_start..global_start+pair_length].copy_from_slice(from);
            });
            //tmp_ri3fn.copy_from_matr(global_start..global_start+pair_length, 0..n_auxbas, &loc_rimatr, 0..pair_length, 0..n_auxbas);
            //time_records.max(&sub_timerecords);
        });
        time_records.count("rec_ri");

        let mut basbas2baspar = MatrixFull::new([n_basis,n_basis],0_usize);
        let mut baspar2basbas = vec![[0_usize;2];n_baspar];
        basbas2baspar.iter_columns_full_mut().enumerate().for_each(|(j,slice_x)| {
            &slice_x.iter_mut().enumerate().for_each(|(i,map_v)| {
                let baspar_ind = if i< j {(j+1)*j/2+i} else {(i+1)*i/2+j};
                *map_v = baspar_ind;
                baspar2basbas[baspar_ind] = [i,j];
            });
        });
        time_records.count("prim ri");

        time_records.new_item("ri_v","for RI dot aux_v^{-1/2}");
        time_records.count_start("ri_v");

        omp_set_num_threads_wrapper(self.ctrl.num_threads.unwrap());
        let mut ri3fn = MatrixFull::new([n_baspar,n_auxbas],0.0);
        _dgemm_full(&tmp_ri3fn, 'N', &aux_v, 'N', &mut ri3fn, 1.0, 0.0);
        time_records.count("ri_v");


        time_records.count("all ri");


        if self.ctrl.print_level>=2 {
            time_records.report_all();
        }

        (ri3fn,basbas2baspar,baspar2basbas)

    }

    // generate the 3-center RI integrals and the basis pair symmetry is used to save the memory
    pub fn prepare_rimatr_for_ri_v_rayon_v04(&self) -> (MatrixFull<f64>,MatrixFull<usize>,Vec<[usize;2]>) {

        let mut time_records = utilities::TimeRecords::new();
        time_records.new_item("all ri", "for the total evaluations of ri3fn");
        time_records.count_start("all ri");

        time_records.new_item("prim ri", "for the pure three-center integrals");
        time_records.new_item("aux_ij", "for the coulomb matrix of auxiliary basis");
        time_records.new_item("inv_sqrt", "for the sqrt inverse of aux_ij");

        let n_basis = self.num_basis;
        let n_auxbas = self.num_auxbas;
        let n_baspar = (self.num_basis+1)*self.num_basis/2;

        // First the Cholesky decomposition `L` of the inverse of the auxiliary 2-center coulumb matrix: V=(\nu|\mu)
        time_records.count_start("aux_ij");
        let mut aux_v = self.int_ij_aux_columb();
        time_records.count("aux_ij");
        time_records.count_start("inv_sqrt");
        aux_v = _power_rayon_for_symmetric_matrix(&aux_v, -0.5, AUXBAS_THRESHOLD).unwrap();
        //aux_v = aux_v.to_matrixfullslicemut().cholesky_decompose_inverse('L').unwrap_or(
        //    _power_rayon(&aux_v, -0.5, AUXBAS_THRESHOLD).unwrap()
        //);
        //let (tmp_aux_v, n_sigular, converge_flag) = _newton_schulz_inverse_square_root(&aux_v, 1.0e-8, 0.1, 100);
        //if ! converge_flag {
        //    aux_v = _power_rayon(&aux_v, -0.5, AUXBAS_THRESHOLD).unwrap();
        //} else {
        //    aux_v = tmp_aux_v.clone()
        //}
        time_records.count("inv_sqrt");

        // Then, prepare the 3-center integrals: O_V = (ij|\nu), and multiple with `L`
        time_records.count_start("prim ri");
        let mut tmp_ri3fn = MatrixFull::new([n_auxbas, n_baspar],0.0);
        let n_basis_shell = self.cint_bas.len();
        let n_auxbas_shell = self.cint_aux_bas.len();
        let cint_type = if self.ctrl.basis_type.to_lowercase()==String::from("spheric") {
            CintType::Spheric
        } else if self.ctrl.basis_type.to_lowercase()==String::from("cartesian") {
            CintType::Cartesian
        } else {
            panic!("Error:: Unknown basis type '{}'. Please use either 'spheric' or 'Cartesian'", 
                   self.ctrl.basis_type);
        };

        let (sender, receiver) = channel();
        (0..n_basis_shell).rev().collect::<Vec<usize>>().par_iter().for_each_with(sender, |s, gj| {

            omp_set_num_threads_wrapper(1);

            let bas_j = *gj;
            // first, initialize rust_cint for each rayon threads
            let mut cint_data = self.initialize_cint(true);

            let basis_start_j = self.cint_fdqc[bas_j][0];
            let basis_len_j = self.cint_fdqc[bas_j][1];
            let global_start = (basis_start_j+1)*basis_start_j/2;

            let mut i_length:usize = 0;
            let mut pair_length = 0; 
            for bas_i in 0..bas_j+1 {
                let basis_start_i = self.cint_fdqc[bas_i][0];
                let basis_len_i = self.cint_fdqc[bas_i][1];
                i_length += basis_len_i;

                if bas_i!=bas_j {
                    pair_length += basis_len_i*basis_len_j
                } else {
                    pair_length += (basis_len_i+1)*basis_len_i/2
                };
            }


            let mut ri_rayon =  RIFull::new([n_auxbas,i_length,basis_len_j],0.0);
            let mut loc_rimatr = MatrixFull::new([n_auxbas,pair_length],0.0);
            for bas_i in 0..bas_j+1 {
                let basis_start_i = self.cint_fdqc[bas_i][0];
                let basis_len_i = self.cint_fdqc[bas_i][1];
                self.cint_aux_fdqc.iter().enumerate().for_each(|(k,bas_info)| {
                    let basis_start_k = bas_info[0];
                    let basis_len_k = bas_info[1];
                    let gk  = k + n_basis_shell;

                    let buf = MatrixFull::from_vec([basis_len_i*basis_len_j, basis_len_k], 
                        cint_data.cint_3c2e(bas_i as i32, bas_j as i32, gk as i32)).unwrap();
                    
                    //(0..basis_len_i*basis_len_j).for_each(|index| {
                    //    let loc_j = index/basis_len_i;
                    //    let gj = loc_j;
                    //    let loc_i = index%basis_len_i;
                    //    let gi = loc_i + basis_start_i;
                    //    let tmp_aux = buf.iter_row(index);
                    //    ri_rayon.get_reducing_matrix_mut(gj).unwrap()
                    //        .get_column_mut(gi)[basis_start_k..basis_start_k+basis_len_k].iter_mut()
                    //        .zip(tmp_aux).for_each(|(to, from)| *to = *from)
                    //});

                    if bas_i < bas_j {
                        for loc_j in (0..basis_len_j) {
                            let index_s = loc_j * basis_len_i;
                            let gj = loc_j + basis_start_j;
                            let loc_x_start = (gj+1)*gj/2 - global_start;
                            for loc_i in (0..basis_len_i) {
                                let index = index_s + loc_i;
                                let gi = loc_i + basis_start_i;
                                let loc_x = loc_x_start + gi;
                                let tmp_aux = buf.iter_row(index);
                                loc_rimatr.slice_column_mut(loc_x)[basis_start_k..basis_start_k+basis_len_k].iter_mut()
                                .zip(tmp_aux).for_each(|(to, from)| *to = *from)
                            }
                        }
                    } else {
                        for loc_j in (0..basis_len_j) {
                            let index_s = loc_j * basis_len_i;
                            let gj = loc_j + basis_start_j;
                            let loc_x_start = (gj+1)*gj/2 - global_start;
                            for loc_i in (0..loc_j+1) {
                                let index = index_s + loc_i;
                                let gi = loc_i + basis_start_i;
                                let loc_x = loc_x_start + gi;
                                let tmp_aux = buf.iter_row(index);
                                loc_rimatr.slice_column_mut(loc_x)[basis_start_k..basis_start_k+basis_len_k].iter_mut()
                                .zip(tmp_aux).for_each(|(to, from)| *to = *from)
                            }
                        }
                    }
                });
            }

            cint_data.final_c2r();


            s.send((loc_rimatr,global_start,pair_length)).unwrap();
        });

        time_records.new_item("rec_ri", "the collection of primitry RI tensor after multi-threads parallization");
        time_records.count_start("rec_ri");

        receiver.into_iter().for_each(|(loc_rimatr, global_start, pair_length)| {
            //tmp_ri3fn.copy_from_matr(0..n_auxbas, global_start..global_start+pair_length, &loc_rimatr, 0..n_auxbas, 0..pair_length);
            tmp_ri3fn.par_iter_columns_mut(global_start..global_start+pair_length).unwrap().zip(loc_rimatr.par_iter_columns_full())
                .for_each(|(to, from)| {
                    to.copy_from_slice(from)
            })
        });
        time_records.count("rec_ri");

        let mut basbas2baspar = MatrixFull::new([n_basis,n_basis],0_usize);
        let mut baspar2basbas = vec![[0_usize;2];n_baspar];
        basbas2baspar.iter_columns_full_mut().enumerate().for_each(|(j,slice_x)| {
            &slice_x.iter_mut().enumerate().for_each(|(i,map_v)| {
                let baspar_ind = if i< j {(j+1)*j/2+i} else {(i+1)*i/2+j};
                *map_v = baspar_ind;
                baspar2basbas[baspar_ind] = [i,j];
            });
        });
        time_records.count("prim ri");

        time_records.new_item("ri_v","for RI dot aux_v^{-1/2}");
        time_records.count_start("ri_v");

        omp_set_num_threads_wrapper(self.ctrl.num_threads.unwrap());
        let mut ri3fn = MatrixFull::new([n_baspar,n_auxbas],0.0);
        _dgemm_full(&tmp_ri3fn, 'T', &aux_v, 'N', &mut ri3fn, 1.0, 0.0);
        time_records.count("ri_v");


        time_records.count("all ri");


        if self.ctrl.print_level>=2 {
            time_records.report_all();
        }

        (ri3fn,basbas2baspar,baspar2basbas)

    }


    pub fn prepare_inv_aux_matr(&self) -> MatrixFull<f64> {
        let mut aux_v = self.int_ij_aux_columb();
        aux_v = _power_rayon_for_symmetric_matrix(&aux_v, -0.5, AUXBAS_THRESHOLD).unwrap();
        aux_v
    }

    //pub fn prepare_prim_ri_for_baspair(&self, ) -> 

    pub fn prepare_baspair_map(&self) -> (MatrixFull<usize>, Vec<[usize;2]>) {
        let n_basis = self.num_basis;
        let n_auxbas = self.num_auxbas;
        let n_baspar = (self.num_basis+1)*self.num_basis/2;

        let mut basbas2baspar = MatrixFull::new([n_basis,n_basis],0_usize);
        let mut baspar2basbas = vec![[0_usize;2];n_baspar];
        basbas2baspar.iter_columns_full_mut().enumerate().for_each(|(j,slice_x)| {
            &slice_x.iter_mut().enumerate().for_each(|(i,map_v)| {
                let baspar_ind = if i< j {(j+1)*j/2+i} else {(i+1)*i/2+j};
                *map_v = baspar_ind;
                baspar2basbas[baspar_ind] = [i,j];
            });
        });

        (basbas2baspar, baspar2basbas)
    }

    /// Distributed (MPI) build of the in-core RI tensor.
    ///
    /// `keep` is the optional build-time pre-screen: a conservative superset of the packed
    /// AO-pair rows that the storage-level pruning will retain (see
    /// `scf_io::prescreen_pair_keep`). When given, each rank carries only the rows in `keep`
    /// through the metric solve; the other rows of the local block stay zero. The block keeps
    /// its shape, so the redistribution that follows is untouched, and the zeroed rows are
    /// exactly the ones the pruning drops anyway — the final row set and `PairMap` are the
    /// same as without the pre-screen.
    pub fn prepare_rimatr_for_ri_v_mpi_rayon(&self, omega: Option<f64>, keep: Option<&[usize]>, mpi_operator: &Option<MPIOperator>) -> (MatrixFull<f64>,MatrixFull<usize>,Vec<[usize;2]>) {
        let n_basis = self.num_basis;
        let n_auxbas = self.num_auxbas;
        let n_baspar = (self.num_basis+1)*self.num_basis/2;

        // AJZ: this will cost n_baspar * n_auxbas * 8 bytes memory
        // for safety, we apply 1.5 factor to limit the memory usage
        let estimated_mem = 1.5 * n_baspar as f64 * n_auxbas as f64 * 8.0 / (1024.0 * 1024.0); // in MB
        let avail_mem = self.ctrl.max_memory.map(|m| m - crate::utilities::memory_batch::detect_used_memory_mb("proc"));
        utilities::memory_batch::handle_memory_exceed(estimated_mem, avail_mem, self.ctrl.abort_on_mem_exceed);

        #[cfg(feature = "mpi")]
        if let (Some(mpi_op), Some(loc_mpi_data)) = (&mpi_operator, &self.mpi_data) {

            if let Some(omega_libcint) = omega {
                // range-separated (RSH) case: delegated to the dedicated MPI module,
                // implemented with rest_tensors (see `mpi_io::rimatr_sr` for details)
                return crate::mpi_io::rimatr_sr::prepare_rimatr_sr_distributed(
                    self,
                    omega_libcint,
                    mpi_op,
                    loc_mpi_data,
                );
            }

            let my_rank = mpi_op.rank;

            if my_rank == 0 {println!("debug: enter the generation of inv_aux_matr")};
            // Build the full 2c-2e metric J following the `j2c_decomp` policy, so that the
            // distributed rimatr is mathematically identical to the serial one
            // (`generate_rimatr_bare`); a fixed eigen power here would mismatch the
            // Cholesky branch of the analytical gradient when `policy = "cholesky"`.
            // The integral (O(naux^2)) is computed only on rank 0 and broadcast,
            // mirroring `diagonalize_hamiltonian_outside`.
            let mut j2c_full = MatrixFull::new([n_auxbas, n_auxbas], 0.0);
            if my_rank == 0 {
                j2c_full = self.int_ij_aux_columb();
            }
            crate::mpi_io::mpi_broadcast_matrixfull(&mpi_op.world, &mut j2c_full, 0);
            if my_rank == 0 {println!("debug: leave the generation of inv_aux_matr")};
            let (basbas2baspar, baspar2basbas) = self.prepare_baspair_map();
            if let (Some(auxbas_distribution), Some(baspar_distribution)) = 
                (&loc_mpi_data.auxbas, &loc_mpi_data.baspar) {

                let mut ri3fn = MatrixFull::new([n_baspar, auxbas_distribution[my_rank].len()], 0.0);

                // metric factor for the non-distributed path, and the distributed
                // (ScaLAPACK) Cholesky context when the sandbox mode selects it
                let mut aux_v = MatrixFull::new([n_auxbas, n_auxbas], 0.0);
                #[cfg(all(feature = "mpi", feature = "scalapack"))]
                let dist_ctx: Option<(
                    tensors::matrix::distributedmatrixfull::DistributedMatrixFull<f64>,
                    i32,
                    Vec<std::ops::Range<usize>>,
                )> = {
                    if crate::mpi_io::j2c_distributed::use_distributed_j2c(
                        &self.ctrl.j2c_decomp,
                        n_auxbas,
                        mpi_op.size,
                    ) {
                        let grid = &mpi_op.cblacsgrid;
                        let u_dist = crate::mpi_io::j2c_distributed::distributed_cholesky(grid, &j2c_full);
                        let nb = u_dist.desc[5];
                        let col_ranges: Vec<std::ops::Range<usize>> = baspar_distribution
                            .iter()
                            .map(|(r, _, _)| r.clone())
                            .collect();
                        Some((u_dist, nb, col_ranges))
                    } else {
                        aux_v = crate::mpi_io::prepare_j2c_solve_factor(&j2c_full, &self.ctrl.j2c_decomp);
                        None
                    }
                };
                #[cfg(not(all(feature = "mpi", feature = "scalapack")))]
                let dist_ctx: Option<()> = {
                    aux_v = crate::mpi_io::prepare_j2c_solve_factor(&j2c_full, &self.ctrl.j2c_decomp);
                    None
                };

                let (baspar, sbsh, ebsh) = &baspar_distribution[my_rank];
                // Build-time pre-screen, per rank: `baspar` is this rank's contiguous range
                // of packed pair indices, kept as a plain `Range` so that the redistribution
                // below needs no change; the rows outside `keep` are solved as zero.
                let local_sel: Option<Vec<usize>> = keep.map(|keep| {
                    keep.iter()
                        .filter(|p| baspar.contains(p))
                        .map(|p| p - baspar.start)
                        .collect()
                });
                // IMPORTANT: scatter_to_blockcyclic / distributed_triangular_solve /
                // gather_from_blockcyclic are COLLECTIVE over the BLACS grid — every
                // rank must participate even when it has no basis-pair work. Ranks
                // with baspar.len() == 0 pass a zero-column matrix (empty participation)
                // rather than skipping the collective pipeline, which would deadlock
                // the ranks that DO have data.
                #[cfg(all(feature = "mpi", feature = "scalapack"))]
                let loc_ri3fn = if dist_ctx.is_some() {
                    // collective path: ALL ranks enter scatter/solve/gather
                    let tmp = if baspar.len() > 0 {
                        self.prepare_rimatr_for_ri_v_mpi_slot(*sbsh, *ebsh)
                    } else {
                        // empty [naux × 0] so scatter sends nothing but still participates
                        MatrixFull::new([n_auxbas, 0], 0.0)
                    };
                    if let Some((u_dist, nb, col_ranges)) = &dist_ctx {
                        let b_dist = crate::mpi_io::j2c_distributed::scatter_to_blockcyclic(
                            &mpi_op.world, &mpi_op.cblacsgrid, *nb, col_ranges, &tmp,
                        );
                        let mut x_dist = b_dist;
                        crate::mpi_io::j2c_distributed::distributed_triangular_solve(
                            &mpi_op.cblacsgrid, u_dist, &mut x_dist,
                        );
                        let x_contig = crate::mpi_io::j2c_distributed::gather_from_blockcyclic(
                            &mpi_op.world, &mpi_op.cblacsgrid, *nb, col_ranges, &x_dist,
                        );
                        x_contig.transpose()
                    } else {
                        unreachable!("dist_ctx is Some but inner None")
                    }
                } else if baspar.len() > 0 {
                    // serial fallback (no distributed j2c): only ranks with work
                    // participate (solve_tmp_to_ri3fn is local)
                    let tmp = self.prepare_rimatr_for_ri_v_mpi_slot(*sbsh, *ebsh);
                    self.solve_tmp_to_ri3fn_local(&aux_v, tmp, &local_sel)
                } else {
                    MatrixFull::empty()
                };
                #[cfg(not(all(feature = "mpi", feature = "scalapack")))]
                let loc_ri3fn = if baspar.len() > 0 {
                    let tmp = self.prepare_rimatr_for_ri_v_mpi_slot(*sbsh, *ebsh);
                    self.solve_tmp_to_ri3fn_local(&aux_v, tmp, &local_sel)
                } else {
                    MatrixFull::empty()
                };

                ////println!("debug mpi 0 of rank {}", my_rank);
                //if my_rank == 0 {println!("debug: enter the mpi communication of ri_v matrix")};
                //let loc_ri3fn = mpi_isend_irecv_wrt_distribution_v02(&mpi_op.world, &loc_ri3fn.data_ref().unwrap(), auxbas_distribution, loc_ri3fn.size()[0]);
                //if my_rank == 0 {println!("debug: leave the mpi communication of ri_v matrix")};
                
                //loc_ri3fn.iter().enumerate().for_each(|(rank_i, v)| {
                //    let (baspar, sbsh, ebsh) = &baspar_distribution[rank_i];
                //    ri3fn.iter_submatrix_mut(baspar.clone(), 0..auxbas_distribution[my_rank].len())
                //    .zip(v.iter()).for_each(|(to, from)| {*to = *from});
                //});

                if my_rank == 0 {println!("debug: enter the mpi communication of ri_v matrix in v03")};
                mpi_isend_irecv_wrt_distribution_v03(&mpi_op.world, &mut ri3fn, &loc_ri3fn.data_ref().unwrap(), auxbas_distribution, baspar_distribution, loc_ri3fn.size()[0]);
                if my_rank == 0 {println!("debug: leave the mpi communication of ri_v matrix in v03")};



                (ri3fn, basbas2baspar, baspar2basbas)
            } else {
                self.build_rimatr_local_or_prescreened(omega, keep)
            }

        } else {
            self.build_rimatr_local_or_prescreened(omega, keep)
        }
        #[cfg(not(feature = "mpi"))]
        { self.build_rimatr_local_or_prescreened(omega, keep) }


    }

    pub fn prepare_rimatr_for_ri_v_mpi_slot(&self, sbsh: usize, ebsh: usize) -> MatrixFull<f64> {
        //let n_basis_shell = self.cint_bas.len() as i32;
        //let n_auxbas_shell = self.cint_aux_bas.len() as i32;

        //let n_basis = self.num_basis;
        //let n_auxbas = self.num_auxbas;
        //let n_baspar = (self.num_basis+1)*self.num_basis/2;

        //let shl_slices: Vec<[i32;2]> = vec![[0, (ebsh+1) as i32], [sbsh as i32, (ebsh + 1) as i32], [n_basis_shell, n_basis_shell + n_auxbas_shell]];

        //let mut cint_data = self.initialize_cint(true);

        ////let mut out = vec![];
        ////let mut out_shape = vec![];
        ////(out, out_shape) = cint_data.integral_s2ij::<int3c2e>(Some(&shl_slices));

        //let mut tmp_ri3fn = MatrixFull::from_vec(out_shape.clone().try_into().unwrap(),out).unwrap();

        //let mut ri3fn = MatrixFull::new(out_shape.clone().try_into().unwrap(),0.0);

        //utilities::omp_set_num_threads_wrapper(self.ctrl.num_threads.unwrap());
        //_dgemm_full(&tmp_ri3fn, 'N', aux_v, 'N', &mut ri3fn, 1.0, 0.0);

        //ri3fn

        let n_basis_shell = self.cint_bas.len();
        let n_auxbas_shell = self.cint_aux_bas.len();
        let n_basis = self.num_basis;
        let n_basis = self.num_basis;
        let n_auxbas = self.num_auxbas;
        let n_baspar = (self.num_basis+1)*self.num_basis/2;

        let basis_start = self.cint_fdqc[sbsh][0];
        let s_baspar = (basis_start+1)*basis_start/2;
        let basis_end = self.cint_fdqc[ebsh][0] + self.cint_fdqc[ebsh][1]-1;
        let e_baspar = (basis_end+1)*basis_end/2 + basis_end;

        let loc_n_baspar = e_baspar - s_baspar + 1;

        let mut tmp_ri3fn = MatrixFull::new([n_auxbas, loc_n_baspar],0.0);
        let (sender, receiver) = channel();
        //(0..n_basis_shell).rev().collect::<Vec<usize>>().par_iter().for_each_with(sender, |s, gj| {
        (sbsh..ebsh+1).collect::<Vec<usize>>().par_iter().for_each_with(sender, |s,gj| {

            omp_set_num_threads_wrapper(1);

            let bas_j = *gj;
            // first, initialize rust_cint for each rayon threads
            let mut cint_data = self.initialize_cint(true);

            let basis_start_j = self.cint_fdqc[bas_j][0];
            let basis_len_j = self.cint_fdqc[bas_j][1];
            let global_start = (basis_start_j+1)*basis_start_j/2;

            let mut i_length:usize = 0;
            let mut pair_length = 0; 
            for bas_i in 0..bas_j+1 {
                let basis_start_i = self.cint_fdqc[bas_i][0];
                let basis_len_i = self.cint_fdqc[bas_i][1];
                i_length += basis_len_i;

                if bas_i!=bas_j {
                    pair_length += basis_len_i*basis_len_j
                } else {
                    pair_length += (basis_len_i+1)*basis_len_i/2
                };
            }


            let mut ri_rayon =  RIFull::new([n_auxbas,i_length,basis_len_j],0.0);
            let mut loc_rimatr = MatrixFull::new([n_auxbas,pair_length],0.0);
            for bas_i in 0..bas_j+1 {
                let basis_start_i = self.cint_fdqc[bas_i][0];
                let basis_len_i = self.cint_fdqc[bas_i][1];
                self.cint_aux_fdqc.iter().enumerate().for_each(|(k,bas_info)| {
                    let basis_start_k = bas_info[0];
                    let basis_len_k = bas_info[1];
                    let gk  = k + n_basis_shell;

                    let buf = MatrixFull::from_vec([basis_len_i*basis_len_j, basis_len_k], 
                        cint_data.cint_3c2e(bas_i as i32, bas_j as i32, gk as i32)).unwrap();
                    
                    if bas_i < bas_j {
                        for loc_j in (0..basis_len_j) {
                            let index_s = loc_j * basis_len_i;
                            let gj = loc_j + basis_start_j;
                            let loc_x_start = (gj+1)*gj/2 - global_start;
                            for loc_i in (0..basis_len_i) {
                                let index = index_s + loc_i;
                                let gi = loc_i + basis_start_i;
                                let loc_x = loc_x_start + gi;
                                let tmp_aux = buf.iter_row(index);
                                loc_rimatr.slice_column_mut(loc_x)[basis_start_k..basis_start_k+basis_len_k].iter_mut()
                                .zip(tmp_aux).for_each(|(to, from)| *to = *from)
                            }
                        }
                    } else {
                        for loc_j in (0..basis_len_j) {
                            let index_s = loc_j * basis_len_i;
                            let gj = loc_j + basis_start_j;
                            let loc_x_start = (gj+1)*gj/2 - global_start;
                            for loc_i in (0..loc_j+1) {
                                let index = index_s + loc_i;
                                let gi = loc_i + basis_start_i;
                                let loc_x = loc_x_start + gi;
                                let tmp_aux = buf.iter_row(index);
                                loc_rimatr.slice_column_mut(loc_x)[basis_start_k..basis_start_k+basis_len_k].iter_mut()
                                .zip(tmp_aux).for_each(|(to, from)| *to = *from)
                            }
                        }
                    }
                });
            }

            cint_data.final_c2r();


            s.send((loc_rimatr,global_start-s_baspar,pair_length)).unwrap();
        });


        receiver.into_iter().for_each(|(loc_rimatr, global_start, pair_length)| {
            //tmp_ri3fn.copy_from_matr(0..n_auxbas, global_start..global_start+pair_length, &loc_rimatr, 0..n_auxbas, 0..pair_length);
            tmp_ri3fn.par_iter_columns_mut(global_start..global_start+pair_length).unwrap().zip(loc_rimatr.par_iter_columns_full())
                .for_each(|(to, from)| {
                    to.copy_from_slice(from)
            })
        });

        // return the raw 3c2e block [naux, loc_n_baspar]; the metric solve is performed
        // by the caller (`solve_tmp_to_ri3fn` or the distributed path)
        tmp_ri3fn

    }

    /// Solve the raw 3c2e block `tmp_ri3fn` (`[naux, loc_n_baspar]`) against the metric
    /// factor `aux_v` following the `j2c_decomp` policy, returning
    /// `ri3fn` (`[loc_n_baspar, naux]`):
    /// - `Cd`: `cderi = j3c · L^{-T}`  ⟺  `U^T · X = tmp_ri3fn` (in-place triangular
    ///   solve with the upper Cholesky factor `U`, then `ri3fn = X^T`),
    /// - `Eig`: `cderi = j3c · J^{-1/2}` via the dense matrix multiplication.
    fn solve_tmp_to_ri3fn(&self, aux_v: &MatrixFull<f64>, tmp_ri3fn: MatrixFull<f64>) -> MatrixFull<f64> {
        let n_auxbas = tmp_ri3fn.size()[0];
        let loc_n_baspar = tmp_ri3fn.size()[1];
        let mut ri3fn = MatrixFull::new([loc_n_baspar, n_auxbas], 0.0);
        omp_set_num_threads_wrapper(self.ctrl.num_threads.unwrap());
        let mut tmp_ri3fn = tmp_ri3fn;
        match self.ctrl.j2c_decomp.policy {
            crate::ri_jk::J2CDecompPolicy::Cd => {
                assert!(
                    _dtrtrs(aux_v, &mut tmp_ri3fn, 'U', 'T', 'N'),
                    "The triangular solve against the Cholesky factor of the 2c-2e metric failed."
                );
                ri3fn = tmp_ri3fn.transpose();
            },
            _ => {
                _dgemm_full(&tmp_ri3fn, 'T', aux_v, 'N', &mut ri3fn, 1.0, 0.0);
            },
        }
        ri3fn
    }

    /// Build the in-core RI tensor locally: the plain rayon builder, or the pre-screened
    /// one when a `keep` row selection is available.
    fn build_rimatr_local_or_prescreened(
        &self,
        omega: Option<f64>,
        keep: Option<&[usize]>,
    ) -> (MatrixFull<f64>, MatrixFull<usize>, Vec<[usize; 2]>) {
        match keep {
            Some(keep) => self.prepare_rimatr_for_ri_v_rayon_prescreened(omega, Some(keep)),
            None => self.prepare_rimatr_for_ri_v_rayon(omega),
        }
    }

    /// Metric solve of one rank's local 3c block, carrying only the rows listed in `sel`
    /// through the solve (`None` solves the whole block).
    ///
    /// The returned block keeps the shape of the full local row space, with the rows left
    /// out set to zero, so the redistribution that follows is byte-for-byte the layout it
    /// expects and no collective changes. The zeroed rows are the ones the build-time
    /// pre-screen proved negligible and the storage-level pruning drops anyway; see
    /// [`Molecule::prepare_rimatr_for_ri_v_mpi_rayon`].
    fn solve_tmp_to_ri3fn_local(
        &self,
        aux_v: &MatrixFull<f64>,
        tmp_ri3fn: MatrixFull<f64>,
        sel: &Option<Vec<usize>>,
    ) -> MatrixFull<f64> {
        let sel = match sel {
            None => return self.solve_tmp_to_ri3fn(aux_v, tmp_ri3fn),
            Some(sel) => sel,
        };
        let n_auxbas = tmp_ri3fn.size()[0];
        let nrows = tmp_ri3fn.size()[1];
        let mut out = MatrixFull::new([nrows, n_auxbas], 0.0);
        if sel.is_empty() {
            return out;
        }
        let mut tmp_sel = MatrixFull::new([n_auxbas, sel.len()], 0.0);
        for (k_sel, &row) in sel.iter().enumerate() {
            tmp_sel
                .iter_column_mut(k_sel)
                .zip(tmp_ri3fn.iter_column(row))
                .for_each(|(to, from)| *to = *from);
        }
        let solved = self.solve_tmp_to_ri3fn(aux_v, tmp_sel);
        for (k_sel, &row) in sel.iter().enumerate() {
            out.iter_row_mut(row)
                .zip(solved.iter_row(k_sel))
                .for_each(|(to, from)| *to = *from);
        }
        out
    }


    // generate the 3-center RI integrals and the basis pair symmetry is used to save the memory
    pub fn prepare_rimatr_for_ri_v_rayon_v05(&self) -> (MatrixFull<f64>,MatrixFull<usize>,Vec<[usize;2]>) {
        use rest_libcint_wrapper::*;

        omp_set_num_threads_wrapper(self.ctrl.num_threads.unwrap());

        let mut time_records = utilities::TimeRecords::new();
        time_records.new_item("all ri", "for the total evaluations of ri3fn");
        time_records.count_start("all ri");

        time_records.new_item("prim ri", "for the pure three-center integrals");
        time_records.new_item("aux_ij", "for the coulomb matrix of auxiliary basis");
        time_records.new_item("inv_sqrt", "for the sqrt inverse of aux_ij");

        let n_basis = self.num_basis;
        let n_auxbas = self.num_auxbas;
        let n_baspar = (self.num_basis+1)*self.num_basis/2;

        // First the Cholesky decomposition `L` of the inverse of the auxiliary 2-center coulumb matrix: V=(\nu|\mu)
        time_records.count_start("aux_ij");
        let mut aux_v = self.int_ij_aux_columb();
        time_records.count("aux_ij");
        time_records.count_start("inv_sqrt");
        aux_v = _power_rayon_for_symmetric_matrix(&aux_v, -0.5, AUXBAS_THRESHOLD).unwrap();
        //aux_v = aux_v.to_matrixfullslicemut().cholesky_decompose_inverse('L').unwrap_or(
        //    _power_rayon(&aux_v, -0.5, AUXBAS_THRESHOLD).unwrap()
        //);
        //let (tmp_aux_v, n_sigular, converge_flag) = _newton_schulz_inverse_square_root(&aux_v, 1.0e-8, 0.1, 100);
        //if ! converge_flag {
        //    aux_v = _power_rayon(&aux_v, -0.5, AUXBAS_THRESHOLD).unwrap();
        //} else {
        //    aux_v = tmp_aux_v.clone()
        //}
        time_records.count("inv_sqrt");

        // Then, prepare the 3-center integrals: O_V = (ij|\nu), and multiple with `L`
        time_records.count_start("prim ri");
        let n_basis_shell = self.cint_bas.len() as i32;
        let n_auxbas_shell = self.cint_aux_bas.len() as i32;
        let cint_type = if self.ctrl.basis_type.to_lowercase()==String::from("spheric") {
            CintType::Spheric
        } else if self.ctrl.basis_type.to_lowercase()==String::from("cartesian") {
            CintType::Cartesian
        } else {
            panic!("Error:: Unknown basis type '{}'. Please use either 'spheric' or 'Cartesian'", 
                   self.ctrl.basis_type);
        };

        let mut out = vec![];
        let mut out_shape = vec![];
        let shl_slices = vec![[0, n_basis_shell], [0, n_basis_shell], [n_basis_shell, n_basis_shell + n_auxbas_shell]];

        let mut cint_data = self.initialize_cint(true);

        (out, out_shape) = cint_data.integral_s2ij::<int3c2e>(Some(&shl_slices));

        cint_data.final_c2r();

        let mut tmp_ri3fn = MatrixFull::from_vec([n_baspar, n_auxbas],out).unwrap();

        let mut basbas2baspar = MatrixFull::new([n_basis,n_basis],0_usize);
        let mut baspar2basbas = vec![[0_usize;2];n_baspar];
        basbas2baspar.iter_columns_full_mut().enumerate().for_each(|(j,slice_x)| {
            &slice_x.iter_mut().enumerate().for_each(|(i,map_v)| {
                let baspar_ind = if i< j {(j+1)*j/2+i} else {(i+1)*i/2+j};
                *map_v = baspar_ind;
                baspar2basbas[baspar_ind] = [i,j];
            });
        });
        time_records.count("prim ri");

        time_records.new_item("ri_v","for RI dot aux_v^{-1/2}");
        time_records.count_start("ri_v");

        let mut ri3fn = MatrixFull::new([n_baspar,n_auxbas],0.0);
        _dgemm_full(&tmp_ri3fn, 'N', &aux_v, 'N', &mut ri3fn, 1.0, 0.0);
        time_records.count("ri_v");


        time_records.count("all ri");


        if self.ctrl.print_level>=2 {
            time_records.report_all();
        }

        (ri3fn,basbas2baspar,baspar2basbas)

    }

    pub fn prepare_rimatr_for_ri_v_rayon(&self, omega: Option<f64>) -> (MatrixFull<f64>,MatrixFull<usize>,Vec<[usize;2]>) {
        let cderi = ri_jk::generate_rimatr_bare(self, omega);
        let (basbas2baspar, baspar2basbas) = ri_jk::generate_baspar(self.num_basis);
        (cderi, basbas2baspar, baspar2basbas)
    }

    /// Build `rimatr` on a pre-screened AO-pair row space.
    ///
    /// `keep` lists full packed AO-pair indices, ascending, and has to be a conservative superset
    /// of the rows the storage-level pruning of the same tensor retains. The returned tensor holds
    /// one row per entry of `keep`, in that order, so it is `keep.len()` rows tall instead of
    /// `num_basis(num_basis+1)/2`. `None` reproduces
    /// [`Molecule::prepare_rimatr_for_ri_v_rayon`].
    pub fn prepare_rimatr_for_ri_v_rayon_prescreened(
        &self,
        omega: Option<f64>,
        keep: Option<&[usize]>,
    ) -> (MatrixFull<f64>,MatrixFull<usize>,Vec<[usize;2]>) {
        let cderi = ri_jk::generate_rimatr_bare_on_cint_prescreened(
            &ri_jk::util::get_cint_mol(self),
            &ri_jk::util::get_cint_aux(self),
            self.ctrl.j2c_decomp,
            omega,
            keep,
        );
        let (basbas2baspar, baspar2basbas) = ri_jk::generate_baspar(self.num_basis);
        (cderi, basbas2baspar, baspar2basbas)
    }

    /// Make an auxiliary molecule from a molecule for calculation.
    /// The auxmol is very similar to the origin mole,
    /// except its basis-related infomation is cloned from auxbasis information of the original mole.
    pub fn make_auxmol_fake(&self) -> Molecule {
        // this code is copied from TYGao's code
        let mut auxmol = self.clone();
        auxmol.num_basis = auxmol.num_auxbas.clone();
        auxmol.fdqc_bas = auxmol.fdqc_aux_bas.clone();
        auxmol.cint_fdqc = auxmol.cint_aux_fdqc.clone();
        auxmol.cint_bas = auxmol.cint_aux_bas.clone();
        auxmol.cint_atm = auxmol.cint_aux_atm.clone();
        auxmol.cint_env = auxmol.cint_aux_env.clone();
        return auxmol;
    }

    /// Get the slice of AO basis for each atom.
    /// 
    /// Output is a vector of 4-element arrays, number of atoms in total,
    /// where each array contains:
    /// - `shl0`: start shell (number of shells)
    /// - `shl1`: end shell (number of shells)
    /// - `p0`: start AO (number of basis functions)
    /// - `p1`: end AO (number of basis functions)
    pub fn aoslice_by_atom(&self) -> Vec<[usize; 4]> {
        const ATOM_OF: usize = rest_libcint::ffi::cint_ffi::ATOM_OF as usize;

        let cint_data = self.initialize_cint(false);
        let cint_bas = self.cint_bas.clone();

        let ao_loc = cint_data.ao_loc();
        let natm = self.natm_all;
        let nbas = cint_bas.len();
        let mut aoslice = vec![[0; 4]; natm];

        // the following code should assume that atoms in `cint_bas` has been sorted by atom index
        let delimiter = (0..(nbas - 1))
            .into_iter()
            .filter(|&idx| cint_bas[idx + 1][ATOM_OF] != cint_bas[idx][ATOM_OF])
            .collect::<Vec<usize>>();
        if delimiter.len() != natm - 1 {
            unimplemented!("Missing basis in atoms. Currently it should be internal problem in program.");
        }
        let shl_idx = 0;
        for atm in 0..natm {
            let shl0 = if atm == 0 { 0 } else { delimiter[atm - 1] + 1 };
            let shl1 = if atm == natm - 1 { nbas } else { delimiter[atm] + 1 };
            let p0 = ao_loc[shl0];
            let p1 = ao_loc[shl1];
            aoslice[atm] = [shl0, shl1, p0, p1];
        }

        // todo: currently we have not consider missing basis in atom
        return aoslice;
    }
}

pub fn generate_ri3fn_from_rimatr(rimatr: &MatrixFull<f64>, basbas2baspar: &MatrixFull<usize>, baspar2basbas: &Vec<[usize;2]>) -> RIFull<f64> {
    let n_basis = basbas2baspar.size[0];
    let n_baspar = rimatr.size[0];
    let n_auxbas = rimatr.size[1];
    let mut ri3fn = RIFull::new([n_basis,n_basis,n_auxbas],0.0);
    for k in 0..n_auxbas {
        for j in 0..n_basis {
            for i in 0..n_basis {
                let ind_baspar = basbas2baspar[[i,j]];
                let val_matr = rimatr[[ind_baspar,k]];
                ri3fn[[i,j,k]] = val_matr;
                //if (val_full-val_matr).abs()>1E-8 {
                //    println!("wrong value: i = {:}, j = {:}, k = {:}, ijpair = {:}, {:16.8}, {:16.8}",
                //             i,j,k,ind_baspar,val_full, val_matr)
                //}
            }
        }
    };
    ri3fn
}


//pub fn basbas2baspar(index: [usize;2]) -> usize {
//    (index[1]+1)*index[1]/2 + index[0]
//}
//pub fn basbas2baspar_range(i_range:Range<usize>,j_range:Range<usize>) -> std::iter::Map<std::iter::Zip<std::ops::Range<usize>, std::ops::Range<usize>>, dyn Fn() -> usize> {
//    i_range.zip(j_range).map(|(i,j)| ((j+1)*j/2+i) as usize)
//    //(index[1]+1)*index[1]/2
//}

pub fn is_int(f: f64) -> bool {
    (f - f.round()).abs() < 1.0e-8
}

pub fn sanity_check_nelec(num_elec: f64, spin: f64) -> (i32, i32) {

    let num_elec_int: i32;
    let spin_int: i32;
    if !is_int(num_elec) {
        panic!("Error:: The total electron number cannot be rounded to an integer: {}.", num_elec);
    } else {
        num_elec_int = num_elec.round() as i32;
    };
    if !is_int(spin) {
        panic!("Error:: The spin multiplicity cannot be rounded to an integer: {}.", spin);
    } else {
        spin_int = spin.round() as i32;
    };
    // odd/even check
    if (num_elec_int - spin_int + 1) % 2 != 0 {
        panic!("Error:: The total number of electrons ({}) and the spin multiplicity ({}) do not match.", num_elec, spin);
    };
    (num_elec_int, spin_int)
}
