
use pyo3::pyclass;
use serde::{Deserialize,Serialize};
//use std::{fs, str::pattern::StrSearcher};
use std::{fs, sync::Arc};
use crate::{geom_io::{GeomCell,MOrC, GeomUnit}, dft::{DFAFamily, DFA4REST}, utilities};
use rayon::ThreadPoolBuilder;

use serde_json;
use toml;

//use crate::geom_io::{GeomCell,GeomCell,CodeSelect,MOrC, GeomUnit, RawGeomCell};

//#[derive(Serialize,Deserialize)]
//pub struct RawCtrl {
//    pub ctrl: Option<RawInputKeywords>,
//    pub geom: Option<RawGeomCell>
//}

mod pyrest_ctrl_io;

/// **InputKeywords** for a specific calculation
///  ### System dependent keywords
///  - `print_level`:  default (1). `0` dose not print anything. larger number with more output information  
/// 
///  ### Basis set keywords
///  - `basis_path`:   `String`. The path where you can find the basis-set file in json format. If the basis-set file is missing, REST will try to download it from BasisSetExchange
///  - `basis_type`:   `String`. It can be `spheric` or `cartesian`
///  - `auxbas_path`:  `String`. The path where you can find the auxiliary basis-set file in json format. If the basis-set file is missing, REST will try to download it from BasisSetExchange
///  - `auxbas_type`:  `String`. It can be `spheric` or `cartesian`
///  - `even_tempered-basis`: `Bool`. True: turn on ETB to generate the auxiliary basis set
///  - `etb_start_atom_number`: `Usize`. Use ETB, for the element with atomic index larger than this value  
///  - `etb_beta`: `f64`. Relevant to the ETB basis set size. Smaller value indicates larger ETB basis set. NOTE: etb_beta should be larger than 1.0
#[derive(Debug,Clone)]
#[pyclass]
pub struct InputKeywords {
    #[pyo3(get, set)]
    pub print_level: usize,
    // Keywords for the (aux) basis sets
    #[pyo3(get, set)]
    pub basis_path: String,
    #[pyo3(get, set)]
    pub basis_type: String,
    #[pyo3(get, set)]
    pub auxbas_path: String,
    #[pyo3(get, set)]
    pub auxbas_type: String,
    #[pyo3(get, set)]
    pub use_auxbas: bool,
    #[pyo3(get, set)]
    pub even_tempered_basis: bool,
    #[pyo3(get, set)]
    pub etb_start_atom_number: usize,
    #[pyo3(get, set)]
    pub etb_beta: f64,
    // Keywords for RI_K
    #[pyo3(get, set)]
    pub ri_k_only: bool,
    // Keywords for Gradient
    #[pyo3(get, set)]
    pub auxbasis_response: bool,
    // Keywords for IDSF
    #[pyo3(get, set)]
    pub use_isdf: bool,
    #[pyo3(get, set)]
    pub isdf_k_only: bool,
    #[pyo3(get, set)]
    pub isdf_k_mu: usize,
    // Keywords for systems
    #[pyo3(get, set)]
    pub isdf_new: bool,
    #[pyo3(get, set)]
    pub eri_type: String,
    #[pyo3(get, set)]
    pub use_ri_symm: bool,
    #[pyo3(get, set)]
    pub xc: String,
    pub post_xc: Vec<String>,
    pub post_correlation: Vec<DFAFamily>,
    pub post_ai_correction: String,
    pub charge: f64,
    #[pyo3(get, set)]
    pub spin: f64,
    #[pyo3(get, set)]
    pub spin_channel: usize,
    #[pyo3(get, set)]
    pub spin_polarization: bool,
    #[pyo3(get, set)]
    pub frozen_core_postscf: i32,
    #[pyo3(get, set)]
    pub frequency_points: usize,
    #[pyo3(get, set)]
    pub lambda_points: usize,
    #[pyo3(get, set)]
    pub freq_grid_type: usize,
    #[pyo3(get, set)]
    pub freq_cut_off: f64,
    // Keywords for DFT numerical integration
    #[pyo3(get, set)]
    pub radial_precision: f64,
    #[pyo3(get, set)]
    pub min_num_angular_points: usize,
    #[pyo3(get, set)]
    pub max_num_angular_points: usize,
    #[pyo3(get, set)]
    pub grid_gen_level: usize,
    #[pyo3(get, set)]
    pub hardness: usize,
    #[pyo3(get, set)]
    pub pruning: String,
    #[pyo3(get, set)]
    pub rad_grid_method: String,
    #[pyo3(get, set)]
    pub external_grids: String,
    // Keywords for the scf procedures
    #[pyo3(get, set)]
    pub mixer: String,
    #[pyo3(get, set)]
    pub mix_param: f64,
    #[pyo3(get, set)]
    pub num_max_diis: usize,
    #[pyo3(get, set)]
    pub start_diis_cycle: usize,
    #[pyo3(get, set)]
    pub start_check_oscillation: usize,
    #[pyo3(get, set)]
    pub level_shift: Option<f64>,
    #[pyo3(get, set)]
    pub max_scf_cycle: usize,
    #[pyo3(get, set)]
    pub scf_acc_rho: f64,
    #[pyo3(get, set)]
    pub scf_acc_eev: f64,
    #[pyo3(get, set)]
    pub scf_acc_etot:f64,
    #[pyo3(get, set)]
    pub restart: bool,
    #[pyo3(get, set)]
    pub chkfile: String,
    #[pyo3(get, set)]
    pub chkfile_type: String,
    #[pyo3(get, set)]
    pub guessfile: String,
    #[pyo3(get, set)]
    pub guessfile_type: String,
    #[pyo3(get, set)]
    pub external_init_guess: bool,
    #[pyo3(get, set)]
    pub initial_guess: String,
    #[pyo3(get, set)]
    pub noiter: bool,
    #[pyo3(get, set)]
    pub check_stab: bool,
    #[pyo3(get, set)]
    pub use_dm_only: bool,
    // Keywords for fciqmc dump
    #[pyo3(get, set)]
    pub fciqmc_dump: bool,
    // Kyewords for post scf analysis
    pub outputs: Vec<String>,
    pub cube_orb_setting: [f64;2],
    pub cube_orb_indices: Vec<[usize;2]>,
    //pub output_wfn_in_real_space: usize,
    //pub output_cube: bool,
    //pub output_molden: bool,
    //pub output_fchk: bool,
    // Keywords for sad initial guess
    #[pyo3(get, set)]
    pub atom_sad: bool,
    // Keywords for parallism
    #[pyo3(get, set)]
    pub num_threads: Option<usize>
}

impl InputKeywords {
    pub fn init_ctrl() -> InputKeywords {
        InputKeywords{
            // keywords for machine and debug info
            print_level: 0,
            num_threads: Some(1),
            // Keywords for (aux)-basis sets
            basis_path: String::from("./STO-3G"),
            basis_type: String::from("spheric"),
            auxbas_path: String::from("./def2-SV(P)-JKFIT"),
            auxbas_type: String::from("spheric"),
            use_auxbas: true,
            auxbasis_response: false,
            use_isdf: false,
            ri_k_only: false,
            isdf_k_only: false,
            isdf_k_mu: 17,
            isdf_new: false,
            // Keywords associated with the method employed
            xc: String::from("x3lyp"),
            post_xc: vec![],
            post_correlation: vec![],
            post_ai_correction: String::from("none"),
            eri_type: String::from("ri_v"),
            use_ri_symm: true,
            charge: 0.0_f64,
            spin: 1.0_f64,
            spin_channel: 1_usize,
            spin_polarization: false,
            // Keywords for frozen-core algorithms
            frozen_core_postscf: 0_i32,
            // Keywords for RPA frequence tabulation
            frequency_points: 20_usize,
            freq_grid_type: 0_usize,
            freq_cut_off: 10.0_f64,
            // Keywords for scsRPA lambda tabulation
            lambda_points: 20_usize,
            // Keywords for DFT numerical integration
            radial_precision: 1.0e-12,
            min_num_angular_points: 110,
            max_num_angular_points: 110,
            hardness: 3,
            grid_gen_level: 3,
            pruning: String::from("nwchem"),
            rad_grid_method: String::from("treutler"),
            external_grids: "none".to_string(),
            // ETB for autogen the auxbasis
            even_tempered_basis: false,
            etb_start_atom_number: 37,
            etb_beta: 2.0,
            // Keywords for the scf procedures
            chkfile: String::from("none"),
            chkfile_type: String::from("hdf5"),
            guessfile: String::from("none"),
            guessfile_type: String::from("hdf5"),
            mixer: String::from("diis"),
            mix_param: 0.6,
            num_max_diis: 8,
            start_diis_cycle: 1,
            start_check_oscillation: 20,
            level_shift : None, 
            max_scf_cycle: 100,
            scf_acc_rho: 1.0e-6,
            scf_acc_eev: 1.0e-5,
            scf_acc_etot:1.0e-8,
            restart: false,
            external_init_guess: false,
            initial_guess: String::from("sad"),
            noiter: false,
            check_stab: false,
            // Kyewords for the manner to evaluate the Vk (and also Vxc) potentials
            // True:  using only density matrix in the evaluation
            // False: use coefficients as well with higher efficiency
            use_dm_only: false,
            // Keywords for the fciqmc dump
            fciqmc_dump: false,
            // Kyewords for post scf
            outputs: vec![],
            cube_orb_setting: [3.0,80.0],
            cube_orb_indices: vec![],
            //output_wfn_in_real_space: 0,
            //output_cube: false,
            //output_molden: false,
            //output_fchk: false,
            // Keyword to turn on atom calculations for the SAD initial guess
            atom_sad: false,
            // Derived keywords of identifying the method used
            //use_dft: false,
            //dft_type: None,
        }
    }



    pub fn parse_ctl_from_json(tmp_keys: &serde_json::Value) -> anyhow::Result<(InputKeywords,GeomCell)> {
        //let tmp_cont = fs::read_to_string(&filename[..])?;
        //let tmp_keys: serde_json::Value = serde_json::from_str(&tmp_cont[..])?;
        let mut tmp_input = InputKeywords::init_ctrl();
        let mut tmp_geomcell = GeomCell::init_geom();

        //==================================================================
        //
        //  parse the keywords from the "ctrl" block
        //
        //==================================================================
        match tmp_keys.get("ctrl").unwrap_or(&serde_json::Value::Null) {
            serde_json::Value::Object(tmp_ctrl) => {
                // =====================================
                //  Keywords for machine info and debug 
                // =====================================
                tmp_input.print_level = match tmp_ctrl.get("print_level").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(1_usize)},
                    serde_json::Value::Number(tmp_num) => {tmp_num.as_i64().unwrap_or(1) as usize},
                    other => {1_usize},
                };
                if tmp_input.print_level>0 {println!("Print level:                {}", tmp_input.print_level)};
                //let default_rayon_current_num_threads = rayon::current_num_threads();
                tmp_input.num_threads = match tmp_ctrl.get("num_threads").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value::String(tmp_str) => {Some(tmp_str.to_lowercase().parse().unwrap_or(1))},
                    serde_json::Value::Number(tmp_num) => {Some(tmp_num.as_i64().unwrap_or(1) as usize)},
                    other => {Some(1)},
                };
                if let Some(num_threads) = tmp_input.num_threads {
                    if tmp_input.print_level>0 {println!("The number of threads used for parallelism:      {}", num_threads)};
                    rayon::ThreadPoolBuilder::new().num_threads(num_threads).build_global()?;
                    utilities::omp_set_num_threads_wrapper(num_threads);
                } else {
                    utilities::omp_set_num_threads_wrapper(rayon::current_num_threads());
                    if tmp_input.print_level>0 {println!("The default rayon num_threads value is used:      {}", rayon::current_num_threads())};
                };
                //println!("max_num_threads: {}, current_num_threads: {}", rayon::max_num_threads(), rayon::current_num_threads());
                // ====================================
                //  Keywords for the (aux) basis sets
                // ====================================
                tmp_input.basis_path = match tmp_ctrl.get("basis_path").unwrap_or(&serde_json::Value::Null) {
                   serde_json::Value::String(tmp_bas) => {
                        if ! std::path::Path::new(tmp_bas).is_dir() {
                            println!("The specified folder for the basis sets is missing: ({})", tmp_bas);
                            println!("REST trys to fetch the basis sets from the basis-set exchange pool (https://www.basissetexchange.org/)");
                        };
                        tmp_bas.clone()
                   },
                   other => {
                        if ! std::path::Path::new(&String::from("./")).is_dir() {
                            panic!("The specified folder for the basis sets is missing. REST trys to find the basis set from the current folder: (./)");
                        };
                        String::from("./")
                   }
                };
                tmp_input.basis_type = match tmp_ctrl.get("basis_type").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value::String(tmp_type) => {tmp_type.to_lowercase()},
                    other => {String::from("spheric")}
                };
                if tmp_input.print_level> 0 {println!("The {}-GTO basis set is taken from {}", tmp_input.basis_type,tmp_input.basis_path)};

                tmp_input.pruning = match tmp_ctrl.get("pruning").unwrap_or(&serde_json::Value::Null){
                    serde_json::Value::String(tmp_type) => {tmp_type.to_lowercase()},
                    other => {String::from("sg1")} //default prune method: sg1
                };
                if tmp_input.print_level>0 {println!("The pruning method will be {}", tmp_input.pruning)};

                tmp_input.rad_grid_method = match tmp_ctrl.get("radial_grid_method").unwrap_or(&serde_json::Value::Null){
                    serde_json::Value::String(tmp_type) => {tmp_type.to_lowercase()},
                    other => {String::from("treutler")} //default prune method: sg1
                };
                if tmp_input.print_level>0 {println!("The radial grid generation method will be {}", tmp_input.rad_grid_method)};

                tmp_input.eri_type = match tmp_ctrl.get("eri_type").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value::String(tmp_eri) => {
                        if tmp_eri.to_lowercase().eq("ri_v") || tmp_eri.to_lowercase().eq("ri-v")
                        {
                            String::from("ri_v")
                        } else {tmp_eri.to_lowercase()}
                    },
                    other => {String::from("ri_v")},
                    other => {String::from("analytic")},
                    other => {String::from("ri_v")},
                };
                let mut eri_type = tmp_input.eri_type.clone();
                if eri_type.eq(&String::from("ri_v"))
                {
                    tmp_input.use_auxbas = true;
                    tmp_input.use_isdf = false;
                    tmp_input.ri_k_only = false;
                } else if eri_type.eq(&String::from("ri_k")) {
                    tmp_input.eri_type = String::from("ri_v");
                    tmp_input.use_auxbas = true;
                    tmp_input.use_isdf = false;
                    tmp_input.ri_k_only = true;
                } else if eri_type.eq(&String::from("isdf_full")) {
                    // =========== for debug use by IGOR =================
                    tmp_input.eri_type = String::from("ri_v");
                    //====================================================
                    tmp_input.use_auxbas = true;
                    tmp_input.use_isdf = true;
                }else if eri_type.eq(&String::from("isdf_k_new")){
                        tmp_input.use_auxbas = true;
                        tmp_input.use_isdf = true;
                        tmp_input.isdf_k_only = true;
                        tmp_input.eri_type = String::from("ri_v");
                        tmp_input.isdf_new = true;
                }else if  eri_type.eq(&String::from("isdf_k")){
                        tmp_input.use_auxbas = true;
                        tmp_input.use_isdf = true;
                        tmp_input.isdf_k_only = true;
                        tmp_input.eri_type = String::from("ri_v");
                        tmp_input.isdf_new = false;
                        //println!("Initial use_isdf: {}", tmp_input.use_isdf);
                }else {
                    tmp_input.use_auxbas = false;
                    tmp_input.use_isdf = false;
                };
                if tmp_input.print_level>0 {println!("ERI Type: {}", tmp_input.eri_type)};

                tmp_input.use_ri_symm = match tmp_ctrl.get("use_ri_symm").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value::Bool(tmp_str) => {*tmp_str},
                    other => {false},
                };
                if tmp_input.print_level>0 {
                    if tmp_input.use_ri_symm {
                        println!("Turn on the basis pair symmetry for RI 3D-tensors")
                    } else {
                        println!("Turn off the basis pair symmetry for RI 3D-tensors")
                    };
                }
                tmp_input.isdf_k_mu = match tmp_ctrl.get("isdf_k_mu").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(8_usize)},
                    serde_json::Value::Number(tmp_num) => {tmp_num.as_i64().unwrap_or(8) as usize},
                    other => {8_usize},
                };            

                tmp_input.auxbas_type = match tmp_ctrl.get("auxbas_type").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value::String(tmp_type) => {tmp_type.to_lowercase()},
                    other => {String::from("spheric")}
                };
                tmp_input.auxbas_path = match tmp_ctrl.get("auxbas_path").unwrap_or(&serde_json::Value::Null) {
                   serde_json::Value::String(tmp_bas) => {
                        if ! std::path::Path::new(tmp_bas).is_dir() {
                            println!("The specified folder for the auxiliar basis sets is missing: ({})", tmp_bas);
                            //tmp_input.use_auxbas = false;
                        }
                        //tmp_input.use_auxbas = true;
                        tmp_bas.clone()
                   },
                   other => {
                        //if ! std::path::Path::new(&String::from("./")).is_dir() {
                        //    println!("The specified folder for the auxiliar basis sets is missing: (./)");
                        //};
                        println!("No auxiliary basis set is specified. REST will try to find the auxiliary basis fromt the current folder: (./)");
                        let default_bas = String::from("./");
                        if ! std::path::Path::new(&default_bas).is_dir() {
                            //tmp_input.use_auxbas = false;
                        } else {
                            //tmp_input.use_auxbas = true;
                        }
                        default_bas
                   }
                };
                if tmp_input.use_auxbas && tmp_input.print_level>0 {
                    println!("The {}-GTO auxiliary basis set is taken from {}", tmp_input.auxbas_type,tmp_input.auxbas_path)
                };
                // ===============================================
                //  Keywords for Gradient calculation
                // ==============================================
                tmp_input.auxbasis_response = match tmp_ctrl.get("auxbasis_response").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value::Bool(tmp_str) => {*tmp_str},
                    other => {false},
                };
                // ==============================================
                //  Keywords associated with the method employed
                // ==============================================
                tmp_input.xc = match tmp_ctrl.get("xc").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value::String(tmp_xc) => {tmp_xc.to_lowercase()},
                    other => {String::from("hf")},
                };
                if tmp_input.print_level>0 {println!("The exchange-correlation method: {}", tmp_input.xc)};
                //let re0 = Regex::new(r"
                //                    (?P<elem>\w{1,2})\s*,?    # the element
                //                    \s+
                //                    (?P<x>[\+-]?\d+.\d+)\s*,? # the 'x' position
                //                    \s+
                //                    (?P<y>[\+-]?\d+.\d+)\s*,? # the 'y' position
                //                    \s+
                //                    (?P<z>[\+-]?\d+.\d+)\s*,? # the 'z' position
                //                    \s*").unwrap();
                tmp_input.post_xc = match tmp_ctrl.get("post_xc").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value::String(tmp_xc) => {vec![tmp_xc.to_lowercase()]},
                    serde_json::Value::Array(tmp_xc) => {
                        let mut tmp_vec:Vec<String> = vec![];
                        tmp_xc.iter().for_each(|x| {
                            let xc_method = x.to_string();
                            let string_len = xc_method.len();
                            tmp_vec.push(xc_method[1..string_len-1].to_string())
                        });
                        tmp_vec
                    },
                    other => {vec![]},
                };
                let post_corr = match tmp_ctrl.get("post_correlation").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value::String(tmp_xc) => {vec![tmp_xc.to_lowercase()]},
                    serde_json::Value::Array(tmp_xc) => {
                        let mut tmp_vec:Vec<String> = vec![];
                        tmp_xc.iter().for_each(|x| {
                            let xc_method = x.to_string();
                            let string_len = xc_method.len();
                            tmp_vec.push(xc_method[1..string_len-1].to_string())
                        });
                        tmp_vec
                    },
                    other => {vec![]},
                };
                tmp_input.post_correlation = vec![];
                post_corr.iter().for_each(|corr| {
                    if corr.to_lowercase().eq("pt2") {
                        tmp_input.post_correlation.push(DFAFamily::PT2)
                    } else if corr.to_lowercase().eq("sbge2") {
                        tmp_input.post_correlation.push(DFAFamily::SBGE2)
                    } else if corr.to_lowercase().eq("rpa") {
                        tmp_input.post_correlation.push(DFAFamily::RPA)
                    } else if corr.to_lowercase().eq("scsrpa") {
                        tmp_input.post_correlation.push(DFAFamily::SCSRPA)
                    } else {
                        println!("Unknown post-scf correlation method: {}", corr)
                    }
                    //if corr.to_lowercase().eq(&pt2) 
                });
                tmp_input.post_ai_correction = match tmp_ctrl.get("post_ai_correction").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value::String(tmp_xc) => {tmp_xc.to_lowercase()},
                    other => {String::from("none")},
                };
                //let post_corr  = if post_ai_corr.eq("scc23") && tmp_input.xc.eq("r-xdh7") {
                //   vec!["sbge2"] 
                //} else {
                //    vec![]
                //};

                //tmp_input.post_ai_correction = vec![];
                //post_corr.iter().for_each(|corr| {
                //    if corr.to_lowercase().eq("pt2") {
                //        tmp_input.post_ai_correction.push(DFAFamily::PT2)
                //    } else if corr.to_lowercase().eq("sbge2") {
                //        tmp_input.post_ai_correction.push(DFAFamily::SBGE2)
                //    } else if corr.to_lowercase().eq("rpa") {
                //        tmp_input.post_ai_correction.push(DFAFamily::RPA)
                //    } else if corr.to_lowercase().eq("scsrpa") {
                //        tmp_input.post_ai_correction.push(DFAFamily::SCSRPA)
                //    } else {
                //        println!("Unknown post-scf correlation method: {}", corr)
                //    }
                //    //if corr.to_lowercase().eq(&pt2) 
                //});
                // ===============================================
                //  Keywords to determine the spin channel, which 
                //   is important to turn on RHF(RKS) or UHF(UKS)
                // ==============================================
                tmp_input.charge = match tmp_ctrl.get("charge").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value::String(tmp_charge) => {tmp_charge.to_lowercase().parse().unwrap_or(0.0)},
                    serde_json::Value::Number(tmp_charge) => {tmp_charge.as_f64().unwrap_or(0.0)},
                    other => {0.0},
                };
                tmp_input.spin = match tmp_ctrl.get("spin").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value::String(tmp_spin) => {tmp_spin.to_lowercase().parse().unwrap_or(0.0)},
                    serde_json::Value::Number(tmp_spin) => {tmp_spin.as_f64().unwrap_or(0.0)},
                    other => {0.0},
                };
                tmp_input.spin_polarization = match tmp_ctrl.get("spin_polarization").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value:: String(tmp_str) => tmp_str.to_lowercase().parse().unwrap_or(false),
                    serde_json::Value:: Bool(tmp_bool) => tmp_bool.clone(),
                    other => false,
                };
                tmp_input.spin_channel = if tmp_input.spin_polarization {
                    if tmp_input.print_level>0 {println!("Spin polarization: On")};
                    2_usize
                } else {
                    if tmp_input.print_level>0 {println!("Spin polarization: Off")};
                    1_usize
                };
                // ==============================================
                //  Keywords of setting the frozen-core algorithm
                // ==============================================
                tmp_input.frozen_core_postscf = match tmp_ctrl.get("frozen_core_postscf").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value::String(tmp_fc) => {tmp_fc.to_lowercase().parse().unwrap_or(0)},
                    serde_json::Value::Number(tmp_fc) => {tmp_fc.as_i64().unwrap_or(0) as i32},
                    other => {0},
                };
                // ==============================================
                //  Keywords of setting the frequency tabulation
                // ==============================================
                tmp_input.frequency_points = match tmp_ctrl.get("frequency_points").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value::String(tmp_fp) => {tmp_fp.to_lowercase().parse().unwrap_or(20_usize)},
                    serde_json::Value::Number(tmp_fp) => {tmp_fp.as_i64().unwrap_or(20) as usize},
                    other => {20_usize},
                };
                tmp_input.freq_grid_type = match tmp_ctrl.get("freq_grid_type").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value::String(tmp_fg) => {tmp_fg.to_lowercase().parse().unwrap_or(0)},
                    serde_json::Value::Number(tmp_fg) => {tmp_fg.as_i64().unwrap_or(0) as usize},
                    other => {0},
                };
                tmp_input.freq_cut_off = match tmp_ctrl.get("freq_cut_off").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value::String(tmp_fg) => {tmp_fg.to_lowercase().parse().unwrap_or(10.0)},
                    serde_json::Value::Number(tmp_fg) => {tmp_fg.as_f64().unwrap_or(10.0)},
                    other => {10.0},
                };
                tmp_input.lambda_points = match tmp_ctrl.get("lambda_points").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value::String(tmp_fp) => {tmp_fp.to_lowercase().parse().unwrap_or(20_usize)},
                    serde_json::Value::Number(tmp_fp) => {tmp_fp.as_i64().unwrap_or(20) as usize},
                    other => {20_usize},
                };
                //===============================================
                // Keywords for fciqmc dump
                //===============================================
                tmp_input.fciqmc_dump = match tmp_ctrl.get("fciqmc_dump").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value::Bool(tmp_bool) => {tmp_bool.clone()},
                    other => {false},
                };


                // ==============================================
                //  Keywords associated with DFT grids
                // ==============================================
                tmp_input.radial_precision = match tmp_ctrl.get("radial_precision").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(1.0e-12)},
                    serde_json::Value::Number(tmp_num) => {tmp_num.as_f64().unwrap_or(1.0e-12)},
                    other => {1.0e-12}
                };
                tmp_input.min_num_angular_points = match tmp_ctrl.get("min_num_angular_points").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(110_usize)},
                    serde_json::Value::Number(tmp_num) => {tmp_num.as_i64().unwrap_or(110) as usize},
                    other => {110_usize}
                };
                tmp_input.max_num_angular_points = match tmp_ctrl.get("max_num_angular_points").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(590_usize)},
                    serde_json::Value::Number(tmp_num) => {tmp_num.as_i64().unwrap_or(590) as usize},
                    other => {590_usize}
                };
                tmp_input.hardness = match tmp_ctrl.get("hardness").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(3_usize)},
                    serde_json::Value::Number(tmp_num) => {tmp_num.as_i64().unwrap_or(3) as usize},
                    other => {3_usize}
                };

                tmp_input.external_grids = match tmp_ctrl.get("external_grids").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value::String(tmp_type) => {
                        if tmp_input.print_level>0 {println!("Read grids from the external file: {}", tmp_type)};
                        tmp_type.to_string()},
                    other => {String::from("grids")}
                };

                tmp_input.grid_gen_level = match tmp_ctrl.get("grid_generation_level").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(3_usize)},
                    serde_json::Value::Number(tmp_num) => {tmp_num.as_i64().unwrap_or(3) as usize},
                    other => {3_usize},
                };

                tmp_input.even_tempered_basis = match tmp_ctrl.get("even_tempered_basis").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value::Bool(tmp_str) => {*tmp_str},
                    other => {false},
                };

                tmp_input.etb_start_atom_number = match tmp_ctrl.get("etb_start_atom_number").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(37_usize)},
                    serde_json::Value::Number(tmp_num) => {tmp_num.as_i64().unwrap_or(37) as usize},
                    other => {37_usize},
                };

                tmp_input.etb_beta = match tmp_ctrl.get("etb_beta").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(2.0_f64)},
                    serde_json::Value::Number(tmp_num) => {tmp_num.as_f64().unwrap_or(2.0_f64)},
                    other => {2.0_f64},
                };

                // ==============================================
                //  Keywords associated with the SCF procedure
                // ==============================================
                tmp_input.max_scf_cycle = match tmp_ctrl.get("max_scf_cycle").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(100_usize)},
                    serde_json::Value::Number(tmp_num) => {tmp_num.as_i64().unwrap_or(100) as usize},
                    other => {100_usize}
                };
                tmp_input.level_shift = match tmp_ctrl.get("level_shift").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value::String(tmp_str) => {
                        let num = tmp_str.to_lowercase().parse().unwrap_or(0.0);
                        if num == 0.0 {
                            None
                        } else {
                            Some(num)
                        }
                    },
                    serde_json::Value::Number(tmp_num) => {
                        let num = tmp_num.as_f64().unwrap_or(0.0);
                        if num == 0.0 {
                            None
                        } else {
                            Some(num)
                        }
                    },
                    other => {None}
                };
                tmp_input.scf_acc_rho = match tmp_ctrl.get("scf_acc_rho").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(1.0e-6)},
                    serde_json::Value::Number(tmp_num) => {tmp_num.as_f64().unwrap_or(1.0e-6)},
                    other => {1.0e-6}
                };
                tmp_input.scf_acc_eev = match tmp_ctrl.get("scf_acc_eev").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(1.0e-6)},
                    serde_json::Value::Number(tmp_num) => {tmp_num.as_f64().unwrap_or(1.0e-6)},
                    other => {1.0e-6}
                };
                tmp_input.scf_acc_etot = match tmp_ctrl.get("scf_acc_etot").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(1.0e-8)},
                    serde_json::Value::Number(tmp_num) => {tmp_num.as_f64().unwrap_or(1.0e-8)},
                    other => {1.0e-6}
                };

                tmp_input.mixer = match tmp_ctrl.get("mixer").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase()},
                    other => {String::from("direct")},
                };
                tmp_input.mix_param = match tmp_ctrl.get("mix_param").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(1.0)},
                    serde_json::Value::Number(tmp_num) => {tmp_num.as_f64().unwrap_or(1.0)},
                    other => {1.0}
                };
                tmp_input.num_max_diis = match tmp_ctrl.get("num_max_diis").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(2_usize)},
                    serde_json::Value::Number(tmp_num) => {tmp_num.as_i64().unwrap_or(2) as usize},
                    other => {2_usize}
                };
                tmp_input.start_diis_cycle = match tmp_ctrl.get("start_diis_cycle").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(2_usize)},
                    serde_json::Value::Number(tmp_num) => {tmp_num.as_i64().unwrap_or(2) as usize},
                    other => {2_usize}
                };
                tmp_input.start_check_oscillation = match tmp_ctrl.get("start_check_oscillation").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(20_usize)},
                    serde_json::Value::Number(tmp_num) => {tmp_num.as_i64().unwrap_or(20) as usize},
                    other => {20_usize}
                };

                // Initial guess relevant keywords
                tmp_input.guessfile = match tmp_ctrl.get("guessfile").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value::String(tmp_guess) => tmp_guess.clone(),
                    other => String::from("none"),
                };
                tmp_input.guessfile_type = match tmp_ctrl.get("guessfile_type").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value::String(tmp_guess) => tmp_guess.to_lowercase().clone(),
                    other => String::from("none"),
                };
                tmp_input.external_init_guess = ! tmp_input.guessfile.to_lowercase().eq(&"none") &&
                            std::path::Path::new(&tmp_input.guessfile).exists();

                tmp_input.chkfile = match tmp_ctrl.get("chkfile").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value::String(tmp_chk) => tmp_chk.clone(),
                    other => String::from("none"),
                };
                tmp_input.chkfile_type = match tmp_ctrl.get("chkfile_type").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value::String(tmp_chk) => tmp_chk.to_lowercase().clone(),
                    other => String::from("hdf5"),
                };

                tmp_input.restart = ! tmp_input.chkfile.to_lowercase().eq(&"none");
                //println!("debug: {:?}", &tmp_input.chkfile);

                tmp_input.initial_guess = match tmp_ctrl.get("initial_guess").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase()},
                    other => {String::from("vsap")},
                };


                tmp_input.noiter = match tmp_ctrl.get("noiter").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value:: String(tmp_str) => tmp_str.to_lowercase().parse().unwrap_or(false),
                    serde_json::Value:: Bool(tmp_bool) => tmp_bool.clone(),
                    other => false,
                };
                tmp_input.check_stab = match tmp_ctrl.get("check_stab").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value:: String(tmp_str) => tmp_str.to_lowercase().parse().unwrap_or(false),
                    serde_json::Value:: Bool(tmp_bool) => tmp_bool.clone(),
                    other => false,
                };
                tmp_input.use_dm_only = match tmp_ctrl.get("use_dm_only").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value:: String(tmp_str) => tmp_str.to_lowercase().parse().unwrap_or(false),
                    serde_json::Value:: Bool(tmp_bool) => tmp_bool.clone(),
                    other => false,
                };
                // ================================================
                //  Keywords associated with the post-SCF analyais
                // ================================================
                tmp_input.outputs = match tmp_ctrl.get("outputs").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value::String(tmp_op) => {vec![tmp_op.to_lowercase()]},
                    serde_json::Value::Array(tmp_op) => {
                        let mut tmp_vec:Vec<String> = vec![];
                        tmp_op.iter().for_each(|x| {
                            let op_type = x.to_string();
                            let string_len = op_type.len();
                            tmp_vec.push(op_type[1..string_len-1].to_lowercase().to_string())
                        });
                        tmp_vec
                    },
                    other => {vec![]},
                };
                tmp_input.cube_orb_setting = match tmp_ctrl.get("cube_orb_setting").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value::Array(tmp_op) => {
                        let mut tmp_array = [0.0;2];
                        tmp_array.iter_mut().zip(tmp_op[0..2].iter()).for_each(|(to, from)| {
                            match from {
                                serde_json::Value::String(tmp_str) => {*to = tmp_str.parse().unwrap_or(0.0)},
                                serde_json::Value::Number(tmp_num) => {*to = tmp_num.as_f64().unwrap_or(0.0)},
                                other => {*to = 0.0},
                            }
                        });
                        tmp_array
                    },
                    other => {[3.0,80.0]},
                };
                tmp_input.cube_orb_indices = match tmp_ctrl.get("cube_orb_indices").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value::Array(tmp_op) => {
                        //let mut tmp_array = [0.0;2];
                        let mut tmp_indices = vec![[0;2];tmp_op.len()];
                        tmp_indices.iter_mut().zip(tmp_op.iter()).for_each(|(to, from)| {
                            let tmp_to = match from {
                                serde_json::Value::Array(tmp_opp) => {
                                    let mut tmp_array = [0_usize;2];
                                    tmp_array.iter_mut().zip(tmp_opp[0..2].iter()).for_each(|(to, from)| {
                                        match from {
                                            serde_json::Value::String(tmp_str) => {*to = tmp_str.parse().unwrap_or(0)},
                                            serde_json::Value::Number(tmp_num) => {*to = tmp_num.as_u64().unwrap_or(0) as usize},
                                            other => {*to = 0},
                                        }
                                    });
                                    Some(tmp_array)
                                },
                                other => {None},
                            };
                            if let Some(tmp_array) = tmp_to {
                                *to = tmp_array;
                            };
                        });
                        tmp_indices 
                    },
                    other => {vec![]},
                };
                //tmp_input.output_wfn_in_real_space = match tmp_ctrl.get("output_wfn_in_real_space").unwrap_or(&serde_json::Value::Null) {
                //    serde_json::Value::String(tmp_wfn) => {tmp_wfn.to_lowercase().parse().unwrap_or(0)},
                //    serde_json::Value::Number(tmp_wfn) => {tmp_wfn.as_i64().unwrap_or(0) as usize},
                //    other => {0_usize},
                //};

                //tmp_input.output_cube = match tmp_ctrl.get("output_cube").unwrap_or(&serde_json::Value::Null) {
                //    serde_json::Value:: String(tmp_str) => tmp_str.to_lowercase().parse().unwrap_or(false),
                //    serde_json::Value:: Bool(tmp_bool) => tmp_bool.clone(),
                //    other => false,
                //}; 

                //tmp_input.output_molden = match tmp_ctrl.get("output_molden").unwrap_or(&serde_json::Value::Null) {
                //    serde_json::Value:: String(tmp_str) => tmp_str.to_lowercase().parse().unwrap_or(false),
                //    serde_json::Value:: Bool(tmp_bool) => tmp_bool.clone(),
                //    other => false,
                //}; 
                //tmp_input.output_fchk = match tmp_ctrl.get("output_fchk").unwrap_or(&serde_json::Value::Null) {
                //    serde_json::Value:: String(tmp_str) => tmp_str.to_lowercase().parse().unwrap_or(false),
                //    serde_json::Value:: Bool(tmp_bool) => tmp_bool.clone(),
                //    other => false,
                //};

                if tmp_input.print_level>0 {
                    println!("Charge: {:3}; Spin: {:3}",tmp_input.charge,tmp_input.spin);
                    println!("min_num_angular_points: {}", tmp_input.min_num_angular_points);
                    println!("max_num_angular_points: {}", tmp_input.max_num_angular_points);
                    println!("hardness: {}", tmp_input.hardness);
                    println!("Grid generation level: {}", tmp_input.grid_gen_level);
                    println!("Even tempered basis generation: {}", tmp_input.even_tempered_basis);
                    if tmp_input.even_tempered_basis == true {
                        if tmp_input.etb_beta<=1.0f64 {
                            println!("WARNING: etb_beta cannot be below 1.0. REST will use etb_beta=2.0 instead in this calculation");
                            tmp_input.etb_beta=2.0f64;
                        }
                        println!("Even tempered basis generation starts at: {}", tmp_input.etb_start_atom_number);
                        println!("Even tempered basis beta is: {}", tmp_input.etb_beta);
                    }
                    println!("SCF convergency thresholds: {:e} for density matrix", tmp_input.scf_acc_rho);
                    println!("                            {:e} Ha. for sum of eigenvalues", tmp_input.scf_acc_eev);
                    println!("                            {:e} Ha. for total energy", tmp_input.scf_acc_etot);
                    println!("Max. SCF cycle number:      {}", tmp_input.max_scf_cycle);
                    let tmp_mixer = tmp_input.mixer.clone();
                    if tmp_mixer.eq(&"direct") {
                        println!("No charge density mixing is employed for the SCF procedure");
                    } else if tmp_mixer.eq(&"linear") {
                        println!("The {} mixing is employed with the mixing parameter of {} for the SCF procedure", 
                                  &tmp_mixer, &tmp_input.mix_param);
                    } else if tmp_mixer.eq(&"ddiis") 
                           || tmp_mixer.eq(&"diis") {
                        println!("The {} mixing with (param, max_vec_len) = ({}, {}) is employed for the SCF procedure", 
                                  &tmp_mixer, &tmp_input.mix_param, &tmp_input.num_max_diis);
                        println!("Turn on the {} mixing after {} step(s) of SCF iteractions with the linear mixing", 
                                  &tmp_mixer, &tmp_input.start_diis_cycle);
                    } else {
                        tmp_input.mixer = String::from("direct");
                        println!("Unknown charge density mixer ({})! No charge density mixing will be invoked.", tmp_input.mixer);
                    };
                    // if guessfile is specified, reading the external initial guess file is prior to reading the restart file
                    if tmp_input.external_init_guess  {
                        println!("The initial guess will be imported from \n({}).\n ",&tmp_input.guessfile)
                    } else if ! std::path::Path::new(&tmp_input.guessfile).exists() {
                        println!("WARNING: The specified external initial guess file (guessfile) is missing \n({}). \n The external initial guess will not be imported.\n",&tmp_input.guessfile)
                    }
                    if tmp_input.restart && ! std::path::Path::new(&tmp_input.chkfile).exists() {
                        println!("The specified checkfile is missing, which will be created after the SCF procedure \n({})",&tmp_input.chkfile)
                    } else if tmp_input.restart && ! tmp_input.external_init_guess {
                        println!("The initial guess will be obtained from the existing checkfile \n({})",&tmp_input.chkfile)
                    } else {
                        println!("The specified checkfile exists but is not loaded because the keyword 'external_init_guess' is specified");
                        println!("It will be updated after the SCF procedure \n({})",&tmp_input.chkfile)
                        //println!("No existing checkfile for restart\n")
                    };
                    println!("Initial guess is prepared by ({}).", &tmp_input.initial_guess);
                }
            },
            other => {
                panic!("Error:: no 'ctrl' keyword or some inproper settings of the 'ctrl' keyword in the input file")
            },
        }
        //==================================================================
        //
        //  parse the keywords from the "geom" block
        //
        //==================================================================
        match tmp_keys.get("geom").unwrap_or(&serde_json::Value::Null) {
            serde_json::Value::Object(tmp_geom) => {
                tmp_geomcell.name = match tmp_geom.get("name").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value::String(tmp_str) => {tmp_str.clone()},
                    other => {String::from("none")},
                };
                let tmp_unit = match tmp_geom.get("unit").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase()},
                    other => {String::from("angstrom")},
                };
                if tmp_unit.to_lowercase()==String::from("angstrom") {
                    tmp_geomcell.unit=GeomUnit::Angstrom;
                } else if tmp_unit.to_lowercase()==String::from("bohr") {
                    tmp_geomcell.unit=GeomUnit::Bohr
                } else {
                    println!("Warning:: unknown geometry unit is specified: {}. Angstrom will be used", tmp_unit);
                    tmp_geomcell.unit=GeomUnit::Angstrom;
                };
                //(tmp_geomcell.elem, tmp_geomcell.fix, tmp_geomcell.position, tmp_geomcell.nfree, )
                match tmp_geom.get("position").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value::Array(tmp_vec) => {
                        let tmp_unit = tmp_geomcell.unit.clone();
                        
                        let (tmp1,tmp2,tmp3,tmp4) = GeomCell::parse_position(tmp_vec, &tmp_unit)?;

                        tmp_geomcell.elem = tmp1;
                        tmp_geomcell.fix = tmp2;
                        tmp_geomcell.position = tmp3;
                        tmp_geomcell.nfree = tmp4;
                    },
                    serde_json::Value::String(tmp_geom) => {
                        let tmp_unit = tmp_geomcell.unit.clone();
                        
                        let (tmp1,tmp2,tmp3,tmp4) = GeomCell::parse_position_from_string(tmp_geom, &tmp_unit)?;

                        tmp_geomcell.elem = tmp1;
                        tmp_geomcell.fix = tmp2;
                        tmp_geomcell.position = tmp3;
                        tmp_geomcell.nfree = tmp4;

                    }
                    other => {
                        panic!("Error in reading the geometry position")
                    }
                };
                match tmp_geom.get("lattice").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value::Array(tmp_vec) => {
                        let tmp_unit = tmp_geomcell.unit.clone();
                        tmp_geomcell.lattice = GeomCell::parse_lattice(tmp_vec, &tmp_unit)?;
                        tmp_geomcell.pbc = MOrC::Crystal;
                        panic!("Find lattice vectors. PBC calculations should be turn on, which, however, is not yet implemented");
                    },
                    other => {
                        if tmp_input.print_level>0 {
                            println!("It is a cluster calculation for finite molecules");
                        }
                        tmp_geomcell.pbc = MOrC::Molecule;
                    }
                }
            },
            other => {
                panic!("Error:: no 'geom' keyword or some inproper settings of 'geom' keyword in the input file");
            },
        }
        Ok((tmp_input,tmp_geomcell))
        
    }

    pub fn parse_ctl(filename: String) -> anyhow::Result<(InputKeywords,GeomCell)> {
        let tmp_cont = fs::read_to_string(&filename[..])?;
        let tmp_keys = if let Ok(tmp_json) = serde_json::from_str::<serde_json::Value>(&tmp_cont[..]) {
            // input file in the json format
            tmp_json
        } else {
            // input file in the toml format
            toml::from_str::<serde_json::Value>(&tmp_cont[..])?
        };

        InputKeywords::parse_ctl_from_json(&tmp_keys)
    }
}
