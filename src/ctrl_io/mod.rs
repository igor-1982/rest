use serde::{Deserialize,Serialize};
//use std::{fs, str::pattern::StrSearcher};
use std::fs;
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

pub enum SCFType {
    RHF,
    ROHF,
    UHF
}


#[derive(Debug,Clone)]
pub struct InputKeywords {
    pub print_level: usize,
    // Keywords for the (aux) basis sets
    pub basis_path: String,
    pub basis_type: String,
    pub auxbas_path: String,
    pub auxbas_type: String,
    pub use_auxbas: bool,
    pub use_auxbas_symm: bool,
    pub even_tempered_basis: bool,
    pub etb_start_atom_number: usize,
    pub etb_beta: f64,
    // Keywords for systems
    pub eri_type: String,
    pub xc: String,
    pub charge: f64,
    pub spin: f64,
    pub spin_channel: usize,
    pub spin_polarization: bool,
    pub frozen_core_postscf: i32,
    pub frequency_points: usize,
    pub freq_grid_type: usize,
    pub freq_cut_off: f64,
    // Keywords for DFT numerical integration
    pub radial_precision: f64,
    pub min_num_angular_points: usize,
    pub max_num_angular_points: usize,
    pub grid_gen_level: usize,
    pub hardness: usize,
    pub pruning: String,
    pub rad_grid_method: String,
    pub external_grids: String,
    // Keywords for the scf procedures
    pub mixer: String,
    pub mix_param: f64,
    pub num_max_diis: usize,
    pub start_diis_cycle: usize,
    pub max_scf_cycle: usize,
    pub scf_acc_rho: f64,
    pub scf_acc_eev: f64,
    pub scf_acc_etot:f64,
    pub chkfile: String,
    pub restart: bool,
    pub guessfile: String,
    pub external_init_guess: bool,
    pub initial_guess: String,
    pub noiter: bool,
    pub check_stab: bool,
    // Keywords for fciqmc dump
    pub fciqmc_dump: bool,
    // Kyewords for post scf analysis
    pub wfn_in_real_space: usize,
    pub cube: bool,
    pub molden: bool,
    // Keywords for parallism
    pub num_threads: Option<usize>
}

impl InputKeywords {
    pub fn new() -> InputKeywords {
        InputKeywords{
            // keywords for machine and debug info
            print_level: 0,
            num_threads: None,
            // Keywords for (aux)-basis sets
            basis_path: String::from("./STO-3G"),
            basis_type: String::from("spheric"),
            auxbas_path: String::from("./def2-SV(P)-JKFIT"),
            auxbas_type: String::from("spheric"),
            use_auxbas: true,
            use_auxbas_symm: true,
            // Keywords associated with the method employed
            xc: String::from("x3lyp"),
            eri_type: String::from("ri-v"),
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
            // Keywords for DFT numerical integration
            radial_precision: 1.0e-12,
            min_num_angular_points: 110,
            max_num_angular_points: 110,
            hardness: 3,
            grid_gen_level: 3,
            pruning: String::from("sg1"),
            rad_grid_method: String::from("treutler"),
            external_grids: "none".to_string(),
            // ETB for autogen the auxbasis
            even_tempered_basis: false,
            etb_start_atom_number: 37,
            etb_beta: 2.0,
            // Keywords for the scf procedures
            chkfile: String::from("none"),
            guessfile: String::from("none"),
            mixer: String::from("direct"),
            mix_param: 1.0,
            num_max_diis: 2,
            start_diis_cycle: 2,
            max_scf_cycle: 100,
            scf_acc_rho: 1.0e-6,
            scf_acc_eev: 1.0e-5,
            scf_acc_etot:1.0e-8,
            restart: false,
            external_init_guess: false,
            initial_guess: String::from("hcore"),
            noiter: false,
            check_stab: false,
            // Keywords for the fciqmc dump
            fciqmc_dump: false,
            // Kyewords for post scf
            wfn_in_real_space: 0,
            cube: false,
            molden: false,
            // Derived keywords of identifying the method used
            //use_dft: false,
            //dft_type: None,
        }
    }



    pub fn parse_ctl_from_json(tmp_keys: &serde_json::Value) -> anyhow::Result<(InputKeywords,GeomCell)> {
        //let tmp_cont = fs::read_to_string(&filename[..])?;
        //let tmp_keys: serde_json::Value = serde_json::from_str(&tmp_cont[..])?;
        let mut tmp_input = InputKeywords::new();
        let mut tmp_geomcell = GeomCell::new();

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
                println!("Print level:                {}", tmp_input.print_level);
                //let default_rayon_current_num_threads = rayon::current_num_threads();
                tmp_input.num_threads = match tmp_ctrl.get("num_threads").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value::String(tmp_str) => {Some(tmp_str.to_lowercase().parse().unwrap())},
                    serde_json::Value::Number(tmp_num) => {Some(tmp_num.as_i64().unwrap() as usize)},
                    other => {None},
                };
                if let Some(num_threads) = tmp_input.num_threads {
                    println!("The number of threads used for parallelism:      {}", num_threads);
                    rayon::ThreadPoolBuilder::new().num_threads(num_threads).build_global()?;
                    unsafe{utilities::openblas_set_num_threads(num_threads as i32)};
                } else {
                    unsafe{utilities::openblas_set_num_threads(rayon::current_num_threads() as i32)};
                    println!("The default rayon num_threads value is used:      {}", rayon::current_num_threads());
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
                            panic!("The specified folder for the basis sets is missing: (./)");
                        };
                        String::from("./")
                   }
                };
                tmp_input.basis_type = match tmp_ctrl.get("basis_type").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value::String(tmp_type) => {tmp_type.to_lowercase()},
                    other => {String::from("spheric")}
                };
                println!("The {}-GTO basis set is taken from {}", tmp_input.basis_type,tmp_input.basis_path);

                tmp_input.pruning = match tmp_ctrl.get("pruning").unwrap_or(&serde_json::Value::Null){
                    serde_json::Value::String(tmp_type) => {tmp_type.to_lowercase()},
                    other => {String::from("sg1")} //default prune method: sg1
                };
                println!("The pruning method will be {}", tmp_input.pruning);

                tmp_input.rad_grid_method = match tmp_ctrl.get("radial_grid_method").unwrap_or(&serde_json::Value::Null){
                    serde_json::Value::String(tmp_type) => {tmp_type.to_lowercase()},
                    other => {String::from("treutler")} //default prune method: sg1
                };
                println!("The radial grid generation method will be {}", tmp_input.rad_grid_method);

                tmp_input.eri_type = match tmp_ctrl.get("eri_type").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value::String(tmp_eri) => {
                        if tmp_eri.to_lowercase().eq("ri_v") || tmp_eri.to_lowercase().eq("ri-v")
                        {
                            String::from("ri_v")
                        } else {tmp_eri.to_lowercase()}
                    },
                    other => {String::from("analytic")},
                };
                if tmp_input.eri_type.eq(&String::from("ri_v"))
                {
                    tmp_input.use_auxbas = true
                } else {
                    tmp_input.use_auxbas = false
                };
                println!("ERI Type: {}", tmp_input.eri_type);

                tmp_input.use_auxbas_symm = match tmp_ctrl.get("use_auxbas_symm").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value::Bool(tmp_str) => {*tmp_str},
                    other => {false},
                };
                if tmp_input.use_auxbas_symm {
                    println!("Turn on the basis pair symmetry for RI 3D-tensors")
                } else {
                    println!("Turn off the basis pair symmetry for RI 3D-tensors")
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
                        println!("No auxiliary basis set is specified");
                        let default_bas = String::from("./");
                        if ! std::path::Path::new(&default_bas).is_dir() {
                            //tmp_input.use_auxbas = false;
                        } else {
                            //tmp_input.use_auxbas = true;
                        }
                        default_bas
                   }
                };
                if tmp_input.use_auxbas {
                    println!("The {}-GTO auxiliary basis set is taken from {}", tmp_input.auxbas_type,tmp_input.auxbas_path)
                };
                // ==============================================
                //  Keywords associated with the method employed
                // ==============================================
                tmp_input.xc = match tmp_ctrl.get("xc").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value::String(tmp_xc) => {tmp_xc.to_lowercase()},
                    other => {String::from("hf")},
                };
                println!("The exchange-correlation method: {}", tmp_input.xc);

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
                println!("Charge: {:3}; Spin: {:3}",tmp_input.charge,tmp_input.spin);
                tmp_input.spin_polarization = match tmp_ctrl.get("spin_polarization").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value:: String(tmp_str) => tmp_str.to_lowercase().parse().unwrap_or(false),
                    serde_json::Value:: Bool(tmp_bool) => tmp_bool.clone(),
                    other => false,
                };
                tmp_input.spin_channel = if tmp_input.spin_polarization {
                    println!("Spin polarization: On");
                    2_usize
                } else {
                    println!("Spin polarization: Off");
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
                println!("min_num_angular_points: {}", tmp_input.min_num_angular_points);
                println!("max_num_angular_points: {}", tmp_input.max_num_angular_points);
                println!("hardness: {}", tmp_input.hardness);

                tmp_input.external_grids = match tmp_ctrl.get("external_grids").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value::String(tmp_type) => {tmp_type.to_string()},
                    other => {String::from("grids")}
                };

                tmp_input.grid_gen_level = match tmp_ctrl.get("grid_generation_level").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(3_usize)},
                    serde_json::Value::Number(tmp_num) => {tmp_num.as_i64().unwrap_or(3) as usize},
                    other => {3_usize},
                };
                println!("Grid generation level: {}", tmp_input.grid_gen_level);

                tmp_input.even_tempered_basis = match tmp_ctrl.get("even_tempered_basis").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value::Bool(tmp_str) => {*tmp_str},
                    other => {false},
                };
                println!("Even tempered basis generation: {}", tmp_input.even_tempered_basis);

                tmp_input.etb_start_atom_number = match tmp_ctrl.get("etb_start_atom_number").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(37_usize)},
                    serde_json::Value::Number(tmp_num) => {tmp_num.as_i64().unwrap_or(37) as usize},
                    other => {37_usize},
                };

                tmp_input.etb_beta = match tmp_ctrl.get("etb_beta").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(2.0_f64)},
                    serde_json::Value::Number(tmp_num) => {tmp_num.as_i64().unwrap_or(2) as f64},
                    other => {2.0_f64},
                };

                if tmp_input.even_tempered_basis == true {
                    println!("Even tempered basis generation starts at: {}", tmp_input.etb_start_atom_number);
                    println!("Even tempered basis beta is: {}", tmp_input.etb_beta);

                }
                




                // ==============================================
                //  Keywords associated with the SCF procedure
                // ==============================================
                tmp_input.max_scf_cycle = match tmp_ctrl.get("max_scf_cycle").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(100_usize)},
                    serde_json::Value::Number(tmp_num) => {tmp_num.as_i64().unwrap_or(100) as usize},
                    other => {100_usize}
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
                println!("SCF convergency thresholds: {:e} for density matrix", tmp_input.scf_acc_rho);
                println!("                            {:e} Ha. for sum of eigenvalues", tmp_input.scf_acc_eev);
                println!("                            {:e} Ha. for total energy", tmp_input.scf_acc_etot);
                println!("Max. SCF cycle number:      {}", tmp_input.max_scf_cycle);

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

                tmp_input.guessfile = match tmp_ctrl.get("guessfile").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value::String(tmp_guess) => tmp_guess.to_lowercase().clone(),
                    other => String::from("none"),
                };
                tmp_input.external_init_guess = ! tmp_input.guessfile.to_lowercase().eq(&"none") &&
                            std::path::Path::new(&tmp_input.guessfile).exists();
                // if guessfile is specified, reading the external initial guess file is prior to reading the restart file
                if tmp_input.external_init_guess  {
                    println!("The initial guess will be imported from \n({}).\n ",&tmp_input.guessfile)
                } else if ! std::path::Path::new(&tmp_input.guessfile).exists() {
                    println!("WARNING: The specified external initial guess file (guessfile) is missing \n({}). \n The external initial guess will not be imported.\n",&tmp_input.guessfile)
                }

                tmp_input.chkfile = match tmp_ctrl.get("chkfile").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value::String(tmp_chk) => tmp_chk.to_lowercase().clone(),
                    other => String::from("none"),
                };
                tmp_input.restart = ! tmp_input.chkfile.to_lowercase().eq(&"none");

                if tmp_input.restart && ! std::path::Path::new(&tmp_input.chkfile).exists() {
                    println!("The specified checkfile is missing, which will be created after the SCF procedure \n({})",&tmp_input.chkfile)
                } else if tmp_input.restart && ! tmp_input.external_init_guess {
                    println!("The initial guess will be obtained from the existing checkfile \n({})",&tmp_input.chkfile)
                } else {
                    println!("The specified checkfile exists but is not loaded because of 'external_init_guess");
                    println!("It will be updated after the SCF procedure \n({})",&tmp_input.chkfile)
                    //println!("No existing checkfile for restart\n")
                };
                tmp_input.initial_guess = match tmp_ctrl.get("initial_guess").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase()},
                    other => {String::from("vsap")},
                };
                println!("Initial guess is prepared by ({}).", &tmp_input.initial_guess);
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
                // ================================================
                //  Keywords associated with the post-SCF analyais
                // ================================================
                tmp_input.wfn_in_real_space = match tmp_ctrl.get("wfn_in_real_space").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value::String(tmp_wfn) => {tmp_wfn.to_lowercase().parse().unwrap_or(0)},
                    serde_json::Value::Number(tmp_wfn) => {tmp_wfn.as_i64().unwrap_or(0) as usize},
                    other => {0_usize},
                };

                tmp_input.cube = match tmp_ctrl.get("cube").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value:: String(tmp_str) => tmp_str.to_lowercase().parse().unwrap_or(false),
                    serde_json::Value:: Bool(tmp_bool) => tmp_bool.clone(),
                    other => false,
                }; 

                tmp_input.molden = match tmp_ctrl.get("molden").unwrap_or(&serde_json::Value::Null) {
                    serde_json::Value:: String(tmp_str) => tmp_str.to_lowercase().parse().unwrap_or(false),
                    serde_json::Value:: Bool(tmp_bool) => tmp_bool.clone(),
                    other => false,
                }; 
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
                        println!("It is a cluster calculation for finite molecules");
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
