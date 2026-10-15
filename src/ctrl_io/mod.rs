#![warn(unused_imports)]
use pyo3::pyclass;
use serde::{Deserialize,Serialize};
use core::panic;
//use std::{fs, str::pattern::StrSearcher};
use std::{fs};
use crate::ctrl_io::geometric_pyo3_io::parse_geometric_keywords;
use crate::ctrl_io::quasiparticle_methods::parse_quasiparticle_keywords;
use crate::ri_jk::decompose::J2CDecompOption;
use crate::analdrv::config::{AnalDrvConfig, AnalDrvTask, parse_analdrv_keywords};
use crate::ctrl_io::tddft_parameters::parse_tddft_keywords;
use crate::ctrl_io::hessian_parameters::parse_hessian_keywords;
use crate::ctrl_io::thermo_parameters::parse_thermo_keywords;
use crate::{scf_io::force_state_occupation::ForceStateOccupation};
use crate::scf_io::smear::SmearingType;
use crate::dft::{DFAFamily, DFTType};
use crate::geom_io::{GeomCell, MOrC, parse_geom_keywords};
use crate::utilities;
// use rayon::ThreadPoolBuilder;
use crate::scf_io::occupation::OCCType;
use tensors::matrix_blas_lapack::omp_set_num_threads_global_wrapper;
use crate::solvent::surface_utils::SmdCavityRadii;
use crate::solvent::{PcmMethod, RadiusScheme};
use crate::x2c::RelativisticMethod;
use serde_json;
use toml;

pub mod ri_jk_io;
pub mod ri_pt2_io;
pub use ri_jk_io::*;
pub use ri_pt2_io::*;
pub mod output;
pub use output::*;

mod pyrest_ctrl_io;
mod geometric_pyo3_io;
mod solvent;
pub mod quasiparticle_methods;
pub mod tddft_parameters;
pub mod hessian_parameters;
pub mod thermo_parameters;
use geometric_pyo3_io::GeomeTRIC;
pub mod path_util;
use quasiparticle_methods::QuasiParticle;
use tddft_parameters::TDDFTParameters;
use hessian_parameters::HessianParameters;
use thermo_parameters::ThermoParameters;
use std::io::Write;
use log::{info, debug, warn, Level};
use env_logger::{Builder, Target};
use utilities::log::printlevel2loglevel;

pub fn parse_ctl(filename: String) -> anyhow::Result<(InputKeywords,GeomCell)> {
    let tmp_cont = fs::read_to_string(&filename[..])?;
    let tmp_keys = if let Ok(tmp_json) = serde_json::from_str::<serde_json::Value>(&tmp_cont[..]) {
        // input file in the json format
        tmp_json
    } else {
        // input file in the toml format
        toml::from_str::<serde_json::Value>(&tmp_cont[..])?
    };
    let (input, mut geomcell) = parse_ctl_from_json(&tmp_keys)?;
    let input_dir = std::path::Path::new(&filename).parent().unwrap_or(std::path::Path::new("."));
    geomcell.resolve_ghost_ep_paths(input_dir);
    Ok((input, geomcell))
}

pub fn parse_ctl_from_json(tmp_keys: &serde_json::Value) -> anyhow::Result<(InputKeywords,GeomCell)> {
    check_top_level_blocks(tmp_keys)?;
    let mut tmp_input = parse_ctrl_keywords(tmp_keys)?;
    let mut tmp_geomcell = parse_geom_keywords(tmp_keys)?;
    let mut tmp_geomtric = parse_geometric_keywords(tmp_keys)?;
    let mut tmp_quasiparticle=parse_quasiparticle_keywords(tmp_keys)?;
    let mut tmp_tddft = parse_tddft_keywords(tmp_keys)?;
    let mut tmp_hessian = parse_hessian_keywords(tmp_keys)?;
    let mut tmp_thermo = parse_thermo_keywords(tmp_keys)?;
    if let Some(tmp_geomtric) = &mut tmp_geomtric {
        tmp_input.geometric_pyo3 = Some(std::mem::take(tmp_geomtric));
    }
    if let Some(tmp_quasiparticle) = &mut tmp_quasiparticle {
        tmp_input.quasiparticle_methods = Some(std::mem::take(tmp_quasiparticle));
    }
    if let Some(tmp_tddft) = &mut tmp_tddft {
        tmp_input.tddft = Some(std::mem::take(tmp_tddft));
    }
    if let Some(tmp_hessian) = &mut tmp_hessian {
        tmp_input.hessian = Some(std::mem::take(tmp_hessian));
    }
    if let Some(tmp_thermo) = &mut tmp_thermo {
        tmp_input.thermo = Some(std::mem::take(tmp_thermo));
    }
    tmp_input.analdrv = parse_analdrv_keywords(tmp_keys);
    tmp_input.output = parse_output_keywords(tmp_keys);
    resolve_cube_keywords(&mut tmp_input, tmp_keys);
    Ok((tmp_input,tmp_geomcell))
}

const VALID_TOP_LEVEL_BLOCKS: &[&str] = &[
    "ctrl",
    "geom",
    "output",
    "md",
    "tddft",
    "hessian",
    "thermo",
    "analdrv",
    "geometric_pyo3",
    "quasiparticle_methods",
];

fn check_top_level_blocks(tmp_keys: &serde_json::Value) -> anyhow::Result<()> {
    if let serde_json::Value::Object(top_level) = tmp_keys {
        let unknown: Vec<&str> = top_level
            .keys()
            .map(|k| k.as_str())
            .filter(|k| !VALID_TOP_LEVEL_BLOCKS.contains(k))
            .collect();
        if !unknown.is_empty() {
            anyhow::bail!(
                "Unknown input block(s): [{}]. Valid top-level blocks are: {}.",
                unknown.join("], ["),
                VALID_TOP_LEVEL_BLOCKS.join(", ")
            );
        }
    }
    Ok(())
}

fn parse_usize_list_keyword(value: &serde_json::Value) -> Vec<usize> {
    match value {
        serde_json::Value::Array(items) => items
            .iter()
            .filter_map(|item| item.as_u64().map(|x| x as usize))
            .collect(),
        serde_json::Value::Number(num) => {
            num.as_u64().map(|x| vec![x as usize]).unwrap_or_default()
        }
        serde_json::Value::String(text) => text
            .trim_matches(|c| c == '[' || c == ']')
            .split(|c: char| c == ',' || c.is_whitespace())
            .filter_map(|part| {
                let part = part.trim();
                if part.is_empty() {
                    None
                } else {
                    part.parse::<usize>().ok()
                }
            })
            .collect(),
        _ => Vec::new(),
    }
}

#[derive(Clone,Copy,Debug, Deserialize, Serialize)]
pub enum JobType {
    SinglePoint,
    Force,
    NumDipole,
    GeomOpt,
    NormalModes,
    MD,
}

/// Whether to use the distributed (ScaLAPACK) solver for diagonalizing the
/// Hamiltonian in MPI runs.
///
/// - `Auto`: use the distributed solver only when the problem is large enough
///   (see `diagonalize_hamiltonian_distributed_check`).
/// - `On`: always use the distributed solver (mainly for testing small systems).
/// - `Off`: never use the distributed solver.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum HamiltonianDistributedMode {
    #[default]
    Auto,
    #[serde(alias = "true", alias = "yes")]
    On,
    #[serde(alias = "false", alias = "no")]
    Off,
}

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
#[derive(Debug,Clone,Serialize, Deserialize)]
#[pyclass]
pub struct InputKeywords {
    pub job_type: JobType,
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
    // Keywords for gradient, evaluate small contribution from auxiliary basis perturbation
    #[pyo3(get, set)]
    pub auxbasis_response: bool,
    // Keywords for gradient, whether evaluate from numerical or analytical derivative
    #[pyo3(get, set)]
    pub numerical_force: bool,
    // Keywords for IDSF
    #[pyo3(get, set)]
    pub use_isdf: bool,
    /// ISDF interpolation-points multiplier k; `None` uses the Rust/PyO3 estimator.
    #[pyo3(get, set)]
    pub isdf_k: Option<usize>,
    #[pyo3(get, set)]
    pub isdf_k_only: bool,
    #[pyo3(get, set)]
    pub isdf_k_mu: usize,
    // Keywords for systems
    #[pyo3(get, set)]
    pub isdf_new: bool,
    /// ISDF interpolation-point generation scheme: "udd" (density-driven) or "cvt".
    #[pyo3(get, set)]
    pub isdf_type: String,
    /// Control input file name used by the ISDF k auto-estimation script.
    #[pyo3(get, set)]
    pub ctrl_file: String,
    #[pyo3(get, set)]
    pub eri_type: String,
    #[pyo3(get, set)]
    pub use_ri_symm: bool,
    #[pyo3(get, set)]
    pub xc: String,
    pub xc_type: DFTType,
    // == for the non_standard setting of DFA ==
    pub xc_namelist: Option<Vec<String>>,
    pub xc_paralist: Option<Vec<f64>>,
    pub dfa_hybrid_scf: Option<f64>,
    // =========================================
    // == for the non_standard setting of DFA ==
    pub xc_model: Option<String>,
    // =========================================
    pub post_xc: Vec<String>,
    pub post_correlation: Vec<DFAFamily>,
    pub post_ai_correction: String,
    pub charge: f64,
    #[pyo3(get, set)]
    pub spin: f64,
    #[pyo3(get, set)]
    pub use_int_nelec: bool,
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
    /// Becke radii adjustment used by the grid partitioning: `"becke"` keeps the raw Bragg radii
    /// (Becke 1988, REST historical behaviour), `"treutler"` uses their square roots
    /// (Treutler-Ahlrichs 1995, the PySCF/PyFock convention).
    #[pyo3(get, set)]
    pub radii_adjust: String,
    // Keywords for the scf procedures
    #[pyo3(get, set)]
    pub mixer: String,
    #[pyo3(get, set)]
    pub mix_param: f64,
    #[pyo3(get, set)]
    pub num_max_diis: usize,
    /// Truncation threshold of the pseudo-inverse used by the DIIS solver (Pulay
    /// coefficients are obtained from the Gram matrix of the error vectors). Larger
    /// values discard more of the near-singular directions of that matrix, which
    /// bounds the extrapolation coefficients.
    pub diis_pinv_threshold: f64,
    /// Whether the DIIS error vector is transformed to the orthonormal basis,
    /// i.e. S^(-1/2) (FDS - SDF) S^(-1/2). PySCF and PyFock use the plain AO-basis
    /// commutator; set this to false to reproduce that behaviour.
    pub diis_orthogonalize_error: bool,
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
    pub scf_conv_criteria: String,
    #[pyo3(get, set)]
    pub scf_acc_g: f64,
    #[pyo3(get, set)]
    pub has_chkfile: bool,
    #[pyo3(get, set)]
    // Keywords for solvent models
    pub solvent_enabled: bool,
    pub solvent_ri: bool,
    pub solv_epsilon: f64,
    pub solvent_model: PcmMethod,
    pub solv_chunk: usize,
    pub pcm_cavity_radii: RadiusScheme,
    /// SMD cavity/CDS radii scheme: "bondi" (default, aligned with PySCF/mnsol.F) | "uff_mixed"
    /// (non-eq.16 elements use the BONDI_UFF_RADII mixed table).
    pub smd_cavity_radii: SmdCavityRadii,
    /// SMD solvent name (e.g. "water", "acetone"). Looked up in solvent_db.
    /// When non-empty and method==SMD, auto-populates solvent_descriptors and solv_epsilon.
    pub solvent_name: String,
    /// SMD solvent descriptors [n, n25, α, β, γ, ε, φ, ψ]. Default: water.
    pub solvent_descriptors: Option<[f64; 8]>,
    #[pyo3(get, set)]
    // The initial MO coefficients and eigenvalues can be imported by setting chkfile
    pub chkfile: String,
    #[pyo3(get, set)]
    // At present, only the hdf5 format is available
    pub chkfile_type: String,
    #[pyo3(get, set)]
    // The initial density matrix can be imported by setting guessfile
    pub guessfile: String,
    #[pyo3(get, set)]
    // At present, only the hdf5 format is available
    pub guessfile_type: String,
    #[pyo3(get, set)]
    pub external_init_guess: Option<String>,
    #[pyo3(get, set)]
    // There are three kinds of available initital guesses: 1) sad (default), 2) hcore, 3) vsap
    pub initial_guess: String,
    #[pyo3(get, set)]
    pub basis_projection: String,
    /// Options of the `[output]` block (see `OutputKeywords`).
    pub output: OutputKeywords,
    #[pyo3(get, set)]
    pub noiter: bool,
    #[pyo3(get, set)]
    pub check_stab: String,
    #[pyo3(get, set)]
    pub use_dm_only: bool,
    #[pyo3(get, set)]
    pub vxc_screen_threshold: f64,
    /// Drop the grid points whose initial-guess density contribution
    /// `|rho_p * w_p|` is below this value, once, before the SCF iterations.
    /// 0.0 (default) keeps every grid point.
    #[pyo3(get, set)]
    pub grid_prune_threshold: f64,
    /// How the one-shot grid pruning materializes the smaller grid. `true` (default)
    /// copies the surviving columns of the already tabulated AO/AOP tables, which costs
    /// about a fifth of a re-tabulation but needs the old and the new table at once.
    /// `false` re-tabulates the surviving points, keeping the peak at one table at the
    /// price of about 0.8x of the first tabulation.
    #[pyo3(get, set)]
    pub grid_prune_compact: bool,
    #[pyo3(get, set)]
    pub ao_cutoff: f64,
    #[pyo3(get, set)]
    pub non0tab_blksize: usize,
    #[pyo3(get, set)]
    pub drop_dense_ao: bool,
    pub algorithm_jk: AlgorithmJK,
    pub algorithm_j: AlgorithmJ,
    pub algorithm_k: AlgorithmK,
    // Keywords for fciqmc dump
    #[pyo3(get, set)]
    pub fciqmc_dump: bool,
    // Kyewords for post scf analysis
    pub outputs: Vec<String>,
    pub outname: Option<String>,
    pub cube_orb_setting: [f64;2],
    pub cube_orb_indices: Vec<[usize;3]>,
    pub cube_orb_type: String,
    // keyword for rrs-pbc output
    pub pbc_eigenval: Option<String>,
    //pub output_wfn_in_real_space: usize,
    //pub output_cube: bool,
    //pub output_molden: bool,
    //pub output_fchk: bool,
    // Keywords for sad initial guess
    #[pyo3(get, set)]
    pub atom_sad: bool,
    pub empirical_dispersion: Option<String>,
    pub occupation_type: OCCType,
    pub frac_tolerant: f64,
    // Keywords for DeepPot
    #[pyo3(get, set)]
    pub deep_pot: bool,
    // Keywords for lib_rint
    #[pyo3(get, set)]
    pub run_lib_rint: bool,
    // Keywords for benchmarking various effective potentials, including ECP, ENXC, and Ghost EP
    #[pyo3(get, set)]
    pub bench_eps: bool,
    // Keywords for parallism
    /// Number of threads of **this MPI process**, not of the node.
    ///
    /// Every rank is an independent process and builds its own global rayon pool (and sets the
    /// OpenMP thread count of the BLAS behind it) to this value, so a node runs `num_threads`
    /// times the number of ranks placed on it. For a hybrid MPI + threads run, set it to
    /// `cores per node / ranks per node`. The default is 1, which serialises the parallel
    /// regions of each process.
    #[pyo3(get, set)]
    pub num_threads: Option<usize>,
    // batch size for each thread
    pub batch_size: usize,
    pub nforce_displacement: f64,
    pub ndipole_displacement: f64,
    pub nhessian_displacement: f64,
    pub force_state_occupation: Vec<ForceStateOccupation>,
    pub auxiliary_reference_states: Vec<(String,usize)>,
    pub rpa_de_excitation_parameters: Option<[f64;4]>,
    /// Maximum memory available to **this MPI process** in MB, `None` if no limit.
    ///
    /// The budget is consumed as `max_memory` minus the resident set of the current process
    /// (`detect_used_memory_mb("proc")`), so under MPI every rank applies it to itself and a node
    /// is allowed `max_memory` times the number of ranks placed on it. For a hybrid MPI + threads
    /// run, set it to `memory per node / ranks per node` with some head room. When it is not set,
    /// the batch decisions fall back to the available memory of the node, which every rank reads
    /// as its own budget, so an explicit value is recommended whenever several ranks share a node.
    ///
    /// The keyword only works where the algorithm is aware of the memory usage and performs batched
    /// computation, and it is not a hard cap: it feeds the batch sizes and the force-phase monitor.
    pub max_memory: Option<f64>,
    /// Back up of max_memory for the ISDF new driver (restored after ISDF preparation).
    #[pyo3(get, set)]
    pub max_memory_backup: Option<f64>,
    /// Abort the calculation when memory usage exceeds max_memory.
    pub abort_on_mem_exceed: bool,
    pub smear: Option<SmearingType>,
    pub smear_sigma: Option<f64>,
    /// Enable dynamic smearing annealing: sigma decays exponentially from smear_sigma
    /// toward smear_sigma_min (or smear_sigma*0.01 by default) over the course of SCF.
    /// A non-zero floor avoids degeneracy-driven orbital-occupation oscillations.
    pub smear_anneal: bool,
    /// Minimum sigma for annealing; defaults to max(smear_sigma * 0.01, 0.001).
    pub smear_sigma_min: Option<f64>,
    /// EDIIS penalty parameter η (default 0.5). Larger η = more conservative extrapolation.
    pub ediis_penalty: Option<f64>,
    /// ADIIS penalty parameter μ (default 0.5). Convex QP, penalizes deviation from D₀.
    pub adiis_penalty: Option<f64>,
    /// HOMO-LUMO gap threshold (Ha) for EDIIS→DIIS auto-switch in "ediis+diis" mode.
    /// Below this gap, EDIIS is preferred. Default 0.1 Ha.
    pub ediis_switch_gap: Option<f64>,
    pub guess_mix: bool,
    pub guess_mix_theta_deg: Vec<f64>,
    pub start_mix_cycle: usize,
    pub spin_correction_scheme: Option<String>,
    pub yamaguchi_triplet_type: Option<String>,
    /// External dipole field (x, y, z) intensity in atomic units
    pub ext_field_dipole: Option<[f64; 3]>,
    pub opt_engine: Option<String>,
    pub geometric_pyo3: Option<GeomeTRIC>,
    pub quasiparticle_methods:Option<QuasiParticle>,
    pub stop_at: Option<String>,
    pub xc_parser: String,
    pub tddft: Option<TDDFTParameters>,
    pub j2c_decomp: J2CDecompOption,
    pub ri_jk: RIJKOption,
    /// Whether to use the distributed (ScaLAPACK) Hamiltonian diagonalization
    /// in MPI runs. `Auto` (default) decides by problem size; `On` forces the
    /// distributed solver; `Off` forces the serial one.
    pub hamiltonian_distributed: HamiltonianDistributedMode,
    /// Whether the dRPA/SCSRPA response matrices are built and factorized in
    /// distributed (ScaLAPACK block-cyclic) form under MPI. `Auto` (default):
    /// naux >= 8192 and nproc >= 32; `On` forces the distributed path (testing);
    /// `Off` keeps the replicated path.
    pub rpa_distributed: HamiltonianDistributedMode,
    pub ri_pt2: RiPt2Option,
    pub hessian: Option<HessianParameters>,
    pub thermo: Option<ThermoParameters>,
    /// Use the optimized fxc_matvec_opt (rayon + pre-allocated workspace).
    #[pyo3(get, set)]
    pub use_fxc_opt: bool,
    pub rel: RelativisticMethod,
    /// Analytical derivative driver configuration.
    pub analdrv: Option<AnalDrvConfig>,
    /// Analytical derivative tasks to perform.
    pub analdrv_tasks: Vec<AnalDrvTask>,
}

impl Default for InputKeywords {
    fn default() -> Self {
        InputKeywords::init_ctrl()
    }
}

impl InputKeywords {
    pub fn init_ctrl() -> InputKeywords {
        InputKeywords{
            // keywords for machine and debug info
            print_level: 0,
            num_threads: Some(1),
            batch_size: 64,
            job_type: JobType::SinglePoint,
            nforce_displacement: 0.0013,
            ndipole_displacement: 3e-4,
            nhessian_displacement: 0.005,
            // Keywords for (aux)-basis sets
            basis_path: String::from("def2-SVP"),
            basis_type: String::from("spheric"),
            auxbas_path: String::from("def2-SV(P)-JKFIT"),
            auxbas_type: String::from("spheric"),
            use_auxbas: true,
            auxbasis_response: true,
            numerical_force: false,
            use_isdf: false,
            ri_k_only: false,
            isdf_k: None,
            isdf_k_only: false,
            isdf_k_mu: 17,
            isdf_new: false,
            isdf_type: String::from("udd"),
            ctrl_file: String::from("ctrl.in"),
            // Keywords associated with the method employed
            xc: String::from("x3lyp"),
            xc_type: DFTType::Standard,
            xc_namelist: None,
            xc_paralist: None,
            dfa_hybrid_scf: None,
            xc_model: None,
            empirical_dispersion: None,
            post_xc: vec![],
            post_correlation: vec![],
            post_ai_correction: String::from("none"),
            eri_type: String::from("ri_v"),
            use_ri_symm: true,
            charge: 0.0_f64,
            spin: 1.0_f64,
            use_int_nelec: true,
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
            radii_adjust: String::from("treutler"),
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
            diis_pinv_threshold: 1.0e-12,
            diis_orthogonalize_error: true,
            start_diis_cycle: 1,
            start_check_oscillation: 20,
            level_shift : None, 
            max_scf_cycle: 100,
            scf_acc_rho: 1.0e-5,
            scf_acc_eev: 1.0e-3,
            scf_acc_etot:1.0e-8,
            scf_conv_criteria: String::from("dm,eev"),
            scf_acc_g: 1.0e-5,
            has_chkfile: false, // not directly set by input
            external_init_guess: None, // not directly set by input
            initial_guess: String::from("sad"),
            basis_projection: String::from("occupied"),
            output: OutputKeywords::default(),
            noiter: false,
            check_stab: String::from("off"),
            // Kyewords for the manner to evaluate the Vk (and also Vxc) potentials
            // True:  using only density matrix in the evaluation
            // False: use coefficients as well with higher efficiency
            use_dm_only: false,
            vxc_screen_threshold: 0.0, // 0.0 = 关闭格点密度筛选（见 README）
            grid_prune_threshold: 0.0,  // 0.0 = 不剪枝（见 README）
            grid_prune_compact: true,   // true = 就地压缩（省时间、多一份表的峰值内存）
            ao_cutoff: 1.0e-9,
            non0tab_blksize: 0,     // 0 = auto-select based on nao
            drop_dense_ao: false,
            algorithm_jk: AlgorithmJK::Default,
            algorithm_j: AlgorithmJ::Default,
            algorithm_k: AlgorithmK::Default,
            // Keywords for the fciqmc dump
            fciqmc_dump: false,
            // Keywords for post scf
            outputs: vec![],
            outname: None,
            cube_orb_setting: [3.0,80.0],
            cube_orb_indices: Vec::new(),
            cube_orb_type: String::from("wavefunction"),
            // keyword for rrs-pbc output
            pbc_eigenval: None,
            //output_wfn_in_real_space: 0,
            //output_cube: false,
            //output_molden: false,
            //output_fchk: false,
            // Keyword to turn on atom calculations for the SAD initial guess
            atom_sad: false,
            // Derived keywords of identifying the method used
            //use_dft: false,
            //dft_type: None,
            deep_pot: false,
            run_lib_rint: false,
            bench_eps: false,
            occupation_type: OCCType::INTEGER,
            frac_tolerant: 1.0e-3,
            auxiliary_reference_states: Vec::new(),
            force_state_occupation: Vec::new(),
            rpa_de_excitation_parameters: None,
            max_memory: None,
            max_memory_backup: None,
            abort_on_mem_exceed: true,
            smear: None,
            smear_sigma: None,
            smear_anneal: false,
            smear_sigma_min: None,
            ediis_penalty: None,
            adiis_penalty: None,
            ediis_switch_gap: None,
            guess_mix: false,
            guess_mix_theta_deg: [15.0, 15.0].to_vec(),
            start_mix_cycle: 0,
            spin_correction_scheme: None,
            yamaguchi_triplet_type: None,
            ext_field_dipole: None,
            opt_engine: None,
            geometric_pyo3: None,
            quasiparticle_methods:None,
            solvent_enabled: false,
            solvent_ri: true,
            solv_epsilon:1.0,
            solvent_model: PcmMethod::CPCM,
            solv_chunk: 16,
            pcm_cavity_radii: RadiusScheme::UFF,
            smd_cavity_radii: SmdCavityRadii::Bondi,
            solvent_name: String::new(),
            solvent_descriptors: None, 
            stop_at: None,
            xc_parser: String::from("legacy"),
            j2c_decomp: J2CDecompOption::default(),
            ri_jk: RIJKOption::default(),
            hamiltonian_distributed: HamiltonianDistributedMode::default(),
            rpa_distributed: HamiltonianDistributedMode::default(),
            ri_pt2: RiPt2Option::default(),
            tddft: None,
            hessian: None,
            thermo: None,
            use_fxc_opt: false,
            rel: RelativisticMethod::None,
            analdrv: None,
            analdrv_tasks: Vec::new(),
        }
    }

    pub fn formated_output_in_toml(&self) -> String {
        toml::to_string(self).unwrap()
    }

}


#[test]
fn iter_inputkeywords()  {
    let dd = InputKeywords::init_ctrl();
    let ff = toml::to_string(&dd).unwrap();
    println!("{}", ff);
}

pub fn overall_parse_and_report_on_ctrl_geom(ctrl: &mut InputKeywords, geom: &mut GeomCell) {
    println!("=========================================================");
    println!("      Input parameters for the REST calculation");
    println!("=========================================================");

    match ctrl.job_type.clone() {
        JobType::SinglePoint => {println!("Calculation type: Single-point energy")},
        JobType::Force => {println!("Calculation type: Force calculation")},
        JobType::NumDipole => {println!("Calculation type: Numerical dipole calculation")},
        JobType::GeomOpt => {
            println!("Calculation type: Geometry optimization");
            // opt_engine: available options: "lbfgs", "geometric-pyo3"; default: "geometric-pyo3"
            if let Some(opt_engine) = &mut ctrl.opt_engine {
                if ctrl.print_level >= 1 {
                    println!("Optimization engine: {}", opt_engine);
                }
            } else {
                ctrl.opt_engine = Some("geometric_pyo3".to_string());
                if ctrl.print_level >= 1 {
                    println!("Optimization engine: Default (geometric_pyo3)");
                }
            }
        },
        JobType::NormalModes => {
            println!("Calculation type: Vibrational normal modes (frequency) calculation");
        },
        JobType::MD => {
            println!("Calculation type: Molecular dynamics (MD) simulation");
        },
    }

    // To make sure the geometric_pyo3 structure is initialized properly
    // The geometric_pyo3 block has higher priority than the corresponding settings in the ctrl block
    if let JobType::GeomOpt = ctrl.job_type.clone() {
        if let Some(opt_engine) = ctrl.opt_engine.clone() {
            if opt_engine.eq("geometric_pyo3") {
                println!("Geometric_pyo3 is used for geometry optimization");
                if let Some(geometric_pyo3) = &ctrl.geometric_pyo3 {
                    if geometric_pyo3.transition {
                        println!("Transition state optimization is enabled");
                    } else if geometric_pyo3.irc {
                        println!("IRC optimization is enabled");
                    } else {
                        println!("Standard geometry optimization is enabled");
                    }
                } else {
                    ctrl.geometric_pyo3 = Some(GeomeTRIC::default());
                    println!("Standard geometry optimization is enabled");
                }
            } else if opt_engine.eq("lbfgs") {
                println!("L-BFGS is used for geometry optimization");
            } else {
                panic!("Error:: Unknown optimization engine: {}", opt_engine);
            }
        }
    };

    if ctrl.cube_orb_type.eq("wavefunction") {
        println!("Cube output: wavefunction");
    } else if ctrl.cube_orb_type.eq("density") {
        println!("Cube output: density");
    } else {
        panic!("Error:: Unknown cube output type: {}", ctrl.cube_orb_type);
    }

    if ctrl.xc.eq("dl_dft") {
        ctrl.xc_type = DFTType::DeepLearning
    };
    match ctrl.xc_type {
        DFTType::Standard => {
            println!("The exchange-correlation method: {}", ctrl.xc);
        },
        DFTType::NonStandard => {
            if let (Some(xc_namelist), Some(xc_paralist), Some(dfa_hybrid_scf)) = (&ctrl.xc_namelist, &ctrl.xc_paralist, &ctrl.dfa_hybrid_scf) {
                println!("Nonstandard exchange-correlation method with {:16.8} exact exchange is employed:", dfa_hybrid_scf);
                xc_namelist.iter().zip(xc_paralist.iter()).for_each(|(xc_name, xc_para)| {
                    println!("    Component: {:>20}, parameter: {:16.8}", xc_name, xc_para);
                });
            } else {
                panic!("Error:: xc_namelist, xc_paralist and xc_hybrid_para should be specified for nonstandard xc method")
            }
        },
        DFTType::DeepLearning => {
            if let Some(xc_model) = &ctrl.xc_model {
                println!("Deep-learning exchange-correlation model is employed");
            } else {
                panic!("Error:: xc_model should be specified for deep-learning xc methods")
            }
        },
    };
    print!("Print level:   {}", ctrl.print_level);
    println!("    max log level: {}", log::max_level());
    if let Some(num_threads) = ctrl.num_threads {
        println!("The number of threads used for parallelism:      {}", num_threads);
    } else {
        println!("The number of threads used for parallelism:      {}", rayon::current_num_threads());
    }
    println!("The {}-GTO basis set is taken from {}", ctrl.basis_type,ctrl.basis_path);
    if ctrl.use_auxbas {
        println!("The {}-GTO auxiliary basis set is taken from {}", ctrl.auxbas_type,ctrl.auxbas_path)
    };
    if ctrl.even_tempered_basis {
        println!("Even tempered basis generation starts at: {}", ctrl.etb_start_atom_number);
    }
    println!("Charge: {:3}; Spin: {:3}",ctrl.charge,ctrl.spin);
    if ctrl.spin_channel == 1 {
        println!("Spin polarization: Off")
    } else if ctrl.spin_channel == 2 {
        println!("Spin polarization: On")
    };

    println!("Input molecular structure (in Angstrom): ----------");
    println!("{}", geom.formated_geometry());
    println!("End of molecular structure ------------------------");
    if geom.has_atom_overlap() {
        panic!("Error:: Overlapping atoms detected")
    }

    if ctrl.print_level>0 {
        println!("ERI Type: {}", ctrl.eri_type);
        println!("SCF convergency thresholds: {:e} for density matrix", ctrl.scf_acc_rho);
        println!("                            {:e} Ha. for sum of eigenvalues", ctrl.scf_acc_eev);
        println!("                            {:e} Ha. for total energy", ctrl.scf_acc_etot);
        println!("                            {:e} Ha. for grad_dm", ctrl.scf_acc_g);
        println!("SCF convergence criteria:       {}", ctrl.scf_conv_criteria);
        println!("Max. SCF cycle number:      {}", ctrl.max_scf_cycle);
        match geom.pbc {
            MOrC::Molecule => println!("It is a finite cluster calculation"),
            MOrC::Crystal => println!("It is a periodic calculation")
        }
    }
    let guessfile_exist = std::path::Path::new(&ctrl.guessfile).exists();
    let chkfile_exist = std::path::Path::new(&ctrl.chkfile).exists();
    if ctrl.guessfile.to_lowercase() != "none" {
        if ! guessfile_exist {
            panic!("The specified guessfile is missing \n({})",&ctrl.guessfile)
        }
        ctrl.external_init_guess = Some(String::from("guessfile"));
        println!("The initial guess is prepared by reading guessfile ({})", &ctrl.guessfile);
        if ctrl.has_chkfile && chkfile_exist {
            println!("The specified chkfile exists but is not loaded because the guessfile is specified");
            println!("It will be overwritten after the SCF procedure \n({})",&ctrl.chkfile)
        }
    } else {
        if ctrl.has_chkfile {
            if ! chkfile_exist {
                println!("The specified chkfile is missing, which will be created after the SCF procedure \n({})",&ctrl.chkfile)
            } else {
                println!("The initial guess is prepared by reading chkfile ({})",&ctrl.chkfile);
                ctrl.external_init_guess = Some(String::from("chkfile"));
            }
        }
    }
    if ctrl.external_init_guess.is_none() {
        println!("Initial guess is prepared by ({}).", &ctrl.initial_guess);

    }
    // The GW/evGW archive is stored inside the chkfile, so the feature needs one.
    if let Some(qp) = ctrl.quasiparticle_methods.as_ref() {
        if (qp.save_gw_checkpoint || qp.resume_from_checkpoint) && !ctrl.has_chkfile {
            panic!(
                "ERROR: `save_gw_checkpoint` / `resume_from_checkpoint` store and read the GW \
                 state inside the chkfile; set `chkfile = \"job.chk\"` in [ctrl] (not \"none\")."
            );
        }
    }
    if ctrl.force_state_occupation.len()>0 {
        if ! ctrl.external_init_guess.is_some() {
            panic!("ERROR: force_state_occupation can not be involved without an existing guessfile/chkfile");
        }
        // Normalize ref_index: entries without explicit reference default to
        // whichever file provides the initial guess.
        let actual_file = match ctrl.external_init_guess.as_ref().unwrap().as_str() {
            "guessfile" => &ctrl.guessfile,
            "chkfile" => &ctrl.chkfile,
            _ => unreachable!(),
        };
        for fso in ctrl.force_state_occupation.iter_mut() {
            fso.normalize_ref_index(actual_file);
        }
    }

    if ctrl.use_ri_symm {
        debug!("Turn on the basis pair symmetry for RI 3D-tensors")
    } else {
        debug!("Turn off the basis pair symmetry for RI 3D-tensors")
    };
    debug!("The pruning method is {}", ctrl.pruning);
    debug!("The radial grid generation method is {}", ctrl.rad_grid_method);
    debug!("The Becke radii adjustment scheme is {}", ctrl.radii_adjust);
    debug!("min_num_angular_points: {}", ctrl.min_num_angular_points);
    debug!("max_num_angular_points: {}", ctrl.max_num_angular_points);
    debug!("hardness: {}", ctrl.hardness);
    debug!("Grid generation level: {}", ctrl.grid_gen_level);
    debug!("Even tempered basis generation: {}", ctrl.even_tempered_basis);

    let tmp_mixer = ctrl.mixer.clone();
    let mut mixer_log = "".to_string();
    if tmp_mixer.eq(&"direct") {
        mixer_log = "No charge density mixing is employed for the SCF procedure".to_string();
    } else if tmp_mixer.eq(&"linear") {
        mixer_log = format!("The {} mixing is employed with the mixing parameter of {} for the SCF procedure", 
                    &tmp_mixer, &ctrl.mix_param);
    } else if tmp_mixer.eq(&"diis")
            || tmp_mixer.eq(&"ediis")
            || tmp_mixer.eq(&"ediis+diis")
            || tmp_mixer.eq(&"adiis+diis") {
        mixer_log = format!("The {} mixing with (param, max_vec_len) = ({}, {}) is employed for the SCF procedure", 
                    &tmp_mixer, &ctrl.mix_param, &ctrl.num_max_diis);
        mixer_log.push_str(&format!("\nTurn on the {} mixing after {} step(s) of SCF iteractions with the linear mixing", 
                    &tmp_mixer, &ctrl.start_diis_cycle));
    } else {
        //ctrl.mixer = String::from("direct");
        panic!("Unknown charge density mixer ({})! No charge density mixing will be invoked.", ctrl.mixer);
    };
    debug!("{}", mixer_log);

    if ctrl.guess_mix {
        info!("Initial guess mixing enabled");
        debug!("HOMO-LUMO rotated with theta = {:.1}° (alpha), {:.1}° (beta) to induce symmetry breaking",
            ctrl.guess_mix_theta_deg[0], ctrl.guess_mix_theta_deg[1]);
    }
    if ctrl.solvent_enabled {
        info!("Current solvent model is {}.",ctrl.solvent_model);
        if ctrl.solvent_name.is_empty() {
            debug!("Solvent: {} (eps = {:.4})", ctrl.solvent_model, ctrl.solv_epsilon);
        } else {
            debug!("Solvent: {} ({}, eps = {:.4})", ctrl.solvent_name, ctrl.solvent_model, ctrl.solv_epsilon);
        }
    }


    println!("=========================================================");

}

pub fn parse_ctrl_keywords(tmp_keys: &serde_json::Value) -> anyhow::Result<InputKeywords> { 
    let mut tmp_input = InputKeywords::init_ctrl();
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
            let mut builder = Builder::new();
            builder.target(Target::Stdout);
            builder.filter_level(printlevel2loglevel(tmp_input.print_level));
            builder.format(|buf, record| {
                if record.level() == Level::Info {
                    writeln!(buf, "{}", record.args())
                } else {
                    writeln!(buf, "[{:<5} {}] {}", record.level(), record.target().split_once("::").map(|(_, rest)| rest).unwrap_or(record.target()), record.args())
                }
            });
            let _ = builder.try_init();
            // log::set_max_level(printlevel2loglevel(tmp_input.print_level));
            //let default_rayon_current_num_threads = rayon::current_num_threads();
            tmp_input.num_threads = match tmp_ctrl.get("num_threads").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {Some(tmp_str.to_lowercase().parse().unwrap_or(1))},
                serde_json::Value::Number(tmp_num) => {Some(tmp_num.as_i64().unwrap_or(1) as usize)},
                other => {Some(1)},
            };
            tmp_input.batch_size = match tmp_ctrl.get("batch_size").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(64)},
                serde_json::Value::Number(tmp_num) => {tmp_num.as_i64().unwrap_or(64) as usize},
                other => {64},
            };
            if let Some(num_threads) = tmp_input.num_threads {
                //if tmp_input.print_level>0 {println!("The number of threads used for parallelism:      {}", num_threads)};
                // Now move the setting of rayon thread numbers to the main.rs
                //rayon::ThreadPoolBuilder::new().num_threads(num_threads);
                rayon::ThreadPoolBuilder::new().num_threads(num_threads).build_global().unwrap_or_else(|x| {println!("{:?}", &x)});
                omp_set_num_threads_global_wrapper(num_threads);
            } else {
                omp_set_num_threads_global_wrapper(rayon::current_num_threads());
                //if tmp_input.print_level>0 {println!("The default rayon num_threads value is used:      {}", rayon::current_num_threads())};
            };
            //println!("max_num_threads: {}, current_num_threads: {}", rayon::max_num_threads(), rayon::current_num_threads());
            // ====================================
            //  Keywords for the (aux) basis sets
            // ====================================

            // if env REST_BASIS_DIR is set, use it as rest_basis_dir, 
            // otherwise use the default path from rest docker's convention or $REST_HOME/rest/basis-set-pool/
            let rest_basis_dir = path_util::get_rest_basis_dir();

            tmp_input.basis_path = match tmp_ctrl.get("basis_path").unwrap_or(&serde_json::Value::Null) {
               serde_json::Value::String(tmp_bas) => {
                    path_util::get_valid_basis_path(&tmp_bas, &rest_basis_dir, "basis")
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
            //if tmp_input.print_level> 0 {println!("The {}-GTO basis set is taken from {}", tmp_input.basis_type,tmp_input.basis_path)};

            tmp_input.pruning = match tmp_ctrl.get("pruning").unwrap_or(&serde_json::Value::Null){
                serde_json::Value::String(tmp_type) => {tmp_type.to_lowercase()},
                other => {String::from("nwchem")} //default prune method: sg1
            };
            //if tmp_input.print_level>0 {println!("The pruning method will be {}", tmp_input.pruning)};

            tmp_input.rad_grid_method = match tmp_ctrl.get("radial_grid_method").unwrap_or(&serde_json::Value::Null){
                serde_json::Value::String(tmp_type) => {tmp_type.to_lowercase()},
                other => {String::from("treutler")} //default prune method: sg1
            };
            //if tmp_input.print_level>0 {println!("The radial grid generation method will be {}", tmp_input.rad_grid_method)};

            tmp_input.radii_adjust = match tmp_ctrl.get("radii_adjust").unwrap_or(&serde_json::Value::Null){
                serde_json::Value::String(tmp_type) => {tmp_type.to_lowercase()},
                other => {String::from("treutler")} // default: Treutler-Ahlrichs sqrt radii (PySCF/PyFock default)
            };
            //if tmp_input.print_level>0 {println!("The Becke radii adjustment will be {}", tmp_input.radii_adjust)};

            tmp_input.eri_type = match tmp_ctrl.get("eri_type").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_eri) => {
                    if tmp_eri.to_lowercase().eq("ri_v") || tmp_eri.to_lowercase().eq("ri-v")
                    {
                        String::from("ri_v")
                    } else {tmp_eri.to_lowercase()}
                },
                other => {String::from("ri_v")},
            };
            let eri_type = tmp_input.eri_type.clone();
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
            }else if eri_type.eq(&String::from("isdf")) {
                // density-driven ISDF (like old version): isdf_new = true
                tmp_input.use_auxbas = true;
                tmp_input.use_isdf = true;
                tmp_input.isdf_new = true;
                tmp_input.eri_type = String::from("ri_v");
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
            //if tmp_input.print_level>0 {println!("ERI Type: {}", tmp_input.eri_type)};

            tmp_input.use_ri_symm = match tmp_ctrl.get("use_ri_symm").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_str) => {*tmp_str},
                other => {true},
            };
            //if tmp_input.print_level>0 {
            //    if tmp_input.use_ri_symm {
            //        println!("Turn on the basis pair symmetry for RI 3D-tensors")
            //    } else {
            //        println!("Turn off the basis pair symmetry for RI 3D-tensors")
            //    };
            //}
            tmp_input.isdf_k_mu = match tmp_ctrl.get("isdf_k_mu").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(8_usize)},
                serde_json::Value::Number(tmp_num) => {tmp_num.as_i64().unwrap_or(8) as usize},
                other => {8_usize},
            };            

            tmp_input.isdf_type = match tmp_ctrl.get("isdf_type").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_type) => {tmp_type.to_lowercase()},
                other => {String::from("udd")},
            };

            tmp_input.isdf_k = match tmp_ctrl.get("isdf_k").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(s) => {
                    let lower = s.to_lowercase();
                    if lower == "none" {
                        None
                    } else {
                        lower.parse::<usize>().ok()
                    }
                }
                serde_json::Value::Number(n) => {
                    n.as_u64().and_then(|v: u64| v.try_into().ok())
                }
                _ => None,
            };

            tmp_input.auxbas_type = match tmp_ctrl.get("auxbas_type").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_type) => {tmp_type.to_lowercase()},
                other => {String::from("spheric")}
            };
            tmp_input.auxbas_path = match tmp_ctrl.get("auxbas_path").unwrap_or(&serde_json::Value::Null) {
               serde_json::Value::String(tmp_bas) => {
                    path_util::get_valid_basis_path(&tmp_bas, &rest_basis_dir, "auxiliary basis")
               },
               other => {
                    println!("No auxiliary basis set is specified. Default auxiliary basis set in REST is def2-universal-jkfit");
                    let default_bas = String::from("def2-universal-jkfit");
                    path_util::get_valid_basis_path(&default_bas, &rest_basis_dir, "auxiliary basis")
               }
            };
            //if tmp_input.use_auxbas && tmp_input.print_level>0 {
            //    println!("The {}-GTO auxiliary basis set is taken from {}", tmp_input.auxbas_type,tmp_input.auxbas_path)
            //};
            // ===============================================
            //  Keywords for Gradient calculation
            // ==============================================
            tmp_input.auxbasis_response = match tmp_ctrl.get("auxbasis_response").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_str) => *tmp_str,
                other => true,
            };
            tmp_input.numerical_force = match tmp_ctrl.get("numerical_force").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_str) => *tmp_str,
                other => false,
            };
            // ==============================================
            //  JobType
            // ==============================================
            tmp_input.job_type = match tmp_ctrl.get("job_type").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_xc) => {
                    let tmp_xc_low = tmp_xc.to_lowercase();
                    if tmp_xc_low.eq("opt") || tmp_xc_low.eq("geometry optimization") ||
                       tmp_xc_low.eq("geometry relaxation") || tmp_xc_low.eq("geom_opt") ||
                       tmp_xc_low.eq("geom_relax") || tmp_xc_low.eq("relax") {
                        JobType::GeomOpt
                    } else if tmp_xc_low.eq("force") || tmp_xc_low.eq("gradient") {
                        JobType::Force
                    } else if tmp_xc_low.eq("numdipole") || tmp_xc_low.eq("numerical dipole") {
                        JobType::NumDipole
                    } else if tmp_xc_low.eq("energy") || tmp_xc_low.eq("single point") ||
                      tmp_xc_low.eq("single_point") || tmp_xc_low.eq("sp") ||
                      tmp_xc_low.eq("singlepoint") {
                        JobType::SinglePoint
                    } else if tmp_xc_low.eq("normal_modes") || tmp_xc_low.eq("freq") ||
                      tmp_xc_low.eq("frequency") || tmp_xc_low.eq("vibration") {
                        JobType::NormalModes
                    } else if tmp_xc_low.eq("md") || tmp_xc_low.eq("molecular dynamics") ||
                      tmp_xc_low.eq("molecular_dynamics") {
                        JobType::MD
                    } else {
                        anyhow::bail!(
                            "Unknown job_type '{}' in the [ctrl] block. Valid values: \
                             energy/single point/single_point/sp/singlepoint, force/gradient, \
                             numdipole/numerical dipole, \
                             opt/geometry optimization/geometry relaxation/geom_opt/geom_relax/relax, \
                             normal_modes/freq/frequency/vibration, md/molecular dynamics/molecular_dynamics.",
                            tmp_xc
                        );
                    }
                },
                other => {JobType::SinglePoint},
            };
            tmp_input.nforce_displacement = match tmp_ctrl.get("nforce_displacement").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_nforce) => {tmp_nforce.to_lowercase().parse().unwrap_or(0.0013)},
                serde_json::Value::Number(tmp_nforce) => {tmp_nforce.as_f64().unwrap_or(0.0013)},
                other => {0.0013},
            };
            tmp_input.ndipole_displacement = match tmp_ctrl.get("nforce_displacement").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_nforce) => {tmp_nforce.to_lowercase().parse().unwrap_or(3e-4)},
                serde_json::Value::Number(tmp_nforce) => {tmp_nforce.as_f64().unwrap_or(3e-4)},
                serde_json::Value::Null => {3e-4},
                other => panic!("The ndipole_displacement is not recognized"),
            };
            tmp_input.nhessian_displacement = match tmp_ctrl.get("nhessian_displacement").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_val) => {tmp_val.to_lowercase().parse().unwrap_or(0.005)},
                serde_json::Value::Number(tmp_val) => {tmp_val.as_f64().unwrap_or(0.005)},
                other => {0.005},
            };
            // ==============================================
            //  Keywords associated with the method employed
            // ==============================================
            tmp_input.xc = match tmp_ctrl.get("xc").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_xc) => {tmp_xc.to_lowercase()},
                other => {String::from("hf")},
            };
            tmp_input.xc_type = match tmp_ctrl.get("xc_type").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_xc) => {
                    if tmp_xc.to_lowercase().eq("nonstandard") || tmp_xc.to_lowercase().eq("non-standard") {
                        DFTType::NonStandard
                    } else if tmp_xc.to_lowercase().eq("deep-learning")  
                           || tmp_xc.to_lowercase().eq("deep_learning") 
                           || tmp_xc.to_lowercase().eq("deep learning") 
                           || tmp_xc.to_lowercase().eq("machine learning") 
                           || tmp_xc.to_lowercase().eq("machine-learning") 
                           || tmp_xc.to_lowercase().eq("machine_learning") 
                    {
                        DFTType::DeepLearning
                    } else if tmp_xc.to_lowercase().eq("standard")  {
                        DFTType::Standard
                    } else {
                        println!("Unknown xc_type: ({}). xc_type is set to `standard`", tmp_xc);
                        DFTType::Standard
                    }
                },
                other => {
                    DFTType::Standard
                },
            };
            // to be deprecated
            tmp_input.xc_namelist = match tmp_ctrl.get("xc_namelist").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_op) => {Some(vec![tmp_op.to_lowercase()])},
                serde_json::Value::Array(tmp_op) => {
                    let mut tmp_vec:Vec<String> = vec![];
                    tmp_op.iter().for_each(|x| {
                        let op_type = x.to_string();
                        let string_len = op_type.len();
                        tmp_vec.push(op_type[1..string_len-1].to_lowercase().to_string())
                    });
                    Some(tmp_vec)
                },
                other => {None},
            };
            tmp_input.xc_paralist = match tmp_ctrl.get("xc_paralist").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Array(tmp_op) => {
                    let tmp_vec:Vec<f64> = tmp_op.iter().map(|x| {
                        match x {
                            serde_json::Value::String(tmp_str) => {tmp_str.parse().unwrap_or(0.0)},
                            serde_json::Value::Number(tmp_num) => {tmp_num.as_f64().unwrap_or(0.0)},
                            other => {0.0},
                        }
                    }).collect::<Vec<f64>>();
                    Some(tmp_vec)
                },
                other => {None},
            };
            //
            tmp_input.dfa_hybrid_scf = match tmp_ctrl.get("xc_hybrid_para").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {Some(tmp_str.parse().unwrap_or(0.0))},
                serde_json::Value::Number(tmp_num) => {Some(tmp_num.as_f64().unwrap_or(0.0))},
                other => {None}
            };
            tmp_input.xc_model = match tmp_ctrl.get("xc_model").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_xc) => {
                    tmp_input.xc_type=DFTType::DeepLearning; 
                    Some(tmp_xc.to_lowercase())
                },
                other => {None},
            };

            tmp_input.empirical_dispersion = match tmp_ctrl.get("empirical_dispersion").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_emp) => {
                    if tmp_emp.to_lowercase() == "none" {
                       None 
                    } else if tmp_emp.to_lowercase() == "true" {
                        Some("d3bj".to_string())
                    } else {
                       Some(tmp_emp.to_lowercase())
                    }
                },
                other => {None},
            };
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
                    println!("WARNNING: Unknown post-scf correlation method: {}", corr)
                }
                //if corr.to_lowercase().eq(&pt2) 
            });
            tmp_input.post_ai_correction = match tmp_ctrl.get("post_ai_correction").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_xc) => {tmp_xc.to_lowercase()},
                other => {String::from("none")},
            };
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
            tmp_input.use_int_nelec = match tmp_ctrl.get("use_int_nelec").unwrap_or(&serde_json::Value::Null) {
                // serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(true)},
                serde_json::Value::Bool(tmp_bool) => tmp_bool.clone(),
                other => {true},
            };
            tmp_input.spin_polarization = match tmp_ctrl.get("spin_polarization").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value:: String(tmp_str) => tmp_str.to_lowercase().parse().unwrap_or(false),
                serde_json::Value:: Bool(tmp_bool) => tmp_bool.clone(),
                other => false,
            };
            tmp_input.spin_channel = if tmp_input.spin_polarization {
                //if tmp_input.print_level>0 {println!("Spin polarization: On")};
                2_usize
            } else {
                //if tmp_input.print_level>0 {println!("Spin polarization: Off")};
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
                    info!("Read grids from the external file: {}", tmp_type);
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
            //  Keywords associated with solvent model
            // ==============================================
            tmp_input.solvent_enabled = match tmp_ctrl.get("solvent_enabled").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value:: String(tmp_str) => tmp_str.to_lowercase().parse().unwrap_or(false),
                serde_json::Value:: Bool(tmp_bool) => tmp_bool.clone(),
                serde_json::Value::Null =>{
                    match tmp_ctrl.get("solvent_model").unwrap_or(&serde_json::Value::Null) {
                        serde_json::Value::String(s) => !s.trim().is_empty(),
                        _ => match tmp_ctrl.get("solvent").unwrap_or(&serde_json::Value::Null) {
                            serde_json::Value::String(s) => !s.trim().is_empty(),
                            _ => false,
                        },
                    }
                },
                    other => false,
            };
            tmp_input.solvent_ri = match tmp_ctrl.get("solvent_ri").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value:: String(tmp_str) => tmp_str.to_lowercase().parse().unwrap_or(true),
                serde_json::Value:: Bool(tmp_bool) => tmp_bool.clone(),
                other => true,
            };
            tmp_input.solvent_model = match tmp_ctrl.get("solvent_model") {
                Some(value) => {
                    serde_json::from_value(value.clone())?
                },
                None => PcmMethod::CPCM,
            };
            tmp_input.pcm_cavity_radii = match tmp_ctrl.get("pcm_cavity_radii") {
                Some(value) => {
                    serde_json::from_value(value.clone())?
                },
                None => RadiusScheme::UFF,
            };
            tmp_input.smd_cavity_radii = match tmp_ctrl.get("smd_cavity_radii") {
                Some(value) => {
                    serde_json::from_value(value.clone())?
                },
                None => SmdCavityRadii::Bondi,
            };
            let has_explicit_eps = matches!(
                tmp_ctrl.get("solv_epsilon").unwrap_or(&serde_json::Value::Null),
                serde_json::Value::String(_) | serde_json::Value::Number(_)
            );
            tmp_input.solv_epsilon = match tmp_ctrl.get("solv_epsilon").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_fc) => {tmp_fc.to_lowercase().parse().unwrap_or(1.0_f64)},
                serde_json::Value::Number(tmp_fc) => {tmp_fc.as_f64().unwrap_or(1.0_f64) as f64},
                serde_json::Value::Null => {
                    1.0_f64
                },
                other => {
                    panic!("ERROR: False form of solvent epsilon provided, please write as string or float number.");
                },
            };
            // Experimental function
            tmp_input.solv_chunk = match tmp_ctrl.get("solv_chunk").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(16)},
                serde_json::Value::Number(tmp_num) => {tmp_num.as_u64().unwrap_or(16) as usize},
                _ => {16},
            };
            // 0 would panic the RI veff path (`par_chunks(0)`); clamp to at least one point.
            tmp_input.solv_chunk = tmp_input.solv_chunk.max(1);
            // ==============================================
            //  Keywords associated with relativistic methods 
            // ==============================================
            tmp_input.rel = match tmp_ctrl.get("rel").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => match tmp_str.to_lowercase().as_str() {
                    // "x2c"   => RelativisticMethod::X2C,
                    "sfx2c"  => RelativisticMethod::SFX2C,
                    "none"  => RelativisticMethod::None,
                    _       => RelativisticMethod::None,
                },
                _other => RelativisticMethod::None,
            };

            // Explicit solvent_descriptors array [n, n25, α, β, γ, ε, φ, ψ]
            tmp_input.solvent_descriptors = match tmp_ctrl.get("solvent_descriptors").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Array(arr) => {
                    let vals: Vec<f64> = arr.iter()
                        .filter_map(|v| v.as_f64())
                        .collect();
                    if vals.len() == 8 {
                        let desc = [vals[0], vals[1], vals[2], vals[3], vals[4], vals[5], vals[6], vals[7]];
                        if has_explicit_eps && (desc[5] - tmp_input.solv_epsilon).abs() > 0.1 {
                            panic!("ERROR: solvent_descriptors epsilon ({:.4}) != solv_epsilon ({:.4}). \
                                   Please set only one or ensure they match.", desc[5], tmp_input.solv_epsilon);
                        }
                        if !has_explicit_eps {
                            tmp_input.solv_epsilon = desc[5];
                        }
                        Some(desc)
                    } else {
                        panic!("ERROR: solvent_descriptors must be an array of 8 floats [n, n25, α, β, γ, ε, φ, ψ].");
                    }
                },
                _ => None,
            };

            if let Some((name, eps, solv_data)) = solvent::parse_solvent_name(tmp_ctrl) {
                tmp_input.solvent_name = name;
                tmp_input.solv_epsilon = eps;
                tmp_input.solvent_descriptors = Some(solv_data);
            }

            if tmp_input.solvent_enabled && tmp_input.solvent_name.is_empty()
                && tmp_input.solvent_descriptors.is_none() && tmp_input.solv_epsilon == 1.0_f64
            {
                panic!("ERROR: solvent_enabled is true but solvent/solvent_descriptors/solv_epsilon is not provided. Check your solvent settings.");
            }

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
            // The default stopping rule stays "dm,eev". Only the density and eigenvalue
            // thresholds are loosened (1e-6 -> 1e-5 and 1e-5 -> 1e-3), matching the level
            // PySCF reaches by default; switching the criterion itself to the gradient
            // norm moved analytic derivatives and thermochemistry by more than their
            // regression tolerances, so "g" is left as an explicit user choice.
            // scf_acc_etot still binds in every mode, so it keeps its historical value.
            tmp_input.scf_acc_rho = match tmp_ctrl.get("scf_acc_rho").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(1.0e-5)},
                serde_json::Value::Number(tmp_num) => {tmp_num.as_f64().unwrap_or(1.0e-5)},
                other => {1.0e-5}
            };
            tmp_input.scf_acc_eev = match tmp_ctrl.get("scf_acc_eev").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(1.0e-3)},
                serde_json::Value::Number(tmp_num) => {tmp_num.as_f64().unwrap_or(1.0e-3)},
                other => {1.0e-3}
            };
            tmp_input.scf_acc_etot = match tmp_ctrl.get("scf_acc_etot").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(1.0e-8)},
                serde_json::Value::Number(tmp_num) => {tmp_num.as_f64().unwrap_or(1.0e-8)},
                other => {1.0e-8}
            };
            tmp_input.scf_conv_criteria = match tmp_ctrl.get("scf_conv_criteria").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase()},
                other => {String::from("dm,eev")}
            };
            tmp_input.scf_acc_g = match tmp_ctrl.get("scf_acc_g").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(1.0e-5)},
                serde_json::Value::Number(tmp_num) => {tmp_num.as_f64().unwrap_or(1.0e-5)},
                other => {1.0e-5}
            };

            tmp_input.mixer = match tmp_ctrl.get("mixer").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {
                    let m = tmp_str.to_lowercase();
                    if m.eq(&"ddiis") { String::from("diis") } else { m }
                },
                other => {String::from("diis")},
            };
            tmp_input.mix_param = match tmp_ctrl.get("mix_param").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(0.2)},
                serde_json::Value::Number(tmp_num) => {tmp_num.as_f64().unwrap_or(0.2)},
                other => {0.2}
            };
            tmp_input.num_max_diis = match tmp_ctrl.get("num_max_diis").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(8_usize)},
                serde_json::Value::Number(tmp_num) => {tmp_num.as_i64().unwrap_or(8) as usize},
                other => {8_usize}
            };
            tmp_input.diis_pinv_threshold = match tmp_ctrl.get("diis_pinv_threshold").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(1.0e-12)},
                serde_json::Value::Number(tmp_num) => {tmp_num.as_f64().unwrap_or(1.0e-12)},
                other => {1.0e-12}
            };
            tmp_input.diis_orthogonalize_error = match tmp_ctrl.get("diis_orthogonalize_error").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_bool) => {*tmp_bool},
                serde_json::Value::String(tmp_str) => {matches!(tmp_str.to_lowercase().as_str(), "true" | "yes" | "on" | "1")},
                other => {true}
            };
            tmp_input.start_diis_cycle = match tmp_ctrl.get("start_diis_cycle").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(1_usize)},
                serde_json::Value::Number(tmp_num) => {tmp_num.as_i64().unwrap_or(1) as usize},
                other => {1_usize}
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
                other => String::from("hdf5"),
            };

            tmp_input.chkfile = match tmp_ctrl.get("chkfile").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_chk) => tmp_chk.clone(),
                other => String::from("none"),
            };
            tmp_input.chkfile_type = match tmp_ctrl.get("chkfile_type").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_chk) => tmp_chk.to_lowercase().clone(),
                other => String::from("hdf5"),
            };

            tmp_input.has_chkfile = ! tmp_input.chkfile.to_lowercase().eq(&"none");

            tmp_input.initial_guess = match tmp_ctrl.get("initial_guess").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase()},
                other => {String::from("sad")},
            };

            tmp_input.basis_projection = match tmp_ctrl.get("basis_projection").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase()},
                other => {String::from("occupied")},
            };

            tmp_input.noiter = match tmp_ctrl.get("noiter").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value:: String(tmp_str) => tmp_str.to_lowercase().parse().unwrap_or(false),
                serde_json::Value:: Bool(tmp_bool) => tmp_bool.clone(),
                other => false,
            };
            tmp_input.check_stab = match tmp_ctrl.get("check_stab").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value:: String(tmp_str) => tmp_str.to_lowercase(),
                serde_json::Value:: Bool(true) => String::from("auto"),
                serde_json::Value:: Bool(false) => String::from("off"),
                other => String::from("off"),
            };
            tmp_input.use_dm_only = match tmp_ctrl.get("use_dm_only").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value:: String(tmp_str) => tmp_str.to_lowercase().parse().unwrap_or(false),
                serde_json::Value:: Bool(tmp_bool) => tmp_bool.clone(),
                other => false,
            };
            tmp_input.vxc_screen_threshold = match tmp_ctrl.get("vxc_screen_threshold").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(num) => num.as_f64().unwrap_or(0.0),
                serde_json::Value::String(s) => s.parse().unwrap_or(0.0),
                _ => 0.0,
            };
            tmp_input.grid_prune_threshold = match tmp_ctrl.get("grid_prune_threshold").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(num) => num.as_f64().unwrap_or(0.0),
                serde_json::Value::String(s) => s.parse().unwrap_or(0.0),
                _ => 0.0,
            };
            tmp_input.grid_prune_compact = match tmp_ctrl.get("grid_prune_compact").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(s) => s.to_lowercase().parse().unwrap_or(true),
                serde_json::Value::Bool(b) => *b,
                _ => true,
            };
            tmp_input.ao_cutoff = match tmp_ctrl.get("ao_cutoff").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(num) => num.as_f64().unwrap_or(1.0e-9),
                serde_json::Value::String(s) => s.parse().unwrap_or(1.0e-9),
                _ => 1.0e-9,
            };
            tmp_input.non0tab_blksize = match tmp_ctrl.get("non0tab_blksize").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(num) => num.as_u64().map(|v| v as usize).unwrap_or(0),
                serde_json::Value::String(s) => s.parse().unwrap_or(0),
                _ => 0,
            };
            tmp_input.drop_dense_ao = match tmp_ctrl.get("drop_dense_ao").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(s) => s.to_lowercase().parse().unwrap_or(false),
                serde_json::Value::Bool(b) => *b,
                _ => false,
            };
            // setup and sanity check of J/K algorithms
            tmp_input.algorithm_jk = tmp_ctrl.get("algorithm_jk").map(serde_from_value).unwrap_or_default();
            tmp_input.algorithm_j = tmp_ctrl.get("algorithm_j").map(serde_from_value).unwrap_or_default();
            tmp_input.algorithm_k = tmp_ctrl.get("algorithm_k").map(serde_from_value).unwrap_or_default();
            tmp_input.j2c_decomp = tmp_ctrl.get("j2c_decomp").map(serde_from_value).unwrap_or_default();
            tmp_input.ri_jk = tmp_ctrl.get("ri_jk").map(serde_from_value).unwrap_or_default();
            tmp_input.hamiltonian_distributed = tmp_ctrl.get("hamiltonian_distributed").map(serde_from_value).unwrap_or_default();
            tmp_input.rpa_distributed = tmp_ctrl.get("rpa_distributed").map(serde_from_value).unwrap_or_default();
            if (tmp_input.algorithm_j != AlgorithmJ::Default || tmp_input.algorithm_k != AlgorithmK::Default) {
                if tmp_input.algorithm_jk != AlgorithmJK::Default {
                    warn!("algorithm_j or algorithm_k are specified, the setting in algorithm_jk will be ignored.");
                }
                tmp_input.algorithm_jk = AlgorithmJK::Separated(tmp_input.algorithm_j, tmp_input.algorithm_k);
            }
            tmp_input.ri_pt2 = tmp_ctrl.get("ri_pt2").map(serde_from_value).unwrap_or_default();
            // ================================================
            //  Keywords associated with the elec occupation 
            // ================================================
            tmp_input.occupation_type = 
            match tmp_ctrl.get("occupation_type").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_type) => {
                    let tmp_occupation_type = tmp_type.to_lowercase();
                    if tmp_occupation_type.eq("integer") {
                        OCCType::INTEGER
                    } else if tmp_occupation_type.eq("sad") {
                        OCCType::ATMSAD
                    } else if tmp_occupation_type.eq("frac") {
                        OCCType::FRAC
                    } else {
                        OCCType::INTEGER
                    }
                },
                other => OCCType::INTEGER,
            };
            tmp_input.frac_tolerant = match tmp_ctrl.get("frac_tolerant").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(1.0e-3)},
                serde_json::Value::Number(tmp_num) => {tmp_num.as_f64().unwrap_or(1.0e-3)},
                other => {1.0e-3}
            };
            tmp_input.force_state_occupation = match tmp_ctrl.get("force_state_occupation").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(_) => vec![],
                serde_json::Value::Array(tmp_op) => {
                    let mut tmp_vec: Vec<ForceStateOccupation> = vec![];
                    tmp_op.iter().for_each(|x| {
                        let tmp_obj = match x {
                            serde_json::Value::Array(tmp_value) => {
                                match tmp_value.len() {
                                    5 => {
                                        // [ref_state, ref_spin, force_occ, min, max]
                                        let ref_state = tmp_value[0].as_u64().unwrap_or(0) as usize;
                                        let ref_spin = tmp_value[1].as_u64().unwrap_or(0) as usize;
                                        let target_spin = ref_spin;
                                        let force_occ = tmp_value[2].as_f64().unwrap_or(0.0);
                                        let check_min = tmp_value[3].as_u64().unwrap_or(0) as usize;
                                        let check_max = tmp_value[4].as_u64().unwrap_or(0) as usize;
                                        Some(ForceStateOccupation::init(
                                            String::new(),
                                            ref_state,
                                            ref_spin,
                                            target_spin,
                                            force_occ,
                                            check_min,
                                            check_max,
                                        ))
                                    }
                                    6 => {
                                        match &tmp_value[0] {
                                            serde_json::Value::String(ref reference) => {
                                                // ["ref.hdf5", ref_state, ref_spin, force_occ, min, max]
                                                let ref_state = tmp_value[1].as_u64().unwrap_or(0) as usize;
                                                let ref_spin = tmp_value[2].as_u64().unwrap_or(0) as usize;
                                                let target_spin = ref_spin;
                                                let force_occ = tmp_value[3].as_f64().unwrap_or(0.0);
                                                let check_min = tmp_value[4].as_u64().unwrap_or(0) as usize;
                                                let check_max = tmp_value[5].as_u64().unwrap_or(0) as usize;
                                                Some(ForceStateOccupation::init(
                                                    reference.to_string(),
                                                    ref_state,
                                                    ref_spin,
                                                    target_spin,
                                                    force_occ,
                                                    check_min,
                                                    check_max,
                                                ))
                                            }
                                            _ => {
                                                // [ref_state, ref_spin, target_spin, force_occ, min, max]
                                                let ref_state = tmp_value[0].as_u64().unwrap_or(0) as usize;
                                                let ref_spin = tmp_value[1].as_u64().unwrap_or(0) as usize;
                                                let target_spin = tmp_value[2].as_u64().unwrap_or(0) as usize;
                                                let force_occ = tmp_value[3].as_f64().unwrap_or(0.0);
                                                let check_min = tmp_value[4].as_u64().unwrap_or(0) as usize;
                                                let check_max = tmp_value[5].as_u64().unwrap_or(0) as usize;
                                                Some(ForceStateOccupation::init(
                                                    String::new(),
                                                    ref_state,
                                                    ref_spin,
                                                    target_spin,
                                                    force_occ,
                                                    check_min,
                                                    check_max,
                                                ))
                                            }
                                        }
                                    }
                                    7 => {
                                        // ["ref.hdf5", ref_state, ref_spin, target_spin, force_occ, min, max]
                                        let reference = tmp_value[0].as_str().unwrap_or("none").to_string();
                                        let ref_state = tmp_value[1].as_u64().unwrap_or(0) as usize;
                                        let ref_spin = tmp_value[2].as_u64().unwrap_or(0) as usize;
                                        let target_spin = tmp_value[3].as_u64().unwrap_or(0) as usize;
                                        let force_occ = tmp_value[4].as_f64().unwrap_or(0.0);
                                        let check_min = tmp_value[5].as_u64().unwrap_or(0) as usize;
                                        let check_max = tmp_value[6].as_u64().unwrap_or(0) as usize;
                                        Some(ForceStateOccupation::init(
                                            reference,
                                            ref_state,
                                            ref_spin,
                                            target_spin,
                                            force_occ,
                                            check_min,
                                            check_max,
                                        ))
                                    }
                                    _ => {
                                        panic!("ERROR:: incorrect force_state_occupation setting: {:?}", &tmp_value);
                                    }
                                }
                            }
                            _ => None,
                        };
                        if let Some(tmp_obj) = tmp_obj {
                            tmp_vec.push(tmp_obj);
                        }
                    });
                    tmp_vec
                }
                _ => vec![],
            };

            //
            tmp_input.auxiliary_reference_states = match tmp_ctrl.get("auxiliary_reference_states").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_chk) => vec![(String::from("none"),0)],
                serde_json::Value::Array(tmp_op) => {
                    let mut tmp_files = vec![];
                    tmp_op.iter().for_each(|x| {
                        match x {
                            serde_json::Value::String(tmp_str) => {tmp_files.push((tmp_str.clone(),0))},
                            serde_json::Value::Array(tmp_value) => {
                                let aux_file_name = tmp_value[0].as_str().unwrap().to_string();
                                let global_start = tmp_value[1].as_u64().unwrap() as usize;
                                tmp_files.push((aux_file_name,global_start));
                            },
                            _ => {}
                        }
                    });
                    tmp_files
                },
                other => Vec::new(),
            };
            tmp_input.rpa_de_excitation_parameters = match tmp_ctrl.get("rpa_de_excitation_parameters").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Array(tmp_op) => {
                    if tmp_op.len() == 4 {
                        let mut tmp_array = [0.0;4];
                        tmp_array.iter_mut().zip(tmp_op.iter()).for_each(|(to, from)| {
                            *to = from.as_f64().unwrap()
                        });
                        Some(tmp_array)
                    } else {
                        None
                    }
                },
                other => None,
            };
            // ================================================
            //  Keywords associated with the post-SCF analyais
            // ================================================
            tmp_input.pbc_eigenval = match tmp_ctrl.get("pbc_eigenval").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_op) => {
                    let tmp = match tmp_op.as_str() {
                        "none" => None,
                        "None" => None,
                        _ => Some(String::from(tmp_op))
                    };
                    tmp
                }
                _ => None
            };
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
            tmp_input.outname = match tmp_ctrl.get("outname").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp) => Some(tmp.to_string()),
                _ => None
            };
            tmp_input.cube_orb_type = match tmp_ctrl.get("cube_orb_type").unwrap_or(&serde_json::Value::Null) {
               serde_json::Value::String(tmp_type) => {
                    String::from(tmp_type).to_lowercase()
               },
               other => {
                    String::from("wavefunction")
               }
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
                    let mut tmp_indices = vec![[0;3];tmp_op.len()];
                    tmp_indices.iter_mut().zip(tmp_op.iter()).for_each(|(to, from)| {
                        let tmp_to = match from {
                            serde_json::Value::Array(tmp_opp) => {
                                let mut tmp_array = [0_usize;3];
                                tmp_array.iter_mut().zip(tmp_opp[0..3].iter()).for_each(|(to, from)| {
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
            tmp_input.deep_pot = match tmp_ctrl.get("deep_potential").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_str) => {*tmp_str},
                other => {false},
            };
            tmp_input.run_lib_rint = match tmp_ctrl.get("run_lib_rint").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_bool) => {*tmp_bool},
                serde_json::Value::String(tmp_str) => tmp_str.to_lowercase().parse().unwrap_or(false),
                _ => false,
            };
            tmp_input.bench_eps = match tmp_ctrl.get("bench_eps").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_str) => {*tmp_str},
                other => {false},
            };

            // for atom_sad setting
            tmp_input.atom_sad = match tmp_ctrl.get("atom_sad").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_str) => {*tmp_str},
                other => {false},
            };

            tmp_input.max_memory = match tmp_ctrl.get("max_memory").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(tmp_num) => Some(tmp_num.as_f64().unwrap()),
                other => None,
            };
            tmp_input.abort_on_mem_exceed = tmp_ctrl.get("abort_on_mem_exceed").map(serde_from_value).unwrap_or(true);

            // for smearing
            tmp_input.smear = match tmp_ctrl.get("smear").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_type) => {
                    let tmp_smear = tmp_type.to_lowercase();
                    if tmp_smear.eq("fermi") {
                        Some(SmearingType::FERMI)
                    } else if tmp_smear.eq("gaussian") || tmp_smear.eq("gauss") {
                        Some(SmearingType::GAUSSIAN)
                    } else {
                        warn!("unknown smear type '{}', smearing not turned on.", tmp_type);
                        None
                    }
                }
                _ => None,
            };

            tmp_input.smear_sigma = match tmp_ctrl.get("smear_sigma").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {
                    let num = tmp_str.to_lowercase().parse().unwrap_or(0.0);
                    if num == 0.0 { None } else { Some(num) }
                },
                serde_json::Value::Number(tmp_num) => {
                    let num = tmp_num.as_f64().unwrap_or(0.0);
                    if num == 0.0 { None } else { Some(num) }
                },
                _ => None,
            };

            // for smear annealing; default = false
            tmp_input.smear_anneal = match tmp_ctrl.get("smear_anneal").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_bool) => *tmp_bool,
                serde_json::Value::String(tmp_str) => tmp_str.to_lowercase().parse().unwrap_or(false),
                _ => false,
            };

            // for smear annealing minimum sigma
            tmp_input.smear_sigma_min = match tmp_ctrl.get("smear_sigma_min").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {
                    let num = tmp_str.to_lowercase().parse().unwrap_or(0.0);
                    if num == 0.0 { None } else { Some(num) }
                },
                serde_json::Value::Number(tmp_num) => {
                    let num = tmp_num.as_f64().unwrap_or(0.0);
                    if num == 0.0 { None } else { Some(num) }
                },
                _ => None,
            };

            // for EDIIS penalty parameter
            tmp_input.ediis_penalty = match tmp_ctrl.get("ediis_penalty").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {
                    let num = tmp_str.to_lowercase().parse().unwrap_or(0.0);
                    if num == 0.0 { None } else { Some(num) }
                },
                serde_json::Value::Number(tmp_num) => {
                    let num = tmp_num.as_f64().unwrap_or(0.0);
                    if num == 0.0 { None } else { Some(num) }
                },
                _ => None,
            };

            // for EDIIS switch gap
            tmp_input.ediis_switch_gap = match tmp_ctrl.get("ediis_switch_gap").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {
                    let num = tmp_str.to_lowercase().parse().unwrap_or(0.0);
                    if num == 0.0 { None } else { Some(num) }
                },
                serde_json::Value::Number(tmp_num) => {
                    let num = tmp_num.as_f64().unwrap_or(0.0);
                    if num == 0.0 { None } else { Some(num) }
                },
                _ => None,
            };

            // for ADIIS penalty parameter
            tmp_input.adiis_penalty = match tmp_ctrl.get("adiis_penalty").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {
                    let num = tmp_str.to_lowercase().parse().unwrap_or(0.0);
                    if num == 0.0 { None } else { Some(num) }
                },
                serde_json::Value::Number(tmp_num) => {
                    let num = tmp_num.as_f64().unwrap_or(0.0);
                    if num == 0.0 { None } else { Some(num) }
                },
                _ => None,
            };

            // for guess_mix setting; default = False
            tmp_input.guess_mix = match tmp_ctrl.get("guess_mix").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_bool) => *tmp_bool,
                serde_json::Value::String(tmp_str) => tmp_str.to_lowercase().parse().unwrap_or(false),
                _ => false,
            };
            
            // for guess_mix_theta_deg: support number, string, or array; default to [15.0, 15.0]
            tmp_input.guess_mix_theta_deg = match tmp_ctrl.get("guess_mix_theta_deg").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => vec![n.as_f64().unwrap_or(15.0); 2],
                serde_json::Value::String(s) => {
                    let val = s.parse::<f64>().unwrap_or(15.0);
                    vec![val; 2]
                }
                serde_json::Value::Array(arr) => {
                    let mut vals = arr.iter().filter_map(|v| v.as_f64()).collect::<Vec<f64>>();
                    if vals.len() == 1 { vec![vals[0]; 2] }
                    else { vals.truncate(2); vals }
                }
                _ => vec![15.0, 15.0],
            };

            // for start_mix_cycle: support number or string; default = 0
            tmp_input.start_mix_cycle = match tmp_ctrl.get("start_mix_cycle").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_i64().unwrap_or(0) as usize,
                serde_json::Value::String(s) => {s.parse::<usize>().unwrap_or(0_usize)}
                _ => 0_usize,
            };

            tmp_input.spin_correction_scheme = match tmp_ctrl.get("spin_correction_scheme").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_emp) => {Some(tmp_emp.to_lowercase())},
                other => {None},
            };

            tmp_input.yamaguchi_triplet_type = match tmp_ctrl.get("yamaguchi_triplet_type").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_emp) => {Some(tmp_emp.to_lowercase())},
                other => {None},
            };

            // opt_engine: available options: "lbfgs", "geometric-pyo3"; default: "geometric-pyo3"
            tmp_input.opt_engine = match tmp_ctrl.get("opt_engine").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => { 
                    let s = tmp_str.to_lowercase();
                    match s.to_lowercase().as_str() {
                        "lbfgs"  => Some(s.to_string()),
                        "geometric_pyo3" | "geometric-pyo3" => Some("geometric_pyo3".to_string()),
                        _ => panic!("Not recognized option for opt_engine: {}", s),
                    }
                },
                serde_json::Value::Null => { None },
                _ => panic!("Not recognized type for opt_engine"),
            };

            tmp_input.stop_at = match tmp_ctrl.get("stop_at").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => { Some(tmp_str.to_lowercase()) },
                other => None,
            };

            tmp_input.xc_parser = match tmp_ctrl.get("xc_parser").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => { tmp_str.to_lowercase() },
                other => String::from("legacy"),
            };
            
            tmp_input.analdrv_tasks = match tmp_ctrl.get("analdrv_tasks").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Null => vec![],
                serde_json::Value::String(tmp_str) => vec![serde_from_value(&tmp_str.to_string().into())],
                serde_json::Value::Array(tmp_arr) => tmp_arr.iter().map(|x| serde_from_value(x)).collect(),
                _ => panic!("analdrv_tasks must be a string or an array of strings"),
            };
            
            //===========================================================
            // Global check of ctrl keywords and futher modification
            //============================================================
            if tmp_input.even_tempered_basis == true {
                if tmp_input.etb_beta<=1.0f64 {
                    warn!("etb_beta cannot be below 1.0. REST will use etb_beta=2.0 instead in this calculation");
                    tmp_input.etb_beta=2.0f64;
                }
                //if tmp_input.print_level>0 {
                //    println!("Even tempered basis generation starts at: {}", tmp_input.etb_start_atom_number);
                //    println!("Even tempered basis beta is: {}", tmp_input.etb_beta);
                //}
            }
        },
        other => {
            panic!("Error:: no 'ctrl' keyword or some inproper settings of the 'ctrl' keyword in the input file")
        },
    }

    Ok(tmp_input)
}
