#![allow(non_snake_case)]
// Solvent gradient module
pub mod grad;
pub mod surface_utils;
pub mod smd_cds;
pub use surface_utils::*;

use std::fmt;
//use std::f64::consts;
use std::time::Instant;
//use rest_libcint::gto::prelude_dev::X;
//use rest_libcint::CintType;
use rest_libcint::prelude::*;
use rest_libcint_wrapper::*;
//use rstsr_core::prelude_dev::shape;
use rstsr::prelude::*;
use std::collections::HashMap;
use std::f64::consts;
use rest_libcint::gto::prelude_dev::X;
use rest_libcint::{CINTR2CDATA, CintType};
use serde::{Deserialize, Serialize};
use tensors::{map_full_to_upper, map_upper_to_full, MatrixFull, MatrixUpper, ri, BasicMatUp, BasicMatrix, MathMatrix, MatrixFullSlice, MatrixUpperSlice};
//use rstsr as rt;
use tensors::matrix_blas_lapack::{_dgemm, _dgemm_full, _dgemm_scaled, omp_get_num_threads_wrapper, omp_set_num_threads_wrapper};
use crate::molecule_io::Molecule;
//use crate::geom_io::{GeomCell, get_mass_charge};
//use crate::constants::solvent as data;
//use crate::dft::gen_grids::angular_grid;
use rayon::prelude::*;
use rand::Rng;
//use rand::thread_rng;
use crate::utilities::memory_batch::*;

type TsrView<'a, T> = TensorView<'a, T, DeviceBLAS, IxD>;
type Tsr<T> = Tensor<T, DeviceBLAS, IxD>;
fn shuffle<T>(slice: &mut [T], rng: &mut impl Rng) {
    for i in (1..slice.len()).rev() {
        let j = rng.random_range(0..(i + 1));
        slice.swap(i, j);
    }
}
//=============================================================================
//  Configuration Structures
//=============================================================================

/// PCM method enumeration

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum PcmMethod {
    #[default]
    CPCM,
    COSMO,
    IEFPCM,
    SSVPE,
    /// SMD = IEFPCM (electrostatics) + CDS (cavitation-dispersion-solvent structure)
    SMD,
}

impl fmt::Display for PcmMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PcmMethod::COSMO => write!(f, "COSMO"),
            PcmMethod::CPCM => write!(f, "CPCM"),
            PcmMethod::IEFPCM => write!(f, "IEFPCM"),
            PcmMethod::SSVPE => write!(f, "SSVPE"),
            PcmMethod::SMD => write!(f, "SMD"),
        }
    }
}

impl<'d> Deserialize<'d> for PcmMethod {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'d>,
    {
        let s = String::deserialize(deserializer)?;
        match s.to_uppercase().as_str() {
            "CPCM" => Ok(PcmMethod::CPCM),
            "COSMO" => Ok(PcmMethod::COSMO),
            "IEFPCM" => Ok(PcmMethod::IEFPCM),
            "SSVPE" | "SS(V)PE" => Ok(PcmMethod::SSVPE),
            "SMD" => Ok(PcmMethod::SMD),
            _ => Err(serde::de::Error::custom(format!("Unknown PCM method: {}", s))),
        }
    }
}

/// PCM configuration
/// #[non_exhaustive]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct PcmObjectCfg {
    pub method: PcmMethod,
    pub epsilon: f64,
    /// SMD solvent descriptors [n, n25, α, β, γ, ε, φ, ψ].
    /// Default: water. Only used when method == SMD.
    pub solvent_descriptors: [f64; 8],
    /// SMD solvent type: 1 = water (ICDS=1, pre-tabulated sigma), 2 = non-aqueous (ICDS=2).
    pub icds: i32,
    /// SMD cavity/CDS radii scheme (bondi default / uff_mixed, see `SmdCavityRadii`).
    pub smd_cavity_radii: SmdCavityRadii,
}

impl PcmObjectCfg {
    pub fn build(
        method: PcmMethod,
        epsilon: f64,
        solvent_descriptors: [f64; 8],
        icds: i32,
        smd_cavity_radii: SmdCavityRadii,
    ) -> Self {
        PcmObjectCfg { method, epsilon, solvent_descriptors, icds, smd_cavity_radii }
    }
}

impl Default for PcmObjectCfg {
    fn default() -> Self {
        PcmObjectCfg {
            method: PcmMethod::default(),
            epsilon: 78.3553,
            solvent_descriptors: SMD_ERROR_DESCRIPTORS,
            icds: 0,
            smd_cavity_radii: SmdCavityRadii::Bondi,
        }
    }
}

/// Water solvent descriptors for SMD: [n, n25, α, β, γ, ε, φ, ψ]
pub const SMD_WATER_DESCRIPTORS: [f64; 8] = [1.3328, 1.3323, 0.82, 0.35, -1.0, 78.355, -1.0, -1.0];

pub const SMD_ERROR_DESCRIPTORS: [f64; 8] = [-1.0; 8];

/// Fuzzy water check — only compares n, α, ε. Sentinel fields ignored.
fn is_water_descriptor(d: &[f64; 8]) -> bool {
    (d[0] - 1.3328).abs() < 0.01
    && (d[2] - 0.82).abs() < 0.01
    && (d[5] - 78.355).abs() < 1.0
}

/// Surface switch function type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize)]
pub enum SurfaceSwitchType {
    SWIG,
    #[default]
    ISWIG,
}

impl<'d> Deserialize<'d> for SurfaceSwitchType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'d>,
    {
        let s = String::deserialize(deserializer)?;
        match s.to_uppercase().as_str() {
            "SWIG" => Ok(SurfaceSwitchType::SWIG),
            "ISWIG" => Ok(SurfaceSwitchType::ISWIG),
            _ => Err(serde::de::Error::custom(format!("Unknown switch type: {}", s))),
        }
    }
}


//=============================================================================
//  PCM Static Structures
//=============================================================================

/// Static PCM data (computed once per calculation, geometry-dependent only)
#[derive(Clone)]
pub struct PcmStatic {
    pub A: Vec<f64>,
    pub D: MatrixFull<f64>,
    pub S: MatrixFull<f64>,
    pub K: MatrixFull<f64>,
    pub R: MatrixFull<f64>,
    pub f_epsilon: f64,
    pub v_grids_n: Vec<f64>,
    pub K_ipiv: Vec<i32>,
    pub K_initial: MatrixFull<f64>,
    /// Lower Cholesky factor of the auxiliary-basis Coulomb metric `V_PQ = (P|Q)`
    /// (with the same `1e-12 · max diag` regularization the per-iteration code used).
    /// Geometry-only, so the RI-PCM path factors it once here instead of rebuilding
    /// and re-factorizing an `naux × naux` matrix twice on every SCF iteration.
    pub v_pq_chol: MatrixFull<f64>,
    /// `M` of the RI-J/K metric decomposition (`cderi = (μν|P)·M`), built through the
    /// very same routine the resident `SCF::rimatr` was built with, so that the two are
    /// consistent. `J^{-1/2}` for the (default) `eig` policy; `None` for `cd` and for
    /// any other policy the RI-reuse prototype does not cover yet.
    ///
    /// Prototype: used by the `REST_PCM_RI_REUSE=1` path to contract against the
    /// resident `rimatr` instead of re-evaluating the 3c integrals every iteration.
    pub j2c_factor: Option<MatrixFull<f64>>,
    /// SMD CDS energy (Hartree), total SASA (Å²), and gradient ([3] × [natm] Hartree/Bohr).
    /// Populated when method == SMD.
    pub e_cds: Option<f64>,
    pub tarea: Option<f64>,
    pub d_cds: Option<Vec<MatrixFull<f64>>>,
}

impl PcmStatic{
    pub fn build_pcm_static(surface: &SurfaceVdwGaussian, cfg: &PcmObjectCfg, mol: &Molecule) -> PcmStatic {
        let mut cint_data = mol.initialize_cint(false);
        let (A, D, S) = get_A_D_S(&surface);
        let (K, R, f_epsilon, K_ipiv, K_initial) = get_K_R_f(&surface, cfg.method, cfg.epsilon);
        let v_grids_n = get_v_grids_n(&surface, &cint_data);
        let v_pq_chol = build_v_pq_chol(mol);
        // Prototype: the factor that the resident RI tensor was built with. Restricted
        // to the `eig` policy (the default), where `M = J^{-1/2}` is symmetric and both
        // PCM contractions reduce to `M` applied once plus one dgemv against `rimatr`.
        // Only built when the prototype path is requested: the eigendecomposition of
        // the `naux × naux` metric is not free, and every other path ignores it.
        let j2c_factor = if std::env::var("REST_PCM_RI_REUSE").map(|v| v == "1").unwrap_or(false) {
            match mol.ctrl.j2c_decomp.policy {
                crate::ri_jk::J2CDecompPolicy::Eig => {
                    let j2c = mol.int_ij_aux_columb();
                    Some(crate::mpi_io::prepare_j2c_solve_factor(&j2c, &mol.ctrl.j2c_decomp))
                }
                _ => None,
            }
        } else {
            None
        };
        let (e_cds, tarea, d_cds) = if cfg.method == PcmMethod::SMD {
            let (e, a, d) = compute_cds_from_surface(surface, cfg);
            let natm = surface.atomic_num.len();
            let d_cds: Vec<MatrixFull<f64>> = (0..3).map(|dir| {
                let mut m = MatrixFull::new([natm, 1], 0.0);
                for i in 0..natm { m[[i, 0]] = d[i][dir]; }
                m
            }).collect();
            (Some(e), Some(a), Some(d_cds))
        } else {
            (None, None, None)
        };
        PcmStatic { A, D, S, K, R, f_epsilon, v_grids_n, K_ipiv, K_initial, v_pq_chol, j2c_factor, e_cds, tarea, d_cds }
    }
}

/// Build and factor the auxiliary-basis Coulomb metric `V_PQ = (P|Q)` used by the
/// RI-PCM path, returning its lower Cholesky factor (`V = L·Lᵀ`, `L` in the lower
/// triangle — the form [`solve_cholesky_factorized`] expects).
///
/// Both the metric and its factorization depend only on the geometry and the basis,
/// so this runs once per calculation (from [`PcmStatic::build_pcm_static`]) instead
/// of on every SCF iteration. The diagonal regularization is the one the
/// per-iteration code applied: `1e-12 · max(1, max diag)`.
pub fn build_v_pq_chol(mol: &Molecule) -> MatrixFull<f64> {
    let cint_data_ri = mol.initialize_cint(true);
    let nbas = mol.cint_bas.len() as i32;
    let nbas_aux = mol.cint_aux_bas.len() as i32;
    // Geometry-only and rank-local, but the message describes the whole calculation.
    let print_root = mol.mpi_data.as_ref().map_or(true, |m| m.rank == 0);

    let shls_slice_vpq = [[nbas, nbas + nbas_aux], [nbas, nbas + nbas_aux]];
    let (v_pq_out, v_pq_shape) = cint_data_ri.integral_s1::<int2c2e>(Some(&shls_slice_vpq));
    let mut v_pq = MatrixFull::from_vec([v_pq_shape[0], v_pq_shape[1]], v_pq_out).unwrap();
    let naux = v_pq_shape[0];
    if print_root {
        println!("naux= {}", naux);
    }

    // Regularize V_{PQ} to avoid numerical issues in Cholesky. The historical value is
    // `1e-12 · max(1, max diag)`; `REST_PCM_VPQ_REG` overrides the relative factor so a
    // run can quantify the one deliberate difference between this path and the
    // `REST_PCM_RI_REUSE` path: the reuse path inverts the *plain* metric that the
    // resident RI-J/K tensor was built with, while this one solves with the
    // regularized metric (see `PcmRiReuse`).
    let reg_rel = std::env::var("REST_PCM_VPQ_REG")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(1.0e-12);
    let v_diag_max = (0..naux).fold(0.0f64, |acc, i| acc.max(v_pq[(i, i)]));
    let lambda = reg_rel * v_diag_max.max(1.0);
    for i in 0..naux {
        v_pq[(i, i)] += lambda;
    }
    if print_root {
        println!("V_PQ regularization: lambda = {:10.6e}, diag_max = {:10.6e}", lambda, v_diag_max);
    }

    v_pq.to_matrixfullslicemut().lapack_dpotrf(b'L');
    v_pq
}

pub fn get_A_D_S(surface: &SurfaceVdwGaussian) -> (Vec<f64>, MatrixFull<f64>, MatrixFull<f64>) {
    const PI: f64 = std::f64::consts::PI;

    let R_vdw = &surface.surface_calc.R_vdw;
    let switch_fun = &surface.surface_calc.switch_fun;
    let weights = &surface.surface_calc.weights;

    //The matrix A is diagonal and consists of the surface element areas, ai.
    let A: Vec<f64> = weights
    .iter().zip(R_vdw.iter()).zip(switch_fun.iter())
    .map(|((&w,&r),&s)| w * r.powi(2) * s).collect();

    let charge_exp = &surface.surface_calc.charge_exp;
    let grid_coords = &surface.surface_calc.grid_coords;
    let n_grids = grid_coords.len();
    let grid_capacity = n_grids * (n_grids + 1) /2;
    let norm_vec = &surface.surface_calc.norm_vec;
    
    if n_grids == 0 {
        panic!("ERROR: No surface grid points generated!");
    }

    let mut zeta_ij = Vec::with_capacity(grid_capacity);
    for j in 0..n_grids{
        for i in 0..j+1{
            zeta_ij.push(charge_exp[i]*charge_exp[j]/(charge_exp[i].powi(2)+charge_exp[j].powi(2)).sqrt() );
        }
    }

    let mut r_ij = Vec::with_capacity(grid_capacity);
    for j in 0..n_grids{
        for i in 0..j+1{
            if i == j {
                r_ij.push(1.0);
            } else {
                r_ij.push(dist_3d(&grid_coords[i], &grid_coords[j]));
            }
        }
    }
    //let r_ij = MatrixUpper::from_vec(grid_capacity, tmp_r_ij).unwrap();
    //let r_ij = r_ij.to_matrixfull();
    let zeta_r_ij: Vec<f64> = r_ij.iter()
        .zip(zeta_ij.iter()).map(|(&r, &z)| z*r).collect();
    
    // S_ij = erf(zeta_ij * r_ij) / r_ij, with S_ii = charge_exp[i] * (2/pi)^0.5 / switch_fun[i]
    let mut tmp_S_ij = zeta_r_ij.iter().zip(r_ij.iter())
        .map(|(&z, &r)| libm::erf(z) / r).collect();
    let mut S_ij: MatrixFull<f64> = MatrixUpper::from_vec(grid_capacity, tmp_S_ij).unwrap().to_matrixfull().unwrap();
    for i in 0..n_grids{
        let s_ii = charge_exp[i] * (2.0 / PI).sqrt() / switch_fun[i];
        S_ij[(i,i)] = s_ii;
    }

    let r_ij_matrix: MatrixFull<f64> = MatrixUpper::from_vec(grid_capacity, r_ij).unwrap().to_matrixfull().unwrap();
    let zeta_r_ij_matrix: MatrixFull<f64> = MatrixUpper::from_vec(grid_capacity, zeta_r_ij).unwrap().to_matrixfull().unwrap();
    // n_r_ij = \vec{n_j} * \vec{r_ij}, where \vec{n_j} is the normal vector at grid point j, and \vec{r_ij} is the vector from grid point j to grid point i.
    let mut n_r_ij: Vec<f64> = Vec::with_capacity(n_grids * n_grids);
    for j in 0..n_grids{
        for i in 0..n_grids{
            let dx = grid_coords[i][0] - grid_coords[j][0];
            let dy = grid_coords[i][1] - grid_coords[j][1];
            let dz = grid_coords[i][2] - grid_coords[j][2];
            n_r_ij.push(dx * norm_vec[j].0 + dy * norm_vec[j].1 + dz * norm_vec[j].2);
        }
    }

    //r_scale = \vec{n_j} * \vec{r_ij}/ r_ij^3, 
    let r_scale: Vec<f64> = n_r_ij.iter().zip(r_ij_matrix.iter())
        .map(|(&n, &r)| n / r.powi(3)).collect();
    let f_zeta_r_ij: Vec<f64> = zeta_r_ij_matrix.iter()
        .map(|&x| libm::erf(x) - 2.0 / PI.sqrt() * x * f64::exp(-x.powi(2)))
        .collect();
    let tmp_D_ij: Vec<f64> = f_zeta_r_ij.iter().zip(r_scale.iter())
        .map(|(&f, &scale)| f * scale).collect();
    let mut D_ij = MatrixFull::from_vec([n_grids, n_grids], tmp_D_ij).unwrap();
    for i in 0..n_grids{
        let d_ii = -charge_exp[i]* (2.0 / PI).sqrt() / (2.0 * R_vdw[i]);
        D_ij[(i,i)] = d_ii;
    }

    (A, D_ij, S_ij)

}

pub fn get_K_R_f(surface: &SurfaceVdwGaussian, method: PcmMethod, epsilon: f64) -> (MatrixFull<f64>, MatrixFull<f64>, f64, Vec<i32>, MatrixFull<f64>) {
    const PI: f64 = std::f64::consts::PI;

    //let device = A.device();
    let (A, D, S) = get_A_D_S(&surface);
    let ngrids = A.len();
    let mut R = MatrixFull::<f64>::new([ngrids, ngrids], 0.0);
    for i in 0..ngrids {
        R[(i, i)] = 1.0 ;
    }
    //let A_matr = MatrixFull::from_vec([ngrids, 1], A.clone()).unwrap();

    let (K, R, f_epsilon, K_ipiv, K_initial) = match method {
        PcmMethod::CPCM => {
            let f_epsilon = (epsilon - 1.0) / epsilon;
            let mut K_initial = S.clone();
            R.self_multiple(-f_epsilon);
            let (K, K_ipiv) = K_initial.clone().to_matrixfullslicemut().lapack_dgetrf_full().unwrap();
            (K, R, f_epsilon, K_ipiv, K_initial)
        },
        PcmMethod::COSMO => {
            let f_epsilon = (epsilon - 1.0) / (epsilon + 0.5);
            let mut K_initial = S.clone();
            R.self_multiple(-f_epsilon);
            let (K, K_ipiv) = K_initial.clone().to_matrixfullslicemut().lapack_dgetrf_full().unwrap();
            (K, R, f_epsilon, K_ipiv, K_initial)
        },
        PcmMethod::IEFPCM | PcmMethod::SMD => {
            let f_epsilon = (epsilon - 1.0) / (epsilon + 1.0);
            let mut DA = D.clone();
            for j in 0..ngrids{
                for i in 0..ngrids{
                    DA[(i,j)] = DA[(i,j)] * A[j];
                }
            }
            //let DA = _dgemm_scaled(&D, 'N', &A_matr, 'N', 1.0);
            let DAS = _dgemm_scaled(&DA, 'N', &S, 'N', 1.0);
            let mut K_initial = S.scaled_add(&DAS, -f_epsilon / (2.0 * PI)).unwrap();
            //let K = S - f_epsilon / (2.0 * PI) * DAS;
            //let R = -f_epsilon * (rt::eye((ngrids, device)) - 1.0 / (2.0 * PI) * DA);
            R.self_scaled_add(&DA, - 1.0 / (2.0 * PI));
            R.self_multiple(-f_epsilon);
            let (K, K_ipiv) = K_initial.clone().to_matrixfullslicemut().lapack_dgetrf_full().unwrap();
            (K, R, f_epsilon, K_ipiv, K_initial)
        },
        PcmMethod::SSVPE => {
            let f_epsilon = (epsilon - 1.0) / (epsilon + 1.0);
            let mut DA = D.clone();
            for j in 0..ngrids{
                for i in 0..ngrids{
                    DA[(i,j)] = DA[(i,j)] * A[j];
                }
            }
            //let DA =  _dgemm_scaled(&D, 'N', &A_matr, 'N', 1.0);;
            let mut DAS = _dgemm_scaled(&DA, 'N', &S, 'N', 1.0);
            DAS.self_add(&DAS.transpose());
            let mut K_initial = S.scaled_add(&DAS,  - f_epsilon / (4.0 * PI)).unwrap();
            //let K = S - f_epsilon / (4.0 * PI) * (&DAS + DAS.t());
            R.self_scaled_add(&DA, - 1.0 / (2.0 * PI));
            R.self_multiple(-f_epsilon);
            //let R = -f_epsilon * (rt::eye((ngrids, device)) - 1.0 / (2.0 * PI) * DA);
            let (K, K_ipiv) = K_initial.clone().to_matrixfullslicemut().lapack_dgetrf_full().unwrap();
            (K, R, f_epsilon, K_ipiv, K_initial)
        },
    };
    warn_if_near_singular(&K, method);
    (K, R, f_epsilon, K_ipiv, K_initial)
}

/// Warn when the LU-factored PCM matrix K is (nearly) singular.
///
/// `K` is the output of `lapack_dgetrf_full`, whose diagonal holds |U_ii|.
/// A tiny min/max ratio means the surface discretization produced a degenerate
/// matrix; subsequent `solve_lu` calls will lose accuracy or fail.
fn warn_if_near_singular(K: &MatrixFull<f64>, method: PcmMethod) {
    let n = K.size[0];
    if n == 0 {
        return;
    }
    let mut diag_min = f64::MAX;
    let mut diag_max = 0.0f64;
    for i in 0..n {
        let d = K[(i, i)].abs();
        diag_min = diag_min.min(d);
        diag_max = diag_max.max(d);
    }
    let ratio = if diag_max > 0.0 { diag_min / diag_max } else { 0.0 };
    if ratio < 1e-12 {
        println!(
            "WARNING: PCM matrix K is nearly singular ({}): min|U_ii|={:10.3e}, max|U_ii|={:10.3e}, min/max={:10.3e}",
            method, diag_min, diag_max, ratio
        );
    }
}

pub fn get_v_grids_n(surface: &SurfaceVdwGaussian, cint_data: &CINTR2CDATA) -> Vec<f64> {
//compute the nuclear contribution to the potential on the surface grid points
    let grid_coords = &surface.surface_calc.grid_coords;
    let charge_exp = &surface.surface_calc.charge_exp;
    let exponents = charge_exp.iter().map(|&x| x * x).collect::<Vec<f64>>();
    let atom_coords = &cint_data.atom_coords();
    let atom_charges = &cint_data.atom_charges();

    let fake_chg_data = CINTR2CDATA::fakemol_for_charges(grid_coords, exponents.as_slice());
    let fake_nuc_data = CINTR2CDATA::fakemol_for_charges(atom_coords, None);
    // Electron distribution satisfy the Gaussian distribution
    //g_i(\vec{r})=q_i(\xi_i^2/\pi)^{3/2}exp(-\xi_i^2|\vec{r}-\vec{s}_i|^2).
    // compute \iint g_i(r1) * 1/|r1 - r2| * dr1 dr2
    let (tmpout, shape) = CINTR2CDATA::integrate_cross("int2c2e", [&fake_nuc_data, &fake_chg_data], None, None).into();
    let tmpshape = [shape[0],shape[1]];
    let tmp_v_ng = MatrixFull::from_vec(tmpshape, tmpout).unwrap();
    let mut v_ng = vec![];
    for j in 0..tmpshape[1]{
        let mut v_ng_j = 0.0;
        for i in 0..tmpshape[0]{
            v_ng_j  += tmp_v_ng[(i,j)] * atom_charges[i];
        }
        v_ng.push(v_ng_j);
    }

    v_ng
}


/// Full PCM object containing configuration, surface data, and static PCM data
#[non_exhaustive]
#[derive(Clone)]
pub struct PcmObject {
    pub cfg: PcmObjectCfg,
    pub surface: SurfaceVdwGaussian,
    pub pstatic: PcmStatic,
}

impl PcmObject {
    pub fn init_Pcm(cfg: PcmObjectCfg, surface: SurfaceVdwGaussian, pstatic: PcmStatic) -> Self {
        PcmObject { cfg, surface, pstatic }
    }

}

//=============================================================================
//  PCM SCF Structures
//=============================================================================

/// PCM data updated during SCF iterations
#[derive(Clone)]
pub struct PcmScf {
    pub v_grids_e: Vec<f64>,
    pub v_grids: Vec<f64>,
    pub veff: MatrixUpper<f64>,
    pub eng: f64,
    pub eng_nuc: f64,
    pub q_sym: MatrixFull<f64>,
}

impl PcmScf{
    pub fn get_pcm_refresh(
        surface: &SurfaceVdwGaussian,
        mol: &Molecule,
        dm: &Vec<MatrixFull<f64>>,
        K: &MatrixFull<f64>,
        K_ipiv: &Vec<i32>,
        R: &MatrixFull<f64>,
        v_grids_n: &Vec<f64>,
        v_pq_chol: &MatrixFull<f64>,
        ri_reuse: Option<&PcmRiReuse>,
        spin_channel: &usize,
        max_memory: &Option<f64>,
        chunk_size: &usize,
        solv_ri: bool,
        mpi_operator: &Option<crate::mpi_io::MPIOperator>,
    ) -> PcmScf {
        let dt_solv0 = time::Local::now();
        let print_root = mpi_operator.as_ref().map_or(true, |op| op.rank == 0);
        // RI-based PCM: use auxiliary basis for (μν|g_i) integrals
        let mut cint_data_ri = mol.initialize_cint(true);
        let mut aux_cint = mol.make_auxmol_fake().initialize_cint(false);
        let mut cint_data = mol.initialize_cint(false);
        // Pre-build integral optimizer for 3c2e integrals
        //let opt_3c = cint_data_ri.optimizer("int3c2e");
        //cint_data_ri.c_opt = Some(Arc::new(opt_3c));

        //let opt_3c_1 = cint_data.optimizer("int3c2e");
        //cint_data.c_opt = Some(Arc::new(opt_3c_1));
        if print_root {
            println!("nao= {}", cint_data_ri.nao());
        }
        // V_{PQ} = (P|Q) and its Cholesky factor are geometry-only; both are built
        // once in `PcmStatic::build_pcm_static` and passed in, instead of being
        // rebuilt and re-factorized here on every SCF iteration.

        let singular_msg = |which: &str| format!(
            "PCM solver failed: K is singular ({} returned None) — method={}, epsilon={}, ngrids={}, solv_ri={}",
            which, mol.ctrl.solvent_model, mol.ctrl.solv_epsilon, K.size[0], solv_ri
        );

        let (v_grids_e, v_grids_matrix, veff, q_sym, dt_4) = if solv_ri{

            let v_grids_e = get_v_grids_e(surface, mol, dm, spin_channel, max_memory, chunk_size,
                &cint_data_ri, &aux_cint, v_pq_chol, ri_reuse, mpi_operator);

            let dt_1 = time::Local::now();
            let timecost_vgrid_e = (dt_1.timestamp_millis()-dt_solv0.timestamp_millis()) as f64 /1000.0;
            println!("v_grids_e costs {:10.2} seconds.", timecost_vgrid_e);

            assert!(v_grids_e.len() == v_grids_n.len(), "Length mismatch: v_grids_e has {} elements but v_grids_n has {} elements", v_grids_e.len(), v_grids_n.len());
            let v_grids: Vec<f64> = v_grids_n.iter().zip(v_grids_e.iter()).map(|(&vn, &ve)| vn - ve).collect();
            let v_grids_matrix = MatrixFull::from_vec([v_grids.len(), 1], v_grids).unwrap();

            let dt_2 = time::Local::now();
            let timecost_vgrid = (dt_2.timestamp_millis()-dt_1.timestamp_millis()) as f64 /1000.0;
            println!("v_grids costs {:10.2} seconds.", timecost_vgrid);

            assert!(v_grids_matrix.size[0] == R.size[1], "Dimension mismatch: v_grids has {} rows but R has {} columns", v_grids_matrix.size[0], R.size[1]);
            let b = _dgemm_scaled(R, 'N', &v_grids_matrix, 'N', 1.0);
            let q = solve_lu(K, K_ipiv, &b.data)
                .unwrap_or_else(|| panic!("{}", singular_msg("solve_lu")));
            let q = MatrixFull::from_vec([q.len(), 1], q).unwrap();

            //solve K^T x = v_grids
            let vK_1 = solve_lu_transpose(K, K_ipiv, &v_grids_matrix.data)
                .unwrap_or_else(|| panic!("{}", singular_msg("solve_lu_transpose")));

            let vK_1 = MatrixFull::from_vec([vK_1.len(), 1], vK_1).unwrap();

            let dt_3 = time::Local::now();
            let timecost_solve = (dt_3.timestamp_millis()-dt_2.timestamp_millis()) as f64 /1000.0;
            println!("Solving linear equations costs {:10.2} seconds.", timecost_solve);

            // `Rᵀ · vK_1`. The transpose goes to BLAS as a flag: `R.transpose()`
            // would clone and re-permute the whole `[ngrids, ngrids]` matrix (several
            // GB) on every SCF iteration.
            let qt = _dgemm_scaled(R, 'T', &vK_1, 'N', 1.0);
            if qt.size[0] != q.size[0] {
            panic!("Dimension mismatch: qt has {} rows but q has {} rows", qt.size[0], q.size[0]);
            }
            let q_sym = (q + qt) * 0.5;

            let dt_4 = time::Local::now();
            let timecost_qsym = (dt_4.timestamp_millis()-dt_3.timestamp_millis()) as f64 /1000.0;
            println!("Symmetrization of q costs {:10.2} seconds.", timecost_qsym);

            let veff = get_veff_pcm_by_q(surface, mol, &q_sym.data, max_memory, chunk_size, &cint_data_ri, &aux_cint, v_pq_chol, ri_reuse, mpi_operator);
    
            (v_grids_e, v_grids_matrix, veff, q_sym, dt_4)
        } else {
            #[cfg(feature = "mpi")]
            let v_grids_e = {
                if mpi_operator.is_some() {
                    get_v_grids_e_old_mpi(surface, &cint_data, dm, spin_channel, max_memory, chunk_size, mpi_operator)
                } else {
                    get_v_grids_e_old(surface, &cint_data, dm, spin_channel, max_memory, chunk_size)
                }
            };
            #[cfg(not(feature = "mpi"))]
            let v_grids_e = get_v_grids_e_old(surface, &cint_data, dm, spin_channel, max_memory, chunk_size);

            let dt_1 = time::Local::now();
            let timecost_vgrid_e = (dt_1.timestamp_millis()-dt_solv0.timestamp_millis()) as f64 /1000.0;
            println!("v_grids_e costs {:10.2} seconds.", timecost_vgrid_e);

            assert!(v_grids_e.len() == v_grids_n.len(), "Length mismatch: v_grids_e has {} elements but v_grids_n has {} elements", v_grids_e.len(), v_grids_n.len());
            let v_grids: Vec<f64> = v_grids_n.iter().zip(v_grids_e.iter()).map(|(&vn, &ve)| vn - ve).collect();
            let v_grids_matrix = MatrixFull::from_vec([v_grids.len(), 1], v_grids).unwrap();

            let dt_2 = time::Local::now();
            let timecost_vgrid = (dt_2.timestamp_millis()-dt_1.timestamp_millis()) as f64 /1000.0;
            println!("v_grids costs {:10.2} seconds.", timecost_vgrid);

            assert!(v_grids_matrix.size[0] == R.size[1], "Dimension mismatch: v_grids has {} rows but R has {} columns", v_grids_matrix.size[0], R.size[1]);
            let b = _dgemm_scaled(R, 'N', &v_grids_matrix, 'N', 1.0);
            let q = solve_lu(K, K_ipiv, &b.data)
                .unwrap_or_else(|| panic!("{}", singular_msg("solve_lu")));
            let q = MatrixFull::from_vec([q.len(), 1], q).unwrap();

            //solve K^T x = v_grids
            let vK_1 = solve_lu_transpose(K, K_ipiv, &v_grids_matrix.data)
                .unwrap_or_else(|| panic!("{}", singular_msg("solve_lu_transpose")));

            let vK_1 = MatrixFull::from_vec([vK_1.len(), 1], vK_1).unwrap();

            let dt_3 = time::Local::now();
            let timecost_solve = (dt_3.timestamp_millis()-dt_2.timestamp_millis()) as f64 /1000.0;
            println!("Solving linear equations costs {:10.2} seconds.", timecost_solve);

            // `Rᵀ · vK_1`. The transpose goes to BLAS as a flag: `R.transpose()`
            // would clone and re-permute the whole `[ngrids, ngrids]` matrix (several
            // GB) on every SCF iteration.
            let qt = _dgemm_scaled(R, 'T', &vK_1, 'N', 1.0);
            if qt.size[0] != q.size[0] {
            panic!("Dimension mismatch: qt has {} rows but q has {} rows", qt.size[0], q.size[0]);
            }
            let q_sym = (q + qt) * 0.5;

            let dt_4 = time::Local::now();
            let timecost_qsym = (dt_4.timestamp_millis()-dt_3.timestamp_millis()) as f64 /1000.0;
            println!("Symmetrization of q costs {:10.2} seconds.", timecost_qsym);

            #[cfg(feature = "mpi")]
            let veff = {
                if mpi_operator.is_some() {
                    get_veff_pcm_by_q_old_mpi(surface, &cint_data, &q_sym.data, max_memory, chunk_size, mpi_operator)
                } else {
                    get_veff_pcm_by_q_old(surface, &cint_data, &q_sym.data, max_memory, chunk_size)
                }
            };
            #[cfg(not(feature = "mpi"))]
            let veff = get_veff_pcm_by_q_old(surface, &cint_data, &q_sym.data, max_memory, chunk_size);
            (v_grids_e, v_grids_matrix, veff, q_sym, dt_4)
        };

        let eng: f64 = 0.5 * q_sym.iter().zip(v_grids_matrix.iter())
            .map(|(&q, &v)| q * v)
            .sum::<f64>();

        let eng_nuc: f64 = 0.5 * q_sym.iter().zip(v_grids_n.iter())
            .map(|(&q, &vn)| q * vn)
            .sum::<f64>();
        let v_grids = v_grids_matrix.data;

        let dt_5 = time::Local::now();
        let timecost_veff = (dt_5.timestamp_millis()-dt_4.timestamp_millis()) as f64 /1000.0;
        println!("Computation of eng costs {:10.2} seconds.", timecost_veff);

        PcmScf{v_grids_e, v_grids, veff, eng, eng_nuc, q_sym}
    }

}   

/// Unified solvent chunk size (number of surface points per work batch).
///
/// A single ctrl knob (`solv_chunk`) controls the thread-level batch size for
/// the PCM energy (RI and non-RI) and gradient paths. The value is clamped by:
/// - **user knob**: `solv_chunk` (upper bound; default 16, benchmark inputs 64);
/// - **memory**: the ~500 MB in-flight tensor budget per process divided by the
///   thread count (`bytes_per_unit` bytes per surface point; gradient tensors
///   are 3× the energy ones);
/// - **load balance**: `range / (nthreads · K)` so every thread gets ≥K tasks;
/// - **floor**: 16 points, silently overridden by the memory cap when the two
///   conflict (per-batch overhead: fake molecule + integral call).
pub(crate) fn solvent_chunk(
    range: usize,          // surface points owned by this rank
    nao: usize,            // number of AO basis functions
    bytes_per_point: f64,  // tensor bytes per point as a multiple of nao² (8 energy, 24 gradient)
    solv_chunk: usize,     // ctrl `solv_chunk`
    nthreads: usize,       // rayon pool size
) -> usize {
    solvent_chunk_bytes(range, bytes_per_point * (nao * nao) as f64, solv_chunk, nthreads)
}

/// Bytes-based variant for tensor shapes whose per-point footprint is not
/// nao²-shaped: the RI paths hold `[naux, chunk]` blocks (8·naux bytes per
/// point), not `[nao², chunk]`. `bytes_per_unit` = in-flight tensor bytes
/// contributed by ONE surface point.
pub(crate) fn solvent_chunk_bytes(
    range: usize,          // surface points owned by this rank
    bytes_per_unit: f64,   // in-flight tensor bytes per surface point
    solv_chunk: usize,     // ctrl `solv_chunk`
    nthreads: usize,       // rayon pool size
) -> usize {
    const C_MIN: usize = 16;
    const K: usize = 2;
    let t = nthreads.max(1);
    let mem_cap = ((500_000_000.0 / bytes_per_unit) as usize / t).max(1);
    let hi = solv_chunk.min(mem_cap).max(1);
    let by_load = (range / (t * K)).max(1);
    let lo = C_MIN.min(range).max(1);
    by_load.max(lo).min(hi).max(1)
}

pub fn get_v_grids_e_old(
    surface: &SurfaceVdwGaussian,
    cint_data: &CINTR2CDATA,
    dm: &Vec<MatrixFull<f64>>,
    spin_channel: &usize,
    max_memory: &Option<f64>,
    chunk_size: &usize) -> Vec<f64> {
        
    let charge_exp = &surface.surface_calc.charge_exp;
    let grid_coords = &surface.surface_calc.grid_coords;
    
    let ngrids = charge_exp.len();
    let nao = cint_data.nao();

    let CHUNK = solvent_chunk(ngrids, nao, 8.0, *chunk_size, rayon::current_num_threads());
    let mut dm_vec = vec![0.0; nao * nao];
    for i_spin in 0..*spin_channel {
        for j in 0..nao {
            for i in 0..nao {
                dm_vec[j*nao + i] += dm[i_spin][(i,j)];
            }
        }
    }
    let dm_mat = MatrixFull::from_vec([1, nao*nao], dm_vec).unwrap();
    
    /**
    let mem_avail_mb = max_memory.map(|max_memory| {
        max_memory - detect_used_memory_mb("proc")
    }).unwrap_or_else(detect_available_memory_mb);

    let CHUNK: usize = ((0.7 * 0.01 * mem_avail_mb * 1024.0 * 1024.0)
    / (8.0 * (nao * nao) as f64))
    .max(1.0) as usize;
    println!("Available memory for v_grids_e computation: {:.2} MB, chunk size: {}", mem_avail_mb, CHUNK);
    
    let mut v_grids_e = Vec::with_capacity(ngrids);
    for p0 in (0..ngrids).step_by(CHUNK) {
        let p1 = (p0 + CHUNK).min(ngrids);
        let grid_coords_chunk = &grid_coords[p0..p1];
        let charge_exp_chunk = charge_exp[p0..p1].to_vec().iter().map(|x| x * x).collect::<Vec<f64>>();
        let fake_chg_data = CINTR2CDATA::fakemol_for_charges(grid_coords_chunk, charge_exp_chunk.as_slice());
        
        let (tmpout, shape) = CINTR2CDATA::integrate_cross("int3c2e", [cint_data, cint_data, &fake_chg_data], None, None).into();
        let tmpshape = [shape[0] * shape[1], shape[2]];
        let v_nj = MatrixFull::from_vec(tmpshape, tmpout).unwrap();

        let v_e_chunk = _dgemm_scaled(&dm_mat, 'N', &v_nj, 'N', 1.0);
        v_grids_e.extend(v_e_chunk.data);
    };
    **/

    let dt0 = time::Local::now();
    
    let mut v_grids_e = vec![0.0; ngrids];
    //let (sender, receiver) = channel();
    let default_omp_num_threads = omp_get_num_threads_wrapper();
    v_grids_e.par_chunks_mut(CHUNK).enumerate().for_each(|(v_chunk, idx)| {
        omp_set_num_threads_wrapper(1);
        let t0 = Instant::now();
        let p0 = v_chunk * CHUNK;
        let p1 = (p0 + CHUNK).min(ngrids);
        let grid_coords_chunk = &grid_coords[p0..p1];
        let mut charge_exp_chunk = Vec::with_capacity(p1 - p0);
        for &x in &charge_exp[p0..p1] {
            charge_exp_chunk.push(x * x);
        }
        //let charge_exp_chunk = charge_exp[p0..p1].to_vec().iter().map(|x| x * x).collect::<Vec<f64>>();
        let fake_chg_data = CINTR2CDATA::fakemol_for_charges(grid_coords_chunk, charge_exp_chunk.as_slice());
        
        let (tmpout, shape) = CINTR2CDATA::integrate_cross("int3c2e", [cint_data, cint_data, &fake_chg_data], None, None).into();
        let max_val = tmpout.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        let v_e_chunk = if max_val < 1.0e-12 {
            MatrixFull::new([1, idx.len()], 0.0)
        } else {
            let tmpshape = [shape[0] * shape[1], shape[2]];
            let v_nj = MatrixFull::from_vec(tmpshape, tmpout).unwrap();
            _dgemm_scaled(&dm_mat, 'N', &v_nj, 'N', 1.0)
        };  
        

        //println!("Chunk cycle of v_e costs {:10.6} seconds.", t0.elapsed().as_secs_f64());
        idx.iter_mut().zip(v_e_chunk.iter()).for_each(|(i, &v)| *i = v);

    });
    omp_set_num_threads_wrapper(default_omp_num_threads);

    let dt1 = time::Local::now();
    let timecost = (dt1.timestamp_millis()-dt0.timestamp_millis()) as f64 /1000.0;
    println!("Parallel computation of v_grids_e costs {:10.2} seconds.", timecost);
    
    
    v_grids_e
}

// ============================================================================
// MPI 并行化：表面格点 chunk 按 rank 分段 → 各 rank 本地计算 → allreduce
// 串行路径完全零改动（仅新增 _mpi 函数）
// ============================================================================

/// MPI 版 `get_v_grids_e_old`：表面点按 rank 分区（`average_distribution`），
/// rank 内按统一 chunk（`solvent_chunk`）用 Rayon 并行批处理（闭包内 BLAS 已 gate）。
/// 通信：allreduce(ngrids × 8B) —— 极小。
#[cfg(feature = "mpi")]
pub fn get_v_grids_e_old_mpi(
    surface: &SurfaceVdwGaussian,
    cint_data: &CINTR2CDATA,
    dm: &Vec<MatrixFull<f64>>,
    spin_channel: &usize,
    max_memory: &Option<f64>,
    chunk_size: &usize,
    mpi_operator: &Option<crate::mpi_io::MPIOperator>,
) -> Vec<f64> {
    use mpi::collective::SystemOperation;
    use mpi::traits::*;

    let mpi_op = match mpi_operator {
        Some(op) => op,
        None => return get_v_grids_e_old(surface, cint_data, dm, spin_channel, max_memory, chunk_size),
    };
    let my_rank = mpi_op.rank;
    let nproc = mpi_op.size;

    let charge_exp = &surface.surface_calc.charge_exp;
    let grid_coords = &surface.surface_calc.grid_coords;
    let ngrids = charge_exp.len();
    let nao = cint_data.nao();

    // Flatten DM
    let mut dm_vec = vec![0.0; nao * nao];
    for i_spin in 0..*spin_channel {
        for j in 0..nao {
            for i in 0..nao {
                dm_vec[j*nao + i] += dm[i_spin][(i,j)];
            }
        }
    }
    let dm_mat = MatrixFull::from_vec([1, nao*nao], dm_vec).unwrap();

    // MPI ownership: contiguous surface-point range per rank; within the range,
    // chunks follow the unified solvent chunk (thread batch size).
    let (g0, g1) = {
        let dist = crate::mpi_io::average_distribution(ngrids, nproc);
        (dist[my_rank].start, dist[my_rank].end)
    };
    let range = g1 - g0;
    let chunk = solvent_chunk(range, nao, 8.0, *chunk_size, rayon::current_num_threads());
    let chunks: Vec<(usize, usize)> = (g0..g1).step_by(chunk)
        .map(|p0| (p0, (p0 + chunk).min(g1)))
        .collect();

    let dt0 = time::Local::now();

    // Thread-parallel chunks; the BLAS call inside the closure is gated to one
    // thread (Rayon × OpenBLAS double pool). Partials are collected in ascending
    // chunk order → deterministic fold.
    let default_omp_num_threads = omp_get_num_threads_wrapper();
    let partials: Vec<(usize, Vec<f64>)> = chunks
        .par_iter()
        .map(|&(p0, p1)| {
            omp_set_num_threads_wrapper(1);
            let grid_coords_chunk = &grid_coords[p0..p1];
            let charge_exp_chunk: Vec<f64> = charge_exp[p0..p1].iter().map(|x| x * x).collect();
            let fake_chg_data = CINTR2CDATA::fakemol_for_charges(grid_coords_chunk, charge_exp_chunk.as_slice());

            let (tmpout, shape) = CINTR2CDATA::integrate_cross("int3c2e", [cint_data, cint_data, &fake_chg_data], None, None).into();
            let max_val = tmpout.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
            if max_val < 1.0e-12 {
                return (p0, vec![0.0; p1 - p0]);
            }
            let tmpshape = [shape[0] * shape[1], shape[2]];
            let v_nj = MatrixFull::from_vec(tmpshape, tmpout).unwrap();
            let v_e_chunk = _dgemm_scaled(&dm_mat, 'N', &v_nj, 'N', 1.0);
            (p0, v_e_chunk.data)
        })
        .collect();
    omp_set_num_threads_wrapper(default_omp_num_threads);

    let mut v_grids_e_local = vec![0.0_f64; ngrids];
    for (p0, vals) in &partials {
        for (off, &v) in vals.iter().enumerate() {
            v_grids_e_local[*p0 + off] = v;
        }
    }

    // Allreduce
    let mut v_grids_e = vec![0.0_f64; ngrids];
    mpi_op.world.any_process().all_reduce_into(&v_grids_e_local[..], &mut v_grids_e[..], &SystemOperation::sum());

    let dt1 = time::Local::now();
    let timecost = (dt1.timestamp_millis()-dt0.timestamp_millis()) as f64 /1000.0;
    if my_rank == 0 {
        println!("MPI v_grids_e ({} chunks / {} ranks) costs {:8.3} s", chunks.len(), nproc, timecost);
    }
    v_grids_e
}

/// MPI 版 `get_veff_pcm_by_q_old`：表面点按 rank 分区（`average_distribution`），
/// rank 内按统一 chunk（`solvent_chunk`）用 Rayon 并行批处理（闭包内 BLAS 已 gate）。
/// 通信：allreduce(nao² × 8B)。
#[cfg(feature = "mpi")]
pub fn get_veff_pcm_by_q_old_mpi(
    surface: &SurfaceVdwGaussian,
    cint_data: &CINTR2CDATA,
    q: &Vec<f64>,
    max_memory: &Option<f64>,
    chunk_size: &usize,
    mpi_operator: &Option<crate::mpi_io::MPIOperator>,
) -> MatrixUpper<f64> {
    use mpi::collective::SystemOperation;
    use mpi::traits::*;

    let mpi_op = match mpi_operator {
        Some(op) => op,
        None => return get_veff_pcm_by_q_old(surface, cint_data, q, max_memory, chunk_size),
    };
    let my_rank = mpi_op.rank;
    let nproc = mpi_op.size;

    let nao = cint_data.nao();
    let charge_exp = &surface.surface_calc.charge_exp;
    let grid_coords = &surface.surface_calc.grid_coords;
    let ngrids = charge_exp.len();
    // MPI ownership: contiguous surface-point range per rank; within the range,
    // chunks follow the unified solvent chunk (thread batch size).
    let (g0, g1) = {
        let dist = crate::mpi_io::average_distribution(ngrids, nproc);
        (dist[my_rank].start, dist[my_rank].end)
    };
    let range = g1 - g0;
    let chunk = solvent_chunk(range, nao, 8.0, *chunk_size, rayon::current_num_threads());
    let chunks: Vec<(usize, usize)> = (g0..g1).step_by(chunk)
        .map(|p0| (p0, (p0 + chunk).min(g1)))
        .collect();

    let dt0 = time::Local::now();

    // Thread-parallel chunks; the BLAS call inside the closure is gated to one
    // thread (Rayon × OpenBLAS double pool). Each chunk's nao² contribution is
    // negated and folded as soon as it is ready (Rayon tree reduction, same
    // scheme as the serial `get_veff_pcm_by_q_old`) — an ordered collect + serial
    // fold would materialize n_chunks·nao² partials and blow up at large nao.
    // Fold order is fixed for a given chunk count and thread count.
    let default_omp_num_threads = omp_get_num_threads_wrapper();
    let veff_local = chunks
        .par_iter()
        .map(|&(p0, p1)| {
            omp_set_num_threads_wrapper(1);
            let chunk_len = p1 - p0;
            let grid_coords_chunk = &grid_coords[p0..p1];
            let charge_exp_chunk_sq: Vec<f64> = charge_exp[p0..p1].iter().map(|x| x * x).collect();
            let fake_chg_data = CINTR2CDATA::fakemol_for_charges(grid_coords_chunk, charge_exp_chunk_sq.as_slice());

            let (tmpout, shape) = CINTR2CDATA::integrate_cross("int3c2e", [cint_data, cint_data, &fake_chg_data], None, None).into();
            let tmpshape = [shape[0] * shape[1], shape[2]];
            let tmp_v_nj = MatrixFull::from_vec(tmpshape, tmpout).unwrap();
            let q_p = MatrixFull::from_vec([chunk_len, 1], q[p0..p1].to_vec()).unwrap();
            let mut v_nj = _dgemm_scaled(&tmp_v_nj, 'N', &q_p, 'N', 1.0);
            v_nj.iter_mut().for_each(|x| *x = -*x);
            v_nj
        })
        .reduce(
            || MatrixFull::<f64>::new([nao, nao], 0.0),
            |mut acc, v| {
                acc.data.iter_mut().zip(v.data.iter()).for_each(|(a, &b)| *a += b);
                acc
            },
        );
    omp_set_num_threads_wrapper(default_omp_num_threads);

    // Allreduce
    let mut veff_full = vec![0.0_f64; nao * nao];
    mpi_op.world.any_process().all_reduce_into(&veff_local.data[..], &mut veff_full[..], &SystemOperation::sum());

    let dt1 = time::Local::now();
    let timecost = (dt1.timestamp_millis()-dt0.timestamp_millis()) as f64 /1000.0;
    if my_rank == 0 {
        println!("MPI veff ({} chunks / {} ranks) costs {:8.3} s", chunks.len(), nproc, timecost);
    }

    let veff_mf = MatrixFull::from_vec([nao, nao], veff_full).unwrap();
    let veff_upper = veff_mf.iter_matrixupper().unwrap().map(|&x| x).collect::<Vec<f64>>();
    MatrixUpper::from_vec(nao*(nao+1)/2 as usize, veff_upper).unwrap()
}

pub fn get_veff_pcm_by_q_old(
    surface: &SurfaceVdwGaussian,
    cint_data: &CINTR2CDATA,
    q: &Vec<f64>,
    max_memory: &Option<f64>,
    chunk_size: &usize) -> MatrixUpper<f64> {
    let nao = cint_data.nao();
    let charge_exp = &surface.surface_calc.charge_exp;
    let grid_coords = &surface.surface_calc.grid_coords;

    let ngrids = charge_exp.len();

    //let mut veff = MatrixFull::<f64>::new([nao, nao], 0.0);

    let CHUNK = solvent_chunk(ngrids, nao, 8.0, *chunk_size, rayon::current_num_threads());
    /** 
    let mem_avail_mb = max_memory.map(|max_memory| {
        max_memory - detect_used_memory_mb("proc")
    }).unwrap_or_else(detect_available_memory_mb);

    let CHUNK: usize = ((0.7 * mem_avail_mb * 1024.0 * 1024.0)
    / (8.0 * (nao * nao) as f64))
    .max(1.0) as usize;
    println!("Available memory for veff computation: {:.2} MB, setting chunk size to {}", mem_avail_mb, CHUNK);


    for p0 in (0..ngrids).step_by(CHUNK) {
        let p1 = (p0 + CHUNK).min(ngrids);
        let grid_coords_chunk = &grid_coords[p0..p1];
        let charge_exp_chunk = charge_exp[p0..p1].iter().map(|x| x * x).collect::<Vec<f64>>();
        let fake_chg_data = CINTR2CDATA::fakemol_for_charges(grid_coords_chunk, charge_exp_chunk.as_slice());

        let (tmpout, shape) = CINTR2CDATA::integrate_cross("int3c2e", [cint_data, cint_data, &fake_chg_data], None, None).into();
        //println!("Block shape={:?}, expected=[{},{},{}]", shape, nao, nao, p1-p0);
        let tmpshape = [shape[0] * shape[1], shape[2]];
        let tmp_v_nj = MatrixFull::from_vec(tmpshape, tmpout).unwrap();
        let q_p =MatrixFull::from_vec([p1 - p0, 1], q[p0..p1].to_vec()).unwrap();
        let v_nj = _dgemm_scaled(&tmp_v_nj, 'N', &q_p, 'N', 1.0);  
        veff.data.par_iter_mut().zip(v_nj.data.par_iter()).for_each(|(ve, &vnj)| *ve -= vnj);

    }
    */
    let dt0 = time::Local::now();
    /**
    let veff = (0..ngrids)
        .into_par_iter()
        .step_by(CHUNK)
        .map(|p0| {
            let t0 = Instant::now();
            let p1 = (p0 + CHUNK).min(ngrids);

            let grid_coords_chunk = &grid_coords[p0..p1];
            let mut charge_exp_chunk = Vec::with_capacity(p1 - p0);
            for &x in &charge_exp[p0..p1] {
                charge_exp_chunk.push(x * x);
            }
            //let charge_exp_chunk = charge_exp[p0..p1].iter().map(|x| x * x).collect::<Vec<f64>>();

            let fake_chg_data =
                CINTR2CDATA::fakemol_for_charges(grid_coords_chunk, charge_exp_chunk.as_slice());

            let (tmpout, shape) =
                CINTR2CDATA::integrate_cross(
                    "int3c2e",
                    [cint_data, cint_data, &fake_chg_data],
                    None,
                    None,
                ).into();

            let tmpshape = [shape[0] * shape[1], shape[2]];
            let tmp_v_nj = MatrixFull::from_vec(tmpshape, tmpout).unwrap();

            let q_p = MatrixFull::from_vec(
                [p1 - p0, 1],
                q[p0..p1].to_vec()
            ).unwrap();

            let mut v_nj = _dgemm_scaled(&tmp_v_nj, 'N', &q_p, 'N', 1.0);

            v_nj.data.iter_mut().for_each(|x| *x = -*x);

            println!("Chunk cycle of veff costs {:10.6} seconds.", t0.elapsed().as_secs_f64());
            v_nj
        })
        .reduce(
            || MatrixFull::<f64>::new([nao, nao], 0.0),
            |mut acc, v| {
                acc.data
                    .iter_mut()
                    .zip(v.data.iter())
                    .for_each(|(a, &b)| *a += b);
                acc
            }
        );
    */
    let default_omp_num_threads = omp_get_num_threads_wrapper();
    let veff = grid_coords
    .par_chunks(CHUNK)
    .zip(charge_exp.par_chunks(CHUNK))
    .zip(q.par_chunks(CHUNK))
    .map(|((grid_coords_chunk, charge_exp_chunk), q_chunk)| {
        omp_set_num_threads_wrapper(1);
        let t0 = Instant::now();

        let chunk_len = grid_coords_chunk.len();

        let charge_exp_chunk_sq: Vec<f64> =
            charge_exp_chunk.iter().map(|x| x * x).collect();

        let fake_chg_data =
            CINTR2CDATA::fakemol_for_charges(grid_coords_chunk, charge_exp_chunk_sq.as_slice());

        let (tmpout, shape) =
            CINTR2CDATA::integrate_cross(
                "int3c2e",
                [cint_data, cint_data, &fake_chg_data],
                None,
                None,
            ).into();

        let tmpshape = [shape[0] * shape[1], shape[2]];
        
        let tmp_v_nj = MatrixFull::from_vec(tmpshape, tmpout).unwrap();

        let q_p = MatrixFull::from_vec([chunk_len, 1],q_chunk.to_vec()).unwrap();

        let mut v_nj = _dgemm_scaled(&tmp_v_nj, 'N', &q_p, 'N', 1.0);

        v_nj.iter_mut().for_each(|x| *x = -*x);

        v_nj
    })
    .reduce(
        || MatrixFull::<f64>::new([nao, nao], 0.0),
        |mut acc, v| {
            acc.data
                .iter_mut()
                .zip(v.data.iter())
                .for_each(|(a, &b)| *a += b);
            acc
        }
    );

    omp_set_num_threads_wrapper(default_omp_num_threads);

    let dt1 = time::Local::now();
    let timecost = (dt1.timestamp_millis() - dt0.timestamp_millis()) as f64 / 1000.0;
    println!("Parallel computation of veff costs {:10.2} seconds.", timecost);
    let veff_upper = veff.iter_matrixupper().unwrap().map(|&x| x).collect::<Vec<f64>>();
    let veff = MatrixUpper::from_vec(nao*(nao+1)/2 as usize, veff_upper).unwrap();
    
    veff
}

//=============================================================================
//  RI-based PCM: get_v_grids_e (using auxiliary basis for (μν|g_i) integrals)
//=============================================================================

/// Prototype (2.1): reuse the resident RI tensor for the two `nao²·naux` contractions.
///
/// The RI-J/K build already holds `cderi = (μν|P)·M` packed as `[nbaspar, naux]`
/// (`SCF::rimatr`), where `M` is the metric factor from `j2c_decomp` and satisfies
/// `M·Mᵀ = V⁻¹`. The PCM needs the same bare `(μν|P)` twice:
///
/// ```text
///   c_vec[P] = Σ_{μν} D_{μν} (μν|P)          →  Y    = V⁻¹ c_vec
///   veff[μν] = -Σ_P (μν|P) Z_P,  Z = V⁻¹ z'  →  veff = -A Z
/// ```
///
/// Writing `A = cderi·M⁻¹` and using `M⁻¹V⁻¹ = Mᵀ`, with the packed density
/// `d` (upper triangle, diagonal halved — the same packing `vj_upper_with_rimatr`
/// uses, so that `2·Σ_p d_p A[p,P] = Σ_{μν} D_{μν}(μν|P)`):
///
/// ```text
///   Y    = 2 · M · (cderiᵀ d)
///   veff = -cderi · (Mᵀ z')          (M symmetric for the `eig` policy)
/// ```
///
/// i.e. one dgemv against the resident tensor plus one `naux × naux` apply each,
/// with no 3c integral evaluation at all.
///
/// One deliberate difference from the 3c path: `M` inverts the *plain* metric
/// `J_PQ = (P|Q)`, because that is the tensor the RI-J/K build produced, while the
/// 3c path solves with `V = J + 1e-12·max(1, max diag)` (see [`build_v_pq_chol`]).
/// The two agree to roughly the regularization, i.e. `~1e-12` relative.
pub struct PcmRiReuse<'a> {
    /// `cderi` — `SCF::rimatr.0`. Rows are packed AO pairs when `map` is `None`, and the
    /// retained rows only (storage-level pruned tensor) when it is `Some`.
    pub cderi: &'a MatrixFull<f64>,
    /// `baspar2basbas` (full-space pair table) — `SCF::rimatr.2`
    pub pairs: &'a Vec<[usize; 2]>,
    /// `Some` for a storage-level pruned tensor (`pair_screen_threshold > 0`):
    /// the compact row indexing. Reading a pruned tensor through `pairs` would silently
    /// read the wrong rows, which is what `require_unpruned_rimatr` guards against.
    pub map: Option<&'a crate::ri_jk::PairMap>,
    /// `M = J^{-1/2}` (eig policy) — `PcmStatic::j2c_factor`. Replicated on every rank,
    /// because the metric is built from the full auxiliary space.
    pub m: &'a MatrixFull<f64>,
    /// Which slice of the full auxiliary space `cderi` holds. In serial this is the whole
    /// range. Under MPI the distributed builder splits the auxiliary dimension over the
    /// ranks (`mpi_data.auxbas`), so a rank holds `[nbaspar, aux_cols.len()]` columns and
    /// `aux_cols.start` is the first global auxiliary index of its block.
    pub aux_cols: std::ops::Range<usize>,
    /// Length of the full auxiliary vectors (`Y`, `Z`, the metric factor's dimension).
    pub naux_full: usize,
}

impl<'a> PcmRiReuse<'a> {
    /// True when the tensor covers only part of the auxiliary space (MPI). Both PCM
    /// contractions are defined over the full auxiliary space, so the rank-local parts
    /// have to be summed back into it; the row dimension needs no treatment because the
    /// distributed tensor keeps every packed AO pair on every rank.
    pub fn is_distributed(&self) -> bool {
        self.aux_cols.len() != self.naux_full
    }
}

/// Sums a rank-local contribution back into the full auxiliary space.
///
/// `local` is the part of the result this rank could compute on its own: the auxiliary
/// block `offset..offset+local.len()` of a full vector, or (for a contraction over the
/// auxiliary index) a full-length row vector whose columns were the rank's own. The
/// missing entries are zero on this rank and live on another one, so the sum over the
/// ranks completes the result. A no-op in serial, where the tensor already covers every
/// auxiliary function.
///
/// The reduction is collective, so it must run on every rank of `world`. The branch only
/// depends on the tensor's layout, which the builders set up identically everywhere.
#[cfg(feature = "mpi")]
fn reduce_aux_contribution(
    ri: &PcmRiReuse,
    mpi_operator: &Option<crate::mpi_io::MPIOperator>,
    local: &[f64],
    full_len: usize,
    offset: usize,
) -> Vec<f64> {
    let mut full = vec![0.0_f64; full_len];
    full[offset..offset + local.len()].copy_from_slice(local);
    if ri.is_distributed() {
        let op = mpi_operator
            .as_ref()
            .expect("RI-PCM reuse: a distributed auxiliary layout requires the MPI operator");
        let mut reduced = vec![0.0_f64; full_len];
        crate::mpi_io::mpi_allreduce(&op.world, &full, &mut reduced, &mpi::collective::SystemOperation::sum());
        full = reduced;
    }
    full
}

/// Serial build: the tensor holds every auxiliary column, so the scatter is the result.
#[cfg(not(feature = "mpi"))]
fn reduce_aux_contribution(
    _ri: &PcmRiReuse,
    _mpi_operator: &Option<crate::mpi_io::MPIOperator>,
    local: &[f64],
    full_len: usize,
    offset: usize,
) -> Vec<f64> {
    let mut full = vec![0.0_f64; full_len];
    full[offset..offset + local.len()].copy_from_slice(local);
    full
}

pub fn get_v_grids_e(
    surface: &SurfaceVdwGaussian,
    mol: &Molecule,
    dm: &Vec<MatrixFull<f64>>,
    spin_channel: &usize,
    max_memory: &Option<f64>,
    chunk_size: &usize,
    cint_data_ri: &CINTR2CDATA,
    aux_cint: &CINTR2CDATA,
    v_pq_chol: &MatrixFull<f64>,
    ri_reuse: Option<&PcmRiReuse>,
    mpi_operator: &Option<crate::mpi_io::MPIOperator>,
) -> Vec<f64> {
    let charge_exp = &surface.surface_calc.charge_exp;
    let grid_coords = &surface.surface_calc.grid_coords;
    let ngrids = charge_exp.len();
    let nao = dm[0].size[0] as usize;
    let naux = aux_cint.ao_loc().last().unwrap().clone();
    // Per grid point the RI chunk holds an [naux, chunk] block (8·naux bytes),
    // not the [nao², chunk] of the non-RI paths.
    let CHUNK = solvent_chunk_bytes(ngrids, 8.0 * naux as f64, *chunk_size, rayon::current_num_threads());
    let nbas = mol.cint_bas.len() as i32;
    let nbas_aux = mol.cint_aux_bas.len() as i32;

    // Flatten DM to row vector [1, nao*nao]
    let mut dm_vec = vec![0.0; nao * nao];
    for i_spin in 0..*spin_channel {
        for j in 0..nao {
            for i in 0..nao {
                dm_vec[j*nao + i] += dm[i_spin][(i,j)];
            }
        }
    }
    let dm_mat = MatrixFull::from_vec([1, nao*nao], dm_vec).unwrap();

    // Partition auxiliary basis into batches
    let aux_loc = &cint_data_ri.ao_loc()[(nbas as usize)..];
    let aux_batch_size = 256;
    let partition = blocksize_partition(&aux_loc, aux_batch_size);

    let dt0 = time::Local::now();

    // Step 1: Compute C_P = Σ_{μν} D_{μν} · (μν|P) over aux batches
    let y_vec: Vec<f64> = if let Some(ri) = ri_reuse {
        // Prototype 2.1: no 3c integrals — one dgemv against the resident cderi
        // plus one application of the metric factor. See `PcmRiReuse`.
        let nrow = ri.cderi.size[0];
        // Gather the density onto the tensor's rows. Unpruned: the packed upper triangle,
        // one value per packed pair. Pruned: one value per stored row, read through the
        // `PairMap` (the pairs whose row was dropped simply do not enter the contraction —
        // the same truncation the storage level applied to the SCF kernels).
        let mut d_packed = vec![0.0f64; nrow];
        match ri.map {
            None => {
                for i_spin in 0..*spin_channel {
                    dm[i_spin]
                        .iter_matrixupper()
                        .unwrap()
                        .enumerate()
                        .for_each(|(p, &v)| d_packed[p] += v);
                }
                // halve the diagonal: the packed contraction below is doubled to recover
                // the full Σ_{μν} (same convention as `vj_upper_with_rimatr`)
                ri.pairs.iter().enumerate().for_each(|(p, ij)| {
                    if ij[0] == ij[1] {
                        d_packed[p] *= 0.5;
                    }
                });
            }
            Some(map) => {
                for i_spin in 0..*spin_channel {
                    let d = &dm[i_spin];
                    for row in 0..map.len() {
                        let (mu, nu) = map.pair_of(row);
                        let v = d[[mu, nu]];
                        d_packed[row] += if mu == nu { 0.5 * v } else { v };
                    }
                }
            }
        }
        let d_mat = MatrixFull::from_vec([nrow, 1], d_packed).unwrap();
        let c_hat_local = _dgemm_scaled(ri.cderi, 'T', &d_mat, 'N', 1.0); // [ncols, 1]
        // Under MPI the tensor carries only this rank's auxiliary columns, so the
        // contraction is completed over the ranks before the metric factor is applied.
        // The factor itself is replicated, so every rank gets the same `Y`.
        let c_hat = reduce_aux_contribution(
            ri, mpi_operator, &c_hat_local.data, naux, ri.aux_cols.start);
        let c_mat = MatrixFull::from_vec([naux, 1], c_hat).unwrap();
        let y = _dgemm_scaled(ri.m, 'N', &c_mat, 'N', 1.0); // Y = M·c_hat
        y.data.iter().map(|&x| 2.0 * x).collect()
    } else {
        let mut c_vec = vec![0.0; naux];
        let mut idx_ao = 0;
        for &[shl0, shl1] in &partition {
            let nbatch = aux_loc[shl1] - aux_loc[shl0];
            let shls_slice = [[0, nbas], [0, nbas], [nbas + shl0 as i32, nbas + shl1 as i32]];
            let (out, _) = cint_data_ri.integral_s1::<int3c2e>(Some(&shls_slice));
            let tmpshape = [nao * nao, nbatch];
            let int3c = MatrixFull::from_vec(tmpshape, out).unwrap();
            let c_batch = _dgemm_scaled(&dm_mat, 'N', &int3c, 'N', 1.0);
            for (i, &v) in c_batch.data.iter().enumerate() {
                c_vec[idx_ao + i] = v;
            }
            idx_ao += nbatch;
        }
        // Step 2: Solve V · Y = C via the precomputed Cholesky factor of V (V is symmetric)
        solve_cholesky_factorized(v_pq_chol, &c_vec)
    };
    let y_mat = MatrixFull::from_vec([naux, 1], y_vec).unwrap();

    // Step 3: Parallel grid chunks — v_grids_e[j] = Σ_P Y_P · (P|g_j)
    let mut v_grids_e = vec![0.0; ngrids];
    let default_omp_num_threads = omp_get_num_threads_wrapper();
    v_grids_e.par_chunks_mut(CHUNK).enumerate().for_each(|(v_chunk, idx)| {
        omp_set_num_threads_wrapper(1);
        let p0 = v_chunk * CHUNK;
        let p1 = (p0 + CHUNK).min(ngrids);
        let grid_coords_chunk = &grid_coords[p0..p1];
        let mut charge_exp_chunk = Vec::with_capacity(p1 - p0);
        for &x in &charge_exp[p0..p1] {
            charge_exp_chunk.push(x * x);
        }
        let fake_chg_data = CINTR2CDATA::fakemol_for_charges(grid_coords_chunk, charge_exp_chunk.as_slice());
        
        //println!("CHUNK {}: exponents[0..5]={:?}, coef check: 2*α^1.5/π = {:?}",
        //v_chunk, &charge_exp_chunk[..5.min(charge_exp_chunk.len())],
        //charge_exp_chunk.iter().map(|&a| 2.0 * a.powf(1.5) / std::f64::consts::PI).take(5).collect::<Vec<_>>());

        //actual code let (tmpout, shape) = CINTR2CDATA::integrate_cross("int2c2e", [&fake_chg_data, aux_cint], None, None).into();
        //let (tmpout, shape) = CINTR2CDATA::integrate_cross("int2c2e", [&fake_chg_data, cint_data_ri], None, None).into();
        //let (tmpout, shape) = CINTR2CDATA::integrate_cross("int2c2e", [&fake_chg_data, &fake_chg_data], None, None).into();
        let shls_slice_2c = [[0, (p1-p0) as i32], [nbas, nbas + nbas_aux]];
        let (tmpout, shape) = CINTR2CDATA::integrate_cross("int2c2e", [&fake_chg_data, cint_data_ri], None, &shls_slice_2c).into();
        let pg_shape = [shape[0], shape[1]];
        let pg_mat = MatrixFull::from_vec(pg_shape, tmpout).unwrap();
        let v_e_chunk = _dgemm_scaled(&pg_mat, 'N', &y_mat, 'N', 1.0);
        idx.iter_mut().zip(v_e_chunk.iter()).for_each(|(i, &v)| *i = v);
    });
    omp_set_num_threads_wrapper(default_omp_num_threads);

    let dt1 = time::Local::now();
    let timecost = (dt1.timestamp_millis() - dt0.timestamp_millis()) as f64 / 1000.0;
    println!("RI-PCM: v_grids_e costs {:10.2} seconds.", timecost);

    v_grids_e
}

//=============================================================================
//  RI-based PCM: get_veff_pcm_by_q (using auxiliary basis for (μν|g_i) integrals)
//=============================================================================

pub fn get_veff_pcm_by_q(
    surface: &SurfaceVdwGaussian,
    mol: &Molecule,
    q: &Vec<f64>,
    max_memory: &Option<f64>,
    chunk_size: &usize,
    cint_data_ri: &CINTR2CDATA,
    aux_cint: &CINTR2CDATA,
    v_pq_chol: &MatrixFull<f64>,
    ri_reuse: Option<&PcmRiReuse>,
    mpi_operator: &Option<crate::mpi_io::MPIOperator>,
) -> MatrixUpper<f64> {

    let nbas = mol.cint_bas.len() as i32;
    let nbas_aux = mol.cint_aux_bas.len() as i32;
    let nao = cint_data_ri.ao_loc()[nbas as usize].clone(); 
    let naux = aux_cint.ao_loc().last().unwrap().clone();
    let charge_exp = &surface.surface_calc.charge_exp;
    let grid_coords = &surface.surface_calc.grid_coords;
    let ngrids = charge_exp.len();
    // Per grid point the RI chunk holds an [naux, chunk] block (8·naux bytes),
    // not the [nao², chunk] of the non-RI paths.
    let CHUNK = solvent_chunk_bytes(ngrids, 8.0 * naux as f64, *chunk_size, rayon::current_num_threads());

    // Auxiliary basis info
    let aux_loc = &cint_data_ri.ao_loc()[(nbas as usize)..];

    let dt0 = time::Local::now();

    // Step 1: Compute Z'_P = Σ_j q_j · (P|g_j) over grid chunks (parallel)
    let default_omp_num_threads = omp_get_num_threads_wrapper();
    let z_prime = grid_coords
        .par_chunks(CHUNK)
        .zip(charge_exp.par_chunks(CHUNK))
        .zip(q.par_chunks(CHUNK))
        .map(|((grid_coords_chunk, charge_exp_chunk), q_chunk)| {
            omp_set_num_threads_wrapper(1);
            let chunk_len = grid_coords_chunk.len();
            let charge_exp_chunk_sq: Vec<f64> =
                charge_exp_chunk.iter().map(|x| x * x).collect();
            let fake_chg_data =
                CINTR2CDATA::fakemol_for_charges(grid_coords_chunk, charge_exp_chunk_sq.as_slice());

            //let (tmpout, shape) = CINTR2CDATA::integrate_cross("int2c2e", [aux_cint, &fake_chg_data], None, None).into();
            let shls_slice_2c = [[nbas, nbas + nbas_aux], [0, chunk_len as i32]];
            let (tmpout, shape) = CINTR2CDATA::integrate_cross("int2c2e", [cint_data_ri, &fake_chg_data], None, &shls_slice_2c).into();
            let pg_shape = [shape[0], shape[1]];
            let pg_mat = MatrixFull::from_vec(pg_shape, tmpout).unwrap();
            let q_mat = MatrixFull::from_vec([chunk_len, 1], q_chunk.to_vec()).unwrap();
            let z_contrib = _dgemm_scaled(&pg_mat, 'N', &q_mat, 'N', 1.0);
            z_contrib.data
        })
        .reduce(
            || vec![0.0; naux],
            |mut a, b| {
                for (i, &v) in b.iter().enumerate() {
                    a[i] += v;
                }
                a
            });
    omp_set_num_threads_wrapper(default_omp_num_threads);

    // Step 2: Solve V · Z = Z' via the precomputed Cholesky factor of V (V is symmetric)
    let z_vec = solve_cholesky_factorized(v_pq_chol, &z_prime);

    // Step 3: Build veff[μ,ν] = -(μν|P) · Z_P — either by re-evaluating the 3c
    // integrals, or (prototype 2.1) by contracting the resident cderi.
    let veff_upper: Vec<f64> = if let Some(ri) = ri_reuse {
        // veff = -cderi·(Mᵀ z'); the result comes out already packed in the same
        // upper-triangular order `MatrixUpper` uses (cderi's rows are `baspar`).
        let z_mat = MatrixFull::from_vec([naux, 1], z_prime).unwrap();
        let w = _dgemm_scaled(ri.m, 'T', &z_mat, 'N', 1.0); // [naux, 1], replicated
        // Contract only the auxiliary columns this rank holds, then complete the sum over
        // the ranks: the row dimension is full everywhere, so the reduced vector is the
        // whole `-Σ_P (μν|P) Z_P`.
        let w_local = MatrixFull::from_vec(
            [ri.aux_cols.len(), 1],
            w.data[ri.aux_cols.clone()].to_vec(),
        ).unwrap();
        let vp_local = _dgemm_scaled(ri.cderi, 'N', &w_local, 'N', 1.0); // [nrow, 1]
        let vp_data = reduce_aux_contribution(ri, mpi_operator, &vp_local.data, vp_local.size[0], 0);
        let vp = MatrixFull::from_vec([vp_data.len(), 1], vp_data).unwrap();
        match ri.map {
            None => vp.data.iter().map(|&x| -x).collect(),
            Some(map) => {
                // Scatter onto the symmetric AO matrix; rows the storage level dropped stay
                // zero, which is the truncation the SCF kernels already apply to this tensor.
                let mut veff = MatrixFull::<f64>::new([nao, nao], 0.0);
                for (row, &v) in vp.data.iter().enumerate() {
                    let [mu, nu] = map.rows[row];
                    let (i, j) = (mu as usize, nu as usize);
                    veff[[i, j]] = -v;
                    veff[[j, i]] = -v;
                }
                veff.iter_matrixupper().unwrap().map(|&x| x).collect()
            }
        }
    } else {
        let aux_batch_size = 256;
        let partition = blocksize_partition(&aux_loc, aux_batch_size);
        let mut veff = MatrixFull::<f64>::new([nao, nao], 0.0);
        let mut idx_ao = 0;
        for &[shl0, shl1] in &partition {
            let nbatch = aux_loc[shl1] - aux_loc[shl0];
            let shls_slice = [[0, nbas], [0, nbas], [nbas + shl0 as i32, nbas + shl1 as i32]];
            let (out, shape) = cint_data_ri.integral_s1::<int3c2e>(Some(&shls_slice));
            let tmpshape = [nao * nao, nbatch];
            let int3c = MatrixFull::from_vec(tmpshape, out).unwrap();
            let z_batch = MatrixFull::from_vec([nbatch, 1], z_vec[idx_ao..idx_ao + nbatch].to_vec()).unwrap();
            let contrib = _dgemm_scaled(&int3c, 'N', &z_batch, 'N', 1.0);
            for (i, &v) in contrib.data.iter().enumerate() {
                veff.data[i] -= v;
            }
            idx_ao += nbatch;
        }
        veff.iter_matrixupper().unwrap().map(|&x| x).collect::<Vec<f64>>()
    };

    // Pack to MatrixUpper
    let dt1 = time::Local::now();
    let timecost = (dt1.timestamp_millis() - dt0.timestamp_millis()) as f64 / 1000.0;
    println!("RI-PCM: veff costs {:10.2} seconds.", timecost);

    // Debug: check veff for NaN
    if veff_upper.iter().any(|&x| x.is_nan() || x.is_infinite()) {
        println!("WARNING: RI-PCM veff contains NaN or Inf!");
    }

    MatrixUpper::from_vec(nao * (nao + 1) / 2, veff_upper).unwrap()
}


/// Main function to prepare PCM object for a given molecule
pub fn solvent_prepare(mol: &Molecule) -> PcmObject {
    let method = mol.ctrl.solvent_model.clone();
    let epsilon = mol.ctrl.solv_epsilon;
    
    let descriptors = {
        if method == PcmMethod::SMD {
            mol.ctrl.solvent_descriptors.unwrap_or_else(|| {
                panic!("SMD solvent model requires solvent descriptors. You should set solvent_descriptors or solvent_name in the control file.");
            })
        } 
        else {SMD_ERROR_DESCRIPTORS}
    };
    let icds = if is_water_descriptor(&descriptors) { 1 } else { 2 };

    if method == PcmMethod::SMD && icds == 2 {
        if descriptors[0] < 0.0 || descriptors[2] < 0.0 || descriptors[3] < 0.0
            || descriptors[4] < 0.0 || descriptors[6] < 0.0 || descriptors[7] < 0.0
        {
            panic!("SMD solvent model requires valid [n, α, β, γ, φ, ψ]. \
                   Found negative value in them. \
                   Check solvent_descriptors or solvent_name in the control file.");
        }
    }
    let pcmcfg = PcmObjectCfg::build(method, epsilon, descriptors, icds, mol.ctrl.smd_cavity_radii);

    // Build cavity surface
    let mut surface = SurfaceVdwGaussian::new(mol.ctrl.pcm_cavity_radii, &mol.geom);
    if method == PcmMethod::SMD {
        // SMD uses intrinsic atomic Coulomb radii (eq. 16, Marenich 2009),
        // no vdW scaling (scale = 1.0)
        let alpha = descriptors[2]; // H-bond acidity
        surface.cfg.atom_radii = Some(smd_radii(alpha, &surface.atomic_num, mol.ctrl.smd_cavity_radii));
        surface.cfg.vdw_scale = Some(1.0);
    }
    surface.build();

    let pstatic = PcmStatic::build_pcm_static(&surface, &pcmcfg, &mol);
    PcmObject::init_Pcm(pcmcfg, surface, pstatic)
}

/// Compute SMD CDS energy and gradient from surface data.
/// Returns (gcds_hartree, tarea_ang2, dcds_hartree_per_bohr).
pub fn compute_cds_from_surface(
    surface: &SurfaceVdwGaussian,
    cfg: &PcmObjectCfg,
) -> (f64, f64, Vec<[f64; 3]>) {
    let natm = surface.atomic_num.len();
    let atomic_numbers = surface.atomic_num.clone();
    // Convert [3, natm] MatrixFull → [natm, 3] Vec<[f64;3]>
    let mut coords = vec![[0.0f64; 3]; natm];
    for i in 0..natm {
        coords[i] = [
            surface.atom_coords[[0, i]],
            surface.atom_coords[[1, i]],
            surface.atom_coords[[2, i]],
        ];
    }
    smd_cds::compute_cds(
        &atomic_numbers,
        &coords,
        cfg.icds,
        &cfg.solvent_descriptors,
        cfg.smd_cavity_radii,
    )
}


fn print_matrix_stats<T: std::fmt::Display + Copy>(matrix: &MatrixFull<T>) 
where
    T: std::fmt::Display + Copy + std::cmp::PartialOrd ,
{
    println!("size of matrix: {}x{}", matrix.size[0], matrix.size[1]);
    print_vec_stats(matrix.data.as_slice());
}

fn print_vec_stats<T: std::fmt::Display + Copy>(slice: &[T]) 
where 
    T: std::cmp::PartialOrd  // 使用 PartialOrd 而不是 Ord
{
    if slice.is_empty() {
        println!("empty vec");
        return;
    }
    
    // 使用 fold 处理浮点数
    let max = slice.iter()
        .fold(&slice[0], |a, b| if a > b { a } else { b });
    
    let min = slice.iter()
        .fold(&slice[0], |a, b| if a < b { a } else { b });
    
    println!("full length: {}", slice.len());
    println!("max: {}", max);
    println!("min: {}", min);
}

pub fn debug_print_pcm(sta: &PcmStatic, scf: &PcmScf){
    println!("A:");
    print_vec_stats(&sta.A);
    //println!("vec of A:{:?}",sta.A);
    println!("D:");
    print_matrix_stats(&sta.D);
    println!("S:");
    print_matrix_stats(&sta.S);
    println!("K:");
    print_matrix_stats(&sta.K_initial);
    println!("R:");
    print_matrix_stats(&sta.R);
    println!("v_grids_n:");
    print_vec_stats(&sta.v_grids_n);
    println!("v_grids_e:");
    print_vec_stats(&scf.v_grids_e);
    println!("q_sym:");
    print_matrix_stats(&scf.q_sym);
    println!("abs(q_sym):");
    print_vec_stats(&scf.q_sym.data.iter().map(|x| x.abs()).collect::<Vec<f64>>());
    println!("v_grids:");
    print_vec_stats(&scf.v_grids);
    println!("veff:");
    print_vec_stats(&scf.veff.data);
    println!("eng:");
    println!("{}", scf.eng);
}

// ============================================================================
// Unit tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// 模糊 water 判定只比较 n, α, ε（容差 0.01 / 0.01 / 1.0）。
    #[test]
    fn test_is_water_descriptor() {
        assert!(is_water_descriptor(&SMD_WATER_DESCRIPTORS));
        // 容差内（严格 < 0.01 / < 0.01 / < 1.0）：Δn=+0.0072, Δα=+0.005, Δε=+0.645
        let near = [1.34, 1.33, 0.825, 0.35, -1.0, 79.0, -1.0, -1.0];
        assert!(is_water_descriptor(&near));
        // 边界：|Δα| = 0.01 不满足严格小于 → false
        let mut d = SMD_WATER_DESCRIPTORS;
        d[2] = 0.83;
        assert!(!is_water_descriptor(&d), "|Δα|=0.01 应为 false（严格小于）");
        // 非水溶剂（octanol）
        let octanol = [1.4295, 1.4279, 0.37, 0.48, 39.01, 9.8629, 0.0, 0.0];
        assert!(!is_water_descriptor(&octanol));
        // n 超容差
        let mut d = SMD_WATER_DESCRIPTORS;
        d[0] = 1.36;
        assert!(!is_water_descriptor(&d));
        // ε 超容差
        let mut d = SMD_WATER_DESCRIPTORS;
        d[5] = 76.0;
        assert!(!is_water_descriptor(&d));
        // 哨兵值
        assert!(!is_water_descriptor(&SMD_ERROR_DESCRIPTORS));
    }

    #[test]
    fn test_pcm_object_cfg_default() {
        let cfg = PcmObjectCfg::default();
        assert_eq!(cfg.method, PcmMethod::CPCM);
        assert_eq!(cfg.epsilon, 78.3553);
        assert_eq!(cfg.icds, 0);
        assert_eq!(cfg.smd_cavity_radii, SmdCavityRadii::Bondi);
        assert_eq!(cfg.solvent_descriptors, SMD_ERROR_DESCRIPTORS);
    }

    #[test]
    fn test_pcm_method_deserialize() {
        assert_eq!(serde_json::from_str::<PcmMethod>("\"cpcm\"").unwrap(), PcmMethod::CPCM);
        assert_eq!(serde_json::from_str::<PcmMethod>("\"COSMO\"").unwrap(), PcmMethod::COSMO);
        assert_eq!(serde_json::from_str::<PcmMethod>("\"iefpcm\"").unwrap(), PcmMethod::IEFPCM);
        // Fortran/文献别名 SS(V)PE
        assert_eq!(serde_json::from_str::<PcmMethod>("\"ss(v)pe\"").unwrap(), PcmMethod::SSVPE);
        assert_eq!(serde_json::from_str::<PcmMethod>("\"ssvpe\"").unwrap(), PcmMethod::SSVPE);
        assert_eq!(serde_json::from_str::<PcmMethod>("\"smd\"").unwrap(), PcmMethod::SMD);
        assert!(serde_json::from_str::<PcmMethod>("\"unknown\"").is_err());
    }

    /// `solvent_chunk_bytes` must never return 0 (`par_chunks(0)` panics) and
    /// the nao²-based wrapper must reproduce the non-RI chunking.
    #[test]
    fn test_solvent_chunk_bytes_clamps_and_matches_wrapper() {
        // user knob 0 -> clamped to 1 (the RI veff panic face)
        assert_eq!(solvent_chunk_bytes(1000, 8.0 * 1000.0, 0, 4), 1);
        // default knob 16 is the upper bound when memory is plentiful
        assert_eq!(solvent_chunk_bytes(1000, 8.0 * 1000.0, 16, 4), 16);
        // range below the floor: chunk collapses to the number of points
        assert_eq!(solvent_chunk_bytes(10, 8.0 * 1000.0, 16, 4), 10);
        // wrapper equivalence: solvent_chunk(nao=100, 8.0) == bytes(8·100²)
        assert_eq!(
            solvent_chunk(1000, 100, 8.0, 16, 4),
            solvent_chunk_bytes(1000, 8.0 * 10_000.0, 16, 4)
        );
        // memory cap: 500 MB / (500 MB per point) -> 1 point per thread
        assert_eq!(solvent_chunk_bytes(1000, 5.0e8, 16, 4), 1);
        // load balance cannot exceed the user knob
        assert_eq!(solvent_chunk_bytes(10_000, 8.0 * 100.0, 64, 4), 64);
    }
}
