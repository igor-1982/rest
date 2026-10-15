use std::fmt;
use std::f64::consts;
use rstsr::prelude::*;
use serde::{Deserialize, Serialize};
use tensors::{MatrixFull, MatrixUpper, BasicMatrix};
use super::SurfaceSwitchType;
use crate::constants::solvent as data;
use crate::constants::BOHR;
use crate::dft::gen_grids::angular_grid;
use crate::geom_io::{GeomCell, get_mass_charge};


//=============================================================================
//  Surface Related Structures
//=============================================================================

/// Container for surface calculation data
#[derive(Clone)]
pub struct SurfaceCalc{
    pub grid_coords: Vec<[f64; 3]>,
    pub weights: Vec<f64>,
    pub switch_fun: Vec<f64>,
    pub norm_vec: Vec<(f64, f64, f64)>,
    pub charge_exp: Vec<f64>,
   // pub charge: Vec<f64>,
    pub R_vdw: Vec<f64>,
    pub area: Vec<f64>,
}

impl SurfaceCalc{
    pub fn new() -> Self{
        SurfaceCalc{
            grid_coords: vec![],
            weights: vec![],
            switch_fun: vec![],
            norm_vec: vec![],
            charge_exp: vec![],
            R_vdw: vec![],
            area: vec![],
        }
    }
}

/// Radius scheme for PCM cavity construction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum RadiusScheme {
    Bondi,
    #[default]
    UFF,
}

impl RadiusScheme {
    pub fn default_vdw_scale(&self) -> f64 {
        match self {
            RadiusScheme::Bondi => 1.2,
            RadiusScheme::UFF => 1.1,
        }
    }
}

impl fmt::Display for RadiusScheme {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RadiusScheme::Bondi => write!(f, "Bondi"),
            RadiusScheme::UFF => write!(f, "UFF"),
        }
    }
}

impl<'d> Deserialize<'d> for RadiusScheme {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'d>,
    {
        let s = String::deserialize(deserializer)?;
        match s.to_uppercase().as_str() {
            "BONDI" => Ok(RadiusScheme::Bondi),
            "UFF" => Ok(RadiusScheme::UFF),
            _ => Err(serde::de::Error::custom(format!("Unknown RadiusScheme: {}", s))),
        }
    }
}

// =============================================================================
//  SMD-specific cavity radii (eq. 16, Marenich et al. JPCB 2009)
// =============================================================================

/// SMD cavity/CDS radii scheme, selected by the `smd_cavity_radii` keyword (default `Bondi`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize)]
pub enum SmdCavityRadii {
    /// Legacy behavior (default): electrostatic fallback uses `VDW_RADII` (PySCF mixed table),
    /// CDS uses the full `BONDI` table — aligned with PySCF / NWChem mnsol.F.
    #[default]
    Bondi,
    /// Non-eq.16 elements use `constants::solvent::BONDI_UFF_RADII`
    /// (Bondi 1964 values ∪ unclassified values ∪ UFF fill-in, measured 2026-08);
    /// the 11 eq.16 elements keep their original paths (eq.16 intrinsic / `BONDI`).
    BondiUff,
}

impl<'d> Deserialize<'d> for SmdCavityRadii {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'d>,
    {
        let s = String::deserialize(deserializer)?;
        match s.to_uppercase().as_str() {
            "BONDI" => Ok(SmdCavityRadii::Bondi),
            "UFF_MIXED" => Ok(SmdCavityRadii::BondiUff),
            _ => Err(serde::de::Error::custom(format!("Unknown SmdCavityRadii: {}", s))),
        }
    }
}

/// SMD intrinsic atomic Coulomb radii (Å), indexed by atomic number Z.
/// Unlisted elements fall back to Bondi vdW radii.
const SMD_RADII_ANG: [f64; 104] = {
    let mut r = [0.0f64; 104];
    r[1]  = 1.20;  // H
    r[6]  = 1.85;  // C
    r[7]  = 1.89;  // N
    r[9]  = 1.73;  // F
    r[14] = 2.47;  // Si
    r[15] = 2.12;  // P
    r[16] = 2.49;  // S
    r[17] = 2.38;  // Cl
    r[35] = 2.60;  // Br  (SMD18)
    r[53] = 2.74;  // I
    r
};

/// Build SMD-specific cavity radii for the given atoms.
///
/// Oxygen radius depends on the solvent H-bond acidity `alpha`:
/// ```text
/// R_O = 1.52          if α ≥ 0.43
///       1.52 + 1.8×(0.43−α)   otherwise
/// ```
/// All other specialized elements (H, C, N, F, Si, P, S, Cl, Br, I) use
/// fixed SMD values. Unparameterized elements fall back per `scheme`:
/// - `Bondi`: `VDW_RADII` (PySCF mixed table, legacy)
/// - `BondiUff`: `BONDI_UFF_RADII` (reference snapshot table; zero entries are defensive only
///   — the table covers Z=1..103 — and fall back to `VDW_RADII`)
///
/// Returns radii in **Bohr**.
pub fn smd_radii(
    alpha: f64,
    atomic_numbers: &[usize],
    scheme: SmdCavityRadii,
) -> Vec<f64> {
    let r_o_ang = if alpha >= 0.43 {
        1.52
    } else {
        1.52 + 1.8 * (0.43 - alpha)
    };
    atomic_numbers.iter().map(|&z| {
        if z == 8 {
            r_o_ang / BOHR
        } else if z < SMD_RADII_ANG.len() && SMD_RADII_ANG[z] > 0.0 {
            SMD_RADII_ANG[z] / BOHR
        } else {
            match scheme {
                SmdCavityRadii::Bondi => data::VDW_RADII[z], // fallback to PySCF mixed table (Bohr)
                SmdCavityRadii::BondiUff => {
                    // reference snapshot table in bohr; 0 entry is defensive only → fall back
                    if z < data::BONDI_UFF_RADII.len() && data::BONDI_UFF_RADII[z] > 0.0 {
                        data::BONDI_UFF_RADII[z]
                    } else {
                        data::VDW_RADII[z]
                    }
                }
            }
        }
    }).collect()
}

/// surface configurations
#[non_exhaustive]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct SurfaceVdwGaussianCfg {
    pub lebedev_degree: usize,
    pub atom_radii: Option<Vec<f64>>,
    pub radius_scheme: RadiusScheme,
    /// VdW scale factor; `None` uses the default scale for the chosen `radius_scheme`
    pub vdw_scale: Option<f64>,
    pub switch_type: SurfaceSwitchType,
}

impl Default for SurfaceVdwGaussianCfg {
    fn default() -> Self {
        SurfaceVdwGaussianCfg {
            lebedev_degree: 302,
            atom_radii: None,
            radius_scheme: RadiusScheme::default(),
            vdw_scale: None,
            switch_type: SurfaceSwitchType::default(),
        }
    }
}
impl SurfaceVdwGaussianCfg {
    /// Resolve the actual vdW scale: use `vdw_scale` if set, otherwise use the default for the chosen `radius_scheme`.
    pub fn effective_vdw_scale(&self) -> f64 {
        self.vdw_scale.unwrap_or_else(|| self.radius_scheme.default_vdw_scale())
    }

    pub fn new(radius_scheme: RadiusScheme) -> Self {
        SurfaceVdwGaussianCfg {
            lebedev_degree: 302,
            atom_radii: None,
            radius_scheme,
            vdw_scale: None,
            switch_type: SurfaceSwitchType::default(),
        }
    }
}
//lebedev_degree is usually 302, vdw_scale is usually 1.2, switch_type is usually SWIG

/// Gaussian VDW surface representation
#[non_exhaustive]
#[derive(Clone)]
pub struct SurfaceVdwGaussian {
    pub cfg: SurfaceVdwGaussianCfg,
    pub atomic_num: Vec<usize>,
    pub atom_coords: MatrixFull<f64>,
    pub surface_calc: SurfaceCalc,
    pub gslice_by_atom: Vec<(usize,usize)>,

}

impl SurfaceVdwGaussian {
    pub fn new(radius_scheme: RadiusScheme, geom: &GeomCell) -> Self {
        let cfg = SurfaceVdwGaussianCfg::new(radius_scheme);
        let mut surface_calc = SurfaceCalc::new();
        let gslice_by_atom = vec![];
        let mass_charge = get_mass_charge(&geom.elem);
        let atomic_num = mass_charge.iter().map(|&(_, charge)| charge as usize).collect();
        let atom_coords = geom.position.clone();
        SurfaceVdwGaussian { cfg, atomic_num, atom_coords, surface_calc, gslice_by_atom }
    }

    /// PySCF's `solvent.pcm.gen_surface`
    pub fn build(&mut self) {
        // unwrap atom_radii or calculate from atomic base radii
        let vdw_scale = self.cfg.effective_vdw_scale();
        let base_radii: &[f64] = match self.cfg.radius_scheme {
            RadiusScheme::Bondi => &data::VDW_RADII,
            RadiusScheme::UFF   => &data::UFF_RADII,
        };
        let atom_radii = self.cfg.atom_radii.clone().unwrap_or_else(|| {
            self.atomic_num.iter()
                .map(|&z| {
                    let mut r0 = base_radii[z];
                    // Bondi 对 H 有特殊的半径值 (1.1 Å vs 表里的 1.2 Å)
                    if z == 1 && self.cfg.radius_scheme == RadiusScheme::Bondi {
                        r0 = 2.0786987370215684;
                    }
                    r0 * vdw_scale
                })
                .collect()
        });

        // sanity check
        let natm = self.atomic_num.len();
        if atom_radii.len() != natm {
            panic!("Number of atom radii must match number of atom charges.");
        }
        if self.atom_coords.size[1] != natm {
            panic!("Number of atom coordinates must match number of atom charges.");
        }

        // basic quantities
        //let device = DeviceBLAS::default();
        let lebedev_degree = self.cfg.lebedev_degree;
        let unit_sphere = angular_grid(lebedev_degree);
        let unit_quads = unit_sphere.0;
        let unit_weights = unit_sphere.1;

        let R_J = atom_radii;

        let R_sw_J: Vec<f64> = R_J.iter().map(|&r| r * (14.0 / (lebedev_degree as f64)).sqrt()).collect();
        let alpha_J: Vec<f64> = R_J.iter().zip(R_sw_J.iter())
            .map(|(&r_vdw, &r_sw)| 0.5 + r_vdw/r_sw - ((r_vdw/r_sw).powi(2) - 1.0/28.0).sqrt())
            .collect();
        let R_in_J: Vec<f64> = R_J.iter().zip(alpha_J.iter()).zip(R_sw_J.iter())
            .map(|((&r_vdw, &alpha), &r_sw)| r_vdw - alpha * r_sw)
            .collect();


        let mut p = 0;
        for ia in 0..natm {
            let r_vdw = R_J[ia];

            // preparation for switch function
            // the grid points for atom ia, which is the unit sphere scaled by r_vdw and shifted by atom_coords
            let atom_grid: Vec<[f64; 3]> = unit_quads.iter()
                .map(|&(x,y,z)| [
                    r_vdw * x + self.atom_coords[(0, ia)],
                    r_vdw * y + self.atom_coords[(1, ia)],
                    r_vdw * z + self.atom_coords[(2, ia)]
                ]).collect();
            //let atom_grid = r_vdw * &unit_quads + atom_coords.i((ia, ..));

            let n_grids = atom_grid.len();
            // distance from the grid points of to all atoms, r_iJ, with shape (natm, n_grids)
            let mut r_iJ = MatrixFull::<f64>::new([natm, n_grids], 0.0);
            for j in 0..n_grids {
                for i in 0..natm {
                    r_iJ[(i, j)] = dist_point_atom(&atom_grid[j], &self.atom_coords, i);
                }
            }

            let w_i: Vec<f64> = unit_weights.iter().map(|&x| x * (4.0 * consts::PI)) .collect();
            //zeta_i = \frac{\zeta}{r_vdw \sqrt{w_i}}
            let zeta_i: Vec<f64> = w_i.iter().map(|&w| data::solvent_lebedev_scale_factor(lebedev_degree) / (r_vdw * w.sqrt())).collect();

            let mut d_iJ =MatrixFull::<f64>::new([natm,n_grids], 0.0);
            for j in 0..n_grids{
                for i in 0..natm{
                    if i == ia {
                        d_iJ[(i,j)] = 1.0;
                    }
                    else{
                        d_iJ[(i,j)] = (r_iJ[(i,j)] - R_in_J[i])/R_sw_J[i];

                    }
                }
            }
            // switch function
            //let f_iJ = match self.cfg.switch_type {
            //    SurfaceSwitchType::SWIG => SurfaceSwitchType::element_switch_swig(R_J.view(), r_iJ.view(), lebedev_degree),
            //    SurfaceSwitchType::ISWIG => SurfaceSwitchType::element_switch_iswig(R_J.view(), r_iJ.view(), zeta_i.view()),
            //};
            //let swf_i = SurfaceSwitchType::switch(f_iJ.view(), ia);
            let f_iJ = switch_matrix(d_iJ);
            // swf_i = numpy.prod(f_iJ, axis=0), means if a grid is in the cavity of any other atom, its switch function is zero
            let swf_i: Vec<f64> = f_iJ.iter_columns_full()
                .map(|col| col.iter().product())
                .collect();

            // zero out the switch function that is too small
            let idx: Vec<usize> = w_i.iter().zip(swf_i.iter())
                .map(|(&w, &s)| (w * s > 1e-16) as usize)
                .collect();
            let nidx = idx.iter().sum();
            //let nidx = idx.iter().filter(|&&x| x).count();
            let (p0, p1) = (p, p + nidx);

            let tmp_area:Vec<f64> = bool_select(&w_i, &idx).iter()
                .zip(bool_select(&swf_i, &idx).iter())
                .map(|(&w,&s)| r_vdw.powi(2) * w * s)
                .collect();
            self.gslice_by_atom.push((p0, p1));
            self.surface_calc.grid_coords.extend(bool_select(&atom_grid, &idx));
            self.surface_calc.weights.extend(bool_select(&w_i, &idx));
            self.surface_calc.switch_fun.extend(bool_select(&swf_i, &idx));
            self.surface_calc.norm_vec.extend(bool_select(&unit_quads, &idx));
            self.surface_calc.charge_exp.extend(bool_select(&zeta_i, &idx));
            self.surface_calc.R_vdw.extend(vec![r_vdw; nidx]);
            self.surface_calc.area.extend(tmp_area);

            p += nidx;
        }

    }

}


//-----------------------------------------------------------------------------
//  3D Coordinate utilities
//-----------------------------------------------------------------------------

/// Distance between two points given as [f64; 3]
#[inline]
pub fn dist_3d(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Distance between a point [f64; 3] and an atom stored in a [3, natm] column-major matrix
#[inline]
pub fn dist_point_atom(point: &[f64; 3], atom_coords: &MatrixFull<f64>, ia: usize) -> f64 {
    let dx = point[0] - atom_coords[(0, ia)];
    let dy = point[1] - atom_coords[(1, ia)];
    let dz = point[2] - atom_coords[(2, ia)];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Dot product of vector (a - b) with normal n (all as [f64; 3])
#[inline]
pub fn diff_dot_normal(a: &[f64; 3], b: &[f64; 3], n: &[f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * n[0] + dy * n[1] + dz * n[2]
}

//=============================================================================
//  Switch Functions
//=============================================================================

/// Utility function for boolean selection based on a mask
pub fn bool_select<T: Clone>(data: &[T], mask: &[usize]) -> Vec<T> {
    data.iter()
        .zip(mask.iter())
        .filter_map(|(d, &m)| if m != 0 { Some(d.clone()) } else { None })
        .collect()
}

/// switch function for a matrix of d_iJ
pub fn switch_matrix(d: MatrixFull<f64>) -> MatrixFull<f64> {
    let mut f = MatrixFull::<f64>::new(d.size, 0.0);
    for i in 0..d.size[0] {
        for j in 0..d.size[1] {
            if d[(i,j)] < 1e-8{
                f[(i,j)] = 0.0;
            }
            else{
                f[(i,j)] = switch_h(d[(i,j)]);
            }
        }
    }
    f
}

pub fn switch_h(x: f64) -> f64 {
    if x < 0.0 {
        0.0
    } else if x >= 1.0 {
        1.0
    } else {
        x.powi(3) * (10.0 - 15.0 * x + 6.0 * x.powi(2))
    }
}

//=============================================================================
//  Linear Algebra Utilities
//=============================================================================

/// Solve V·x = b where V is symmetric positive definite, using Cholesky decomposition.
/// Uses LAPACK's dpotrf (Cholesky) + forward/backward substitution.
///
/// This factors `V` on every call. When the same `V` is solved repeatedly (the
/// RI-PCM path solves the auxiliary-basis metric on every SCF iteration), factor
/// it once and use [`solve_cholesky_factorized`] instead: an `naux × naux`
/// factorization dominates the solve and is geometry-only.
pub fn solve_cholesky(v: &MatrixFull<f64>, b: &[f64]) -> Vec<f64> {
    let n = v.size[0];
    assert!(v.size[1] == n && b.len() == n, "solve_cholesky: dimension mismatch");

    // Copy V to mutable working matrix for in-place Cholesky
    let mut v_work = v.clone();

    // Cholesky factorization: V = L·L^T (L stored in lower triangle)
    v_work.to_matrixfullslicemut().lapack_dpotrf(b'L');

    solve_cholesky_factorized(&v_work, b)
}

/// Solve V·x = b from an **already computed** lower Cholesky factor of V.
///
/// `chol` must be the output of `lapack_dpotrf(b'L')` applied to V, i.e. V = L·Lᵀ
/// with L in the lower triangle; only that triangle is read. The two triangular
/// substitutions below are the same ones [`solve_cholesky`] performs, so the
/// result is bit-identical to factor-and-solve on the same V.
pub fn solve_cholesky_factorized(chol: &MatrixFull<f64>, b: &[f64]) -> Vec<f64> {
    let n = chol.size[0];
    assert!(chol.size[1] == n && b.len() == n, "solve_cholesky_factorized: dimension mismatch");

    let mut x = b.to_vec();

    // Forward substitution: L·y = b
    // L[i,j] = chol[(i, j)] for j <= i (lower triangle)
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..i {
            sum += chol[(i, j)] * x[j];
        }
        x[i] = (x[i] - sum) / chol[(i, i)];
    }

    // Backward substitution: L^T·x = y
    // L^T[i,j] = L[j,i] = chol[(j, i)]
    for i in (0..n).rev() {
        let mut sum = 0.0;
        for j in i+1..n {
            sum += chol[(j, i)] * x[j];
        }
        x[i] = (x[i] - sum) / chol[(i, i)];
    }

    x
}

/// Solve linear system Ax = b using LU decomposition without explicitly computing the inverse of A
pub fn solve_lu(a: &MatrixFull<f64>, a_ipiv: &Vec<i32>, b: &[f64]) -> Option<Vec<f64>>
{
    let n = a.size[0];
    // 2. 应用行置换到右端项 b
    let mut x = b.to_vec();
    for i in 0..n {
        if a_ipiv[i] as usize != i + 1 {
            x.swap(i, a_ipiv[i] as usize - 1);
        }
    }

    // 3. 前向替换求解 L·y = Pb (L 是单位下三角)
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..i {
            sum += a[(i, j)] * x[j];
        }
        x[i] = (x[i] - sum); // L[i,i] = 1.0
    }

    // 4. 后向替换求解 U·x = y
    for i in (0..n).rev() {
        let mut sum = 0.0;
        for j in i+1..n {
            sum += a[(i,j)] * x[j];
        }
        if a[(i,i)].abs() < 1e-15 {
            println!(
                "Singular matrix detected in solve_lu: n={}, pivot index={}, |U_ii|={:10.3e} (< 1e-15)",
                n, i, a[(i,i)].abs()
            );
            return None;
        }
        x[i] = (x[i] - sum) / a[(i,i)];
    }

    Some(x)
}

pub fn solve_lu_transpose(
    lu: &MatrixFull<f64>,   // 紧凑存储的 L 和 U
    ipiv: &[i32],           // 置换信息
    b: &[f64],              // 右端项
) -> Option<Vec<f64>> {
    let n = lu.size()[0];
    if b.len() != n { return None; }

    let mut x = b.to_vec();

    // 1. 解 U^T y = x (前向替换)
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..i {
            sum += lu[(j, i)] * x[j];
        }
        if lu[(i, i)].abs() < 1e-15 {
            println!(
                "Singular matrix detected in solve_lu_transpose: n={}, pivot index={}, |U_ii|={:10.3e} (< 1e-15)",
                n, i, lu[(i, i)].abs()
            );
            return None;
        }
        x[i] = (x[i] - sum) / lu[(i, i)];
    }

    // 2. 解 L^T z = y (后向替换，L^T 是单位上三角)
    for i in (0..n).rev() {
        let mut sum = 0.0;
        for j in i+1..n {
            sum += lu[(j, i)] * x[j];
        }
        x[i] -= sum;
    }

    // 3. 应用置换 P 得到最终解 x = P z
    // 注意：ipiv 记录的是分解时的行交换，正向应用即可
    for i in (0..n).rev() {   // 从后往前应用，避免覆盖问题
        let p = ipiv[i] as usize;
        if p != i + 1 {
            x.swap(i, p - 1);
        }
    }

    Some(x)
}

// ============================================================================
// Unit tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_radius_scheme_scales_and_parsing() {
        assert_eq!(RadiusScheme::Bondi.default_vdw_scale(), 1.2);
        assert_eq!(RadiusScheme::UFF.default_vdw_scale(), 1.1);
        assert_eq!(serde_json::from_str::<RadiusScheme>("\"bondi\"").unwrap(), RadiusScheme::Bondi);
        assert_eq!(serde_json::from_str::<RadiusScheme>("\"uff\"").unwrap(), RadiusScheme::UFF);
        assert!(serde_json::from_str::<RadiusScheme>("\"smd\"").is_err());
    }

    #[test]
    fn test_smd_cavity_radii_parsing() {
        assert_eq!(
            serde_json::from_str::<SmdCavityRadii>("\"bondi\"").unwrap(),
            SmdCavityRadii::Bondi
        );
        assert_eq!(
            serde_json::from_str::<SmdCavityRadii>("\"uff_mixed\"").unwrap(),
            SmdCavityRadii::BondiUff
        );
        assert!(serde_json::from_str::<SmdCavityRadii>("\"bogus\"").is_err());
    }

    /// R_O(α) 分段：α ≥ 0.43 → 1.52 Å；否则 1.52 + 1.8×(0.43−α)。返回值为 Bohr。
    #[test]
    fn test_smd_radii_oxygen_alpha_dependence() {
        let r = smd_radii(0.43, &[8], SmdCavityRadii::Bondi);
        assert!((r[0] - 1.52 / BOHR).abs() < 1e-12);
        let r = smd_radii(0.35, &[8], SmdCavityRadii::Bondi);
        assert!((r[0] - (1.52 + 1.8 * 0.08) / BOHR).abs() < 1e-12);
        let r = smd_radii(0.0, &[8], SmdCavityRadii::Bondi);
        assert!((r[0] - (1.52 + 1.8 * 0.43) / BOHR).abs() < 1e-12);
    }

    #[test]
    fn test_smd_radii_special_elements() {
        let r = smd_radii(0.82, &[1, 6, 7, 9, 15, 16, 17], SmdCavityRadii::Bondi);
        assert!((r[0] - 1.20 / BOHR).abs() < 1e-12); // H
        assert!((r[1] - 1.85 / BOHR).abs() < 1e-12); // C
        assert!((r[2] - 1.89 / BOHR).abs() < 1e-12); // N
        assert!((r[3] - 1.73 / BOHR).abs() < 1e-12); // F
        assert!((r[4] - 2.12 / BOHR).abs() < 1e-12); // P
        assert!((r[5] - 2.49 / BOHR).abs() < 1e-12); // S
        assert!((r[6] - 2.38 / BOHR).abs() < 1e-12); // Cl
    }

    /// 非 eq.16 元素按 scheme 走不同 fallback 表（Be 两表取值不同）。
    #[test]
    fn test_smd_radii_fallback_scheme() {
        let r_bondi = smd_radii(0.5, &[4], SmdCavityRadii::Bondi);
        let r_uff = smd_radii(0.5, &[4], SmdCavityRadii::BondiUff);
        assert!((r_bondi[0] - data::VDW_RADII[4]).abs() < 1e-12);
        assert!((r_uff[0] - data::BONDI_UFF_RADII[4]).abs() < 1e-12);
        assert!((r_bondi[0] - r_uff[0]).abs() > 0.1, "两套 fallback 表应显著不同");
    }
}
