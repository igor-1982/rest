//! Shared locality-screening utilities for grid-based kernels.
//!
//! Both the SCF AO tabulation (`Grids::prepare_tabulated_ao_sparse`) and the
//! analytical-gradient block pipeline (`dft::num_int::GridBlockProducer`) need
//! the same two decisions:
//!
//! 1. **Geometric pre-screen** — which atoms can contribute basis values above
//!    a cutoff anywhere inside a block of grid points
//!    ([`atom_influence_radii`] + [`block_bbox`] + [`atom_reaches_block`]).
//! 2. **Row screen** — which basis rows of an evaluated atom table survive a
//!    value cutoff ([`screen_rows_by_value`]).
//!
//! Keeping them in one place pins the two paths to identical screening
//! semantics. The radius envelope below is a rigorous upper bound of the
//! spherical GTO magnitude, so a geometric skip never drops a row that the
//! value screen would have kept (the SCF tabulation therefore stays
//! bit-identical when the pre-screen is added).

use rest_tensors::MatrixFull;

use crate::molecule_io::Molecule;

/// Conservative per-atom influence radii (bohr) for AO screening: beyond
/// `radius(a)` no basis function on atom `a` can exceed `cutoff` in magnitude.
///
/// The envelope of one shell is `A_l * C_contr * r^l * exp(-alpha_min r^2)`:
///
/// - `C_contr = sum_p |c_p| * Norm0(alpha_p, l)` bounds the primitive sum,
///   with `Norm0 = (2a/pi)^(3/4) * (4a)^(l/2)` the primitive normalization of
///   `cartesian_gto_batch_v04` (whose angular `Fang <= 1` and monomials obey
///   `|x^lx y^ly z^lz| <= r^l`).
/// - `A_l = sqrt((l+1)(l+2)/2)` bounds the 1-norm of every row of the
///   (orthogonal) c2s transform, i.e. the worst amplification of the
///   cartesian-to-spherical combination.
///
/// A 0.5 bohr margin is added on top. Derivatives amplify the value by a
/// modest polynomial factor at these distances; consumers that screen
/// derivative-driven contractions should pass a cutoff tightened accordingly
/// (the gradient pipeline uses `ao_cutoff * 1e-3`).
pub fn atom_influence_radii(mol: &Molecule, cutoff: f64) -> Vec<f64> {
    mol.basis4elem
        .iter()
        .map(|elem| {
            let mut r_max = 0.0_f64;
            for shell in &elem.electron_shells {
                let l = shell.angular_momentum.iter().map(|&x| x.max(0) as usize).max().unwrap_or(0);
                let alpha_min = shell.exponents.iter().cloned().fold(f64::INFINITY, f64::min);
                if !alpha_min.is_finite() || alpha_min <= 0.0 {
                    continue;
                }
                let n_cart = (l + 1) * (l + 2) / 2;
                let a_l = (n_cart as f64).sqrt();
                for coeffs in &shell.coefficients {
                    let c_contr: f64 = coeffs
                        .iter()
                        .zip(shell.exponents.iter())
                        .map(|(c, a)| {
                            c.abs()
                                * (2.0 * a / std::f64::consts::PI).powf(0.75)
                                * (4.0 * a).powf(l as f64 / 2.0)
                        })
                        .sum();
                    r_max = f64::max(r_max, envelope_radius(a_l * c_contr, l, alpha_min, cutoff));
                }
            }
            r_max + 0.5
        })
        .collect()
}

/// Largest r (bohr) at which `C * r^l * exp(-alpha r^2) >= cutoff`, solved by
/// bracketing the tail root and bisecting; 0 when the envelope never reaches
/// the cutoff (a function that is everywhere negligible).
pub fn envelope_radius(c: f64, l: usize, alpha: f64, cutoff: f64) -> f64 {
    if c <= 0.0 {
        return 0.0;
    }
    let f = |r: f64| c * r.powi(l as i32) * (-alpha * r * r).exp();
    let r_peak = (l as f64 / (2.0 * alpha)).sqrt();
    if f(r_peak) < cutoff {
        return 0.0;
    }
    let mut lo = r_peak.max(1.0e-8);
    let mut hi = (r_peak * 2.0).max(1.0);
    while f(hi) >= cutoff {
        hi *= 2.0;
        if hi > 1.0e4 {
            return hi;
        }
    }
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if f(mid) >= cutoff {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

/// Axis-aligned bounding box of a block of grid points, `[lo, hi]` per axis.
pub fn block_bbox(coords: &[[f64; 3]]) -> ([f64; 3], [f64; 3]) {
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for c in coords {
        for k in 0..3 {
            lo[k] = lo[k].min(c[k]);
            hi[k] = hi[k].max(c[k]);
        }
    }
    (lo, hi)
}

/// Whether an atom at `center` with influence `radius` can reach any point of
/// the block bounding box (center-to-box distance not exceeding the radius).
pub fn atom_reaches_block(center: &[f64; 3], radius: f64, lo: &[f64; 3], hi: &[f64; 3]) -> bool {
    let mut d2 = 0.0_f64;
    for k in 0..3 {
        let d = (lo[k] - center[k]).max(0.0) + (center[k] - hi[k]).max(0.0);
        d2 += d * d;
    }
    d2 <= radius * radius
}

/// Row screen of a per-atom AO table `[nbas, num_grids]` (column-major): keep
/// the indices of rows whose absolute value exceeds `cutoff` at some grid
/// point. This is the same predicate the SCF sparse AO tabulation applies.
pub fn screen_rows_by_value(vals: &MatrixFull<f64>, num_grids: usize, cutoff: f64) -> Vec<usize> {
    let nbas = vals.size[0];
    (0..nbas)
        .filter(|&r| (0..num_grids).any(|g| vals[[r, g]].abs() > cutoff))
        .collect()
}
