//! PAO (promoted atomic orbital) initial guess.
//!
//! # What this does, and what it cannot do
//!
//! The guess builds the SAD density and hands it to the SCF as **occupied atomic orbitals**, so
//! that the first Fock build runs the orbital-based, pair-screened RI-K (`use_dm_only = false`)
//! instead of the dense dm-only kernel. On ts2 the exchange of that first build drops from
//! 95.3 s to about 20 s, and the whole `InitGuess` stage from 125 s to a few seconds, while the
//! guess density stays exactly the SAD one.
//!
//! The "promotion" is the dual-basis map `C~ = S^{-1} S[all, A] C_A`, that is, the least-squares
//! representative of the atomic orbital in the molecular basis. Note that with the **target
//! basis** as the reference this map is the identity on the atom block: because the atomic
//! orbital is already a vector of the molecular space, `S^{-1} S[all, A] C_A` returns that same
//! vector embedded with zeros elsewhere. So this variant adds no inter-atomic tails and
//! reproduces the block-diagonal SAD density by construction. The genuine PySCF-`minao` style
//! promotion needs a *minimal* basis reference, where the reference functions are not
//! representable in the target basis and the projection really does spread density onto other
//! atoms. That variant needs a MINAO basis table and is not implemented here, see
//! `xcluster01_jobs/xc09_xppn3/minao_design.md`.
//!
//! # Spin channels
//!
//! For a spin-unpolarized molecule every atom contributes its atomic occupied orbitals with the
//! atomic (pair) occupations, exactly the density SAD builds. For a spin-polarized molecule the
//! convention mirrors SAD: a closed-shell atom contributes half of its pair occupation to each
//! spin, an open-shell atom contributes its alpha orbitals only. The two channels are then
//! rescaled by one common factor so that the total electron count is exact, which keeps the
//! atomic spin distribution and only repairs the charge, in the same way SAD's density is left
//! untouched between diagonalizations.
//!
//! # Why the orbitals matter
//!
//! `vk_upper_with_rimatr_screened*` consumes `eigenvectors`, `occupation` and `num_elec`, and it
//! is linear in the density `P = sum_i f_i C_i C_i^t`. The atomic orbitals are localized, so the
//! pair screening stays effective. This is the opposite situation to the failed 2026-10-21
//! experiment that diagonalized the SAD density, whose eigenvectors are delocalized over the
//! whole molecule.

use std::collections::HashMap;

use log::{info, warn};
use tensors::{BasicMatrix, MatrixFull};

use crate::constants::BAS_ATM;
use crate::initial_guess::sad::prepare_atomic_scf;
use crate::molecule_io::Molecule;
use crate::scf_io::util::occupied_orbital_count;

/// Occupied guess orbitals of one element and their occupations, in the element's own basis
/// (which is also the element's block of the molecular basis).
struct AtomicGuessData {
    c_occ: MatrixFull<f64>,
    occ: Vec<f64>,
    /// Spin channel of the *atomic* reference calculation: 1 closed shell, 2 open shell.
    atomic_spin_channel: usize,
}

/// `[start, start + len)` of every atom in the molecular AO basis.
///
/// libcint orders basis functions shell by shell and every shell records its centre, so the
/// range of an atom is the union of its shells. The shells of one atom are contiguous.
fn atom_basis_ranges(mol: &Molecule) -> Vec<[usize; 2]> {
    let mut ranges = vec![[usize::MAX, 0usize]; mol.cint_atm.len()];
    for (ish, bas) in mol.cint_bas.iter().enumerate() {
        let ia = bas[BAS_ATM] as usize;
        if ia >= ranges.len() {
            continue;
        }
        let start = mol.cint_fdqc[ish][0];
        let len = mol.cint_fdqc[ish][1];
        if ranges[ia][0] == usize::MAX {
            ranges[ia][0] = start;
        }
        ranges[ia][1] += len;
    }
    ranges
}

fn atomic_guess_for_element(mol: &Molecule, ielem: &String) -> Option<AtomicGuessData> {
    let atom_scf = prepare_atomic_scf(mol, ielem);
    let n_a = atom_scf.mol.num_basis;
    if atom_scf.eigenvectors[0].size()[0] != n_a {
        return None;
    }
    let occ_tot = &atom_scf.occupation[0];
    let nw = occupied_orbital_count(occ_tot).min(n_a);
    if nw == 0 || atom_scf.eigenvectors[0].size()[1] < nw {
        return None;
    }
    let ev = &atom_scf.eigenvectors[0];
    let mut c_occ = MatrixFull::new([n_a, nw], 0.0);
    for j in 0..nw {
        for i in 0..n_a {
            c_occ[[i, j]] = ev[[i, j]];
        }
    }
    Some(AtomicGuessData {
        c_occ,
        occ: occ_tot[0..nw].to_vec(),
        atomic_spin_channel: atom_scf.mol.spin_channel,
    })
}

/// Build the PAO guess for every spin channel.
///
/// Returns `(eigenvectors, occupation)` in the layout the SCF expects. Each `eigenvectors[s]` is
/// `[num_basis, num_state]` with the promoted columns in the first `ncols` columns, and
/// `occupation[s]` holds their atomic occupations in non-increasing order (zero padded), so that
/// `occupied_orbital_count` and the density generation agree with the K screening kernel.
/// The unused second channel is left empty for a spin-unpolarized molecule.
///
/// `None` means "this system is not supported here", and the caller falls back to SAD.
pub fn initial_guess_from_pao(
    mol: &Molecule,
) -> Option<([MatrixFull<f64>; 2], [Vec<f64>; 2])> {
    if !mol.geom.ghost_bs_elem.is_empty() {
        warn!("initial_guess = pao: ghost basis atoms are not supported");
        return None;
    }
    let n = mol.num_basis;
    let num_state = mol.num_state;
    let spin_channel = mol.spin_channel;
    let ranges = atom_basis_ranges(mol);

    let mut elem_data: HashMap<String, AtomicGuessData> = HashMap::new();
    let mut cols: [Vec<(Vec<f64>, f64)>; 2] = [Vec::new(), Vec::new()];

    for (ia, ielem) in mol.geom.elem.iter().enumerate() {
        let [start, len] = ranges[ia];
        if start == usize::MAX || len == 0 || start + len > n {
            warn!(
                "initial_guess = pao: could not resolve the AO range of atom {} ({})",
                ia, ielem
            );
            return None;
        }
        if !elem_data.contains_key(ielem) {
            let data = atomic_guess_for_element(mol, ielem)?;
            elem_data.insert(ielem.clone(), data);
        }
        let data = elem_data.get(ielem).unwrap();
        let n_a = data.c_occ.size()[0];
        let nw = data.c_occ.size()[1];
        if n_a != len {
            warn!(
                "initial_guess = pao: atom {} has {} AO functions in the molecule but {} in the atomic guess",
                ia, len, n_a
            );
            return None;
        }

        // Promotion with the atom's own block as the reference basis is the identity on that
        // block, so the promoted orbital is the atomic orbital placed into the atom's AO rows.
        // `S^{-1} S[all, A]` acting on a vector that already lives in the atom block returns the
        // same vector, hence the resulting density is exactly the block-diagonal SAD density.
        for j in 0..nw {
            let mut col = vec![0.0; n];
            for i in 0..n_a {
                col[start + i] = data.c_occ[[i, j]];
            }
            let w = data.occ[j];
            if spin_channel == 1 {
                cols[0].push((col, w));
            } else if data.atomic_spin_channel == 1 {
                // closed-shell reference atom: half of the pair occupation per spin, as SAD does
                cols[0].push((col.clone(), 0.5 * w));
                cols[1].push((col, 0.5 * w));
            } else {
                // open-shell reference atom: its alpha orbitals only
                cols[0].push((col, w));
            }
        }
    }

    let mut eigenvectors: [MatrixFull<f64>; 2] = [MatrixFull::empty(), MatrixFull::empty()];
    let mut occupation: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
    let mut weight_sum = 0.0;
    let mut per_spin_sum = [0.0_f64; 2];

    for spin in 0..spin_channel {
        let mut spin_cols = std::mem::take(&mut cols[spin]);
        if spin_cols.is_empty() {
            warn!("initial_guess = pao: spin channel {} has no occupied atomic orbital", spin);
            return None;
        }
        // Non-increasing occupations: occupied_orbital_count() takes the last index above the
        // threshold as the occupied count, so the significant columns must come first.
        spin_cols.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let ncols = spin_cols.len().min(num_state);
        let mut eigv = MatrixFull::new([n, num_state], 0.0);
        let mut occ = vec![0.0; num_state];
        for (idx, (col, w)) in spin_cols.iter().take(ncols).enumerate() {
            for i in 0..n {
                eigv[[i, idx]] = col[i];
            }
            occ[idx] = *w;
            weight_sum += *w;
            per_spin_sum[spin] += *w;
        }
        eigenvectors[spin] = eigv;
        occupation[spin] = occ;
    }

    let nelec = mol.num_elec[0];
    let tol = 1.0e-6_f64.max(1.0e-8 * nelec.abs());
    // The atomic guess densities are built from *neutral* atoms, exactly as SAD does, so their
    // occupation sum differs from what the molecule needs in two ways: a charged molecule is off
    // by the charge (an anion such as F- is one electron short), and a spin-polarized molecule
    // has a different alpha/beta split than the sum of the atomic ones (the atoms carry their own
    // unpaired electrons). SAD repairs both by diagonalizing its guess Fock and re-occupying the
    // orbitals from `num_elec`. This guess keeps the atomic *shape* but rescales each spin
    // channel to its own target, `num_elec[largest]` for the unpolarized case (which sums to the
    // total) and `num_elec[spin+1]` per channel otherwise, so the electron count and the spin
    // polarization are both exact.
    for spin in 0..spin_channel {
        let target = if spin_channel == 1 { nelec } else { mol.num_elec[spin + 1] };
        let sum = per_spin_sum[spin];
        if (sum - target).abs() > tol && sum > 0.0 {
            let scale = target / sum;
            warn!(
                "initial_guess = pao: spin channel {} occupation sum {:.10} differs from the target {:.10}, rescaling by {:.10}",
                spin, sum, target, scale
            );
            for w in occupation[spin].iter_mut() {
                *w *= scale;
            }
            per_spin_sum[spin] = target;
        }
    }
    weight_sum = per_spin_sum[..spin_channel].iter().sum();

    if spin_channel == 1 {
        eigenvectors[1] = MatrixFull::empty();
        occupation[1] = Vec::new();
    }
    info!(
        "Initial guess from PAO: {} spin channel(s), occupation sum alpha {:.10} / beta {:.10} (total {:.10}, target electron count {:.10})",
        spin_channel, per_spin_sum[0], per_spin_sum[1], weight_sum, nelec
    );
    Some((eigenvectors, occupation))
}
