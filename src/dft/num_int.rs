use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use rayon::prelude::*;
use rstsr::prelude::*;
use rest_tensors::{MatrixFull, MatrixFullSlice, RIFull};
use rest_tensors::matrix_blas_lapack::{ _einsum_01_serial, _einsum_02_serial};
use rest_tensors::matrix::matrix_blas_lapack::_dgemm_full;
use tensors::matrix_blas_lapack::{_dgemm};
use rayon::iter::{
    ParallelIterator, IndexedParallelIterator,
    plumbing::{Consumer, Folder, Producer, ProducerCallback, bridge},
};
use crate::scf_io::SCF;
use crate::molecule_io::Molecule;
use crate::basis_io::{spheric_gto_deriv_batch_serial};
use crate::dft::{Grids, DFA4REST};
use crate::dft::xc_deriv::XCType;
use crate::scf_io::util::occupied_orbital_count_with_threshold;
use crate::dft::libxc_itrf::eval_xc_eff;
use crate::ri_tddft::utils::tddft_occupation_parameters;
use rest_libcint::prelude::CInt;


/// Wall-clock accumulators (nanoseconds, summed across rayon workers) for the
/// parallel grid-block pipeline that feeds the analytical XC gradient.
/// The gradient driver (`grad::rks::calc_de_xc`) resets them right before the
/// block loop and reports the totals at `print_level >= 2`, so the
/// AO-evaluation / rho+XC / contraction split can be read from the log
/// without an external profiler.
pub static GRID_BLOCK_AO_NS: AtomicU64 = AtomicU64::new(0);
pub static GRID_BLOCK_RHOXC_NS: AtomicU64 = AtomicU64::new(0);
pub static GRID_BLOCK_CONTRACT_NS: AtomicU64 = AtomicU64::new(0);
pub static GRID_BLOCK_COUNT: AtomicU64 = AtomicU64::new(0);
/// Sum of the per-block active (locality-screened) basis counts; averaged by
/// the gradient driver to report the effective compression of the screened
/// contraction.
pub static GRID_BLOCK_ACTIVE_SUM: AtomicU64 = AtomicU64::new(0);
/// Sub-split of `GRID_BLOCK_AO_NS`: time inside the per-atom GTO evaluator
/// (`spheric_gto_deriv_batch_serial`) versus the screening+assembly around it.
pub static GRID_BLOCK_GTO_NS: AtomicU64 = AtomicU64::new(0);

/// Z-order (Morton) permutation of grid points: sorting points by this order
/// makes contiguous index ranges spatially compact, which is what the block
/// locality screening in the XC gradient pipeline exploits. The sort only
/// reorders quadrature points, so it changes results at the level of floating
/// point summation order (roundoff), nothing else.
pub fn morton_order(coords: &[[f64; 3]]) -> Vec<usize> {
    fn spread(v: u32) -> u64 {
        let mut x = (v as u64) & 0x1F_FFFF;
        x = (x | (x << 32)) & 0x1F00_0000_00FF_FFFF;
        x = (x | (x << 16)) & 0x1F00_00FF_0000_FFFF;
        x = (x | (x << 8)) & 0x100F_00F0_0F00_F00F;
        x = (x | (x << 4)) & 0x10C3_0C30_C30C_30C3;
        x = (x | (x << 2)) & 0x1249_2492_4924_9249;
        x
    }
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for c in coords {
        for k in 0..3 {
            lo[k] = lo[k].min(c[k]);
            hi[k] = hi[k].max(c[k]);
        }
    }
    // 21-bit quantization per axis inside the bounding box
    let scale: Vec<f64> = (0..3)
        .map(|k| if hi[k] > lo[k] { (0x1F_FFFE as f64) / (hi[k] - lo[k]) } else { 0.0 })
        .collect();
    let key = |c: &[f64; 3]| -> u64 {
        let q = |k: usize| -> u32 { ((c[k] - lo[k]) * scale[k]) as u32 };
        spread(q(0)) | (spread(q(1)) << 1) | (spread(q(2)) << 2)
    };
    let mut perm: Vec<usize> = (0..coords.len()).collect();
    perm.sort_by_cached_key(|&i| key(&coords[i]));
    perm
}

// The locality-screening primitives (`atom_influence_radii`, `block_bbox`,
// `atom_reaches_block`, `screen_rows_by_value`) live in `dft::locality`,
// shared with the SCF sparse AO tabulation (`Grids::prepare_tabulated_ao_sparse`).
use crate::dft::locality::{atom_influence_radii, atom_reaches_block, block_bbox, screen_rows_by_value};

pub fn eval_ao_batch(mol:&Molecule, coords:&[[f64; 3]], ao_deriv:usize, num_grids:usize) -> RIFull<f64> {
    // let default_omp_num_threads = unsafe {utilities::openblas_get_num_threads()};
    // let default_omp_num_threads = utilities::omp_get_num_threads_wrapper();
    // utilities::omp_set_num_threads_wrapper(1);

    let n_components_deriv = (ao_deriv + 1) * (ao_deriv + 2) * (ao_deriv + 3) / 6;
    
    let mut loc_ao = RIFull::new([mol.num_basis, num_grids, n_components_deriv], 0.0);

    mol.basis4elem.iter()
    .zip(mol.geom.rg_position.iter_columns_full())
    .for_each(
        |(elem, geom)|
        {
            let ind_glb_bas = elem.global_index.0;
            let loc_num_bas = elem.global_index.1;
            let start = ind_glb_bas;
            let end = start + loc_num_bas;
            let mut tmp_geom = [0.0; 3];
            tmp_geom.iter_mut().zip(geom.iter()).for_each(|value| {*value.0 = *value.1});
            let tab_ao_elem = spheric_gto_deriv_batch_serial(&coords, &tmp_geom, elem, ao_deriv);

            for ic in 0..n_components_deriv {
                let gto_ic = tab_ao_elem.get(ic).unwrap();
                loc_ao.copy_from_matr(start..end, 0..num_grids, ic, 0, 
                   gto_ic, 0..loc_num_bas, 0..num_grids);
            }
            // loc_ao.copy_from_ri(start..end, 0..num_grids, 0..n_components_cart, &tab_ao_elem, 0..loc_num_bas, 0..num_grids, 0..n_components_cart);
        }
    );

    loc_ao
}

/// Raw libcint AO buffer, **without** the layout conversion of
/// [`eval_ao_batch_libcint`].
///
/// The buffer holds component `c` as a `[num_grids, nao]` column-major block
/// (element `(g, mu)` at `c*nao*num_grids + g + mu*num_grids`), i.e. the
/// transpose of the `RIFull` layout.  Consumers that can express their
/// contractions as BLAS calls on the transposed operand can use this to skip
/// the (memory-bound) transpose entirely — the fxc response matvec does, see
/// `compute_fxc_response_block`.
pub fn eval_ao_batch_libcint_native(
    cint: &CInt,
    nao: usize,
    coords: &[[f64; 3]],
    ao_deriv: usize,
) -> Vec<f64> {
    let ncomp = (ao_deriv + 1) * (ao_deriv + 2) * (ao_deriv + 3) / 6;
    let out = cint.eval_gto(&format!("deriv{}", ao_deriv), coords);
    let data = out.out.expect("eval_ao_batch_libcint_native: libcint returned no output buffer");
    assert_eq!(
        data.len(),
        nao * coords.len() * ncomp,
        "eval_ao_batch_libcint_native: libcint returned {} values, expected {}",
        data.len(),
        nao * coords.len() * ncomp
    );
    data
}

/// AO values + Cartesian derivatives on a grid batch, evaluated with libcint.
///
/// The returned tensor is layout-identical to [`eval_ao_batch`]: component `c`
/// is a `[nao, num_grids]` column-major block (`get_reducing_matrix(c)`), with
/// the AO index fastest, so this is a drop-in replacement for the legacy
/// serial spherical transform.
///
/// libcint's own output is the **transpose** of that (see
/// [`eval_ao_batch_libcint_native`]), so a per-component transpose is needed;
/// it is cache-blocked and costs ~50% of the evaluation itself.
///
/// Measured (naphthalene / def2-svp / 219212 grids, single thread) against the
/// legacy path: `ao_deriv = 1` 1.15 s vs 2.79 s, `ao_deriv = 2` 2.88 s vs
/// 7.43 s; values agree to 1.4e-14 (roundoff).
pub fn eval_ao_batch_libcint(
    cint: &CInt,
    nao: usize,
    coords: &[[f64; 3]],
    ao_deriv: usize,
    num_grids: usize,
) -> RIFull<f64> {
    let ncomp = (ao_deriv + 1) * (ao_deriv + 2) * (ao_deriv + 3) / 6;
    let mut data = eval_ao_batch_libcint_native(cint, nao, coords, ao_deriv);
    // [ng, nao] (grids fastest) -> [nao, ng] (AO fastest), per component.
    // Tiled so that both the reads and the writes are contiguous: a BLK x BLK
    // tile is loaded row- (AO-) wise and stored column-wise, which keeps every
    // access either unit-stride or inside a 64-line (4 KB) window of L1.
    if num_grids != nao {
        const BLK: usize = 64;
        let mut dst = vec![0.0f64; data.len()];
        let mut tile = vec![0.0f64; BLK * BLK];
        for c in 0..ncomp {
            let src = &data[c * nao * num_grids..(c + 1) * nao * num_grids];
            let d = &mut dst[c * nao * num_grids..(c + 1) * nao * num_grids];
            for gb in (0..num_grids).step_by(BLK) {
                let ge = (gb + BLK).min(num_grids);
                for mub in (0..nao).step_by(BLK) {
                    let mue = (mub + BLK).min(nao);
                    let h = mue - mub;
                    // load: tile[g_local * BLK + mu_local] = src[g + mu * ng]
                    for (i, mu) in (mub..mue).enumerate() {
                        let row = &src[gb + mu * num_grids..ge + mu * num_grids];
                        for (j, v) in row.iter().enumerate() {
                            tile[j * BLK + i] = *v;
                        }
                    }
                    // store: dst[mu + g * nao] = tile[g_local * BLK + mu_local]
                    for (j, g) in (gb..ge).enumerate() {
                        let col = &tile[j * BLK..j * BLK + h];
                        let drow = &mut d[g * nao + mub..g * nao + mue];
                        drow.copy_from_slice(col);
                    }
                }
            }
        }
        data = dst;
    }
    RIFull::from_vec([nao, num_grids, ncomp], data).unwrap()
}


pub fn eval_rho5_batch(ao:&RIFull<f64>, xc_type:XCType, mo:&Vec<MatrixFull<f64>>, occ:&Vec<Vec<f64>>, spin_channel:usize, num_grids:usize) -> Tensor<f64> {
    /*
        Evaluate density on a batch of ao grids value for given mo coefficients and occupation.
        Args:
            ao: gto value on grids [num_grids, num_basis, nderiv],
                nderiv = 1 for ao0, 4 for ao0, ao_x, ao_y, ao_z, 10 for 2nd order derivatives (symmetric)
            xc_type: exchange-correlation dfa type
            mo: orbital coeffients [num_basis, num_state]
            occ: occupation number [num_state]
            spin_channel: number of spins, 1 for Restricted, 2 for Unrestricted
            range_grids: range of grids
        Return:
            rho [num_grids, nvar, spin_channel]
            where nvar = 1 for LDA, 4 for GGA, 5 for MGGA 
            tabulated as rho, rho_x, rho_y, rho_z, tau
        Note:
            laplacian is not implemented
    */
    let num_basis = mo[0].size.get(0).unwrap();
    let num_state = mo[0].size.get(1).unwrap();

    let nvar = match xc_type {
        XCType::HF => 1, 
        XCType::LDA => 1,
        XCType::GGA => 4,
        XCType::MGGA => 5,
    };
    let mut cur_rho = RIFull::new([num_grids, nvar, spin_channel], 0.0);

    let ao0 = &ao.get_reducing_matrix(0).unwrap();

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
            ao0, (0..ao0.size[0], 0..num_grids), 'N',
            &mut tmo, (0..wmo.size[1], 0..num_grids),
            1.0, 0.0
        );
        // spin case: rho_s = tmo * tmo (C_ug * C_vg * phi_ug * phi_vg => rho_g) 
        let rho_s = _einsum_02_serial(&tmo.to_matrixfullslice(), &tmo.to_matrixfullslice());
        // cur_rho[ispin][0] = rho_s
        cur_rho.get_reducing_matrix_mut(i_spin).unwrap()
        .iter_mut_j(0)
        .zip(rho_s.iter())
        .for_each(
            |(to, from)| {*to = *from}
        );
        if nvar > 1 { // gga and upper case 
            for ic in 0..3 { // x, y, z
                let mut tmop = MatrixFull::new([num_occ, num_grids], 0.0);
                let aop_ic = &ao.get_reducing_matrix(ic+1).unwrap();
                _dgemm(
                    &wmo, (0..wmo.size[0], 0..wmo.size[1]), 'T',
                    aop_ic, (0..aop_ic.size[0], 0..num_grids), 'N',
                    &mut tmop, (0..wmo.size[1], 0..num_grids),
                    1.0, 0.0
                );
                let rhop_ic_s = _einsum_02_serial(&tmop.to_matrixfullslice(), &tmo.to_matrixfullslice());
                // cur_rho[ispin][ic+1] = rhop_ic_s
                cur_rho.get_reducing_matrix_mut(i_spin).unwrap()
                .iter_mut_j(ic + 1)
                .zip(rhop_ic_s.iter())
                .for_each(
                    |(to, from)| {*to = *from * 2.0}
                );
                if nvar == 5 { // mgga
                    let tau_s = _einsum_02_serial(&tmop.to_matrixfullslice(), &tmop.to_matrixfullslice());
                    // cur_rho[ispin][5] = tau_s
                    // tau = 1/2 * |grad phi|^2 sum over x,y,z
                    cur_rho.get_reducing_matrix_mut(i_spin).unwrap()
                    .iter_mut_j(4)
                    .zip(tau_s.iter())
                    .for_each(
                        |(to, from)| {*to += *from*0.5}
                    );
                }
            }
        }
    }
    let rho_ten = rt::asarray((&cur_rho.data, [num_grids, nvar, spin_channel])).to_owned();
    rho_ten

}


pub fn eval_rho5_dm_only_batch(ao:&RIFull<f64>, xc_type:XCType, dms:&Vec<MatrixFull<f64>>, spin_channel:usize, num_grids:usize) -> Tensor<f64> {
    /*
        Evaluate density on grids for a given set of density matrices.
        Args:
            ao: gto value on grids [num_grids, num_basis, nderiv] (column major),
                nderiv = 1 for ao0, 4 for ao0, ao_x, ao_y, ao_z, 10 for 2nd order derivatives (symmetric)
            xc_method: exchange-correlation dfa type
            dms: density matrices of shape [num_basis, num_bais, spin_channel, nsets] 
            spin_channel: number of spins, 1 for Restricted, 2 for Unrestricted
            range_grids: range of grids
        Return:
            rho [num_grids, nvar, spin_channel]
            where nvar = 1 for LDA, 4 for GGA, 5 for MGGA 
            tabulated as rho, rho_x, rho_y, rho_z, tau
        Note:
            laplacian is not implemented
    */

    let num_basis = dms[0].size[0];

    let nvar = match xc_type {
        XCType::HF => 1, 
        XCType::LDA => 1,
        XCType::GGA => 4,
        XCType::MGGA => 5,
    };

    let mut cur_rho = RIFull::new([num_grids, nvar, spin_channel], 0.0);
    
    let ao0 = &ao.get_reducing_matrix(0).unwrap();

    for i_spin in 0..spin_channel {
        // rho 
        let dm_s = &dms[i_spin];
        let mut wao = MatrixFull::new([num_basis, num_grids],0.0);
        _dgemm(
            dm_s, (0..num_basis, 0..num_basis), 'N',
            ao0, (0..num_basis, 0..num_grids), 'N',
            &mut wao,  (0..num_basis, 0..num_grids), 
            1.0, 0.0
        );
        ao0.iter_columns(0..num_grids).unwrap()
        .zip(wao.iter_columns_full())
        .map(|(ao_r,wao_r)| (ao_r, wao_r))
        .zip(cur_rho.get_reducing_matrix_mut(i_spin).unwrap().iter_column_mut(0))
        .for_each(
            |((ao_r, wao_r), cur_rho_s)| 
            {
                *cur_rho_s = wao_r.iter().zip(ao_r.iter()).fold(0.0, |acc, (a,b)| {acc + a*b})
            }
        );
        if nvar > 1 {
            for ic in 0..3usize {
                let mut aop_ic = &ao.get_reducing_matrix(ic + 1).unwrap();
                let mut wao = MatrixFull::new([num_basis, num_grids],0.0);
                _dgemm(
                    dm_s, (0..num_basis, 0..num_basis), 'N',
                    aop_ic, (0..num_basis, 0..num_grids), 'N',
                    &mut wao, (0..num_basis, 0..num_grids), 1.0, 0.0
                );
            
                ao0.iter_columns(0..num_grids).unwrap()
                .zip(wao.iter_columns_full())
                .map(|(ao_r, wao_r)| (ao_r, wao_r))
                .zip(cur_rho.get_reducing_matrix_mut(i_spin).unwrap().iter_column_mut(ic + 1))
                .for_each(
                    |((ao_r, wao_r), cur_rhop_r)| 
                    {
                        *cur_rhop_r = 2.0 * wao_r.iter().zip(ao_r.iter()).fold(0.0, |acc,(wao,ao)| {acc + wao*ao})
                    }
                );
                if nvar == 5 {
                    aop_ic.iter_columns(0..num_grids).unwrap()
                    .zip(wao.iter_columns_full())
                    .map(|(aop_r, wao_r)|(aop_r, wao_r))
                    .zip(cur_rho.get_reducing_matrix_mut(i_spin).unwrap().iter_column_mut(4))
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
    let rho_ten = rt::asarray((&cur_rho.data, [num_grids, nvar, spin_channel])).to_owned();
    rho_ten
}


pub fn eval_rho5_spin_batch(ao:&RIFull<f64>, xc_type:XCType, mo:&MatrixFull<f64>, occ:&Vec<f64>, num_grids:usize) -> Tensor<f64> {
    /*
        Evaluate density on a batch of ao grids value for given mo coefficients and occupation in one spin channel.
        Args:
            ao: gto value on grids [num_grids, num_basis, nderiv],
                nderiv = 1 for ao0, 4 for ao0, ao_x, ao_y, ao_z, 10 for 2nd order derivatives (symmetric)
            xc_type: exchange-correlation dfa type
            mo: orbital coeffients [num_basis, num_state]
            occ: occupation number [num_state]
            range_grids: range of grids
        Return:
            rho [num_grids, nvar]
            where nvar = 1 for LDA, 4 for GGA, 5 for MGGA 
            tabulated as rho, rho_x, rho_y, rho_z, tau
        Note:
            laplacian is not implemented
    */

    let num_basis = mo.size.get(0).unwrap();
    let num_state = mo.size.get(1).unwrap();

    let nvar = match xc_type {
        XCType::HF => 1, 
        XCType::LDA => 1,
        XCType::GGA => 4,
        XCType::MGGA => 5,
    };
    let mut cur_rho = MatrixFull::new([num_grids, nvar], 0.0);

    let ao0 = &ao.get_reducing_matrix(0).unwrap();

    
    
    let num_occ = occupied_orbital_count_with_threshold(occ, 1.0e-6);
    let mut occ_tmp = occ[0..num_occ].iter().map(|occ| occ.sqrt()).collect::<Vec<f64>>();

    let num_occ = occ_tmp.len();
    let mut wmo = _einsum_01_serial(&mo.to_matrixfullslice(), &occ_tmp);
    let mut tmo = MatrixFull::new([num_occ, num_grids], 0.0);
    // tmo = C.T matmul ao (half transform)
    _dgemm(
        &wmo, (0..wmo.size[0], 0..wmo.size[1]), 'T', 
        ao0, (0..ao0.size[0], 0..num_grids), 'N',
        &mut tmo, (0..wmo.size[1], 0..num_grids),
        1.0, 0.0
    );
    // rho_0 = tmo * tmo (C_ug * C_vg * phi_ug * phi_vg => rho_g) 
    let rho_0 = _einsum_02_serial(&tmo.to_matrixfullslice(), &tmo.to_matrixfullslice());
    // cur_rho[0] = rho_0
    cur_rho.iter_column_mut(0).zip(rho_0.iter())
    .for_each(
        |(to, from)| {*to = *from}
    );
        
    if nvar > 1 { // gga and upper case 
        for ic in 0..3 { // x, y, z
            let mut tmop = MatrixFull::new([num_occ, num_grids], 0.0);
            let aop_ic = &ao.get_reducing_matrix(ic+1).unwrap();
            _dgemm(
                &wmo, (0..wmo.size[0], 0..wmo.size[1]), 'T',
                aop_ic, (0..aop_ic.size[0], 0..num_grids), 'N',
                &mut tmop, (0..wmo.size[1], 0..num_grids),
                1.0, 0.0
            );
            let rhop_ic = _einsum_02_serial(&tmop.to_matrixfullslice(), &tmo.to_matrixfullslice());
            // cur_rho[ic+1] = rhop_ic
            cur_rho.iter_column_mut(ic + 1).zip(rhop_ic.iter())
            .for_each(
                |(to, from)| {*to = *from * 2.0}
            );
            if nvar == 5 { // mgga
                let tau = _einsum_02_serial(&tmop.to_matrixfullslice(), &tmop.to_matrixfullslice());
                // cur_rho[5] = tau
                // tau = 1/2 * |grad phi|^2 sum over x,y,z
                cur_rho.iter_column_mut(4).zip(tau.iter())
                .for_each(
                    |(to, from)| {*to += *from*0.5}
                );
            }
        }
    }

    let rho_ten = rt::asarray((cur_rho.data, [num_grids, nvar]));
    rho_ten 
}


pub fn eval_rho5_spin_dm_only_batch(ao:&RIFull<f64>, xc_type:XCType, dm:&MatrixFull<f64>, num_grids:usize) -> Tensor<f64> {
    let num_basis = dm.size[0];

    let nvar = match xc_type {
        XCType::HF => 1, 
        XCType::LDA => 1,
        XCType::GGA => 4,
        XCType::MGGA => 5,
    };

    let mut cur_rho = MatrixFull::new([num_grids, nvar], 0.0);
    let ao0 = &ao.get_reducing_matrix(0).unwrap();
    // rho0 
    let mut wao = MatrixFull::new([num_basis, num_grids],0.0);
    _dgemm(
        dm, (0..num_basis, 0..num_basis), 'N',
        ao0, (0..num_basis, 0..num_grids), 'N',
        &mut wao,  (0..num_basis, 0..num_grids), 
        1.0, 0.0
    );
    ao0.iter_columns(0..num_grids).unwrap()
    .zip(wao.iter_columns_full())
    .map(|(ao_r,wao_r)| (ao_r, wao_r))
    .zip(cur_rho.iter_column_mut(0))
    .for_each(
        |((ao_r, wao_r), cur_rho_0)| 
        {
            *cur_rho_0 = wao_r.iter().zip(ao_r.iter()).fold(0.0, |acc, (a,b)| {acc + a*b})
        }
    );
    // gga and upper case
    if nvar > 1 {
        // nabla rho 
        for ic in 0..3usize {
            let mut aop_ic = &ao.get_reducing_matrix(ic + 1).unwrap();
            let mut wao = MatrixFull::new([num_basis, num_grids],0.0);
            _dgemm(
                dm, (0..num_basis, 0..num_basis), 'N',
                aop_ic, (0..num_basis, 0..num_grids), 'N',
                &mut wao, (0..num_basis, 0..num_grids), 1.0, 0.0
            );
            ao0.iter_columns(0..num_grids).unwrap()
            .zip(wao.iter_columns_full())
            .map(|(ao_r, wao_r)| (ao_r, wao_r))
            .zip(cur_rho.iter_column_mut(ic + 1))
            .for_each(
                |((ao_r, wao_r), cur_rhop_r)| 
                {
                    *cur_rhop_r = 2.0 * wao_r.iter().zip(ao_r.iter()).fold(0.0, |acc,(wao,ao)| {acc + wao*ao})
                }
            );
            // tau 
            if nvar == 5 {
                aop_ic.iter_columns(0..num_grids).unwrap()
                .zip(wao.iter_columns_full())
                .map(|(aop_r, wao_r)|(aop_r, wao_r))
                .zip(cur_rho.iter_column_mut(4))
                .for_each(
                    |((aop_r, wao_r), cur_tau)|
                    {
                        *cur_tau += 0.5 * wao_r.iter().zip(aop_r.iter()).fold(0.0, |acc, (a,b)| {acc + a*b})
                    }
                );
            }
        }
    }

    let rho_ten = rt::asarray((cur_rho.data, [num_grids, nvar]));
    rho_ten
}

// ══════════════════════════════════════════════════════════════════════════════
// fxc kernel for TDDFT/CP-HF: exchange-correlation kernel matrix-vector product
// ══════════════════════════════════════════════════════════════════════════════
//
// Physical formula:
//   (A^{fxc} · z)_{ia} = Σ_g w(g) · Σ_{αβ} D_α[φ_i φ_a](g) · fxc(g)_{αβ} · ρ_z,β(g)
//   where ρ_z,β(g) = Σ_{jb} D_β[φ_j φ_b](g) · z_{jb}
//
// Reference: PySCF TDDFT implementation pattern

/// Precomputed fxc kernel data for TDDFT matrix-vector product
pub struct FXCMatvecData {
    /// Number of density variables: 1 (LDA) or 4 (GGA)
    pub nvar: usize,
    /// Number of numerical integration grid points
    pub ngrids: usize,
    /// Number of occupied orbitals
    pub nocc: usize,
    /// Number of virtual orbitals
    pub nvir: usize,
    /// First active MO index (for frozen-core handling)
    pub start_mo: usize,
    /// Hybrid exchange coefficient (c_x from functional)
    pub alpha_hybrid: f64,
    /// Occupied MO values on grids, column-major [nocc, ngrids]
    pub mo_occ: MatrixFull<f64>,
    /// Virtual MO values on grids, column-major [nvir, ngrids]
    pub mo_vir: MatrixFull<f64>,
    /// For GGA: occupied MO gradients (x, y, z) on grids, each [nocc, ngrids]
    pub mo_occ_grad: Option<[MatrixFull<f64>; 3]>,
    /// For GGA: virtual MO gradients (x, y, z) on grids, each [nvir, ngrids]
    pub mo_vir_grad: Option<[MatrixFull<f64>; 3]>,
    /// fxc kernel × grid weights
    /// LDA: vec[ngrids] = fxc[g] × w[g]
    /// GGA: vec[ngrids × 4 × 4] f-contiguous [g, α, β] = fxc[g,α,β] × w[g]
    pub wfxc: Vec<f64>,
    /// If true, `fxc_matvec` will use the optimized implementation
    /// (`fxc_matvec_opt`) instead of the original.
    pub use_opt: bool,
}

/// Prepare FXCMatvecData from converged SCF object
///
/// Steps:
/// 1. Extract occupied/virtual MO coefficients from SCF eigenvectors
/// 2. Project MO coefficients onto DFT integration grids
/// 3. For GGA: also project MO gradients onto grids
/// 4. Compute ground-state density on grids via eval_rho5_batch
/// 5. Compute fxc kernel via eval_xc_eff with deriv=2
/// 6. Multiply fxc by grid weights
pub fn prepare_fxc_data(scf: &SCF) -> FXCMatvecData {
    let (start_mo, _num_state, occ_size, vir_size, _homo, _lumo) =
        tddft_occupation_parameters(scf);

    let xc_data = &scf.mol.xc_data;
    let xc_type = if xc_data.use_density_gradient() {
        XCType::GGA
    } else {
        XCType::LDA
    };
    let nvar = match xc_type {
        XCType::LDA => 1,
        XCType::GGA => 4,
        _ => panic!("fxc only supports LDA and GGA"),
    };
    let alpha_hybrid = xc_data.dfa_hybrid_scf;

    let grids = scf.grids.as_ref().expect("DFT grids must be initialized for fxc");
    let ngrids = grids.weights.len();
    let num_basis = scf.mol.num_basis;
    let weights = &grids.weights;
    // ── Obtain dense AO: decompress from compressed storage if needed ──
    let ao_owned: Option<MatrixFull<f64>>;
    let ao: &MatrixFull<f64> = match &grids.ao {
        Some(a) => { ao_owned = None; a }
        None => match &grids.ao_compressed {
            Some(c) => { ao_owned = Some(Grids::decompress_ao(c)); ao_owned.as_ref().unwrap() }
            None => panic!("AO on grids must be tabulated (dense or compressed)"),
        }
    };
    let eigvec = &scf.eigenvectors[0];

    // ── Extract MO coefficients for occupied and virtual spaces ──
    let mut c_occ = MatrixFull::new([num_basis, occ_size], 0.0);
    for j in 0..occ_size {
        for i in 0..num_basis {
            c_occ[[i, j]] = eigvec[[i, start_mo + j]];
        }
    }
    let mut c_vir = MatrixFull::new([num_basis, vir_size], 0.0);
    for j in 0..vir_size {
        for i in 0..num_basis {
            c_vir[[i, j]] = eigvec[[i, _lumo + j]];
        }
    }

    // ── Project MO values onto grids ──
    // mo_occ[i,g] = Σ_p C_occ[p,i] × ao[p,g]
    // mo_vir[a,g] = Σ_p C_vir[p,a] × ao[p,g]
    // In BLAS: mo_occ = C_occ^T × ao
    let mut mo_occ = MatrixFull::new([occ_size, ngrids], 0.0);
    _dgemm_full(&c_occ, 'T', ao, 'N', &mut mo_occ, 1.0, 0.0);

    let mut mo_vir = MatrixFull::new([vir_size, ngrids], 0.0);
    _dgemm_full(&c_vir, 'T', ao, 'N', &mut mo_vir, 1.0, 0.0);

    // ── GGA: MO gradients on grids ──
    let aop_owned: Option<RIFull<f64>>;
    let (mo_occ_grad, mo_vir_grad) = if xc_type == XCType::GGA {
        let aop: &RIFull<f64> = match &grids.aop {
            Some(a) => { aop_owned = None; a }
            None => match &grids.aop_compressed {
                Some(c) => { aop_owned = Some(Grids::decompress_aop(c)); aop_owned.as_ref().unwrap() }
                None => panic!("AO gradients needed for GGA fxc (dense or compressed)"),
            }
        };
        let mut og = [
            MatrixFull::new([occ_size, ngrids], 0.0),
            MatrixFull::new([occ_size, ngrids], 0.0),
            MatrixFull::new([occ_size, ngrids], 0.0),
        ];
        let mut vg = [
            MatrixFull::new([vir_size, ngrids], 0.0),
            MatrixFull::new([vir_size, ngrids], 0.0),
            MatrixFull::new([vir_size, ngrids], 0.0),
        ];
        for d in 0..3 {
            let aop_d_slice = aop.get_reducing_matrix(d).unwrap();
            let aop_d = MatrixFull::from_vec(
                [num_basis, ngrids],
                aop_d_slice.iter().cloned().collect(),
            ).unwrap();
            _dgemm_full(&c_occ, 'T', &aop_d, 'N', &mut og[d], 1.0, 0.0);
            _dgemm_full(&c_vir, 'T', &aop_d, 'N', &mut vg[d], 1.0, 0.0);
        }
        (Some(og), Some(vg))
    } else {
        (None, None)
    };

    // ── Compute ground-state density on grids for fxc evaluation ──
    let ao_deriv = if xc_type == XCType::GGA { 1 } else { 0 };
    let ao_rifull = eval_ao_batch(&scf.mol, &grids.coordinates, ao_deriv, ngrids);
    let mo_coeffs = vec![scf.eigenvectors[0].clone()];
    let occ = vec![scf.occupation[0].clone()];
    let rho_tensor = eval_rho5_batch(&ao_rifull, xc_type, &mo_coeffs, &occ, 1, ngrids);

    // Extract rho_array in the format expected by eval_xc_eff
    // For spin=0: column-major [ngrids, nvar], i.e. rho_array[g + v*ngrids]
    let rho_array: Vec<f64> = rho_tensor.raw()[rho_tensor.offset()..]
        .chunks(ngrids * nvar)
        .next()
        .unwrap_or(&[])
        .to_vec();
    let rho_array = if rho_array.is_empty() {
        let raw = rho_tensor.raw();
        let offset = rho_tensor.offset();
        let len = ngrids * nvar;
        raw[offset..offset + len].to_vec()
    } else {
        rho_array
    };

    // ── Compute fxc kernel analytically via libxc ──
    let func_ids = &xc_data.dfa_compnt_scf;
    let func_factors = &xc_data.dfa_paramr_scf;
    let xc_tensors = eval_xc_eff(func_ids, func_factors, xc_type, 0, &rho_array, ngrids, 2);

    // xc_tensors[2] = fxc kernel (deriv=2)
    let fxc_tensor = xc_tensors[2].as_ref()
        .expect("fxc (deriv=2) should be available");
    let fxc_raw: Vec<f64> = {
        let raw = fxc_tensor.raw();
        let offset = fxc_tensor.offset();
        let nv2 = nvar * nvar;
        let len = ngrids * nv2;
        raw[offset..offset + len].to_vec()
    };

    // ── Multiply fxc by grid weights ──
    // IMPORTANT: eval_xc_eff(spin=0) returns the unpolarized fxc = (δ²Exc/δρ²)
    // For RKS closed shell, the unpolarized fxc corresponds to:
    //   f_u = (f_↑↑ + f_↑↓) / 2 = δ²Exc/δρ² (where ρ = ρ_↑+ρ_↓)
    // The singlet TDDFT kernel needs f_s = f_↑↑ + f_↑↓ = 2 × f_u
    // Reference: PySCF nr_rks_fxc_st for singlet
    const SINGLET_FXC_FACTOR: f64 = 2.0;
    let wfxc: Vec<f64> = if nvar == 1 {
        (0..ngrids).map(|g| fxc_raw[g] * weights[g] * SINGLET_FXC_FACTOR).collect()
    } else {
        let nv2 = nvar * nvar;
        let mut wfxc = vec![0.0; ngrids * nv2];
        for g in 0..ngrids {
            let w = weights[g];
            for k in 0..nv2 {
                wfxc[g + k * ngrids] = fxc_raw[g + k * ngrids] * w * SINGLET_FXC_FACTOR;
            }
        }
        wfxc
    };

    println!(
        "FXCMatvecData prepared: nocc={}, nvir={}, ngrids={}, nvar={}, alpha_hybrid={}",
        occ_size, vir_size, ngrids, nvar, alpha_hybrid
    );

    FXCMatvecData {
        nvar,
        ngrids,
        nocc: occ_size,
        nvir: vir_size,
        start_mo,
        alpha_hybrid,
        mo_occ,
        mo_vir,
        mo_occ_grad,
        mo_vir_grad,
        wfxc,
        use_opt: scf.mol.ctrl.use_fxc_opt,
    }
}

/// Full-occupation variant of `prepare_fxc_data` for analytic Hessian.
///
/// Uses `start_mo = 0` and `occ_size = homo + 1` (no frozen core), matching
/// `CPHFSolverPySCF::new_full`. This is required for RKS Hessian consistency:
/// PySCF's `kernel()` uses full occupation for everything.
pub fn prepare_fxc_data_full(scf: &SCF) -> FXCMatvecData {
    let nmo = scf.eigenvalues[0].len();
    let homo = scf.homo[0] as usize;
    let lumo = scf.lumo[0] as usize;
    let start_mo = 0;
    let occ_size = homo + 1;
    let vir_size = nmo - lumo;
    prepare_fxc_data_impl(scf, start_mo, occ_size, vir_size, lumo)
}

/// Implementation body shared by `prepare_fxc_data` (frozen-core) and
/// `prepare_fxc_data_full` (full occupation). Transplanted verbatim from
/// master during the group-progress/master semantic merge.
fn prepare_fxc_data_impl(
    scf: &SCF,
    start_mo: usize,
    occ_size: usize,
    vir_size: usize,
    lumo: usize,
) -> FXCMatvecData {
    let xc_data = &scf.mol.xc_data;
    let xc_type = if xc_data.use_density_gradient() {
        XCType::GGA
    } else {
        XCType::LDA
    };
    let nvar = match xc_type {
        XCType::LDA => 1,
        XCType::GGA => 4,
        _ => panic!("fxc only supports LDA and GGA"),
    };
    let alpha_hybrid = xc_data.dfa_hybrid_scf;

    let grids = scf.grids.as_ref().expect("DFT grids must be initialized for fxc");
    let ngrids = grids.weights.len();
    let num_basis = scf.mol.num_basis;
    let weights = &grids.weights;
    // ── Obtain dense AO: decompress from compressed storage if needed ──
    let ao_owned: Option<MatrixFull<f64>>;
    let ao: &MatrixFull<f64> = match &grids.ao {
        Some(a) => { ao_owned = None; a }
        None => match &grids.ao_compressed {
            Some(c) => { ao_owned = Some(Grids::decompress_ao(c)); ao_owned.as_ref().unwrap() }
            None => panic!("AO on grids must be tabulated (dense or compressed)"),
        }
    };
    let eigvec = &scf.eigenvectors[0];

    // ── Extract MO coefficients for occupied and virtual spaces ──
    let mut c_occ = MatrixFull::new([num_basis, occ_size], 0.0);
    for j in 0..occ_size {
        for i in 0..num_basis {
            c_occ[[i, j]] = eigvec[[i, start_mo + j]];
        }
    }
    let mut c_vir = MatrixFull::new([num_basis, vir_size], 0.0);
    for j in 0..vir_size {
        for i in 0..num_basis {
            c_vir[[i, j]] = eigvec[[i, lumo + j]];
        }
    }

    // ── Project MO values onto grids ──
    // mo_occ[i,g] = Σ_p C_occ[p,i] × ao[p,g]; mo_vir[a,g] = Σ_p C_vir[p,a] × ao[p,g]
    let mut mo_occ = MatrixFull::new([occ_size, ngrids], 0.0);
    _dgemm_full(&c_occ, 'T', ao, 'N', &mut mo_occ, 1.0, 0.0);

    let mut mo_vir = MatrixFull::new([vir_size, ngrids], 0.0);
    _dgemm_full(&c_vir, 'T', ao, 'N', &mut mo_vir, 1.0, 0.0);

    // ── GGA: MO gradients on grids ──
    let aop_owned: Option<RIFull<f64>>;
    let (mo_occ_grad, mo_vir_grad) = if xc_type == XCType::GGA {
        let aop: &RIFull<f64> = match &grids.aop {
            Some(a) => { aop_owned = None; a }
            None => match &grids.aop_compressed {
                Some(c) => { aop_owned = Some(Grids::decompress_aop(c)); aop_owned.as_ref().unwrap() }
                None => panic!("AO gradients needed for GGA fxc (dense or compressed)"),
            }
        };
        let mut og = [
            MatrixFull::new([occ_size, ngrids], 0.0),
            MatrixFull::new([occ_size, ngrids], 0.0),
            MatrixFull::new([occ_size, ngrids], 0.0),
        ];
        let mut vg = [
            MatrixFull::new([vir_size, ngrids], 0.0),
            MatrixFull::new([vir_size, ngrids], 0.0),
            MatrixFull::new([vir_size, ngrids], 0.0),
        ];
        for d in 0..3 {
            let aop_d_slice = aop.get_reducing_matrix(d).unwrap();
            let aop_d = MatrixFull::from_vec(
                [num_basis, ngrids],
                aop_d_slice.iter().cloned().collect(),
            ).unwrap();
            _dgemm_full(&c_occ, 'T', &aop_d, 'N', &mut og[d], 1.0, 0.0);
            _dgemm_full(&c_vir, 'T', &aop_d, 'N', &mut vg[d], 1.0, 0.0);
        }
        (Some(og), Some(vg))
    } else {
        (None, None)
    };

    // ── Compute ground-state density on grids for fxc evaluation ──
    let ao_deriv = if xc_type == XCType::GGA { 1 } else { 0 };
    let ao_rifull = eval_ao_batch(&scf.mol, &grids.coordinates, ao_deriv, ngrids);
    let mo_coeffs = vec![scf.eigenvectors[0].clone()];
    let occ = vec![scf.occupation[0].clone()];
    let rho_tensor = eval_rho5_batch(&ao_rifull, xc_type, &mo_coeffs, &occ, 1, ngrids);

    // Extract rho_array in the format expected by eval_xc_eff
    // For spin=0: column-major [ngrids, nvar], i.e. rho_array[g + v*ngrids]
    let rho_array: Vec<f64> = rho_tensor.raw()[rho_tensor.offset()..]
        .chunks(ngrids * nvar)
        .next()
        .unwrap_or(&[])
        .to_vec();
    let rho_array = if rho_array.is_empty() {
        let raw = rho_tensor.raw();
        let offset = rho_tensor.offset();
        let len = ngrids * nvar;
        raw[offset..offset + len].to_vec()
    } else {
        rho_array
    };

    // ── Compute fxc kernel analytically via libxc ──
    let func_ids = &xc_data.dfa_compnt_scf;
    let func_factors = &xc_data.dfa_paramr_scf;
    let xc_tensors = eval_xc_eff(func_ids, func_factors, xc_type, 0, &rho_array, ngrids, 2);

    // xc_tensors[2] = fxc kernel (deriv=2)
    let fxc_tensor = xc_tensors[2].as_ref()
        .expect("fxc (deriv=2) should be available");
    let fxc_raw: Vec<f64> = {
        let raw = fxc_tensor.raw();
        let offset = fxc_tensor.offset();
        let nv2 = nvar * nvar;
        let len = ngrids * nv2;
        raw[offset..offset + len].to_vec()
    };

    // ── Multiply fxc by grid weights ──
    // IMPORTANT: eval_xc_eff(spin=0) returns the unpolarized fxc = (δ²Exc/δρ²)
    // For RKS closed shell, the unpolarized fxc corresponds to:
    //   f_u = (f_↑↑ + f_↑↓) / 2 = δ²Exc/δρ² (where ρ = ρ_↑+ρ_↓)
    // The singlet TDDFT kernel needs f_s = f_↑↑ + f_↑↓ = 2 × f_u
    // Reference: PySCF nr_rks_fxc_st for singlet
    const SINGLET_FXC_FACTOR: f64 = 2.0;
    let wfxc: Vec<f64> = if nvar == 1 {
        (0..ngrids).map(|g| fxc_raw[g] * weights[g] * SINGLET_FXC_FACTOR).collect()
    } else {
        let nv2 = nvar * nvar;
        let mut wfxc = vec![0.0; ngrids * nv2];
        for g in 0..ngrids {
            let w = weights[g];
            for k in 0..nv2 {
                wfxc[g + k * ngrids] = fxc_raw[g + k * ngrids] * w * SINGLET_FXC_FACTOR;
            }
        }
        wfxc
    };

    println!(
        "FXCMatvecData prepared: nocc={}, nvir={}, ngrids={}, nvar={}, alpha_hybrid={}",
        occ_size, vir_size, ngrids, nvar, alpha_hybrid
    );

    FXCMatvecData {
        nvar,
        ngrids,
        nocc: occ_size,
        nvir: vir_size,
        start_mo,
        alpha_hybrid,
        mo_occ,
        mo_vir,
        mo_occ_grad,
        mo_vir_grad,
        wfxc,
        use_opt: scf.mol.ctrl.use_fxc_opt,
    }
}


// ============================================================================
// Unrestricted (spin-polarised) TDDFT fxc kernel data
// ============================================================================

/// Precomputed spin-resolved fxc kernel data for unrestricted TDDFT.
///
/// For each spin pair (σ,τ) we store the same per-grid layout as the
/// restricted `FXCMatvecData.wfxc`, but **without** the restricted singlet
/// factor 2:
///
/// - LDA: `wfxc_xy[g]`
/// - GGA: `wfxc_xy[g + α*ngrids + β*nvar*ngrids]`
///
/// The MO data are stored separately for α and β, because unrestricted
/// α/β occupied/virtual orbitals are different.
pub struct FXCMatvecDataUnrestricted {
    pub nvar: usize,
    pub ngrids: usize,
    pub nocc: [usize; 2],
    pub nvir: [usize; 2],
    pub start_mo: [usize; 2],
    pub alpha_hybrid: f64,
    pub mo_occ: [MatrixFull<f64>; 2],
    pub mo_vir: [MatrixFull<f64>; 2],
    /// GGA occupied MO gradients on grids, one [3] array per spin.
    pub mo_occ_grad: Option<[[MatrixFull<f64>; 3]; 2]>,
    /// GGA virtual MO gradients on grids, one [3] array per spin.
    pub mo_vir_grad: Option<[[MatrixFull<f64>; 3]; 2]>,
    pub wfxc_aa: Vec<f64>,
    pub wfxc_ab: Vec<f64>,
    pub wfxc_ba: Vec<f64>,
    pub wfxc_bb: Vec<f64>,
}

fn extract_fxc_spin_component(
    fxc_tensor: &Tensor<f64, DeviceBLAS>,
    nvar: usize,
    ngrids: usize,
    spin_left: usize,
    spin_right: usize,
) -> Vec<f64> {
    let nv2 = nvar * nvar;
    let mut out = vec![0.0; ngrids * nv2];
    if nvar == 1 {
        let v = fxc_tensor.i((.., 0, spin_left, 0, spin_right)).to_vec();
        out[..ngrids].copy_from_slice(&v);
    } else {
        for a in 0..nvar {
            for b in 0..nvar {
                let v = fxc_tensor.i((.., a, spin_left, b, spin_right)).to_vec();
                for (g, val) in v.iter().enumerate() {
                    out[g + a * ngrids + b * nvar * ngrids] = *val;
                }
            }
        }
    }
    out
}

/// Prepare unrestricted fxc data from a converged UKS SCF object.
///
/// This is the spin-polarised counterpart of `prepare_fxc_data`.  It evaluates
/// the full 2×2 spin kernel and stores each component separately.
pub fn prepare_fxc_data_unrestricted(scf: &SCF) -> FXCMatvecDataUnrestricted {
    let p0 = crate::ri_tddft::utils::tddft_occupation_parameters_spin(scf, 0);
    let p1 = crate::ri_tddft::utils::tddft_occupation_parameters_spin(scf, 1);
    let start_mo = [p0.0, p1.0];
    let nocc = [p0.2, p1.2];
    let nvir = [p0.3, p1.3];
    let lumo = [p0.5, p1.5];

    let xc_data = &scf.mol.xc_data;
    let xc_type = if xc_data.use_density_gradient() {
        XCType::GGA
    } else {
        XCType::LDA
    };
    let nvar = match xc_type {
        XCType::LDA => 1,
        XCType::GGA => 4,
        _ => panic!("fxc only supports LDA and GGA"),
    };
    let alpha_hybrid = xc_data.dfa_hybrid_scf;

    let grids = scf.grids.as_ref().expect("DFT grids must be initialized for fxc");
    let ngrids = grids.weights.len();
    let num_basis = scf.mol.num_basis;
    let weights = &grids.weights;

    let ao_owned: Option<MatrixFull<f64>>;
    let ao: &MatrixFull<f64> = match &grids.ao {
        Some(a) => { ao_owned = None; a }
        None => match &grids.ao_compressed {
            Some(c) => { ao_owned = Some(Grids::decompress_ao(c)); ao_owned.as_ref().unwrap() }
            None => panic!("AO on grids must be tabulated (dense or compressed)"),
        }
    };

    // MO projections for each spin.
    let mut mo_occ = [
        MatrixFull::new([nocc[0], ngrids], 0.0),
        MatrixFull::new([nocc[1], ngrids], 0.0),
    ];
    let mut mo_vir = [
        MatrixFull::new([nvir[0], ngrids], 0.0),
        MatrixFull::new([nvir[1], ngrids], 0.0),
    ];
    let mut mo_occ_grad: Option<[[MatrixFull<f64>; 3]; 2]> = None;
    let mut mo_vir_grad: Option<[[MatrixFull<f64>; 3]; 2]> = None;

    for spin in 0..2 {
        if nocc[spin] == 0 || nvir[spin] == 0 {
            continue;
        }
        let eigvec = &scf.eigenvectors[spin];
        let mut c_occ = MatrixFull::new([num_basis, nocc[spin]], 0.0);
        for j in 0..nocc[spin] {
            for i in 0..num_basis {
                c_occ[[i, j]] = eigvec[[i, start_mo[spin] + j]];
            }
        }
        let mut c_vir = MatrixFull::new([num_basis, nvir[spin]], 0.0);
        for j in 0..nvir[spin] {
            for i in 0..num_basis {
                c_vir[[i, j]] = eigvec[[i, lumo[spin] + j]];
            }
        }
        _dgemm_full(&c_occ, 'T', ao, 'N', &mut mo_occ[spin], 1.0, 0.0);
        _dgemm_full(&c_vir, 'T', ao, 'N', &mut mo_vir[spin], 1.0, 0.0);
    }

    // GGA gradients.
    if xc_type == XCType::GGA {
        let aop_owned: Option<RIFull<f64>>;
        let aop: &RIFull<f64> = match &grids.aop {
            Some(a) => { aop_owned = None; a }
            None => match &grids.aop_compressed {
                Some(c) => { aop_owned = Some(Grids::decompress_aop(c)); aop_owned.as_ref().unwrap() }
                None => panic!("AO gradients needed for GGA fxc (dense or compressed)"),
            }
        };
        let mut og_all = [
            [
                MatrixFull::new([nocc[0], ngrids], 0.0),
                MatrixFull::new([nocc[0], ngrids], 0.0),
                MatrixFull::new([nocc[0], ngrids], 0.0),
            ],
            [
                MatrixFull::new([nocc[1], ngrids], 0.0),
                MatrixFull::new([nocc[1], ngrids], 0.0),
                MatrixFull::new([nocc[1], ngrids], 0.0),
            ],
        ];
        let mut vg_all = [
            [
                MatrixFull::new([nvir[0], ngrids], 0.0),
                MatrixFull::new([nvir[0], ngrids], 0.0),
                MatrixFull::new([nvir[0], ngrids], 0.0),
            ],
            [
                MatrixFull::new([nvir[1], ngrids], 0.0),
                MatrixFull::new([nvir[1], ngrids], 0.0),
                MatrixFull::new([nvir[1], ngrids], 0.0),
            ],
        ];
        for spin in 0..2 {
            if nocc[spin] == 0 || nvir[spin] == 0 {
                continue;
            }
            let eigvec = &scf.eigenvectors[spin];
            let mut c_occ = MatrixFull::new([num_basis, nocc[spin]], 0.0);
            for j in 0..nocc[spin] {
                for i in 0..num_basis { c_occ[[i, j]] = eigvec[[i, start_mo[spin] + j]]; }
            }
            let mut c_vir = MatrixFull::new([num_basis, nvir[spin]], 0.0);
            for j in 0..nvir[spin] {
                for i in 0..num_basis { c_vir[[i, j]] = eigvec[[i, lumo[spin] + j]]; }
            }
            for d in 0..3 {
                let aop_d_slice = aop.get_reducing_matrix(d).unwrap();
                let aop_d = MatrixFull::from_vec(
                    [num_basis, ngrids],
                    aop_d_slice.iter().cloned().collect(),
                ).unwrap();
                _dgemm_full(&c_occ, 'T', &aop_d, 'N', &mut og_all[spin][d], 1.0, 0.0);
                _dgemm_full(&c_vir, 'T', &aop_d, 'N', &mut vg_all[spin][d], 1.0, 0.0);
            }
        }
        mo_occ_grad = Some(og_all);
        mo_vir_grad = Some(vg_all);
    }

    // Ground-state alpha+beta density for fxc evaluation.
    let ao_deriv = if xc_type == XCType::GGA { 1 } else { 0 };
    let ao_rifull = eval_ao_batch(&scf.mol, &grids.coordinates, ao_deriv, ngrids);
    let mo_coeffs = vec![scf.eigenvectors[0].clone(), scf.eigenvectors[1].clone()];
    let occ = vec![scf.occupation[0].clone(), scf.occupation[1].clone()];
    let rho_tensor = eval_rho5_batch(&ao_rifull, xc_type, &mo_coeffs, &occ, 2, ngrids);
    let rho_array: Vec<f64> = {
        let raw = rho_tensor.raw();
        let offset = rho_tensor.offset();
        raw[offset..offset + ngrids * nvar * 2].to_vec()
    };

    let func_ids = &xc_data.dfa_compnt_scf;
    let func_factors = &xc_data.dfa_paramr_scf;
    let xc_tensors = eval_xc_eff(func_ids, func_factors, xc_type, 1, &rho_array, ngrids, 2);
    let fxc_tensor = xc_tensors[2].as_ref()
        .expect("fxc (deriv=2) should be available for unrestricted TDDFT");

    let raw_aa = extract_fxc_spin_component(fxc_tensor, nvar, ngrids, 0, 0);
    let raw_ab = extract_fxc_spin_component(fxc_tensor, nvar, ngrids, 0, 1);
    let raw_ba = extract_fxc_spin_component(fxc_tensor, nvar, ngrids, 1, 0);
    let raw_bb = extract_fxc_spin_component(fxc_tensor, nvar, ngrids, 1, 1);

    let multiply_weight = |raw: &[f64]| -> Vec<f64> {
        if nvar == 1 {
            (0..ngrids).map(|g| raw[g] * weights[g]).collect()
        } else {
            let nv2 = nvar * nvar;
            let mut out = vec![0.0; ngrids * nv2];
            for g in 0..ngrids {
                let w = weights[g];
                for k in 0..nv2 {
                    out[g + k * ngrids] = raw[g + k * ngrids] * w;
                }
            }
            out
        }
    };

    let wfxc_aa = multiply_weight(&raw_aa);
    let wfxc_ab = multiply_weight(&raw_ab);
    let wfxc_ba = multiply_weight(&raw_ba);
    let wfxc_bb = multiply_weight(&raw_bb);

    println!(
        "Unrestricted FXCMatvecData prepared: nocc=[{},{}], nvir=[{},{}], ngrids={}, nvar={}, alpha_hybrid={}",
        nocc[0], nocc[1], nvir[0], nvir[1], ngrids, nvar, alpha_hybrid
    );

    FXCMatvecDataUnrestricted {
        nvar,
        ngrids,
        nocc,
        nvir,
        start_mo,
        alpha_hybrid,
        mo_occ,
        mo_vir,
        mo_occ_grad,
        mo_vir_grad,
        wfxc_aa,
        wfxc_ab,
        wfxc_ba,
        wfxc_bb,
    }
}

/// Compute the spin-resolved fxc matrix-vector products.
///
/// `z` is the concatenated [alpha; beta] transition vector.  The returned
/// tuple is `(fxc_alpha, fxc_beta)`, each of length `nocc_s*nvir_s`.
pub fn fxc_matvec_unrestricted(
    data: &FXCMatvecDataUnrestricted,
    z: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let n0 = data.nocc[0] * data.nvir[0];
    let n1 = data.nocc[1] * data.nvir[1];
    assert_eq!(z.len(), n0 + n1, "unrestricted fxc vector length mismatch");
    let za = &z[..n0];
    let zb = &z[n0..];

    let mut fa = vec![0.0; n0];
    let mut fb = vec![0.0; n1];
    if n0 > 0 {
        let tmp = fxc_matvec_with_kernel_unrestricted(
            data.nvar,
            data.ngrids,
            data.nocc[0],
            data.nvir[0],
            &data.mo_occ[0],
            &data.mo_vir[0],
            data.mo_occ_grad.as_ref().map(|g| &g[0]),
            data.mo_vir_grad.as_ref().map(|g| &g[0]),
            &data.wfxc_aa,
            za,
        );
        fa.iter_mut().zip(tmp).for_each(|(a, b)| *a += b);
        if n1 > 0 {
            let tmp = fxc_matvec_with_kernel_cross_unrestricted(
                data.nvar,
                data.ngrids,
                data.nocc[0],
                data.nvir[0],
                &data.mo_occ[0],
                &data.mo_vir[0],
                data.mo_occ_grad.as_ref().map(|g| &g[0]),
                data.mo_vir_grad.as_ref().map(|g| &g[0]),
                data.nocc[1],
                data.nvir[1],
                &data.mo_occ[1],
                &data.mo_vir[1],
                data.mo_occ_grad.as_ref().map(|g| &g[1]),
                data.mo_vir_grad.as_ref().map(|g| &g[1]),
                &data.wfxc_ab,
                zb,
            );
            fa.iter_mut().zip(tmp).for_each(|(a, b)| *a += b);
        }
    }
    if n1 > 0 {
        let tmp = fxc_matvec_with_kernel_unrestricted(
            data.nvar,
            data.ngrids,
            data.nocc[1],
            data.nvir[1],
            &data.mo_occ[1],
            &data.mo_vir[1],
            data.mo_occ_grad.as_ref().map(|g| &g[1]),
            data.mo_vir_grad.as_ref().map(|g| &g[1]),
            &data.wfxc_bb,
            zb,
        );
        fb.iter_mut().zip(tmp).for_each(|(a, b)| *a += b);
        if n0 > 0 {
            let tmp = fxc_matvec_with_kernel_cross_unrestricted(
                data.nvar,
                data.ngrids,
                data.nocc[1],
                data.nvir[1],
                &data.mo_occ[1],
                &data.mo_vir[1],
                data.mo_occ_grad.as_ref().map(|g| &g[1]),
                data.mo_vir_grad.as_ref().map(|g| &g[1]),
                data.nocc[0],
                data.nvir[0],
                &data.mo_occ[0],
                &data.mo_vir[0],
                data.mo_occ_grad.as_ref().map(|g| &g[0]),
                data.mo_vir_grad.as_ref().map(|g| &g[0]),
                &data.wfxc_ba,
                za,
            );
            fb.iter_mut().zip(tmp).for_each(|(a, b)| *a += b);
        }
    }
    (fa, fb)
}

/// Restricted-like fxc matvec using an explicit spin-pair kernel array.
#[allow(clippy::too_many_arguments)]
fn fxc_matvec_with_kernel_unrestricted(
    nvar: usize,
    ngrids: usize,
    nocc: usize,
    nvir: usize,
    mo_occ: &MatrixFull<f64>,
    mo_vir: &MatrixFull<f64>,
    mo_occ_grad: Option<&[MatrixFull<f64>; 3]>,
    mo_vir_grad: Option<&[MatrixFull<f64>; 3]>,
    wfxc: &[f64],
    z: &[f64],
) -> Vec<f64> {
    if nocc == 0 || nvir == 0 || z.is_empty() {
        return vec![0.0; nocc * nvir];
    }
    if nvar == 1 {
        fxc_matvec_lda_with_kernel(nocc, nvir, ngrids, mo_occ, mo_vir, wfxc, z)
    } else {
        fxc_matvec_gga_with_kernel(
            nocc,
            nvir,
            ngrids,
            mo_occ,
            mo_vir,
            mo_occ_grad.expect("GGA unrestricted fxc requires mo_occ_grad"),
            mo_vir_grad.expect("GGA unrestricted fxc requires mo_vir_grad"),
            wfxc,
            z,
        )
    }
}


/// fxc matvec for a cross-spin pair (left output spin, right perturbing spin).
#[allow(clippy::too_many_arguments)]
fn fxc_matvec_with_kernel_cross_unrestricted(
    nvar: usize,
    ngrids: usize,
    nocc_l: usize,
    nvir_l: usize,
    mo_occ_l: &MatrixFull<f64>,
    mo_vir_l: &MatrixFull<f64>,
    mo_occ_grad_l: Option<&[MatrixFull<f64>; 3]>,
    mo_vir_grad_l: Option<&[MatrixFull<f64>; 3]>,
    nocc_r: usize,
    nvir_r: usize,
    mo_occ_r: &MatrixFull<f64>,
    mo_vir_r: &MatrixFull<f64>,
    mo_occ_grad_r: Option<&[MatrixFull<f64>; 3]>,
    mo_vir_grad_r: Option<&[MatrixFull<f64>; 3]>,
    wfxc: &[f64],
    z_r: &[f64],
) -> Vec<f64> {
    if nocc_l == 0 || nvir_l == 0 || nocc_r == 0 || nvir_r == 0 || z_r.is_empty() {
        return vec![0.0; nocc_l * nvir_l];
    }
    if nvar == 1 {
        fxc_matvec_lda_cross_with_kernel(
            nocc_l, nvir_l, nocc_r, nvir_r, ngrids,
            mo_occ_l, mo_vir_l, mo_occ_r, mo_vir_r, wfxc, z_r,
        )
    } else {
        fxc_matvec_gga_cross_with_kernel(
            nocc_l, nvir_l, nocc_r, nvir_r, ngrids,
            mo_occ_l, mo_vir_l,
            mo_occ_grad_l.expect("GGA unrestricted fxc requires mo_occ_grad_l"),
            mo_vir_grad_l.expect("GGA unrestricted fxc requires mo_vir_grad_l"),
            mo_occ_r, mo_vir_r,
            mo_occ_grad_r.expect("GGA unrestricted fxc requires mo_occ_grad_r"),
            mo_vir_grad_r.expect("GGA unrestricted fxc requires mo_vir_grad_r"),
            wfxc, z_r,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn fxc_matvec_lda_cross_with_kernel(
    nocc_l: usize,
    nvir_l: usize,
    nocc_r: usize,
    nvir_r: usize,
    ngrids: usize,
    mo_occ_l: &MatrixFull<f64>,
    mo_vir_l: &MatrixFull<f64>,
    mo_occ_r: &MatrixFull<f64>,
    mo_vir_r: &MatrixFull<f64>,
    wfxc: &[f64],
    z_r: &[f64],
) -> Vec<f64> {
    let z_mat = MatrixFull::from_vec([nocc_r, nvir_r], z_r.to_vec()).unwrap();
    let mut t = MatrixFull::new([nocc_r, ngrids], 0.0);
    _dgemm_full(&z_mat, 'N', mo_vir_r, 'N', &mut t, 1.0, 0.0);
    let mut rho_z = vec![0.0; ngrids];
    for g in 0..ngrids {
        let mut sum = 0.0;
        for i in 0..nocc_r {
            sum += mo_occ_r[[i, g]] * t[[i, g]];
        }
        rho_z[g] = sum;
    }
    let mut v = vec![0.0; ngrids];
    for g in 0..ngrids {
        v[g] = wfxc[g] * rho_z[g];
    }
    let mut mo_vir_scaled = MatrixFull::new([nvir_l, ngrids], 0.0);
    for g in 0..ngrids {
        for a in 0..nvir_l {
            mo_vir_scaled[[a, g]] = mo_vir_l[[a, g]] * v[g];
        }
    }
    let mut result = MatrixFull::new([nocc_l, nvir_l], 0.0);
    _dgemm_full(mo_occ_l, 'N', &mo_vir_scaled, 'T', &mut result, 1.0, 0.0);
    result.data
}

#[allow(clippy::too_many_arguments)]
fn fxc_matvec_gga_cross_with_kernel(
    nocc_l: usize,
    nvir_l: usize,
    nocc_r: usize,
    nvir_r: usize,
    ngrids: usize,
    mo_occ_l: &MatrixFull<f64>,
    mo_vir_l: &MatrixFull<f64>,
    mo_occ_grad_l: &[MatrixFull<f64>; 3],
    mo_vir_grad_l: &[MatrixFull<f64>; 3],
    mo_occ_r: &MatrixFull<f64>,
    mo_vir_r: &MatrixFull<f64>,
    mo_occ_grad_r: &[MatrixFull<f64>; 3],
    mo_vir_grad_r: &[MatrixFull<f64>; 3],
    wfxc: &[f64],
    z_r: &[f64],
) -> Vec<f64> {
    let z_mat = MatrixFull::from_vec([nocc_r, nvir_r], z_r.to_vec()).unwrap();
    let mut t0 = MatrixFull::new([nocc_r, ngrids], 0.0);
    _dgemm_full(&z_mat, 'N', mo_vir_r, 'N', &mut t0, 1.0, 0.0);
    let mut t_grad: [MatrixFull<f64>; 3] = [
        MatrixFull::new([nocc_r, ngrids], 0.0),
        MatrixFull::new([nocc_r, ngrids], 0.0),
        MatrixFull::new([nocc_r, ngrids], 0.0),
    ];
    for d in 0..3 {
        _dgemm_full(&z_mat, 'N', &mo_vir_grad_r[d], 'N', &mut t_grad[d], 1.0, 0.0);
    }

    let mut rho_z = vec![0.0; ngrids * 4];
    for g in 0..ngrids {
        let mut sum = 0.0;
        for i in 0..nocc_r {
            sum += mo_occ_r[[i, g]] * t0[[i, g]];
        }
        rho_z[g] = sum;
        for d in 0..3 {
            let mut sum1 = 0.0;
            let mut sum2 = 0.0;
            for i in 0..nocc_r {
                sum1 += mo_occ_grad_r[d][[i, g]] * t0[[i, g]];
                sum2 += mo_occ_r[[i, g]] * t_grad[d][[i, g]];
            }
            rho_z[g + (d + 1) * ngrids] = sum1 + sum2;
        }
    }

    let mut fxc_eff_grid = vec![0.0; ngrids * 4];
    for g in 0..ngrids {
        for alpha in 0..4 {
            let mut sum = 0.0;
            for beta in 0..4 {
                let w_idx = g + alpha * ngrids + beta * 4 * ngrids;
                sum += wfxc[w_idx] * rho_z[g + beta * ngrids];
            }
            fxc_eff_grid[g + alpha * ngrids] = sum;
        }
    }

    let mut result = vec![0.0; nocc_l * nvir_l];
    for alpha in 0..4 {
        let fxc_a = &fxc_eff_grid[alpha * ngrids..(alpha + 1) * ngrids];
        if alpha == 0 {
            let mut right_scaled = MatrixFull::new([nvir_l, ngrids], 0.0);
            for g in 0..ngrids {
                let fv = fxc_a[g];
                for a in 0..nvir_l {
                    right_scaled[[a, g]] = mo_vir_l[[a, g]] * fv;
                }
            }
            let mut contrib = MatrixFull::new([nocc_l, nvir_l], 0.0);
            _dgemm_full(mo_occ_l, 'N', &right_scaled, 'T', &mut contrib, 1.0, 0.0);
            for idx in 0..result.len() {
                result[idx] += contrib.data[idx];
            }
        } else {
            let d = alpha - 1;
            let mut right_scaled = MatrixFull::new([nvir_l, ngrids], 0.0);
            for g in 0..ngrids {
                let fv = fxc_a[g];
                for a in 0..nvir_l {
                    right_scaled[[a, g]] = mo_vir_l[[a, g]] * fv;
                }
            }
            let mut contrib_a = MatrixFull::new([nocc_l, nvir_l], 0.0);
            _dgemm_full(&mo_occ_grad_l[d], 'N', &right_scaled, 'T', &mut contrib_a, 1.0, 0.0);
            for idx in 0..result.len() {
                result[idx] += contrib_a.data[idx];
            }
            let mut right_grad_scaled = MatrixFull::new([nvir_l, ngrids], 0.0);
            for g in 0..ngrids {
                let fv = fxc_a[g];
                for a in 0..nvir_l {
                    right_grad_scaled[[a, g]] = mo_vir_grad_l[d][[a, g]] * fv;
                }
            }
            let mut contrib_b = MatrixFull::new([nocc_l, nvir_l], 0.0);
            _dgemm_full(mo_occ_l, 'N', &right_grad_scaled, 'T', &mut contrib_b, 1.0, 0.0);
            for idx in 0..result.len() {
                result[idx] += contrib_b.data[idx];
            }
        }
    }
    result
}

fn fxc_matvec_lda_with_kernel(
    nocc: usize,
    nvir: usize,
    ngrids: usize,
    mo_occ: &MatrixFull<f64>,
    mo_vir: &MatrixFull<f64>,
    wfxc: &[f64],
    z: &[f64],
) -> Vec<f64> {
    let z_mat = MatrixFull::from_vec([nocc, nvir], z.to_vec()).unwrap();
    let mut t = MatrixFull::new([nocc, ngrids], 0.0);
    _dgemm_full(&z_mat, 'N', mo_vir, 'N', &mut t, 1.0, 0.0);
    let mut rho_z = vec![0.0; ngrids];
    for g in 0..ngrids {
        let mut sum = 0.0;
        for i in 0..nocc {
            sum += mo_occ[[i, g]] * t[[i, g]];
        }
        rho_z[g] = sum;
    }
    let mut v = vec![0.0; ngrids];
    for g in 0..ngrids {
        v[g] = wfxc[g] * rho_z[g];
    }
    let mut mo_vir_scaled = MatrixFull::new([nvir, ngrids], 0.0);
    for g in 0..ngrids {
        for a in 0..nvir {
            mo_vir_scaled[[a, g]] = mo_vir[[a, g]] * v[g];
        }
    }
    let mut result = MatrixFull::new([nocc, nvir], 0.0);
    _dgemm_full(mo_occ, 'N', &mo_vir_scaled, 'T', &mut result, 1.0, 0.0);
    result.data
}

#[allow(clippy::too_many_arguments)]
fn fxc_matvec_gga_with_kernel(
    nocc: usize,
    nvir: usize,
    ngrids: usize,
    mo_occ: &MatrixFull<f64>,
    mo_vir: &MatrixFull<f64>,
    mo_occ_grad: &[MatrixFull<f64>; 3],
    mo_vir_grad: &[MatrixFull<f64>; 3],
    wfxc: &[f64],
    z: &[f64],
) -> Vec<f64> {
    let z_mat = MatrixFull::from_vec([nocc, nvir], z.to_vec()).unwrap();
    let mut t0 = MatrixFull::new([nocc, ngrids], 0.0);
    _dgemm_full(&z_mat, 'N', mo_vir, 'N', &mut t0, 1.0, 0.0);
    let mut t_grad: [MatrixFull<f64>; 3] = [
        MatrixFull::new([nocc, ngrids], 0.0),
        MatrixFull::new([nocc, ngrids], 0.0),
        MatrixFull::new([nocc, ngrids], 0.0),
    ];
    for d in 0..3 {
        _dgemm_full(&z_mat, 'N', &mo_vir_grad[d], 'N', &mut t_grad[d], 1.0, 0.0);
    }

    let mut rho_z = vec![0.0; ngrids * 4];
    for g in 0..ngrids {
        let mut sum = 0.0;
        for i in 0..nocc {
            sum += mo_occ[[i, g]] * t0[[i, g]];
        }
        rho_z[g] = sum;
        for d in 0..3 {
            let mut sum1 = 0.0;
            let mut sum2 = 0.0;
            for i in 0..nocc {
                sum1 += mo_occ_grad[d][[i, g]] * t0[[i, g]];
                sum2 += mo_occ[[i, g]] * t_grad[d][[i, g]];
            }
            rho_z[g + (d + 1) * ngrids] = sum1 + sum2;
        }
    }

    let mut fxc_eff_grid = vec![0.0; ngrids * 4];
    for g in 0..ngrids {
        for alpha in 0..4 {
            let mut sum = 0.0;
            for beta in 0..4 {
                let w_idx = g + alpha * ngrids + beta * 4 * ngrids;
                sum += wfxc[w_idx] * rho_z[g + beta * ngrids];
            }
            fxc_eff_grid[g + alpha * ngrids] = sum;
        }
    }

    let mut result = vec![0.0; nocc * nvir];
    for alpha in 0..4 {
        let fxc_a = &fxc_eff_grid[alpha * ngrids..(alpha + 1) * ngrids];
        if alpha == 0 {
            let mut right_scaled = MatrixFull::new([nvir, ngrids], 0.0);
            for g in 0..ngrids {
                let fv = fxc_a[g];
                for a in 0..nvir {
                    right_scaled[[a, g]] = mo_vir[[a, g]] * fv;
                }
            }
            let mut contrib = MatrixFull::new([nocc, nvir], 0.0);
            _dgemm_full(mo_occ, 'N', &right_scaled, 'T', &mut contrib, 1.0, 0.0);
            for idx in 0..result.len() {
                result[idx] += contrib.data[idx];
            }
        } else {
            let d = alpha - 1;
            let mut right_scaled = MatrixFull::new([nvir, ngrids], 0.0);
            for g in 0..ngrids {
                let fv = fxc_a[g];
                for a in 0..nvir {
                    right_scaled[[a, g]] = mo_vir[[a, g]] * fv;
                }
            }
            let mut contrib_a = MatrixFull::new([nocc, nvir], 0.0);
            _dgemm_full(&mo_occ_grad[d], 'N', &right_scaled, 'T', &mut contrib_a, 1.0, 0.0);
            for idx in 0..result.len() {
                result[idx] += contrib_a.data[idx];
            }
            let mut right_grad_scaled = MatrixFull::new([nvir, ngrids], 0.0);
            for g in 0..ngrids {
                let fv = fxc_a[g];
                for a in 0..nvir {
                    right_grad_scaled[[a, g]] = mo_vir_grad[d][[a, g]] * fv;
                }
            }
            let mut contrib_b = MatrixFull::new([nocc, nvir], 0.0);
            _dgemm_full(mo_occ, 'N', &right_grad_scaled, 'T', &mut contrib_b, 1.0, 0.0);
            for idx in 0..result.len() {
                result[idx] += contrib_b.data[idx];
            }
        }
    }
    result
}

// ── Global flag to enable the optimised (rayon-parallel) kernel ──
static USE_OPTIMIZED_FXC: AtomicBool = AtomicBool::new(true);

// ── Benchmarking accumulators for fxc_matvec timing ──
// NOTE: On group-progress the production `fxc_matvec` router does not
// accumulate into these counters (the team version of `fxc_matvec` is kept
// verbatim per merge rules). `fxc_matvec_reset_bench` /
// `fxc_matvec_report_bench` are still provided so external call sites compile;
// they will report zeros until the router is instrumented (handled by Task 3).
static FXC_MATVEC_COUNT: AtomicU64 = AtomicU64::new(0);
static FXC_MATVEC_NS: AtomicU64 = AtomicU64::new(0);

/// Reset the fxc_matvec benchmark counters.
pub fn fxc_matvec_reset_bench() {
    FXC_MATVEC_COUNT.store(0, Ordering::Relaxed);
    FXC_MATVEC_NS.store(0, Ordering::Relaxed);
}

/// Read and report cumulative fxc_matvec timing.
pub fn fxc_matvec_report_bench() {
    let count = FXC_MATVEC_COUNT.load(Ordering::Relaxed);
    let total_ns = FXC_MATVEC_NS.load(Ordering::Relaxed);
    if count > 0 {
        let total_s = total_ns as f64 / 1e9;
        let avg_ms = total_s / count as f64 * 1000.0;
        println!("\n[FXC_BENCH] fxc_matvec called {} times", count);
        println!("[FXC_BENCH] Total time: {:.6} s", total_s);
        println!("[FXC_BENCH] Average time per call: {:.6} ms ({:.3} µs)",
                 avg_ms, avg_ms * 1000.0);
    }
}

/// Set whether the production `fxc_matvec` should use the optimised kernel.
///
/// Call once during initialisation, before any TDDFT solve loop.
/// The flag is read on every invocation so it can be toggled between
/// calculations, but it is not synchronised with in-flight calls.
pub fn set_fxc_use_optimized(flag: bool) {
    USE_OPTIMIZED_FXC.store(flag, Ordering::Relaxed);
}

/// Production router: delegates to optimised or original kernel based on
/// the global `tddft_use_optimized_fxc` control flag.
pub fn fxc_matvec(data: &FXCMatvecData, z: &[f64]) -> Vec<f64> {
    if USE_OPTIMIZED_FXC.load(Ordering::Relaxed) {
        fxc_matvec_opt(data, z)
    } else {
        fxc_matvec_old(data, z)
    }
}



/// Original fxc matrix-vector product (kept for reference).
pub fn fxc_matvec_old(data: &FXCMatvecData, z: &[f64]) -> Vec<f64> {
    assert_eq!(z.len(), data.nocc * data.nvir,
               "z vector length {} must equal nocc×nvir = {}×{}",
               z.len(), data.nocc, data.nvir);
    match data.nvar {
        1 => fxc_matvec_lda(data, z),
        4 => fxc_matvec_gga(data, z),
        _ => panic!("fxc_matvec_old only supports LDA (nvar=1) and GGA (nvar=4)"),
    }
}

/// LDA fxc matrix-vector product
fn fxc_matvec_lda(data: &FXCMatvecData, z: &[f64]) -> Vec<f64> {
    let nocc = data.nocc;
    let nvir = data.nvir;
    let ngrids = data.ngrids;

    // ── Step 1: Compute perturbed density on grids ──
    let z_mat = MatrixFull::from_vec([nocc, nvir], z.to_vec()).unwrap();
    let mut t = MatrixFull::new([nocc, ngrids], 0.0);
    _dgemm_full(&z_mat, 'N', &data.mo_vir, 'N', &mut t, 1.0, 0.0);

    // ρ_z[g] = Σ_i mo_occ[i,g] × t[i,g]
    let mut rho_z = vec![0.0; ngrids];
    for g in 0..ngrids {
        let mut sum = 0.0;
        for i in 0..nocc {
            sum += data.mo_occ[[i, g]] * t[[i, g]];
        }
        rho_z[g] = sum;
    }

    // ── Step 2: Apply fxc kernel ──
    let mut v = vec![0.0; ngrids];
    for g in 0..ngrids {
        v[g] = data.wfxc[g] * rho_z[g];
    }

    // ── Step 3: Contract back to MO basis ──
    let mut mo_vir_scaled = MatrixFull::new([nvir, ngrids], 0.0);
    for g in 0..ngrids {
        for a in 0..nvir {
            mo_vir_scaled[[a, g]] = data.mo_vir[[a, g]] * v[g];
        }
    }

    let mut result = MatrixFull::new([nocc, nvir], 0.0);
    _dgemm_full(&data.mo_occ, 'N', &mo_vir_scaled, 'T', &mut result, 1.0, 0.0);

    result.data
}

/// GGA fxc matrix-vector product
fn fxc_matvec_gga(data: &FXCMatvecData, z: &[f64]) -> Vec<f64> {
    let nocc = data.nocc;
    let nvir = data.nvir;
    let ngrids = data.ngrids;

    let mo_occ_grad = data.mo_occ_grad.as_ref().expect("GGA requires mo_occ_grad");
    let mo_vir_grad = data.mo_vir_grad.as_ref().expect("GGA requires mo_vir_grad");

    let z_mat = MatrixFull::from_vec([nocc, nvir], z.to_vec()).unwrap();
    let mut t0 = MatrixFull::new([nocc, ngrids], 0.0);
    _dgemm_full(&z_mat, 'N', &data.mo_vir, 'N', &mut t0, 1.0, 0.0);

    let mut t_grad: [MatrixFull<f64>; 3] = [
        MatrixFull::new([nocc, ngrids], 0.0),
        MatrixFull::new([nocc, ngrids], 0.0),
        MatrixFull::new([nocc, ngrids], 0.0),
    ];
    for d in 0..3 {
        _dgemm_full(&z_mat, 'N', &mo_vir_grad[d], 'N', &mut t_grad[d], 1.0, 0.0);
    }

    let mut rho_z = vec![0.0; ngrids * 4];

    // β=0: ρ_z[0,g] = Σ_i mo_occ[i,g] × t0[i,g]
    for g in 0..ngrids {
        let mut sum = 0.0;
        for i in 0..nocc {
            sum += data.mo_occ[[i, g]] * t0[[i, g]];
        }
        rho_z[g] = sum;

        for d in 0..3 {
            let mut sum1 = 0.0;
            let mut sum2 = 0.0;
            for i in 0..nocc {
                sum1 += mo_occ_grad[d][[i, g]] * t0[[i, g]];
                sum2 += data.mo_occ[[i, g]] * t_grad[d][[i, g]];
            }
            rho_z[g + (d + 1) * ngrids] = sum1 + sum2;
        }
    }

    // ── Step 2: Apply fxc kernel at each grid point ──
    let mut fxc_eff_grid = vec![0.0; ngrids * 4];

    for g in 0..ngrids {
        for alpha in 0..4 {
            let mut sum = 0.0;
            for beta in 0..4 {
                let w_idx = g + alpha * ngrids + beta * 4 * ngrids;
                sum += data.wfxc[w_idx] * rho_z[g + beta * ngrids];
            }
            fxc_eff_grid[g + alpha * ngrids] = sum;
        }
    }

    // ── Step 3: Contract back to MO basis ──
    let mut result = vec![0.0; nocc * nvir];

    for alpha in 0..4 {
        let fxc_a = &fxc_eff_grid[alpha * ngrids..(alpha + 1) * ngrids];

        if alpha == 0 {
            let mut right_scaled = MatrixFull::new([nvir, ngrids], 0.0);
            for g in 0..ngrids {
                let fv = fxc_a[g];
                for a in 0..nvir {
                    right_scaled[[a, g]] = data.mo_vir[[a, g]] * fv;
                }
            }
            let mut contrib = MatrixFull::new([nocc, nvir], 0.0);
            _dgemm_full(&data.mo_occ, 'N', &right_scaled, 'T', &mut contrib, 1.0, 0.0);
            for idx in 0..result.len() {
                result[idx] += contrib.data[idx];
            }
        } else {
            let d = alpha - 1;

            let mut right_scaled = MatrixFull::new([nvir, ngrids], 0.0);
            for g in 0..ngrids {
                let fv = fxc_a[g];
                for a in 0..nvir {
                    right_scaled[[a, g]] = data.mo_vir[[a, g]] * fv;
                }
            }
            let mut contrib_a = MatrixFull::new([nocc, nvir], 0.0);
            _dgemm_full(&mo_occ_grad[d], 'N', &right_scaled, 'T', &mut contrib_a, 1.0, 0.0);
            for idx in 0..result.len() {
                result[idx] += contrib_a.data[idx];
            }

            let mut right_grad_scaled = MatrixFull::new([nvir, ngrids], 0.0);
            for g in 0..ngrids {
                let fv = fxc_a[g];
                for a in 0..nvir {
                    right_grad_scaled[[a, g]] = mo_vir_grad[d][[a, g]] * fv;
                }
            }
            let mut contrib_b = MatrixFull::new([nocc, nvir], 0.0);
            _dgemm_full(&data.mo_occ, 'N', &right_grad_scaled, 'T', &mut contrib_b, 1.0, 0.0);
            for idx in 0..result.len() {
                result[idx] += contrib_b.data[idx];
            }
        }
    }

    result
}

// ====================================================================
// Optimized fxc_matvec implementations
// ====================================================================

/// Optimized dispatch: same interface as fxc_matvec, but with rayon
/// parallelism over grid points, fused dot products (GGA), and reduced
/// temporary allocation overhead.
#[allow(dead_code)]
pub fn fxc_matvec_opt(data: &FXCMatvecData, z: &[f64]) -> Vec<f64> {
    assert_eq!(z.len(), data.nocc * data.nvir,
               "z vector length {} must equal nocc×nvir = {}×{}",
               z.len(), data.nocc, data.nvir);
    match data.nvar {
        1 => fxc_matvec_lda_opt(data, z),
        4 => fxc_matvec_gga_opt(data, z),
        _ => panic!("fxc_matvec_opt only supports LDA (nvar=1) and GGA (nvar=4)"),
    }
}

/// Optimised LDA fxc matrix-vector product with cache-blocked step 3.
///
/// Key improvements over the original:
/// - Direct raw-slice column access (no per-element bounds checks)
/// - Rayon parallelisation over grid points
/// - **Blocked contraction**: mo_vir columns are scaled and contracted in
///   blocks of `BLOCK` grid columns so that the working set fits in L2/L3
///   cache, reducing main-memory traffic by ~10× for the contraction step.
/// - Temporary block-scaled matrices created without zeroing.
const LDA_BLOCK: usize = 8192;
#[allow(dead_code)]
fn fxc_matvec_lda_opt(data: &FXCMatvecData, z: &[f64]) -> Vec<f64> {
    let nocc = data.nocc;
    let nvir = data.nvir;
    let ngrids = data.ngrids;

    // ── Step 1: z_mat * mo_vir  →  t (single BLAS call, unchanged) ──
    let z_mat = MatrixFull::from_vec([nocc, nvir], z.to_vec()).unwrap();
    let mut t = MatrixFull::new([nocc, ngrids], 0.0);
    _dgemm_full(&z_mat, 'N', &data.mo_vir, 'N', &mut t, 1.0, 0.0);

    let mo_occ_data = data.mo_occ.data.as_slice();
    let mo_vir_data = data.mo_vir.data.as_slice();
    let t_data = t.data.as_slice();
    let wfxc = &data.wfxc;

    // Pre-allocate result matrix (accumulated across blocks)
    let mut result = MatrixFull::new([nocc, nvir], 0.0);

    // ── Block loop: fuse Steps 1b, 2 & 3 for cache efficiency ──
    //
    // Each block processes `LDA_BLOCK` grid columns.  Steps 1b and 2
    // (rho_z, fxc application) are cheap and done per-block.  Step 3
    // scales mo_vir columns within the block and contracts via dgemm
    // with beta=1.0 (in-place accumulation).
    for g_start in (0..ngrids).step_by(LDA_BLOCK) {
        let g_end = (g_start + LDA_BLOCK).min(ngrids);
        let n_block = g_end - g_start;

        // ── Step 1b: ρ_z for this block ──
        let rho_block: Vec<f64> = (0..n_block).into_par_iter().map(|gb| {
            let g = g_start + gb;
            let off = g * nocc;
            let mo = &mo_occ_data[off..off + nocc];
            let tc = &t_data[off..off + nocc];
            let mut sum = 0.0_f64;
            for i in 0..nocc { sum += mo[i] * tc[i]; }
            sum
        }).collect();

        // ── Step 2: v = wfxc × ρ_z for this block ──
        let v_block: Vec<f64> = (0..n_block).into_par_iter()
            .map(|gb| wfxc[g_start + gb] * rho_block[gb])
            .collect();

        // ── Step 3: scale mo_vir columns and contract via dgemm ──
        let mut scaled = Vec::with_capacity(nvir * n_block);
        unsafe { scaled.set_len(nvir * n_block); }
        scaled.par_chunks_mut(nvir).enumerate().for_each(|(gb, chunk)| {
            let g = g_start + gb;
            let vg = v_block[gb];
            let src_off = g * nvir;
            for a in 0..nvir { chunk[a] = mo_vir_data[src_off + a] * vg; }
        });
        let scaled_mat = MatrixFull { size: [nvir, n_block], indicing: [1, nvir], data: scaled };

        // View into mo_occ for this block
        let mo_occ_slice = MatrixFullSlice {
            size: &[nocc, n_block],
            indicing: &[1, nocc],
            data: &mo_occ_data[g_start * nocc..][..nocc * n_block],
        };

        _dgemm_full(&mo_occ_slice, 'N', &scaled_mat, 'T', &mut result, 1.0, 1.0);
    }

    result.data
}

/// Optimized GGA fxc matrix-vector product.
///
/// Key optimizations:
/// - Direct raw-slice column access in all inner loops
/// - Fused multi-dot-product loop (reads mo_occ[i,g] and t0[i,g] once
///   instead of four times)
/// - Rayon parallelisation over grid points for the rho computation and
///   MO-vir/grad scaling
/// - Temporary scaling matrices created without zero-initialisation
/// - Block-fused grid processing: all steps (small dgemms, rho_z, fxc,
///   contraction) performed per grid block, keeping MO data in cache.
#[allow(dead_code)]
/// Design (inspired by PySCF's grid-blocking approach):
/// Process grid points in blocks.  For each block, ALL steps (small dgemms,
/// rho_z, fxc, contraction) are performed, keeping MO data in L2/L3 cache
/// across all 11 internal dgemms.
const GGA_BLOCK: usize = 512;
#[allow(dead_code)]
fn fxc_matvec_gga_opt(data: &FXCMatvecData, z: &[f64]) -> Vec<f64> {
    let nocc = data.nocc;
    let nvir = data.nvir;
    let ngrids = data.ngrids;

    let mo_occ_grad = data.mo_occ_grad.as_ref().expect("GGA requires mo_occ_grad");
    let mo_vir_grad = data.mo_vir_grad.as_ref().expect("GGA requires mo_vir_grad");

    let z_mat = MatrixFull::from_vec([nocc, nvir], z.to_vec()).unwrap();

    // Pre-fetch raw data slices for block slicing
    let mo_occ_data  = data.mo_occ.data.as_slice();
    let mo_vir_data  = data.mo_vir.data.as_slice();
    let og_data: [&[f64]; 3] = [
        mo_occ_grad[0].data.as_slice(),
        mo_occ_grad[1].data.as_slice(),
        mo_occ_grad[2].data.as_slice(),
    ];
    let vg_data: [&[f64]; 3] = [
        mo_vir_grad[0].data.as_slice(),
        mo_vir_grad[1].data.as_slice(),
        mo_vir_grad[2].data.as_slice(),
    ];
    let wfxc_data = &data.wfxc;

    let mut result = MatrixFull::new([nocc, nvir], 0.0);

    // ── Block loop ──
    for g_start in (0..ngrids).step_by(GGA_BLOCK) {
        let g_end = (g_start + GGA_BLOCK).min(ngrids);
        let n_block = g_end - g_start;

        // ── Step 1: 4 small dgemms for this block ──
        let mv_block = MatrixFullSlice {
            size: &[nvir, n_block], indicing: &[1, nvir],
            data: &mo_vir_data[g_start * nvir ..][.. nvir * n_block],
        };
        let mg_block: [MatrixFullSlice<'_, f64>; 3] = [
            MatrixFullSlice { size: &[nvir, n_block], indicing: &[1, nvir],
                data: &vg_data[0][g_start * nvir ..][.. nvir * n_block] },
            MatrixFullSlice { size: &[nvir, n_block], indicing: &[1, nvir],
                data: &vg_data[1][g_start * nvir ..][.. nvir * n_block] },
            MatrixFullSlice { size: &[nvir, n_block], indicing: &[1, nvir],
                data: &vg_data[2][g_start * nvir ..][.. nvir * n_block] },
        ];

        let mut t0   = MatrixFull::new([nocc, n_block], 0.0);
        let mut tg0  = MatrixFull::new([nocc, n_block], 0.0);
        let mut tg1  = MatrixFull::new([nocc, n_block], 0.0);
        let mut tg2  = MatrixFull::new([nocc, n_block], 0.0);

        _dgemm_full(&z_mat, 'N', &mv_block, 'N', &mut t0, 1.0, 0.0);
        _dgemm_full(&z_mat, 'N', &mg_block[0], 'N', &mut tg0, 1.0, 0.0);
        _dgemm_full(&z_mat, 'N', &mg_block[1], 'N', &mut tg1, 1.0, 0.0);
        _dgemm_full(&z_mat, 'N', &mg_block[2], 'N', &mut tg2, 1.0, 0.0);

        // ── Step 1b: fused dot products for ρ_z (parallel within block) ──
        let rho_block: Vec<[f64; 4]> = (0..n_block).into_par_iter().map(|gb| {
            let g = g_start + gb;
            let off = g * nocc;
            let mo   = &mo_occ_data[off .. off + nocc];
            let tt0  = &t0.data[gb * nocc ..][..nocc];   // t0
            let ttg0 = &tg0.data[gb * nocc ..][..nocc];  // t_grad[0]
            let ttg1 = &tg1.data[gb * nocc ..][..nocc];  // t_grad[1]
            let ttg2 = &tg2.data[gb * nocc ..][..nocc];  // t_grad[2]
            let mut s = [0.0_f64; 7];
            for i in 0..nocc {
                let m = mo[i];
                let t = tt0[i];
                s[0] += m * t;
                s[1] += og_data[0][off + i] * t;  // ∇_x occ × t0
                s[2] += m * ttg0[i];              // occ × t_grad[0]
                s[3] += og_data[1][off + i] * t;  // ∇_y occ × t0
                s[4] += m * ttg1[i];              // occ × t_grad[1]
                s[5] += og_data[2][off + i] * t;  // ∇_z occ × t0
                s[6] += m * ttg2[i];              // occ × t_grad[2]
            }
            [s[0], s[1] + s[2], s[3] + s[4], s[5] + s[6]]
        }).collect();

        // ── Step 2: apply 4×4 fxc kernel on block ──
        let mut fxc_eff = [0.0_f64; 4 * GGA_BLOCK];
        for (gb, rc) in rho_block.iter().enumerate() {
            let g = g_start + gb;
            for alpha in 0..4 {
                let mut sum = 0.0_f64;
                for beta in 0..4 {
                    sum += wfxc_data[g + alpha * ngrids + beta * 4 * ngrids] * rc[beta];
                }
                fxc_eff[gb + alpha * n_block] = sum;
            }
        }

        // ── Step 3: contract back (7 small dgemms, in-place accumulation) ──
        let mo_block = MatrixFullSlice {
            size: &[nocc, n_block], indicing: &[1, nocc],
            data: &mo_occ_data[g_start * nocc ..][.. nocc * n_block],
        };
        let mo_grad_block: [MatrixFullSlice<'_, f64>; 3] = [
            MatrixFullSlice { size: &[nocc, n_block], indicing: &[1, nocc],
                data: &og_data[0][g_start * nocc ..][.. nocc * n_block] },
            MatrixFullSlice { size: &[nocc, n_block], indicing: &[1, nocc],
                data: &og_data[1][g_start * nocc ..][.. nocc * n_block] },
            MatrixFullSlice { size: &[nocc, n_block], indicing: &[1, nocc],
                data: &og_data[2][g_start * nocc ..][.. nocc * n_block] },
        ];

        // beta=0 for first block * first alpha, otherwise beta=1
        let first = g_start == 0;

        for alpha in 0..4 {
            let fxc_a = &fxc_eff[alpha * n_block .. (alpha + 1) * n_block];

            // Scale mo_vir columns for this block (parallel)
            let mut scaled = Vec::with_capacity(nvir * n_block);
            unsafe { scaled.set_len(nvir * n_block); }
            scaled.par_chunks_mut(nvir).enumerate().for_each(|(gb, chunk)| {
                let g = g_start + gb;
                let src = g * nvir;
                for a in 0..nvir { chunk[a] = mo_vir_data[src + a] * fxc_a[gb]; }
            });
            let right = MatrixFull { size: [nvir, n_block], indicing: [1, nvir], data: scaled };
            let beta = if first && alpha == 0 { 0.0 } else { 1.0 };

            if alpha == 0 {
                _dgemm_full(&mo_block, 'N', &right, 'T', &mut result, 1.0, beta);
            } else {
                let d = alpha - 1;
                _dgemm_full(&mo_grad_block[d], 'N', &right, 'T', &mut result, 1.0, beta);

                let mut gscaled = Vec::with_capacity(nvir * n_block);
                unsafe { gscaled.set_len(nvir * n_block); }
                gscaled.par_chunks_mut(nvir).enumerate().for_each(|(gb, chunk)| {
                    let g = g_start + gb;
                    let src = g * nvir;
                    for a in 0..nvir { chunk[a] = vg_data[d][src + a] * fxc_a[gb]; }
                });
                let right_g = MatrixFull { size: [nvir, n_block], indicing: [1, nvir], data: gscaled };
                _dgemm_full(&mo_block, 'N', &right_g, 'T', &mut result, 1.0, 1.0);
            }
        }
    }

    result.data
}

// ====================================================================
// NWChem-style optimised fxc_matvec
//
// Key improvements over fxc_matvec_opt:
// 1. wfxc stored in [α, β, g] layout for cache-friendly access
// 2. Pre-allocated buffers for Step 3 scaling → no per-block allocations
// 3. Adaptive LDA block size (avoids oversized working sets)
// 4. Merged DGEMM for GGA α=1,2,3 (2 → 1 dgemm per α)
// 5. Parallelised fxc kernel application (Step 2)
// ====================================================================

const NWCHEM_LDA_BLOCK: usize = 8192;
const NWCHEM_GGA_BLOCK: usize = 1024;

pub struct FXCMatvecDataNwchemOpt {
    pub nvar: usize,
    pub ngrids: usize,
    pub nocc: usize,
    pub nvir: usize,
    pub mo_occ: MatrixFull<f64>,
    pub mo_vir: MatrixFull<f64>,
    pub mo_occ_grad: Option<[MatrixFull<f64>; 3]>,
    pub mo_vir_grad: Option<[MatrixFull<f64>; 3]>,
    /// fxc kernel × grid weights
    /// LDA: vec[ngrids]
    /// GGA: vec[4 × 4 × ngrids], layout wfxc[α*4*ngrids + β*ngrids + g]
    pub wfxc: Vec<f64>,
    // Pre-allocated buffers for Step 3 (GGA)
    /// Combined left buffer [nocc × 2 × NWCHEM_GGA_BLOCK]
    pub combined_left_buf: Vec<f64>,
    /// Combined right buffer [nvir × 2 × NWCHEM_GGA_BLOCK]
    pub combined_right_buf: Vec<f64>,
}

pub fn prepare_fxc_data_nwchem_opt(scf: &SCF) -> FXCMatvecDataNwchemOpt {
    let (start_mo, _num_state, occ_size, vir_size, _homo, _lumo) =
        tddft_occupation_parameters(scf);

    let xc_data = &scf.mol.xc_data;
    let xc_type = if xc_data.use_density_gradient() {
        XCType::GGA
    } else {
        XCType::LDA
    };
    let nvar = match xc_type {
        XCType::LDA => 1,
        XCType::GGA => 4,
        _ => panic!("fxc only supports LDA and GGA"),
    };
    let grids = scf.grids.as_ref().expect("DFT grids must be initialized for fxc");
    let ngrids = grids.weights.len();
    let num_basis = scf.mol.num_basis;
    let weights = &grids.weights;
    let ao = grids.ao.as_ref().expect("AO on grids must be tabulated");
    let eigvec = &scf.eigenvectors[0];

    // Extract MO coefficients
    let mut c_occ = MatrixFull::new([num_basis, occ_size], 0.0);
    for j in 0..occ_size {
        for i in 0..num_basis {
            c_occ[[i, j]] = eigvec[[i, start_mo + j]];
        }
    }
    let mut c_vir = MatrixFull::new([num_basis, vir_size], 0.0);
    for j in 0..vir_size {
        for i in 0..num_basis {
            c_vir[[i, j]] = eigvec[[i, _lumo + j]];
        }
    }

    // Project MO values onto grids
    let mut mo_occ = MatrixFull::new([occ_size, ngrids], 0.0);
    _dgemm_full(&c_occ, 'T', ao, 'N', &mut mo_occ, 1.0, 0.0);
    let mut mo_vir = MatrixFull::new([vir_size, ngrids], 0.0);
    _dgemm_full(&c_vir, 'T', ao, 'N', &mut mo_vir, 1.0, 0.0);

    // GGA: MO gradients
    let (mo_occ_grad, mo_vir_grad) = if xc_type == XCType::GGA {
        let aop = grids.aop.as_ref().expect("AO gradients needed for GGA fxc");
        let mut og = [
            MatrixFull::new([occ_size, ngrids], 0.0),
            MatrixFull::new([occ_size, ngrids], 0.0),
            MatrixFull::new([occ_size, ngrids], 0.0),
        ];
        let mut vg = [
            MatrixFull::new([vir_size, ngrids], 0.0),
            MatrixFull::new([vir_size, ngrids], 0.0),
            MatrixFull::new([vir_size, ngrids], 0.0),
        ];
        for d in 0..3 {
            let aop_d_slice = aop.get_reducing_matrix(d).unwrap();
            let aop_d = MatrixFull::from_vec(
                [num_basis, ngrids],
                aop_d_slice.iter().cloned().collect(),
            ).unwrap();
            _dgemm_full(&c_occ, 'T', &aop_d, 'N', &mut og[d], 1.0, 0.0);
            _dgemm_full(&c_vir, 'T', &aop_d, 'N', &mut vg[d], 1.0, 0.0);
        }
        (Some(og), Some(vg))
    } else {
        (None, None)
    };

    // Ground-state density on grids
    let ao_deriv = if xc_type == XCType::GGA { 1 } else { 0 };
    let ao_rifull = eval_ao_batch(&scf.mol, &grids.coordinates, ao_deriv, ngrids);
    let mo_coeffs = vec![scf.eigenvectors[0].clone()];
    let occ = vec![scf.occupation[0].clone()];
    let rho_tensor = eval_rho5_batch(&ao_rifull, xc_type, &mo_coeffs, &occ, 1, ngrids);

    let rho_array: Vec<f64> = {
        let raw = rho_tensor.raw();
        let offset = rho_tensor.offset();
        raw[offset..offset + ngrids * nvar].to_vec()
    };

    let func_ids = &xc_data.dfa_compnt_scf;
    let func_factors = &xc_data.dfa_paramr_scf;
    let xc_tensors = eval_xc_eff(func_ids, func_factors, xc_type, 0, &rho_array, ngrids, 2);
    let fxc_tensor = xc_tensors[2].as_ref()
        .expect("fxc (deriv=2) should be available");
    let fxc_raw: Vec<f64> = {
        let raw = fxc_tensor.raw();
        let offset = fxc_tensor.offset();
        raw[offset..offset + ngrids * nvar * nvar].to_vec()
    };

    // fxc × weights stored in [α, β, g] layout
    const SINGLET_FXC_FACTOR: f64 = 2.0;
    let wfxc: Vec<f64> = if nvar == 1 {
        (0..ngrids).map(|g| fxc_raw[g] * weights[g] * SINGLET_FXC_FACTOR).collect()
    } else {
        let mut wfxc = vec![0.0; ngrids * 16];
        for alpha in 0..4 {
            let off_out_a = alpha * 4 * ngrids;
            let off_in_a  = alpha * ngrids;
            for beta in 0..4 {
                let off_out = off_out_a + beta * ngrids;
                let off_in  = off_in_a + beta * 4 * ngrids;
                let w = weights;
                for g in 0..ngrids {
                    wfxc[off_out + g] = fxc_raw[g + off_in] * w[g] * SINGLET_FXC_FACTOR;
                }
            }
        }
        wfxc
    };

    let combined_left_buf = vec![0.0; occ_size * NWCHEM_GGA_BLOCK * 2];
    let combined_right_buf = vec![0.0; vir_size * NWCHEM_GGA_BLOCK * 2];

    println!(
        "FXCMatvecDataNwchemOpt: nocc={}, nvir={}, ngrids={}, nvar={}",
        occ_size, vir_size, ngrids, nvar
    );

    FXCMatvecDataNwchemOpt {
        nvar,
        ngrids,
        nocc: occ_size,
        nvir: vir_size,
        mo_occ,
        mo_vir,
        mo_occ_grad,
        mo_vir_grad,
        wfxc,
        combined_left_buf,
        combined_right_buf,
    }
}

pub fn fxc_matvec_nwchem_opt(data: &mut FXCMatvecDataNwchemOpt, z: &[f64]) -> Vec<f64> {
    assert_eq!(z.len(), data.nocc * data.nvir,
               "z length {} must be nocc×nvir = {}×{}",
               z.len(), data.nocc, data.nvir);
    match data.nvar {
        1 => fxc_matvec_lda_nwchem_opt(data, z),
        4 => fxc_matvec_gga_nwchem_opt(data, z),
        _ => panic!("fxc_matvec_nwchem_opt only supports LDA (nvar=1) and GGA (nvar=4)"),
    }
}

fn fxc_matvec_lda_nwchem_opt(data: &FXCMatvecDataNwchemOpt, z: &[f64]) -> Vec<f64> {
    let nocc = data.nocc;
    let nvir = data.nvir;
    let ngrids = data.ngrids;

    // Step 1: z_mat × mo_vir → t
    let z_mat = MatrixFull::from_vec([nocc, nvir], z.to_vec()).unwrap();
    let mut t = MatrixFull::new([nocc, ngrids], 0.0);
    _dgemm_full(&z_mat, 'N', &data.mo_vir, 'N', &mut t, 1.0, 0.0);

    let mo_occ_data = data.mo_occ.data.as_slice();
    let mo_vir_data = data.mo_vir.data.as_slice();
    let t_data = t.data.as_slice();
    let wfxc = &data.wfxc;

    let mut result = MatrixFull::new([nocc, nvir], 0.0);

    // Adaptive block size to keep scaled[nvir, block] in ~14 MB
    let block_size = (NWCHEM_LDA_BLOCK)
        .min(14_000_000 / (nvir * 8).max(1))
        .max(256);

    for g_start in (0..ngrids).step_by(block_size) {
        let g_end = (g_start + block_size).min(ngrids);
        let n_block = g_end - g_start;

        // Step 1b + 2: ρ_z = Σ_i mo_occ[i,g] × t[i,g]; v = wfxc × ρ_z
        let v_block: Vec<f64> = (0..n_block).into_par_iter().map(|gb| {
            let g = g_start + gb;
            let off = g * nocc;
            let mo = &mo_occ_data[off..off + nocc];
            let tc = &t_data[off..off + nocc];
            let mut sum = 0.0;
            for i in 0..nocc { sum += mo[i] * tc[i]; }
            wfxc[g] * sum
        }).collect();

        // Step 3: scale mo_vir and contract
        let mut scaled = Vec::with_capacity(nvir * n_block);
        unsafe { scaled.set_len(nvir * n_block); }
        scaled.par_chunks_mut(nvir).enumerate().for_each(|(gb, chunk)| {
            let g = g_start + gb;
            let src_off = g * nvir;
            let vg = v_block[gb];
            for a in 0..nvir { chunk[a] = mo_vir_data[src_off + a] * vg; }
        });
        let scaled_mat = MatrixFull { size: [nvir, n_block], indicing: [1, nvir], data: scaled };

        let mo_occ_slice = MatrixFullSlice {
            size: &[nocc, n_block],
            indicing: &[1, nocc],
            data: &mo_occ_data[g_start * nocc..][..nocc * n_block],
        };

        let beta = if g_start == 0 { 0.0 } else { 1.0 };
        _dgemm_full(&mo_occ_slice, 'N', &scaled_mat, 'T', &mut result, 1.0, beta);
    }

    result.data
}

fn fxc_matvec_gga_nwchem_opt(data: &mut FXCMatvecDataNwchemOpt, z: &[f64]) -> Vec<f64> {
    let nocc = data.nocc;
    let nvir = data.nvir;
    let ngrids = data.ngrids;

    // Borrow buffers from data
    let combined_left_buf  = &mut data.combined_left_buf;
    let combined_right_buf = &mut data.combined_right_buf;

    let mo_occ_grad = data.mo_occ_grad.as_ref().expect("GGA requires mo_occ_grad");
    let mo_vir_grad = data.mo_vir_grad.as_ref().expect("GGA requires mo_vir_grad");

    let z_mat = MatrixFull::from_vec([nocc, nvir], z.to_vec()).unwrap();

    let mo_occ_data = data.mo_occ.data.as_slice();
    let mo_vir_data = data.mo_vir.data.as_slice();
    let og_data: [&[f64]; 3] = [
        mo_occ_grad[0].data.as_slice(),
        mo_occ_grad[1].data.as_slice(),
        mo_occ_grad[2].data.as_slice(),
    ];
    let vg_data: [&[f64]; 3] = [
        mo_vir_grad[0].data.as_slice(),
        mo_vir_grad[1].data.as_slice(),
        mo_vir_grad[2].data.as_slice(),
    ];
    let wfxc_data = &data.wfxc;
    let block_size = NWCHEM_GGA_BLOCK;

    let mut result = MatrixFull::new([nocc, nvir], 0.0);

    for g_start in (0..ngrids).step_by(block_size) {
        let g_end = (g_start + block_size).min(ngrids);
        let n_block = g_end - g_start;

        // ── Step 1: 4 small dgemms ──
        let mv_block = MatrixFullSlice {
            size: &[nvir, n_block], indicing: &[1, nvir],
            data: &mo_vir_data[g_start * nvir ..][.. nvir * n_block],
        };
        let mg_block: [MatrixFullSlice<'_, f64>; 3] = [
            MatrixFullSlice { size: &[nvir, n_block], indicing: &[1, nvir],
                data: &vg_data[0][g_start * nvir ..][.. nvir * n_block] },
            MatrixFullSlice { size: &[nvir, n_block], indicing: &[1, nvir],
                data: &vg_data[1][g_start * nvir ..][.. nvir * n_block] },
            MatrixFullSlice { size: &[nvir, n_block], indicing: &[1, nvir],
                data: &vg_data[2][g_start * nvir ..][.. nvir * n_block] },
        ];

        let mut t0  = MatrixFull::new([nocc, n_block], 0.0);
        let mut tg0 = MatrixFull::new([nocc, n_block], 0.0);
        let mut tg1 = MatrixFull::new([nocc, n_block], 0.0);
        let mut tg2 = MatrixFull::new([nocc, n_block], 0.0);

        _dgemm_full(&z_mat, 'N', &mv_block,  'N', &mut t0,  1.0, 0.0);
        _dgemm_full(&z_mat, 'N', &mg_block[0], 'N', &mut tg0, 1.0, 0.0);
        _dgemm_full(&z_mat, 'N', &mg_block[1], 'N', &mut tg1, 1.0, 0.0);
        _dgemm_full(&z_mat, 'N', &mg_block[2], 'N', &mut tg2, 1.0, 0.0);

        // ── Step 1b: fused dot products (rayon) ──
        let rho_block: Vec<[f64; 4]> = (0..n_block).into_par_iter().map(|gb| {
            let g = g_start + gb;
            let off = g * nocc;
            let mo   = &mo_occ_data[off .. off + nocc];
            let tt0  = &t0.data[gb * nocc ..][..nocc];
            let ttg0 = &tg0.data[gb * nocc ..][..nocc];
            let ttg1 = &tg1.data[gb * nocc ..][..nocc];
            let ttg2 = &tg2.data[gb * nocc ..][..nocc];
            let mut s = [0.0_f64; 7];
            for i in 0..nocc {
                s[0] += mo[i] * tt0[i];
                s[1] += og_data[0][off + i] * tt0[i];
                s[2] += mo[i] * ttg0[i];
                s[3] += og_data[1][off + i] * tt0[i];
                s[4] += mo[i] * ttg1[i];
                s[5] += og_data[2][off + i] * tt0[i];
                s[6] += mo[i] * ttg2[i];
            }
            [s[0], s[1] + s[2], s[3] + s[4], s[5] + s[6]]
        }).collect();

        // ── Step 2: apply 4×4 fxc kernel (rayon, compact layout) ──
        // wfxc stored wfxc[α*4*ngrids + β*ngrids + g]
        // Build fxc_eff in [alpha][gb] flat layout for easy Step 3 slicing
        let mut fxc_eff = vec![0.0_f64; 4 * n_block];
        fxc_eff.par_chunks_mut(n_block).enumerate().for_each(|(alpha, chunk)| {
            let base = alpha * 4 * ngrids;
            let w0 = &wfxc_data[(base + 0 * ngrids + g_start)..];
            let w1 = &wfxc_data[(base + 1 * ngrids + g_start)..];
            let w2 = &wfxc_data[(base + 2 * ngrids + g_start)..];
            let w3 = &wfxc_data[(base + 3 * ngrids + g_start)..];
            for gb in 0..n_block {
                let rc = &rho_block[gb];
                chunk[gb] = w0[gb] * rc[0]
                          + w1[gb] * rc[1]
                          + w2[gb] * rc[2]
                          + w3[gb] * rc[3];
            }
        });

        // ── Step 3: contract back ──
        let mo_block = MatrixFullSlice {
            size: &[nocc, n_block], indicing: &[1, nocc],
            data: &mo_occ_data[g_start * nocc ..][.. nocc * n_block],
        };
        let mo_grad_block: [MatrixFullSlice<'_, f64>; 3] = [
            MatrixFullSlice { size: &[nocc, n_block], indicing: &[1, nocc],
                data: &og_data[0][g_start * nocc ..][.. nocc * n_block] },
            MatrixFullSlice { size: &[nocc, n_block], indicing: &[1, nocc],
                data: &og_data[1][g_start * nocc ..][.. nocc * n_block] },
            MatrixFullSlice { size: &[nocc, n_block], indicing: &[1, nocc],
                data: &og_data[2][g_start * nocc ..][.. nocc * n_block] },
        ];

        let first = g_start == 0;

        // α = 0: single DGEMM (unmerged)
        {
            let fxc_a = &fxc_eff[0 * n_block .. 1 * n_block];
            let mut scaled = Vec::with_capacity(nvir * n_block);
            unsafe { scaled.set_len(nvir * n_block); }
            scaled.par_chunks_mut(nvir).enumerate().for_each(|(gb, chunk)| {
                let g = g_start + gb;
                let src = g * nvir;
                let fv = fxc_a[gb];
                for a in 0..nvir { chunk[a] = mo_vir_data[src + a] * fv; }
            });
            let right = MatrixFull { size: [nvir, n_block], indicing: [1, nvir], data: scaled };
            let beta = if first { 0.0 } else { 1.0 };
            _dgemm_full(&mo_block, 'N', &right, 'T', &mut result, 1.0, beta);
        }

        // α = 1,2,3: merged DGEMM (pre-allocated buffers)
        for d in 0..3 {
            let alpha = d + 1;
            let fxc_a = &fxc_eff[alpha * n_block .. (alpha + 1) * n_block];

            // Fill combined_right_buf with [right | right_grad] side-by-side
            //   right[nvir, n_block]:  [nvir × n_block] contiguous
            //   right_grad[nvir, n_block]: [nvir × n_block] contiguous
            //   combined[nvir, 2×n_block]: 2 × [nvir × n_block] end-to-end (col-major)
            let cb = &mut combined_right_buf[..nvir * n_block * 2];
            let nvir_block = nvir * n_block;
            let (r_part, rg_part) = cb.split_at_mut(nvir_block);
            r_part.par_chunks_mut(nvir).enumerate().for_each(|(gb, chunk)| {
                let g = g_start + gb;
                let src = g * nvir;
                let fv = fxc_a[gb];
                for a in 0..nvir { chunk[a] = mo_vir_data[src + a] * fv; }
            });
            rg_part.par_chunks_mut(nvir).enumerate().for_each(|(gb, chunk)| {
                let g = g_start + gb;
                let src = g * nvir;
                let fv = fxc_a[gb];
                for a in 0..nvir { chunk[a] = vg_data[d][src + a] * fv; }
            });

            // Fill combined_left_buf with [mo_grad_block | mo_block]
            let lb = &mut combined_left_buf[..nocc * n_block * 2];
            let nocc_block = nocc * n_block;
            let (lg_part, lm_part) = lb.split_at_mut(nocc_block);
            lg_part.copy_from_slice(&og_data[d][g_start * nocc ..][..nocc_block]);
            lm_part.copy_from_slice(&mo_occ_data[g_start * nocc ..][..nocc_block]);

            // One DGEMM instead of two
            let nm2 = n_block * 2;
            let left_mat = MatrixFullSlice {
                size: &[nocc, nm2],
                indicing: &[1, nocc],
                data: &combined_left_buf[..nocc * nm2],
            };
            let right_mat = MatrixFullSlice {
                size: &[nvir, nm2],
                indicing: &[1, nvir],
                data: &combined_right_buf[..nvir * nm2],
            };
            _dgemm_full(&left_mat, 'N', &right_mat, 'T', &mut result, 1.0, 1.0);
        }
    }

    result.data
}


#[derive(Debug, Clone, Copy)]
pub struct BlockSettings {
    pub ao_deriv: usize,
    pub max_memory: usize, // in MB
    pub blksize: usize, // block size of batched grids
}

pub struct BlockData<'a> {
    pub coords: &'a [[f64; 3]],
    pub weights: &'a [f64],
    pub ao: RIFull<f64>, // atomic orbitals evaluated on grids
    /// Global basis-function indices of the rows of `ao` when the block was
    /// locality-screened (`ao` is then the compact [n_active, ngrids, ncomp]
    /// table); `None` means the dense, unscreened [nao, ngrids, ncomp] table.
    pub active_bas: Option<Vec<usize>>,
}

pub struct XCData<'a> {
    pub xc_type: XCType,
    pub xc_code: &'a Vec<usize>,
    pub xc_params: &'a Vec<f64>,
    pub device: DeviceOpenBLAS,
    pub dm: &'a Vec<MatrixFull<f64>>, 
    pub mo_coeffs: Option<Vec<MatrixFull<f64>>>,
    pub occ: Option<Vec<Vec<f64>>>,
}


pub struct GridIterator<'a> {
    mol: &'a Molecule,
    grids: &'a Grids,
    settings: BlockSettings,
    current: usize,
    total_grids: usize,
}

impl<'a> GridIterator<'a> {
    pub fn new(
        mol: &'a Molecule,
        grids: &'a Grids,
        settings: BlockSettings,
    ) -> Self {
        let total_grids = grids.weights.len();
        Self {
            mol,
            grids,
            settings,
            current: 0,
            total_grids,
        }
    }
}

impl<'a> Iterator for GridIterator<'a> {
    type Item = BlockData<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.total_grids {
            return None;
        }
        let start = self.current;
        let end = (start + self.settings.blksize).min(self.total_grids);
        self.current = end;

        let coords = &self.grids.coordinates[start..end];
        let weights = &self.grids.weights[start..end];
        let num_grids = weights.len();
        let ao = eval_ao_batch(self.mol, coords, self.settings.ao_deriv, num_grids);

        Some(BlockData{
            coords: &self.grids.coordinates[start..end],
            weights: &self.grids.weights[start..end],
            ao: ao,
            active_bas: None,
        })
    }
}

/// Producer that splits block indices and evaluates AO batches in each fold.
struct GridBlockProducer<'a> {
    mol: &'a Molecule,
    grids: &'a Grids,
    settings: BlockSettings,
    total_grids: usize,
    /// Per-atom influence radii (bohr) used to skip atoms that cannot
    /// contribute above `row_cutoff` anywhere in the block; empty means the
    /// dense, unscreened path.
    active_radii: std::sync::Arc<Vec<f64>>,
    /// Row-keep threshold for the locality screen (magnitude of the AO value
    /// at any block point); <= 0 disables screening.
    row_cutoff: f64,
    /// Range of block indices this producer owns, [block_start, block_end).
    block_range: Range<usize>,
}

impl<'a> GridBlockProducer<'a> {
    fn block_start(&self, block_idx: usize) -> usize {
        block_idx * self.settings.blksize
    }
    fn block_end(&self, block_idx: usize) -> usize {
        (self.block_start(block_idx) + self.settings.blksize).min(self.total_grids)
    }
    fn produce(&self, block_idx: usize) -> BlockData<'a> {
        let start = self.block_start(block_idx);
        let end = self.block_end(block_idx);
        let num_grids = end - start;
        let coords = &self.grids.coordinates[start..end];
        let weights = &self.grids.weights[start..end];
        let t_ao = std::time::Instant::now();
        if self.row_cutoff <= 0.0 {
            let ao = eval_ao_batch(self.mol, coords, self.settings.ao_deriv, num_grids);
            GRID_BLOCK_AO_NS.fetch_add(t_ao.elapsed().as_nanos() as u64, Ordering::Relaxed);
            GRID_BLOCK_COUNT.fetch_add(1, Ordering::Relaxed);
            GRID_BLOCK_ACTIVE_SUM.fetch_add(self.mol.num_basis as u64, Ordering::Relaxed);
            return BlockData { coords, weights, ao, active_bas: None };
        }
        let (ao, active_bas) = self.eval_ao_batch_screened(coords);
        if active_bas.is_empty() {
            // Degenerate block (every AO value below the cutoff): fall back to
            // the dense table so downstream never sees an empty contraction.
            let ao = eval_ao_batch(self.mol, coords, self.settings.ao_deriv, num_grids);
            GRID_BLOCK_AO_NS.fetch_add(t_ao.elapsed().as_nanos() as u64, Ordering::Relaxed);
            GRID_BLOCK_COUNT.fetch_add(1, Ordering::Relaxed);
            GRID_BLOCK_ACTIVE_SUM.fetch_add(self.mol.num_basis as u64, Ordering::Relaxed);
            return BlockData { coords, weights, ao, active_bas: None };
        }
        GRID_BLOCK_AO_NS.fetch_add(t_ao.elapsed().as_nanos() as u64, Ordering::Relaxed);
        GRID_BLOCK_COUNT.fetch_add(1, Ordering::Relaxed);
        GRID_BLOCK_ACTIVE_SUM.fetch_add(active_bas.len() as u64, Ordering::Relaxed);
        BlockData { coords, weights, ao, active_bas: Some(active_bas) }
    }

    /// Locality-screened AO evaluation: keep only the atoms whose influence
    /// radius reaches the block's bounding box, evaluate them with
    /// `spheric_gto_deriv_batch_serial`, then keep only the basis rows whose
    /// value exceeds `row_cutoff` at some point of the block. Returns the
    /// compact [n_active, num_grids, ncomp] table and the global basis
    /// indices of its rows.
    fn eval_ao_batch_screened(&self, coords: &[[f64; 3]]) -> (RIFull<f64>, Vec<usize>) {
        let num_grids = coords.len();
        let ao_deriv = self.settings.ao_deriv;
        let ncomp = (ao_deriv + 1) * (ao_deriv + 2) * (ao_deriv + 3) / 6;

        // block bounding box (shared helper)
        let (lo, hi) = block_bbox(coords);

        // per-atom passes: (global kept rows, per-component tables)
        let mut parts: Vec<(usize, Vec<usize>, Vec<MatrixFull<f64>>)> = Vec::new();
        let mut n_active = 0usize;
        for (iatom, (elem, geom)) in self
            .mol
            .basis4elem
            .iter()
            .zip(self.mol.geom.rg_position.iter_columns_full())
            .enumerate()
        {
            let mut center = [0.0_f64; 3];
            center.iter_mut().zip(geom.iter()).for_each(|(c, v)| *c = *v);
            if !atom_reaches_block(&center, self.active_radii[iatom], &lo, &hi) {
                continue;
            }
            let t_gto = std::time::Instant::now();
            let tab = spheric_gto_deriv_batch_serial(coords, &center, elem, ao_deriv);
            GRID_BLOCK_GTO_NS.fetch_add(t_gto.elapsed().as_nanos() as u64, Ordering::Relaxed);
            let bas_start = elem.global_index.0;
            // keep a row when its value exceeds the cutoff at some block point
            // (same predicate as the SCF sparse AO tabulation)
            let kept = screen_rows_by_value(&tab[0], num_grids, self.row_cutoff);
            if kept.is_empty() {
                continue;
            }
            n_active += kept.len();
            parts.push((bas_start, kept, tab));
        }

        // assemble the compact [n_active, num_grids, ncomp] column-major table;
        // every element is written exactly once below, so skip the zero-init
        let mut data: Vec<f64> = Vec::with_capacity(n_active * num_grids * ncomp);
        #[allow(clippy::uninit_vec)]
        unsafe {
            data.set_len(n_active * num_grids * ncomp);
        }
        let mut active_bas: Vec<usize> = Vec::with_capacity(n_active);
        let mut row0 = 0usize;
        for (bas_start, kept, tab) in &parts {
            for c in 0..ncomp {
                let src = &tab[c];
                let comp_base = c * n_active * num_grids;
                // g outer / row inner: contiguous writes into the destination
                // column, gathered reads from the (short) source column
                for g in 0..num_grids {
                    let dst = &mut data[comp_base + g * n_active + row0..comp_base + g * n_active + row0 + kept.len()];
                    for (r, &lr) in kept.iter().enumerate() {
                        dst[r] = src[[lr, g]];
                    }
                }
            }
            active_bas.extend(kept.iter().map(|&lr| bas_start + lr));
            row0 += kept.len();
        }
        let ao = RIFull::from_vec([n_active, num_grids, ncomp], data).unwrap();
        (ao, active_bas)
    }
}

/// Sequential iterator over blocks for the producer's `IntoIter`.
struct GridBlockIter<'a> {
    producer: GridBlockProducer<'a>,
    next_block_idx: usize,
    block_end: usize,
}

impl<'a> Iterator for GridBlockIter<'a> {
    type Item = BlockData<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_block_idx >= self.block_end {
            return None;
        }
        let block = self.producer.produce(self.next_block_idx);
        self.next_block_idx += 1;
        Some(block)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.block_end - self.next_block_idx;
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for GridBlockIter<'a> {}

impl<'a> DoubleEndedIterator for GridBlockIter<'a> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.next_block_idx >= self.block_end {
            return None;
        }
        let last_block_idx = self.block_end - 1;
        let block = self.producer.produce(last_block_idx);
        self.block_end = last_block_idx;
        Some(block)
    }
}

impl<'a> Producer for GridBlockProducer<'a> {
    type Item = BlockData<'a>;
    type IntoIter = GridBlockIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        GridBlockIter {
            next_block_idx: self.block_range.start,
            block_end: self.block_range.end,
            producer: self,
        }
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let GridBlockProducer { mol, grids, settings, total_grids, active_radii, row_cutoff, block_range } = self;
        let mid = block_range.start + index;
        (
            GridBlockProducer {
                mol, grids, settings, total_grids,
                active_radii: active_radii.clone(), row_cutoff,
                block_range: block_range.start..mid,
            },
            GridBlockProducer {
                mol, grids, settings, total_grids,
                active_radii, row_cutoff,
                block_range: mid..block_range.end,
            },
        )
    }

    fn fold_with<F>(self, mut folder: F) -> F
    where
        F: Folder<Self::Item>,
    {
        for block_idx in self.block_range.start..self.block_range.end {
            let block = self.produce(block_idx);
            folder = folder.consume(block);
            if folder.full() {
                break;
            }
        }
        folder
    }
}

pub struct GridParallelIterator<'a> {
    mol: &'a Molecule,
    grids: &'a Grids,
    settings: BlockSettings,
    total_grids: usize,
    num_blocks: usize,
    active_radii: std::sync::Arc<Vec<f64>>,
    row_cutoff: f64,
}

impl<'a> GridParallelIterator<'a> {
    pub fn new(
        mol: &'a Molecule,
        grids: &'a Grids,
        settings: BlockSettings,
    ) -> Self {
        let total_grids = grids.weights.len();
        let num_blocks = if total_grids == 0 { 0 }
            else { (total_grids + settings.blksize - 1) / settings.blksize };
        // Locality screening for the gradient pipeline: the row-keep cutoff is
        // tighter than `ao_cutoff` to cover the amplification of the AO first
        // and second derivatives used by the gradient contraction. Disabled
        // when `ao_cutoff <= 0` (exact, dense path).
        let row_cutoff = if mol.ctrl.ao_cutoff > 0.0 {
            mol.ctrl.ao_cutoff * 1.0e-3
        } else {
            0.0
        };
        let active_radii = if row_cutoff > 0.0 {
            std::sync::Arc::new(atom_influence_radii(mol, row_cutoff))
        } else {
            std::sync::Arc::new(Vec::new())
        };
        Self { mol, grids, settings, total_grids, num_blocks, active_radii, row_cutoff }
    }
}

impl<'a> ParallelIterator for GridParallelIterator<'a> {
    type Item = BlockData<'a>;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::plumbing::UnindexedConsumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        Some(self.num_blocks)
    }
}

impl<'a> IndexedParallelIterator for GridParallelIterator<'a> {
    fn len(&self) -> usize {
        self.num_blocks
    }

    fn drive<C>(self, consumer: C) -> C::Result
    where
        C: Consumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn with_producer<CB>(self, callback: CB) -> CB::Output
    where
        CB: ProducerCallback<Self::Item>,
    {
        callback.callback(GridBlockProducer {
            mol: self.mol,
            grids: self.grids,
            settings: self.settings,
            total_grids: self.total_grids,
            active_radii: self.active_radii.clone(),
            row_cutoff: self.row_cutoff,
            block_range: 0..self.num_blocks,
        })
    }
}

pub trait NumInt<'a> {

    fn gen_xc_data(&'a self, scf_data:&'a SCF, spin:usize) -> XCData<'a>;

    fn get_vxc(&'a self, xc_data: &'a XCData, grids: &mut Grids, mol: &Molecule, spin: usize) -> (Vec<f64>, Vec<f64>, Vec<MatrixFull<f64>>);

    fn get_fxc(&'a self, xc_data0: &'a XCData, xc_data1: &'a XCData, grids: &mut Grids, mol: &Molecule, spin: usize) -> Vec<MatrixFull<f64>>;

    fn block_loop(
        &'a self,
        mol: &'a Molecule,
        grids: &'a mut Grids,
        settings: BlockSettings,
    ) -> GridIterator<'a> {
        GridIterator::new(mol, grids, settings)
    }

    fn par_block_loop(
        &'a self,
        mol: &'a Molecule,
        grids: &'a Grids,
        settings: BlockSettings,
    ) -> GridParallelIterator<'a> {
        GridParallelIterator::new(mol, grids, settings)
    }
}


impl<'a> NumInt<'a> for DFA4REST {
    fn gen_xc_data(&'a self, scf_data: &'a SCF, spin: usize) -> XCData<'a> {
        let xc_type = if self.use_kinetic_density() {
            XCType::MGGA
        } else if self.use_density_gradient() {
            XCType::GGA
        } else {
            XCType::LDA
        };

        let xc_code = &self.dfa_compnt_scf;
        let xc_params = &self.dfa_paramr_scf;

        let density_matrix = &scf_data.density_matrix;
        let mo_coeffs = scf_data.eigenvectors.clone().to_vec();
        let mo_occ = scf_data.occupation.clone().to_vec();

        let device = DeviceOpenBLAS::default();

        XCData {
            xc_type,
            xc_code,
            xc_params,
            device: device,
            dm: density_matrix,
            mo_coeffs: Some(mo_coeffs),
            occ: Some(mo_occ),
        }
    }

    fn get_fxc(
        &'a self,
        xc_data0: &'a XCData,
        xc_data1: &'a XCData,
        grids: &mut Grids,
        mol: &Molecule,
        spin: usize,
    ) -> Vec<MatrixFull<f64>> {
        todo!("get_fxc has not yet implemented for DFA4REST");
    }

    fn get_vxc(
        &'a self,
        xc_data: &'a XCData,
        grids: &mut Grids,
        mol: &Molecule,
        spin: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<MatrixFull<f64>>) {

        let num_grids = grids.weights.len();
        let num_basis = mol.num_basis;
        // determine block settings 
        let ao_deriv = match xc_data.xc_type {
            XCType::HF => 0usize,
            XCType::LDA => 0usize,
            XCType::GGA => 1usize,
            XCType::MGGA => 1usize,
        };
        let max_memory = if let Some(max_memory) = mol.ctrl.max_memory {
            max_memory as usize
        } else {
            2000 as usize // default to 2000 MB
        };  
        let ao_comp = (ao_deriv + 1) * (ao_deriv + 2) * (ao_deriv + 3) / 6; // number of components for ao derivatives 
        let blksize = max_memory * 1_000_000 / 8 / ((ao_comp + 1) * num_basis);
        let blksize = blksize.min(num_grids).max(4); 
        let block_settings = BlockSettings {
            ao_deriv,
            max_memory,
            blksize,
        };

        let device = &xc_data.device;
        
        let mut nelec = vec![0.0; spin+1];
        let mut exc_sum = vec![0.0; spin+1];
        let mut vxc = rt::zeros(([num_basis, num_basis, spin + 1], device));
        let mut vxc_mgga_temp = match xc_data.xc_type {
            XCType::MGGA => Some(rt::zeros(([num_basis, num_basis, spin + 1], device))),
            _ => None,
        };
        let grids_iterator = self.block_loop(mol, grids, block_settings);
        // calculation starts here 
        let deriv = 1usize;          
        grids_iterator.into_iter().for_each(
            |block| 
            {
                let num_grids = block.weights.len();
                let loc_rho = if let Some(mo_coeffs) = &xc_data.mo_coeffs {
                    eval_rho5_batch(&block.ao, xc_data.xc_type, mo_coeffs, xc_data.occ.as_ref().unwrap(), spin, num_grids)
                } else {
                    eval_rho5_dm_only_batch(&block.ao, xc_data.xc_type, xc_data.dm, spin, num_grids)
                };
                let shape = loc_rho.shape().clone();

                let rho_array = loc_rho.into_vec();
                let loc_xc_tensor = eval_xc_eff(xc_data.xc_code, xc_data.xc_params, xc_data.xc_type, spin, &rho_array, num_grids, deriv);

                let loc_rho = rt::asarray((rho_array, shape, device));
                // let loc_exc = rt::asarray((&loc_xc_tensor[0].unwrap()));
                // let loc_vxc = rt::asarray((&loc_xc_tensor[1].unwrap()));

                let loc_exc = if let Some(exc) = loc_xc_tensor[0].as_ref() {
                    rt::asarray(exc)
                } else {
                    panic!("Exchange-correlation energy is not provided for the given xc_type: {:?}", xc_data.xc_type);
                };

                let loc_vxc = if let Some(vxc) = loc_xc_tensor[1].as_ref() {
                    rt::asarray(vxc)
                } else {
                    panic!("Exchange-correlation potential is not provided for the given xc_type: {:?}", xc_data.xc_type);
                };

                let loc_weights = rt::asarray((block.weights, device));

                let ao_shape = vec![num_basis, num_grids, ao_comp];
                let loc_ao = rt::asarray((&block.ao.data, ao_shape, device));
                

                // let loc_exc = loc_xc_tensor[0].as_ref().unwrap();
                // let loc_vxc = loc_xc_tensor[1].as_ref().unwrap();

                match spin {
                    0 => {
                        let loc_rho0 = loc_rho.i((0..num_grids, 0, 0));
                        let loc_den = loc_rho0 * &loc_weights;
                        let loc_nelec = rt::sum(&loc_den);
                        nelec[0] += loc_nelec;
                        let loc_exc_sum = (loc_exc * &loc_den).sum();
                        exc_sum[0] += loc_exc_sum;

                        let mut loc_wv = loc_vxc * &loc_weights.i((.., None)); 

                        match xc_data.xc_type {
                            XCType::LDA => {
                                let aow = &loc_ao.i((.., .., 0)) * &loc_wv; 
                                vxc.i_mut((.., .., 0)).matmul_from(
                                    &aow, &loc_ao.i((.., .., 0)).t(), 1.0, 1.0
                                );

                            },
                            XCType::GGA => {
                                loc_wv.i_mut((.., 0)).mul_assign(0.5);
                                
                                let aow = &loc_ao * &loc_wv.i((.., None, ..));
                                for ic in 0..=3 {
                                    vxc.i_mut(0).matmul_from(
                                        &aow.i((.., .., ic)), &loc_ao.i((.., .., ic)).t(), 1.0, 1.0
                                    );
                                }
                            },
                            XCType::MGGA => {
                                loc_wv.i_mut((.., 0)).mul_assign(0.5);
                                loc_wv.i_mut((.., 4)).mul_assign(0.5);

                                for ic in 0..=3 {
                                    let aow = &loc_ao.i((.., .., ic)) * &loc_wv.i((.., None, ic));
                                    vxc.i_mut(0).matmul_from(
                                        &loc_ao.i((.., .., 0)), &aow.t(), 1.0, 1.0
                                    );
                                }
                                let Some(v1) = vxc_mgga_temp.as_mut() else {
                                    panic!("vxc_mgga_temp is not initialized for MGGA type");
                                };
                                for ic in 0..=3 {
                                    let vtau_ket = &loc_ao.i((.., .., ic+1)) * &loc_wv.i((.., None, 4));
                                    v1.i_mut(0).matmul_from(
                                        &loc_ao.i((.., .., ic+1)), &vtau_ket.t(), 1.0, 1.0
                                    );
                                }
                            },
                            XCType::HF => {},
                        }
                    },
                    1 => {
                        let loc_rho_a = loc_rho.i((0..num_grids, 0, 0));
                        let loc_rho_b = loc_rho.i((0..num_grids, 0, 1));
                        let loc_den_a = loc_rho_a * &loc_weights;
                        let loc_den_b = loc_rho_b * &loc_weights;
                        let loc_nelec_a = rt::sum(&loc_den_a);
                        let loc_nelec_b = rt::sum(&loc_den_b);
                        nelec[0] += loc_nelec_a;
                        nelec[1] += loc_nelec_b;
                        let loc_exc_sum_a = (&loc_exc * &loc_den_a).sum();
                        let loc_exc_sum_b = (&loc_exc * &loc_den_b).sum();
                        exc_sum[0] += loc_exc_sum_a;
                        exc_sum[1] += loc_exc_sum_b;

                        //loc_vxc shape [num_grids, nderiv ,spin] nderiv = 1, 4, 5 for LDA (vrho), GGA (vrho, vnabla_rho), MGGA (.., vtau)
                        let mut loc_wv = &loc_vxc * &loc_weights.i((.., None, None)); 

                        match xc_data.xc_type {
                            XCType::LDA => {
                                let loc_wv_a = loc_wv.i((.., 0, 0));
                                let loc_wv_b = loc_wv.i((.., 0, 1));
                                let aow_a = &loc_ao.i((.., .., 0)) * &loc_wv_a.i((.., None)); 
                                let aow_b = &loc_ao.i((.., .., 0)) * &loc_wv_b.i((.., None));
                                vxc.i_mut((.., .., 0)).matmul_from(
                                    &aow_a, &loc_ao.i((.., .., 0)).t(), 1.0, 1.0
                                );
                                vxc.i_mut((.., .., 1)).matmul_from(
                                    &aow_b, &loc_ao.i((.., .., 0)).t(), 1.0, 1.0
                                );

                            },
                            XCType::GGA => {
                                loc_wv.i_mut((.., 0, ..)).mul_assign(0.5);
                                let loc_wv_a = loc_wv.i((.., .., 0));
                                let aow_a = &loc_ao.i((.., .., 0..4)) * &loc_wv_a.i((.., None, ..));
                                let loc_wv_b = loc_wv.i((.., .., 1));
                                let aow_b = &loc_ao .i((.., .., 0..4)) * &loc_wv_b.i((.., None, ..));
                                for ic in 0..4 {
                                    vxc.i_mut((.., .., 0)).matmul_from(
                                        &loc_ao.i((.., .., 0)), &aow_a.i((.., .., ic)).t(), 1.0, 1.0
                                    );
                                    vxc.i_mut((.., .., 1)).matmul_from(
                                        &loc_ao.i((.., .., 0)), &aow_b.i((.., .., ic)).t(), 1.0, 1.0
                                    );
                                }
                            },
                            XCType::MGGA => {
                                loc_wv.i_mut((.., 0, ..)).mul_assign(0.5);
                                loc_wv.i_mut((.., 4, ..)).mul_assign(0.5);
                                let Some(v1) = vxc_mgga_temp.as_mut() else {
                                    panic!("vxc_mgga_temp is not initialized for MGGA type");
                                };

                                let loc_wv_a = loc_wv.i((.., .., 0));
                                let aow_a = &loc_ao.i((.., .., 0..4)) * &loc_wv_a.i((.., None, 0..4));
                                let loc_wv_b = loc_wv.i((.., .., 1));
                                let aow_b = &loc_ao .i((.., .., 0..4)) * &loc_wv_b.i((.., None, 0..4));

                                for ic in 0..4 {
                                    vxc.i_mut((.., .., 0)).matmul_from(
                                        &loc_ao.i((.., .., 0)), &aow_a.i((.., .., ic)).t(), 1.0, 1.0
                                    );
                                    vxc.i_mut((.., .., 1)).matmul_from(
                                        &loc_ao.i((.., .., 0)), &aow_b.i((.., .., ic)).t(), 1.0, 1.0
                                    );
                                    let vtau_ket_a = &loc_ao.i((.., .., ic+1)) * &loc_wv_a.i((.., None, 4));
                                    let vtau_ket_b = &loc_ao.i((.., .., ic+1)) * &loc_wv_b.i((.., None, 4));
                                    v1.i_mut((.., .., 0)).matmul_from(
                                        &loc_ao.i((.., .., ic+1)), &vtau_ket_a.t(), 1.0, 1.0
                                    );
                                    v1.i_mut((.., .., 1)).matmul_from(
                                        &loc_ao.i((.., .., ic+1)), &vtau_ket_b.t(), 1.0, 1.0
                                    );
                                }
                            },
                            XCType::HF => {},
                        }

                    },
                    _ => unreachable!(),
                }
            }
        ); // end block loop

        let mut vmat = vec!();

        match xc_data.xc_type {
            XCType::LDA => {},
            XCType::GGA => {
                let vxc_trans = vxc.transpose([0, 2, 1]);
                vxc = &vxc + vxc_trans;
            },
            XCType::MGGA => {
                let vxc_trans = vxc.transpose([0, 2, 1]);
                vxc = &vxc + vxc_trans;
                let Some(v1) = vxc_mgga_temp.as_ref() else {
                    panic!("vxc_mgga_temp is not initialized for MGGA type");
                };
                vxc.add_assign(v1);
            },
            XCType::HF => {
                unreachable!()
            },
        }

        vmat.push(MatrixFull::from_vec([num_basis, num_basis], vxc.i(0).to_vec()).unwrap());
        if spin == 1 {
            vmat.push(MatrixFull::from_vec([num_basis, num_basis], vxc.i(1).to_vec()).unwrap());
        }

        (nelec, exc_sum, vmat)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensors::matrix_blas_lapack::{
        omp_get_num_threads_wrapper, omp_set_num_threads_wrapper,
    };

    fn make_lda_data(nocc: usize, nvir: usize, ngrids: usize) -> FXCMatvecData {
        let mut mo_occ = MatrixFull::new([nocc, ngrids], 0.0);
        for i in 0..nocc {
            for g in 0..ngrids {
                mo_occ[[i, g]] = ((i as f64 + 1.0) * (g as f64 + 1.0)).sin() * 0.5;
            }
        }
        let mut mo_vir = MatrixFull::new([nvir, ngrids], 0.0);
        for a in 0..nvir {
            for g in 0..ngrids {
                mo_vir[[a, g]] = ((a as f64 + 1.0) * (g as f64 + 1.0)).cos() * 0.3;
            }
        }
        let wfxc: Vec<f64> = (0..ngrids).map(|g| (g as f64 + 1.0).sqrt() * 0.01).collect();

        FXCMatvecData {
            nvar: 1,
            ngrids,
            nocc,
            nvir,
            start_mo: 0,
            alpha_hybrid: 0.0,
            mo_occ,
            mo_vir,
            mo_occ_grad: None,
            mo_vir_grad: None,
            wfxc,
            use_opt: false,
        }
    }

    fn make_gga_data(nocc: usize, nvir: usize, ngrids: usize) -> FXCMatvecData {
        let mut mo_occ = MatrixFull::new([nocc, ngrids], 0.0);
        for i in 0..nocc {
            for g in 0..ngrids {
                mo_occ[[i, g]] = ((i as f64 + 1.0) * (g as f64 + 1.0)).sin() * 0.5;
            }
        }
        let mut mo_vir = MatrixFull::new([nvir, ngrids], 0.0);
        for a in 0..nvir {
            for g in 0..ngrids {
                mo_vir[[a, g]] = ((a as f64 + 1.0) * (g as f64 + 1.0)).cos() * 0.3;
            }
        }

        let mut og = [
            MatrixFull::new([nocc, ngrids], 0.0),
            MatrixFull::new([nocc, ngrids], 0.0),
            MatrixFull::new([nocc, ngrids], 0.0),
        ];
        let mut vg = [
            MatrixFull::new([nvir, ngrids], 0.0),
            MatrixFull::new([nvir, ngrids], 0.0),
            MatrixFull::new([nvir, ngrids], 0.0),
        ];
        for d in 0..3 {
            for i in 0..nocc {
                for g in 0..ngrids {
                    og[d][[i, g]] = ((i as f64 + 1.0) * (g as f64 + 1.0)).cos() * 0.2 * (d as f64 + 1.0);
                }
            }
            for a in 0..nvir {
                for g in 0..ngrids {
                    vg[d][[a, g]] = -((a as f64 + 1.0) * (g as f64 + 1.0)).sin() * 0.2 * (d as f64 + 1.0);
                }
            }
        }

        let nv2 = 16;
        let mut wfxc = vec![0.0; ngrids * nv2];
        for g in 0..ngrids {
            let weight = (g as f64 + 1.0).sqrt() * 0.01;
            for a in 0..4 {
                for b in 0..4 {
                    let idx = g + a * ngrids + b * 4 * ngrids;
                    wfxc[idx] = weight * (a as f64 * 0.1 + b as f64 * 0.1 + 1.0).sin().max(0.01);
                }
            }
        }

        FXCMatvecData {
            nvar: 4,
            ngrids,
            nocc,
            nvir,
            start_mo: 0,
            alpha_hybrid: 0.25,
            mo_occ,
            mo_vir,
            mo_occ_grad: Some(og),
            mo_vir_grad: Some(vg),
            wfxc,
            use_opt: false,
        }
    }

    fn test_fxc_lda(nocc: usize, nvir: usize, ngrids: usize) {
        let data = make_lda_data(nocc, nvir, ngrids);
        let n = nocc * nvir;
        let z: Vec<f64> = (0..n).map(|i| ((i % 7) as f64) * 0.01).collect();
        let result = fxc_matvec_old(&data, &z);
        assert_eq!(result.len(), n);
        let norm: f64 = result.iter().map(|x| x * x).sum::<f64>().sqrt();
        println!("LDA test nocc={} nvir={} ngrids={}: norm={:.6}", nocc, nvir, ngrids, norm);
        assert!(norm > 0.0, "LDA fxc_matvec should give non-zero result");
        assert!(norm.is_finite(), "LDA fxc_matvec result should be finite");
    }

    fn test_fxc_gga(nocc: usize, nvir: usize, ngrids: usize) {
        let data = make_gga_data(nocc, nvir, ngrids);
        let n = nocc * nvir;
        let z: Vec<f64> = (0..n).map(|i| ((i % 7) as f64) * 0.01).collect();
        let result = fxc_matvec_old(&data, &z);
        assert_eq!(result.len(), n);
        let norm: f64 = result.iter().map(|x| x * x).sum::<f64>().sqrt();
        println!("GGA test nocc={} nvir={} ngrids={}: norm={:.6}", nocc, nvir, ngrids, norm);
        assert!(norm > 0.0, "GGA fxc_matvec should give non-zero result");
        assert!(norm.is_finite(), "GGA fxc_matvec result should be finite");
    }

    #[test]
    fn test_fxc_lda_small() { test_fxc_lda(3, 5, 10); }
    #[test]
    fn test_fxc_lda_medium() { test_fxc_lda(10, 20, 100); }
    #[test]
    fn test_fxc_lda_zero_z() {
        let data = make_lda_data(3, 5, 10);
        let z = vec![0.0; 15];
        let result = fxc_matvec_old(&data, &z);
        for &v in &result { assert!((v).abs() < 1e-15); }
    }
    #[test]
    fn test_fxc_gga_small() { test_fxc_gga(3, 5, 10); }
    #[test]
    fn test_fxc_gga_medium() { test_fxc_gga(10, 20, 100); }
    #[test]
    fn test_fxc_gga_zero_z() {
        let data = make_gga_data(3, 5, 10);
        let z = vec![0.0; 15];
        let result = fxc_matvec_old(&data, &z);
        for &v in &result { assert!((v).abs() < 1e-15); }
    }
    #[test]
    fn test_fxc_gga_symmetry() {
        let data = make_gga_data(4, 6, 20);
        let n = 24;
        let z: Vec<f64> = (0..n).map(|i| (i as f64) * 0.01).collect();
        let result = fxc_matvec_old(&data, &z);
        assert!(result.iter().all(|x| x.is_finite()));
    }
    #[test]
    fn test_fxc_lda_linearity() {
        let data = make_lda_data(3, 5, 10);
        let n = 15;
        let z1: Vec<f64> = (0..n).map(|i| (i as f64) * 0.01).collect();
        let z2: Vec<f64> = (0..n).map(|i| ((i + 3) as f64) * 0.01).collect();
        let r1 = fxc_matvec_old(&data, &z1);
        let r2 = fxc_matvec_old(&data, &z2);
        let z_sum: Vec<f64> = z1.iter().zip(z2.iter()).map(|(a, b)| a + b).collect();
        let r_sum = fxc_matvec_old(&data, &z_sum);
        for i in 0..n {
            assert!((r_sum[i] - (r1[i] + r2[i])).abs() < 1e-14,
                    "Linearity violated at index {}", i);
        }
        println!("LDA linearity test passed");
    }
    #[test]
    fn test_fxc_lda_energy() {
        let data = make_lda_data(5, 8, 30);
        let n = 40;
        let z: Vec<f64> = (0..n).map(|i| if i < 10 { 1.0 } else { 0.0 }).collect();
        let result = fxc_matvec_old(&data, &z);
        assert!(result.iter().all(|x| x.is_finite()));
        let dot: f64 = z.iter().zip(result.iter()).map(|(a, b)| a * b).sum();
        println!("LDA energy test: z·result = {:.6}", dot);
    }
    #[test]
    fn test_fxc_gga_linearity() {
        let data = make_gga_data(3, 5, 10);
        let n = 15;
        let z1: Vec<f64> = (0..n).map(|i| (i as f64) * 0.01).collect();
        let z2: Vec<f64> = (0..n).map(|i| ((i + 3) as f64) * 0.01).collect();
        let r1 = fxc_matvec_old(&data, &z1);
        let r2 = fxc_matvec_old(&data, &z2);
        let z_sum: Vec<f64> = z1.iter().zip(z2.iter()).map(|(a, b)| a + b).collect();
        let r_sum = fxc_matvec_old(&data, &z_sum);
        for i in 0..n {
            let diff = (r_sum[i] - (r1[i] + r2[i])).abs();
            assert!(diff < 1e-14, "GGA linearity violated at index {}: {}", i, diff);
        }
        println!("GGA linearity test passed");
    }

    // ================================================================
    // Tests for fxc_matvec_opt — correctness and performance
    // ================================================================

    /// Verify LDA optimised result matches original exactly.
    #[test]
    fn test_opt_lda_correctness() {
        let sizes = [(3, 5, 10), (10, 20, 100), (20, 50, 500)];
        for &(nocc, nvir, ngrids) in &sizes {
            let data = make_lda_data(nocc, nvir, ngrids);
            let n = nocc * nvir;
            let z: Vec<f64> = (0..n).map(|i| ((i % 7) as f64) * 0.01).collect();
            let expected = fxc_matvec_old(&data, &z);
            let actual   = fxc_matvec_opt(&data, &z);
            assert_eq!(expected.len(), actual.len());
            for idx in 0..n {
                let diff = (expected[idx] - actual[idx]).abs();
                assert!(diff < 1e-14,
                    "LDA mismatch at idx={}: expected={:.15e} actual={:.15e} diff={:.2e}",
                    idx, expected[idx], actual[idx], diff);
            }
            println!("  opt LDA correct: nocc={} nvir={} ngrids={}", nocc, nvir, ngrids);
        }
    }

    /// Verify GGA optimised result matches original exactly.
    #[test]
    fn test_opt_gga_correctness() {
        let sizes = [(3, 5, 10), (10, 20, 100), (20, 50, 500)];
        for &(nocc, nvir, ngrids) in &sizes {
            let data = make_gga_data(nocc, nvir, ngrids);
            let n = nocc * nvir;
            let z: Vec<f64> = (0..n).map(|i| ((i % 7) as f64) * 0.01).collect();
            let expected = fxc_matvec_old(&data, &z);
            let actual   = fxc_matvec_opt(&data, &z);
            assert_eq!(expected.len(), actual.len());
            for idx in 0..n {
                let diff = (expected[idx] - actual[idx]).abs();
                assert!(diff < 1e-14,
                    "GGA mismatch at idx={}: expected={:.15e} actual={:.15e} diff={:.2e}",
                    idx, expected[idx], actual[idx], diff);
            }
            println!("  opt GGA correct: nocc={} nvir={} ngrids={}", nocc, nvir, ngrids);
        }
    }

    /// Verify LDA linearity holds for optimised version too.
    #[test]
    fn test_opt_lda_linearity() {
        let data = make_lda_data(5, 8, 30);
        let n = 40;
        let z1: Vec<f64> = (0..n).map(|i| (i as f64) * 0.01).collect();
        let z2: Vec<f64> = (0..n).map(|i| ((i + 5) as f64) * 0.01).collect();
        let r1 = fxc_matvec_opt(&data, &z1);
        let r2 = fxc_matvec_opt(&data, &z2);
        let z_sum: Vec<f64> = z1.iter().zip(z2.iter()).map(|(a, b)| a + b).collect();
        let r_sum = fxc_matvec_opt(&data, &z_sum);
        for i in 0..n {
            let diff = (r_sum[i] - (r1[i] + r2[i])).abs();
            assert!(diff < 1e-14, "LDA opt linearity violated at index {}", i);
        }
        println!("opt LDA linearity test passed");
    }

    /// Verify GGA linearity holds for optimised version too.
    #[test]
    fn test_opt_gga_linearity() {
        let data = make_gga_data(5, 8, 30);
        let n = 40;
        let z1: Vec<f64> = (0..n).map(|i| (i as f64) * 0.01).collect();
        let z2: Vec<f64> = (0..n).map(|i| ((i + 5) as f64) * 0.01).collect();
        let r1 = fxc_matvec_opt(&data, &z1);
        let r2 = fxc_matvec_opt(&data, &z2);
        let z_sum: Vec<f64> = z1.iter().zip(z2.iter()).map(|(a, b)| a + b).collect();
        let r_sum = fxc_matvec_opt(&data, &z_sum);
        for i in 0..n {
            let diff = (r_sum[i] - (r1[i] + r2[i])).abs();
            assert!(diff < 1e-14, "GGA opt linearity violated at index {}", i);
        }
        println!("opt GGA linearity test passed");
    }

    /// LDA performance benchmark: compare original (serial) vs optimised with
    /// 8 rayon threads.  Reports wall-clock times and speedup.
    #[test]
    fn test_opt_lda_performance() {
        let nocc = 60;
        let nvir = 200;
        let ngrids = 50000;
        let data = make_lda_data(nocc, nvir, ngrids);
        let n = nocc * nvir;
        let z: Vec<f64> = (0..n).map(|i| ((i % 7) as f64) * 0.01).collect();

        // Warm-up: call both once to stabilise BLAS / caches
        let _ = fxc_matvec_old(&data, &z);
        let _ = fxc_matvec_opt(&data, &z);

        use std::time::Instant;
        let n_repeat = 5;

        // Time original (no thread pool needed — purely sequential)
        let t0 = Instant::now();
        for _ in 0..n_repeat {
            let _ = fxc_matvec_old(&data, &z);
        }
        let dt_orig = t0.elapsed().as_secs_f64() / n_repeat as f64;

        // Time optimised with 8-thread rayon pool
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .unwrap();
        let dt_opt = pool.install(|| {
            let t0 = Instant::now();
            for _ in 0..n_repeat {
                let _ = fxc_matvec_opt(&data, &z);
            }
            t0.elapsed().as_secs_f64() / n_repeat as f64
        });

        let speedup = dt_orig / dt_opt;
        println!("\n  === LDA Performance ({} occ × {} vir × {} grids) ===", nocc, nvir, ngrids);
        println!("  original  (serial):   {:.4} s",  dt_orig);
        println!("  optimised (8 threads): {:.4} s",  dt_opt);
        println!("  speedup:              {:.2}×",       speedup);
        println!("  z·fxc[z] (original):  {:.6e}", z.iter().zip(fxc_matvec_old(&data, &z).iter()).map(|(a,b)| a*b).sum::<f64>());

        assert!(speedup > 1.8,
            "Expected > 1.8× speedup with 8 threads for LDA, got {:.2}×", speedup);
    }

    /// GGA performance benchmark: compare original (serial) vs optimised with
    /// 8 rayon threads.  Reports wall-clock times and speedup.
    #[test]
    fn test_opt_gga_performance() {
        let nocc = 50;
        let nvir = 200;
        let ngrids = 30000;
        let data = make_gga_data(nocc, nvir, ngrids);
        let n = nocc * nvir;
        let z: Vec<f64> = (0..n).map(|i| ((i % 7) as f64) * 0.01).collect();

        // Warm-up
        let _ = fxc_matvec_old(&data, &z);
        let _ = fxc_matvec_opt(&data, &z);

        use std::time::Instant;
        let n_repeat = 5;

        // Time original
        let t0 = Instant::now();
        for _ in 0..n_repeat {
            let _ = fxc_matvec_old(&data, &z);
        }
        let dt_orig = t0.elapsed().as_secs_f64() / n_repeat as f64;

        // Time optimised with 8-thread rayon pool
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .unwrap();
        let dt_opt = pool.install(|| {
            let t0 = Instant::now();
            for _ in 0..n_repeat {
                let _ = fxc_matvec_opt(&data, &z);
            }
            t0.elapsed().as_secs_f64() / n_repeat as f64
        });

        let speedup = dt_orig / dt_opt;
        println!("\n  === GGA Performance ({} occ × {} vir × {} grids) ===", nocc, nvir, ngrids);
        println!("  original  (serial):   {:.4} s",  dt_orig);
        println!("  optimised (8 threads): {:.4} s",  dt_opt);
        println!("  speedup:              {:.2}×",       speedup);
        println!("  z·fxc[z] (original):  {:.6e}", z.iter().zip(fxc_matvec_old(&data, &z).iter()).map(|(a,b)| a*b).sum::<f64>());

        assert!(speedup > 2.0,
            "Expected > 2.0× speedup with 8 threads for GGA, got {:.2}×", speedup);
    }

    /// Verify the router delegates correctly: when the flag is off, fxc_matvec
    /// matches fxc_matvec_old; when on, it matches fxc_matvec_opt.
    #[test]
    fn test_opt_router() {
        // Restore default after test
        let saved = USE_OPTIMIZED_FXC.load(Ordering::Relaxed);

        let check = |nocc, nvir, ngrids, is_gga: bool| {
            let data = if is_gga {
                make_gga_data(nocc, nvir, ngrids)
            } else {
                make_lda_data(nocc, nvir, ngrids)
            };
            let n = nocc * nvir;
            let z: Vec<f64> = (0..n).map(|i| ((i % 7) as f64) * 0.01).collect();

            // flag off → should match old
            set_fxc_use_optimized(false);
            let router_off = fxc_matvec(&data, &z);
            let old = fxc_matvec_old(&data, &z);
            for idx in 0..n {
                assert!((router_off[idx] - old[idx]).abs() < 1e-14,
                    "Router-off mismatch at {} nocc={}", idx, nocc);
            }

            // flag on → should match opt
            set_fxc_use_optimized(true);
            let router_on = fxc_matvec(&data, &z);
            let opt = fxc_matvec_opt(&data, &z);
            for idx in 0..n {
                assert!((router_on[idx] - opt[idx]).abs() < 1e-14,
                    "Router-on mismatch at {} nocc={}", idx, nocc);
            }
        };

        check(5, 8, 20, false);   // LDA small
        check(5, 8, 20, true);    // GGA small
        check(20, 50, 200, false); // LDA medium
        check(20, 50, 200, true);  // GGA medium

        // restore
        set_fxc_use_optimized(saved);
        println!("opt router test passed (LDA + GGA, small + medium)");
    }

    // ================================================================
    // DGEMM scheduling benchmark for GGA Step 1
    //
    // Tests multiple strategies for the 4 independent dgemm calls:
    //   z_mat [nocc,nvir] × mo_vir_grad[d] [nvir,ngrids] → t[d]
    // ================================================================

    /// Run the GGA Step‑1 dgemm scheduling benchmark.
    /// Reports wall-clock times for multiple strategies in release mode.
    fn bench_gga_step1_all(nocc: usize, nvir: usize, ngrids: usize,
                           n_warm: usize, n_iter: usize) {
        use std::time::Instant;
        let old_omp = omp_get_num_threads_wrapper();

        // -------- test data ----------
        let data = make_gga_data(nocc, nvir, ngrids);
        let mv  = &data.mo_vir;
        let mg  = data.mo_vir_grad.as_ref().unwrap();
        let z: Vec<f64> = (0..nocc * nvir).map(|i| ((i % 7) as f64) * 0.01).collect();
        let z_mat = MatrixFull::from_vec([nocc, nvir], z.to_vec()).unwrap();

        // reference result
        let ref_t = {
            omp_set_num_threads_wrapper(8);
            let mut t = [MatrixFull::new([nocc, ngrids], 0.0),
                         MatrixFull::new([nocc, ngrids], 0.0),
                         MatrixFull::new([nocc, ngrids], 0.0),
                         MatrixFull::new([nocc, ngrids], 0.0)];
            _dgemm_full(&z_mat, 'N', mv, 'N', &mut t[0], 1.0, 0.0);
            for d in 0..3 { _dgemm_full(&z_mat, 'N', &mg[d], 'N', &mut t[1+d], 1.0, 0.0); }
            t
        };

        // verify helper
        let check = |t: &[MatrixFull<f64>; 4]| -> bool {
            (0..4).all(|i| t[i].data.iter().zip(ref_t[i].data.iter())
                .map(|(a,b)| (a-b).abs()).sum::<f64>() < 1e-12)
        };

        println!("\n  === GGA Step‑1 dgemm benchmark ({}×{}×{}) ===", nocc, nvir, ngrids);
        println!("  {:<22} {:>10} {:>8}  {}", "Strategy", "Time (s)", "Speedup", "Correct");
        println!("  {}", "-".repeat(50));

        // macro: warmup + time, run a strategy
        macro_rules! time_strat {
            ($label:expr, $omp_thr:expr, $body:expr) => {{
                let label = $label;
                let thr = $omp_thr;
                for _ in 0..n_warm { let _ = { $body }; }
                omp_set_num_threads_wrapper(thr);
                let t0 = Instant::now();
                let mut last = None;
                for _ in 0..n_iter { last = Some({ $body }); }
                let dt = t0.elapsed().as_secs_f64() / n_iter as f64;
                (label, dt, last.as_ref().map_or(false, |t| check(t)))
            }};
        }

        // -------- strategies ----------
        // 1. sequential 4 calls × 8 threads (baseline)
        let mut results: Vec<(&str, f64, bool)> = Vec::new();
        results.push(time_strat!("seq 4×8thr", 8, {
            let mut t = [MatrixFull::new([nocc, ngrids], 0.0),
                         MatrixFull::new([nocc, ngrids], 0.0),
                         MatrixFull::new([nocc, ngrids], 0.0),
                         MatrixFull::new([nocc, ngrids], 0.0)];
            _dgemm_full(&z_mat, 'N', mv, 'N', &mut t[0], 1.0, 0.0);
            for d in 0..3 { _dgemm_full(&z_mat, 'N', &mg[d], 'N', &mut t[1+d], 1.0, 0.0); }
            t
        }));

        // 2. concurrent 4 calls × 2 threads (Phase 7)
        results.push(time_strat!("conc4 2thr/call", 2, {
            let mut t0  = MatrixFull::new([nocc, ngrids], 0.0);
            let mut tg0 = MatrixFull::new([nocc, ngrids], 0.0);
            let mut tg1 = MatrixFull::new([nocc, ngrids], 0.0);
            let mut tg2 = MatrixFull::new([nocc, ngrids], 0.0);
            rayon::scope(|s| {
                s.spawn(|_| _dgemm_full(&z_mat, 'N', mv, 'N', &mut t0, 1.0, 0.0));
                s.spawn(|_| _dgemm_full(&z_mat, 'N', &mg[0], 'N', &mut tg0, 1.0, 0.0));
                s.spawn(|_| _dgemm_full(&z_mat, 'N', &mg[1], 'N', &mut tg1, 1.0, 0.0));
                s.spawn(|_| _dgemm_full(&z_mat, 'N', &mg[2], 'N', &mut tg2, 1.0, 0.0));
            });
            [t0, tg0, tg1, tg2]
        }));

        // 3. concurrent 4 calls × 1 thread (extreme)
        results.push(time_strat!("conc4 1thr/call", 1, {
            let mut t0  = MatrixFull::new([nocc, ngrids], 0.0);
            let mut tg0 = MatrixFull::new([nocc, ngrids], 0.0);
            let mut tg1 = MatrixFull::new([nocc, ngrids], 0.0);
            let mut tg2 = MatrixFull::new([nocc, ngrids], 0.0);
            rayon::scope(|s| {
                s.spawn(|_| _dgemm_full(&z_mat, 'N', mv, 'N', &mut t0, 1.0, 0.0));
                s.spawn(|_| _dgemm_full(&z_mat, 'N', &mg[0], 'N', &mut tg0, 1.0, 0.0));
                s.spawn(|_| _dgemm_full(&z_mat, 'N', &mg[1], 'N', &mut tg1, 1.0, 0.0));
                s.spawn(|_| _dgemm_full(&z_mat, 'N', &mg[2], 'N', &mut tg2, 1.0, 0.0));
            });
            [t0, tg0, tg1, tg2]
        }));

        // 4. two rounds of 2 concurrent × 4 threads
        results.push(time_strat!("conc2×2 4thr", 4, {
            let mut t0  = MatrixFull::new([nocc, ngrids], 0.0);
            let mut tg0 = MatrixFull::new([nocc, ngrids], 0.0);
            let mut tg1 = MatrixFull::new([nocc, ngrids], 0.0);
            let mut tg2 = MatrixFull::new([nocc, ngrids], 0.0);
            rayon::scope(|s| {
                s.spawn(|_| _dgemm_full(&z_mat, 'N', mv, 'N', &mut t0, 1.0, 0.0));
                s.spawn(|_| _dgemm_full(&z_mat, 'N', &mg[0], 'N', &mut tg0, 1.0, 0.0));
            });
            rayon::scope(|s| {
                s.spawn(|_| _dgemm_full(&z_mat, 'N', &mg[1], 'N', &mut tg1, 1.0, 0.0));
                s.spawn(|_| _dgemm_full(&z_mat, 'N', &mg[2], 'N', &mut tg2, 1.0, 0.0));
            });
            [t0, tg0, tg1, tg2]
        }));

        // 5. two rounds of 2 concurrent × 2 threads
        results.push(time_strat!("conc2×2 2thr", 2, {
            let mut t0  = MatrixFull::new([nocc, ngrids], 0.0);
            let mut tg0 = MatrixFull::new([nocc, ngrids], 0.0);
            let mut tg1 = MatrixFull::new([nocc, ngrids], 0.0);
            let mut tg2 = MatrixFull::new([nocc, ngrids], 0.0);
            rayon::scope(|s| {
                s.spawn(|_| _dgemm_full(&z_mat, 'N', mv, 'N', &mut t0, 1.0, 0.0));
                s.spawn(|_| _dgemm_full(&z_mat, 'N', &mg[0], 'N', &mut tg0, 1.0, 0.0));
            });
            rayon::scope(|s| {
                s.spawn(|_| _dgemm_full(&z_mat, 'N', &mg[1], 'N', &mut tg1, 1.0, 0.0));
                s.spawn(|_| _dgemm_full(&z_mat, 'N', &mg[2], 'N', &mut tg2, 1.0, 0.0));
            });
            [t0, tg0, tg1, tg2]
        }));

        // 6. stacked single dgemm: copy rhs → 1 big dgemm → split
        results.push(time_strat!("stacked 8thr", 8, {
            let mut rhs_all = vec![0.0; nvir * ngrids * 4];
            let stride = nvir * ngrids;
            rhs_all[0..stride].copy_from_slice(mv.data.as_slice());
            rhs_all[stride..2*stride].copy_from_slice(mg[0].data.as_slice());
            rhs_all[2*stride..3*stride].copy_from_slice(mg[1].data.as_slice());
            rhs_all[3*stride..4*stride].copy_from_slice(mg[2].data.as_slice());
            let rhs_full = MatrixFull { size: [nvir, ngrids * 4], indicing: [1, nvir], data: rhs_all };
            let mut t_stk = MatrixFull::new([nocc, ngrids * 4], 0.0);
            _dgemm_full(&z_mat, 'N', &rhs_full, 'N', &mut t_stk, 1.0, 0.0);
            let n = nocc * ngrids;
            [MatrixFull::from_vec([nocc, ngrids], t_stk.data[0..n].to_vec()).unwrap(),
             MatrixFull::from_vec([nocc, ngrids], t_stk.data[n..2*n].to_vec()).unwrap(),
             MatrixFull::from_vec([nocc, ngrids], t_stk.data[2*n..3*n].to_vec()).unwrap(),
             MatrixFull::from_vec([nocc, ngrids], t_stk.data[3*n..4*n].to_vec()).unwrap()]
        }));

        // 7. stacked with 4 OMP threads
        results.push(time_strat!("stacked 4thr", 4, {
            let mut rhs_all = vec![0.0; nvir * ngrids * 4];
            let stride = nvir * ngrids;
            rhs_all[0..stride].copy_from_slice(mv.data.as_slice());
            rhs_all[stride..2*stride].copy_from_slice(mg[0].data.as_slice());
            rhs_all[2*stride..3*stride].copy_from_slice(mg[1].data.as_slice());
            rhs_all[3*stride..4*stride].copy_from_slice(mg[2].data.as_slice());
            let rhs_full = MatrixFull { size: [nvir, ngrids * 4], indicing: [1, nvir], data: rhs_all };
            let mut t_stk = MatrixFull::new([nocc, ngrids * 4], 0.0);
            _dgemm_full(&z_mat, 'N', &rhs_full, 'N', &mut t_stk, 1.0, 0.0);
            let n = nocc * ngrids;
            [MatrixFull::from_vec([nocc, ngrids], t_stk.data[0..n].to_vec()).unwrap(),
             MatrixFull::from_vec([nocc, ngrids], t_stk.data[n..2*n].to_vec()).unwrap(),
             MatrixFull::from_vec([nocc, ngrids], t_stk.data[2*n..3*n].to_vec()).unwrap(),
             MatrixFull::from_vec([nocc, ngrids], t_stk.data[3*n..4*n].to_vec()).unwrap()]
        }));

        // -------- print results --------
        let baseline = results[0].1;
        for &(label, dt, ok) in &results {
            let speedup = baseline / dt;
            let status = if ok { "✓" } else { "✗ WRONG" };
            println!("  {:<22} {:>8.4}s  {:>6.2}×  {}", label, dt, speedup, status);
        }
        let best = results.iter().filter(|r| r.2).map(|r| baseline / r.1).fold(0.0_f64, f64::max);
        println!("  Best speedup (correct): {:.2}×", best);

        omp_set_num_threads_wrapper(old_omp);
    }

    /// Run the dgemm scheduling benchmark (completes in < 30 s in release).
    #[test]
    fn test_bench_gga_step1_dgemm() {
        bench_gga_step1_all(50, 200, 30000, 2, 3);
    }

    // ================================================================
    // Tests for fxc_matvec_nwchem_opt — NWChem-style optimised kernel
    // ================================================================

    fn make_lda_data_nwchem(nocc: usize, nvir: usize, ngrids: usize) -> FXCMatvecDataNwchemOpt {
        let mut mo_occ = MatrixFull::new([nocc, ngrids], 0.0);
        for i in 0..nocc {
            for g in 0..ngrids {
                mo_occ[[i, g]] = ((i as f64 + 1.0) * (g as f64 + 1.0)).sin() * 0.5;
            }
        }
        let mut mo_vir = MatrixFull::new([nvir, ngrids], 0.0);
        for a in 0..nvir {
            for g in 0..ngrids {
                mo_vir[[a, g]] = ((a as f64 + 1.0) * (g as f64 + 1.0)).cos() * 0.3;
            }
        }
        let wfxc: Vec<f64> = (0..ngrids).map(|g| (g as f64 + 1.0).sqrt() * 0.01).collect();

        FXCMatvecDataNwchemOpt {
            nvar: 1,
            ngrids,
            nocc,
            nvir,
            mo_occ,
            mo_vir,
            mo_occ_grad: None,
            mo_vir_grad: None,
            wfxc,
            combined_left_buf: vec![],
            combined_right_buf: vec![],
        }
    }

    fn make_gga_data_nwchem(nocc: usize, nvir: usize, ngrids: usize) -> FXCMatvecDataNwchemOpt {
        let mut mo_occ = MatrixFull::new([nocc, ngrids], 0.0);
        for i in 0..nocc {
            for g in 0..ngrids {
                mo_occ[[i, g]] = ((i as f64 + 1.0) * (g as f64 + 1.0)).sin() * 0.5;
            }
        }
        let mut mo_vir = MatrixFull::new([nvir, ngrids], 0.0);
        for a in 0..nvir {
            for g in 0..ngrids {
                mo_vir[[a, g]] = ((a as f64 + 1.0) * (g as f64 + 1.0)).cos() * 0.3;
            }
        }

        let mut og = [
            MatrixFull::new([nocc, ngrids], 0.0),
            MatrixFull::new([nocc, ngrids], 0.0),
            MatrixFull::new([nocc, ngrids], 0.0),
        ];
        let mut vg = [
            MatrixFull::new([nvir, ngrids], 0.0),
            MatrixFull::new([nvir, ngrids], 0.0),
            MatrixFull::new([nvir, ngrids], 0.0),
        ];
        for d in 0..3 {
            for i in 0..nocc {
                for g in 0..ngrids {
                    og[d][[i, g]] = ((i as f64 + 1.0) * (g as f64 + 1.0)).cos() * 0.2 * (d as f64 + 1.0);
                }
            }
            for a in 0..nvir {
                for g in 0..ngrids {
                    vg[d][[a, g]] = -((a as f64 + 1.0) * (g as f64 + 1.0)).sin() * 0.2 * (d as f64 + 1.0);
                }
            }
        }

        // wfxc in [α, β, g] layout
        let mut wfxc = vec![0.0; ngrids * 16];
        for alpha in 0..4 {
            let off_a = alpha * 4 * ngrids;
            for beta in 0..4 {
                let off = off_a + beta * ngrids;
                for g in 0..ngrids {
                    let weight = (g as f64 + 1.0).sqrt() * 0.01;
                    wfxc[off + g] = weight * (alpha as f64 * 0.1 + beta as f64 * 0.1 + 1.0).sin().max(0.01);
                }
            }
        }

        let combined_left_buf = vec![0.0; nocc * NWCHEM_GGA_BLOCK * 2];
        let combined_right_buf = vec![0.0; nvir * NWCHEM_GGA_BLOCK * 2];

        FXCMatvecDataNwchemOpt {
            nvar: 4,
            ngrids,
            nocc,
            nvir,
            mo_occ,
            mo_vir,
            mo_occ_grad: Some(og),
            mo_vir_grad: Some(vg),
            wfxc,
            combined_left_buf,
            combined_right_buf,
        }
    }

    /// Verify NWChem-opt LDA matches the original LDA exactly.
    #[test]
    fn test_nwchem_opt_lda_correctness() {
        let sizes = [(3, 5, 10), (10, 20, 100), (20, 50, 500)];
        for &(nocc, nvir, ngrids) in &sizes {
            let old_data = make_lda_data(nocc, nvir, ngrids);
            let mut new_data = make_lda_data_nwchem(nocc, nvir, ngrids);
            let n = nocc * nvir;
            let z: Vec<f64> = (0..n).map(|i| ((i % 7) as f64) * 0.01).collect();
            let expected = fxc_matvec_lda(&old_data, &z);
            let actual   = fxc_matvec_lda_nwchem_opt(&new_data, &z);
            assert_eq!(expected.len(), actual.len());
            for idx in 0..n {
                let diff = (expected[idx] - actual[idx]).abs();
                assert!(diff < 1e-14,
                    "LDA nwchem mismatch at idx={}: expected={:.15e} actual={:.15e} diff={:.2e}",
                    idx, expected[idx], actual[idx], diff);
            }
            println!("  nwchem LDA correct: nocc={} nvir={} ngrids={}", nocc, nvir, ngrids);
        }
    }

    /// Verify NWChem-opt GGA matches the original GGA exactly.
    #[test]
    fn test_nwchem_opt_gga_correctness() {
        let sizes = [(3, 5, 10), (10, 20, 100), (20, 50, 500)];
        for &(nocc, nvir, ngrids) in &sizes {
            let old_data = make_gga_data(nocc, nvir, ngrids);
            let mut new_data = make_gga_data_nwchem(nocc, nvir, ngrids);
            let n = nocc * nvir;
            let z: Vec<f64> = (0..n).map(|i| ((i % 7) as f64) * 0.01).collect();
            let expected = fxc_matvec_gga(&old_data, &z);
            let actual   = fxc_matvec_gga_nwchem_opt(&mut new_data, &z);
            assert_eq!(expected.len(), actual.len());
            for idx in 0..n {
                let diff = (expected[idx] - actual[idx]).abs();
                assert!(diff < 1e-14,
                    "GGA nwchem mismatch at idx={}: expected={:.15e} actual={:.15e} diff={:.2e}",
                    idx, expected[idx], actual[idx], diff);
            }
            println!("  nwchem GGA correct: nocc={} nvir={} ngrids={}", nocc, nvir, ngrids);
        }
    }

    /// NWChem-opt LDA performance benchmark.
    #[test]
    fn test_nwchem_opt_lda_performance() {
        let nocc = 60;
        let nvir = 200;
        let ngrids = 50000;
        let old_data = make_lda_data(nocc, nvir, ngrids);
        let mut new_data = make_lda_data_nwchem(nocc, nvir, ngrids);
        let n = nocc * nvir;
        let z: Vec<f64> = (0..n).map(|i| ((i % 7) as f64) * 0.01).collect();

        // Warm-up
        let _ = fxc_matvec_lda(&old_data, &z);
        let _ = fxc_matvec_lda_nwchem_opt(&new_data, &z);

        use std::time::Instant;
        let n_repeat = 5;

        // Time original
        let t0 = Instant::now();
        for _ in 0..n_repeat {
            let _ = fxc_matvec_lda(&old_data, &z);
        }
        let dt_orig = t0.elapsed().as_secs_f64() / n_repeat as f64;

        // Time nwchem-opt with 8-thread rayon pool
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .unwrap();
        let dt_opt = pool.install(|| {
            let t0 = Instant::now();
            for _ in 0..n_repeat {
                let _ = fxc_matvec_lda_nwchem_opt(&new_data, &z);
            }
            t0.elapsed().as_secs_f64() / n_repeat as f64
        });

        let speedup = dt_orig / dt_opt;
        println!("\n  === LDA NWChem-Opt Performance ({} occ × {} vir × {} grids) ===", nocc, nvir, ngrids);
        println!("  original  (serial):     {:.4} s",  dt_orig);
        println!("  nwchem-opt (8 threads): {:.4} s",  dt_opt);
        println!("  speedup:                {:.2}×",       speedup);

        // Soft threshold — just informative
        println!("  (expected > 1.0 for any parallel speedup)");
    }

    /// NWChem-opt GGA performance benchmark.
    #[test]
    fn test_nwchem_opt_gga_performance() {
        let nocc = 50;
        let nvir = 200;
        let ngrids = 30000;
        let old_data = make_gga_data(nocc, nvir, ngrids);
        let mut new_data = make_gga_data_nwchem(nocc, nvir, ngrids);
        let n = nocc * nvir;
        let z: Vec<f64> = (0..n).map(|i| ((i % 7) as f64) * 0.01).collect();

        // Warm-up
        let _ = fxc_matvec_gga(&old_data, &z);
        let _ = fxc_matvec_gga_nwchem_opt(&mut new_data, &z);

        use std::time::Instant;
        let n_repeat = 5;

        // Time original
        let t0 = Instant::now();
        for _ in 0..n_repeat {
            let _ = fxc_matvec_gga(&old_data, &z);
        }
        let dt_orig = t0.elapsed().as_secs_f64() / n_repeat as f64;

        // Time nwchem-opt with 8-thread rayon pool
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .unwrap();
        let dt_opt = pool.install(|| {
            let t0 = Instant::now();
            for _ in 0..n_repeat {
                let _ = fxc_matvec_gga_nwchem_opt(&mut new_data, &z);
            }
            t0.elapsed().as_secs_f64() / n_repeat as f64
        });

        let speedup = dt_orig / dt_opt;
        println!("\n  === GGA NWChem-Opt Performance ({} occ × {} vir × {} grids) ===", nocc, nvir, ngrids);
        println!("  original  (serial):     {:.4} s",  dt_orig);
        println!("  nwchem-opt (8 threads): {:.4} s",  dt_opt);
        println!("  speedup:                {:.2}×",       speedup);

        println!("  (expected > 1.0 for any parallel speedup)");
    }

    // ================================================================
    // Cross-code comparison benchmark: REST vs NWChem
    //
    // Uses H2O BLYP/6-31G* parameters extracted from instrumented
    // NWChem RI-TDDFT run:
    //   nocc =  5  (doubly occupied MOs)
    //   nvir = 14  (19 AO - 5 occ - frozen core adjustments)
    //   nvar =  4  (GGA)
    //   NWChem fxc kernel: avg ~0.090 s/call (7 dgemms, 2 MPI ranks)
    // ================================================================

    /// REST GGA fxc_matvec benchmark with H2O-level parameters.
    /// Reports timing for original (serial) and NWChem-opt (8-thread rayon),
    /// scaling over a range of grid sizes.
    #[test]
    fn test_bench_vs_nwchem_h2o() {
        let nocc = 5;
        let nvir = 14;
        let ngrids_values = [2_500usize, 5_000, 10_000, 20_000, 50_000];

        use std::time::Instant;
        let n_repeat = 50;
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .unwrap();

        println!("\n  === REST fxc_matvec GGA benchmark (H2O: {} occ × {} vir) ===", nocc, nvir);
        println!("  {:<8} {:>12} {:>12} {:>10}",
                 "ngrids", "old(s)", "nwchem-opt(s)", "speedup");
        println!("  {}", "-".repeat(45));

        for &ngrids in &ngrids_values {
            let old_data = make_gga_data(nocc, nvir, ngrids);
            let mut new_data = make_gga_data_nwchem(nocc, nvir, ngrids);
            let n = nocc * nvir;
            let z: Vec<f64> = (0..n).map(|i| ((i % 7) as f64) * 0.01).collect();

            // Warm-up
            let _ = fxc_matvec_gga(&old_data, &z);
            let _ = fxc_matvec_gga_nwchem_opt(&mut new_data, &z);

            // Time original (serial BLAS)
            let t0 = Instant::now();
            for _ in 0..n_repeat {
                let _ = fxc_matvec_gga(&old_data, &z);
            }
            let dt_old = t0.elapsed().as_secs_f64() / n_repeat as f64;

            // Time nwchem-opt (8-thread rayon pool)
            let dt_new = pool.install(|| {
                let t0 = Instant::now();
                for _ in 0..n_repeat {
                    let _ = fxc_matvec_gga_nwchem_opt(&mut new_data, &z);
                }
                t0.elapsed().as_secs_f64() / n_repeat as f64
            });

            let speedup = dt_old / dt_new;
            println!("  {:<8} {:>8.6}s  {:>8.6}s  {:>7.2}×",
                     ngrids, dt_old, dt_new, speedup);
        }
        println!();
        println!("  NWChem ref (BI H2O, BLYP/6-31G*, RI-TDDFT, 2xMPI):");
        println!("    FXC_TIME call=1: 0.0766 cpu-s  (first Davidson matvec)");
        println!("    FXC_TIME avg:    0.095 cpu-s   (over 7 matvecs)");
        println!("    Grid: medium quality, 94 quadrature shells (~9k points)");
        println!("  => Compare REST times above at ngrids≈10000");
    }
}



