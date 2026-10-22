use crate::molecule_io;
use crate::geom_io;
use crate::dft;
use rayon::prelude::IndexedParallelIterator;
use rayon::prelude::IntoParallelRefIterator;
use rayon::prelude::IntoParallelRefMutIterator;
use rayon::prelude::ParallelIterator;
use rest_tensors::{MatrixFull, matrix_blas_lapack::_einsum_01_rayon};
use tensors::matrix_blas_lapack::_dgemm_full;
use tensors::BasicMatrix;
use crate::constants::ARR;

    
pub fn mid_zeff(z: f64, r: f64, arr: &[[f64;1501];119]) -> f64 {
    if r <= 0.0 {
        return 0.0;
    }
    let arr0 = &arr[0];
    let arrz = &arr[z as usize];
    if r > 40.0 {
        // outside the tabulated range, same as the original scan
        return 0.0;
    }
    // `arr0` is a sorted (non-uniform, log-like) radial grid. Binary search for
    // the bracketing interval: partition_point returns the first index with
    // arr0[i] > r, so the interval start is idx-1. The original linear scan
    // walked up to 1500 entries per (grid point x atom); this is log2(1501)~11.
    let idx = arr0.partition_point(|&x| x <= r);
    let i = if idx == 0 { 0 } else { (idx - 1).min(1499) };
    let zeff_r_a = arrz[i] + (arrz[i + 1] - arrz[i]) / (arr0[i + 1] - arr0[i]) * (r - arr0[i]);
    zeff_r_a / r
}   

#[derive(Clone,Debug,PartialEq)]
pub struct SimpleAtomInfo<'a> {
    pub position_a: &'a [f64],
    pub z_a: f64,
}

pub fn get_vsap(mol: &molecule_io::Molecule, grids: &dft::Grids, print_level: usize) -> MatrixFull<f64> {
    let dt1 = time::Local::now();
    let num_grids = grids.weights.len();

    // Compute the VSAP effective potential on each grid point
    // This step is independent of the AO matrix representation (dense or compressed)
    let atom_info = &mol.geom;
    let atom_pos = &atom_info.position;
    let num_atoms = atom_info.elem.len();
    let atom_mass_charge = geom_io::get_mass_charge(&atom_info.elem).clone();

    let init_sai = SimpleAtomInfo { position_a: &[0.0,0.0,0.0], z_a: 0.0 };
    let mut atoms_info = vec![init_sai;num_atoms];
    atoms_info.iter_mut().zip(MatrixFull::iter_columns_full(atom_pos).zip(&atom_mass_charge)).for_each(|(a,(xyz, (mass,charge)))|{
        let charge_local = charge.clone();
        *a = SimpleAtomInfo{
            position_a: xyz,
            z_a: charge_local,
        };
    });

    let mut v_diag_grid = vec![0.0;num_grids];
    v_diag_grid.par_iter_mut().zip(grids.coordinates.par_iter()).zip(grids.weights.par_iter())
        .map(|((v_grid,c),w)| {(v_grid,c,w)}).for_each(|(v_grid,c,w)| {
        *v_grid = 0.0;
        atoms_info.iter().for_each(|ai| {
            // squared distance by multiplication (powf(2.0) is an expensive libm
            // call on the per-(grid point x atom) hot path)
            let r2 = ai.position_a.iter().zip(c.iter())
                .fold(0.0, |acc, (ac, gc)| { let d = ac - gc; acc + d * d });
            *v_grid += mid_zeff(ai.z_a, r2.sqrt(), &ARR);
        });
        *v_grid *= - w;
    });

    // Contract the potential with AO basis: choose dense or compressed path
    if let Some(a_matrix) = &grids.ao {
        // Dense AO path
        let mut int_mat = _einsum_01_rayon(&a_matrix.to_matrixfullslice(),&v_diag_grid);
        let mut f_mat = MatrixFull::new([a_matrix.size()[0],int_mat.size[0]],0.0);
        _dgemm_full(a_matrix, 'N', &int_mat, 'T', &mut f_mat, 1.0, 0.0);

        let dt2 = time::Local::now();
        let timecost = (dt2.timestamp_millis()-dt1.timestamp_millis()) as f64 /1000.0;
        if print_level>0 {println!("The evaluation of vmat costs {:16.2} seconds",timecost)};

        return f_mat
    } else if let Some(compressed) = &grids.ao_compressed {
        // Compressed (non0tab) AO path: process batch-by-batch
        let num_basis = compressed.nao_total;
        let mut f_mat = MatrixFull::new([num_basis, num_basis], 0.0);

        for ibatch in 0..compressed.batches.len() {
            let batch_ao = &compressed.batches[ibatch];
            let indices = &compressed.batch_ao_map[ibatch];
            let g_range = &compressed.batch_grid_ranges[ibatch];
            let n_active = indices.len();
            if n_active == 0 { continue; }

            let v_batch = &v_diag_grid[g_range.clone()];

            // AO-weighted potential for this batch: int_batch[lμ, g] = AO[lμ, g] * V[g]
            let int_batch = _einsum_01_rayon(&batch_ao.to_matrixfullslice(), v_batch);

            // Batch-local Fock matrix contribution: f_batch = AO_batch * int_batch^T
            let mut f_batch = MatrixFull::new([n_active, n_active], 0.0);
            _dgemm_full(batch_ao, 'N', &int_batch, 'T', &mut f_batch, 1.0, 0.0);

            // Scatter the batch contribution into the global Fock matrix.
            // Raw slices: the Index<[usize;2]> chain costs 4+ calls and 2 bounds
            // checks per element; measured -69% on the equivalent XC scatter.
            {
                let stride = num_basis;
                let dst: &mut [f64] = &mut f_mat.data[..];
                let src: &[f64] = &f_batch.data[..];
                for (l_nu, &g_nu) in indices.iter().enumerate() {
                    let dst_col = g_nu * stride;
                    let src_col = l_nu * n_active;
                    for (l_mu, &g_mu) in indices.iter().enumerate() {
                        dst[g_mu + dst_col] += src[l_mu + src_col];
                    }
                }
            }
        }

        let dt2 = time::Local::now();
        let timecost = (dt2.timestamp_millis()-dt1.timestamp_millis()) as f64 /1000.0;
        if print_level>0 {println!("The evaluation of vmat (compressed) costs {:16.2} seconds",timecost)};

        return f_mat
    } else {
        panic!("VSAP initial guess requires tabulated AO on grids, but neither dense ao nor ao_compressed is available.");
    }

}