use crate::basis_io::Basis4Elem;
use crate::{geom_io};
use crate::scf_io::SCF;
use rest_tensors::{MatrixFull};
use tensors::matrix_blas_lapack::_dgemm_full;

fn ao_population(dm: &MatrixFull<f64>, s: &MatrixFull<f64>, nao: usize) -> Vec<f64> {
    let mut m = MatrixFull::from_vec([nao, nao], vec![0.0; nao * nao]).unwrap();
    _dgemm_full(dm, 'N', s, 'T', &mut m, 1.0, 0.0);
    m.iter_diagonal().unwrap().map(|x| *x).collect::<Vec<f64>>()
}

fn atom_population(pop: &[f64], basis4elem: &[Basis4Elem]) -> Vec<f64> {
    let mut atom_pop = vec![0.0; basis4elem.len()];
    atom_pop.iter_mut().zip(basis4elem.iter()).for_each(|(ap, b4e)| {
        for i in b4e.global_index.0..(b4e.global_index.0 + b4e.global_index.1) {
            *ap += pop[i];
        }
    });
    atom_pop
}

/// AO-level gross Mulliken population of a set of density matrices (summed over the set)
pub fn mulliken_pop_ao(scf_data: &SCF, dm_list: &[MatrixFull<f64>]) -> Vec<f64> {
    let mol = &scf_data.mol;
    let nao = mol.num_basis;
    let s = scf_data.ovlp.to_matrixfull().unwrap();
    let mut dm = dm_list[0].clone();
    for d in dm_list.iter().skip(1) {
        dm += d.clone();
    }
    ao_population(&dm, &s, nao)
}

/// atom-resolved gross Mulliken population of a set of density matrices (summed over the set)
pub fn mulliken_pop_dm(scf_data: &SCF, dm_list: &[MatrixFull<f64>]) -> Vec<f64> {
    let pop = mulliken_pop_ao(scf_data, dm_list);
    atom_population(&pop, &scf_data.mol.basis4elem)
}

pub fn mulliken_pop(scf_data: &SCF) -> Vec<f64> {
    let mol = &scf_data.mol;

    // consider the mulliken population among atoms and ghost atoms
    let atm_tot = mol.geom.elem.clone().into_iter().chain(mol.geom.ghost_bs_elem.clone().into_iter()).collect::<Vec<_>>();

    let atom_mass_charge = geom_io::get_mass_charge(&mol.geom.elem).clone();
    let mut charge = atom_mass_charge.iter().map(|x| x.1).collect::<Vec<f64>>();

    charge.extend(vec![0.0; mol.geom.ghost_bs_elem.len()]);

    let dm_list: Vec<MatrixFull<f64>> = if mol.spin_channel == 2 {
        vec![scf_data.density_matrix[0].clone(), scf_data.density_matrix[1].clone()]
    } else {
        vec![scf_data.density_matrix[0].clone()]
    };

    let pop = mulliken_pop_ao(scf_data, &dm_list);

    if scf_data.mol.ctrl.print_level > 2 {
        let basis_info = &scf_data.mol.fdqc_bas;
        println!("Mulliken population:");
        pop.iter().zip(basis_info).for_each(|(x, bi)| {
            println!("{:2}-{}-{:10}{:14.5}", bi.elem_index0, atm_tot[bi.elem_index0], bi.bas_name, x);
        });
    }

    let mut mulliken = atom_population(&pop, &mol.basis4elem).iter().zip(charge.iter()).map(|(sum, cha)| cha - sum).collect::<Vec<f64>>();

    // check if ecp is used
    let ghost_atom_index = scf_data.mol.geom.get_start_index_of_ghost_atoms();
    mulliken.iter_mut().enumerate().zip(scf_data.mol.basis4elem.iter()).for_each(|((atm_indx, mcharge), b4e)| {
        if atm_indx < ghost_atom_index {
            if let Some(i_ecp) = b4e.ecp_electrons { *mcharge -= i_ecp as f64; }
        }
    });

    mulliken
}

pub fn mulliken_spin_pop(scf_data: &SCF) -> Vec<f64> {
    let mol = &scf_data.mol;
    if mol.spin_channel != 2 {
        return vec![0.0; mol.basis4elem.len()];
    }
    let pop_a = mulliken_pop_dm(scf_data, &[scf_data.density_matrix[0].clone()]);
    let pop_b = mulliken_pop_dm(scf_data, &[scf_data.density_matrix[1].clone()]);
    pop_a.iter().zip(pop_b.iter()).map(|(a, b)| a - b).collect::<Vec<f64>>()
}
