use std::borrow::BorrowMut;

use crate::{molecule_io,geom_io,scf_io};
use crate::scf_io::SCF;
use rest_tensors::{BasicMatrix, MathMatrix, MatrixFull};
use tensors::matrix_blas_lapack::_dgemm_full;

pub fn mulliken_pop(scf_data: &SCF) -> Vec<f64>{
    let mol = &scf_data.mol;

    // consider the mulliken population among atoms and ghost atoms
    let atm_tot = mol.geom.elem.clone().into_iter().chain(mol.geom.ghost_bs_elem.clone().into_iter()).collect::<Vec<_>>();
    
    let num_atoms = atm_tot.len();
    let nao = mol.num_basis;

    let mut m = MatrixFull::from_vec([nao,nao], vec![0.0; nao*nao]).unwrap();

    let atom_mass_charge = geom_io::get_mass_charge(&mol.geom.elem).clone();
    let mut charge = atom_mass_charge.iter().map(|x| x.1).collect::<Vec<f64>>();

    charge.extend(vec![0.0; mol.geom.ghost_bs_elem.len()]);

    let mut dm = scf_data.density_matrix[0].clone();
    if mol.ctrl.spin_polarization {
        dm += scf_data.density_matrix[1].clone();
    };
    
    let mut s = scf_data.ovlp.to_matrixfull().unwrap();

    //dm.formated_output(19, "full");
    //// The ovlp matrix is symmetric
    //let s_t = &s;
    //m.data.iter_mut().zip(s_t.data.iter()).zip(dm.data.iter()).for_each(|((x,s),d)|{
    //    *x = *s * *d
    //});
    _dgemm_full(&dm, 'N', &s, 'T', &mut m, 1.0, 0.0);
    

    //let mut pop = vec![0.0; nao];
    //pop.iter_mut().zip(m.iter_columns_full()).for_each(|(charge, column)|{
    //    column.iter().for_each(|x|{
    //        *charge += *x 
    //    })
    //});
    let pop = m.iter_diagonal().unwrap().map(|x| *x).collect::<Vec<f64>>();

    if scf_data.mol.ctrl.print_level>1 {
        let basis_info = &scf_data.mol.fdqc_bas;
        println!("Mulliken population:");
        pop.iter().zip(basis_info).for_each(|(x, bi)|{
            println!("{:2}-{}-{:10}{:14.5}", bi.elem_index0, atm_tot[bi.elem_index0], bi.bas_name, x);
        });
        
    }


    let basis = &mol.basis4elem;
    
    let mut mulliken = vec![0.0; num_atoms];
    basis.iter().zip(charge.iter()).zip(mulliken.iter_mut()).for_each(|(((b4e,cha),mull))|{
        let mut sum = 0.0;
        for i in b4e.global_index.0..(b4e.global_index.0 + b4e.global_index.1){
            sum += pop[i];
        }
        *mull = *cha - sum;
    });

    // check if ecp is used
    mulliken.iter_mut().zip(scf_data.mol.basis4elem.iter()).for_each(|(mcharge,b4e)| {
        if let Some(i_ecp) = b4e.ecp_electrons {*mcharge -= i_ecp as f64;}
    });
    
    mulliken
}