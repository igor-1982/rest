use std::borrow::BorrowMut;

use crate::{molecule_io,geom_io,scf_io};
use crate::scf_io::SCF;
use rest_tensors::{BasicMatrix, MathMatrix, MatrixFull};

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

    let dm0 = &scf_data.density_matrix[0];
    let dm1 = &scf_data.density_matrix[1];
    let mut dm = dm0.clone();
    if mol.ctrl.spin_polarization{
        dm += dm1.clone();
        
    };
    
    let mut s = scf_data.ovlp.to_matrixfull().unwrap();
    // The ovlp matrix is symmetric
    let s_t = &s;
    m.data.iter_mut().zip(s_t.data.iter()).zip(dm.data.iter()).for_each(|((x,s),d)|{
        *x = *s * *d
    });
    

    let mut pop = vec![0.0; nao];
    pop.iter_mut().zip(m.iter_columns_full()).for_each(|(charge, column)|{
        column.iter().for_each(|x|{
            *charge += *x 
        })
    });

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