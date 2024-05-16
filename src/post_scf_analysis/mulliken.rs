use std::borrow::BorrowMut;

use crate::{molecule_io,geom_io,scf_io};
use crate::scf_io::SCF;
use rest_tensors::{BasicMatrix, MathMatrix, MatrixFull};

pub fn mulliken_pop(scf_data: &SCF) -> Vec<f64>{
    let mol = &scf_data.mol;
    let atom_mass_charge = geom_io::get_mass_charge(&mol.geom.elem).clone();
    let num_atoms = mol.geom.elem.len();
    let mut charge = vec![0.0; num_atoms];
    charge.iter_mut().zip(atom_mass_charge.iter()).for_each(|(x,(a,b))|{
        *x = *b;
    });
    let nao = mol.num_basis;
    let mut m = MatrixFull::from_vec([nao,nao], vec![0.0; nao*nao]).unwrap();
    let dm0 = &scf_data.density_matrix[0];
    //&dm0.formated_output_e(5, "full");
    let dm1 = &scf_data.density_matrix[1];
    let mut dm = MatrixFull::from_vec([nao,nao], vec![0.0; nao*nao]).unwrap();
    if mol.ctrl.spin_polarization{
        dm.data.iter_mut().zip(dm0.data.iter().zip(dm1.data.iter())).for_each(|(x,(a,b))|{
            *x = *a + *b;
        });
        //&dm.formated_output_e(5, "full");
    }else{
        dm = dm0.clone();
    }
    
    let mut s = scf_data.ovlp.to_matrixfull().unwrap();
    let s_t = s.transpose_and_drop();
    m.data.iter_mut().zip(s_t.data.iter()).zip(dm.data.iter()).for_each(|((x,s),d)|{
        *x = *s * *d
    });
    

    let mut pop = vec![0.0; nao];
    pop.iter_mut().zip(m.iter_columns_full()).for_each(|(charge, column)|{
        column.iter().for_each(|x|{
            *charge += *x 
        })
    });
    //println!("pop: {:?}", &pop);

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