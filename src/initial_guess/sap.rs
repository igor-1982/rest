use crate::molecule_io;
use crate::geom_io;
use crate::dft::Grids as dftgrids;
use crate::dft;
use rayon::prelude::IndexedParallelIterator;
use rayon::prelude::IntoParallelRefIterator;
use rayon::prelude::IntoParallelRefMutIterator;
use rayon::prelude::ParallelIterator;
//use crate::scf::io;
use rest_tensors::{MatrixFull, matrix_blas_lapack::_einsum_01_rayon,matrix_blas_lapack::_dgemm_nn};
use crate::constants::ARR;

    
pub fn mid_zeff(z: f64, r: f64, arr: &[[f64;1501];119]) -> f64 {
    let mut zeff_r_a = 0.0;

    if r >= 0.0  {
        let arr0 = &arr[0];
        let arrz = &arr[z as usize];
        if r <= 40.0{
            for i in 0..1501 {
                if (r >=arr0[i]&& r <= arr0[i+1]) {
                    //println!("i is {}",&arrz[i]);
                    zeff_r_a  = arrz[i] + (arrz[i+1]-arrz[i]) / (arr0[i+1]-arr0[i]) * (r - arr0[i]);
                    //println!("{}",&zeff_r_a);
                    break; 
                }
            } 
        }else{
            zeff_r_a = 0.0_f64;//arrz[1500];
        }
    }else{
        println!("{} is over range",r);
    }
    zeff_r_a/r
}   

#[derive(Clone,Debug,PartialEq)]
pub struct SimpleAtomInfo<'a> {
    pub position_a: &'a [f64],
    pub z_a: f64,
}

pub fn get_vsap(mol: &molecule_io::Molecule, grids: &dft::Grids) -> MatrixFull<f64> {
    let dt1 = time::Local::now();
    
    let num_grids = grids.weights.len();
    //grids.prepare_tabulated_ao(mol);
    let mut a_matrix = &grids.ao.as_ref();
    let dt4 = time::Local::now();

    //get V
    let atom_info = &mol.geom;
    println!("{}",atom_info.name);
    let atom_pos = &atom_info.position; //atom_pos : matrixfull
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
    //println!("{:?}",&atoms_info[0]._a);
    let mut v_diag_grid = vec![0.0;num_grids];
    //v_diag_grid.iter_mut().zip(grids.coordinates.iter().zip(grids.weights.iter())).for_each(|(v_grid,(c,w))| {
    v_diag_grid.par_iter_mut().zip(grids.coordinates.par_iter()).zip(grids.weights.par_iter())
        .map(|((v_grid,c),w)| {(v_grid,c,w)}).for_each(|(v_grid,c,w)| {
        *v_grid = 0.0;
        atoms_info.iter().for_each(|ai| {
            let r = ai.position_a.iter().zip(c.iter()).fold(0.0,|r,(ac,gc)| {r + (ac-gc).powf(2.0)}).sqrt();
            *v_grid += mid_zeff(ai.z_a, r, &ARR);
            //println!("{},",&v_grid);
        });
        *v_grid *= - w;
    });
    //println!("v_grid:{:?}",&v_diag_grid);

    let mut int_mat = _einsum_01_rayon(&a_matrix.unwrap().to_matrixfullslice(),&v_diag_grid);
    //&int_mat.formated_output(5, "full");

    let mut f_mat = MatrixFull::new([a_matrix.unwrap().size[0],int_mat.size[0]],0.0);
    f_mat.to_matrixfullslicemut().lapack_dgemm(&mut a_matrix.unwrap().to_matrixfullslice(), &mut int_mat.to_matrixfullslice(), 'N', 'T', 1.0, 0.0);
    //&f_mat.formated_output(5, "full");

    let dt2 = time::Local::now();
    let timecost = (dt2.timestamp_millis()-dt1.timestamp_millis()) as f64 /1000.0;
    println!("The evaluation of vmat costs {:16.2} seconds",timecost);
    f_mat

}