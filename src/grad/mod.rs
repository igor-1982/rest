pub mod rhf;
pub mod uhf;
pub mod rks;
pub mod uks;

use crate::{collect_total_energy, constants::{ANG, EV}, performance_essential_calculations, scf_io::{initialize_scf, scf_without_build, SCF}, utilities};
use tensors::MatrixFull;


pub fn numerical_force(scf_data: &SCF, displace: f64) -> (f64,MatrixFull<f64>) {
    let num_atoms =  scf_data.mol.geom.nfree;
    let mut num_force = MatrixFull::new([3,num_atoms],0.0);
    (0..num_atoms).into_iter().for_each(|atm_idx| {
        let mut num_force_atm = &mut num_force[(..,atm_idx)];
        num_force_atm.iter_mut().enumerate().for_each(|(xyz,per_force)| {
            let mut time_mark = utilities::TimeRecords::new();
            
            // move the atom along + direction
            let mut vec_xyz = vec![0.0;3];
            vec_xyz[xyz] = displace;
            let mut new_scf = scf_data.clone();
            // update the geometry
            new_scf.mol.geom.geom_shift(atm_idx, vec_xyz);
            // update the control file
            new_scf.mol.ctrl.print_level = 0;
            new_scf.mol.ctrl.initial_guess = String::from("inherit");
            initialize_scf(&mut new_scf);
            let de0 = performance_essential_calculations(&mut new_scf, &mut time_mark);

            // move the atom along - direction
            let mut vec_xyz = vec![0.0;3];
            vec_xyz[xyz] = -displace;
            new_scf = scf_data.clone();
            // update the geometry
            new_scf.mol.geom.geom_shift(atm_idx, vec_xyz);
            // update the control file
            new_scf.mol.ctrl.print_level = 0;
            new_scf.mol.ctrl.initial_guess = String::from("inherit");
            initialize_scf(&mut new_scf);
            let de1 = performance_essential_calculations(&mut new_scf, &mut time_mark);

            *per_force = 0.5*(de0-de1)/displace;

        })
    });

    (collect_total_energy(scf_data),num_force)
}

pub fn formated_force(force: &MatrixFull<f64>, elem: &Vec<String>) -> String {
    let mut output = String::new();
    force.iter_columns_full().zip(elem.iter()).for_each(|(force, elem)| {
        output  = format!("{}{:3}{:16.8}{:16.8}{:16.8}\n", output, elem, force[0],force[1],force[2]);
    });

    output

}

pub fn formated_force_ev(force: &MatrixFull<f64>, elem: &Vec<String>) -> String {
    let mut output = String::new();
    force.iter_columns_full().zip(elem.iter()).for_each(|(force, elem)| {
        output  = format!("{}{:3}{:16.8}{:16.8}{:16.8}\n", output, elem, force[0]*EV/ANG,force[1]*EV/ANG,force[2]*EV/ANG);
    });

    output

}


//pub fn evaluate(x: &[f64], gx: &mut [f64]) -> f64 {
//
//    
//    fn to_matrixfull(x: &[f64]) -> MatrixFull<f64> {
//        let num_atoms = x.len()/3;
//        MatrixFull::from_vec([3,num_atoms],x.to_vec()).unwrap()
//    }
//
//}