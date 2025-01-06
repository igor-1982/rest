pub mod rhf;
pub mod uhf;
pub mod rks;
pub mod uks;

use std::io::{self, Write};

use crate::{collect_total_energy, constants::{ANG, EV}, performance_essential_calculations, scf_io::{initialize_scf, scf_without_build, SCF}, utilities};
use crate::mpi_io::{MPIData, MPIOperator};
use tensors::MatrixFull;


pub fn numerical_force(scf_data: &SCF, displace: f64, mpi_operator: &Option<MPIOperator>) -> (f64,MatrixFull<f64>) {
    let num_atoms =  scf_data.mol.geom.nfree;
    let mut num_force = MatrixFull::new([3,num_atoms],0.0);
    if scf_data.mol.ctrl.print_level > 0 {
        if let Some(mp_op) = mpi_operator {
            if mp_op.rank == 0 {
                print!("Numerical force calculation ...");
                io::stdout().flush().unwrap();
            }
        } else {
            print!("Numerical force calculation ...");
            io::stdout().flush().unwrap();
        }
    }
    (0..num_atoms).into_iter().for_each(|atm_idx| {
        if scf_data.mol.ctrl.print_level > 0 {
            if let Some(mp_op) = mpi_operator {
                if mp_op.rank == 0 {
                    print!("| {:3}", &atm_idx);
                    io::stdout().flush().unwrap();
                }
            } else {
                print!("| {:3}", &atm_idx);
                io::stdout().flush().unwrap();
            }
        }
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
            // ==== DEBUG IGOR ====
            if let Some(mp_op) = &mpi_operator {
                if mp_op.rank == 0 {
                    new_scf.mol.ctrl.print_level = 0;
                } else {
                    new_scf.mol.ctrl.print_level = 0;
                }
            } else {
                new_scf.mol.ctrl.print_level = 0;
            }
            // ==== DEBUG IGOR ====
            new_scf.mol.ctrl.initial_guess = String::from("inherit");
            initialize_scf(&mut new_scf, mpi_operator);
            let de0 = performance_essential_calculations(&mut new_scf, &mut time_mark, mpi_operator);

            // move the atom along - direction
            let mut vec_xyz = vec![0.0;3];
            vec_xyz[xyz] = -displace;
            new_scf = scf_data.clone();
            // update the geometry
            new_scf.mol.geom.geom_shift(atm_idx, vec_xyz);
            // update the control file
            // ==== DEBUG IGOR ====
            if let Some(mp_op) = &mpi_operator {
                if mp_op.rank == 0 {
                    new_scf.mol.ctrl.print_level = 0;
                } else {
                    new_scf.mol.ctrl.print_level = 0;
                }
            } else {
                new_scf.mol.ctrl.print_level = 0;
            }
            // ==== DEBUG IGOR ====
            new_scf.mol.ctrl.initial_guess = String::from("inherit");
            initialize_scf(&mut new_scf, mpi_operator);
            let de1 = performance_essential_calculations(&mut new_scf, &mut time_mark, mpi_operator);

            *per_force = 0.5*(de0-de1)/displace;

        })
    });

    if scf_data.mol.ctrl.print_level > 0 {
        if let Some(mp_op) = mpi_operator {
            if mp_op.rank == 0 {
                print!("|\n");
                io::stdout().flush().unwrap();
            }
        } else {
            print!("|");
            io::stdout().flush().unwrap();
        }
    }

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