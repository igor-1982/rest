use std::num;
use std::ops::Range;
use std::sync::Arc;
use std::sync::mpsc::channel;
use rayon::prelude::{IndexedParallelIterator, ParallelIterator, IntoParallelRefIterator};
use rayon::slice::ParallelSlice;
use rest_tensors::{TensorOpt,RIFull, MatrixFull};
use rest_tensors::matrix_blas_lapack::{_dgemm_nn,_dgemm_tn};
use tensors::BasicMatrix;
use tensors::matrix_blas_lapack::{_dsymm, _dgemm};

use crate::ri_pt2::sbge2::{close_shell_sbge2_rayon,open_shell_sbge2_rayon};
use crate::scf_io::determine_ri3mo_size_for_pt2_and_rpa;
use crate::molecule_io::Molecule;
use crate::scf_io::SCF;
use crate::utilities::{TimeRecords, self};

pub mod sbge2;

#[derive(Clone)]
pub struct PT2 {
    pub pt2_type: usize,
    pub pt2_param: [f64;2],
    pub pt2_energy: [f64;2]
}

impl PT2 {
    pub fn new(mol: &Molecule) -> PT2 {
        PT2 {
            pt2_type: 0,
            pt2_param:[1.0,1.0],
            pt2_energy:[0.0,0.0]
        }
    }
}

pub fn xdh_calculations(scf_data: &mut SCF) -> anyhow::Result<f64> {
    let postscf_method = match scf_data.mol.xc_data.dfa_family_pos.clone().unwrap() {
        crate::dft::DFAFamily::PT2 => "PT2".to_string(),
        crate::dft::DFAFamily::SBGE2 => "SBGE2".to_string(),
        crate::dft::DFAFamily::RPA => "RPA".to_string(),
        crate::dft::DFAFamily::SCSRPA => "RPA".to_string(),
        _ => panic!("Error: No post-scf calculation is needed"),
    };
    println!("==========================================");
    println!("Now evaluate the {:>6} correlation energy", postscf_method);
    println!("==========================================");

    let mut pt2_c = [
        // total pt2 correlation energy;
        0.0_f64, 
        // opposite-spin pt2 correlation energy;
        0.0_f64,
        // same-spin pt2 correlation energy;
        0.0_f64
    ];

    let x_energy = scf_data.evaluate_exact_exchange_ri_v();
    let xc_energy_scf = scf_data.evaluate_xc_energy(0);
    let xc_energy_xdh = scf_data.evaluate_xc_energy(1);

    if scf_data.mol.ctrl.use_ri_symm {
        let (occ_range, vir_range) = determine_ri3mo_size_for_pt2_and_rpa(&scf_data);
        if scf_data.mol.ctrl.print_level>1 {
            println!("generate RI3MO only for occ_range:{:?}, vir_range:{:?}", &occ_range, &vir_range)
        };
        scf_data.generate_ri3mo_rayon(vir_range, occ_range);
        let dfa_family_pos = scf_data.mol.xc_data.dfa_family_pos.clone().unwrap();
        pt2_c = if scf_data.mol.spin_channel == 1 {
            match  dfa_family_pos {
                crate::dft::DFAFamily::PT2 => close_shell_pt2_rayon(&scf_data).unwrap(),
                crate::dft::DFAFamily::SBGE2 => close_shell_sbge2_rayon(scf_data).unwrap(),
                _ => [0.0,0.0,0.0]
            }
        } else {
            match  dfa_family_pos {
                crate::dft::DFAFamily::PT2 => open_shell_pt2_rayon(&scf_data).unwrap(),
                crate::dft::DFAFamily::SBGE2 => open_shell_sbge2_rayon(scf_data).unwrap(),
                _ => [0.0,0.0,0.0]
            }
            
        }
    } else {
        pt2_c = if scf_data.mol.spin_channel == 1 {
            close_shell_pt2(&scf_data).unwrap()
        } else {
            open_shell_pt2(&scf_data).unwrap()
        };
    };
    //println!("{:?}",&pt2_c);
    let hy_coeffi_scf = scf_data.mol.xc_data.dfa_hybrid_scf;
    let hy_coeffi_xdh = if let Some(coeff) = scf_data.mol.xc_data.dfa_hybrid_pos {coeff} else {0.0};
    let hy_coeffi_pt2 = if let Some(coeff) = &scf_data.mol.xc_data.dfa_paramr_adv {coeff.clone()} else {vec![0.0,0.0]};
    let xdh_pt2_energy: f64 = pt2_c[1..3].iter().zip(hy_coeffi_pt2.iter()).map(|(e,c)| e*c).sum();
    //println!("Exc_scf: ({:?},{:?}),Exc_pos: ({:?},{:?})",xc_energy_scf,hy_coeffi_scf,xc_energy_xdh,hy_coeffi_xdh);
    println!("Fifth-rung correlation energy : {:20.12} Ha", xdh_pt2_energy);
    let total_energy = scf_data.scf_energy +
                            x_energy * (hy_coeffi_xdh-hy_coeffi_scf) +
                            xc_energy_xdh-xc_energy_scf +
                            xdh_pt2_energy;
    println!("E[{:5}]=: {:16.8} Ha, Ex[HF]: {:16.8} Ha, Ec[PT2]: {:16.8} Ha", 
        scf_data.mol.ctrl.xc.to_uppercase(), 
        total_energy, 
        x_energy, 
        pt2_c[0]);
    Ok(total_energy)
}


/// ==========================================================================================
///    `E_c[PT2]`=\sum_{i<j}^{occ}\sum_{a<b}^{vir}         |(ia||jb)|^2
///                                                x --------------------------
///                                                    e_i+e_j-e_a-e_b
///  For each electron-pair correlation e_{ij}:
///    e_{ij} = \sum_{a<b}^{vir}          |(ia||jb)|^2
///                              x --------------------------
///                                 e_i+e_j-e_a-e_b
///           = \sum_{a<b}^{vir}      |(ia|jb)-(ib|ja)|^2
///                              x --------------------------
///                                 e_i+e_j-e_a-e_b
///  Then:
///   `E_c[PT2]`=\sum{i<j}^{occ}e_{ij}
/// ==========================================================================================
fn close_shell_pt2(scf_data: &SCF) -> anyhow::Result<[f64;3]> {
    if let Some(riao)=&scf_data.ri3fn {
        let mut e_mp2_ss = 0.0_f64;
        let mut e_mp2_os = 0.0_f64;
        let eigenvector = scf_data.eigenvectors.get(0).unwrap();
        let eigenvalues = scf_data.eigenvalues.get(0).unwrap();

        let homo = scf_data.homo.get(0).unwrap().clone();
        let lumo = scf_data.lumo.get(0).unwrap().clone();
        let num_basis = eigenvector.size.get(0).unwrap().clone();
        let num_state = eigenvector.size.get(1).unwrap().clone();
        let start_mo: usize = scf_data.mol.start_mo;
        let num_occu = homo + 1;
        //let num_virt = num_state - num_occu;
        //println!("{:?},{:?},{:?},{:?}",homo,lumo,num_state, num_basis);
        //for i in 0..homo {
        //    for j in i..homo {
        //    }
        //}
        let mut tmp_record = TimeRecords::new();
        tmp_record.new_item("rimo", "the generation of three-center RI tensor for MO");
        tmp_record.count_start("rimo");
        let mut rimo = riao.ao2mo(eigenvector).unwrap();
        tmp_record.count("rimo");

        tmp_record.new_item("dgemm", "prepare four-center integrals from RI-MO");
        tmp_record.new_item("get2d", "get the ERI values");
        for i_state in start_mo..num_occu {
            let i_state_eigen = eigenvalues.get(i_state).unwrap();
            for j_state in i_state..num_occu {

                let mut e_mp2_term_ss = 0.0_f64;
                let mut e_mp2_term_os = 0.0_f64;

                let j_state_eigen = eigenvalues.get(j_state).unwrap();
                let ij_state_eigen = i_state_eigen + j_state_eigen;

                tmp_record.count_start("dgemm");
                let ri_i = rimo.get_reducing_matrix(i_state).unwrap();
                let ri_j = rimo.get_reducing_matrix(j_state).unwrap();
                //if (i_state == start_mo && j_state == i_state) {println!("debug {:?}", &ri_i.get_slice_x(lumo))};
                let eri_virt = _dgemm_tn(&ri_i,&ri_j);
                tmp_record.count("dgemm");

                for i_virt in lumo..num_state {
                    let i_virt_eigen = eigenvalues.get(i_virt).unwrap();
                    for j_virt in lumo..num_state {

                        let j_virt_eigen = eigenvalues.get(j_virt).unwrap();
                        let ij_virt_eigen = i_virt_eigen + j_virt_eigen;

                        let mut double_gap = ij_virt_eigen - ij_state_eigen;
                        if double_gap.abs()<=10E-6 {
                            println!("Warning: too close to degeneracy")
                        };

                        tmp_record.count_start("get2d");
                        let e_mp2_a = eri_virt.get2d([i_virt,j_virt]).unwrap();
                        let e_mp2_b = eri_virt.get2d([j_virt,i_virt]).unwrap();
                        tmp_record.count("get2d");
                        e_mp2_term_ss += (e_mp2_a - e_mp2_b) * e_mp2_a / double_gap;
                        e_mp2_term_os += e_mp2_a * e_mp2_a / double_gap;

                    }
                }
                if i_state != j_state {
                    e_mp2_term_ss *= 2.0;
                    e_mp2_term_os *= 2.0;
                }
                //println!("{:?},{:?},{:?}, {:?}",i_state, j_state,e_mp2_term_os,e_mp2_term_ss);
                e_mp2_ss -= e_mp2_term_ss;
                e_mp2_os -= e_mp2_term_os;
            }
        }
        tmp_record.report_all();
        return(Ok([e_mp2_ss+e_mp2_os,e_mp2_os,e_mp2_ss]));
    } else {
        panic!("ri3fn should be initialized for RI-PT2 calculations")
    };

}

fn open_shell_pt2(scf_data: &SCF) -> anyhow::Result<[f64;3]> {
    if let Some(riao)=&scf_data.ri3fn {
        let start_mo: usize = scf_data.mol.start_mo;
        let mut e_mp2_ss = 0.0_f64;
        let mut e_mp2_os = 0.0_f64;
        let num_basis = scf_data.mol.num_basis;
        let num_state = scf_data.mol.num_state;
        let spin_channel = scf_data.mol.spin_channel;
        let i_spin_pair: [(usize,usize);3] = [(0,0),(0,1),(1,1)];
        for (i_spin_1,i_spin_2) in i_spin_pair {
            if i_spin_1 == i_spin_2 {

                let i_spin = i_spin_1;
                let eigenvector = scf_data.eigenvectors.get(i_spin).unwrap();
                let eigenvalues = scf_data.eigenvalues.get(i_spin).unwrap();

                let homo = scf_data.homo.get(i_spin).unwrap().clone();
                let lumo = scf_data.lumo.get(i_spin).unwrap().clone();
                let num_occu = homo + 1;

                let mut rimo = riao.ao2mo(eigenvector).unwrap();

                for i_state in start_mo..num_occu {
                    let i_state_eigen = eigenvalues.get(i_state).unwrap();
                    for j_state in i_state+1..num_occu {

                        let mut e_mp2_term_ss = 0.0_f64;

                        let j_state_eigen = eigenvalues.get(j_state).unwrap();
                        let ij_state_eigen = i_state_eigen + j_state_eigen;
                        let ri_i = rimo.get_reducing_matrix(i_state).unwrap();
                        let ri_j = rimo.get_reducing_matrix(j_state).unwrap();
                        let eri_virt = _dgemm_tn(&ri_i,&ri_j);

                        for i_virt in lumo..num_state {
                            let i_virt_eigen = eigenvalues.get(i_virt).unwrap();
                            for j_virt in i_virt+1..num_state {
                                let j_virt_eigen = eigenvalues.get(j_virt).unwrap();
                                let ij_virt_eigen = i_virt_eigen + j_virt_eigen;

                                let mut double_gap = ij_virt_eigen - ij_state_eigen;
                                if double_gap.abs()<=10E-6 {
                                    println!("Warning: too close to degeneracy")
                                };

                                let e_mp2_a = eri_virt.get2d([i_virt,j_virt]).unwrap();
                                let e_mp2_b = eri_virt.get2d([j_virt,i_virt]).unwrap();
                                e_mp2_term_ss += (e_mp2_a - e_mp2_b).powf(2.0) / double_gap;
                                //e_mp2_term_os += e_mp2_a * e_mp2_a / double_gap;

                            }
                        }
                        e_mp2_ss -= e_mp2_term_ss;
                    }
                }
            } else {
                let eigenvector_1 = scf_data.eigenvectors.get(i_spin_1).unwrap();
                let eigenvalues_1 = scf_data.eigenvalues.get(i_spin_1).unwrap();
                let homo_1 = scf_data.homo.get(i_spin_1).unwrap().clone();
                let lumo_1 = scf_data.lumo.get(i_spin_1).unwrap().clone();
                let num_occu_1 = homo_1 + 1;
                let mut rimo_1 = riao.ao2mo_v01(eigenvector_1).unwrap();

                let eigenvector_2 = scf_data.eigenvectors.get(i_spin_2).unwrap();
                let eigenvalues_2 = scf_data.eigenvalues.get(i_spin_2).unwrap();
                let homo_2 = scf_data.homo.get(i_spin_2).unwrap().clone();
                let lumo_2 = scf_data.lumo.get(i_spin_2).unwrap().clone();
                let num_occu_2 = homo_2 + 1;
                let mut rimo_2 = riao.ao2mo_v01(eigenvector_2).unwrap();
                for i_state in start_mo..num_occu_1 {
                    let i_state_eigen = eigenvalues_1.get(i_state).unwrap();
                    let ri_i = rimo_1.get_reducing_matrix(i_state).unwrap();
                    for j_state in start_mo..num_occu_2 {

                        let mut e_mp2_term_os = 0.0_f64;

                        let j_state_eigen = eigenvalues_2.get(j_state).unwrap();
                        let ri_j = rimo_2.get_reducing_matrix(j_state).unwrap();

                        let ij_state_eigen = i_state_eigen + j_state_eigen;
                        let eri_virt = _dgemm_tn(&ri_i,&ri_j);

                        for i_virt in lumo_1..num_state {
                            let i_virt_eigen = eigenvalues_1.get(i_virt).unwrap();
                            for j_virt in lumo_2..num_state {
                                let j_virt_eigen = eigenvalues_2.get(j_virt).unwrap();
                                let ij_virt_eigen = i_virt_eigen + j_virt_eigen;

                                let mut double_gap = ij_virt_eigen - ij_state_eigen;
                                if double_gap.abs()<=10E-6 {
                                    println!("Warning: too close to degeneracy")
                                };

                                let e_mp2_a = eri_virt.get2d([i_virt,j_virt]).unwrap();
                                e_mp2_term_os += e_mp2_a * e_mp2_a / double_gap;

                            }
                        }
                        e_mp2_os -= e_mp2_term_os;
                    }
                }
            }
        }
        return(Ok([e_mp2_ss+e_mp2_os,e_mp2_os,e_mp2_ss]));
    } else {
        panic!("ri3fn should be initialized for RI-PT2 calculations")
    };

}

pub fn close_shell_pt2_rayon(scf_data: &SCF) -> anyhow::Result<[f64;3]> {
    //let mut tmp_record = TimeRecords::new();

    // In this subroutine, we call the lapack dgemm in a rayon parallel environment.
    // In order to ensure the efficiency, we disable the openmp ability and re-open it in the end of subroutien
    let default_omp_num_threads = utilities::omp_get_num_threads_wrapper();
    utilities::omp_set_num_threads_wrapper(1);

    let mut e_mp2_ss = 0.0_f64;
    let mut e_mp2_os = 0.0_f64;

    if let Some(ri3mo_vec) = &scf_data.ri3mo {
        let (rimo, vir_range, occ_range) = &ri3mo_vec[0];

        let eigenvector = scf_data.eigenvectors.get(0).unwrap();
        let eigenvalues = scf_data.eigenvalues.get(0).unwrap();

        let homo = scf_data.homo.get(0).unwrap().clone();
        let lumo = scf_data.lumo.get(0).unwrap().clone();
        let num_basis = eigenvector.size.get(0).unwrap().clone();
        let num_auxbas = rimo.size[0];
        let num_state = eigenvector.size.get(1).unwrap().clone();
        let start_mo: usize = scf_data.mol.start_mo;
        let num_occu = homo + 1;
        //tmp_record.new_item("dgemm", "prepare four-center integrals from RI-MO");
        //tmp_record.new_item("get2d", "get the ERI values");
        let mut elec_pair: Vec<[usize;2]> = vec![];
        for i_state in start_mo..num_occu {
            for j_state in i_state..num_occu {
                elec_pair.push([i_state,j_state])
            }
        };
        let (sender, receiver) = channel();
        elec_pair.par_iter().for_each_with(sender,|s,i_pair| {
            let mut e_mp2_term_ss = 0.0_f64;
            let mut e_mp2_term_os = 0.0_f64;

            let i_state = i_pair[0];
            let j_state = i_pair[1];
            let i_state_eigen = eigenvalues.get(i_state).unwrap();
            let j_state_eigen = eigenvalues.get(j_state).unwrap();
            let ij_state_eigen = i_state_eigen + j_state_eigen;

            // because we generate ri3mo for [lumo..num_state, start_mo..num_occ], 
            // the indices in rimo are shifted
            let i_loc_state = i_state-occ_range.start;
            let j_loc_state = j_state-occ_range.start;
            let ri_i = rimo.get_reducing_matrix(i_loc_state).unwrap();
            let ri_j = rimo.get_reducing_matrix(j_loc_state).unwrap();
            let mut eri_virt = MatrixFull::new([vir_range.len(),vir_range.len()],0.0_f64);
            _dgemm(
                &ri_i, (0..num_auxbas,0..vir_range.len()), 'T', 
                &ri_j,(0..num_auxbas,0..vir_range.len()) , 'N', 
                &mut eri_virt, (0..vir_range.len(),0..vir_range.len()), 
                1.0,0.0);

            for i_virt in lumo..num_state {
                let i_virt_eigen = eigenvalues.get(i_virt).unwrap();
                for j_virt in lumo..num_state {

                    let j_virt_eigen = eigenvalues.get(j_virt).unwrap();
                    let ij_virt_eigen = i_virt_eigen + j_virt_eigen;

                    let mut double_gap = ij_virt_eigen - ij_state_eigen;
                    if double_gap.abs()<=10E-6 {
                        println!("Warning: too close to degeneracy")
                    };

                    // because we generate ri3mo for [lumo..num_state, start_mo..num_occ], 
                    // the indices in rimo are shifted
                    let i_loc_virt = i_virt-vir_range.start;
                    let j_loc_virt = j_virt-vir_range.start;

                    let e_mp2_a = eri_virt.get2d([i_loc_virt,j_loc_virt]).unwrap();
                    let e_mp2_b = eri_virt.get2d([j_loc_virt,i_loc_virt]).unwrap();
                    e_mp2_term_ss += (e_mp2_a - e_mp2_b) * e_mp2_a / double_gap;
                    e_mp2_term_os += e_mp2_a * e_mp2_a / double_gap;

                }
            }
            if i_state != j_state {
                e_mp2_term_ss *= 2.0;
                e_mp2_term_os *= 2.0;
            }

            s.send((e_mp2_term_os, e_mp2_term_ss)).unwrap()

        });
        receiver.into_iter().for_each(|(e_mp2_term_os,e_mp2_term_ss)| {
            e_mp2_os -= e_mp2_term_os;
            e_mp2_ss -= e_mp2_term_ss;
        });
    } else {
        panic!("RI3MO should be initialized before the PT2 calculations")
    };

    // reuse the default omp_num_threads setting
    utilities::omp_set_num_threads_wrapper(default_omp_num_threads);
    //tmp_record.report_all();
    Ok([e_mp2_ss+e_mp2_os,e_mp2_os,e_mp2_ss])

}

pub fn open_shell_pt2_rayon(scf_data: &SCF) -> anyhow::Result<[f64;3]> {
    // In this subroutine, we call the lapack dgemm in a rayon parallel environment.
    // In order to ensure the efficiency, we disable the openmp ability and re-open it in the end of subroutien
    let default_omp_num_threads = utilities::omp_get_num_threads_wrapper();
    utilities::omp_set_num_threads_wrapper(1);

    let mut e_mp2_ss = 0.0_f64;
    let mut e_mp2_os = 0.0_f64;

    if let Some(ri3mo_vec) = &scf_data.ri3mo {

        let start_mo: usize = scf_data.mol.start_mo;
        let num_basis = scf_data.mol.num_basis;
        let num_state = scf_data.mol.num_state;
        let num_auxbas = scf_data.mol.num_auxbas;
        let spin_channel = scf_data.mol.spin_channel;
        let i_spin_pair: [(usize,usize);3] = [(0,0),(0,1),(1,1)];

        for (i_spin_1,i_spin_2) in i_spin_pair {
            if i_spin_1 == i_spin_2 {

                let i_spin = i_spin_1;
                let eigenvector = scf_data.eigenvectors.get(i_spin).unwrap();
                let eigenvalues = scf_data.eigenvalues.get(i_spin).unwrap();

                let homo = scf_data.homo.get(i_spin).unwrap().clone();
                let lumo = scf_data.lumo.get(i_spin).unwrap().clone();
                let num_occu = homo + 1;

                let (rimo, vir_range, occ_range) = &ri3mo_vec[i_spin];

                //let mut rimo = riao.ao2mo(eigenvector).unwrap();
                let mut elec_pair: Vec<[usize;2]> = vec![];
                for i_state in start_mo..num_occu {
                    for j_state in i_state..num_occu {
                        elec_pair.push([i_state,j_state])
                    }
                };
                let (sender, receiver) = channel();
                elec_pair.par_iter().for_each_with(sender,|s,i_pair| {

                    let mut e_mp2_term_ss = 0.0_f64;

                    let i_state = i_pair[0];
                    let j_state = i_pair[1];
                    let i_state_eigen = eigenvalues.get(i_state).unwrap();
                    let j_state_eigen = eigenvalues.get(j_state).unwrap();
                    let ij_state_eigen = i_state_eigen + j_state_eigen;

                    // because we generate ri3mo for [lumo..num_state, start_mo..num_occ], 
                    // the indices in rimo are shifted
                    let i_loc_state = i_state-occ_range.start;
                    let j_loc_state = j_state-occ_range.start;
                    let ri_i = rimo.get_reducing_matrix(i_loc_state).unwrap();
                    let ri_j = rimo.get_reducing_matrix(j_loc_state).unwrap();
                    let mut eri_virt = MatrixFull::new([vir_range.len(),vir_range.len()],0.0_f64);
                    _dgemm(
                        &ri_i, (0..num_auxbas,0..vir_range.len()), 'T', 
                        &ri_j,(0..num_auxbas,0..vir_range.len()) , 'N', 
                        &mut eri_virt, (0..vir_range.len(),0..vir_range.len()), 
                        1.0,0.0);
                    for i_virt in lumo..num_state {
                        let i_virt_eigen = eigenvalues[i_virt];
                        for j_virt in i_virt+1..num_state {
                            let j_virt_eigen = eigenvalues[j_virt];
                            let ij_virt_eigen = i_virt_eigen + j_virt_eigen;

                            let mut double_gap = ij_virt_eigen - ij_state_eigen;
                            if double_gap.abs()<=10E-6 {
                                println!("Warning: too close to degeneracy")
                            };

                            // because we generate ri3mo for [lumo..num_state, start_mo..num_occ], 
                            // the indices in rimo are shifted
                            let i_loc_virt = i_virt-vir_range.start;
                            let j_loc_virt = j_virt-vir_range.start;
                            let e_mp2_a = eri_virt.get2d([i_loc_virt,j_loc_virt]).unwrap();
                            let e_mp2_b = eri_virt.get2d([j_loc_virt,i_loc_virt]).unwrap();
                            e_mp2_term_ss += (e_mp2_a - e_mp2_b).powf(2.0) / double_gap;
                        }
                    }
                    s.send(e_mp2_term_ss).unwrap()
                });

                e_mp2_ss -= receiver.into_iter().sum::<f64>();

            } else {
                let eigenvector_1 = scf_data.eigenvectors.get(i_spin_1).unwrap();
                let eigenvalues_1 = scf_data.eigenvalues.get(i_spin_1).unwrap();
                let homo_1 = scf_data.homo.get(i_spin_1).unwrap().clone();
                let lumo_1 = scf_data.lumo.get(i_spin_1).unwrap().clone();
                let num_occu_1 = homo_1 + 1;
                let (rimo_1, vir_range, occ_range) = &ri3mo_vec[i_spin_1];

                let eigenvector_2 = scf_data.eigenvectors.get(i_spin_2).unwrap();
                let eigenvalues_2 = scf_data.eigenvalues.get(i_spin_2).unwrap();
                let homo_2 = scf_data.homo.get(i_spin_2).unwrap().clone();
                let lumo_2 = scf_data.lumo.get(i_spin_2).unwrap().clone();
                let num_occu_2 = homo_2 + 1;
                let (rimo_2, _, _) = &ri3mo_vec[i_spin_2];


                // prepare the elec_pair for the rayon parallelization
                let mut elec_pair: Vec<[usize;2]> = vec![];
                for i_state in start_mo..num_occu_1 {
                    for j_state in start_mo..num_occu_2 {
                        elec_pair.push([i_state,j_state])
                    }
                };
                let (sender, receiver) = channel();
                elec_pair.par_iter().for_each_with(sender,|s,i_pair| {
                    let mut e_mp2_term_os = 0.0_f64;
                    let i_state = i_pair[0];
                    let j_state = i_pair[1];
                    let i_state_eigen = eigenvalues_1.get(i_state).unwrap();
                    let j_state_eigen = eigenvalues_2.get(j_state).unwrap();
                    let ij_state_eigen = i_state_eigen + j_state_eigen;
                    // because we generate ri3mo for [lumo..num_state, start_mo..num_occ], 
                    // the indices in rimo are shifted
                    let i_loc_state = i_state-occ_range.start;
                    let j_loc_state = j_state-occ_range.start;
                    let ri_i = rimo_1.get_reducing_matrix(i_loc_state).unwrap();
                    let ri_j = rimo_2.get_reducing_matrix(j_loc_state).unwrap();
                    let mut eri_virt = MatrixFull::new([vir_range.len(),vir_range.len()],0.0_f64);
                    _dgemm(
                        &ri_i, (0..num_auxbas,0..vir_range.len()), 'T', 
                        &ri_j,(0..num_auxbas,0..vir_range.len()) , 'N', 
                        &mut eri_virt, (0..vir_range.len(),0..vir_range.len()), 
                        1.0,0.0);
                    for i_virt in lumo_1..num_state {
                        let i_virt_eigen = eigenvalues_1.get(i_virt).unwrap();
                        for j_virt in lumo_2..num_state {
                            let j_virt_eigen = eigenvalues_2.get(j_virt).unwrap();
                            let ij_virt_eigen = i_virt_eigen + j_virt_eigen;

                            let mut double_gap = ij_virt_eigen - ij_state_eigen;
                            if double_gap.abs()<=10E-6 {
                                println!("Warning: too close to degeneracy")
                            };

                            // because we generate ri3mo for [lumo..num_state, start_mo..num_occ], 
                            // the indices in rimo are shifted
                            let i_loc_virt = i_virt-vir_range.start;
                            let j_loc_virt = j_virt-vir_range.start;
                            let e_mp2_a = eri_virt.get2d([i_loc_virt,j_loc_virt]).unwrap();
                            e_mp2_term_os += e_mp2_a * e_mp2_a / double_gap;

                        }
                    }
                    s.send(e_mp2_term_os).unwrap()
                });

                e_mp2_os -= receiver.into_iter().sum::<f64>();
            }
        }
    } else {
        panic!("RI3MO should be initialized before the PT2 calculations")
    };
    // reuse the default omp_num_threads setting
    utilities::omp_set_num_threads_wrapper(default_omp_num_threads);

    Ok([e_mp2_ss+e_mp2_os,e_mp2_os,e_mp2_ss])

}

