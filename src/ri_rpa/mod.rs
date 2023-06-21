pub mod scsrpa;

use std::num;
use std::sync::mpsc::channel;

use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
//use core::num;
//use std::sync::Arc;
use rest_tensors::{TensorOpt,RIFull, MatrixFull, TensorSlice};
use rest_tensors::matrix_blas_lapack::{_dgemm_nn,_dgemm_tn};
use libm::{exp,cos};
//use serde::__private::de;
use tensors::ParMathMatrix;
use tensors::matrix_blas_lapack::_dgemm;

use crate::molecule_io::Molecule;
use crate::scf_io::{SCF,scf};
use crate::constants::{E, PI};
use crate::utilities::{debug_print_slices, self};

pub fn rpa_calculations(scf_data: &mut SCF) -> anyhow::Result<f64> {

    let x_energy = scf_data.evaluate_exact_exchange_ri_v();
    let xc_energy_scf = scf_data.evaluate_xc_energy(0);
    let xc_energy_xdh = scf_data.evaluate_xc_energy(1);

    println!("=======================================");
    println!("Now evaluate the RPA correlation energy");
    println!("=======================================");
    let mut rpa_c_energy = 0.0_f64;
    let spin_channel = scf_data.mol.spin_channel;
    let num_freq = scf_data.mol.ctrl.frequency_points;
    let freq_grid_type = scf_data.mol.ctrl.freq_grid_type;
    let max_freq = scf_data.mol.ctrl.freq_cut_off;
    let mut sp = format!("The frequency integration is tabulated by {:3} grids using", num_freq);
    let (omega,weight) = if freq_grid_type==0 {
        sp = format!("{} the modified Gauss-Legendre grids",sp);
        trans_gauss_legendre_grids(1.0, num_freq)
    } else if freq_grid_type==1 {
        sp = format!("{} the standard Gauss-Legendre grids",sp);
        gauss_legendre_grids([0.0,max_freq], num_freq)
    } else if freq_grid_type== 2 {
        sp = format!("{} the logarithmic grids",sp);
        logarithmic_grid([0.0,max_freq], num_freq)
    } else {
        sp = format!("{} the modified Gauss-Legendre grids",sp);
        trans_gauss_legendre_grids(1.0, num_freq)
    };
    if scf_data.mol.ctrl.print_level>1 {
        println!("{}", sp);
    }

    if scf_data.mol.ctrl.use_ri_symm {

        let (occ_range, vir_range) = crate::scf_io::determine_ri3mo_size_for_pt2_and_rpa(&scf_data);
        if scf_data.mol.ctrl.print_level>1 {
            println!("generate RI3MO only for occ_range:{:?}, vir_range:{:?}", &occ_range, &vir_range)
        };
        scf_data.generate_ri3mo_rayon(vir_range, occ_range);

        rpa_c_energy = evaluate_rpa_correlation_rayon(scf_data).unwrap();
        //rpa_c_energy = evaluate_rpa_correlation(scf_data).unwrap();

    } else {
        let mut rimo: Vec<RIFull<f64>> = if let Some(riao)=&scf_data.ri3fn {
            let spin_channel = scf_data.mol.spin_channel;
            let mut rimo: Vec<RIFull<f64>> =vec![];
            for i_spin in 0..spin_channel {
                let eigenvector = scf_data.eigenvectors.get(i_spin).unwrap();
                rimo.push(riao.ao2mo(eigenvector).unwrap());
            }
            rimo
        } else {
            panic!("ri3fn should be initialized for RI-RPA calculations")
        };
        omega.iter().zip(weight.iter()).for_each(|(omega,weight)| {
            println!(" (freq, weight): ({:16.8},{:16.8})", omega, weight);
            let mut response_freq = evaluate_response(scf_data, &mut rimo,*omega).unwrap();
            if scf_data.mol.spin_channel == 1 {
                response_freq.par_self_multiple(2.0);
            }
            let rpa_c_integrand = evaluate_rpa_integrand(&mut response_freq);

            rpa_c_energy += rpa_c_integrand*weight

        });

        rpa_c_energy = rpa_c_energy*0.5/PI;
    }


    let hy_coeffi_scf = scf_data.mol.xc_data.dfa_hybrid_scf;
    let hy_coeffi_pot = if let Some(coeff) = scf_data.mol.xc_data.dfa_hybrid_pos {coeff} else {0.0};
    //let hy_coeffi_rpa = if let Some(coeff) = &scf_data.mol.xc_data.dfa_paramr_adv {coeff.clone()} else {vec![0.0,0.0]};
    let xdh_rpa_energy: f64 = rpa_c_energy;
    //println!("Exc_scf: ({:?},{:?}),Exc_pos: ({:?},{:?})",xc_energy_scf,hy_coeffi_scf,xc_energy_xdh,hy_coeffi_xdh);
    let total_energy = scf_data.scf_energy +
                            x_energy * (hy_coeffi_pot-hy_coeffi_scf) +
                            xc_energy_xdh-xc_energy_scf +
                            xdh_rpa_energy;
    println!("E[{:?}]=: {:?}, Ex[HF]: {:?}, Ec[RPA]: {:?}", scf_data.mol.ctrl.xc, total_energy, x_energy, rpa_c_energy);
    Ok(total_energy)
}

pub fn evaluate_rpa_correlation(scf_data: &SCF) -> anyhow::Result<f64>  {
    let mut rpa_c_energy = 0.0_f64;
    let spin_channel = scf_data.mol.spin_channel;
    let num_freq = scf_data.mol.ctrl.frequency_points;
    let freq_grid_type = scf_data.mol.ctrl.freq_grid_type;
    let max_freq = scf_data.mol.ctrl.freq_cut_off;
    let mut sp = format!("The frequency integration is tabulated by {:3} grids using", num_freq);
    let (omega,weight) = if freq_grid_type==0 {
        sp = format!("{} the modified Gauss-Legendre grids",sp);
        trans_gauss_legendre_grids(1.0, num_freq)
    } else if freq_grid_type==1 {
        sp = format!("{} the standard Gauss-Legendre grids",sp);
        gauss_legendre_grids([0.0,max_freq], num_freq)
    } else if freq_grid_type== 2 {
        sp = format!("{} the logarithmic grids",sp);
        logarithmic_grid([0.0,max_freq], num_freq)
    } else {
        sp = format!("{} the modified Gauss-Legendre grids",sp);
        trans_gauss_legendre_grids(1.0, num_freq)
    };
    if scf_data.mol.ctrl.print_level>1 {
        println!("{}", sp);
    }
    omega.iter().zip(weight.iter()).for_each(|(omega,weight)| {
        let mut response_freq = evaluate_response_rayon(scf_data, *omega).unwrap();
        if scf_data.mol.spin_channel == 1 {
            response_freq.par_self_multiple(2.0);
        }
        let rpa_c_integrand = evaluate_rpa_integrand(&mut response_freq);

        if scf_data.mol.ctrl.print_level>1 {println!(" (freq, weight, rpa_c): ({:16.8},{:16.8},{:16.8})", omega, weight, rpa_c_integrand)};

        rpa_c_energy += rpa_c_integrand*weight

    });
    rpa_c_energy = rpa_c_energy*0.5/PI;

    Ok(rpa_c_energy)
}

pub fn evaluate_rpa_correlation_rayon(scf_data: &SCF) -> anyhow::Result<f64>  {
    // In this subroutine, we call the lapack dgemm in a rayon parallel environment.
    // In order to ensure the efficiency, we disable the openmp ability and re-open it in the end of subroutien
    let default_omp_num_threads = utilities::omp_get_num_threads_wrapper();
    utilities::omp_set_num_threads_wrapper(1);

    let mut rpa_c_energy = 0.0_f64;
    let spin_channel = scf_data.mol.spin_channel;
    let num_freq = scf_data.mol.ctrl.frequency_points;
    let freq_grid_type = scf_data.mol.ctrl.freq_grid_type;
    let max_freq = scf_data.mol.ctrl.freq_cut_off;
    let mut sp = format!("The frequency integration is tabulated by {:3} grids using", num_freq);
    let (omega,weight) = if freq_grid_type==0 {
        sp = format!("{} the modified Gauss-Legendre grids",sp);
        trans_gauss_legendre_grids(1.0, num_freq)
    } else if freq_grid_type==1 {
        sp = format!("{} the standard Gauss-Legendre grids",sp);
        gauss_legendre_grids([0.0,max_freq], num_freq)
    } else if freq_grid_type== 2 {
        sp = format!("{} the logarithmic grids",sp);
        logarithmic_grid([0.0,max_freq], num_freq)
    } else {
        sp = format!("{} the modified Gauss-Legendre grids",sp);
        trans_gauss_legendre_grids(1.0, num_freq)
    };
    if scf_data.mol.ctrl.print_level>1 {
        println!("{}", sp);
    }

    let (sender,receiver) = channel();
    rayon::prelude::IndexedParallelIterator::zip(omega.par_iter(), weight.par_iter())
        .for_each_with(sender, |s, (omega,weight)| {
        let mut response_freq = evaluate_response_serial(scf_data, *omega).unwrap();
        if scf_data.mol.spin_channel == 1 {
            response_freq *= 2.0;
        }
        let rpa_c_integrand = evaluate_rpa_integrand(&mut response_freq);

        if scf_data.mol.ctrl.print_level>1 {println!(" (freq, weight, rpa_c): ({:16.8},{:16.8},{:16.8})", omega, weight, rpa_c_integrand)};

        //rpa_c_energy += rpa_c_integrand*weight;

        s.send(rpa_c_integrand*weight).unwrap()

    });

    rpa_c_energy = receiver.into_iter().sum();

    rpa_c_energy = rpa_c_energy*0.5/PI;

    utilities::omp_set_num_threads_wrapper(default_omp_num_threads);

    Ok(rpa_c_energy)
}

fn evaluate_response(scf_data: &SCF, rimo: &mut Vec<RIFull<f64>>, freq: f64) -> anyhow::Result<MatrixFull<f64>> {
    let num_auxbas = scf_data.mol.num_auxbas;
    let num_basis = scf_data.mol.num_basis;
    let num_state = scf_data.mol.num_state;
    let start_mo = scf_data.mol.start_mo;
    let spin_channel = scf_data.mol.spin_channel;
    let num_spin = spin_channel as f64;
    let mut polar_freq = MatrixFull::new([num_auxbas,num_auxbas],0.0);

    for i_spin in 0..spin_channel {
        let eigenvector = scf_data.eigenvectors.get(i_spin).unwrap();
        let eigenvalues = scf_data.eigenvalues.get(i_spin).unwrap();
        let occ_numbers = scf_data.occupation.get(i_spin).unwrap();
        let rimo_s = rimo.get_mut(i_spin).unwrap();
        let homo = scf_data.homo.get(i_spin).unwrap().clone();
        let lumo = scf_data.lumo.get(i_spin).unwrap().clone();
        let num_occu = homo + 1;
        for j_state in start_mo .. num_occu {
            let j_state_eigen = eigenvalues.get(j_state).unwrap();
            let j_state_occ = occ_numbers.get(j_state).unwrap();
            let mut tmp_matrix = MatrixFull::new([num_auxbas,num_state],0.0);
            for k_state in lumo .. num_state {
                //let k_state_rel = k_state - num_occu;
                let k_state_eigen = eigenvalues.get(k_state).unwrap();
                let k_state_occ = occ_numbers.get(k_state).unwrap();
                let zeta = num_spin*(j_state_eigen-k_state_eigen) /
                    ((j_state_eigen-k_state_eigen).powf(2.0) + freq*freq)*
                    (j_state_occ-k_state_occ);
                let from_iter = rimo_s.get_slices(0..num_auxbas, k_state..k_state+1, j_state..j_state+1);
                let to_iter = tmp_matrix.iter_submatrix_mut(0..num_auxbas,k_state..k_state+1);
                to_iter.zip(from_iter).for_each(|(to, from)| {
                    *to = from * zeta
                });
            }
            let mut rimo_j = rimo_s.get_reducing_matrix(j_state).unwrap();
            polar_freq.to_matrixfullslicemut().lapack_dgemm(&tmp_matrix.to_matrixfullslice(), 
                &rimo_j, 'N', 'T', 1.0, 1.0);
        }
    }
    //let dt = polar_freq.data.iter().map(|a| *a).collect::<Vec<f64>>();
    //debug_print_slices(&dt);
    return(Ok(polar_freq))
}

fn evaluate_response_rayon(scf_data: &SCF, freq: f64) -> anyhow::Result<MatrixFull<f64>> {
    // In this subroutine, we call the lapack dgemm in a rayon parallel environment.
    // In order to ensure the efficiency, we disable the openmp ability and re-open it in the end of subroutien
    let default_omp_num_threads = utilities::omp_get_num_threads_wrapper();
    utilities::omp_set_num_threads_wrapper(1);

    let num_auxbas = scf_data.mol.num_auxbas;
    let num_basis = scf_data.mol.num_basis;
    let num_state = scf_data.mol.num_state;
    let start_mo = scf_data.mol.start_mo;
    let spin_channel = scf_data.mol.spin_channel;
    let num_spin = spin_channel as f64;
    let mut polar_freq = MatrixFull::new([num_auxbas,num_auxbas],0.0);

    if let Some(ri3mo_vec) = &scf_data.ri3mo {
        for i_spin in 0..spin_channel {
            let eigenvector = scf_data.eigenvectors.get(i_spin).unwrap();
            let eigenvalues = scf_data.eigenvalues.get(i_spin).unwrap();
            let occ_numbers = scf_data.occupation.get(i_spin).unwrap();
            let homo = scf_data.homo.get(i_spin).unwrap().clone();
            let lumo = scf_data.lumo.get(i_spin).unwrap().clone();
            let num_occu = homo + 1;

            let (ri3mo, vir_range, occ_range) = ri3mo_vec.get(i_spin).unwrap();

            
            //let mut rimo = riao.ao2mo(eigenvector).unwrap();
            let mut elec_pair: Vec<[usize;2]> = vec![];
            for j_state in start_mo..num_occu {
                for k_state in lumo..num_state {
                    elec_pair.push([j_state,k_state])
                }
            };
            let (sender,receiver) = channel();
            elec_pair.par_iter().for_each_with(sender,|s,i_pair| {

                let mut loc_polar_freq = MatrixFull::new([num_auxbas,num_auxbas],0.0);
                let j_state = i_pair[0];
                let k_state = i_pair[1];

                let j_state_eigen = eigenvalues[j_state];
                let j_state_occ = occ_numbers[j_state];
                let mut tmp_matrix = MatrixFull::new([num_auxbas,vir_range.len()],0.0);

                let j_loc_state = j_state - occ_range.start;
                let rimo_j = ri3mo.get_reducing_matrix(j_loc_state).unwrap();

                let k_state_eigen = eigenvalues.get(k_state).unwrap();
                let k_state_occ = occ_numbers.get(k_state).unwrap();
                let zeta = num_spin*(j_state_eigen-k_state_eigen) /
                    ((j_state_eigen-k_state_eigen).powf(2.0) + freq*freq)*
                    (j_state_occ-k_state_occ);

                let k_loc_state = k_state - vir_range.start;
                let from_iter = ri3mo.get_slices(0..num_auxbas, k_loc_state..k_loc_state+1, j_loc_state..j_loc_state+1);
                let to_iter = tmp_matrix.iter_submatrix_mut(0..num_auxbas,k_loc_state..k_loc_state+1);
                to_iter.zip(from_iter).for_each(|(to, from)| {
                    *to = from * zeta
                });
                _dgemm(
                    &tmp_matrix, (0..num_auxbas, 0..vir_range.len()), 'N',
                    &rimo_j, (0..num_auxbas, 0..vir_range.len()), 'T',
                    &mut loc_polar_freq, (0..num_auxbas, 0..num_auxbas),
                    1.0, 1.0
                );

                s.send(loc_polar_freq).unwrap()

            });

            receiver.into_iter().for_each(|loc_polar_freq| {
                polar_freq += loc_polar_freq
            })

        }
    } else {
        panic!("RI3MO should be initialized before the RPA calculations")
    };

    utilities::omp_set_num_threads_wrapper(default_omp_num_threads);

    Ok(polar_freq)

}

fn evaluate_response_serial(scf_data: &SCF, freq: f64) -> anyhow::Result<MatrixFull<f64>> {

    let num_auxbas = scf_data.mol.num_auxbas;
    let num_basis = scf_data.mol.num_basis;
    let num_state = scf_data.mol.num_state;
    let start_mo = scf_data.mol.start_mo;
    let spin_channel = scf_data.mol.spin_channel;
    let num_spin = spin_channel as f64;
    let mut polar_freq = MatrixFull::new([num_auxbas,num_auxbas],0.0);

    if let Some(ri3mo_vec) = &scf_data.ri3mo {
        for i_spin in 0..spin_channel {
            let eigenvector = scf_data.eigenvectors.get(i_spin).unwrap();
            let eigenvalues = scf_data.eigenvalues.get(i_spin).unwrap();
            let occ_numbers = scf_data.occupation.get(i_spin).unwrap();
            let homo = scf_data.homo.get(i_spin).unwrap().clone();
            let lumo = scf_data.lumo.get(i_spin).unwrap().clone();
            let num_occu = homo + 1;

            let (ri3mo, vir_range, occ_range) = ri3mo_vec.get(i_spin).unwrap();

            
            //let mut rimo = riao.ao2mo(eigenvector).unwrap();
            let mut elec_pair: Vec<[usize;2]> = vec![];
            for j_state in start_mo..num_occu {
                    let j_state_eigen = eigenvalues[j_state];
                    let j_state_occ = occ_numbers[j_state];
                    let j_loc_state = j_state - occ_range.start;
                    let rimo_j = ri3mo.get_reducing_matrix(j_loc_state).unwrap();

                    let mut tmp_matrix = MatrixFull::new([num_auxbas,vir_range.len()],0.0);

                    for k_state in lumo..num_state {

                    let k_state_eigen = eigenvalues.get(k_state).unwrap();
                    let k_state_occ = occ_numbers.get(k_state).unwrap();
                    let zeta = num_spin*(j_state_eigen-k_state_eigen) /
                        ((j_state_eigen-k_state_eigen).powf(2.0) + freq*freq)*
                        (j_state_occ-k_state_occ);

                    let k_loc_state = k_state - vir_range.start;
                    let from_iter = ri3mo.get_slices(0..num_auxbas, k_loc_state..k_loc_state+1, j_loc_state..j_loc_state+1);
                    let to_iter = tmp_matrix.iter_submatrix_mut(0..num_auxbas,k_loc_state..k_loc_state+1);
                    to_iter.zip(from_iter).for_each(|(to, from)| {
                        *to = from * zeta
                    });
                }
                _dgemm(
                    &tmp_matrix, (0..num_auxbas, 0..vir_range.len()), 'N',
                    &rimo_j, (0..num_auxbas, 0..vir_range.len()), 'T',
                    &mut polar_freq, (0..num_auxbas, 0..num_auxbas),
                    1.0, 1.0
                );
            }
        }
    } else {
        panic!("RI3MO should be initialized before the RPA calculations")
    };

    Ok(polar_freq)

}

fn evaluate_rpa_integrand(polar_freq: &mut MatrixFull<f64>) -> f64 {
    let mut rpa_c_integrand = 0.0;
    let num_auxbas = polar_freq.size.get(0).unwrap();

    let trace_v_times_polar = polar_freq.iter_diagonal().unwrap()
        .fold(0.0, |acc,value| acc+(*value));

    //let mut tmp_v = polar_freq.get_diagonal_terms_mut().unwrap();
    //tmp_v.iter_mut().for_each(|data| **data -= 1.0);
    polar_freq.iter_diagonal_mut().unwrap().for_each(|data| *data -= 1.0);
    //let mut tmp_v = polar_freq.get_diagonal_terms_mut().unwrap();
    //tmp_v.iter_mut().for_each(|data| **data -= 1.0);
    *polar_freq *= -1.0;

    let v_times_polar = polar_freq.to_matrixfullslicemut().lapack_dgetrf().unwrap();

    //let mut det_v_times_polar = v_times_polar.get_diagonal_terms().unwrap().iter()
    //    .fold(1.0,|acc,value| acc*(*value)
    //);
    let mut det_v_times_polar = v_times_polar.iter_diagonal().unwrap()
        .fold(1.0,|acc,value| acc*(*value)
    );
    if det_v_times_polar<0.0 {println!("WARNING: Determinant of V_TIMES_POLAR is negetive !")};
    //println!("debug {:16.8}, {:16.8}",trace_v_times_polar, det_v_times_polar);


    rpa_c_integrand = det_v_times_polar.abs().log(E) + trace_v_times_polar;

    rpa_c_integrand
}


fn logarithmic_grid(score:[f64;2],num_grids:usize) -> (Vec<f64>, Vec<f64>) {
    let e = std::f64::consts::E;
    let w_0 = 0.01_f64;
    let h = 1.0_f64/(num_grids as f64)*((score[1] - score[0])/w_0).log(e);
    let mut weight = vec![0.0;num_grids];
    let mut abcsia = vec![0.0;num_grids];

    //println!("{:?},{:?}",w_0,h);
    weight.iter_mut().zip(abcsia.iter_mut()).map(|(w,a)| (w,a)).enumerate()
    .for_each(|(i,(w,a))| {
        let ii = i as f64;
        *a = w_0*(exp(ii*h)-1.0);
        *w = h* w_0 * exp(ii*h);
    });
    (abcsia, weight)

}

fn trans_gauss_legendre_grids(omega_max:f64,num_grids:usize) -> (Vec<f64>, Vec<f64>) {
    //specific for the frequence generation
    let score = [-omega_max, omega_max];
    let (s_abcsia, s_weight) = gauss_legendre_grids(score, num_grids);

    let mut weight = vec![0.0;num_grids];
    let mut abcsia = vec![0.0;num_grids];

    let s_grids = s_weight.iter().zip(s_abcsia.iter()).map(|(from_w,from_a)| (from_w,from_a));
    let f_grids = weight.iter_mut().zip(abcsia.iter_mut()).map(|(to_w,to_a)| (to_w,to_a));

    s_grids.zip(f_grids).for_each(|((from_w,from_a),(to_w,to_a))| {
        *to_w = from_w / (1.0-from_a).powf(2.0);
        *to_a = 0.5*(1.0+from_a)/(1.0-from_a);

    });

    (abcsia,weight)
}

fn gauss_legendre_grids(score:[f64;2],num_grids:usize) -> (Vec<f64>, Vec<f64>) {
    let eps = 3E-14;
    let pi = std::f64::consts::PI;
    let m = (num_grids+1)/2;
    let xm = 0.5*(score[1]+score[0]);
    let xl = 0.5*(score[1]-score[0]);
    let nn = num_grids as f64;
    //println!("{:?},{:?},{:?},{:?}",num_grids,m,xm,xl);

    let mut weight = vec![0.0;num_grids];
    let mut abcsia = vec![0.0;num_grids];
    weight[0..m].iter_mut().zip(abcsia[0..m].iter_mut()).map(|(w,a)| (w,a)).enumerate()
    .for_each(|(i,(w,a))| {
        let ii = (i as f64) + 1.0;
        let mut z = cos(pi*(ii-0.25)/(nn+0.5));
        //println!("{:?},{:?}",ii, z);
        let mut z1 = z+10.0*eps;
        let mut pp =0.0;
        while (z-z1).abs() > eps  {
            let mut p1 = 1.0;
            let mut p2 = 0.0;
            let mut p3 = 0.0;
            for j in 0..num_grids {
                let jj = (j as f64) + 1.0;
                p3 = p2;
                p2 = p1;
                p1 = ((2.0*jj-1.0)*z*p2-(jj-1.0)*p3)/jj;
            }
            pp = nn*(z*p1-p2)/(z*z-1.0);
            z1 = z;
            z = z1-p1/pp;
        }
        *a = xm - xl*z;
        *w = 2.0*xl/((1.0-z*z)*pp.powf(2.0));
    });
    let tmp_w = weight[0..m].to_vec();
    tmp_w.iter().zip(weight[m..num_grids].iter_mut().rev()).for_each(|(from,to)| {
        *to = *from
    });
    let tmp_a = abcsia[0..m].to_vec();
    tmp_a.iter().zip(abcsia[m..num_grids].iter_mut().rev()).for_each(|(from,to)| {
        *to = 2.0*xm-from
    });
    (abcsia, weight)
}

#[test]
fn test_gauleg() {
    let (p,w) = gauss_legendre_grids([0.0,1.0], 6);
    println!("{:?}",p);
    println!("{:?}",w);
    let (p,w) = trans_gauss_legendre_grids(1.0, 6);
    println!("{:?}",p);
    println!("{:?}",w);
    let (p,w) = logarithmic_grid([0.0,1.0], 6);
    println!("{:?}",p);
    println!("{:?}",w);
}

#[test]
fn test_get_mut_diagonal_terms() {
    let mut dd = MatrixFull::new([5,5],2.0);
    let mut tmp_v = dd.get_diagonal_terms_mut().unwrap();
    tmp_v.iter_mut().for_each(|data| **data -= 1.0);
    dd.par_self_multiple(-1.0);
    dd.formated_output(5, "full");
    
}