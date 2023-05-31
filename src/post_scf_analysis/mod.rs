use pyrest::dft::DFAFamily;

use crate::ri_pt2::sbge2::{close_shell_sbge2_rayon, open_shell_sbge2_rayon};
use crate::ri_rpa::scsrpa::evaluate_osrpa_correlation_rayon;
use crate::ri_rpa::{evaluate_rpa_correlation, evaluate_rpa_correlation_rayon};
use crate::scf_io::SCF;

pub mod rand_wf_real_space;
pub mod cube_build;
pub mod molden_build;

use crate::ri_pt2::{close_shell_pt2_rayon, open_shell_pt2_rayon};

pub fn post_scf_analysis(scf_data: &SCF) {
    if scf_data.mol.ctrl.output_fchk {
        scf_data.save_fchk_of_gaussian()
    };
    if scf_data.mol.ctrl.fciqmc_dump {
        fciqmc_dump(&scf_data);
    }
    if scf_data.mol.ctrl.output_wfn_in_real_space>0 {
        let np = scf_data.mol.ctrl.output_wfn_in_real_space;
        let slater_determinant = rand_wf_real_space::slater_determinant(&scf_data, np);
        let output = serde_json::to_string(&slater_determinant).unwrap();
        let mut file = std::fs::File::create("./wf_in_real_space.txt").unwrap();
        std::io::Write::write(&mut file, output.as_bytes());
    }

    if scf_data.mol.ctrl.output_cube == true {
        let cube_file = cube_build::get_cube(&scf_data,80);
    }

    if scf_data.mol.ctrl.output_molden == true {
        let molden_file = molden_build::gen_molden(&scf_data);
    }
}

pub fn print_out_dfa(scf_data: &SCF) {
    let dfa = crate::dft::DFA4REST::new_xc(scf_data.mol.spin_channel, scf_data.mol.ctrl.print_level);
    let post_xc_energy = if let Some(grids) = &scf_data.grids {
        dfa.post_xc_exc(&scf_data.mol.ctrl.post_xc, grids, &scf_data.density_matrix, &scf_data.eigenvectors, &scf_data.occupation)
    } else {
        vec![[0.0,0.0]]
    };
    post_xc_energy.iter().zip(scf_data.mol.ctrl.post_xc.iter()).for_each(|(energy, name)| {
        println!("{:<16}: {:16.8} Ha", name, energy[0]+energy[1]);
    });
}

/// NOTE: only support symmetric RI-V tensors
pub fn post_scf_correlation(scf_data: &mut SCF) {
    let spin_channel = scf_data.mol.spin_channel;
    let dfa_family_pos = if let Some(tmp_dfa) = &scf_data.mol.xc_data.dfa_family_pos {
        tmp_dfa.clone()
    } else {crate::dft::DFAFamily::Unknown};

    if let None = scf_data.ri3mo {
        let (occ_range, vir_range) = crate::scf_io::determine_ri3mo_size_for_pt2_and_rpa(&scf_data);
        if scf_data.mol.ctrl.print_level>1 {
            println!("generate RI3MO only for occ_range:{:?}, vir_range:{:?}", &occ_range, &vir_range)
        };
        scf_data.generate_ri3mo_rayon(vir_range, occ_range);
    }

    let mut post_corr: Vec<(crate::dft::DFAFamily, [f64;3])> = vec![];
    scf_data.mol.ctrl.post_correlation.iter().filter(|corr| *corr!=&dfa_family_pos).for_each(|corr| {
        match corr {
            crate::dft::DFAFamily::PT2 => {
                println!("Evaluating the PT2 correlation");
                let energy_post = if spin_channel == 1 {
                    close_shell_pt2_rayon(&scf_data).unwrap()
                } else {
                    open_shell_pt2_rayon(&scf_data).unwrap()
                };
                post_corr.push((crate::dft::DFAFamily::PT2, energy_post));
            },
            crate::dft::DFAFamily::SBGE2 => {
                println!("Evaluating the sBGE2 correlation");
                let energy_post = if spin_channel == 1 {
                    close_shell_sbge2_rayon(&scf_data).unwrap()
                } else {
                    //[0.0,0.0,0.0]
                    open_shell_sbge2_rayon(&scf_data).unwrap()
                };
                post_corr.push((crate::dft::DFAFamily::SBGE2, energy_post));

            },
            crate::dft::DFAFamily::RPA => {
                println!("Evaluating the dRPA correlation");
                let energy_post = evaluate_rpa_correlation_rayon(&scf_data).unwrap();
                post_corr.push((crate::dft::DFAFamily::RPA, [energy_post, 0.0,0.0]));
            },
            crate::dft::DFAFamily::SCSRPA => {
                println!("Evaluating the scsRPA correlation");
                let energy_post = evaluate_osrpa_correlation_rayon(&scf_data).unwrap();
                post_corr.push((crate::dft::DFAFamily::SCSRPA, energy_post));

            }
            _ => {println!("Unknown post-scf correlation methods")}
        }
    });

    println!("----------------------------------------------------------------------");
    println!("{:16}: {:>16}, {:>16}, {:>16}","Methods","Total Corr", "OS Corr", "SS Corr");
    println!("----------------------------------------------------------------------");

    post_corr.iter().for_each(|(name,energy)| {
        println!("{:16}: {:16.8}, {:16.8}, {:16.8}", name.to_name(), energy[0], energy[1], energy[2]);

    });
    println!("----------------------------------------------------------------------");
}

fn fciqmc_dump(scf_data: &SCF) {
    if let Some(ri3fn) = &scf_data.ri3fn {
        // prepare RI-V three-center coefficients for HF orbitals
        for i_spin in 0..scf_data.mol.spin_channel {
            let ri3mo = ri3fn.ao2mo_v01(&scf_data.eigenvectors[i_spin]).unwrap();
            for i in 0.. scf_data.mol.num_state {
                let ri3mo_i = ri3mo.get_reducing_matrix(i).unwrap();
                for j in 0.. scf_data.mol.num_state {
                    let ri3mo_ij = ri3mo_i.get_slice_x(j);
                    for k in 0.. scf_data.mol.num_state {
                        let ri3mo_k = ri3mo.get_reducing_matrix(k).unwrap();
                        for l in 0.. scf_data.mol.num_state {
                            let ri3mo_kl = ri3mo_k.get_slice_x(l);
                            let ijkl = ri3mo_ij.iter().zip(ri3mo_kl.iter())
                                .fold(0.0, |acc, (val1, val2)| acc + val1*val2);
                            if ijkl.abs() > 1.0E-8 {
                                println! ("{:16.8} {:5} {:5} {:5} {:5}",ijkl, i,j,k,l);
                            }
                        }
                    }
                }
            }
        }
    }
}