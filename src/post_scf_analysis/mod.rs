use crate::scf_io::SCF;

pub mod rand_wf_real_space;
pub mod cube_build;
pub mod molden_build;

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