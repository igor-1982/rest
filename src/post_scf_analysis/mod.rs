pub mod rand_wf_real_space;
pub mod cube_build;
pub mod molden_build;
pub mod mulliken;
pub mod strong_correlation_correction;

use std::path::Path;
use rest_libcint::prelude::int1e_r;
use tensors::{MathMatrix, MatrixFull, RIFull};

use crate::constants::{ANG, AU2DEBYE, SPECIES_INFO};
use crate::dft::DFAFamily;
use crate::geom_io::get_mass_charge;
use crate::grad::{formated_force, formated_force_ev, numerical_force};
use crate::ri_pt2::sbge2::{close_shell_sbge2_rayon, open_shell_sbge2_rayon, close_shell_sbge2_detailed_rayon, open_shell_sbge2_detailed_rayon};
use crate::ri_rpa::scsrpa::{evaluate_osrpa_correlation_rayon, evaluate_spin_response_rayon, evaluate_special_radius_only};
use crate::ri_rpa::{evaluate_rpa_correlation, evaluate_rpa_correlation_rayon};
use crate::scf_io::SCF;
use crate::ri_pt2::{close_shell_pt2_rayon, open_shell_pt2_rayon};
use crate::utilities::TimeRecords;

use self::molden_build::{gen_header, gen_molden};
use self::strong_correlation_correction::scc15_for_rxdh7;

pub fn post_scf_output(scf_data: &SCF) {
    scf_data.mol.ctrl.outputs.iter().for_each(|output_type| {
        if output_type.eq("fchk") {
            scf_data.save_fchk_of_gaussian();
        } else if output_type.eq("fciqmc_dump") {
            fciqmc_dump(&scf_data);
        } else if output_type.eq("wfn_in_real_space") {
            let np = 100;
            let slater_determinant = rand_wf_real_space::slater_determinant(&scf_data, np);
            let output = serde_json::to_string(&slater_determinant).unwrap();
            let mut file = std::fs::File::create("./wf_in_real_space.txt").unwrap();
            std::io::Write::write(&mut file, output.as_bytes());
        } else if output_type.eq("cube_orb") {
            println!("Now generating the cube files for given orbitals");
            //let grids = scf_data.mol.ctrl.cube_orb_setting[1] as usize;
            //cube_build::get_cube_orb(&scf_data,grids);
            cube_build::get_cube_orb(&scf_data);
        } else if output_type.eq("molden") {
            molden_build::gen_molden(&scf_data);
        } else if output_type.eq("hamiltonian") {
           save_hamiltonian(&scf_data);
        } else if output_type.eq("geometry") {
           save_geometry(&scf_data);
           scf_data.mol.geom.to_xyz("geometry.xyz".to_string());
        } else if output_type.eq("overlap") {
           save_overlap(&scf_data);
        } else if output_type.eq("multiwfn") {
            gen_molden(&scf_data);
        } else if output_type.eq("deeph") {
           save_hamiltonian(&scf_data);
           save_geometry(&scf_data);
           scf_data.mol.geom.to_xyz("geometry.xyz".to_string());
        } else if output_type.eq("dipole") {
            let dp = evaluate_dipole_moment(scf_data, None);

            println!("Dipole Moment in DEBYE: {:16.8}, {:16.8}, {:16.8}", dp[0], dp[1], dp[2]);
        } else if output_type.eq("force") {
            let displace = match scf_data.mol.geom.unit {
                crate::geom_io::GeomUnit::Angstrom => scf_data.mol.ctrl.nforce_displacement/ANG,
                crate::geom_io::GeomUnit::Bohr => scf_data.mol.ctrl.nforce_displacement,
            };
            let (energy, num_force) = numerical_force(scf_data, displace);
            println!("Total atomic forces [a.u.]: ");
            //num_force.formated_output(5, "full");
            println!("{}", formated_force(&num_force, &scf_data.mol.geom.elem));
            println!("Total atomic forces [ev/ang]: ");
            //num_force.formated_output(5, "full");
            println!("{}", formated_force_ev(&num_force, &scf_data.mol.geom.elem));
        }
    });
}

pub fn save_chkfile(scf_data: &SCF) {
    let chkfile= &scf_data.mol.ctrl.chkfile;
    let path = Path::new(chkfile);
    //if path.exists() {std::fs::remove_file(chkfile).unwrap()};
    //let file = hdf5::File::create(chkfile).unwrap();
    //let scf = file.create_group("scf").unwrap();
    let file = if path.exists() {
        hdf5::File::open_rw(chkfile).unwrap()
    } else {
        hdf5::File::create(chkfile).unwrap()
    };
    let is_exist = file.member_names().unwrap().iter().fold(false,|is_exist,x| {is_exist || x.eq("scf")});
    let scf = if is_exist {
        file.group("scf").unwrap()
    } else {
        file.create_group("scf").unwrap()
    };

    let is_exist = scf.member_names().unwrap().iter().fold(false,|is_exist,x| {is_exist || x.eq("e_tot")});
    if is_exist {
        let dataset = scf.dataset("e_tot").unwrap();
        dataset.write(&ndarray::arr0(scf_data.scf_energy));
    } else {
        let builder = scf.new_dataset_builder();
        builder.with_data(&ndarray::arr0(scf_data.scf_energy)
        ).create("e_tot").unwrap();
    }

    let is_exist = scf.member_names().unwrap().iter().fold(false,|is_exist,x| {is_exist || x.eq("num_basis")});
    if is_exist {
        let dataset = scf.dataset("num_basis").unwrap();
        dataset.write(&ndarray::arr0(scf_data.mol.num_basis));
    } else {
        let builder = scf.new_dataset_builder();
        builder.with_data(&ndarray::arr0(scf_data.mol.num_basis)
        ).create("num_basis").unwrap();
    }
    let is_exist = scf.member_names().unwrap().iter().fold(false,|is_exist,x| {is_exist || x.eq("spin_channel")});
    if is_exist {
        let dataset = scf.dataset("spin_channel").unwrap();
        dataset.write(&ndarray::arr0(scf_data.mol.spin_channel));
    } else {
        let builder = scf.new_dataset_builder();
        builder.with_data(&ndarray::arr0(scf_data.mol.spin_channel)
        ).create("spin_channel").unwrap();
    }

    let is_exist = scf.member_names().unwrap().iter().fold(false,|is_exist,x| {is_exist || x.eq("num_state")});
    if is_exist {
        let dataset = scf.dataset("num_state").unwrap();
        dataset.write(&ndarray::arr0(scf_data.mol.num_state));
    } else {
        let builder = scf.new_dataset_builder();
        builder.with_data(&ndarray::arr0(scf_data.mol.num_state)
        ).create("num_state").unwrap();
    }

    let is_exist = scf.member_names().unwrap().iter().fold(false,|is_exist,x| {is_exist || x.eq("mo_coeff")});
    let mut eigenvectors: Vec<f64> = vec![];
    for i_spin in 0..scf_data.mol.spin_channel {
        let tmp_eigenvectors = scf_data.eigenvectors[i_spin].transpose();
        eigenvectors.extend(tmp_eigenvectors.data.iter());
    }
    if is_exist {
        let dataset = scf.dataset("mo_coeff").unwrap();
        dataset.write(&ndarray::arr1(&eigenvectors));
    } else {
        let builder = scf.new_dataset_builder();
        builder.with_data(&ndarray::arr1(&eigenvectors)).create("mo_coeff");
    }

    let mut eigenvalues: Vec<f64> = vec![];
    for i_spin in 0..scf_data.mol.spin_channel {
        eigenvalues.extend(scf_data.eigenvalues[i_spin].iter());
    }
    if is_exist {
        let dataset = scf.dataset("mo_energy").unwrap();
        dataset.write(&ndarray::arr1(&eigenvalues));
    } else {
        let builder = scf.new_dataset_builder();
        builder.with_data(&ndarray::arr1(&eigenvalues)).create("mo_energy");
    }

    let is_exist = scf.member_names().unwrap().iter().fold(false,|is_exist,x| {is_exist || x.eq("mo_occupation")});
    let mut occ: Vec<f64> = vec![];
    for i_spin in 0..scf_data.mol.spin_channel {
        occ.extend(scf_data.occupation[i_spin].iter());
    }
    if is_exist {
        let dataset = scf.dataset("mo_occupation").unwrap();
        dataset.write(&ndarray::arr1(&occ));
    } else {
        let builder = scf.new_dataset_builder();
        builder.with_data(&ndarray::arr1(&occ)).create("mo_occupation");
    }

    file.close();
}

pub fn save_hamiltonian(scf_data: &SCF) {
    let chkfile= &scf_data.mol.ctrl.chkfile;
    let path = Path::new(chkfile);
    let file = if path.exists() {
        hdf5::File::open_rw(chkfile).unwrap()
    } else {
        hdf5::File::create(chkfile).unwrap()
    };
    let is_exist = file.member_names().unwrap().iter().fold(false,|is_exist,x| {is_exist || x.eq("scf")});
    let scf = if is_exist {
        file.group("scf").unwrap()
    } else {
        file.create_group("scf").unwrap()
    };
    let mut hamiltonians: Vec<f64> = vec![];
    for i_spin in 0..scf_data.mol.spin_channel {
        let tmp_eigenvectors = scf_data.hamiltonian[i_spin].to_matrixfull().unwrap();
        hamiltonians.extend(scf_data.eigenvalues[i_spin].iter());
    }
    let is_hamiltonian = scf.member_names().unwrap().iter().fold(false,|is_exist,x| {is_exist || x.eq("hamiltonian")});
    if is_hamiltonian {
        let dataset = scf.dataset("hamiltonian").unwrap();
        dataset.write(&ndarray::arr1(&hamiltonians));
    } else {
        let builder = scf.new_dataset_builder();
        builder.with_data(&ndarray::arr1(&hamiltonians)).create("hamiltonian");
    };
    file.close();
}

pub fn save_overlap(scf_data: &SCF) {
    let chkfile= &scf_data.mol.ctrl.chkfile;
    let path = Path::new(chkfile);
    let file = if path.exists() {
        hdf5::File::open_rw(chkfile).unwrap()
    } else {
        hdf5::File::create(chkfile).unwrap()
    };
    let is_exist = file.member_names().unwrap().iter().fold(false,|is_exist,x| {is_exist || x.eq("scf")});
    let scf = if is_exist {
        file.group("scf").unwrap()
    } else {
        file.create_group("scf").unwrap()
    };
    let mut overlap: Vec<f64> = vec![];
    let overlap = scf_data.ovlp.to_matrixfull().unwrap();
    let is_exist = scf.member_names().unwrap().iter().fold(false,|is_exist,x| {is_exist || x.eq("overlap")});
    if is_exist {
        let dataset = scf.dataset("overlap").unwrap();
        dataset.write(&ndarray::arr1(&overlap.data));
    } else {
        let builder = scf.new_dataset_builder();
        builder.with_data(&ndarray::arr1(&overlap.data)).create("overlap");
    };
    file.close();
}

pub fn save_geometry(scf_data: &SCF) {
    let ang = crate::constants::ANG;
    let chkfile= &scf_data.mol.ctrl.chkfile;
    let path = Path::new(chkfile);
    let file = if path.exists() {
        hdf5::File::open_rw(chkfile).unwrap()
    } else {
        hdf5::File::create(chkfile).unwrap()
    };
    let is_geom = file.member_names().unwrap().iter().fold(false,|is_exist,x| {is_exist || x.eq("geom")});
    let geom = if is_geom {
        file.group("geom").unwrap()
    } else {
        file.create_group("geom").unwrap()
    };
    let mass_charge = get_mass_charge(&scf_data.mol.geom.elem);
    let mut geometry: Vec<(f64,f64,f64,f64)> = vec![];
    mass_charge.iter().zip(scf_data.mol.geom.position.iter_columns_full()).for_each(|(mass_charge, position)| {
        geometry.push((mass_charge.1,position[0]*ang,position[1]*ang,position[2]*ang));
    });

    let is_geom = geom.member_names().unwrap().iter().fold(false,|is_exist,x| {is_exist || x.eq("position")});
    if is_geom {
        let dataset = geom.dataset("position").unwrap();
        dataset.write(&ndarray::arr1(&geometry));
    } else {
        let builder = geom.new_dataset_builder();
        builder.with_data(&ndarray::arr1(&geometry)).create("position");
    }
    file.close();
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

pub fn post_ai_correction(scf_data: &mut SCF) -> Option<Vec<f64>> {
    let xc_method = &scf_data.mol.ctrl.xc.to_lowercase();
    let post_ai_corr = &scf_data.mol.ctrl.post_ai_correction.to_lowercase();
    let mut scc = 0.0;
    if post_ai_corr.eq("scc15") && xc_method.eq("r-xdh7") {
        scc = scc15_for_rxdh7(scf_data);
        let total_energy = scf_data.energies.get("xdh_energy").unwrap()[0];
        if scf_data.mol.ctrl.print_level>0 {
            println!("E(R-xDH7-SCC15): {:16.8} Ha", total_energy + scc);
        }
        //scf_data.energies.insert("scc23".to_string(), vec![scc]);
        return Some(vec![scc])
    };
    None
}

/// NOTE: only support symmetric RI-V tensors
pub fn post_scf_correlation(scf_data: &mut SCF) {

    let mut timerecords = TimeRecords::new();
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
                timerecords.new_item("PT2", "the PT2 calculation");
                timerecords.count_start("PT2");
                println!("Evaluating the PT2 correlation");
                let energy_post = if spin_channel == 1 {
                    close_shell_pt2_rayon(&scf_data).unwrap()
                } else {
                    open_shell_pt2_rayon(&scf_data).unwrap()
                };
                post_corr.push((crate::dft::DFAFamily::PT2, energy_post));
                timerecords.count("PT2");
            },
            crate::dft::DFAFamily::SBGE2 => {
                timerecords.new_item("sBGE2", "the sBGE2 calculation");
                timerecords.count_start("sBGE2");
                println!("Evaluating the sBGE2 correlation");
                let energy_post = if spin_channel == 1 {
                    close_shell_sbge2_rayon(&scf_data).unwrap()
                } else {
                    //[0.0,0.0,0.0]
                    open_shell_sbge2_rayon(&scf_data).unwrap()
                };
                post_corr.push((crate::dft::DFAFamily::SBGE2, energy_post));
                timerecords.count("sBGE2");

            },
            crate::dft::DFAFamily::RPA => {
                timerecords.new_item("dRPA", "the dRPA calculation");
                timerecords.count_start("dRPA");
                println!("Evaluating the dRPA correlation");
                let energy_post = evaluate_rpa_correlation_rayon(&scf_data).unwrap();
                post_corr.push((crate::dft::DFAFamily::RPA, [energy_post, 0.0,0.0]));
                timerecords.count("dRPA");
            },
            crate::dft::DFAFamily::SCSRPA => {
                timerecords.new_item("SCSRPA", "the scsRPA calculation");
                timerecords.count_start("SCSRPA");
                println!("Evaluating the scsRPA correlation");
                let energy_post = evaluate_osrpa_correlation_rayon(&scf_data).unwrap();
                post_corr.push((crate::dft::DFAFamily::SCSRPA, energy_post));
                timerecords.count("SCSRPA");
            }
            _ => {println!("Unknown post-scf correlation methods")}
        }
    });

    if scf_data.mol.ctrl.print_level>0 {
        println!("----------------------------------------------------------------------");
        println!("{:16}: {:>16}, {:>16}, {:>16}","Methods","Total Corr", "OS Corr", "SS Corr");
        println!("----------------------------------------------------------------------");

        post_corr.iter().for_each(|(name,energy)| {
            println!("{:16}: {:16.8}, {:16.8}, {:16.8}", name.to_name(), energy[0], energy[1], energy[2]);
        });
        println!("----------------------------------------------------------------------");
        if scf_data.mol.ctrl.print_level>1 {timerecords.report_all()};
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


//pub fn test_hdf5_string() {
//    let file = hdf5::File::create("test_string").unwrap();
//    let geom = file.group("geom").unwrap_or(file.create_group("geom").unwrap());
//    let dd = vec!["string1", "string2"];
//    let builder = geom.new_dataset_builder();
//    builder.with_data(&ndarray::arr1(&dd)).create("elem");
//    file.close();
//}

// evaluate the dipole moment based on the converged density matrix (dm): scf_data.density_matrix
// the dipole moment of the nuclear part (nucl_dip) is given by scf_data.mol.geom.evaluate_dipole_moment()
// the dipole moment of the atomic orbitals (ao_dip) is given by scf_data.mol.int_ij_matrixuppers()
// the dipole moment of the electronic part (el_dip) is given ('ij,ji', ao_dip[x], dm)
pub fn evaluate_dipole_moment(scf_data: &SCF, orig: Option<[f64;3]>) -> [f64;3] {

    let (mut tot_dip, mass_tot) = scf_data.mol.geom.evaluate_dipole_moment(None);

    let mut dm = scf_data.density_matrix[0].clone();
    if scf_data.mol.spin_channel == 2 {
        dm.self_add(&scf_data.density_matrix[1]);
    }

    let mut cint_data = scf_data.mol.initialize_cint(false);

    let p_orig = cint_data.get_common_origin();

    let r_orig: [f64;3] = if let Some(u_orig) = orig {
        u_orig.try_into().unwrap()
    } else {
        p_orig.clone()
    };
    cint_data.set_common_origin(&r_orig);

    let (out, out_shape)= cint_data.integral_s1::<int1e_r>(None);
    //let mut out_shape_1 = [0;3];
    //out_shape_1.iter_mut().zip(out_shape.iter()).for_each(|(out_shape_1, &out_shape)| {*out_shape_1 = out_shape});

    let ao_dip = RIFull::from_vec(out_shape.try_into().unwrap(), out).unwrap();

    let mut el_dip = [0.0;3];
    for i in 0..3 {
        let ao_dip_tmp = ao_dip.get_reducing_matrix(i).unwrap();
        el_dip[i] = dm.iter_columns_full().zip(ao_dip_tmp.iter_columns_full()).fold(0.0,|acc_c,(dm_col, ao_dip_col)| {
            let acc_r = dm_col.iter().zip(ao_dip_col.iter()).fold(0.0, |acc_r, (dm_val, ao_dip_val)| {acc_r + dm_val*ao_dip_val});
            acc_c + acc_r
        });
    }

    cint_data.set_common_origin(&p_orig);

    //nucl_dip.iter().zip(el_dip.iter()).map(|(nucl, el)| (*nucl - *el)*AU2DEBYE).collect::<Vec<f64>>()

    tot_dip.iter_mut().zip(el_dip.iter()).for_each(|(nucl, el)| *nucl = (*nucl - *el)*AU2DEBYE);

    tot_dip

}

