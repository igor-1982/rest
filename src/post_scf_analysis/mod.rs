pub mod rand_wf_real_space;
pub mod cube_build;
pub mod molden_build;

use std::path::Path;
use crate::constants::SPECIES_INFO;
use crate::dft::DFAFamily;
use crate::geom_io::get_mass_charge;
use crate::ri_pt2::sbge2::{close_shell_sbge2_rayon, open_shell_sbge2_rayon};
use crate::ri_rpa::scsrpa::evaluate_osrpa_correlation_rayon;
use crate::ri_rpa::{evaluate_rpa_correlation, evaluate_rpa_correlation_rayon};
use crate::scf_io::SCF;
use crate::ri_pt2::{close_shell_pt2_rayon, open_shell_pt2_rayon};
use crate::utilities::TimeRecords;

use self::molden_build::{gen_header, gen_molden};

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
        } else if output_type.eq("cube") {
            cube_build::get_cube(&scf_data,80);
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
        //builder.with_data_as(&scf_data.scf_energy,f64).create("e_tot");
        builder.with_data(&ndarray::arr0(scf_data.scf_energy)
        ).create("e_tot").unwrap();
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

    println!("----------------------------------------------------------------------");
    println!("{:16}: {:>16}, {:>16}, {:>16}","Methods","Total Corr", "OS Corr", "SS Corr");
    println!("----------------------------------------------------------------------");

    post_corr.iter().for_each(|(name,energy)| {
        println!("{:16}: {:16.8}, {:16.8}, {:16.8}", name.to_name(), energy[0], energy[1], energy[2]);
    });
    println!("----------------------------------------------------------------------");
    if scf_data.mol.ctrl.print_level>1 {timerecords.report_all()};
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