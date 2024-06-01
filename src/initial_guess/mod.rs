use tensors::{MatrixFull, MatrixUpper, BasicMatrix};

use crate::constants::E;
use crate::initial_guess::enxc::effective_nxc_matrix;
use crate::scf_io::{scf, SCFType};
use crate::{molecule_io::Molecule, scf_io::SCF, dft::Grids};

use crate::initial_guess::sap::get_vsap;

use self::sad::initial_guess_from_sad;

//use crate::initial_guess::*;
pub mod sap;
pub mod sad;
pub mod enxc;
mod pyrest_enxc;

enum RESTART {
    HDF5,
    Inherit
}

pub fn initial_guess(scf_data: &mut SCF) {
    // inherit the initial guess from the previous SCF procedure
    if scf_data.mol.ctrl.initial_guess.eq(&"inherit") {
        scf_data.generate_occupation();
        scf_data.generate_density_matrix();
    // import the initial guess density from a hdf5 file
    } else if scf_data.mol.ctrl.external_init_guess && scf_data.mol.ctrl.guessfile_type.eq(&"hdf5") {
        scf_data.density_matrix = initial_guess_from_hdf5guess(&scf_data.mol);
        // for DFT methods, it needs the eigenvectors to generate the hamiltoniam. In consequence, we use the hf method to prepare the eigenvectors from the guess dm
        scf_data.generate_hf_hamiltonian_for_guess();
        //scf_data.generate_hf_hamiltonian();
        if scf_data.mol.ctrl.print_level>0 {println!("Initial guess energy: {:16.8}", scf_data.evaluate_hf_total_energy())};
        scf_data.diagonalize_hamiltonian();
        scf_data.generate_occupation();
        scf_data.generate_density_matrix();
    // import the eigenvalues and eigen vectors from a hdf5 file directly
    } else if scf_data.mol.ctrl.restart && std::path::Path::new(&scf_data.mol.ctrl.chkfile).exists()  {
        if scf_data.mol.ctrl.chkfile_type.eq(&"hdf5") {
            let (eigenvectors, eigenvalues, is_occupation) = initial_guess_from_hdf5chk(&scf_data.mol);

            //=============================
            // for MOM projection
            //=============================
            if scf_data.mol.ctrl.force_state_occupation.len()>0 {
                let restart = scf_data.mol.ctrl.chkfile.clone();
                //let is_exist = scf_data.ref_eigenvectors.contains_key(&restart);
                //if ! is_exist {
                scf_data.ref_eigenvectors.insert(
                    restart, 
                    (eigenvectors.clone(),[0,scf_data.mol.num_basis,scf_data.mol.num_state,scf_data.mol.spin_channel])
                );
                //};
                match scf_data.scftype {
                    SCFType::RHF => {
                        scf_data.mol.ctrl.force_state_occupation.iter().enumerate().for_each(|(i,x)| {
                            if x.get_force_occ() > 2.0 {
                                println!("ERROR: the orbital occupation number for RHF cannot be larger than 2.0");
                                panic!("{}", x.formated_output_check());
                            }
                            if x.get_occ_spin() > 0 {
                                println!("ERROR: the spin is unpolarized for RHF, and thus cannot manipulate the orbitals in BETA spin-channel");
                                panic!("{}", x.formated_output_check());
                            }
                        })
                    },
                    _ => {
                        scf_data.mol.ctrl.force_state_occupation.iter().enumerate().for_each(|(i,x)| {
                            if x.get_force_occ() > 1.0 {
                                println!("ERROR: the orbital occupation number for UHF and ROHF cannot be larger than 1.0");
                                panic!("{}", x.formated_output_check());
                            }
                        })

                    }
                }
            }
            //println!("{:?}", &scf_data.mol.ctrl.auxiliary_reference_states);
            if scf_data.mol.ctrl.auxiliary_reference_states.len() > 0 {
                scf_data.mol.ctrl.auxiliary_reference_states.iter().for_each(|(chkname,global_index)| {
                    println!("{}", chkname);
                    let is_exist = scf_data.ref_eigenvectors.contains_key(chkname);
                    if ! is_exist {
                        let (reference,[num_basis, num_state, spin_channel]) = import_mo_coeff_from_hdf5chkfile(chkname);
                        println!("{},{},{},{},{}", chkname,global_index, num_basis, num_state, spin_channel);
                        scf_data.ref_eigenvectors.insert(chkname.clone(), (reference,[global_index.clone(),num_basis, num_state, spin_channel]));
                    }
                });
 ;           }
            //=============================
            scf_data.eigenvalues = eigenvalues;
            scf_data.eigenvectors = eigenvectors;
            if let Some(occupation) = is_occupation {
                scf_data.occupation = occupation;
            } else {
                scf_data.generate_occupation();
            }
            scf_data.generate_density_matrix();
        } else {
            panic!("WARNNING: at present only hdf5 type check file is supported");
        }
    // generate the machine-learning enxc potential initial guess
    } else if scf_data.mol.ctrl.initial_guess.eq(&"deep_enxc") {
        let mut init_fock = scf_data.h_core.clone();
        if scf_data.mol.spin_channel==1 {
            let mut cur_mol = scf_data.mol.clone();
            let mut effective_hamiltonian = effective_nxc_matrix(&mut cur_mol);
            init_fock.data.iter_mut().zip(effective_hamiltonian.data.iter()).for_each(|(to, from)| {*to += *from});
            scf_data.hamiltonian = [init_fock,MatrixUpper::new(1,0.0)];
        } else {
            panic!("Error: at present the 'deep_enxc' initial guess can only be used for the close-shell problem");
        };
        scf_data.diagonalize_hamiltonian();
        scf_data.generate_occupation();
        scf_data.generate_density_matrix();
        scf_data.generate_hf_hamiltonian();
        let homo_id = scf_data.homo[0];
        let lumo_id = scf_data.lumo[0];
        println!("homo: {}, lumo: {}", &scf_data.eigenvalues[0][homo_id], &scf_data.eigenvalues[0][lumo_id]);
        println!("initial_energy by deep_enxc: {}", scf_data.scf_energy);

    // generate the VSAP initial guess
    } else if scf_data.mol.ctrl.initial_guess.eq(&"vsap") {
        let init_fock = initial_guess_from_vsap(&scf_data.mol,&scf_data.grids);
        if scf_data.mol.spin_channel==1 {
            scf_data.hamiltonian = [init_fock,MatrixUpper::new(1,0.0)];
        } else {
            let init_fock_beta = init_fock.clone();
            scf_data.hamiltonian = [init_fock,init_fock_beta];
        };
        scf_data.diagonalize_hamiltonian();
        scf_data.generate_occupation();
        scf_data.generate_density_matrix();
        //scf_data.generate_hf_hamiltonian();
    } else if scf_data.mol.ctrl.initial_guess.eq(&"sad") {
        scf_data.density_matrix = initial_guess_from_sad(&scf_data.mol);
        //for DFT methods, it needs the eigenvectors to generate the hamiltoniam. In consequence, we use the hf method to prepare the eigenvectors from the guess dm
        //scf_data.generate_hf_hamiltonian_for_guess();
        //if scf_data.mol.ctrl.print_level>0 {println!("Initial guess HF energy: {:16.8}", scf_data.evaluate_hf_total_energy())};
        let original_flag = scf_data.mol.ctrl.use_dm_only;
        scf_data.mol.ctrl.use_dm_only = true;
        scf_data.generate_hf_hamiltonian();
        scf_data.mol.ctrl.use_dm_only = original_flag;
        //println!("{:?}",scf_data.);
        if scf_data.mol.ctrl.print_level>0 {println!("Initial guess HF energy: {:24.16}", scf_data.scf_energy)};

        scf_data.diagonalize_hamiltonian();
        scf_data.generate_occupation();
        scf_data.generate_density_matrix();
        //scf_data.generate_hf_hamiltonian();
        //if scf_data.mol.ctrl.print_level>0 {println!("Initial guess HF energy: {:16.8}", scf_data.scf_energy)};
        //===============================see====================================
    // generate the initial guess from hcore
    } else if scf_data.mol.ctrl.initial_guess.eq(&"hcore") {
        let init_fock = scf_data.h_core.clone();
        if scf_data.mol.spin_channel==1 {
            scf_data.hamiltonian = [init_fock,MatrixUpper::new(1,0.0)];
        } else {
            let init_fock_beta = init_fock.clone();
            scf_data.hamiltonian = [init_fock,init_fock_beta];
        };
        scf_data.diagonalize_hamiltonian();
        scf_data.generate_occupation();
        scf_data.generate_density_matrix();
        scf_data.generate_hf_hamiltonian();
        let homo_id = scf_data.homo[0];
        let lumo_id = scf_data.lumo[0];
        println!("homo: {}, lumo: {}", &scf_data.eigenvalues[0][homo_id], &scf_data.eigenvalues[0][lumo_id]);
        println!("initial_energy: {}", scf_data.scf_energy);
    } else {
        println!("WARNNING: unknown initial_guess method ({}), invoke the \"hcore\" method", &scf_data.mol.ctrl.initial_guess);
        let init_fock = scf_data.h_core.clone();
        if scf_data.mol.spin_channel==1 {
            scf_data.hamiltonian = [init_fock,MatrixUpper::new(1,0.0)];
        } else {
            let init_fock_beta = init_fock.clone();
            scf_data.hamiltonian = [init_fock,init_fock_beta];
        };
        scf_data.diagonalize_hamiltonian();
        scf_data.generate_occupation();
        scf_data.generate_density_matrix();
        scf_data.generate_hf_hamiltonian();
        let homo_id = scf_data.homo[0];
        let lumo_id = scf_data.lumo[0];
        println!("homo: {}, lumo: {}", &scf_data.eigenvalues[0][homo_id], &scf_data.eigenvalues[0][lumo_id]);
        println!("initial_energy: {}", scf_data.scf_energy);

    };
}


pub fn initial_guess_from_hdf5guess(mol: &Molecule) -> Vec<MatrixFull<f64>> {
    if mol.ctrl.print_level>0 {println!("Importing density matrix from external inital guess file")};
    let file = hdf5::File::open(&mol.ctrl.guessfile).unwrap();
    let init_guess = file.dataset("init_guess").unwrap().read_raw::<f64>().unwrap();
    let mut dm = vec![MatrixFull::empty(),MatrixFull::empty()];
    for i_spin in 0..mol.spin_channel {
        let start = (0+i_spin)*mol.num_basis.pow(2);
        let end = (1+i_spin)*mol.num_basis.pow(2);
        dm[i_spin]= MatrixFull::from_vec([mol.num_basis,mol.num_basis],init_guess[start..end].to_vec()).unwrap();
    }
    dm
}

pub fn initial_guess_from_hdf5chk(mol: &Molecule) -> ([MatrixFull<f64>;2],[Vec<f64>;2],Option<[Vec<f64>;2]>) {

    initial_guess_from_hdf5chkfile(&mol.ctrl.chkfile, 
        mol.spin_channel,
        mol.num_state,
        mol.num_basis,
        mol.ctrl.print_level)
}

pub fn initial_guess_from_hdf5chkfile(
    chkname: &str, 
    spin_channel: usize,
    num_state: usize,
    num_basis: usize,
    print_level: usize,
) -> ([MatrixFull<f64>;2],[Vec<f64>;2], Option<[Vec<f64>;2]>) {
    
    //let mut tmp_scf = SCF::new(&mol);
    //tmp_scf.generate_occupation();

    let file = hdf5::File::open(chkname).unwrap();
    let scf = file.group("scf").unwrap();
    let member = scf.member_names().unwrap();
    let e_tot = scf.dataset("e_tot").unwrap().read_scalar::<f64>().unwrap();
    if print_level>1 {
        println!("HDF5 Group: {:?} \nMembers: {:?}", scf, member);
    }
    if print_level>0 {println!("E_tot from chkfile: {:18.10}", e_tot)};
    // importing MO coefficients
    let buf01 = scf.dataset("mo_coeff").unwrap().read_raw::<f64>().unwrap();
    if buf01.len() != num_state*num_basis*spin_channel {
        panic!("Inconsistency happens when importing the molecular coefficients from \'{}\':\n buf01 length: {}, num_state*num_basis*spin_channel: {}", 
        chkname, buf01.len(), num_state*num_basis*spin_channel);
    }
    // importing MO eigenvalues
    let buf02 = scf.dataset("mo_energy").unwrap().read_raw::<f64>().unwrap();
    if buf02.len() != num_state*spin_channel {
        panic!("Inconsistency happens when importing the molecular orbital eigenvalues from \'{}\':\n buf02 length: {}, num_state*spin_channel: {}", 
        chkname, buf02.len(), num_state*spin_channel);
    }
    let mut tmp_eigenvectors: [MatrixFull<f64>;2] = [MatrixFull::empty(),MatrixFull::empty()];
    let mut tmp_eigenvalues: [Vec<f64>;2] = [vec![],vec![]];
    (0..spin_channel).into_iter().for_each(|i| {
        let start = (0 + i)*num_state*num_basis;
        let end = (1 + i)*num_state*num_basis;

        let tmp_eigen = MatrixFull::from_vec([num_state,num_basis], buf01[start..end].to_vec()).unwrap();
        tmp_eigenvectors[i]=tmp_eigen.transpose_and_drop();

        //tmp_scf.eigenvectors[i] = tmp_eigenvectors[i].transpose();

        tmp_eigenvalues[i]=buf02[ (0+i)*num_state..(1+i)*num_state].to_vec();
    });
    if print_level>3 {
        (0..spin_channel).into_iter().for_each(|i| {
            tmp_eigenvectors[i].formated_output(5, "full");
            println!("eigenval {:?}", &tmp_eigenvalues[i]);
        });
    }

    // importing MO occupation
    let is_exist = scf.member_names().unwrap().iter().fold(false, |is_exist, x| x.eq("mo_occupation"));
    let buf03 = if is_exist {
        Some(scf.dataset("mo_occupation").unwrap().read_raw::<f64>().unwrap())
    } else {
        None
    };
    let tmp_occupation = if let Some(tmp_buf03) = &buf03 {
        if tmp_buf03.len() != num_state*spin_channel {
            panic!("Inconsistency happens when importing the molecular orbital occupation from \'{}\':\n buf03 length: {}, num_state*spin_channel: {}", 
            chkname, tmp_buf03.len(), num_state*spin_channel);
            let mut tmp_occupation_0 = [vec![],vec![]];
            (0..spin_channel).into_iter().for_each(|i_spin| {
                tmp_occupation_0[i_spin]=tmp_buf03[ (0+i_spin)*num_state..(1+i_spin)*num_state].to_vec();
            });
            Some(tmp_occupation_0)
        } else {
            None
        }
    } else {
        None
    };

    //tmp_scf.generate_density_matrix();

    //tmp_scf.density_matrix
    (tmp_eigenvectors,tmp_eigenvalues,tmp_occupation)

}

pub fn import_mo_coeff_from_hdf5chkfile(chkname: &str) -> ([MatrixFull<f64>;2], [usize;3]) {
    
    //let mut tmp_scf = SCF::new(&mol);
    //tmp_scf.generate_occupation();

    let file = hdf5::File::open(chkname).unwrap();
    let scf = file.group("scf").unwrap();
    let member = scf.member_names().unwrap();

    let num_basis = scf.dataset("num_basis").unwrap().read_scalar::<usize>().unwrap();
    let num_state = scf.dataset("num_state").unwrap().read_scalar::<usize>().unwrap();
    let spin_channel = scf.dataset("spin_channel").unwrap().read_scalar::<usize>().unwrap();

    // importing MO coefficients
    let buf01 = scf.dataset("mo_coeff").unwrap().read_raw::<f64>().unwrap();
    if buf01.len() != num_state*num_basis*spin_channel {
        panic!("Inconsistency happens when importing the molecular coefficients from \'{}\':\n buf01 length: {}, num_state*num_basis*spin_channel: {}", 
        chkname, buf01.len(), num_state*num_basis*spin_channel);
    }
    let mut tmp_eigenvectors: [MatrixFull<f64>;2] = [MatrixFull::empty(),MatrixFull::empty()];

    (0..spin_channel).into_iter().for_each(|i| {
        let start = (0 + i)*num_state*num_basis;
        let end = (1 + i)*num_state*num_basis;

        let tmp_eigen = MatrixFull::from_vec([num_state,num_basis], buf01[start..end].to_vec()).unwrap();
        tmp_eigenvectors[i]=tmp_eigen.transpose_and_drop();

    });

    (tmp_eigenvectors, [num_basis, num_state, spin_channel])

}



pub fn initial_guess_from_vsap(mol: &Molecule, grids: &Option<Grids>) -> MatrixUpper<f64> {
    //time_mark.new_item("SAP", "Generation of SAP initial guess");
    //time_mark.count_start("SAP");

    //let mut tmp_scf = SCF::new(&mol);
    //tmp_scf.generate_occupation();

    if mol.ctrl.print_level>0 {println!("Initial guess from SAP")};
    let mut tmp_mol = mol.clone();
    let mut h_sap = tmp_mol.int_ij_matrixupper(String::from("kinetic"));
    let tmp_v = if let Some(grids) = grids {
        get_vsap(&mol, grids, mol.ctrl.print_level)
    } else {
        panic!("Density grids should be prepared at first for SAP initial guess")
    };

    //tmp_v.formated_output_e(5, "full");

    h_sap.data.iter_mut().zip(tmp_v.iter_matrixupper().unwrap()).for_each(|(t,f)| if f.abs() > 1.0e-8 {*t += f});

    //time_mark.count("SAP");

    h_sap

}

//pub fn initial_guess_from_hcore(mol: &Molecule) -> Vec<MatrixFull<f64>> {
//    let mut tmp_scf = SCF::new(&mol);
//    tmp_scf.generate_occupation();
//    let init_fock = mol.int_ij_matrixupper(String::from("hcore"));
//    let ovlp = mol.int_ij_matrixupper(String::from("hcore"));
//    let (eigenvectors_alpha,eigenvalues_alpha)=init_fock.to_matrixupperslicemut().lapack_dspgvx(ovlp.to_matrixupperslicemut(),mol.num_state).unwrap();
//    (0..mol.spin_channel).into_iter().for_each(|i| {
//        tmp_scf.eigenvectors[i] = eigenvectors_alpha.clone();
//    });
//
//    (0..mol.spin_channel).into_iter().for_each(|i| {
//        tmp_scf.eigenvalues[i] = eigenvalues_alpha.clone();
//    });
//
//    tmp_scf.generate_density_matrix();
//
//    tmp_scf.density_matrix
//}