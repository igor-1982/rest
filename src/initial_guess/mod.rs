use tensors::{MatrixFull, MatrixUpper};

use crate::{molecule_io::Molecule, scf_io::SCF, dft::Grids, get_vsap};

use self::sad::initial_guess_from_sad;

//use crate::initial_guess::*;
pub mod sap;
pub mod sad;



pub fn initial_guess(scf_data: &mut SCF) {
    // import the initial guess density from a hdf5 file
    if scf_data.mol.ctrl.external_init_guess && scf_data.mol.ctrl.guessfile_type.eq(&"hdf5") {
        scf_data.density_matrix = initial_guess_from_hdf5guess(&scf_data.mol);
        // for DFT methods, it needs the eigenvectors to generate the hamiltoniam. In consequence, we use the hf method to prepare the eigenvectors from the guess dm
        scf_data.generate_hf_hamiltonian_for_guess();
        //scf_data.generate_hf_hamiltonian();
        println!("Initial guess HF energy: {:16.8}", scf_data.evaluate_hf_total_energy());
        scf_data.diagonalize_hamiltonian();
        scf_data.generate_density_matrix();

    // import the eigenvalues and eigen vectors from a hdf5 file directly
    } else if scf_data.mol.ctrl.restart 
        && std::path::Path::new(&scf_data.mol.ctrl.chkfile).exists() 
        && scf_data.mol.ctrl.chkfile_type.eq(&"hdf5") {
        let (eigenvectors, eigenvalues) = initial_guess_from_hdf5chk(&scf_data.mol);
        scf_data.eigenvalues = eigenvalues;
        scf_data.eigenvectors = eigenvectors;
        scf_data.generate_density_matrix();
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
        scf_data.generate_density_matrix();
        scf_data.generate_hf_hamiltonian();
    } else if scf_data.mol.ctrl.initial_guess.eq(&"sad") {
        scf_data.density_matrix = initial_guess_from_sad(&scf_data.mol);
        // for DFT methods, it needs the eigenvectors to generate the hamiltoniam. In consequence, we use the hf method to prepare the eigenvectors from the guess dm
        //for i in 0..scf_data.mol.spin_channel {
        //    println!("debug {}", if i==0 {"Alpha"} else {"Beta"});
        //    scf_data.density_matrix[i].formated_output(5, "full");
        //    scf_data.hamiltonian[i] = scf_data.h_core.clone();
        //}
        scf_data.generate_hf_hamiltonian_for_guess();
        //scf_data.generate_hf_hamiltonian();
        println!("Initial guess HF energy: {:16.8}", scf_data.evaluate_hf_total_energy());
        scf_data.diagonalize_hamiltonian();
        scf_data.generate_density_matrix();
    // generate the initial guess from hcore
    } else {
        let init_fock = scf_data.h_core.clone();
        if scf_data.mol.spin_channel==1 {
            scf_data.hamiltonian = [init_fock,MatrixUpper::new(1,0.0)];
        } else {
            let init_fock_beta = init_fock.clone();
            scf_data.hamiltonian = [init_fock,init_fock_beta];
        };
        scf_data.diagonalize_hamiltonian();
        scf_data.generate_density_matrix();
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

pub fn initial_guess_from_hdf5chk(mol: &Molecule) -> ([MatrixFull<f64>;2],[Vec<f64>;2]) {
    
    //let mut tmp_scf = SCF::new(&mol);
    //tmp_scf.generate_occupation();

    let file = hdf5::File::open(&mol.ctrl.chkfile).unwrap();
    let scf = file.group("scf").unwrap();
    let member = scf.member_names().unwrap();
    let e_tot = scf.dataset("e_tot").unwrap().read_scalar::<f64>().unwrap();
    if mol.ctrl.print_level>1 {
        println!("HDF5 Group: {:?} \nMembers: {:?}", scf, member);
    }
    if mol.ctrl.print_level>0 {println!("E_tot from chkfile: {:18.10}", e_tot)};
    let buf01 = scf.dataset("mo_coeff").unwrap().read_raw::<f64>().unwrap();
    let buf02 = scf.dataset("mo_energy").unwrap().read_raw::<f64>().unwrap();
    let mut tmp_eigenvectors: [MatrixFull<f64>;2] = [MatrixFull::empty(),MatrixFull::empty()];
    let mut tmp_eigenvalues: [Vec<f64>;2] = [vec![],vec![]];
    (0..mol.spin_channel).into_iter().for_each(|i| {
        let start = (0 + i)*mol.num_state*mol.num_basis;
        let end = (1 + i)*mol.num_state*mol.num_basis;

        let tmp_eigen = MatrixFull::from_vec([mol.num_state,mol.num_basis], buf01[start..end].to_vec()).unwrap();
        tmp_eigenvectors[i]=tmp_eigen.transpose_and_drop();

        //tmp_scf.eigenvectors[i] = tmp_eigenvectors[i].transpose();

        tmp_eigenvalues[i]=buf02[ (0+i)*mol.num_state..(1+i)*mol.num_state].to_vec();
    });
    if mol.ctrl.print_level>1 {
        (0..mol.spin_channel).into_iter().for_each(|i| {
            tmp_eigenvectors[i].formated_output(5, "full");
            println!("eigenval {:?}", &tmp_eigenvalues[i]);
        });
    }

    //tmp_scf.generate_density_matrix();

    //tmp_scf.density_matrix
    (tmp_eigenvectors,tmp_eigenvalues)

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

    h_sap.data.iter_mut().zip(tmp_v.iter_matrixupper().unwrap()).for_each(|(t,f)| *t += f);

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