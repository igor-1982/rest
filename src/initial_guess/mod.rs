#![warn(unused_imports)]
use tensors::{MatrixFull, MatrixUpper};
use crate::initial_guess::enxc::effective_nxc_matrix;
#[cfg(feature = "mpi")]
use crate::mpi_io::{mpi_broadcast_matrixfull, mpi_broadcast_vector};
use crate::mpi_io::MPIOperator;
use crate::scf_io::{SCFType};
use crate::{molecule_io::Molecule, scf_io::SCF, dft::Grids};
use crate::initial_guess::sap::get_vsap;
use self::sad::initial_guess_from_sad;
use log::{self, info, warn, debug, trace, LevelFilter};

pub mod sap;
pub mod sad;
pub mod enxc;
pub mod proj;
pub mod pao;
mod pyrest_enxc;
use crate::fileop::chkfile;

enum RESTART {
    HDF5,
    Inherit
}

pub fn initial_guess(scf_data: &mut SCF, mpi_operator: &Option<MPIOperator>) {
    // inherit the initial guess from the previous SCF procedure
    if scf_data.mol.ctrl.initial_guess.eq(&"inherit") {
        scf_data.generate_occupation();
        scf_data.generate_density_matrix();
    // import the initial guess from guessfile
    } else if scf_data.mol.ctrl.external_init_guess.is_some() { match scf_data.mol.ctrl.external_init_guess.as_ref().unwrap().as_str() {
        "guessfile" => {
            assert!(std::path::Path::new(&scf_data.mol.ctrl.guessfile).exists(), "The specified guessfile is missing \n({})", &scf_data.mol.ctrl.guessfile);
            assert!(scf_data.mol.ctrl.guessfile_type.eq(&"hdf5"), "at present only hdf5 type guess file is supported");
            let file = hdf5::File::open(&scf_data.mol.ctrl.guessfile).unwrap();
            if chkfile::has_dm(&file) {
                scf_data.density_matrix = initial_guess_from_hdf5guess(&scf_data.mol);
                // for DFT methods, it needs the eigenvectors to generate the hamiltonian. In consequence, we use the hf method to prepare the eigenvectors from the guess dm
                scf_data.generate_hf_hamiltonian_for_guess();
                scf_data.grad_dm = scf_data.get_grad_dm();
                //scf_data.generate_hf_hamiltonian();
                info!("Initial guess energy: {:16.8}", scf_data.evaluate_hf_total_energy());
                scf_data.diagonalize_hamiltonian(mpi_operator);
                scf_data.generate_occupation();
                scf_data.generate_density_matrix();
            } else if chkfile::has_mo_coeff(&file) {
                info!("Read MO coefficients from guessfile: {}", &scf_data.mol.ctrl.guessfile);
                // let (eigenvectors, eigenvalues, is_occupation) = initial_guess_from_hdf5chk(
                //     &scf_data.mol, &scf_data.scftype, &scf_data.mol.ctrl.guessfile);
                update_scf_from_hdf5chk(scf_data, scf_data.mol.ctrl.guessfile.clone());
            }
        },
        "chkfile" => {
            // import the eigenvalues and eigen vectors from chkfile
            assert!(std::path::Path::new(&scf_data.mol.ctrl.chkfile).exists(), "The specified chkfile is missing \n({})", &scf_data.mol.ctrl.chkfile);
            assert!(scf_data.mol.ctrl.chkfile_type.eq(&"hdf5"), "at present only hdf5 type check file is supported");
            info!("Read MO coefficients from chkfile: {}", &scf_data.mol.ctrl.chkfile);
            warn!("However, the chkfile will be overwritten in the SCF procedure.");
            warn!("To prevent that, use `guessfile = your_chkfile` instead");
            update_scf_from_hdf5chk(scf_data, scf_data.mol.ctrl.chkfile.clone());
        }
        _ => {
            panic!("Error: unknown external initial guess file type ({})", &scf_data.mol.ctrl.external_init_guess.as_ref().unwrap());
        }
    }
    // generate the machine-learning enxc potential initial guess
    } else if scf_data.mol.ctrl.initial_guess.eq(&"deep_enxc") {
        let mut init_fock = scf_data.h_core.clone();
        if scf_data.mol.spin_channel==1 {
            let mut cur_mol = scf_data.mol.clone();
            let mut effective_hamiltonian = effective_nxc_matrix(&mut cur_mol);
            init_fock.data.iter_mut().zip(effective_hamiltonian.data.iter()).for_each(|(to, from)| {*to += *from});
            scf_data.hamiltonian = [init_fock,MatrixUpper::new(1,0.0)];
            //scf_data.hamiltonian[0].formated_output(10, "full");
        } else {
            panic!("Error: at present the 'deep_enxc' initial guess is only available for close-shell calculations");
        };
        scf_data.diagonalize_hamiltonian(mpi_operator);
        scf_data.generate_occupation();
        scf_data.generate_density_matrix();
        scf_data.generate_hf_hamiltonian(mpi_operator);
        let homo_id = scf_data.homo[0];
        let lumo_id = scf_data.lumo[0];
        info!("homo: {}, lumo: {}", &scf_data.eigenvalues[0][homo_id], &scf_data.eigenvalues[0][lumo_id]);
        info!("initial_energy by deep_enxc: {}", scf_data.scf_energy);

    // generate the VSAP initial guess
    } else if scf_data.mol.ctrl.initial_guess.eq(&"vsap") {
        let init_fock = initial_guess_from_vsap(&scf_data.mol,&scf_data.grids);
        if let SCFType::ROHF = scf_data.scftype {
            scf_data.roothaan_hamiltonian = Some(init_fock);
        } else if scf_data.mol.spin_channel==1 {
            scf_data.hamiltonian = [init_fock,MatrixUpper::new(1,0.0)];
        } else {
            let init_fock_beta = init_fock.clone();
            scf_data.hamiltonian = [init_fock,init_fock_beta];
        };
        scf_data.diagonalize_hamiltonian(mpi_operator);
        scf_data.generate_occupation();
        scf_data.generate_density_matrix();
        //scf_data.generate_hf_hamiltonian();
    } else if scf_data.mol.ctrl.initial_guess.eq(&"sad")
        || scf_data.mol.ctrl.initial_guess.eq(&"pao") {
        // The PAO guess reuses the SAD atomic SCF step and hands the atomic guess orbitals to
        // the SCF, per spin channel. When it is not available (ghost basis, ISDF enabled, or a
        // construction failure) the branch falls through to plain SAD below.
        let mut pao_done = false;
        if scf_data.mol.ctrl.initial_guess.eq(&"pao") && !scf_data.mol.ctrl.use_isdf {
            match crate::initial_guess::pao::initial_guess_from_pao(&scf_data.mol) {
                Some((mut eigv, mut occ)) => {
                    #[cfg(feature = "mpi")]
                    if let Some(mpi_op) = mpi_operator {
                        for i_spin in 0..scf_data.mol.spin_channel {
                            if !eigv[i_spin].data.is_empty() {
                                mpi_broadcast_matrixfull(&mpi_op.world, &mut eigv[i_spin], 0);
                            }
                            if !occ[i_spin].is_empty() {
                                mpi_broadcast_vector(&mpi_op.world, &mut occ[i_spin], 0);
                            }
                        }
                    }
                    let ncols = crate::scf_io::util::occupied_orbital_count(&occ[0]);
                    let ncols_beta = if scf_data.mol.spin_channel == 2 {
                        crate::scf_io::util::occupied_orbital_count(&occ[1])
                    } else {
                        0
                    };
                    scf_data.eigenvectors = [eigv[0].clone(), eigv[1].clone()];
                    scf_data.occupation = [occ[0].clone(), occ[1].clone()];
                    scf_data.generate_density_matrix();
                    // The guess carries occupied orbitals, so every Fock build here and below
                    // takes the orbital-based (pair-screened) RI-K instead of the dense dm-only
                    // kernel. One Roothaan step then runs on that Fock: it is cheap on the orbital
                    // path and it makes the effective starting density the same kind of object
                    // SAD hands over (idempotent, occupations taken from `num_elec`), which keeps
                    // the downstream SCF trajectory and the converged density bit-comparable.
                    scf_data.mol.ctrl.use_dm_only = false;
                    scf_data.generate_hf_hamiltonian(mpi_operator);
                    scf_data.diagonalize_hamiltonian(mpi_operator);
                    scf_data.generate_occupation();
                    scf_data.generate_density_matrix();
                    if ncols_beta > 0 {
                        info!("PAO guess: {} alpha and {} beta occupied atomic orbitals drive the first Fock build", ncols, ncols_beta);
                    } else {
                        info!("PAO guess: {} occupied atomic orbitals drive the first Fock build", ncols);
                    }
                    pao_done = true;
                },
                None => {
                    warn!("initial_guess = pao is not available for this system, falling back to SAD");
                }
            }
        }
        if !pao_done {
        if !scf_data.mol.ctrl.use_isdf {
            let cur_log_level = log::max_level();
            log::set_max_level(LevelFilter::Info);
            scf_data.density_matrix = initial_guess_from_sad(&scf_data.mol, mpi_operator);
            log::set_max_level(cur_log_level);
        }
        //for DFT methods, it needs the eigenvectors to generate the hamiltoniam. In consequence, we use the hf method to prepare the eigenvectors from the guess dm
        //scf_data.generate_hf_hamiltonian_for_guess();
        //if scf_data.mol.ctrl.print_level>0 {println!("Initial guess HF energy: {:16.8}", scf_data.evaluate_hf_total_energy())};
        #[cfg(feature = "mpi")]
        if let Some(mpi_op) = mpi_operator {
            mpi_broadcast_matrixfull(&mpi_op.world, &mut scf_data.density_matrix[0], 0);
            mpi_broadcast_matrixfull(&mpi_op.world, &mut scf_data.density_matrix[1], 0);
        }
        // The SAD guess has no orbitals, so the first Fock build runs on the
        // dm-only paths. NOTE: do NOT replace this with pseudo-orbitals
        // (diagonalizing P_SAD): the SAD pseudo-eigenvectors are delocalized,
        // which defeats the pair screening of the RI-K/XC contraction
        // (measured 15-28x slower on ts2/water30) and breaks SCF convergence
        // (fractional occupations, 2026-10-21 experiment).
        let original_flag = scf_data.mol.ctrl.use_dm_only;
        scf_data.mol.ctrl.use_dm_only = true;

        //println!("======== IGOR debug for dfa components using SAD density matrix =======");
        //let dfa = crate::dft::DFA4REST::new_xc(scf_data.mol.spin_channel, scf_data.mol.ctrl.print_level);
        //let post_xc_energy = if let Some(grids) = &scf_data.grids {
        //    dfa.post_xc_exc(&scf_data.mol.ctrl.post_xc, grids, &scf_data.density_matrix, &scf_data.eigenvectors, &scf_data.occupation)
        //} else {
        //    vec![[0.0,0.0]]
        //};
        //post_xc_energy.iter().zip(scf_data.mol.ctrl.post_xc.iter()).for_each(|(energy, name)| {
        //    println!("{:<16}: {:16.8} Ha", name, energy[0]+energy[1]);
        //});
        //println!("======= IGOR debug ========");

        scf_data.generate_hf_hamiltonian(mpi_operator);
        scf_data.mol.ctrl.use_dm_only = original_flag;
        //println!("{:?}",scf_data.);
        info!("Initial guess energy using single atom density (SAD): {:24.16}", scf_data.scf_energy);

        scf_data.diagonalize_hamiltonian(mpi_operator);
        scf_data.generate_occupation();
        scf_data.generate_density_matrix();
        //println!("======== IGOR debug for tabulated orbital densities =======");
        //if let Some(grids) = &scf_data.grids {
        //    let dd = numerical_orbital_population(grids, &scf_data.mol);
        //    println!("debug orbital densities: {:?}", &dd);
        //    let dd = numerical_density_rayon(grids, &scf_data.mol, &scf_data.density_matrix);
        //    println!("debug density: {:?}", &dd);
        //}
        //println!("======== IGOR debug for dfa components using SAD density matrix =======");
        //scf_data.generate_hf_hamiltonian(mpi_operator);
        //if scf_data.mol.ctrl.print_level>0 {println!("Initial guess HF energy: {:16.8}", scf_data.scf_energy)};
        //===============================see====================================
        }
    // generate the initial guess from hcore
    } else if scf_data.mol.ctrl.initial_guess.eq(&"hcore") {
        let init_fock = scf_data.h_core.clone();
        if scf_data.mol.spin_channel==1 {
            scf_data.hamiltonian = [init_fock,MatrixUpper::new(1,0.0)];
        } else {
            let init_fock_beta = init_fock.clone();
            scf_data.hamiltonian = [init_fock,init_fock_beta];
        };
        scf_data.diagonalize_hamiltonian(mpi_operator);
        scf_data.generate_occupation();
        scf_data.generate_density_matrix();
        scf_data.generate_hf_hamiltonian(mpi_operator);
        let homo_id = scf_data.homo[0];
        let lumo_id = scf_data.lumo[0];
        info!("homo: {}, lumo: {}", &scf_data.eigenvalues[0][homo_id], &scf_data.eigenvalues[0][lumo_id]);
        info!("initial_energy: {}", scf_data.scf_energy);
    } else {
        warn!("unknown initial_guess method ({}), invoke the \"hcore\" method", &scf_data.mol.ctrl.initial_guess);
        let init_fock = scf_data.h_core.clone();
        if scf_data.mol.spin_channel==1 {
            scf_data.hamiltonian = [init_fock,MatrixUpper::new(1,0.0)];
        } else {
            let init_fock_beta = init_fock.clone();
            scf_data.hamiltonian = [init_fock,init_fock_beta];
        };
        scf_data.diagonalize_hamiltonian(mpi_operator);
        scf_data.generate_occupation();
        scf_data.generate_density_matrix();
        scf_data.generate_hf_hamiltonian(mpi_operator);
        let homo_id = scf_data.homo[0];
        let lumo_id = scf_data.lumo[0];
        info!("homo: {}, lumo: {}", &scf_data.eigenvalues[0][homo_id], &scf_data.eigenvalues[0][lumo_id]);
        info!("initial_energy: {}", scf_data.scf_energy);
    };
}

pub fn update_scf_from_hdf5chk(scf_data: &mut SCF, chkfile: String) {
            let (eigenvectors, eigenvalues, is_occupation) = initial_guess_from_hdf5chk(
                &mut scf_data.mol, &scf_data.scftype, &chkfile);

            //=============================
            // for MOM projection
            //=============================
            if scf_data.mol.ctrl.force_state_occupation.len()>0 {
                let restart = chkfile.clone();
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
                                panic!("ERROR: the orbital occupation number for RHF cannot be larger than 2.0. {}", x.formated_output_check());
                            }
                            if x.get_occ_spin() > 0 {
                                panic!("ERROR: the spin is unpolarized for RHF, and thus cannot manipulate the orbitals in BETA spin-channel. {}", x.formated_output_check());
                            }
                        })
                    },
                    _ => {
                        scf_data.mol.ctrl.force_state_occupation.iter().enumerate().for_each(|(i,x)| {
                            if x.get_force_occ() > 1.0 {
                                panic!("ERROR: the orbital occupation number for UHF and ROHF cannot be larger than 1.0. {}", x.formated_output_check());
                            }
                        })

                    }
                }
            }
            //println!("{:?}", &scf_data.mol.ctrl.auxiliary_reference_states);
            if scf_data.mol.ctrl.auxiliary_reference_states.len() > 0 {
                scf_data.mol.ctrl.auxiliary_reference_states.iter().for_each(|(chkname,global_index)| {
                    debug!("{}", chkname);
                    let is_exist = scf_data.ref_eigenvectors.contains_key(chkname);
                    if ! is_exist {
                        let (reference,[num_basis, num_state, spin_channel]) = import_mo_coeff_from_hdf5chkfile(chkname);
                        debug!("{},{},{},{},{}", chkname,global_index, num_basis, num_state, spin_channel);
                        scf_data.ref_eigenvectors.insert(chkname.clone(), (reference,[global_index.clone(),num_basis, num_state, spin_channel]));
                    }
                });
            }
            //=============================
            scf_data.eigenvalues = eigenvalues;
            scf_data.eigenvectors = eigenvectors;
            if let Some(occupation) = is_occupation {
                scf_data.occupation = occupation;
                // let (homo, lumo) = scf_io::util::get_homo_lumo_from_occ_integer(&scf_data.occupation, &scf_data.scftype);
                // scf_data.homo = homo;
                // scf_data.lumo = lumo;
            } else {
            // Since homo, lumo is needed, re-generate occupation temporarily
                scf_data.generate_occupation();
            }
            // println!("homo {:?}", scf_data.homo);
            // println!("occ {:?}", scf_data.occupation);
            scf_data.generate_density_matrix();
        }

pub fn initial_guess_from_hdf5guess(mol: &Molecule) -> Vec<MatrixFull<f64>> {
    info!("Importing density matrix from external initial guess file");
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

pub fn update_basis_from_hdf5chk(scf_data: &mut SCF) {
    let chkbasis = scf_data.mol.ctrl.basis_path == "chkfile";
    if !chkbasis {
        return;
    }
    if scf_data.mol.ctrl.initial_guess.eq(&"inherit") {
        return;
    } else if scf_data.mol.ctrl.external_init_guess.is_some() { 
        let chkfile = match scf_data.mol.ctrl.external_init_guess.as_ref().unwrap().as_str() {
            "guessfile" => {
                assert!(std::path::Path::new(&scf_data.mol.ctrl.guessfile).exists(), "The specified guessfile is missing \n({})", &scf_data.mol.ctrl.guessfile);
                assert!(scf_data.mol.ctrl.guessfile_type.eq(&"hdf5"), "at present only hdf5 type guess file is supported");
                scf_data.mol.ctrl.guessfile.clone()
            },
            "chkfile" => {
                assert!(std::path::Path::new(&scf_data.mol.ctrl.chkfile).exists(), "The specified chkfile is missing \n({})", &scf_data.mol.ctrl.chkfile);
                assert!(scf_data.mol.ctrl.chkfile_type.eq(&"hdf5"), "at present only hdf5 type check file is supported");
                scf_data.mol.ctrl.chkfile.clone()
            }
            _ => {
                panic!("Error: unknown external initial guess file type ({})", &scf_data.mol.ctrl.external_init_guess.as_ref().unwrap());
            }
        };
    

        info!("taking basis set information from chkfile");
        let (cint_raw_data, ecp_raw, basis4elem, cint_type, fdqc_bas, cint_fdqc) = chkfile::reconstruct_cint_data(&chkfile, Some(&scf_data.mol.geom));
        if let Some((atm, bas, env)) = cint_raw_data {
            if let Some(ct) = cint_type {
                scf_data.mol.cint_type = ct;
            }
            scf_data.mol.set_cint_data(atm, bas, env, ecp_raw, None, basis4elem, fdqc_bas, cint_fdqc);
            scf_data.mol.update_num_elec();
            scf_data.mol.start_mo = scf_data.mol.generate_start_mo(scf_data.mol.ecp_electrons);
            // Fix (chkfile-basis + MPI/size>=2): 轨道基组在 initialize_scf 才经 set_cint_data 整体
            // 替换 cint_env，而辅助基 cint/env 是按替换前的 env 布局建立的 → 2c2e 壳层指针错位
            // → aux V 矩阵 NaN（full-RI 路径 n>=1 即触发；symm/MPI 路径 n>=2 触发，rank 集体处挂起）。
            // 在 env 替换后重建辅助基，使壳层偏移与当前 env 一致。
            if scf_data.mol.ctrl.use_auxbas {
                scf_data.mol.initialize_auxbas();
            }
        } else {
            panic!("Failed to load the basis set information from chkfile");
        }
    }
}

pub fn initial_guess_from_hdf5chk(mol: &mut Molecule, scftype: &SCFType, chkfile: &String) -> ([MatrixFull<f64>;2],[Vec<f64>;2],Option<[Vec<f64>;2]>) {
    match proj::decide_guess(chkfile, &*mol) {
        proj::GuessAction::Refuse(reason) => {
            panic!("Cannot use chkfile '{}' for initial guess: {}", chkfile, reason);
        }
        proj::GuessAction::DirectReuse => {
            let (loaded_eigenvectors, loaded_eigenvalues, loaded_occupation) = import_guess_from_hdf5chkfile(chkfile);
            initial_guess_from_raw(
                loaded_eigenvectors,
                loaded_eigenvalues,
                loaded_occupation.unwrap(),
                scftype,
                mol.num_state,
                mol.num_basis,
                mol.ctrl.print_level
            )
        }
        proj::GuessAction::Project(source) => {
            let (loaded_eigenvectors, loaded_eigenvalues, loaded_occupation) = import_guess_from_hdf5chkfile(chkfile);
            info!("The basis set in chkfile does not match the input basis set (loaded: {} basis, {} MOs; target: {} basis, {} MOs),", source.num_basis, source.num_state, mol.num_basis, mol.num_state);
            info!("invoke basis projection");
            let (mo2, _, occ2) = initial_guess_from_raw(
                loaded_eigenvectors,
                loaded_eigenvalues,
                loaded_occupation.unwrap(),
                scftype,
                source.num_state,
                source.num_basis,
                mol.ctrl.print_level
            );
            let ne = mol.num_elec;
            let nocc = [ne[1].round() as usize, ne[2].round() as usize];
            let mo_range = if matches!(scftype, SCFType::ROHF) {
                // ROKS/ROHF: shared spatial MOs stored in the alpha channel only
                match mol.ctrl.basis_projection.as_str() {
                    "full" => [0..source.num_state, 0..0],
                    _ => [0..nocc[0], 0..0],
                }
            } else {
                match mol.ctrl.basis_projection.as_str() {
                    "full" => [0..source.num_state, 0..source.num_state],
                    _ => [0..nocc[0], 0..nocc[1]],
                }
            };
            let mo = proj::proj_mo(mol, &source, mo2, mo_range);
            (mo, [vec![], vec![]], occ2)
        }
    }
}

pub fn import_guess_from_hdf5chkfile(chkname: &str) -> (Vec<f64>,Vec<f64>, Option<Vec<f64>>) {
    let file = hdf5::File::open(chkname).unwrap();
    let scf = file.group("scf").unwrap();
    let member = scf.member_names().unwrap();
    let e_tot = scf.dataset("e_tot").unwrap().read_raw::<f64>().unwrap()[0];
    debug!("HDF5 Group: {:?} \nMembers: {:?}", scf, member);
    info!("E_tot from chkfile: {:18.10}", e_tot);
    // importing MO coefficients
    let buf01 = scf.dataset("mo_coeff").unwrap().read_raw::<f64>().unwrap();
    // importing MO eigenvalues
    let buf02 = scf.dataset("mo_energy").unwrap().read_raw::<f64>().unwrap();
    // importing MO occupation
    // let is_exist = scf.member_names().unwrap().iter().fold(false, |is_exist, x| x.eq("mo_occupation"));
    let buf03 = if scf.dataset("mo_occ").is_ok() {
        Some(scf.dataset("mo_occ").unwrap().read_raw::<f64>().unwrap())
    } else {
        None
    };
    (buf01, buf02, buf03)
}

#[derive(Clone, Copy)]
enum ImportMode {
    /// 1-channel source spatial MOs → restricted target (RKS → RKS)
    R2R,
    /// 1-channel source spatial MOs → unrestricted target (RKS → UKS)
    R2U,
    /// 1-channel source spatial MOs → restricted-open target (ROKS → ROKS)
    RO2RO,
    /// 1-channel source spatial MOs → unrestricted target (ROKS → UKS)
    RO2U,
    /// 2-channel source → unrestricted target (UKS → UKS)
    U2U,
    /// 2-channel source → restricted target (UKS → RKS/ROKS), unsupported
    U2R,
}

fn derive_source_scftype(nch_evec: usize, nch_occ: usize) -> SCFType {
    match (nch_evec, nch_occ) {
        (1, 1) => SCFType::RHF,   // RKS
        (1, 2) => SCFType::ROHF,  // ROKS
        (2, _) => SCFType::UHF,   // UKS
        _ => panic!(
            "Cannot determine source SCF type from eigenvector channels ({}) and occupation channels ({})",
            nch_evec, nch_occ
        ),
    }
}

fn import_mode(target_scftype: &SCFType, source: SCFType) -> ImportMode {
    match (source, target_scftype) {
        (SCFType::RHF, SCFType::RHF) => ImportMode::R2R,
        (SCFType::RHF, SCFType::UHF) => ImportMode::R2U,
        (SCFType::RHF, SCFType::ROHF) => {
            panic!("RKS/RHF source cannot map to an ROKS/ROHF target (r2ro)");
        },
        (SCFType::ROHF, SCFType::RHF) => {
            panic!("ROKS/ROHF source cannot map to an RKS/RHF target (ro2r)");
        },
        (SCFType::ROHF, SCFType::ROHF) => ImportMode::RO2RO,
        (SCFType::ROHF, SCFType::UHF) => ImportMode::RO2U,
        (SCFType::UHF, SCFType::UHF) => ImportMode::U2U,
        (SCFType::UHF, SCFType::RHF) | (SCFType::UHF, SCFType::ROHF) => ImportMode::U2R,
    }
}

pub fn initial_guess_from_raw(
    loaded_eigenvectors: Vec<f64>,
    loaded_eigenvalues: Vec<f64>,
    loaded_occupation: Vec<f64>,
    target_scftype: &SCFType,
    num_state: usize,
    num_basis: usize,
    print_level: usize
) -> ([MatrixFull<f64>;2],[Vec<f64>;2], Option<[Vec<f64>;2]>) {

    assert_eq!(
        loaded_eigenvectors.len() % (num_state * num_basis), 0,
        "Inconsistent eigenvector buffer length {} for num_state*num_basis = {}",
        loaded_eigenvectors.len(), num_state * num_basis
    );
    assert!(
        loaded_occupation.len() == num_state || loaded_occupation.len() == 2 * num_state,
        "Unexpected occupation size in chkfile: {} (expected {} or {} for num_state={})",
        loaded_occupation.len(), num_state, 2 * num_state, num_state
    );

    let source = derive_source_scftype(
        loaded_eigenvectors.len() / (num_state * num_basis),
        loaded_occupation.len() / num_state,
    );
    let mode = import_mode(target_scftype, source);

    let mut tmp_eigenvectors: [MatrixFull<f64>;2] = [MatrixFull::empty(),MatrixFull::empty()];
    let mut tmp_eigenvalues: [Vec<f64>;2] = [vec![],vec![]];
    let mut tmp_occupation: [Vec<f64>;2] = [vec![],vec![]];
    match mode {
        ImportMode::R2R => {
            info!("Importing MO coefficients in restricted format (RKS/RHF)");
            let tmp_eigen = MatrixFull::from_vec([num_state,num_basis], loaded_eigenvectors[0..num_state*num_basis].to_vec()).unwrap();
            tmp_eigenvectors[0] = tmp_eigen.transpose_and_drop();
            tmp_eigenvalues[0] = loaded_eigenvalues[0..num_state].to_vec();
            tmp_occupation[0] = loaded_occupation[0..num_state].to_vec();
            if print_level>3 {
                tmp_eigenvectors[0].formated_output(5, "full");
                trace!("eigenval {:?}", &tmp_eigenvalues[0]);
            }
        },
        ImportMode::R2U => {
            info!("Importing MO coefficients in RHF/RKS format and converting them to UHF/UKS format");
            // UKS target needs both channels populated; both start from the shared RKS spatial MOs
            (0..2).into_iter().for_each(|i| {
                let tmp_eigen = MatrixFull::from_vec([num_state,num_basis], loaded_eigenvectors[0..num_state*num_basis].to_vec()).unwrap();
                tmp_eigenvectors[i]=tmp_eigen.transpose_and_drop();
                tmp_eigenvalues[i]=loaded_eigenvalues[0..num_state].to_vec();
            });
            // RKS/RHF source: occupation is total (e.g. [2,2,0]), divide by 2 for per-spin
            (0..2).into_iter().for_each(|i_spin| {
                tmp_occupation[i_spin] = loaded_occupation[0..num_state].iter()
                    .map(|x| x * 0.5)
                    .collect();
            });
        },
        ImportMode::RO2RO => {
            info!("Importing ROKS/ROHF MOs in ROKS/ROHF format: shared spatial MOs in the alpha channel only");
            // ROKS/ROHF stores the shared spatial set in channel 0 only; channel 1 stays empty
            let tmp_eigen = MatrixFull::from_vec([num_state,num_basis], loaded_eigenvectors[0..num_state*num_basis].to_vec()).unwrap();
            tmp_eigenvectors[0]=tmp_eigen.transpose_and_drop();
            tmp_eigenvalues[0]=loaded_eigenvalues[0..num_state].to_vec();
            // ROKS/ROHF source: alpha and beta occupation stored separately
            tmp_occupation[0] = loaded_occupation[0..num_state].to_vec();
            tmp_occupation[1] = loaded_occupation[num_state..2*num_state].to_vec();
        },
        ImportMode::RO2U => {
            info!("Importing ROKS/ROHF MOs in UHF/UKS format: sharing spatial MOs across alpha and beta channels");
            // UKS target needs both channels populated; both start from the shared ROKS spatial MOs
            (0..2).into_iter().for_each(|i| {
                let tmp_eigen = MatrixFull::from_vec([num_state,num_basis], loaded_eigenvectors[0..num_state*num_basis].to_vec()).unwrap();
                tmp_eigenvectors[i]=tmp_eigen.transpose_and_drop();
                tmp_eigenvalues[i]=loaded_eigenvalues[0..num_state].to_vec();
            });
            // ROKS/ROHF source: alpha and beta occupation stored separately
            tmp_occupation[0] = loaded_occupation[0..num_state].to_vec();
            tmp_occupation[1] = loaded_occupation[num_state..2*num_state].to_vec();
        },
        ImportMode::U2U => {
            info!("Importing MO coefficients in unrestricted format (UKS/UHF)");
            (0..2).into_iter().for_each(|i| {
                let start = (0 + i)*num_state*num_basis;
                let end = (1 + i)*num_state*num_basis;
                let tmp_eigen = MatrixFull::from_vec([num_state,num_basis], loaded_eigenvectors[start..end].to_vec()).unwrap();
                tmp_eigenvectors[i]=tmp_eigen.transpose_and_drop();
                tmp_eigenvalues[i]=loaded_eigenvalues[ (0+i)*num_state..(1+i)*num_state].to_vec();
            });
            if print_level>3 {
                (0..2).into_iter().for_each(|i| {
                    tmp_eigenvectors[i].formated_output(5, "full");
                    trace!("eigenval {:?}", &tmp_eigenvalues[i]);
                });
            }
            tmp_occupation[0] = loaded_occupation[0..num_state].to_vec();
            tmp_occupation[1] = loaded_occupation[num_state..2*num_state].to_vec();
        },
        ImportMode::U2R => {
            panic!("Importing UHF/UKS MOs in an RHF/RKS calculation is not supported at present");
        }
    }

    (tmp_eigenvectors,tmp_eigenvalues,Some(tmp_occupation))
}

pub fn import_mo_coeff_from_hdf5chkfile(chkname: &str) -> ([MatrixFull<f64>;2], [usize;3]) {
    
    //let mut tmp_scf = SCF::new(&mol);
    //tmp_scf.generate_occupation();

    let file = hdf5::File::open(chkname).unwrap();
    let scf = file.group("scf").unwrap();
    let member = scf.member_names().unwrap();

    let num_basis = scf.dataset("num_basis").unwrap().read_1d::<usize>().unwrap()[0];
    let num_state = scf.dataset("num_state").unwrap().read_1d::<usize>().unwrap()[0];
    let spin_channel = scf.dataset("spin_channel").unwrap().read_1d::<usize>().unwrap()[0];

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

    info!("Initial guess from SAP");
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
