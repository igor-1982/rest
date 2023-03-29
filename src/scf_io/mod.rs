use clap::value_parser;
use tensors::{ERIFull,MatrixFull, ERIFold4, MatrixUpper, TensorSliceMut, RIFull, MatrixFullSlice, MatrixFullSliceMut, BasicMatrix, MathMatrix, MatrixUpperSlice};
use itertools::{Itertools, iproduct, izip};
use libc::SCHED_OTHER;
use core::num;
use std::fmt::format;
use std::io::Write;
//use std::{fs, vec};
use std::thread::panicking;
use std::{vec, fs};
use rest_libcint::{CINTR2CDATA, CintType};
use crate::geom_io::{GeomCell,MOrC, GeomUnit};
use crate::basis_io::{Basis4Elem,BasInfo};
use crate::molecule_io::{Molecule, generate_ri3fn_from_rimatr};
use crate::tensors::{TensorOpt,TensorOptMut,TensorSlice};
use crate::dft::{Grids, numerical_density, par_numerical_density};
use crate::{utilities, parse_input};
use crate::initial_guess::sap::get_vsap;
use rayon::prelude::*;
use hdf5;
//use blas::{ddot,daxpy};
use std::sync::{Mutex, Arc,mpsc};
use std::thread;
use crossbeam::{channel::{unbounded,bounded},thread::{Scope,scope}};
use std::sync::mpsc::{channel, Receiver};
//use blas_src::openblas::dgemm;
mod addons;

use crate::constants::SPECIES_INFO;




pub struct SCF {
    pub mol: Molecule,
    pub ovlp: MatrixUpper<f64>,
    pub h_core: MatrixUpper<f64>,
    //pub ijkl: Option<Tensors<f64>>,
    //pub ijkl: Option<ERIFull<f64>>,
    pub ijkl: Option<ERIFold4<f64>>,
    pub ri3fn: Option<RIFull<f64>>,
    pub rimatr: Option<(MatrixFull<f64>,MatrixFull<usize>,Vec<[usize;2]>)>,
    pub eigenvalues: [Vec<f64>;2],
    //pub eigenvectors: Vec<Tensors<f64>>,
    pub eigenvectors: [MatrixFull<f64>;2],
    //pub density_matrix: Vec<Tensors<f64>>,
    //pub density_matrix: [MatrixFull<f64>;2],
    pub density_matrix: Vec<MatrixFull<f64>>,
    //pub hamiltonian: Vec<Tensors<f64>>,
    pub hamiltonian: [MatrixUpper<f64>;2],
    pub scftype: SCFType,
    pub occupation: [Vec<f64>;2],
    pub homo: [usize;2],
    pub lumo: [usize;2],
    pub nuc_energy: f64,
    pub scf_energy: f64,
    pub grids: Option<Grids>,
}

#[derive(Clone,Copy)]
pub enum SCFType {
    RHF,
    ROHF,
    UHF
}

impl SCF {
    pub fn new(mol: &mut Molecule) -> SCF {
        SCF {
            mol: mol.clone(),
            ovlp: MatrixUpper::new(1,0.0),
            h_core: MatrixUpper::new(1,0.0),
            ijkl: None,
            ri3fn: None,
            rimatr: None,
            eigenvalues: [vec![],vec![]],
            hamiltonian: [MatrixUpper::new(1,0.0),
                              MatrixUpper::new(1,0.0)],
            eigenvectors: [MatrixFull::new([1,1],0.0),
                              MatrixFull::new([1,1],0.0)],
            //density_matrix: [MatrixFull::new([1,1],0.0),
            //                     MatrixFull::new([1,1],0.0)],
            density_matrix: vec![MatrixFull::empty(),
                                 MatrixFull::empty()],
            scftype: SCFType::RHF,
            occupation: [vec![],vec![]],
            homo: [0,0],
            lumo: [0,0],
            nuc_energy: 0.0,
            scf_energy: 0.0,
            grids: None,
        }
    }
    pub fn build(mut mol: Molecule) -> SCF {

        let mut time_mark = utilities::TimeRecords::new();
        time_mark.new_item("Overall", "SCF Preparation");
        time_mark.count_start("Overall");

        time_mark.new_item("CInt", "Two, Three, and Four-center integrals");
        time_mark.count_start("CInt");

        let nuc_energy = mol.geom.calc_nuc_energy();
        let dt1 = time::Local::now();
        let mut ovlp = mol.int_ij_matrixupper(String::from("ovlp"));
        let mut h_core = mol.int_ij_matrixupper(String::from("hcore"));
        let dt2 = time::Local::now();
        let timecost = (dt2.timestamp_millis()-dt1.timestamp_millis()) as f64 /1000.0;
        println!("The evaluation of 2D-tensors spends {:16.2} seconds",timecost);
        let dt1 = time::Local::now();

        // check and prepare the auxiliary basis sets
        if mol.ctrl.use_auxbas {mol.initialize_auxbas()};

        let mut ri3fn = if mol.ctrl.use_auxbas && !mol.ctrl.use_auxbas_symm {
            //Some(mol.prepare_ri3fn_for_ri_v_rayon())
            Some(mol.prepare_ri3fn_for_ri_v_full_rayon())
            //println!("generate ri3fn from rimatr");
            //let (rimatr, basbas2baspar, baspar2basbas) = mol.prepare_rimatr_for_ri_v_rayon();
            //Some(generate_ri3fn_from_rimatr(&rimatr, &basbas2baspar, &baspar2basbas))
        } else {
            None
        };

        let mut rimatr = if mol.ctrl.use_auxbas && mol.ctrl.use_auxbas_symm {
            //println!("generate ri3fn from rimatr");
            let (rimatr, basbas2baspar, baspar2basbas) = mol.prepare_rimatr_for_ri_v_rayon();
            Some((rimatr, basbas2baspar, baspar2basbas))
        } else {
            None
        };

        let dt2 = time::Local::now();
        let timecost = (dt2.timestamp_millis()-dt1.timestamp_millis()) as f64 /1000.0;
        if mol.ctrl.use_auxbas {
            println!("The evaluation of 3D-tensors spends {:16.2} seconds",timecost);
        } else {
            println!("The evaluation of 4D-tensors spends {:16.2} seconds",timecost);
        }

        let mut eris = if mol.ctrl.use_auxbas {
            if let Some(tmp_r3fn) = &ri3fn {
                None
                //Some(mol.int_ijkl_from_r3fn(tmp_r3fn))
            } else {
                None
            }
        } else {
            Some(mol.int_ijkl_erifold4())
            //None
        };
 
        //let mut tmp_eris = mol.int_ijkl_erifold4();
        //let mut tmp_ri3fn = mol.prepare_ri3fn_for_ri_v();
        //let mut t

        
        if mol.ctrl.print_level>3 {
            println!("The S matrix:");
            ovlp.formated_output(5, "lower");
            let mut kin = mol.int_ij_matrixupper(String::from("kinetic"));
            println!("The Kinetic matrix:");
            kin.formated_output(5, "lower");
            println!("The H-core matrix:");
            h_core.formated_output(5, "lower");
        }

        if mol.ctrl.print_level>4 {
            //(ij|kl)
            if let Some(tmp_eris) = &eris {
                println!("The four-center ERIs:");
                let mut tmp_num = 0;
                let (i_len,j_len) =  (mol.num_basis,mol.num_basis);
                let (k_len,l_len) =  (mol.num_basis,mol.num_basis);
                (0..k_len).into_iter().for_each(|k| {
                    (0..k+1).into_iter().for_each(|l| {
                        (0..i_len).into_iter().for_each(|i| {
                            (0..i+1).into_iter().for_each(|j| {
                                if let Some(tmp_value) = tmp_eris.get(&[i,j,k,l]) {
                                    if tmp_value.abs()>1.0e-1 {
                                        println!("I= {:2} J= {:2} K= {:2} L= {:2} Int= {:16.8}",i+1, j+1, k+1,l+1, tmp_value);
                                        tmp_num+= 1;
                                    }
                                } else {
                                    println!("Error: unknown value for eris[{},{},{},{}]",i,j,k,l)
                                };
                            })
                        })
                    })
                });
                println!("Print out {} ERIs", tmp_num);
            }
        }

        time_mark.count("CInt");

        let (eigenvectors, eigenvalues,n_found)=ovlp.to_matrixupperslicemut().lapack_dspevx().unwrap();

        if (n_found as usize) < mol.fdqc_bas.len() {
            println!("Overlap matrix is singular:");
            println!("  Using {} out of a possible {} specified basis functions",n_found, mol.fdqc_bas.len());
            println!("  Lowest remaining eigenvalue: {:16.8}",eigenvalues[0]);
            mol.num_state = n_found as usize;
        } else {
            println!("Overlap matrix is nonsigular:");
            println!("  Lowest eigenvalue: {:16.8} with the total number of basis functions: {:6}",eigenvalues[0],mol.num_state);
        };

        //let mut eigenvectors = [MatrixFull::new([1,1],0.0),MatrixFull::new([1,1],0.0)];
        //let mut eigenvalues = [vec![],vec![]];
        let mut tmp_scf = SCF::new(&mut mol);

        // at first check the scf type: RHF, ROHF or UHF
        tmp_scf.scftype = if mol.num_elec[1]==mol.num_elec[2] && ! mol.ctrl.spin_polarization {
            SCFType::RHF
        } else if mol.num_elec[1]!=mol.num_elec[2] && ! mol.ctrl.spin_polarization {
            SCFType::ROHF
        } else {      
            SCFType::UHF
        };
        match &tmp_scf.scftype {
            SCFType::RHF => {
                println!("Restricted Hartree-Fock (or Kohn-Sham) algorithm is invoked.")},
            SCFType::ROHF => {
                println!("Restricted-orbital Hartree-Fock (or Kohn-Sham) algorithm is invoked.");
                mol.ctrl.spin_channel=2;
                mol.spin_channel=2;
            },
            SCFType::UHF => {println!("Unrestricted Hartree-Fock (or Kohn-Sham) algorithm is invoked.")},
        };

        time_mark.new_item("DFT Grids", "the generation of DFT grids");
        time_mark.count_start("DFT Grids");
        let mut grids = if mol.xc_data.is_dfa_scf() {Some(Grids::build(&mol))} else {None};
        time_mark.count("DFT Grids");

        time_mark.new_item("Grids AO", "the generation of the tabulated AO");
        time_mark.count_start("Grids AO");
        if let Some(grids) = &mut grids {
            grids.prepare_tabulated_ao(&mol);
        }
        time_mark.count("Grids AO");

        if mol.ctrl.external_init_guess {
            println!("Importing density matrix from external inital guess file");
            let file = hdf5::File::open(&mol.ctrl.guessfile).unwrap();
            let init_guess = file.dataset("init_guess").unwrap().read_raw::<f64>().unwrap();
            let mut tmp_dm = MatrixFull::from_vec([mol.num_basis,mol.num_basis],init_guess).unwrap();
            (0..mol.spin_channel).into_iter().for_each(|i| {
                tmp_scf.density_matrix[i] = tmp_dm.clone();
            });
        }

        
        
        if mol.ctrl.restart && std::path::Path::new(&mol.ctrl.chkfile).exists() {
            let file = hdf5::File::open(&mol.ctrl.chkfile).unwrap();
            let scf = file.group("scf").unwrap();
            let member = scf.member_names().unwrap();
            let e_tot = scf.dataset("e_tot").unwrap().read_scalar::<f64>().unwrap();
            if mol.ctrl.print_level>=1 {
                println!("HDF5 Group: {:?} \nMembers: {:?}", scf, member);
            }
            println!("E_tot from chkfile: {:18.10}", e_tot);
            let buf01 = scf.dataset("mo_coeff").unwrap().read_raw::<f64>().unwrap();
            let buf02 = scf.dataset("mo_energy").unwrap().read_raw::<f64>().unwrap();
            let mut tmp_eigenvectors = vec![MatrixFull::empty(), MatrixFull::empty()];
            (0..mol.spin_channel).into_iter().for_each(|i| {
                let start = (0 + i)*mol.num_state*mol.num_basis;
                let end = (1 + i)*mol.num_state*mol.num_basis;
                tmp_eigenvectors[i] = MatrixFull::from_vec([mol.num_state,mol.num_basis], buf01[start..end].to_vec()).unwrap();
                //tmp_eigenvectors[i] = tmp_eigenvectors[i].transpose();
            //} 
            //let eigenvector_alpha = tmp_eigenvectors.transpose();
            //(0..mol.spin_channel).into_iter().for_each(|i| {
                tmp_scf.eigenvectors[i] = tmp_eigenvectors[i].transpose().clone();
            //});
            //let eigenvalues_alpha = buf02;
            //(0..mol.spin_channel).into_iter().for_each(|i| {
                tmp_scf.eigenvalues[i] = buf02[ (0+i)*mol.num_state..(1+i)*mol.num_state].to_vec().clone();
            });
            if mol.ctrl.print_level>1 {
                (0..mol.spin_channel).into_iter().for_each(|i| {
                    tmp_scf.eigenvectors[i].formated_output(5, "full");
                    println!("eigenval {:?}", &tmp_scf.eigenvalues[i]);
                });
            }
        } else {
            let mut init_fock = if mol.ctrl.initial_guess.eq(&"vsap") {

                time_mark.new_item("SAP", "Generation of SAP initial guess");
                time_mark.count_start("SAP");

                println!("Initial guess from SAP");
                let mut h_sap = mol.int_ij_matrixupper(String::from("kinetic"));
                //let tmp_kinetic = MatrixUpper::to_matrixfull(&kinetic).unwrap();
                //let tmp_v = if let Some( ref grids1 ) = grids {
                //    get_vsap(&mol, grids1)
                //} else {
                //    MatrixFull::new([mol.num_basis,mol.num_basis],0.0)
                //};
                //let tmp_h = MatrixFull::add(&tmp_kinetic, &tmp_v).unwrap();
                //tmp_h.to_matrixupper()
                let tmp_v = if let Some( ref grids1 ) = grids {
                    get_vsap(&mol, grids1)
                } else {
                    MatrixFull::new([mol.num_basis,mol.num_basis],0.0)
                };

                h_sap.data.iter_mut().zip(tmp_v.iter_matrixupper().unwrap()).for_each(|(t,f)| *t += f);

                time_mark.count("SAP");

                h_sap
                //(tmp_kinetic + tmp_v).to_matrixupper()
                //MatrixFull::add(&tmp_kinetic, &tmp_v).unwrap().to_matrixupper()

            } else if mol.ctrl.initial_guess.eq(&"hcore") {
                h_core.clone()
            } else {
                h_core.clone()
            };
            let (eigenvectors_alpha,eigenvalues_alpha)=init_fock.to_matrixupperslicemut().lapack_dspgvx(ovlp.to_matrixupperslicemut(),mol.num_state).unwrap();
            (0..mol.spin_channel).into_iter().for_each(|i| {
                tmp_scf.eigenvectors[i] = eigenvectors_alpha.clone();
            });
            (0..mol.spin_channel).into_iter().for_each(|i| {
                tmp_scf.eigenvalues[i] = eigenvalues_alpha.clone();
            });
        }

        tmp_scf.generate_occupation();

        //tmp_scf.formated_eigenvalues(mol.num_state);

        if ! mol.ctrl.external_init_guess {tmp_scf.generate_density_matrix()};
        let mut hamiltonian = [MatrixUpper::new(1,0.0),MatrixUpper::new(1,0.0)];

        //println!("debug dm_a");
        //tmp_scf.density_matrix[0].formated_output(10, "full");
        //println!("debug dm_b");
        //tmp_scf.density_matrix[1].formated_output(10, "full");
        //println!("debug ov");
        //ovlp.formated_output(10, "full");

        time_mark.count("Overall");
        if mol.ctrl.print_level>=2 {
            time_mark.report_all();
        }


        SCF {
            mol,
            ovlp,
            h_core,
            hamiltonian,
            ijkl: eris,
            ri3fn,
            rimatr,
            eigenvalues: tmp_scf.eigenvalues,
            eigenvectors: tmp_scf.eigenvectors,
            density_matrix: tmp_scf.density_matrix,
            scftype: tmp_scf.scftype,
            occupation: tmp_scf.occupation,
            homo: tmp_scf.homo,
            lumo: tmp_scf.lumo,
            nuc_energy,
            scf_energy: 0.0,
            grids,
        }
    }

    pub fn generate_occupation(&mut self) {
        self.generate_occupation_integer()
    }

    pub fn generate_occupation_integer(&mut self) {
        let num_state = self.mol.num_state;
        let spin_channel = self.mol.ctrl.spin_channel;
        let num_elec = &self.mol.num_elec;
        let mut occupation:[Vec<f64>;2] = [vec![],vec![]];
        let mut lumo:[usize;2] = [0,0];
        let mut homo:[usize;2] = [0,0];
        //let occ_num = 
        match self.scftype {
            SCFType::RHF => {
                let occ_num = 2.0;
                let i_spin = 0_usize;
                occupation[i_spin] = vec![0.0;num_state];
                let mut left_elec_spin = num_elec[i_spin+1];
                let mut index_i = 0_usize;
                while  left_elec_spin > 0.0 && index_i<=num_state {
                    occupation[i_spin][index_i] = (left_elec_spin*occ_num).min(occ_num);
                    index_i += 1;
                    left_elec_spin -= 1.0;
                }
                // make sure there is at least one LUMO
                if index_i > num_state-1 && left_elec_spin>0.0 {
                    panic!("Error:: the number of molecular orbitals is smaller than the number of electrons in the {}-spin channel\n num_state: {}; num_elec_alpha: {}", 
                           i_spin,num_state, num_elec[i_spin]);
                } else {
                    lumo[i_spin] = index_i;
                    homo[i_spin] = index_i-1;
                }; 
            },
            SCFType::ROHF => {
                let occ_num = 1.0;
                (0..2).for_each(|i_spin| {
                    occupation[i_spin] = vec![0.0;num_state];
                    let mut left_elec_spin = num_elec[i_spin+1];
                    let mut index_i = 0_usize;
                    while  left_elec_spin > 0.0 && index_i<=num_state {
                        occupation[i_spin][index_i] = (left_elec_spin*occ_num).min(occ_num);
                        index_i += 1;
                        left_elec_spin -= 1.0;
                    }
                    // make sure there is at least one LUMO
                    if index_i > num_state-1 && left_elec_spin>0.0 {
                        panic!("Error:: the number of molecular orbitals is smaller than the number of electrons in the {}-spin channel\n num_state: {}; num_elec_alpha: {}", 
                               i_spin,num_state, num_elec[i_spin]);
                    } else {
                        lumo[i_spin] = index_i;
                        homo[i_spin] = index_i-1;
                    }; 
                });
            },
            SCFType::UHF => {
                let occ_num = 1.0;
                (0..2).for_each(|i_spin| {
                    occupation[i_spin] = vec![0.0;num_state];
                    let mut left_elec_spin = num_elec[i_spin+1];
                    let mut index_i = 0_usize;
                    while  left_elec_spin > 0.0 && index_i<=num_state {
                        occupation[i_spin][index_i] = (left_elec_spin*occ_num).min(occ_num);
                        index_i += 1;
                        left_elec_spin -= 1.0;
                    }
                    // make sure there is at least one LUMO
                    if index_i > num_state-1 && left_elec_spin>0.0 {
                        panic!("Error:: the number of molecular orbitals is smaller than the number of electrons in the {}-spin channel\n num_state: {}; num_elec_alpha: {}", 
                               i_spin,num_state, num_elec[i_spin]);
                    } else {
                        lumo[i_spin] = index_i;
                        homo[i_spin] = if index_i==0 {0} else {index_i-1};
                    }; 
                });
            }
        };
        self.occupation = occupation;
        self.lumo = lumo;
        self.homo = homo;
        //println!("Occupation: {:?}, {:?}, {:?}, {}, {}",&self.homo,&self.lumo,&self.occupation,self.mol.num_state,self.mol.num_basis);
    }
    pub fn generate_density_matrix(&mut self) {
        let num_basis = self.mol.num_basis;
        let num_state = self.mol.num_state;
        let spin_channel = self.mol.spin_channel;
        let homo = &self.homo;
        let mut dm = vec![
            MatrixFull::empty(),
            MatrixFull::empty()
            ];
        (0..spin_channel).into_iter().for_each(|i_spin| {
            //println!("debug density_matrix spin: {}",i_spin);
            let mut dm_s = &mut dm[i_spin];
            *dm_s = MatrixFull::new([num_basis,num_basis],0.0);
            let mut eigv_s = &mut self.eigenvectors[i_spin];
            let occ_s =  &self.occupation[i_spin];
            let nw =  self.homo[i_spin]+1;

            let mut weight_eigv = MatrixFull::new([num_basis, num_state],0.0_f64);
            //let mut weight_eigv = eigv_s.clone();
            weight_eigv.par_iter_columns_mut(0..nw).unwrap().zip(eigv_s.par_iter_columns(0..nw).unwrap())
                .for_each(|value| {
                    value.0.into_iter().zip(value.1.into_iter()).for_each(|value| {
                        *value.0 = *value.1
                    })
                });
            // prepare weighted eigenvalue matrix wC
            weight_eigv.par_iter_columns_mut(0..nw).unwrap().zip(occ_s[0..nw].par_iter()).for_each(|(we,occ)| {
            //weight_eigv.data.chunks_exact_mut(weight_eigv.size[0]).zip(occ_s.iter()).for_each(|(we,occ)| {
                we.iter_mut().for_each(|c| *c = *c*occ);
            });

            // dm = wC*C^{T}
            dm_s.lapack_dgemm(&mut weight_eigv, eigv_s, 'N', 'T', 1.0, 0.0);
            //dm_s.formated_output(5, "lower".to_string());
        });
        //if let SCFType::ROHF = self.scftype {dm[1]=dm[0].clone()};
        self.density_matrix = dm;
        //self.density_matrix[0].formated_output(10, "upper".to_string());
        //if let Some(grids) = &self.grids {
        //    //println!("debug print grids");
        //    //println!("{:?}", &grids.weights);
        //    //grids.formated_output();
        //    //let total_elec = par_numerical_density(grids, &self.mol, &mut self.density_matrix);
        //    let total_elec = grids.evaluate_density(&mut self.density_matrix);
        //    if self.mol.spin_channel==1 {
        //        println!("total electron number: ({:16.8},{:16.8})", self.mol.num_elec[0],total_elec[0])
        //    } else {
        //        println!("electron number in alpha-channel: ({:12.8},{:12.8})", self.mol.num_elec[1],total_elec[0]);
        //        println!("electron number in beta-channel:  ({:12.8},{:12.8})", self.mol.num_elec[2],total_elec[1]);

        //    }
        //}
    }

    pub fn generate_vj_with_erifold4(&mut self, scaling_factor: f64) -> Vec<MatrixUpper<f64>> {
        let num_basis = self.mol.num_basis;
        let npair = num_basis*(num_basis+1)/2;
        let spin_channel = self.mol.spin_channel;
        let mut vj: Vec<MatrixUpper<f64>> = vec![MatrixUpper::new(1,0.0f64),MatrixUpper::new(1,0.0f64)];
        let dm = &self.density_matrix;
        if let Some(ijkl) = &self.ijkl {
            for i_spin in (0..spin_channel) {
                vj[i_spin] = MatrixUpper::new(npair,0.0f64);
                for jc in (0..num_basis) {
                    for ic in (0..jc) {
                        let dm_ij = dm[i_spin].get1d(ic*num_basis + jc).unwrap() + 
                                        dm[i_spin].get1d(jc*num_basis + ic).unwrap();
                        let ijkl_start = (jc*(jc+1)/2+ic)*npair;
                        let reduce_ij = ijkl.get1d_slice(ijkl_start,npair).unwrap();
                        //let reduce_ij = ijkl.get4d_slice([0,0,ic,jc],npair).unwrap();
                        //unsafe{daxpy(npair as i32, dm_ij, reduce_ij, 1, vj[i_spin].to_slice_mut(), 1)};
                        vj[i_spin].data.iter_mut().zip(reduce_ij.iter()).for_each(|(vj_ij,eri_ij)| {
                            *vj_ij += eri_ij*dm_ij
                        });
                        // Rayon parallellism. 
                        //vj[i_spin].data.par_iter_mut().zip(reduce_ij.par_iter()).for_each(|(vj_ij,eri_ij)| {
                        //    *vj_ij += eri_ij*dm_ij
                        //});
                    }
                }
                for jc in (0..num_basis) {
                    let dm_ij = dm[i_spin].get1d(jc*num_basis + jc).unwrap(); 
                    let ijkl_start = (jc*(jc+1)/2+jc)*npair;
                    let reduce_ij = ijkl.get1d_slice(ijkl_start,npair).unwrap();
                    //let reduce_ij = ijkl.get4d_slice([0,0,jc,jc],npair).unwrap();
                    //unsafe{daxpy(npair as i32, *dm_ij, reduce_ij, 1, vj[i_spin].to_slice_mut(), 1)};
                    vj[i_spin].data.iter_mut().zip(reduce_ij.iter()).for_each(|(vj_ij,eri_ij)| {
                        *vj_ij += eri_ij*dm_ij
                    });
                    //vj[i_spin].data.par_iter_mut().zip(reduce_ij.par_iter()).for_each(|(vj_ij,eri_ij)| {
                    //    *vj_ij += eri_ij*dm_ij
                    //});
                }
            }
        }
        if scaling_factor!=1.0f64 {
            for i_spin in (0..spin_channel) {
                vj[i_spin].data.par_iter_mut().for_each(|f| *f = *f*scaling_factor)
            }
        };

        //debug 
        //let i_spin:usize = 0;
        //for j in (0..num_basis) {
        //    for i in (0..j+1) {
        //        println!("i: {}, j: {}, vj_ij: {}", i,j, vj[0].get2d([i,j]).unwrap());
        //    }
        //}

        vj
    }
    pub fn generate_vj_with_erifold4_sync(&mut self, scaling_factor: f64) -> Vec<MatrixUpper<f64>> {
        let num_basis = self.mol.num_basis;
        let npair = num_basis*(num_basis+1)/2;
        let spin_channel = self.mol.spin_channel;
        let mut vj: Vec<MatrixUpper<f64>> = vec![MatrixUpper::new(1,0.0f64),MatrixUpper::new(1,0.0f64)];
        let dm = &self.density_matrix;
        let num_para: usize = if let Some(num_para) = self.mol.ctrl.num_threads {
            num_para
        } else {
            1
        };
        let num_chunck = if num_basis%num_para==0 {
            (num_basis/num_para,num_basis/num_para)
        } else {
            (num_basis/num_para+1,num_basis-(num_basis/num_para+1)*(num_para-1))
        };
        if let Some(ijkl) = &self.ijkl {
            for i_spin in (0..spin_channel) {
                vj[i_spin] = MatrixUpper::new(npair,0.0f64);
                scope(|s_thread| {
                    let (tx_jc,rx_jc) = unbounded();
                    for f in (0..num_para-1) {
                        let jc_start_thread = f*num_chunck.0;
                        let jc_end_thread = jc_start_thread + num_chunck.0;
                        let tx_jc_thread = tx_jc.clone();
                        let handle = s_thread.spawn(move |_| {
                            let mut vj_thread = MatrixUpper::new(npair,0.0f64);
                            for jc in (jc_start_thread..jc_end_thread) {
                                for ic in (0..jc) {
                                    let dm_ij = dm[i_spin].get1d(ic*num_basis + jc).unwrap() + 
                                                    dm[i_spin].get1d(jc*num_basis + ic).unwrap();
                                    let ijkl_start = (jc*(jc+1)/2+ic)*npair;
                                    let reduce_ij = ijkl.get1d_slice(ijkl_start,npair).unwrap();
                                    vj_thread.data.iter_mut().zip(reduce_ij.iter()).for_each(|(vj_ij,eri_ij)| {
                                        *vj_ij += eri_ij*dm_ij
                                    });
                                }
                                let dm_ij = dm[i_spin].get1d(jc*num_basis + jc).unwrap(); 
                                let ijkl_start = (jc*(jc+1)/2+jc)*npair;
                                let reduce_ij = ijkl.get1d_slice(ijkl_start,npair).unwrap();
                                vj_thread.data.iter_mut().zip(reduce_ij.iter()).for_each(|(vj_ij,eri_ij)| {
                                    *vj_ij += eri_ij*dm_ij
                                });
                            }
                            tx_jc_thread.send(vj_thread).unwrap();
                        });
                    }
                    let jc_start_thread = (num_para-1)*num_chunck.0;
                    let jc_end_thread = jc_start_thread + num_chunck.1;
                    let tx_jc_thread = tx_jc;
                    let handle = s_thread.spawn(move |_| {
                        let mut vj_thread = MatrixUpper::new(npair,0.0f64);
                        for jc in (jc_start_thread..jc_end_thread) {
                            for ic in (0..jc) {
                                let dm_ij = dm[i_spin].get1d(ic*num_basis + jc).unwrap() + 
                                                dm[i_spin].get1d(jc*num_basis + ic).unwrap();
                                let ijkl_start = (jc*(jc+1)/2+ic)*npair;
                                let reduce_ij = ijkl.get1d_slice(ijkl_start,npair).unwrap();
                                vj_thread.data.iter_mut().zip(reduce_ij.iter()).for_each(|(vj_ij,eri_ij)| {
                                    *vj_ij += eri_ij*dm_ij
                                });
                            }
                            let dm_ij = dm[i_spin].get1d(jc*num_basis + jc).unwrap(); 
                            let ijkl_start = (jc*(jc+1)/2+jc)*npair;
                            let reduce_ij = ijkl.get1d_slice(ijkl_start,npair).unwrap();
                            vj_thread.data.iter_mut().zip(reduce_ij.iter()).for_each(|(vj_ij,eri_ij)| {
                                *vj_ij += eri_ij*dm_ij
                            });
                        }
                        tx_jc_thread.send(vj_thread).unwrap();
                    });
                    for received in rx_jc {
                        vj[i_spin].data.iter_mut()
                            .zip(received.data).for_each(|(i,j)| {*i += j});
                    }

                }).unwrap();
            }
        }
        if scaling_factor!=1.0f64 {
            for i_spin in (0..spin_channel) {
                vj[i_spin].data.par_iter_mut().for_each(|f| *f = *f*scaling_factor)
            }
        };

        //debug 
        //let i_spin:usize = 0;
        //for j in (0..num_basis) {
        //    for i in (0..j+1) {
        //        println!("i: {}, j: {}, vj_ij: {}", i,j, vj[0].get2d([i,j]).unwrap());
        //    }
        //}

        vj
    }
    pub fn generate_vk_with_erifold4(&mut self, scaling_factor: f64) -> Vec<MatrixFull<f64>> {
        let num_basis = self.mol.num_basis;
        let npair = num_basis*(num_basis+1)/2;
        let spin_channel = self.mol.spin_channel;
        let mut vk: Vec<MatrixFull<f64>> = vec![MatrixFull::new([1,1],0.0f64),MatrixFull::new([1,1],0.0f64)];
        let dm = &self.density_matrix;
        if let Some(ijkl) = &self.ijkl {
            for i_spin in (0..spin_channel) {
                vk[i_spin] = MatrixFull::new([num_basis,num_basis],0.0f64);
                for jc in (0..num_basis) {
                    for ic in (0..jc) {
                        let ijkl_start = (jc*(jc+1)/2+ic)*npair;
                        let reduce_ij = ijkl.get1d_slice(ijkl_start,npair).unwrap();
                        //let reduce_ij = ijkl.get4d_slice([0,0,ic,jc],npair).unwrap();
                        let dm_ic = dm[i_spin].get1d_slice(ic*num_basis,num_basis).unwrap();
                        let dm_jc = dm[i_spin].get1d_slice(jc*num_basis,num_basis).unwrap();
                        let mut vk_ic = vk[i_spin].get1d_slice_mut(ic*num_basis,num_basis).unwrap();
                        let mut kl = 0_usize;
                        for k in (0..num_basis) {
                            // The psuedo-code for the next several ten lines
                            //for l in (0..k) {
                            //    vk_ic[l] += reduce_ij[kl] *dm_jc[k];
                            //    vk_ic[k] += reduce_ij[kl] *dm_jc[l];
                            //    kl += 1;
                            //}
                            vk_ic[..k].iter_mut()
                                .zip(reduce_ij[kl..kl+k].iter())
                                .for_each(|(i,j)| {*i += j*dm_jc[k]});
                            vk_ic[k] += reduce_ij[kl..kl+k]
                                .iter()
                                .zip(dm_jc[..k].iter()).fold(0.0, |acc,(i,j)| acc+i*j);
                            kl += k;
                            //============================================
                            vk_ic[k] += reduce_ij[kl] *dm_jc[k];
                            kl += 1;
                        }
                        let mut vk_jc = vk[i_spin].get1d_slice_mut(jc*num_basis,num_basis).unwrap();
                        let mut kl = 0_usize;
                        for k in (0..num_basis) {
                            // The psuedo-code for the next several ten lines
                            //for l in (0..k) {
                            //    vk_jc[l] += reduce_ij[kl] *dm_ic[k];
                            //    vk_jc[k] += reduce_ij[kl] *dm_ic[l];
                            //    kl += 1;
                            //}
                            vk_jc[..k].iter_mut()
                                .zip(reduce_ij[kl..kl+k].iter())
                                .for_each(|(i,j)| {*i += j*dm_ic[k]});
                            vk_jc[k] += reduce_ij[kl..kl+k]
                                .iter()
                                .zip(dm_ic[..k].iter()).fold(0.0, |acc,(i,j)| acc+i*j);
                            kl += k;
                            //============================================
                            vk_jc[k] += reduce_ij[kl] *dm_ic[k];
                            kl += 1;
                        }
                    }
                }
                for ic in (0..num_basis) {
                    let ijkl_start = (ic*(ic+1)/2+ic)*npair;
                    let reduce_ij = ijkl.get1d_slice(ijkl_start,npair).unwrap();
                    let dm_ic = dm[i_spin].get1d_slice(ic*num_basis,num_basis).unwrap();
                    let mut vk_ic = vk[i_spin].get1d_slice_mut(ic*num_basis,num_basis).unwrap();
                    let mut kl = 0_usize;
                    for k in (0..num_basis) {
                        // The psuedo-code for the next several ten lines
                        //for l in (0..k) {
                        //    vk_ic[l] += reduce_ij[kl] *dm_ic[k];
                        //    vk_ic[k] += reduce_ij[kl] *dm_ic[l];
                        //    kl += 1;
                        //}
                        vk_ic[..k].par_iter_mut()
                            .zip(reduce_ij[kl..kl+k].par_iter())
                            .for_each(|(i,j)| {*i += j*dm_ic[k]});
                        vk_ic[k] += reduce_ij[kl..kl+k]
                            .iter()
                            .zip(dm_ic[..k].iter()).fold(0.0, |acc,(i,j)| acc+i*j);
                        kl += k;
                        //=================================================
                        vk_ic[k] += reduce_ij[kl] *dm_ic[k];
                        kl += 1;
                    }
                }
            }
        }
        if scaling_factor!=1.0f64 {
            for i_spin in (0..spin_channel) {
                vk[i_spin].data.par_iter_mut().for_each(|f| *f = *f*scaling_factor)
            }
        };
        vk
    }
    pub fn generate_vk_with_erifold4_v02(&mut self, scaling_factor: f64) -> Vec<MatrixUpper<f64>> {
        let num_basis = self.mol.num_basis;
        let npair = num_basis*(num_basis+1)/2;
        let spin_channel = self.mol.spin_channel;
        let mut vk: Vec<MatrixUpper<f64>> = vec![MatrixUpper::new(1,0.0f64),MatrixUpper::new(1,0.0f64)];
        let dm = &self.density_matrix;
        if let Some(ijkl) = &self.ijkl {
            for i_spin in (0..spin_channel) {
                vk[i_spin] = MatrixUpper::new(npair,0.0f64);
                for jc in (0..num_basis) {
                    for ic in (0..jc) {
                        let ijkl_start = (jc*(jc+1)/2+ic)*npair;
                        let reduce_ij = ijkl.get1d_slice(ijkl_start,npair).unwrap();
                        //let reduce_ij = ijkl.get4d_slice([0,0,ic,jc],npair).unwrap();
                        let dm_ic = dm[i_spin].get1d_slice(ic*num_basis,num_basis).unwrap();
                        let dm_jc = dm[i_spin].get1d_slice(jc*num_basis,num_basis).unwrap();
                        let mut vk_ic = vk[i_spin].get1d_slice_mut(ic*(ic+1)/2,ic+1).unwrap();
                        let mut kl = 0_usize;
                        // The psuedo-code for the next several ten lines
                        //for k in (0..num_basis) {
                        //    for l in (0..k) {
                        //        vk_ic[l] += reduce_ij[kl] *dm_jc[k];
                        //        vk_ic[k] += reduce_ij[kl] *dm_jc[l];
                        //        kl += 1;
                        //    }
                        //}
                        //    vk_ic[k] += reduce_ij[kl] *dm_jc[k];
                        //    kl += 1;
                        //==============================================
                        for k in (0..num_basis) {
                            if k<=ic {
                                vk_ic[..k].iter_mut()
                                    .zip(reduce_ij[kl..kl+k].iter())
                                    .for_each(|(i,j)| {*i += j*dm_jc[k]});
                                vk_ic[k] += reduce_ij[kl..kl+k]
                                    .iter()
                                    .zip(dm_jc[..k].iter()).fold(0.0, |acc,(i,j)| acc+i*j);
                                kl += k;
                                vk_ic[k] += reduce_ij[kl] *dm_jc[k];
                                kl += 1;
                            } else {
                                vk_ic[..ic+1].iter_mut()
                                    .zip(reduce_ij[kl..kl+ic+1].iter())
                                    .for_each(|(i,j)| {*i += j*dm_jc[k]});
                                kl += k+1;
                            }
                            //if ic==4 && k==35 {println!("{}",kl)};
                        }
                        //=================================================
                        // try rayon parallel version
                        //for k in (0..num_basis) {
                        //    if k<=ic {
                        //        vk_ic[..k].iter_mut()
                        //            .zip(reduce_ij[kl..kl+k].iter())
                        //            .for_each(|(i,j)| {*i += j*dm_jc[k]});
                        //        vk_ic[k] += reduce_ij[kl..kl+k]
                        //            .par_iter()
                        //            .zip(dm_jc[..k].par_iter()).map(|(i,j)| i*j).sum::<f64>();
                        //        kl += k;
                        //        vk_ic[k] += reduce_ij[kl] *dm_jc[k];
                        //        kl += 1;
                        //    } else {
                        //        vk_ic[..ic+1].iter_mut()
                        //            .zip(reduce_ij[kl..kl+ic+1].iter())
                        //            .for_each(|(i,j)| {*i += j*dm_jc[k]});
                        //        kl += k+1;
                        //    }
                        //}
                        //=================================================
                        let mut vk_jc = vk[i_spin].get1d_slice_mut(jc*(jc+1)/2,jc+1).unwrap();
                        let mut kl = 0_usize;
                        // The psuedo-code for the next several ten lines
                        //for k in (0..num_basis) {
                        //    for l in (0..k) {
                        //        vk_jc[l] += reduce_ij[kl] *dm_ic[k];
                        //        vk_jc[k] += reduce_ij[kl] *dm_ic[l];
                        //        kl += 1;
                        //    }
                        //    vk_jc[k] += reduce_ij[kl] *dm_ic[k];
                        //    kl += 1;
                        //}
                        for k in (0..num_basis) {
                            if k<=jc {
                                vk_jc[..k].iter_mut()
                                    .zip(reduce_ij[kl..kl+k].iter())
                                    .for_each(|(i,j)| {*i += j*dm_ic[k]});
                                vk_jc[k] += reduce_ij[kl..kl+k]
                                    .iter()
                                    .zip(dm_ic[..k].iter()).fold(0.0, |acc,(i,j)| acc+i*j);
                                kl += k;
                                vk_jc[k] += reduce_ij[kl] *dm_ic[k];
                                kl += 1;
                            } else {
                                vk_jc[..jc+1].iter_mut()
                                    .zip(reduce_ij[kl..kl+jc+1].iter())
                                    .for_each(|(i,j)| {*i += j*dm_ic[k]});
                                kl += k+1;
                            }
                        }
                        //=================================================
                    }
                }
                for ic in (0..num_basis) {
                    let ijkl_start = (ic*(ic+1)/2+ic)*npair;
                    let reduce_ij = ijkl.get1d_slice(ijkl_start,npair).unwrap();
                    //let reduce_ij = ijkl.get4d_slice([0,0,ic,ic],npair).unwrap();
                    let dm_ic = dm[i_spin].get1d_slice(ic*num_basis,num_basis).unwrap();
                    let mut vk_ic = vk[i_spin].get1d_slice_mut(ic*(ic+1)/2,ic+1).unwrap();
                    let mut kl = 0_usize;
                    // The psuedo-code for the next several ten lines
                    //for k in (0..num_basis) {
                    //    for l in (0..k) {
                    //        vk_ic[l] += reduce_ij[kl] *dm_ic[k];
                    //        vk_ic[k] += reduce_ij[kl] *dm_ic[l];
                    //        kl += 1;
                    //    }
                    //    vk_ic[k] += reduce_ij[kl] *dm_ic[k];
                    //    kl += 1;
                    //}
                    for k in (0..num_basis) {
                        if k<=ic {
                            vk_ic[..k].iter_mut()
                                .zip(reduce_ij[kl..kl+k].iter())
                                .for_each(|(i,j)| {*i += j*dm_ic[k]});
                            vk_ic[k] += reduce_ij[kl..kl+k]
                                .iter()
                                .zip(dm_ic[..k].iter()).fold(0.0, |acc,(i,j)| acc+i*j);
                            kl += k;
                            vk_ic[k] += reduce_ij[kl] *dm_ic[k];
                            kl += 1;
                        } else {
                            vk_ic[..ic+1].iter_mut()
                                .zip(reduce_ij[kl..kl+ic+1].iter())
                                .for_each(|(i,j)| {*i += j*dm_ic[k]});
                            kl += k+1;
                        }
                    }
                    //=================================================
                }
            }
        }
        if scaling_factor!=1.0f64 {
            for i_spin in (0..spin_channel) {
                vk[i_spin].data.par_iter_mut().for_each(|f| *f = *f*scaling_factor)
            }
        };
        vk
    }
    pub fn generate_vk_with_erifold4_sync(&mut self, scaling_factor: f64) -> Vec<MatrixUpper<f64>> {
        let num_basis = self.mol.num_basis;
        let npair = num_basis*(num_basis+1)/2;
        let spin_channel = self.mol.spin_channel;
        let mut vk: Vec<MatrixUpper<f64>> = vec![MatrixUpper::new(1,0.0f64),MatrixUpper::new(1,0.0f64)];
        let dm = &self.density_matrix;
        let num_para: usize = if let Some(num_para) = self.mol.ctrl.num_threads {
            num_para
        } else {
            1
        };
        let num_chunck = if num_basis%num_para==0 {
            (num_basis/num_para,num_basis/num_para)
        } else {
            (num_basis/num_para+1,num_basis-(num_basis/num_para+1)*(num_para-1))
        };
        println!("num_threads: ({},{}),num_chunck: ({},{})",
                num_para,num_basis,
                num_chunck.0,
                num_chunck.1);
                //if num_basis%num_para==0 {num_chunck.0} else {num_basis%num_para});
        if let Some(ijkl) = &self.ijkl {
            for i_spin in (0..spin_channel) {
                vk[i_spin] = MatrixUpper::new(npair,0.0f64);
                for jc in (0..num_basis) {
                    for ic in (0..jc) {
                        scope(|s_thread| {
                            let ijkl_start = (jc*(jc+1)/2+ic)*npair;
                            let reduce_ij = ijkl.get1d_slice(ijkl_start,npair).unwrap();
                            //let reduce_ij = ijkl.get4d_slice([0,0,ic,jc],npair).unwrap();
                            let dm_ic = dm[i_spin].get1d_slice(ic*num_basis,num_basis).unwrap();
                            let dm_jc = dm[i_spin].get1d_slice(jc*num_basis,num_basis).unwrap();
                            let (tx_ic,rx_ic) = unbounded();
                            let (tx_jc,rx_jc) = unbounded();
                            //println!("Main thread: {:?}",thread::current().id());
                            for f in (0..num_para-1) {
                                let ic_thread = ic;
                                let jc_thread = jc;
                                let k_start_thread = f*num_chunck.0;
                                let k_end_thread = k_start_thread + num_chunck.0;
                                let tx_ic_thread = tx_ic.clone();
                                let tx_jc_thread = tx_jc.clone();
                                let mut kl_thread = k_start_thread*(k_start_thread+1)/2;
                                let handle = s_thread.spawn(move |_| {
                                    let mut vk_ic_thread = vec![0.0;ic_thread+1];
                                    let mut vk_jc_thread = vec![0.0;jc_thread+1];
                                    //let handle_thread = thread::current();
                                    //if ic_thread == 4 {println!("Fork thread: {:?}, kl: {}, k: ({}, {})",handle_thread.id(),kl_thread, k_start_thread, k_end_thread)};
                                    for k in (k_start_thread..k_end_thread) {
                                        let mut kl_jc_thread = kl_thread;
                                        if k<=ic_thread {
                                            vk_ic_thread[..k].iter_mut()
                                                .zip(reduce_ij[kl_thread..kl_thread+k].iter())
                                                .for_each(|(i,j)| {*i += j*dm_jc[k]});
                                            vk_ic_thread[k] += reduce_ij[kl_thread..kl_thread+k]
                                                .iter()
                                                .zip(dm_jc[..k].iter()).fold(0.0, |acc,(i,j)| acc+i*j);
                                            kl_thread += k;
                                            vk_ic_thread[k] += reduce_ij[kl_thread] *dm_jc[k];
                                            kl_thread += 1;
                                        } else {
                                            vk_ic_thread[..ic+1].iter_mut()
                                                .zip(reduce_ij[kl_thread..kl_thread+ic+1].iter())
                                                .for_each(|(i,j)| {*i += j*dm_jc[k]});
                                            kl_thread += k+1;
                                        }
                                        if k<=jc_thread {
                                            vk_jc_thread[..k].iter_mut()
                                                .zip(reduce_ij[kl_jc_thread..kl_jc_thread+k].iter())
                                                .for_each(|(i,j)| {*i += j*dm_ic[k]});
                                            vk_jc_thread[k] += reduce_ij[kl_jc_thread..kl_jc_thread+k]
                                                .iter()
                                                .zip(dm_ic[..k].iter()).fold(0.0, |acc,(i,j)| acc+i*j);
                                            kl_jc_thread += k;
                                            vk_jc_thread[k] += reduce_ij[kl_jc_thread] *dm_ic[k];
                                            kl_jc_thread += 1;
                                        } else {
                                            vk_jc_thread[..jc+1].iter_mut()
                                                .zip(reduce_ij[kl_jc_thread..kl_jc_thread+jc_thread+1].iter())
                                                .for_each(|(i,j)| {*i += j*dm_ic[k]});
                                            kl_jc_thread += k+1;
                                        }
                                    }
                                    //if ic_thread == 4 {println!("Fork thread: {:?}, kl: {}, k: ({}, {})",handle_thread.id(),kl_thread, k_start_thread, k_end_thread)};
                                    tx_ic_thread.send(vk_ic_thread).unwrap();
                                    tx_jc_thread.send(vk_jc_thread).unwrap();
                                });
                                //handles.push(handle);
                            }
                            let ic_thread = ic;
                            let jc_thread = jc;
                            let k_start_thread = (num_para-1)*num_chunck.0;
                            let k_end_thread = k_start_thread+num_chunck.1;
                            //let reduce_ij_thread = reduce_ij.clone();
                            //let dm_ic_thread = dm_ic.clone();
                            //let dm_jc_thread = dm_jc.clone();
                            let mut kl_thread = k_start_thread*(k_start_thread+1)/2;
                            let tx_ic_thread = tx_ic;
                            let tx_jc_thread = tx_jc;
                            let handle = s_thread.spawn(move |_| {
                                let mut vk_ic_thread = vec![0.0;ic_thread+1];
                                let mut vk_jc_thread = vec![0.0;jc_thread+1];
                                //let handle_thread = thread::current();
                                //if ic_thread == 4 {println!("Fork thread: {:?}, kl: {}, k: ({}, {})",handle_thread.id(),kl_thread, k_start_thread, k_end_thread)};
                                for k in (k_start_thread..k_end_thread) {
                                    let mut kl_jc_thread = kl_thread;
                                    if k<=ic_thread {
                                        vk_ic_thread[..k].iter_mut()
                                            .zip(reduce_ij[kl_thread..kl_thread+k].iter())
                                            .for_each(|(i,j)| {*i += j*dm_jc[k]});
                                        vk_ic_thread[k] += reduce_ij[kl_thread..kl_thread+k]
                                            .iter()
                                            .zip(dm_jc[..k].iter()).fold(0.0, |acc,(i,j)| acc+i*j);
                                        kl_thread += k;
                                        vk_ic_thread[k] += reduce_ij[kl_thread] *dm_jc[k];
                                        kl_thread += 1;
                                    } else {
                                        vk_ic_thread[..ic+1].iter_mut()
                                            .zip(reduce_ij[kl_thread..kl_thread+ic+1].iter())
                                            .for_each(|(i,j)| {*i += j*dm_jc[k]});
                                        kl_thread += k+1;
                                    }
                                    if k<=jc_thread {
                                        vk_jc_thread[..k].iter_mut()
                                            .zip(reduce_ij[kl_jc_thread..kl_jc_thread+k].iter())
                                            .for_each(|(i,j)| {*i += j*dm_ic[k]});
                                        vk_jc_thread[k] += reduce_ij[kl_jc_thread..kl_jc_thread+k]
                                            .iter()
                                            .zip(dm_ic[..k].iter()).fold(0.0, |acc,(i,j)| acc+i*j);
                                        kl_jc_thread += k;
                                        vk_jc_thread[k] += reduce_ij[kl_jc_thread] *dm_ic[k];
                                        kl_jc_thread += 1;
                                    } else {
                                        vk_jc_thread[..jc+1].iter_mut()
                                            .zip(reduce_ij[kl_jc_thread..kl_jc_thread+jc_thread+1].iter())
                                            .for_each(|(i,j)| {*i += j*dm_ic[k]});
                                        kl_jc_thread += k+1;
                                    }
                                }
                                //if ic_thread == 4 {println!("Fork thread: {:?}, kl: {}, k: ({}, {})",handle_thread.id(),kl_thread, k_start_thread, k_end_thread)};
                                tx_ic_thread.send(vk_ic_thread).unwrap();
                                tx_jc_thread.send(vk_jc_thread).unwrap();
                            });
                            //handles.push(handle);
                            {
                                let mut vk_ic = vk[i_spin].get1d_slice_mut(ic*(ic+1)/2,ic+1).unwrap();
                                for received in rx_ic {
                                    vk_ic.iter_mut()
                                        .zip(received)
                                        .for_each(|(i,j)| {*i += j});
                                }
                            }
                            {
                                let mut vk_jc = vk[i_spin].get1d_slice_mut(jc*(jc+1)/2,jc+1).unwrap();
                                for received in rx_jc {
                                    vk_jc.iter_mut()
                                        .zip(received)
                                        .for_each(|(i,j)| {*i += j});
                                }
                            }
                        }).unwrap();
                    }
                }
                for ic in (0..num_basis) {
                    let ijkl_start = (ic*(ic+1)/2+ic)*npair;
                    let reduce_ij = ijkl.get1d_slice(ijkl_start,npair).unwrap();
                    //let reduce_ij = ijkl.get4d_slice([0,0,ic,ic],npair).unwrap();
                    let dm_ic = dm[i_spin].get1d_slice(ic*num_basis,num_basis).unwrap();
                    let mut vk_ic = vk[i_spin].get1d_slice_mut(ic*(ic+1)/2,ic+1).unwrap();
                    let mut kl = 0_usize;
                    // The psuedo-code for the next several ten lines
                    //for k in (0..num_basis) {
                    //    for l in (0..k) {
                    //        vk_ic[l] += reduce_ij[kl] *dm_ic[k];
                    //        vk_ic[k] += reduce_ij[kl] *dm_ic[l];
                    //        kl += 1;
                    //    }
                    //    vk_ic[k] += reduce_ij[kl] *dm_ic[k];
                    //    kl += 1;
                    //}
                    for k in (0..num_basis) {
                        if k<=ic {
                            vk_ic[..k].iter_mut()
                                .zip(reduce_ij[kl..kl+k].iter())
                                .for_each(|(i,j)| {*i += j*dm_ic[k]});
                            vk_ic[k] += reduce_ij[kl..kl+k]
                                .iter()
                                .zip(dm_ic[..k].iter()).fold(0.0, |acc,(i,j)| acc+i*j);
                            kl += k;
                            vk_ic[k] += reduce_ij[kl] *dm_ic[k];
                            kl += 1;
                        } else {
                            vk_ic[..ic+1].iter_mut()
                                .zip(reduce_ij[kl..kl+ic+1].iter())
                                .for_each(|(i,j)| {*i += j*dm_ic[k]});
                            kl += k+1;
                        }
                    }
                    //=================================================
                }
            }
        }
        if scaling_factor!=1.0f64 {
            for i_spin in (0..spin_channel) {
                vk[i_spin].data.par_iter_mut().for_each(|f| *f = *f*scaling_factor)
            }
        };
        vk
    }
    pub fn generate_vk_with_erifold4_sync_v02(&mut self, scaling_factor: f64) -> Vec<MatrixUpper<f64>> {
        let num_basis = self.mol.num_basis;
        let npair = num_basis*(num_basis+1)/2;
        let spin_channel = self.mol.spin_channel;
        let mut vk: Vec<MatrixUpper<f64>> = vec![MatrixUpper::new(1,0.0f64),MatrixUpper::new(1,0.0f64)];
        let dm = &self.density_matrix;
        let num_para: usize = if let Some(num_para) = self.mol.ctrl.num_threads {
            num_para
        } else {
            1
        };
        let num_chunck = if num_basis%num_para==0 {
            (num_basis/num_para,num_basis/num_para)
        } else {
            (num_basis/num_para+1,num_basis-(num_basis/num_para+1)*(num_para-1))
        };
        println!("num_threads: ({},{}),num_chunck: ({},{})",
                num_para,num_basis,
                num_chunck.0,
                num_chunck.1);
        if let Some(ijkl) = &self.ijkl {
            for i_spin in (0..spin_channel) {
                vk[i_spin] = MatrixUpper::new(npair,0.0f64);
                scope(|s_thread| {
                    let (tx_jc,rx_jc) = unbounded();
                    for f in (0..num_para-1) {
                        let jc_start_thread = f*num_chunck.0;
                        let jc_end_thread = jc_start_thread + num_chunck.0;
                        let tx_jc_thread = tx_jc.clone();
                        let handle = s_thread.spawn(move |_| {
                            let mut vk_thread =  MatrixUpper::new(npair,0.0f64);
                            for jc in (jc_start_thread..jc_end_thread) {
                                for ic in (0..jc) {
                                    let ijkl_start = (jc*(jc+1)/2+ic)*npair;
                                    let reduce_ij = ijkl.get1d_slice(ijkl_start,npair).unwrap();
                                    //let reduce_ij = ijkl.get4d_slice([0,0,ic,jc],npair).unwrap();
                                    let dm_ic = dm[i_spin].get1d_slice(ic*num_basis,num_basis).unwrap();
                                    let dm_jc = dm[i_spin].get1d_slice(jc*num_basis,num_basis).unwrap();
                                    let mut vk_ic = vk_thread.get1d_slice_mut(ic*(ic+1)/2,ic+1).unwrap();
                                    let mut kl = 0_usize;
                                    for k in (0..num_basis) {
                                        if k<=ic {
                                            vk_ic[..k].iter_mut()
                                                .zip(reduce_ij[kl..kl+k].iter())
                                                .for_each(|(i,j)| {*i += j*dm_jc[k]});
                                            vk_ic[k] += reduce_ij[kl..kl+k]
                                                .iter()
                                                .zip(dm_jc[..k].iter()).fold(0.0, |acc,(i,j)| acc+i*j);
                                            kl += k;
                                            vk_ic[k] += reduce_ij[kl] *dm_jc[k];
                                            kl += 1;
                                        } else {
                                            vk_ic[..ic+1].iter_mut()
                                                .zip(reduce_ij[kl..kl+ic+1].iter())
                                                .for_each(|(i,j)| {*i += j*dm_jc[k]});
                                            kl += k+1;
                                        }
                                    }
                                    //=================================================
                                    let mut vk_jc = vk_thread.get1d_slice_mut(jc*(jc+1)/2,jc+1).unwrap();
                                    let mut kl = 0_usize;
                                    for k in (0..num_basis) {
                                        if k<=jc {
                                            vk_jc[..k].iter_mut()
                                                .zip(reduce_ij[kl..kl+k].iter())
                                                .for_each(|(i,j)| {*i += j*dm_ic[k]});
                                            vk_jc[k] += reduce_ij[kl..kl+k]
                                                .iter()
                                                .zip(dm_ic[..k].iter()).fold(0.0, |acc,(i,j)| acc+i*j);
                                            kl += k;
                                            vk_jc[k] += reduce_ij[kl] *dm_ic[k];
                                            kl += 1;
                                        } else {
                                            vk_jc[..jc+1].iter_mut()
                                                .zip(reduce_ij[kl..kl+jc+1].iter())
                                                .for_each(|(i,j)| {*i += j*dm_ic[k]});
                                            kl += k+1;
                                        }
                                    }
                                    //=================================================
                                }
                                let ijkl_start = (jc*(jc+1)/2+jc)*npair;
                                let reduce_ij = ijkl.get1d_slice(ijkl_start,npair).unwrap();
                                //let reduce_ij = ijkl.get4d_slice([0,0,ic,ic],npair).unwrap();
                                let dm_jc = dm[i_spin].get1d_slice(jc*num_basis,num_basis).unwrap();
                                let mut vk_jc = vk_thread.get1d_slice_mut(jc*(jc+1)/2,jc+1).unwrap();
                                let mut kl = 0_usize;
                                for k in (0..num_basis) {
                                    if k<=jc {
                                        vk_jc[..k].iter_mut()
                                            .zip(reduce_ij[kl..kl+k].iter())
                                            .for_each(|(i,j)| {*i += j*dm_jc[k]});
                                        vk_jc[k] += reduce_ij[kl..kl+k]
                                            .iter()
                                            .zip(dm_jc[..k].iter()).fold(0.0, |acc,(i,j)| acc+i*j);
                                        kl += k;
                                        vk_jc[k] += reduce_ij[kl] *dm_jc[k];
                                        kl += 1;
                                    } else {
                                        vk_jc[..jc+1].iter_mut()
                                            .zip(reduce_ij[kl..kl+jc+1].iter())
                                            .for_each(|(i,j)| {*i += j*dm_jc[k]});
                                        kl += k+1;
                                    }
                                }
                            }
                            tx_jc_thread.send(vk_thread);
                        });
                    }
                    let jc_start_thread = (num_para-1)*num_chunck.0;
                    let jc_end_thread = jc_start_thread + num_chunck.1;
                    let tx_jc_thread = tx_jc;
                    let handle = s_thread.spawn(move |_| {
                        let mut vk_thread =  MatrixUpper::new(npair,0.0f64);
                        for jc in (jc_start_thread..jc_end_thread) {
                            for ic in (0..jc) {
                                let ijkl_start = (jc*(jc+1)/2+ic)*npair;
                                let reduce_ij = ijkl.get1d_slice(ijkl_start,npair).unwrap();
                                //let reduce_ij = ijkl.get4d_slice([0,0,ic,jc],npair).unwrap();
                                let dm_ic = dm[i_spin].get1d_slice(ic*num_basis,num_basis).unwrap();
                                let dm_jc = dm[i_spin].get1d_slice(jc*num_basis,num_basis).unwrap();
                                let mut vk_ic = vk_thread.get1d_slice_mut(ic*(ic+1)/2,ic+1).unwrap();
                                let mut kl = 0_usize;
                                for k in (0..num_basis) {
                                    if k<=ic {
                                        vk_ic[..k].iter_mut()
                                            .zip(reduce_ij[kl..kl+k].iter())
                                            .for_each(|(i,j)| {*i += j*dm_jc[k]});
                                        vk_ic[k] += reduce_ij[kl..kl+k]
                                            .iter()
                                            .zip(dm_jc[..k].iter()).fold(0.0, |acc,(i,j)| acc+i*j);
                                        kl += k;
                                        vk_ic[k] += reduce_ij[kl] *dm_jc[k];
                                        kl += 1;
                                    } else {
                                        vk_ic[..ic+1].iter_mut()
                                            .zip(reduce_ij[kl..kl+ic+1].iter())
                                            .for_each(|(i,j)| {*i += j*dm_jc[k]});
                                        kl += k+1;
                                    }
                                }
                                //=================================================
                                let mut vk_jc = vk_thread.get1d_slice_mut(jc*(jc+1)/2,jc+1).unwrap();
                                let mut kl = 0_usize;
                                for k in (0..num_basis) {
                                    if k<=jc {
                                        vk_jc[..k].iter_mut()
                                            .zip(reduce_ij[kl..kl+k].iter())
                                            .for_each(|(i,j)| {*i += j*dm_ic[k]});
                                        vk_jc[k] += reduce_ij[kl..kl+k]
                                            .iter()
                                            .zip(dm_ic[..k].iter()).fold(0.0, |acc,(i,j)| acc+i*j);
                                        kl += k;
                                        vk_jc[k] += reduce_ij[kl] *dm_ic[k];
                                        kl += 1;
                                    } else {
                                        vk_jc[..jc+1].iter_mut()
                                            .zip(reduce_ij[kl..kl+jc+1].iter())
                                            .for_each(|(i,j)| {*i += j*dm_ic[k]});
                                        kl += k+1;
                                    }
                                }
                                //=================================================
                            }
                            let ijkl_start = (jc*(jc+1)/2+jc)*npair;
                            let reduce_ij = ijkl.get1d_slice(ijkl_start,npair).unwrap();
                            //let reduce_ij = ijkl.get4d_slice([0,0,ic,ic],npair).unwrap();
                            let dm_jc = dm[i_spin].get1d_slice(jc*num_basis,num_basis).unwrap();
                            let mut vk_jc = vk_thread.get1d_slice_mut(jc*(jc+1)/2,jc+1).unwrap();
                            let mut kl = 0_usize;
                            for k in (0..num_basis) {
                                if k<=jc {
                                    vk_jc[..k].iter_mut()
                                        .zip(reduce_ij[kl..kl+k].iter())
                                        .for_each(|(i,j)| {*i += j*dm_jc[k]});
                                    vk_jc[k] += reduce_ij[kl..kl+k]
                                        .iter()
                                        .zip(dm_jc[..k].iter()).fold(0.0, |acc,(i,j)| acc+i*j);
                                    kl += k;
                                    vk_jc[k] += reduce_ij[kl] *dm_jc[k];
                                    kl += 1;
                                } else {
                                    vk_jc[..jc+1].iter_mut()
                                        .zip(reduce_ij[kl..kl+jc+1].iter())
                                        .for_each(|(i,j)| {*i += j*dm_jc[k]});
                                    kl += k+1;
                                }
                            }
                        }
                        tx_jc_thread.send(vk_thread);
                    });
                    for received in rx_jc {
                        vk[i_spin].data.iter_mut()
                            .zip(received.data)
                            .for_each(|(i,j)| {*i += j});
                    }
                }).unwrap();
            }
        }
        if scaling_factor!=1.0f64 {
            for i_spin in (0..spin_channel) {
                vk[i_spin].data.par_iter_mut().for_each(|f| *f = *f*scaling_factor)
            }
        };
        vk
    }
    pub fn generate_hf_hamiltonian_erifold4(&mut self) {
        let num_basis = self.mol.num_basis;
        let num_state = self.mol.num_state;
        let spin_channel = self.mol.spin_channel;
        //let homo = &self.homo;
        //let vj = if self.mol.ctrl.num_threads>1 {
        //    self.generate_vj_with_erifold4_sync(1.0);
        //} else {
        //    self.generate_vj_with_erifold4(1.0)
        //};
        //let vk = if self.mol.ctrl.num_threads>1 {
        //    self.generate_vk_with_erifold4_sync_v02(-0.5);
        //} else {
        //    self.generate_vk_with_erifold4_v02(-0.5)
        //};
        let vj = self.generate_vj_with_erifold4_sync(1.0);
        let scaling_factor = match self.scftype {
            SCFType::RHF => -0.5,
            _ => -1.0,
        };
        let vk = self.generate_vk_with_erifold4_sync(scaling_factor);
        // let tmp_matrix = &self.h_core;
        // let mut tmp_num = 0;
        // let (i_len,j_len) =  (self.mol.num_basis,self.mol.num_basis);
        // let (k_len,l_len) =  (self.mol.num_basis,self.mol.num_basis);
        // tmp_matrix.data.iter().enumerate().for_each(|value| {
        //     if value.1.abs()>1.0e-1 {
        //         println!("I= {:2} Value= {:16.8}",value.0,value.1);
        //     }
        // });
        for i_spin in (0..spin_channel) {
            self.hamiltonian[i_spin] = self.h_core.clone();
            self.hamiltonian[i_spin].data
                            .par_iter_mut()
                            .zip(vj[0].data.par_iter())
                            .for_each(|(h_ij,vj_ij)| {
                                *h_ij += vj_ij
                            });
            self.hamiltonian[i_spin].data
                            .par_iter_mut()
                            .zip(vj[1].data.par_iter())
                            .for_each(|(h_ij,vj_ij)| {
                                *h_ij += vj_ij
                            });
            self.hamiltonian[i_spin].data
                            .par_iter_mut()
                            .zip(vk[i_spin].data.par_iter())
                            .for_each(|(h_ij,vk_ij)| {
                                *h_ij += vk_ij
                            });
        };
    }
    pub fn generate_hf_hamiltonian_ri_v(&mut self) {
        let num_basis = self.mol.num_basis;
        let num_state = self.mol.num_state;
        let spin_channel = self.mol.spin_channel;
        //let homo = &self.homo;
        let dt1 = time::Local::now();
        let vj = self.generate_vj_with_ri_v_sync(1.0);
        let dt2 = time::Local::now();
        let scaling_factor = match self.scftype {
            SCFType::RHF => -0.5,
            _ => -1.0,
        };
        let vk = self.generate_vk_with_ri_v(scaling_factor);
        let dt3 = time::Local::now();
        let timecost1 = (dt2.timestamp_millis()-dt1.timestamp_millis()) as f64 /1000.0;
        let timecost2 = (dt3.timestamp_millis()-dt2.timestamp_millis()) as f64 /1000.0;
        if self.mol.ctrl.print_level>2 {
            println!("The evaluation of Vj and Vk matrices cost {:10.2} and {:10.2} seconds, respectively",
                      timecost1,timecost2);
        }
        for i_spin in (0..spin_channel) {
            self.hamiltonian[i_spin] = self.h_core.clone();
            self.hamiltonian[i_spin].data
                            .par_iter_mut()
                            .zip(vj[0].data.par_iter())
                            .for_each(|(h_ij,vj_ij)| {
                                *h_ij += vj_ij
                            });
            self.hamiltonian[i_spin].data
                            .par_iter_mut()
                            .zip(vj[1].data.par_iter())
                            .for_each(|(h_ij,vj_ij)| {
                                *h_ij += vj_ij
                            });
            self.hamiltonian[i_spin].data
                            .par_iter_mut()
                            .zip(vk[i_spin].data.par_iter())
                            .for_each(|(h_ij,vk_ij)| {
                                *h_ij += vk_ij
                            });
        };
    }
    pub fn generate_ks_hamiltonian_erifold4(&mut self) -> (f64,f64) {
        let num_basis = self.mol.num_basis;
        let num_state = self.mol.num_state;
        let spin_channel = self.mol.spin_channel;
        let mut exc_total = 0.0;
        let mut vxc_total = 0.0;
        //let homo = &self.homo;
        for i_spin in (0..spin_channel) {
            self.hamiltonian[i_spin] = self.h_core.clone();
        }
        let dt1 = time::Local::now();
        //let vj = self.generate_vj_with_ri_v_sync(1.0);
        let vj = self.generate_vj_with_erifold4_sync(1.0);
        for i_spin in (0..spin_channel) {
            self.hamiltonian[i_spin].data
                .par_iter_mut()
                .zip(vj[0].data.par_iter())
                .for_each(|(h_ij,vj_ij)| {
                    *h_ij += vj_ij
                });
            self.hamiltonian[i_spin].data
                .par_iter_mut()
                .zip(vj[1].data.par_iter())
                .for_each(|(h_ij,vj_ij)| {
                    *h_ij += vj_ij
                });
        }
        /////for debug purpose
        //let mut ecoul = 0.0;
        //for i_spin in (0..spin_channel) {
        //    let dm_s = &self.density_matrix[i_spin];
        //    let dm_upper = dm_s.to_matrixupper();
        //    ecoul += SCF::par_energy_contraction(&dm_upper, &vj[i_spin]);
        //}
        //println!("debug ecoul: {:16.8}", ecoul);
        let dt2 = time::Local::now();
        let scaling_factor = match self.scftype {
            SCFType::RHF => -0.5,
            _ => -1.0,
        }*self.mol.xc_data.dfa_hybrid_scf ;
        if ! scaling_factor.eq(&0.0) {
            //let vk = self.generate_vk_with_ri_v(scaling_factor);
            let vk = self.generate_vk_with_erifold4_sync(scaling_factor);
            for i_spin in (0..spin_channel) {
                self.hamiltonian[i_spin].data
                    .par_iter_mut()
                    .zip(vk[i_spin].data.par_iter())
                    .for_each(|(h_ij,vk_ij)| {
                        *h_ij += vk_ij
                    });
            };
        }
        let dt3 = time::Local::now();
        if self.mol.xc_data.dfa_compnt_scf.len()!=0 {
            let (exc,vxc) = self.generate_vxc(1.0);
            //println!("{:?}",vxc[0].data);
            for i_spin in (0..spin_channel) {
                self.hamiltonian[i_spin].data
                                .par_iter_mut()
                                .zip(vxc[i_spin].data.par_iter())
                                .for_each(|(h_ij,vk_ij)| {
                                    *h_ij += vk_ij
                                });
            };
            exc_total = exc;
            for i_spin in (0..spin_channel) {
                let dm_s = &self.density_matrix[i_spin];
                let dm_upper = dm_s.to_matrixupper();
                vxc_total += SCF::par_energy_contraction(&dm_upper, &vxc[i_spin]);
            }
        }

        let dt4 = time::Local::now();
        
        let timecost1 = (dt2.timestamp_millis()-dt1.timestamp_millis()) as f64 /1000.0;
        let timecost2 = (dt3.timestamp_millis()-dt2.timestamp_millis()) as f64 /1000.0;
        let timecost3 = (dt4.timestamp_millis()-dt3.timestamp_millis()) as f64 /1000.0;
        if self.mol.ctrl.print_level>2 {
            println!("The evaluation of Vj, Vk and Vxc matrices cost {:10.2}, {:10.2} and {:10.2} seconds, respectively",
                      timecost1,timecost2, timecost3);
        };
        (exc_total, vxc_total)

    }
    pub fn generate_ks_hamiltonian_ri_v(&mut self) -> (f64,f64) {
        let num_basis = self.mol.num_basis;
        let num_state = self.mol.num_state;
        let spin_channel = self.mol.spin_channel;
        let mut exc_total = 0.0;
        let mut vxc_total = 0.0;
        let mut vk_total = 0.0;
        //let homo = &self.homo;
        for i_spin in (0..spin_channel) {
            self.hamiltonian[i_spin] = self.h_core.clone();
        }
        let dt1 = time::Local::now();
        let vj = self.generate_vj_with_ri_v_sync(1.0);
        for i_spin in (0..spin_channel) {
            self.hamiltonian[i_spin].data
                .par_iter_mut()
                .zip(vj[0].data.par_iter())
                .for_each(|(h_ij,vj_ij)| {
                    *h_ij += vj_ij
                });
            self.hamiltonian[i_spin].data
                .par_iter_mut()
                .zip(vj[1].data.par_iter())
                .for_each(|(h_ij,vj_ij)| {
                    *h_ij += vj_ij
                });
        }
        /////for debug purpose
        //let mut ecoul = 0.0;
        //for i_spin in (0..spin_channel) {
        //    let dm_s = &self.density_matrix[i_spin];
        //    let dm_upper = dm_s.to_matrixupper();
        //    ecoul += SCF::par_energy_contraction(&dm_upper, &vj[i_spin]);
        //}
        //println!("debug ecoul: {:16.8}", ecoul);
        let dt2 = time::Local::now();
        let scaling_factor = match self.scftype {
            SCFType::RHF => -0.5,
            _ => -1.0,
        }*self.mol.xc_data.dfa_hybrid_scf ;
        if ! scaling_factor.eq(&0.0) {
            let vk = self.generate_vk_with_ri_v(scaling_factor);
            for i_spin in (0..spin_channel) {
                self.hamiltonian[i_spin].data
                    .par_iter_mut()
                    .zip(vk[i_spin].data.par_iter())
                    .for_each(|(h_ij,vk_ij)| {
                        *h_ij += vk_ij
                    });
            };
        }
        let dt3 = time::Local::now();
        if self.mol.xc_data.dfa_compnt_scf.len()!=0 {
            let (exc,vxc) = self.generate_vxc_rayon(1.0);
            //let (exc,vxc) = self.generate_vxc(1.0);
            let _ = utilities::timing(&dt3, Some("evaluate vxc total"));
            for i_spin in (0..spin_channel) {
                self.hamiltonian[i_spin].data
                                .par_iter_mut()
                                .zip(vxc[i_spin].data.par_iter())
                                .for_each(|(h_ij,vk_ij)| {
                                    *h_ij += vk_ij
                                });
            };
            exc_total = exc;
            for i_spin in (0..spin_channel) {
                let dm_s = &self.density_matrix[i_spin];
                let dm_upper = dm_s.to_matrixupper();
                vxc_total += SCF::par_energy_contraction(&dm_upper, &vxc[i_spin]);
            }
        }

        let dt4 = time::Local::now();
        
        let timecost1 = (dt2.timestamp_millis()-dt1.timestamp_millis()) as f64 /1000.0;
        let timecost2 = (dt3.timestamp_millis()-dt2.timestamp_millis()) as f64 /1000.0;
        let timecost3 = (dt4.timestamp_millis()-dt3.timestamp_millis()) as f64 /1000.0;
        if self.mol.ctrl.print_level>2 {
            println!("The evaluation of Vj, Vk and Vxc matrices cost {:10.2}, {:10.2} and {:10.2} seconds, respectively",
                      timecost1,timecost2, timecost3);
        };
        (exc_total, vxc_total)

    }
    pub fn generate_hf_hamiltonian(&mut self) {
        let num_basis = self.mol.num_basis;
        let num_state = self.mol.num_state;
        let spin_channel = self.mol.spin_channel;
        let mut exc_total = 0.0;
        let mut vxc_total = 0.0;
        if self.mol.xc_data.dfa_compnt_scf.len() == 0 {
            if self.mol.ctrl.eri_type.eq("analytic") {
                self.generate_hf_hamiltonian_erifold4();
            } else if  self.mol.ctrl.eri_type.eq("ri_v") {
                self.generate_hf_hamiltonian_ri_v();
            }
        } else {
            if self.mol.ctrl.eri_type.eq("analytic") {
                //panic!("Hybrid DFA is not implemented with analytic ERI.");
                let (tmp_exc_total,tmp_vxc_total) = self.generate_ks_hamiltonian_erifold4();
                exc_total = tmp_exc_total;
                vxc_total = tmp_vxc_total;
            } else {
                let (tmp_exc_total,tmp_vxc_total) = self.generate_ks_hamiltonian_ri_v();
                exc_total = tmp_exc_total;
                vxc_total = tmp_vxc_total;
            }
        }

        let dm = &self.density_matrix;

        // The following scf energy evaluation formula is obtained from 
        // the quantum chemistry book of Szabo A. and Ostlund N.S. P 150, Formula (3.184)
        self.scf_energy = self.nuc_energy;
        //println!("debug: {}", self.scf_energy);
        // for DFT calculations, we should replace the exchange-correlation (xc) potential by the xc energy
        self.scf_energy = self.scf_energy - vxc_total + exc_total;
        println!("Exc: {:?}, Vxc: {:?}", exc_total, vxc_total);
        match self.scftype {
            SCFType::RHF => {
                // D*(H^{core}+F)
                let dm_s = &dm[0];
                let hc = &self.h_core;
                let ht_s = &self.hamiltonian[0];
                let dm_upper = dm_s.to_matrixupper();
                let mut hc_and_ht = hc.clone();
                hc_and_ht.data.par_iter_mut().zip(ht_s.data.par_iter()).for_each(|value| {
                    *value.0 += value.1
                });
                self.scf_energy += SCF::par_energy_contraction(&dm_upper, &hc_and_ht);
            },
            _ => {
                let dm_a = &dm[0];
                let dm_a_upper = dm_a.to_matrixupper();
                let dm_b = &dm[1];
                let dm_b_upper = dm_b.to_matrixupper();
                let mut dm_t_upper = dm_a_upper.clone();
                dm_t_upper.data.par_iter_mut().zip(dm_b_upper.data.par_iter()).for_each(|value| {*value.0+=value.1});

                // Now for D^{tot}*H^{core} term
                self.scf_energy += SCF::par_energy_contraction(&dm_t_upper, &self.h_core);
                //println!("debug: {}", self.scf_energy);
                // Now for D^{alpha}*F^{alpha} term
                self.scf_energy += SCF::par_energy_contraction(&dm_a_upper, &self.hamiltonian[0]);
                //println!("debug: {}", self.scf_energy);
                // Now for D^{beta}*F^{beta} term
                self.scf_energy += SCF::par_energy_contraction(&dm_b_upper, &self.hamiltonian[1]);
                //println!("debug: {}", self.scf_energy);

            },
        }
    }

    /// about total energy contraction:
    /// E0 = 1/2*\sum_{i}\sum_{j}a_{ij}*b_{ij}
    /// einsum('ij,ij')
    pub fn par_energy_contraction(a:&MatrixUpper<f64>, b:&MatrixUpper<f64>) -> f64 {
        let (sender, receiver) = channel();
        a.data.par_iter().zip(b.data.par_iter()).for_each_with(sender, |s,(dm,hc)| {
            let mut tmp_scf_energy = 0.0_f64;
            tmp_scf_energy += dm*(hc);
            s.send(tmp_scf_energy).unwrap();
        });
        let mut tmp_energy = 2.0_f64 * receiver.into_iter().sum::<f64>();
        let a_diag = a.get_diagonal_terms().unwrap();
        let b_diag = b.get_diagonal_terms().unwrap();
        let double_count = a_diag.par_iter().zip(b_diag.par_iter()).fold(|| 0.0_f64,|acc,(a,b)| {
            acc + *a*(*b)
        }).sum::<f64>();

        (tmp_energy - double_count) * 0.5
    }

    pub fn evaluate_exact_exchange_ri_v(&mut self) -> f64 {
        let mut x_energy = 0.0;
        let mut vk = self.generate_vk_with_ri_v(1.0);
        let spin_channel = self.mol.spin_channel;
        for i_spin in 0..spin_channel {
            let dm_s = &self.density_matrix[i_spin];
            let dm_upper = dm_s.to_matrixupper();
            x_energy += SCF::par_energy_contraction(&dm_upper, &vk[i_spin]);
        }
        if self.mol.spin_channel==1 {
            // the factor of 0.5 is due to the use of full density matrix for the exchange energy evaluation
            x_energy*-0.5
        } else {
            x_energy*-1.0
        }
    }

    pub fn evaluate_xc_energy(&mut self, iop: usize) -> f64 {
        let num_basis = self.mol.num_basis;
        let num_state = self.mol.num_state;
        let num_auxbas = self.mol.num_auxbas;
        let npair = num_basis*(num_basis+1)/2;
        let spin_channel = self.mol.spin_channel;
        //let mut vxc: MatrixUpper<f64> = MatrixUpper::new(1,0.0f64);
        let mut exc_spin:Vec<f64> = vec![];
        let dm = &mut self.density_matrix;
        let mo = &mut self.eigenvectors;
        let occ = &mut self.occupation;
        if let Some(grids) = &mut self.grids {
            exc_spin = self.mol.xc_data.xc_exc(grids, spin_channel,dm, mo, occ,iop);
        }
        let exc:f64 = exc_spin.iter().sum();
        exc
    }
       

    pub fn diagonalize_hamiltonian(&mut self) {
        let spin_channel = self.mol.spin_channel;
        let num_state = self.mol.num_state;
        let dt1 = time::Local::now();
        match self.scftype {
            SCFType::ROHF => {
                let (eigenvector_spin, eigenvalue_spin)=
                    self.hamiltonian[0].to_matrixupperslicemut()
                    .lapack_dspgvx(self.ovlp.to_matrixupperslicemut(),num_state).unwrap();
                self.eigenvectors[0] = eigenvector_spin;
                self.eigenvalues[0] = eigenvalue_spin;
                self.eigenvectors[1] = self.eigenvectors[0].clone();
                self.eigenvalues[1] = self.eigenvalues[0].clone();
            },
            _ => {
                for i_spin in (0..spin_channel) {
                    let (eigenvector_spin, eigenvalue_spin)=
                        self.hamiltonian[i_spin].to_matrixupperslicemut()
                        .lapack_dspgvx(self.ovlp.to_matrixupperslicemut(),num_state).unwrap();
                    self.eigenvectors[i_spin] = eigenvector_spin;
                    self.eigenvalues[i_spin] = eigenvalue_spin;
                }
            }
        }
        //self.formated_eigenvalues(num_state);
        let dt2 = time::Local::now();
        let timecost1 = (dt2.timestamp_millis()-dt1.timestamp_millis()) as f64 /1000.0;
        if self.mol.ctrl.print_level>1 {
            println!("Hamiltonian eigensolver:  {:10.2}s", timecost1);
        }
    }

    pub fn check_scf_convergence(&mut self, scftracerecode: &ScfTraceRecord) -> [bool;2] {
        //println!("debug spin_channel: {}", self.mol.spin_channel);
        if scftracerecode.num_iter<2 {
            return [false,false]
        }
        let spin_channel = self.mol.spin_channel;
        let num_basis = self.mol.num_basis as f64;
        let max_scf_cycle = self.mol.ctrl.max_scf_cycle;
        let scf_acc_rho   = self.mol.ctrl.scf_acc_rho;   
        let scf_acc_eev   = self.mol.ctrl.scf_acc_eev;  
        let scf_acc_etot  = self.mol.ctrl.scf_acc_etot; 
        let mut flag   = [true,true];

        let cur_index = 1;
        let pre_index = 0;
        let cur_energy = self.scf_energy;
        let pre_energy = scftracerecode.scf_energy;
        let diff_energy = cur_energy-pre_energy;
        let etot_converge = diff_energy.abs()<=scf_acc_etot;

        let cur_energy = &self.eigenvalues;
        let pre_energy = &scftracerecode.eigenvalues;
        //let eev_converge = true
        let mut eev_err = 0.0;
        for i_spin in 0..spin_channel {
            //eev_err += cur_energy[i_spin].iter()
            //    .zip(pre_energy[i_spin].iter())
            //    .fold(0.0,|acc,(c,p)| acc + (c-p).powf(2.0));
            // rayon parallel version
            eev_err += cur_energy[i_spin].par_iter()
                .zip(pre_energy[i_spin].par_iter())
                .map(|(c,p)| (c-p).powf(2.0)).sum::<f64>();
        }
        eev_err = eev_err.sqrt();

        let mut dm_err = [0.0;2];
        //let cur_index = &scftracerecode.residual_density.len()-1;
        //let cur_residual_density = &scftracerecode.residual_density[cur_index];
        let cur_dm = &self.density_matrix;
        let pre_dm = &scftracerecode.density_matrix[1];

        for i_spin in 0..spin_channel {
            //dm_err[i_spin] = cur_dm[i_spin].data.iter()
            //    .zip(pre_dm[i_spin].data.iter())
            //    .fold(0.0,|acc,(c,p)| acc + (c-p).powf(2.0)).sqrt()/num_basis;
            // rayon parallel version
            dm_err[i_spin] = cur_dm[i_spin].data.par_iter()
                .zip(pre_dm[i_spin].data.par_iter())
                .map(|(c,p)| (c-p).powf(2.0)).sum::<f64>().sqrt()/num_basis;
            //dm_err[i_spin] = cur_residual_density[i_spin].data.par_iter()
            //    .map(|c| c.powf(2.0)).sum::<f64>().sqrt()/num_basis;
        }

        if self.mol.ctrl.print_level>=1 {
            if spin_channel==1 {
                println!("SCF Change: DM {:10.5e}; eev {:10.5e} Ha; etot {:10.5e} Ha",
                          dm_err[0],eev_err,diff_energy);
                flag[0] = diff_energy.abs()<=scf_acc_etot &&
                          dm_err[0] <=scf_acc_rho &&
                          eev_err <= scf_acc_eev
            } else {
                println!("SCF Change: DM ({:10.5e},{:10.5e}); eev {:10.5e} Ha; etot {:10.5e} Ha",
                          dm_err[0],dm_err[1],eev_err,diff_energy);
                flag[0] = diff_energy.abs()<=scf_acc_etot &&
                          dm_err[0] <=scf_acc_rho &&
                          dm_err[1] <=scf_acc_rho &&
                          eev_err <= scf_acc_eev
            }
        }

                  
        // Now check if max_scf_cycle is reached or not
        flag[1] = scftracerecode.num_iter >= max_scf_cycle;

        flag
    }

    pub fn formated_eigenvalues(&mut self,num_state_to_print:usize) {
        let spin_channel = self.mol.spin_channel;
        if spin_channel==1 {
            println!("{:>8}{:>14}{:>18}",String::from("State"),
                                        String::from("Occupation"),
                                        String::from("Eigenvalue"));
            for i_state in (0..num_state_to_print) {
                println!("{:>8}{:>14.5}{:>18.6}",i_state,self.occupation[0][i_state],self.eigenvalues[0][i_state]);
            }
        } else {
            for i_spin in (0..spin_channel) {
                if i_spin == 0 {
                    println!("Spin-up eigenvalues");
                    println!(" ");
                } else {
                    println!(" ");
                    println!("Spin-down eigenvalues");
                    println!(" ");
                }
                println!("{:>8}{:>14}{:>18}",String::from("State"),
                                            String::from("Occupation"),
                                            String::from("Eigenvalue"));
                for i_state in (0..num_state_to_print) {
                    println!("{:>8}{:>14.5}{:>18.6}",i_state,self.occupation[i_spin][i_state],self.eigenvalues[i_spin][i_state]);
                }
            }
        }
    }
    pub fn formated_eigenvectors(&self) {
        let spin_channel = self.mol.spin_channel;
        if spin_channel==1 {
            self.eigenvectors[0].formated_output(5, "full");
        } else {
            (0..spin_channel).into_iter().for_each(|i_spin|{
                if i_spin == 0 {
                    println!("Spin-up eigenvalues");
                    println!(" ");
                } else {
                    println!(" ");
                    println!("Spin-down eigenvalues");
                    println!(" ");
                }
                self.eigenvectors[i_spin].formated_output(5, "full");
            });
        }
    }
    pub fn save_chkfile(&self) {
        if self.mol.ctrl.restart {
        }
    }
    pub fn save_fchk_of_gaussian(&self) {
        use crate::utilities::convert_scientific_notation_to_fortran_format as r2f;
        use rest_libcint::CINTR2CDATA;
        use regex::Regex;
        let re_basis = Regex::new(r"/?(?P<basis>[^/]*)/?$").unwrap();
        let cap = re_basis.captures(&self.mol.ctrl.basis_path).unwrap();
        let basis_name = cap.name("basis").unwrap().to_string();

        let mut input = fs::File::create(&format!("{}.fchk",&self.mol.geom.name)).unwrap();
        write!(input, "{:}\n", &self.mol.geom.name);
        if self.mol.spin_channel==1 {
            write!(input, "Freq      R{:-59}{:-20}\n",
                self.mol.ctrl.xc.to_uppercase(),
                basis_name.to_uppercase()
            );
        } else {
            write!(input, "Freq      U{:-59}{:-20}\n",
                self.mol.ctrl.xc.to_uppercase(),
                basis_name.to_uppercase()
            );
        };
        write!(input, "Number of atom                             I {:16}\n", self.mol.geom.elem.len());
        write!(input, "Number of electrons                        I {:16}\n", self.mol.num_elec[0]);
        write!(input, "Number of alpha electrons                  I {:16}\n", self.mol.num_elec[1]);
        write!(input, "Number of beta electrons                   I {:16}\n", self.mol.num_elec[2]);
        write!(input, "Number of basis functions                  I {:16}\n", self.mol.num_basis);
        write!(input, "Number of independent functions            I {:16}\n", self.mol.num_state);
        // ==============================
        // Now for atomic numbers
        // ==============================
        write!(input, "Atomic numbers                             I   N={:12}\n", self.mol.geom.elem.len());
        let mut i_index = 0;
        self.mol.geom.elem.iter().for_each(|x| {
            let (mass, charge) = SPECIES_INFO.get(x.as_str()).unwrap();
            if (i_index + 1)%5 == 0 {
                write!(input, "{:12}\n",*charge as i32)
            } else {
                write!(input, "{:12}",*charge as i32)
            };
            i_index += 1;
        });
        if i_index % 5 != 0 {write!(input, "\n");}
        // ==============================
        // Now for Nuclear charges
        // ==============================
        write!(input, "Nuclear charges                            R   N={:12}\n", self.mol.geom.elem.len());
        let mut i_index = 0;
        self.mol.geom.elem.iter().for_each(|x| {
            let (mass, charge) = SPECIES_INFO.get(x.as_str()).unwrap();
            let sdd = format!("{:16.8E}", charge);
            if (i_index + 1)%5 == 0 {
                write!(input, "{}\n",r2f(&sdd))
            } else {
                write!(input, "{}",r2f(&sdd))
            };
            i_index += 1;
        });
        if i_index % 5 != 0 {write!(input, "\n");}
        // ==============================
        // Now for geometry coordinates
        // ==============================
        write!(input, "Current cartesian coordinates              R   N={:12}\n", self.mol.geom.elem.len()*3);
        let mut i_index = 0;
        self.mol.geom.position.iter_columns_full().for_each(|x_position| {
            x_position.iter().for_each(|x| {
                let sdd = format!("{:16.8E}\n",x);
                if (i_index + 1)%5 ==0 {
                    write!(input,"{}\n", r2f(&sdd));
                } else {
                    write!(input,"{}",r2f(&sdd));
                }
                i_index +=1;
            })
        });
        if i_index % 5 != 0 {write!(input, "\n");}
        // ==============================
        // Now for atomic weights
        // ==============================
        write!(input, "Real atomic weights                        R   N={:12}\n", self.mol.geom.elem.len());
        let mut i_index = 0;
        self.mol.geom.elem.iter().for_each(|x| {
            let (mass, charge) = SPECIES_INFO.get(x.as_str()).unwrap();
            let sdd = format!("{:16.8E}", mass);
            if (i_index + 1)%5 == 0 {
                write!(input, "{}\n",r2f(&sdd))
            } else {
                write!(input, "{}",r2f(&sdd))
            };
            i_index += 1;
        });
        if i_index % 5 != 0 {write!(input, "\n");}
        // ==============================
        // Now for MicOpt
        // ==============================
        write!(input, "MicOpt                                     I   N={:12}\n", self.mol.geom.elem.len());
        let mut i_index = 0;
        self.mol.geom.elem.iter().for_each(|x| {
            if (i_index + 1)%5 == 0 {
                write!(input, "{:12}\n", -1)
            } else {
                write!(input, "{:12}", -1)
            };
            i_index += 1;
        });
        if i_index % 5 != 0 {write!(input, "\n");}

        //================================
        // Now for basis set info.
        //================================
        let mut num_contract = 0;
        let mut num_primitiv = 0;
        let mut num_d_shell = 0;
        let mut num_f_shell = 0;
        let mut max_ang = 0;
        let mut max_contract = 0;
        let mut shell_type: Vec<i32> = vec![];
        let mut num_primitiv_vec: Vec<usize> = vec![];
        let mut shell_to_atom_map: Vec<usize> = vec![];
        let mut primitive_exp: Vec<f64> = vec![];
        let mut coord_each_shell: Vec<f64> = vec![];
        let mut coeff_vec:Vec<f64> = vec![];
        let shell_type_fac = match self.mol.cint_type {
            CintType::Spheric => -1,
            CintType::Cartesian => 1,
        };
        self.mol.basis4elem.iter().enumerate().for_each(|(i_atom,ibas)| {
            ibas.electron_shells.iter().for_each(|ibascell| {
                //let mut tmp_coeff_vec =  ibascell.coefficients.clone();
                ibascell.coefficients.iter().for_each(|coeff_i| {
                    // De-normalization
                    let mut tmp_coeff_i = coeff_i.clone();
                    let tmp_ang = ibascell.angular_momentum[0];
                    tmp_coeff_i.iter_mut().zip(ibascell.exponents.iter()).for_each(|(i, e)| {
                        *i /= CINTR2CDATA::gto_norm(tmp_ang, *e);
                    });
                    coeff_vec.extend(tmp_coeff_i.iter());
                    num_contract += 1;
                    num_primitiv += ibascell.exponents.len();
                    max_ang = std::cmp::max(max_ang, tmp_ang);
                    max_contract = std::cmp::max(max_contract, ibascell.exponents.len());
                    if ibascell.angular_momentum[0] == 2 {num_d_shell += 1};
                    if ibascell.angular_momentum[0] == 3 {num_f_shell += 1};
                    shell_type.push(ibascell.angular_momentum[0]*shell_type_fac);
                    num_primitiv_vec.push(ibascell.exponents.len());
                    shell_to_atom_map.push(i_atom+1);
                    primitive_exp.extend(ibascell.exponents.iter());
                    coord_each_shell.extend(self.mol.geom.position.iter_column(i_atom));

                });
            });
        });
        write!(input, "Number of contracted shells                I {:16}\n",num_contract);
        write!(input, "Number of primitive shells                 I {:16}\n",num_primitiv);
        write!(input, "Pure/Cartesian d shells                    I {:16}\n", num_d_shell);
        write!(input, "Pure/Cartesian f shells                    I {:16}\n", num_f_shell);
        write!(input, "Highest angular momentum                   I {:16}\n", max_ang);
        write!(input, "Largest degree of contraction              I {:16}\n", max_contract);

        // ==============================
        // Now for shell infor.
        // ==============================
        write!(input, "Shell types                                I   N={:12}\n", num_contract);
        let mut i_index = 0;
        shell_type.iter().for_each(|x| {
            if (i_index + 1)%6 == 0 {
                write!(input, "{:12}\n", x)
            } else {
                write!(input, "{:12}", x)
            };
            i_index += 1;
        });
        if i_index % 6 != 0 {write!(input, "\n");}

        write!(input, "Number of primitives per shell             I   N={:12}\n", num_contract);
        let mut i_index = 0;
        num_primitiv_vec.iter().for_each(|x| {
            if (i_index + 1)%6 == 0 {
                write!(input, "{:12}\n", x)
            } else {
                write!(input, "{:12}", x)
            };
            i_index += 1;
        });
        if i_index % 6 != 0 {write!(input, "\n");}

        write!(input, "Shell to atom map                          I   N={:12}\n", num_contract);
        let mut i_index = 0;
        shell_to_atom_map.iter().for_each(|x| {
            if (i_index + 1)%6 == 0 {
                write!(input, "{:12}\n", x)
            } else {
                write!(input, "{:12}", x)
            };
            i_index += 1;
        });
        if i_index % 6 != 0 {write!(input, "\n");}

        write!(input, "Primitive exponents                        R   N={:12}\n", primitive_exp.len());
        let mut i_index = 0;
        primitive_exp.iter().for_each(|x| {
            let sdd = format!("{:16.8E}", x);
            if (i_index + 1)%5 == 0 {
                write!(input, "{}\n",r2f(&sdd))
            } else {
                write!(input, "{}",r2f(&sdd))
            };
            i_index += 1;
        });
        if i_index % 5 != 0 {write!(input, "\n");}

        write!(input, "Contraction coefficients                   R   N={:12}\n", coeff_vec.len());
        let mut i_index = 0;
        coeff_vec.iter().for_each(|x| {
            let sdd = format!("{:16.8E}", x);
            if (i_index + 1)%5 == 0 {
                write!(input, "{}\n",r2f(&sdd))
            } else {
                write!(input, "{}",r2f(&sdd))
            };
            i_index += 1;
        });
        if i_index % 5 != 0 {write!(input, "\n");}

        write!(input, "P(S=P) Contraction coefficients            R   N={:12}\n", coeff_vec.len());
        let mut i_index = 0;
        coeff_vec.iter().for_each(|x| {
            let sdd = format!("{:16.8E}", 0);
            if (i_index + 1)%5 == 0 {
                write!(input, "{}\n",r2f(&sdd))
            } else {
                write!(input, "{}",r2f(&sdd))
            };
            i_index += 1;
        });
        if i_index % 5 != 0 {write!(input, "\n");}

        write!(input, "Coordinates of each shell                  R   N={:12}\n", coord_each_shell.len());
        let mut i_index = 0;
        coord_each_shell.iter().for_each(|x| {
            let sdd = format!("{:16.8E}", x);
            if (i_index + 1)%5 == 0 {
                write!(input, "{}\n",r2f(&sdd))
            } else {
                write!(input, "{}",r2f(&sdd))
            };
            i_index += 1;
        });
        if i_index % 5 != 0 {write!(input, "\n");}

        // ==============================
        // Now for total and orb energies
        // ==============================
        let dd = format!("{:27.15E}", self.scf_energy);
        write!(input, "SCF Energy                                 R {}\n",r2f(&dd));
        write!(input, "Total Energy                               R {}\n",r2f(&dd));
        for i_spin in 0..self.mol.spin_channel {
            if i_spin ==0  {
                write!(input, "Alpha Orbital Energies                     R   N={:12}\n", self.eigenvalues[i_spin].len());
            } else {
                write!(input, "Beta Orbital Energies                      R   N={:12}\n", self.eigenvalues[i_spin].len());
            }
            let mut i_index = 0;
            self.eigenvalues[i_spin].iter().for_each(|x| {
                let sdd = format!("{:16.8E}", x);
                if (i_index + 1)%5 == 0 {
                    write!(input, "{}\n",r2f(&sdd))
                } else {
                    write!(input, "{}",r2f(&sdd))
                };
                i_index += 1;
            });
            if i_index % 5 != 0 {write!(input, "\n");}
        }

        // ==============================
        // Now for orbital coefficients
        // ==============================
        for i_spin in 0..self.mol.spin_channel {
            if i_spin ==0  {
                write!(input, "Alpha MO coefficients                      R   N={:12}\n", self.eigenvectors[i_spin].data.len());
            } else {
                write!(input, "Beta MO coefficients                       R   N={:12}\n", self.eigenvectors[i_spin].data.len());
            }
            let mut i_index = 0;
            self.eigenvectors[i_spin].iter().for_each(|x| {
                let sdd = format!("{:16.8E}", x);
                if (i_index + 1)%5 == 0 {
                    write!(input, "{}\n",r2f(&sdd))
                } else {
                    write!(input, "{}",r2f(&sdd))
                };
                i_index += 1;
            });
            if i_index % 5 != 0 {write!(input, "\n");}
        }

        write!(input, "Contraction coefficients                   R   N={:12}\n", coeff_vec.len());
        let mut i_index = 0;
        coeff_vec.iter().for_each(|x| {
            let sdd = format!("{:16.8E}", x);
            if (i_index + 1)%5 == 0 {
                write!(input, "{}\n",r2f(&sdd))
            } else {
                write!(input, "{}",r2f(&sdd))
            };
            i_index += 1;
        });
        if i_index % 5 != 0 {write!(input, "\n");}


        input.sync_all().unwrap();
    }

    // relevant to RI-V
    pub fn generate_vj_with_ri_v(&mut self, scaling_factor: f64) -> Vec<MatrixUpper<f64>> {
        //let num_basis = self.mol.num_basis;
        //let num_auxbas = self.mol.num_auxbas;
        //let npair = num_basis*(num_basis+1)/2;
        let spin_channel = self.mol.spin_channel;
        let dm = &self.density_matrix;

        vj_upper_with_ri_v(&self.ri3fn, dm, spin_channel, scaling_factor)
    }
    pub fn generate_vj_with_ri_v_sync(&mut self, scaling_factor: f64) -> Vec<MatrixUpper<f64>> {
        //let num_basis = self.mol.num_basis;
        //let num_auxbas = self.mol.num_auxbas;
        //let npair = num_basis*(num_basis+1)/2;
        let spin_channel = self.mol.spin_channel;
        let dm = &self.density_matrix;

        if self.mol.ctrl.use_auxbas_symm {
            vj_upper_with_ri_v_sync(&self.ri3fn, dm, spin_channel, scaling_factor)
        } else {
            vj_upper_with_ri_v_sync(&self.ri3fn, dm, spin_channel, scaling_factor)
        }
    }

    pub fn generate_vk_with_ri_v(&mut self, scaling_factor: f64) -> Vec<MatrixUpper<f64>> {
        //let num_basis = self.mol.num_basis;
        //let num_state = self.mol.num_state;
        //let num_auxbas = self.mol.num_auxbas;
        //let npair = num_basis*(num_basis+1)/2;
        let spin_channel = self.mol.spin_channel;
        //let dm = &self.density_matrix;
        let eigv = &self.eigenvectors;

        vk_upper_with_ri_v_sync(&mut self.ri3fn, eigv, self.homo, &self.occupation, 
                                spin_channel, scaling_factor)
    }

    pub fn generate_vxc(&mut self, scaling_factor: f64) -> (f64, Vec<MatrixUpper<f64>>) {
        let num_basis = self.mol.num_basis;
        let num_state = self.mol.num_state;
        let num_auxbas = self.mol.num_auxbas;
        let npair = num_basis*(num_basis+1)/2;
        let spin_channel = self.mol.spin_channel;
        //let mut vxc: MatrixUpper<f64> = MatrixUpper::new(1,0.0f64);
        let mut vxc:Vec<MatrixUpper<f64>> = vec![MatrixUpper::empty();spin_channel];
        let mut exc_spin:Vec<f64> = vec![];
        let mut exc_total:f64 = 0.0;
        let mut vxc_mf:Vec<MatrixFull<f64>> = vec![MatrixFull::empty();spin_channel];
        let dm = &mut self.density_matrix;
        let mo = &mut self.eigenvectors;
        let occ = &mut self.occupation;
        if let Some(grids) = &mut self.grids {
            let dt0 = utilities::init_timing();
            let (exc,mut vxc_ao) = self.mol.xc_data.xc_exc_vxc(grids, spin_channel,dm, mo, occ);
            let dt1 = utilities::timing(&dt0, Some("Total vxc_ao time"));
            exc_spin = exc;
            if let Some(ao) = &mut grids.ao {
                // Evaluate the exchange-correlation energy
                //exc_total = izip!(grids.weights.iter(),exc.data.iter()).fold(0.0,|acc,(w,e)| {
                //    acc + w*e
                //});
                for i_spin in 0..spin_channel {
                    let vxc_mf_s = vxc_mf.get_mut(i_spin).unwrap();
                    *vxc_mf_s = MatrixFull::new([num_basis,num_basis],0.0f64);
                    let vxc_ao_s = vxc_ao.get_mut(i_spin).unwrap();
                    vxc_mf_s.lapack_dgemm(ao, vxc_ao_s, 'N', 'T', 1.0, 0.0);
                    //// zip(ao, vxc, weight)
                    //ao.iter_columns_full().zip(vxc_ao_s.iter_columns_full())
                    //.map(|(ao_r,vxc_r)| (ao_r,vxc_r))
                    //.zip(grids.weights.iter()).for_each(|((ao_r,vxc_r),w)| {
                    ////izip!(ao.iter_columns_full(),vxc_ao_s.iter_columns_full(),grids.weights.iter())
                    ////    .for_each(|(ao_r,vxc_r,w)| {
                    //    // generate Vxc for a given grid
                    //    iproduct!(ao_r.iter(),vxc_r.iter()).map(|(y,x)| {(y,x)})
                    //    .zip(vxc_mf_s.data.iter_mut()).for_each(|(ao,vxc)| {
                    //        *vxc += ao.0*ao.1*w
                    //    });
                    //});
                }
            }
            let dt2 = utilities::timing(&dt1, Some("From vxc_ao to vxc"));
        }

        println!("debug {:?}", exc_spin);

        let dt0 = utilities::init_timing();
        for i_spin in (0..spin_channel) {
            let mut vxc_s = vxc.get_mut(i_spin).unwrap();
            let mut vxc_mf_s = vxc_mf.get_mut(i_spin).unwrap();

            vxc_mf_s.self_add(&vxc_mf_s.transpose());
            vxc_mf_s.self_multiple(0.5);
            //println!("debug vxc{}",i_spin);
            //vxc_mf_s.formated_output(10, "full");
            *vxc_s = vxc_mf_s.to_matrixupper();
        }

        utilities::timing(&dt0, Some("symmetrize vxc"));

        exc_total = exc_spin.iter().sum();


        if scaling_factor!=1.0f64 {
            exc_total *= scaling_factor;
            for i_spin in (0..spin_channel) {
                vxc[i_spin].data.par_iter_mut().for_each(|f| *f = *f*scaling_factor)
            }
        };

        (exc_total, vxc)

    }

    pub fn generate_vxc_rayon(&self, scaling_factor: f64) -> (f64, Vec<MatrixUpper<f64>>) {
        //In this subroutine, we call the lapack dgemm in a rayon parallel environment.
        //In order to ensure the efficiency, we disable the openmp ability and re-open it in the end of subroutien
        let default_omp_num_threads = utilities::omp_get_num_threads_wrapper();
        //println!("debug: default_omp_num_threads: {}", default_omp_num_threads);
        utilities::omp_set_num_threads_wrapper(1);

        let num_basis = self.mol.num_basis;
        let num_state = self.mol.num_state;
        let num_auxbas = self.mol.num_auxbas;
        let npair = num_basis*(num_basis+1)/2;
        let spin_channel = self.mol.spin_channel;
        //let mut vxc: MatrixUpper<f64> = MatrixUpper::new(1,0.0f64);
        let mut vxc:Vec<MatrixUpper<f64>> = vec![MatrixUpper::empty();spin_channel];
        let mut exc_spin:Vec<f64> = vec![0.0;spin_channel];
        let mut total_elec = [0.0,0.0];
        let mut exc_total:f64 = 0.0;
        let mut vxc_mf:Vec<MatrixFull<f64>> = vec![MatrixFull::new([num_basis,num_basis],0.0);spin_channel];
        let dm = &self.density_matrix;
        let mo = &self.eigenvectors;
        let occ = &self.occupation;
        if let Some(grids) = &self.grids {
            //println!("debug: {:?}",grids.parallel_balancing);
            let (sender, receiver) = channel();
            grids.parallel_balancing.par_iter().for_each_with(sender,|s,range_grids| {
                //println!("debug 0: {}", rayon::current_thread_index().unwrap());
                let (exc,vxc_ao,total_elec) = self.mol.xc_data.xc_exc_vxc_slots(range_grids.clone(), grids, spin_channel,dm, mo, occ);
                //println!("debug loc_exc: {:?}, {:?}, {:?}", exc, vxc_ao[0].size(), vxc_ao[0].data.iter().sum::<f64>());
                //exc_spin = exc;
                let mut vxc_mf: Vec<MatrixFull<f64>> = vec![MatrixFull::new([num_basis,num_basis],0.0f64);spin_channel];;
                //println!("debug 1: {}", rayon::current_thread_index().unwrap());
                if let Some(ao) = &grids.ao {
                    for i_spin in 0..spin_channel {
                        //println!("debug 1-1: {} with spin {}", rayon::current_thread_index().unwrap(), i_spin);

                        let vxc_mf_s = vxc_mf.get_mut(i_spin).unwrap();
                        let vxc_ao_s = vxc_ao.get(i_spin).unwrap();
                        //println!("debug 1-2: {} with spin {}", rayon::current_thread_index().unwrap(), i_spin);
                        rest_tensors::matrix::matrix_blas_lapack::_dgemm(
                            ao,(0..num_basis, range_grids.clone()),'N',
                            vxc_ao_s,(0..num_basis,0..range_grids.len()),'T',
                            vxc_mf_s, (0..num_basis,0..num_basis),
                            1.0,0.0);

                        //println!("debug 1-3: {} with spin {}", rayon::current_thread_index().unwrap(), i_spin);
                        //vxc_mf_s.to_matrixfullslicemut().lapack_dgemm(
                        //    &ao.to_matrixfullslice(), 
                        //    &vxc_ao_s.to_matrixfullslice(),
                        //    'N', 'T', 1.0, 0.0);
                    }
                }
                //println!("debug 2: {}", rayon::current_thread_index().unwrap());
                s.send((vxc_mf,exc,total_elec)).unwrap()
            });
            receiver.into_iter().for_each(|(vxc_mf_local,exc_local,loc_total_elec)| {
                vxc_mf.iter_mut().zip(vxc_mf_local.iter()).for_each(|(to_matr,from_matr)| {
                    to_matr.self_add(from_matr);
                });
                exc_spin.iter_mut().zip(exc_local.iter()).for_each(|(to_exc,from_exc)| {
                    *to_exc += from_exc
                });
                total_elec.iter_mut().zip(loc_total_elec.iter()).for_each(|(to_elec, from_elec)| {
                    *to_elec += from_elec

                })
            })
        }
        //println!("debug total_exc: {:?}, {:?}, {:?}", exc_spin, vxc_mf[0].size(),vxc_mf[0].data.iter().sum::<f64>());

        if spin_channel==1 {
            println!("total electron number: {:16.8}", total_elec[0]);
        } else {
            println!("electron number in alpha-channel: {:12.8}", total_elec[0]);
            println!("electron number in beta-channel:  {:12.8}", total_elec[1]);
        }


        for i_spin in (0..spin_channel) {
            let mut vxc_s = vxc.get_mut(i_spin).unwrap();
            let mut vxc_mf_s = vxc_mf.get_mut(i_spin).unwrap();

            vxc_mf_s.self_add(&vxc_mf_s.transpose());
            vxc_mf_s.self_multiple(0.5);
            *vxc_s = vxc_mf_s.to_matrixupper();
        }

        exc_total = exc_spin.iter().sum();


        if scaling_factor!=1.0f64 {
            exc_total *= scaling_factor;
            for i_spin in (0..spin_channel) {
                vxc[i_spin].data.par_iter_mut().for_each(|f| *f = *f*scaling_factor)
            }
        };

        utilities::omp_set_num_threads_wrapper(default_omp_num_threads);

        (exc_total, vxc)

    }

}




// vj, vk without dependency on SCF struct
//
    pub fn vj_upper_with_ri_v(
                        ri3fn: &Option<RIFull<f64>>,
                        dm: &Vec<MatrixFull<f64>>, 
                        spin_channel: usize, scaling_factor: f64)  -> Vec<MatrixUpper<f64>> {
        
        let mut vj: Vec<MatrixUpper<f64>> = vec![MatrixUpper::new(1,0.0f64),MatrixUpper::new(1,0.0f64)];
        if let Some(ri3fn) = ri3fn {
            let num_basis = ri3fn.size[0];
            let num_auxbas = ri3fn.size[2];
            let npair = num_basis*(num_basis+1)/2;
            for i_spin in (0..spin_channel) {
                //let mut tmp_mu = vec![0.0f64;num_auxbas];
                let mut vj_spin = &mut vj[i_spin];
                *vj_spin = MatrixUpper::new(npair,0.0f64);
                ri3fn.iter_auxbas(0..num_auxbas).unwrap().enumerate().for_each(|(i,m)| {
                    //prepare \sum_{kl}D_{kl}*M_{kl}^{\mu}
                    let tmp_mu =
                        m.chunks_exact(num_basis).zip(dm[i_spin].data.chunks_exact(num_basis))
                            .fold(0.0_f64,|acc, (m,d)| {
                                acc + m.iter().zip(d.iter()).map(|value| value.0*value.1).sum::<f64>()
                            });
                    // filter out the upper part of  M_{ij}^{\mu}
                    let m_ij_upper = m.iter().enumerate().filter(|(i,v)| i%num_basis<=i/num_basis)
                        .map(|(i,v)| v );
    
                    // fill vj[i_spin] with the contribution from the given {\mu}:
                    //
                    // M_{ij}^{\mu}*(\sum_{kl}D_{kl}*M_{kl}^{\mu})
                    //
                    vj_spin.data.iter_mut().zip(m_ij_upper)
                        .for_each(|value| *value.0 += *value.1*tmp_mu); 
                });
            }
        };
    
        if scaling_factor!=1.0f64 {
            for i_spin in (0..spin_channel) {
                vj[i_spin].data.iter_mut().for_each(|f| *f = *f*scaling_factor)
            }
        };
        vj
    }
    pub fn vj_upper_with_ri_v_sync(
                    ri3fn: &Option<RIFull<f64>>,
                    dm: &Vec<MatrixFull<f64>>, 
                    spin_channel: usize, scaling_factor: f64)  -> Vec<MatrixUpper<f64>> {
        let mut vj: Vec<MatrixUpper<f64>> = vec![MatrixUpper::new(1,0.0f64),MatrixUpper::new(1,0.0f64)];
        //// In this subroutine, we call the lapack dgemm in a rayon parallel environment.
        //// In order to ensure the efficiency, we disable the openmp ability and re-open it in the end of subroutien
        //let default_omp_num_threads = unsafe {openblas_get_num_threads()};
        ////println!("debug: default omp_num_threads: {}", default_omp_num_threads);
        //unsafe{openblas_set_num_threads(1)};
        
        //println!("debug rayon local thread number: {}", rayon::current_num_threads());

        if let Some(ri3fn) = ri3fn {
        let num_basis = ri3fn.size[0];
        let num_auxbas = ri3fn.size[2];
        let npair = num_basis*(num_basis+1)/2;
            for i_spin in (0..spin_channel) {
                //let mut tmp_mu = vec![0.0f64;num_auxbas];
                let mut vj_spin = &mut vj[i_spin];
                *vj_spin = MatrixUpper::new(npair,0.0f64);

                let (sender, receiver) = channel();
                ri3fn.par_iter_auxbas(0..num_auxbas).unwrap().enumerate().for_each_with(sender, |s, (i,m)| {
                    //prepare \sum_{kl}D_{kl}*M_{kl}^{\mu} for each \mu -> tmp_mu
                    let tmp_mu =
                        m.chunks_exact(num_basis).zip(dm[i_spin].data.chunks_exact(num_basis))
                            .fold(0.0_f64,|acc, (m,d)| {
                                acc + m.iter().zip(d.iter()).map(|value| value.0*value.1).sum::<f64>()
                            });
                    // filter out the upper part (ij pair) of M_{ij}^{\mu} for each \mu -> m_ij_upper
                    let m_ij_upper = m.iter().enumerate().filter(|(i,v)| i%num_basis<=i/num_basis)
                        .map(|(i,v)| v.clone() ).collect_vec();
                    s.send((m_ij_upper,tmp_mu)).unwrap();
                });
                // fill vj[i_spin] with the contribution from the given {\mu}:
                //
                // M_{ij}^{\mu}*(\sum_{kl}D_{kl}*M_{kl}^{\mu})
                //
                receiver.iter().for_each(|(m_ij_upper, tmp_mu)| {
                    vj_spin.data.iter_mut().zip(m_ij_upper.iter())
                        .for_each(|value| *value.0 += *value.1*tmp_mu); 
                });


                //vj_spin.data.par_iter_mut().zip(m_ij_upper.par_iter())
                //    .for_each(|value| *value.0 += *value.1*tmp_mu); 
            }
        };

        if scaling_factor!=1.0f64 {
            for i_spin in (0..spin_channel) {
                vj[i_spin].data.par_iter_mut().for_each(|f| *f = *f*scaling_factor)
            }
        };

        //// reuse the default omp_num_threads setting
        //unsafe{openblas_set_num_threads(default_omp_num_threads)};

        vj
    }

    pub fn vj_upper_with_rimatr_v_sync(
                    ri3fn: &Option<(MatrixFull<f64>,MatrixFull<usize>,Vec<[usize;2]>)>,
                    dm: &Vec<MatrixFull<f64>>, 
                    spin_channel: usize, scaling_factor: f64)  -> Vec<MatrixUpper<f64>> {
        let mut vj: Vec<MatrixUpper<f64>> = vec![MatrixUpper::new(1,0.0f64),MatrixUpper::new(1,0.0f64)];
        //// In this subroutine, we call the lapack dgemm in a rayon parallel environment.
        //// In order to ensure the efficiency, we disable the openmp ability and re-open it in the end of subroutien
        //let default_omp_num_threads = unsafe {openblas_get_num_threads()};
        ////println!("debug: default omp_num_threads: {}", default_omp_num_threads);
        //unsafe{openblas_set_num_threads(1)};
        
        //println!("debug rayon local thread number: {}", rayon::current_num_threads());

        if let Some((ri3fn,basbas2baspar,baspar2basbas)) = ri3fn {
        let num_basis = basbas2baspar[0];
        let num_baspar = ri3fn.size[0];
        let num_auxbas = ri3fn.size[1];
        let npair = num_basis*(num_basis+1)/2;
            for i_spin in (0..spin_channel) {
                //let mut tmp_mu = vec![0.0f64;num_auxbas];
                let mut vj_spin = &mut vj[i_spin];
                *vj_spin = MatrixUpper::new(npair,0.0f64);

                let (sender, receiver) = channel();
                ri3fn.par_iter_columns_full().enumerate().for_each_with(sender, |s, (i,m)| {
                    //prepare \sum_{kl}D_{kl}*M_{kl}^{\mu} for each \mu -> tmp_mu
                    let riupper = MatrixUpperSlice::from_vec(m);
                    //let full_m = riupper.to_matrixfull().unwrap();
                    //let tmp_mu =
                    //    full_m.data.chunks_exact(num_basis).zip(dm[i_spin].data.chunks_exact(num_basis))
                    //        .fold(0.0_f64,|acc, (m,d)| {
                    //            acc + m.iter().zip(d.iter()).map(|value| value.0*value.1).sum::<f64>()
                    //        });
                    let tmp_mu =
                        m.iter().zip(dm[i_spin].iter_matrixupper().unwrap()).fold(0.0_f64, |acc,(m,d)| {
                            acc + *m * (*d)
                        });
                        //m.chunks_exact(num_basis).zip(dm[i_spin].data.chunks_exact(num_basis))
                        //    .fold(0.0_f64,|acc, (m,d)| {
                        //        acc + m.iter().zip(d.iter()).map(|value| value.0*value.1).sum::<f64>()
                        //    });
                    // filter out the upper part (ij pair) of M_{ij}^{\mu} for each \mu -> m_ij_upper
                    //let m_ij_upper = full_m.data.iter().enumerate().filter(|(i,v)| i%num_basis<=i/num_basis)
                    //    .map(|(i,v)| v.clone() ).collect_vec();
                    let m_ij_upper = m.iter().enumerate().filter(|(i,v)| i%num_basis<=i/num_basis)
                        .map(|(i,v)| v.clone() ).collect_vec();
                    s.send((m_ij_upper,tmp_mu)).unwrap();
                });
                // fill vj[i_spin] with the contribution from the given {\mu}:
                //
                // M_{ij}^{\mu}*(\sum_{kl}D_{kl}*M_{kl}^{\mu})
                //
                receiver.iter().for_each(|(m_ij_upper, tmp_mu)| {
                    vj_spin.data.iter_mut().zip(m_ij_upper.iter())
                        .for_each(|value| *value.0 += *value.1*tmp_mu); 
                });


                //vj_spin.data.par_iter_mut().zip(m_ij_upper.par_iter())
                //    .for_each(|value| *value.0 += *value.1*tmp_mu); 
            }
        };

        if scaling_factor!=1.0f64 {
            for i_spin in (0..spin_channel) {
                vj[i_spin].data.par_iter_mut().for_each(|f| *f = *f*scaling_factor)
            }
        };

        //// reuse the default omp_num_threads setting
        //unsafe{openblas_set_num_threads(default_omp_num_threads)};

        vj
    }

// Just for test, no need to use vj_full because it's always symmetric
    pub fn vj_full_with_ri_v(
                        ri3fn: &Option<RIFull<f64>>,
                        dm: &Vec<MatrixFull<f64>>, 
                        spin_channel: usize, scaling_factor: f64)  -> Vec<MatrixFull<f64>> {
        
        let mut vj: Vec<MatrixFull<f64>> = vec![MatrixFull::new([1,1],0.0f64),MatrixFull::new([1,1],0.0f64)];
        if let Some(ri3fn) = ri3fn {
            let num_basis = ri3fn.size[0];
            let num_auxbas = ri3fn.size[2];
            //let npair = num_basis*(num_basis+1)/2;
            for i_spin in (0..spin_channel) {
                //let mut tmp_mu = vec![0.0f64;num_auxbas];
                let mut vj_spin = &mut vj[i_spin];
                *vj_spin = MatrixFull::new([num_basis, num_basis],0.0f64);
                ri3fn.iter_auxbas(0..num_auxbas).unwrap().enumerate().for_each(|(i,m)| {
                    //prepare \sum_{kl}D_{kl}*M_{kl}^{\mu}
                    let tmp_mu =
                        m.chunks_exact(num_basis).zip(dm[i_spin].data.chunks_exact(num_basis))
                            .fold(0.0_f64,|acc, (m,d)| {
                                acc + m.iter().zip(d.iter()).map(|value| value.0*value.1).sum::<f64>()
                            });
    
                    // fill vj[i_spin] with the contribution from the given {\mu}:
                    //
                    // M_{ij}^{\mu} * (\sum_{kl}D_{kl}*M_{kl}^{\mu})
                    //
                    vj_spin.data.iter_mut().zip(m)
                        .for_each(|value| *value.0 += *value.1*tmp_mu); 
                });
            }
        };
    
        if scaling_factor!=1.0f64 {
            for i_spin in (0..spin_channel) {
                vj[i_spin].data.iter_mut().for_each(|f| *f = *f*scaling_factor)
            }
        };
        vj
    }
    pub fn vk_full_fromdm_with_ri_v(
                        ri3fn: &Option<RIFull<f64>>,
                        dm: &Vec<MatrixFull<f64>>, 
                        spin_channel: usize, scaling_factor: f64)  -> Vec<MatrixFull<f64>> {
        
        let mut vk: Vec<MatrixFull<f64>> = vec![MatrixFull::new([1,1],0.0f64),MatrixFull::new([1,1],0.0f64)];
        if let Some(ri3fn) = ri3fn {
            let num_basis = ri3fn.size[0];
            let num_auxbas = ri3fn.size[2];
            //let npair = num_basis*(num_basis+1)/2;
            for i_spin in (0..spin_channel) {
                //let mut tmp_mu = vec![0.0f64;num_auxbas];
                let mut vk_spin = &mut vk[i_spin];
                *vk_spin = MatrixFull::new([num_basis, num_basis],0.0f64);
                ri3fn.iter_auxbas(0..num_auxbas).unwrap().enumerate().for_each(|(i,m)| {
                    //prepare \sum_{l}D_{jl}*M_{kl}^{\mu}
                    let mut tmp_mu = MatrixFull::from_vec([num_basis, num_basis], m.to_vec()).unwrap();
                    let mut dm_m = MatrixFull::new([num_basis, num_basis], 0.0f64);
                    dm_m.lapack_dgemm(&mut dm[i_spin].clone(), &mut tmp_mu, 'N', 'T', 1.0, 0.0);
    
                    // fill vk[i_spin] with the contribution from the given {\mu}:
                    //
                    // \sum_j M_{ij}^{\mu} * (\sum_{l}D_{jl}*M_{kl}^{\mu})
                    //
                    vk_spin.lapack_dgemm(&mut tmp_mu.clone(), &mut dm_m, 'N', 'N', 1.0, 1.0);
                });
            }
        };
    
        if scaling_factor!=1.0f64 {
            for i_spin in (0..spin_channel) {
                vk[i_spin].data.iter_mut().for_each(|f| *f = *f*scaling_factor)
            }
        };
        vk
    }

    pub fn vk_upper_with_ri_v_sync(
                    ri3fn: &mut Option<RIFull<f64>>,
                    eigv: &[MatrixFull<f64>;2], 
                    homo: [usize;2], occupation: &[Vec<f64>;2],
                    spin_channel: usize, scaling_factor: f64)  -> Vec<MatrixUpper<f64>> {
        // In this subroutine, we call the lapack dgemm in a rayon parallel environment.
        // In order to ensure the efficiency, we disable the openmp ability and re-open it in the end of subroutien
        let default_omp_num_threads = utilities::omp_get_num_threads_wrapper();
        //println!("debug: default omp_num_threads: {}", default_omp_num_threads);
        utilities::omp_set_num_threads_wrapper(1);

        //let mut bm = RIFull::new([num_state,num_basis,num_auxbas], 0.0f64);
        let mut vk: Vec<MatrixUpper<f64>> = vec![MatrixUpper::new(1,0.0f64),MatrixUpper::new(1,0.0f64)];

        if let Some(ri3fn) = ri3fn {
            let num_basis = eigv[0].size[0];
            let num_state = eigv[0].size[1];
            let num_auxbas = ri3fn.size[2];
            let npair = num_basis*(num_basis+1)/2;
            for i_spin in 0..spin_channel {
                let mut vk_s = &mut vk[i_spin];
                *vk_s = MatrixUpper::new(npair,0.0_f64);
                let eigv_s = &eigv[i_spin];
                let nw = homo[i_spin]+1;
                let occ_s = &occupation[i_spin][0..nw];
                //let mut tmp_b = MatrixFull::new([num_basis,num_basis],0.0_f64);
                let (sender, receiver) = channel();
                ri3fn.par_iter_mut_auxbas(0..num_auxbas).unwrap().for_each_with(sender, |s, m| {
                    let mut tmp_mat = MatrixFull::new([num_basis,nw],0.0_f64);
                    tmp_mat.data.iter_mut().zip(eigv_s.iter_submatrix(0..num_basis,0..nw))
                        .for_each(|value| {*value.0 = *value.1});
                    let mut reduced_ri3fn = MatrixFullSlice {
                        size:  &[num_basis,num_basis],
                        indicing: &[1,num_basis],
                        data: m,
                    };
                    let mut tmp_mc = MatrixFull::new([num_basis,nw],0.0_f64);
                    tmp_mc.to_matrixfullslicemut().lapack_dgemm(&reduced_ri3fn, &tmp_mat.to_matrixfullslice(), 'N', 'N', 1.0, 0.0);

                    let mut tmp_mat = tmp_mc.clone();
                    tmp_mat.data.chunks_exact_mut(tmp_mat.size[0]).zip(occ_s.iter()).for_each(|(to_value, from_value)| {
                        to_value.iter_mut().for_each(|to_value| {*to_value = *to_value*from_value});
                    });

                    let mut vk_mu = MatrixFull::new([num_basis,num_basis],0.0_f64);
                    vk_mu.lapack_dgemm(&mut tmp_mat, &mut tmp_mc, 'N', 'T', 1.0, 0.0);

                    // filter out the upper part of vk_mu
                    let mut tmp_mat = MatrixUpper::from_vec(npair, vk_mu.data.iter().enumerate().filter(|(i,v)| i%num_basis<=i/num_basis)
                        .map(|(i,v)| v.clone() ).collect_vec()).unwrap();

                    s.send(tmp_mat).unwrap()
                });
                receiver.into_iter().for_each(|vk_mu_upper| {
                    vk_s.data.par_iter_mut()
                        .zip(vk_mu_upper.data.par_iter()).for_each(|value| {
                        *value.0 += *value.1
                    })
                });
            }
            //// for each spin channel
            //vk.iter_mut().zip(eigv.iter()).for_each(|(vk_s,eigv_s)| {
            //});

        };

        if scaling_factor!=1.0f64 {
            for i_spin in (0..spin_channel) {
                vk[i_spin].data.par_iter_mut().for_each(|f| *f = *f*scaling_factor)
            }
        };

        // reuse the default omp_num_threads setting
        utilities::omp_set_num_threads_wrapper(default_omp_num_threads);

        vk
    }


#[derive(Clone)]
pub struct ScfTraceRecord {
    pub num_iter: usize,
    pub mixer: String,
    // the maximum number of stored residual densities
    pub num_max_records: usize,
    pub mix_param: f64,
    pub start_diis_cycle: usize,
    //pub scf_energy : Vec<f64>,
    //pub density_matrix: Vec<[MatrixFull<f64>;2]>,
    //pub eigenvectors: Vec<[MatrixFull<f64>;2]>,
    //pub eigenvalues: Vec<[Vec<f64>;2]>,
    pub scf_energy : f64,
    pub eigenvectors: [MatrixFull<f64>;2],
    pub eigenvalues: [Vec<f64>;2],
    pub density_matrix: [Vec<MatrixFull<f64>>;2],
    pub target_vector: Vec<[MatrixFull<f64>;2]>,
    pub error_vector: Vec<Vec<f64>>,
}

impl ScfTraceRecord {
    pub fn new(num_max_records: usize, mix_param: f64, mixer: String,start_diis_cycle: usize) -> ScfTraceRecord {
        if num_max_records==0 {
            println!("Error: num_max_records cannot be 0");
        }
        ScfTraceRecord {
            num_iter: 0,
            mixer,
            mix_param,
            num_max_records,
            start_diis_cycle,
            scf_energy : 0.0,
            eigenvectors: [MatrixFull::new([1,1],0.0),
                              MatrixFull::new([1,1],0.0)],
            eigenvalues: [Vec::<f64>::new(),Vec::<f64>::new()],
            density_matrix: [vec![MatrixFull::new([1,1],0.0),
                              MatrixFull::new([1,1],0.0)],
                             vec![MatrixFull::new([1,1],0.0),
                              MatrixFull::new([1,1],0.0)]],
            target_vector: Vec::<[MatrixFull<f64>;2]>::new(),
            error_vector: Vec::<Vec::<f64>>::new(),
        }
    }
    pub fn initialize(scf: &SCF) -> ScfTraceRecord {
        /// 
        /// Initialize the scf records which should be involked after the initial guess 
        /// 
        let mut tmp_records = ScfTraceRecord::new(
            scf.mol.ctrl.num_max_diis, 
            scf.mol.ctrl.mix_param.clone(), 
            scf.mol.ctrl.mixer.clone(),
            scf.mol.ctrl.start_diis_cycle.clone()
        );
        tmp_records.scf_energy=scf.scf_energy;
        tmp_records.eigenvectors=scf.eigenvectors.clone();
        tmp_records.eigenvalues=scf.eigenvalues.clone();
        tmp_records.density_matrix=[scf.density_matrix.clone(),scf.density_matrix.clone()];
        if tmp_records.mixer.eq(&"ddiis") {
            tmp_records.target_vector.push([scf.density_matrix[0].clone(),scf.density_matrix[1].clone()]);
        }
        tmp_records
    }
    /// This subroutine updates:  
    ///     scf_energy:         from previous to the current value  
    ///     scf_eigenvalues:    from previous to the current value  
    ///     scf_eigenvectors:   from previous to the current value  
    ///     scf_density_matrix: [pre, cur]  
    ///     num_iter  
    /// This subroutine should be called after [`scf.check_scf_convergence`] and before [`self.prepare_next_input`]"
    pub fn update(&mut self, scf: &SCF) {

        let spin_channel = scf.mol.spin_channel;

        // now store the scf energy, eigenvectors and eigenvalues of the last two steps
        //let tmp_data =  self.scf_energy[1].clone();
        self.scf_energy=scf.scf_energy;
        //let tmp_data =  self.eigenvectors[1].clone();
        self.eigenvectors=scf.eigenvectors.clone();
        //let tmp_data =  self.eigenvalues[1].clone();
        self.eigenvalues=scf.eigenvalues.clone();
        let tmp_data =  self.density_matrix[1].clone();
        self.density_matrix=[tmp_data,scf.density_matrix.clone()];

        self.num_iter +=1;
    }
    /// This subroutine prepares the fock matrix for the next step according different mixing algorithm  
    ///
    /// self.mixer =  
    /// * "direct": the output density matrix in the current step `n0[out]` will be used directly 
    ///             to generate the the input fock matrix of the next step
    /// * "linear": the density matrix used in the next step `n1[in]` is a mix between
    ///             the input density matrix in the current step `n0[in]` and `n0[out]`  
    ///             <span style="text-align:right">`n1[in] = alpha*n0[out] + (1-alpha)*n0[in]`</span>  
    ///             <span style="text-align:right">`       = n0[in] + alpha * Rn0            ` </span>  
    ///             where alpha the mixing parameter obtained from self.mix_param
    ///              and `Rn0 = n0[out]-n0[in]` is the density matrix change in the current step.
    ///             `n1[in]` is then be used to generate the input fock matrix of the next step
    /// * "diis":   the input fock matrix of the next step `f1[in] = sum_{i} c_i*f_i[in]`,
    ///            where `f_i[in]` is the input fock matrix of the ith step and 
    ///            c_i is obtained by the diis altogirhm against the error vector
    ///            of the commutator `(f_i[out]*d_i[out]*s-s*d_i[out]*f_i[out])`, where  
    ///            - `f_i[out]` is the ith output fock matrix,   
    ///            - `d_i[out]` is the ith output density matrix,  
    ///            - `s` is the overlap matrix  
    /// * **Ref**: P. Pulay, Improved SCF Convergence Acceleration, JCC, 1982, 3:556-560.
    ///
    pub fn prepare_next_input(&mut self, scf: &mut SCF) {
        let spin_channel = scf.mol.spin_channel;
        let start_pulay = self.start_diis_cycle;
        //if self.residual_density.len()>=2 {
        let alpha = self.mix_param;
        let beta = 1.0-self.mix_param;
        if self.mixer.eq(&"direct") {
            scf.generate_hf_hamiltonian();
        }
        else if self.mixer.eq(&"linear") 
            || (self.mixer.eq(&"ddiis") && self.num_iter<start_pulay) 
            || (self.mixer.eq(&"diis") && self.num_iter<start_pulay) 
        {
            // n1[in] = a*n0[out] + (1-a)*n0[in] = n0[out]-(1-a)*Rn0 = n0[in] + a*Rn0
            // Rn0 = n0[out]-n0[in]; the residual density in the current iteration
            // n1[in] is the input density for the next iteration
            for i_spin in (0..spin_channel) {
                let residual_dm = self.density_matrix[1][i_spin].sub(&self.density_matrix[0][i_spin]).unwrap();
                scf.density_matrix[i_spin] = self.density_matrix[0][i_spin]
                    .scaled_add(&residual_dm, alpha)
                    .unwrap();
            }
            scf.generate_hf_hamiltonian();
        } else if self.mixer.eq(&"diis") && self.num_iter>=start_pulay {
            // 
            // Reference: P. Pulay, Improved SCF Convergence Acceleration, JCC, 1982, 3:556-560.
            // 
            //let num_diis_vec = if self.num_iter >= self.num_max_records + self.start_diis_cycle - 1 {
            //    self.num_max_records
            //} else {
            //    self.num_iter - self.start_diis_cycle + 1
            //};
            //let start_dim = if (self.target_vector.len()>=num_diis_vec) {
            //    self.target_vector.len()-num_diis_vec
            //} else {
            //    0
            //};
            let start_dim = 0usize;
            //
            // prepare the fock matrix according to the output density matrix of the previous step
            //
            let dt1 = time::Local::now();
            scf.generate_hf_hamiltonian();
            let dt2 = time::Local::now();

            // check if the storage of fock matrix reaches the maximum setting
            if self.target_vector.len() == self.num_max_records {
                self.target_vector.remove(0);
                self.error_vector.remove(0);
            };

            //
            // prepare and store the fock matrix in full formate and the error vector in the current step
            //
            let (cur_error_vec, cur_target) = generate_diis_error_vector(&scf.hamiltonian, &scf.ovlp, &mut self.density_matrix, spin_channel);
            self.error_vector.push(cur_error_vec);
            self.target_vector.push(cur_target);


            // solve the diss against the error vector
            let coeff = diis_solver(&self.error_vector, &self.error_vector.len());
            //println!("debug: diis_c: {:?}", &coeff);

            // now extrapolate the fock matrix for the next step
            (0..spin_channel).into_iter().for_each(|i_spin| {
                let mut next_hamiltonian = MatrixFull::new(self.target_vector[0][i_spin].size.clone(),0.0);
                coeff.iter().enumerate().for_each(|(i,value)| {
                    next_hamiltonian.self_scaled_add(&self.target_vector[i+start_dim][i_spin], *value);
                });
                scf.hamiltonian[i_spin] = next_hamiltonian.to_matrixupper();
            });
            let dt3 = time::Local::now();
            let timecost1 = (dt2.timestamp_millis()-dt1.timestamp_millis()) as f64 /1000.0;
            let timecost2 = (dt3.timestamp_millis()-dt2.timestamp_millis()) as f64 /1000.0;
            if scf.mol.ctrl.print_level>2 {
                println!("Hamiltonian: generation by {:10.2}s and DIIS extrapolation by {:10.2}s", timecost1,timecost2);
            }
            
        };
        //}
        //now store the target_vector for diis and its variants.
        //if self.mixer.eq(&"ddiis") {
        //}
    }
}

pub fn generate_diis_error_vector(hamiltonian: &[MatrixUpper<f64>;2], 
                                ovlp: &MatrixUpper<f64>, 
                                density_matrix: &mut [Vec<MatrixFull<f64>>;2],
                                spin_channel: usize) -> (Vec<f64>, [MatrixFull<f64>;2]) {
            let mut cur_error = [
                MatrixFull::new([1,1],0.0),
                MatrixFull::new([1,1],0.0)
            ];
            let mut cur_target = [hamiltonian[0].to_matrixfull().unwrap(),
                 hamiltonian[1].to_matrixfull().unwrap()];

            let mut full_ovlp = ovlp.to_matrixfull().unwrap();

            //let mut sqrt_inv_ovlp = full_ovlp.lapack_power(-0.5,10.0E-6).unwrap();

            //let mut tmp_num = 0_usize;
            //sqrt_inv_ovlp.data.iter().enumerate().for_each(|ig| {
            //    let j = ig.0/sqrt_inv_ovlp.indicing[1];
            //    let i = ig.0%sqrt_inv_ovlp.indicing[1];
            //    if ig.1.abs()>0.1 {
            //        println!("I= {:3} J= {:3} Int= {:16.8}",i, j, ig.1);
            //        tmp_num += 1;
            //    }
            //});
            //println!("Print out {} elements of sqrt_inv_ovlp", tmp_num);

            // now generte the error as the commutator of [fds-sdf]
            (0..spin_channel).into_iter().for_each(|i_spin| {
                cur_error[i_spin] = cur_target[i_spin]
                    .ddot(&mut density_matrix[1][i_spin]).unwrap();
                cur_error[i_spin] = cur_error[i_spin].ddot(&mut full_ovlp).unwrap();
                let mut dsf = full_ovlp.ddot(&mut density_matrix[1][i_spin]).unwrap();
                dsf = dsf.ddot(&mut cur_target[i_spin]).unwrap();
                cur_error[i_spin].self_sub(&dsf);

                // //transfer to an orthorgonal basis
                // let mut tmp_mat = cur_error[i_spin].ddot(&mut sqrt_inv_ovlp).unwrap();
                // cur_error[i_spin].lapack_dgemm(
                //   &mut sqrt_inv_ovlp, &mut tmp_mat,
                //   'T', 'N',
                //   1.0,0.0);
            });

            let mut norm = 0.0;
            (0..spin_channel).for_each(|i_spin| {
                let dd = cur_error[i_spin].data.par_iter().fold(|| 0.0, |acc, x| {
                    acc + x*x
                }).sum::<f64>();
                //println!("diis-norm for {}-spin: {:16.8}",i_spin,dd.sqrt());
                norm += dd
            });
            println!("diis-norm : {:16.8}",norm.sqrt());

            ([cur_error[0].data.clone(),cur_error[1].data.clone()].concat(),
            cur_target)

}

pub fn diis_solver(em: &Vec<Vec<f64>>,
                   num_vec:&usize) -> Vec<f64> {

    //if ! tm.len()==em.len() {
    //    println!("ERROR: the length of target vector is not the same as the corresponding error vector");
    //}
    //let dim = dm.len();

    let dim_vec = em.len();
    let start_dim = if (em.len()>=*num_vec) {em.len()-*num_vec} else {0};
    let dim = if (em.len()>=*num_vec) {*num_vec} else {em.len()};
    let mut coeff = Vec::<f64>::new();
    //let mut norm_rdm = [Vec::<f64>::new(),Vec::<f64>::new()];
    let mut odm = MatrixFull::new([1,1],0.0);
    let mut inv_opta = MatrixFull::new([dim,dim],0.0);
    //let mut sum_inv_norm_rdm = [0.0,0.0];
    let mut sum_inv_norm_rdm = 0.0_f64;

    // now prepare the norm matrix of the residual density matrix
    let mut opta = MatrixFull::new([dim,dim],0.0);
    (start_dim..dim_vec).into_iter().for_each(|i| {
        (start_dim..dim_vec).into_iter().for_each(|j| {
            let mut inv_norm_rdm = em[i].iter()
                .zip(em[j].iter())
                .fold(0.0,|c,(d,e)| {c + d*e});
            opta.set2d([i-start_dim,j-start_dim],inv_norm_rdm);
            //sum_inv_norm_rdm += inv_norm_rdm;
        })
    });
    inv_opta = opta.lapack_inverse().unwrap();
    sum_inv_norm_rdm = inv_opta.data.iter().sum::<f64>().powf(-1.0f64);
    //(0..*spin_channel).into_iter().for_each(|i_spin| {
    //    let mut opta = MatrixFull::new([dim,dim],0.0);
    //    (start_dim..dim_vec).into_iter().for_each(|i| {
    //        (start_dim..dim_vec).into_iter().for_each(|j| {
    //            let inv_norm_rdm = em[i][i_spin].data.iter()
    //                .zip(em[j][i_spin].data.iter())
    //                .fold(0.0,|c,(d,e)| {c + d*e});
    //            opta.set2d([i-start_dim,j-start_dim],inv_norm_rdm);
    //            //sum_inv_norm_rdm += inv_norm_rdm;
    //        })
    //    });
    //    inv_opta[i_spin] = opta.lapack_inverse().unwrap();
    //    sum_inv_norm_rdm[i_spin] = inv_opta[i_spin].data.iter().sum::<f64>().powf(-1.0f64);
    //});

    // now prepare the coefficients for the pulay mixing
    coeff = vec![sum_inv_norm_rdm;dim];
    //coeff.iter().for_each(|i| {println!("coeff: {}",i)});
    (0..dim).zip(coeff.iter_mut()).for_each(|(i,value)| {
        //println!("{:?}",*value);
        *value *= inv_opta.get2d_slice([0,i], dim)
                 .unwrap()
                 .iter()
                 .sum::<f64>();
    });
    //(0..*spin_channel).into_iter().for_each(|i_spin| {
    //    coeff[i_spin] = vec![sum_inv_norm_rdm[i_spin];dim];
    //    coeff[i_spin].iter().for_each(|i| {println!("coeff: {}",i)});
    //    (0..dim).zip(coeff[i_spin].iter_mut()).for_each(|(i,value)| {
    //        //println!("{:?}",*value);
    //        *value *= inv_opta[i_spin].get2d_slice([0,i], dim)
    //                 .unwrap()
    //                 .iter()
    //                 .sum::<f64>();
    //    });
    //});

    coeff

}

pub fn scf(mol:Molecule) -> anyhow::Result<SCF> {
    let dt0 = time::Local::now();

    let mut scf_data = SCF::build(mol);
    // now generate the hamiltonian and the total energy according the initial guess density matrix
    scf_data.generate_hf_hamiltonian();

    let mut scf_records=ScfTraceRecord::initialize(&scf_data);

    println!("The total energy: {:20.10} Ha by the initial guess",scf_data.scf_energy);
    //let mut scf_continue = true;
    if scf_data.mol.ctrl.noiter {
        println!("Warning: the SCF iteration is skipped!");
        return Ok(scf_data);
    }

    // now prepare the input density matrix for the first iteration and initialize the records
    scf_data.diagonalize_hamiltonian();
    scf_data.generate_density_matrix();
    scf_records.update(&scf_data);

    let mut scf_converge = [false;2];
    while ! (scf_converge[0] || scf_converge[1]) {
        let dt1 = time::Local::now();

        scf_records.prepare_next_input(&mut scf_data);

        let dt1_1 = time::Local::now();

        scf_data.diagonalize_hamiltonian();
        let dt1_2 = time::Local::now();
        scf_data.generate_density_matrix();
        let dt1_3 = time::Local::now();


        scf_converge = scf_data.check_scf_convergence(&scf_records);
        let dt1_4 = time::Local::now();

        scf_records.update(&scf_data);
        let dt1_5 = time::Local::now();


        let dt2 = time::Local::now();
        let timecost = (dt2.timestamp_millis()-dt1.timestamp_millis()) as f64 /1000.0;
        println!("Energy: {:18.10} Ha after {:4} iterations (in {:10.2} seconds).",
                 scf_records.scf_energy,
                 scf_records.num_iter-1,
                 timecost);
        if scf_data.mol.ctrl.print_level>1 {
            println!("Detailed timing info in this SCF step:");
            let timecost = (dt1_1.timestamp_millis()-dt1.timestamp_millis()) as f64 /1000.0;
            println!("prepare_next_input:      {:10.2}s", timecost);
            let timecost = (dt1_2.timestamp_millis()-dt1_1.timestamp_millis()) as f64 /1000.0;
            println!("diagonalize_hamiltonian: {:10.2}s", timecost);
            let timecost = (dt1_3.timestamp_millis()-dt1_2.timestamp_millis()) as f64 /1000.0;
            println!("generate_density_matrix: {:10.2}s", timecost);
            let timecost = (dt1_4.timestamp_millis()-dt1_3.timestamp_millis()) as f64 /1000.0;
            println!("check_scf_convergence:   {:10.2}s", timecost);
            let timecost = (dt1_5.timestamp_millis()-dt1_4.timestamp_millis()) as f64 /1000.0;
            println!("scf_records.update:      {:10.2}s", timecost);
        }
    }
    if scf_converge[0] {
        println!("SCF is converged after {:4} iterations.", scf_records.num_iter-1);
        if scf_data.mol.ctrl.print_level>1 {
            scf_data.formated_eigenvalues((scf_data.homo.iter().max().unwrap()+4).min(scf_data.mol.num_state));
        }
        if scf_data.mol.ctrl.print_level>3 {
            scf_data.formated_eigenvectors();
        }
        // not yet implemented. Just an empty subroutine
        scf_data.save_chkfile();
    } else {
        println!("SCF does not converge within {:03} iterations",scf_records.num_iter);
    }
    let dt2 = time::Local::now();
    println!("the job spends {:16.2} seconds",(dt2.timestamp_millis()-dt0.timestamp_millis()) as f64 /1000.0);

    Ok(scf_data)
}


#[test]
fn test_max() {
    println!("{}, {}, {}",1,2,std::cmp::max(1, 2));
}