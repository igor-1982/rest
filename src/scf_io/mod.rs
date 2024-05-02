use crate::check_norm::{self, generate_occupation_frac_occ, generate_occupation_integer, generate_occupation_sad, OCCType};
//use clap::value_parser;
//use pyo3::{pyclass, pymethods, pyfunction};
//use tensors::matrix_blas_lapack::{_dgemm, _dinverse, _dsymm};
//use tensors::{ERIFull,MatrixFull, ERIFold4, MatrixUpper, TensorSliceMut, RIFull, MatrixFullSlice, MatrixFullSliceMut, BasicMatrix, MathMatrix, MatrixUpperSlice, ParMathMatrix, ri};
//use itertools::{Itertools, iproduct, izip};
//use libc::SCHED_OTHER;
//use core::num;
//use std::fmt::format;
//use std::io::Write;
//use std::{fs, vec};
//use std::thread::panicking;
//use std::{vec, fs};
//use rest_libcint::{CINTR2CDATA, CintType};
//use crate::geom_io::{GeomCell,MOrC, GeomUnit};
//use crate::basis_io::{Basis4Elem,BasInfo};
//use crate::isdf::{prepare_for_ri_isdf, init_by_rho, prepare_m_isdf};
////use crate::initial_guess::sad::sad_dm;
//use crate::molecule_io::{Molecule, generate_ri3fn_from_rimatr};
//use crate::tensors::{TensorOpt,TensorOptMut,TensorSlice};
//use crate::dft::{Grids, numerical_density, par_numerical_density};
//use crate::{utilities, initial_guess};
//use crate::initial_guess::initial_guess;
//use rayon::prelude::*;
//use hdf5;
////use blas::{ddot,daxpy};
//use std::sync::{Mutex, Arc,mpsc};
//use std::thread;
//use crossbeam::{channel::{unbounded,bounded},thread::{Scope,scope}};
////use std::sync::mpsc::{channel, Receiver};
//use std::sync::mpsc::channel;
use crate::dft::gen_grids::prune::prune_by_rho;
use crate::geom_io::calc_nuc_energy_with_ecp;
use crate::utilities::TimeRecords;
////use blas_src::openblas::dgemm;
mod addons;
mod fchk;
mod pyrest_scf_io;

//use crate::basis_io::ecp::test_ecp;

//use libc::{abs, _SC_AIO_LISTIO_MAX};
//use clap::value_parser;
use pyo3::{pyclass, pymethods, pyfunction};
use tensors::matrix_blas_lapack::{_dgemm, _dgemm_full, _dgemv, _dinverse, _dspgvx, _dsymm, _dsyrk, _power, _power_rayon};
use tensors::{ERIFull,MatrixFull, ERIFold4, MatrixUpper, TensorSliceMut, RIFull, MatrixFullSlice, MatrixFullSliceMut, BasicMatrix, MathMatrix, MatrixUpperSlice, ParMathMatrix, ri};
use itertools::{Itertools, iproduct, izip};
use rayon::prelude::*;
use std::collections::HashMap;
use std::mem::size_of;
use std::sync::{Mutex, Arc,mpsc};
use std::thread;
use crossbeam::{channel::{unbounded,bounded},thread::{Scope,scope}};
use std::sync::mpsc::{channel, Receiver};
//use std::arch;
//use ndarray;
//use hdf5;
//use libc::SCHED_OTHER;
//use core::num;
//use std::fmt::format;
//use std::io::Write;
//use std::path::Path;
//use std::{fs, vec};
//use std::thread::panicking;
//use std::{vec, fs};
//use rest_libcint::{CINTR2CDATA, CintType};
//use crate::geom_io::{GeomCell,MOrC, GeomUnit};
//use crate::basis_io::{Basis4Elem,BasInfo};
//use crate::post_scf_analysis::save_chkfile;
//use crate::dft::{Grids, numerical_density, par_numerical_density};
use crate::isdf::{prepare_for_ri_isdf, init_by_rho, prepare_m_isdf};
use crate::molecule_io::{Molecule, generate_ri3fn_from_rimatr};
use crate::tensors::{TensorOpt,TensorOptMut,TensorSlice};
use crate::dft::Grids;
use crate::{utilities, initial_guess};
use crate::initial_guess::initial_guess;
use crate::constants::{SPECIES_INFO, INVERSE_THRESHOLD};




#[pyclass]
#[derive(Clone)]
pub struct SCF {
    #[pyo3(get,set)]
    pub mol: Molecule,
    pub ovlp: MatrixUpper<f64>,
    pub h_core: MatrixUpper<f64>,
    //pub ijkl: Option<Tensors<f64>>,
    //pub ijkl: Option<ERIFull<f64>>,
    pub ijkl: Option<ERIFold4<f64>>,
    pub ri3fn: Option<RIFull<f64>>,
    pub ri3fn_isdf: Option<RIFull<f64>>,
    pub tab_ao: Option<MatrixFull<f64>>,
    pub m: Option<MatrixFull<f64>>,
    pub rimatr: Option<(MatrixFull<f64>,MatrixFull<usize>,Vec<[usize;2]>)>,
    pub ri3mo: Option<Vec<(RIFull<f64>,std::ops::Range<usize> , std::ops::Range<usize>)>>,
    #[pyo3(get,set)]
    pub eigenvalues: [Vec<f64>;2],
    //pub eigenvectors: Vec<Tensors<f64>>,
    pub eigenvectors: [MatrixFull<f64>;2],
    //pub density_matrix: Vec<Tensors<f64>>,
    //pub density_matrix: [MatrixFull<f64>;2],
    pub density_matrix: Vec<MatrixFull<f64>>,
    //pub hamiltonian: Vec<Tensors<f64>>,
    pub hamiltonian: [MatrixUpper<f64>;2],
    pub scftype: SCFType,
    #[pyo3(get,set)]
    pub occupation: [Vec<f64>;2],
    #[pyo3(get,set)]
    pub homo: [usize;2],
    #[pyo3(get,set)]
    pub lumo: [usize;2],
    #[pyo3(get,set)]
    pub nuc_energy: f64,
    #[pyo3(get,set)]
    pub scf_energy: f64,
    pub grids: Option<Grids>,
    pub energies: HashMap<String,Vec<f64>>,
}

#[derive(Clone,Copy)]
pub enum SCFType {
    RHF,
    ROHF,
    UHF
}


impl SCF {
    pub fn init_scf(mol: &Molecule) -> SCF {
        let mut scf_data = SCF {
            mol: mol.clone(),
            ovlp: MatrixUpper::new(1,0.0),
            h_core: MatrixUpper::new(1,0.0),
            ijkl: None,
            ri3fn: None,
            ri3fn_isdf: None,
            tab_ao: None,
            m: None,
            rimatr: None,
            ri3mo: None,
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
            energies: HashMap::new(),
        };

        // at first check the scf type: RHF, ROHF or UHF
        scf_data.scftype = if mol.num_elec[1]==mol.num_elec[2] && ! mol.ctrl.spin_polarization {
            SCFType::RHF
        } else if mol.num_elec[1]!=mol.num_elec[2] && ! mol.ctrl.spin_polarization {
            SCFType::ROHF
        } else {      
            SCFType::UHF
        };
        match &scf_data.scftype {
            SCFType::RHF => {
                if mol.ctrl.print_level>0 {println!("Restricted Hartree-Fock (or Kohn-Sham) algorithm is invoked.")}},
            SCFType::ROHF => {
                if mol.ctrl.print_level>0 {println!("Restricted-orbital Hartree-Fock (or Kohn-Sham) algorithm is invoked.")};
                scf_data.mol.ctrl.spin_channel=2;
                scf_data.mol.spin_channel=2;
                if mol.ctrl.print_level>0 {
                    panic!("Restricted-orbital Hartree-Fock (or Kohn-Sham) algorithm is invoked, which, however, is not yet stable.")
                };
            },
            SCFType::UHF => {
                if mol.ctrl.print_level>0 {println!("Unrestricted Hartree-Fock (or Kohn-Sham) algorithm is invoked.")}
            },
        };

        scf_data
    }


    pub fn prepare_necessary_integrals(&mut self) {
        // prepare standard two, three, and four-center integrals.
        // for ISDF integrals, they needs density grids, and thus should be prepared after the grid initialization

        let print_level = self.mol.ctrl.print_level;

        self.nuc_energy = calc_nuc_energy_with_ecp(&self.mol.geom, &self.mol.basis4elem);
        if print_level>0 {println!("Nuc_energy: {}",self.nuc_energy)};

        self.ovlp = self.mol.int_ij_matrixupper(String::from("ovlp"));
        self.h_core = self.mol.int_ij_matrixupper(String::from("hcore"));
        self.ijkl = if self.mol.ctrl.use_auxbas {
            None
        } else {
            Some(self.mol.int_ijkl_erifold4())
        };

        if self.mol.ctrl.print_level>3 {
            println!("The S matrix:");
            self.ovlp.formated_output(5, "lower");
            let mut kin = self.mol.int_ij_matrixupper(String::from("kinetic"));
            println!("The Kinetic matrix:");
            kin.formated_output(5, "lower");
            println!("The H-core matrix:");
            self.h_core.formated_output(5, "lower");
        }

        if self.mol.ctrl.print_level>4 {
            //(ij|kl)
            if let Some(tmp_eris) = &self.ijkl {
                println!("The four-center ERIs:");
                let mut tmp_num = 0;
                let (i_len,j_len) =  (self.mol.num_basis,self.mol.num_basis);
                let (k_len,l_len) =  (self.mol.num_basis,self.mol.num_basis);
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

        let isdf = self.mol.ctrl.eri_type.eq("ri_v") && self.mol.ctrl.use_isdf;
        let ri3fn_full = self.mol.ctrl.use_auxbas && !self.mol.ctrl.use_ri_symm;
        let ri3fn_symm = self.mol.ctrl.use_auxbas && self.mol.ctrl.use_ri_symm;

        // preparing the three-center integrals in the full format
        self.ri3fn = if ri3fn_full && !isdf {
            Some(self.mol.prepare_ri3fn_for_ri_v_full_rayon())
        }else if self.mol.ctrl.isdf_k_only{ 
            Some(self.mol.prepare_ri3fn_for_ri_v_full_rayon())
        }else {
            None
        };

        // preparing the three-center integrals using the symmetry
        self.rimatr = if ri3fn_symm  && ! isdf {
            let (rimatr, basbas2baspar, baspar2basbas) = self.mol.prepare_rimatr_for_ri_v_rayon();
            Some((rimatr, basbas2baspar, baspar2basbas))
        } else if ri3fn_symm  && isdf {
            None
        } else {
            None
        };


        // initial eigenvectors and eigenvalues
        let (eigenvectors, eigenvalues,n_found)=self.ovlp.to_matrixupperslicemut().lapack_dspevx().unwrap();

        if (n_found as usize) < self.mol.fdqc_bas.len() {
            println!("Overlap matrix is singular:");
            println!("  Using {} out of a possible {} specified basis functions",n_found, self.mol.fdqc_bas.len());
            println!("  Lowest remaining eigenvalue: {:16.8}",eigenvalues[0]);
            self.mol.num_state = n_found as usize;
        } else {
            if self.mol.ctrl.print_level>0 {
                println!("Overlap matrix is nonsigular:");
                println!("  Lowest eigenvalue: {:16.8} with the total number of basis functions: {:6}",eigenvalues[0],self.mol.num_state);
            }
        };

    }

    pub fn prepare_density_grids(&mut self) {

        self.grids = if self.mol.xc_data.is_dfa_scf() || self.mol.ctrl.use_isdf || self.mol.ctrl.initial_guess == "vsap" {
            let grids = Grids::build(&self.mol);
            if self.mol.ctrl.print_level>0 {
                println!("Grid size: {:}", grids.coordinates.len());
            }
            Some(grids)
        } else {None};

        if let Some(grids) = &mut self.grids {
            grids.prepare_tabulated_ao(&self.mol);
        }

    }

    pub fn prepare_isdf(&mut self) {
        if let Some(grids) = &self.grids {
            if self.mol.ctrl.use_isdf {
                let init_fock = self.h_core.clone();
                if self.mol.spin_channel==1 {
                    self.hamiltonian = [init_fock,MatrixUpper::new(1,0.0)];
                } else {
                    let init_fock_beta = init_fock.clone();
                    self.hamiltonian = [init_fock,init_fock_beta];
                };
                (self.eigenvectors,self.eigenvalues) = diagonalize_hamiltonian_outside(&self);
                (self.occupation, self.homo, self.lumo) = generate_occupation_outside(&self);
                self.density_matrix = generate_density_matrix_outside(&self);

                self.grids = Some(prune_by_rho(grids, &self.density_matrix, self.mol.spin_channel));
                
            };

            let isdf = self.mol.ctrl.eri_type.eq("ri_v") && self.mol.ctrl.use_isdf;
            let ri3fn_full = self.mol.ctrl.use_auxbas && !self.mol.ctrl.use_ri_symm;
            let ri3fn_symm = self.mol.ctrl.use_auxbas && self.mol.ctrl.use_ri_symm;

            self.ri3fn_isdf = if ri3fn_full && isdf && !self.mol.ctrl.isdf_new{
                if let Some(grids) = &self.grids {
                    Some(prepare_for_ri_isdf(self.mol.ctrl.isdf_k_mu, &self.mol, &grids))
                } else {
                    None
                }
            } else {
                None
            };

            (self.tab_ao, self.m) = if isdf && self.mol.ctrl.isdf_new{
                if let Some(grids) = &self.grids {
                    let isdf = prepare_m_isdf(self.mol.ctrl.isdf_k_mu, &self.mol, &grids);
                    (Some(isdf.0), Some(isdf.1))
                } else {
                    (None,None)
                }
            } else {
                (None,None)
            };
        } else {
            panic!("SCF.grids should be initialized before the preparation of ISDF");
        }

    }



    pub fn build(mol: Molecule) -> SCF {

        let mut new_scf = SCF::init_scf(&mol);
        //new_scf.generate_occupation();

        initialize_scf(&mut new_scf);

        new_scf

    }


    pub fn generate_occupation(&mut self) {
        (self.occupation, self.homo, self.lumo) = generate_occupation_outside(&self);
    }

    pub fn generate_density_matrix(&mut self) {
        self.density_matrix = generate_density_matrix_outside(&self);
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

        vj
    }

    pub fn generate_vj_on_the_fly(&mut self) -> Vec<MatrixUpper<f64>>{
        let num_shell = self.mol.cint_bas.len();
        //let num_shell = self.mol.cint_fdqc.len();
        let num_basis = self.mol.num_basis;
        let spin_channel = self.mol.spin_channel;
        let dm = &self.density_matrix;
        let mut vj: Vec<MatrixUpper<f64>> = vec![];
        let mol = &self.mol;
        for i_spin in 0..spin_channel{
            let mut vj_i = MatrixFull::new([num_basis, num_basis], 0.0);
            let mut dm_s = &self.density_matrix[i_spin];
            for k in 0..num_shell{
                let bas_start_k = mol.cint_fdqc[k][0];
                let bas_len_k = mol.cint_fdqc[k][1];

                for l in 0..num_shell{
                    let bas_start_l = mol.cint_fdqc[l][0];
                    let bas_len_l = mol.cint_fdqc[l][1];
                    let mut klij = &mol.int_ijkl_given_kl(k, l);
                    
                    // ao_k & ao_l are index of ao
                    let mut sum =0.0;
                    for ao_k in bas_start_k..bas_start_k+bas_len_k{
                        for ao_l in bas_start_l..bas_start_l+bas_len_l{
                            let mut sum =0.0;
                            let mut index_k = ao_k-bas_start_k;
                            let mut index_l = ao_l-bas_start_l;
                            let mut eri_cd = &klij[index_l * bas_len_k + index_k];
                            let eri_full = eri_cd.to_matrixupper().to_matrixfull().unwrap();
                            let mut v_cd = MatrixFull::new([num_basis, num_basis], 0.0);
                            v_cd.data.iter_mut().zip(dm_s.data.iter()).zip(eri_full.data.iter()).for_each(|((v,p),eri)|{
                                *v = *p * *eri
                            });

                            v_cd.data.iter().for_each(|x|{
                                sum += *x
                            });
                            vj_i[(ao_k,ao_l)] = sum;
                        }
                    }
                }
            }

            vj.push(vj_i.to_matrixupper());
        }
        if spin_channel == 1{
            vj.push(MatrixUpper::new(1, 0.0));       
        }
        vj
    }

    pub fn generate_vj_on_the_fly_par_old(&mut self) -> Vec<MatrixUpper<f64>>{
        //utilities::omp_set_num_threads_wrapper(1);
        let num_shell = self.mol.cint_bas.len();
        let num_basis = self.mol.num_basis;
        let spin_channel = self.mol.spin_channel;
        let dm = &self.density_matrix;
        let mut vj: Vec<MatrixUpper<f64>> = vec![];
        let mol = &self.mol;
        for i_spin in 0..spin_channel{
            let mut vj_i = MatrixFull::new([num_basis, num_basis], 0.0);
            let mut dm_s = &self.density_matrix[i_spin];
            let par_tasks = utilities::balancing(num_shell*num_shell, rayon::current_num_threads());
            let (sender, receiver) = channel();
            let mut index = vec![0usize; num_shell*num_shell];
            for i in 0..num_shell*num_shell{index[i] = i};
            index.par_iter().for_each_with(sender,|s,i|{
                let k = i/num_shell;
                let bas_start_k = mol.cint_fdqc[k][0];
                let bas_len_k = mol.cint_fdqc[k][1];
                let l = i%num_shell;
                let bas_start_l = mol.cint_fdqc[l][0];
                let bas_len_l = mol.cint_fdqc[l][1];
                let mut klij = &mol.int_ijkl_given_kl(k, l);
                let mut sum =0.0;
                //let mut out = vec![(0.0, 0usize, 0usize); ];
                let mut out:Vec<(f64, usize, usize)> = Vec::new();
                    for ao_k in bas_start_k..bas_start_k+bas_len_k{
                        for ao_l in bas_start_l..bas_start_l+bas_len_l{
                            let mut sum =0.0;
                            let mut index_k = ao_k-bas_start_k;
                            let mut index_l = ao_l-bas_start_l;
                            let mut eri_cd = &klij[index_l * bas_len_k + index_k];
                            let eri_full = eri_cd.to_matrixupper().to_matrixfull().unwrap();
                            let mut v_cd = MatrixFull::new([num_basis, num_basis], 0.0);
                            v_cd.data.iter_mut().zip(dm_s.data.iter()).zip(eri_full.data.iter()).for_each(|((v,p),eri)|{
                                *v = *p * *eri
                            });

                            v_cd.data.iter().for_each(|x|{
                                sum += *x
                            });
                            out.push((sum, ao_k, ao_l));
                        }
                    }
                s.send(out).unwrap();
            });
            receiver.into_iter().for_each(|out_vec| {
                out_vec.iter().for_each(|(value,index_k,index_l)|{
                    vj_i[(*index_k, *index_l)] = *value
                })
            });


            vj.push(vj_i.to_matrixupper());
        }
        if spin_channel == 1{
            vj.push(MatrixUpper::new(1, 0.0));
        }
        vj

    }

    pub fn generate_vj_on_the_fly_par_new(&mut self) -> Vec<MatrixUpper<f64>>{
        let num_shell = self.mol.cint_bas.len();
        let num_basis = self.mol.num_basis;
        let spin_channel = self.mol.spin_channel;
        let dm = &self.density_matrix;
        let mut vj: Vec<MatrixUpper<f64>> = vec![];
        let mol = &self.mol;
        //utilities::omp_set_num_threads_wrapper(1);
        for i_spin in 0..spin_channel{
            let mut vj_i = MatrixFull::new([num_basis, num_basis], 0.0);
            let mut dm_s = &self.density_matrix[i_spin];
            let par_tasks = utilities::balancing(num_shell*num_shell, rayon::current_num_threads());
            let (sender, receiver) = channel();
            let mut index = Vec::new();
            for l in 0..num_shell {
                for k in 0..l+1 {
                    index.push((k,l))
                }
            };
            index.par_iter().for_each_with(sender,|s,(k,l)|{
                let bas_start_k = mol.cint_fdqc[*k][0];
                let bas_len_k = mol.cint_fdqc[*k][1];
                let bas_start_l = mol.cint_fdqc[*l][0];
                let bas_len_l = mol.cint_fdqc[*l][1];

                let mut klij = mol.int_ijkl_given_kl_v02(*k, *l);
                let mut sum =0.0;
                //let mut out = vec![(0.0, 0usize, 0usize); ];
                //let mut out:Vec<(f64, usize, usize)> = Vec::new();
                let mut out = MatrixFull::new([bas_len_k, bas_len_l],0.0);
                //for ao_k in bas_start_k..bas_start_k+bas_len_k{
                //    for ao_l in bas_start_l..bas_start_l+bas_len_l{
                out.iter_columns_full_mut().enumerate().for_each(|(loc_l,x)|{
                    x.iter_mut().enumerate().for_each(|(loc_k,elem)|{
                        let ao_k = loc_k + bas_start_k;
                        let ao_l = loc_l + bas_start_l;
                        let mut eri_cd = klij.get(&[loc_k, loc_l]).unwrap();
                        let mut sum = dm_s.iter_matrixupper().unwrap().zip(eri_cd.iter_matrixupper().unwrap()).fold(0.0,|sum, (p,eri)| {
                            sum + *p * *eri
                        });

                        let mut diagonal = dm_s.iter_diagonal().unwrap().zip(eri_cd.iter_diagonal().unwrap()).fold(0.0,|diagonal, (p,eri)| {
                            diagonal + *p * *eri
                        });

                        sum = sum*2.0 - diagonal;

                        *elem = sum;
                    })
                });
                s.send((out,*k,*l)).unwrap();
            });
            receiver.into_iter().for_each(|(out,k,l)| {
                let bas_start_k = mol.cint_fdqc[k][0];
                let bas_len_k = mol.cint_fdqc[k][1];
                let bas_start_l = mol.cint_fdqc[l][0];
                let bas_len_l = mol.cint_fdqc[l][1];
                vj_i.copy_from_matr(bas_start_k..bas_start_k+bas_len_k, bas_start_l..bas_start_l+bas_len_l, 
                    &out, 0..bas_len_k,0..bas_len_l);
                //vj_i.iter_submatrix_mut(bas_start_k..bas_start_k+bas_len_k, bas_start_l..bas_start_l+bas_len_l).zip(out.iter())
                //    .for_each(|(to, from)| {*to = *from});
            });
            

            vj.push(vj_i.to_matrixupper());
        }
        if spin_channel == 1{
            vj.push(MatrixUpper::new(1, 0.0));       
        }
        vj
    }

    pub fn generate_vj_on_the_fly_par(&mut self) -> Vec<MatrixUpper<f64>> {
        self.generate_vj_on_the_fly_par_new()
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
                        //        kl += 1;我们的描述子是1*10
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

    pub fn generate_vk_with_isdf_new(&mut self, scaling_factor: f64) -> Vec<MatrixUpper<f64>>{
        let num_basis = self.mol.num_basis;
        let num_state = self.mol.num_state;
        //let npair = num_basis*(num_basis+1)/2;
        let spin_channel = self.mol.spin_channel;
        let mut vk: Vec<MatrixUpper<f64>> = vec![];
        let spin_channel = self.mol.spin_channel;
        let m = self.m.clone().unwrap();
        let tab_ao = self.tab_ao.clone().unwrap();
        let n_ip = m.size[0];

        for i_spin in 0..spin_channel{
            let mut dm_s = &self.density_matrix[i_spin];
            let nw =  self.homo[i_spin]+1;
            let mut kernel_mid = MatrixFull::new([n_ip,num_basis], 0.0);
            _dgemm(&tab_ao,(0..num_basis, 0..n_ip),'T',
                dm_s,(0..num_basis,0..num_basis),'N',
                &mut kernel_mid, (0..n_ip, 0..num_basis),
                1.0,0.0);

            let mut kernel = MatrixFull::new([n_ip,n_ip], 0.0);
            _dgemm(&kernel_mid,(0..n_ip, 0..num_basis),'N',
            &tab_ao, (0..num_basis, 0..n_ip),'N',
            &mut kernel, (0..n_ip,0..n_ip),
            1.0, 0.0);

            kernel.data.iter_mut().zip(m.data.iter()).for_each(|(x,y)|{
                *x *= *y * scaling_factor
            });

            let mut tmp = MatrixFull::new([num_basis, n_ip], 0.0);
            _dgemm(&tab_ao,(0..num_basis,0..n_ip),'N',
            &kernel,(0..n_ip,0..n_ip),'N',
            &mut tmp, (0..num_basis,0..n_ip),
            1.0, 0.0);
            let mut vk_i = MatrixFull::new([num_basis, num_basis], 0.0);
            _dgemm(&tmp,(0..num_basis,0..n_ip),'N',
            &tab_ao,(0..num_basis,0..n_ip),'T',
            &mut vk_i, (0..num_basis,0..num_basis),
            1.0, 0.0);
            vk.push(vk_i.to_matrixupper());
        }
        vk
    }

    pub fn generate_vk_with_isdf_dm_only(&mut self) -> Vec<MatrixUpper<f64>>{
        let num_basis = self.mol.num_basis;
        let num_state = self.mol.num_state;
        //let npair = num_basis*(num_basis+1)/2;
        let spin_channel = self.mol.spin_channel;
        let mut vk: Vec<MatrixUpper<f64>> = vec![];
        let eigv = &self.eigenvectors;
        let spin_channel = self.mol.spin_channel;
        let m = self.m.clone().unwrap();
        let tab_ao = self.tab_ao.clone().unwrap();
        let n_ip = m.size[0];

        for i_spin in 0..spin_channel{
            let occ_s =  &self.occupation[i_spin];
            let nw =  self.homo[i_spin]+1;

            let mut tab_mo = MatrixFull::new([nw,n_ip], 0.0);
            _dgemm(&eigv[i_spin],(0..num_basis, 0..nw),'T',
                &tab_ao,(0..num_basis,0..n_ip),'N',
                &mut tab_mo, (0..nw, 0..n_ip),
                1.0,0.0);

            let mut zip_m_mo = MatrixFull::new([n_ip,n_ip], 0.0);
            _dgemm(&tab_mo,(0..nw, 0..n_ip),'T',
            &tab_mo, (0..nw, 0..n_ip),'N',
            &mut zip_m_mo, (0..n_ip,0..n_ip),
            1.0, 0.0);

            zip_m_mo.data.iter_mut().zip(m.data.iter()).for_each(|(x,y)|{
                *x *= *y * (-1.0)
            });

            let mut tmp = MatrixFull::new([num_basis, n_ip], 0.0);
            _dgemm(&tab_ao,(0..num_basis,0..n_ip),'N',
            &zip_m_mo,(0..n_ip,0..n_ip),'N',
            &mut tmp, (0..num_basis,0..n_ip),
            1.0, 0.0);

            let mut vk_i = MatrixFull::new([num_basis, num_basis], 0.0);
            _dgemm(&tmp,(0..num_basis,0..n_ip),'N',
            &tab_ao,(0..num_basis,0..n_ip),'T',
            &mut vk_i, (0..num_basis,0..num_basis),
            1.0, 0.0);

            vk.push(vk_i.to_matrixupper());

        }
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
        let dt1 = time::Local::now();
        let vj = if self.mol.ctrl.isdf_new || self.mol.ctrl.ri_k_only {
            self.generate_vj_on_the_fly_par()
        }else{
            self.generate_vj_with_ri_v_sync(1.0)
        };

        let dt2 = time::Local::now();
        let scaling_factor = match self.scftype {
            SCFType::RHF => -0.5,
            _ => -1.0,
        };

        let use_dm_only = self.mol.ctrl.use_dm_only;
        let vk = if self.mol.ctrl.use_isdf && !self.mol.ctrl.isdf_new{
            self.generate_vk_with_isdf(scaling_factor, use_dm_only)
        }else if self.mol.ctrl.isdf_new{
            self.generate_vk_with_isdf_new(scaling_factor)
        }else{
            self.generate_vk_with_ri_v(scaling_factor, use_dm_only)
        };


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
    pub fn generate_hf_hamiltonian_ri_v_dm_only(&mut self) {
        let num_basis = self.mol.num_basis;
        let num_state = self.mol.num_state;
        let spin_channel = self.mol.spin_channel;
        //let homo = &self.homo;
        let dt1 = time::Local::now();
        let vj = if self.mol.ctrl.isdf_new || self.mol.ctrl.ri_k_only {
            self.generate_vj_on_the_fly_par()
        } else {
            self.generate_vj_with_ri_v_sync(1.0)
        };
        let dt2 = time::Local::now();
        let scaling_factor = match self.scftype {
            SCFType::RHF => -0.5,
            _ => -1.0,
        };
        let vk = self.generate_vk_with_ri_v(scaling_factor, true);
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
        let dt2 = time::Local::now();
        let scaling_factor = match self.scftype {
            SCFType::RHF => -0.5,
            _ => -1.0,
        }*self.mol.xc_data.dfa_hybrid_scf ;
        if ! scaling_factor.eq(&0.0) {
            // DEBUG IGOR 1
            let use_dm_only = self.mol.ctrl.use_dm_only;
            //self.mol.ctrl.use_dm_only
            let vk = self.generate_vk_with_ri_v(scaling_factor, use_dm_only);
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
            //let (exc,vxc) = self.generate_vxc_rayon_dm_only(1.0);
            let (exc,vxc) = if self.mol.ctrl.use_dm_only {
                self.generate_vxc_rayon_dm_only(1.0)
            } else {
                self.generate_vxc_rayon(1.0)
            };
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

    pub fn generate_ks_hamiltonian_ri_v_dm_only(&mut self) -> (f64,f64) {
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
        let dt2 = time::Local::now();
        let scaling_factor = match self.scftype {
            SCFType::RHF => -0.5,
            _ => -1.0,
        }*self.mol.xc_data.dfa_hybrid_scf ;
        if ! scaling_factor.eq(&0.0) {
            let use_dm_only = self.mol.ctrl.use_auxbas;
            let vk = if self.mol.ctrl.use_isdf{
                self.generate_vk_with_isdf(scaling_factor, use_dm_only)
            }else{
                self.generate_vk_with_ri_v(scaling_factor, use_dm_only)
            };
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
            let (exc,vxc) = self.generate_vxc_rayon_dm_only(1.0);
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

    pub fn generate_hf_hamiltonian_for_guess(&mut self) {
        if self.mol.xc_data.dfa_compnt_scf.len() == 0 {
            if self.mol.ctrl.eri_type.eq("analytic") {
                self.generate_hf_hamiltonian_erifold4();
            } else if  self.mol.ctrl.eri_type.eq("ri_v") {
                self.generate_hf_hamiltonian_ri_v_dm_only();
            }
        } else {
            if self.mol.ctrl.eri_type.eq("analytic") {
                self.generate_ks_hamiltonian_erifold4();
            } else if  self.mol.ctrl.eri_type.eq("ri_v") {
                self.generate_ks_hamiltonian_ri_v_dm_only();
            }
        }
    }

    pub fn evaluate_hf_total_energy(&self) -> f64 {
        let num_basis = self.mol.num_basis;
        let num_state = self.mol.num_state;
        let spin_channel = self.mol.spin_channel;
        let dm = &self.density_matrix;
        let mut total_energy = self.nuc_energy;
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
                total_energy += SCF::par_energy_contraction(&dm_upper, &hc_and_ht);
            },
            _ => {
                let dm_a = &dm[0];
                let dm_a_upper = dm_a.to_matrixupper();
                let dm_b = &dm[1];
                let dm_b_upper = dm_b.to_matrixupper();
                let mut dm_t_upper = dm_a_upper.clone();
                dm_t_upper.data.par_iter_mut().zip(dm_b_upper.data.par_iter()).for_each(|value| {*value.0+=value.1});

                // Now for D^{tot}*H^{core} term
                total_energy += SCF::par_energy_contraction(&dm_t_upper, &self.h_core);
                //println!("debug: {}", total_energy);
                // Now for D^{alpha}*F^{alpha} term
                total_energy += SCF::par_energy_contraction(&dm_a_upper, &self.hamiltonian[0]);
                //println!("debug: {}", total_energy);
                // Now for D^{beta}*F^{beta} term
                total_energy += SCF::par_energy_contraction(&dm_b_upper, &self.hamiltonian[1]);
                //println!("debug: {}", self.scf_energy);

            },
        }
        total_energy
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

        // The following scf energy evaluation follow the formula presented in
        // the quantum chemistry book of Szabo A. and Ostlund N.S. P 150, Formula (3.184)
        self.scf_energy = self.nuc_energy;
        //println!("debug: {}", self.scf_energy);
        // for DFT calculations, we should replace the exchange-correlation (xc) potential by the xc energy
        self.scf_energy = self.scf_energy - vxc_total + exc_total;
        if self.mol.ctrl.print_level>1 {println!("Exc: {:?}, Vxc: {:?}", exc_total, vxc_total)};
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
        let use_dm_only = self.mol.ctrl.use_dm_only;
        //let mut vk = self.generate_vk_with_ri_v(1.0, use_dm_only);
        let mut vk = if self.mol.ctrl.use_isdf{
            self.generate_vk_with_isdf(1.0, use_dm_only)
        }else{
            self.generate_vk_with_ri_v(1.0, use_dm_only)
        };
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

    //pub fn evaluate_xc_energy
       

    pub fn diagonalize_hamiltonian(&mut self) {
        (self.eigenvectors, self.eigenvalues) = diagonalize_hamiltonian_outside(&self);

        //let spin_channel = self.mol.spin_channel;
        //let num_state = self.mol.num_state;
        //let dt1 = time::Local::now();
        //match self.scftype {
        //    SCFType::ROHF => {
        //        let (eigenvector_spin, eigenvalue_spin)=
        //            self.hamiltonian[0].to_matrixupperslicemut()
        //            .lapack_dspgvx(self.ovlp.to_matrixupperslicemut(),num_state).unwrap();
        //        self.eigenvectors[0] = eigenvector_spin;
        //        self.eigenvalues[0] = eigenvalue_spin;
        //        self.eigenvectors[1] = self.eigenvectors[0].clone();
        //        self.eigenvalues[1] = self.eigenvalues[0].clone();
        //    },
        //    _ => {
        //        for i_spin in (0..spin_channel) {
        //            let (eigenvector_spin, eigenvalue_spin)=
        //                self.hamiltonian[i_spin].to_matrixupperslicemut()
        //                .lapack_dspgvx(self.ovlp.to_matrixupperslicemut(),num_state).unwrap();
        //            self.eigenvectors[i_spin] = eigenvector_spin;
        //            self.eigenvalues[i_spin] = eigenvalue_spin;
        //        }
        //    }
        //}
        ////self.formated_eigenvalues(num_state);
        //let dt2 = time::Local::now();
        //let timecost1 = (dt2.timestamp_millis()-dt1.timestamp_millis()) as f64 /1000.0;
        //if self.mol.ctrl.print_level>1 {
        //    println!("Hamiltonian eigensolver:  {:10.2}s", timecost1);
        //}
    }

    pub fn check_scf_convergence(&self, scftracerecode: &ScfTraceRecord) -> [bool;2] {
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
        //scftracerecode.energy_change.push(diff_energy);

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

        if spin_channel==1 {
            if self.mol.ctrl.print_level>0 {
                println!("SCF Change: DM {:10.5e}; eev {:10.5e} Ha; etot {:10.5e} Ha",dm_err[0],eev_err,diff_energy)
            };
            flag[0] = diff_energy.abs()<=scf_acc_etot &&
                      dm_err[0] <=scf_acc_rho &&
                      eev_err <= scf_acc_eev
        } else {
            if self.mol.ctrl.print_level>0 {
                println!("SCF Change: DM ({:10.5e},{:10.5e}); eev {:10.5e} Ha; etot {:10.5e} Ha",dm_err[0],dm_err[1],eev_err,diff_energy)
            };
            flag[0] = diff_energy.abs()<=scf_acc_etot &&
                      dm_err[0] <=scf_acc_rho &&
                      dm_err[1] <=scf_acc_rho &&
                      eev_err <= scf_acc_eev
        }

                  
        // Now check if max_scf_cycle is reached or not
        flag[1] = scftracerecode.num_iter >= max_scf_cycle;

        flag
    }

    pub fn formated_eigenvalues(&self,num_state_to_print:usize) {
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

    // relevant to RI-V
    pub fn generate_vj_with_ri_v(&mut self, scaling_factor: f64) -> Vec<MatrixUpper<f64>> {

        let spin_channel = self.mol.spin_channel;
        let dm = &self.density_matrix;

        vj_upper_with_ri_v(&self.ri3fn, dm, spin_channel, scaling_factor)
    }

    pub fn generate_vj_with_ri_v_sync(&mut self, scaling_factor: f64) -> Vec<MatrixUpper<f64>> {

        let spin_channel = self.mol.spin_channel;
        let dm = &self.density_matrix;

        if self.mol.ctrl.use_ri_symm {
            vj_upper_with_rimatr_sync(&self.rimatr, dm, spin_channel, scaling_factor)
        } else {
            //vj_upper_with_ri_v_sync(&self.ri3fn, dm, spin_channel, scaling_factor)
            if self.mol.ctrl.use_isdf && !self.mol.ctrl.isdf_k_only && !self.mol.ctrl.isdf_new{
                vj_upper_with_ri_v_sync(&self.ri3fn_isdf, dm, spin_channel, scaling_factor)
            }else{
                vj_upper_with_ri_v_sync(&self.ri3fn, dm, spin_channel, scaling_factor)
            }
        }
    }

    pub fn generate_vj_with_isdf(&mut self, scaling_factor: f64) -> Vec<MatrixUpper<f64>> {

        let spin_channel = self.mol.spin_channel;
        let dm = &self.density_matrix;

        vj_upper_with_ri_v(&self.ri3fn_isdf, dm, spin_channel, scaling_factor)
    }

    pub fn generate_vk_with_ri_v(&mut self, scaling_factor: f64, use_dm_only: bool) -> Vec<MatrixUpper<f64>> {
        //let num_basis = self.mol.num_basis;
        //let num_state = self.mol.num_state;
        //let num_auxbas = self.mol.num_auxbas;
        //let npair = num_basis*(num_basis+1)/2;
        let spin_channel = self.mol.spin_channel;

        if self.mol.ctrl.use_ri_symm {
            if use_dm_only {
                let dm = &self.density_matrix;
                vk_upper_with_rimatr_use_dm_only_sync(&mut self.rimatr, dm, spin_channel, scaling_factor)
            } else {
                let eigv = &self.eigenvectors;
                let occupation = &self.occupation;
                let num_elec = &self.mol.num_elec;
                vk_upper_with_rimatr_sync(&mut self.rimatr, eigv, num_elec, occupation, spin_channel, scaling_factor)
            }
        } else if self.mol.ctrl.isdf_new{
            self.generate_vk_with_isdf_new(scaling_factor)
        }else{
            if use_dm_only {
                let dm = &self.density_matrix;
                vk_upper_with_ri_v_use_dm_only_sync(&mut self.ri3fn, dm, spin_channel, scaling_factor)
            } else {
                let eigv = &self.eigenvectors;
                vk_upper_with_ri_v_sync(&mut self.ri3fn, eigv, &self.mol.num_elec, &self.occupation, 
                                        spin_channel, scaling_factor)
            }
        }

    

    }

    pub fn generate_vk_with_isdf(&mut self, scaling_factor: f64, use_dm_only: bool) -> Vec<MatrixUpper<f64>> {
        //let num_basis = self.mol.num_basis;
        //let num_state = self.mol.num_state;
        //let num_auxbas = self.mol.num_auxbas;
        //let npair = num_basis*(num_basis+1)/2;
        let spin_channel = self.mol.spin_channel;

        if self.mol.ctrl.use_ri_symm {
            let dm = &self.density_matrix;
            vk_upper_with_rimatr_use_dm_only_sync(&mut self.rimatr, dm, spin_channel, scaling_factor)
        } else {
            if use_dm_only {
                //println!("use isdf to generate k");
                let dm = &self.density_matrix;
                //&dm[0].formated_output_e(5, "full");
                vk_upper_with_ri_v_use_dm_only_sync(&mut self.ri3fn_isdf, dm, spin_channel, scaling_factor)
            } else {
                let eigv = &self.eigenvectors;
                vk_upper_with_ri_v_sync(&mut self.ri3fn_isdf, eigv, &self.mol.num_elec, &self.occupation, 
                                        spin_channel, scaling_factor)
            }
        }
        

    }

    pub fn generate_vxc(&self, scaling_factor: f64) -> (f64, Vec<MatrixUpper<f64>>) {
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
        let dm = &self.density_matrix;
        let mo = &self.eigenvectors;
        let occ = &self.occupation;
        let print_level = self.mol.ctrl.print_level;
        if let Some(grids) = &self.grids {
            let dt0 = utilities::init_timing();
            let (exc,mut vxc_ao) = self.mol.xc_data.xc_exc_vxc(grids, spin_channel,dm, mo, occ, print_level);
            let dt1 = utilities::timing(&dt0, Some("Total vxc_ao time"));
            exc_spin = exc;
            if let Some(ao) = &grids.ao {
                // Evaluate the exchange-correlation energy
                //exc_total = izip!(grids.weights.iter(),exc.data.iter()).fold(0.0,|acc,(w,e)| {
                //    acc + w*e
                //});
                for i_spin in 0..spin_channel {
                    let vxc_mf_s = vxc_mf.get_mut(i_spin).unwrap();
                    *vxc_mf_s = MatrixFull::new([num_basis,num_basis],0.0f64);
                    let vxc_ao_s = vxc_ao.get(i_spin).unwrap();
                    _dgemm_full(ao, 'N', vxc_ao_s, 'T', vxc_mf_s, 1.0, 0.0);
                    //vxc_mf_s.lapack_dgemm(ao, vxc_ao_s, 'N', 'T', 1.0, 0.0);
                }
            }
            let dt2 = utilities::timing(&dt1, Some("From vxc_ao to vxc"));
        }

        //println!("debug {:?}", exc_spin);

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

    pub fn generate_vxc_rayon_dm_only(&self, scaling_factor: f64) -> (f64, Vec<MatrixUpper<f64>>) {
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
        //println!("debug: {:?}", &self.occupation);
        if let Some(grids) = &self.grids {
            //println!("debug: {:?}",grids.parallel_balancing);
            let (sender, receiver) = channel();
            grids.parallel_balancing.par_iter().for_each_with(sender,|s,range_grids| {
                //println!("debug 0: {}", rayon::current_thread_index().unwrap());
                let (exc,vxc_ao,total_elec) = self.mol.xc_data.xc_exc_vxc_slots_dm_only(range_grids.clone(), grids, spin_channel,dm, mo, occ);
                //println!("debug: {:?}", vxc_ao[0].size());
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

        if self.mol.ctrl.print_level>1 {
            if spin_channel==1 {
                println!("total electron number: {:16.8}", total_elec[0]);
            } else {
                println!("electron number in alpha-channel: {:12.8}", total_elec[0]);
                println!("electron number in beta-channel:  {:12.8}", total_elec[1]);
            }
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
        //println!("debug: {:?}", &self.occupation);
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

        if self.mol.ctrl.print_level>1 {
            if spin_channel==1 {
                println!("total electron number: {:16.8}", total_elec[0]);
            } else {
                println!("electron number in alpha-channel: {:12.8}", total_elec[0]);
                println!("electron number in beta-channel:  {:12.8}", total_elec[1]);
            }
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

    pub fn generate_ri3mo_rayon(&mut self, row_range: std::ops::Range<usize>, col_range: std::ops::Range<usize>) {
        let (mut ri3ao, mut basbas2baspair, mut baspar2basbas) =  if let Some((riao,basbas2baspair, baspar2basbas))=&mut self.rimatr {
            (riao,basbas2baspair, baspar2basbas)
        } else {
            panic!("rimatr should be initialized in the preparation of ri3mo");
        };
        let mut ri3mo: Vec<(RIFull<f64>,std::ops::Range<usize>, std::ops::Range<usize>)> = vec![];
        for i_spin in 0..self.mol.spin_channel {
            let eigenvector = &self.eigenvectors[i_spin];
            ri3mo.push(
                ao2mo_rayon(
                    eigenvector, ri3ao, 
                    row_range.clone(), 
                    col_range.clone()
                ).unwrap()
            )
        }

        // deallocate the rimatr to save the memory
        self.rimatr = None;
        self.ri3mo = Some(ri3mo);


    }

}


/// return the occupation range and virtual range for the preparation of ri3mo;
pub fn determine_ri3mo_size_for_pt2_and_rpa(scf_data: &SCF) -> (std::ops::Range<usize>, std::ops::Range<usize>) {
    let num_state = scf_data.mol.num_state;
    let mut homo = 0_usize;
    let mut lumo = num_state;
    let start_mo = scf_data.mol.start_mo;

    for i_spin in 0..scf_data.mol.spin_channel {

        let i_homo = scf_data.homo.get(i_spin).unwrap().clone();
        let i_lumo = scf_data.lumo.get(i_spin).unwrap().clone();

        homo = homo.max(i_homo);
        lumo = lumo.min(i_lumo);
    }

    (start_mo..homo+1, lumo..num_state)
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

pub fn vj_upper_with_rimatr_sync(
                ri3fn: &Option<(MatrixFull<f64>,MatrixFull<usize>,Vec<[usize;2]>)>,
                dm: &Vec<MatrixFull<f64>>, 
                spin_channel: usize, scaling_factor: f64)  -> Vec<MatrixUpper<f64>> {
    vj_upper_with_rimatr_sync_v02(ri3fn,dm,spin_channel,scaling_factor)
}

pub fn vj_upper_with_rimatr_sync_v01(
                ri3fn: &Option<(MatrixFull<f64>,MatrixFull<usize>,Vec<[usize;2]>)>,
                dm: &Vec<MatrixFull<f64>>, 
                spin_channel: usize, scaling_factor: f64)  -> Vec<MatrixUpper<f64>> {
    let mut vj: Vec<MatrixUpper<f64>> = vec![MatrixUpper::new(1,0.0f64),MatrixUpper::new(1,0.0f64)];
    //// In this subroutine, we call the lapack dgemm in a rayon parallel environment.
    //// In order to ensure the efficiency, we disable the openmp ability and re-open it in the end of subroutien
    //let default_omp_num_threads = unsafe {openblas_get_num_threads()};
    ////println!("debug: default omp_num_threads: {}", default_omp_num_threads);
    //unsafe{openblas_set_num_threads(1)};
    
    if let Some((ri3fn,basbas2baspar,baspar2basbas)) = ri3fn {
        let num_basis = basbas2baspar.size[0];
        let num_baspar = ri3fn.size[0];
        let num_auxbas = ri3fn.size[1];
        //let npair = num_basis*(num_basis+1)/2;
        //println!("debug npair:{}, num_baspar:{}", npair, num_baspar);
        for i_spin in (0..spin_channel) {
            //let mut tmp_mu = vec![0.0f64;num_auxbas];
            let mut vj_spin = &mut vj[i_spin];
            *vj_spin = MatrixUpper::new(num_baspar,0.0f64);
            let dm_s = &dm[i_spin];

            let (sender, receiver) = channel();
            ri3fn.par_iter_columns_full().enumerate().for_each_with(sender, |s, (i,m)| {
                //prepare \sum_{kl}D_{kl}*M_{kl}^{\mu} for each \mu -> tmp_mu
                let riupper = MatrixUpperSlice::from_vec(m);
                let mut tmp_mu =
                    m.iter().zip(dm_s.iter_matrixupper().unwrap()).fold(0.0_f64, |acc,(m,d)| {
                        acc + *m * (*d)
                    });
                //let diagonal_term = riupper.get_diagonal_terms().unwrap()
                //    .iter().zip(dm[i_spin].iter_diagonal().unwrap()).fold(0.0f64, |acc, (v1,v2)| {
                //        acc + *v1*v2
                //});
                let diagonal_term = riupper.iter_diagonal()
                    .zip(dm[i_spin].iter_diagonal().unwrap()).fold(0.0f64, |acc, (v1,v2)| {
                        acc + *v1*v2
                });

                tmp_mu = 2.0_f64*tmp_mu - diagonal_term;

                let m_ij_upper = m.iter().map(|v| *v*tmp_mu).collect_vec();
                s.send(m_ij_upper).unwrap();
            });
            // fill vj[i_spin] with the contribution from the given {\mu}:
            //
            // M_{ij}^{\mu}*(\sum_{kl}D_{kl}*M_{kl}^{\mu})
            //
            receiver.iter().for_each(|(m_ij_upper)| {
                vj_spin.data.iter_mut().zip(m_ij_upper.iter())
                    .for_each(|value| *value.0 += *value.1); 
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
    //vj[0].formated_output(5, "full");
    //println!("debug: {:?}", vj[0]);

    vj
}

pub fn vj_upper_with_rimatr_sync_v02(
                ri3fn: &Option<(MatrixFull<f64>,MatrixFull<usize>,Vec<[usize;2]>)>,
                dm: &Vec<MatrixFull<f64>>, 
                spin_channel: usize, scaling_factor: f64)  -> Vec<MatrixUpper<f64>> {
    let mut vj: Vec<MatrixUpper<f64>> = vec![MatrixUpper::new(1,0.0f64),MatrixUpper::new(1,0.0f64)];
    if let Some((ri3fn,basbas2baspar,baspar2basbas)) = ri3fn {
        let num_basis = basbas2baspar.size[0];
        let num_baspar = ri3fn.size[0];
        let num_auxbas = ri3fn.size[1];
        for i_spin in (0..spin_channel) {
            //let mut tmp_mu = vec![0.0f64;num_auxbas];
            let mut vj_spin = &mut vj[i_spin];
            *vj_spin = MatrixUpper::new(num_baspar,0.0f64);
            //let mut dm_s = dm[i_spin].clone();
            //dm_s.iter_diagonal_mut().unwrap().for_each(|x| *x = *x/2.0);

            let mut dm_s_upper = MatrixUpper::from_vec(num_baspar,dm[i_spin].iter_matrixupper().unwrap().map(|x| *x).collect_vec()).unwrap();
            dm_s_upper.iter_diagonal_mut().for_each(|x| {*x = *x/2.0});

            let mut tmp_v = vec![0.0;num_auxbas];

            _dgemv(ri3fn, &dm_s_upper.data, &mut tmp_v, 'T', 2.0, 0.0, 1, 1);

            _dgemv(ri3fn, &tmp_v, &mut vj_spin.data, 'N',1.0,0.0,1,1);

        }
    }

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

pub fn vk_upper_with_ri_v_use_dm_only_sync(
                ri3fn: &Option<RIFull<f64>>,
                dm: &Vec<MatrixFull<f64>>,
                spin_channel: usize, scaling_factor: f64)  -> Vec<MatrixUpper<f64>> {
    // In this subroutine, we call the lapack dgemm in a rayon parallel environment.
    // In order to ensure the efficiency, we disable the openmp ability and re-open it in the end of subroutien
    let default_omp_num_threads = utilities::omp_get_num_threads_wrapper();
    //println!("debug: default omp_num_threads: {}", default_omp_num_threads);
    utilities::omp_set_num_threads_wrapper(1);
    //let mut bm = RIFull::new([num_state,num_basis,num_auxbas], 0.0f64);
    let mut vk: Vec<MatrixUpper<f64>> = vec![MatrixUpper::new(1,0.0f64),MatrixUpper::new(1,0.0f64)];

    //============================debug===================================
    //&dm[0].formated_output_e(5, "full");
    //====================================================================

    if let Some(ri3fn) = ri3fn {
        let num_basis = dm[0].size()[0];
        let num_baspair = (num_basis+1)*num_basis/2;
        let num_auxbas = ri3fn.size[2];
        for i_spin in 0..spin_channel {
            let mut vk_s = &mut vk[i_spin];
            *vk_s = MatrixUpper::new(num_baspair,0.0_f64);
            let dm_s = &dm[i_spin];
            //dm_s.formated_output(5, "upper");
            let (sender, receiver) = channel();
            ri3fn.par_iter_auxbas(0..num_auxbas).unwrap().for_each_with(sender,|s, m| {
                let mut tmp_mat = MatrixFull::new([num_basis,num_basis],0.0_f64);
                let mut reduced_ri3fn = MatrixFullSlice {
                    size:  &[num_basis,num_basis],
                    indicing: &[1,num_basis],
                    data: m,
                };
                //_dgemm(&reduced_ri3fn, (0..num_basis,0..num_basis), 'N', 
                //       dm_s, (0..num_basis,0..num_basis), 'N', 
                //       &mut tmp_mat, (0..num_basis,0..num_basis), 1.0, 0.0);
                //let mut vk_sm = MatrixFull::new([num_basis,num_basis],0.0_f64);
                //_dgemm(&tmp_mat, (0..num_basis,0..num_basis), 'N', 
                //       &reduced_ri3fn, (0..num_basis,0..num_basis), 'T', 
                //       &mut vk_sm, (0..num_basis,0..num_basis), 1.0, 0.0);
                //tmp_mat = ri3fn \cdot dm
                _dsymm(&reduced_ri3fn, dm_s, &mut tmp_mat, 'L', 'U', 1.0, 0.0);
                let mut vk_sm = MatrixFull::new([num_basis,num_basis],0.0_f64);
                //vk_sm = ri3fn \cdot dm \cdot ri3fn
                _dsymm(&reduced_ri3fn, &tmp_mat, &mut vk_sm, 'R', 'U', 1.0, 0.0);

                s.send(vk_sm.to_matrixupper()).unwrap();
            });

            receiver.into_iter().for_each(|vk_mu_upper| {
                vk_s.data.par_iter_mut()
                    .zip(vk_mu_upper.data.par_iter()).for_each(|value| {
                    *value.0 += *value.1
                })
            });
        }
    }

    if scaling_factor!=1.0f64 {
        for i_spin in (0..spin_channel) {
            vk[i_spin].data.par_iter_mut().for_each(|f| *f = *f*scaling_factor)
        }
    };

    // reuse the default omp_num_threads setting
    utilities::omp_set_num_threads_wrapper(default_omp_num_threads);

    //============================debug==================================

    /* let mut data = vec![0.0; vk[0].data.len()];
    for i in 0..vk[0].data.len(){
        data[i] = vk[0].data[i] * 2.0;
    }
    let final_vk = MatrixUpper{size: vk[0].size, data: data};
    &final_vk.formated_output(5, "upper"); */
    //===================================================================
    vk
}
pub fn vk_upper_with_rimatr_use_dm_only_sync(
                ri3fn: &Option<(MatrixFull<f64>,MatrixFull<usize>,Vec<[usize;2]>)>,
                dm: &Vec<MatrixFull<f64>>,
                spin_channel: usize, scaling_factor: f64)  -> Vec<MatrixUpper<f64>> {

    vk_upper_with_rimatr_use_dm_only_sync_v02(ri3fn, dm, spin_channel, scaling_factor)
}

pub fn vk_upper_with_rimatr_use_dm_only_sync_v01(
                ri3fn: &Option<(MatrixFull<f64>,MatrixFull<usize>,Vec<[usize;2]>)>,
                dm: &Vec<MatrixFull<f64>>,
                spin_channel: usize, scaling_factor: f64)  -> Vec<MatrixUpper<f64>> {
    // In this subroutine, we call the lapack dgemm in a rayon parallel environment.
    // In order to ensure the efficiency, we disable the openmp ability and re-open it in the end of subroutien
    let default_omp_num_threads = utilities::omp_get_num_threads_wrapper();
    //println!("debug: default omp_num_threads: {}", default_omp_num_threads);
    utilities::omp_set_num_threads_wrapper(1);
    //let mut bm = RIFull::new([num_state,num_basis,num_auxbas], 0.0f64);
    let mut vk: Vec<MatrixUpper<f64>> = vec![MatrixUpper::new(1,0.0f64),MatrixUpper::new(1,0.0f64)];

    if let Some((ri3fn,basbas2baspar,baspar2basbas)) = ri3fn {
        let num_basis = dm[0].size()[0];
        let num_baspair = (num_basis+1)*num_basis/2;
        //let num_auxbas = ri3fn.size[2];
        for i_spin in 0..spin_channel {
            let mut vk_s = &mut vk[i_spin];
            *vk_s = MatrixUpper::new(num_baspair,0.0_f64);
            let dm_s = &dm[i_spin];
            let (sender, receiver) = channel();
            ri3fn.par_iter_columns_full().for_each_with(sender,|s, m| {
                let mut tmp_mat = MatrixFull::new([num_basis,num_basis],0.0_f64);
                let mut reduced_ri3fn = MatrixFull::new([num_basis,num_basis],0.0_f64);

                reduced_ri3fn.iter_matrixupper_mut().unwrap().zip(m.iter()).for_each(|(to, from)| {*to = *from});

                _dsymm(&reduced_ri3fn, dm_s, &mut tmp_mat, 'L', 'U', 1.0, 0.0);
                let mut vk_sm = MatrixFull::new([num_basis,num_basis],0.0_f64);
                _dsymm(&reduced_ri3fn, &tmp_mat, &mut vk_sm, 'R', 'U', 1.0, 0.0);

                s.send(vk_sm.to_matrixupper()).unwrap();
            });

            receiver.into_iter().for_each(|vk_mu_upper| {
                vk_s.data.iter_mut()
                    .zip(vk_mu_upper.data.iter()).for_each(|value| {
                    *value.0 += *value.1
                })
            });
        }
    }

    if scaling_factor!=1.0f64 {
        for i_spin in (0..spin_channel) {
            vk[i_spin].data.par_iter_mut().for_each(|f| *f = *f*scaling_factor)
        }
    };

    // reuse the default omp_num_threads setting
    utilities::omp_set_num_threads_wrapper(default_omp_num_threads);

    vk
}

pub fn vk_upper_with_rimatr_use_dm_only_sync_v02(
                ri3fn: &Option<(MatrixFull<f64>,MatrixFull<usize>,Vec<[usize;2]>)>,
                dm: &Vec<MatrixFull<f64>>,
                spin_channel: usize, scaling_factor: f64)  -> Vec<MatrixUpper<f64>> {
    // In this subroutine, we call the lapack dgemm in a rayon parallel environment.
    // In order to ensure the efficiency, we disable the openmp ability and re-open it in the end of subroutien
    let default_omp_num_threads = utilities::omp_get_num_threads_wrapper();
    //utilities::omp_set_num_threads_wrapper(1);
    //let mut bm = RIFull::new([num_state,num_basis,num_auxbas], 0.0f64);
    let mut vk: Vec<MatrixUpper<f64>> = vec![MatrixUpper::new(1,0.0f64),MatrixUpper::new(1,0.0f64)];

    if let Some((ri3fn,basbas2baspar,baspar2basbas)) = ri3fn {
        let num_basis = dm[0].size()[0];
        let num_baspair = ri3fn.size()[0];
        let num_auxbas = ri3fn.size()[1];
        //let num_auxbas = ri3fn.size[2];
        for i_spin in 0..spin_channel {
            let mut vk_s = &mut vk[i_spin];
            *vk_s = MatrixUpper::new(num_baspair,0.0_f64);
            //let dm_s = &dm[i_spin];
            utilities::omp_set_num_threads_wrapper(default_omp_num_threads);
            let dm_s = _power_rayon(&dm[i_spin], 0.5, 1.0e-8).unwrap();
            utilities::omp_set_num_threads_wrapper(1);
            let batch_num_auxbas = utilities::balancing(num_auxbas, rayon::current_num_threads());
            let (sender, receiver) = channel();
            batch_num_auxbas.par_iter().for_each_with(sender, |s,loc_auxbas| {
                let mut tmp_mat = MatrixFull::new([num_basis,num_basis],0.0_f64);
                let mut reduced_ri3fn = MatrixFull::new([num_basis,num_basis],0.0_f64);
                let mut vk_sm = MatrixFull::new([num_basis,num_basis],0.0_f64);
                ri3fn.iter_columns(loc_auxbas.clone()).for_each(|m| {
                    reduced_ri3fn.iter_matrixupper_mut().unwrap().zip(m.iter()).for_each(|(to, from)| {*to = *from});
                    //_dsymm(&reduced_ri3fn, dm_s, &mut tmp_mat, 'L', 'U', 1.0, 0.0);
                    //_dsymm(&reduced_ri3fn, &tmp_mat, &mut vk_sm, 'R', 'U', 1.0, 1.0);
                    _dsymm(&reduced_ri3fn, &dm_s, &mut tmp_mat, 'L', 'U', 1.0, 0.0);
                    _dsyrk(&tmp_mat, &mut vk_sm, 'U', 'N', 1.0, 1.0)
                });
                s.send(vk_sm.to_matrixupper()).unwrap();
            });

            receiver.into_iter().for_each(|vk_mu_upper| {
                vk_s.data.par_iter_mut()
                    .zip(vk_mu_upper.data.par_iter()).for_each(|value| {
                    *value.0 += *value.1
                })
            });
        }
    }

    if scaling_factor!=1.0f64 {
        for i_spin in (0..spin_channel) {
            vk[i_spin].data.par_iter_mut().for_each(|f| *f = *f*scaling_factor)
        }
    };

    // reuse the default omp_num_threads setting
    utilities::omp_set_num_threads_wrapper(default_omp_num_threads);

    vk
}
pub fn vk_upper_with_rimatr_sync(
                ri3fn: &Option<(MatrixFull<f64>,MatrixFull<usize>,Vec<[usize;2]>)>,
                eigv: &[MatrixFull<f64>;2], 
                num_elec: &[f64;3], occupation: &[Vec<f64>;2],
                //dm: &Vec<MatrixFull<f64>>,
                spin_channel: usize, scaling_factor: f64)  -> Vec<MatrixUpper<f64>> {
    vk_upper_with_rimatr_sync_v03(ri3fn,eigv,num_elec,occupation,spin_channel,scaling_factor)
}

pub fn vk_upper_with_rimatr_sync_v01(
                ri3fn: &Option<(MatrixFull<f64>,MatrixFull<usize>,Vec<[usize;2]>)>,
                eigv: &[MatrixFull<f64>;2], 
                num_elec: &[f64;3], occupation: &[Vec<f64>;2],
                //dm: &Vec<MatrixFull<f64>>,
                spin_channel: usize, scaling_factor: f64)  -> Vec<MatrixUpper<f64>> {
    // In this subroutine, we call the lapack dgemm in a rayon parallel environment.
    // In order to ensure the efficiency, we disable the openmp ability and re-open it in the end of subroutien
    let default_omp_num_threads = utilities::omp_get_num_threads_wrapper();
    //println!("debug: default omp_num_threads: {}", default_omp_num_threads);
    utilities::omp_set_num_threads_wrapper(1);
    //let mut bm = RIFull::new([num_state,num_basis,num_auxbas], 0.0f64);
    let mut vk: Vec<MatrixUpper<f64>> = vec![MatrixUpper::new(1,0.0f64),MatrixUpper::new(1,0.0f64)];

    if let Some((ri3fn,basbas2baspar,baspar2basbas)) = ri3fn {
        let num_basis = eigv[0].size()[0];
        let num_baspair = (num_basis+1)*num_basis/2;
        //let num_auxbas = ri3fn.size[2];
        for i_spin in 0..spin_channel {
            let mut vk_s = &mut vk[i_spin];
            *vk_s = MatrixUpper::new(num_baspair,0.0_f64);
            let eigv_s = &eigv[i_spin];
            let nw = num_elec[i_spin+1].ceil() as usize;
            if nw>0 {
                let mut tmp_mat = MatrixFull::new([num_basis,nw],0.0_f64);
                tmp_mat.data.iter_mut().zip(eigv_s.iter_submatrix(0..num_basis,0..nw))
                    .for_each(|value| {*value.0 = *value.1});
                let occ_s = &occupation[i_spin][0..nw];
                tmp_mat.data.par_chunks_exact_mut(tmp_mat.size[0]).zip(occ_s.par_iter()).for_each(|(to_value, from_value)| {
                        to_value.iter_mut().for_each(|to_value| {*to_value = *to_value*from_value.sqrt()});
                });
                let reduced_eigv_s = tmp_mat;

                //let dm_s = &dm[i_spin];

                let (sender, receiver) = channel();
                ri3fn.par_iter_columns_full().for_each_with(sender,|s, m| {
                    //let mut tmp_mat = MatrixFull::new([num_basis,num_basis],0.0_f64);
                    let mut reduced_ri3fn = MatrixFull::new([num_basis,num_basis],0.0_f64);

                    reduced_ri3fn.iter_matrixupper_mut().unwrap().zip(m.iter()).for_each(|(to, from)| {*to = *from});

                    let mut tmp_mc = MatrixFull::new([num_basis,nw],0.0_f64);
                    //tmp_mc = ri3fn \cdot eigv \cdot occ.sqrt()
                    _dsymm(&reduced_ri3fn, &reduced_eigv_s, &mut tmp_mc, 'L', 'U', 1.0, 0.0);

                    let mut vk_sm = MatrixFull::new([num_basis,num_basis],0.0_f64);
                    _dsyrk(&tmp_mc, &mut vk_sm, 'U', 'N', 1.0, 0.0);

                    s.send(vk_sm.to_matrixupper()).unwrap();
                });

                receiver.into_iter().for_each(|vk_mu_upper| {
                    vk_s.data.par_iter_mut()
                        .zip(vk_mu_upper.data.par_iter()).for_each(|value| {
                        *value.0 += *value.1
                    })
                });
            }
        }
    }

    if scaling_factor!=1.0f64 {
        for i_spin in (0..spin_channel) {
            vk[i_spin].data.par_iter_mut().for_each(|f| *f = *f*scaling_factor)
        }
    };

    // reuse the default omp_num_threads setting
    utilities::omp_set_num_threads_wrapper(default_omp_num_threads);

    vk
}

/// a new vk version with the parallelization giving to openmk.
pub fn vk_upper_with_rimatr_sync_v02(
                ri3fn: &Option<(MatrixFull<f64>,MatrixFull<usize>,Vec<[usize;2]>)>,
                eigv: &[MatrixFull<f64>;2], 
                num_elec: &[f64;3], occupation: &[Vec<f64>;2],
                //dm: &Vec<MatrixFull<f64>>,
                spin_channel: usize, scaling_factor: f64)  -> Vec<MatrixUpper<f64>> {

    let mut vk: Vec<MatrixUpper<f64>> = vec![MatrixUpper::new(1,0.0f64),MatrixUpper::new(1,0.0f64)];

    if let Some((ri3fn,basbas2baspar,baspar2basbas)) = ri3fn {
        let num_basis = eigv[0].size()[0];
        let num_baspair = (num_basis+1)*num_basis/2;
        //let num_auxbas = ri3fn.size[2];
        for i_spin in 0..spin_channel {
            let mut vk_s = &mut vk[i_spin];
            let mut vk_sm = MatrixFull::new([num_basis,num_basis],0.0_f64);
            //*vk_s = MatrixUpper::new(num_baspair,0.0_f64);
            let eigv_s = &eigv[i_spin];
            let nw = num_elec[i_spin+1].ceil() as usize;
            if nw>0 {
                let mut tmp_mat = MatrixFull::new([num_basis,nw],0.0_f64);
                tmp_mat.data.iter_mut().zip(eigv_s.iter_submatrix(0..num_basis,0..nw))
                    .for_each(|value| {*value.0 = *value.1});
                let occ_s = &occupation[i_spin][0..nw];
                tmp_mat.data.chunks_exact_mut(tmp_mat.size[0]).zip(occ_s.iter()).for_each(|(to_value, from_value)| {
                        to_value.iter_mut().for_each(|to_value| {*to_value = *to_value*from_value.sqrt()});
                });
                let reduced_eigv_s = tmp_mat;

                //let dm_s = &dm[i_spin];

                //let (sender, receiver) = channel();

                let mut tmp_mc = MatrixFull::new([num_basis,nw],0.0_f64);
                let mut reduced_ri3fn = MatrixFull::new([num_basis,num_basis],0.0_f64);

                ri3fn.iter_columns_full().for_each(|m| {
                    //let mut tmp_mat = MatrixFull::new([num_basis,num_basis],0.0_f64);
                    reduced_ri3fn.iter_matrixupper_mut().unwrap().zip(m.iter()).for_each(|(to, from)| {*to = *from});

                    //tmp_mc = ri3fn \cdot eigv
                    _dsymm(&reduced_ri3fn, &reduced_eigv_s, &mut tmp_mc, 'L', 'U', 1.0, 0.0);

                    _dsyrk(&tmp_mc, &mut vk_sm, 'U', 'N', 1.0, 1.0);

                    //s.send(vk_sm.to_matrixupper()).unwrap();
                });
                *vk_s = vk_sm.to_matrixupper();
            } else {
              *vk_s = MatrixUpper::new(num_baspair,0.0_f64);
            }
        }
    }

    if scaling_factor!=1.0f64 {
        for i_spin in (0..spin_channel) {
            vk[i_spin].data.par_iter_mut().for_each(|f| *f = *f*scaling_factor)
        }
    };

    //// reuse the default omp_num_threads setting
    //utilities::omp_set_num_threads_wrapper(default_omp_num_threads);

    vk
}

pub fn vk_upper_with_rimatr_sync_v03(
                ri3fn: &Option<(MatrixFull<f64>,MatrixFull<usize>,Vec<[usize;2]>)>,
                eigv: &[MatrixFull<f64>;2], 
                num_elec: &[f64;3], occupation: &[Vec<f64>;2],
                //dm: &Vec<MatrixFull<f64>>,
                spin_channel: usize, scaling_factor: f64)  -> Vec<MatrixUpper<f64>> {
    // In this subroutine, we call the lapack dgemm in a rayon parallel environment.
    // In order to ensure the efficiency, we disable the openmp ability and re-open it in the end of subroutien
    let default_omp_num_threads = utilities::omp_get_num_threads_wrapper();
    //println!("debug: default omp_num_threads: {}", default_omp_num_threads);
    utilities::omp_set_num_threads_wrapper(1);
    //let mut bm = RIFull::new([num_state,num_basis,num_auxbas], 0.0f64);
    let mut vk: Vec<MatrixUpper<f64>> = vec![MatrixUpper::new(1,0.0f64),MatrixUpper::new(1,0.0f64)];

    if let Some((ri3fn,basbas2baspar,baspar2basbas)) = ri3fn {
        let num_basis = eigv[0].size()[0];
        let num_baspair = ri3fn.size()[0];
        let num_auxbas = ri3fn.size()[1];
        //let num_auxbas = ri3fn.size[2];
        for i_spin in 0..spin_channel {
            let mut vk_s = &mut vk[i_spin];
            *vk_s = MatrixUpper::new(num_baspair,0.0_f64);
            let eigv_s = &eigv[i_spin];
            let nw = num_elec[i_spin+1].ceil() as usize;
            if nw>0 {
                let mut tmp_mat = MatrixFull::new([num_basis,nw],0.0_f64);
                tmp_mat.data.iter_mut().zip(eigv_s.iter_submatrix(0..num_basis,0..nw))
                    .for_each(|value| {*value.0 = *value.1});
                let occ_s = &occupation[i_spin][0..nw];
                tmp_mat.data.par_chunks_exact_mut(tmp_mat.size[0]).zip(occ_s.par_iter()).for_each(|(to_value, from_value)| {
                        to_value.iter_mut().for_each(|to_value| {*to_value = *to_value*from_value.sqrt()});
                });
                let reduced_eigv_s = tmp_mat;

                //let dm_s = &dm[i_spin];

                let batch_num_auxbas = utilities::balancing(num_auxbas, rayon::current_num_threads());
                let (sender, receiver) = channel();
                batch_num_auxbas.par_iter().for_each_with(sender, |s, loc_auxbas| {
                    let mut reduced_ri3fn = MatrixFull::new([num_basis,num_basis],0.0_f64);
                    let mut vk_sm = MatrixFull::new([num_basis,num_basis],0.0_f64);
                    let mut tmp_mc = MatrixFull::new([num_basis,nw],0.0_f64);
                    ri3fn.iter_columns(loc_auxbas.clone()).for_each(|m| {
                        reduced_ri3fn.iter_matrixupper_mut().unwrap().zip(m.iter()).for_each(|(to, from)| {*to = *from});
                        _dsymm(&reduced_ri3fn, &reduced_eigv_s, &mut tmp_mc, 'L', 'U', 1.0, 0.0);
                        _dsyrk(&tmp_mc, &mut vk_sm, 'U', 'N', 1.0, 1.0);
                    });
                    s.send(vk_sm.to_matrixupper()).unwrap();
                });

                receiver.into_iter().for_each(|vk_mu_upper| {
                    vk_s.data.par_iter_mut()
                        .zip(vk_mu_upper.data.par_iter()).for_each(|value| {
                        *value.0 += *value.1
                    })
                });
            }
        }
    }

    if scaling_factor!=1.0f64 {
        for i_spin in (0..spin_channel) {
            vk[i_spin].data.par_iter_mut().for_each(|f| *f = *f*scaling_factor)
        }
    };

    // reuse the default omp_num_threads setting
    utilities::omp_set_num_threads_wrapper(default_omp_num_threads);

    vk
}

//==========================need to be checked=============================
pub fn vk_upper_with_ri_v_sync(
                ri3fn: &mut Option<RIFull<f64>>,
                eigv: &[MatrixFull<f64>;2], 
                num_elec: &[f64;3], occupation: &[Vec<f64>;2],
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
            let nw = num_elec[i_spin+1].ceil() as usize;
            if nw>0 {
                let mut tmp_mat = MatrixFull::new([num_basis,nw],0.0_f64);
                tmp_mat.data.iter_mut().zip(eigv_s.iter_submatrix(0..num_basis,0..nw))
                    .for_each(|value| {*value.0 = *value.1});
                let reduced_eigv_s = tmp_mat;
                let occ_s = &occupation[i_spin][0..nw];
                //let mut tmp_b = MatrixFull::new([num_basis,num_basis],0.0_f64);
                let (sender, receiver) = channel();
                ri3fn.par_iter_mut_auxbas(0..num_auxbas).unwrap().for_each_with(sender, |s, m| {

                    let mut reduced_ri3fn = MatrixFullSlice {
                        size:  &[num_basis,num_basis], 
                        indicing: &[1,num_basis],
                        data: m,
                    };
                    //tmp_mat: copy of related eigenvalue; reduced_ri3fn: certain part of ri3fn
                    let mut tmp_mc = MatrixFull::new([num_basis,nw],0.0_f64);
                    //tmp_mc = ri3fn \cdot eigv
                    tmp_mc.to_matrixfullslicemut().lapack_dgemm(&reduced_ri3fn, &reduced_eigv_s.to_matrixfullslice(), 'N', 'N', 1.0, 0.0);
                    //tmp_mat = tmp_mc (ri3fn \cdot eigv)
                    let mut tmp_mat = tmp_mc.clone();
                    //tmp_mat = tmp_mc * occ
                    tmp_mat.data.chunks_exact_mut(tmp_mat.size[0]).zip(occ_s.iter()).for_each(|(to_value, from_value)| {
                        to_value.iter_mut().for_each(|to_value| {*to_value = *to_value*from_value});
                    });

                    let mut vk_mu = MatrixFull::new([num_basis,num_basis],0.0_f64);
                    // vk_mu = tmp_mat \cdot tmp_mc.T  ((ri3fn \cdot eigv * occ) \cdot (ri3fn \cdot eigv)^T)
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

    //============================debug==================================

    /* let mut data = vec![0.0; vk[0].data.len()];
    for i in 0..vk[0].data.len(){
        data[i] = vk[0].data[i] * 2.0;
    }
    let final_vk = MatrixUpper{size: vk[0].size, data: data};
    &final_vk.formated_output(5, "upper"); */
    //===================================================================

    vk
}
//=============================================================================

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
    pub energy_records: Vec<f64>,
    pub prev_hamiltonian: Vec<[MatrixUpper<f64>;2]>,
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
            energy_records: vec![],
            prev_hamiltonian: vec![[MatrixUpper::empty(),MatrixUpper::empty()]],
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
            let mut alpha = self.mix_param;
            let mut beta = 1.0-alpha;
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
            let start_dim = 0usize;
            let mut start_check_oscillation = scf.mol.ctrl.start_check_oscillation;
            //
            // prepare the fock matrix according to the output density matrix of the previous step
            //
            let dt1 = time::Local::now();

            scf.generate_hf_hamiltonian();


            // update the energy records and check the oscillation
            self.energy_records.push(scf.scf_energy);
            let num_step = self.energy_records.len();
            let oscillation_flag = if num_step >=2 {
                let change_1 = self.energy_records[num_step-1] - self.energy_records[num_step-2];
                num_step > start_check_oscillation && change_1 > 0.0
                //false
            }else {
                false
            };

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


            // solve the DIIS against the error vector
            if let Some(coeff) = diis_solver(&self.error_vector, &self.error_vector.len()) {
                // now extrapolate the fock matrix for the next step
                (0..spin_channel).into_iter().for_each(|i_spin| {
                    let mut next_hamiltonian = MatrixFull::new(self.target_vector[0][i_spin].size.clone(),0.0);
                    coeff.iter().enumerate().for_each(|(i,value)| {
                        next_hamiltonian.self_scaled_add(&self.target_vector[i+start_dim][i_spin], *value);
                    });
                    let next_hamiltonian = next_hamiltonian.to_matrixupper();

                    if oscillation_flag {
                        println!("Energy increase is detected. Turn on the linear mixing algorithm with (H[DIIS, i-1] + H[DIIS, i+1]).");
                        let length = self.energy_records.len();
                        println!("Prev_Energies: ({:16.8}, {:16.8})", self.energy_records[length-2], self.energy_records[length-1]);
                        let mut alpha: f64 = self.mix_param;
                        let mut beta = 1.0-alpha;
                        scf.hamiltonian[i_spin].data.par_iter_mut().zip(self.prev_hamiltonian[0][i_spin].data.par_iter()).zip(next_hamiltonian.data.par_iter())
                        .for_each(|((to, prev), new)| {
                            *to = prev*beta + new*alpha;
                        });
                    } else {
                        scf.hamiltonian[i_spin] = next_hamiltonian;
                    }
                });

                // update the previous hamiltonian list to make sure the first item is H[DIIS, i-1]
                // and the second term is H[DIIS, i]
                if self.prev_hamiltonian.len() == 2 {self.prev_hamiltonian.remove(0);};
                self.prev_hamiltonian.push(scf.hamiltonian.clone());



            } else {
                let mut alpha = self.mix_param;
                let mut beta = 1.0-alpha;
                println!("WARNING: fail to obtain the DIIS coefficients. Turn to use the linear mixing algorithm, and re-invoke DIIS  8 steps later");
                for i_spin in (0..spin_channel) {
                    let residual_dm = self.density_matrix[1][i_spin].sub(&self.density_matrix[0][i_spin]).unwrap();
                    scf.density_matrix[i_spin] = self.density_matrix[0][i_spin]
                        .scaled_add(&residual_dm, alpha)
                        .unwrap();
                }
                scf.generate_hf_hamiltonian();
                //let index = self.target_vector.len()-1;
                //self.target_vector.remove(index);
                //self.error_vector.remove(index);
                self.start_diis_cycle = self.num_iter + 8;
                self.target_vector =  Vec::<[MatrixFull<f64>;2]>::new();
                self.error_vector =  Vec::<Vec::<f64>>::new();
            }
            let dt3 = time::Local::now();
            let timecost1 = (dt2.timestamp_millis()-dt1.timestamp_millis()) as f64 /1000.0;
            let timecost2 = (dt3.timestamp_millis()-dt2.timestamp_millis()) as f64 /1000.0;
            if scf.mol.ctrl.print_level>2 {
                println!("Hamiltonian: generation by {:10.2}s and DIIS extrapolation by {:10.2}s", timecost1,timecost2);
            }
            
        };

        // now consider if level_shift is applied
        // at present only a constant level shift is implemented for both spin channels and for the whole SCF procedure
        if let Some(level_shift) = scf.mol.ctrl.level_shift {

            if scf.mol.spin_channel == 1 {
                let mut fock = scf.hamiltonian.get_mut(0).unwrap();
                let ovlp = &scf.ovlp;
                let dm = scf.density_matrix.get(0).unwrap();
                let dm_scaling_factor = 0.5;
                level_shift_fock(fock, ovlp, level_shift, dm, dm_scaling_factor)
            } else {
                for i_spin in 0..scf.mol.spin_channel {
                    let mut fock = scf.hamiltonian.get_mut(i_spin).unwrap();
                    let ovlp = &scf.ovlp;
                    let dm = scf.density_matrix.get(i_spin).unwrap();
                    let dm_scaling_factor = 1.0;
                    level_shift_fock(fock, ovlp, level_shift, dm, dm_scaling_factor)
                }
            }
        }
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
                norm += dd
            });
            //println!("diis-norm : {:16.8}",norm.sqrt());

            ([cur_error[0].data.clone(),cur_error[1].data.clone()].concat(),
            cur_target)

}

pub fn diis_solver(em: &Vec<Vec<f64>>,
                   num_vec:&usize) -> Option<Vec<f64>> {

    let dim_vec = em.len();
    let start_dim = if (em.len()>=*num_vec) {em.len()-*num_vec} else {0};
    let dim = if (em.len()>=*num_vec) {*num_vec} else {em.len()};
    let mut coeff = Vec::<f64>::new();
    //let mut norm_rdm = [Vec::<f64>::new(),Vec::<f64>::new()];
    let mut odm = MatrixFull::new([1,1],0.0);
    //let mut inv_opta = MatrixFull::new([dim,dim],0.0);
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
    //println!("debug");
    let inv_opta = if let Some(inv_opta) = _dinverse(&mut opta) {
        //println!("diis_solver: _dinverse");
        inv_opta
    //} else if let Some(inv_opta) = opta.lapack_power(-1.0, INVERSE_THRESHOLD) {
    //    println!("diis_solver: lapack_power");
    //    inv_opta
    } else {
        //println!("diis_solver: none");
        return None
    };
    sum_inv_norm_rdm = inv_opta.data.iter().sum::<f64>().powf(-1.0f64);

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

    Some(coeff)

}

pub fn scf(mol:Molecule) -> anyhow::Result<SCF> {
    let dt0 = time::Local::now();

    let mut scf_data = SCF::build(mol);

    scf_without_build(&mut scf_data);

    let dt2 = time::Local::now();
    if scf_data.mol.ctrl.print_level>0 {
        println!("the job spends {:16.2} seconds",(dt2.timestamp_millis()-dt0.timestamp_millis()) as f64 /1000.0)
    };

    Ok(scf_data)
}

fn ao2mo_rayon<'a, T, P>(eigenvector: &T, rimat_chunk: &P, row_dim: std::ops::Range<usize>, column_dim: std::ops::Range<usize>)
-> anyhow::Result<(RIFull<f64>, std::ops::Range<usize>, std::ops::Range<usize>)>
    where T: BasicMatrix<'a, f64>+std::marker::Sync,
          P: BasicMatrix<'a, f64>
{
    ao2mo_rayon_v02(eigenvector, rimat_chunk, row_dim, column_dim)
}

fn ao2mo_rayon_v01<'a, T, P>(eigenvector: &T, rimat_chunk: &P, row_dim: std::ops::Range<usize>, column_dim: std::ops::Range<usize>)
-> anyhow::Result<(RIFull<f64>, std::ops::Range<usize>, std::ops::Range<usize>)>
    where T: BasicMatrix<'a, f64>+std::marker::Sync,
          P: BasicMatrix<'a, f64>
{
    // In this subroutine, we call the lapack dgemm in a rayon parallel environment.
    // In order to ensure the efficiency, we disable the openmp ability and re-open it in the end of subroutien
    let default_omp_num_threads = utilities::omp_get_num_threads_wrapper();
    utilities::omp_set_num_threads_wrapper(1);

    let num_basis = eigenvector.size()[0];
    let num_state = eigenvector.size()[1];
    let num_bpair = rimat_chunk.size()[0];
    let num_auxbs = rimat_chunk.size()[1];
    let num_loc_row = row_dim.len();
    let num_loc_col = column_dim.len();
    let mut rimo = RIFull::new([num_auxbs, num_loc_row, num_loc_col],0.0);
    let (sender, receiver) = channel();

    rimat_chunk.data_ref().unwrap().par_chunks_exact(num_bpair).enumerate().for_each_with(sender, |s, (i_auxbs, m)| {
        let mut loc_ri3mo = MatrixFull::new([row_dim.len(), column_dim.len()],0.0_f64);
        let mut reduced_ri = MatrixFull::new([num_basis, num_basis], 0.0_f64);
        reduced_ri.iter_matrixupper_mut().unwrap().zip(m.iter()).for_each(|(to, from)| {*to = *from});

        let mut tmp_mat = MatrixFull::new([num_basis,num_state], 0.0_f64);
        _dsymm(&reduced_ri, eigenvector, &mut tmp_mat, 'L', 'U', 1.0, 0.0);

        _dgemm(
            &tmp_mat, ((0..num_basis),row_dim.clone()), 'T',
            eigenvector, ((0..num_basis),column_dim.clone()), 'N',
            &mut loc_ri3mo, (0..row_dim.len(), 0..column_dim.len()),
            1.0, 0.0
        );
        s.send((loc_ri3mo, i_auxbs)).unwrap()
    });
    receiver.into_iter().for_each(|(loc_ri3mo, i_auxbs)| {
        rimo.copy_from_matr(0..num_loc_row, 0..num_loc_col, i_auxbs, 2, &loc_ri3mo, 0..num_loc_row, 0..num_loc_col)
    });

    utilities::omp_set_num_threads_wrapper(default_omp_num_threads);

    Ok((rimo, row_dim, column_dim))
}

fn ao2mo_rayon_v02<'a, T, P>(eigenvector: &T, rimat_chunk: &P, row_dim: std::ops::Range<usize>, column_dim: std::ops::Range<usize>)
-> anyhow::Result<(RIFull<f64>, std::ops::Range<usize>, std::ops::Range<usize>)>
    where T: BasicMatrix<'a, f64>+std::marker::Sync,
          P: BasicMatrix<'a, f64>
{
    // In this subroutine, we call the lapack dgemm in a rayon parallel environment.
    // In order to ensure the efficiency, we disable the openmp ability and re-open it in the end of subroutien
    let default_omp_num_threads = utilities::omp_get_num_threads_wrapper();
    utilities::omp_set_num_threads_wrapper(1);

    let num_basis = eigenvector.size()[0];
    let num_state = eigenvector.size()[1];
    let num_bpair = rimat_chunk.size()[0];
    let num_auxbs = rimat_chunk.size()[1];
    let num_loc_row = row_dim.len();
    let num_loc_col = column_dim.len();
    let mut rimo = RIFull::new([num_auxbs, num_loc_row, num_loc_col],0.0);
    let (sender, receiver) = channel();

    rimat_chunk.data_ref().unwrap().par_chunks_exact(num_bpair).enumerate().for_each_with(sender, |s, (i_auxbs, m)| {
        let mut loc_ri3mo = MatrixFull::new([row_dim.len(), column_dim.len()],0.0_f64);
        let mut reduced_ri = MatrixFull::new([num_basis, num_basis], 0.0_f64);
        reduced_ri.iter_matrixupper_mut().unwrap().zip(m.iter()).for_each(|(to, from)| {*to = *from});

        let mut tmp_mat = MatrixFull::new([num_basis,num_state], 0.0_f64);
        _dsymm(&reduced_ri, eigenvector, &mut tmp_mat, 'L', 'U', 1.0, 0.0);

        _dgemm(
            &tmp_mat, ((0..num_basis),row_dim.clone()), 'T',
            eigenvector, ((0..num_basis),column_dim.clone()), 'N',
            &mut loc_ri3mo, (0..row_dim.len(), 0..column_dim.len()),
            1.0, 0.0
        );
        s.send((loc_ri3mo, i_auxbs)).unwrap()
    });
    receiver.into_iter().for_each(|(loc_ri3mo, i_auxbs)| {
        rimo.copy_from_matr(0..num_loc_row, 0..num_loc_col, i_auxbs, 2, &loc_ri3mo, 0..num_loc_row, 0..num_loc_col)
    });

    utilities::omp_set_num_threads_wrapper(default_omp_num_threads);

    Ok((rimo, row_dim, column_dim))
}

pub fn level_shift_fock(fock: &mut MatrixUpper<f64>, ovlp: &MatrixUpper<f64>, level_shift: f64,  dm: &MatrixFull<f64>, dm_scaling_factor: f64) {
    // FC = SCE
    // F' = F + SC \Lambda C^\dagger S
    // F' = F + LF * (S - SDS) 

    let num_basis = dm.size()[0];

    let mut tmp_s = MatrixFull::new([num_basis, num_basis], 0.0);
    tmp_s.iter_matrixupper_mut().unwrap().zip(ovlp.data.iter()).for_each(|(to, from)| {*to = *from});
    let mut tmp_s2 = tmp_s.clone();
    let mut tmp_s3 = tmp_s.clone();
    _dsymm(&tmp_s, dm, &mut tmp_s2, 'L', 'U', -dm_scaling_factor, 0.0);
    _dsymm(&tmp_s, &mut tmp_s2, &mut tmp_s3, 'R', 'U', 1.0, 1.0);
    fock.data.iter_mut().zip(tmp_s3.iter_matrixupper().unwrap()).for_each(|(to, from)| {*to += *from*level_shift});
}

pub fn diagonalize_hamiltonian_outside(scf_data: &SCF) -> ([MatrixFull<f64>;2], [Vec<f64>;2]) {
    let spin_channel = scf_data.mol.spin_channel;
    let num_state = scf_data.mol.num_state;
    let dt1 = time::Local::now();

    let mut eigenvectors = [MatrixFull::empty(),MatrixFull::empty()];
    let mut eigenvalues = [Vec::new(),Vec::new()];
    match scf_data.scftype {
        SCFType::ROHF => {
            let (eigenvector_spin, eigenvalue_spin)=
                _dspgvx(&scf_data.hamiltonian[0], &scf_data.ovlp, num_state).unwrap();
                //self.hamiltonian[0].to_matrixupperslicemut()
                //.lapack_dspgvx(self.ovlp.to_matrixupperslicemut(),num_state).unwrap();
            //self.eigenvectors[0] = eigenvector_spin;
            //self.eigenvalues[0] = eigenvalue_spin;
            //self.eigenvectors[1] = self.eigenvectors[0].clone();
            //self.eigenvalues[1] = self.eigenvalues[0].clone();
            eigenvectors[0] = eigenvector_spin;
            eigenvectors[1] = eigenvectors[0].clone();
            eigenvalues[0] = eigenvalue_spin;
            eigenvalues[1] = eigenvalues[0].clone();
        },
        _ => {
            for i_spin in (0..spin_channel) {
                let (eigenvector_spin, eigenvalue_spin)=
                    _dspgvx(&scf_data.hamiltonian[0], &scf_data.ovlp, num_state).unwrap();
                    //self.hamiltonian[i_spin].to_matrixupperslicemut()
                    //.lapack_dspgvx(self.ovlp.to_matrixupperslicemut(),num_state).unwrap();
                eigenvectors[i_spin] = eigenvector_spin;
                eigenvalues[i_spin] = eigenvalue_spin;
            }
        }
    }

    (eigenvectors,eigenvalues)
}

pub fn generate_occupation_outside(scf_data: &SCF) -> ([Vec<f64>;2], [usize;2], [usize;2]) {
    let mut occ = [vec![],vec![]];
    let mut homo = [0,0];
    let mut lumo = [0,0];
    match scf_data.mol.ctrl.occupation_type {
        OCCType::INTEGER => {
            //println!("debug generate_occupation_integer");
            (occ,homo,lumo) = generate_occupation_integer(&scf_data.mol,&scf_data.scftype);
            //self.occupation = occ;
            //self.homo = homo;
            //self.lumo = lumo;
        },
        OCCType::ATMSAD => {
            //println!("debug generate_occupation_sad");
            (occ,homo,lumo) = generate_occupation_sad(scf_data.mol.geom.elem.get(0).unwrap(),scf_data.mol.num_state, scf_data.mol.ecp_electrons);
            //self.occupation = occ;
            //self.homo = homo;
            //self.lumo = lumo;
        },
        OCCType::FRAC => {
            //println!("debug generate_occupation_frac");
            (occ,homo,lumo) = generate_occupation_frac_occ(&scf_data.mol,&scf_data.scftype, &scf_data.eigenvalues, scf_data.mol.ctrl.frac_tolerant);
            //self.occupation = occ;
            //self.homo = homo;
            //self.lumo = lumo;
        }
    }
    if scf_data.mol.ctrl.print_level>3 {
        println!("Occupation in Alpha Channel: {:?}", &occ[0]);
        if scf_data.mol.spin_channel == 2{
            println!("Occupation in Beta Channel:  {:?}", &occ[1]);
        }
    }

    (occ, homo, lumo)
}

pub fn generate_density_matrix_outside(scf_data: &SCF) -> Vec<MatrixFull<f64>>{

    let num_basis = scf_data.mol.num_basis;
    let num_state = scf_data.mol.num_state;
    let spin_channel = scf_data.mol.spin_channel;
    let homo = &scf_data.homo;
    let mut dm = vec![
        MatrixFull::empty(),
        MatrixFull::empty()
        ];
    (0..spin_channel).into_iter().for_each(|i_spin| {
        //println!("debug density_matrix spin: {}",i_spin);
        let mut dm_s = &mut dm[i_spin];
        *dm_s = MatrixFull::new([num_basis,num_basis],0.0);
        let eigv_s = &scf_data.eigenvectors[i_spin];
        let occ_s =  &scf_data.occupation[i_spin];

        let nw =  scf_data.homo[i_spin]+1;
        //println!("number of occupied orbitals from dm generation: {}", nw);

        let mut weight_eigv = MatrixFull::new([num_basis, num_state],0.0_f64);
        //let mut weight_eigv = eigv_s.clone();
        weight_eigv.par_iter_columns_mut(0..nw).unwrap().zip(eigv_s.par_iter_columns(0..nw).unwrap())
            .for_each(|value| {
                value.0.into_iter().zip(value.1.into_iter()).for_each(|value| {
                    *value.0 = *value.1
                })
            });
        // prepare weighted eigenvalue matrix wC
        //println!("debug: {:?}", nw);
        //println!("debug: {:?}", &occ_s);
        weight_eigv.par_iter_columns_mut(0..nw).unwrap().zip(occ_s[0..nw].par_iter()).for_each(|(we,occ)| {
        //weight_eigv.data.chunks_exact_mut(weight_eigv.size[0]).zip(occ_s.iter()).for_each(|(we,occ)| {
            we.iter_mut().for_each(|c| *c = *c*occ);
        });

        // dm = wC*C^{T}
        _dgemm_full(&weight_eigv,'N',eigv_s, 'T',dm_s, 1.0, 0.0);
        //dm_s.lapack_dgemm(&mut weight_eigv, eigv_s, 'N', 'T', 1.0, 0.0);
        //dm_s.formated_output(5, "full");
    });
    //if let SCFType::ROHF = scf_data.scftype {dm[1]=dm[0].clone()};
    //scf_data.density_matrix = dm;

    dm

}

pub fn initialize_scf(scf_data: &mut SCF) {

    // update the corresponding geometry information, which is crucial 
    // for preparing the following integrals accurately
    let position = &scf_data.mol.geom.position;
    scf_data.mol.cint_env = scf_data.mol.update_geom_poisition_in_cint_env(position);

    let mut time_mark = utilities::TimeRecords::new();
    time_mark.new_item("Overall", "SCF Preparation");
    time_mark.count_start("Overall");

    time_mark.new_item("CInt", "Two, Three, and Four-center integrals");
    time_mark.count_start("CInt");
    scf_data.prepare_necessary_integrals();
    time_mark.count("CInt");


    time_mark.new_item("DFT Grids", "Initialization of the tabulated Grids and AOs");
    time_mark.count_start("DFT Grids");
    scf_data.prepare_density_grids();
    time_mark.count("DFT Grids");

    time_mark.new_item("ISDF", "ISDF initialization");
    time_mark.count_start("ISDF");
    scf_data.prepare_isdf();
    time_mark.count("ISDF");

    time_mark.new_item("InitGuess", "Prepare initial guess");
    time_mark.count_start("InitGuess");
    initial_guess(scf_data);
    if ! scf_data.mol.ctrl.atom_sad && scf_data.mol.ctrl.print_level>4 {
        println!("Initial density matrix by Atom SAD:");
        scf_data.density_matrix[0].formated_output(5, "full");
    }
    time_mark.count("InitGuess");

    time_mark.count("Overall");
    if scf_data.mol.ctrl.print_level>=2 {
        time_mark.report_all();
    }

}

pub fn scf_without_build(scf_data: &mut SCF) {
    scf_data.generate_hf_hamiltonian();

    let mut scf_records=ScfTraceRecord::initialize(&scf_data);

    if scf_data.mol.ctrl.print_level>0 {println!("The total energy: {:20.10} Ha by the initial guess",scf_data.scf_energy)};
    //let mut scf_continue = true;
    if scf_data.mol.ctrl.noiter {
        println!("Warning: the SCF iteration is skipped!");
        return;
    }

    // now prepare the input density matrix for the first iteration and initialize the records
    scf_data.diagonalize_hamiltonian();
    scf_data.generate_occupation();
    scf_data.generate_density_matrix();
    scf_records.update(&scf_data);

    let mut scf_converge = [false;2];
    while ! (scf_converge[0] || scf_converge[1]) {
        let dt1 = time::Local::now();

        scf_records.prepare_next_input(scf_data);

        let dt1_1 = time::Local::now();

        scf_data.diagonalize_hamiltonian();
        let dt1_2 = time::Local::now();
        scf_data.generate_occupation();
        scf_data.generate_density_matrix();
        let dt1_3 = time::Local::now();
        scf_converge = scf_data.check_scf_convergence(&scf_records);
        let dt1_4 = time::Local::now();
        scf_records.update(&scf_data);
        let dt1_5 = time::Local::now();


        let dt2 = time::Local::now();
        let timecost = (dt2.timestamp_millis()-dt1.timestamp_millis()) as f64 /1000.0;
        if scf_data.mol.ctrl.print_level>0 {println!("Energy: {:18.10} Ha after {:4} iterations (in {:10.2} seconds).",
                 scf_records.scf_energy,
                 scf_records.num_iter-1,
                 timecost)};
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
        if scf_data.mol.ctrl.print_level>0 {println!("SCF is converged after {:4} iterations.", scf_records.num_iter-1)};
        if scf_data.mol.ctrl.print_level>1 {
            scf_data.formated_eigenvalues((scf_data.homo.iter().max().unwrap()+4).min(scf_data.mol.num_state));
        }
        if scf_data.mol.ctrl.print_level>3 {
            scf_data.formated_eigenvectors();
        }
        match scf_data.mol.ctrl.occupation_type {
            OCCType::FRAC => {
                //println!("debug: final energy evaluation after occupation_type = frac");
                let (occupation, homo, lumo) = check_norm::generate_occupation_integer(&scf_data.mol, &scf_data.scftype);
                scf_data.occupation = occupation;
                scf_data.homo = homo;
                scf_data.lumo = lumo;
                scf_data.generate_density_matrix();
                scf_data.generate_hf_hamiltonian();

            }
            _ => {}
        }
        // not yet implemented. Just an empty subroutine
    } else {
        //if scf_data.mol.ctrl.restart {save_chkfile(&scf_data)};
        println!("SCF does not converge within {:03} iterations",scf_records.num_iter);
    }

}



#[test]
fn test_max() {
    println!("{}, {}, {}",1,2,std::cmp::max(1, 2));
}
