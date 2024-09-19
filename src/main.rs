//! # Rust-based Electronic-Structure Tool (REST)
//! 
//! ### Installation
//!   At present, REST can be compiled in Linux only  
//! 
//!   0) Prerequisites:  
//!     - libopenblas.so  
//!     - libcint.so  
//!     - libhdf5.so  
//!     - libxc.so  
//!   1) git clone git@github.com:igor-1982/rest_workspace.git rest_workspace  
//!   2) cd rest_workspace; cp Config.templet Config
//!   3) edit `Config` to make the prerequeisite libraries aformationed accessable via some global variables heading with `REST`.  
//!   4) bash Config; source $HOME/.bash_profile
//!   5-1) cargo build (--release) 
//! 
//! ### Usage
//!   - Some examples are provided in the folder of `rest/examples/`.  
//!   - Detailed user manual is in preparation.  
//!   - Basic usage of varying keywords can be found on the page of [`InputKeywords`](crate::ctrl_io::InputKeywords).
//! 
//! ### Features
//!   1) Use Gaussian Type Orbital (GTO) basis sets  
//!   2) Provide Density Functional Approximations (DFAs) at varying levels from LDA, GGA, Hybrid, to Fifth-rungs, including doubly hybrid approximations, like XYG3, XYGJ-OS, and random-phase approximation (RPA).  
//!   3) Provide some Wave Function Methods (WFMs), like Hartree-Fock approximation (HF) and Moller-Plesset Second-order perturbation (MP2)
//!   4) Provide analytic electronic-repulsive integrals (ERI)s as well as the Resolution-of-idensity (RI) approximation. The RI algorithm is the recommended choice.  
//!   5) High Share Memory Parallelism (SMP) efficiency
//! 
//! 
//! ### Development
//!   1) Provide a tensor library, namely [`rest_tensors`](https://igor-1982.github.io/rest_tensors/rest_tensors/). `rest_tensors` is developed to manipulate
//!    different kinds multi-rank arrays in REST. Thanks to the sophisticated generic, type, and trait systems, `rest_tensors` can be used as easy as `Numpy` and `Scipy` without losing the computation efficiency. 
//!   2) It is very easy and save to develop a parallel program using the Rust language. Please refer to [`rayon`](rayon) for more details.
//!   3) However, attention should pay if you want to use (Sca)Lapcke functions together in the rayon-spawn threads. 
//!    It is because the (Sca)Lapack functions, like `dgemm`, were compiled with OpenMP by default.  The competetion between OpenMP and Rayon threads 
//!    could dramatically deteriorate the final performance.  
//!    Please use [`utilities::omp_set_num_threads_wrapper`] to set the OpenMP treads number in the runtime.
//! 
//! 
//! 
//! ### Presentation
//! ![image](/home/igor/Documents/Package-Pool/rest_workspace/rest/figures/REST电子结构程序1.png) 
//! ![image](/home/igor/Documents/Package-Pool/rest_workspace/rest/figures/REST电子结构程序2.png) 
//! ![image](/home/igor/Documents/Package-Pool/rest_workspace/rest/figures/REST电子结构程序3.png) 
//! ![image](/home/igor/Documents/Package-Pool/rest_workspace/rest/figures/REST电子结构程序4.png) 
//! ![image](/home/igor/Documents/Package-Pool/rest_workspace/rest/figures/REST电子结构程序5-2.png) 
//! ![image](/home/igor/Documents/Package-Pool/rest_workspace/rest/figures/REST电子结构程序6-2.png) 
//! 
#![allow(unused)]
extern crate rest_tensors as tensors;
//extern crate rest_libxc as libxc;
extern crate chrono as time;
#[macro_use]
extern crate lazy_static;
use std::{f64, fs::File, io::Write};
use std::path::PathBuf;
use basis_io::ecp::ghost_effective_potential_matrix;
use num_traits::Pow;
use pyo3::prelude::*;
use autocxx::prelude::*;
use ctrl_io::JobType;
use constants::ANG;
use scf_io::{SCF,scf_without_build};
use tensors::{MathMatrix, MatrixFull};

mod geom_io;
mod basis_io;
mod ctrl_io;
mod grad;
mod dft;
mod utilities;
mod molecule_io;
mod scf_io;
mod initial_guess;
mod check_norm;
mod ri_pt2;
//mod grad;
mod ri_rpa;
mod isdf;
mod constants;
mod post_scf_analysis;
mod external_libs;
//use rayon;
#[global_allocator]
static GLOBAL: jemallocator::Jemalloc = jemallocator::Jemalloc;
use crate::constants::EV;
use crate::grad::{formated_force, numerical_force};
use crate::initial_guess::enxc::{effective_nxc_matrix, effective_nxc_tensors};
//static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;
//use crate::grad::rhf::Gradient;
use crate::initial_guess::sap::*;

use anyhow;
//use crate::isdf::error_isdf;
use crate::dft::DFA4REST;
use crate::post_scf_analysis::mulliken::mulliken_pop;
//use crate::post_scf_analysis::{post_scf_correlation, print_out_dfa, save_chkfile};
use crate::scf_io::{initialize_scf, scf};
use time::{DateTime,Local};
use crate::molecule_io::Molecule;
//use crate::isdf::error_isdf;
//use crate::dft::DFA4REST;
use crate::post_scf_analysis::{post_scf_correlation, print_out_dfa, save_chkfile, rand_wf_real_space, cube_build, molden_build, post_ai_correction};
use liblbfgs::{lbfgs,Progress};

//use autocxx::prelude::*;
//
//include_cpp! {
//    #include "libecpint/api.hpp"
//    #include "run.h"
//    safety!(unsafe)
//    generate!("ECPIntWrapper")
//    generate!("rest_ecpint")
//}

//#[test]
//pub fn test_ecp() -> anyhow::Result<()> {
//    let mut ecpint = ffi::ECPIntWrapper::new("/usr/share/libecpint").within_box();
//    let ints = ecpint.as_mut().get_integrals();
//    println!("length: {}", ints.len());
//    println!("{:?}", ints);
//
//    //let basis_list = 
//    Ok(())
//}

//pub use crate::initial_guess::sap::*;
//use crate::{post_scf_analysis::{rand_wf_real_space, cube_build, molden_build}, isdf::error_isdf, molecule_io::Molecule};

fn main() -> anyhow::Result<()> {

                                  
    let mut time_mark = utilities::TimeRecords::new();
    time_mark.new_item("Overall", "the whole job");
    time_mark.count_start("Overall");


    let ctrl_file = utilities::parse_input().value_of("input_file").unwrap_or("ctrl.in").to_string();
    if ! PathBuf::from(ctrl_file.clone()).is_file() {
        panic!("Input file ({:}) does not exist", ctrl_file);
    }
    let mut mol = Molecule::build(ctrl_file)?;
    if mol.ctrl.print_level>0 {println!("Molecule_name: {}", &mol.geom.name)};
    if mol.ctrl.print_level>=2 {
        println!("{}", mol.ctrl.formated_output_in_toml());
    }

    if mol.ctrl.deep_pot {
        //let mut scf_data = scf_io::SCF::build(&mut mol);
        let mut effective_hamiltonian = mol.int_ij_matrixupper(String::from("hcore"));
        //effective_hamiltonian.formated_output(5, "full");
        let effective_nxc = effective_nxc_matrix(&mut mol);
        effective_nxc.formated_output(5, "full");
        effective_hamiltonian.data.iter_mut().zip(effective_nxc.data.iter()).for_each(|(to,from)| {*to += from});

        let mut ecp = mol.int_ij_matrixupper(String::from("ecp"));
        //println!("ecp: {:?}", ecp);
        ecp.iter_mut().zip(effective_nxc.iter()).for_each(|(to, from)| {*to -= from});

        ecp.formated_output(5, "full");
        let acc_error = ecp.iter().fold(0.0, |acc, x| {acc + x.abs()});
        println!("acc_error: {}", acc_error);
        
        return Ok(())
    }

    if mol.ctrl.bench_eps {
        let ecp = mol.int_ij_matrixupper(String::from("ecp"));
        let enxc = effective_nxc_matrix(&mut mol);
        let gep = ghost_effective_potential_matrix(
            &mol.cint_env, &mol.cint_atm, &mol.cint_bas, &mol.cint_type, mol.num_basis, 
            &mol.geom.ghost_ep_path, &mol.geom.ghost_ep_pos);

        let d12 = ecp.data.iter().zip(enxc.data.iter()).fold(0.0, |acc, dt| {acc + (dt.0 -dt.1).powf(2.0)});
        let d13 = ecp.data.iter().zip(gep.data.iter()).fold(0.0, |acc, dt| {acc + (dt.0 -dt.1).powf(2.0)});
        let num_data = ecp.data.len() as f64;
        println!("Compare between ECP, ENXC and GEP with the matrix sizes of");
        println!(" {:?}, {:?}, and {:?}, respectively", ecp.size(), enxc.size(), gep.size());
        println!("RMSDs between (ECP, ENXC) and (ECP, GEP): ({:16.8}, {:16.8})", 
            (d12/num_data).powf(0.5), (d13/num_data).powf(0.5)
        );

        return Ok(())
    }

    // initialize the SCF procedure
    time_mark.new_item("SCF", "the scf procedure");
    time_mark.count_start("SCF");
    let mut scf_data = scf_io::SCF::build(mol);
    time_mark.count("SCF");
    // perform the SCF and post SCF evaluation for the specified xc method
    performance_essential_calculations(&mut scf_data, &mut time_mark);

    let jobtype = scf_data.mol.ctrl.job_type.clone();
    match jobtype {
        JobType::GeomOpt => {
            let mut geom_time_mark = utilities::TimeRecords::new();
            geom_time_mark.new_item("geom_opt", "geometry optimization");
            geom_time_mark.count_start("geom_opt");
            if scf_data.mol.ctrl.print_level>0 {
                println!("Geometry optimization invoked");
            }
            let displace = 0.0013/ANG;

            //let (energy,nforce) = numerical_force(&scf_data, displace);
            //println!("Total atomic forces [a.u.]: ");
            //nforce.formated_output(5, "full");
            //let mut nnforce = nforce.clone();
            //nnforce.iter_mut().for_each(|x| *x *= ANG/EV);
            //println!("Total atomic forces [EV/Ang]: ");
            //nnforce.formated_output(5, "full");

            let mut position = scf_data.mol.geom.position.iter().map(|x| *x).collect::<Vec<f64>>();
            lbfgs().minimize(
                &mut position, 
                |x: &[f64], gx: &mut [f64]| {
                    scf_data.mol.geom.position = MatrixFull::from_vec([3,x.len()/3], x.to_vec()).unwrap();
                    if scf_data.mol.ctrl.print_level>0 {
                        println!("Input geometry in this round is:");
                        println!("{}", scf_data.mol.geom.formated_geometry());
                    }
                    scf_data.mol.ctrl.initial_guess = String::from("inherit");
                    initialize_scf(&mut scf_data);
                    performance_essential_calculations(&mut scf_data, &mut geom_time_mark);
                    let (energy, nforce) = numerical_force(&scf_data, displace);
                    gx.iter_mut().zip(nforce.iter()).for_each(|(to, from)| {*to = *from});

                    if scf_data.mol.ctrl.print_level>0 {
                        println!("Output force in this round [a.u.] is:");
                        println!("{}", formated_force(&nforce, &scf_data.mol.geom.elem));
                    }

                    Ok(energy)
                },
                |prgr| {
                    println!("Iteration {}, Evaluation: {}", &prgr.niter, &prgr.neval);
                    println!(" xnorm = {}, gnorm = {}, step = {}",
                        &prgr.xnorm, &prgr.gnorm, &prgr.step
                    );
                    false
                },
            );
            println!("Geometry after relaxation [Ang]:");
            println!("{}", scf_data.mol.geom.formated_geometry());
            geom_time_mark.count("geom_opt");

            geom_time_mark.report("geom_opt");

        },
        _ => {}
    }

    //let mut grad_data = Gradient::build(&scf_data.mol, &scf_data);

    //grad_data.calc_j(&scf_data.density_matrix);
    //print!("occ, {:?}", scf_data.occupation);

    //time_mark.count("SCF");

    if scf_data.mol.ctrl.restart {
        println!("now save the converged SCF results");
        save_chkfile(&scf_data)
    };

    if scf_data.mol.ctrl.check_stab {
        time_mark.new_item("Stability", "the scf stability check");
        time_mark.count_start("Stability");

        scf_data.stability();

        time_mark.count("Stability");
    }

    //====================================
    // Now for post-xc calculations
    //====================================
    if scf_data.mol.ctrl.post_xc.len()>=1 {
        print_out_dfa(&scf_data);
    }

    //====================================
    // Now for post-SCF analysis
    //====================================
    if scf_data.mol.ctrl.print_level > 0 {
        let mulliken = mulliken_pop(&scf_data);
        println!("Mulliken population analysis:");
        for (i, (pop, atom)) in mulliken.iter().zip(scf_data.mol.geom.elem.iter()).enumerate() {
            println!("{:3}-{:3}: {:10.6}", i, atom, pop);
        }
    }

    post_scf_analysis::post_scf_output(&scf_data);

    //====================================
    // Now for post-correlation calculations
    //====================================
    if scf_data.mol.ctrl.post_correlation.len()>=1 {
        post_scf_correlation(&mut scf_data);
    }

    time_mark.count("Overall");

    println!("");
    println!("====================================================");
    println!("              REST: Mission accomplished");
    println!("====================================================");

    output_result(&scf_data);


    time_mark.report_all();

    Ok(())
}


pub fn output_result(scf_data: &scf_io::SCF) {
    println!("The SCF energy        : {:18.10} Ha", 
        //scf_data.mol.ctrl.xc.to_uppercase(),
        scf_data.scf_energy);
    
    let xc_name = scf_data.mol.ctrl.xc.to_lowercase();
    if xc_name.eq("mp2") || xc_name.eq("xyg3") || xc_name.eq("xygjos") || xc_name.eq("r-xdh7") || xc_name.eq("xyg7") || xc_name.eq("zrps") || xc_name.eq("scsrpa") {
        let total_energy = scf_data.energies.get("xdh_energy").unwrap()[0];
        let post_ai_correction = scf_data.mol.ctrl.post_ai_correction.to_lowercase();
        //let ai_correction = if xc_name.eq("r-xdh7") && post_ai_correction.eq("scc15") {
        //    let ai_correction = scf_data.energies.get("ai_correction").unwrap()[0];
        //    println!("AI Correction         : {:18.10} Ha", ai_correction);
        //    ai_correction
        //} else {
        //    0.0
        //};
        let ai_correction = if let Some(ai_correction) = scf_data.energies.get("ai_correction") {
            ai_correction[0]
        } else {
            0.0
        };
        println!("The (R)-xDH energy    : {:18.10} Ha", total_energy+ ai_correction);
    }
    if xc_name.eq("rpa@pbe") {
        let total_energy = scf_data.energies.get("rpa_energy").unwrap()[0];
        println!("The RPA energy        : {:18.10} Ha", total_energy);
    }

}

/// Perform key SCF and post-SCF calculations
/// Return the total energy of the specfied xc method
/// Assume the initialization of SCF is ready
pub fn performance_essential_calculations(scf_data: &mut SCF, time_mark: &mut utilities::TimeRecords) -> f64 {

    let mut total_energy = 0.0;

    //=================================================================
    // Now evaluate the SCF energy for the given method
    //=================================================================
    time_mark.count_start("SCF");
    scf_without_build(scf_data);
    time_mark.count("SCF");

    //==================================================================
    // Now evaluate the advanced correction energy for the given method
    //==================================================================
    //let mut time_mark = utilities::TimeRecords::new();
    if let Some(dft_method) = &scf_data.mol.xc_data.dfa_family_pos {
        match dft_method {
            dft::DFAFamily::PT2 | dft::DFAFamily::SBGE2 => {
                time_mark.new_item("PT2", "the PT2 evaluation");
                time_mark.count_start("PT2");
                ri_pt2::xdh_calculations(scf_data);
                time_mark.count("PT2");
            },
            dft::DFAFamily::RPA => {
                time_mark.new_item("RPA", "the RPA evaluation");
                time_mark.count_start("RPA");
                ri_rpa::rpa_calculations(scf_data);
                time_mark.count("RPA");
            }
            dft::DFAFamily::SCSRPA => {
                time_mark.new_item("SCS-RPA", "the SCS-RPA evaluation");
                time_mark.count_start("SCS-RPA");
                ri_pt2::xdh_calculations(scf_data);
                time_mark.count("SCS-RPA");
            }
            _ => {}
        }
    }
    //====================================
    // Now for post ai correction
    //====================================
    if let Some(scc) = post_ai_correction(scf_data) {
        scf_data.energies.insert("ai_correction".to_string(), scc);
    }

    collect_total_energy(scf_data)

}

pub fn collect_total_energy(scf_data: &SCF) -> f64 {
    //====================================
    // Determine the total energy
    //====================================
    let mut total_energy = scf_data.scf_energy;
    
    let xc_name = scf_data.mol.ctrl.xc.to_lowercase();
    if xc_name.eq("mp2") || xc_name.eq("xyg3") || xc_name.eq("xygjos") || xc_name.eq("r-xdh7") || xc_name.eq("xyg7") || xc_name.eq("zrps") || xc_name.eq("scsrpa") {
        total_energy = scf_data.energies.get("xdh_energy").unwrap()[0];
    } else if xc_name.eq("rpa@pbe") {
        total_energy = scf_data.energies.get("rpa_energy").unwrap()[0];
    }
    if let Some(post_ai_correction) = scf_data.energies.get("ai_correction") {
        total_energy += post_ai_correction[0]
    };

    total_energy

}
