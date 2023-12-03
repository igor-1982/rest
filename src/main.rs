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
use pyo3::prelude::*;

mod geom_io;
mod basis_io;
mod ctrl_io;
mod dft;
mod utilities;
mod molecule_io;
mod scf_io;
mod initial_guess;
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
//static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;
//use crate::grad::rhf::Gradient;
use crate::initial_guess::sap::*;

use anyhow;
//use crate::isdf::error_isdf;
use crate::dft::DFA4REST;
use crate::post_scf_analysis::mulliken::mulliken_pop;
//use crate::post_scf_analysis::{post_scf_correlation, print_out_dfa, save_chkfile};
use crate::scf_io::scf;
use time::{DateTime,Local};
use crate::molecule_io::Molecule;
//use crate::isdf::error_isdf;
//use crate::dft::DFA4REST;
use crate::post_scf_analysis::{post_scf_correlation, print_out_dfa, save_chkfile, rand_wf_real_space, cube_build, molden_build, post_ai_correction};


//pub use crate::initial_guess::sap::*;
//use crate::{post_scf_analysis::{rand_wf_real_space, cube_build, molden_build}, isdf::error_isdf, molecule_io::Molecule};

fn main() -> anyhow::Result<()> {
    let mut time_mark = utilities::TimeRecords::new();
    time_mark.new_item("Overall", "the whole job");
    time_mark.count_start("Overall");

    time_mark.new_item("SCF", "the scf procedure");
    time_mark.count_start("SCF");

    let ctrl_file = utilities::parse_input().value_of("input_file").unwrap_or("ctrl.in").to_string();
    if ! PathBuf::from(ctrl_file.clone()).is_file() {
        panic!("Input file ({:}) does not exist", ctrl_file);
    }


    let mut mol = Molecule::build(ctrl_file)?;
    println!("Molecule_name: {}", &mol.geom.name);

    let mut scf_data = scf_io::scf(mol).unwrap();

    //let mut grad_data = Gradient::build(&scf_data.mol, &scf_data);

    //grad_data.calc_j(&scf_data.density_matrix);
    //print!("occ, {:?}", scf_data.occupation);

    time_mark.count("SCF");

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
    let mulliken = mulliken_pop(&scf_data);
    println!("Mulliken population analysis:");
    for (i, (pop, atom)) in mulliken.iter().zip(scf_data.mol.geom.elem.iter()).enumerate() {
        println!("{:3}-{:3}: {:10.6}", i, atom, pop);
    }

    post_scf_analysis::post_scf_output(&scf_data);

    //let error_isdf = error_isdf(12..20, &scf_data);
    //println!("k_mu:{:?}, abs_error: {:?}, rel_error: {:?}", error_isdf.0, error_isdf.1, error_isdf.2);

    if let Some(dft_method) = &scf_data.mol.xc_data.dfa_family_pos {
        match dft_method {
            dft::DFAFamily::PT2 | dft::DFAFamily::SBGE2 => {
                time_mark.new_item("PT2", "the PT2 evaluation");
                time_mark.count_start("PT2");
                ri_pt2::xdh_calculations(&mut scf_data);
                time_mark.count("PT2");
            },
            dft::DFAFamily::RPA => {
                time_mark.new_item("RPA", "the RPA evaluation");
                time_mark.count_start("RPA");
                ri_rpa::rpa_calculations(&mut scf_data);
                time_mark.count("RPA");
            }
            dft::DFAFamily::SCSRPA => {
                time_mark.new_item("SCS-RPA", "the SCS-RPA evaluation");
                time_mark.count_start("SCS-RPA");
                ri_pt2::xdh_calculations(&mut scf_data);
                time_mark.count("SCS-RPA");
            }
            _ => {}
        }
    }
    //====================================
    // Now for post ai correction
    //====================================
    if let Some(scc) = post_ai_correction(&mut scf_data) {
        scf_data.energies.insert("ai_correction".to_string(), scc);
    }

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
        let ai_correction = if xc_name.eq("r-xdh7") && post_ai_correction.eq("scc23") {
            let ai_correction = scf_data.energies.get("ai_correction").unwrap()[0];
            println!("AI Correction         : {:18.10} Ha", ai_correction);
            ai_correction
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


