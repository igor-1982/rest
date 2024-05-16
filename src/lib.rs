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
extern crate chrono as time;
extern crate anyhow;
extern crate lazy_static;
pub mod basis_io;
pub mod constants;
pub mod check_norm;
pub mod ctrl_io;
pub mod dft;
pub mod geom_io;
pub mod initial_guess;
pub mod isdf;
pub mod molecule_io;
pub mod scf_io;
pub mod utilities;
pub mod external_libs;

//extern crate rest;

use initial_guess::enxc::ENXC;
use pyo3::prelude::*;
use tensors::MatrixUpper;

use crate::initial_guess::enxc::PotCell;
use crate::{molecule_io::Molecule, scf_io::SCF, scf_io::scf};
use crate::geom_io::GeomCell;
use crate::ctrl_io::InputKeywords;

#[pyfunction]
fn read_ctrl(ctrlfile: String) -> PyResult<Molecule> {
    Ok(Molecule::build(ctrlfile).unwrap())
}
#[pyfunction]
fn do_scf(mol: Molecule) -> PyResult<SCF> {
    Ok(scf(mol).unwrap())
}



#[pymodule]
#[pyo3(name = "pyrest")]
fn pyrest(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(read_ctrl, m)?)?;
    m.add_function(wrap_pyfunction!(do_scf, m)?)?;
    m.add_class::<Molecule>()?;
    m.add_class::<SCF>()?;
    m.add_class::<GeomCell>()?;
    m.add_class::<InputKeywords>()?;

    // ==================================================
    // add functions for effective potential
    // ==================================================
    m.add_class::<ENXC>()?;
    m.add_class::<PotCell>()?;
    m.add_function(wrap_pyfunction!(effective_nxc_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(effective_nxc_tensors, m)?)?;
    m.add_function(wrap_pyfunction!(effective_nxc_pymatrix, m)?)?;
    m.add_function(wrap_pyfunction!(parse_enxc_potential, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_derive_enxc, m)?)?;
    
    Ok(())
}


#[pyfunction]
fn effective_nxc_pymatrix(mut mol: Molecule, exnc: Vec<ENXC>) -> PyResult<Vec<f64>> {
    let output_matrix = crate::initial_guess::enxc::effective_nxc_matrix_v02(&mut mol, &exnc);
    Ok(output_matrix.data)
}

#[pyfunction]
fn parse_enxc_potential(file_name: String) -> PyResult<ENXC> {
    let output_enxc = crate::initial_guess::enxc::parse_enxc_potential(&file_name[..]).unwrap();
    Ok(output_enxc)
}

#[pyfunction]
fn effective_nxc_matrix(mut mol: Molecule) -> PyResult<Vec<f64>> {
    let output_matrix = crate::initial_guess::enxc::effective_nxc_matrix(&mut mol);
    Ok(output_matrix.data)
}

#[pyfunction]
fn effective_nxc_tensors(mut mol: Molecule) -> PyResult<Vec<f64>> {
    let output_matrix = crate::initial_guess::enxc::effective_nxc_tensors(&mut mol);
    Ok(output_matrix.data)
}

#[pyfunction]
fn evaluate_derive_enxc(mut mol: Molecule, enxc: Vec<ENXC>, atm_index: usize, coeff_index: usize) -> PyResult<Vec<f64>> {
    let output_matrix = crate::initial_guess::enxc::evaluate_derive_enxc(&mut mol, &enxc, atm_index, coeff_index);
    Ok(output_matrix.data)
}