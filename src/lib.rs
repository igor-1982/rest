#![allow(unused)]
extern crate rest_tensors as tensors;
extern crate chrono as time;
extern crate anyhow;
extern crate lazy_static;
pub mod basis_io;
pub mod constants;
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

use pyo3::prelude::*;

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
fn pyrest(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(read_ctrl, m)?)?;
    m.add_function(wrap_pyfunction!(do_scf, m)?)?;
    m.add_class::<Molecule>()?;
    m.add_class::<SCF>()?;
    m.add_class::<GeomCell>()?;
    m.add_class::<InputKeywords>()?;
    
    Ok(())
}
