use pyo3::{pymethods,PyResult};
use crate::molecule_io::Molecule;
use crate::initial_guess::initial_guess;

use super::SCF;

#[pymethods]
impl SCF {
    #[new]
    pub fn new(mol: Molecule) -> PyResult<SCF> {
        Ok(SCF::build(mol))
    }
    pub fn py_get_dm(&mut self) -> PyResult<([Vec<f64>;2], [usize;2])> {
        Ok((
            [self.density_matrix[0].data.clone(), self.density_matrix[1].data.clone()], 
            [self.mol.num_basis, self.mol.num_basis]
        ))
    }
    pub fn py_get_hamiltonian(&mut self) -> PyResult<([Vec<f64>;2], [usize;2])> {
        self.generate_hf_hamiltonian();
        Ok((
            [
                self.hamiltonian[0].to_matrixfull().unwrap().data.clone(), 
                self.hamiltonian[1].to_matrixfull().unwrap().data.clone()
            ], 
            [self.mol.num_basis, self.mol.num_basis]
        ))
    }
    pub fn py_get_ovlp(&mut self) -> PyResult<(Vec<f64>, [usize;2])> {
        Ok((
            self.ovlp.to_matrixfull().unwrap().data.clone(),
            [self.mol.num_basis, self.mol.num_basis]
        ))
    }
    pub fn py_get_hcore(&mut self) -> PyResult<(Vec<f64>, [usize;2])> {
        Ok((
            self.h_core.to_matrixfull().unwrap().data.clone(), 
            [self.mol.num_basis, self.mol.num_basis]
        ))
    }
    //pub fn py_get_init_hamiltonian(&mut self) -> PyResult<([Vec<f64>;2], [usize;2])> {
    //    initial_guess(self);
    //    self.generate_hf_hamiltonian();
    //    Ok((
    //        [   self.hamiltonian[0].to_matrixfull().unwrap().data.clone(), 
    //            self.hamiltonian[1].to_matrixfull().unwrap().data.clone()
    //        ], 
    //        [self.mol.num_basis, self.mol.num_basis]
    //    ))
    //}

}