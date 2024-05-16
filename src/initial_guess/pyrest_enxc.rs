//use pyo3::prelude::*;
use pyo3::{pymethods, PyResult};
use crate::initial_guess::enxc::ENXC;

use super::enxc::PotCell;

#[pymethods]
impl ENXC {
    #[new]
    pub fn new(atm_index: usize, position: Vec<f64>) -> PyResult<ENXC> {
        Ok(ENXC {
            enxc_potentials: vec![], 
            atm_index,
            position, 
            masscharge: (0.0, 0.0)
        })
    }


    pub fn py_count_parameters(&self) -> PyResult<(usize, usize)> {
        Ok(self.count_parameters())
    }

    pub fn py_add_a_potcell(&mut self, coeff: Vec<f64>, gaussian_exponents: Vec<f64>, angular_molentum: i32, r_exponents: i32) {
        self.add_a_potcell(&coeff, &gaussian_exponents, angular_molentum, r_exponents);
    }

    pub fn py_sort_by_angular_momentum(&mut self) {
        self.sort_by_angular_momentum();
    }

    pub fn py_change_a_potcell(&mut self, cur_potcel: PotCell, index: usize) {
        self.change_a_potcell(&cur_potcel, index);
    }


    pub fn py_get_enxc_potential_by_index(&self, index: usize) -> PyResult<PotCell> {
        Ok(self.get_enxc_potential_by_index(index))
    }

    pub fn py_allocate_coeff(&self, index: usize) -> PyResult<(usize, PotCell)> {
        Ok(self.allocate_coeff(index))
    }

    pub fn py_allocate_coeff_index(&self, index: usize) -> PyResult<(usize, usize, usize)> {
        Ok(self.allocate_coeff_index(index))
    }
}

#[pymethods]
impl PotCell {
    #[new]
    pub fn new(coeffs: Vec<f64>, gaussian_exponents: Vec<f64>, angular_momentum: i32, r_exponent: i32) -> Self {
        PotCell {
            coefficients: vec![coeffs],
            gaussian_exponents,
            angular_momentum: vec![angular_momentum],
            r_exponents: vec![r_exponent] 
        }
    }
    pub fn py_set_coeff(&mut self, coeff: Vec<f64>) {
        self.coefficients[0]=coeff;
    }

    pub fn py_set_gaussian_exponents(&mut self, gaussian_exponents: Vec<f64>) {
        self.gaussian_exponents = gaussian_exponents;
    }

    pub fn py_set_r_exponents(&mut self, r_exponents: i32) {
        self.r_exponents = vec![r_exponents];
    }
}

