use pyo3::{pymethods, PyResult};

use crate::{ctrl_io::InputKeywords, geom_io::GeomCell};

use super::Molecule;

#[pymethods]
impl Molecule {
    #[new]
    #[pyo3(signature = (ctrl, geom))]
    pub fn new(ctrl: InputKeywords, geom: GeomCell) -> PyResult<Molecule> {
        Ok(Molecule::build_native(ctrl, geom).unwrap())
    }

    pub fn py_build(&mut self, ctrl: InputKeywords, geom: GeomCell) -> PyResult<Molecule> {
        Ok(Molecule::build_native(ctrl, geom).unwrap())
    }

    pub fn py_get_2dmatrix(&self, op_name:String) -> (Vec<f64>, [usize;2]) {
        let matr = self.int_ij_matrixupper(op_name);
        let matr_full = matr.to_matrixfull().unwrap();
        (matr_full.data.clone(), matr_full.size.clone())
    }

    pub fn calc_nuc_energy(&self) -> f64 {
        self.geom.calc_nuc_energy()
    }

}