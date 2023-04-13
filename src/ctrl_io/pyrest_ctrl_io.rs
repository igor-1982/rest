use pyo3::{pymethods, PyResult};
//use crate::molecule_io::Molecule;
//use crate::initial_guess::initial_guess;
use crate::ctrl_io::InputKeywords;

#[pymethods]
impl InputKeywords {
    #[new]
    pub fn new() -> PyResult<InputKeywords> {
        Ok(InputKeywords::init_ctrl())
    }
    pub fn py_set_print_level(&mut self, print_level: usize) {
        self.print_level = print_level
    }
    pub fn py_set_xc(&mut self, xc: String) {
        self.xc = xc
    }
    pub fn py_set_basis_path(&mut self, basis_path: String) {
        self.basis_path = basis_path
    }
    pub fn py_set_basis_type(&mut self, basis_type: String) {
        self.basis_type = basis_type
    }
    pub fn py_set_auxbasis_path(&mut self, auxbasis_path: String) {
        self.auxbas_path = auxbasis_path
    }
    pub fn py_set_auxbasis_type(&mut self, auxbasis_type: String) {
        self.auxbas_type = auxbasis_type
    }
    pub fn py_set_charge_spin(&mut self, charge_and_spin: [f64;2]) {
        self.charge = charge_and_spin[0];
        self.spin = charge_and_spin[0];
    }
    pub fn py_set_initial_guess(&mut self, init_guess: String) {
        self.initial_guess = init_guess
    }
    pub fn py_set_spin_polarization(&mut self, flag: bool) {
        self.spin_channel = if flag {2} else {1};
        self.spin_polarization = flag;
    }
    pub fn py_set_num_threads(&mut self, num_threads: usize) {
        rayon::ThreadPoolBuilder::new().num_threads(num_threads).build_global().unwrap();
        self.num_threads = Some(num_threads);
    }
}