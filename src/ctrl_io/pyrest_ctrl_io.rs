use pyo3::{pymethods, PyResult};
//use crate::molecule_io::Molecule;
//use crate::initial_guess::initial_guess;
use crate::ctrl_io::InputKeywords;

#[pymethods]
impl InputKeywords {
    #[new]
    #[pyo3(signature=(xc, charge, spin, basis_path, auxbas_path, print_level=0, num_threads=1, **basis_type))]   
    pub fn new(
        xc: String, 
        charge: f64, 
        spin: f64,
        basis_path:String,
        auxbas_path: String,
        print_level: usize,
        num_threads: usize,
        basis_type: Option<String>,
    ) -> PyResult<InputKeywords> {
        let mut new_ctrl = InputKeywords::init_ctrl();
        new_ctrl.py_set_num_threads(num_threads);
        new_ctrl.py_set_print_level(print_level);
        new_ctrl.py_set_xc(xc);
        new_ctrl.py_set_charge_spin([charge,spin]);
        new_ctrl.py_set_basis_path(basis_path);
        new_ctrl.py_set_basis_type(basis_type.clone().unwrap_or("spheric".to_string()));
        new_ctrl.py_set_auxbasis_path(auxbas_path);
        new_ctrl.py_set_auxbasis_type(basis_type.unwrap_or("spheric".to_string()));
        Ok(new_ctrl)
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
    pub fn py_set_eri_type(&mut self, eri_type: String) {
        if eri_type.to_lowercase().eq("ri-v") {
            self.eri_type = "ri_v".to_string()
        } else {
            self.eri_type = eri_type.to_lowercase()
        }
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
        self.spin = charge_and_spin[1];
    }
    pub fn py_set_initial_guess(&mut self, init_guess: String) {
        self.initial_guess = init_guess
    }
    pub fn py_set_spin_polarization(&mut self, flag: bool) {
        self.spin_channel = if flag {2} else {1};
        self.spin_polarization = flag;
    }
    pub fn py_set_num_threads(&mut self, num_threads: usize) {
        rayon::ThreadPoolBuilder::new().num_threads(num_threads).build_global();
        self.num_threads = Some(num_threads);
    }
    pub fn py_set_mixer(&mut self, mixer: String) {
        self.mixer = mixer.to_lowercase();
        let flag = self.mixer.eq("direct") || self.mixer.eq("linear") || self.mixer.eq("diis");
        if ! flag {
            println!("Warning: please use either 'direct', 'linear', or 'diis'")
        }
    }
    pub fn py_set_mix_param(&mut self, mix_param:f64) {
        self.mix_param = mix_param
    }

    pub fn py_set_etb(&mut self, etb_flag:bool) {
        self.even_tempered_basis = etb_flag
    }
}