use pyo3::pymethods;
use crate::geom_io::GeomUnit;

use super::GeomCell;

#[pymethods]
impl GeomCell {
    #[new]
    #[pyo3(signature = (pos, **unit))]
    pub fn new(pos: String, unit: Option<String>) -> GeomCell {
        let mut new_geom = GeomCell::init_geom();
        new_geom.py_set_unit(unit.unwrap_or(String::from("Angstrom")));
        new_geom.py_set_position_from_string(pos);
        new_geom
    }
    pub fn py_get_unit(&self) -> String {
        match self.unit {
            super::GeomUnit::Angstrom => "Anstronm".to_string(),
            super::GeomUnit::Bohr => "Bohr".to_string(),
        }
    }
    pub fn py_get_position(&self) -> (Vec<f64>, [usize;2]) {
        (self.position.data.clone(), self.position.size.clone())
    }
    pub fn py_get_elem(&self) -> (Vec<String>, usize) {
        (self.elem.clone(), self.elem.len())
    }
    pub fn py_set_unit(&mut self, unit: String) {
        if unit.to_lowercase()==String::from("angstrom") {
            self.unit=GeomUnit::Angstrom;
        } else if unit.to_lowercase()==String::from("bohr") {
            self.unit=GeomUnit::Bohr
        } else {
            println!("Warning:: unknown geometry unit is specified: {}. Angstrom will be used", unit);
            self.unit=GeomUnit::Angstrom;
        };
    }
    pub fn py_set_position_from_string(&mut self, pos: String) {
        let (elem, fix, pos,n_free) = GeomCell::parse_position_from_string(&pos, &self.unit).unwrap();
        self.elem = elem;
        self.fix = fix;
        self.position = pos;
        self.nfree = n_free;

    }
    pub fn py_set_position(&mut self, pos: Vec<String>) {
        let (elem, fix, pos,n_free) = GeomCell::parse_position_from_string_vec(&pos, &self.unit).unwrap();
        self.elem = elem;
        self.fix = fix;
        self.position = pos;
        self.nfree = n_free;
    }
    pub fn py_calc_nuc_energy(&self) -> f64 {
        self.calc_nuc_energy()
    }
    pub fn py_to_xyz(&self, filename:String) {
        self.to_xyz(filename)
    }
}