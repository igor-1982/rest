#![allow(unused)]
use std::collections::binary_heap::Iter;
use std::fs::{File, self};
use std::io::{Write,BufRead, BufReader};
use pyo3::{pyclass, pymethods};
use rest_tensors::MatrixFull;
use rayon::str::Chars;
use regex::{Regex,RegexSet};
use itertools::izip;
use tensors::MathMatrix;
use std::collections::HashMap;
use serde::{Deserialize,Serialize};
use serde_json::Value;
//use tensors::Tensors;

use crate::basis_io::Basis4Elem;
use crate::constants::{SPECIES_NAME,MASS_CHARGE,ANG,SPECIES_INFO};
mod pyrest_geom_io;



//const SPECIES_INFO: HashMap<&str, &(f64, f64)> = 
//    (("H",(1.00794,1.0))).collect();



#[derive(Clone)]
#[pyclass]
pub struct GeomCell {
    #[pyo3(get,set)]
    pub name: String,
    #[pyo3(get,set)]
    pub elem: Vec<String>,
    #[pyo3(get,set)]
    pub fix:  Vec<bool>,
    pub unit: GeomUnit,
    //pub position: MatrixXx3<f64>, 
    pub position: MatrixFull<f64>, 
    pub lattice: MatrixFull<f64>,
    #[pyo3(get,set)]
    pub nfree: usize,
    pub pbc: MOrC,
    #[pyo3(get,set)]
    pub rest : Vec<(usize,String)>,
}

//impl GeomCell {
//    pub fn new() -> GeomCell {
//        GeomCell{
//            name            : String::from("a molecule"),
//            nfree           : 0,
//            elem            : vec![], 
//            fix             : vec![],
//            unit            : GeomUnit::Angstrom,
//            //position        : Tensors::new('F',vec![3,1],0.0f64),
//            //lattice         : Tensors::new('F',vec![3,1],0.0f64),
//            position        : MatrixFull::empty(),
//            lattice         : MatrixFull::empty(),
//            pbc             : MOrC::Molecule,
//            rest            : vec![],
//        }
//    }
//}

#[derive(Clone,Copy)]
pub enum MOrC {
    Molecule,
    Crystal,
}

#[derive(Clone,Copy)]
pub enum GeomUnit {
    Angstrom,
    Bohr,
}

pub enum EnergyUnit {
    Hartree,
    EV,
}

pub enum GeomType {
    Cartesian,
    Fraction,
}


pub fn formated_element_name(elem: &String) -> String {
    /// make sure the first letter of the element name is in uppercase.
    let mut tmp_elem = elem.to_lowercase();
    let mut c = tmp_elem.chars();
    match c.next() {
        None => String::new(),
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
    }
}


pub fn get_mass_charge(elem_list: &Vec<String>) -> Vec<(f64,f64)> {

    //let name_mass_charge: HashMap<&String,&(f64,f64)> = element_name.iter().zip(atomic_mass_charge.iter()).collect();
    //SPECIES_NAME.in
    //let name_mass_charge: HashMap<&str,&(f64,f64)> = SPECIES_NAME.iter().zip(MASS_CHARGE.iter())
    //.map(|(name,mc)| (*name,mc)).collect();

    let mut final_list: Vec<(f64,f64)> = vec![];

    for elem in elem_list {
        let formated_elem = formated_element_name(&elem);
        let tmp_result = SPECIES_INFO.get(formated_elem.as_str());
        if let Some(tmp_value) = tmp_result {
            final_list.push(**tmp_value);
        } else {
            panic!("Specify unknown element: {}. Please check the input file", elem);
        }
    }

    final_list
}

pub fn get_charge(elem_list: &Vec<String>) -> Vec<f64> {
    let mass_charge = get_mass_charge(&elem_list);
    let charge: Vec<f64> = mass_charge.into_iter().map(|(m,c)| c).collect();

    charge
}

impl GeomCell {
    pub fn init_geom() -> GeomCell {
        GeomCell{
            name            : String::from("a molecule"),
            nfree           : 0,
            elem            : vec![], 
            fix             : vec![],
            unit            : GeomUnit::Angstrom,
            //position        : Tensors::new('F',vec![3,1],0.0f64),
            //lattice         : Tensors::new('F',vec![3,1],0.0f64),
            position        : MatrixFull::empty(),
            lattice         : MatrixFull::empty(),
            pbc             : MOrC::Molecule,
            rest            : vec![],
        }
    }
    pub fn copy(&mut self, name:String) -> GeomCell {
        let mut new_mol = GeomCell::init_geom();
        new_mol.name = name;
        new_mol.nfree = self.nfree;
        new_mol.unit = self.unit;
        new_mol.position = self.position.to_owned();
        new_mol.lattice = self.lattice.to_owned();
        new_mol.pbc = match &self.pbc {
            MOrC::Crystal => MOrC::Crystal,
            MOrC::Molecule => MOrC::Molecule,
        };
        new_mol.rest = self.rest.to_owned();
        for (elem,fix) in izip!(&mut self.elem, &mut self.fix) {
            new_mol.elem.push(elem.to_string());
            new_mol.fix.push(*fix);
        }
        new_mol
    }
    pub fn get_nfree(&self) -> anyhow::Result<usize> {
        Ok(self.nfree)
    }
    pub fn get_elem(&self, index_a:usize) -> anyhow::Result<String> {
        Ok(self.elem[index_a].to_owned())
    }
    pub fn get_elems_iter(&self) ->  std::slice::Iter<'_, std::string::String> {
        self.elem.iter()
    }
    pub fn calc_nuc_energy(&self) -> f64 {
        let mass_charge = get_mass_charge(&self.elem);
        let mut nuc_energy = 0.0;
        let tmp_range1 = (0..self.position.size[1]);
        self.position.iter_columns(tmp_range1).enumerate().for_each(|(i,ri)| {
            let i_charge = mass_charge[i].1;
            let tmp_range2 = (0..i);
            self.position.iter_columns(tmp_range2).enumerate().for_each(|(j,rj)| {
                let j_charge = mass_charge[j].1;
                //println!("debug: {:?}, {:?}, i_charge: {:16.4}, j_charge: {:16.4}", &ri,&rj, i_charge, j_charge);
                let dd = ri.iter().zip(rj.iter())
                    .fold(0.0,|acc,(ri,rj)| acc + (ri-rj).powf(2.0)).sqrt();
                nuc_energy += i_charge*j_charge/dd;
            });
        });
        nuc_energy
    }

    pub fn calc_nuc_energy_deriv(&self) -> MatrixFull<f64> {
        let natm = self.elem.len();
        //let mut gs = vec![vec![0.0_f64;3];natm];
        let mut gs = MatrixFull::new([3,natm], 0.0_f64);
        for j in 0..natm {
            let q2 = get_mass_charge(&vec![self.elem[j].clone()])[0].1;
            let r2 = &self.position[(..,j)];
            for i in 0..natm {
                if i != j {
                    let q1 = get_mass_charge(&vec![self.elem[i].clone()])[0].1;
                    let r1 = &self.position[(..,i)];
                    let rdiff: Vec<f64> = r2.iter().zip(r1).map(|(r2,r1)| r1 - r2).collect();
                    let r = rdiff.iter().fold(0.0, |acc, rdiff| acc + rdiff*rdiff).sqrt();
                    //gs[j] -= q1 * q2 * (r2-r1) / r**3
                    //gs[j] += q1 * q2 * (r1-r2) / r**3
                    gs.iter_column_mut(j).zip(rdiff).for_each(|(gs,rdiff)| *gs += q1*q2*rdiff / r.powf(3.0) );
                }
            }
        }
        gs
    }
    
/*
    pub fn calc_nuc_energy_deriv(&self) -> Vec<Vec<f64>> {
        let natm = self.elem.len();
        let mut gs = vec![vec![0.0_f64;3];natm];
        for j in 0..natm {
            let q2 = get_mass_charge(&vec![self.elem[j].clone()])[0].1;
            let r2 = &self.position[(..,j)];
            for i in 0..natm {
                if i != j {
                    let q1 = get_mass_charge(&vec![self.elem[i].clone()])[0].1;
                    let r1 = &self.position[(..,i)];
                    let rdiff: Vec<f64> = r2.iter().zip(r1).map(|(r2,r1)| r1 - r2).collect();
                    let r = rdiff.iter().fold(0.0, |acc, rdiff| acc + rdiff*rdiff).sqrt();
                    //gs[j] -= q1 * q2 * (r2-r1) / r**3
                    //gs[j] += q1 * q2 * (r1-r2) / r**3
                    gs[j].iter_mut().zip(rdiff).for_each(|(gs,rdiff)| *gs += q1*q2*rdiff / r.powf(3.0) );
                }
            }
        }
        gs
    }
 */

    //pub fn get_elems_(&mut self,Vec<T>) ->  std::iter::Enumerate<std::slice::Iter<'_, std::string::String>> {
    //    self.elem.iter().enumerate()
    //}
    
    /// Get atom postion(coordinates) of the given atom index (eg. 1, 2, 3...).
    pub fn get_coord(&self, atm_id: usize) -> Vec<f64> {
        let coord = &self.position[(..,atm_id)];
        coord.to_vec()
    }

    pub fn get_fix(&self, index_a:usize) -> anyhow::Result<bool> {
        Ok(self.fix[index_a])
    }
    pub fn get_relax_index(&self, index_a:usize) -> anyhow::Result<usize> {
        let mut gi:usize = 0;
        let mut ci:usize = 0;
        while ci <= index_a {
            if !self.fix[gi] {ci += 1};
            gi += 1;
        }
        Ok(gi-1)
    }

    pub fn geom_shift(&mut self, atm_idx:usize, vec_xyz:Vec<f64>) {
        let mut gi = self.get_relax_index(atm_idx).unwrap();
        let mut given_atm = &mut self.position[(..,atm_idx)];
        given_atm.iter_mut().zip(vec_xyz.iter()).for_each(|(to, from)| {
            *to += from
        });
    }


    pub fn parse_position(position:&Vec<Value>,unit:&GeomUnit) -> anyhow::Result<(Vec<String>,Vec<bool>,MatrixFull<f64>,usize)> {
        // re0: the standard Cartesian position format with or without ',' as seperator
        //      no fix atom information
        let re0 = Regex::new(r"(?x)\s*
                            (?P<elem>\w{1,2})\s*,?    # the element
                            \s+
                            (?P<x>[\+-]?\d+.\d+)\s*,? # the 'x' position
                            \s+
                            (?P<y>[\+-]?\d+.\d+)\s*,? # the 'y' position
                            \s+
                            (?P<z>[\+-]?\d+.\d+)\s*,? # the 'z' position
                            \s*").unwrap();
        // re1: the standard Cartesian position format with or without ',' as seperator
        //      info. of fix atom is specified following the element name
        let re1 = Regex::new(r"(?x)\s*
                            (?P<elem>\w{1,2})\s*,?    # the element
                            \s+
                            (?P<fix>\d)\s*,? # 1 for geometry relazation; 0 for fix
                            \s+
                            (?P<x>[\+-]?\d+.\d+)\s*,? # the 'x' position
                            \s+
                            (?P<y>[\+-]?\d+.\d+)\s*,? # the 'y' position
                            \s+
                            (?P<z>[\+-]?\d+.\d+)\s*,? # the 'z' position
                            \s*").unwrap();
        let mut tmp_nfree:usize = 0;
        let mut tmp_ele: Vec<String> = vec![];
        let mut tmp_fix: Vec<bool> = vec![];
        let mut tmp_pos: Vec<f64> = vec![];
        for line in position {
            let tmpp = match line {
                Value::String(tmp_str)=>{tmp_str.clone()},
                other => {String::from("none")}
            };
            if let Some(cap) = re0.captures(&tmpp) {
                tmp_ele.push(cap[1].to_string());
                tmp_pos.push(cap[2].parse().unwrap());
                tmp_pos.push(cap[3].parse().unwrap());
                tmp_pos.push(cap[4].parse().unwrap());
                tmp_fix.push(false);
                tmp_nfree += 1;
            } else if let Some(cap) = re1.captures(&tmpp) {
                tmp_ele.push(cap[1].to_string());
                tmp_pos.push(cap[3].parse().unwrap());
                tmp_pos.push(cap[4].parse().unwrap());
                tmp_pos.push(cap[5].parse().unwrap());
                let tmp_num: i32 = cap[2].parse().unwrap();
                if tmp_num==0 {
                    tmp_fix.push(true);
                } else {
                    tmp_fix.push(false);
                    tmp_nfree += 1;
                }
            } else {
                panic!("Error: unknown geometry format: {}", &tmpp);
            }
        }
        let tmp_size: [usize;2] = [3,tmp_pos.len()/3];
        let mut tmp_pos_tensor = MatrixFull::from_vec(tmp_size, tmp_pos).unwrap();
        if let GeomUnit::Angstrom = unit {
            // To store the geometry position in "Bohr" according to the convention of quantum chemistry. 
            tmp_pos_tensor.self_multiple(ANG.powf(-1.0));
        };
        Ok((tmp_ele, tmp_fix, tmp_pos_tensor, tmp_nfree))
    }

    pub fn parse_lattice(lattice:&Vec<Value>, unit: &GeomUnit) -> anyhow::Result<MatrixFull<f64>> {
        //
        // re2: the standard Cartesian position format with or without ',' as seperator
        //
        let re2 = Regex::new(r"(?x)\s*
                            (?P<x>[\+-]?\d+.\d+)\s*,? # the 'x' position
                            \s+
                            (?P<y>[\+-]?\d+.\d+)\s*,? # the 'y' position
                            \s+
                            (?P<z>[\+-]?\d+.\d+)\s*,? # the 'z' position
                            \s*").unwrap();
        
        let mut tmp_vec: Vec<f64> = vec![];
        for line in lattice {
            let tmpp = match line {
                Value::String(tmp_str)=>{tmp_str.clone()},
                other => {String::from("none")}
            };
            if let Some(cap) = re2.captures(&tmpp) {
                tmp_vec.push(cap[1].parse().unwrap());
                tmp_vec.push(cap[2].parse().unwrap());
                tmp_vec.push(cap[3].parse().unwrap());
            } else {
                panic!("Error in reading the lattice constant: {}", &tmpp);
            }
        }
        let mut tmp_lat = unsafe{MatrixFull::from_vec_unchecked([3,3],tmp_vec)};
        if let GeomUnit::Angstrom = unit {
            // To store the lattice vector in "Bohr" according to the convention of quantum chemistry. 
            tmp_lat.self_multiple(ANG.powf(-1.0));
        };
        Ok(tmp_lat)
        //if frac_bool {
        //    new_geom.position = &new_geom.lattice * new_geom.position
        //}; 
    }
    pub fn to_numgrid_io(&self) -> Vec<(f64,f64,f64)> {
        let mut tmp_vec: Vec<(f64,f64,f64)> = vec![];
        self.position.data.chunks_exact(3).for_each(|value| {
            tmp_vec.push((value[0],value[1],value[2]))
        });
        tmp_vec
    }

    pub fn parse_position_from_string_vec(position:&Vec<String>,unit:&GeomUnit) -> anyhow::Result<(Vec<String>,Vec<bool>,MatrixFull<f64>,usize)> {
        // re0: the standard Cartesian position format with or without ',' as seperator
        //      no fix atom information
        let re0 = Regex::new(r"(?x)\s*
                            (?P<elem>\w{1,2})\s*,?    # the element
                            \s+
                            (?P<x>[\+-]?\d+.\d+)\s*,? # the 'x' position
                            \s+
                            (?P<y>[\+-]?\d+.\d+)\s*,? # the 'y' position
                            \s+
                            (?P<z>[\+-]?\d+.\d+)\s*,? # the 'z' position
                            \s*").unwrap();
        // re1: the standard Cartesian position format with or without ',' as seperator
        //      info. of fix atom is specified following the element name
        let re1 = Regex::new(r"(?x)\s*
                            (?P<elem>\w{1,2})\s*,?    # the element
                            \s+
                            (?P<fix>\d)\s*,? # 1 for geometry relazation; 0 for fix
                            \s+
                            (?P<x>[\+-]?\d+.\d+)\s*,? # the 'x' position
                            \s+
                            (?P<y>[\+-]?\d+.\d+)\s*,? # the 'y' position
                            \s+
                            (?P<z>[\+-]?\d+.\d+)\s*,? # the 'z' position
                            \s*").unwrap();
        let mut tmp_nfree:usize = 0;
        let mut tmp_ele: Vec<String> = vec![];
        let mut tmp_fix: Vec<bool> = vec![];
        let mut tmp_pos: Vec<f64> = vec![];
        for line in position {
            let tmpp = line.clone();
            //let tmpp = match line {
            //    Value::String(tmp_str)=>{tmp_str.clone()},
            //    other => {String::from("none")}
            //};
            if let Some(cap) = re0.captures(&tmpp) {
                tmp_ele.push(cap[1].to_string());
                tmp_pos.push(cap[2].parse().unwrap());
                tmp_pos.push(cap[3].parse().unwrap());
                tmp_pos.push(cap[4].parse().unwrap());
                tmp_fix.push(false);
                tmp_nfree += 1;
            } else if let Some(cap) = re1.captures(&tmpp) {
                tmp_ele.push(cap[1].to_string());
                tmp_pos.push(cap[3].parse().unwrap());
                tmp_pos.push(cap[4].parse().unwrap());
                tmp_pos.push(cap[5].parse().unwrap());
                let tmp_num: i32 = cap[2].parse().unwrap();
                if tmp_num==0 {
                    tmp_fix.push(true);
                } else {
                    tmp_fix.push(false);
                    tmp_nfree += 1;
                }
            } else {
                panic!("Error: unknown geometry format: {}", &tmpp);
            }
        }
        let tmp_size: [usize;2] = [3,tmp_pos.len()/3];
        let mut tmp_pos_tensor = MatrixFull::from_vec(tmp_size, tmp_pos).unwrap();
        if let GeomUnit::Angstrom = unit {
            // To store the geometry position in "Bohr" according to the convention of quantum chemistry. 
            tmp_pos_tensor.self_multiple(ANG.powf(-1.0));
        };
        Ok((tmp_ele, tmp_fix, tmp_pos_tensor, tmp_nfree))
    }


    pub fn parse_position_from_string(position: &String, unit: &GeomUnit) -> anyhow::Result<(Vec<String>,Vec<bool>,MatrixFull<f64>,usize)> {
        // re0: the standard Cartesian position format with or without ',' as seperator
        //      no fix atom information
        let re0 = Regex::new(r"(?x)\s*
                            (?P<elem>\w{1,2})\s*,?    # the element
                            \s+
                            (?P<x>[\+-]?\d+.\d+)\s*,? # the 'x' position
                            \s+
                            (?P<y>[\+-]?\d+.\d+)\s*,? # the 'y' position
                            \s+
                            (?P<z>[\+-]?\d+.\d+)\s*,? # the 'z' position
                            \s*").unwrap();
        // re1: the standard Cartesian position format with or without ',' as seperator
        //      info. of fix atom is specified following the element name
        let re1 = Regex::new(r"(?x)\s*
                            (?P<elem>\w{1,2})\s*,?    # the element
                            \s+
                            (?P<fix>\d)\s*,? # 1 for geometry relazation; 0 for fix
                            \s+
                            (?P<x>[\+-]?\d+.\d+)\s*,? # the 'x' position
                            \s+
                            (?P<y>[\+-]?\d+.\d+)\s*,? # the 'y' position
                            \s+
                            (?P<z>[\+-]?\d+.\d+)\s*,? # the 'z' position
                            \s*").unwrap();
        let mut tmp_nfree:usize = 0;
        let mut tmp_ele: Vec<String> = vec![];
        let mut tmp_fix: Vec<bool> = vec![];
        let mut tmp_pos: Vec<f64> = vec![];
        for cap in re0.captures_iter(&position) {
            tmp_ele.push(cap[1].to_string());
            tmp_pos.push(cap[2].parse().unwrap());
            tmp_pos.push(cap[3].parse().unwrap());
            tmp_pos.push(cap[4].parse().unwrap());
            tmp_fix.push(false);
            tmp_nfree += 1;
        };
        for cap in re1.captures_iter(&position) {
            tmp_ele.push(cap[1].to_string());
            tmp_pos.push(cap[3].parse().unwrap());
            tmp_pos.push(cap[4].parse().unwrap());
            tmp_pos.push(cap[5].parse().unwrap());
            let tmp_num: i32 = cap[2].parse().unwrap();
            if tmp_num==0 {
                tmp_fix.push(true);
            } else {
                tmp_fix.push(false);
                tmp_nfree += 1;
            }
        };
        let tmp_size: [usize;2] = [3,tmp_pos.len()/3];
        let mut tmp_pos_tensor = MatrixFull::from_vec(tmp_size, tmp_pos).unwrap();
        if let GeomUnit::Angstrom = unit {
            // To store the geometry position in "Bohr" according to the convention of quantum chemistry. 
            tmp_pos_tensor.self_multiple(ANG.powf(-1.0));
        };
        Ok((tmp_ele, tmp_fix, tmp_pos_tensor, tmp_nfree))

    }

    pub fn to_xyz(&self, filename: String) {
        let ang = crate::constants::ANG;
        let mut input = fs::File::create(&filename).unwrap();
        write!(input, "{}\n\n", self.elem.len());
        self.position.iter_columns_full().zip(self.elem.iter()).for_each(|(pos, elem)| {
            write!(input, "{:3}{:16.8}{:16.8}{:16.8}\n", elem, pos[0]*ang,pos[1]*ang,pos[2]*ang);
        });
    }

    pub fn formated_geometry(&self) -> String {
        let ang = crate::constants::ANG;
        let mut input = String::new();
        //write!(input, "{}\n\n", self.elem.len());
        self.position.iter_columns_full().zip(self.elem.iter()).for_each(|(pos, elem)| {
            //write!(input, "{:3}{:16.8}{:16.8}{:16.8}\n", elem, pos[0]*ang,pos[1]*ang,pos[2]*ang);
            input = format!("{}{:3}{:16.8}{:16.8}{:16.8}\n", input, elem, pos[0]*ang,pos[1]*ang,pos[2]*ang);
        });
        input
    }
    
    // evaluate the center of mass
    pub fn evaluate_center_of_mass(&self) -> (Vec<f64>, f64) {
        let mut mass_charge = get_mass_charge(&self.elem);
        let mut com = vec![0.0;3];
        let mut mass_sum = 0.0;
        self.position.iter_columns_full().zip(mass_charge.iter()).for_each(|(pos, (mass, charge))| {
            mass_sum += mass;
            com[0] += mass*pos[0];
            com[1] += mass*pos[1];
            com[2] += mass*pos[2];
        });
        com[0] /= mass_sum;
        com[1] /= mass_sum;
        com[2] /= mass_sum;
        (com, mass_sum)
    }

    // evalate the dipole moment of the nuclear charge
    pub fn evaluate_dipole_moment(&self) -> (Vec<f64>, f64) {
        let mut mass_charge = get_mass_charge(&self.elem);
        let mut dipole = vec![0.0;3];
        let mut mass_sum = 0.0;
        self.position.iter_columns_full().zip(mass_charge.iter()).for_each(|(pos, (mass, charge))| {
            mass_sum += mass;
            dipole[0] += charge*pos[0];
            dipole[1] += charge*pos[1];
            dipole[2] += charge*pos[2];
        });
        //dipole[0] /= mass_sum;
        //dipole[1] /= mass_sum;
        //dipole[2] /= mass_sum;
        (dipole, mass_sum)
    }


}

pub fn calc_nuc_energy_with_ecp(geom: &GeomCell, basis4elem: &Vec<Basis4Elem>) -> f64 {
    let mass_charge = get_mass_charge(&geom.elem);
    let mut nuc_energy = 0.0;
    let tmp_range1 = (0..geom.position.size[1]);
    geom.position.iter_columns(tmp_range1).enumerate().for_each(|(i,ri)| {
        let mut i_charge = mass_charge[i].1;
        if let Some(i_ecp) = basis4elem.get(i).unwrap().ecp_electrons {
            i_charge -= i_ecp as f64;
        };
        let tmp_range2 = (0..i);
        geom.position.iter_columns(tmp_range2).enumerate().for_each(|(j,rj)| {
            let mut j_charge = mass_charge[j].1;
            if let Some(j_ecp) = basis4elem.get(j).unwrap().ecp_electrons {
                j_charge -= j_ecp as f64;
            };
            let dd = ri.iter().zip(rj.iter())
                .fold(0.0,|acc,(ri,rj)| acc + (ri-rj).powf(2.0)).sqrt();
            nuc_energy += i_charge*j_charge/dd;
        });
    });
    nuc_energy
}

#[test]
fn test_string_parse() {
    let geom_str = "
        C   -6.1218053484 -0.7171513386  0.0000000000
        C   -4.9442285958 -1.4113046519  0.0000000000
        C   -3.6803098659 -0.7276441672  0.0000000000
        C   -2.4688049693 -1.4084918967  0.0000000000
        C   -1.2270315983 -0.7284607452  0.0000000000
        C    0.0000000000 -1.4090846909  0.0000000000
        C    1.22703159Do83 -0.7284607452  0.0000000000
        C    2.4688049693 -1.4084918967  0.0000000000
        C    3.6803098659 -0.7276441672  0.0000000000
        C    4.9442285958 -1.4113046519  0.0000000000
        C    6.1218053484 -0.7171513386  0.0000000000
        C    6.1218053484  0.7171513386  0.0000000000
        C    4.9442285958  1.4113046519  0.0000000000
        C    3.6803098659  0.7276441672  0.0000000000
        C    2.4688049693  1.4084918967  0.0000000000
        C    1.2270315983  0.7284607452  0.0000000000
        C    0.0000000000  1.4090846909  0.0000000000
        C   -1.2270315983  0.7284607452  0.0000000000
        C   -2.4688049693  1.4084918967  0.0000000000
        C   -3.6803098659  0.7276441672  0.0000000000
        C   -4.9442285958  1.4113046519  0.0000000000
        C   -6.1218053484  0.7171513386  0.0000000000
        H   -7.0692917090 -1.2490690741  0.0000000000
        H   -4.9430735200 -2.4988605526  0.0000000000
        H   -2.4690554105 -2.4968374995  0.0000000000
        H    0.0000000000 -2.4973235097  0.0000000000
        H    2.4690554105 -2.4968374995  0.0000000000
        H    4.9430735200 -2.4988605526  0.0000000000
        H    7.0692917090 -1.2490690741  0.0000000000
        H    7.0692917090  1.2490690741  0.0000000000
        H    4.9430735200  2.4988605526  0.0000000000
        H    2.4690554105  2.4968374995  0.0000000000
        H    0.0000000000  2.4973235097  0.0000000000
        H   -2.4690554105  2.4968374995  0.0000000000
        H   -4.9430735200  2.4988605526  0.0000000000
        H   -7.0692917090  1.2490690741  0.0000000000
    ".to_string();
    // re0: the standard Cartesian position format with or without ',' as seperator
    //      no fix atom information
    let re0 = Regex::new(r"(?x)\s*
                        (?P<elem>\w{1,2})\s*,?    # the element
                        \s+
                        (?P<x>[\+-]?\d+.\d+)\s*,? # the 'x' position
                        \s+
                        (?P<y>[\+-]?\d+.\d+)\s*,? # the 'y' position
                        \s+
                        (?P<z>[\+-]?\d+.\d+)\s*,? # the 'z' position
                        \s*").unwrap();
    // re1: the standard Cartesian position format with or without ',' as seperator
    //      info. of fix atom is specified following the element name
    let re1 = Regex::new(r"(?x)\s*
                        (?P<elem>\w{1,2})\s*,?    # the element
                        \s+
                        (?P<fix>\d)\s*,? # 1 for geometry relazation; 0 for fix
                        \s+
                        (?P<x>[\+-]?\d+.\d+)\s*,? # the 'x' position
                        \s+
                        (?P<y>[\+-]?\d+.\d+)\s*,? # the 'y' position
                        \s+
                        (?P<z>[\+-]?\d+.\d+)\s*,? # the 'z' position
                        \s*").unwrap();
    let re_set = RegexSet::new(&[
        // first
        r"(?x)\s*
        (?P<elem>\w{1,2})\s*,?    # the element
        \s+
        (?P<x>[\+-]?\d+.\d+)\s*,? # the 'x' position
        \s+
        (?P<y>[\+-]?\d+.\d+)\s*,? # the 'y' position
        \s+
        (?P<z>[\+-]?\d+.\d+)\s*,? # the 'z' position
        \s*",
        // second
        r"(?x)\s*
        (?P<elem>\w{1,2})\s*,?    # the element
        \s+
        (?P<fix>\d)\s*,? # 1 for geometry relazation; 0 for fix
        \s+
        (?P<x>[\+-]?\d+.\d+)\s*,? # the 'x' position
        \s+
        (?P<y>[\+-]?\d+.\d+)\s*,? # the 'y' position
        \s+
        (?P<z>[\+-]?\d+.\d+)\s*,? # the 'z' position
        \s*"
    ]).unwrap();
    //if let Some(cap) = re0.captures(&geom_str) {
    //    println!("{:?}", &cap);
    //}
    for cap in re1.captures_iter(&geom_str) {
        println!("{}, {}, {}, {}", 
            cap[1].to_string(), 
            cap[2].to_string(), 
            cap[3].to_string(), 
            cap[4].to_string());
    }
}