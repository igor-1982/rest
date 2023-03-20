#![allow(unused)]
use std::collections::binary_heap::Iter;
use std::fs::File;
use std::io::{Write,BufRead, BufReader};
use rest_tensors::MatrixFull;
use rayon::str::Chars;
use regex::Regex;
use itertools::izip;
use tensors::MathMatrix;
use std::collections::HashMap;
use serde::{Deserialize,Serialize};
use serde_json::Value;
//use tensors::Tensors;

use crate::constants::{SPECIES_NAME,MASS_CHARGE,ANG,SPECIES_INFO};



//const SPECIES_INFO: HashMap<&str, &(f64, f64)> = 
//    (("H",(1.00794,1.0))).collect();



#[derive(Clone)]
pub struct GeomCell {
    pub name: String,
    pub elem: Vec<String>,
    pub fix:  Vec<bool>,
    pub unit: GeomUnit,
    //pub position: MatrixXx3<f64>, 
    pub position: MatrixFull<f64>, 
    pub lattice: MatrixFull<f64>,
    pub nfree: usize,
    pub pbc: MOrC,
    pub rest : Vec<(usize,String)>,
}

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

impl GeomCell {
    pub fn new() -> GeomCell {
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
        let mut new_mol = GeomCell::new();
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
    pub fn get_elems_iter(&mut self) ->  std::slice::Iter<'_, std::string::String> {
        self.elem.iter()
    }
    pub fn calc_nuc_energy(&mut self) -> f64 {
        let mass_charge = get_mass_charge(&self.elem);
        let mut nuc_energy = 0.0;
        //(0..self.elem.len()).into_iter().for_each(|i| {
        //    let i_charge =  mass_charge[i].1;
        //    let mut i_position = self.position.get_reducing_tensor(i).unwrap();
        //    (0..i).into_iter().for_each(|j| {
        //        let j_charge =  mass_charge[j].1;
        //        let j_position = self.position.get_reducing_tensor(j).unwrap();
        //        let mut dd = (i_position.clone() - j_position).abs();
        //        //println!("Debug {}={},{}={},{}",i, i_charge, j, j_charge, dd);
        //        nuc_energy += i_charge*j_charge/dd;
        //    });
        //});
        let tmp_range1 = (0..self.position.size[1]);
        self.position.iter_columns(tmp_range1).enumerate().for_each(|(i,ri)| {
            let i_charge = mass_charge[i].1;
            let tmp_range2 = (0..i);
            self.position.iter_columns(tmp_range2).enumerate().for_each(|(j,rj)| {
                let j_charge = mass_charge[j].1;
                let dd = ri.iter().zip(rj.iter())
                    .fold(0.0,|acc,(ri,rj)| acc + (ri-rj).powf(2.0)).sqrt();
                nuc_energy += i_charge*j_charge/dd;
            });
        });

        println!("Nuc_energy: {}",nuc_energy);
        
        nuc_energy
    }
    //pub fn get_elems_(&mut self,Vec<T>) ->  std::iter::Enumerate<std::slice::Iter<'_, std::string::String>> {
    //    self.elem.iter().enumerate()
    //}
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
}