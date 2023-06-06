use crate::molecule_io::Molecule;
use crate::geom_io;
use crate::basis_io;
use crate::scf_io::SCF;
use rest_tensors::MatrixFull;
use rest_libcint::CINTR2CDATA;
use rest_libcint::CintType;
use std::borrow::BorrowMut;
use std::collections::HashMap;
use rayon::iter::{IntoParallelRefIterator, IndexedParallelIterator, ParallelIterator, IntoParallelRefMutIterator};
//mod lib;
use std::fs::{self, File};
use std::io::prelude::*;
use std::io::LineWriter;

pub fn gen_header(mol: &Molecule) -> String{
    let mut header = "[Molden Format]\n".to_owned();
    header += "[Title]\n";
    header += "Molden file created by REST.\n";
    header += "\n[Atoms] AU\n";
    let geometry = &mol.geom;
    let atom_mass_charge = geom_io::get_mass_charge(&geometry.elem).clone();
    geometry.elem.iter().zip(1..(geometry.elem.len()+1)).zip(geometry.position.iter_columns_full()).zip(atom_mass_charge.iter())
        .for_each(|(((elem,index),position),(mass,charge))|{
        header += elem;
        header += "    ";
        header += &index.to_string();
        header += "    ";
        let charge_u = *charge as usize;
        header += &charge_u.to_string();
        position.iter().for_each(|x|{
            header += "    ";
            header += &x.to_string();
        });
        header += "\n";

    });
    header += "[GTO]\n";
    let num_atom = mol.geom.elem.len();
    mol.basis4elem.iter().zip(1..(num_atom+1)).for_each(|(bas,index)|{
        header += "  ";
        header += &index.to_string();
        header += "  0\n";
        let mut ele_shells =bas.electron_shells.clone();
        ele_shells.iter_mut().for_each(|ao_ele|{
            let mut tmp_ang = ao_ele.angular_momentum[0];
            let mut tmp_coefficients: Vec<Vec<f64>> = vec![];
            for coe_vec in ao_ele.coefficients.iter() {
                let mut tmp_coefficients_column: Vec<f64> = vec![];
                coe_vec.iter().enumerate().for_each(|(ix,x)| {
                    let tmp_value = CINTR2CDATA::gto_norm(tmp_ang as std::os::raw::c_int,
                        ao_ele.exponents[ix]);
                    //println!("gto_norm for {}: {}",tmp_exponents[ix],tmp_value);
                    tmp_coefficients_column.push(*x/tmp_value);
                    //env.push(*x);
                    //tmp_coefficients_column.push(*x);
                });
                tmp_coefficients.push(tmp_coefficients_column);
            };
            ao_ele.coefficients = tmp_coefficients.clone();

            //ao_ele.divide_normalization(&CintType::Spheric);
            ao_ele.coefficients.iter().for_each(|co|{
                header += " ";
                header += &match_angular_momentum(ao_ele.angular_momentum[0]);
                header += "    ";
                header += &ao_ele.exponents.len().to_string();
                header += " 1.00\n";
                co.iter().zip(ao_ele.exponents.iter()).for_each(|(c,alpha)|{
                    header += "                ";
                    header += &alpha.to_string();
                    header += "  ";
                    header += &c.to_string();
                    header += "\n";
                });
       
            })
            
        });
    });
    header += "\n[5D]\n[9G]\n";
    
    header
}

 pub fn gen_mo_info(scf_data: &SCF) -> String {
    let mut mo_info = "[MO]\n".to_owned();
    let energy = &scf_data.eigenvalues[0];
    let coeff = &scf_data.eigenvectors[0];
    let occup = &scf_data.occupation[0];
    coeff.iter_columns_full().zip(energy.iter().zip(occup.iter())).for_each(|(mo_co,(eig,occ))|{
        mo_info +=  " Sym=     1a\n Ene= ";
        mo_info += &eig.to_string();
        mo_info += "\n Spin= Alpha\n Occup= ";
        mo_info += &occ.to_string();
        mo_info += "\n";
        let num_ao = mo_co.len();
        mo_co.iter().zip(1..(num_ao+1)).for_each(|(ao_co,index)|{
            mo_info += " ";
            mo_info += &index.to_string();
            mo_info += "      ";
            mo_info += &ao_co.to_string();
            mo_info += "\n";
        });   
    });
    mo_info
 }

pub fn gen_molden(scf_data: &SCF) -> String {
    let mol = &scf_data.mol;
    let mut text = gen_header(&mol);
    text += &gen_mo_info(scf_data);
    //println!("{}",&text);

    let mut path = "src/post_scf_analysis/generated_file/".to_owned();
    path +=  &mol.geom.name;
    path += ".molden";
    let file = File::create(path);
    let mut file = LineWriter::new(file.unwrap());
    file.write_all(&text.into_bytes());
    
    "Molden successfull built.".to_owned()
}
pub fn match_angular_momentum(angular_momentum: i32) -> String{
    let i = HashMap::from([
        (0, "s".to_owned()),
        (1, "p".to_owned()),
        (2, "d".to_owned()),
        (3, "f".to_owned()),
        (4, "g".to_owned()),
    ]);
    i.get(&angular_momentum).unwrap().clone()

    
}
