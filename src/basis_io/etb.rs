use std::collections::HashMap;
use std::f64::consts::E;
use rest_tensors::MatrixFull;
use itertools::Itertools;
use libm::log;
use rust_libcint::CINTR2CDATA;
use statrs::statistics::{Statistics, Min};
use std::fmt::format;
use std::cmp::{max, max_by};
use std::vec;
use serde::{Deserialize, Serialize};
use crate::geom_io::{get_mass_charge, GeomCell};
use crate::molecule_io::Molecule;
use super::{Basis4Elem, BasCell};
use super::bse_downloader::{ctrl_element_checker};
use super::bse_downloader::Info;
use crate::constants::ATOM_CONFIGURATION;
use rest_tensors::matrix::matrix_blas_lapack::{_einsum_03, _einsum_03_forvec};



#[derive(Clone, Debug)]
pub struct InfoV2 {
    pub elements: HashMap<String, Basis4Elem>,
}


impl InfoV2 {
    pub fn new() -> InfoV2 {
        InfoV2 {
            elements: HashMap::new()
        }
    }
}

pub fn get_basis_from_mol(mol: &Molecule, atom: &String) -> anyhow::Result<Basis4Elem> {
    let mut index = 0;
    for item in mol.geom.elem.iter() {
        if atom == item {
            break;
        }
        index += 1;
    }
    Ok(mol.basis4elem[index].clone())
}

//generate even-tempered basis
//dfbasis: default aux basis. For pyscf, it is weigend, aka. def2-universal-jfit
//auxbas_path: path to auxbas
//beta: controls the size of generated auxbas set
// start_at: determine the first atom number to generate auxbas. Generally, heavy atoms 
//          use generated auxbas, while light atoms use dfbasis.


//give etbs for atom list
pub fn etb_gen_for_atom_list(mol: &Molecule, beta: &f64, required_elem: &Vec<String>) -> InfoV2 {

    //get elements in the given molecule
    let mut newbasis = InfoV2::new();
    //let nuc_start = elem_indexer(start_at)
    let required_elem_mass_charge = get_mass_charge(required_elem);
    let required_elem_charge: Vec<usize> = required_elem_mass_charge.iter().map(|(a, b)| (*b as usize)).collect();

    //println!("test1");

    let mut index = 0;
    for elem in required_elem {
        let nuc_charge = required_elem_charge[index];
        let conf = ATOM_CONFIGURATION[nuc_charge];
        let mut max_shells = 0;
        for shell in conf {
            if shell != 0 {
                max_shells += 1;
            }
        }

        //println!("test2, elem = {}", elem);

        let basis = get_basis_from_mol(&mol, elem).unwrap();
        let ang_array: Vec<usize> = basis.electron_shells.iter().map(|b| b.angular_momentum[0] as usize).collect();
        //max shell number, not max angular momentum (4 = 3 + 1)
        //let max_ang = ang_array.into_iter().max().unwrap() + 1;

        let mut emin_by_l = vec![1.0e99_f64; 8];
        let mut emax_by_l = vec![0.0_f64; 8];
        let mut l_max = 0;
        
        for b in basis.electron_shells {
            let l = b.angular_momentum[0] as usize;
            l_max = max(l, l_max);
            if l >= (max_shells + 1) {
                //println!("maxshell = {}, l={}", max_shells, l);
                continue;
            }
            let es = b.exponents;
            let cs = b.coefficients; 
            let mut es_new = vec![];    

            //println!("test3, l = {}", l);

            for row_num in 0..cs[0].len() {
                //let mut max = 0.0;
                let mut tmp_max = 0.0;
                //println!("test4, row_num = {}", row_num);
                //tmp = cs[row_num].iter().map(|v| v.abs()).collect_vec().max();
                let mut tmp = cs.iter().map(|col| col[row_num].abs()).collect_vec();
                let tmp_max = tmp.clone().max();
                //println!("test4.5, tmp = {:?}", tmp);
                if tmp_max > 1.0e-3*CINTR2CDATA::gto_norm(l as std::os::raw::c_int, es[row_num]) {
                    es_new.push(es[row_num]);
                    //println!("test5, es_new = {:?}", es_new);
                }

            }

            emax_by_l[l] = emax_by_l[l].max(es_new.clone().max());
            emin_by_l[l] = emin_by_l[l].min(es_new.min());
            //println!("test6, emax_by_l = {:?}, emin_by_l = {:?}", emax_by_l, emin_by_l);

        }

        let l_max1 = l_max + 1;
        emax_by_l = emax_by_l[0..l_max1].to_vec();
        emin_by_l = emin_by_l[0..l_max1].to_vec();
        //Estimate the exponents ranges by geometric average

        //println!("test7, emax_by_l = {:?}, emin_by_l = {:?}", emax_by_l, emin_by_l);

        let emax_by_l_root: Vec<f64> = emax_by_l.iter().map(|a| a.sqrt()).collect();
        let emin_by_l_root: Vec<f64> = emin_by_l.iter().map(|a| a.sqrt()).collect();

        //println!("test8, emax_by_l_root = {:?}, emin_by_l_root = {:?}", emax_by_l_root, emin_by_l_root);

        let emax = _einsum_03_forvec(&emax_by_l_root, &emax_by_l_root);
        let emin = _einsum_03_forvec(&emin_by_l_root, &emin_by_l_root);

        //println!("test9, emax = {:?}, emin = {:?}", emax, emin);

        //l_max1 is max_ang
        //let liljsum = MatrixFull::<f64>::new_ijsum([l_max1; 2]);
        let mut emax_by_l_new = vec![];
        let mut emin_by_l_new = vec![];
        for i in 0..(l_max1*2-1) {
            emax_by_l_new.push(emax.get_sub_antidiag_terms(i).unwrap().max());
            emin_by_l_new.push(emin.get_sub_antidiag_terms(i).unwrap().min());
            //println!("test10, i = {}", i);
        }
        // Tune emin and emax

        emax_by_l_new = emax_by_l_new.into_iter().map(|v| 2.0*v).collect(); //*2 for alpha+alpha on same center
        emin_by_l_new = emin_by_l_new.into_iter().map(|v| 2.0*v).collect(); // (numpy.arange(l_max1*2-1)*.5+1)
        
        let mut ns: Vec<f64> = emax_by_l_new.iter().zip(&emin_by_l_new).map(|(max,min)| ((max+min)/min).log(E)/beta.log(E)).collect();

        //let etb = ns.iter().enumerate().map(|(i, n)|(i, n, emin_by_l_new[i], beta)).collect()
        ns = ns.iter().map(|v| v.ceil()).filter(|v| *v > 0.0).collect();

        let etb_para: Vec<(usize, usize, f64, f64)> = ns.iter().zip(emin_by_l_new).enumerate().map(|(l, (n, min))|(l, *n as usize, min, *beta)).collect();
        let etb_result = etbs_gen(etb_para);

        newbasis.elements.insert(elem.clone(), etb_result);

        index += 1;
    }
    
    //println!("newbasis = {:?}", newbasis);
    newbasis

}

pub fn etbs_gen(input: Vec<(usize, usize, f64, f64)>) -> Basis4Elem {

    let mut bas = Basis4Elem {
        electron_shells: vec![],
        references: None,
        global_index: (0,0),
    };
    
    bas.electron_shells = input.iter()
    .map(|(l, n, alpha, beta)| etb_gen(l, n, alpha, beta)).collect();

    bas

}

//generate etb for a single electron shell
pub fn etb_gen(l: &usize, n: &usize, alpha: &f64, beta: &f64) -> BasCell {
    let mut shell = BasCell {
        function_type: Some(String::from("gto")),
        region: None,
        angular_momentum: vec![*l as i32],
        exponents: vec![],
        coefficients: vec![vec![1.0_f64; *n]],
    };
/*
    for i in (0..n).rev() {
        shell.exponents.push(alpha*beta*(i as f64));
        shell.coefficients[0].push(1.0_f64);
    }
 */
    shell.exponents = (0..*n).rev().map(|i| alpha*beta.powf((i as f64))).collect();


    /* 
        Generate the exponents of even tempered basis for :attr:`Mole.basis`.
    .. math::

        e = e^{-\alpha * \beta^{i-1}} for i = 1 .. n

    Args:
        l : int
            Angular momentum
        n : int
            Number of GTOs

     */
    //[[l, [alpha*beta**i, 1]] for i in reversed(range(n))]

    shell

}

pub fn get_etb_elem(mol: &GeomCell, start_atom: &usize) -> Vec<String>{
    let elem_list = ctrl_element_checker(mol);
    let elem_mass_charge = get_mass_charge(&elem_list);
    let elem_charge_list: Vec<usize> = elem_mass_charge.iter().map(|(a, b)| (*b as usize)).collect();
    let etb_elem = elem_list.iter().zip(elem_charge_list)
        .filter(|(elem, charge)| *charge >= *start_atom)
        .map(|(elem, charge)| elem.clone()).collect();

    etb_elem
}


 #[test]
 fn test_einsum_03() {
     let mut vec_a = [1.0, 2.0, 3.0];
     let mut vec_b = [1.0, 2.0];
     let mut mat_c = _einsum_03(&vec_a, &vec_b);
     println!("{:?}", mat_c);
 
 }

#[test]
fn test_diag() {
    let a = MatrixFull::from_vec([3,3], vec![1,2,3,4,5,6,7,8,9]).unwrap();
    let v = a.get_diagonal_terms().unwrap();
    println!("{:?}", v);


}

#[test]
fn test_antidiag() {
    let a = MatrixFull::from_vec([4,4], vec![1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]).unwrap();
    let v = a.get_sub_antidiag_terms(4).unwrap();
    
    println!("{:?}", v);


}



#[test]
fn test_iter() {
    let mut a = vec![1,2,3,4,5,6];
    a = a.into_iter().map(|v| v*2).collect();
    println!("{:?}", a);
}

