//! This mod is designed to download basis set or auxiliary basis set from [basissetexchange.org] 
//! if required basis sets for elements are missing in local. 
//! 
//! [basissetexchange.org]: https://www.basissetexchange.org/


use libc::PTHREAD_CREATE_JOINABLE;
use regex::Regex;
use reqwest::header::{HeaderMap, HeaderValue, USER_AGENT};
use rest_libcint::CintType;
use serde::{Deserialize, Serialize};
use serde_json::{from_str,to_string};
use std::collections::HashMap;
use std::fmt::format;
use std::fs::{File, self, create_dir_all, read_dir};
use std::io::{Write, copy, Cursor};
use crate::basis_io::Basis4Elem;
use crate::geom_io::{GeomCell, get_mass_charge};
use array_tool::vec::{self, Intersect};

use super::Basis4ElemRaw;

//use super::Basis4Elem;

/// This structure works to hold results received from basissetexchange.org. It includes an hashmap 
/// named 'elements' where the keys are element names and the values are structures named [`crate::basis_io::bse_downloader::Basis`] which
/// stores basis set information for certain elements.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Info {
    //pub elements: HashMap<String, Basis>,
    pub elements: HashMap<String, Basis4ElemRaw>,
 }


///// This structure works to hold basis set information for certain elements. It contains two fields, 
///// 'electron_shells' and 'references'. Similar to Structure [`crate::basis_io::Basis4Elem`].
// #[derive(Serialize, Deserialize, Debug, Clone)]
//pub struct Basis {
//    electron_shells: Vec<Shell>,
//    references: Vec<Refs>,
// }
//
///// This structure works to hold certain electron shell information for certain elements. It contains five fields, 
///// 'function_type', 'region', 'angular_momentum', 'exponents' and 'coefficients'. Typically one Shell only holds the electron
///// shell for only one angular momentum. Similar to Structure [`crate::basis_io::BasCell`].
// #[derive(Serialize, Deserialize, Debug, Clone)]
//pub struct Shell {
//    function_type: String,
//    region: String,
//    angular_momentum: Vec<usize>,
//    exponents: Vec<String>,
//    coefficients: Vec<Vec<String>>,
//}

///// This structure works to hold certain reference information for certain elements. It contains two fields, 
///// reference_description and reference_keys. Similar to Structure [`crate::basis_io::RefCell`].
//#[derive(Serialize, Deserialize, Debug, Clone)]
//pub struct Refs {
//    reference_description: String,
//    reference_keys: Vec<String>,
//}

/// This function returns the matching atom number to the input element name. Deprecated due to low efficiency.
/// Alternative method is [`crate::geom_io::get_mass_charge`].
#[deprecated(note = "Low efficiency. Users should instead use crate::geom_io::get_mass_charge")]
fn elem_indexer(elem: &String) -> usize {

    let keys = vec![String::from("H"), String::from("He"),
         String::from("Li"),String::from("Be"),String::from("B"), String::from("C"),
         String::from("N"), String::from("O"), String::from("F"), String::from("Ne"),
         String::from("Na"),String::from("Mg"),String::from("Al"),String::from("Si"),
         String::from("P"), String::from("S"), String::from("Cl"),String::from("Ar"),
         String::from("K"), String::from("Ca"),String::from("Ga"),String::from("Ge"),
         String::from("As"),String::from("Se"),String::from("Br"),String::from("Kr"),
         String::from("Sc"),String::from("Ti"),String::from("V"), String::from("Cr"),String::from("Mn"),
         String::from("Fe"),String::from("Co"),String::from("Ni"),String::from("Cu"),String::from("Zn"),
         String::from("Rb"),String::from("Sr"),String::from("In"),String::from("Sn"),
         String::from("Sb"),String::from("Te"),String::from("I"), String::from("Xe"),
         String::from("Y"), String::from("Zr"),String::from("Nb"),String::from("Mo"),String::from("Tc"),
         String::from("Ru"),String::from("Rh"),String::from("Pd"),String::from("Ag"),String::from("Cd"),
         String::from("Cs"),String::from("Ba"),String::from("Tl"),String::from("Pb"),
         String::from("Bi"),String::from("Po"),String::from("At"),String::from("Rn"),
         String::from("La"),String::from("Hf"),String::from("Ta"),String::from("W"), String::from("Re"),
         String::from("Os"),String::from("Ir"),String::from("Pt"),String::from("Au"),String::from("Hg")];
    
    let values = vec![1,2,
    3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,
    19,20,31,32,33,34,35,36,
    21,22,23,24,25,26,27,28,29,30,
    37,38,49,50,51,52,53,54,
    39,40,41,42,43,44,45,46,47,48,
    55,56,81,82,83,84,85,86,
    57,72,73,74,75,76,77,78,79,80];

    let chem_elem: HashMap<String, usize> = keys.into_iter().zip(values.into_iter()).collect();
    let num = chem_elem.get(elem).unwrap();
    *num

}

/// This function download missing basis set from www.basissetexchange.org to a certain path. Basis set information
/// will be written into local json files.
pub fn bse_basis_getter(basis_set: &String, cell: &GeomCell, path: &String) {

    //check if path exists
    //create_dir_all(path);
    //element checker
    let ctrl_elem = ctrl_element_checker(cell);
    let local_elem = local_element_checker(path);
    //println!("local elements are {:?}", local_elem);
    //println!("ctrl elements are {:?}", ctrl_elem);

    let elem_intersection = ctrl_elem.intersect(local_elem.clone());
    let mut required_elem = vec![];
    for ctrl_item in ctrl_elem {
        if !elem_intersection.contains(&ctrl_item) {
            required_elem.push(ctrl_item)
        }
    }

    //println!("required elements are {:?}", required_elem);

    if required_elem.len() == 0 {
        return
    }

    //make HTTP request

    let mut headers = HeaderMap::new();
    headers.insert(USER_AGENT, HeaderValue::from_static("reqwest"));

    let elem_set: String = required_elem.iter().map(|a| format!("{},",a)).collect();
    let elem_num = required_elem.len();

    let url = format!("http://basissetexchange.org/api/basis/{}/format/json?elements={}", basis_set, elem_set);
    
    let resp = reqwest::blocking::Client::new()
        .get(&url)
        .headers(headers)
        .send()
        .expect("Download failed, please check basis set name or element.")
        .json::<Info>()
        .unwrap();
    //let response = reqwest::blocking::Client::new().get(&url).headers(headers).send().unwrap()..unwrap();
    if elem_num == resp.elements.len() {
        println!("{} for {:?} has been successfully downloaded.", basis_set, required_elem);
    }
    else {
        println!("Warning: the number of elements downloaded ({}) is not equal to the number of required elements ({})",
        resp.elements.len(), elem_num);
    }

    let required_elem_mass_charge = get_mass_charge(&required_elem);
    let required_elem_charge: Vec<String> = required_elem_mass_charge.iter().map(|(a, b)| (*b as usize).to_string()).collect();

    let mut index = 0;
    for elem in required_elem {
        let path_to_elem = format!("{}/{}.json", path, elem); 
        let atom_num = required_elem_charge[index].clone();
        //let atom_num = elem_indexer(&elem).to_string();
        let basis = resp.elements.get(&atom_num).unwrap();
        let mut f = File::create(&path_to_elem).unwrap();
        let basis_final = serde_json::to_writer_pretty(f, basis);
        //basis_modifier(&path_to_elem);
        index += 1;
    }


}

/// This function download missing basis set from www.basissetexchange.org to a certain path. Basis set information
/// will be written into local json files.
pub fn bse_basis_getter_v2(basis_set: &String, cell: &GeomCell, path: &String, required_elem: &Vec<String>) {

    //check if path exists
    //create_dir_all(path);
/*     
    //element checker
    let ctrl_elem = ctrl_element_checker(cell);
    let local_elem = local_element_checker(path);
    //println!("local elements are {:?}", local_elem);
    //println!("ctrl elements are {:?}", ctrl_elem);

    let elem_intersection = ctrl_elem.intersect(local_elem.clone());
    let mut required_elem = vec![];
    for ctrl_item in ctrl_elem {
        if !elem_intersection.contains(&ctrl_item) {
            required_elem.push(ctrl_item)
        }
    }
 */
    //println!("required elements are {:?}", required_elem);

    if required_elem.len() == 0 {
        return
    }

    //make HTTP request

    let mut headers = HeaderMap::new();
    headers.insert(USER_AGENT, HeaderValue::from_static("reqwest"));

    let elem_set: String = required_elem.iter().map(|a| format!("{},",a)).collect();
    let elem_num = required_elem.len();

    let url = format!("http://basissetexchange.org/api/basis/{}/format/json?elements={}", basis_set, elem_set);
    
    let resp = reqwest::blocking::Client::new()
        .get(&url)
        .headers(headers)
        .send()
        .expect("Download failed, please check basis set name or element.")
        .json::<Info>()
        .unwrap();
    //let response = reqwest::blocking::Client::new().get(&url).headers(headers).send().unwrap()..unwrap();
    if elem_num == resp.elements.len() {
        println!("{} for {:?} has been successfully downloaded.", basis_set, required_elem);
    }
    else {
        println!("Warning: the number of elements downloaded ({}) is not equal to the number of required elements ({})",
        resp.elements.len(), elem_num);
    }

    let required_elem_mass_charge = get_mass_charge(&required_elem);
    let required_elem_charge: Vec<String> = required_elem_mass_charge.iter().map(|(a, b)| (*b as usize).to_string()).collect();

    let mut index = 0;
    for elem in required_elem {
        let path_to_elem = format!("{}/{}.json", path, elem); 
        let atom_num = required_elem_charge[index].clone();
        //let atom_num = elem_indexer(&elem).to_string();
        let basis = resp.elements.get(&atom_num).unwrap();
        let mut f = File::create(&path_to_elem).unwrap();
        let basis_final = serde_json::to_writer_pretty(f, basis);
        //basis_modifier(&path_to_elem);
        index += 1;
    }


}

/// This function download missing auxiliary basis set from www.basissetexchange.org to a certain path. Basis set information
/// will be written into local json files.
pub fn bse_auxbas_getter(basis_set: &String, cell: &GeomCell, path: &String) {

    //check if path exists
    //create_dir_all(path);
    //element checker
    let ctrl_elem = ctrl_element_checker(cell);
    let local_elem = local_element_checker(path);
    //println!("local elements are {:?}", local_elem);
    //println!("ctrl elements are {:?}", ctrl_elem);

    let elem_intersection = ctrl_elem.intersect(local_elem.clone());
    let mut required_elem = vec![];
    for ctrl_item in ctrl_elem {
        if !elem_intersection.contains(&ctrl_item) {
            required_elem.push(ctrl_item)
        }
    }

    //println!("required elements are {:?}", required_elem);

    if required_elem.len() == 0 {
        return
    }

    //make HTTP request

    let mut headers = HeaderMap::new();
    headers.insert(USER_AGENT, HeaderValue::from_static("reqwest"));

    let elem_set: String = required_elem.iter().map(|a| format!("{},",a)).collect();
    let elem_num = required_elem.len();

    let url = format!("http://basissetexchange.org/api/basis/{}/format/json?elements={}", basis_set, elem_set);

    //println!("url = {}", url);
    
    let resp = reqwest::blocking::Client::new()
        .get(&url)
        .headers(headers)
        .send()
        .expect("Download failed, please check auxiliary basis set name or element.")
        .json::<Info>()
        .unwrap();
    //let response = reqwest::blocking::Client::new().get(&url).headers(headers).send().unwrap()..unwrap();
    if elem_num == resp.elements.len() {
        println!("{} for {:?} has been successfully downloaded.", basis_set, required_elem);
    }
    else {
        println!("Warning: the number of elements downloaded ({}) is not equal to the number of required elements ({})",
        resp.elements.len(), elem_num);
    }

    let required_elem_mass_charge = get_mass_charge(&required_elem);
    let required_elem_charge: Vec<String> = required_elem_mass_charge.iter().map(|(a, b)| (*b as usize).to_string()).collect();

    let mut index = 0;
    for elem in required_elem {
        let path_to_elem = format!("{}/{}.json", path, elem); 
        let atom_num = required_elem_charge[index].clone();
        //let atom_num = elem_indexer(&elem).to_string();
        let basis = resp.elements.get(&atom_num).unwrap();
        let mut f = File::create(&path_to_elem).unwrap();
        let basis_final = serde_json::to_writer_pretty(f, basis);
        //basis_modifier(&path_to_elem);
    }

}


pub fn bse_auxbas_getter_v2(basis_set: &String, cell: &GeomCell, path: &String, required_elem: &Vec<String>, print_level: usize) {

    //check if path exists
    //create_dir_all(path);

/*     
    //element checker
    let ctrl_elem = ctrl_element_checker(cell);
    let local_elem = local_element_checker(path);
    //println!("local elements are {:?}", local_elem);
    //println!("ctrl elements are {:?}", ctrl_elem);

    let elem_intersection = ctrl_elem.intersect(local_elem.clone());
    let mut required_elem = vec![];
    for ctrl_item in ctrl_elem {
        if !elem_intersection.contains(&ctrl_item) {
            required_elem.push(ctrl_item)
        }
    }
 */
    //println!("required elements are {:?}", required_elem);

    if required_elem.len() == 0 {
        return
    }

    //make HTTP request

    let mut headers = HeaderMap::new();
    headers.insert(USER_AGENT, HeaderValue::from_static("reqwest"));

    let elem_set: String = required_elem.iter().map(|a| format!("{},",a)).collect();
    let elem_num = required_elem.len();

    //println!("required_elem: {:?}", &required_elem);
    //println!("elem_set: {:?}", &elem_set);

    let url = format!("http://basissetexchange.org/api/basis/{}/format/json?elements={}", basis_set, elem_set);

    if print_level >=2 {println!("url = {}", url)};
    
    let resp = reqwest::blocking::Client::new()
        .get(&url)
        .headers(headers)
        .send()
        .expect("Download failed, please check auxiliary basis set name or element.")
        .json::<Info>()
        .unwrap();

    if print_level >= 1 { 
        if elem_num == resp.elements.len() {
            println!("{} for {:?} has been successfully downloaded.", basis_set, required_elem);
        }
        else {
            println!("Warning: the number of elements downloaded ({}) is not equal to the number of required elements ({})",
            resp.elements.len(), elem_num);
        }
    }

    let required_elem_mass_charge = get_mass_charge(&required_elem);
    let required_elem_charge: Vec<String> = required_elem_mass_charge.iter().map(|(a, b)| (*b as usize).to_string()).collect();

    println!("debug required_elem_charge: {:?}", &required_elem_charge);

    //let mut index = 0;
    required_elem.iter().zip(required_elem_charge.iter()).for_each(|(elem, atom_num)| {
        let path_to_elem = format!("{}/{}.json", path, elem);
        let basis = resp.elements.get(atom_num).unwrap();
        let mut f = File::create(&path_to_elem).unwrap();
        let basis_final = serde_json::to_writer_pretty(f, basis);
        //basis_modifier(&path_to_elem);
    });
    //for elem in required_elem {
    //    let path_to_elem = format!("{}/{}.json", path, elem); 
    //    let atom_num = required_elem_charge[index].clone();
    //    //let atom_num = elem_indexer(&elem).to_string();
    //    let basis = resp.elements.get(&atom_num).unwrap();
    //    let mut f = File::create(&path_to_elem).unwrap();
    //    let basis_final = serde_json::to_writer_pretty(f, basis);
    //    //basis_modifier(&path_to_elem);
    //}

}

/// This function modify basis set information if some coeifficients are non-zero. Works for cc-*VTZ basis sets, 
/// but not stable for other basis sets. Deprecated therefore.  
#[deprecated(note = "Not stable for basis sets other than cc type")]
pub fn basis_modifier(basis_path: &String) {
    let file_content = fs::read_to_string(basis_path).unwrap();
    let origin_basis = serde_json::from_str::<Basis4ElemRaw>(&file_content).unwrap();
    let mut modified_basis = origin_basis.clone();

    let mut modify_coord: Vec<(usize, usize, usize)> = vec![]; // (shell number, row number, line number)
    //note that shell number may not be equal to angular momentum, since the shells is out of order 

    //find certain coefficients, elements before which should be modified to 0.
    let mut shell_index = 0;
    for shell in origin_basis.electron_shells.iter() {
        let mut row_index = 0;
        for coeff_row in shell.coefficients.iter() {
            let mut coeff_index = 0;
            for coeff in coeff_row {
                if coeff.parse::<f64>().unwrap() == 1.0 && row_index != 0  {
                    modify_coord.push((shell_index,row_index,coeff_index));
                }
                coeff_index += 1;
            }
            row_index += 1;
        }
        shell_index += 1;
    }

    //println!("{:?}", modify_coord);

    for coord in modify_coord {
        for row_num in 0..coord.1 {
            modified_basis.electron_shells[coord.0].coefficients[row_num][coord.2] = String::from("0.000000E+00");
        }
    }

/*     
    let path_m = format!("{}.modified", basis_path);
    let mut f = File::create(path_m).unwrap();
    let basis_final = serde_json::to_writer_pretty(f, &modified_basis);
    */

    let mut f = File::create(basis_path).unwrap();
    let basis_final = serde_json::to_writer_pretty(f, &modified_basis);
    println!("Basis set {} has been successfully modified.", basis_path);
    

}

/// Checks elements information (.json) of a certain basis set in a given path. Returning a vector of element names.
pub fn local_element_checker(path: &String) -> Vec<String> {
    create_dir_all(&path);
    let elem_pattern = format!("{}/*.json", path);
    let mut elem_set = vec![];
    let re = Regex::new(r".*\.json$").unwrap();

    for item in read_dir(&path).unwrap() {
        //println!("{:?}", item.unwrap().file_name())
        let filename = item.unwrap().file_name();
        let text = filename.to_str().unwrap();
        if let Some(cap) = re.captures(text) {
            let len = cap[0].len();
            elem_set.push(cap[0][..(len-5)].to_string());
        }
    }
    //println!("{:?}, {}", elem_set, elem_set.len());

    elem_set
}

/// Checks elements in the given molecule. Returning a vector of element names.
pub fn ctrl_element_checker(cell: &GeomCell) -> Vec<String> {
    let raw_elem = &cell.elem;
    let mut elem_set = vec![];
    for item in raw_elem.iter() {
        if !elem_set.contains(item) {
            elem_set.push(item.clone())
        }
    }
    let raw_elem = &cell.ghost_bs_elem;
    for item in raw_elem.iter() {
        if !elem_set.contains(item) {
            elem_set.push(item.clone())
        }
    }
    elem_set
}


#[test]
fn local_test() {
    let path = String::from("/share/home/tygao/REST2.0/basis-set-pool/cc-pVTZ");
    local_element_checker(&path);
}    
//passed


/* 
#[test]
fn modifier() {
    let basis_path = String::from("/share/home/tygao/REST/BasisSets/cc-pVTZ/Cl.json");
    basis_modifier(&basis_path);
}
 */
#[test]
fn getter() {
    //bse_basis_getter(&String::from("6-31G*"), &String::from("Cl"), &"test".to_string());
}
/* 
#[test]
fn identifier() {
    let num = elem_indexer(&String::from("C"));
    println!("{}", num);
}
 */

#[test]
fn final_test1() {
    //let tmp_path = String::from("/share/home/tygao/REST/BasisSets/cc-pVTZ");
    let tmp_path = String::from("/home/igor/Documents/Package-Pool/rest_workspace/rest/basis-set-pool/def2-TZVP");
    let cint_type = CintType::Spheric;
    let atm_elem = String::from("Au");
    let re = Regex::new(r"/{1}[^/]*$").unwrap();
    let cap = re.captures(&tmp_path).unwrap();
    let basis_name = cap[0][1..].to_string();

    //println!("basis name is {}", basis_name);
    //bse_basis_getter(&basis_name,&atm_elem, &tmp_path);
    //let bas = Basis4Elem::parse_json_from_file(tmp_path,&cint_type).unwrap();

}
