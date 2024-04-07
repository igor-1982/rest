use core::panic;
use std::fs;
use libc::TCA_DUMP_INVISIBLE;
use rest_libcint::CINTR2CDATA;
use serde::{Deserialize, Serialize};
use serde_json::{Result,Value};
use anyhow;
use tensors::{MatrixFull, MatrixUpper, TensorSlice, TensorSliceMut};

use crate::{molecule_io::Molecule, scf_io::SCF};
use crate::constants::{ATM_NUC_MOD_OF, NUC_ECP, SPECIES_INFO};

#[derive(Clone, Debug,Serialize,Deserialize)]
pub struct PotCellRaw {
    pub angular_momentum: Vec<i32>,
    pub coefficients: Vec<Vec<String>>,
    pub r_exponents: Vec<i32>,
    pub gaussian_exponents: Vec<String>,
}

#[derive(Clone, Debug,Serialize,Deserialize)]
pub struct PotCell {
    pub angular_momentum: Vec<i32>,
    pub coeffcients: Vec<Vec<f64>>,
    pub r_exponents: Vec<i32>,
    pub gaussian_exponents: Vec<f64>,
}

#[derive(Clone, Debug,Serialize,Deserialize)]
pub struct ENXCRaw {
    pub enxc_potentials: Vec<PotCellRaw>,
    pub elem_type: String,
    pub position: Vec<String>,
}

#[derive(Clone, Debug,Serialize,Deserialize)]
pub struct ENXC {
    pub enxc_potentials: Vec<PotCell>,
    pub masscharge: (f64,f64),
    pub position: Vec<f64>,
}


pub fn parse_enxc_from_string(cont: String) -> anyhow::Result<ENXC> {

    let raw:Value = serde_json::from_str(&cont[..])?;
    let tmp_enxc_raw: ENXCRaw = serde_json::from_value(raw)?;

    let enxc_potentials: Vec<PotCell> = tmp_enxc_raw.enxc_potentials.iter().map(|x| x.parse()).collect::<Vec<PotCell>>();

    let masscharge = SPECIES_INFO[&tmp_enxc_raw.elem_type[..]].clone();

    let position: Vec<f64> = tmp_enxc_raw.position.iter().map(|x| x.parse().unwrap()).collect();

    Ok(ENXC { enxc_potentials, masscharge, position})
}

pub fn parse_enxc_potential(file_name: &str) -> anyhow::Result<ENXC> {

    let tmp_cont = fs::read_to_string(file_name)?;
    parse_enxc_from_string(tmp_cont)

}

//pub fn parse_enxc_from_deepnet(index: usize) -> anyhow::Result<ENXC> {
//
//    let tmp_cont = ;
//    parse_enxc_from_string(tmp_cont)
//
//}

pub fn effective_nxc_matrix(mol: &mut Molecule) -> MatrixUpper<f64> {

    let org_env = mol.cint_env.clone();
    let org_atm = mol.cint_atm.clone();

    let nbas = mol.fdqc_bas.len();
    let mut enxc = MatrixUpper::new(nbas*(nbas+1)/2,0.0);

    // now import enxc potentials info.
    let mut enxcbas: Vec<Vec<i32>> = vec![];

    let mut atm = &mut mol.cint_atm;
    let mut env = &mut mol.cint_env;
    mol.geom.elem.iter().zip(atm.iter_mut()).enumerate().for_each(|(atm_index, (elem, cur_atm))| {
        let file_name = format!("./enxc/{}.json", atm_index);
        //println!("{}",&file_name);
        let tmp_enxc = parse_enxc_potential(&file_name[..]).unwrap();

        cur_atm[ATM_NUC_MOD_OF] = NUC_ECP;

        let enxc_ang_start = (tmp_enxc.enxc_potentials.len() -1) as i32; 

        for potcell in tmp_enxc.enxc_potentials.iter() {
            let angl = potcell.angular_momentum[0];
            let coeffs = &potcell.coeffcients;
            let r_exponents = *potcell.r_exponents.get(0).unwrap();
            let gaussian_exponents = &potcell.gaussian_exponents;
            let num_exp = gaussian_exponents.len() as i32;
            let num_coeffs = coeffs.len() as i32;

            let enxc_exp_start = env.len() as i32;
            env.extend(gaussian_exponents.iter());

            coeffs.iter().for_each(|each_coeffs| {
                let len_coeffs = each_coeffs.len() as i32;
                if len_coeffs != num_exp {
                    panic!("effective_nxc_matrix: coeffs.len() != num_exp");
                }
                let enxc_coeff_start = env.len() as i32;
                let mut tmp_enxcbas_vec: Vec<i32> = vec![atm_index as i32, 
                            if angl==enxc_ang_start {-1} else {angl},
                            num_exp,
                            r_exponents,
                            0,
                            enxc_exp_start,
                            enxc_coeff_start,
                            0];
                env.extend(each_coeffs.iter());
                enxcbas.push(tmp_enxcbas_vec);
            });
        }
    });

    println!("{:?}", &enxcbas);

    // now initialize for libcint and libgto, fake with "ecp" 
    let final_cint_atm = &mol.cint_atm;
    let final_cint_bas = &mol.cint_bas;
    let final_cint_env = &mol.cint_env;
    let natm = final_cint_atm.len() as i32;
    let nbas = final_cint_bas.len() as i32;
    let mut cint_data = CINTR2CDATA::new();
    cint_data.set_cint_type(&mol.cint_type);
    let nenxc = enxcbas.len() as i32;
    cint_data.initial_r2c_with_ecp(&final_cint_atm, natm, &final_cint_bas, nbas, &enxcbas, nenxc, &final_cint_env);
    cint_data.cint1e_ecp_optimizer_rust();
    let mut cur_op = String::from("ecp");

    //now fill the ENXC matrix
    let nbas_shell = mol.cint_bas.len();
    for j in 0..nbas_shell {
        let bas_start_j = mol.cint_fdqc[j][0];
        let bas_len_j = mol.cint_fdqc[j][1];
        // for i < j
        for i in 0..j {
            let bas_start_i = mol.cint_fdqc[i][0];
            let bas_len_i = mol.cint_fdqc[i][1];
            let tmp_size = [bas_len_i,bas_len_j];
            let mat_local = MatrixFull::from_vec(tmp_size,
                cint_data.cint_ij(i as i32, j as i32, &cur_op)).unwrap();
            (0..bas_len_j).into_iter().for_each(|tmp_j| {
                let gj = tmp_j + bas_start_j;
                let global_ij_start = gj*(gj+1)/2+bas_start_i;
                let local_ij_start = tmp_j*bas_len_i;
                //let length = if bas_start_i+bas_len_i <= gj+1 {bas_len_i} else {gj+1-bas_start_i};
                let length = bas_len_i;
                let mat_global_j = enxc.get1d_slice_mut(global_ij_start,length).unwrap();
                let mat_local_j = mat_local.get1d_slice(local_ij_start,length).unwrap();
                mat_global_j.iter_mut().zip(mat_local_j.iter()).for_each(|(gij,lij)| {
                    *gij = *lij
                });
            });
        };
        // for i = j 
        let tmp_size = [bas_len_j,bas_len_j];
        let mat_local = MatrixFull::from_vec(tmp_size,
            cint_data.cint_ij(j as i32, j as i32, &cur_op)).unwrap();
        (0..bas_len_j).into_iter().for_each(|tmp_j| {
            let gj = bas_start_j + tmp_j;
            let global_ij_start = gj*(gj+1)/2+bas_start_j;
            let local_ij_start = tmp_j*bas_len_j;
            let length = tmp_j + 1;
            let mat_global_j = enxc.get1d_slice_mut(global_ij_start,length).unwrap();
            let mat_local_j = mat_local.get1d_slice(local_ij_start,length).unwrap();
            mat_global_j.iter_mut().zip(mat_local_j.iter()).for_each(|(gij,lij)| {
                *gij = *lij
            });
        });
    }

    mol.cint_env = org_env;
    mol.cint_atm = org_atm;

    enxc
}

impl PotCellRaw {
    pub fn parse(&self) -> PotCell {
        let to_potcell = PotCell {
            angular_momentum: self.angular_momentum.clone(),
            coeffcients: self.coefficients.iter().map(|x| x.iter().map(|y| y.parse().unwrap()).collect()).collect(),
            r_exponents: self.r_exponents.clone(),
            gaussian_exponents: self.gaussian_exponents.iter().map(|x| x.parse().unwrap()).collect(),
        };

        if to_potcell.angular_momentum.len() != to_potcell.coeffcients.len() {
            panic!("PotCellRaw::parse: angular_momentum.len() != coeffcients.len()");
        };
        if to_potcell.r_exponents.len()!=to_potcell.gaussian_exponents.len() {
            panic!("PotCellRaw::parse: r_exponents.len() != gaussian_exponents.len()");
        };
        to_potcell
    }
}