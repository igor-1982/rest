use core::panic;
use std::fs;
use libc::TCA_DUMP_INVISIBLE;
use rest_libcint::{CintType, CINTR2CDATA};
use serde::{Deserialize, Serialize};
use serde_json::{Result,Value};
use anyhow;
use tensors::{MatrixFull, MatrixUpper, TensorSlice, TensorSliceMut};
use pyo3::pyclass;

use crate::{molecule_io::Molecule, scf_io::SCF};
use crate::constants::{ATM_NUC_MOD_OF, NUC_ECP, SPECIES_INFO};



#[derive(Clone, Debug,Serialize,Deserialize)]
#[pyclass]
pub struct PotCell {
    #[pyo3(get,set)]
    pub angular_momentum: Vec<i32>,
    #[pyo3(get,set)]
    pub coefficients: Vec<Vec<f64>>,
    #[pyo3(get,set)]
    pub r_exponents: Vec<i32>,
    #[pyo3(get,set)]
    pub gaussian_exponents: Vec<f64>,
}

#[derive(Clone, Debug,Serialize,Deserialize)]
#[pyclass]
pub struct ENXC {
    #[pyo3(get,set)]
    pub enxc_potentials: Vec<PotCell>,
    #[pyo3(get,set)]
    pub masscharge: (f64,f64),
    #[pyo3(get,set)]
    pub position: Vec<f64>,
    #[pyo3(get,set)]
    pub atm_index: usize,
}


/// introduce the effective potential for the given atom
///
/// # Arguments
/// * `enxc_potentials` - store the potentials for different angular momentum channels (l, pc_index), sorted by l, 
/// * `masscharge` - mass and charge of the atom (atm_index)
/// * `position` - position of the atom
/// * `atm_index` - index of the atom in the molecule
///
/// # Returns
/// * `ENXC` - the effective potential
///
/// # Info.
/// 
/// \hat{P} = \sum_{l=0}^{n} \hat{P}_{l}  
///  # `l` is the angumar momentum number
/// \hat{P}_{l} = \sum_{i=0}^{n} c_i r^{2} exp(-\alpha_i * r^2)
///  # `{c_i} and {\alpha_i}` are the corresponding coefficients generated by the DeEP^2Net
/// 
///  The coefficients of DeEP^2Net -->>   the coefficients of ENXC 
///         (atm_index, ep_index)  -->>   (atm_index, pc_index, c_i, \alpha_i)
/// 
impl ENXC {
    pub fn init(elem: &String, position: &Vec<f64>, atm_index: usize) -> ENXC {
        ENXC {
            enxc_potentials: vec![],
            masscharge: **SPECIES_INFO.get(&elem[..]).unwrap(),
            position: position.clone(),
            atm_index
        }
    }

    pub fn count_parameters(&self) -> (usize, usize) {
        let num_coeffs = self.enxc_potentials.iter().map(|x| x.coefficients[0].len() * x.coefficients.len()).sum() ;
        let num_gaexps = self.enxc_potentials.iter().map(|x| x.gaussian_exponents.len()).sum();
        (num_coeffs, num_gaexps)
    }

    pub fn add_a_potcell(&mut self, coeff: &Vec<f64>, gaussian_exponents: &Vec<f64>, angular_molentum: i32, r_exponents: i32) {
        self.enxc_potentials.push(PotCell {
            angular_momentum: vec![angular_molentum],
            coefficients: vec![coeff.clone()],
            r_exponents: vec![r_exponents; coeff.len()],
            gaussian_exponents: gaussian_exponents.clone(),
        });
    }
    pub fn change_a_potcell(&mut self, cur_potcell: &PotCell, potcell_index: usize) {
        let mut pot_cell = &mut self.enxc_potentials[potcell_index];
        *pot_cell = cur_potcell.clone()
    }

    pub fn sort_by_angular_momentum(&mut self) {
        self.enxc_potentials.sort_by(|a,b| a.angular_momentum[0].cmp(&b.angular_momentum[0]));
    }

    pub fn get_enxc_potential_by_index(&self, index: usize) -> PotCell {
        self.enxc_potentials[index].clone()
    }

    pub fn get_angular_momentum_range(&self) -> (i32, i32) {
        let angle_max = self.enxc_potentials.iter().map(|x| x.angular_momentum[0]).max().unwrap();
        let angle_min = self.enxc_potentials.iter().map(|x| x.angular_momentum[0]).min().unwrap();
        (angle_max, angle_min)
    }

    pub fn allocate_coeff_index(&self, index: usize) -> (usize, usize, usize) {
        let mut final_coeff_type = 0;
        let mut final_potcell_index = 0;
        let mut final_coeff_index = 0;
        let mut tmp_index: i32 = index as i32;
        self.enxc_potentials.iter().enumerate().for_each(|(potcell_index,x)| {
            let num_coeffs = x.coefficients[0].len() as i32;
            if tmp_index < num_coeffs && tmp_index >= 0 {
                final_coeff_type = 0;
                final_potcell_index = potcell_index;
                final_coeff_index = tmp_index as usize;

                tmp_index = -1;
            } else if tmp_index >= num_coeffs && tmp_index < num_coeffs*2 {
                let tmp_tmp_index = tmp_index - num_coeffs;

                final_coeff_type = 1;
                final_potcell_index = potcell_index;
                final_coeff_index = tmp_tmp_index as usize;

                tmp_index = -1;
            } else {
                tmp_index -= num_coeffs*2
            }
        });

        (final_coeff_type, final_potcell_index, final_coeff_index)

    }

    pub fn allocate_coeff(&self, index: usize) -> (usize, PotCell) {

        let mut coeff_type = 0;
        let mut coeff = 0.0 ;
        let mut coeff_partner = 0.0;
        let mut r_exponent = 0;
        let mut angular_momentum = 0;

        let mut tmp_index: i32 = index as i32;
        self.enxc_potentials.iter().for_each(|x| {
            let num_coeffs = x.coefficients[0].len() as i32;
            if tmp_index < num_coeffs && tmp_index >= 0 {
                coeff_type = 0;
                coeff = x.coefficients[0][tmp_index as usize];
                coeff_partner = x.gaussian_exponents[tmp_index as usize];
                angular_momentum = x.angular_momentum[0];
                r_exponent = x.r_exponents[tmp_index as usize];
                tmp_index = -1;
            } else if tmp_index >= num_coeffs && tmp_index < num_coeffs*2 {
                let tmp_tmp_index = tmp_index - num_coeffs;
                coeff_type = 1;
                coeff_partner = -x.coefficients[0][tmp_tmp_index as usize];
                coeff = x.gaussian_exponents[tmp_tmp_index as usize];
                angular_momentum = x.angular_momentum[0];
                r_exponent = x.r_exponents[tmp_tmp_index as usize];
                tmp_index = -1;
            } else {
                tmp_index -= num_coeffs*2
            }
        });


        (coeff_type, PotCell {
            angular_momentum: vec![angular_momentum],
            coefficients: vec![vec![if coeff_type==0 {coeff} else {coeff_partner}]],
            r_exponents: vec![r_exponent],
            gaussian_exponents: vec![if coeff_type==1 {coeff} else {coeff_partner}],
        })

    }

}

pub fn evaluate_derive_enxc(mol: &mut Molecule, enxc: &Vec<ENXC>, atm_index: usize, coeff_index: usize) -> MatrixUpper<f64> {
    let cur_enxc = &enxc[atm_index];
    let (coeff_type, mut cur_potcell) = cur_enxc.allocate_coeff(coeff_index);

    if coeff_type == 0 {
        cur_potcell.coefficients[0][0] = 1.0;
    } else {
        cur_potcell.r_exponents[0] += 2;
    }

    effective_nxc_for_potcell(mol, &cur_potcell, atm_index)

}


pub fn parse_enxc_from_string(cont: String) -> anyhow::Result<ENXC> {

    let raw:Value = serde_json::from_str(&cont[..])?;
    let tmp_enxc_raw: ENXCRaw = serde_json::from_value(raw)?;

    let enxc_potentials: Vec<PotCell> = tmp_enxc_raw.enxc_potentials.iter().map(|x| x.parse()).collect::<Vec<PotCell>>();

    let masscharge = SPECIES_INFO[&tmp_enxc_raw.elem_type[..]].clone();

    let position: Vec<f64> = tmp_enxc_raw.position.iter().map(|x| x.parse().unwrap()).collect();

    let atm_index =  tmp_enxc_raw.atm_index;

    Ok(ENXC { enxc_potentials, masscharge, position, atm_index})
}

pub fn parse_enxc_potential(file_name: &str) -> anyhow::Result<ENXC> {

    let tmp_cont = fs::read_to_string(file_name)?;
    let mut out_enxc = parse_enxc_from_string(tmp_cont).unwrap();

    out_enxc.sort_by_angular_momentum();

    Ok(out_enxc)

}

pub fn effective_nxc_for_potcell(mol: &mut Molecule, cur_potcell: &PotCell, atm_index: usize) -> MatrixUpper<f64> {
    let org_env = mol.cint_env.clone();
    let org_atm = mol.cint_atm.clone();
    let mut env = &mut mol.cint_env;
    let mut atm = &mut mol.cint_atm;

    atm[atm_index][ATM_NUC_MOD_OF] = NUC_ECP;
    let nbas = mol.fdqc_bas.len();
    let mut enxc = MatrixUpper::new(nbas*(nbas+1)/2, 0.0);

    // now import enxc potentials info.
    let mut enxcbas: Vec<Vec<i32>> = vec![];
    load_enxc_operator_to_cint(env, &mut enxcbas, &cur_potcell, atm_index);
    evaluate_primitive_enxc_operator(&mut enxc, &mol.cint_atm, &mol.cint_bas, 
        &mol.cint_env, &mol.cint_type, &mol.cint_fdqc, &enxcbas);

    mol.cint_env = org_env.clone();
    mol.cint_atm = org_atm.clone();

    enxc
}


pub fn effective_nxc_tensors(mol: &mut Molecule) -> MatrixUpper<f64> {
    let org_env = mol.cint_env.clone();
    let org_atm = mol.cint_atm.clone();

    let nbas = mol.fdqc_bas.len();
    let mut enxc = MatrixUpper::new(nbas*(nbas+1)/2,0.0);

    // now import enxc potentials info.
    let mut enxcbas: Vec<Vec<i32>> = vec![];

    mol.geom.elem.iter().enumerate().for_each(|(atm_index, elem)| {
        let file_name = format!("./enxc/{}.json", atm_index);

        let tmp_enxc = parse_enxc_potential(&file_name[..]).unwrap();

        let (enxc_ang_max, enxc_ang_min) = tmp_enxc.get_angular_momentum_range();
        let enxc_ang_l = if enxc_ang_min == -1 {-1} else {enxc_ang_max};

        mol.cint_atm = org_atm.clone();
        let mut atm = &mut mol.cint_atm;
        atm[atm_index][ATM_NUC_MOD_OF] = NUC_ECP;


        for potcell in tmp_enxc.enxc_potentials.iter() {
            let angl = if potcell.angular_momentum[0]==enxc_ang_l {-1} else {potcell.angular_momentum[0]};
            let coeffs = &potcell.coefficients;
            let r_exponents = *potcell.r_exponents.get(0).unwrap();
            let gaussian_exponents = &potcell.gaussian_exponents;
            let num_exp = gaussian_exponents.len() as i32;
            let num_coeffs = coeffs.len() as i32;

            coeffs.iter().for_each(|each_coeffs| {
                let len_coeffs = each_coeffs.len() as i32;
                if len_coeffs != num_exp {
                    panic!("effective_nxc_matrix: coeffs.len() != num_exp");
                }

                each_coeffs.iter().zip(gaussian_exponents.iter()).for_each(|(coeff, exp)| {

                    let mut enxcbas: Vec<Vec<i32>> = vec![];
                    let mut tmp_enxc = MatrixUpper::new(nbas*(nbas+1)/2,0.0);

                    // now initialize the primitive operator for each pair of (coeff, exp)
                    mol.cint_env = org_env.clone();
                    let mut env = &mut mol.cint_env;

                    let cur_potcell = PotCell {
                        angular_momentum: vec![angl],
                        coefficients: vec![vec![1.0]],
                        r_exponents: vec![r_exponents],
                        gaussian_exponents: vec![*exp],
                    };
                    load_enxc_operator_to_cint(env, &mut enxcbas, &cur_potcell, atm_index);

                    // now generate the corresponding hamiltonian for this primitive operator
                    evaluate_primitive_enxc_operator(&mut tmp_enxc, &mol.cint_atm, &mol.cint_bas, 
                        &mol.cint_env, &mol.cint_type, &mol.cint_fdqc, &enxcbas);

                    enxc.data.iter_mut().zip(tmp_enxc.data.iter()).for_each(|(x, y)| {
                        *x += y*coeff;
                    });
                });

            });
        }
    });

    enxc

}

pub fn effective_nxc_matrix_v02(mol: &mut Molecule, exnc: &Vec<ENXC>) -> MatrixUpper<f64> {
    let org_env = mol.cint_env.clone();
    let org_atm = mol.cint_atm.clone();

    let nbas = mol.fdqc_bas.len();
    let mut enxc = MatrixUpper::new(nbas*(nbas+1)/2,0.0);

    // now import enxc potentials info.
    let mut enxcbas: Vec<Vec<i32>> = vec![];

    let mut atm = &mut mol.cint_atm;
    let mut env = &mut mol.cint_env;

    mol.geom.elem.iter_mut().enumerate().for_each(|(atm_index, elem )| {
        //let file_name = format!("./enxc/{}.json", atm_index);
        //println!("{}",&file_name);
        //let tmp_enxc = parse_enxc_potential(&file_name[..]).unwrap();
        let tmp_enxc = &exnc[atm_index];
        let mut cur_atm = &mut atm[atm_index];

        cur_atm[ATM_NUC_MOD_OF] = NUC_ECP;

        let (enxc_ang_max, enxc_ang_min) = tmp_enxc.get_angular_momentum_range();
        let enxc_ang_l = if enxc_ang_min == -1 {-1} else {enxc_ang_max};

        for potcell in tmp_enxc.enxc_potentials.iter() {
            let angl = if potcell.angular_momentum[0]==enxc_ang_l {-1} else {potcell.angular_momentum[0]};
            let mut cur_potcell = potcell.clone();
            cur_potcell.angular_momentum[0] = angl;

            load_enxc_operator_to_cint(env, &mut enxcbas, &cur_potcell, atm_index);
        }
    });

    evaluate_primitive_enxc_operator(&mut enxc, &mol.cint_atm, &mol.cint_bas, 
        &mol.cint_env, &mol.cint_type, &mol.cint_fdqc, &enxcbas);

    mol.cint_env = org_env;
    mol.cint_atm = org_atm;

    enxc

}

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
        let mut tmp_enxc = parse_enxc_potential(&file_name[..]).unwrap();

        cur_atm[ATM_NUC_MOD_OF] = NUC_ECP;

        let (enxc_ang_max, enxc_ang_min) = tmp_enxc.get_angular_momentum_range();
        let enxc_ang_l = if enxc_ang_min == -1 {-1} else {enxc_ang_max};

        for potcell in tmp_enxc.enxc_potentials.iter_mut() {
            let angl = if potcell.angular_momentum[0]==enxc_ang_l {-1} else {potcell.angular_momentum[0]};
            potcell.angular_momentum[0] = angl;

            load_enxc_operator_to_cint(env, &mut enxcbas, potcell, atm_index);

        }
    });

    evaluate_primitive_enxc_operator(&mut enxc, &mol.cint_atm, &mol.cint_bas, 
        &mol.cint_env, &mol.cint_type, &mol.cint_fdqc, &enxcbas);

    mol.cint_env = org_env;
    mol.cint_atm = org_atm;

    enxc
}

pub fn load_enxc_operator_to_cint(env: &mut Vec<f64>, enxcbas: &mut Vec<Vec<i32>>, potcell: &PotCell, atm_index: usize) {
    let angl = potcell.angular_momentum[0];
    let coeffs = &potcell.coefficients;
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
                    angl,
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



pub fn evaluate_primitive_enxc_operator(enxc: &mut MatrixUpper<f64>, 
    final_cint_atm: &Vec<Vec<i32>>, 
    final_cint_bas: &Vec<Vec<i32>>, 
    final_cint_env: &Vec<f64>, 
    cint_type: &CintType, 
    cint_fdqc: &Vec<Vec<usize>>,
    enxcbas: &Vec<Vec<i32>>, ) {

    let natm = final_cint_atm.len() as i32;
    let nbas_shell = final_cint_bas.len() as i32;

    let mut cint_data = CINTR2CDATA::new();
    cint_data.set_cint_type(cint_type);
    let nenxc = enxcbas.len() as i32;
    cint_data.initial_r2c_with_ecp(&final_cint_atm, natm, &final_cint_bas, nbas_shell, enxcbas, nenxc, &final_cint_env);
    cint_data.cint1e_ecp_optimizer_rust();
    let mut cur_op = String::from("ecp");

    //now fill the ENXC matrix
    //let nbas_shell = mol.cint_bas.len();
    for j in 0..nbas_shell as usize{
        let bas_start_j = cint_fdqc[j][0];
        let bas_len_j = cint_fdqc[j][1];
        // for i < j
        for i in 0..j {
            let bas_start_i = cint_fdqc[i][0];
            let bas_len_i = cint_fdqc[i][1];
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
}


#[derive(Clone, Debug,Serialize,Deserialize)]
pub struct PotCellRaw {
    pub angular_momentum: Vec<i32>,
    pub coefficients: Vec<Vec<String>>,
    pub r_exponents: Vec<i32>,
    pub gaussian_exponents: Vec<String>,
}
impl PotCellRaw {
    pub fn parse(&self) -> PotCell {
        let to_potcell = PotCell {
            angular_momentum: self.angular_momentum.clone(),
            coefficients: self.coefficients.iter().map(|x| x.iter().map(|y| y.parse().unwrap()).collect()).collect(),
            r_exponents: self.r_exponents.clone(),
            gaussian_exponents: self.gaussian_exponents.iter().map(|x| x.parse().unwrap()).collect(),
        };

        if to_potcell.angular_momentum.len() != to_potcell.coefficients.len() {
            panic!("PotCellRaw::parse: angular_momentum.len() != coefficients.len()");
        };
        if to_potcell.r_exponents.len()!=to_potcell.gaussian_exponents.len() {
            panic!("PotCellRaw::parse: r_exponents.len() != gaussian_exponents.len()");
        };
        to_potcell
    }
}

#[derive(Clone, Debug,Serialize,Deserialize)]
pub struct ENXCRaw {
    pub enxc_potentials: Vec<PotCellRaw>,
    pub elem_type: String,
    pub position: Vec<String>,
    pub atm_index: usize
}
