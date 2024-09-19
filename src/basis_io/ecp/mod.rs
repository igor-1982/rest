use std::{fs, path::PathBuf};

use hdf5::file;
use pyo3::{pyclass, pymethods};
use rest_libcint::{prelude::ECPscalar, CintType, CINTR2CDATA};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tensors::{MatrixFull, MatrixUpper};
use time::ParseResult;

use crate::{constants::NUC_ECP, geom_io::GeomCell, molecule_io::Molecule};

pub fn ghost_effective_potential_matrix(
    env: &Vec<f64>, 
    atm: &Vec<Vec<i32>>, 
    bas: &Vec<Vec<i32>>, 
    cint_type: &CintType,
    nbas: usize, 
    ep_path: &Vec<String>, 
    ep_pos: &MatrixFull<f64>) -> MatrixUpper<f64> {

    //let nbas = mol.fdqc_bas.len();
    let mut gp_matr = MatrixUpper::new(nbas*(nbas+1)/2,0.0);

    let (final_cint_env, final_cint_atm, gpbas) = initialize_gp_operator_for_cint(&env, &atm, &ep_path, &ep_pos);
    //let final_cint_bas = &mut mol.cint_bas;

    evaluate_gp_matrix(&mut gp_matr, &final_cint_atm, bas, &final_cint_env, &cint_type, &gpbas);

    gp_matr
}

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
pub struct PotCellRaw {
    pub angular_momentum: Vec<i32>,
    pub coefficients: Vec<Vec<String>>,
    pub r_exponents: Vec<i32>,
    pub gaussian_exponents: Vec<String>,
}

#[pymethods]
impl PotCell {
    #[new]
    pub fn new(coeffs: Vec<f64>, gaussian_exponents: Vec<f64>, angular_momentum: i32, r_exponent: i32) -> Self {
        PotCell {
            coefficients: vec![coeffs],
            gaussian_exponents,
            angular_momentum: vec![angular_momentum],
            r_exponents: vec![r_exponent] 
        }
    }
    pub fn py_set_coeff(&mut self, coeff: Vec<f64>) {
        self.coefficients[0]=coeff;
    }

    pub fn py_set_gaussian_exponents(&mut self, gaussian_exponents: Vec<f64>) {
        self.gaussian_exponents = gaussian_exponents;
    }

    pub fn py_set_r_exponents(&mut self, r_exponents: i32) {
        self.r_exponents = vec![r_exponents];
    }
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


#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GhostPotential {
    pub potentials: Vec<PotCell>,
    pub position: Option<[f64;3]>
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GhostPotentialRaw {
    pub potentials: Vec<PotCellRaw>,
    pub position: Option<[f64;3]>
}

impl GhostPotential {
    pub fn sort_by_angular_momentum(&mut self) {
        self.potentials.sort_by(|a, b| a.angular_momentum.cmp(&b.angular_momentum));
    }
}

fn parse_gp_from_string(cont: String) -> anyhow::Result<GhostPotential> {

    let raw:Value = serde_json::from_str(&cont[..])?;
    let tmp_gp_raw: GhostPotentialRaw = serde_json::from_value(raw)?;

    let potentials: Vec<PotCell> = tmp_gp_raw.potentials.iter().map(|x| x.parse()).collect::<Vec<PotCell>>();

    Ok(GhostPotential {
        potentials,
        position: tmp_gp_raw.position
    })
}

fn parse_ghost_potential(file_name: &str) -> anyhow::Result<GhostPotential> {
    let tmp_cont = fs::read_to_string(file_name).unwrap();
    let mut out_gp = parse_gp_from_string(tmp_cont).unwrap();
    out_gp.sort_by_angular_momentum();
    Ok(out_gp)
}

fn loading_gp_operator_per_atom(gpbas: &mut Vec<Vec<i32>>, env: &mut Vec<f64>, cint_atm: &mut Vec<Vec<i32>>, gpot: &GhostPotential, position: &[f64]) {
    //let position = gpot.position.unwrap();
    let mut gp_atm_start = env.len() as i32;
    let gp_atm_index = cint_atm.len() as i32;

    cint_atm.push(vec![0, gp_atm_start, NUC_ECP, gp_atm_start+3, 0, 0]);
    env.extend(position);

    
    for potcell in gpot.potentials.iter() {
        let angl = potcell.angular_momentum[0];
        let coeffs = &potcell.coefficients;
        let r_exponents = *potcell.r_exponents.get(0).unwrap();
        let gaussian_exponents = &potcell.gaussian_exponents;
        let num_exp = gaussian_exponents.len() as i32;
        let num_coeffs = coeffs.len() as i32;

        let gp_exp_start = env.len() as i32;
        env.extend(gaussian_exponents.iter());

        coeffs.iter().for_each(|each_coeffs| {
            let len_coeffs = each_coeffs.len() as i32;
            if len_coeffs != num_exp {
                panic!("effective_nxc_matrix: coeffs.len() != num_exp");
            }
            let gp_coeff_start = env.len() as i32;
            let mut tmp_gpbas_vec: Vec<i32> = vec![gp_atm_index as i32, 
                        angl,
                        num_exp,
                        r_exponents,
                        0,
                        gp_exp_start,
                        gp_coeff_start,
                        0];
            env.extend(each_coeffs.iter());
            gpbas.push(tmp_gpbas_vec);
        });
    }
}

fn initialize_gp_operator_for_cint(env: &Vec<f64>, atm: &Vec<Vec<i32>>, ep_path: &Vec<String>, ep_pos: &MatrixFull<f64>) -> (Vec<f64>, Vec<Vec<i32>>, Vec<Vec<i32>>) {
    let mut out_env = env.clone();
    let mut out_atm = atm.clone();
    let mut gpbas: Vec<Vec<i32>> = vec![];
    ep_path.iter().zip(ep_pos.iter_columns_full()).for_each(|(path, position)| {
        let file_path = PathBuf::from(path);
        if file_path.exists()  {
            let tmp_cont = fs::read_to_string(path).unwrap();
            let mut tmp_potcell = parse_gp_from_string(tmp_cont).unwrap();
            tmp_potcell.sort_by_angular_momentum();
            loading_gp_operator_per_atom(&mut gpbas, &mut out_env, &mut out_atm, &tmp_potcell, position);
        } else {
            panic!("ghost_effective_potential: file not found: {:?}", file_path);
        }
    });

    (out_env, out_atm, gpbas)
}


fn evaluate_gp_matrix(gp_matr: &mut MatrixUpper<f64>, 
    final_cint_atm: &Vec<Vec<i32>>, 
    final_cint_bas: &Vec<Vec<i32>>, 
    final_cint_env: &Vec<f64>, 
    cint_type: &CintType, 
    gpbas: &Vec<Vec<i32>>) {

    let natm = final_cint_atm.len() as i32;
    let nbas_shell = final_cint_bas.len() as i32;
    let ngp = gpbas.len() as i32;

    let mut cint_data = CINTR2CDATA::new();
    cint_data.set_cint_type(cint_type);
    cint_data.initial_r2c_with_ecp(&final_cint_atm, natm, &final_cint_bas, nbas_shell, gpbas, ngp, &final_cint_env);
    cint_data.cint1e_ecp_optimizer_rust();

    let (out, shape) = cint_data.integral_ecp_s1::<ECPscalar>(None);
    let out_matr = MatrixFull::from_vec(shape.try_into().unwrap(), out).unwrap();
    gp_matr.iter_mut().zip(out_matr.iter_matrixupper().unwrap()).for_each(|(o,i)| {*o = *i});
}

#[test]
fn test_parse_gp() {
    //"position": [[0.0, 0.0, 0.0]],
    let cont = r#"{
        "potentials": [
          {
            "angular_momentum": [0],
            "coefficients": [
              ["0.0"]
            ],
            "r_exponents": [0],
            "gaussian_exponents": ["0.0"]
          }
        ]
        }"#.to_string();
    let raw:Value = serde_json::from_str(&cont[..]).unwrap();
    println!("debug raw: {:?}", raw);
    let dd: GhostPotential = serde_json::from_value(raw).unwrap();
    println!("{:?}", dd);
}

