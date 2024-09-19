use rest_tensors::matrix_blas_lapack::_dgemm_nn;
use rest_tensors::{MatrixFull, RIFull};
use itertools::izip;
use rayon::prelude::*;
use serde::{Deserialize,Serialize};
use serde_json::{Result,Value};
use anyhow;
use tensors::{MathMatrix,ParMathMatrix};
use tensors::matrix_blas_lapack::_dgemm_nn_serial;
use std::collections::HashMap;
use std::convert::TryInto;
use std::fs;
use std::io::{Write,BufRead, BufReader};
use rest_libcint::{CINTR2CDATA, CintType};
use std::f64::consts::{PI, E};
use libm;
use crate::constants::{C2S_L0, C2S_L1, C2S_L2, C2S_L3, C2S_L4, c2s_matrix_const, CarBasInfo, cartesian_gto_const, self};
use crate::utilities;
pub mod bse_downloader;
pub mod basis_list;
pub mod etb;
pub mod ecp;
use self::basic_math::{double_factorial, specific_double_factorial};

//use crate::geom_io::GeomCell;
//use ndarray::{Axis};

#[derive(Debug,Clone,Serialize,Deserialize)]
pub struct BasCellRaw {
    pub function_type: Option<String>,
    pub region: Option<String>,
    pub angular_momentum: Vec<i32>,
    pub exponents: Vec<String>,
    pub coefficients: Vec<Vec<String>>,
}
#[derive(Serialize,Deserialize,Clone, Debug)]
pub struct RefCell {
    pub reference_description: Option<String>,
    pub reference_keys: Option<Vec<String>>,
}

#[derive(Clone, Debug,Serialize,Deserialize)]
pub struct ECPCellRaw {
    pub angular_momentum: Vec<i32>,
    pub coefficients: Vec<Vec<String>>,
    pub ecp_type: Option<String>,
    pub r_exponents: Vec<i32>,
    pub gaussian_exponents: Vec<String>,
}
#[derive(Debug,Clone,Serialize,Deserialize)]
pub struct Basis4ElemRaw {
    pub electron_shells: Vec<BasCellRaw>,
    pub references: Option<Vec<RefCell>>,
    pub ecp_potentials: Option<Vec<ECPCellRaw>>,
    pub ecp_electrons: Option<usize>,
}

#[test]
fn import_ecp()-> anyhow::Result<()> {
    let tmp_string = fs::read_to_string(String::from("/home/igor/Documents/Package-Pool/rest_workspace/rest/basis-set-pool/def2-SVP/Au.json"))?;
    let tmp_basis:Basis4ElemRaw = serde_json::from_str(&tmp_string[..])?;
    if let (Some(ecp_electrons), Some(ecp_potentials))= (&tmp_basis.ecp_electrons, &tmp_basis.ecp_potentials)  {
        println!("debug ecp electrons: {}", &ecp_electrons);
        ecp_potentials.iter().for_each(|ecp| {
            println!("debug ecp: {:?}", ecp.parse());
        });
    };
    Ok(())
}

/// #BasCell
/// `BasCell` has the same organization structure of the basis set files from the basis set exchange (BSE) 
///  - BasCell.function_type:     "gto_spherical" or "gto"  
///  - BasCell.region:  
///  - BasCell.angular_momentum:  The angular momentums of the GTOs in this cell  
///  - BasCell.exponents:         The exponents  
///  - BasCell.coefficients:      The coefficients 
#[derive(Debug,Clone)]
pub struct BasCell {
    pub function_type: Option<String>,
    pub region: Option<String>,
    pub angular_momentum: Vec<i32>,
    pub exponents: Vec<f64>,
    pub coefficients: Vec<Vec<f64>>,
    pub native_coefficients: Vec<Vec<f64>>,
}

#[derive(Clone, Debug,Serialize,Deserialize)]
pub struct ECPCell {
    pub angular_momentum: Vec<i32>,
    pub coefficients: Vec<Vec<f64>>,
    pub ecp_type: Option<String>,
    pub r_exponents: Vec<i32>,
    pub gaussian_exponents: Vec<f64>,
}

///  # BasInfo
/// `BasInfo` contains the elemental information of each group of GTOs, including
///   self.bas_name: The GTOs in the group has the same angular momentum. 
///                  It is the name for the angularm momentum, for example, S, P, D, F, and so forth.
///   self.bas_type: The GTOs are `Primitive` or `Contractive`
///   self.elem_index0: The index of atom where the GTOs are located
///   self.cint_index0: The global start index of this basis function group 
///   self.cint_index1: The number of basis functions in this group
#[derive(Clone, Debug)]
pub struct BasInfo {
    pub bas_name: String,
    pub bas_type: String,
    pub elem_index0: usize,
    pub cint_index0: usize,
    pub cint_index1: usize,
}

impl  BasInfo {
    pub fn new() -> BasInfo {
        BasInfo {
            bas_name : String::from("S"),
            bas_type : String::from("Primitive"),
            elem_index0 : 0,
            cint_index0 : 0,
            cint_index1 : 0,
        }
    }
    pub fn formated_name(&self) -> String {
        format!("{}-e{}-cint{}-{}",self.bas_name,self.elem_index0, self.cint_index0,self.cint_index1)
    }
}

/// #Basis4Elem
/// `Basis4Elem` contains all informations of the GTO basis functions located in a given element
///     self.electron_shells: the GTO basis functions organized cell by cell [`BasCell`](BasCell)
///     self.references: The reference of each basis cell.
///     self.global_index: The global index for the given atom
#[derive(Clone, Debug)]
pub struct Basis4Elem {
    pub electron_shells: Vec<BasCell>,
    pub references: Option<Vec<RefCell>>,
    pub ecp_potentials: Option<Vec<ECPCell>>,
    pub ecp_electrons: Option<usize>,
    pub global_index: (usize,usize)
}

impl Basis4Elem {
    pub fn parse_json(json: &String)-> anyhow::Result<Basis4Elem> {
        let tmp_basis: Basis4ElemRaw = serde_json::from_str(&json)?;
        let mut tmp_vec:Vec<BasCell> = vec![];
        &tmp_basis.electron_shells.iter().for_each(|x: &BasCellRaw| {
            tmp_vec.extend(x.parse());
        });
        let (ecp_electrons, ecp_potentials) = if let (Some(ecp_electrons), Some(ecp_potentials))= (&tmp_basis.ecp_electrons, &tmp_basis.ecp_potentials)  {
            (
                Some(ecp_electrons.clone()), 
                Some(ecp_potentials.iter().map(|x| x.parse()).collect::<Vec<ECPCell>>())
            )
        } else {
            (None, None)
        };
        //println!("debug 0");
        if let Some(ecp_electrons_unwrap)=&ecp_electrons {
            println!("ecp_electrons: {}", &ecp_electrons_unwrap)
        };
        if let Some(ecp_potentials_unwrap)=&ecp_potentials {
            for tmp_ecp in ecp_potentials_unwrap {
                println!("ecp_potentials: {:?}", tmp_ecp);
            }
        };
        Ok(Basis4Elem{
            electron_shells: tmp_vec,
            references: tmp_basis.references,
            ecp_electrons,
            ecp_potentials,
            global_index: (0,0)
        })
    }
    pub fn parse_json_from_file(file_name: String, cint_type: &CintType)-> anyhow::Result<Basis4Elem> {
        let tmp_cont = fs::read_to_string(&file_name[..])?;
        //let tmp_basis:Basis4ElemRaw = serde_json::from_str(&tmp_cont[..])?;
        let raw:Value = serde_json::from_str(&tmp_cont[..])?;
        let tmp_basis:Basis4ElemRaw = if !raw["electron_shells"].is_null() {
            serde_json::from_value(raw)?
        } else if !raw["elements"].is_null() {
            // Treat json from `bse convert-basis`, which looks like
            //     { "elements": { "1": { "electron_shells": [ ... ] } }
            // assume there's only one element
            serde_json::from_value(raw["elements"].as_object().unwrap().values().next().unwrap().clone())?
        } else {
            panic!("The basis set file is not in the right format: {:?}", &file_name);
        };
        let mut tmp_vec:Vec<BasCell> = vec![];
        &tmp_basis.electron_shells.iter().for_each(|x: &BasCellRaw| {
            let tmp_bas_cell = x.parse();
            //println!("{:?}",tmp_bas_cell);
            for mut x_bascell in tmp_bas_cell {
                x_bascell.basis_normalization(cint_type);
                tmp_vec.push(x_bascell);
            }
        });
        let (ecp_electrons, ecp_potentials) = if let (Some(ecp_electrons), Some(ecp_potentials))= (&tmp_basis.ecp_electrons, &tmp_basis.ecp_potentials)  {
            (
                Some(ecp_electrons.clone()), 
                Some(ecp_potentials.iter().map(|x| x.parse()).collect::<Vec<ECPCell>>())
            )
        } else {
            (None, None)
        };
        //println!("debug 1");
        //if let Some(ecp_electrons_unwrap)=&ecp_electrons {
        //    println!("ecp_electrons: {}", &ecp_electrons_unwrap)
        //};
        //if let Some(ecp_potentials_unwrap)=&ecp_potentials {
        //    for tmp_ecp in ecp_potentials_unwrap {
        //        println!("ecp_potentials: {:?}", tmp_ecp);
        //    }
        //};
        // Re-ordering the basis function shells according to the angular momentum
        tmp_vec.sort_by(|a,b| a.angular_momentum[0].cmp(&b.angular_momentum[0]));
        Ok(Basis4Elem{
            electron_shells: tmp_vec,
            references: tmp_basis.references,
            ecp_electrons,
            ecp_potentials,
            global_index: (0,0)
        })
    }
    pub fn to_numgrid_io(&self) -> (HashMap<usize,f64>,f64) {
        let mut alpha_max = -std::f64::MAX;
        let mut alpha_min = HashMap::new();
        self.electron_shells.iter().for_each(|value| {
            let angular_momentum = value.angular_momentum[0] as usize;
            value.exponents.iter().for_each(|exponent| {
                alpha_max = alpha_max.max(*exponent);
                let s = alpha_min.entry(angular_momentum).or_insert(std::f64::MAX);
                if exponent < s {
                    *s = *exponent;
                }
            });
        });
        (alpha_min,alpha_max)
    }


}


impl ECPCellRaw {
    pub fn parse(&self) -> ECPCell {
        let to_ecpcell = ECPCell {
            angular_momentum: self.angular_momentum.clone(),    
            ecp_type: self.ecp_type.clone(),
            r_exponents: self.r_exponents.clone(),
            coefficients: {
                self.coefficients.iter().map(|i| i.iter().map(|j| j.parse().unwrap()).collect::<Vec<f64>>()).collect::<Vec<Vec<f64>>>()
            },
            gaussian_exponents: {
                self.gaussian_exponents.iter().map(|i| i.parse().unwrap()).collect::<Vec<f64>>()
            },
        };
        if to_ecpcell.angular_momentum.len()!=to_ecpcell.coefficients.len() {
            panic!("The number of angular momentum is not equal to the number of coefficients\n")
        };
        if to_ecpcell.r_exponents.len()!=to_ecpcell.gaussian_exponents.len() {
            panic!("The number of r_exponents is not equal to the number of gaussian_exponents\n")
        };
        to_ecpcell
    }
}


impl BasCellRaw {
    pub fn parse(&self) -> Vec<BasCell> {
        let mut bascell_vec: Vec<BasCell> = vec![];
        let mut tmp_vec_0: Vec<i32> = self.angular_momentum.clone();

        // parse the vector of exponents
        let mut tmp_vec_1 = self.exponents.iter()
            .map(|i| i.parse().unwrap())
            .collect::<Vec<f64>>();

        // parse the vector of coefficients
        let mut tmp_vec_2 = self.coefficients.iter().map(|i| i.iter()
            .map(|j| j.parse().unwrap()).collect::<Vec<f64>>())
            .collect::<Vec<Vec<f64>>>();

        if self.angular_momentum.len()>1 && self.angular_momentum.len()==self.coefficients.len() {
            // for Pople basis sets (for example, 6-31G)
            for i_bas in (0..self.angular_momentum.len()) {
                bascell_vec.push(
                    BasCell { 
                        function_type: self.function_type.clone(), 
                        region: self.region.clone(), 
                        angular_momentum: vec![tmp_vec_0[i_bas]], 
                        exponents: tmp_vec_1.clone(), 
                        coefficients: vec![tmp_vec_2[i_bas].clone()],
                        native_coefficients: vec![tmp_vec_2[i_bas].clone()]
                    }
                );
            }
        } else if self.angular_momentum.len()>1 && self.angular_momentum.len()!=self.coefficients.len(){
            // improper setting for the Pople basis sets
            panic!("The basis set with a wrong format: The number of angular momentums is more than one, but is unequal to the number of the coefficient set\n
                    {:?}", &self);
        } else if self.angular_momentum.len()==1 {
            // for any other standard basis set
            //let mut tmp_vec_2_0:Vec<Vec<f64>> = vec![];
            //let mut tmp_vec_2_1:Vec<Vec<f64>> = vec![];
            //for i_coeff in (0..tmp_vec_2.len()) {
            //    //let full_power = tmp_vec_2[i_coeff].iter().fold(true,|check,i| check && *i!=0.0f64);
            //    let full_power = true;
            //    if full_power {
            //        tmp_vec_2_0.push(tmp_vec_2[i_coeff].clone())
            //    } else {
            //        tmp_vec_2_1.push(tmp_vec_2[i_coeff].clone())
            //    };
            //}
            // consider the contracted GTOs that use all primitive GTOs in the given shell
            bascell_vec.push(
                BasCell { 
                    function_type: self.function_type.clone(), 
                    region: self.region.clone(), 
                    angular_momentum: tmp_vec_0.clone(), 
                    exponents: tmp_vec_1.clone(), 
                    coefficients: tmp_vec_2.clone(), 
                    native_coefficients: tmp_vec_2.clone()
                }
            );
            //// now consider the contracted (or primitive) GTOs that use part of primitive GTOs in the given shell
            //for i_coeff in (0..tmp_vec_2_1.len()) {
            //    let mut tmp_vec_1_1:Vec<f64> = vec![];
            //    let mut tmp_vec_2_2:Vec<f64> = vec![];
            //    for (i,j) in tmp_vec_2_1[i_coeff].iter().zip(tmp_vec_1.iter()) {
            //        if *i!=0.0 {
            //            tmp_vec_1_1.push(*j);
            //            tmp_vec_2_2.push(*i);
            //        }
            //    }
            //    bascell_vec.push(
            //        BasCell { 
            //            function_type: self.function_type.clone(), 
            //            region: self.region.clone(), 
            //            angular_momentum: tmp_vec_0.clone(), 
            //            exponents: tmp_vec_1_1.clone(), 
            //            coefficients: vec![tmp_vec_2_2.clone()] 
            //        }
            //    );
            //}
        } else {
            panic!("The angular momentum is missing for the basis set : {:?}", &self);
        }
        bascell_vec
    }
}

impl BasCell {
   pub fn basis_normalization(&mut self, cint_type: &CintType) {
       /// To use libcint properly, we should rescale the coefficients of the contractive GTOs,
       /// A detailed interpreation is necessary, which, however, is missing at present.
       let mut bas_start: i32 = 0;
       let mut atm: Vec<Vec<i32>> = vec![vec![1,0,0,0,0,0]];
       let mut env: Vec<f64> = vec![0.0_f64,0.0,0.0]; 
       bas_start += 3;
       let bas: Vec<Vec<i32>> = vec![vec![0,
                                self.angular_momentum[0],
                                self.exponents.len() as i32, 
                                self.coefficients.len() as i32,
                                0,
                                bas_start,
                                bas_start + self.exponents.len() as i32,
                                0
                                ]];
        env.extend(self.exponents.iter());
        //self.exponents.iter().for_each(|x| {
        //    env.push(*x);
        //});
        //(0..self.exponents.len()).into_iter().for_each(|x| {
        //    println!("gto_norm for {}: {}",self.exponents[x],
        //        CINTR2CDATA::gto_norm(self.angular_momentum[0] as std::os::raw::c_int,self.exponents[x]));
        //});
        //println!(" before basis-normalization: self.coefficients {:?}",self.coefficients);
        let mut tmp_exponents = self.exponents.to_owned();
        if self.angular_momentum.len()>1 && self.angular_momentum.len()==self.coefficients.len() {
            // for Pople basis sets (for example, 6-31G)
            for i_bas in (0..self.angular_momentum.len()) {
                let mut tmp_ang = self.angular_momentum[i_bas];
                let mut tmp_coefficients: Vec<Vec<f64>> = vec![];
                for coe_vec in self.coefficients.iter() {
                    let mut tmp_coefficients_column: Vec<f64> = vec![];
                    coe_vec.iter().enumerate().for_each(|(ix,x)| {
                        let tmp_value = CINTR2CDATA::gto_norm(tmp_ang as std::os::raw::c_int,
                                                          self.exponents[ix]);
                        //println!("gto_norm for {}: {}",tmp_exponents[ix],tmp_value);
                        env.push(*x*tmp_value);
                        tmp_coefficients_column.push(*x*tmp_value);
                        //env.push(*x);
                        //tmp_coefficients_column.push(*x);
                    });
                    tmp_coefficients.push(tmp_coefficients_column);
                };
            }
        } else if self.angular_momentum.len()>1 && self.angular_momentum.len()!=self.coefficients.len(){
            // improper setting for the Pople basis sets
            panic!("Error: the Pople basis set is specified with a wrong format: {:?}", &self);
        } else if self.angular_momentum.len()==1 {
            // for any other standard basis set
            let mut tmp_ang = self.angular_momentum[0];
            let mut tmp_coefficients: Vec<Vec<f64>> = vec![];
            //println!("before basis-normalization: self.coefficients {:?}",self.coefficients);
            //for (index, coe_vec) in self.coefficients.iter().enumerate() {
            for coe_vec in self.coefficients.iter() {
                let mut tmp_coefficients_column: Vec<f64> = vec![];
                coe_vec.iter().enumerate().for_each(|(ix,x)| {
                    let tmp_value = CINTR2CDATA::gto_norm(tmp_ang as std::os::raw::c_int,
                                                      self.exponents[ix]);
                    //println!("gto_norm for {}: {}",tmp_exponents[ix],tmp_value);
                    env.push(*x*tmp_value);
                    tmp_coefficients_column.push(*x*tmp_value);
                    //env.push(*x);
                    //tmp_coefficients_column.push(*x);
                });
                tmp_coefficients.push(tmp_coefficients_column);
            };
            self.coefficients = tmp_coefficients.clone();

            // now for normalization
            let mut cint_data = CINTR2CDATA::new();
            let natm = atm.len() as i32;
            let nbas = bas.len() as i32;
            let (ang,n_len) = match &cint_type {
                CintType::Cartesian => {let ang = tmp_ang as usize; (ang,(ang+1)*(ang+2)/2)},
                CintType::Spheric => {let ang = tmp_ang as usize; (ang, ang*2+1)},
                _ => panic!("The angular momentum is missing for the basis set : {:?}", &self)
            };
            //let n_len = (tmp_ang*2+1) as usize;
            cint_data.initial_r2c(&atm, natm, &bas, nbas, &env);
            //cint_data.set_cint_type(CintType::Spheric);
            cint_data.set_cint_type(&cint_type);
            cint_data.cint1e_ovlp_optimizer_rust();
            let num_bas = self.coefficients.len();
            let num_pri = self.exponents.len();
            let buf:Vec<f64> = cint_data.cint_ij(0, 0,&String::from("ovlp"));
            //println!("buf: {:?}",&buf);
            //println!("before coef {:?}",&self.coefficients);
            (0..num_bas).into_iter().for_each(|x| {
                let xx = (n_len*num_bas)*(x*n_len)+n_len*x;
                let norm_fac = 1.0_f64/buf.get(xx).unwrap().sqrt();
                //println!("debug norm: {}", norm_fac);
                self.coefficients[x].iter_mut().for_each(|y| *y *= norm_fac);
            });
            //println!("after coef {:?}",&self.coefficients);
            cint_data.final_c2r();
            //println!(" after basis-normalization: self.coefficients {:?}",self.coefficients);
        } else {
            panic!("The angular momentum is missing for the basis set : {:?}", &self);
        }
   }
}

/// produce the value of cartesian gaussian-type orbital "gau(a,l,c)" in a given coordinate "r=(x,y,z)"
///
///       gau(a,l,c) = Norm*x^{lx}*y^{ly}*z^{lz}*exp[-a*r^2],
///                  = Norm*Fang*exp[-a*r^2],
///
/// The GTO is normalized with the normalization factor "Norm" as
///
///   Norm = (2a/Pi)^(3/4)*(4a)^{l/2}/((2*lx-1)!!(2*ly-1)!!(2*lz-1)!!)^{1/2}
///
pub fn cartesian_gto_std(a: f64, l: usize, c:&[f64;3],r:&[f64;3]) -> MatrixFull<f64> {
    // produce the value of cartesian gaussian-type orbital "gau(a,l,c)" in a given coordinate "r=(x,y,z)"
    //
    //       gau(a,l,c) = Norm*x^{lx}*y^{ly}*z^{lz}*exp[-a*r^2],
    //
    // The GTO is normalized with the normalization factor "Norm" as
    //
    //   Norm = (2a/Pi)^(3/4)*(4a)^{l/2}/((2*lx-1)!!(2*ly-1)!!(2*lz-1)!!)^{1/2}
    //

    // results in the same order that are used in libcint
    let mut tmp_v: Vec<f64> = vec![];
    let mut rr:[f64;3] = r.clone();
    rr.iter_mut().zip(c.iter()).for_each(|(r,c)| *r -= c);
    let mut rdot = rr.iter().fold(0.0f64,|acc,rr| acc + rr*rr);
    let exp_part = libm::exp(-a*rdot);
    let norm0 = (2.0*a/PI).powf(0.75)*(4.0*a).powf((l as f64)/2.0);
    for lx in (0..=l).rev() {
        let xp = rr[0].powf(lx as f64);
        //debug here
        let normx = double_factorial((2*lx) as i32-1) as f64;
        let rl = l - lx;
        for ly in (0..=rl).rev() {
            let lz = rl - ly;
            let normy = double_factorial((2*ly) as i32-1) as f64;
            let normz = double_factorial((2*lz) as i32-1) as f64;

            //tmp_v.push(
            //        xp*rr[1].powf(ly as f64)*rr[2].powf(lz as f64)
            //        *exp_part);
            tmp_v.push(norm0/(normx*normy*normz).sqrt()
                    *xp*rr[1].powf(ly as f64)*rr[2].powf(lz as f64)
                    *exp_part);
            //println!("debug norm 2: {}", norm0/(normx*normy*normz).sqrt());
        }
    }
    unsafe{MatrixFull::from_vec_unchecked([(l+1)*(l+2)/2,1], tmp_v)}
}

pub fn cartesian_gto_batch(a: f64, l: usize, c:&[f64;3],r:&[[f64;3]]) -> MatrixFull<f64> {
    // produce the value of cartesian gaussian-type orbital "gau(a,l,c)" in a given coordinate "r=(x,y,z)"
    //
    //       gau(a,l,c) = Norm*x^{lx}*y^{ly}*z^{lz}*exp[-a*r^2],
    //                  = Norm*Fang*exp[-a*r^2],
    //
    // The GTO is normalized with the normalization factor "Norm" as
    //
    //   Norm = (2a/Pi)^(3/4)*(4a)^{l/2}/((2*lx-1)!!(2*ly-1)!!(2*lz-1)!!)^{1/2}
    //

    // results in the same order that are used in libcint
    let num_grids = r.len();
    let num_bas = (l+1)*(l+2)/2;
    let norm0 = (2.0*a/PI).powf(0.75)*(4.0*a).powf((l as f64)/2.0);
    //let mut tmp_v: Vec<f64> = vec![];
    let mut tmp_mat = MatrixFull::new([num_grids,num_bas],0.0);
    let mut i_bas:usize = 0;
    for lx in (0..=l).rev() {
        let normx = double_factorial((2*lx) as i32-1) as f64;
        let lxf = lx as f64;
        let rl = l - lx;
        for ly in (0..=rl).rev() {
            let lz = rl - ly;
            let normy = double_factorial((2*ly) as i32-1) as f64;
            let normz = double_factorial((2*lz) as i32-1) as f64;
            let norm  = norm0/(normx*normy*normz).sqrt();

            let lyf = ly as f64;
            let lzf = lz as f64;
            tmp_mat.iter_column_mut(i_bas).zip(r.iter()).for_each(|(bas_r,r)| {
                //r.iter().zip(c.iter())
                let mut rdot = 0.0;
                let mut fang = 1.0;
                izip!(r.iter(),c.iter(),[lxf,lyf,lzf].iter()).for_each(|(rx,cx,lx)| {
                    let rrx = rx-cx;
                    rdot += rrx*rrx;
                    fang *= rrx.powf(*lx);
                });
                let exp_part = libm::exp(-a*rdot);
                *bas_r = norm*fang*exp_part;
            });
            i_bas +=1;
        }
    }
    tmp_mat
}

#[inline]
pub fn cartesian_gto_batch_par(a: f64, l: usize, c:&[f64;3],r:&[[f64;3]]) -> MatrixFull<f64> {
    // produce the value of cartesian gaussian-type orbital "gau(a,l,c)" in a given coordinate "r=(x,y,z)"
    //
    //       gau(a,l,c) = Norm*x^{lx}*y^{ly}*z^{lz}*exp[-a*r^2],
    //                  = Norm*Fang*exp[-a*r^2],
    //
    // The GTO is normalized with the normalization factor "Norm" as
    //
    //   Norm = (2a/Pi)^(3/4)*(4a)^{l/2}/((2*lx-1)!!(2*ly-1)!!(2*lz-1)!!)^{1/2}
    //

    // results in the same order that are used in libcint
    let num_grids = r.len();
    let num_bas = (l+1)*(l+2)/2;
    let norm0 = (2.0*a/PI).powf(0.75)*(4.0*a).powf((l as f64)/2.0);
    //let mut tmp_v: Vec<f64> = vec![];
    let mut tmp_mat = MatrixFull::new([num_grids,num_bas],0.0);
    let mut i_bas:usize = 0;
    //let mut lf = [0.0,0.0,0.0];
    for lx in (0..=l).rev() {
        //let normx = double_factorial((2*lx) as i32-1) as f64;
        let normx = specific_double_factorial((2*lx) as i32-1);
        let lxf = lx as f64;
        //lf[0] = lx as f64;
        let rl = l - lx;
        for ly in (0..=rl).rev() {
            let lz = rl - ly;
            let normy = specific_double_factorial((2*ly) as i32-1);
            let normz = specific_double_factorial((2*lz) as i32-1);
            let norm  = norm0/(normx*normy*normz).sqrt();
            //println!("debug: {:16.8},{:16.8},{:2},{:16.8}, {:16.8}, {:16.8},{:16.8}",cint_norm_factor(l as i32, a), a,l, norm0, normx, normy,normz);

            //lf[1] = ly as f64;
            //lf[2] = (lz-ly) as f64;
            let lyf = ly as f64;
            let lzf = lz as f64;
            tmp_mat.par_iter_column_mut(i_bas).zip(r.par_iter()).for_each(|(bas_r,r)| {
                //r.iter().zip(c.iter())
                //let mut rdot = 0.0;
                //let mut fang = 1.0;
                //izip!(r.iter(),c.iter(),[lxf,lyf,lzf].iter()).for_each(|(rx,cx,lx)| {
                //    let rrx = rx-cx;
                //    rdot += rrx.powf(2.0);
                //    fang *= rrx.powf(*lx);
                //});
                let (rdot, fang) = izip!(r.iter(),c.iter(),[lxf,lyf,lzf].iter())
                //let (rdot, fang) = izip!(r.iter(),c.iter(),(&lf).iter())
                    .fold((0.0,1.0), |(rdot, fang), (rx,cx,lx)| {
                        let rrx = rx-cx;
                        (rdot + rrx.powf(2.0), fang * rrx.powf(*lx))
                });
                let exp_part = libm::exp(-a*rdot);
                *bas_r = norm*fang*exp_part;
            });
            i_bas +=1;
        }
    }
    tmp_mat
}
#[inline]
pub fn cartesian_gto_batch_par_v02(a: f64, l: usize, c:&[f64;3],r:&Vec<[f64;3]>) -> MatrixFull<f64> {
    // produce the value of cartesian gaussian-type orbital "gau(a,l,c)" in a given coordinate "r=(x,y,z)"
    //
    //       gau(a,l,c) = Norm*x^{lx}*y^{ly}*z^{lz}*exp[-a*r^2],
    //                  = Norm*Fang*exp[-a*r^2],
    //
    // The GTO is normalized with the normalization factor "Norm" as
    //
    //   Norm = (2a/Pi)^(3/4)*(4a)^{l/2}/((2*lx-1)!!(2*ly-1)!!(2*lz-1)!!)^{1/2}
    //

    // results in the same order that are used in libcint
    let num_grids = r.len();
    let num_bas = (l+1)*(l+2)/2;
    let norm0 = (2.0*a/PI).powf(0.75)*(4.0*a).powf((l as f64)/2.0);

    let binding = cartesian_gto_const(l);
    let basinfo = &binding.to_matrixfullslice();;

    let mut tmp_mat = MatrixFull::new([num_grids,num_bas],0.0_f64);
    tmp_mat.iter_columns_full_mut().zip(basinfo.iter_columns_full()).for_each(|(bas_r,bas_i)| {
        bas_r.par_iter_mut().zip(r.par_iter()).for_each(|(bas_r,r)| {
            let (rdot, fang) = izip!(r.iter(),c.iter(),bas_i[0..3].iter())
                .fold((0.0,1.0), |(rdot, fang), (rx,cx,lx)| {
                    let rrx = rx-cx;
                    (rdot + rrx.powf(2.0), fang * rrx.powf(*lx))
            });
            let exp_part = libm::exp(-a*rdot);
            *bas_r = norm0*bas_i[3]*fang*exp_part;
        })
    });
    tmp_mat
}
#[inline]
pub fn cartesian_gto_batch_par_v03(a: f64, l: usize, c:&[f64;3],r:&[[f64;3]]) -> MatrixFull<f64> {
    // produce the value of cartesian gaussian-type orbital "gau(a,l,c)" in a given coordinate "r=(x,y,z)"
    //
    //       gau(a,l,c) = Norm*x^{lx}*y^{ly}*z^{lz}*exp[-a*r^2],
    //                  = Norm*Fang*exp[-a*r^2],
    //
    // The GTO is normalized with the normalization factor "Norm" as
    //
    //   Norm = (2a/Pi)^(3/4)*(4a)^{l/2}/((2*lx-1)!!(2*ly-1)!!(2*lz-1)!!)^{1/2}
    //

    // results in the same order that are used in libcint
    let num_grids = r.len();
    let num_bas = (l+1)*(l+2)/2;
    let norm0 = (2.0*a/PI).powf(0.75)*(4.0*a).powf((l as f64)/2.0);
    let cut_off:f64 = (constants::E9).log(constants::E)/a;

    let binding = cartesian_gto_const(l);
    let basinfo = &binding.to_matrixfullslice();;

    let mut tmp_mat = MatrixFull::new([num_bas,num_grids],0.0_f64);
    tmp_mat.par_iter_columns_full_mut().zip(r.par_iter())
        .filter(|(r_bas,r)| {
            let rr = r.iter().zip(c.iter()).fold(0.0, |rdot, (r,c)| {rdot + (r-c).powf(2.0_f64)});
            rr < cut_off
            //true
        }).for_each(|(r_bas,r)| {
            let mut rr = [0.0;3];
            let rdot = izip!(rr.iter_mut(),r.iter(),c.iter()).fold(0.0, |rdot, (rr,r,c)| {
                *rr= r-c;
                rdot + rr.powf(2.0_f64)}
            );
            let exp_part = norm0*libm::exp(-a*rdot);
            r_bas.iter_mut().zip(basinfo.iter_columns_full()).for_each(|(bas_r,bas_i)| {
                let fang = rr.iter().zip(bas_i[0..3].iter())
                    .fold(1.0, |fang, (rr,lx)| {
                        fang * rr.powf(*lx)
                });
                *bas_r = bas_i[3]*fang*exp_part;
            });
    });
    tmp_mat.transpose_and_drop()

}
pub fn cartesian_gto_batch_v03(a: f64, l: usize, c:&[f64;3],r:&[[f64;3]]) -> MatrixFull<f64> {
    // produce the value of cartesian gaussian-type orbital "gau(a,l,c)" in a given coordinate "r=(x,y,z)"
    //
    //       gau(a,l,c) = Norm*x^{lx}*y^{ly}*z^{lz}*exp[-a*r^2],
    //                  = Norm*Fang*exp[-a*r^2],
    //
    // The GTO is normalized with the normalization factor "Norm" as
    //
    //   Norm = (2a/Pi)^(3/4)*(4a)^{l/2}/((2*lx-1)!!(2*ly-1)!!(2*lz-1)!!)^{1/2}
    //

    // results in the same order that are used in libcint
    let num_grids = r.len();
    let num_bas = (l+1)*(l+2)/2;
    let norm0 = (2.0*a/PI).powf(0.75)*(4.0*a).powf((l as f64)/2.0);
    let cut_off:f64 = (constants::E9).log(constants::E)/a;

    let binding = cartesian_gto_const(l);
    let basinfo = &binding.to_matrixfullslice();;

    let mut tmp_mat = MatrixFull::new([num_bas,num_grids],0.0_f64);
    tmp_mat.iter_columns_full_mut().zip(r.iter())
        //.filter(|(r_bas,r)| {
        //    let rr = r.iter().zip(c.iter()).fold(0.0, |rdot, (r,c)| {rdot + (r-c).powf(2.0_f64)});
        //    //rr < cut_off
        //    true
        //}).for_each(|(r_bas,r)| {
        .for_each(|(r_bas,r)| {
            let mut rr = [0.0;3];
            let rdot = izip!(rr.iter_mut(),r.iter(),c.iter()).fold(0.0, |rdot, (rr,r,c)| {
                *rr= r-c;
                rdot + rr.powf(2.0_f64)}
            );
            let exp_part = norm0*libm::exp(-a*rdot);
            r_bas.iter_mut().zip(basinfo.iter_columns_full()).for_each(|(bas_r,bas_i)| {
                let fang = rr.iter().zip(bas_i[0..3].iter())
                    .fold(1.0, |fang, (rr,lx)| {
                        fang * rr.powf(*lx)
                });
                *bas_r = bas_i[3]*fang*exp_part;
            });
    });
    tmp_mat.transpose_and_drop()

}

pub fn cartesian_gto_batch_v04(a: f64, l: usize, c:&[f64;3],r:&[[f64;3]]) -> MatrixFull<f64> {
    // produce the value of cartesian gaussian-type orbital "gau(a,l,c)" in a given coordinate "r=(x,y,z)"
    //
    //       gau(a,l,c) = Norm*x^{lx}*y^{ly}*z^{lz}*exp[-a*r^2],
    //                  = Norm*Fang*exp[-a*r^2],
    //
    // The GTO is normalized with the normalization factor "Norm" as
    //
    //   Norm = (2a/Pi)^(3/4)*(4a)^{l/2}/((2*lx-1)!!(2*ly-1)!!(2*lz-1)!!)^{1/2}
    //

    // results in the same order that are used in libcint
    let num_grids = r.len();
    let num_bas = (l+1)*(l+2)/2;
    let norm0 = (2.0*a/PI).powf(0.75)*(4.0*a).powf((l as f64)/2.0);
    let cut_off:f64 = (constants::E7).log(constants::E)/a;

    let binding = cartesian_gto_const(l);
    let basinfo = &binding.to_matrixfullslice();;

    let mut tmp_mat = MatrixFull::new([num_bas,num_grids],0.0_f64);
    tmp_mat.iter_columns_full_mut().zip(r.iter())
        //.filter(|(r_bas,r)| {
        //    let rr = r.iter().zip(c.iter()).fold(0.0, |rdot, (r,c)| {rdot + (r-c).powf(2.0_f64)});
        //    rr < cut_off
        //    //true
        //}).for_each(|(r_bas,r)| {
        .for_each(|(r_bas,r)| {
            let mut rr = [0.0;3];
            let rdot = izip!(rr.iter_mut(),r.iter(),c.iter()).fold(0.0, |rdot, (rr,r,c)| {
                *rr= r-c;
                rdot + rr.powf(2.0_f64)}
            );
            let exp_part = norm0*libm::exp(-a*rdot);
            r_bas.iter_mut().zip(basinfo.iter_columns_full()).for_each(|(bas_r,bas_i)| {
                let fang = rr.iter().zip(bas_i[0..3].iter())
                    .fold(1.0, |fang, (rr,lx)| {
                        fang * rr.powf(*lx)
                });
                *bas_r = bas_i[3]*fang*exp_part;
            });
    });
    tmp_mat

}

pub fn cartesian_gto_1st_batch(a: f64, l: usize, c:&[f64;3],r:&[[f64;3]]) -> Vec<MatrixFull<f64>> {
    let num_grids = r.len();
    let num_bas = (l+1)*(l+2)/2;
    let norm0 = (2.0*a/PI).powf(0.75)*(4.0*a).powf((l as f64)/2.0);
    //let mut pao = RIFull::new([num_grids,num_bas,3],0.0);
    let mut aox =  MatrixFull::new([num_grids,num_bas], 0.0);
    let mut aoy =  MatrixFull::new([num_grids,num_bas], 0.0);
    let mut aoz =  MatrixFull::new([num_grids,num_bas], 0.0);
    let ll = l as i32;
    let mut i_bas:usize = 0;
    for lx in (0..=ll).rev() {
        let lxf = lx as f64;
        let normx = double_factorial(2*lx-1) as f64;
        let rl = ll - lx;
        for ly in (0..=rl as i32).rev() {
            let lz = rl - ly;
            let lyf = ly as f64;
            let lzf = lz as f64;
            let normy = double_factorial(2*ly-1) as f64;
            let normz = double_factorial(2*lz-1) as f64;
            let norm  = norm0/(normx*normy*normz).sqrt();
            izip!(aox.iter_column_mut(i_bas),aoy.iter_column_mut(i_bas),aoz.iter_column_mut(i_bas),r.iter())
            .for_each(|(aox_r,aoy_r,aoz_r,r)| {

                let mut rdot = 0.0;
                let mut xyz = [0.0;3];
                let mut pxyz = [0.0;3];
                izip!(r.iter(),c.iter(),[lxf,lyf,lzf].iter(),xyz.iter_mut(),pxyz.iter_mut())
                    .for_each(|(rx,cx,lx,x,px)| {
                    let rrx = rx-cx;
                    rdot += rrx*rrx;
                    *x = rrx.powf(*lx);
                    *px = -2.0*a*rrx.powf(lx+1.0);
                    *px += if (*lx!=0.0) {lx*rrx.powf(lx-1.0)} else {0.0};
                });
                let exp_part = norm*libm::exp(-a*rdot);
                let [x,y,z] = xyz;
                let [px,py,pz] = pxyz;
                *aox_r = exp_part*px*y*z;
                *aoy_r = exp_part*x*py*z;
                *aoz_r = exp_part*x*y*pz;
            });
            i_bas +=1;
        }
    }
    vec![aox,aoy,aoz]
}
#[inline]
pub fn cartesian_gto_1st_batch_par(a: f64, l: usize, c:&[f64;3],r:&[[f64;3]]) -> Vec<MatrixFull<f64>> {
    let num_grids = r.len();
    let num_bas = (l+1)*(l+2)/2;
    let norm0 = (2.0*a/PI).powf(0.75)*(4.0*a).powf((l as f64)/2.0);
    //let mut pao = RIFull::new([num_grids,num_bas,3],0.0);
    let mut aox =  MatrixFull::new([num_grids,num_bas], 0.0);
    let mut aoy =  MatrixFull::new([num_grids,num_bas], 0.0);
    let mut aoz =  MatrixFull::new([num_grids,num_bas], 0.0);

    //let binding = cartesian_gto_const(l);
    //let basinfo = &binding.to_matrixfullslice();;
    let ll = l as i32;
    let mut i_bas:usize = 0;
    for lx in (0..=ll).rev() {
        let lxf = lx as f64;
        let normx = double_factorial(2*lx-1) as f64;
        let rl = ll - lx;
        for ly in (0..=rl as i32).rev() {
            let lz = rl - ly;
            let lyf = ly as f64;
            let lzf = lz as f64;
            let normy = double_factorial(2*ly-1) as f64;
            let normz = double_factorial(2*lz-1) as f64;
            let norm  = norm0/(normx*normy*normz).sqrt();

            //let (sender, receiver) = channel();
            aox.par_iter_column_mut(i_bas).zip(aoy.par_iter_column_mut(i_bas))
            .map(|(aox_r,aoy_r)| (aox_r,aoy_r)).zip(
            aoz.par_iter_column_mut(i_bas).zip(r.par_iter())
            .map(|(aoz_r,r)| (aoz_r,r)))
            .for_each(|((aox_r,aoy_r),(aoz_r,r))| {
                let mut rdot = 0.0;
                let mut xyz = [0.0;3];
                let mut pxyz = [0.0;3];
                izip!(r.iter(),c.iter(),[lxf,lyf,lzf].iter(),xyz.iter_mut(),pxyz.iter_mut())
                    .for_each(|(rx,cx,lx,x,px)| {
                    let rrx = rx-cx;
                    rdot += rrx*rrx;
                    *x = rrx.powf(*lx);
                    *px = -2.0*a*rrx.powf(lx+1.0);
                    *px += if (*lx!=0.0) {lx*rrx.powf(lx-1.0)} else {0.0};
                });
                let exp_part = norm*libm::exp(-a*rdot);
                let [x,y,z] = xyz;
                let [px,py,pz] = pxyz;
                *aox_r = exp_part*px*y*z;
                *aoy_r = exp_part*x*py*z;
                *aoz_r = exp_part*x*y*pz;

            });

            i_bas +=1;
        }
    }
    vec![aox,aoy,aoz]
}

#[inline]
pub fn cartesian_gto_1st_batch_par_v02(a: f64, l: usize, c:&[f64;3],r:&[[f64;3]]) -> Vec<MatrixFull<f64>> {
    let num_grids = r.len();
    let num_bas = (l+1)*(l+2)/2;
    let norm0 = (2.0*a/PI).powf(0.75)*(4.0*a).powf((l as f64)/2.0);
    let cut_off:f64 = (constants::E9).log(constants::E)/a;

    //let mut pao = RIFull::new([num_grids,num_bas,3],0.0);
    //let mut aox =  MatrixFull::new([num_grids,num_bas], 0.0);
    //let mut aoy =  MatrixFull::new([num_grids,num_bas], 0.0);
    //let mut aoz =  MatrixFull::new([num_grids,num_bas], 0.0);

    let binding = cartesian_gto_const(l);
    let basinfo = &binding.to_matrixfullslice();;

    let mut aox =  MatrixFull::new([num_bas,num_grids], 0.0);
    let mut aoy =  MatrixFull::new([num_bas,num_grids], 0.0);
    let mut aoz =  MatrixFull::new([num_bas,num_grids], 0.0);

    aox.par_iter_columns_full_mut()
    .zip(aoy.par_iter_columns_full_mut())
    .zip(aoz.par_iter_columns_full_mut())
    .zip(r.par_iter()).map(|(((aox_r,aoy_r),aoz_r),r)| (aox_r,aoy_r,aoz_r,r))
    .filter(|(aox_r,aoy_r,aoz_r,r)| {
            let rr = r.iter().zip(c.iter()).fold(0.0, |rdot, (r,c)| {rdot + (r-c).powf(2.0_f64)});
            rr < cut_off
            //true
    }).for_each(|(aox_r,aoy_r,aoz_r,r)| {
        let mut rr = [0.0;3];
        let rdot = izip!(rr.iter_mut(),r.iter(),c.iter()).fold(0.0, |rdot, (rr,r,c)| {
            *rr= r-c;
            rdot + rr.powf(2.0_f64)}
        );
        let exp_part = norm0*libm::exp(-a*rdot);
        let mut xyz = [0.0;3];
        let mut pxyz = [0.0;3];
        aox_r.iter_mut().zip(aoy_r.iter_mut()).zip(aoz_r.iter_mut())
        .zip(basinfo.iter_columns_full())
        .map(|((((aox_r,aoy_r),aoz_r)),bas_i)| (aox_r, aoy_r, aoz_r,bas_i))
        .for_each(|(aox_r,aoy_r,aoz_r,bas_i)| {
            //xyz = [0.0;3];
            //pxyz = [0.0;3];
            izip!(xyz.iter_mut(),pxyz.iter_mut(),bas_i[0..3].iter(),rr.iter())
                .for_each(|(x,px,lx,rrx)| {
                    *x = rrx.powf(*lx);
                    *px = -2.0*a*rrx.powf(lx+1.0);
                    *px += if (*lx!=0.0) {lx*rrx.powf(lx-1.0)} else {0.0};
                });
            let [x,y,z] = xyz;
            let [px,py,pz] = pxyz;
            *aox_r = bas_i[3]*exp_part*px*y*z;
            *aoy_r = bas_i[3]*exp_part*x*py*z;
            *aoz_r = bas_i[3]*exp_part*x*y*pz;
        });
    });

    vec![aox.transpose_and_drop(),aoy.transpose_and_drop(),aoz.transpose_and_drop()]
}

#[inline]
pub fn cartesian_gto_1st_batch_v02(a: f64, l: usize, c:&[f64;3],r:&[[f64;3]]) -> Vec<MatrixFull<f64>> {
    let num_grids = r.len();
    let num_bas = (l+1)*(l+2)/2;
    let norm0 = (2.0*a/PI).powf(0.75)*(4.0*a).powf((l as f64)/2.0);
    let cut_off:f64 = (constants::E9).log(constants::E)/a;

    //let mut pao = RIFull::new([num_grids,num_bas,3],0.0);
    //let mut aox =  MatrixFull::new([num_grids,num_bas], 0.0);
    //let mut aoy =  MatrixFull::new([num_grids,num_bas], 0.0);
    //let mut aoz =  MatrixFull::new([num_grids,num_bas], 0.0);

    let binding = cartesian_gto_const(l);
    let basinfo = &binding.to_matrixfullslice();;

    let mut aox =  MatrixFull::new([num_bas,num_grids], 0.0);
    let mut aoy =  MatrixFull::new([num_bas,num_grids], 0.0);
    let mut aoz =  MatrixFull::new([num_bas,num_grids], 0.0);

    aox.iter_columns_full_mut()
    .zip(aoy.iter_columns_full_mut())
    .zip(aoz.iter_columns_full_mut())
    .zip(r.iter()).map(|(((aox_r,aoy_r),aoz_r),r)| (aox_r,aoy_r,aoz_r,r))
    .filter(|(aox_r,aoy_r,aoz_r,r)| {
            let rr = r.iter().zip(c.iter()).fold(0.0, |rdot, (r,c)| {rdot + (r-c).powf(2.0_f64)});
            //rr < cut_off
            true
    }).for_each(|(aox_r,aoy_r,aoz_r,r)| {
        let mut rr = [0.0;3];
        let rdot = izip!(rr.iter_mut(),r.iter(),c.iter()).fold(0.0, |rdot, (rr,r,c)| {
            *rr= r-c;
            rdot + rr.powf(2.0_f64)}
        );
        let exp_part = norm0*libm::exp(-a*rdot);
        let mut xyz = [0.0;3];
        let mut pxyz = [0.0;3];
        aox_r.iter_mut().zip(aoy_r.iter_mut()).zip(aoz_r.iter_mut())
        .zip(basinfo.iter_columns_full())
        .map(|((((aox_r,aoy_r),aoz_r)),bas_i)| (aox_r, aoy_r, aoz_r,bas_i))
        .for_each(|(aox_r,aoy_r,aoz_r,bas_i)| {
            //xyz = [0.0;3];
            //pxyz = [0.0;3];
            izip!(xyz.iter_mut(),pxyz.iter_mut(),bas_i[0..3].iter(),rr.iter())
                .for_each(|(x,px,lx,rrx)| {
                    *x = rrx.powf(*lx);
                    *px = -2.0*a*rrx.powf(lx+1.0);
                    *px += if (*lx!=0.0) {lx*rrx.powf(lx-1.0)} else {0.0};
                });
            let [x,y,z] = xyz;
            let [px,py,pz] = pxyz;
            *aox_r = bas_i[3]*exp_part*px*y*z;
            *aoy_r = bas_i[3]*exp_part*x*py*z;
            *aoz_r = bas_i[3]*exp_part*x*y*pz;
        });
    });

    vec![aox.transpose_and_drop(),aoy.transpose_and_drop(),aoz.transpose_and_drop()]
}
#[inline]
pub fn cartesian_gto_1st_batch_v03(a: f64, l: usize, c:&[f64;3],r:&[[f64;3]]) -> Vec<MatrixFull<f64>> {
    let num_grids = r.len();
    let num_bas = (l+1)*(l+2)/2;
    let norm0 = (2.0*a/PI).powf(0.75)*(4.0*a).powf((l as f64)/2.0);
    let cut_off:f64 = (constants::E7).log(constants::E)/a;

    //let mut pao = RIFull::new([num_grids,num_bas,3],0.0);
    //let mut aox =  MatrixFull::new([num_grids,num_bas], 0.0);
    //let mut aoy =  MatrixFull::new([num_grids,num_bas], 0.0);
    //let mut aoz =  MatrixFull::new([num_grids,num_bas], 0.0);

    let binding = cartesian_gto_const(l);
    let basinfo = &binding.to_matrixfullslice();;

    let mut aox =  MatrixFull::new([num_bas,num_grids], 0.0);
    let mut aoy =  MatrixFull::new([num_bas,num_grids], 0.0);
    let mut aoz =  MatrixFull::new([num_bas,num_grids], 0.0);

    aox.iter_columns_full_mut()
    .zip(aoy.iter_columns_full_mut())
    .zip(aoz.iter_columns_full_mut())
    .zip(r.iter()).map(|(((aox_r,aoy_r),aoz_r),r)| (aox_r,aoy_r,aoz_r,r))
    //.filter(|(aox_r,aoy_r,aoz_r,r)| {
    //        let rr = r.iter().zip(c.iter()).fold(0.0, |rdot, (r,c)| {rdot + (r-c).powf(2.0_f64)});
    //        rr < cut_off
    //        //true
    //}).for_each(|(aox_r,aoy_r,aoz_r,r)| {
    .for_each(|(aox_r,aoy_r,aoz_r,r)| {
        let mut rr = [0.0;3];
        let rdot = izip!(rr.iter_mut(),r.iter(),c.iter()).fold(0.0, |rdot, (rr,r,c)| {
            *rr= r-c;
            rdot + rr.powf(2.0_f64)}
        );
        let exp_part = norm0*libm::exp(-a*rdot);
        let mut xyz = [0.0;3];
        let mut pxyz = [0.0;3];
        aox_r.iter_mut().zip(aoy_r.iter_mut()).zip(aoz_r.iter_mut())
        .zip(basinfo.iter_columns_full())
        .map(|((((aox_r,aoy_r),aoz_r)),bas_i)| (aox_r, aoy_r, aoz_r,bas_i))
        .for_each(|(aox_r,aoy_r,aoz_r,bas_i)| {
            //xyz = [0.0;3];
            //pxyz = [0.0;3];
            izip!(xyz.iter_mut(),pxyz.iter_mut(),bas_i[0..3].iter(),rr.iter())
                .for_each(|(x,px,lx,rrx)| {
                    *x = rrx.powf(*lx);
                    *px = -2.0*a*rrx.powf(lx+1.0);
                    *px += if (*lx!=0.0) {lx*rrx.powf(lx-1.0)} else {0.0};
                });
            let [x,y,z] = xyz;
            let [px,py,pz] = pxyz;
            *aox_r = bas_i[3]*exp_part*px*y*z;
            *aoy_r = bas_i[3]*exp_part*x*py*z;
            *aoz_r = bas_i[3]*exp_part*x*y*pz;
        });
    });

    vec![aox,aoy,aoz]
}

pub fn cartesian_gto_1st_std(a: f64, l: usize, c:&[f64;3],r:&[f64;3]) -> [MatrixFull<f64>;3] {
    let mut gto_dx: Vec<f64> = vec![];
    let mut gto_dy: Vec<f64> = vec![];
    let mut gto_dz: Vec<f64> = vec![];
    let mut rr:[f64;3] = r.clone();
    rr.iter_mut().zip(c.iter()).for_each(|(r,c)| *r -= c);
    let mut rdot = rr.iter().fold(0.0f64,|acc,rr| acc + rr*rr);
    let exp_part = libm::exp(-a*rdot);
    let norm0 = (2.0*a/PI).powf(0.75)*(4.0*a).powf((l as f64)/2.0);
    for lx in (0..=l as i32).rev() {
        let lx_f64 = lx as f64;
        let xp = rr[0].powf(lx_f64);
        let mut xp_1s = -2.0*a*rr[0].powf(lx_f64+1.0);
        xp_1s += if (lx != 0) {lx_f64*rr[0].powf(lx_f64-1.0)} else {0.0};
        //debug here
        let normx = double_factorial(2*lx-1) as f64;
        let rl = l as i32 - lx;
        for ly in (0..=rl as i32).rev() {
            let lz = rl - ly;
            let ly_f64 = ly as f64;
            let lz_f64 = lz as f64;
            let normy = double_factorial(2*ly-1) as f64;
            let normz = double_factorial(2*lz-1) as f64;
            let norm = norm0/(normx*normy*normz).sqrt();

            let yp = rr[1].powf(ly_f64);
            let mut yp_1s = -2.0*a*rr[1].powf(ly_f64+1.0);
            yp_1s += if (ly != 0) {ly_f64*rr[1].powf(ly_f64-1.0)} else {0.0};
            let zp = rr[2].powf(lz_f64);
            let mut zp_1s = -2.0*a*rr[2].powf(lz_f64+1.0);
            zp_1s += if (lz != 0)  {lz_f64*rr[2].powf(lz_f64-1.0)} else {0.0};

            //tmp_v.push(
            //        xp*rr[1].powf(ly as f64)*rr[2].powf(lz as f64)
            //        *exp_part);
            gto_dx.push(norm*xp_1s*yp*zp*exp_part);
            gto_dy.push(norm*xp*yp_1s*zp*exp_part);
            gto_dz.push(norm*xp*yp*zp_1s*exp_part);
            //println!("debug norm 2: {}", norm0/(normx*normy*normz).sqrt());
            //println!("debug: {},{},{}", xp_1s, yp_1s, zp_1s);
            //println!("yp_1s comp: ly_f64: {}, rr[1]: {}, a: {}", ly_f64, rr[1], a);
            //println!("yp_1s comp: ly_f64: {}, rr[1]: {}, a: {}", ly_f64, rr[1].powf(ly_f64-1.0), a);
        }
    }
    //println!("debug: gto_d(x,y,z): ({:?}, {:?}, {:?})", &gto_dx, &gto_dy, &gto_dz);
    [unsafe{MatrixFull::from_vec_unchecked([(l+1)*(l+2)/2,1], gto_dx)},
     unsafe{MatrixFull::from_vec_unchecked([(l+1)*(l+2)/2,1], gto_dy)},
     unsafe{MatrixFull::from_vec_unchecked([(l+1)*(l+2)/2,1], gto_dz)}]

}

pub fn cartesian_gto_cint(a: f64, l: usize, c:&[f64;3],r:&[f64;3]) -> MatrixFull<f64> {
    // WARNNING: In libcint, the normalization of radial part should be defined in coefficients.
    //           In other words, the native GTO integrals from libcint are not normlized with respect to "\int dr^3 r^2 g(r)"
    //           In consequence, 
    let mut std_gto = cartesian_gto_std(a, l, c, r);
    std_gto.self_multiple(cint_norm_factor(l as i32, a));
    std_gto
}

pub fn cartesian_gto_cint_batch_rayon(a: f64, l: usize, c:&[f64;3],r:&[[f64;3]]) -> MatrixFull<f64> {
    // WARNNING: In libcint, the normalization of radial part should be defined in coefficients.
    //           In other words, the native GTO integrals from libcint are not normlized with respect to "\int dr^3 r^2 g(r)"
    //           In consequence, 
    //let mut time_records = utilities::TimeRecords::new();

    //time_records.new_item("1-1-1", "cartesian_gto_batch_par");
    //time_records.count_start("1-1-1");
    let mut std_gto = cartesian_gto_batch_par_v03(a, l, c, r);

    //time_records.new_item("1-1-2", "self_multiple/norm_factor");
    //time_records.count("1-1-1");
    //time_records.count_start("1-1-2");
    std_gto.self_multiple(cint_norm_factor(l as i32, a));
    //time_records.count("1-1-2");

    //time_records.report_all();

    std_gto
}
pub fn cartesian_gto_cint_batch(a: f64, l: usize, c:&[f64;3],r:&[[f64;3]]) -> MatrixFull<f64> {
    // WARNNING: In libcint, the normalization of radial part should be defined in coefficients.
    //           In other words, the native GTO integrals from libcint are not normlized with respect to "\int dr^3 r^2 g(r)"
    //           In consequence, 
    let mut std_gto = cartesian_gto_batch_v03(a, l, c, r);

    std_gto.self_multiple(cint_norm_factor(l as i32, a));

    std_gto
}

pub fn cartesian_gto_cint_batch_v04(a: f64, l: usize, c:&[f64;3],r:&[[f64;3]]) -> MatrixFull<f64> {
    // WARNNING: In libcint, the normalization of radial part should be defined in coefficients.
    //           In other words, the native GTO integrals from libcint are not normlized with respect to "\int dr^3 r^2 g(r)"
    //           In consequence, 
    let mut std_gto = cartesian_gto_batch_v04(a, l, c, r);

    std_gto.self_multiple(cint_norm_factor(l as i32, a));

    std_gto
}

pub fn cartesian_gto_1st_cint(a: f64, l: usize, c:&[f64;3],r:&[f64;3]) -> [MatrixFull<f64>;3] {
    // WARNNING: In libcint, the normalization of radial part should be defined in coefficients.
    //           In other words, the native GTO integrals from libcint are not normlized with respect to "\int dr^3 r^2 g(r)"
    //           In consequence, 
    let mut std_gto_1st = cartesian_gto_1st_std(a, l, c, r);
    //println!("debug: {:?}, {:?}, {:?}", &std_gto_1st[0].data,&std_gto_1st[1].data,&std_gto_1st[2].data);
    std_gto_1st.iter_mut().for_each(|std_gto| {
        std_gto.self_multiple(cint_norm_factor(l as i32, a))
    });
    std_gto_1st
}

pub fn cartesian_gto_1st_cint_batch(a: f64, l: usize, c:&[f64;3],r:&[[f64;3]]) -> Vec<MatrixFull<f64>> {
    // WARNNING: In libcint, the normalization of radial part should be defined in coefficients.
    //           In other words, the native GTO integrals from libcint are not normlized with respect to "\int dr^3 r^2 g(r)"
    //           In consequence, 
    let mut std_gto_1st = cartesian_gto_1st_batch_par_v02(a, l, c, r);
    //println!("debug: {:?}, {:?}, {:?}", &std_gto_1st[0].data,&std_gto_1st[1].data,&std_gto_1st[2].data);
    let fac = cint_norm_factor(l as i32, a);
    //std_gto_1st.data.iter_mut().for_each(|value| {*value *=fac});
    std_gto_1st.iter_mut().for_each(|std_gto| {
        std_gto.self_multiple(fac)
    });
    std_gto_1st
}

pub fn cartesian_gto_1st_cint_batch_serial(a: f64, l: usize, c:&[f64;3],r:&[[f64;3]]) -> Vec<MatrixFull<f64>> {
    // WARNNING: In libcint, the normalization of radial part should be defined in coefficients.
    //           In other words, the native GTO integrals from libcint are not normlized with respect to "\int dr^3 r^2 g(r)"
    //           In consequence, 
    let mut std_gto_1st = cartesian_gto_1st_batch_v02(a, l, c, r);
    //println!("debug: {:?}, {:?}, {:?}", &std_gto_1st[0].data,&std_gto_1st[1].data,&std_gto_1st[2].data);
    let fac = cint_norm_factor(l as i32, a);
    //std_gto_1st.data.iter_mut().for_each(|value| {*value *=fac});
    std_gto_1st.iter_mut().for_each(|std_gto| {
        std_gto.self_multiple(fac)
    });
    std_gto_1st
}

pub fn cartesian_gto_1st_cint_batch_serial_v03(a: f64, l: usize, c:&[f64;3],r:&[[f64;3]]) -> Vec<MatrixFull<f64>> {
    // WARNNING: In libcint, the normalization of radial part should be defined in coefficients.
    //           In other words, the native GTO integrals from libcint are not normlized with respect to "\int dr^3 r^2 g(r)"
    //           In consequence, 
    let mut std_gto_1st = cartesian_gto_1st_batch_v03(a, l, c, r);
    //println!("debug: {:?}, {:?}, {:?}", &std_gto_1st[0].data,&std_gto_1st[1].data,&std_gto_1st[2].data);
    let fac = cint_norm_factor(l as i32, a);
    //std_gto_1st.data.iter_mut().for_each(|value| {*value *=fac});
    std_gto_1st.iter_mut().for_each(|std_gto| {
        std_gto.self_multiple(fac)
    });
    std_gto_1st
}


pub fn spheric_gto_value_matrixfull(r: &[[f64;3]],gto_center:&[f64;3], bas:&Basis4Elem) -> MatrixFull<f64> {

    //let mut time_records = utilities::TimeRecords::new();
    //time_records.new_item("1-1", "cartesian_gto_cint_batch");
    //time_records.new_item("1-2", "_dgemm_nn");

    let num_grids:usize = r.len();
    let mut num_bas_c = 0;
    let mut num_bas_s = 0;
    bas.electron_shells.iter().for_each(|ibas| {
        let iang = ibas.angular_momentum[0] as usize;
        num_bas_c += (iang+1)*(iang+2)/2*ibas.coefficients.len();
        num_bas_s += (iang*2+1)*ibas.coefficients.len();
    });
    //println!("debug num_bas_c and num_bas_s: {},{}", num_bas_c, num_bas_s);
    //let mut outmat_c = MatrixFull::new([num_grids,num_bas_c],0.0);
    let mut outmat_s = MatrixFull::new([num_grids,num_bas_s],0.0);
    let mut ibas_start = 0;
    for ibas in &bas.electron_shells {
        let iang = ibas.angular_momentum[0] as usize;
        //let mut c2s_mat = &mut c2s_matrix(iang);
        let c2s_mat = &c2s_matrix_const(iang);
        let s_len = 2*iang + 1;
        let c_len = (iang+1)*(iang+2)/2;
        for coeff in ibas.coefficients.iter() {
            let mut tmp_cart = MatrixFull::new([num_grids,c_len],0.0);
            coeff.iter().zip(ibas.exponents.iter()).for_each(|(icoeff,iexp)| {

                //time_records.count_start("1-1");
                let mut tmp_cart_0 = cartesian_gto_cint_batch_rayon(*iexp, iang, gto_center, r);
                //time_records.count("1-1");

                tmp_cart.par_self_scaled_add(&tmp_cart_0, *icoeff);
            });

            //time_records.count_start("1-2");
            let tmp_spheric = _dgemm_nn(&tmp_cart.to_matrixfullslice(), &c2s_mat.to_matrixfullslice());
            //time_records.count("1-2");

            outmat_s.iter_columns_mut(ibas_start..ibas_start+s_len)
            .zip(tmp_spheric.iter_columns_full()).for_each(|(to,from)| {
                to.par_iter_mut().zip(from.par_iter()).for_each(|(to,from)| {*to = *from});
            });
            ibas_start += s_len;
        }
    }

    //time_records.report_all();

    outmat_s
}

pub fn spheric_gto_value_matrixfull_serial(r: &[[f64;3]],gto_center:&[f64;3], bas:&Basis4Elem) -> MatrixFull<f64> {

    //let mut time_records = utilities::TimeRecords::new();
    //time_records.new_item("1-1", "cartesian_gto_cint_batch");
    //time_records.new_item("1-2", "_dgemm_nn");

    let num_grids:usize = r.len();
    let mut num_bas_c = 0;
    let mut num_bas_s = 0;
    bas.electron_shells.iter().for_each(|ibas| {
        let iang = ibas.angular_momentum[0] as usize;
        num_bas_c += (iang+1)*(iang+2)/2*ibas.coefficients.len();
        num_bas_s += (iang*2+1)*ibas.coefficients.len();
    });
    //println!("debug num_bas_c and num_bas_s: {},{}", num_bas_c, num_bas_s);
    //let mut outmat_c = MatrixFull::new([num_grids,num_bas_c],0.0);
    let mut outmat_s = MatrixFull::new([num_grids,num_bas_s],0.0);
    let mut ibas_start = 0;
    for ibas in &bas.electron_shells {
        let iang = ibas.angular_momentum[0] as usize;
        //let mut c2s_mat = &mut c2s_matrix(iang);
        let c2s_mat = &c2s_matrix_const(iang);
        let s_len = 2*iang + 1;
        let c_len = (iang+1)*(iang+2)/2;
        for coeff in ibas.coefficients.iter() {
            let mut tmp_cart = MatrixFull::new([num_grids,c_len],0.0);
            coeff.iter().zip(ibas.exponents.iter()).for_each(|(icoeff,iexp)| {

                //time_records.count_start("1-1");
                let mut tmp_cart_0 = cartesian_gto_cint_batch(*iexp, iang, gto_center, r);
                //time_records.count("1-1");

                tmp_cart.self_scaled_add(&tmp_cart_0, *icoeff);
            });

            //time_records.count_start("1-2");
            let tmp_spheric = _dgemm_nn_serial(&tmp_cart.to_matrixfullslice(), &c2s_mat.to_matrixfullslice());
            //time_records.count("1-2");

            outmat_s.iter_columns_mut(ibas_start..ibas_start+s_len)
            .zip(tmp_spheric.iter_columns_full()).for_each(|(to,from)| {
                to.iter_mut().zip(from.iter()).for_each(|(to,from)| {*to = *from});
            });
            ibas_start += s_len;
        }
    }
    //time_records.report_all();

    outmat_s
}

pub fn spheric_gto_value_serial(r: &[[f64;3]],gto_center:&[f64;3], bas:&Basis4Elem) -> MatrixFull<f64> {

    //let mut time_records = utilities::TimeRecords::new();
    //time_records.new_item("1-1", "cartesian_gto_cint_batch");
    //time_records.new_item("1-2", "_dgemm_nn");

    let num_grids:usize = r.len();
    let mut num_bas_c = 0;
    let mut num_bas_s = 0;
    bas.electron_shells.iter().for_each(|ibas| {
        let iang = ibas.angular_momentum[0] as usize;
        num_bas_c += (iang+1)*(iang+2)/2*ibas.coefficients.len();
        num_bas_s += (iang*2+1)*ibas.coefficients.len();
    });
    //println!("debug num_bas_c and num_bas_s: {},{}", num_bas_c, num_bas_s);
    //let mut outmat_c = MatrixFull::new([num_grids,num_bas_c],0.0);
    let mut outmat_s = MatrixFull::new([num_bas_s, num_grids],0.0);
    let mut ibas_start = 0;
    for ibas in &bas.electron_shells {
        let iang = ibas.angular_momentum[0] as usize;
        //let mut c2s_mat = &mut c2s_matrix(iang);
        let c2s_mat = &c2s_matrix_const(iang);
        let s_len = 2*iang + 1;
        let c_len = (iang+1)*(iang+2)/2;
        for coeff in ibas.coefficients.iter() {
            let mut tmp_cart = MatrixFull::new([c_len,num_grids],0.0);
            coeff.iter().zip(ibas.exponents.iter()).for_each(|(icoeff,iexp)| {

                ////time_records.count_start("1-1");
                //let mut tmp_cart_0 = cartesian_gto_cint_batch_v04(*iexp, iang, gto_center, r);
                ////time_records.count("1-1");

                //tmp_cart.self_scaled_add(&tmp_cart_0, *icoeff);
                tmp_cart += cartesian_gto_cint_batch_v04(*iexp, iang, gto_center, r)* (*icoeff);
            });

            //time_records.count_start("1-2");
            let mut tmp_spheric= MatrixFull::new([s_len,num_grids],0.0);
            tmp_spheric.to_matrixfullslicemut().lapack_dgemm(
                &c2s_mat.to_matrixfullslice(),
                &tmp_cart.to_matrixfullslice(), 
                'T', 'N', 1.0, 0.0);
            //let tmp_spheric = _dgemm_nn_serial(&tmp_cart.to_matrixfullslice(), &c2s_mat.to_matrixfullslice());
            //time_records.count("1-2");

            outmat_s.copy_from_matr(ibas_start..ibas_start+s_len, 0..num_grids, 
                &tmp_spheric, 0..s_len, 0..num_grids);
            //outmat_s.iter_columns_mut(ibas_start..ibas_start+s_len)
            //.zip(tmp_spheric.iter_columns_full()).for_each(|(to,from)| {
            //    to.iter_mut().zip(from.iter()).for_each(|(to,from)| {*to = *from});
            //});
            ibas_start += s_len;
        }
    }
    //time_records.report_all();

    outmat_s
}

pub fn gto_value(r:&[f64;3], gto_center:&[f64;3], bas:&Basis4Elem, basis_type: &String) -> Vec<f64> {
    let mut value:Vec<f64> = vec![];
    for ibas in &bas.electron_shells {
        let iang = ibas.angular_momentum[0];
        for coeff in ibas.coefficients.iter() {
            let mut tmp_cart = MatrixFull::new([((iang+1)*(iang+2)/2) as usize,1],0.0);
            coeff.iter().zip(ibas.exponents.iter()).for_each(|(icoeff,iexp)| {
                //let scoeff = *icoeff * _gaussian_int(iang*2 + 2, *iexp*2.0).sqrt();
                let mut tmp_cart_0 = cartesian_gto_cint(*iexp, iang as usize, gto_center, r);
                //tmp_cart.data.iter_mut().zip(tmp_cart_0.data.iter()).for_each(|(to,from)| {
                //    *to += *from*icoeff
                //});
                tmp_cart.self_scaled_add(&tmp_cart_0, *icoeff);
            });
            if basis_type.eq(&"spheric".to_string()) {
                let mut tmp_spheric = MatrixFull::new([(2*iang + 1) as usize, 1],0.0);
                tmp_spheric.to_matrixfullslicemut().lapack_dgemm(
                    &tmp_cart.to_matrixfullslice(), 
                    &c2s_matrix_const(iang as usize).to_matrixfullslice(),
                     'T','N',1.0,1.0);
                //tmp_spheric.lapack_dgemm(&mut tmp_cart, &mut c2s_matrix(iang as usize), 'T', 'N', 1.0, 0.0);
                value.extend(tmp_spheric.data);
            } else {
                value.extend(tmp_cart.data);
            }
        }
        
    }
    value
}
pub fn spheric_gto_1st_value_batch(r:&[[f64;3]], gto_center:&[f64;3], bas:&Basis4Elem) -> Vec<MatrixFull<f64>> {
    let num_grids:usize = r.len();
    let mut num_bas_c = 0;
    let mut num_bas_s = 0;
    bas.electron_shells.iter().for_each(|ibas| {
        let iang = ibas.angular_momentum[0] as usize;
        num_bas_c += (iang+1)*(iang+2)/2*ibas.coefficients.len();
        num_bas_s += (iang*2+1)*ibas.coefficients.len();
    });
    //let mut paox = RIFull::new([num_grids,num_bas_s,3],0.0);
    let mut paox = vec![MatrixFull::new([num_grids,num_bas_s],0.0);3];
    //let mut paox = MatrixFull::new([num_grids,num_bas_s],0.0);
    //let mut aopy = MatrixFull::new([num_grids,num_bas_s],0.0);
    //let mut aopz = MatrixFull::new([num_grids,num_bas_s],0.0);
    let mut ibas_start = 0;
    for ibas in &bas.electron_shells {
        let iang = ibas.angular_momentum[0] as usize;
        //let mut c2s_mat = &mut c2s_matrix(iang);
        let mut c2s_mat = &c2s_matrix_const(iang);
        let s_len = 2*iang + 1;
        let c_len = (iang+1)*(iang+2)/2;
        for coeff in ibas.coefficients.iter() {
            let mut tmp_cart = vec![MatrixFull::new([num_grids,c_len],0.0);3];
            coeff.iter().zip(ibas.exponents.iter()).for_each(|(icoeff,iexp)| {
                let mut tmp_cart_0 = cartesian_gto_1st_cint_batch(*iexp, iang, gto_center, r);

                //tmp_cart.self_scaled_add(&tmp_cart_0, *icoeff);
                tmp_cart.iter_mut().zip(tmp_cart_0.iter()).for_each(|(to,from)| {
                    to.par_self_scaled_add(from, *icoeff);
                });
            });
            //let mut tmp_spheric = MatrixFull::new([num_grids,s_len],0.0);
            paox.iter_mut().zip(tmp_cart.iter_mut()).for_each(|(paox_x,tmp_cart_x)| {
                //tmp_spheric.lapack_dgemm(tmp_cart_x, &mut c2s_mat, 'N','N',1.0,0.0);
                let tmp_spheric = _dgemm_nn(&tmp_cart_x.to_matrixfullslice(),&c2s_mat.to_matrixfullslice());
                paox_x.iter_columns_mut(ibas_start..ibas_start+s_len)
                .zip(tmp_spheric.iter_columns_full()).for_each(|(to,from)| {
                    to.par_iter_mut().zip(from.par_iter()).for_each(|(to,from)| {*to = *from});
                });

            });
            ibas_start += s_len;
        }
    }
    paox
}
pub fn spheric_gto_1st_value_batch_serial(r:&[[f64;3]], gto_center:&[f64;3], bas:&Basis4Elem) -> Vec<MatrixFull<f64>> {
    let num_grids:usize = r.len();
    let mut num_bas_c = 0;
    let mut num_bas_s = 0;
    bas.electron_shells.iter().for_each(|ibas| {
        let iang = ibas.angular_momentum[0] as usize;
        num_bas_c += (iang+1)*(iang+2)/2*ibas.coefficients.len();
        num_bas_s += (iang*2+1)*ibas.coefficients.len();
    });
    //let mut paox = RIFull::new([num_grids,num_bas_s,3],0.0);
    let mut paox = vec![MatrixFull::new([num_grids,num_bas_s],0.0);3];
    //let mut paox = MatrixFull::new([num_grids,num_bas_s],0.0);
    //let mut aopy = MatrixFull::new([num_grids,num_bas_s],0.0);
    //let mut aopz = MatrixFull::new([num_grids,num_bas_s],0.0);
    let mut ibas_start = 0;
    for ibas in &bas.electron_shells {
        let iang = ibas.angular_momentum[0] as usize;
        //let mut c2s_mat = &mut c2s_matrix(iang);
        let mut c2s_mat = &c2s_matrix_const(iang);
        let s_len = 2*iang + 1;
        let c_len = (iang+1)*(iang+2)/2;
        for coeff in ibas.coefficients.iter() {
            let mut tmp_cart = vec![MatrixFull::new([num_grids,c_len],0.0);3];
            coeff.iter().zip(ibas.exponents.iter()).for_each(|(icoeff,iexp)| {
                let mut tmp_cart_0 = cartesian_gto_1st_cint_batch_serial(*iexp, iang, gto_center, r);

                //tmp_cart.self_scaled_add(&tmp_cart_0, *icoeff);
                tmp_cart.iter_mut().zip(tmp_cart_0.iter()).for_each(|(to,from)| {
                    to.self_scaled_add(from, *icoeff);
                });
            });
            //let mut tmp_spheric = MatrixFull::new([num_grids,s_len],0.0);
            paox.iter_mut().zip(tmp_cart.iter_mut()).for_each(|(paox_x,tmp_cart_x)| {
                //tmp_spheric.lapack_dgemm(tmp_cart_x, &mut c2s_mat, 'N','N',1.0,0.0);
                let tmp_spheric = _dgemm_nn_serial(&tmp_cart_x.to_matrixfullslice(),&c2s_mat.to_matrixfullslice());
                paox_x.iter_columns_mut(ibas_start..ibas_start+s_len)
                .zip(tmp_spheric.iter_columns_full()).for_each(|(to,from)| {
                    to.iter_mut().zip(from.iter()).for_each(|(to,from)| {*to = *from});
                });

            });
            ibas_start += s_len;
        }
    }
    paox
}

pub fn spheric_gto_1st_value_serial(r:&[[f64;3]], gto_center:&[f64;3], bas:&Basis4Elem) -> Vec<MatrixFull<f64>> {
    let num_grids:usize = r.len();
    let mut num_bas_c = 0;
    let mut num_bas_s = 0;
    bas.electron_shells.iter().for_each(|ibas| {
        let iang = ibas.angular_momentum[0] as usize;
        num_bas_c += (iang+1)*(iang+2)/2*ibas.coefficients.len();
        num_bas_s += (iang*2+1)*ibas.coefficients.len();
    });
    //let mut paox = RIFull::new([num_grids,num_bas_s,3],0.0);
    let mut paox = vec![MatrixFull::new([num_bas_s,num_grids],0.0);3];
    //let mut paox = MatrixFull::new([num_grids,num_bas_s],0.0);
    //let mut aopy = MatrixFull::new([num_grids,num_bas_s],0.0);
    //let mut aopz = MatrixFull::new([num_grids,num_bas_s],0.0);
    let mut ibas_start = 0;
    for ibas in &bas.electron_shells {
        let iang = ibas.angular_momentum[0] as usize;
        //let mut c2s_mat = &mut c2s_matrix(iang);
        let mut c2s_mat = &c2s_matrix_const(iang);
        let s_len = 2*iang + 1;
        let c_len = (iang+1)*(iang+2)/2;
        for coeff in ibas.coefficients.iter() {
            let mut tmp_cart = vec![MatrixFull::new([c_len, num_grids],0.0);3];
            coeff.iter().zip(ibas.exponents.iter()).for_each(|(icoeff,iexp)| {
                let mut tmp_cart_0 = cartesian_gto_1st_cint_batch_serial_v03(*iexp, iang, gto_center, r);

                //tmp_cart.self_scaled_add(&tmp_cart_0, *icoeff);
                tmp_cart.iter_mut().zip(tmp_cart_0.iter()).for_each(|(to,from)| {
                    to.self_scaled_add(from, *icoeff);
                });
            });
            //let mut tmp_spheric = MatrixFull::new([num_grids,s_len],0.0);
            //Stop here
            paox.iter_mut().zip(tmp_cart.iter_mut()).for_each(|(paox_x,tmp_cart_x)| {
                //tmp_spheric.lapack_dgemm(tmp_cart_x, &mut c2s_mat, 'N','N',1.0,0.0);
                let mut tmp_spheric = MatrixFull::new([s_len,num_grids],0.0);
                tmp_spheric.to_matrixfullslicemut().lapack_dgemm(
                    &c2s_mat.to_matrixfullslice(),
                    &tmp_cart_x.to_matrixfullslice(),
                    'T','N',1.0,0.0);
                paox_x.copy_from_matr(ibas_start..ibas_start+s_len,0..num_grids,  &tmp_spheric, 0..s_len,0..num_grids);
                //let tmp_spheric = _dgemm_nn_serial(&tmp_cart_x.to_matrixfullslice(),&c2s_mat.to_matrixfullslice());
                //paox_x.iter_columns_mut(ibas_start..ibas_start+s_len)
                //.zip(tmp_spheric.iter_columns_full()).for_each(|(to,from)| {
                //    to.iter_mut().zip(from.iter()).for_each(|(to,from)| {*to = *from});
                //});

            });
            ibas_start += s_len;
        }
    }
    paox
}

pub fn gto_1st_value(r:&[f64;3], gto_center:&[f64;3], bas:&Basis4Elem, basis_type: &String) -> Vec<Vec<f64>> {
    let mut value:Vec<Vec<f64>> = vec![vec![];3];
    //let mut value:[Vec<f64>;3] = [Vec::new();3];
    for ibas in &bas.electron_shells {
        let iang = ibas.angular_momentum[0];
        for coeff in ibas.coefficients.iter() {
            let mut tmp_cart = [
                MatrixFull::new([((iang+1)*(iang+2)/2) as usize,1],0.0),
                MatrixFull::new([((iang+1)*(iang+2)/2) as usize,1],0.0),
                MatrixFull::new([((iang+1)*(iang+2)/2) as usize,1],0.0)
            ];
            coeff.iter().zip(ibas.exponents.iter()).for_each(|(icoeff,iexp)| {
                //let scoeff = *icoeff * _gaussian_int(iang*2 + 2, *iexp*2.0).sqrt();
                let mut tmp_cart_0 = cartesian_gto_1st_cint(*iexp, iang as usize, gto_center, r);
                //println!("debug: {:?}, {:?}, {:?}", &tmp_cart_0[0].data,&tmp_cart_0[1].data,&tmp_cart_0[2].data);
                tmp_cart.iter_mut().zip(tmp_cart_0.iter()).for_each(|(tmp_cart,tmp_cart_0)| {
                    tmp_cart.self_scaled_add(tmp_cart_0, *icoeff)
                });
            });
            if basis_type.eq(&"spheric".to_string()) {
                tmp_cart.iter().zip(value.iter_mut()).for_each(|(tmp_cart, value)| {
                    let mut tmp_spheric = MatrixFull::new([(2*iang + 1) as usize, 1],0.0);
                    //tmp_spheric.lapack_dgemm(tmp_cart, &mut c2s_matrix(iang as usize), 'T', 'N', 1.0, 0.0);
                    tmp_spheric.to_matrixfullslicemut().lapack_dgemm(
                        &tmp_cart.to_matrixfullslice(), 
                        &c2s_matrix_const(iang as usize).to_matrixfullslice(), 
                        'T', 'N', 1.0, 0.0);
                    value.extend(tmp_spheric.data.iter());
                });
            } else {
                tmp_cart.iter().zip(value.iter_mut()).for_each(|(tmp_cart, value)| {
                    value.extend(tmp_cart.data.iter())
                });
            }
        }
        
    }
    value
}
pub fn gto_value_debug(r:&[f64;3], gto_center:&[f64;3], bas:&Basis4Elem, basis_type: &String,count: usize) -> Vec<f64> {
    let mut value:Vec<f64> = vec![];
    for ibas in &bas.electron_shells {
        let iang = ibas.angular_momentum[0];
        for coeff in ibas.coefficients.iter() {
            let mut tmp_cart = MatrixFull::new([((iang+1)*(iang+2)/2) as usize,1],0.0);
            coeff.iter().zip(ibas.exponents.iter()).for_each(|(icoeff,iexp)| {
                //let scoeff = *icoeff * _gaussian_int(iang*2 + 2, *iexp*2.0).sqrt();
                let mut tmp_cart_0 = cartesian_gto_cint(*iexp, iang as usize, gto_center, r);
                //tmp_cart.data.iter_mut().zip(tmp_cart_0.data.iter()).for_each(|(to,from)| {
                //    *to += *from*icoeff
                //});
                tmp_cart.self_scaled_add(&tmp_cart_0, *icoeff);
                //if count<10 {println!("iang: {:2} coeff: {:8.4}, exp: {:8.4}, tmp_cart: {:?}", iang, scoeff, iexp, &tmp_cart)};
            });
            if basis_type.eq(&"spheric".to_string()) {
                let mut tmp_spheric = MatrixFull::new([(2*iang + 1) as usize, 1],0.0);
                //tmp_spheric.lapack_dgemm(&mut tmp_cart, &mut c2s_matrix(iang as usize), 'T', 'N', 1.0, 0.0);
                tmp_spheric.to_matrixfullslicemut().lapack_dgemm(
                    &tmp_cart.to_matrixfullslice(), 
                    &c2s_matrix_const(iang as usize).to_matrixfullslice(),
                    'T', 'N', 1.0, 0.0);
                value.extend(tmp_spheric.data);
            } else {
                value.extend(tmp_cart.data);
            }
        }
        
    }
    value
}

/// Normalization factor for GTO radial part g=r^l e^{-\alpha r^2}
/// 
/// \frac{1}{\sqrt{\int g^r r^2 dr}} = \sqrt{\frac{2^{2l+3} (l+1)! (2a)^{1+1.5}}{(2l+2)!\sqrt{\pi}}}
/// 
/// Ref: H. B. Schlegel and M. J. Frisch, Int. J. Quant. Chem., 54(1995, 83-87)
/// 
/// This part is absorbed in the BasCell.coefficients for libcint, which, however, 
/// should be removed for numerical integration used.
pub fn cint_norm_factor(iang: i32, alpha: f64) -> f64 {
    return _gaussian_int(iang*2+2, alpha*2.0).sqrt()
}

pub fn _gaussian_int(iang: i32, alpha: f64) -> f64 {
    let dang = (iang as f64 + 1.0)*0.5;
    return libm::exp(libm::lgamma(dang))/(2.0*alpha.powf(dang))
}





pub(crate) mod basic_math {
    pub fn pi() -> f64 {
        return std::f64::consts::PI;
    }

    pub fn rad2deg(x:f64) -> f64 {
        return x*180.00/pi();
    }
    
    pub fn arctan(x: f64) -> f64 {
        let y = x;
        let sqrt_three = (3.0_f64).sqrt();
        if y < 0.0 {
            return -1.0 * arctan(-1.0 * y);
        } else
        /* if positive */
        {
            if y < 1.0 {
                if y <= 0.267949 {
                    return y - (y.powf(3.0) / 3.0) + (y.powf(5.0) / 5.0);
                } else {
                    return rad2deg(
                        (pi() / 6.0)
                            + arctan(( sqrt_three * y) - 1.0)
                                / arctan(sqrt_three + y),
                            //+ arctan(((3.0_f64).sqrt() * y) - 1.0)
                            //    / arctan((3.0_f64).sqrt() + y),
                    );
                }
                
            } else {
                return (pi() / 2.0) - arctan(1.0 / y);
            }
        }
    }
    pub fn arcsin(x: f64) -> f64 {
        return arctan(x / (1.0 - x.powf(2.0)).sqrt());
    }
    pub fn arccos(x: f64) -> f64 {
        return arctan((1.0 - x.powf(2.0)).sqrt() / x);
    }

    pub fn factorial(end:i32) -> i32 {
        if end == 0 {return 1
        } else {
            return (1..=end).fold(1,|acc,x| {acc * x})
        }
    }
    #[inline]
    pub fn double_factorial(end:i32) -> i32 {
        // for positive number, it yields the corresponding double factorial number
        // for negtive number, it yields the reciprocal ef the double factorial number
        if end == 0 || end == 1 || end == -1 {return 1};
        if end>0 {
            let (tmp_start, tmp_end) = (1,end/2);
            if end%2 == 1 {   //odd
                return (tmp_start..=tmp_end).fold(1,|acc,x| {acc*(2*x+1)});
            } else {  // even
                return (tmp_start..=tmp_end).fold(1,|acc,x| {acc * 2* x});
            }
        } else {
            let (tmp_start, tmp_end) = (end/2,-1);
            if end%2 == -1 {   //odd
                return (tmp_start..=tmp_end).fold(1,|acc,x| {acc*(2*x+1)});
            } else {  // even
                return 0;
            }
        }
    }
    #[inline]
    pub fn specific_double_factorial(end:i32) -> f64 {
        match end {
           -1 => 1.0_f64,
            0 => 1.0_f64,
            1 => 1.0_f64,
            3 => 3.0_f64,
            5 => 15.0_f64,
            7 => 105.0_f64,
            9 => 945.0_f64,
           11 => 10395.0_f64,
           13 => 135135.0_f64,
            _ => panic!("Error: at present only provide the double factorial for the odd numbers from -2 to 10"),
        }
    }
}

#[test]
fn test_double_factorial() {
    for x in 0..=19 {
        println!("{},{}",x,double_factorial(x));
    }
    for x in -10..=-1 {
        println!("{},{}",x,double_factorial(x));
    }
}
#[test]
fn debug_cartesian_gto() {
    let a = 3.0;
    let l = 2usize;
    let c = [0.0, 0.0,0.0];
    let r = [1.0, 2.0,3.0];
    //cartesian_gto(a, l, c, r);
}
#[test]
fn debug_par() {
    let vec1:Vec<i32> = (0..1000).map(|i| i).collect();
    let vec2:Vec<i32> = (1000..2000).map(|i| i).collect();
    let vec3:Vec<i32> = (2000..3000).map(|i| i).collect();
    let vec4:Vec<i32> = (3000..4000).map(|i| i).collect();
    vec1.par_iter().zip(vec2.par_iter()).map(|(a,b)| (a,b))
    .zip(vec3.par_iter().zip(vec4.par_iter()).map(|(c,d)| (c,d)))
    .for_each(|(a,b)| {println!("{:4}, {:4}, {:4}, {:4}, {:4}, {:4}",a.0,a.1,b.0,b.1, b.0-a.0,b.1-a.1)});
}


#[test]
fn norm_factor_gto_radial() {
    let l = 2_i32;
    let a = 2.0_f64;
    let value1 = cint_norm_factor(l,a);
    let value2 = _gaussian_int(l*2+2,a*2.0).sqrt();
    let norm0 = ((2.0*a/PI).powf(0.75)*(4.0*a).powf((l as f64)/2.0));
    println!("{},{},{},{}",value1,value2,norm0, value1*norm0);
}
