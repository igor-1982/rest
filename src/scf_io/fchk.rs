use std::fs;
use std::io::Write;
use crate::utilities::convert_scientific_notation_to_fortran_format as r2f;
use rest_libcint::{CINTR2CDATA, CintType};
use regex::Regex;
use crate::constants::{SPECIES_INFO, INVERSE_THRESHOLD};

use crate::external_libs::py2fch;

use crate::scf_io::SCF;


macro_rules! dump_real_r2f {
    ($input:expr, $data:expr) => {
    let mut i_index = 0;
    $data.iter().for_each(|x| {
        let sdd = format!("{:16.8E}", x);
        if (i_index + 1)%5 == 0 {
            write!($input, "{}\n",r2f(&sdd))
        } else {
            write!($input, "{}",r2f(&sdd))
        };
        i_index += 1;
    });
    if i_index % 5 != 0 {write!($input, "\n");}
}
}

macro_rules! dump_int {
    ($input:expr, $data:expr) => {
    let mut i_index = 0;
    $data.iter().for_each(|x| {
        if (i_index + 1)%6 == 0 {
            write!($input, "{:12}\n", x)
        } else {
            write!($input, "{:12}", x)
        };
        i_index += 1;
    });
    if i_index % 6 != 0 {write!($input, "\n");}
}
}


impl SCF {
    pub fn save_fchk_of_gaussian(&self) {
        self.create_fchk_head();
        self.fchk_write_mo();
    }

    pub fn create_fchk_head(&self) {
        let re_basis = Regex::new(r"/?(?P<basis>[^/]*)/?$").unwrap();
        let cap = re_basis.captures(&self.mol.ctrl.basis_path).unwrap();
        let basis_name = cap.name("basis").unwrap().to_string();
        let mut input = fs::File::create(&format!("{}.fchk",&self.mol.geom.name)).unwrap();
        write!(input, "{:} generated by REST\n", &self.mol.geom.name);
        if self.mol.spin_channel==1 {
            write!(input, "SP        R{:-59}{:-20}\n",
                self.mol.ctrl.xc.to_uppercase(),
                basis_name.to_uppercase()
            );
        } else {
            write!(input, "SP      U{:-59}{:-20}\n",
                self.mol.ctrl.xc.to_uppercase(),
                basis_name.to_uppercase()
            );
        };
        let natom = self.mol.geom.elem.len();
        write!(input, "Number of atoms                            I {:16}\n", self.mol.geom.elem.len());
        write!(input, "Charge                                     I {:16}\n", self.mol.ctrl.charge as i32);
        write!(input, "Multiplicity                               I {:16}\n", self.mol.ctrl.spin as i32);
        write!(input, "Number of electrons                        I {:16}\n", self.mol.num_elec[0]);
        write!(input, "Number of alpha electrons                  I {:16}\n", self.mol.num_elec[1]);
        write!(input, "Number of beta electrons                   I {:16}\n", self.mol.num_elec[2]);
        write!(input, "Number of basis functions                  I {:16}\n", self.mol.num_basis);
        write!(input, "Number of independent functions            I {:16}\n", self.mol.num_state);
        // ==============================
        // Now for atomic numbers
        // ==============================
        write!(input, "Atomic numbers                             I   N={:12}\n", natom);
        let mut i_index = 0;
        self.mol.geom.elem.iter().for_each(|x| {
            let (mass, charge) = SPECIES_INFO.get(x.as_str()).unwrap();
            if (i_index + 1)%6 == 0 {
                write!(input, "{:12}\n",*charge as i32)
            } else {
                write!(input, "{:12}",*charge as i32)
            };
            i_index += 1;
        });
        if i_index % 6 != 0 {write!(input, "\n");}
        // ==============================
        // Now for Nuclear charges
        // ==============================
        write!(input, "Nuclear charges                            R   N={:12}\n", natom);
        let mut i_index = 0;
        self.mol.geom.elem.iter().for_each(|x| {
            let (mass, charge) = SPECIES_INFO.get(x.as_str()).unwrap();
            let sdd = format!("{:16.8E}", charge);
            if (i_index + 1)%5 == 0 {
                write!(input, "{}\n",r2f(&sdd))
            } else {
                write!(input, "{}",r2f(&sdd))
            };
            i_index += 1;
        });
        if i_index % 5 != 0 {write!(input, "\n");}
        // ==============================
        // Now for geometry coordinates
        // ==============================
        write!(input, "Current cartesian coordinates              R   N={:12}\n", natom*3);
        let mut i_index = 0;
        self.mol.geom.position.iter_columns_full().for_each(|x_position| {
            x_position.iter().for_each(|x| {
                let sdd = format!("{:16.8E}",x);
                if (i_index + 1)%5 ==0 {
                    write!(input,"{}\n", r2f(&sdd));
                } else {
                    write!(input,"{}",r2f(&sdd));
                }
                i_index +=1;
            })
        });
        if i_index % 5 != 0 {write!(input, "\n");}
        // ==============================
        // Now for atomic weights
        // ==============================
        write!(input, "Real atomic weights                        R   N={:12}\n", natom);
        let mut i_index = 0;
        self.mol.geom.elem.iter().for_each(|x| {
            let (mass, charge) = SPECIES_INFO.get(x.as_str()).unwrap();
            let sdd = format!("{:16.8E}", mass);
            if (i_index + 1)%5 == 0 {
                write!(input, "{}\n",r2f(&sdd))
            } else {
                write!(input, "{}",r2f(&sdd))
            };
            i_index += 1;
        });
        if i_index % 5 != 0 {write!(input, "\n");}
        // ==============================
        // Now for MicOpt
        // ==============================
        write!(input, "MicOpt                                     I   N={:12}\n", natom);
        let mut i_index = 0;
        self.mol.geom.elem.iter().for_each(|x| {
            if (i_index + 1)%6 == 0 {
                write!(input, "{:12}\n", -1)
            } else {
                write!(input, "{:12}", -1)
            };
            i_index += 1;
        });
        if i_index % 6 != 0 {write!(input, "\n");}
        //================================
        // Now for basis set info.
        //================================
        let mut num_contract = 0;
        let mut num_primitiv = 0;
        let mut num_d_shell = 0;
        let mut num_f_shell = 0;
        let mut max_ang = 0;
        let mut max_contract = 0;
        let mut shell_type: Vec<i32> = vec![];
        let mut num_primitiv_vec: Vec<usize> = vec![];
        let mut shell_to_atom_map: Vec<usize> = vec![];
        let mut primitive_exp: Vec<f64> = vec![];
        let mut coord_each_shell: Vec<f64> = vec![];
        let mut coeff_vec:Vec<f64> = vec![];
        let shell_type_fac = match self.mol.cint_type {
            CintType::Spheric => -1,
            CintType::Cartesian => 1,
        };
        self.mol.basis4elem.iter().enumerate().for_each(|(i_atom,ibas)| {
            ibas.electron_shells.iter().for_each(|ibascell| {
                //let mut tmp_coeff_vec =  ibascell.coefficients.clone();
                ibascell.coefficients.iter().for_each(|coeff_i| {
                    // De-normalization
                    let mut tmp_coeff_i = coeff_i.clone();
                    let tmp_ang = ibascell.angular_momentum[0];
                    tmp_coeff_i.iter_mut().zip(ibascell.exponents.iter()).for_each(|(i, e)| {
                        *i /= CINTR2CDATA::gto_norm(tmp_ang, *e);
                    });
                    coeff_vec.extend(tmp_coeff_i.iter());
                    num_contract += 1;
                    num_primitiv += ibascell.exponents.len();
                    max_ang = std::cmp::max(max_ang, tmp_ang);
                    max_contract = std::cmp::max(max_contract, ibascell.exponents.len());
                    if ibascell.angular_momentum[0] == 2 {num_d_shell += 1};
                    if ibascell.angular_momentum[0] == 3 {num_f_shell += 1};
                    let tmp_shell_type = if ibascell.angular_momentum[0] == 1 {
                        ibascell.angular_momentum[0]
                    } else {
                        ibascell.angular_momentum[0]*shell_type_fac
                    };
                    shell_type.push(tmp_shell_type);
                    num_primitiv_vec.push(ibascell.exponents.len());
                    shell_to_atom_map.push(i_atom+1);
                    primitive_exp.extend(ibascell.exponents.iter());
                    coord_each_shell.extend(self.mol.geom.position.iter_column(i_atom));
                });
            });
        });
        write!(input, "Number of contracted shells                I {:16}\n",num_contract);
        write!(input, "Number of primitive shells                 I {:16}\n",num_primitiv);
        write!(input, "Pure/Cartesian d shells                    I {:16}\n", num_d_shell);
        write!(input, "Pure/Cartesian f shells                    I {:16}\n", num_f_shell);
        write!(input, "Highest angular momentum                   I {:16}\n", max_ang);
        write!(input, "Largest degree of contraction              I {:16}\n", max_contract);
        // ==============================
        // Now for shell infor.
        // ==============================
        write!(input, "Shell types                                I   N={:12}\n", num_contract);
        dump_int!(input, shell_type);
        write!(input, "Number of primitives per shell             I   N={:12}\n", num_contract);
        dump_int!(input, num_primitiv_vec);
        write!(input, "Shell to atom map                          I   N={:12}\n", num_contract);
        dump_int!(input, shell_to_atom_map);
        write!(input, "Primitive exponents                        R   N={:12}\n", primitive_exp.len());
        dump_real_r2f!(input, primitive_exp);
        write!(input, "Contraction coefficients                   R   N={:12}\n", coeff_vec.len());
        dump_real_r2f!(input, &coeff_vec);
//        write!(input, "P(S=P) Contraction coefficients            R   N={:12}\n", coeff_vec.len());
//        let mut i_index = 0;
//        coeff_vec.iter().for_each(|x| {
//            let sdd = format!("{:16.8E}", 0);
//            if (i_index + 1)%5 == 0 {
//                write!(input, "{}\n",r2f(&sdd))
//            } else {
//                write!(input, "{}",r2f(&sdd))
//            };
//            i_index += 1;
//        });
//        if i_index % 5 != 0 {write!(input, "\n");}
        write!(input, "Coordinates of each shell                  R   N={:12}\n", coord_each_shell.len());
        dump_real_r2f!(input, &coord_each_shell);
        // ==============================
        // Now for total and orb energies
        // ==============================
        let mut ilsw:Vec<i32> = vec![0;100];
        if self.mol.spin_channel==2 {
            ilsw[0] = 1
        }
        ilsw[4] = 2;
        ilsw[11] = -1;
        ilsw[12] = 5;
        ilsw[24] = 1;
        ilsw[25] = 1;
        ilsw[32] = 100000;
        ilsw[34] = -1;
        ilsw[45] = 1;
        ilsw[49] = -2000000000;
        ilsw[50] = 1;
        ilsw[56] = 4;
        ilsw[57] = 52;
        ilsw[69] = natom as i32;
        ilsw[71] = 1;  
        write!(input, "Num ILSW                                   I     {:12}\n", 100);
        write!(input, "ILSW                                       I   N={:12}\n", 100);
        dump_int!(input, ilsw);
        // TODO: ECP
        let dd = format!("{:27.15E}", self.scf_energy);
        write!(input, "SCF Energy                                 R {}\n",r2f(&dd));
        write!(input, "Total Energy                               R {}\n",r2f(&dd));
        for i_spin in 0..self.mol.spin_channel {
            if i_spin ==0  {
                write!(input, "Alpha Orbital Energies                     R   N={:12}\n", self.eigenvalues[i_spin].len());
            } else {
                write!(input, "Beta Orbital Energies                      R   N={:12}\n", self.eigenvalues[i_spin].len());
            }
            dump_real_r2f!(input, self.eigenvalues[i_spin]);
        }
        // ==============================
        // Now for orbital coefficients
        // ==============================
        let nbf = self.mol.num_basis;
        //let nif = self.mol.num_state;
        for i_spin in 0..self.mol.spin_channel {
            if i_spin ==0  {
                write!(input, "Alpha MO coefficients                      R   N={:12}\n", self.eigenvectors[i_spin].data.len());
            } else {
                write!(input, "Beta MO coefficients                       R   N={:12}\n", self.eigenvectors[i_spin].data.len());
            }
        }
        // leave MOs blank, it will be written by librest2fch
        let n_dm = nbf*(nbf+1)/2;
        // write all zeros as dm
        // for compatibility with librest2fch.so
        write!(input, "Total SCF Density                          R   N={:12}\n", n_dm);
        for i_index in 0..n_dm {
                let sdd = format!("{:16.8E}", 0.0f64); 
                if (i_index + 1)%5 == 0 {
                    write!(input, "{}\n",r2f(&sdd));
                } else {
                    write!(input, "{}",r2f(&sdd));
                }
        }
        if n_dm % 5 != 0 {write!(input, "\n");}
        input.sync_all().unwrap();
    }

    pub fn fchk_write_mo(&self) {
        let nbf = self.mol.num_basis;
        let nif = self.mol.num_state;
    
        for i_spin in 0..self.mol.spin_channel {
            if i_spin ==0  {
                py2fch(format!("{}.fchk", self.mol.geom.name), nbf, nif, &self.eigenvectors[i_spin].data, 'a', &self.eigenvalues[i_spin], 0);
            } else {
                py2fch(format!("{}.fchk", self.mol.geom.name), nbf, nif, &self.eigenvectors[i_spin].data, 'b', &self.eigenvalues[i_spin], 0);
            }
        }
    }

} 