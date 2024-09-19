use crate::constants::{F_SHELL, XE_SHELL, NELE_IN_SHELLS, SPECIES_INFO};
use crate::geom_io::formated_element_name;
use crate::scf_io::SCFType;
use crate::molecule_io::Molecule;
use serde::{Deserialize, Serialize};

#[derive(Clone,Copy,Debug, Deserialize, Serialize)]
pub enum OCCType {
    INTEGER,
    FRAC,
    ATMSAD,
}

pub mod force_state_occupation;

pub fn generate_occupation_frac_occ(mol: &Molecule, scftype: &SCFType, eigenvalues: &[Vec<f64>;2], tolerant: f64) -> ([Vec<f64>;2],[usize;2],[usize;2]) {
    let num_state = mol.num_state;
    let spin_channel = mol.ctrl.spin_channel;
    let num_elec = &mol.num_elec;
    let mut occupation:[Vec<f64>;2] = [vec![],vec![]];
    let mut lumo:[usize;2] = [0,0];
    let mut homo:[usize;2] = [0,0];
    match scftype {
        SCFType::RHF => {
            let occ_num = 2.0;
            let i_spin = 0_usize;

            let num_occs = (num_elec[0]/2.0).ceil() as usize;
            let mo_energy = &eigenvalues[i_spin];
            let tmp_homo = mo_energy[num_occs-1];
            let tmp_lumo = mo_energy[num_occs];
            let mut frac_orb_list: Vec<usize> = Vec::new();
            // determine which orbitals are degenerated with HOMO
            mo_energy.iter().enumerate().for_each(|(i, orb_ene)| {
                if (orb_ene - tmp_homo).abs() < tolerant {
                    frac_orb_list.push(i)
                };
            });

            // fill in the orbitals with integer occupation
            occupation[i_spin] = vec![0.0;num_state];
            occupation[i_spin][..num_occs].iter_mut().enumerate().for_each(|(i,each_occ)| {
                if ! frac_orb_list.contains(&i) {
                    *each_occ = occ_num;
                }
            });
            let occupied_elec = occupation[i_spin].iter().fold(0.0, |sum, i| sum+i);
            lumo[i_spin] = (occupied_elec/2.0) as usize+frac_orb_list.len();
            homo[i_spin] = lumo[i_spin]-1;

            let left_elec = num_elec[0] as f64 - occupied_elec;
            let frac_elec = left_elec/frac_orb_list.len() as f64;
            frac_orb_list.iter().for_each(|i| {
                occupation[i_spin][*i] = frac_elec
            });
        },
        SCFType::UHF => {
            let occ_num = 1.0;
            //let i_spin = 0_usize;
            for i_spin in 0..spin_channel {
                let num_occs = num_elec[i_spin + 1].ceil() as usize;
                let mo_energy = &eigenvalues[i_spin];
                let tmp_homo = mo_energy[num_occs-1];
                let tmp_lumo = mo_energy[num_occs];
                let mut frac_orb_list: Vec<usize> = vec![];
                // determine which orbitals are degenerated with HOMO
                mo_energy.iter().enumerate().for_each(|(i, orb_ene)| {
                    if (orb_ene - tmp_homo).abs() < 1.0e-3 {
                        frac_orb_list.push(i)
                    };
                });
                // fill in the orbitals with integer occupation
                occupation[i_spin] = vec![0.0;num_state];
                occupation[i_spin][..num_occs].iter_mut().enumerate().for_each(|(i,each_occ)| {
                    if ! frac_orb_list.contains(&i) {
                        *each_occ = occ_num;
                    }
                });
                let occupied_elec = occupation[i_spin].iter().fold(0.0, |sum, i| sum+i);
                lumo[i_spin] = (occupied_elec/2.0) as usize+frac_orb_list.len();
                homo[i_spin] = lumo[i_spin]-1;

                let left_elec = num_elec[0] as f64 - occupied_elec;
                let frac_elec = left_elec/frac_orb_list.len() as f64;
                frac_orb_list.iter().for_each(|i| {
                    occupation[i_spin][*i] = frac_elec
                });
            }

        },
        _ => {}
    }
    (occupation,homo,lumo)
}


pub fn generate_occupation_integer(mol: &Molecule, scftype: &SCFType) -> ([Vec<f64>;2],[usize;2],[usize;2]) {
    let num_state = mol.num_state;
    let spin_channel = mol.ctrl.spin_channel;
    let num_elec = &mol.num_elec;
    let mut occupation:[Vec<f64>;2] = [vec![],vec![]];
    let mut lumo:[usize;2] = [0,0];
    let mut homo:[usize;2] = [0,0];
    //let occ_num = 
    match scftype {
        SCFType::RHF => {
            let occ_num = 2.0;
            let i_spin = 0_usize;
            occupation[i_spin] = vec![0.0;num_state];
            let mut left_elec_spin = num_elec[i_spin+1];
            let mut index_i = 0_usize;
            while  left_elec_spin > 0.0 && index_i<=num_state {
                occupation[i_spin][index_i] = (left_elec_spin*occ_num).min(occ_num);
                index_i += 1;
                left_elec_spin -= 1.0;
            }
            // make sure there is at least one LUMO
            if index_i > num_state-1 && left_elec_spin>0.0 {
                panic!("Error:: the number of molecular orbitals is smaller than the number of electrons in the {}-spin channel\n num_state: {}; num_elec_alpha: {}", 
                       i_spin,num_state, num_elec[i_spin]);
            } else {
                lumo[i_spin] = index_i;
                homo[i_spin] = index_i-1;
            }; 
        },
        SCFType::ROHF => {
            let occ_num = 1.0;
            (0..2).for_each(|i_spin| {
                occupation[i_spin] = vec![0.0;num_state];
                let mut left_elec_spin = num_elec[i_spin+1];
                let mut index_i = 0_usize;
                while  left_elec_spin > 0.0 && index_i<=num_state {
                    occupation[i_spin][index_i] = (left_elec_spin*occ_num).min(occ_num);
                    index_i += 1;
                    left_elec_spin -= 1.0;
                }
                // make sure there is at least one LUMO
                if index_i > num_state-1 && left_elec_spin>0.0 {
                    panic!("Error:: the number of molecular orbitals is smaller than the number of electrons in the {}-spin channel\n num_state: {}; num_elec_alpha: {}", 
                           i_spin,num_state, num_elec[i_spin]);
                } else {
                    lumo[i_spin] = index_i;
                    homo[i_spin] = index_i-1;
                }; 
            });
        },
        SCFType::UHF => {
            let occ_num = 1.0;
            (0..2).for_each(|i_spin| {
                occupation[i_spin] = vec![0.0;num_state];
                let mut left_elec_spin = num_elec[i_spin+1];
                let mut index_i = 0_usize;
                while  left_elec_spin > 0.0 && index_i<=num_state {
                    occupation[i_spin][index_i] = (left_elec_spin*occ_num).min(occ_num);
                    index_i += 1;
                    left_elec_spin -= 1.0;
                }
                // make sure there is at least one LUMO
                if index_i > num_state-1 && left_elec_spin>0.0 {
                    panic!("Error:: the number of molecular orbitals is smaller than the number of electrons in the {}-spin channel\n num_state: {}; num_elec_alpha: {}", 
                           i_spin,num_state, num_elec[i_spin]);
                } else {
                    lumo[i_spin] = index_i;
                    homo[i_spin] = if index_i==0 {0} else {index_i-1};
                }; 
            });
        }
    };
    //self.occupation = occupation;
    //self.lumo = lumo;
    //self.homo = homo;
    //println!("Occupation: {:?}, {:?}, {:?}, {}, {}",&self.homo,&self.lumo,&self.occupation,self.mol.num_state,self.mol.num_basis);
    (occupation,homo,lumo)
}

pub fn generate_occupation_sad(elem: &String,num_basis: usize, num_ecp: usize) -> ([Vec<f64>;2],[usize;2],[usize;2]) {
    generate_occupation_sad_v02(elem,num_basis, num_ecp)
}

pub fn generate_occupation_sad_v02(elem: &String,num_basis: usize, num_ecp: usize) -> ([Vec<f64>;2],[usize;2],[usize;2]) {
    let frozen_orb = num_ecp/2;
    let num_elec = SPECIES_INFO.get(&elem.as_str()).unwrap().1;
    let mut occ_o: Vec<f64> = vec![];
    let mut rest_elem = num_elec;
    NELE_IN_SHELLS.iter().for_each(|x| {
        let num_orbs = (x/2.0) as usize;
        if rest_elem >= *x {
            occ_o.extend(vec![2.0; num_orbs]);
            rest_elem -= x;
        } else if rest_elem < *x && rest_elem >= 1.0 {
            let num_elec_per_orb = rest_elem / (num_orbs as f64);
            occ_o.extend(vec![num_elec_per_orb; num_orbs]);
            rest_elem = -1.0;
        }
    });
    let num_occ_o = occ_o.len();
    let mut occ_a = cut_ecp_occ(occ_o,num_ecp);
    let num_occ_a = occ_a.len();
    occ_a.extend(vec![0.0;frozen_orb+num_basis-num_occ_o]);
    //println!("{:?}", &occ_a);
    ([occ_a, vec![0.0;num_occ_a]], [num_occ_a-1, 0], [num_occ_a, 0])
}

pub fn generate_occupation_sad_v01(elem: &String,num_basis: usize, num_ecp: usize) -> ([Vec<f64>;2],[usize;2],[usize;2]) {
    let frozen_orb = num_ecp/2;
    match &formated_element_name(elem)[..] {
        "H" => {
            let mut occ_a = cut_ecp_occ(vec![1.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-1]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[0-frozen_orb,0],[1-frozen_orb,0])
        },
        "Li" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 1.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-2]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[1-frozen_orb,0],[2-frozen_orb,0])
        },
        "Na" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 1.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-6]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[5-frozen_orb,0],[6-frozen_orb,0])
        },
        "K" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-10]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[9-frozen_orb,0],[10-frozen_orb,0])
        },

        "Be" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-2]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[1-frozen_orb,0],[2-frozen_orb,0])
        },
        "Mg" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-6]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[5-frozen_orb,0],[6-frozen_orb,0])
        },
        "Ca" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-10]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[9-frozen_orb,0],[10-frozen_orb,0])
        },
        "Sr" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-19]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[18-frozen_orb,0],[19-frozen_orb,0])
        },

        "Sc" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.2, 0.2, 0.2, 0.2, 0.2, 2.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-15]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[14-frozen_orb,0],[15-frozen_orb,0])
        },
        "Ti" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.4, 0.4, 0.4, 0.4, 0.4, 2.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-15]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[14-frozen_orb,0],[15-frozen_orb,0])
        },
        "V" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.6, 0.6, 0.6, 0.6, 0.6, 2.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-15]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[14-frozen_orb,0],[15-frozen_orb,0])
        },
        "Cr" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-15]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[14-frozen_orb,0],[15-frozen_orb,0])
        },
        "Mn" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-15]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[14-frozen_orb,0],[15-frozen_orb,0])
        },
        "Fe" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.2, 1.2, 1.2, 1.2, 1.2, 2.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-15]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[14-frozen_orb,0],[15-frozen_orb,0])
        },
        "Co" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.4, 1.4, 1.4, 1.4, 1.4, 2.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-15]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[14-frozen_orb,0],[15-frozen_orb,0])
        },
        "Ni" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.6, 1.6, 1.6, 1.6, 1.6, 2.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-15]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[14-frozen_orb,0],[15-frozen_orb,0])
        },
        "Cu" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-15]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[14-frozen_orb,0],[15-frozen_orb,0])
        },
        "Zn" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-15]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[14-frozen_orb,0],[15-frozen_orb,0])
        },

        "Y" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.2, 0.2, 0.2, 0.2, 0.2, 2.0, 2.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-24]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[23-frozen_orb,0],[24-frozen_orb,0])
        },
        "Zr" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.4, 0.4, 0.4, 0.4, 0.4, 2.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-24]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[23-frozen_orb,0],[24-frozen_orb,0])
        },
        "Nb" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.8, 0.8, 0.8, 0.8, 0.8, 1.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-24]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[23-frozen_orb,0],[24-frozen_orb,0])
        },
        "Mo" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-24]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[23-frozen_orb,0],[24-frozen_orb,0])
        },
        "Tc" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-24]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[23-frozen_orb,0],[24-frozen_orb,0])
        },
        "Ru" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.4, 1.4, 1.4, 1.4, 1.4, 1.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-24]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[23-frozen_orb,0],[24-frozen_orb,0])
        },
        "Rh" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.6, 1.6, 1.6, 1.6, 1.6, 1.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-24]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[23-frozen_orb,0],[24-frozen_orb,0])
        },
        "Pd" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-24]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[23-frozen_orb,0],[24-frozen_orb,0])
        },
        "Ag" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-24]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[23-frozen_orb,0],[24-frozen_orb,0])
        },
        "Cd" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-24]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[23-frozen_orb,0],[24-frozen_orb,0])
        },

        "B" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 0.333333333, 0.33333333, 0.333333333],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-5]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[4-frozen_orb,0],[5-frozen_orb,0])
        },
        "Al" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.333333333, 0.33333333, 0.333333333],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-9]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[8-frozen_orb,0],[9-frozen_orb,0])
        },
        "Ga" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.333333333, 0.33333333, 0.333333333],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-18]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[17-frozen_orb,0],[18-frozen_orb,0])
        },
        "In" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0 ,2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.333333333, 0.33333333, 0.333333333],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-27]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[26-frozen_orb,0],[27-frozen_orb,0])
        },
        "C" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 0.666666667, 0.666666667, 0.666666667],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-5]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[4-frozen_orb,0],[5-frozen_orb,0])
        },
        "Si" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.666666667, 0.666666667, 0.666666667],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-9]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[8-frozen_orb,0],[9-frozen_orb,0])
        },
        "Ge" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.666666667, 0.666666667, 0.666666667],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-18]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[17-frozen_orb,0],[18-frozen_orb,0])
        },
        "Sn" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0 ,2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.666666667, 0.666666667, 0.666666667],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-27]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[26-frozen_orb,0],[27-frozen_orb,0])
        },
        "O" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 0.0, 1.333333, 1.333333, 1.333333],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-6]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[5-frozen_orb,0],[6-frozen_orb,0])
        },
        "N" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 1.0, 1.0, 1.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-5]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[4-frozen_orb,0],[5-frozen_orb,0])
        },
        "P" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-9]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[8-frozen_orb,0],[9-frozen_orb,0])
        },
        "As" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-18]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[17-frozen_orb,0],[18-frozen_orb,0])
        },
        "Sb" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0 ,2.0, 2.0, 
                                             2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 
                                             2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-27]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[26-frozen_orb,0],[27-frozen_orb,0])
        },
        "S" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.33333333,1.33333333,1.33333333],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-9]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[8-frozen_orb,0],[9-frozen_orb,0])
        },
        "Se" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.33333333,1.33333333,1.33333333],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-18]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[17-frozen_orb,0],[18-frozen_orb,0])
        },
        "Te" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0 ,2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.33333333,1.33333333,1.33333333],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-27]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[26-frozen_orb,0],[27-frozen_orb,0])
        },
        "F" => {
            let mut occ_a = cut_ecp_occ(vec![2.0,2.0,1.666666667,1.666666667,1.666666667],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-5]);
            //occ_b.extend(vec![0.0;frozen_orb+num_basis-5]);
            //([occ_a,occ_b],[4,2],[5,3])
            ([occ_a,vec![]],[4-frozen_orb,0],[5-frozen_orb,0])
        },
        "Cl" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.666666667,1.666666667,1.666666667],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-9]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[8-frozen_orb,0],[9-frozen_orb,0])
        },
        "Br" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.666666667,1.666666667,1.666666667],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-18]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[17-frozen_orb,0],[18-frozen_orb,0])
        },
        "I" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0 ,2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.666666667,1.666666667,1.666666667],num_ecp);
            //println!("{}, {}, {:?}", num_basis, frozen_orb, &occ_a);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-27]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[26-frozen_orb,0],[27-frozen_orb,0])
        },
        "He" => {
            let mut occ_a = cut_ecp_occ(vec![2.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-1]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[0-frozen_orb,0],[1-frozen_orb,0])
        },
        "Ne" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-5]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[4-frozen_orb,0],[5-frozen_orb,0])
        },
        "Ar" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-9]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[8-frozen_orb,0],[9-frozen_orb,0])
        },
        "Kr" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-18]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[17-frozen_orb,0],[18-frozen_orb,0])
        },
        // for 5d elements
        "Os" => {
            let mut occ_o: Vec<f64> = vec![];
            occ_o.extend_from_slice(&XE_SHELL);
            occ_o.extend_from_slice(&F_SHELL);
            occ_o.extend_from_slice(&[1.6;5]);
            let num_occ_o = occ_o.len();
            println!("{:?}", &occ_o);
            let mut occ_a = cut_ecp_occ(occ_o,num_ecp);
            let num_occ_a = occ_a.len();
            println!("{:?}", &occ_a);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-num_occ_o]);
            ([occ_a, vec![0.0;num_occ_a]], [num_occ_a-1, 0], [num_occ_a, 0])
        },
        
        _ => ([vec![],vec![]],[0,0],[0,0])
    }
}

pub fn cut_ecp_occ(occ: Vec<f64>, num_ecp: usize) -> Vec<f64> {
    occ[num_ecp/2..].iter().map(|x| *x).collect::<Vec<f64>>()
}