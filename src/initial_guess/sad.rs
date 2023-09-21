use tensors::{MatrixFull, BasicMatrix};
use crate::molecule_io::Molecule;
use crate::ctrl_io::InputKeywords;
use crate::geom_io::{GeomCell, formated_element_name};
use crate::scf_io::scf;
use std::collections::HashMap;

pub fn initial_guess_from_sad(mol: &Molecule) -> Vec<MatrixFull<f64>> {
    let mut elem_name: Vec<String> = vec![];
    let mut dms_alpha: Vec<MatrixFull<f64>> = vec![];
    let mut dms_beta: Vec<MatrixFull<f64>> = vec![];
    //let mut index_elem: Vec<usize> = vec![];
    mol.geom.elem.iter().for_each(|ielem| {
        let mut index = 0;

        //check if the dm of the given element available.
        let flag =  elem_name.iter().enumerate()
        .fold(false, |flag, (j,jelem)| {
            if jelem.eq(ielem) {
                index = j;
                true
            } else {
                flag
            }
        });
        elem_name.push(ielem.to_string());
        //println!("?:{:?}", &flag);
        if ! flag {
            if mol.ctrl.print_level>0 {println!("Generate SAD of {}", &ielem)};

            let mut atom_ctrl = InputKeywords::init_ctrl();
            atom_ctrl.xc = String::from("hf");
            atom_ctrl.basis_path = mol.ctrl.basis_path.clone();
            atom_ctrl.basis_type = mol.ctrl.basis_type.clone();
            atom_ctrl.auxbas_path = mol.ctrl.auxbas_path.clone();
            atom_ctrl.auxbas_type = mol.ctrl.auxbas_type.clone();
            atom_ctrl.eri_type = String::from("ri_v");
            atom_ctrl.num_threads = Some(1);
            atom_ctrl.mixer = "diis".to_string();
            atom_ctrl.initial_guess = "hcore".to_string();
            atom_ctrl.print_level = 0;
            atom_ctrl.atom_sad = true;
            atom_ctrl.charge = 0.0_f64;
            let (spin, spin_channel, spin_polarization) = ctrl_setting_atom_sad(ielem);
            atom_ctrl.spin = spin;
            atom_ctrl.spin_channel = spin_channel;
            atom_ctrl.spin_polarization = spin_polarization;
            //atom_ctrl.spin = 1.0;
            //atom_ctrl.spin_channel = 1;
            //atom_ctrl.spin_polarization = false;

            let mut atom_geom = GeomCell::init_geom();
            atom_geom.name = ielem.to_string();
            atom_geom.position = MatrixFull::from_vec([3,1], vec![0.000,0.000,0.000]).unwrap();
            atom_geom.elem = vec![ielem.to_string()];

            let mut atom_mol = Molecule::build_native(atom_ctrl,atom_geom).unwrap();

            let mut atom_scf = scf(atom_mol).unwrap();

            if atom_scf.mol.spin_channel == 1 {
                let dm = atom_scf.density_matrix[0].clone()*0.5;
                //println!("dm: {:?}", dm.size);
                dms_alpha.push(dm.clone());
                dms_beta.push(dm);
            } else {
                dms_alpha.push(atom_scf.density_matrix[0].clone());
                dms_beta.push(atom_scf.density_matrix[1].clone());
            }
        } else {
            let dm = dms_alpha[index].clone();
            dms_alpha.push(dm);
            let dm = dms_beta[index].clone();
            dms_beta.push(dm);
            //println!("index: {:?}", &index);
        }

    });

    if mol.spin_channel == 1{
        let dm_all = vec![block_diag(&dms_alpha)+block_diag(&dms_beta), MatrixFull::empty()];
        //println!("dm size: {:?}", dm_all[0].size);
        dm_all
    } else {
        vec![block_diag(&dms_alpha), block_diag(&dms_beta)]
    }



}

pub fn block_diag_specific(atom_dms: &HashMap<String,Vec<MatrixFull<f64>>>,elem: &Vec<String>) -> (MatrixFull<f64>, MatrixFull<f64>) {
    let mut atom_size = 0;
    elem.iter().for_each(|ielem| {
        if let Some(dm) = &atom_dms.get(ielem) {
            atom_size += &dm[0].size[0];
        } else {
            panic!("Error: Unknown elemement ({}), for which the density matrix is not yet prepared", ielem);
        }
    });
    let mut dms_alpha = MatrixFull::new([atom_size;2], 0.0);
    let mut dms_beta = MatrixFull::new([atom_size;2], 0.0);
    //let dms_alpha = dms.get_mut(0).unwrap();
    //let dms_beta = dms.get_mut(1).unwrap();
    let mut ao_index = 0;
    elem.iter().for_each(|ielem| {
        if let Some(dm_atom_vec) = &atom_dms.get(ielem) {
            let dm_atom_alpha = dm_atom_vec.get(0).unwrap();
            let dm_atom_beta  = dm_atom_vec.get(1).unwrap();
            let loc_length = dm_atom_alpha.size[0];

            dms_alpha.copy_from_matr(ao_index..ao_index+loc_length, ao_index..ao_index+loc_length,
                dm_atom_alpha, 0..loc_length, 0..loc_length);
            dms_beta.copy_from_matr(ao_index..ao_index+loc_length, ao_index..ao_index+loc_length,
                dm_atom_beta, 0..loc_length, 0..loc_length);
            ao_index += loc_length;
        }
    });

    (dms_alpha, dms_beta)
}

pub fn block_diag(dms: &Vec<MatrixFull<f64>>) -> MatrixFull<f64>{
    //scipy.linalg.block_diag()
    let mut atom_size = 0;
    dms.iter().for_each(|dm_atom|{
        atom_size += dm_atom.size[0];
    });
    let mut dm = MatrixFull::new([atom_size;2], 0.0);
    let mut ao_index = 0;
    dms.iter().for_each(|dm_atom|{
        dm_atom.iter_columns_full().zip(dm.iter_columns_mut(ao_index..ao_index+dm_atom.size[0]))
            .for_each(|(x,y)|{
                for i in (0..dm_atom.size[0]){
                    y[ao_index+i] = x[i];
                }

            });
        ao_index += dm_atom.size[0];
    });
    dm
}

pub fn ctrl_setting_atom_sad(elem: &String) -> (f64,usize,bool) {
    match &formated_element_name(elem)[..] {
        "H" | "Li" | "Na" | "K"  | "Rb" | "Cs" | "Fr" => (2.0, 2, true),
        "B" | "Al" | "Ga" | "In" | "Tl" | "Nh" => (1.0, 1, false),
        "C" | "Si" | "Ge" | "Sn" | "Pb" | "Fl" => (1.0, 1, false),
        "N" | "P"  | "As" | "Sb" | "Bi" | "Mc" => (1.0, 1, false),
        "O" | "S"  | "Se" | "Te" | "Po" | "Lv" => (1.0, 1, false),
        "F" | "Cl" | "Br" | "I"  | "At" | "Ts" => (1.0, 1, false),
        _ => (1.0,1,false)
    }

}

pub fn generate_occupation(elem: &String,num_basis: usize) -> ([Vec<f64>;2],[usize;2],[usize;2]) {
    match &formated_element_name(elem)[..] {
        "H" => {
            let mut occ_a = vec![1.0];
            occ_a.extend(vec![0.0;num_basis-1]);
            ([occ_a,vec![0.0;num_basis]],[0,0],[1,0])
        },
        "Li" => {
            let mut occ_a = vec![2.0, 1.0];
            occ_a.extend(vec![0.0;num_basis-2]);
            ([occ_a,vec![0.0;num_basis]],[1,0],[2,0])
        },
        "Na" => {
            let mut occ_a = vec![2.0, 2.0, 2.0, 2.0, 2.0, 1.0];
            occ_a.extend(vec![0.0;num_basis-6]);
            ([occ_a,vec![0.0;num_basis]],[5,0],[6,0])
        },
        "K" => {
            let mut occ_a = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0];
            occ_a.extend(vec![0.0;num_basis-10]);
            ([occ_a,vec![0.0;num_basis]],[9,0],[10,0])
        },
        "Rb" => {
            let mut occ_a = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0];
            occ_a.extend(vec![0.0;num_basis-19]);
            ([occ_a,vec![0.0;num_basis]],[18,0],[19,0])
        },
        

        "Be" => {
            let mut occ_a = vec![2.0, 2.0];
            occ_a.extend(vec![0.0;num_basis-2]);
            ([occ_a,vec![0.0;num_basis]],[1,0],[2,0])
        },
        "Mg" => {
            let mut occ_a = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0];
            occ_a.extend(vec![0.0;num_basis-6]);
            ([occ_a,vec![0.0;num_basis]],[5,0],[6,0])
        },
        "Ca" => {
            let mut occ_a = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0];
            occ_a.extend(vec![0.0;num_basis-10]);
            ([occ_a,vec![0.0;num_basis]],[9,0],[10,0])
        },
        "Sr" => {
            let mut occ_a = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0];
            occ_a.extend(vec![0.0;num_basis-19]);
            ([occ_a,vec![0.0;num_basis]],[18,0],[19,0])
        },

        "Sc" => {
            let mut occ_a = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.2, 0.2, 0.2, 0.2, 0.2, 2.0];
            occ_a.extend(vec![0.0;num_basis-15]);
            ([occ_a,vec![0.0;num_basis]],[14,0],[15,0])
        },
        "Ti" => {
            let mut occ_a = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.4, 0.4, 0.4, 0.4, 0.4, 2.0];
            occ_a.extend(vec![0.0;num_basis-15]);
            ([occ_a,vec![0.0;num_basis]],[14,0],[15,0])
        },
        "V" => {
            let mut occ_a = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.6, 0.6, 0.6, 0.6, 0.6, 2.0];
            occ_a.extend(vec![0.0;num_basis-15]);
            ([occ_a,vec![0.0;num_basis]],[14,0],[15,0])
        },
        "Cr" => {
            let mut occ_a = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
            occ_a.extend(vec![0.0;num_basis-15]);
            ([occ_a,vec![0.0;num_basis]],[14,0],[15,0])
        },
        "Mn" => {
            let mut occ_a = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0];
            occ_a.extend(vec![0.0;num_basis-15]);
            ([occ_a,vec![0.0;num_basis]],[14,0],[15,0])
        },
        "Fe" => {
            let mut occ_a = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.2, 1.2, 1.2, 1.2, 1.2, 2.0];
            occ_a.extend(vec![0.0;num_basis-15]);
            ([occ_a,vec![0.0;num_basis]],[14,0],[15,0])
        },
        "Co" => {
            let mut occ_a = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.4, 1.4, 1.4, 1.4, 1.4, 2.0];
            occ_a.extend(vec![0.0;num_basis-15]);
            ([occ_a,vec![0.0;num_basis]],[14,0],[15,0])
        },
        "Ni" => {
            let mut occ_a = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.6, 1.6, 1.6, 1.6, 1.6, 2.0];
            occ_a.extend(vec![0.0;num_basis-15]);
            ([occ_a,vec![0.0;num_basis]],[14,0],[15,0])
        },
        "Cu" => {
            let mut occ_a = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0];
            occ_a.extend(vec![0.0;num_basis-15]);
            ([occ_a,vec![0.0;num_basis]],[14,0],[15,0])
        },
        "Zn" => {
            let mut occ_a = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0];
            occ_a.extend(vec![0.0;num_basis-15]);
            ([occ_a,vec![0.0;num_basis]],[14,0],[15,0])
        },

        "Y" => {
            let mut occ_a = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.2, 0.2, 0.2, 0.2, 0.2, 2.0, 2.0];
            occ_a.extend(vec![0.0;num_basis-24]);
            ([occ_a,vec![0.0;num_basis]],[23,0],[24,0])
        },
        "Zr" => {
            let mut occ_a = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.4, 0.4, 0.4, 0.4, 0.4, 2.0];
            occ_a.extend(vec![0.0;num_basis-24]);
            ([occ_a,vec![0.0;num_basis]],[23,0],[24,0])
        },
        "Nb" => {
            let mut occ_a = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.8, 0.8, 0.8, 0.8, 0.8, 1.0];
            occ_a.extend(vec![0.0;num_basis-24]);
            ([occ_a,vec![0.0;num_basis]],[23,0],[24,0])
        },
        "Mo" => {
            let mut occ_a = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
            occ_a.extend(vec![0.0;num_basis-24]);
            ([occ_a,vec![0.0;num_basis]],[23,0],[24,0])
        },
        "Tc" => {
            let mut occ_a = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0];
            occ_a.extend(vec![0.0;num_basis-24]);
            ([occ_a,vec![0.0;num_basis]],[23,0],[24,0])
        },
        "Ru" => {
            let mut occ_a = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.4, 1.4, 1.4, 1.4, 1.4, 1.0];
            occ_a.extend(vec![0.0;num_basis-24]);
            ([occ_a,vec![0.0;num_basis]],[23,0],[24,0])
        },
        "Rh" => {
            let mut occ_a = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.6, 1.6, 1.6, 1.6, 1.6, 1.0];
            occ_a.extend(vec![0.0;num_basis-24]);
            ([occ_a,vec![0.0;num_basis]],[23,0],[24,0])
        },
        "Pd" => {
            let mut occ_a = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0];
            occ_a.extend(vec![0.0;num_basis-24]);
            ([occ_a,vec![0.0;num_basis]],[23,0],[24,0])
        },
        "Ag" => {
            let mut occ_a = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0];
            occ_a.extend(vec![0.0;num_basis-24]);
            ([occ_a,vec![0.0;num_basis]],[23,0],[24,0])
        },
        "Cd" => {
            let mut occ_a = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0];
            occ_a.extend(vec![0.0;num_basis-24]);
            ([occ_a,vec![0.0;num_basis]],[23,0],[24,0])
        },

        "B" => {
            let mut occ_a = vec![2.0, 2.0, 0.333333333, 0.33333333, 0.333333333];
            occ_a.extend(vec![0.0;num_basis-5]);
            ([occ_a,vec![0.0;num_basis]],[4,0],[5,0])
        },
        "Al" => {
            let mut occ_a = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.333333333, 0.33333333, 0.333333333];
            occ_a.extend(vec![0.0;num_basis-9]);
            ([occ_a,vec![0.0;num_basis]],[8,0],[9,0])
        },
        "Ga" => {
            let mut occ_a = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.333333333, 0.33333333, 0.333333333];
            occ_a.extend(vec![0.0;num_basis-18]);
            ([occ_a,vec![0.0;num_basis]],[17,0],[18,0])
        },
        "In" => {
            let mut occ_a = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0 ,2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.333333333, 0.33333333, 0.333333333];
            occ_a.extend(vec![0.0;num_basis-27]);
            ([occ_a,vec![0.0;num_basis]],[26,0],[27,0])
        },
        "C" => {
            let mut occ_a = vec![2.0, 2.0, 0.666666667, 0.666666667, 0.666666667];
            occ_a.extend(vec![0.0;num_basis-5]);
            ([occ_a,vec![0.0;num_basis]],[4,0],[5,0])
        },
        "Si" => {
            let mut occ_a = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.666666667, 0.666666667, 0.666666667];
            occ_a.extend(vec![0.0;num_basis-9]);
            ([occ_a,vec![0.0;num_basis]],[8,0],[9,0])
        },
        "Ge" => {
            let mut occ_a = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.666666667, 0.666666667, 0.666666667];
            occ_a.extend(vec![0.0;num_basis-18]);
            ([occ_a,vec![0.0;num_basis]],[17,0],[18,0])
        },
        "Sn" => {
            let mut occ_a = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0 ,2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.666666667, 0.666666667, 0.666666667];
            occ_a.extend(vec![0.0;num_basis-27]);
            ([occ_a,vec![0.0;num_basis]],[26,0],[27,0])
        },
        "N" => {
            let mut occ_a = vec![2.0, 2.0, 1.0, 1.0, 1.0];
            occ_a.extend(vec![0.0;num_basis-5]);
            ([occ_a,vec![0.0;num_basis]],[4,0],[5,0])
        },
        "P" => {
            let mut occ_a = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0];
            occ_a.extend(vec![0.0;num_basis-9]);
            ([occ_a,vec![0.0;num_basis]],[8,0],[9,0])
        },
        "As" => {
            let mut occ_a = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0];
            occ_a.extend(vec![0.0;num_basis-18]);
            ([occ_a,vec![0.0;num_basis]],[17,0],[18,0])
        },
        "Sb" => {
            let mut occ_a = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0 ,2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0];
            occ_a.extend(vec![0.0;num_basis-27]);
            ([occ_a,vec![0.0;num_basis]],[26,0],[27,0])
        },
        "O" => {
            let mut occ_a = vec![2.0,2.0,1.33333333,1.33333333,1.33333333];
            occ_a.extend(vec![0.0;num_basis-5]);
            //occ_b.extend(vec![0.0;num_basis-5]);
            //([occ_a,occ_b],[4,2],[5,3])
            ([occ_a,vec![]],[4,0],[5,0])
        },
        "S" => {
            let mut occ_a = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.33333333,1.33333333,1.33333333];
            occ_a.extend(vec![0.0;num_basis-9]);
            ([occ_a,vec![0.0;num_basis]],[8,0],[9,0])
        },
        "Se" => {
            let mut occ_a = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.33333333,1.33333333,1.33333333];
            occ_a.extend(vec![0.0;num_basis-18]);
            ([occ_a,vec![0.0;num_basis]],[17,0],[18,0])
        },
        "Te" => {
            let mut occ_a = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0 ,2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.33333333,1.33333333,1.33333333];
            occ_a.extend(vec![0.0;num_basis-27]);
            ([occ_a,vec![0.0;num_basis]],[26,0],[27,0])
        },
        "F" => {
            let mut occ_a = vec![2.0,2.0,1.666666667,1.666666667,1.666666667];
            occ_a.extend(vec![0.0;num_basis-5]);
            //occ_b.extend(vec![0.0;num_basis-5]);
            //([occ_a,occ_b],[4,2],[5,3])
            ([occ_a,vec![]],[4,0],[5,0])
        },
        "Cl" => {
            let mut occ_a = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.666666667,1.666666667,1.666666667];
            occ_a.extend(vec![0.0;num_basis-9]);
            ([occ_a,vec![0.0;num_basis]],[8,0],[9,0])
        },
        "Br" => {
            let mut occ_a = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.666666667,1.666666667,1.666666667];
            occ_a.extend(vec![0.0;num_basis-18]);
            ([occ_a,vec![0.0;num_basis]],[17,0],[18,0])
        },
        "I" => {
            let mut occ_a = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0 ,2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.666666667,1.666666667,1.666666667];
            occ_a.extend(vec![0.0;num_basis-27]);
            ([occ_a,vec![0.0;num_basis]],[26,0],[27,0])
        },

        
        _ => ([vec![],vec![]],[0,0],[0,0])
    }
}