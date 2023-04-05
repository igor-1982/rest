use tensors::MatrixFull;
use crate::molecule_io::Molecule;
use crate::ctrl_io::InputKeywords;
use crate::geom_io::{GeomCell, formated_element_name};
use crate::scf_io::scf;

pub fn initial_guess_from_sad(mol: &Molecule) -> Vec<MatrixFull<f64>> {
    let mut elem_name: Vec<String> = vec![];
    let mut dms_alpha: Vec<MatrixFull<f64>> = vec![];
    let mut dms_beta: Vec<MatrixFull<f64>> = vec![];
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

        if ! flag {
            if mol.ctrl.print_level>0 {println!("Generate SAD of {}", &ielem)};

            elem_name.push(ielem.to_string());

            let mut atom_ctrl = InputKeywords::new();
            atom_ctrl.xc = String::from("hf");
            atom_ctrl.basis_path = mol.ctrl.basis_path.clone();
            atom_ctrl.basis_type = mol.ctrl.basis_type.clone();
            atom_ctrl.auxbas_path = mol.ctrl.auxbas_path.clone();
            atom_ctrl.auxbas_type = mol.ctrl.auxbas_type.clone();
            atom_ctrl.eri_type = mol.ctrl.eri_type.clone();
            atom_ctrl.num_threads = Some(1);
            atom_ctrl.mixer = "diis".to_string();
            atom_ctrl.initial_guess = "hcore".to_string();
            atom_ctrl.print_level = 1;
            atom_ctrl.atom_sad = true;
            atom_ctrl.charge = 0.0_f64;
            let (spin, spin_channel, spin_polarization) = ctrl_setting_atom_sad(ielem);
            atom_ctrl.spin = spin;
            atom_ctrl.spin_channel = spin_channel;
            atom_ctrl.spin_polarization = spin_polarization;
            //atom_ctrl.spin = 1.0;
            //atom_ctrl.spin_channel = 1;
            //atom_ctrl.spin_polarization = false;

            let mut atom_geom = GeomCell::new();
            atom_geom.name = ielem.to_string();
            atom_geom.position = MatrixFull::from_vec([3,1], vec![0.000,0.000,0.000]).unwrap();
            atom_geom.elem = vec![ielem.to_string()];

            let mut atom_mol = Molecule::build_native(atom_ctrl,atom_geom).unwrap();

            let mut atom_scf = scf(atom_mol).unwrap();

            if atom_scf.mol.spin_channel == 1 {
                let dm = atom_scf.density_matrix[0].clone()*0.5;
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
        }

    });

    if mol.spin_channel == 1{
        vec![block_diag(&dms_alpha)+block_diag(&dms_beta), MatrixFull::empty()]
    } else {
        vec![block_diag(&dms_alpha), block_diag(&dms_beta)]
    }


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
        "B" | "Al" | "Ga" | "In" | "Tl" | "Nh" => (2.0, 2, true),
        "C" | "Si" | "Ge" | "Sn" | "Pb" | "Fl" => (3.0, 2, true),
        "N" | "P"  | "As" | "Sb" | "Bi" | "Mc" => (4.0, 2, true),
        "O" | "S"  | "Se" | "Te" | "Po" | "Lv" => (3.0, 2, true),
        "F" | "Cl" | "Br" | "I"  | "At" | "Ts" => (2.0, 2, true),
        _ => (1.0,1,false)
    }

}