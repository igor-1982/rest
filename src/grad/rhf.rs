use std::path::PathBuf;

use tensors::MatrixFull;

use crate::molecule_io::Molecule;
use crate::geom_io::{self, get_charge, get_mass_charge};
use crate::parse_input;

pub const PTR_ZETA: usize = 3;
pub const PTR_RINV_ORIG: usize = 4;
pub const PTR_RINV_ZETA: usize = 7;
pub const AS_RINV_ORIG_ATOM: usize = 17;
pub const PTR_ENV_START: usize = 20;
pub const ATOM_OF: usize = 0;

impl Molecule {

    /// Update origin for operator: $\frac{1}{|r-R_O|}$ <br>
    /// **Note** the unit is Bohr
    pub fn set_rinv_origin(&mut self, coord: Vec<f64>) {
        let mut tmp_slice = &mut self.cint_env[PTR_RINV_ORIG..(PTR_RINV_ORIG+3)].iter_mut();
        let coord_slice = &coord[..3];
        tmp_slice.zip(coord_slice.iter()).for_each(|(tmp, coord)| *tmp = *coord);
    }

    /// Return a temporary mol context which has the rquired origin of 1/r
    /// operator.  The required origin has no effects out of the temporary
    /// context.
    pub fn with_rinv_origin(&mut self, coord: Vec<f64>) {
        // let coord0 = self.cint_env[PTR_RINV_ORIG..(PTR_RINV_ORIG+3)].to_vec();
        self.set_rinv_origin(coord);
        // If set coord failed, set coord0 then. (PySCF may use this to ensure thread safe)

    }

    fn set_rinv(&mut self, zeta: f64, rinv: Vec<f64> ) {
        self.cint_env[PTR_RINV_ZETA] = zeta;
        let mut tmp_slice = &mut self.cint_env[PTR_RINV_ORIG..(PTR_RINV_ORIG+3)].iter_mut();
        let rinv_slice = &rinv[..3];
        tmp_slice.zip(rinv_slice.iter()).for_each(|(tmp, rinv)| *tmp = *rinv);
    }
    
    fn set_mole_with_env_start(&mut self) {
        for i in 0..self.cint_atm.len() {
            self.cint_atm[i][1] += (PTR_ENV_START as i32);
            self.cint_atm[i][3] += (PTR_ENV_START as i32);
        }
        for j in 0..self.cint_bas.len() {
            self.cint_bas[j][5] += (PTR_ENV_START as i32);
            self.cint_bas[j][6] += (PTR_ENV_START as i32);
        }
        let mut env_new = vec![0.0_f64; PTR_ENV_START];
        env_new.append(&mut self.cint_env.clone());
        self.cint_env = env_new;
    }
/// set temporary mol context for rinv usage
/// In python:
/// Return a temporary mol context in which the rinv operator (1/r) is
/// treated like the Coulomb potential of a Gaussian charge distribution
/// rho(r) = Norm * exp(-zeta * r^2) at the place of the input atm_id.
/// 
/// In REST:
/// returning a temporary molecule would just be fine
    pub fn with_rinv_at_nucleus(&self, atm_id: usize) -> Molecule {
        let mut temp_mol = self.clone();
        temp_mol.set_mole_with_env_start();
        let zeta_index = self.cint_atm[atm_id][PTR_ZETA] as usize;
        let zeta = self.cint_env[zeta_index];
        let rinv = self.geom.get_coord(atm_id); //untested
        
        if zeta == 0.0 {
            temp_mol.cint_env[AS_RINV_ORIG_ATOM] = atm_id as f64; // required by ecp gradients
            temp_mol.with_rinv_origin(rinv);
            return temp_mol
        } else {
            temp_mol.cint_env[AS_RINV_ORIG_ATOM] = atm_id as f64; // required by ecp gradients
            // If set zeta&rinv failed, set zeta0&rinv0 then. (PySCF may use this to ensure thread safe)
            //let rinv0 = self.cint_env[PTR_RINV_ORIG..(PTR_RINV_ORIG+3)].to_vec();
            //let zeta0 = self.cint_env[PTR_RINV_ZETA];
            temp_mol.set_rinv(zeta, rinv);
            return temp_mol
        };

    }

    /// Offset of every shell in the spherical basis function spectrum
    /// Same as mol.ao_loc_nr in PySCF
    pub fn ao_loc(&self) -> Vec<usize> {
        let mut ao: Vec<usize> = self.cint_fdqc.iter().map(|a| a[0]).collect();
        let len = self.cint_fdqc.len();
        let end = self.cint_fdqc[len-1][0] + self.cint_fdqc[len-1][1];
        ao.push(end);
        ao
    }

    /// AO offsets for each atom.  Return a list, each item of the list gives
    /// start-shell-id, stop-shell-id, start-AO-id, stop-AO-id
    pub fn aoslice_by_atom(&self) -> Vec<Vec<usize>> {

        let natm = self.geom.elem.len();
        let mut aorange = vec![vec![0_usize;4];natm];
        let ao_loc = self.ao_loc();

        if natm == 0 {
            return aorange
        }

        let bas_atom: Vec<usize> = self.cint_bas.iter().map(|vec| vec[ATOM_OF] as usize).collect();
        let mut delimiter = vec![];
        let bas_len = self.cint_bas.len();
        let slice1 = &bas_atom[0..bas_len-1];
        let slice2 = &bas_atom[1..bas_len];
        //println!("slice1 = {:?}", slice1);
        //println!("slice2 = {:?}", slice2);
        
        for i in 0..(bas_len-1) {
            if slice1[i] != slice2[i] {
                delimiter.push(i+1)
            }
        }
        //println!("delimiter = {:?}", delimiter);

        let mut shell_start = vec![1145141919810_usize;natm];
        let mut shell_end = vec![0_usize;natm];

        if natm == delimiter.len() + 1 {
            shell_start[0] = 0;
            let mut shell_start_slice = &mut shell_start[1..natm];
            shell_start_slice.iter_mut().zip(&delimiter).for_each(|(slice, deli)| *slice = *deli);
            shell_end[natm-1] = bas_len;
            let mut shell_end_slice = &mut shell_end[0..natm-1];
            shell_end_slice.iter_mut().zip(&delimiter).for_each(|(slice, deli)| *slice = *deli);
            //println!("shell_start = {:?}", shell_start);
            //println!("shell_end = {:?}", shell_end);

        }
        else {
            // for some atoms missing basis
            shell_start[0] = 0;
            shell_start[bas_atom[0]] = 0;
            //shell_start[bas_atom[delimiter]] = delimiter;
            let indexing_start: Vec<usize> = delimiter.iter().map(|deli| bas_atom[*deli]).collect();
            indexing_start.iter().enumerate().for_each(|(i, idx)| shell_start[*idx] = delimiter[i]);
            //println!("indexing_start = {:?}", indexing_start);

            shell_end[0] = 0;
            shell_end[bas_atom[bas_atom.len()-1]] = bas_len;
            //shell_end[bas_atom[delimiter-1]] = delimiter;
            let indexing_end: Vec<usize> = delimiter.iter().map(|deli| bas_atom[*deli-1]).collect();
            indexing_end.iter().enumerate().for_each(|(i, idx)| shell_end[*idx] = delimiter[i]);
            //println!("indexing_end = {:?}", indexing_end);


            for i in 1..natm {
                /// 1145141919810 here is just a random number to detect if there is any atom without basis set
                if shell_start[i] == 1145141919810 {
                    shell_end[i] = shell_end[i-1];
                    shell_start[i] = shell_end[i];
                }
            }
            //println!("shell_start = {:?}", shell_start);
            //println!("shell_end = {:?}", shell_end);

        }

        (0..4).into_iter().for_each(|i| aorange[i][0] = shell_start[i]);
        (0..4).into_iter().for_each(|i| aorange[i][1] = shell_end[i]);
        let ao_start: Vec<usize> = shell_start.iter().map(|shell| ao_loc[*shell]).collect();
        let ao_end: Vec<usize> = shell_end.iter().map(|shell| ao_loc[*shell]).collect();
        ao_start.iter().enumerate().for_each(|(i, ao)| aorange[i][2] = *ao);
        ao_end.iter().enumerate().for_each(|(i, ao)| aorange[i][3] = *ao);
        
        aorange

    }

    /// Return the corresponding atomic obital slice (start-AO-id, stop-AO-id) of the given atom.
    pub fn mol_slice(&self, atm_id: usize) -> (usize, usize) {
        let aoslice = self.aoslice_by_atom();

        (aoslice[atm_id][2], aoslice[atm_id][3])
    }

}

pub struct Gradient {
    pub mol: Molecule, 
    pub nuc_deriv: MatrixFull<f64>,
    pub ovlp_deriv: Vec<MatrixFull<f64>>,
    pub hcore_deriv: Vec<MatrixFull<f64>>,
    //pub eri_deriv: Vec<RIFull<f64>>,
}

impl Gradient {
    pub fn new(mol: &Molecule) -> Gradient {
        let grad_data = Gradient {
            mol: mol.clone(),
            nuc_deriv: MatrixFull::empty(),
            ovlp_deriv: vec![],
            hcore_deriv: vec![],
        };
        grad_data
    }

    /// Takes ownership of Molecule.
    /// For rinv usage.
    pub fn new_from_mol(mol: Molecule) -> Gradient {
        Gradient {
            mol,
            nuc_deriv: MatrixFull::empty(),
            ovlp_deriv: vec![],
            hcore_deriv: vec![],
        }
    }

    pub fn build(mol: &Molecule) -> Gradient {
        let mut grad_data = Gradient::new(mol);
        grad_data.nuc_deriv = grad_data.calc_nuc_energy_deriv();
        grad_data.ovlp_deriv = grad_data.calc_ovlp_deriv();
        grad_data.hcore_deriv = grad_data.calc_hcore_deriv();

        //println!("ovlp");
        //for i in 0..grad_data.ovlp_deriv.len() {
        //    //grad_data.hcore_deriv[i].formated_output_e_with_threshold(29,"full",1.0e-14);
        //    grad_data.ovlp_deriv[i].formated_output_e_with_threshold(29,"full", 1e-14);
        //    //println!("{:?}", grad_data.hcore_deriv[i])
        //} 

        //println!("hcore");
        //for i in 0..grad_data.hcore_deriv.len() {
        //    //grad_data.hcore_deriv[i].formated_output_e_with_threshold(29,"full",1.0e-14);
        //    grad_data.hcore_deriv[i].formated_output_e_with_threshold(29,"full", 1e-14);
        //    //println!("{:?}", grad_data.hcore_deriv[i])
        //} 

        grad_data

    }

    /// Giving integral of given operation
    pub fn ip_intor(&self, cur_op: &String) -> Vec<MatrixFull<f64>> {
        let mut cint_data = self.mol.initialize_cint(false);

        if cur_op == &String::from("ipkin") {
            cint_data.int1e_ipkin_optimizer_rust();
        } else if cur_op == &String::from("ipovlp") {
            cint_data.int1e_ipovlp_optimizer_rust();

        } else if cur_op == &String::from("ipnuc") {
            cint_data.int1e_ipnuc_optimizer_rust();

        } else if cur_op == &String::from("iprinv") {
            cint_data.int1e_iprinv_optimizer_rust();
        } else {
            panic!("No such operation: {}", cur_op)
        }

        let nbas = self.mol.fdqc_bas.len();
        let mut mat_full = vec![];


        let mut mat_partial_x = MatrixFull::new([nbas,nbas], 0.0_f64);
        let mut mat_partial_y = MatrixFull::new([nbas,nbas], 0.0_f64);
        let mut mat_partial_z = MatrixFull::new([nbas,nbas], 0.0_f64);

        let nbas_shell = self.mol.cint_bas.len();
        for j in 0..nbas_shell {
            let bas_start_j = self.mol.cint_fdqc[j][0];
            let bas_len_j = self.mol.cint_fdqc[j][1];
            for i in 0..nbas_shell {
                let bas_start_i = self.mol.cint_fdqc[i][0];
                let bas_len_i = self.mol.cint_fdqc[i][1];
                let tmp_size = [3*bas_len_i,bas_len_j];
                let mat_local = MatrixFull::from_vec(tmp_size,
                    cint_data.cint_ip_ij(i as i32, j as i32, &cur_op)).unwrap();
                
                let buf_len = mat_local.data.len() / 3;
                
                let mut tmp_slices_x = mat_partial_x.iter_submatrix_mut(
                    bas_start_i..bas_start_i+bas_len_i,
                    bas_start_j..bas_start_j+bas_len_j);
                mat_local.iter().enumerate().filter(|(i,local)| *i < buf_len).zip(tmp_slices_x).for_each(|((i,local),x)| {*x = *local});
                
                let mut tmp_slices_y = mat_partial_y.iter_submatrix_mut(
                    bas_start_i..bas_start_i+bas_len_i,
                    bas_start_j..bas_start_j+bas_len_j);
                mat_local.iter().enumerate().filter(|(i,local)| *i>=buf_len && *i < 2*buf_len).zip(tmp_slices_y).for_each(|((i,local),y)| {*y = *local});

                let mut tmp_slices_z = mat_partial_z.iter_submatrix_mut(
                    bas_start_i..bas_start_i+bas_len_i,
                    bas_start_j..bas_start_j+bas_len_j);
                mat_local.iter().enumerate().filter(|(i,local)| *i>= 2*buf_len && *i< 3*buf_len).zip(tmp_slices_z).for_each(|((i,local),z)| {*z = *local});
            };

        }
        mat_full.push(mat_partial_x.clone());
        mat_full.push(mat_partial_y.clone());
        mat_full.push(mat_partial_z.clone());
        
        cint_data.final_c2r();
        //vec_2d
        
        mat_full
    }

    /// Slicing IP Matrix by atomic orbital 
    pub fn orbital_slice(&self, mat: Vec<MatrixFull<f64>>) -> Vec<MatrixFull<f64>> {
        let natm = self.mol.geom.elem.len();
        let nao = self.mol.num_basis;
        let mut mat_final = vec![MatrixFull::new([nao;2], 0.0_f64); natm*3];
        for atm_id in 0..natm {
            let (sa0, sa1) = self.mol.mol_slice(atm_id);
            for axis in 0..3 {
                let data_slice = &mat[axis][(.., sa0..sa1)];
                let mut mut_slice = mat_final[atm_id*3 + axis].iter_columns_mut(sa0..sa1).flatten();
                mut_slice.zip(data_slice).for_each(|(mu, data)| *mu += *data);
            }
        };
        mat_final
    }

    /// Derivatives of nuclear repulsion energy with reference to nuclear coordinates
    pub fn calc_nuc_energy_deriv(&self) -> MatrixFull<f64> {
        self.mol.geom.calc_nuc_energy_deriv()
    }

    pub fn calc_ovlp_deriv(&self) -> Vec<MatrixFull<f64>>  {
        let mut cur_op = String::from("ipovlp");
        //let mut temp_mol = self.mol.clone();
        //temp_mol.set_mole_with_env_start();
        //let mut temp_grad = Gradient::new_from_mol(temp_mol);
        //let mut mat = temp_grad.ip_intor(&cur_op);
        //let mut mat_final = temp_grad.orbital_slice(mat);
        let mut mat = self.ip_intor(&cur_op);
        let mut mat_final = self.orbital_slice(mat);
        
        // swapaxes(-1,-2), equals transposing every MatrixFull in vec
        let mat_final_t: Vec<MatrixFull<f64>> = mat_final.iter().map(|mat| mat.transpose()).collect();

        mat_final.iter_mut().zip(mat_final_t).for_each(|(mat, mat_t)| *mat += mat_t);
        
        mat_final
    }


    pub fn calc_hcore_deriv(&self) -> Vec<MatrixFull<f64>> {
        let mut kin = self.calc_ipkin();
        let mut nuc = self.calc_ipnuc();
        // nuc is not symmetric, therefore it is transposed first
        nuc.iter_mut().for_each(|nuc| *nuc = nuc.transpose());
        // kin + nuc
        kin.iter_mut().zip(nuc).for_each(|(kin,nuc)| *kin -= nuc);
        let mut mat_final = self.orbital_slice(kin);

        let mut rinv = self.calc_iprinv();
        mat_final.iter_mut().zip(rinv).for_each(|(mat,rinv)| *mat += rinv); //rinv is already negative in calc_rinv

        // swapaxes(-1,-2), equals transposing every MatrixFull in vec
        let mat_final_t: Vec<MatrixFull<f64>> = mat_final.iter().map(|mat| mat.transpose()).collect();

        mat_final.iter_mut().zip(mat_final_t).for_each(|(mat, mat_t)| *mat += mat_t);
        
        mat_final

    }


    pub fn calc_ipkin(&self) -> Vec<MatrixFull<f64>> {
        let cur_op = String::from("ipkin");
        let mat = self.ip_intor(&cur_op);
        //let mut mat_final = self.orbital_slice(mat);
        
        // now orbital slice and swapaxes are operated in calc_hcore_deriv
        // swapaxes(-1,-2), equals transposing every MatrixFull in vec
        //let mat_final_t: Vec<MatrixFull<f64>> = mat_final.iter().map(|mat| mat.transpose()).collect();

        //mat_final.iter_mut().zip(mat_final_t).for_each(|(mat, mat_t)| *mat += mat_t);
        
        //mat_final
        mat
    }


    pub fn calc_ipnuc(&self) -> Vec<MatrixFull<f64>> {
        let cur_op = String::from("ipnuc");
        let mut mat = self.ip_intor(&cur_op);

        // nuc is not symmetric, therefore it is transposed first in calc_hcore_deriv
        // mat.iter_mut().for_each(|mat| *mat = mat.transpose());

        //let mut mat_final = self.orbital_slice(mat);
        
        // now orbital slice and swapaxes are operated in calc_hcore_deriv
        // swapaxes(-1,-2), equals transposing every MatrixFull in vec
        //let mat_final_t: Vec<MatrixFull<f64>> = mat_final.iter().map(|mat| mat.transpose()).collect();

        //mat_final.iter_mut().zip(mat_final_t).for_each(|(mat, mat_t)| {  *mat += mat_t; 
        //                                                                 *mat *= -1.0 });
        //mat_final
        mat

    }

    pub fn iprinv_intor(&self) -> Vec<MatrixFull<f64>> {
        let cur_op = String::from("iprinv");
        let mut mat = self.ip_intor(&cur_op);
        mat
    }

    // shape (natm*3, nao, nao)
    pub fn calc_iprinv(&self) -> Vec<MatrixFull<f64>> {
        let mut mat_full = vec![];
        let charge = get_charge(&self.mol.geom.elem);
        let natm = self.mol.geom.elem.len();
        for atm_id in 0..natm {
            let temp_mol = self.mol.with_rinv_at_nucleus(atm_id);
            //println!("atm = {:?}", temp_mol.cint_atm);
            //println!("atm_len = {:?}", temp_mol.cint_atm.len());
            //println!("bas = {:?}", temp_mol.cint_bas);
            //println!("bas_len = {:?}", temp_mol.cint_bas.len());
            //println!("env = {:?}", temp_mol.cint_env);
            //println!("env_len = {:?}", temp_mol.cint_env.len());
            let temp_grad = Gradient::new_from_mol(temp_mol);
            let mut temp_iprinv = temp_grad.iprinv_intor();
            temp_iprinv.iter_mut().for_each(|mat| 
                mat.data.iter_mut().for_each(|data| *data *= -charge[atm_id]));
            mat_full.append(&mut temp_iprinv)
        }
        
        // swapaxes(-1,-2), equals transposing every MatrixFull in vec
        // already swap in calc_hcore
        //let mat_full_t: Vec<MatrixFull<f64>> = mat_full.iter().map(|mat| mat.transpose()).collect();
        //mat_full.iter_mut().zip(mat_full_t).for_each(|(mat, mat_t)| *mat += mat_t);

        mat_full


    }


    pub fn calc_xc_deriv(){

    }

    pub fn calc_veff_deriv(){

    }


}




#[test]
fn test_charge() {
    let elem_list = vec![String::from("N"), String::from("H"), String::from("H"), String::from("H")];
    let mass_charge = get_mass_charge(&elem_list);
    let charge: Vec<f64> = mass_charge.into_iter().map(|(m,c)| c).collect();
    println!("charge = {:?}", charge);
}

#[test]
fn test_charge_2nd() {
    let elem_list = vec![String::from("N"), String::from("H"), String::from("H"), String::from("H")];
    let charge = get_charge(&elem_list);
    println!("charge = {:?}", charge);
}

