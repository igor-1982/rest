use std::path::PathBuf;
use std::sync::Arc;
use std::vec;
use itertools::Itertools;
use rest_tensors::RIFull;
use tensors::{MatrixFull, ri};
use tensors::matrix_blas_lapack::{_dgemm_nn, _dgemm_tn};
//use crate::ctrl_io::SCFType;
use crate::molecule_io::Molecule;
use crate::geom_io::{self, get_charge, get_mass_charge};
use crate::scf_io::{SCF, SCFType, scf};

pub const PTR_ZETA: usize = 3;
pub const PTR_RINV_ORIG: usize = 4;
pub const PTR_RINV_ZETA: usize = 7;
pub const AS_RINV_ORIG_ATOM: usize = 17;
pub const PTR_ENV_START: usize = 20;
pub const ATOM_OF: usize = 0;


/// Generate index numbers of diagonal elements in MatrixUpper
fn diag_idx_generator(n: usize) -> Vec<usize> {
    let mut idx = vec![0usize; n];
    let mut value = 2;
    for i in 1..n {
        idx[i] = value;
        value += (i+2);
    }

    println!("idx = {:?}", idx);
    idx
}



/// Expand [i*j, k] MatrixFull to [i,j,k] RIFull
fn matfull_to_rifull(matfull: &MatrixFull<f64>, i: &usize, j: &usize) -> RIFull<f64> {
    let k = matfull.size[1];
    let result = RIFull::from_vec([*i,*j,k], matfull.data.clone()).unwrap();
    result
}


impl Molecule {
    /// Make an auxiliary molecule from a molecule for calculation. 
    /// The auxmol is very similar to the origin mole, 
    /// except its basis-related infomation is cloned from auxbasis information of the original mole.
    pub fn make_auxmol(&self) -> Molecule {

        let mut auxmol = self.clone();
        auxmol.num_basis = auxmol.num_auxbas.clone();
        auxmol.fdqc_bas = auxmol.fdqc_aux_bas.clone();
        auxmol.cint_fdqc = auxmol.cint_aux_fdqc.clone();
        auxmol.cint_bas = auxmol.cint_aux_bas.clone();
        auxmol.cint_atm = auxmol.cint_aux_atm.clone();
        auxmol.cint_env = auxmol.cint_aux_env.clone();

        auxmol
    }

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

    pub fn build(mol: &Molecule, scf: &SCF) -> Gradient {
        let mut grad_data = Gradient::new(mol);
        grad_data.nuc_deriv = grad_data.calc_nuc_energy_deriv();
        grad_data.ovlp_deriv = grad_data.calc_ovlp_deriv();
        grad_data.hcore_deriv = grad_data.calc_hcore_deriv();
        let (vj, vjaux) = grad_data.calc_j(scf);
        let (vk, vkaux) = grad_data.calc_k(scf);



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

    fn ip_2c2e_intor(&self) -> Vec<MatrixFull<f64>> {
        let mut cint_data = self.mol.initialize_cint(true);
        let n_auxbas = self.mol.num_auxbas;
        let mut mat_full = vec![];
        let mut mat_partial_x = MatrixFull::new([n_auxbas,n_auxbas], 0.0_f64);
        let mut mat_partial_y = MatrixFull::new([n_auxbas,n_auxbas], 0.0_f64);
        let mut mat_partial_z = MatrixFull::new([n_auxbas,n_auxbas], 0.0_f64);

        let n_basis_shell = self.mol.cint_bas.len();
        let n_auxbas_shell = self.mol.cint_aux_bas.len();
        cint_data.cint2c2e_ip1_optimizer_rust();
        let mut aux_v = MatrixFull::new([n_auxbas,n_auxbas],0.0);
        for l in 0..n_auxbas_shell {
            let basis_start_l = self.mol.cint_aux_fdqc[l][0];
            let basis_len_l = self.mol.cint_aux_fdqc[l][1];
            let gl  = l + n_basis_shell;
            for k in 0..n_auxbas_shell {
                let basis_start_k = self.mol.cint_aux_fdqc[k][0];
                let basis_len_k = self.mol.cint_aux_fdqc[k][1];
                let gk  = k + n_basis_shell;
                let tmp_size = [3*basis_len_k,basis_len_l];
                let mat_local = MatrixFull::from_vec(tmp_size,
                    cint_data.cint_ip_2c2e(gk as i32, gl as i32)).unwrap();
                let buf_len = mat_local.data.len() / 3;
                
                let mut tmp_slices_x = mat_partial_x.iter_submatrix_mut(
                    basis_start_k..basis_start_k+basis_len_k,
                    basis_start_l..basis_start_l+basis_len_l);
                mat_local.iter().enumerate().filter(|(i,local)| *i < buf_len).zip(tmp_slices_x).for_each(|((i,local),x)| {*x = *local});
                
                let mut tmp_slices_y = mat_partial_y.iter_submatrix_mut(
                    basis_start_k..basis_start_k+basis_len_k,
                    basis_start_l..basis_start_l+basis_len_l);
                mat_local.iter().enumerate().filter(|(i,local)| *i>=buf_len && *i < 2*buf_len).zip(tmp_slices_y).for_each(|((i,local),y)| {*y = *local});

                let mut tmp_slices_z = mat_partial_z.iter_submatrix_mut(
                    basis_start_k..basis_start_k+basis_len_k,
                    basis_start_l..basis_start_l+basis_len_l);
                mat_local.iter().enumerate().filter(|(i,local)| *i>= 2*buf_len && *i< 3*buf_len).zip(tmp_slices_z).for_each(|((i,local),z)| {*z = *local});
            
            }
        }
        mat_full.push(mat_partial_x.clone());
        mat_full.push(mat_partial_y.clone());
        mat_full.push(mat_partial_z.clone());

        cint_data.final_c2r();
        //aux_v.formated_output_e(4, "full");
        mat_full
    }

    fn ip_3c2e_intor(&self, cur_op: &String) -> Vec<RIFull<f64>> {
        let mut cint_data = self.mol.initialize_cint(true);

        if cur_op == &String::from("ip1") {
            cint_data.cint3c2e_ip1_optimizer_rust();
        } else if cur_op == &String::from("ip2") {
            cint_data.cint3c2e_ip2_optimizer_rust();

        } else {
            panic!("No such operation: {}", cur_op)
        }
        let n_basis = self.mol.num_basis;
        let n_auxbas = self.mol.num_auxbas;
        let mut ipri_x = RIFull::new([n_basis,n_basis,n_auxbas],0.0);
        let mut ipri_y = RIFull::new([n_basis,n_basis,n_auxbas],0.0);
        let mut ipri_z = RIFull::new([n_basis,n_basis,n_auxbas],0.0);
        let n_basis_shell = self.mol.cint_bas.len();
        let n_auxbas_shell = self.mol.cint_aux_bas.len();

        let mut mat_partial_x = RIFull::new([n_basis,n_basis,n_auxbas], 0.0_f64);
        let mut mat_partial_y = RIFull::new([n_basis,n_basis,n_auxbas], 0.0_f64);
        let mut mat_partial_z = RIFull::new([n_basis,n_basis,n_auxbas], 0.0_f64);

        for k in 0..n_auxbas_shell {
            let basis_start_k = self.mol.cint_aux_fdqc[k][0];
            let basis_len_k = self.mol.cint_aux_fdqc[k][1];
            let gk  = k + n_basis_shell;
            for j in 0..n_basis_shell {
                let basis_start_j = self.mol.cint_fdqc[j][0];
                let basis_len_j = self.mol.cint_fdqc[j][1];
                // can be optimized with "for i in 0..(j+1)"
                // currently we still use RIFull, to be optimized later
                for i in 0..n_basis_shell {
                    let basis_start_i = self.mol.cint_fdqc[i][0];
                    let basis_len_i = self.mol.cint_fdqc[i][1];
                    let buf = cint_data.cint_ip_3c2e(i as i32, j as i32, gk as i32, &cur_op);
                    let buf_len = buf.len()/3;
                    let mut buf_chunk = buf.chunks_exact(buf_len);
                    let buf_x = buf_chunk.next().unwrap().to_vec();
                    let buf_y = buf_chunk.next().unwrap().to_vec();
                    let buf_z = buf_chunk.next().unwrap().to_vec();

                    let ri_x = RIFull::from_vec([basis_len_i, basis_len_j,basis_len_k], buf_x).unwrap();
                    let ri_y = RIFull::from_vec([basis_len_i, basis_len_j,basis_len_k], buf_y).unwrap();
                    let ri_z = RIFull::from_vec([basis_len_i, basis_len_j,basis_len_k], buf_z).unwrap();

                    ipri_x.copy_from_ri(
                        basis_start_i..basis_start_i+basis_len_i,
                        basis_start_j..basis_start_j+basis_len_j,
                        basis_start_k..basis_start_k+basis_len_k,
                        &ri_x, 
                        0..basis_len_i, 
                        0..basis_len_j, 
                        0..basis_len_k);
                    ipri_y.copy_from_ri(
                        basis_start_i..basis_start_i+basis_len_i,
                        basis_start_j..basis_start_j+basis_len_j,
                        basis_start_k..basis_start_k+basis_len_k,
                        &ri_y, 
                        0..basis_len_i, 
                        0..basis_len_j, 
                        0..basis_len_k);
                    ipri_z.copy_from_ri(
                        basis_start_i..basis_start_i+basis_len_i,
                        basis_start_j..basis_start_j+basis_len_j,
                        basis_start_k..basis_start_k+basis_len_k,
                        &ri_z, 
                        0..basis_len_i, 
                        0..basis_len_j, 
                        0..basis_len_k);
                    
                    //let mut tmp_slices = ipri.get_slices_mut(
                    //    basis_start_i..basis_start_i+basis_len_i,
                    //    basis_start_j..basis_start_j+basis_len_j,
                    //    basis_start_k..basis_start_k+basis_len_k);
                    //tmp_slices.zip(buf.iter()).for_each(|value| {*value.0 = *value.1});
                }
            }
        }
        cint_data.final_c2r();
        let mut mat = vec![];
        mat.push(ipri_x);
        mat.push(ipri_y);
        mat.push(ipri_z);

        mat
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

    pub fn calc_j(&self, scf_data: &SCF) -> (Vec<MatrixFull<f64>>,Option<Vec<Vec<f64>>>) {
        let ao_loc = self.mol.ao_loc();
        let nbas = self.mol.fdqc_bas.len();
        let nauxbas = self.mol.fdqc_aux_bas.len();
        let nao = self.mol.num_basis;
        let naux = self.mol.num_auxbas;
        let auxmol = self.mol.make_auxmol();
        let auxslices = auxmol.aoslice_by_atom();
        let aux_loc = auxmol.ao_loc();
        //println!("auxslices = {:?}", auxslices);
        //println!("aux_loc = {:?}", aux_loc);

        let nset = 1_usize;  // temporarily set to 1
        // let nset = dm.len();    //number of density matrix?
        // dm[0] is density matrix

        // 
        // python: 
        // get a lower matrix
        // REST:
        // trying to use matrix upper
        let mut dm_upper = scf_data.density_matrix[0].to_matrixupper();
        let diag_idx = diag_idx_generator(scf_data.density_matrix[0].size[0]);
        dm_upper.data.iter_mut().enumerate().filter(|(i, data)| !diag_idx.contains(i))
                    .for_each(|(i, data)| *data *= 2.0);
        //println!("dm_upper = {:?}", dm_upper);

        // (i,j|P)
        // already accomplished in previous functions
        let mut rhoj = MatrixFull::new([naux,1], 0.0_f64);
        let dm_mat = MatrixFull::from_vec([dm_upper.data.len(),1], dm_upper.data).unwrap(); // flatten matrixupper for dgemm
        // Here pyscf uses memory control(ao_ranges), but we temporarily not use it
        let int3c = self.mol.int_ijk_rifull();
        let mut ijp_symm = int3c.rifull_to_matfull_symm(); // reduce RIFull according to its symmetry
        // [nao*(nao+1)/2, naux]**T * [nao*(nao+1)/2, 1] -> [naux,1]
        let mut rho = _dgemm_nn(&ijp_symm.transpose().to_matrixfullslice(),&dm_mat.to_matrixfullslice()); //[naux,1]
        // In this step rho does not need copying
        //rhoj.iter_mut().zip(rho.data.clone()).for_each(|(rj, r)| *rj = r);
        //rho.formated_output_e(150, "full"); // test passed
        // (P|Q)

        let mut int2c = self.mol.int_ij_aux_columb();
        // [naux, naux]
        //rhoj = scipy.linalg.solve(int2c, rhoj.T, sym_pos=True).T 
        //assuming C: int2c, rhoj.T: R^T, 
        // C*X = R^T
        //then if CX = R^T, the result is X
        // We just use C^(-1)*R^T 
        // [naux, naux]*[naux,1] -> [naux,1]
        let int2c_inv = int2c.lapack_inverse().unwrap(); //C^(-1)
        //let rho = rho.transpose(); //transpose not needed
        let rho2 = _dgemm_nn(&int2c_inv.to_matrixfullslice(), &rho.to_matrixfullslice()); //[naux, 1]
        rhoj.data.iter_mut().zip(rho2.data).for_each(|(rj, r)| *rj = r);
        rhoj.formated_output_e(150, "full"); 


        // auxbasis_response
        // (d/dX i,j|P)
        let origin_mat = MatrixFull::new([nao,nao], 0.0_f64);
        let mut vj = vec![origin_mat;3];
        let int3c_ip1 = self.ip_3c2e_intor(&String::from("ip1")); // [3, nao, nao, naux], xijp
        // vj += numpy.einsum('xijp,np->nxij', int3c, rhoj[:,p0:p1])
        // REST just einsum by x, y, z respectively
        let mut vj: Vec<MatrixFull<f64>> = int3c_ip1.iter()
            .map(|ri| { let mat = ri.rifull_to_matfull_ij_k();
                                      // [nao*nao,naux]*[naux,1] -> [nao*nao,1]
                                      let mut c = _dgemm_nn(&mat.to_matrixfullslice(), &rhoj.to_matrixfullslice());
                                      c.reshape([nao,nao]);
                                      c
                                    }).collect();
        
        if self.mol.ctrl.auxbasis_response == true {
            // here a variable should be added in ctrl.in: auxbasis_response: True/False
            // if self.mol.ctrl.auxb_resp == True 
            // (i,j|d/dX P)
            //let mut vjaux = MatrixFull::new([3,naux], 0.0_f64);
            let mut int3c_ip2 = self.ip_3c2e_intor(&String::from("ip2")); // [3, nao, nao, naux], xijp
            let mut int3c_ip2_matfull: Vec<MatrixFull<f64>> = int3c_ip2.iter().map(|x| x.rifull_to_matfull_symm()).collect(); // [3, nao*(nao+1)/2, naux], xwp
            // 'xwp,mw,np->xp'  [3,36,131] [1,36] [1,131] -> [3,1,131] -> [3,131]
            // actually 3*[36,131] [36,1] [131,1]
            // [3,131,1] 
            let mut vjaux: Vec<MatrixFull<f64>> = int3c_ip2_matfull.iter()
                .map(|x| {
                                            let temp = _dgemm_nn(&dm_mat.transpose().to_matrixfullslice(), &x.to_matrixfullslice()).transpose(); //[1,131]*3
                                            temp+rhoj.clone()
                                            }).collect();
            // Here we use Matrixfull for substraction later
            // let vjaux: Vec<Vec<f64>> = c1.iter().map(|x| x.data).collect(); 
                                        
            // (d/dX P|Q)
            // [3,131,131]
            let int2c_e1 = self.ip_2c2e_intor(); //[3,naux,naux]
            let mut vjaux2: Vec<MatrixFull<f64>> = int2c_e1.iter()
                .map(|x| { let temp = _dgemm_nn(&x.to_matrixfullslice(), &rhoj.to_matrixfullslice());
                                            temp+rhoj.clone()
                                            }).collect();
            //vjaux -= numpy.einsum('xpq,mp,nq->xp', int2c_e1, rhoj, rhoj)
            //actually 3*[131,131] [131,1] [131,1] -> [3,131,1]
            vjaux.iter_mut().zip(vjaux2).for_each(|(x1,x2)| *x1-=x2);

            let mut vjaux_final_data = vec![];
            for slice in &auxslices {
                let p0 = slice[2];
                let p1 = slice[3];
                let mut x_sum: f64 = vjaux[0][(p0..p1,0)].iter().sum();
                let mut y_sum: f64 = vjaux[1][(p0..p1,0)].iter().sum();
                let mut z_sum: f64 = vjaux[2][(p0..p1,0)].iter().sum();
                x_sum *= -1.0_f64;
                y_sum *= -1.0_f64;
                z_sum *= -1.0_f64;
                //vjaux_final_data.push(x_sum);
                //vjaux_final_data.push(y_sum);
                //vjaux_final_data.push(z_sum);
                let data = vec![x_sum, y_sum, z_sum];
                vjaux_final_data.push(data)
            }

            // vj needs to be multiplied by -1, but currently we just test its number
            return (vj, Some(vjaux_final_data))
        } else {
            return (vj, None)
        }
        


    }

    pub fn calc_k(&self, scf_data: &SCF) -> (Vec<MatrixFull<f64>>,Option<Vec<Vec<f64>>>) {
        // in pyscf, it uses HDF5 file to temporarily save data, but REST currently just use memory to save data
        let nao = self.mol.num_basis;
        let naux = self.mol.num_auxbas;
        let auxmol = self.mol.make_auxmol();
        let auxslices = auxmol.aoslice_by_atom();
        //vk [3, nao, nao]
        let mo_coeff = &scf_data.eigenvectors[0];
        let mo_occ = &scf_data.occupation[0];
        let nmo = mo_occ.len();
        // Here pyscf uses assert to judge if the mol is ROHF/RHF, 
        // if (mo_occ > 0) + (mo_occ == 2) != (mo_occ), return RUNTIME_ERROR

        match scf_data.scftype {
            SCFType::ROHF => {
                let mo_occa = mo_occ.iter().filter(|v| **v>0.0).count();
                let mo_occb = mo_occ.iter().filter(|v| **v==2.0).count();
                assert_eq!((mo_occa+mo_occb), mo_occ.iter().sum::<f64>() as usize)
            }
            _ => () 
        }  
        //mo_coeff = numpy.asarray(mo_coeff).reshape(-1,nao,nmo)
        //mo_occ   = numpy.asarray(mo_occ).reshape(-1,nmo)
        // here pyscf convert mo_coeff shape to [nset,nao,nmo], mo_occ [nset,1,nmo]
        //
        let einsum_b: Vec<f64> = mo_occ.iter().filter(|v| **v>0.0).map(|v| v.sqrt()).collect();
        let orbo_idx: Vec<usize> = mo_occ.iter().enumerate().filter(|(i,v)| **v>0.0).map(|(i,v)| i).collect();
        let occ = orbo_idx.len();        
        let einsum_a_vec: Vec<f64> = orbo_idx.iter().map(|i| mo_coeff[(..,*i)].to_vec()).flatten().collect();
        let einsum_a = MatrixFull::from_vec([nao,occ], einsum_a_vec).unwrap();
        let orbo_data: Vec<f64> = einsum_a.iter_columns_full().zip(einsum_b).map(|(a,b)| a.to_vec().iter().map(|i| i*b).collect::<Vec<f64>>()).flatten().collect();
        let orbo = MatrixFull::from_vec([nao,occ], orbo_data).unwrap();
        //(P|Q)
        
        let mut int2c = self.mol.int_ij_aux_columb();
        // [naux, naux]
        int2c.to_matrixfullslicemut().cholesky_decompose_inverse('L').unwrap();
        let int3c = self.mol.int_ijk_rifull();
        // orbo [nao, occ] int3c [nao, nao, naux] -> [naux, nao, occ]
        let int3c_iter = int3c.data.chunks_exact(nao*nao);
        let v_data: Vec<f64> = int3c_iter.map(|c| { let tmp = MatrixFull::from_vec([nao,nao], c.to_vec()).unwrap();
                                                            let tmp2 = _dgemm_nn(&tmp.to_matrixfullslice(), &orbo.to_matrixfullslice()); //[lk][ko] -> [lo]
                                                            tmp2.data }).flatten().collect(); //[nao,occ,naux]
        let v = MatrixFull::from_vec([nao*occ,naux], v_data).unwrap(); //[nao*occ, naux]
        let v2 = _dgemm_nn(&v.to_matrixfullslice(),&int2c.to_matrixfullslice());//[nao*occ, naux]*[naux,naux]->[nao*occ,naux]
        let rhok = RIFull::from_vec([nao,occ,naux], v2.data).unwrap(); //[nao,occ,naux]
    
        // (d/dX i,j|P)
        let int3c_ip1 = self.ip_3c2e_intor(&String::from("ip1")); // [3, nao, nao, naux], xijp
        //pyscf[3, nao, nao, naux]*[nao,occ]-> [3,nao,naux,occ]
        //actually 3*[nao, nao]*[nao,occ]*naux [3,nao,occ,naux]
        let tmp: Vec<RIFull<f64>> = int3c_ip1.iter().map(|ri| {
                                let tmp_data: Vec<f64> = ri.data.chunks_exact(nao*nao).map(|chunk| {
                                                                            let tmp = MatrixFull::from_vec([nao,nao], chunk.to_vec()).unwrap();
                                                                            let tmp2 = _dgemm_nn(&tmp.to_matrixfullslice(), &orbo.to_matrixfullslice()); //[nao,nao][nao,occ]->[nao,occ] 
                                                                            tmp2.data
                                                                        }).flatten().collect();
                                RIFull::from_vec([nao,occ,naux], tmp_data).unwrap() }).collect(); //[3,nao,occ,naux]
        // pyscf xipo,pok->xik [3,nao,naux,occ]*[naux,occ,nao] -> [3,nao,nao]
        // actually 3*[nao,occ,naux]*[nao,occ,naux] -> [3,nao,nao]
        let rhok_matfull = rhok.transpose_jki().rifull_to_matfull_ij_k(); //[occ*naux,nao]
        let vk: Vec<MatrixFull<f64>> = tmp.iter().map(|tmp| {let c1 = tmp.rifull_to_matfull_i_jk();  //[nao,occ*naux]
                                                                          _dgemm_nn(&c1.to_matrixfullslice(), &rhok_matfull.to_matrixfullslice())}).collect();
        let rhok = rhok.transpose_kji(); // [naux,occ,nao]
        let rhok_matfull2 = rhok.rifull_to_matfull_ij_k(); //[naux*occ,nao]
        let rhok_oo = _dgemm_nn(&rhok_matfull2.to_matrixfullslice(), &orbo.to_matrixfullslice()); //[naux*occ,nao][nao,occ] -> [naux*occ,occ]
        let rhok_oo = matfull_to_rifull(&rhok_oo, &naux, &occ); //[naux,occ,occ]

        /* 
        tmp = rhok_oo[i][p0:p1]
                tmp = lib.einsum('por,ir->pio', tmp, orbo[i])
                tmp = lib.einsum('pio,jo->pij', tmp, orbo[i])
                vkaux[:,p0:p1] += lib.einsum('xpij,pij->xp', int3c, tmp)

         */

        if self.mol.ctrl.auxbasis_response == true {
            //if mf_grad.auxbasis_response:
            // (i,j|d/dX P)

            //pyscf por,ir->pio [naux,occ,occ]*[nao,occ]-> [naux,nao,occ]
            //actually [naux*occ,occ]*[nao,occ]**T-> [naux,occ,nao]
            let mut int3c_ip2 = self.ip_3c2e_intor(&String::from("ip2")); // [3, nao, nao, naux], xijp
            
            let tmp = _dgemm_nn(&rhok_oo.rifull_to_matfull_ij_k().to_matrixfullslice(), &orbo.transpose().to_matrixfullslice()); //[naux*occ,occ]*[nao,occ]**T-> [naux,occ,nao]
            let tmp = matfull_to_rifull(&tmp, &naux, &occ); //[naux,occ,nao] poi
            let tmp = tmp.transpose_ikj(); //[naux,nao,occ] pio
            let tmp = _dgemm_nn(&tmp.rifull_to_matfull_ij_k().to_matrixfullslice(), &orbo.transpose().to_matrixfullslice()); //[naux*nao,occ]*[nao,occ]**T-> [naux*nao,nao]
            let tmp = matfull_to_rifull(&tmp, &naux, &nao); //[naux,nao,nao] pij
            //pyscf 'xpij,pij->xp' [3,naux, nao, nao] [naux,nao,nao] -> [3,naux]
            //actually 3*[nao, nao, naux] [naux,nao,nao] -> 
            let mut vkaux: Vec<Vec<f64>> = int3c_ip2.iter().map(|ri| { let int3c_mat = ri.rifull_to_matfull_ij_k().transpose(); //[naux,nao*nao]
                                                      let tmp_ri = tmp.transpose_jki(); // [nao,nao,naux]
                                                      let tmp_mat = tmp_ri.rifull_to_matfull_ij_k(); //  
                                                      let result_mat = _dgemm_nn(&int3c_mat.to_matrixfullslice(), &tmp_mat.to_matrixfullslice()); // [naux,nao*nao]*[nao*nao,naux]-> [naux,naux]
                                                      result_mat.get_diagonal_terms().unwrap().into_iter().map(|v| *v).collect_vec()}).collect(); 

            // (d/dX P|Q)
            /* 
             int2c_e1 = auxmol.intor('int2c2e_ip1')
            vjaux -= numpy.einsum('xpq,mp,nq->xp', int2c_e1, rhoj, rhoj)
            for i in range(nset):
                tmp = lib.einsum('pij,qij->pq', rhok_oo[i], rhok_oo[i])
                vkaux -= numpy.einsum('xpq,pq->xp', int2c_e1, tmp)

            vjaux = [-vjaux[:,p0:p1].sum(axis=1) for p0, p1 in auxslices[:,2:]]
            vkaux = [-vkaux[:,p0:p1].sum(axis=1) for p0, p1 in auxslices[:,2:]]
            vj = lib.tag_array(-vj.reshape(out_shape), aux=numpy.array(vjaux))
            vk = lib.tag_array(-vk.reshape(out_shape), aux=numpy.array(vkaux))
             */
            let int2c_e1 = self.ip_2c2e_intor(); //[3,naux,naux]
            // rhok_oo [naux,occ,occ] pij 
            let rhok_oo_mat = rhok_oo.rifull_to_matfull_i_jk(); //[naux,occ*occ]
            let tmp2 = _dgemm_nn(&rhok_oo_mat.to_matrixfullslice(), &rhok_oo_mat.transpose().to_matrixfullslice()); //[naux,occ*occ]*[naux,occ*occ]**T-> [naux,naux]
            let vkaux2: Vec<Vec<f64>> = int2c_e1.iter().map(|mat| { let result_mat = _dgemm_nn(&mat.to_matrixfullslice(), &tmp2.transpose().to_matrixfullslice()); //[naux,naux]*[naux,naux]**T -> [naux,naux]
                                                         result_mat.get_diagonal_terms().unwrap().into_iter().map(|v| *v).collect_vec()}).collect(); //[naux,naux]->[naux]
            vkaux.iter_mut().zip(vkaux2).for_each(|(vk1, vk2)| vk1.iter_mut().zip(vk2).for_each(|(v1,v2)| *v1-=v2));
            
            let vkaux: Vec<MatrixFull<f64>> = vkaux.iter().map(|v| MatrixFull::from_vec([naux,1], v.clone()).unwrap()).collect();

            let mut vkaux_final_data = vec![];
            for slice in &auxslices {
                let p0 = slice[2];
                let p1 = slice[3];
                let mut x_sum: f64 = vkaux[0][(p0..p1,0)].iter().sum();
                let mut y_sum: f64 = vkaux[1][(p0..p1,0)].iter().sum();
                let mut z_sum: f64 = vkaux[2][(p0..p1,0)].iter().sum();
                x_sum *= -1.0_f64;
                y_sum *= -1.0_f64;
                z_sum *= -1.0_f64;
                //vjaux_final_data.push(x_sum);
                //vjaux_final_data.push(y_sum);
                //vjaux_final_data.push(z_sum);
                let data = vec![x_sum, y_sum, z_sum];
                vkaux_final_data.push(data)
            }

            // vk needs to be multiplied by -1, but currently we just test its number
            return (vk, Some(vkaux_final_data))
        }
        else {
            return (vk, None)
        }
        

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


#[test]
fn test_diag_idx(){
    diag_idx_generator(8);
}


#[test]
fn test_r2m(){
    let data = vec![1.1067269298728097, 0.16577931765462367, 1.1316486011894959e-16, 1.0717756889866899e-16, 0.0, 0.041055077382229056, 0.04105507738606492, 0.04105507737900729, 0.16577931765462367, 0.2433476817813402, 1.3881199170863872e-16, -1.4314535040399662e-16, 0.0, 0.09257476815598732, 0.09257476816187377, 0.09257476815104326, 1.1316486011894959e-16, 1.3881199170863872e-16, 0.24249931779865294, -4.8818713224198074e-32, 0.0, 0.07116589349082135, -0.023721706124913082, -0.02372202908177368, 1.07177568898669e-16, -1.4314535040399664e-16, -4.8818713224198074e-32, 0.24249931779865294, 0.0, -3.538248156308559e-17, -0.05615328167972683, 0.05988069548378212, 0.0, 0.0, 0.0, 0.0, 0.24249931779865294, 0.0, -0.03672429729684275, -0.030268003577478712, 0.04105507738222905, 0.09257476815598732, 0.07116589349082134, -3.53824815630856e-17, 0.0, 0.12592041492280376, 0.0313501285643936, 0.03135002080065368, 0.04105507738606492, 0.09257476816187378, -0.023721706124913082, -0.056153281679726816, -0.036724297296842755, 0.0313501285643936, 0.1259204149270262, 0.031350020797584234, 0.0410550773790073, 0.09257476815104325, -0.023722029081773684, 0.05988069548378213, -0.030268003577478712, 0.03135002080065368, 0.03135002079758423, 0.1259204149192573, 
    2.0776443093039516, 0.3709473153193498, 2.586690979916184e-16, 2.2989501078606204e-16, 0.0, 0.09108470952936638, 0.09108470953784345, 0.09108470952224643, 0.3709473153193499, 0.646951990511637, 3.740211627145463e-16, -3.837804405456467e-16, 0.0, 0.24754697838934697, 0.24754697840500506, 0.2475469783761957, 2.586690979916184e-16, 3.7402116271454625e-16, 0.6470314501076488, -1.3192297174181583e-31, 0.0, 0.19184485856040961, -0.06394758968257624, -0.06394846029083087, 2.2989501078606204e-16, -3.837804405456467e-16, -1.3192297174181583e-31, 0.6470314501076488, 0.0, -9.577251068532419e-17, -0.1513747365925795, 0.16142288099099084, 0.0, 0.0, 0.0, 0.0, 0.6470314501076488, 0.0, -0.09899921542544819, -0.08159471595725742, 0.09108470952936637, 0.24754697838934697, 0.19184485856040956, -9.577251068532417e-17, 0.0, 0.33966938518039674, 0.08408403616691149, 0.08408374501908782, 0.09108470953784346, 0.247546978405005, -0.06394758968257624, -0.15137473659257947, -0.09899921542544818, 0.08408403616691149, 0.3396693851916775, 0.0840837450107355, 0.09108470952224643, 0.2475469783761957, -0.06394846029083087, 0.1614228809909908, -0.08159471595725744, 0.08408374501908782, 0.0840837450107355, 0.33966938517092204,
    2.616499186134411, 0.5081744764677162, 3.487184123863622e-16, 3.0234589827400317e-16, 0.0, 0.12433484132370207, 0.12433484133524306, 0.12433484131400874, 0.5081744764677162, 1.015346397636462, 5.96907910716508e-16, -6.069115204078888e-16, 0.0, 0.3921379919352661, 0.3921379919598922, 0.39213799191458243, 3.487184123863622e-16, 5.96907910716508e-16, 1.0162910904078175, -2.119379098100008e-31, 0.0, 0.30675102851531527, -0.10224922916046483, -0.10225062122272477, 3.0234589827400317e-16, -6.069115204078888e-16, -2.119379098100008e-31, 1.0162910904078175, 0.0, -1.5411930118991651e-16, -0.24204118103261255, 0.2581076977588685, 0.0, 0.0, 0.0, 0.0, 1.0162910904078175, 0.0, -0.1582951525614886, -0.1304661653646977, 0.12433484132370207, 0.3921379919352661, 0.30675102851531527, -1.5411930118991651e-16, 0.0, 0.5448046641481702, 0.13385826357920538, 0.13385779585880697, 0.12433484133524306, 0.3921379919598922, -0.10224922916046483, -0.2420411810326125, -0.15829515256148854, 0.13385826357920538, 0.5448046641660165, 0.13385779584526994, 0.12433484131400874, 0.39213799191458243, -0.10225062122272476, 0.2581076977588685, -0.1304661653646977, 0.13385779585880694, 0.13385779584526994, 0.544804664133181];
    let ri = RIFull::from_vec([8,8,3], data).unwrap();
    let m = ri.rifull_to_matfull_symm();
    m.formated_output_e(36, "full")
}