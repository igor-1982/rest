use rayon::prelude::*;
use rstsr::prelude::*;
use std::collections::HashMap;
use std::sync::mpsc::channel;
use tensors::{matrix_blas_lapack::_power_rayon_for_symmetric_matrix, MatrixFull};

type Tsr<T> = Tensor<T, DeviceBLAS, IxD>;
type TsrView<'a, T> = TensorView<'a, T, DeviceBLAS, IxD>;
type TsrMut<'a, T> = TensorMut<'a, T, DeviceBLAS, IxD>;

use crate::constants::AUXBAS_THRESHOLD;
use crate::grad::traits::GradAPI;
use crate::scf_io;
use crate::scf_io::SCF;
use crate::Molecule;
use super::rhf::{RIHFGradientFlags};
use super::uhf::{RIUHFGradient};
use crate::dft::{Grids, DFA4REST};
use crate::dft::xc_deriv::XCType;
use crate::dft::num_int::{NumInt, XCData, BlockSettings};
use crate::dft::num_int::{eval_rho5_batch, eval_rho5_dm_only_batch, eval_ao_batch};
use crate::dft::libxc_itrf::{eval_xc_eff};
use super::rks::{gga_grad_sum, tau_grad_dot, gather_rows_full, gather_submatrix_full, scatter_to_full};
use crate::utilities::{self, balancing};

use tensors::matrix_blas_lapack::{omp_get_num_threads_wrapper, omp_set_num_threads_wrapper};


impl<'a> NumInt<'a> for RIUHFGradient<'a> {
    fn gen_xc_data(&'a self, scf_data: &'a SCF, spin: usize) -> XCData<'a> {
        let xc_method = &self.scf_data.mol.xc_data;
        let xc_type = if xc_method.use_kinetic_density() {
            XCType::MGGA
        } else if xc_method.use_density_gradient() {
            XCType::GGA
        } else {
            XCType::LDA
        };

        let xc_code = &xc_method.dfa_compnt_scf;
        let xc_params = &xc_method.dfa_paramr_scf;

        let density_matrix = &scf_data.density_matrix;
        let mo_coeffs = scf_data.eigenvectors.clone().to_vec();
        let mo_occ = scf_data.occupation.clone().to_vec();

        let device = DeviceOpenBLAS::default();

        XCData {
            xc_type,
            xc_code,
            xc_params,
            device: device,
            dm: density_matrix,
            mo_coeffs: Some(mo_coeffs),
            occ: Some(mo_occ),
        }
    }


    fn get_vxc(
        &'a self,
        xc_data: &'a XCData,
        grids: &mut Grids,
        mol: &Molecule,
        spin: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<MatrixFull<f64>>) {
        unimplemented!("Gradient calculation does not goes here.");
    }


    fn get_fxc(
        &'a self,
        xc_data0: &'a XCData,
        xc_data1: &'a XCData,
        grids: &mut Grids,
        mol: &Molecule,
        spin: usize,
    ) -> Vec<MatrixFull<f64>> {
        unreachable!("Gradient calculation does not support get_fxc");
    }
}


impl RIUHFGradient<'_> {
    pub fn calc_de_xc(&mut self) -> &mut Self {
        // get dx vxc [nao, nao, 3, 2]
        // println!("Calculating dx vxc");
        let scf_data = self.scf_data; 
        let xc_data = self.gen_xc_data(scf_data, 1);
        // Reuse the SCF grid (points/weights/partition), mirroring the RKS
        // path: identical deterministic content (rank-local under MPI),
        // consistent with any SCF grid pruning, and one grid generation saved.
        // Fall back to a fresh build only when the SCF data holds no grids.
        let mut owned_grids: Option<Grids> = None;
        let mut owned_mol: Option<Molecule> = None;
        let grids: &Grids = match scf_data.grids.as_ref() {
            Some(g) => g,
            None => {
                let mut mol = scf_data.mol.clone();
                let g = Grids::build(&mut mol);
                owned_mol = Some(mol);
                owned_grids = Some(g);
                owned_grids.as_ref().unwrap()
            },
        };
        let mol: &Molecule = owned_mol.as_ref().unwrap_or(&scf_data.mol);
        // Same Morton reorder as the RKS path: a private, spatially compact
        // copy of the grid points so per-block locality screening pays off.
        let perm = crate::dft::num_int::morton_order(&grids.coordinates);
        let grad_grids = Grids {
            coordinates: perm.iter().map(|&i| grids.coordinates[i]).collect(),
            weights: perm.iter().map(|&i| grids.weights[i]).collect(),
            ao: None,
            aop: None,
            parallel_balancing: balancing(grids.coordinates.len(), rayon::current_num_threads()),
            non0tab: None,
            ao_cutoff: grids.ao_cutoff,
            ao_compressed: None,
            aop_compressed: None,
            atm_idx: perm.iter().map(|&i| grids.atm_idx[i]).collect(),
            quadrature_weights: perm.iter().map(|&i| grids.quadrature_weights[i]).collect(),
        };
        let mem_probe_on =
            utilities::memory_batch::mem_probe_enabled(self.scf_data.mol.ctrl.print_level);
        utilities::memory_batch::mem_probe("calc_de_xc (uks): before get_vxc_rayon_new", mem_probe_on);
        // let dao_vxc = get_vxc(&self, &xc_data, grids, &scf_data.mol, self.flags.max_memory.unwrap_or(2000.0) as usize);
        // let dao_vxc = get_vxc_rayon(&self, &xc_data, grids, &scf_data.mol);
        // 每线程每块的 XC 工作集预算，由 [ctrl] max_memory 推得，未设置时保持历史上的 16 MiB
        let xc_block_mb = utilities::memory_batch::xc_grad_block_mb(self.flags.max_memory, 16.0);
        let dao_vxc = get_vxc_rayon_new(&self, &xc_data, &grad_grids, mol, xc_block_mb as usize);
        utilities::memory_batch::mem_probe("calc_de_xc (uks): after get_vxc_rayon_new", mem_probe_on);
        // println!("Finished calculating dx vxc");
        
        // contract dao_vxc and dm (tuv, uv -> tu) and sum over spin case
        let nao = mol.num_basis;
        let mut dao_xc:Tsr<f64> = rt::zeros(([nao, 3], &xc_data.device));        
        let dm = get_dm(scf_data, &xc_data.device);
        for ispin in 0..2usize {
            let dm_s = dm[ispin].view();
            dao_xc += (&dao_vxc[ispin] * &dm_s.i((.., .., None))).sum_axes(1) * 2.0;
        }
        // println!("Finished calculating dao_xc");
        
        // scatter result to natm  
        let natm = mol.geom.elem.len();
        let mut de_xc: Tsr<f64> = rt::zeros(([3, natm], &xc_data.device));
        let ao_slice = mol.aoslice_by_atom();
        for atm in 0..natm {
            let [_, _, p0, p1] = ao_slice[atm];
            *&mut de_xc.i_mut((.., atm)) += &dao_xc.i((p0..p1)).sum_axes(0);
        }

        // return to REST, note f-contiguous transpose
        let mut de_xc_raw = de_xc.into_raw_parts().0.into_cpu_vec().unwrap();
        // MPI: the numerical grids are distributed across ranks (`Grids::build`), so each rank
        // only integrates its local grid portion; sum the XC gradient contribution over ranks.
        #[cfg(feature = "mpi")]
        if let Some(mpi_op) = self.mpi_operator {
            let mut reduced = vec![0.0_f64; de_xc_raw.len()];
            crate::mpi_io::mpi_allreduce(
                &mpi_op.world,
                &de_xc_raw,
                &mut reduced,
                &mpi::collective::SystemOperation::sum(),
            );
            de_xc_raw.copy_from_slice(&reduced);
        }
        let de_xc = MatrixFull::from_vec([3, natm], de_xc_raw).unwrap();
        self.result.insert("de_xc".into(), de_xc);
        
        // println!("Finished calculating de_xc");

        self
    }


    pub fn calc_uks(&mut self) -> &MatrixFull<f64> {
        let mut time_records = crate::utilities::TimeRecords::new();
        time_records.new_item("uks grad", "uks grad");
        time_records.new_item("uks grad calc_de_nuc", "uks grad calc_de_nuc");
        time_records.new_item("uks grad calc_de_ovlp", "uks grad calc_de_ovlp");
        time_records.new_item("uks grad calc_de_hcore", "uks grad calc_de_hcore");
        time_records.new_item("uks grad calc_de_ext_field", "uks grad calc_de_ext_field");
        time_records.new_item("uks grad calc_de_jk", "uks grad calc_de_jk");
        time_records.new_item("uks grad calc_de_xc", "uks grad calc_de_xc");
        time_records.new_item("uks grad calc_de_solvent", "uks grad calc_de_solvent");
        time_records.new_item("uks grad calc_de_qmmm", "uks grad calc_de_qmmm");

        time_records.count_start("uks grad");

        time_records.count_start("uks grad calc_de_nuc");
        self.calc_de_nuc();
        time_records.count("uks grad calc_de_nuc");

        time_records.count_start("uks grad calc_de_ovlp");
        self.calc_de_ovlp();
        time_records.count("uks grad calc_de_ovlp");

        time_records.count_start("uks grad calc_de_hcore");
        self.calc_de_hcore();
        time_records.count("uks grad calc_de_hcore");

        if self.flags.ext_field_dipole.is_some() {
            time_records.count_start("uks grad calc_de_ext_field");
            self.calc_de_ext_field();
            time_records.count("uks grad calc_de_ext_field");
        }

        time_records.count_start("uks grad calc_de_qmmm");
        self.calc_de_qmmm();
        time_records.count("uks grad calc_de_qmmm");

        if self.flags.factor_j.is_some() || self.flags.factor_k.is_some() || self.flags.omega.is_some() {
            time_records.count_start("uks grad calc_de_jk");
            self.calc_de_jk();
            time_records.count("uks grad calc_de_jk");
        }

        time_records.count_start("uks grad calc_de_xc");
        self.calc_de_xc();
        time_records.count("uks grad calc_de_xc");

        time_records.count_start("uks grad calc_de_solvent");
        self.calc_de_solvent();
        time_records.count("uks grad calc_de_solvent");

        time_records.count("uks grad");

        let mut de = self.result.get("de_nuc").unwrap().clone();
        de += self.result.get("de_ovlp").unwrap().clone();
        de += self.result.get("de_hcore").unwrap().clone();
        self.result.get("de_j").map(|x| de += x.clone());
        self.result.get("de_k").map(|x| de += x.clone());
        self.result.get("de_sr").map(|x| de += x.clone());
        self.result.get("de_jaux").map(|x| de += x.clone());
        self.result.get("de_kaux").map(|x| de += x.clone());
        self.result.get("de_sraux").map(|x| de += x.clone());
        self.result.get("de_xc").map(|x| de += x.clone());
        self.result.get("de_qmmm").map(|x| de += x.clone());
        self.result.get("de_solvent").map(|x| de += x.clone());
        self.result.get("de_ext_field").map(|x| de += x.clone());
        self.result.insert("de".into(), de);

        if self.flags.print_level >= 2 {
            time_records.report_all();
        }

        return self.result.get("de").unwrap();
    }


}


fn get_dm(scf_data: &SCF, device: &DeviceBLAS) -> Vec<Tsr<f64>> {
    // This can be reconsidered if 3-D (spin, ao, ao) is better.
    // Currently, vector of 2-D (ao, ao) is used.
    // from uhf.rs 
    let mut result = vec![];
    for spin in [0, 1] {
        let dm = &scf_data.density_matrix[spin];
        result.push(rt::asarray((&dm.data, dm.size, device)).to_owned());
    }
    return result;
}


fn get_vxc(gradient_method: &RIUHFGradient, xc_data: &XCData, grids: &mut Grids, mol: &Molecule, max_memory: usize) -> Vec<Tsr<f64>> {
    let num_grids = grids.weights.len();
    let num_basis = mol.num_basis;
    // determine block settings 
    let ao_deriv = match xc_data.xc_type {
        XCType::HF => 1usize,
        XCType::LDA => 1usize,
        XCType::GGA => 2usize,
        XCType::MGGA => 2usize,
    };
    
    let ao_comp = (ao_deriv + 1) * (ao_deriv + 2) * (ao_deriv + 3) / 6; // number of components for ao derivatives 
    let blksize = max_memory * 1_000_000 / 8 / ((ao_comp + 1) * num_basis);
    let blksize = blksize.min(num_grids).max(4); 
    let block_settings = BlockSettings {
        ao_deriv,
        max_memory,
        blksize,
    };
   
    let device = &xc_data.device;

    // UKS case 
    let spin = 1usize;
    let nspin = spin + 1;
    let mut vmat_a = rt::zeros(([num_basis, num_basis, 3], device));
    let mut vmat_b = rt::zeros(([num_basis, num_basis, 3], device));
    let mut vmat = vec![vmat_a, vmat_b];

    let grids_iterator = gradient_method.block_loop(mol, grids, block_settings);
    // calculation starts here 
    let deriv = 1 as usize;          
    grids_iterator.into_iter().for_each(
        |block| 
        {
            let num_grids = block.weights.len();
            let loc_rho = if let Some(mo_coeffs) = &xc_data.mo_coeffs {
                eval_rho5_batch(&block.ao, xc_data.xc_type, mo_coeffs, xc_data.occ.as_ref().unwrap(), nspin, num_grids)
            } else {
                eval_rho5_dm_only_batch(&block.ao, xc_data.xc_type, xc_data.dm, nspin, num_grids)
            };
            // println!("Shape of loc_rho: {:?}", loc_rho.shape());

            let loc_xc_tensor = eval_xc_eff(xc_data.xc_code, xc_data.xc_params, xc_data.xc_type, spin, &loc_rho.raw(), num_grids, deriv);
            let loc_vxc = if let Some(vxc) = loc_xc_tensor[1].as_ref() {
                rt::asarray(vxc)
            } else {
                panic!("Exchange-correlation potential is not provided for the given xc_type: {:?}", xc_data.xc_type);
            };
            // println!("Shape of loc_vxc: {:?}", loc_vxc.shape()); // assume to be [num_grids, num_components, nspin]

            let loc_weights = rt::asarray((block.weights, device));
            let ao_shape = vec![num_basis, num_grids, ao_comp];
            let loc_ao = rt::asarray((&block.ao.data, ao_shape, device));

            let mut loc_wv = loc_vxc * &loc_weights.i((.., None, None));

            match xc_data.xc_type {
                XCType::LDA => {
                    for ispin in 0..nspin {
                        let aow_s = &loc_ao.i((.., .., 0)) * &loc_wv.i((None, .., 0, ispin)); 
                        let aow_s = aow_s.t();
                        for ic in 0..3 {
                            vmat[ispin].i_mut((.., .., ic)).matmul_from(
                                &loc_ao.i((.., .., ic+1)), &aow_s, 1.0, 1.0
                            );
                        }
                    }
                }, 
                XCType::GGA => {
                    for ispin in 0..nspin {
                        // let mut vmat_s = vmat.i((.., .., .., ispin)).to_owned();
                        loc_wv.i_mut((.., 0, ispin)).mul_assign(0.5);
                        gga_grad_sum(&mut vmat[ispin], loc_ao.view(), loc_wv.i((.., .., ispin)));
                    }
                },
                XCType::MGGA => {
                    for ispin in 0..nspin {
                        // let mut vmat_s = vmat.i((.., .., .., ispin)).to_owned();
                        loc_wv.i_mut((.., 0, ispin)).mul_assign(0.5);
                        loc_wv.i_mut((.., 4, ispin)).mul_assign(0.5);
                        gga_grad_sum(&mut vmat[ispin], loc_ao.view(), loc_wv.i((.., .., ispin)));
                        tau_grad_dot(&mut vmat[ispin], loc_ao.view(), loc_wv.i((.., .., ispin)));
                    }
                },
                XCType::HF => {
                    unreachable!("HF gradient calculation does not support here in get_vxc");
                },
            }
        }
    );
    // nabla R = - nabla r 
    for ispin in 0..nspin {
        vmat[ispin].mul_assign(-1.0);
    }

    vmat
}


fn get_vxc_rayon(gradient_method: &RIUHFGradient, xc_data: &XCData, grids: &mut Grids, mol: &Molecule) -> Vec<Tsr<f64>> {
    let default_omp_num_threads = omp_get_num_threads_wrapper();

    let num_basis = mol.num_basis;
    // determine block settings 
    let ao_deriv = match xc_data.xc_type {
        XCType::HF => 1usize,
        XCType::LDA => 1usize,
        XCType::GGA => 2usize,
        XCType::MGGA => 2usize,
    };
    let ao_comp = (ao_deriv + 1) * (ao_deriv + 2) * (ao_deriv + 3) / 6;
    let device = &xc_data.device;
    let deriv = 1 as usize;
    // UKS case 
    let spin = 1usize;
    let nspin = spin + 1;
    let mut vmat_a = rt::zeros(([num_basis, num_basis, 3], device));
    let mut vmat_b = rt::zeros(([num_basis, num_basis, 3], device));
    let mut vmat = vec![vmat_a, vmat_b];

    let (sender, receiver) = channel();
    grids.parallel_balancing.par_iter()
    .for_each_with(
        sender,|s, range_grids| 
        {
            omp_set_num_threads_wrapper(1);
            // eval batch ao 
            let num_grids = range_grids.len();
            let loc_coordinates = &grids.coordinates[range_grids.clone()];
            let loc_ao = eval_ao_batch(mol, loc_coordinates, ao_deriv, num_grids);
            // eval batch rho 
            let loc_rho = if let Some(mo_coeffs) = &xc_data.mo_coeffs {
                eval_rho5_batch(&loc_ao, xc_data.xc_type, mo_coeffs, xc_data.occ.as_ref().unwrap(), nspin, num_grids)
            } else {
                eval_rho5_dm_only_batch(&loc_ao, xc_data.xc_type, xc_data.dm, nspin, num_grids)
            };
            // eval vxc on grids 
            let loc_xc_tensor = eval_xc_eff(xc_data.xc_code, xc_data.xc_params, xc_data.xc_type, spin, &loc_rho.raw(), num_grids, deriv);
            // let loc_rho = rt::asarray((rho_array, shape, device));
            let loc_vxc = if let Some(vxc) = loc_xc_tensor[1].as_ref() {
                rt::asarray(vxc)
            } else {
                panic!("Exchange-correlation potential is not provided for the given xc_type: {:?}", xc_data.xc_type);
            };
            // eval batch vxc_ao 
            let mut loc_vmat_a = rt::zeros(([num_basis, num_basis, 3], device));
            let mut loc_vmat_b = rt::zeros(([num_basis, num_basis, 3], device));
            let mut loc_vmat = vec![loc_vmat_a, loc_vmat_b];
            let loc_weights = rt::asarray((&grids.weights[range_grids.clone()], device));
            let ao_shape = vec![num_basis, num_grids, ao_comp];
            let loc_ao = rt::asarray((&loc_ao.data, ao_shape, device));
            
            let mut loc_wv = loc_vxc * &loc_weights.i((.., None, None));

            match xc_data.xc_type {
                XCType::LDA => {
                    for ispin in 0..nspin {
                        let aow_s = &loc_ao.i((.., .., 0)) * &loc_wv.i((None, .., 0, ispin)); 
                        let aow_s = aow_s.t();
                        for ic in 0..3 {
                            loc_vmat[ispin].i_mut((.., .., ic, ispin)).matmul_from(
                                &loc_ao.i((.., .., ic+1)), &aow_s, 1.0, 1.0
                            );
                        }
                    }
                }, 
                XCType::GGA => {
                    for ispin in 0..nspin {
                        loc_wv.i_mut((.., 0, ispin)).mul_assign(0.5);
                        gga_grad_sum(&mut loc_vmat[ispin], loc_ao.view(), loc_wv.i((.., .., ispin)));
                    }
                },
                XCType::MGGA => {
                    for ispin in 0..nspin {
                        loc_wv.i_mut((.., 0, ispin)).mul_assign(0.5);
                        loc_wv.i_mut((.., 4, ispin)).mul_assign(0.5);
                        gga_grad_sum(&mut loc_vmat[ispin], loc_ao.view(), loc_wv.i((.., .., ispin)));
                        tau_grad_dot(&mut loc_vmat[ispin], loc_ao.view(), loc_wv.i((.., .., ispin)));
                    }
                },
                XCType::HF => {
                    unreachable!("HF gradient calculation does not support here in get_vxc");
                },
            }
            s.send((loc_vmat)).unwrap();
        } 
    );
    receiver.into_iter().for_each(
        |(loc_vmat)| 
        {
            vmat[0] += loc_vmat[0].view();
            vmat[1] += loc_vmat[1].view();
        }
    );
    omp_set_num_threads_wrapper(default_omp_num_threads);
    // nabla R = - nabla r
    for ispin in 0..nspin {
        vmat[ispin].mul_assign(-1.0);
    }
    vmat
}


/// Grid contribution of the UKS XC skeleton derivative, `vmat_s[nao, nao, 3]` per spin (the
/// row-scatter, times `2`, contracted with the per-spin density, is the XC skeleton gradient), for
/// the XC functional carried by `xc_data`.
///
/// Public so that post-SCF (doubly-hybrid) gradients can evaluate the skeleton derivative of
/// the *final* functional, whose component list is not the SCF one (`xc_data` is built by the
/// caller, e.g. from `dfa_compnt_pos` / `dfa_paramr_pos`).
///
/// `block_mem_mb` bounds the workspace of **one block of one worker**, so callers pass a
/// per-thread share of the declared budget (`utilities::memory_batch::xc_grad_block_mb`),
/// never `[ctrl] max_memory` itself.
pub fn get_vxc_rayon_new(gradient_method: &RIUHFGradient, xc_data: &XCData, grids: &Grids, mol: &Molecule, block_mem_mb: usize) -> Vec<Tsr<f64>> {
    let default_omp_num_threads = omp_get_num_threads_wrapper();

    let num_grids = grids.weights.len();
    let num_basis = mol.num_basis;

    let ao_deriv = match xc_data.xc_type {
        XCType::HF    => 1,
        XCType::LDA   => 1,
        XCType::GGA   => 2,
        XCType::MGGA  => 2,
    };
    let ao_comp = (ao_deriv + 1) * (ao_deriv + 2) * (ao_deriv + 3) / 6;
    let batch_size = 64;
    let blksize = (block_mem_mb * 1_000_000 / 8 / ((ao_comp + 1) * num_basis * batch_size))
        .min(num_grids / batch_size + 1)
        .max(4) * batch_size;
    let blksize = blksize.min(num_grids.max(1));
    // 自检：一块的工作集是 (ao_comp + 1) 个分量 x nao x blksize 的 f64，应与预算同量级
    let block_mb = ((ao_comp + 1) * num_basis * blksize * std::mem::size_of::<f64>()) as f64 / 1.0e6;
    if gradient_method.flags.print_level >= 2 {
        println!(
            "In get_vxc_rayon_new: BLKSIZE={} grids, block budget={} MiB, per-block workspace={:.1} MiB, blocks={}.",
            blksize, block_mem_mb, block_mb, (num_grids + blksize - 1) / blksize.max(1)
        );
    }
    if block_mb > 1.5 * block_mem_mb as f64 && gradient_method.flags.print_level >= 1 {
        println!(
            "[WARN] XC gradient block workspace {:.1} MiB exceeds the {:.1} MiB budget; the block size formula and the budget disagree.",
            block_mb, block_mem_mb as f64
        );
    }

    let block_settings = BlockSettings { ao_deriv, max_memory: block_mem_mb, blksize };
    let device = &xc_data.device;
    let deriv = 1usize;
    // UKS case 
    let spin = 1usize;
    let nspin = spin + 1;
    // Accumulate grid block by grid block with one accumulator per worker, see the closed-shell
    // twin in `rks.rs`: an unbounded channel holding one [nao, nao, 3, nspin] entry per block
    // grows like (ngrids / blksize) * nao^2 and dominates the memory of a large gradient.
    let mut vmat = gradient_method
        .par_block_loop(mol, grids, block_settings)
        .map(|block| {
            omp_set_num_threads_wrapper(1);

            let ng = block.weights.len();
            // Locality-screened compact block, mirroring the RKS twin:
            // contract on the compact table and scatter into full [nao,nao,3].
            let idx_opt: Option<&Vec<usize>> =
                block.active_bas.as_ref().filter(|idx| idx.len() < num_basis);
            let nbb = idx_opt.map_or(num_basis, |idx| idx.len());

            let mo_compact: Vec<MatrixFull<f64>>;
            let dm_compact: Vec<MatrixFull<f64>>;
            let loc_rho = if let Some(mo) = &xc_data.mo_coeffs
            {
                let mo_use = match &idx_opt {
                    Some(idx) => {
                        mo_compact = mo.iter().map(|m| gather_rows_full(m, idx)).collect();
                        &mo_compact
                    },
                    None => mo,
                };
                eval_rho5_batch(&block.ao, xc_data.xc_type, mo_use, xc_data.occ.as_ref().unwrap(), nspin, ng)
            } else {
                // Gather exactly the consumed spin entries (the Vec may carry
                // degenerate placeholders), mirroring the RKS twin.
                let dm_usable = !xc_data.dm.is_empty()
                    && xc_data.dm.iter().take(nspin).all(|m| m.size[0] == num_basis && !m.data.is_empty());
                let dm_use = match (&idx_opt, dm_usable) {
                    (Some(idx), true) => {
                        dm_compact = xc_data.dm.iter().take(nspin).map(|m| gather_submatrix_full(m, idx)).collect();
                        &dm_compact
                    },
                    _ => xc_data.dm,
                };
                eval_rho5_dm_only_batch(&block.ao, xc_data.xc_type, dm_use, nspin, ng)
            };
            let xc_ten = eval_xc_eff(xc_data.xc_code, xc_data.xc_params, xc_data.xc_type, spin, &loc_rho.raw(), ng, deriv);
            let loc_vxc = rt::asarray(xc_ten[1].as_ref().expect("vxc not provided for this xc_type"));

            let loc_w = rt::asarray((block.weights, device));
            let ao_shape = vec![nbb, ng, ao_comp];
            let loc_ao = rt::asarray((&block.ao.data, ao_shape, device));

            let mut loc_wv = loc_vxc * &loc_w.i((.., None));

            let mut loc_vmat = vec![
                rt::zeros(([nbb, nbb, 3], device)),
                rt::zeros(([nbb, nbb, 3], device)),
            ];
            match xc_data.xc_type {
                XCType::LDA => {
                    for ispin in 0..nspin {
                        let aow_s = &loc_ao.i((.., .., 0)) * &loc_wv.i((None, .., 0, ispin));
                        let aow_s = aow_s.t();
                        for ic in 0..3 {
                            loc_vmat[ispin].i_mut((.., .., ic, ispin)).matmul_from(
                                &loc_ao.i((.., .., ic+1)), &aow_s, 1.0, 1.0
                            );
                        }
                    }
                },
                XCType::GGA => {
                    for ispin in 0..nspin {
                        loc_wv.i_mut((.., 0, ispin)).mul_assign(0.5);
                        gga_grad_sum(&mut loc_vmat[ispin], loc_ao.view(), loc_wv.i((.., .., ispin)));
                    }
                },
                XCType::MGGA => {
                    for ispin in 0..nspin {
                        loc_wv.i_mut((.., 0, ispin)).mul_assign(0.5);
                        loc_wv.i_mut((.., 4, ispin)).mul_assign(0.5);
                        gga_grad_sum(&mut loc_vmat[ispin], loc_ao.view(), loc_wv.i((.., .., ispin)));
                        tau_grad_dot(&mut loc_vmat[ispin], loc_ao.view(), loc_wv.i((.., .., ispin)));
                    }
                },
                XCType::HF => {
                    unreachable!("HF gradient calculation does not support here in get_vxc");
                },
            }
            match idx_opt {
                Some(idx) => loc_vmat
                    .iter()
                    .map(|vc| scatter_to_full(vc, idx, num_basis, device))
                    .collect(),
                None => loc_vmat,
            }
        })
        .reduce(
            || {
                vec![
                    rt::zeros(([num_basis, num_basis, 3], device)),
                    rt::zeros(([num_basis, num_basis, 3], device)),
                ]
            },
            |mut acc, part| {
                for (a, p) in acc.iter_mut().zip(part.iter()) {
                    *a += p.view();
                }
                acc
            },
        );
    omp_set_num_threads_wrapper(default_omp_num_threads);
    // nabla R = - nabla r
    for ispin in 0..nspin {
        vmat[ispin].mul_assign(-1.0);
    }
    vmat

}
