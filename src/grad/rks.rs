// use num_traits::ToPrimitive;
use rayon::prelude::*;
// use rest_libcint::prelude::*;
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
use crate::grad::rhf::{RIRHFGradient, RIHFGradientFlags};
use crate::dft::{Grids, DFA4REST};
use crate::dft::xc_deriv::XCType;
use crate::dft::num_int::{NumInt, XCData, BlockSettings};
use crate::dft::num_int::{GRID_BLOCK_AO_NS, GRID_BLOCK_RHOXC_NS, GRID_BLOCK_CONTRACT_NS, GRID_BLOCK_COUNT};
use crate::dft::num_int::{eval_rho5_batch, eval_rho5_dm_only_batch, eval_ao_batch};
use crate::dft::libxc_itrf::{eval_xc_eff};
use crate::utilities::{self, balancing};

use tensors::matrix_blas_lapack::{_dgemm_full, omp_get_num_threads_wrapper, omp_set_num_threads_wrapper};
use tensors::matrix::matrixfullslice::MatrixFullSlice;


impl<'a> NumInt<'a> for RIRHFGradient<'a> {
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
        let num_grids = grids.weights.len();
        let num_basis = mol.num_basis;
        // determine block settings 
        let ao_deriv = match xc_data.xc_type {
            XCType::HF => 1usize,
            XCType::LDA => 1usize,
            XCType::GGA => 2usize,
            XCType::MGGA => 2usize,
        };
        let max_memory = if let Some(max_memory) = self.flags.max_memory {
            max_memory as usize
        } else {
            2000 as usize // default to 2000 MB
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
        
        let mut nelec = vec![0.0; 1];
        let mut exc_sum = vec![0.0; 1];
        let mut vxc = rt::zeros(([num_basis, num_basis, 3], device));

        let grids_iterator = self.block_loop(mol, grids, block_settings);
        // calculation starts here 
        let deriv = 1usize;          
        grids_iterator.into_iter().for_each(
            |block| 
            {
                let num_grids = block.weights.len();
                let loc_rho = if let Some(mo_coeffs) = &xc_data.mo_coeffs {
                    eval_rho5_batch(&block.ao, xc_data.xc_type, mo_coeffs, xc_data.occ.as_ref().unwrap(), spin + 1, num_grids)
                } else {
                    eval_rho5_dm_only_batch(&block.ao, xc_data.xc_type, xc_data.dm, spin + 1, num_grids)
                };

                let loc_xc_tensor = eval_xc_eff(xc_data.xc_code, xc_data.xc_params, xc_data.xc_type, spin, loc_rho.raw(), num_grids, deriv);
                
                // let loc_rho = rt::asarray((rho_array, shape, device));
                let loc_vxc = if let Some(vxc) = loc_xc_tensor[1].as_ref() {
                    rt::asarray(vxc)
                } else {
                    panic!("Exchange-correlation potential is not provided for the given xc_type: {:?}", xc_data.xc_type);
                };
                let loc_weights = rt::asarray((block.weights, device));
                let ao_shape = vec![num_basis, num_grids, ao_comp];
                let loc_ao = rt::asarray((&block.ao.data, ao_shape, device));
                
                let mut loc_wv = loc_vxc * &loc_weights.i((.., None));

                match xc_data.xc_type {
                    XCType::LDA => {
                        let aow = &loc_ao.i((.., .., 0)) * &loc_wv.i((None, ..)); 
                        let aow = aow.t();
                        for ic in 0..3 {
                            vxc.i_mut(ic).matmul_from(
                                &loc_ao.i((.., .., ic+1)), &aow, 1.0, 1.0
                            );
                        }
                    }, 
                    XCType::GGA => {
                        loc_wv.i_mut((.., 0)).mul_assign(0.5);
                        gga_grad_sum(&mut vxc, loc_ao.view(), loc_wv.view());
                    },
                    XCType::MGGA => {
                        loc_wv.i_mut((.., 0)).mul_assign(0.5);
                        loc_wv.i_mut((.., 4)).mul_assign(0.5);
                        gga_grad_sum(&mut vxc, loc_ao.view(), loc_wv.view());
                        tau_grad_dot(&mut vxc, loc_ao.view(), loc_wv.view());
                    },
                    XCType::HF => {
                        unreachable!("HF gradient calculation does not support here in get_vxc");
                    },
                }
            }
        );
        vxc.mul_assign(-1.0);

        let mut vmat = vec!();
        for ic in 0..3 {
            vmat.push(MatrixFull::from_vec([num_basis, num_basis], vxc.i((.., .., ic)).to_vec()).unwrap());
        }


        (nelec, exc_sum, vmat)

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

impl RIRHFGradient<'_> {
    pub fn calc_de_xc(&mut self) -> &mut Self {
        // get dx vxc [nao, nao, 3]
        let scf_data = self.scf_data; 
        let xc_data = self.gen_xc_data(scf_data, 0);
        // Reuse the SCF grid (points/weights/partition) instead of regenerating
        // it. The SCF grid carries the identical deterministic content (the
        // rank-local portion under MPI) plus any grid pruning applied during
        // `initialize_scf`, so the XC gradient is evaluated on the same grid
        // as the XC energy and one full grid generation is saved. AO values
        // (up to second derivatives) are still evaluated per block on the fly.
        // Fall back to a fresh build only when the SCF data holds no grids.
        let mut owned_grids: Option<Grids> = None;
        let mut owned_mol: Option<Molecule> = None;
        let t_grid_build = std::time::Instant::now();
        let grids_reused = scf_data.grids.is_some();
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
        let t_grid_build = t_grid_build.elapsed();
        let mol: &Molecule = owned_mol.as_ref().unwrap_or(&scf_data.mol);
        let mem_probe_on =
            utilities::memory_batch::mem_probe_enabled(self.scf_data.mol.ctrl.print_level);
        utilities::memory_batch::mem_probe("calc_de_xc: before get_vxc_rayon_new", mem_probe_on);
        // let dao_vxc = get_vxc(&self, &xc_data, grids, &scf_data.mol, self.flags.max_memory.unwrap_or(2000.0) as usize);
        // let dao_vxc = get_vxc_rayon(&self, &xc_data, grids, &scf_data.mol);
        // 每线程每块的 XC 工作集预算，由 [ctrl] max_memory 推得，未设置时保持历史上的 16 MiB
        let xc_block_mb = utilities::memory_batch::xc_grad_block_mb(self.flags.max_memory, 16.0);
        GRID_BLOCK_AO_NS.store(0, std::sync::atomic::Ordering::Relaxed);
        GRID_BLOCK_RHOXC_NS.store(0, std::sync::atomic::Ordering::Relaxed);
        GRID_BLOCK_CONTRACT_NS.store(0, std::sync::atomic::Ordering::Relaxed);
        GRID_BLOCK_COUNT.store(0, std::sync::atomic::Ordering::Relaxed);
        let t_pipeline = std::time::Instant::now();
        let dao_vxc = get_vxc_rayon_new(&self, &xc_data, grids, mol, xc_block_mb as usize);
        let t_pipeline = t_pipeline.elapsed();
        if self.scf_data.mol.ctrl.print_level >= 2 {
            let nthr = rayon::current_num_threads() as f64;
            let nblk = GRID_BLOCK_COUNT.load(std::sync::atomic::Ordering::Relaxed);
            let ao_s = GRID_BLOCK_AO_NS.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1.0e9;
            let rho_s = GRID_BLOCK_RHOXC_NS.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1.0e9;
            let con_s = GRID_BLOCK_CONTRACT_NS.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1.0e9;
            println!("XC grad profile: grid {} ({:.3} s); block pipeline {:.3} s wall over {} blocks / {} threads; worker-sums: ao_eval {:.3} s, rho+xc {:.3} s, contraction {:.3} s (est. wall shares: {:.3}, {:.3}, {:.3} s)",
                if grids_reused { "reused from SCF" } else { "freshly built" },
                t_grid_build.as_secs_f64(), t_pipeline.as_secs_f64(), nblk, nthr as usize,
                ao_s, rho_s, con_s, ao_s / nthr, rho_s / nthr, con_s / nthr);
        }
        utilities::memory_batch::mem_probe("calc_de_xc: after get_vxc_rayon_new", mem_probe_on);
        // println!("Print dao_vxc: {:?}", dao_vxc);
        
        // contract dao_vxc and dm (tuv, uv -> tu)
        let nao = mol.num_basis;
        let mut dao_xc:Tsr<f64> = rt::zeros(([nao, 3], &xc_data.device));        
        let dm = get_dm(scf_data, &xc_data.device);
        dao_xc = 2.0 * (&dao_vxc * &dm.i((.., .., None))).sum_axes(1);
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

    pub fn calc_rks(&mut self) -> &MatrixFull<f64> {
        let mut time_records = crate::utilities::TimeRecords::new();
        time_records.new_item("rks grad", "rks grad");
        time_records.new_item("rks grad calc_de_nuc", "rks grad calc_de_nuc");
        time_records.new_item("rks grad calc_de_ovlp", "rks grad calc_de_ovlp");
        time_records.new_item("rks grad calc_de_hcore", "rks grad calc_de_hcore");
        time_records.new_item("rks grad calc_de_ext_field", "rks grad calc_de_ext_field");
        time_records.new_item("rks grad calc_de_jk", "rks grad calc_de_jk");
        time_records.new_item("rks grad calc_de_xc", "rks grad calc_de_xc");
        time_records.new_item("rks grad calc_de_qmmm", "rks grad calc_de_qmmm");
        time_records.new_item("rks grad calc_de_solvent", "rks grad calc_de_solvent");
        time_records.new_item("rks grad calc_de_ext_field", "rks grad calc_de_ext_field");

        time_records.count_start("rks grad");

        time_records.count_start("rks grad calc_de_nuc");
        self.calc_de_nuc();
        time_records.count("rks grad calc_de_nuc");

        time_records.count_start("rks grad calc_de_ovlp");
        self.calc_de_ovlp();
        time_records.count("rks grad calc_de_ovlp");

        time_records.count_start("rks grad calc_de_hcore");
        self.calc_de_hcore();
        time_records.count("rks grad calc_de_hcore");

        if self.flags.ext_field_dipole.is_some() {
            time_records.count_start("rhf grad calc_de_ext_field");
            self.calc_de_ext_field();
            time_records.count("rhf grad calc_de_ext_field");
        }

        time_records.count_start("rks grad calc_de_qmmm");
        self.calc_de_qmmm();
        time_records.count("rks grad calc_de_qmmm");

        if self.flags.factor_j.is_some() || self.flags.factor_k.is_some() {
            time_records.count_start("rks grad calc_de_jk");
            self.calc_de_jk();
            time_records.count("rks grad calc_de_jk");
        }

        time_records.count_start("rks grad calc_de_xc");
        self.calc_de_xc();
        time_records.count("rks grad calc_de_xc");

        time_records.count_start("rks grad calc_de_solvent");
        self.calc_de_solvent();
        time_records.count("rks grad calc_de_solvent");

        time_records.count("rks grad");

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


fn get_dm(scf_data: &SCF, device: &DeviceBLAS) -> Tsr<f64> {
    let dm = &scf_data.density_matrix[0];
    return rt::asarray((&dm.data, dm.size, device)).to_owned();
}



/// Grid contribution of a GGA functional to the AO derivative matrix
/// `vmat[nao,nao,3]`, from the AOs `ao[nao,ng,10]` and the weighted kernel
/// `wv[ng,4]`.
///
/// This is the contraction PySCF performs in `grad.rks._gga_grad_sum_`
/// (15 matmuls of shape `nao x ng x nao`), but expressed as explicit OpenBLAS
/// `dgemm` calls on contiguous col-major operands.  Driving the same matmuls
/// through `rstsr`'s `matmul_from` on the permuted views measured ~2.4x slower
/// (`tests/bench_gga.rs`: 2.15 s vs 0.91 s per call at nao=86, ng=115752),
/// which dominated the TDDFT response gradient.  The arithmetic is unchanged.
pub fn gga_grad_sum(vmat: &mut Tsr<f64>, ao: TsrView<f64>, wv: TsrView<f64>) {
    let nao = ao.shape()[0];
    let ng = ao.shape()[1];
    let ao_raw = ao.raw();
    let ao_off = ao.offset();
    // col-major [ng, 4]: channel c at g + c*ng
    let wv_raw = wv.raw();
    let wv_off = wv.offset();

    // Zero-copy col-major [nao,ng] views of the 10 AO components (col-major
    // [nao,ng,10]: component c starts at ao_off + c*nao*ng).
    let sz = [nao, ng];
    let ind = [1usize, nao];
    let comp = |c: usize| MatrixFullSlice {
        size: &sz,
        indicing: &ind,
        data: &ao_raw[ao_off + c * nao * ng..ao_off + (c + 1) * nao * ng],
    };

    let mut acc: Vec<MatrixFull<f64>> =
        (0..3).map(|_| MatrixFull::new([nao, nao], 0.0)).collect();

    // Single reusable [nao,ng] scratch buffer: aow_c[mu,g] = ao_c[mu,g]*wv_c[g].
    let mut aow = MatrixFull::new([nao, ng], 0.0);
    for ic in 0..4 {
        aow.data.copy_from_slice(&ao_raw[ao_off + ic * nao * ng..ao_off + (ic + 1) * nao * ng]);
        for g in 0..ng {
            let wc = wv_raw[wv_off + g + ic * ng];
            for mu in 0..nao {
                aow.data[mu + g * nao] *= wc;
            }
        }
        for d in 0..3 {
            _dgemm_full(&comp(1 + d), 'N', &aow, 'T', &mut acc[d], 1.0, 1.0);
        }
    }

    // `_make_dR_dao_w`: second-derivative AO combinations
    //   aow[d] = ao[1+d]*wv[0] + ao[..]*wv[1] + ...   with the
    // (XX,XY,XZ)=4,5,6 / (YX,YY,YZ)=5,7,8 / (ZX,ZY,ZZ)=6,8,9 map.
    let idx = [[1usize, 4, 5, 6], [2, 5, 7, 8], [3, 6, 8, 9]];
    for d in 0..3 {
        aow.data.copy_from_slice(
            &ao_raw[ao_off + idx[d][0] * nao * ng..ao_off + (idx[d][0] + 1) * nao * ng],
        );
        for g in 0..ng {
            let w0 = wv_raw[wv_off + g];
            for mu in 0..nao {
                aow.data[mu + g * nao] *= w0;
            }
        }
        for c in 1..4 {
            let src = ao_off + idx[d][c] * nao * ng;
            for g in 0..ng {
                let wg = wv_raw[wv_off + g + c * ng];
                let base = g * nao;
                for mu in 0..nao {
                    let k = base + mu;
                    aow.data[k] += ao_raw[src + k] * wg;
                }
            }
        }
        _dgemm_full(&aow, 'N', &comp(0), 'T', &mut acc[d], 1.0, 1.0);
    }

    let voff = vmat.offset();
    let vraw = vmat.raw_mut();
    for d in 0..3 {
        let off = voff + d * nao * nao;
        for (dst, src) in vraw[off..off + nao * nao].iter_mut().zip(acc[d].data.iter()) {
            *dst += *src;
        }
    }
}


pub fn tau_grad_dot(vmat: &mut Tsr<f64>, ao:TsrView<f64>, wv:TsrView<f64>) {
    // vtau ao_x
    let aow = &ao.i((.., .., 1)) * &wv.i((None, .., 4));
    let aow = aow.t();
    // contract with ao_xx, ao_xy, ao_xz 
    vmat.i_mut((.., .., 0)).matmul_from(
        &ao.i((.., .., 4)), &aow, 1.0, 1.0 
    );
    vmat.i_mut((.., .., 1)).matmul_from(
        &ao.i((.., .., 5)), &aow, 1.0, 1.0 
    );
    vmat.i_mut((.., .., 2)).matmul_from(
        &ao.i((.., .., 6)), &aow, 1.0, 1.0 
    );
    // vtau ao_y
    let aow = &ao.i((.., .., 2)) * &wv.i((None, .., 4));
    let aow = aow.t();
    // contract with ao_xy, ao_yy, ao_yz
    vmat.i_mut((.., .., 0)).matmul_from(
        &ao.i((.., .., 5)), &aow, 1.0, 1.0 
    );
    vmat.i_mut((.., .., 1)).matmul_from(
        &ao.i((.., .., 7)), &aow, 1.0, 1.0 
    );
    vmat.i_mut((.., .., 2)).matmul_from(
        &ao.i((.., .., 8)), &aow, 1.0, 1.0 
    );
    // vtau ao_z
    let aow = &ao.i((.., .., 3)) * &wv.i((None, .., 4));
    let aow = aow.t();
    // contract with ao_xz, ao_yz, ao_zz
    vmat.i_mut((.., .., 0)).matmul_from(
        &ao.i((.., .., 6)), &aow, 1.0, 1.0 
    );
    vmat.i_mut((.., .., 1)).matmul_from(
        &ao.i((.., .., 8)), &aow, 1.0, 1.0 
    );
    vmat.i_mut((.., .., 2)).matmul_from(
        &ao.i((.., .., 9)), &aow, 1.0, 1.0 
    );
}    


#[allow(non_snake_case)]
fn make_dR_dao_w(ao:TsrView<f64>, wv:TsrView<f64>) -> Tsr<f64> {
    let mut aow = &ao.i((.., .., 1..4)) * &wv.i((None, .., 0));
    aow.i_mut((.., .., 0)).add_assign(&ao.i((.., .., 4)) * &wv.i((None, .., 1))); // dX nabla_x
    aow.i_mut((.., .., 0)).add_assign(&ao.i((.., .., 5)) * &wv.i((None, .., 2))); // dX nabla_y
    aow.i_mut((.., .., 0)).add_assign(&ao.i((.., .., 6)) * &wv.i((None, .., 3))); // dX nabla_z
    aow.i_mut((.., .., 1)).add_assign(&ao.i((.., .., 5)) * &wv.i((None, .., 1))); // dY nabla_x
    aow.i_mut((.., .., 1)).add_assign(&ao.i((.., .., 7)) * &wv.i((None, .., 2))); // dY nabla_y
    aow.i_mut((.., .., 1)).add_assign(&ao.i((.., .., 8)) * &wv.i((None, .., 3))); // dY nabla_z
    aow.i_mut((.., .., 2)).add_assign(&ao.i((.., .., 6)) * &wv.i((None, .., 1))); // dZ nabla_x
    aow.i_mut((.., .., 2)).add_assign(&ao.i((.., .., 8)) * &wv.i((None, .., 2))); // dZ nabla_y
    aow.i_mut((.., .., 2)).add_assign(&ao.i((.., .., 9)) * &wv.i((None, .., 3))); // dZ nabla_z
    
    aow 
}


fn get_vxc(gradient_method: &RIRHFGradient, xc_data: &XCData, grids: &mut Grids, mol: &Molecule, max_memory:usize) -> Tsr<f64> {
    let num_grids = grids.weights.len();
    let num_basis = mol.num_basis;
    // determine block settings 
    let ao_deriv = match xc_data.xc_type {
        XCType::HF => 1usize,
        XCType::LDA => 1usize,
        XCType::GGA => 2usize,
        XCType::MGGA => 2usize,
    };
    // let max_memory = if let Some(max_memory) = gradient_method.flags.max_memory {
    //     max_memory as usize
    // } else {
    //     2000 as usize // default to 2000 MB
    // };
    let ao_comp = (ao_deriv + 1) * (ao_deriv + 2) * (ao_deriv + 3) / 6; // number of components for ao derivatives 
    let blksize = max_memory * 1_000_000 / 8 / ((ao_comp + 1) * num_basis);
    let blksize = blksize.min(num_grids).max(4); 
    // let blksize:usize = 1024;
    let block_settings = BlockSettings {
        ao_deriv,
        max_memory,
        blksize,
    };

    let device = &xc_data.device;
    
    // let mut nelec = vec![0.0; 1];
    // let mut exc_sum = vec![0.0; 1];
    let mut vxc = rt::zeros(([num_basis, num_basis, 3], device));

    let grids_iterator = gradient_method.block_loop(mol, grids, block_settings);
    // calculation starts here 
    let deriv = 1 as usize;          
    grids_iterator.into_iter().for_each(
        |block| 
        {
            let num_grids = block.weights.len();
            let loc_rho = if let Some(mo_coeffs) = &xc_data.mo_coeffs {
                eval_rho5_batch(&block.ao, xc_data.xc_type, mo_coeffs, xc_data.occ.as_ref().unwrap(), 1, num_grids)
            } else {
                eval_rho5_dm_only_batch(&block.ao, xc_data.xc_type, xc_data.dm, 1, num_grids)
            };

            let loc_xc_tensor = eval_xc_eff(xc_data.xc_code, xc_data.xc_params, xc_data.xc_type, 0, &loc_rho.raw(), num_grids, deriv);
            
            // let loc_rho = rt::asarray((rho_array, shape, device));
            let loc_vxc = if let Some(vxc) = loc_xc_tensor[1].as_ref() {
                rt::asarray(vxc)
            } else {
                panic!("Exchange-correlation potential is not provided for the given xc_type: {:?}", xc_data.xc_type);
            };
            // println!("[RIRHFGradient] In get_vxc, shape of loc_vxc: {:?}", loc_vxc);
            let loc_weights = rt::asarray((block.weights, device));
            let ao_shape = vec![num_basis, num_grids, ao_comp];
            let loc_ao = rt::asarray((&block.ao.data, ao_shape, device));

            let mut loc_wv = loc_vxc * &loc_weights.i((.., None));

            // println!("In get_vxc shape of loc_wv: {:?}", loc_wv.shape());

            match xc_data.xc_type {
                XCType::LDA => {
                    let aow = &loc_ao.i((.., .., 0)) * &loc_wv.i((None, .., 0)); 
                    // let aow = aow.t();
                    for ic in 0..3 {
                        vxc.i_mut((.., .., ic)).matmul_from(
                            &loc_ao.i((.., .., ic+1)), &aow.t(), 1.0, 1.0
                        );
                    }
                }, 
                XCType::GGA => {
                    loc_wv.i_mut((.., 0)).mul_assign(0.5);
                    gga_grad_sum(&mut vxc, loc_ao.view(), loc_wv.view());
                },
                XCType::MGGA => {
                    loc_wv.i_mut((.., 0)).mul_assign(0.5);
                    loc_wv.i_mut((.., 4)).mul_assign(0.5);
                    gga_grad_sum(&mut vxc, loc_ao.view(), loc_wv.view());
                    tau_grad_dot(&mut vxc, loc_ao.view(), loc_wv.view());
                },
                XCType::HF => {
                    unreachable!("HF gradient calculation does not support here in get_vxc");
                },
            }
        }
    );
    // nabla R = - nabla r 
    vxc *= -1.0;

    vxc

}


fn get_vxc_rayon(gradient_method: &RIRHFGradient, xc_data: &XCData, grids: &mut Grids, mol: &Molecule) -> Tsr<f64> {
    //In order to ensure the efficiency, we disable the openmp ability and re-open it in the end of subroutien
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
    let mut vmat = rt::zeros(([num_basis, num_basis, 3], device));

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
                eval_rho5_batch(&loc_ao, xc_data.xc_type, mo_coeffs, xc_data.occ.as_ref().unwrap(), 1, num_grids)
            } else {
                eval_rho5_dm_only_batch(&loc_ao, xc_data.xc_type, xc_data.dm, 1, num_grids)
            };
            // eval vxc on grids 
            let loc_xc_tensor = eval_xc_eff(xc_data.xc_code, xc_data.xc_params, xc_data.xc_type, 0, &loc_rho.raw(), num_grids, deriv);
            // let loc_rho = rt::asarray((rho_array, shape, device));
            let loc_vxc = if let Some(vxc) = loc_xc_tensor[1].as_ref() {
                rt::asarray(vxc)
            } else {
                panic!("Exchange-correlation potential is not provided for the given xc_type: {:?}", xc_data.xc_type);
            };
            // eval batch vxc_ao 
            let mut loc_vmat = rt::zeros(([num_basis, num_basis, 3], device));
            let loc_weights = rt::asarray((&grids.weights[range_grids.clone()], device));
            let ao_shape = vec![num_basis, num_grids, ao_comp];
            let loc_ao = rt::asarray((&loc_ao.data, ao_shape, device));
            
            let mut loc_wv = loc_vxc * &loc_weights.i((.., None));

            match xc_data.xc_type {
                XCType::LDA => {
                    let aow = &loc_ao.i((.., .., 0)) * &loc_wv.i((None, .., 0)); 
                    let aow = aow.t();
                    for ic in 0..3 {
                        loc_vmat.i_mut((.., .., ic)).matmul_from(
                            &loc_ao.i((.., .., ic+1)), &aow, 1.0, 1.0
                        );
                    }
                }, 
                XCType::GGA => {
                    loc_wv.i_mut((.., 0)).mul_assign(0.5);
                    gga_grad_sum(&mut loc_vmat, loc_ao.view(), loc_wv.view());
                },
                XCType::MGGA => {
                    loc_wv.i_mut((.., 0)).mul_assign(0.5);
                    loc_wv.i_mut((.., 4)).mul_assign(0.5);
                    gga_grad_sum(&mut loc_vmat, loc_ao.view(), loc_wv.view());
                    tau_grad_dot(&mut loc_vmat, loc_ao.view(), loc_wv.view());
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
            vmat += loc_vmat;
        }
    );
    omp_set_num_threads_wrapper(default_omp_num_threads);
    // nabla R = - nabla r
    vmat *= -1.0;
    vmat
}


/// Grid contribution of the XC skeleton gradient, `dao_vxc[nao, nao, 3]` (the AO-derivative
/// potential contraction whose row-scatter, times `2`, is the XC skeleton gradient), for the
/// XC functional carried by `xc_data`.
///
/// Public so that post-SCF (doubly-hybrid) gradients can evaluate the skeleton derivative of
/// the *final* functional, whose component list is not the SCF one (`xc_data` is built by the
/// caller, e.g. from `dfa_compnt_pos` / `dfa_paramr_pos`).
///
/// `block_mem_mb` bounds the workspace of **one block of one worker**, so callers pass a
/// per-thread share of the declared budget (`utilities::memory_batch::xc_grad_block_mb`),
/// never `[ctrl] max_memory` itself.
pub fn get_vxc_rayon_new(gradient_method: &RIRHFGradient, xc_data: &XCData, grids: &Grids, mol: &Molecule, block_mem_mb: usize) -> Tsr<f64> {
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

    // Accumulate grid block by grid block, one accumulator per worker, instead of buffering every
    // block result in an unbounded channel. Each block produces a [nao, nao, 3] matrix, so a
    // channel holding one entry per block grows like (ngrids / blksize) * nao^2, which reaches tens
    // of GB for a few hundred basis functions. `reduce` keeps the live set at O(workers) and
    // produces the identical sum.
    let mut vxc = gradient_method
        .par_block_loop(mol, grids, block_settings)
        .map(|block| {
            omp_set_num_threads_wrapper(1);

            let ng = block.weights.len();
            let t_rhoxc = std::time::Instant::now();
            let loc_rho = if let Some(mo) = &xc_data.mo_coeffs
            {
                eval_rho5_batch(&block.ao, xc_data.xc_type, mo, xc_data.occ.as_ref().unwrap(), 1, ng)
            } else {
                eval_rho5_dm_only_batch(&block.ao, xc_data.xc_type, xc_data.dm, 1, ng)
            };

            let xc_ten = eval_xc_eff(xc_data.xc_code, xc_data.xc_params, xc_data.xc_type, 0, &loc_rho.raw(), ng, deriv);
            GRID_BLOCK_RHOXC_NS.fetch_add(t_rhoxc.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
            let loc_vxc = rt::asarray(xc_ten[1].as_ref().expect("vxc not provided for this xc_type"));

            let loc_w = rt::asarray((block.weights, device));
            let ao_shape = vec![num_basis, ng, ao_comp];
            let loc_ao = rt::asarray((&block.ao.data, ao_shape, device));

            let mut wv = loc_vxc * &loc_w.i((.., None));
            let mut loc_vmat = rt::zeros(([num_basis, num_basis, 3], device));

            let t_contract = std::time::Instant::now();
            match xc_data.xc_type {
                XCType::LDA => {
                    // `loc_vmat` has the Cartesian axis LAST ([nao,nao,3]), like
                    // the GGA/MGGA branches and `gga_grad_sum`; indexing it with
                    // a bare `ic` addresses the wrong axis.
                    let aow = &loc_ao.i((.., .., 0)) * &wv.i((None, .., 0));
                    let aow = aow.t();
                    for ic in 0..3 {
                        loc_vmat.i_mut((.., .., ic)).matmul_from(
                            &loc_ao.i((.., .., ic + 1)), &aow, 1.0, 1.0);
                    }
                }
                XCType::GGA => {
                    wv.i_mut((.., 0)).mul_assign(0.5);
                    gga_grad_sum(&mut loc_vmat, loc_ao.view(), wv.view());
                }
                XCType::MGGA => {
                    wv.i_mut((.., 0)).mul_assign(0.5);
                    wv.i_mut((.., 4)).mul_assign(0.5);
                    gga_grad_sum(&mut loc_vmat, loc_ao.view(), wv.view());
                    tau_grad_dot(&mut loc_vmat, loc_ao.view(), wv.view());
                }
                XCType::HF => unreachable!("HF gradient not supported in get_vxc_rayon_new"),
            }
            GRID_BLOCK_CONTRACT_NS.fetch_add(t_contract.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
            loc_vmat
        })
        .reduce(
            || rt::zeros(([num_basis, num_basis, 3], device)),
            |mut acc, part| {
                acc += part.view();
                acc
            },
        );

    omp_set_num_threads_wrapper(default_omp_num_threads);
    vxc *= -1.0;
    vxc
}


// #[cfg(test)]
// #[allow(non_snake_case)]
// mod debug {
//     use super::*;
//     use crate::ctrl_io::InputKeywords;
//     use crate::scf_io::scf_without_build;

//     #[test]
//     fn test_nh3() {
//         let scf_data = initialize_nh3();
//         let time = std::time::Instant::now();
//         let scf_grad = test_with_scf(&scf_data);
//         println!("Time elapsed: {:?}", time.elapsed());

//         let de = scf_grad.result.get("de").unwrap().clone();
//         let de = rt::asarray((de.data, de.size));
//         #[rustfmt::skip]
//         let de_ref = vec![
//             -0.1137786866, -0.1161365056, -0.1125150713,
//              0.0004215289,  0.0659156136,  0.0553809300,
//              0.0630962392,  0.0488708661, -0.0140011525,
//              0.0502609185,  0.0013500258,  0.0711352938,
//         ];
//         let de_ref = rt::asarray((&de_ref, [3, 4]));
//         println!("Maximum Error {:?}", (&de_ref - &de).abs().max_all());
//         assert!((de_ref - de).abs().max_all() < 1.0e-5);
//     }


//     fn test_with_scf(scf_data: &SCF) -> RIRHFGradient {
//         let mut scf_grad = RIRHFGradient::new(scf_data);
//         scf_grad.calc_rks();

//         println!("=== de ===");
//         let de = scf_grad.result.get("de").unwrap().clone();
//         let de = rt::asarray((de.data, de.size));
//         println!("{:12.6}", de.t());

//         println!("=== de_nuc ===");
//         scf_grad.result.get("de_nuc").map(|de| {
//             let de = rt::asarray((&de.data, de.size));
//             println!("{:12.6}", de.t());
//         });

//         println!("=== de_ovlp ===");
//         scf_grad.result.get("de_ovlp").map(|de| {
//             let de = rt::asarray((&de.data, de.size));
//             println!("{:12.6}", de.t());
//         });

//         println!("=== de_hcore ===");
//         scf_grad.result.get("de_hcore").map(|de| {
//             let de = rt::asarray((&de.data, de.size));
//             println!("{:12.6}", de.t());
//         });

//         println!("=== de_j ===");
//         scf_grad.result.get("de_j").map(|de| {
//             let de = rt::asarray((&de.data, de.size));
//             println!("{:12.6}", de.t());
//         });

//         println!("=== de_k ===");
//         scf_grad.result.get("de_k").map(|de| {
//             let de = rt::asarray((&de.data, de.size));
//             println!("{:12.6}", de.t());
//         });

//         println!("=== de_jaux ===");
//         scf_grad.result.get("de_jaux").map(|de| {
//             let de = rt::asarray((&de.data, de.size));
//             println!("{:12.6}", de.t());
//         });

//         println!("=== de_kaux ===");
//         scf_grad.result.get("de_kaux").map(|de| {
//             let de = rt::asarray((&de.data, de.size));
//             println!("{:12.6}", de.t());
//         });

//         return scf_grad;
//     }

//     fn initialize_nh3() -> SCF {
//         let input_token = r##"
// [ctrl]
//      print_level =          2
//      xc =                   "pbe"
//      basis_path =           "basis-set-pool/def2-SVP"
//      auxbas_path =          "basis-set-pool/def2-SVP-JKFIT"
//      basis_type =           "spheric"
//      eri_type =             "ri-v"
//      auxbas_type =          "spheric"
//      guessfile =            "none"
//      chkfile =              "none"
//      charge =               0.0
//      spin =                 1.0
//      spin_polarization =    false
//      auxbasis_response =    true
//      external_grids =       "none"
//      initial_guess=         "sad"
//      mixer =                "diis"
//      num_max_diis =         8
//      start_diis_cycle =     3
//      mix_param =            0.8
//      max_scf_cycle =        100
//      scf_acc_rho =          1.0e-10
//      scf_acc_eev =          1.0e-10
//      scf_acc_etot =         1.0e-11
//      num_threads =          16

// [geom]
//     name = "NH3"
//     unit = "Angstrom"
//     position = """
//         N  0.0  0.0  0.0
//         H  0.0  1.5  1.0
//         H  1.4  1.1  0.0
//         H  1.2  0.0  1.3
//     """
// "##;
//         let keys = toml::from_str::<serde_json::Value>(&input_token[..]).unwrap();
//         let (mut ctrl, mut geom) = InputKeywords::parse_ctl_from_json(&keys).unwrap();
//         let mol = Molecule::build_native(ctrl, geom, None).unwrap();
//         let mut scf_data = scf_io::SCF::build(mol, &None);
//         scf_without_build(&mut scf_data, &None);
//         return scf_data;
//     }

// }


