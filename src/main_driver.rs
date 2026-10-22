#![allow(unused)]
extern crate rest_tensors as tensors;
//extern crate rest_libxc as libxc;
extern crate chrono as time;
extern crate hdf5_metno as hdf5;
use std::{f64, fs::File, io::{Write,Read}};
use std::io;
use std::path::PathBuf;
use crate::basis_io::ecp::ghost_effective_potential_matrix;
use crate::external_field::num_dipole::numerical_dipole;
use crate::geom_io::{GeomUnit, get_mass_charge};
use num_traits::Pow;
use pyo3::prelude::*;
//use autocxx::prelude::*;
use crate::ctrl_io::JobType;
use crate::constants::{BOHR, AU2DEBYE};
use crate::scf_io::{scf_without_build, SCFType, SCF};
use tensors::{MathMatrix, MatrixFull};
use tensors::matrix_blas_lapack::_dsyevd;
use crate::{utilities, ri_pt2, ri_rpa, dft, scf_io, post_scf_analysis, lib_rint};

//use rayon;
use crate::constants::EV;
use crate::grad::{formated_force, numerical_force};
use crate::initial_guess::enxc::{effective_nxc_matrix, effective_nxc_tensors};
//use crate::grad::rhf::Gradient;
use crate::initial_guess::sap::*;

use anyhow;
//use crate::isdf::error_isdf;
use crate::dft::DFA4REST;
use crate::post_scf_analysis::mulliken::{mulliken_pop, mulliken_spin_pop};
use crate::post_scf_analysis::spin_correction::apply_yamaguchi_spin_correction;

//use crate::post_scf_analysis::{post_scf_correlation, print_out_dfa, save_chkfile};
use crate::scf_io::{initialize_scf, scf};
use time::{DateTime,Local};
use crate::molecule_io::Molecule;
//use crate::isdf::error_isdf;
//use crate::dft::DFA4REST;
use crate::post_scf_analysis::{post_scf_correlation, print_out_dfa, save_chkfile, rand_wf_real_space, cube_build, molden_build, post_ai_correction,quasiparticle_methods,rrs_pbc};
use liblbfgs::{lbfgs,Progress};
use crate::mpi_io::{MPIOperator,MPIData};
use std::collections::HashMap;
use serde_json::json;

//use crate::mpi_io::initialization;

fn print_r12inv2_export(observables: &lib_rint::RhfVeeObservables) {
    println!("E_J(r2)            : {:.16e}", observables.ej);
    println!("E_K(r2)            : {:.16e}", observables.ek);
    println!("E(r2)              : {:.16e}", observables.total);
}


pub fn main_driver() -> anyhow::Result<()> {

                                  
    //let mut time_mark = utilities::TimeRecords::new();
    //time_mark.new_item("Overall", "the whole job");
    //time_mark.new_item("SCF", "the scf procedure");




    // VERY IMPORTANCE: introduce mpi_operator:
    let (mut mpi_operator , mut mpi_data)= MPIData::initialization();

    // Under MPI, every rank executes the same code, so an ungated print would appear once
    // per process in the merged output. The `print_level` gating in `Molecule::build`
    // already suppresses rank-gated prints on non-root ranks, but many prints (and the
    // `log` macros, whose stdout target is not print_level-gated) are unconditional.
    // As a blanket fix, redirect the standard output of all non-root ranks to /dev/null;
    // stderr is intentionally kept so that warnings and MPI runtime errors remain visible.
    #[cfg(feature = "mpi")]
    if let Some(mpi_op) = &mpi_operator {
        if mpi_op.rank != 0 {
            use std::os::unix::io::AsRawFd;
            let devnull = std::fs::OpenOptions::new()
                .write(true)
                .open("/dev/null")
                .expect("Failed to open /dev/null for redirecting the standard output of non-root MPI ranks");
            unsafe {
                libc::dup2(devnull.as_raw_fd(), libc::STDOUT_FILENO);
            }
        }
    }

    // Stamp the very first lines of the output with the git revision of the source
    // tree this binary was compiled from. Non-root MPI ranks have already had their
    // stdout redirected to /dev/null above, so the lines are printed exactly once.
    println!("REST git commit: {}", utilities::git_commit_string());
    println!("REST git commit date: {}", utilities::git_commit_date_string());

    let ctrl_file = utilities::parse_input().value_of("input_file").unwrap_or("ctrl.in").to_string();
    if ! PathBuf::from(ctrl_file.clone()).is_file() {
        panic!("Input file ({:}) does not exist", ctrl_file);
    }
    if crate::md::is_pure_mm_run(&ctrl_file) {
        crate::md::run_pure_mm(&ctrl_file, &mpi_operator)?;
        return Ok(());
    }
    let mut mol = Molecule::build(ctrl_file.clone(), mpi_data)?;
    mol.ctrl.ctrl_file = ctrl_file;
    if mol.ctrl.print_level>0 {println!("Molecule_name: {}", &mol.geom.name)};

    // Storage-level AO-pair pruning ([ctrl.ri_jk] pair_screen_threshold > 0) rewrites the row
    // space of the in-core RI tensor `rimatr`, and every consumer of that tensor has to address
    // it through `SCF::rimatr_pair_map`. That is wired for the SCF J/K contractions (S1), for
    // the AO-to-MO transform that feeds the post-SCF correlation energy: the `ri3mo` route and the
    // streaming PT2 driver both go through the map-aware `scf_io::ao2mo_rayon*` kernels (S2), and
    // for the analytic gradient of a hybrid functional (`grad/rhf.rs`, `grad/uhf.rs`, S3c). The
    // response, Hessian, TDDFT and quasiparticle paths still index the tensor with the full-space
    // pair tables, and would silently read the wrong rows instead of failing, so they stay
    // refused here, and the entry points that were reachable from two of them now fail closed on
    // their own (`ri_jk::require_unpruned_rimatr`).
    if mol.ctrl.ri_jk.pair_screen_threshold > 0.0 {
        let mut unsupported: Vec<&str> = Vec::new();
        if !mol.ctrl.analdrv_tasks.is_empty() {
            unsupported.push("analytical derivative tasks (analdrv_tasks)");
        }
        match mol.ctrl.job_type {
            // The analytic gradient contracts `rimatr` through its pair map, and it shares the
            // row set of the energy of the very same geometry, so a single point, a force
            // calculation and a geometry optimization driven by that force are all consistent.
            JobType::SinglePoint | JobType::Force | JobType::GeomOpt => {},
            _ => unsupported.push(
                "a job_type other than a single-point energy, a force calculation or a geometry \
                 optimization (MD, normal modes and the numerical dipole change the geometry or \
                 the field without freezing the pair mask)",
            ),
        }
        if mol.ctrl.tddft.is_some() {
            unsupported.push("TDDFT and response TDDFT (tddft)");
        }
        if mol.ctrl.hessian.is_some() {
            unsupported.push("Hessian / frequency analysis (hessian)");
        }
        if mol.ctrl.quasiparticle_methods.is_some() {
            unsupported.push("quasiparticle methods such as GW and BSE (quasiparticle_methods)");
        }
        // A numerical force differentiates the pruned energy across geometries, and the pair
        // mask is rebuilt at every displaced geometry: the finite difference then crosses a
        // change of the row set, and its error does not converge with the displacement. Freezing
        // the mask of the reference geometry for the whole stencil is still to be implemented.
        if mol.ctrl.numerical_force {
            unsupported.push("a numerical force ([ctrl] numerical_force = true)");
        }
        if mol.ctrl.outputs.iter().any(|output| output.eq("num_force")) {
            unsupported.push("numerical force output (outputs = \"num_force\")");
        }
        if !unsupported.is_empty() {
            return Err(anyhow::anyhow!(
                "Storage-level AO-pair pruning is enabled ([ctrl.ri_jk] pair_screen_threshold = \
                 {:e}). It is implemented for a single-point energy and for the post-SCF \
                 correlation energy of that single point, but not for: {}. Set \
                 pair_screen_threshold = 0.0 to run these paths.",
                mol.ctrl.ri_jk.pair_screen_threshold,
                unsupported.join(", ")
            ));
        }
    }
    if mol.ctrl.print_level>=2 {
        println!("{}", mol.ctrl.formated_output_in_toml());
    }
    // `[ctrl] num_threads` and `[ctrl] max_memory` are per-process values, so a hybrid
    // MPI + threads job has to divide the cores and the memory of a node by the number of ranks
    // placed on it. Warn once at startup when the requested thread count of all processes
    // together exceeds what this node offers. Only the root rank reaches this point with a
    // standard output (the other ranks are redirected to /dev/null above).
    utilities::memory_batch::warn_thread_oversubscription(
        mol.ctrl.num_threads,
        mpi_operator.as_ref().map(|mpi_op| mpi_op.size),
    );
    let mut time_mark = initialize_time_record(&mol);
    time_mark.count_start("Overall");

    if mol.ctrl.deep_pot {
        //let mut scf_data = scf_io::SCF::build(&mut mol);
        let mut effective_hamiltonian = mol.int_ij_matrixupper(String::from("hcore"));
        //effective_hamiltonian.formated_output(5, "full");
        let effective_nxc = effective_nxc_matrix(&mut mol);
        effective_nxc.formated_output(5, "full");
        effective_hamiltonian.data.iter_mut().zip(effective_nxc.data.iter()).for_each(|(to,from)| {*to += from});

        let mut ecp = mol.int_ij_matrixupper(String::from("ecp"));
        //println!("ecp: {:?}", ecp);
        ecp.iter_mut().zip(effective_nxc.iter()).for_each(|(to, from)| {*to -= from});

        ecp.formated_output(5, "full");
        let acc_error = ecp.iter().fold(0.0, |acc, x| {acc + x.abs()});
        println!("acc_error: {}", acc_error);
        
        return Ok(())
    }

    if mol.ctrl.bench_eps {
        let ecp = mol.int_ij_matrixupper(String::from("ecp"));
        let enxc = effective_nxc_matrix(&mut mol);
        let gep = ghost_effective_potential_matrix(
            &mol.cint_env, &mol.cint_atm, &mol.cint_bas, &mol.cint_type, mol.num_basis, 
            &mol.geom.ghost_ep_path, &mol.geom.ghost_ep_pos);

        let d12 = ecp.data.iter().zip(enxc.data.iter()).fold(0.0, |acc, dt| {acc + (dt.0 -dt.1).powf(2.0)});
        let d13 = ecp.data.iter().zip(gep.data.iter()).fold(0.0, |acc, dt| {acc + (dt.0 -dt.1).powf(2.0)});
        let num_data = ecp.data.len() as f64;
        println!("Compare between ECP, ENXC and GEP with the matrix sizes of");
        println!(" {:?}, {:?}, and {:?}, respectively", ecp.size(), enxc.size(), gep.size());
        println!("RMSDs between (ECP, ENXC) and (ECP, GEP): ({:16.8}, {:16.8})", 
            (d12/num_data).powf(0.5), (d13/num_data).powf(0.5)
        );

        return Ok(())
    }
    //check rrs-pbc
    let mut rrs_pbc_index_map = HashMap::new();
    let mut unit_cell_elem = vec![];
    if mol.geom.rrs_pbc {
        rrs_pbc::rrs_pbc_check(&mol);
        (rrs_pbc_index_map, unit_cell_elem) = rrs_pbc::rrs_pbc_match(&mol);
        if mol.ctrl.print_level >= 1 {
            println!("RRS-PBC check result:");
            for key in rrs_pbc_index_map.keys() {
                println!("For step path = {:?}, found target index = {:?}",key,rrs_pbc_index_map.get(key).unwrap());
            };
        };
    }



    // initialize the time record
    // initialize the SCF procedure
    time_mark.count_start("SCF");
    // `mol` is consumed by `SCF::build`, so read the resume switch first.
    let resume_from_checkpoint = mol
        .ctrl
        .quasiparticle_methods
        .as_ref()
        .map(|qp| qp.resume_from_checkpoint)
        .unwrap_or(false);
    let mut scf_data = scf_io::SCF::build(mol,&mpi_operator);
    time_mark.count("SCF");
    if resume_from_checkpoint {
        // Resume path (see `crate::fileop::gw_checkpoint`): the SCF *iterations*
        // and the GW/evGW work already stored in the archive are skipped, and
        // the converged electronic structure is installed directly onto
        // `scf_data`.
        //
        // `SCF::build` above already ran `initialize_scf`, which builds the
        // molecule/cint data, the DFT grids and -- crucially -- the RI
        // integrals (`prepare_necessary_integrals`: nuc_energy, ovlp, h_core,
        // ri3fn, rimatr) that every downstream GW/BSE step consumes.  Those are
        // deliberately NOT archived, so the setup still runs; only
        // `scf_without_build` (the iteration loop) and the correlation part of
        // `performance_essential_calculations` are skipped.
        crate::fileop::gw_checkpoint::announce_scf_skipped(&scf_data);
        let gw_state = crate::fileop::gw_checkpoint::load_gw_checkpoint(&mut scf_data)?;
        // Solvent runs refresh the PCM state from the current density inside
        // `scf_without_build`; since we skip that, do it explicitly here so the
        // restored density and the solvent stay consistent.
        if scf_data.mol.ctrl.solvent_enabled {
            scf_data.refresh_solvent_with_mpi(&mpi_operator);
        }
        scf_data.gw_checkpoint_state = Some(gw_state);
        // `performance_essential_calculations` (which normally fills
        // `energies["total_energy"]`) is skipped together with the SCF, so
        // restore that bookkeeping from the archived SCF energy.
        scf_data
            .energies
            .insert("total_energy".to_string(), vec![scf_data.scf_energy]);
        println!(
            "[gw_checkpoint] resume complete: SCF iterations skipped, GW state loaded; the run \
             continues with '{}'.",
            scf_data
                .mol
                .ctrl
                .quasiparticle_methods
                .as_ref()
                .map(|qp| qp.gw_or_bse.clone())
                .unwrap_or_default()
        );
    } else {
        // perform the SCF and post SCF evaluation for the specified xc method
        performance_essential_calculations(&mut scf_data, &mut time_mark, &mpi_operator);
    }

    let mut json_extra: HashMap<String, serde_json::Value> = HashMap::new();

    let spin_correction_scheme: Option<String> = scf_data.mol.ctrl.spin_correction_scheme.clone();
    match spin_correction_scheme.as_deref() {
        Some("yamaguchi") => {
            apply_yamaguchi_spin_correction(&mut scf_data, &mut time_mark, &mpi_operator);
        },
        None => {
        },
        Some(other) => {
            println!(
                "Warning: Unrecognized spin correction scheme '{}'.\nOnly 'yamaguchi' is currently supported. Spin correction will be skipped.",
                other
            );            
        }
    }


    let jobtype = scf_data.mol.ctrl.job_type.clone();
    match jobtype {
        JobType::Force => {
            let (_, gradient) = eval_force(&mut scf_data, &mut time_mark, &mpi_operator);
            json_extra.insert("gradient".to_string(), json!(gradient.data));
        },
        JobType::NumDipole => {
            time_mark.count_start("numerical dipole");
            if scf_data.mol.ctrl.print_level>0 {
                println!("Numerical dipole calculation invoked");
            }

            let displace = scf_data.mol.ctrl.ndipole_displacement;
            let dp_au = numerical_dipole(&scf_data, displace);
            let dp = dp_au.iter().map(|x| *x * AU2DEBYE).collect::<Vec<f64>>();
            println!("Dipole Moment in A.U. : {:16.8}, {:16.8}, {:16.8}", dp_au[0], dp_au[1], dp_au[2]);
            println!("Dipole Moment in DEBYE: {:16.8}, {:16.8}, {:16.8}", dp[0], dp[1], dp[2]);
        },
        JobType::GeomOpt => {
            //let opt_engine = scf_data.mol.ctrl.opt_engine.clone().unwrap_or("lbfgs".to_string());
            let opt_engine = scf_data.mol.ctrl.opt_engine.clone().expect("Optimization engine is not specified");
            if opt_engine == "lbfgs" {
                println!("LBFGS geometry optimization invoked");
                time_mark.count_start("geom_opt");
                if scf_data.mol.ctrl.print_level>0 {
                    println!("Geometry optimization invoked");
                }
                let displace = scf_data.mol.ctrl.nforce_displacement/BOHR;

                let mut position = scf_data.mol.geom.position.iter().map(|x| *x).collect::<Vec<f64>>();
                lbfgs().minimize(
                    &mut position, 
                    |x: &[f64], gx: &mut [f64]| {
                        scf_data.mol.geom.geom_update(x, GeomUnit::Bohr);
                        if scf_data.mol.ctrl.print_level>0 {
                            println!("Input geometry in this round is:");
                            println!("{}", scf_data.mol.geom.formated_geometry());
                        }
                        //scf_data.mol.ctrl.initial_guess = String::from("inherit");
                        //initialize_scf(&mut scf_data, &mpi_operator);
                        //performance_essential_calculations(&mut scf_data, &mut time_mark, &mpi_operator);
                        //let (energy, nforce) = numerical_force(&scf_data, displace, &mpi_operator);
                        
                        let coords = MatrixFull::from_vec([3, x.len()/3], x.to_vec()).unwrap();
                        let (energy, force) = eval_force_with_position(&mut scf_data, &mut time_mark, &mpi_operator, &coords);

                        gx.iter_mut().zip(force.iter()).for_each(|(to, from)| {*to = *from});

                        if scf_data.mol.ctrl.print_level>0 {
                            println!("Output force in this round [a.u.] is:");
                            println!("{}", formated_force(&force, &scf_data.mol.geom.elem));
                        }

                        Ok(energy)
                    },
                    |prgr| {
                        println!("Iteration {}, Evaluation: {}", &prgr.niter, &prgr.neval);
                        println!(" xnorm = {}, gnorm = {}, step = {}",
                            &prgr.xnorm, &prgr.gnorm, &prgr.step
                        );
                        false
                    },
                );
                println!("Geometry after relaxation [Ang]:");
                println!("{}", scf_data.mol.geom.formated_geometry());
                time_mark.count("geom_opt");

                time_mark.report("geom_opt");
            } else if opt_engine == "geometric_pyo3" {
                #[cfg(feature = "geometric-pyo3")]
                {
                    //println!("Geometric geometry optimization invoked");
                    time_mark.count_start("geom_opt");
                    if scf_data.mol.ctrl.print_level>0 {
                        println!("Geometry optimization invoked using the optimization engine of geometric_pyo3");
                    }
                    geometric_pyo3_impl::optimize_geometric_pyo3(&mut scf_data, &mut time_mark, &mpi_operator)
                        .map_err(|e| anyhow::anyhow!("Geometry optimization (geometric_pyo3 engine) failed: {}", e))?;
                    println!("Geometry after relaxation [Ang]:");
                    println!("{}", scf_data.mol.geom.formated_geometry());
                    time_mark.count("geom_opt");
                    time_mark.report("geom_opt");
                }
                #[cfg(not(feature = "geometric-pyo3"))]
                panic!("Geometric-Pyo3 feature is not enabled. Please enable it in Cargo.toml.");
            } else {
                panic!("Invalid optimization engine: {}", opt_engine);
            }
        },
        // UNVERIFIED NORMAL MODES CALCULATION
        JobType::NormalModes => {
            eval_normal_modes(&mut scf_data, &mut time_mark, &mpi_operator);
        },
        JobType::MD => {
            let ctrl_file = utilities::parse_input().value_of("input_file").unwrap_or("ctrl.in").to_string();
            let (sd, tm, mo) = crate::md::run_md(scf_data, time_mark, mpi_operator, &ctrl_file);
            scf_data = sd;
            time_mark = tm;
            mpi_operator = mo;
        },
        // ------------
        _ => {}
    }

    // Stability mode resolution: `[tddft] stability` wins; the top-level
    // `check_stab` (same string values) is the fallback. "auto" is the
    // recommended value: it currently resolves to "full" (internal + external)
    // inside `stability::stability` and may become method-aware in the future,
    // whereas the meaning of "full" stays frozen.
    let mut stab_mode = scf_data
        .mol
        .ctrl
        .tddft
        .as_ref()
        .map_or(String::from("off"), |t| t.stability.clone());
    if stab_mode == "off" {
        stab_mode = scf_data.mol.ctrl.check_stab.clone();
    }
    if stab_mode != "off" {
        time_mark.new_item("Stability", "the SCF stability analysis (TDDFT Hessian)");
        time_mark.count_start("Stability");

        match crate::ri_tddft::stability::stability(&scf_data, &stab_mode) {
            Ok(report) => {
                // Expose the Hessian roots in rest_results.json, grouped under
                // "stability" like the "tddft" block (machine-readable
                // regression input; the lowest internal root is the stability
                // verdict quantity).
                if !report.roots_internal.is_empty() || !report.roots_external.is_empty() {
                    json_extra.insert("stability".to_string(), json!({
                        "roots_internal": report.roots_internal,
                        "roots_external": report.roots_external,
                    }));
                }
            }
            Err(e) => return Err(anyhow::anyhow!("stability analysis failed: {e}")),
        }

        time_mark.count("Stability");
    }

    // check lib_rint
    if scf_data.mol.ctrl.run_lib_rint {
        time_mark.count_start("RI-r2");
        match scf_data.scftype {
            SCFType::RHF => {
                let p_rhf = &scf_data.density_matrix[0];
                let r2 = lib_rint::lib_vee_rhf_r2_observables_advanced(
                    &scf_data.mol.geom,
                    &scf_data.mol.basis4elem,
                    p_rhf,
                    Some(&scf_data.eigenvectors),
                    Some(&scf_data.occupation),
                );
                print_r12inv2_export(&r2);
            }
            SCFType::ROHF | SCFType::UHF => {
                if scf_data.density_matrix.len() < 2 {
                    println!("lib_rint skipped: open-shell run needs alpha/beta density matrices");
                } else {
                    let dm_spin = &scf_data.density_matrix[0..2];
                    let r2 = lib_rint::lib_vee_uhf_r2_observables_advanced(
                        &scf_data.mol.geom,
                        &scf_data.mol.basis4elem,
                        dm_spin,
                        Some(&scf_data.eigenvectors),
                        Some(&scf_data.occupation),
                    );
                    print_r12inv2_export(&r2);
                }
            }
        }
        time_mark.count("RI-r2");
    }

    //====================================
    // Now for post-xc calculations
    //====================================
    if scf_data.mol.ctrl.post_xc.len()>=1 {
        print_out_dfa(&scf_data);
    }

    //====================================
    // Now for post-SCF analysis
    //====================================
    if scf_data.mol.ctrl.print_level > 0 {
        let mulliken = mulliken_pop(&scf_data);
        println!("Mulliken population analysis:");
        let elem_tot = scf_data.mol.geom.elem.clone().into_iter().chain(scf_data.mol.geom.ghost_bs_elem.clone().into_iter()).collect::<Vec<_>>();
        let ghost_atm_start = scf_data.mol.geom.elem.len();
        for (i, (pop, atom)) in mulliken.iter().zip(elem_tot.iter()).enumerate() {
            if i < ghost_atm_start {
                println!("{:3}-{:3}: {:10.6}", i, atom, pop)
            } else {
                println!("{:3}-{:3}: {:10.6}, Ghost Atom", i, atom, pop)
            };
        }
        if scf_data.mol.spin_channel == 2 {
            let spin_pop = mulliken_spin_pop(&scf_data);
            println!("Mulliken spin population analysis (alpha - beta):");
            for (i, (pop, atom)) in spin_pop.iter().zip(elem_tot.iter()).enumerate() {
                if i < ghost_atm_start {
                    println!("{:3}-{:3}: {:10.6}", i, atom, pop)
                } else {
                    println!("{:3}-{:3}: {:10.6}, Ghost Atom", i, atom, pop)
                };
            }
        }
    }
    
    post_scf_analysis::post_scf_output(&mut scf_data, &mpi_operator);

    //====================================
    // Now for post-correlation calculations
    //====================================
    if scf_data.mol.ctrl.post_correlation.len()>=1 {
        post_scf_correlation(&mut scf_data);
    }

    //===================================
    // Now for rrs-pbc calculations
    //===================================
    if scf_data.mol.geom.rrs_pbc {
        time_mark.count_start("RRS-PBC");
        rrs_pbc::rrs_pbc_output(&scf_data,unit_cell_elem,rrs_pbc_index_map);
        time_mark.count("RRS-PBC");
    }    
    if let Some(qp_ctrl)=scf_data.mol.ctrl.quasiparticle_methods.clone(){
        print!("Now starts quasiparticle method computation!\n");
        let qp_output = quasiparticle_methods(&mut scf_data,&mpi_operator);
        if qp_output.first_excitation.is_some() || !qp_output.excitation_energies.is_empty() {
            json_extra.insert("bse".to_string(), json!({
                "first_excitation": qp_output.first_excitation,
                "excitation_energies": qp_output.excitation_energies,
            }));
        }
    }

    //===================================
    // Now for TDDFT calculations
    //===================================
            if let Some(tddft_ctrl) = &scf_data.mol.ctrl.tddft {
        // A stability analysis (`[tddft] stability != "off"` or the top-level
        // `check_stab != "off"`, run above) is mutually exclusive with the
        // excitation-energy run: a stability-only deck carries the [tddft]
        // section for the stability keywords alone.
        let stability_requested =
            tddft_ctrl.stability != "off" || scf_data.mol.ctrl.check_stab != "off";
        if !tddft_ctrl.response_tddft && !stability_requested {
            time_mark.new_item("TDDFT", "the TDDFT eigenvalue calculation");
            time_mark.count_start("TDDFT");
            match crate::ri_tddft::tddft_main(&mut scf_data) {
                Ok(output) => {
                    json_extra.insert("tddft".to_string(), json!({
                        "energies": output.energies,
                        "osc": output.osc,
                    }));
                    // Keep the raw eigenvectors for the TDDFT analytic gradient.
                    scf_data.tddft_excitations = Some(output.excitations.clone());
                }
                Err(e) => eprintln!("Error in TDDFT calculation: {}", e),
            }
            time_mark.count("TDDFT");
        }
    }

    //===================================
    // Now for response TDDFT calculations
    //===================================
    if let Some(tddft_ctrl) = &scf_data.mol.ctrl.tddft {
        if tddft_ctrl.response_tddft {
            time_mark.new_item("ResponseTDDFT", "the response TDDFT calculation");
            time_mark.count_start("ResponseTDDFT");
            if let Err(e) = crate::ri_tddft::response_tddft(&mut scf_data) {
                eprintln!("Error in response TDDFT calculation: {}", e);
            }
            time_mark.count("ResponseTDDFT");
        }
    }

    //===================================
    // CP-HF / Hessian / Frequency calculations
    //===================================
    if let Some(ref hess_ctrl) = scf_data.mol.ctrl.hessian {
        let ho = crate::hessian::rhf_hessian_main(&scf_data, hess_ctrl, &mut time_mark);
        let mut hessian = json!({
            "total_max_abs": ho.total_max_abs,
            "h_partial_max_abs": ho.h_partial_max_abs,
            "cphf_contrib_max_abs": ho.cphf_contrib_max_abs,
        });
        if !ho.frequencies_cm.is_empty() {
            hessian["frequencies_cm"] = json!(ho.frequencies_cm);
        }
        json_extra.insert("hessian".to_string(), hessian);
    }
    
    //===================================
    // Analytical derivative module calculations
    //===================================
    if !scf_data.mol.ctrl.analdrv_tasks.is_empty() {
        time_mark.new_item("AnalDrv", "analytical derivative module");
        time_mark.count_start("AnalDrv");
        use crate::analdrv::interface::{analdrv_interface, analdrv_json_interface};
        let tasks = scf_data.mol.ctrl.analdrv_tasks.clone();
        let config = scf_data.mol.ctrl.analdrv.clone().unwrap_or_default();
        // the results-JSON expansion (the "analdrv"/"thermo" entries) is returned by
        // analdrv_json_interface, so this driver holds no task-specific knowledge
        let analdrv_out = analdrv_interface(&mut scf_data, &tasks, &config);
        json_extra.extend(analdrv_json_interface(&analdrv_out));
        time_mark.count("AnalDrv");
    }

    time_mark.count("Overall");

    if scf_data.mol.ctrl.print_level > 0 {
        println!("");
        println!("====================================================");
        println!("              REST: Mission accomplished");
        println!("====================================================");
        output_result(&scf_data);
        time_mark.report_all();
    }

    if let Some(mpi_op) = &mpi_operator {
        if mpi_op.rank == 0 {
            crate::fileop::json_dump::dump_json(&scf_data, &json_extra);
        }
    } else {
        crate::fileop::json_dump::dump_json(&scf_data, &json_extra);
    }

    //if let Some(mpi_op) = &mpi_operator {
    //    // I would like to finish the mpi world
    //}


    Ok(())
}

pub fn output_result(scf_data: &scf_io::SCF) {
    
    //--------------------------
    // 0. Solvent energy
    //--------------------------
    if scf_data.mol.ctrl.solvent_enabled{
        println!("The solvent energy    : {:18.10} Ha", scf_data.energies.get("solvent_energy").unwrap()[0]);
    }
    let xc_name = scf_data.mol.ctrl.xc.to_lowercase();

    //===========================================================
    // 1. Print SCF energy (Yamaguchi-corrected overrides SCF)
    //===========================================================
    if let Some(v) = scf_data.energies.get("yamaguchi_scf_corrected") {
        println!("The SCF energy        : {:18.10} Ha", v[0]);
    } else {
        if scf_data.mol.ctrl.smear.is_some() {
            println!("The SCF energy (E)    : {:18.10} Ha", scf_data.scf_energy);
            let sigma = scf_data.current_smear_sigma;
            let s = scf_data.smearing_entropy;
            println!("Free energy  (E-TS)  : {:18.10} Ha", scf_data.scf_energy - sigma * s);
            println!("Zero-temp energy (E0): {:18.10} Ha", scf_data.scf_energy - 0.5 * sigma * s);
        } else {
            println!("The SCF energy        : {:18.10} Ha", scf_data.scf_energy);
        }
    }


    //===========================================================
    // 2. Helper: fetch corrected or normal total energy
    //===========================================================
    let yamaguchi_tot = scf_data.energies.get("yamaguchi_tot_corrected").map(|v| v[0]);


    //===========================================================
    // 3. Print energies corresponding to the functional family
    //===========================================================
    //--------------------------
    // RPA
    //--------------------------
    if scf_data.mol.xc_data.is_rpa() {
        if let Some(e) = yamaguchi_tot {
            println!("The RPA energy        : {:18.10} Ha", e);
        } else {
            let total = scf_data.energies.get("rpa_energy").unwrap()[0];
            println!("The RPA energy        : {:18.10} Ha", total);
        }
        return;
    }
    //--------------------------
    // DH / ZRPS / SCSRPA
    //--------------------------
    if scf_data.mol.xc_data.is_fifth_dfa() {
        if let Some(e) = yamaguchi_tot {
            println!("The (R)-xDH energy    : {:18.10} Ha", e);
        } else {
            let total = scf_data.energies.get("xdh_energy").unwrap()[0];
            //let post_ai_correction = scf_data.mol.ctrl.post_ai_correction.to_lowercase();
            //let ai_correction = if xc_name.eq("r-xdh7") && post_ai_correction.eq("scc15") {
            //    let ai_correction = scf_data.energies.get("ai_correction").unwrap()[0];
            //    println!("AI Correction         : {:18.10} Ha", ai_correction);
            //    ai_correction
            //} else {
            //    0.0
            //};
            let ai_corr = scf_data.energies.get("ai_correction").map(|v| v[0]).unwrap_or(0.0);
            println!("The (R)-xDH energy    : {:18.10} Ha", total + ai_corr);
        }
        return;
    }



}

/// Perform key SCF and post-SCF calculations
/// Return the total energy of the specfied xc method
/// Assume the initialization of SCF is ready
pub fn performance_essential_calculations(scf_data: &mut SCF, time_mark: &mut utilities::TimeRecords, mpi_operator: &Option<MPIOperator>) -> f64 {

    let mut total_energy = 0.0;

    //=================================================================
    // Now evaluate the SCF energy for the given method
    //=================================================================
    time_mark.count_start("SCF");
    scf_without_build(scf_data, mpi_operator);
    //println!("debug time mark SCF turn off");
    time_mark.count("SCF");
    if scf_data.mol.ctrl.max_memory_backup.is_some() {
        scf_data.mol.ctrl.max_memory = scf_data.mol.ctrl.max_memory_backup.clone();
    }

    //==================================================================
    // Save the converged SCF results to the chkfile
    //==================================================================
    if scf_data.mol.ctrl.has_chkfile {
        if let Some(mp_op) = mpi_operator {
            if mp_op.rank == 0 {
                println!("Rank 0: now save the converged SCF results");
                save_chkfile(scf_data)
            }
        } else {
            println!("now save the converged SCF results");
            save_chkfile(scf_data)
        }
    }

    //==================================================================
    // Now evaluate the advanced correction energy for the given method
    //==================================================================
    //let mut time_mark = utilities::TimeRecords::new();
    if let Some(dft_method) = &scf_data.mol.xc_data.dfa_family_pos {
        match dft_method {
            dft::DFAFamily::PT2 | dft::DFAFamily::SBGE2 => {
                //time_mark.new_item("PT2", "the PT2 evaluation");
                time_mark.count_start("PT2");
                ri_pt2::xdh_calculations(scf_data, mpi_operator);
                time_mark.count("PT2");
            },
            dft::DFAFamily::RPA => {
                //time_mark.new_item("RPA", "the RPA evaluation");
                time_mark.count_start("RPA");
                ri_rpa::rpa_calculations(scf_data, mpi_operator);
                time_mark.count("RPA");
            }
            dft::DFAFamily::SCSRPA => {
                //time_mark.new_item("SCS-RPA", "the SCS-RPA evaluation");
                time_mark.count_start("SCS-RPA");
                ri_pt2::xdh_calculations(scf_data, mpi_operator);
                time_mark.count("SCS-RPA");
            }
            _ => {}
        }
    }
    //====================================
    // Now for post ai correction
    //====================================
    if let Some(scc) = post_ai_correction(scf_data, mpi_operator) {
        scf_data.energies.insert("ai_correction".to_string(), scc);
    }

    let total_energy = collect_total_energy(scf_data);
    scf_data.energies.insert("total_energy".to_string(), vec![total_energy]);
    total_energy

}

pub fn collect_total_energy(scf_data: &SCF) -> f64 {
    //====================================
    // Determine the total energy
    //====================================
    let mut total_energy = scf_data.scf_energy;
    
    if scf_data.mol.xc_data.is_rpa() {
        total_energy = scf_data.energies.get("rpa_energy").unwrap()[0];
    } else if scf_data.mol.xc_data.is_fifth_dfa() {
        total_energy = scf_data.energies.get("xdh_energy").unwrap()[0];
    } else {
        total_energy = scf_data.scf_energy;
    }

    if let Some(post_ai_correction) = scf_data.energies.get("ai_correction") {
        total_energy += post_ai_correction[0]
    };

    total_energy

}


fn initialize_time_record(mol: &Molecule) -> utilities::TimeRecords {
    let mut time_mark = utilities::TimeRecords::new();
    time_mark.new_item("Overall", "the whole job");
    time_mark.new_item("SCF", "the scf procedure");
    let jobtype = mol.ctrl.job_type.clone();
    match jobtype {
        JobType::GeomOpt => {
            time_mark.new_item("geom_opt", "geometry optimization");
        },
        JobType::Force => {
            time_mark.new_item("force", "force calculation");
        },
        _ => {}
    };
    if let Some(dft_method) = &mol.xc_data.dfa_family_pos {
        match dft_method {
            dft::DFAFamily::PT2 | dft::DFAFamily::SBGE2 => {
                time_mark.new_item("PT2", "the PT2 evaluation");
            },
            dft::DFAFamily::RPA => {
                time_mark.new_item("RPA", "the RPA evaluation");
            }
            dft::DFAFamily::SCSRPA => {
                time_mark.new_item("SCS-RPA", "the SCS-RPA evaluation");
            }
            _ => {}
        }
    }
    if mol.geom.rrs_pbc {
        time_mark.new_item("RRS-PBC", "the RRS-PBC calculation");
    }

    time_mark

}

/* #region force and geomopt utilities */


fn eval_force(scf_data: &mut SCF, time_mark: &mut utilities::TimeRecords, mpi_operator: &Option<MPIOperator>) -> (f64, MatrixFull<f64>) {
    // this is a temporary workaround for the force evaluation
    // currently, this framework could not work for post-scf,
    // especially that currently there's no class that represents post-scf computation

    time_mark.count_start("force");
    if scf_data.mol.ctrl.print_level>0 {
        println!("Force calculation invoked");
    }

    // 力阶段的内存监控：后台线程每 20 ms 采样 RSS 并跟踪峰值；只有用户要求时
    // （`abort_on_mem_exceed`，缺省 true）才按 `[ctrl] max_memory` 中止。SCF 阶段不在覆盖范围内。
    let mem_monitor = crate::utilities::memory_monitor::MemMonitor::start_from_ctrl(
        scf_data.mol.ctrl.max_memory,
        scf_data.mol.ctrl.abort_on_mem_exceed,
        std::time::Duration::from_millis(20),
    );
    if scf_data.mol.ctrl.print_level > 1 {
        println!(
            "  Memory monitor (force phase): limit = {}",
            mem_monitor
                .limit_mb()
                .map(|m| format!("{:.1} MiB ({:.3} GiB, abort on exceed)", m, m / 1024.0))
                .unwrap_or_else(|| "NONE (peak tracking only)".to_string())
        );
    }

    let (energy, gradient) = if scf_data.mol.ctrl.numerical_force {
        if scf_data.mol.ctrl.print_level > 1 {
            println!("Gradient evaluation using numerical differentiation");
        }
        let displace = scf_data.mol.ctrl.nforce_displacement / BOHR;
        let (energy, nforce) = numerical_force(scf_data, displace, &mpi_operator);
        println!("------ Output gradient [a.u.] ------");
        println!("{}", formated_force(&nforce, &scf_data.mol.geom.elem));
        println!("------------------------------------");
        (energy, nforce)
    } else {
        // current available analytical gradients methods:
        // 1) analytical RHF, UHF force
        // 2) dftd force
        // 3) doubly-hybrid (PT2-family) correlation gradient, as a separate entry
        let is_dh = matches!(scf_data.mol.xc_data.dfa_family_pos, Some(crate::dft::DFAFamily::PT2));
        if scf_data.mol.xc_data.is_fifth_dfa() && !is_dh {
            panic!("Analytic Gradient calculation is currently not available for post-SCF methods.");
        }
        if is_dh && mpi_operator.is_some() {
            panic!("Analytic Gradient calculation of post-SCF methods is not MPI-parallelized yet.");
        }

        if scf_data.mol.ctrl.print_level > 1 {
            println!("Gradient evaluation using Analytical differentiation");
        }

        // In MPI-parallel runs, `scf_data.rimatr` (decomposed ERI / cderi) is distributed along
        // the auxiliary-basis dimension: each rank stores only its column block
        // `[n_baspar, naux_local]` (the SCF J/K build reduces partial contractions over ranks).
        // The analytic gradient reads exactly that local block: `grad::rhf::get_itm_j_columns` and
        // `get_itm_k_occtp_columns` restrict the pair-row contraction to the columns this rank
        // owns and all-gather only the small intermediates (a `[naux]` vector and a
        // `[nocc_tp, naux]` array), so the complete `[n_baspar, naux]` matrix is *not* replicated
        // on every rank any more. For (H2O)30 that replication cost 6.56 GiB per rank, which made
        // the node footprint of a force job grow linearly with the number of ranks.
        //
        // The exception is the RI-TDDFT response gradient, which reads the complete tensor
        // through the full-space pair tables and is not part of the validated MPI scope, so its
        // path still gets the gathered form here. Any other consumer of the full matrix has to
        // call `mpi_io::gather_full_rimatr` itself (it is idempotent, and returns the input when
        // the tensor already carries every column).
        let needs_full_rimatr = scf_data
            .mol
            .ctrl
            .tddft
            .as_ref()
            .map_or(false, |tddft| tddft.tddft_grad_state > 0);
        if needs_full_rimatr {
            let naux_total = scf_data.mol.num_auxbas;
            if scf_data.rimatr.is_some() {
                let (rimatr, basbas2baspar, baspar2basbas) = scf_data.rimatr.take().unwrap();
                let full_rimatr =
                    crate::mpi_io::gather_full_rimatr(&rimatr, naux_total, mpi_operator)
                        .unwrap_or(rimatr);
                scf_data.rimatr = Some((full_rimatr, basbas2baspar, baspar2basbas));
            }
            if scf_data.rimatr_sr.is_some() {
                let (rimatr_sr, basbas2baspar, baspar2basbas) = scf_data.rimatr_sr.take().unwrap();
                let full_rimatr_sr =
                    crate::mpi_io::gather_full_rimatr(&rimatr_sr, naux_total, mpi_operator)
                        .unwrap_or(rimatr_sr);
                scf_data.rimatr_sr = Some((full_rimatr_sr, basbas2baspar, baspar2basbas));
            }
        }

        let is_hf = scf_data.mol.xc_data.dfa_compnt_scf.is_empty();

        // Please note that this is only a temporary workaround implemented gradients.
        // Totally refactor the following code if necessary if other types of gradients to be implemented.
        use crate::grad::traits::GradAPI;

        // we will collect the gradient data from different components into a list,
        // and then summarize the total gradient at the end
        // list of (gradient name, gradient data)
        let mut grad_data_list: Vec<(String, Box<dyn GradAPI>)> = vec![];

        // the DH (PT2-family) entry hands the total relaxed density back for the fchk density
        // dump (`SCF::dh_rdm1_resp`, written out by the `outputs` pass)
        let mut dh_rdm1_resp: Option<Vec<MatrixFull<f64>>> = None;

        // post-SCF (doubly-hybrid) gradients need the DFT grids again whenever the DH entry runs
        // on a grid (final-functional XC part / hybrid difference, or the SCF functional's own XC
        // response); `xdh_calculations` frees them, so regenerate (pure MP2 needs none)
        let is_dh = matches!(scf_data.mol.xc_data.dfa_family_pos, Some(crate::dft::DFAFamily::PT2));
        let dh_needs_grids = {
            let xc_data = &scf_data.mol.xc_data;
            let final_xc = xc_data.dfa_compnt_pos.as_ref().map_or(false, |v| !v.is_empty());
            let delta_hyb =
                xc_data.dfa_hybrid_pos.unwrap_or(xc_data.dfa_hybrid_scf) - xc_data.dfa_hybrid_scf;
            final_xc || delta_hyb.abs() > 1.0e-10 || !xc_data.dfa_compnt_scf.is_empty()
        };
        if is_dh && dh_needs_grids && scf_data.grids.is_none() {
            scf_data.grids = Some(crate::dft::Grids::build(&mut scf_data.mol));
        }

        // 1. self-consistent gradient data
        let grad_data_scf: Box<dyn crate::grad::traits::GradAPI> = {
            if !scf_data.mol.ctrl.spin_polarization {
                let mut grad_data_scf = crate::grad::rhf::RIRHFGradient::new(&scf_data, mpi_operator);

                if is_hf {
                    grad_data_scf.calc();
                } else {
                    grad_data_scf.calc_rks();
                }

                Box::new(grad_data_scf)
            } else {
                let mut grad_data_scf = crate::grad::uhf::RIUHFGradient::new(&scf_data, mpi_operator);

                if is_hf {
                    grad_data_scf.calc();
                } else {
                    grad_data_scf.calc_uks();
                }
                
                Box::new(grad_data_scf)
            }
        };
        grad_data_list.push(("SCF".into(), grad_data_scf));

        // 1.5 doubly-hybrid (RI-PT2) correlation gradient; requires the decomposed rimatr
        if is_dh {
            if !scf_data.mol.ctrl.spin_polarization {
                let mut grad_data_dh = crate::grad::rdh::RDHGradient::new(&scf_data, mpi_operator);
                grad_data_dh.calc();
                dh_rdm1_resp = grad_data_dh.rdm1_dump.take();
                grad_data_list.push(("DH".into(), Box::new(grad_data_dh)));
            } else {
                let mut grad_data_dh = crate::grad::udh::UDHGradient::new(&scf_data, mpi_operator);
                grad_data_dh.calc();
                dh_rdm1_resp = grad_data_dh.rdm1_dump.take();
                grad_data_list.push(("DH".into(), Box::new(grad_data_dh)));
            }
        }

        // 2. dftd gradient data
        //    we will force to evaluate dftd gradient, since dftd3 is not bottleneck for small to medium molecules
        use crate::dftd::grad::DFTDGrad;
        let mut grad_data_dftd = DFTDGrad::new(&scf_data);
        grad_data_dftd.make_grad();
        // only append the dftd gradient if dftd really exists
        if grad_data_dftd.result.is_some() {
            grad_data_list.push(("DFTD".into(), Box::new(grad_data_dftd)));
        }

        // summarize the gradient contributions by different components
        let natm = scf_data.mol.geom.elem.len();
        let mut gradient = MatrixFull::new([3, natm], 0.0);
        for (grad_name, grad_data) in grad_data_list.iter() {
            let grad_contrib = grad_data.get_gradient();
            gradient.data.iter_mut().zip(grad_contrib.data.iter()).for_each(|(to, from)| {*to += from});
            if scf_data.mol.ctrl.print_level > 1 {
                println!("Gradient contribution from {:} [a.u.]:", grad_name);
                println!("{}", formated_force(&grad_contrib, &scf_data.mol.geom.elem));
            }
        }

        // the gradient objects borrow `scf_data`; release them before handing the DH relaxed
        // density to the fchk output pass — force jobs only
        drop(grad_data_list);
        if is_dh && matches!(scf_data.mol.ctrl.job_type, JobType::Force) {
            scf_data.dh_rdm1_resp = dh_rdm1_resp;
        }

        // TDDFT analytic-gradient response for the requested excited state.
        if let Some(tddft_ctrl) = scf_data.mol.ctrl.tddft.clone() {
            if tddft_ctrl.tddft_grad_state > 0 {
                let state = tddft_ctrl.tddft_grad_state;
                if scf_data.mol.spin_channel == 2 {
                    panic!("TDDFT analytic gradient currently supports only spin-restricted references (spin_polarization = false)");
                }
                let exc = scf_data
                    .tddft_excitations
                    .as_ref()
                    .expect("tddft_grad_state > 0 requires a TDDFT calculation to have run first");
                assert!(
                    state <= exc.len(),
                    "tddft_grad_state = {} exceeds the {} computed TDDFT states",
                    state,
                    exc.len()
                );
                let raw = &exc[state - 1].1;
                let tda = tddft_ctrl.tddft_method.eq_ignore_ascii_case("tda");
                let singlet = tddft_ctrl.restricted_spin() != "triplet";
                let norm = crate::ri_bse::dipoles::normalize(raw, tda);
                let (x, y) = if tda {
                    (norm, vec![0.0; raw.len()])
                } else {
                    let dim = raw.len() / 2;
                    (norm[..dim].to_vec(), norm[dim..].to_vec())
                };
                let engine = crate::ri_tddft::TddftGradEngine::new(
                    &scf_data, state, singlet, tda, x, y,
                );
                let response = engine.response_gradient();
                gradient
                    .data
                    .iter_mut()
                    .zip(response.data.iter())
                    .for_each(|(to, from)| *to += from);
                println!(
                    "Gradient contribution from TDDFT state {} [a.u.]:",
                    state
                );
                println!("{}", formated_force(&response, &scf_data.mol.geom.elem));
            }
        }

        println!("------ Output gradient [a.u.] ------");
        println!("{}", formated_force(&gradient, &scf_data.mol.geom.elem));
        println!("------------------------------------");

        // the total energy (fifth-DFA/rpa/ai correction included), consistent with the gradient
        (collect_total_energy(scf_data), gradient)
    };

    // 力阶段峰值（监控线程 20 ms 一次采样），随后关掉监控线程
    let force_peak_mb = mem_monitor.stage_peak_mb();
    if scf_data.mol.ctrl.print_level > 0 {
        println!(
            "  [mem] force phase peak RSS = {:.1} MiB ({:.3} GiB)",
            force_peak_mb,
            force_peak_mb / 1024.0
        );
    }
    mem_monitor.stop();

    time_mark.count("force");
    time_mark.report("force");

    (energy, gradient)
}

pub(crate) fn eval_force_with_position(scf_data: &mut SCF, time_mark: &mut utilities::TimeRecords, mpi_operator: &Option<MPIOperator>, position: &MatrixFull<f64>) -> (f64, MatrixFull<f64>) {
    scf_data.mol.geom.geom_update(&position.data(), GeomUnit::Bohr);
    if scf_data.mol.ctrl.print_level>0 {
        println!("Input geometry in this round is:");
        println!("{}", scf_data.mol.geom.formated_geometry());
    }
    scf_data.free_large_tensors();
    scf_data.mol.ctrl.initial_guess = String::from("inherit");
    initialize_scf(scf_data, mpi_operator);
    performance_essential_calculations(scf_data, time_mark, mpi_operator);
    let (energy, gradient) = eval_force(scf_data, time_mark, mpi_operator);
    return (energy, gradient);
}

// UNVERIFIED NORMAL MODES CALCULATION
fn eval_normal_modes(
    scf_data: &mut SCF,
    time_mark: &mut utilities::TimeRecords,
    mpi_operator: &Option<MPIOperator>,
) {
    if scf_data.mol.xc_data.is_fifth_dfa() {
        panic!("Normal modes calculation is currently not available for post-SCF methods.");
    }

    let num_atoms = scf_data.mol.geom.nfree;
    let dim = num_atoms * 3;
    let displace_ang = scf_data.mol.ctrl.nhessian_displacement;
    let displace = displace_ang / BOHR; // convert Angstrom to Bohr

    if scf_data.mol.ctrl.print_level > 0 {
        println!("");
        println!("=========================================================");
        println!("      Vibrational Normal Modes (Frequency) Calculation");
        println!("=========================================================");
        println!("Number of atoms:               {}", num_atoms);
        println!("Hessian dimension:             {}", dim);
        println!("Finite-difference displacement: {:.6} Bohr ({:.6} Ang)", displace, displace_ang);
        println!("Number of displaced SCF jobs:  {}", dim * 2);
        println!("---------------------------------------------------------");
    }

    // Build Hessian via central finite difference of analytical gradients
    let mut hessian = MatrixFull::new([dim, dim], 0.0);

    // Every displaced job calls `initialize_scf`, which rebuilds the three-center tensors and the
    // density grids from the displaced geometry. Cloning the SCF with those tensors resident would
    // therefore copy them 6N times only to overwrite them immediately, and would hold a second
    // copy of them for the whole loop: on a large system that is GBs per displacement. Park them
    // for the duration of the loop and hand them back afterwards, so `scf_data.clone()` below
    // copies only the cheap state (geometry, control, density, orbitals).
    let parked = (
        scf_data.ijkl.take(),
        scf_data.ri3fn.take(),
        scf_data.ri3fn_sr.take(),
        scf_data.ri3fn_isdf.take(),
        scf_data.ri3fn_bse.take(),
        scf_data.rimatr.take(),
        scf_data.rimatr_sr.take(),
        scf_data.rimatr_bse.take(),
        scf_data.ri3mo.take(),
        scf_data.ri3mo_full.take(),
        scf_data.tab_ao.take(),
        scf_data.m.take(),
        scf_data.grids.take(),
    );

    if scf_data.mol.ctrl.print_level > 0 {
        print!("  Hessian finite difference progress: ");
        io::stdout().flush().unwrap();
    }

    for atm_idx in 0..num_atoms {
        for xyz in 0..3 {
            let col_idx = atm_idx * 3 + xyz;

            // +δ displacement
            let mut vec_plus = vec![0.0; 3];
            vec_plus[xyz] = displace;
            let mut scf_plus = scf_data.clone();
            scf_plus.mol.geom.geom_shift(atm_idx, vec_plus);
            scf_plus.mol.ctrl.print_level = 0;
            scf_plus.mol.ctrl.initial_guess = String::from("inherit");
            initialize_scf(&mut scf_plus, mpi_operator);
            let _ = performance_essential_calculations(&mut scf_plus, time_mark, mpi_operator);
            let (_, force_plus) = eval_force(&mut scf_plus, time_mark, mpi_operator);

            // -δ displacement
            let mut vec_minus = vec![0.0; 3];
            vec_minus[xyz] = -displace;
            let mut scf_minus = scf_data.clone();
            scf_minus.mol.geom.geom_shift(atm_idx, vec_minus);
            scf_minus.mol.ctrl.print_level = 0;
            scf_minus.mol.ctrl.initial_guess = String::from("inherit");
            initialize_scf(&mut scf_minus, mpi_operator);
            let _ = performance_essential_calculations(&mut scf_minus, time_mark, mpi_operator);
            let (_, force_minus) = eval_force(&mut scf_minus, time_mark, mpi_operator);

            // Hessian column = (F(+) - F(-)) / (2δ)
            // force_plus/minus are [3, num_atoms], we flatten to 3*num_atoms vector
            for a in 0..num_atoms {
                for d in 0..3 {
                    let row_idx = a * 3 + d;
                    hessian[(row_idx, col_idx)] = (force_plus[(d, a)] - force_minus[(d, a)]) / (2.0 * displace);
                }
            }

            if scf_data.mol.ctrl.print_level > 0 {
                print!(".");
                io::stdout().flush().unwrap();
            }
        }
    }

    if scf_data.mol.ctrl.print_level > 0 {
        println!(" done");
    }

    // hand the parked tensors back, so the caller sees the SCF it passed in
    (
        scf_data.ijkl,
        scf_data.ri3fn,
        scf_data.ri3fn_sr,
        scf_data.ri3fn_isdf,
        scf_data.ri3fn_bse,
        scf_data.rimatr,
        scf_data.rimatr_sr,
        scf_data.rimatr_bse,
        scf_data.ri3mo,
        scf_data.ri3mo_full,
        scf_data.tab_ao,
        scf_data.m,
        scf_data.grids,
    ) = parked;

    // Symmetrize Hessian: H = (H + H^T) / 2
    for i in 0..dim {
        for j in 0..i {
            let avg = 0.5 * (hessian[(i, j)] + hessian[(j, i)]);
            hessian[(i, j)] = avg;
            hessian[(j, i)] = avg;
        }
    }

    // Get atomic masses in amu
    let mass_charge = get_mass_charge(&scf_data.mol.geom.elem);

    // Mass-weight Hessian: H̃_ij = H_ij / sqrt(m_i * m_j)
    let mut mw_hessian = MatrixFull::new([dim, dim], 0.0);
    for i in 0..dim {
        let atom_i = i / 3;
        let mass_i = mass_charge[atom_i].0; // mass in amu
        for j in 0..=i {
            let atom_j = j / 3;
            let mass_j = mass_charge[atom_j].0;
            let mw_val = hessian[(i, j)] / (mass_i * mass_j).sqrt();
            mw_hessian[(i, j)] = mw_val;
            mw_hessian[(j, i)] = mw_val;
        }
    }

    // Diagonalize mass-weighted Hessian
    let (eigenvectors, eigenvalues, info) = _dsyevd(&mw_hessian, 'V');
    if info != 0 {
        panic!("DSYEVD (diagonalization of mass-weighted Hessian) failed with info = {}", info);
    }

    let eigvec = eigenvectors.unwrap();

    // Conversion factor: sqrt(λ / (amu)) to cm⁻¹
    // λ is eigenvalue of mass-weighted Hessian in Hartree/(Bohr²·amu)
    // ν̃ (cm⁻¹) = sign(λ) × sqrt(|λ|) × 5140.487
    const FREQ_CONV: f64 = 5140.487;

    // Print frequencies sorted by mode number (already sorted by DSYEVD)
    if scf_data.mol.ctrl.print_level > 0 {
        println!("");
        println!("=========================================================");
        println!("      Vibrational Frequencies");
        println!("=========================================================");
        println!("{:>7} {:>22} {:>22}", "Mode", "Eigenvalue", "Freq (cm⁻¹)");
        println!("---------------------------------------------------------");

        for i in 0..dim {
            let lambda = eigenvalues[i];
            let freq_sign = if lambda < 0.0 { -1.0 } else { 1.0 };
            let freq = freq_sign * lambda.abs().sqrt() * FREQ_CONV;
            let lambda_str = if lambda >= 0.0 {
                format!("{:22.8}", lambda)
            } else {
                format!("{:22.8}", lambda)
            };
            println!("{:7} {:>22} {:>22.4}", i + 1, lambda_str, freq);
        }
        println!("=========================================================");

        // Identify near-zero modes (translations + rotations)
        println!("");
        println!("Near-zero modes (translations/rotations):");
        let threshold = 100.0; // cm⁻¹
        for i in 0..dim {
            let lambda = eigenvalues[i];
            let freq_sign = if lambda < 0.0 { -1.0 } else { 1.0 };
            let freq = freq_sign * lambda.abs().sqrt() * FREQ_CONV;
            if freq.abs() < threshold {
                println!("  Mode {:4}: {:12.4} cm⁻¹", i + 1, freq);
            }
        }
    }

    // Print Cartesian normal mode displacements at higher print levels
    if scf_data.mol.ctrl.print_level > 1 {
        println!("");
        println!("Cartesian normal mode displacements:");
        for i in 0..dim {
            let lambda = eigenvalues[i];
            let freq_sign = if lambda < 0.0 { -1.0 } else { 1.0 };
            let freq = freq_sign * lambda.abs().sqrt() * FREQ_CONV;
            println!("");
            println!("Mode {:4} (ω = {:12.4} cm⁻¹):", i + 1, freq);
            println!("{:>6} {:>14} {:>14} {:>14}", "Atom", "dX", "dY", "dZ");
            for a in 0..num_atoms {
                let dx = eigvec[(a * 3,     i)];
                let dy = eigvec[(a * 3 + 1, i)];
                let dz = eigvec[(a * 3 + 2, i)];
                println!("{:>6} {:14.8} {:14.8} {:14.8}",
                    scf_data.mol.geom.elem[a], dx, dy, dz);
            }
        }
    }
}
//------------------

#[cfg(feature = "geometric-pyo3")]
mod geometric_pyo3_impl {
    use super::*;
    use geometric_pyo3::prelude::*;
    use pyo3::prelude::*;

    /// `MPIOperator` is not `Sync` (the `mpi` crate's `SimpleCommunicator`
    /// wraps a raw `MPI_Comm` pointer), so a plain `&Option<MPIOperator>` can
    /// not satisfy the `Send` bound required by `GeomDriverAPI`. The reference
    /// is only ever dereferenced inside `GeometricOptDriver::calc_new`, which
    /// geomeTRIC invokes single-threaded on the master rank while
    /// `optimize_geometric_pyo3` is still on the stack, so the pointee always
    /// outlives every use. This makes the `unsafe impl Send` sound. (Without
    /// the `mpi` feature, `MPIOperator` is a plain-data stub that is `Sync`,
    /// so the `unsafe impl` is not needed.)
    struct MpiOptRef<'a>(&'a Option<MPIOperator>);
    #[cfg(feature = "mpi")]
    unsafe impl Send for MpiOptRef<'_> {}

    pub(crate) struct GeometricOptDriver<'a> {
        scf_data: &'a mut SCF,
        time_mark: &'a mut utilities::TimeRecords,
        mpi_operator: MpiOptRef<'a>,
    }

    impl GeomDriverAPI for GeometricOptDriver<'_> {
        fn calc_new(&mut self, coords: &[f64], _dirname: &str) -> GradOutput {
            let coords = coords.to_vec();

            // Under MPI, the master rank (rank 0) first broadcasts the new
            // coordinates to every other rank, which are blocked in
            // `mpi_opt_slave_loop` waiting for them. Then ALL ranks (master
            // included) enter the collective energy/gradient evaluation, so
            // that each optimization step is MPI-parallelized and every rank
            // keeps an identical SCF state.
            #[cfg(feature = "mpi")]
            if let Some(mpi_op) = &*self.mpi_operator.0 {
                let mut coords_buf = coords.clone();
                crate::mpi_io::mpi_broadcast_vector(&mpi_op.world, &mut coords_buf, 0);
            }

            let coords = MatrixFull::from_vec([3, coords.len()/3], coords).unwrap();
            let mpi_operator = self.mpi_operator.0;
            let (scf_data, time_mark) = (&mut self.scf_data, &mut self.time_mark);
            let (energy, gradient) = eval_force_with_position(scf_data, time_mark, mpi_operator, &coords);
            //gradient.formated_output(3, "full");
            //gradient *= -1.0;
            let gradient = gradient.data();
            return GradOutput {
                energy,
                gradient,
            }
        }
    }

    /// Non-root ranks: wait for the master's per-step coordinates, evaluate the
    /// energy and the gradient collectively (the very same code path as
    /// `GeometricOptDriver::calc_new` on the master), and loop until the master
    /// signals the end of the optimization with an empty coordinate vector.
    #[cfg(feature = "mpi")]
    fn mpi_opt_slave_loop(
        scf_data: &mut SCF,
        time_mark: &mut utilities::TimeRecords,
        mpi_operator: &Option<MPIOperator>,
    ) {
        let mpi_op = mpi_operator.as_ref().expect("mpi_opt_slave_loop requires an MPI operator");
        let world = &mpi_op.world;
        loop {
            let mut coords_buf: Vec<f64> = Vec::new();
            crate::mpi_io::mpi_broadcast_vector(world, &mut coords_buf, 0);
            if coords_buf.is_empty() {
                // The master has finished (or aborted) the optimization.
                break;
            }
            let coords = MatrixFull::from_vec([3, coords_buf.len() / 3], coords_buf).unwrap();
            let _ = eval_force_with_position(scf_data, time_mark, mpi_operator, &coords);
        }
    }

    pub(crate) fn optimize_geometric_pyo3(scf_data: &mut SCF, time_mark: &mut utilities::TimeRecords, mpi_operator: &Option<MPIOperator>) -> PyResult<(f64, MatrixFull<f64>)> {
        // Under MPI, only the master rank (rank 0) drives the geomeTRIC Python
        // optimization loop; every other rank joins each energy/gradient
        // evaluation through MPI (see `mpi_opt_slave_loop` and
        // `GeometricOptDriver::calc_new`). Without this split, every rank would
        // run its own full geomeTRIC instance, redundantly redo all the SCF
        // work, and write its own GeomeTRIC_<n>.log / GeomeTRIC.tmp /
        // GeomeTRIC_optim.xyz files (racing on the shared ones).
        #[cfg(feature = "mpi")]
        if let Some(mpi_op) = mpi_operator {
            if mpi_op.rank != 0 {
                mpi_opt_slave_loop(scf_data, time_mark, mpi_operator);
                let coords = scf_data.mol.geom.position.clone();
                return Ok((scf_data.scf_energy, coords));
            }
        }

        pyo3::prepare_freethreaded_python();
        
        let elem = scf_data.mol.geom.elem.iter().map(|x| x.as_str()).collect::<Vec<&str>>();
        let xyz = scf_data.mol.geom.position.data.iter().map(|x| x * BOHR).collect::<Vec<f64>>();
        //let xyz = scf_data.mol.geom.position.iter().map(|x| *x).collect::<Vec<f64>>();
        let xyzs = vec![xyz];
        let molecule = init_pyo3_molecule(&elem, &xyzs).unwrap();
        // Build topology. Fac / radii control bond detection and must be written into
        // molecule.top_settings (read by build_bonds), not passed as kwargs, so they also
        // apply to the rebuild that geomeTRIC does inside makePrimitives.
        //   fac   : multiplicative factor to covalent radii (default 1.2).
        //   radii : per-element radius overrides, e.g. [("Sr", 0.0)] to make Sr non-bonding.
        let geo_params = scf_data.mol.ctrl.geometric_pyo3.as_ref();
        let fac = geo_params.and_then(|g| g.fac);
        let radii = geo_params.and_then(|g| g.radii.as_ref());
        Python::with_gil(|py| -> PyResult<()> {
            let ts = molecule.getattr(py, "top_settings")?;
            if let Some(f) = fac {
                ts.call_method1(py, "__setitem__", ("Fac", f))?;
            }
            if let Some(ref pairs) = radii {
                let radii_dict = pyo3::types::PyDict::new(py);
                for (elem, r) in pairs.iter() {
                    radii_dict.set_item(elem, r)?;
                }
                ts.call_method1(py, "__setitem__", ("radii", radii_dict))?;
            }
            molecule.call_method(py, "build_topology", (), None)?;
            Ok(())
        }).unwrap();
        
        //let optimizer_params = r#"
        //    convergence_energy   = 1.0e-6  # Eh
        //    convergence_grms     = 3.0e-4  # Eh/Bohr
        //    convergence_gmax     = 4.5e-4  # Eh/Bohr
        //    convergence_drms     = 1.2e-3  # Angstrom
        //    convergence_dmax     = 1.8e-3  # Angstrom
        //"#;
        //let mut params = tomlstr2py(optimizer_params)?;

        let rust_params = scf_data.mol.ctrl.geometric_pyo3.as_ref().expect("For geometric_pyo3, you must specify the parameters in the control file.");
        let params = toml2py(&rust_params.to_toml())?;

        // ── Compute analytical Hessian before optimization (if requested) ──
        let hessian_analytic_path: Option<std::path::PathBuf> =
            if rust_params.analytic_hessian {
                let pl = scf_data.mol.ctrl.print_level;
                if pl > 0 {
                    println!("  Computing analytical Hessian for geomeTRIC initial guess...");
                }
                let is_hessian_enabled = scf_data.mol.ctrl.hessian.is_some();
                let use_analdrv = scf_data.mol.ctrl.geometric_pyo3.as_ref().map_or(None, |g| g.use_analdrv);
                // if `[hessian]` is specified in the control file, then we will use it to compute the Hessian.
                // otherwise, use `analdrv`.
                let use_analdrv = use_analdrv.unwrap_or(!is_hessian_enabled);
                let hess_total = if !use_analdrv {
                    crate::hessian::compute_hessian(&*scf_data)
                        .expect("Analytical Hessian computation failed for geometry optimization")
                } else {
                    use crate::analdrv::hessian::hess_interface;
                    use crate::analdrv::response::rresp_interface::rscf_resp_interface;
                    use crate::analdrv::response::uresp_interface::uscf_resp_interface;
                    use crate::analdrv::response::RespSCF;
                    use rstsr::prelude::*;

                    let config = scf_data.mol.ctrl.analdrv.clone().unwrap_or_default();
                    // post-SCF (fifth-DFA) methods have no analytic hessian; reject before any
                    // response object is built (the DFT grids it would need are already freed)
                    if scf_data.mol.xc_data.is_fifth_dfa() {
                        panic!("Normal modes calculation is currently not available for post-SCF methods.");
                    }
                    // shared response object for the hessian, per the SCF type
                    let mut resp_objs = match scf_data.scftype {
                        crate::scf_io::SCFType::RHF => Some(RespSCF::R(rscf_resp_interface(&scf_data, &config))),
                        crate::scf_io::SCFType::UHF => Some(RespSCF::U(uscf_resp_interface(&scf_data, &config))),
                        _ => None,
                    };
                    let hess_out = hess_interface(&scf_data, &config, resp_objs.as_mut());
                    let hess_raw = hess_out.hessian;
                    let natm = (hess_raw.len() / 9).isqrt();
                    assert!(natm * natm * 9 == hess_raw.len(), "Hessian raw data length does not match expected size for {} atoms", natm);
                    let device = DeviceBLAS::default();
                    let hess_rt = rt::asarray((&hess_raw, [3, 3, natm, natm].f(), &device));
                    let hess_rt = hess_rt.transpose([0, 2, 1, 3]).into_shape((3 * natm, 3 * natm));
                    let hess_vec = hess_rt.into_shape(-1).into_vec();
                    MatrixFull::from_vec([3 * natm, 3 * natm], hess_vec).unwrap()
                };
                let natm = scf_data.mol.geom.nfree;
                let n3 = natm * 3;
                let mut out = String::new();
                for i in 0..n3 {
                    let mut row = String::new();
                    for j in 0..n3 {
                        use std::fmt::Write;
                        let _ = write!(&mut row, " {:17.10e}", hess_total[[i, j]]);
                    }
                    out.push_str(row.trim_start());
                    out.push('\n');
                }
                let hess_path = std::env::temp_dir()
                    .join(format!("rest_hess_analytic_{}.txt", std::process::id()));
                std::fs::write(&hess_path, &out)
                    .expect("Failed to write analytical Hessian file");
                if pl > 0 {
                    println!("  Analytical Hessian saved to {}", hess_path.display());
                }
                Some(hess_path)
            } else {
                None
            };

        let constraint_path = if let Some(constraint_str) = scf_data.mol.geom.to_geometric_freeze_str() {
            if scf_data.mol.ctrl.print_level > 0 {
                println!("Fixed atoms detected, generating constraint file.");
            }
            let dir = std::env::temp_dir();
            let path = dir.join(format!("rest_constraints_{}.txt", std::process::id()));
            std::fs::write(&path, &constraint_str).expect("Failed to write constraint file");
            Some(path)
        } else {
            None
        };

        let input = None;

        let pyo3_engine_cls = get_pyo3_engine_cls()?;
        let geometric_opt_driver = GeometricOptDriver {
            scf_data,
            time_mark,
            mpi_operator: MpiOptRef(mpi_operator),
        };
        let driver: PyGeomDriver = geometric_opt_driver.into();

        
        let py_result = Python::with_gil(|py| -> PyResult<(f64, Vec<f64>)> {
            let custom_engine = pyo3_engine_cls.call1(py, (molecule,))?;
            custom_engine.call_method1(py, "set_driver", (driver,))?;

            if let Some(ref cst_path) = constraint_path {
                params.bind(py).set_item("constraints", cst_path.to_str().unwrap())?;
            }

            // Inject analytical Hessian file path into geomeTRIC params
            if let Some(ref hess_path) = hessian_analytic_path {
                params.bind(py).set_item("hessian", format!("file:{}", hess_path.display()))?;
            }

            let res = run_optimization(custom_engine, &params, input)?;

            let last_energy = res
                .getattr(py, "qm_energies")?
                .call_method1(py, "__getitem__", (-1,))?
                .extract::<f64>(py)?;

            let last_coords = res
                .getattr(py, "xyzs")?
                .call_method1(py, "__getitem__", (-1,))?
                .call_method0(py, "flatten")?
                .call_method0(py, "tolist")?
                .extract::<Vec<f64>>(py)?;

            Ok((last_energy, last_coords))
        });

        // Signal the slave ranks that the optimization has finished, so that
        // they leave `mpi_opt_slave_loop`. This must also be sent when the
        // optimization failed, otherwise the slaves would block forever.
        #[cfg(feature = "mpi")]
        if let Some(mpi_op) = mpi_operator {
            if mpi_op.rank == 0 {
                let mut done: Vec<f64> = Vec::new();
                crate::mpi_io::mpi_broadcast_vector(&mpi_op.world, &mut done, 0);
            }
        }

        if let Some(ref path) = constraint_path {
            let _ = std::fs::remove_file(path);
        }
        if let Some(ref path) = hessian_analytic_path {
            let _ = std::fs::remove_file(path);
        }

        let (last_energy, last_coords) = py_result?;

        let last_coords = MatrixFull::from_vec([3, last_coords.len()/3], last_coords).unwrap();

        return Ok((last_energy, last_coords));
    }
}

/* #endregion */
