//! # Rust-based Electronic-Structure Tool (REST)
//! 
//! ![image](/home/igor/Documents/Package-Pool/rest_workspace/rest/figures/REST电子结构程序1.png) 
//! ![image](/home/igor/Documents/Package-Pool/rest_workspace/rest/figures/REST电子结构程序2.png) 
//! ![image](/home/igor/Documents/Package-Pool/rest_workspace/rest/figures/REST电子结构程序3.png) 
//! ![image](/home/igor/Documents/Package-Pool/rest_workspace/rest/figures/REST电子结构程序4.png) 
//! ![image](/home/igor/Documents/Package-Pool/rest_workspace/rest/figures/REST电子结构程序5-2.png) 
//! ![image](/home/igor/Documents/Package-Pool/rest_workspace/rest/figures/REST电子结构程序6-2.png) 
//! 
#![allow(unused)]
extern crate rest_tensors as tensors;
extern crate chrono as time;
#[macro_use]
extern crate lazy_static;
use std::{f64, fs::File, io::Write};
use std::path::PathBuf;

use anyhow;
use clap::{Command, Arg, ArgMatches};
use time::{DateTime,Local};
mod geom_io;
mod basis_io;
mod ctrl_io;
mod dft;
mod utilities;
mod molecule_io;
mod scf_io;
mod initial_guess;
mod ri_pt2;
mod ri_rpa;
mod isdf;
mod constants;
pub mod post_scf_analysis;
//use rayon;
#[global_allocator]
static GLOBAL: jemallocator::Jemalloc = jemallocator::Jemalloc;
//static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

pub use crate::initial_guess::sap::*;
use crate::{post_scf_analysis::{rand_wf_real_space, cube_build, molden_build}, isdf::error_isdf, molecule_io::Molecule};

fn main() -> anyhow::Result<()> {
    let mut time_mark = utilities::TimeRecords::new();
    time_mark.new_item("Overall", "the whole job");
    time_mark.count_start("Overall");

    time_mark.new_item("SCF", "the scf procedure");
    time_mark.count_start("SCF");

    let ctrl_file = parse_input().value_of("input_file").unwrap_or("ctrl.in").to_string();
    if ! PathBuf::from(ctrl_file.clone()).is_file() {
        panic!("Input file ({:}) does not exist", ctrl_file);
    }


    let mut mol = Molecule::build(ctrl_file)?;

    let mut scf_data = scf_io::scf(mol).unwrap();

    time_mark.count("SCF");

    if scf_data.mol.ctrl.check_stab {
        time_mark.new_item("Stability", "the scf stability check");
        time_mark.count_start("Stability");

        scf_data.stability();

        time_mark.count("Stability");
    }

    //====================================
    // Now for post-SCF analysis
    //====================================
    if scf_data.mol.ctrl.fciqmc_dump {
        fciqmc_dump(&scf_data);
    }
    if scf_data.mol.ctrl.wfn_in_real_space>0 {
        let np = scf_data.mol.ctrl.wfn_in_real_space;
        let slater_determinant = rand_wf_real_space::slater_determinant(&scf_data, np);
        let output = serde_json::to_string(&slater_determinant).unwrap();
        let mut file = File::create("./wf_in_real_space.txt")?;
        file.write(output.as_bytes());
    }

    if scf_data.mol.ctrl.cube == true {
        let cube_file = cube_build::get_cube(&scf_data,80);
    }

    if scf_data.mol.ctrl.molden == true {
        let molden_file = molden_build::gen_molden(&scf_data);
    }

    /* let error_isdf = error_isdf(3..4, &scf_data);
    println!("k_mu:{:?}, abs_error: {:?}, rel_error: {:?}", error_isdf.0, error_isdf.1, error_isdf.2); */


    if let Some(dft_method) = &scf_data.mol.xc_data.dfa_family_pos {
        match dft_method {
            dft::DFAFamily::XDH => {
                time_mark.new_item("PT2", "the PT2 evaluation");
                time_mark.count_start("PT2");
                ri_pt2::xdh_calculations(&mut scf_data);
                time_mark.count("PT2");
            },
            dft::DFAFamily::RPA => {
                time_mark.new_item("RPA", "the RPA evaluation");
                time_mark.count_start("RPA");
                ri_rpa::rpa_calculations(&mut scf_data);
                time_mark.count("RPA");
            }
            _ => {}
        }
    }

    time_mark.count("Overall");

    println!("");
    println!("====================================================");
    println!("              REST: Mission accomplished");
    println!("====================================================");

    time_mark.report_all();

    Ok(())
}


fn parse_input() -> ArgMatches {
    Command::new("fdqc")
        .version("0.1")
        .author("Igor Ying Zhang <igor_zhangying@fudan.edu.cn>")
        .about("Rust-based Electronic-Structure Tool (REST)")
        .arg(Arg::new("input_file")
             .short('i')
             .long("input-file")
             .value_name("input_file")
             .help("Input file including \"ctrl\" and \"geom\" block, in the format of either \"json\" or \"toml\"")
             .takes_value(true))
        .get_matches()
}

fn fciqmc_dump(scf_data: &scf_io::SCF) {
    if let Some(ri3fn) = &scf_data.ri3fn {
        // prepare RI-V three-center coefficients for HF orbitals
        for i_spin in 0..scf_data.mol.spin_channel {
            let ri3mo = ri3fn.ao2mo_v01(&scf_data.eigenvectors[i_spin]).unwrap();
            for i in 0.. scf_data.mol.num_state {
                let ri3mo_i = ri3mo.get_reducing_matrix(i).unwrap();
                for j in 0.. scf_data.mol.num_state {
                    let ri3mo_ij = ri3mo_i.get_slice_x(j);
                    for k in 0.. scf_data.mol.num_state {
                        let ri3mo_k = ri3mo.get_reducing_matrix(k).unwrap();
                        for l in 0.. scf_data.mol.num_state {
                            let ri3mo_kl = ri3mo_k.get_slice_x(l);
                            let ijkl = ri3mo_ij.iter().zip(ri3mo_kl.iter())
                                .fold(0.0, |acc, (val1, val2)| acc + val1*val2);
                            if ijkl.abs() > 1.0E-8 {
                                println! ("{:16.8} {:5} {:5} {:5} {:5}",ijkl, i,j,k,l);
                            }
                        }
                    }
                }
            }
        }
    }
}
