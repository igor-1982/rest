use tensors::MatrixFull;

use crate::constants::{E, EV};
use crate::mpi_io::MPIOperator;
use crate::scf_io::SCF;
use crate::ri_rpa::scsrpa::{evaluate_special_radius_only, evaluate_osrpa_correlation_rayon}; 
use crate::ri_pt2::sbge2::{close_shell_sbge2_detailed_rayon, open_shell_sbge2_detailed_rayon};

use libm::{erf,erfc,pow};

pub fn scc15_for_rxdh7(scf_data: &mut SCF, mpi_operator: &Option<MPIOperator>) -> f64 {
    if let (Some(mpi_op), Some(mpi_ix)) = (mpi_operator, &scf_data.mol.mpi_data) {
        panic!("The MPI implementation for SCC15 is not yet available");
    };
    let spin_channel = scf_data.mol.spin_channel;
    let xc_method = &scf_data.mol.ctrl.xc;
    let num_elec = &scf_data.mol.num_elec;

    if num_elec[0].abs()<10e-8 || (num_elec[0].abs() - 1.0)<10e-8 {
        println!("There is no electron correlation effect in the system. No strong-correlation correction is needed");
        return 0.0
    }

    let start_mo = scf_data.mol.start_mo;
    let num_occu_0 = scf_data.lumo[0];
    let num_occu_1 = if spin_channel ==1 {scf_data.lumo[0]} else {scf_data.lumo[1]};

    let mut scc = 0.0;
    // prepare energy gaps
    let energy_gap = if spin_channel == 1 {
        let homo = scf_data.homo[0];
        let lumo = scf_data.lumo[0];
        let e_homo = scf_data.eigenvalues[0][homo];
        let e_lumo = scf_data.eigenvalues[0][lumo];
        (e_lumo - e_homo)*EV
    } else {
        let homo_0 = scf_data.homo[0];
        let lumo_0 = scf_data.lumo[0];
        let e_homo_0 = scf_data.eigenvalues[0][homo_0];
        let e_lumo_0 = scf_data.eigenvalues[0][lumo_0];
        let homo_1 = scf_data.homo[1];
        let lumo_1 = scf_data.lumo[1];
        let e_homo_1 = scf_data.eigenvalues[1][homo_1];
        let e_lumo_1 = scf_data.eigenvalues[1][lumo_1];
        (e_lumo_0.min(e_lumo_1) - e_homo_0.max(e_homo_1))*EV
    };
    let special_radius = evaluate_special_radius_only(scf_data).unwrap();
    let x_max = if special_radius[0] > special_radius[1] {special_radius[0]} else {special_radius[1]};
    let x_min = if special_radius[0] < special_radius[1] {special_radius[0]} else {special_radius[1]};


    // collect the hf exchange
    let x_hf = if let Some(x_hf) =scf_data.energies.get("x_hf") {
        x_hf[0]
    } else {
        scf_data.evaluate_exact_exchange_ri_v(mpi_operator)
    };
    // collect the pbe exchange
    let dfa = crate::dft::DFA4REST::new_xc(scf_data.mol.spin_channel, scf_data.mol.ctrl.print_level);
    let post_xc_energy = if let Some(grids) = &scf_data.grids {
        dfa.post_xc_exc(&vec![String::from("gga_x_pbe")], grids, &scf_data.density_matrix, &scf_data.eigenvectors, &scf_data.occupation)
    } else {
        vec![[0.0,0.0]]
    };
    let x_pbe = post_xc_energy[0][0]+post_xc_energy[0][1];

    let dxpbe = (x_pbe-x_hf)/x_hf*100.0f64;

    let ([c_sbge2,sbge2_os,sbge2_ss],[eij_00,eij_01,eij_11]) = if spin_channel == 1 {
        close_shell_sbge2_detailed_rayon(scf_data).unwrap()
    } else {
        open_shell_sbge2_detailed_rayon(scf_data).unwrap()
    };

    scf_data.energies.insert(String::from("sbge2"), vec![c_sbge2,sbge2_os,sbge2_ss]);

    let c_scsrpa = if let Some(scsrpa_c) = scf_data.energies.get("scsrpa") {
        let os1 = scsrpa_c[1];
        let ssp = scsrpa_c[0]-scsrpa_c[1];
        os1*1.2 + ssp*0.75
    } else {
        let scsrpa_c = evaluate_osrpa_correlation_rayon(scf_data).unwrap();
        let os1 = scsrpa_c[1];
        let ssp = scsrpa_c[0]-scsrpa_c[1];
        os1*1.2 + ssp*0.75
    };

    let c_pt2 = if let Some(pt2_c) = scf_data.energies.get("pt2") {
        pt2_c[0]
    } else {
        let mut pt2_c = 0.0f64;
        for i_state in start_mo..num_occu_0 {
            for j_state  in i_state+1..num_occu_0 {
                pt2_c += eij_00[(i_state,j_state)].0;
            }
        };
        for i_state in start_mo..num_occu_1 {
            for j_state  in i_state+1..num_occu_1 {
                pt2_c += eij_11[(i_state,j_state)].0;
            }
        };
        for i_state in start_mo..num_occu_0 {
            for j_state  in start_mo..num_occu_1 {
                pt2_c += eij_01[(i_state,j_state)].0;
            }
        };

        pt2_c
    };


    let fr_vec = prepare_fr_for_scc23(&eij_00, &eij_01, &eij_11, start_mo, [num_occu_0,num_occu_1]);

    let e_scc_1 = if fr_vec.len() > 0 {

        let fr1 = fr_vec[0];
        let lnfr1 = fr1.ln();

        //println!("fr_vec: {:?}", &fr_vec);

        if scf_data.mol.ctrl.print_level >=2 {
            println!("dxpbe: {:16.8}, fr1: {:16.8}, energy_gap: {:16.8}", dxpbe, fr1, energy_gap);
            println!("x_max: {:16.8}, x_min: {:16.8}", x_max, x_min);
        }
        if scf_data.mol.ctrl.print_level>0 {
            println!("c_pt2: {:16.8} Ha, c_sbge2: {:16.8} Ha, c_scsrpa: {:16.8} Ha", c_pt2,c_sbge2, c_scsrpa);
        }

        //let screen0 = (2.3*dxpbe).exp()*fr1.ln().powf(4.0);
        let screen0 = fr1.ln().powf(2.0);
        let screen1 = 0.000005*dxpbe*(dxpbe/x_max).exp()/fr1;

        (2.5734773954673504*lnfr1.powf(3.0)/(x_max*fr1.exp())*erf(screen0)-
        0.6687902606008397*lnfr1.powf(3.0)/(x_max*fr1.powf(1.0/3.0))*erf(screen0)-
        0.00002756*dxpbe*(dxpbe/x_max).exp()/fr1*erfc(screen1))/EV
    } else {
        0.0
    };

    //let e_scc_r = 0.0f64;

    //let pref = 0.5/EV;
    //println!("debug pref: {:?}", (pref, x_max, energy_gap));

    let e_scc_r = if fr_vec.len()>=2 {
        let pref = 0.5*(x_max.ln()*energy_gap/x_max).exp()/EV;
        pref * fr_vec[1..].iter().fold(0.0f64, |e_scc_r,fr| {
        //let screen0 = (2.3*dxpbe).exp()*fr.ln().powf(4.0);
        let screen0 = fr.ln().powf(2.0);
        let pref = erf(screen0);
        let a1 = fr.ln().powf(3.0)/x_max;

        //e_scc_r + pref * (1.298042947*a1/fr.exp() - 0.340437683*a1/fr.cbrt())
        e_scc_r + pref * (2.5734773954673504*a1/fr.exp() -0.6687902606008397*a1/fr.cbrt())
    })} else {0.0};

    let e_scc_limit = if fr_vec.len() > 0 {
        let fr1 = fr_vec[0];
        let screen2 = fr1.ln().powf(4.0);
        let pref = erf(screen2)/EV;
        let s2p = c_scsrpa/c_pt2;
        let s2b = c_scsrpa/c_sbge2;
        pref*(
        -0.7206*(x_max-x_min).cbrt()/(energy_gap/x_max).exp() 
        +2.8648*(x_max-x_min)*x_min.cbrt()/x_max/energy_gap.exp() 
        +1.5173*s2p/(energy_gap+dxpbe.cbrt()).exp()
        -0.3992*((s2b+s2p)/energy_gap.exp()-(s2b-s2p)/s2b.cbrt()).abs() 
        +0.2023*energy_gap.powf(2.0)*dxpbe.cbrt()/energy_gap.exp()/x_max.sqrt())
    } else {
        0.0
    };

    //println!("debug 2, {:?}", sbge2_tot);

    if scf_data.mol.ctrl.print_level>0 {
        println!("SCC_1: {:16.8} Ha, SCC_R: {:16.8} Ha, SCC_Limit: {:16.8} Ha", e_scc_1,e_scc_r,e_scc_limit);
    }

    e_scc_1 + e_scc_r + e_scc_limit
}

fn prepare_fr_for_scc23(
    eij_00: &MatrixFull<(f64,f64)>, 
    eij_01: &MatrixFull<(f64,f64)>, 
    eij_11: &MatrixFull<(f64,f64)>,
    start_mo: usize,
    num_occ: [usize;2]    
) -> Vec<f64> {
    let num_eij_00 = if num_occ[0] > start_mo {
        (num_occ[0]-start_mo) * (num_occ[0] - start_mo- 1) / 2
    } else {
        0
    };
    let num_eij_11 = if num_occ[1] > start_mo {
        (num_occ[1]-start_mo) * (num_occ[1] - start_mo- 1) / 2 
    } else {
        0
    };
    let num_eij_01 = (num_occ[0]-start_mo) * (num_occ[1] - start_mo);
    let num_eij = num_eij_00 + num_eij_11 + num_eij_01;
    let mut fr_mat = vec![0.0; num_eij];
    //println!("{},{},{},{}",num_eij_00,num_eij_11,num_eij_01,num_eij);

    let eij_00_slice  = &mut fr_mat[0..num_eij_00];

    for i_index in start_mo..num_occ[0] {
        for j_index in i_index+1..num_occ[0] {
            //println!("{:?}", eij_00[[i_index,j_index]]);
        }
    }

    eij_00_slice.iter_mut()
    .zip(eij_00.iter().enumerate().filter(|(i,val)| {
        let c_row = i%num_occ[0];
        let c_col = i/num_occ[0];
        c_row >= start_mo && c_col >= start_mo && c_col > c_row
        //*i >= start_mo*num_occ[0]+start_mo && i/num_occ[0] > i%num_occ[0]
    }))
    .for_each(|(i,(index,val))| {
        let r = if val.0.abs() < 1e-10 {1.0} else {val.1/val.0};
        *i = (0.2602*r-0.4102*r.powf(2.0)+0.1528*r.powf(3.0))/(1.0-1.8257*r+0.8285*r.powf(2.0));
        //println!("eij_00: {:16.8},{:16.8},{:16.8}, {:16.8}", val.0,val.1,r,*i);
    });

    let eij_01_slice  = &mut fr_mat[num_eij_00..num_eij_00+num_eij_01];

    eij_01_slice.iter_mut()
    .zip(eij_01.iter().enumerate().filter(|(i,val)| {
        let c_row = i%num_occ[0];
        let c_col = i/num_occ[0];
        c_row >= start_mo && c_col >= start_mo
        //*i >= start_mo * start_mo + start_mo
    }))
    .for_each(|(i,(index, val))| {
        let r = if val.0.abs() < 1e-10 {1.0} else {val.1/val.0};
        *i = (0.2602*r-0.4102*r.powf(2.0)+0.1528*r.powf(3.0))/(1.0-1.8257*r+0.8285*r.powf(2.0));
        //println!("eij_01: {:16.8},{:16.8},{:16.8}, {:16.8}", val.0,val.1,r,*i);
    });

    let eij_11_slice  = &mut fr_mat[num_eij_00+num_eij_01..num_eij];

    eij_11_slice.iter_mut()
    .zip(eij_11.iter().enumerate().filter(|(i,val)| {
        let c_row = i%num_occ[1];
        let c_col = i/num_occ[1];
        c_row >= start_mo && c_col >= start_mo && c_col > c_row
        //*i >= start_mo * num_occ[1] + start_mo && *i/num_occ[1] > i%num_occ[1]
    }))
    .for_each(|(i,(index,val))| {
        let r = if val.0.abs() < 1e-10 {1.0} else {val.1/val.0};
        *i = (0.2602*r-0.4102*r.powf(2.0)+0.1528*r.powf(3.0))/(1.0-1.8257*r+0.8285*r.powf(2.0));
        //println!("eij_11: {:16.8},{:16.8},{:16.8}, {:16.8}", val.0,val.1,r,*i);
    });

    fr_mat.sort_by(|a,b| a.partial_cmp(b).unwrap());

    fr_mat
}
