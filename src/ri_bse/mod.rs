//use std::simd::num;
use std::ops::Range;
use crate::scf_io::SCF;
use rayon::result;
use rest_tensors::{RIFull};
use std::ops::Index;
use std::cmp;
use tensors::{matrix_blas_lapack::_dinverse, MathMatrix, MatrixFull};
use crate::ri_gw::{get_occupation_parameters, get_occ_params_per_spin, select_ri3mo_spin};
//use libc::select;
//use rest::molecule_io::Molecule;
use crate::ri_gw;
use crate::molecule_io::Molecule;
use rest_tensors::matrix::matrix_blas_lapack::{_dgeev,_dgemm_full,_newton_schulz_inverse_square_root_v02,_dsvd,_dsyevd};
use std::fs::OpenOptions;
use crate::solvers::davidson::{lr_davidson_solver, tda_davidson_solver, DavidsonConfig, generate_initial_guess};
use crate::solvers::feast;
use std::time::Instant;
use std::{fs::File, io::Write};
pub mod dipoles;
pub mod matvec;
pub mod matvec_fast;
pub mod response;
pub mod feast_solver;
pub mod nonlinbse_matvec;
pub mod nonlinbse;
pub mod matvec_trace;
pub mod dynamicbse;
pub mod pysoc_export;
pub mod bse_grad;


#[cfg(target_os = "linux")]
use libc::seccomp_notif;

pub struct BseOutput {
    pub first_excitation: Option<f64>,
    pub excitation_energies: Vec<f64>,
}

/// Report which one-particle orbital set the BSE stage is expanded in.
///
/// Everything that enters the BSE Hamiltonian is derived from
/// `scf_data.eigenvectors` at call time (RI three-centre AO2MO, dipole matrix,
/// `v`, `W`, ...), so when the renormalized-singles orbitals have been installed
/// the whole BSE stage — kernel *and* the printed/saved transition amplitudes
/// `#i -> #a`, NTOs and PySOC coefficients — lives in the RS MO basis.
fn report_orbital_representation(scf_data:&SCF){
    let rs_orbitals=scf_data.mol.ctrl.quasiparticle_methods.clone()
        .map(|qp|qp.renormalized_singles&&qp.rs_use_rs_orbitals)
        .unwrap_or(false);
    if rs_orbitals{
        println!("BSE orbital representation: renormalized-singles (RS) orbitals.");
        println!("  v, W, the dipole matrix and the transition amplitudes are all expanded in the RS MO basis");
        println!("  (C_rs = C_ks * U); the RS coefficients and eigenvalues are in rs_orbitals.dat.");
    }else{
        println!("BSE orbital representation: Kohn-Sham orbitals.");
    }
}
/// Save the BSE transition amplitudes (the eigenvector `X`, plus `Y` for the
/// non-TDA case) of every computed excitation to `bse_terms.dat`.
///
/// The header states the orbital representation the amplitudes are expressed in
/// (`renormalized-singles(RS)` when `rs_use_rs_orbitals` installed the RS
/// orbitals, `kohn-sham` otherwise).  In the RS case the accompanying
/// `rs_orbitals.dat` (written by `renormalized_singles::save_rs_orbitals`) holds
/// the rotation `U` needed to map the amplitudes back to the Kohn-Sham basis,
/// `A_KS = U_o A_RS U_vᵀ`.
///
/// Layout (`#` starts a comment line):
///
/// ```text
/// # REST BSE transition amplitudes
/// # orbital_representation: renormalized-singles(RS)
/// # spin tda occ_size vir_size nstates
/// singlet 1 5 24 3
/// # state <n> energy <E>  (X layout: index = a*occ_size + i)
/// state 0 0.3140005370402806
/// x ...
/// y ...            (non-TDA only)
/// ```
pub fn save_bse_amplitudes(excitations:&Vec<(f64,Vec<f64>)>,occ_size:usize,vir_size:usize,
                           tda:bool,spin:&str,rs_orbitals:bool,path:&str){
    use std::io::Write;
    let mut buffer=String::new();
    buffer.push_str("# REST BSE transition amplitudes\n");
    buffer.push_str(&format!("# orbital_representation: {}\n",
        if rs_orbitals{"renormalized-singles(RS)"}else{"kohn-sham"}));
    buffer.push_str(&format!("# spin tda occ_size vir_size nstates\n{} {} {} {} {}\n",
                             spin,if tda{1}else{0},occ_size,vir_size,excitations.len()));
    let join=|values:&[f64]|values.iter().map(|v|format!("{:.16e}",v)).collect::<Vec<_>>().join(" ");
    for (n,(energy,vector)) in excitations.iter().enumerate(){
        buffer.push_str(&format!("state {} {:.16e}\n",n,energy));
        if tda{
            buffer.push_str(&format!("x {}\n",join(vector)));
        }else{
            let n_pair=occ_size*vir_size;
            buffer.push_str(&format!("x {}\n",join(&vector[0..n_pair.min(vector.len())])));
            if vector.len()>n_pair{
                buffer.push_str(&format!("y {}\n",join(&vector[n_pair..])));
            }
        }
    }
    match std::fs::File::create(path){
        Ok(mut file)=>{
            if let Err(err)=file.write_all(buffer.as_bytes()){
                println!("Warning: could not write the BSE amplitudes to {}: {}",path,err);
            }else{
                println!("The BSE transition amplitudes have been written to {}",path);
            }
        },
        Err(err)=>println!("Warning: could not create {}: {}",path,err),
    }
}
pub fn bse_main(scf_data:&mut SCF) -> BseOutput {
    if scf_data.mol.spin_channel == 2 {
        return bse_main_unrestricted(scf_data);
    }
    report_orbital_representation(scf_data);
    let start=Instant::now();
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=get_occupation_parameters(scf_data,'N');
    let quasiparticle_energies=scf_data.gwqp.0.clone();
    let dipole_matrix=dipoles::compute_dipole_matrix(scf_data);
    let qp_ctrl=scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap();
    if qp_ctrl.bse_spin =="none"{
        println!("No BSE Calculations are triggered");
        return BseOutput { first_excitation: None, excitation_energies: Vec::new() };
    }else if qp_ctrl.bse_spin=="both"{
        let (mut excitations_singlets,mut excitations_triplets)=bse_both_spins(scf_data,&quasiparticle_energies);
        if qp_ctrl.pysoc {
            pysoc_export::export_pysoc_json(
                scf_data,
                &excitations_singlets,
                &excitations_triplets,
                qp_ctrl.bse_tda,
                "rest_pysoc_export.json",
                "BSE",
                start_mo, num_state, occ_size, vir_size,
            );
        }
        if qp_ctrl.bse_tda==true{
            println!("BSE Calculation Results of Both Singlets and Triplets with TDA:");
            let number=excitations_singlets.len();
            println!("First {} Singlet Excitations:",number);
            excitations_singlets[0..number].iter().enumerate().for_each(|(n,(e,vec))|{
                let v=dipoles::normalize(vec,true);
                let vec_norm: f64 = vec.iter().map(|x| x*x).sum::<f64>().sqrt();
                println!("#{} Excitation energy={}, norm={:.6}",n,e,vec_norm);
                let dipole_square=dipoles::transition_dipole_square(&dipole_matrix,&v,true);
                println!("Transition Dipole Square:{}; Oscillator Strength:{}",dipole_square,dipole_square*e*2.0/3.0);
                leading_components(&v,occ_size,vir_size,qp_ctrl.print_nto)});
            println!("The first singlet excitation obtained by BSE is {}",excitations_singlets[0].0);
            let t_number = number.min(excitations_triplets.len());
            println!("First {} Triplet Excitations:",t_number);
            excitations_triplets[0..t_number].iter().enumerate().for_each(|(n,(e,vec))|{
                let v=dipoles::normalize(vec,true);
                let vec_norm: f64 = vec.iter().map(|x| x*x).sum::<f64>().sqrt();
                println!("#{} Excitation energy={}, norm={:.6}",n,e,vec_norm);
                let dipole_square=dipoles::transition_dipole_square(&dipole_matrix,&v,true);
                println!("Transition Dipole Square:{}; Oscillator Strength:{}",dipole_square,dipole_square*e*2.0/3.0);
                leading_components(&v,occ_size,vir_size,qp_ctrl.print_nto)});
            println!("The first triplet excitation obtained by BSE is {}",excitations_triplets[0].0);
        }else{
            println!("BSE Calculation Results of Both Singlets and Triplets without TDA:");
            let number=excitations_singlets.len();
            println!("First {} Singlet Excitations:",number);
            excitations_singlets[0..number].iter().enumerate().for_each(|(n,(e,vec))|{
                let vec_norm: f64 = vec.iter().map(|x| x*x).sum::<f64>().sqrt();
                println!("#{} Excitation energy={}, norm={:.6}",n,e,vec_norm);
                let v=dipoles::normalize(vec,false);
                let dipole_square=dipoles::transition_dipole_square(&dipole_matrix,&v,false);
                println!("Transition Dipole Square:{}; Oscillator Strength:{}",dipole_square,dipole_square*e*2.0/3.0);
                leading_components(&v,occ_size,vir_size,qp_ctrl.print_nto)
            });
            println!("The first singlet excitation obtained by BSE is {}",excitations_singlets[0].0);
            let t_number = number.min(excitations_triplets.len());
            println!("First {} Triplet Excitations:",t_number);
            excitations_triplets[0..t_number].iter().enumerate().for_each(|(n,(e,vec))|{
                let vec_norm: f64 = vec.iter().map(|x| x*x).sum::<f64>().sqrt();
                println!("#{} Excitation energy={}, norm={:.6}",n,e,vec_norm);
                let v=dipoles::normalize(vec,false);
                let dipole_square=dipoles::transition_dipole_square(&dipole_matrix,&v,false);
                println!("Transition Dipole Square:{}; Oscillator Strength:{}",dipole_square,dipole_square*e*2.0/3.0);
                leading_components(&v,occ_size,vir_size,qp_ctrl.print_nto)
            });
            println!("The first triplet excitation obtained by BSE is {}",excitations_triplets[0].0);
        }
        return BseOutput { first_excitation: Some(excitations_singlets[0].0), excitation_energies: Vec::new() };
    }else{
        println!("Specific BSE calculations are triggered");
        let bse_spin=qp_ctrl.bse_spin.clone();
        println!("BSE Type:{}",bse_spin);
        let xlet=if bse_spin=="triplet"{'T'}else if bse_spin=="singlet"{'S'}else{panic!("invalid choice for bse_spin!")};

        // ── FEAST solver path for individual spins ──
        if !qp_ctrl.bse_davidson_solver && qp_ctrl.bse_feast_solver {
            eprintln!("Warning: Using FEAST solver for BSE (experimental). \
                       To use the Davidson solver instead, set bse_davidson_solver=true.");
            if crate::ctrl_io::quasiparticle_methods::style_is_ao(&qp_ctrl.bse_matvec_style) {
                eprintln!("Warning: bse_matvec_style = \"ao\" is not wired into the FEAST solver \
                           yet; the FEAST path still uses the MO-basis matvec.");
            }
            let excitations = match bse_spin.as_str() {
                "singlet" => feast_solver::feast_solve_bse_singlet(scf_data),
                "triplet" => feast_solver::feast_solve_bse_triplet(scf_data),
                _ => panic!("invalid choice for bse_spin!"),
            };
            // Print results (same format as below)
            let number = excitations.len();
            println!("{} excitations within the window:", number);
            for (n, (e, vec)) in excitations[..].iter().enumerate() {
                let vec_norm: f64 = vec.iter().map(|x| x*x).sum::<f64>().sqrt();
                println!("#{} Excitation energy={}, norm={:.6}", n, e, vec_norm);
                // Non-TDA solvers return [X;Y] vectors (length 2*O*V); the dipole
                // helpers must be told, exactly as the non-TDA Davidson branch does.
                let v = dipoles::normalize(vec, qp_ctrl.bse_tda);
                let dipole_square =
                    dipoles::transition_dipole_square(&dipole_matrix, &v, qp_ctrl.bse_tda);
                println!("\tTransition Dipole Square:{}; Oscillator Strength:{}",
                    dipole_square, dipole_square * e * 2.0 / 3.0);
                leading_components(&v, occ_size, vir_size,qp_ctrl.print_nto);
            }
            println!("The first excitation obtained by BSE is {}", excitations[0].0);
            if qp_ctrl.save_bse_terms {
                save_bse_amplitudes(&excitations,occ_size,vir_size,qp_ctrl.bse_tda,&bse_spin,
                    qp_ctrl.renormalized_singles&&qp_ctrl.rs_use_rs_orbitals,"bse_terms.dat");
            }
            if qp_ctrl.save_bse_excitations {
                let line = excitations.iter().map(|(num,_)| num.to_string()).collect::<Vec<_>>().join(",");
                let mut file = OpenOptions::new().append(true).create(true).open("bse_excitations.txt");
                writeln!(file.expect("write failure"), "{}", line);
            }
            if qp_ctrl.save_first_excitation {
                let save_path = qp_ctrl.save_first_excitation_path.clone();
                let mut file = OpenOptions::new().append(true).create(true).open(save_path);
                writeln!(file.expect("write failure"), "{}", excitations[0].0);
            }
            return BseOutput { first_excitation: Some(excitations[0].0), excitation_energies: Vec::new() };
        }

        if qp_ctrl.bse_tda==false{
            let mut excitations=non_tda_calculations(&scf_data,&quasiparticle_energies,xlet);
            if scf_data.mol.ctrl.print_level>2{
                show_all_eigenpairs(&excitations);
            }
            let number=excitations.len();
            println!("{} excitations within the window:",number);
            excitations[..].iter().enumerate().for_each(|(n,(e,vec))|{
            let vec_norm: f64 = vec.iter().map(|x| x*x).sum::<f64>().sqrt();
            println!("#{} Excitation energy={}, norm={:.6}",n,e,vec_norm);
            let v=dipoles::normalize(vec,false);
            let dipole_square=dipoles::transition_dipole_square(&dipole_matrix,&v,false);
            println!("Transition Dipole Square:{}; Oscillator Strength:{}",dipole_square,dipole_square*e*2.0/3.0);
            leading_components(&v,occ_size,vir_size,qp_ctrl.print_nto)});
            println!("\n\nThe first excitation obtained by BSE is {}",excitations[0].0);
            if qp_ctrl.save_bse_terms{
                save_bse_amplitudes(&excitations,occ_size,vir_size,false,&bse_spin,
                    qp_ctrl.renormalized_singles&&qp_ctrl.rs_use_rs_orbitals,"bse_terms.dat");
            }
            if qp_ctrl.save_bse_excitations==true{
                let line = excitations.iter().map(|(num,vec)| num.to_string()).collect::<Vec<_>>().join(",");
                let mut file = OpenOptions::new().append(true).create(true).open("bse_excitations.txt");
                writeln!(file.expect("write failure"), "{}", line);
            }
            if qp_ctrl.save_first_excitation==true{
                let save_path=qp_ctrl.save_first_excitation_path.clone();
                let mut file = OpenOptions::new().append(true).create(true).open(save_path);
                writeln!(file.expect("write failure"), "{}", excitations[0].0);
            }
            return BseOutput { first_excitation: Some(excitations[0].0), excitation_energies: Vec::new() };
        }else{
            let excitations=tda_calculations(&scf_data,&quasiparticle_energies,xlet);
            if scf_data.mol.ctrl.print_level>2{
                show_all_eigenpairs(&excitations);
            }
            let number=excitations.len();
            println!("{} excitations within the window:",number);
            excitations[..].iter().enumerate().for_each(|(n,(e,vec))|{
                let v=dipoles::normalize(vec,true);
                let vec_norm: f64 = vec.iter().map(|x| x*x).sum::<f64>().sqrt();
                println!("#{} Excitation energy={}, norm={:.6}",n,e,vec_norm);
                let dipole_square=dipoles::transition_dipole_square(&dipole_matrix,&v,true);
                println!("\tTransition Dipole Square:{}; Oscillator Strength:{}",dipole_square,dipole_square*e*2.0/3.0);
                leading_components(&v,occ_size,vir_size,qp_ctrl.print_nto)});
            println!("The first excitation obtained by BSE is {}",excitations[0].0);
            if qp_ctrl.save_bse_terms{
                save_bse_amplitudes(&excitations,occ_size,vir_size,true,&bse_spin,
                    qp_ctrl.renormalized_singles&&qp_ctrl.rs_use_rs_orbitals,"bse_terms.dat");
            }
            if qp_ctrl.save_bse_excitations==true{
                let line = excitations.iter().map(|(num,vec)| num.to_string()).collect::<Vec<_>>().join(",");
                let mut file = OpenOptions::new().append(true).create(true).open("bse_excitations.txt");
                writeln!(file.expect("write failure"), "{}", line);
            }
            if qp_ctrl.save_first_excitation==true{
                let save_path=qp_ctrl.save_first_excitation_path.clone();
                let mut file = OpenOptions::new().append(true).create(true).open(save_path);
                writeln!(file.expect("write failure"), "{}", excitations[0].0);
            }
            return BseOutput { first_excitation: Some(excitations[0].0), excitation_energies: Vec::new() };
        }
    }
    BseOutput { first_excitation: None, excitation_energies: Vec::new() }
}
pub fn get_submatrix(scf_data:&SCF,choice_a:char,choice_b:char,response_or_not:char)->MatrixFull<f64>{
    let mut vector:Vec<(RIFull<f64>,Range<usize>,Range<usize>)>=Vec::new();
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=ri_gw::get_occupation_parameters(scf_data,response_or_not);
    let range_oo=(start_mo..homo+1, start_mo..homo+1);
    let range_vv=(lumo..num_state, lumo..num_state);
    let range_ov=(start_mo..homo+1, lumo..num_state);
    let range_ff=(start_mo..num_state,start_mo..num_state);

    // Check if BSE-specific RI integrals are available
    let use_bse_integrals = scf_data.ri3fn_bse.is_some() || scf_data.rimatr_bse.is_some();

    if choice_a=='F'&&choice_b=='F'{
        vector=scf_data.generate_ri3mo_rayon_for_multiple_times(range_ff.0,range_ff.1);
    }else if choice_a=='O'&&choice_b=='O'{
        vector = if use_bse_integrals {
            scf_data.generate_ri3mo_bse(range_oo.0, range_oo.1)
        } else {
            scf_data.generate_ri3mo_rayon_for_multiple_times(range_oo.0, range_oo.1)
        };
    }else if choice_a=='V'&&choice_b=='V'{
        vector = if use_bse_integrals {
            scf_data.generate_ri3mo_bse(range_vv.0, range_vv.1)
        } else {
            scf_data.generate_ri3mo_rayon_for_multiple_times(range_vv.0, range_vv.1)
        };
    }else if choice_a=='O'&&choice_b=='V'{
        vector = if use_bse_integrals {
            scf_data.generate_ri3mo_bse(range_ov.0, range_ov.1)
        } else {
            scf_data.generate_ri3mo_rayon_for_multiple_times(range_ov.0, range_ov.1)
        };
    }else {
        panic!("invalid choice of ri subspace!")
    };

    let auxbas_type = if use_bse_integrals { "BSE-specific" } else { "Regular" };
    if scf_data.mol.ctrl.print_level>1{println!("Allocated RI Tensor: {}-{}, Size={:?}, For Response={}, AuxBas Type={}",
             choice_a, choice_b, vector[0].0.size, response_or_not, auxbas_type)};

    let matrix:MatrixFull<f64>=vector.into_iter().next().unwrap().0.into_matfull_i_jk();
    matrix
}
pub fn get_submatrix_spin(scf_data:&SCF,choice_a:char,choice_b:char,response_or_not:char,spin:usize)->MatrixFull<f64>{
    let op = get_occ_params_per_spin(scf_data, response_or_not)[spin];
    let (start_mo, num_state, _occ_size, _vir_size, homo, lumo) = (op.start_mo, op.num_state, op.occ_size, op.vir_size, op.homo, op.lumo);
    let range_oo=(start_mo..homo+1, start_mo..homo+1);
    let range_vv=(lumo..num_state, lumo..num_state);
    let range_ov=(start_mo..homo+1, lumo..num_state);
    let range_ff=(start_mo..num_state,start_mo..num_state);

    let use_bse_integrals = scf_data.ri3fn_bse.is_some() || scf_data.rimatr_bse.is_some();
    let vector:Vec<(RIFull<f64>,Range<usize>,Range<usize>)> = if choice_a=='F'&&choice_b=='F'{
        scf_data.generate_ri3mo_rayon_for_multiple_times(range_ff.0,range_ff.1)
    }else if choice_a=='O'&&choice_b=='O'{
        if use_bse_integrals { scf_data.generate_ri3mo_bse(range_oo.0, range_oo.1) } else { scf_data.generate_ri3mo_rayon_for_multiple_times(range_oo.0, range_oo.1) }
    }else if choice_a=='V'&&choice_b=='V'{
        if use_bse_integrals { scf_data.generate_ri3mo_bse(range_vv.0, range_vv.1) } else { scf_data.generate_ri3mo_rayon_for_multiple_times(range_vv.0, range_vv.1) }
    }else if choice_a=='O'&&choice_b=='V'{
        if use_bse_integrals { scf_data.generate_ri3mo_bse(range_ov.0, range_ov.1) } else { scf_data.generate_ri3mo_rayon_for_multiple_times(range_ov.0, range_ov.1) }
    }else {
        panic!("invalid choice of ri subspace!")
    };
    select_ri3mo_spin(vector, spin)
}

pub fn construct_inverse_dielectric_spin(scf_data:&SCF, epsilon:&[Vec<f64>;2])->MatrixFull<f64>{
    let occ_params = get_occ_params_per_spin(scf_data,'Y');
    let ri_ov = [
        get_submatrix_spin(scf_data,'O','V','Y',0),
        get_submatrix_spin(scf_data,'O','V','Y',1),
    ];
    let response = ri_gw::response_matrix_total(scf_data, epsilon, &occ_params, &ri_ov, 0.0, 'R', 0.0);
    ri_gw::inverse_dielectric_matrix(response, 'R')
}

pub fn construct_raw_w(ri_left:&MatrixFull<f64>,ri_right:&MatrixFull<f64>,inverse_dielectric:&MatrixFull<f64>)->MatrixFull<f64>{
    let mut first_product:MatrixFull<f64>=MatrixFull::new([ri_left.size[1],ri_left.size[0]],0.0);
    let mut w:MatrixFull<f64>=MatrixFull::new([ri_left.size[1],ri_right.size[1]],0.0);
    _dgemm_full(ri_left,'T',inverse_dielectric,'N',&mut first_product,1.0,0.0);
    _dgemm_full(&first_product,'N',ri_right,'N',&mut w,1.0,0.0);
    w
}
pub fn construct_coulomb(ri_left:&MatrixFull<f64>,ri_right:&MatrixFull<f64>)->MatrixFull<f64>{
    let mut coulomb:MatrixFull<f64>=MatrixFull::new([ri_left.size[1],ri_right.size[1]],0.0);
    _dgemm_full(ri_left,'T',ri_right,'N',&mut coulomb,1.0,0.0);
    coulomb
}
pub fn reorganize_w(w:MatrixFull<f64>,part:char,occ_size:usize,vir_size:usize)->MatrixFull<f64>{
    //println!("starts reoorganizing W!!!");
    let mut reorganized_w:MatrixFull<f64>=MatrixFull::new([occ_size*vir_size,occ_size*vir_size],0.0);
    if part=='A'{
        for i in 0..occ_size {
            for j in 0..occ_size {
                for a in 0..vir_size {
                    for b in 0..vir_size {
                        // 根据给定的转换规则计算目标位置
                        //let target_row = i * vir_size + k;
                        let target_row=a*occ_size+i;
                        //let target_col = j * vir_size + l;
                        let target_col=b*occ_size+j;
                        // 计算在原矩阵a中的位置
                        //let source_row = i * occ_size + j;
                        let source_row=j*occ_size+i;
                        let source_col=b*vir_size+a;
                       // let source_col = k * vir_size + l;
                        // 将值从a复制到b中相应的位置
                        reorganized_w[[target_row,target_col]] = w[[source_row,source_col]];
                    }
                }
            }
        }
    }
    if part=='B'{
        for i in 0..occ_size {
            for j in 0..occ_size {
                for a in 0..vir_size {
                    for b in 0..vir_size {
                        
                        // 根据给定的转换规则计算目标位置
                        let target_row=a*occ_size+i;
                        let target_col=b*occ_size+j;
                        //TEMPORARY INDICIES:
                        //let temp_row=a*occ_size+i;
                        //let temp_col=j*vir_size+b;


                        // 计算在原矩阵a中的位置
                        //SAVED SCHEME
                        //let source_row=b*occ_size+i;
                        //let source_col=j*vir_size+a;
                        //TEST SCHEME
                        let source_row=b*occ_size+i;
                        let source_col=a*occ_size+j;
                        // 将值从a复制到b中相应的位置
                        reorganized_w[[target_row,target_col]] = w[[source_row,source_col]];
                    }
                }
            }
        }
    }
    reorganized_w
}
pub fn identity_matrix(n: usize) -> MatrixFull<f64> {
    // 创建一个大小为 n * n 的矩阵，初始值为 0.0
    let mut mat = MatrixFull::new([n, n], 0.0);
    // 设置主对角线上的元素为 1.0
    for i in 0..n {
        mat[[i, i]] = 1.0;
    }
    mat
}
pub fn construct_energy_diag_for_a(quasiparticle_energies:&Vec<f64>,occ_size:usize,vir_size:usize)->Vec<f64>{
    let mut energy_diag = Vec::new();
    for a in 0..vir_size{
        for i in 0..occ_size{
            let energy_gap=quasiparticle_energies[occ_size+a]-quasiparticle_energies[i];
            energy_diag.push(energy_gap);
        }
    }
    energy_diag
}
/// Energies used to dress the independent-particle response `chi0` (and hence the
/// screened interaction `W`) in the BSE stage.
///
/// Historically the caller decides between the Kohn-Sham eigenvalues (default)
/// and the GW quasiparticle energies (`bse_qp_polarization = true`).  When the
/// renormalized-singles orbitals have been installed (`rs_use_rs_orbitals`),
/// neither of those is the right choice on its own: the RI three-centre tensor —
/// and therefore every `v`/`W` matrix element of the BSE kernel — is now
/// expanded in the *RS* orbitals, so `chi0` must be dressed with the *RS*
/// eigenvalues `scf_data.renormalized_singles_particles`.  Otherwise the
/// screening would mix RS orbitals with KS energy denominators and `W` would be
/// in neither representation.
///
/// The RS orbital coefficients are dumped by
/// `renormalized_singles::save_rs_orbitals` when this mode is active, so the BSE
/// transition amplitudes (which live in the RS `(i,a)` product basis) can always
/// be transformed back to the Kohn-Sham basis.
pub fn bse_screening_energies(scf_data:&SCF,epsilon:&Vec<f64>)->Vec<f64>{
    let rs_orbitals_installed=scf_data.mol.ctrl.quasiparticle_methods.clone()
        .map(|qp|qp.renormalized_singles&&qp.rs_use_rs_orbitals)
        .unwrap_or(false);
    if rs_orbitals_installed
        && scf_data.renormalized_singles_particles.len()==scf_data.eigenvalues[0].len()
    {
        println!("BSE screening: using the renormalized-singles eigenvalues (RS representation).");
        scf_data.renormalized_singles_particles.clone()
    }else{
        epsilon.clone()
    }
}
pub fn construct_inverse_dielectric(scf_data:&SCF,epsilon:&Vec<f64>)->MatrixFull<f64>{
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=ri_gw::get_occupation_parameters(scf_data,'Y');
    // Keep the orbital basis and the energy denominators in the same representation.
    let epsilon=bse_screening_energies(scf_data,epsilon);

    // Check if BSE-specific auxiliary basis is being used
    let use_bse_integrals = scf_data.ri3fn_bse.is_some() || scf_data.rimatr_bse.is_some();

    // For response function, use BSE-specific integrals if available
    // This ensures dimensional consistency with BSE Hamiltonian construction
    let mut ri_ov = if use_bse_integrals {
        get_submatrix(scf_data,'O','V','Y')  // Use BSE-specific integrals with full ov space
    } else {
        get_submatrix(scf_data,'O','V','Y')  // Use regular integrals
    };

    let qp_ctrl=scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap();
    if scf_data.mol.ctrl.print_level>1{
        println!("occ_size={},vir_size(for response)={}",occ_size,vir_size);
    }
    let response=ri_gw::response_matrix(&epsilon,occ_size,vir_size,&ri_ov,0.0,'R',0.0);
    let inverse_dielectric=ri_gw::inverse_dielectric_matrix(response,'R');
    inverse_dielectric
}
pub fn construct_submat_a(scf_data:&SCF,inverse_dielectric:&MatrixFull<f64>,quasiparticle_energies:&Vec<f64>,xlet:char)->MatrixFull<f64>{
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=get_occupation_parameters(scf_data,'N');
    let ri_ov=get_submatrix(scf_data,'O','V','N');
    let mut v=construct_coulomb(&ri_ov,&ri_ov);
    let ri_oo=get_submatrix(scf_data,'O','O','N');
    let ri_vv=get_submatrix(scf_data,'V','V','N');
    let raw_w=construct_raw_w(&ri_oo,&ri_vv,&inverse_dielectric);
    let mut w=reorganize_w(raw_w, 'A', occ_size, vir_size);
    w.self_multiple(-1.0);
    let mut energy_diag=construct_energy_diag_for_a(quasiparticle_energies,occ_size,vir_size);
    let mut a=w;
    a.iter_diagonal_mut().unwrap().zip(energy_diag.iter_mut()).for_each(|(x,e)|{(*x,*e)=(*x+*e,*e)});
    if xlet=='S'{
        v.self_multiple(2.0);
        a=MatrixFull::add(&v,&a).unwrap();
    }
    if xlet=='R'{
        a=MatrixFull::add(&v,&a).unwrap();
    }
    a
}
pub fn construct_submat_b(scf_data:&SCF,xlet:char,inverse_dielectric:&MatrixFull<f64>)->MatrixFull<f64>{
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=get_occupation_parameters(scf_data,'N');
    let ri_ov=get_submatrix(scf_data,'O','V','N');
    let mut v=construct_coulomb(&ri_ov,&ri_ov);
    let raw_w=construct_raw_w(&ri_ov,&ri_ov,&inverse_dielectric);
    let mut w=reorganize_w(raw_w, 'B', occ_size, vir_size);
    w.self_multiple(-1.0);
    let mut b=w;
    if xlet=='S'{
        v.self_multiple(2.0);
        b=MatrixFull::add(&b,&v).unwrap();
    }
    if xlet=='R'{
        b=MatrixFull::add(&b,&v).unwrap();
    }
    b
}
pub fn construct_full_bse_hamitonian(scf_data:&SCF,xlet:char,inverse_dielectric:&MatrixFull<f64>,quasiparticle_energies:&Vec<f64>)->MatrixFull<f64>{
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=get_occupation_parameters(scf_data,'N');
    let mut hamiltonian:MatrixFull<f64>=MatrixFull::new([2*occ_size*vir_size,2*occ_size*vir_size],0.0);
    let a:MatrixFull<f64>=construct_submat_a(scf_data,inverse_dielectric,quasiparticle_energies, xlet);
    let mut minus_a=a.transpose();
    let b=construct_submat_b(scf_data,xlet,inverse_dielectric);
    let transpose_b=b.transpose();
    let mut minus_b=b;
    minus_a.self_multiple(-1.0);
    minus_b.self_multiple(-1.0);
    for i in 0..occ_size*vir_size {
        for j in 0..vir_size*occ_size{
            hamiltonian[[i,j]]=a[[i,j]];
            hamiltonian[[occ_size*vir_size+i,j]]=minus_b[[i,j]];
            hamiltonian[[i,occ_size*vir_size+j]]=transpose_b[[i,j]];
            hamiltonian[[occ_size*vir_size+i,occ_size*vir_size+j]]=minus_a[[i,j]];
        }
    }
    hamiltonian
}
pub fn construct_response_full_bse_hamitonian(scf_data:&SCF,xlet:char,inverse_dielectric:&MatrixFull<f64>,quasiparticle_energies:&Vec<f64>)->MatrixFull<f64>{
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=get_occupation_parameters(scf_data,'N');
    let mut hamiltonian:MatrixFull<f64>=MatrixFull::new([2*occ_size*vir_size,2*occ_size*vir_size],0.0);
    let a:MatrixFull<f64>=construct_submat_a(scf_data,inverse_dielectric,quasiparticle_energies, xlet);
    let b=construct_submat_b(scf_data,xlet,inverse_dielectric);
    for i in 0..occ_size*vir_size {
        for j in 0..vir_size*occ_size{
            hamiltonian[[i,j]]=a[[i,j]];
            hamiltonian[[occ_size*vir_size+i,j]]=b[[i,j]];
            hamiltonian[[i,occ_size*vir_size+j]]=b[[i,j]];
            hamiltonian[[occ_size*vir_size+i,occ_size*vir_size+j]]=a[[i,j]];
        }
    }
    hamiltonian
}
pub fn construct_tdhf_tda_hamiltonian(scf_data:&SCF,xlet:char,quasiparticle_energies:&Vec<f64>)->MatrixFull<f64>{
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=get_occupation_parameters(scf_data,'N');
    let ri_ov=get_submatrix(scf_data,'O','V','N');
    let mut v=construct_coulomb(&ri_ov,&ri_ov);
    v.self_multiple(2.0);
    let ri_oo=get_submatrix(scf_data,'O','O','N');
    let ri_vv=get_submatrix(scf_data,'V','V','N');
    let raw_exchange=construct_coulomb(&ri_oo,&ri_vv);
    let mut exchange=reorganize_w(raw_exchange, 'A', occ_size, vir_size);
    exchange.self_multiple(-1.0);
    let mut energy_diag=construct_energy_diag_for_a(quasiparticle_energies,occ_size,vir_size);
    let mut a=exchange;
    a.iter_diagonal_mut().unwrap().zip(energy_diag.iter_mut()).for_each(|(x,e)| {(*x,*e)=(*x+*e,*e)});
    if xlet=='S'{
        a=MatrixFull::add(&v,&a).unwrap();
    }
    a
}
extern "C" fn select(ar: *const f64, ai: *const f64) -> i32 {
    unsafe{if ((*ar)*(*ar)).sqrt() < 1.0e-8 || ((*ai)*(*ai)).sqrt() > 1.0e-8 { 0 } else { 1 }} // Example criterion
}
pub fn evaluate_all_excitations(scf_data:&SCF,quasiparticle_energies:&Vec<f64>,xlet:char)->(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>){
    let ks_energies:Vec<f64>=scf_data.eigenvalues[0].clone();
    let mut epsilon=ks_energies.clone();
    let qp_ctrl=scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap();
    if qp_ctrl.bse_qp_polarization==true{
        epsilon=quasiparticle_energies.clone();
    }
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=get_occupation_parameters(scf_data,'N');
    println!("homo:{},lumo:{},occ_size:{},vir_size:{},start_mo:{},num_state:{}",homo,lumo,occ_size,vir_size,start_mo,num_state);
    println!("--------------------------------");
    println!("starts contructing Full BSE hamiltonian!!!");
    let inverse_dielectric=construct_inverse_dielectric(scf_data,&epsilon);
    println!("the inverse dielectric that is passed into CFBH is of size:{},{}",inverse_dielectric.size[0],inverse_dielectric.size[1]);
    let bse_hamiltonian=construct_full_bse_hamitonian(scf_data, xlet,&inverse_dielectric,quasiparticle_energies);
    println!("--------------------------------");
    let (matr_b_1, wr_1, wi_1,vl_1,vr_1,info_1)=_dgeev(&bse_hamiltonian, 'N', 'V');
    println!("--------------------------------");
    println!("starts contructing TDA BSE hamiltonian!!!");
    let tda_bse_hamiltonian=construct_submat_a(scf_data,&inverse_dielectric,quasiparticle_energies, xlet);
    println!("--------------------------------");
    //tda_bse_hamiltonian.formated_output(occ_size*vir_size,"full");
    let (matr_b_2, wr_2, wi_2,vl_2,vr_2,info_2)=_dgeev(&tda_bse_hamiltonian, 'N', 'V');
    //println!("_dgees has been finished!!!");
    //let excitation_energies=excitations.2;
    //excitation_energies
    println!("--------------------------------");
    println!("starts contructing TDA TDHF hamiltonian!!!");
    let tda_tdhf_hamiltonian=construct_tdhf_tda_hamiltonian(scf_data, xlet, quasiparticle_energies);
    println!("--------------------------------");
    //tda_tdhf_hamiltonian.formated_output(occ_size*vir_size,"full");
    let (matr_b_3, wr_3, wi_3,vl_3,vr_3,info_3)=_dgeev(&tda_bse_hamiltonian, 'N', 'V');
    (wr_1,wi_1,wr_2,wi_2,wr_3,wi_3)
}
/// The MO-basis RI tensors the historical matvec needs.
struct MoBseTensors {
    ri_vv: MatrixFull<f64>,
    ri_oo_tilde: MatrixFull<f64>,
    ri_ov: MatrixFull<f64>,
    ri_ov_b: Option<MatrixFull<f64>>,
    ri_ov_tilde: Option<MatrixFull<f64>>,
}

/// A-block / B-block BSE matvec behind one interface, dispatching between the
/// historical MO-basis implementation and the AO-basis one (`bse_matvec_style`).
///
/// `BseMatvec::new(scf_data, inverse_dielectric, need_b)` builds **whichever**
/// backend the control file selects; in AO mode it never materialises a MO-basis
/// RI tensor.  `need_b = false` additionally skips `ri_ov_b`/`ri_ov_tilde`,
/// which only the B block needs.
///
/// The packed AO tensor is looked up when the object is constructed
/// (`rimatr_bse` first, then the regular `rimatr`), so constructing one of these
/// **per stage** follows the renormalized-doubles workflow automatically:
/// Round 1 runs while the BSE-specific integrals are still loaded, the caller
/// then clears `rimatr_bse` (and `ri3fn_bse`), and the Rayleigh-Ritz refinement
/// consequently builds its matvec from the regular `rimatr`.
pub struct BseMatvec {
    ao: Option<matvec_fast::FastBseContext>,
    mo: Option<MoBseTensors>,
}

impl BseMatvec {
    pub fn new(scf_data: &SCF, inverse_dielectric: &MatrixFull<f64>, need_b: bool) -> Self {
        if let Some(ctx) = build_ao_matvec_context(scf_data, inverse_dielectric) {
            return Self { ao: Some(ctx), mo: None };
        }
        let (_start_mo, _num_state, occ_size, vir_size, _homo, _lumo) =
            get_occupation_parameters(scf_data, 'N');
        let num_auxbas = inverse_dielectric.size[0];
        let ri_ov = get_submatrix(scf_data, 'O', 'V', 'N');
        let mut ri_vv = get_submatrix(scf_data, 'V', 'V', 'N');
        ri_vv.reshape([num_auxbas * vir_size, vir_size]);
        let ri_oo = get_submatrix(scf_data, 'O', 'O', 'N');
        let mut ri_oo_tilde: MatrixFull<f64> = MatrixFull::new(ri_oo.size, 0.0);
        _dgemm_full(inverse_dielectric, 'N', &ri_oo, 'N', &mut ri_oo_tilde, 1.0, 0.0);
        drop(ri_oo);
        ri_oo_tilde.reshape([num_auxbas * occ_size, occ_size]);
        ri_oo_tilde = ri_oo_tilde.transpose_and_drop();
        ri_oo_tilde.reshape([occ_size * num_auxbas, occ_size]);
        let (ri_ov_b, ri_ov_tilde) = if need_b {
            let mut ri_ov_b = ri_ov.clone();
            ri_ov_b.reshape([num_auxbas * occ_size, vir_size]);
            let mut ri_ov_tilde: MatrixFull<f64> = MatrixFull::new(ri_ov.size, 0.0);
            _dgemm_full(inverse_dielectric, 'N', &ri_ov, 'N', &mut ri_ov_tilde, 1.0, 0.0);
            ri_ov_tilde.reshape([num_auxbas * occ_size, vir_size]);
            (Some(ri_ov_b), Some(ri_ov_tilde))
        } else {
            (None, None)
        };
        Self {
            ao: None,
            mo: Some(MoBseTensors { ri_vv, ri_oo_tilde, ri_ov, ri_ov_b, ri_ov_tilde }),
        }
    }

    /// `true` when the AO backend is in use.
    pub fn is_ao(&self) -> bool {
        self.ao.is_some()
    }

    /// `A z`.
    pub fn a(
        &self,
        scf_data: &SCF,
        qp_ctrl: &crate::ctrl_io::quasiparticle_methods::QuasiParticle,
        z: &Vec<f64>,
    ) -> Vec<f64> {
        match (&self.ao, &self.mo) {
            (Some(ctx), _) => ctx.a_block_matvec(scf_data, qp_ctrl, z),
            (None, Some(mo)) => {
                matvec::a_block_matvec(scf_data, qp_ctrl, &mo.ri_vv, &mo.ri_ov, &mo.ri_oo_tilde, z)
            }
            _ => unreachable!("BseMatvec has no backend"),
        }
    }

    /// `B z`.  Requires `need_b = true` at construction.
    pub fn b(
        &self,
        scf_data: &SCF,
        qp_ctrl: &crate::ctrl_io::quasiparticle_methods::QuasiParticle,
        z: &Vec<f64>,
    ) -> Vec<f64> {
        match (&self.ao, &self.mo) {
            (Some(ctx), _) => ctx.b_block_matvec(scf_data, qp_ctrl, z),
            (None, Some(mo)) => matvec::b_block_matvec(
                scf_data,
                qp_ctrl,
                &mo.ri_ov,
                mo.ri_ov_b.as_ref().expect("BseMatvec::b requires need_b = true"),
                mo.ri_ov_tilde.as_ref().expect("BseMatvec::b requires need_b = true"),
                z,
            ),
            _ => unreachable!("BseMatvec has no backend"),
        }
    }

    /// `(A - B) z`.
    pub fn amb(
        &self,
        scf_data: &SCF,
        qp_ctrl: &crate::ctrl_io::quasiparticle_methods::QuasiParticle,
        z: &Vec<f64>,
    ) -> Vec<f64> {
        let a = self.a(scf_data, qp_ctrl, z);
        let b = self.b(scf_data, qp_ctrl, z);
        a.into_iter().zip(b.into_iter()).map(|(ai, bi)| ai - bi).collect()
    }

    /// `(A + B) z`.
    pub fn apb(
        &self,
        scf_data: &SCF,
        qp_ctrl: &crate::ctrl_io::quasiparticle_methods::QuasiParticle,
        z: &Vec<f64>,
    ) -> Vec<f64> {
        let a = self.a(scf_data, qp_ctrl, z);
        let b = self.b(scf_data, qp_ctrl, z);
        a.into_iter().zip(b.into_iter()).map(|(ai, bi)| ai + bi).collect()
    }
}

/// Build the AO-basis ("memory-efficient") matvec context when
/// `[quasiparticle_methods] bse_matvec_style = "ao"`, else `None`.
///
/// When `Some` is returned the caller must **not** build the MO-basis RI
/// tensors (`ri_vv`/`ri_oo_tilde`/`ri_ov`/`ri_ov_tilde`): the whole point of the
/// AO path is that those `O(M*O*V)` objects never exist.
pub fn build_ao_matvec_context(
    scf_data: &SCF,
    inverse_dielectric: &MatrixFull<f64>,
) -> Option<matvec_fast::FastBseContext> {
    let qp = scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap_or_default();
    if !crate::ctrl_io::quasiparticle_methods::style_is_ao(&qp.bse_matvec_style) {
        return None;
    }
    if scf_data.mol.spin_channel > 1 {
        println!(
            "bse_matvec_style = \"ao\": the AO-basis matvec is implemented for the restricted \
             (spin_channel = 1) reference only; keeping the MO-basis matvec."
        );
        return None;
    }
    println!(
        "BSE matvec style: \"ao\" (memory-efficient AO-basis matvec, bse_matvec_style = {})",
        qp.bse_matvec_style
    );
    match matvec_fast::FastBseContext::build(scf_data, inverse_dielectric) {
        Some(ctx) => Some(ctx),
        None => {
            println!(
                "bse_matvec_style = \"ao\": the AO context could not be built (see the message \
                 above); falling back to the MO-basis matvec."
            );
            None
        }
    }
}

pub fn non_tda_calculations(scf_data:&SCF,quasiparticle_energies:&Vec<f64>,xlet:char)->Vec<(f64,Vec<f64>)>{
    let start=Instant::now();
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=get_occupation_parameters(scf_data,'N');
    let ks_energies:Vec<f64>=scf_data.eigenvalues[0].clone();
    let mut epsilon=ks_energies.clone();
    let qp_ctrl=scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap();
    if qp_ctrl.bse_qp_polarization==true{
        epsilon=quasiparticle_energies.clone();
    }
    let inverse_dielectric=construct_inverse_dielectric(scf_data,&epsilon);
    let mut eigenpairs:Vec<(f64,Vec<f64>)>=Vec::new();
    if qp_ctrl.bse_davidson_solver==true{
        // ---- AO ("ao") or MO ("mo") matvec ----
        let ao_ctx = build_ao_matvec_context(scf_data, &inverse_dielectric);
        let mo_ri: Option<(MatrixFull<f64>,MatrixFull<f64>,MatrixFull<f64>,MatrixFull<f64>,MatrixFull<f64>)> =
            if ao_ctx.is_some() {
                drop(inverse_dielectric);
                None
            } else {
                let ri_oo=get_submatrix(scf_data,'O','O','N');
                println!("num_auxbas={}",ri_oo.size[0]);
                let num_auxbas=ri_oo.size[0];
                let mut ri_oo_tilde:MatrixFull<f64>=MatrixFull::new(ri_oo.size,0.0);
                _dgemm_full(&inverse_dielectric,'N',&ri_oo,'N',&mut ri_oo_tilde,1.0,0.0);
                drop(ri_oo);
                ri_oo_tilde.reshape([num_auxbas*occ_size,occ_size]);
                ri_oo_tilde=ri_oo_tilde.transpose_and_drop();
                ri_oo_tilde.reshape([occ_size*num_auxbas,occ_size]);
                let ri_ov=get_submatrix(scf_data,'O','V','N');
                let mut ri_ov_b=ri_ov.clone();
                ri_ov_b.reshape([num_auxbas*occ_size,vir_size]);
                let mut ri_vv=get_submatrix(scf_data,'V','V','N');
                ri_vv.reshape([num_auxbas*vir_size,vir_size]);
                let mut ri_ov_tilde:MatrixFull<f64>=MatrixFull::new(ri_ov.size,0.0);
                _dgemm_full(&inverse_dielectric,'N',&ri_ov,'N',&mut ri_ov_tilde,1.0,0.0);
                drop(inverse_dielectric);
                ri_ov_tilde.reshape([num_auxbas*occ_size,vir_size]);
                Some((ri_vv,ri_ov,ri_oo_tilde,ri_ov_b,ri_ov_tilde))
            };
        let energy_diag=construct_energy_diag_for_a(&scf_data.gwqp.0,occ_size,vir_size);
        let initial_guess=generate_initial_guess(&energy_diag,qp_ctrl.davidson_target_excitations);
        let preptime=start.elapsed();
        println!("BSE Preparation Time:{:?}",preptime);
        let davidson_cfg = DavidsonConfig {
            max_subspace: qp_ctrl.davidson_maximum_subspace_size,
            add_dim: qp_ctrl.davidson_add_dimensions,
            restart_dim: qp_ctrl.davidson_restart_dimensions,
            max_iter: qp_ctrl.davidson_max_iter,
            tol: qp_ctrl.davidson_converge_threshold,
            ..Default::default()
        };
        let a_matvec = |z:&Vec<f64>| -> Vec<f64> {
            match (&ao_ctx,&mo_ri) {
                (Some(ctx),_) => ctx.a_block_matvec(scf_data,&qp_ctrl,z),
                (None,Some((ri_vv,ri_ov,ri_oo_tilde,_,_))) =>
                    matvec::a_block_matvec(scf_data,&qp_ctrl,ri_vv,ri_ov,ri_oo_tilde,z),
                _ => unreachable!("no matvec backend selected"),
            }
        };
        let b_matvec = |z:&Vec<f64>| -> Vec<f64> {
            match (&ao_ctx,&mo_ri) {
                (Some(ctx),_) => ctx.b_block_matvec(scf_data,&qp_ctrl,z),
                (None,Some((_,ri_ov,_,ri_ov_b,ri_ov_tilde))) =>
                    matvec::b_block_matvec(scf_data,&qp_ctrl,ri_ov,ri_ov_b,ri_ov_tilde,z),
                _ => unreachable!("no matvec backend selected"),
            }
        };
        eigenpairs=lr_davidson_solver(a_matvec,b_matvec,qp_ctrl.davidson_target_excitations,&energy_diag,initial_guess,&davidson_cfg);
        println!("Davidson Solver took {:?}",start.elapsed()-preptime);
    }else{
        println!("starts contructing Full BSE hamiltonian!!!");
        let bse_hamiltonian=construct_full_bse_hamitonian(scf_data, xlet,&inverse_dielectric,quasiparticle_energies);
        let preptime=start.elapsed();
        println!("BSE Preparation Time:{:?}",preptime);
        let (matr_b_1, wr_1, wi_1,vl_1,vr_1,info_1)=_dgeev(&bse_hamiltonian, 'N', 'V');
        println!("BSE DGEES Time:{:?}",start.elapsed()-preptime);
        if scf_data.mol.ctrl.print_level > 1 {
            let n = occ_size * vir_size;
            let a_mat = construct_submat_a(scf_data, &inverse_dielectric, quasiparticle_energies, xlet);
            let b_mat = construct_submat_b(scf_data, xlet, &inverse_dielectric);
            // (A+B) matrix
            let apb = MatrixFull::add(&a_mat, &b_mat).unwrap();
            let (_, wr_apb, _, _, _, _) = _dgeev(&apb, 'N', 'N');
            let min_apb = wr_apb.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            println!("Minimum eigenvalue of (A+B) matrix: {}", min_apb);
            if min_apb < 0.0 {
                println!("WARNING:NEGATIVE EIGENVALUE OF DIAGNOSTIC MATRIX!!!");
            }
            // (A-B) matrix
            let mut neg_b = b_mat.clone();
            neg_b.self_multiple(-1.0);
            let amb = MatrixFull::add(&a_mat, &neg_b).unwrap();
            let (_, wr_amb, _, _, _, _) = _dgeev(&amb, 'N', 'N');
            let min_amb = wr_amb.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            println!("Minimum eigenvalue of (A-B) matrix: {}", min_amb);
            if min_amb < 0.0 {
                println!("WARNING:NEGATIVE EIGENVALUE OF DIAGNOSTIC MATRIX!!!");
            }
            // (A+B)(A-B) matrix
            let mut apb_amb = MatrixFull::new([n, n], 0.0);
            _dgemm_full(&apb, 'N', &amb, 'N', &mut apb_amb, 1.0, 0.0);
            let (_, wr_apb_amb, _, _, _, _) = _dgeev(&apb_amb, 'N', 'N');
            let min_apb_amb = wr_apb_amb.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            println!("Minimum eigenvalue of (A+B)(A-B) matrix: {}", min_apb_amb);
            if min_apb_amb < 0.0 {
                println!("WARNING:NEGATIVE EIGENVALUE OF DIAGNOSTIC MATRIX!!!");
            }
        }
        eigenpairs=zip_and_sort(&wr_1,&vr_1);
        eigenpairs = eigenpairs.into_iter()
            .filter(|(val, _)| *val >= qp_ctrl.bse_eigenrange_min && *val <= qp_ctrl.bse_eigenrange_max)
            .collect();
    }
    eigenpairs
}
pub fn tda_calculations(scf_data:&SCF,quasiparticle_energies:&Vec<f64>,xlet:char)->Vec<(f64,Vec<f64>)>{
    let start=Instant::now();
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=get_occupation_parameters(scf_data,'N');
    let ks_energies:Vec<f64>=scf_data.eigenvalues[0].clone();
    let mut epsilon=ks_energies.clone();
    let qp_ctrl=scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap();
    if qp_ctrl.bse_qp_polarization==true{
        epsilon=quasiparticle_energies.clone();
    }
    let inverse_dielectric=construct_inverse_dielectric(scf_data,&epsilon);
    let mut eigenpairs:Vec<(f64,Vec<f64>)>=Vec::new();
    if qp_ctrl.bse_davidson_solver==true{
        let start=Instant::now();
        // ---- AO ("ao") or MO ("mo") matvec ----
        let ao_ctx = build_ao_matvec_context(scf_data, &inverse_dielectric);
        let mo_ri: Option<(MatrixFull<f64>,MatrixFull<f64>,MatrixFull<f64>)> = if ao_ctx.is_some() {
            drop(inverse_dielectric);
            None
        } else {
            let ri_ov=get_submatrix(scf_data,'O','V','N');
            let duration1=start.elapsed();
            let _ = duration1;
            let mut ri_vv=get_submatrix(scf_data,'V','V','N');
            let num_auxbas=inverse_dielectric.size[0];
            let duration2=start.elapsed();
            let _ = duration2;
            let ri_oo=get_submatrix(scf_data,'O','O','N');
            let duration3=start.elapsed();
            let _ = duration3;
            let mut ri_oo_tilde:MatrixFull<f64>=MatrixFull::new(ri_oo.size,0.0);
            _dgemm_full(&inverse_dielectric,'N',&ri_oo,'N',&mut ri_oo_tilde,1.0,0.0);
            drop(ri_oo);
            drop(inverse_dielectric);
            ri_oo_tilde.reshape([num_auxbas*occ_size,occ_size]);
            ri_oo_tilde=ri_oo_tilde.transpose_and_drop();
            ri_oo_tilde.reshape([occ_size*num_auxbas,occ_size]);
            ri_vv.reshape([num_auxbas*vir_size,vir_size]);
            Some((ri_vv,ri_ov,ri_oo_tilde))
        };
        let energy_diag=construct_energy_diag_for_a(&scf_data.gwqp.0,occ_size,vir_size);
        let initial_guess=generate_initial_guess(&energy_diag,qp_ctrl.davidson_target_excitations);
        let preptime=start.elapsed();
        println!("BSE Preparation Time:{:?}",preptime);
        let davidson_cfg = DavidsonConfig {
            max_subspace: qp_ctrl.davidson_maximum_subspace_size,
            add_dim: qp_ctrl.davidson_add_dimensions,
            restart_dim: qp_ctrl.davidson_restart_dimensions,
            max_iter: qp_ctrl.davidson_max_iter,
            tol: qp_ctrl.davidson_converge_threshold,
            ..Default::default()
        };
        let a_matvec = |z:&Vec<f64>| -> Vec<f64> {
            match (&ao_ctx,&mo_ri) {
                (Some(ctx),_) => ctx.a_block_matvec(scf_data,&qp_ctrl,z),
                (None,Some((ri_vv,ri_ov,ri_oo_tilde))) =>
                    matvec::a_block_matvec(scf_data,&qp_ctrl,ri_vv,ri_ov,ri_oo_tilde,z),
                _ => unreachable!("no matvec backend selected"),
            }
        };
        eigenpairs=tda_davidson_solver(a_matvec,qp_ctrl.davidson_target_excitations,&energy_diag,initial_guess,&davidson_cfg);
        println!("Davidson Solver took {:?}",start.elapsed());
    }else{
        println!("starts contructing TDA BSE hamiltonian!!!");
        let tda_bse_hamiltonian=construct_submat_a(scf_data,&inverse_dielectric,quasiparticle_energies, xlet);
        let preptime=start.elapsed();
        println!("BSE Preparation Time:{:?}",preptime);
        let (matr_b_1, wr_1, wi_1,vl_1,vr_1,info_1)=_dgeev(&tda_bse_hamiltonian, 'N', 'V');
        println!("BSE DGEES Time:{:?}",start.elapsed()-preptime);
        eigenpairs=zip_and_sort(&wr_1,&vr_1);
        eigenpairs = eigenpairs.into_iter()
            .filter(|(val, _)| *val >= qp_ctrl.bse_eigenrange_min && *val <= qp_ctrl.bse_eigenrange_max)
            .collect();
    }
    eigenpairs
}
pub fn bse_both_spins(scf_data:&mut SCF,quasiparticle_energies:&Vec<f64>)->(Vec<(f64,Vec<f64>)>,Vec<(f64,Vec<f64>)>){
    let qp_ctrl_ref=scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap();

    // ── FEAST solver path ──
    // Triggered when bse_davidson_solver=false and bse_feast_solver=true.
    if !qp_ctrl_ref.bse_davidson_solver && qp_ctrl_ref.bse_feast_solver {
        eprintln!("Warning: Using FEAST solver for BSE (experimental). \
                   To use the Davidson solver instead, set bse_davidson_solver=true.");
        return feast_solver::feast_solve_bse(scf_data);
    }

    // ── Davidson solver path (also fallback) ──
    let start=Instant::now();
    let mut qp_ctrl=qp_ctrl_ref;
    let mut eigenpairs_singlet:Vec<(f64,Vec<f64>)>=Vec::new();
    let mut eigenpairs_triplet:Vec<(f64,Vec<f64>)>=Vec::new();
    if qp_ctrl.bse_tda==true{
        let (start_mo,num_state,occ_size,vir_size,homo,lumo)=get_occupation_parameters(scf_data,'N');
        let ks_energies:Vec<f64>=scf_data.eigenvalues[0].clone();
        let mut epsilon=ks_energies.clone();
        if qp_ctrl.bse_qp_polarization==true{
            epsilon=quasiparticle_energies.clone();
        }
        let inverse_dielectric=construct_inverse_dielectric(scf_data,&epsilon);
        let start=Instant::now();
        let ri_ov=get_submatrix(scf_data,'O','V','N');
        let duration1=start.elapsed();
        //println!("RI-OV耗时: {:?}", duration1);
        let mut ri_vv=get_submatrix(scf_data,'V','V','N');
        let num_auxbas=inverse_dielectric.size[0];
        //ri_vv.reshape([num_auxbas*vir_size,vir_size]);
        let duration2=start.elapsed();
        //println!("RI-VV作耗时: {:?}", duration2-duration1);
        let ri_oo=get_submatrix(scf_data,'O','O','N');
        let duration3=start.elapsed();
        //println!("RI-OO耗时: {:?}", duration3-duration2);
        let mut ri_oo_tilde:MatrixFull<f64>=MatrixFull::new(ri_oo.size,0.0);
        println!("Created RI-OO-Tilde. Size=[{},{},{}]",occ_size,occ_size,num_auxbas);
        _dgemm_full(&inverse_dielectric,'N',&ri_oo,'N',&mut ri_oo_tilde,1.0,0.0);
        drop(inverse_dielectric);
        ri_oo_tilde.reshape([num_auxbas*occ_size,occ_size]);
        ri_oo_tilde=ri_oo_tilde.transpose_and_drop();
        ri_oo_tilde.reshape([occ_size*num_auxbas,occ_size]);
        ri_vv.reshape([num_auxbas*vir_size,vir_size]);
        //ri_oo_tilde.reshape([num_auxbas*occ_size,occ_size]);
        let energy_diag=construct_energy_diag_for_a(&scf_data.gwqp.0,occ_size,vir_size);
        let preptime=start.elapsed();
        println!("BSE Preparation Time:{:?}",preptime);
        let davidson_cfg = DavidsonConfig {
            max_subspace: qp_ctrl.davidson_maximum_subspace_size,
            add_dim: qp_ctrl.davidson_add_dimensions,
            restart_dim: qp_ctrl.davidson_restart_dimensions,
            max_iter: qp_ctrl.davidson_max_iter,
            tol: qp_ctrl.davidson_converge_threshold,
            ..Default::default()
        };
        qp_ctrl.bse_spin=String::from("singlet");
        let initial_guess=generate_initial_guess(&energy_diag,qp_ctrl.davidson_target_excitations);
        eigenpairs_singlet=tda_davidson_solver(|z|matvec::a_block_matvec(scf_data,&qp_ctrl,&ri_vv,&ri_ov,&ri_oo_tilde,&z),qp_ctrl.davidson_target_excitations,&energy_diag,initial_guess,&davidson_cfg);
        let one_dav_time=start.elapsed();
        println!("Singlets calculation took {:?}",one_dav_time-preptime);
        qp_ctrl.bse_spin=String::from("triplet");
        let initial_guess=generate_initial_guess(&energy_diag,qp_ctrl.davidson_target_excitations);
        eigenpairs_triplet=tda_davidson_solver(|z|matvec::a_block_matvec(scf_data,&qp_ctrl,&ri_vv,&ri_ov,&ri_oo_tilde,&z),qp_ctrl.davidson_target_excitations,&energy_diag,initial_guess,&davidson_cfg);
        println!("Triplets calculation took {:?}",start.elapsed()-one_dav_time);
    }else{
        let (start_mo,num_state,occ_size,vir_size,homo,lumo)=get_occupation_parameters(scf_data,'N');
        let ks_energies:Vec<f64>=scf_data.eigenvalues[0].clone();
        let mut epsilon=ks_energies.clone();
        if qp_ctrl.bse_qp_polarization==true{
            epsilon=quasiparticle_energies.clone();
        }
        let inverse_dielectric=construct_inverse_dielectric(scf_data,&epsilon);
        let ri_oo=get_submatrix(scf_data,'O','O','N');
        println!("num_auxbas={}",ri_oo.size[0]);
        let num_auxbas=ri_oo.size[0];
        let mut ri_oo_tilde:MatrixFull<f64>=MatrixFull::new(ri_oo.size,0.0);
        println!("Created RI-OO-Tilde. Size=[{},{},{}]",occ_size,occ_size,num_auxbas);
        _dgemm_full(&inverse_dielectric,'N',&ri_oo,'N',&mut ri_oo_tilde,1.0,0.0);
        drop(ri_oo);
        println!("Deallocated RI-OO. Size=[{},{},{}]",occ_size,occ_size,num_auxbas);
        ri_oo_tilde.reshape([num_auxbas*occ_size,occ_size]);
        ri_oo_tilde=ri_oo_tilde.transpose_and_drop();
        ri_oo_tilde.reshape([occ_size*num_auxbas,occ_size]);
        let ri_ov=get_submatrix(scf_data,'O','V','N');
        let mut ri_vv=get_submatrix(scf_data,'V','V','N');
        ri_vv.reshape([num_auxbas*vir_size,vir_size]);
        let mut ri_ov_tilde:MatrixFull<f64>=MatrixFull::new(ri_ov.size,0.0);
        println!("Created RI-OV-Tilde. Size=[{},{},{}]",occ_size,vir_size,num_auxbas);
        _dgemm_full(&inverse_dielectric,'N',&ri_ov,'N',&mut ri_ov_tilde,1.0,0.0);
        ri_ov_tilde.reshape([num_auxbas*occ_size,vir_size]);
        let mut ri_ov_b=ri_ov.clone();
        println!("Created RI-OV for B block. Size=[{},{},{}]",occ_size,vir_size,num_auxbas);
        ri_ov_b.reshape([num_auxbas*occ_size,vir_size]);
        drop(inverse_dielectric);
        let energy_diag=construct_energy_diag_for_a(&scf_data.gwqp.0,occ_size,vir_size);
        let initial_guess=generate_initial_guess(&energy_diag,qp_ctrl.davidson_target_excitations);
        let preptime=start.elapsed();
        println!("BSE Preparation Time:{:?}",preptime);
        let davidson_cfg = DavidsonConfig {
            max_subspace: qp_ctrl.davidson_maximum_subspace_size,
            add_dim: qp_ctrl.davidson_add_dimensions,
            restart_dim: qp_ctrl.davidson_restart_dimensions,
            max_iter: qp_ctrl.davidson_max_iter,
            tol: qp_ctrl.davidson_converge_threshold,
            ..Default::default()
        };
        qp_ctrl.bse_spin=String::from("singlet");
        eigenpairs_singlet=lr_davidson_solver(|z|matvec::a_block_matvec(scf_data,&qp_ctrl,&ri_vv,&ri_ov,&ri_oo_tilde,&z),|z|matvec::b_block_matvec(scf_data,&qp_ctrl,&ri_ov,&ri_ov_b,&ri_ov_tilde,&z),qp_ctrl.davidson_target_excitations,&energy_diag,initial_guess.clone(),&davidson_cfg);
        let one_dav_time=start.elapsed();
        println!("Singlets calculation took {:?}",one_dav_time-preptime);
        qp_ctrl.bse_spin=String::from("triplet");
        eigenpairs_triplet=lr_davidson_solver(|z|matvec::a_block_matvec(scf_data,&qp_ctrl,&ri_vv,&ri_ov,&ri_oo_tilde,&z),|z|matvec::b_block_matvec(scf_data,&qp_ctrl,&ri_ov,&ri_ov_b,&ri_ov_tilde,&z),qp_ctrl.davidson_target_excitations,&energy_diag,initial_guess,&davidson_cfg);
        println!("Triplets calculation took {:?}",start.elapsed()-one_dav_time);
    }
    (eigenpairs_singlet,eigenpairs_triplet)
}
fn place_block(dst:&mut MatrixFull<f64>, src:&MatrixFull<f64>, row0:usize, col0:usize){
    for j in 0..src.size[1] {
        for i in 0..src.size[0] {
            dst[[row0+i, col0+j]] = src[[i,j]];
        }
    }
}

/// Unrestricted A block of the BSE Hamiltonian.
///
/// The direct (screened Coulomb) term always enters: it is spin-independent and
/// is the only term coupling the alpha and beta blocks, so it is part of the
/// single physical channel of an unrestricted reference.  There is no
/// "triplet-like" (direct-term-free) variant -- see `bse_main_unrestricted`.
fn construct_u_submat_a(
    scf_data:&SCF,
    inverse_dielectric:&MatrixFull<f64>,
    quasiparticle_energies:&[Vec<f64>;2],
)->MatrixFull<f64>{
    let ops = get_occ_params_per_spin(scf_data,'N');
    let nspin = scf_data.mol.spin_channel;
    let ntot:usize = (0..nspin).map(|s| ops[s].occ_size*ops[s].vir_size).sum();
    let mut a = MatrixFull::new([ntot, ntot], 0.0);
    let mut offsets = [0usize;2];
    for s in 0..nspin {
        offsets[s] = if s==0 {0} else {offsets[s-1] + ops[s-1].occ_size*ops[s-1].vir_size};
    }
    for s in 0..nspin {
        let ri_ov_s = get_submatrix_spin(scf_data,'O','V','N',s);
        let ri_oo_s = get_submatrix_spin(scf_data,'O','O','N',s);
        let ri_vv_s = get_submatrix_spin(scf_data,'V','V','N',s);
        let v_ss = construct_coulomb(&ri_ov_s, &ri_ov_s);
        let raw_w = construct_raw_w(&ri_oo_s, &ri_vv_s, inverse_dielectric);
        let mut w_a = reorganize_w(raw_w, 'A', ops[s].occ_size, ops[s].vir_size);
        w_a.self_multiple(-1.0);
        let mut diag = construct_energy_diag_for_a(&quasiparticle_energies[s], ops[s].occ_size, ops[s].vir_size);
        let mut block = w_a;
        block.iter_diagonal_mut().unwrap().zip(diag.iter_mut()).for_each(|(x,e)|{*x += *e;});
        block = MatrixFull::add(&v_ss, &block).unwrap();
        place_block(&mut a, &block, offsets[s], offsets[s]);
        for t in 0..nspin {
            if s == t { continue; }
            let ri_ov_t = get_submatrix_spin(scf_data,'O','V','N',t);
            let v_st = construct_coulomb(&ri_ov_s, &ri_ov_t);
            place_block(&mut a, &v_st, offsets[s], offsets[t]);
        }
    }
    a
}

/// Unrestricted B block of the BSE Hamiltonian (direct term always included).
fn construct_u_submat_b(
    scf_data:&SCF,
    inverse_dielectric:&MatrixFull<f64>,
)->MatrixFull<f64>{
    let ops = get_occ_params_per_spin(scf_data,'N');
    let nspin = scf_data.mol.spin_channel;
    let ntot:usize = (0..nspin).map(|s| ops[s].occ_size*ops[s].vir_size).sum();
    let mut b = MatrixFull::new([ntot, ntot], 0.0);
    let mut offsets = [0usize;2];
    for s in 0..nspin {
        offsets[s] = if s==0 {0} else {offsets[s-1] + ops[s-1].occ_size*ops[s-1].vir_size};
    }
    for s in 0..nspin {
        let ri_ov_s = get_submatrix_spin(scf_data,'O','V','N',s);
        let raw_w = construct_raw_w(&ri_ov_s, &ri_ov_s, inverse_dielectric);
        let mut w_b = reorganize_w(raw_w, 'B', ops[s].occ_size, ops[s].vir_size);
        w_b.self_multiple(-1.0);
        let mut block = w_b;
        let v_ss = construct_coulomb(&ri_ov_s, &ri_ov_s);
        block = MatrixFull::add(&v_ss, &block).unwrap();
        place_block(&mut b, &block, offsets[s], offsets[s]);
        for t in 0..nspin {
            if s == t { continue; }
            let ri_ov_t = get_submatrix_spin(scf_data,'O','V','N',t);
            let v_st = construct_coulomb(&ri_ov_s, &ri_ov_t);
            place_block(&mut b, &v_st, offsets[s], offsets[t]);
        }
    }
    b
}

fn construct_u_full_bse_hamiltonian(
    scf_data:&SCF,
    inverse_dielectric:&MatrixFull<f64>,
    quasiparticle_energies:&[Vec<f64>;2],
)->MatrixFull<f64>{
    let a = construct_u_submat_a(scf_data, inverse_dielectric, quasiparticle_energies);
    let b = construct_u_submat_b(scf_data, inverse_dielectric);
    let n = a.size[0];
    let mut h = MatrixFull::new([2*n,2*n], 0.0);
    let mut minus_a = a.transpose();
    let b_t = b.transpose();
    let mut minus_b = b.clone();
    minus_a.self_multiple(-1.0);
    minus_b.self_multiple(-1.0);
    for j in 0..n {
        for i in 0..n {
            h[[i,j]] = a[[i,j]];
            h[[n+i,j]] = minus_b[[i,j]];
            h[[i,n+j]] = b_t[[i,j]];
            h[[n+i,n+j]] = minus_a[[i,j]];
        }
    }
    h
}

fn solve_dense_eigenpairs(mat:&MatrixFull<f64>, qp_ctrl:&crate::ctrl_io::quasiparticle_methods::QuasiParticle)->Vec<(f64,Vec<f64>)>{
    let (_, wr, _wi, _vl, vr, _info) = _dgeev(mat, 'N', 'V');
    let mut pairs = zip_and_sort(&wr, &vr);
    pairs = pairs.into_iter().filter(|(val,_)| *val >= qp_ctrl.bse_eigenrange_min && *val <= qp_ctrl.bse_eigenrange_max).collect();
    pairs
}

struct UnrestrictedBseData {
    ri_vv: [MatrixFull<f64>; 2],
    ri_ov: [MatrixFull<f64>; 2],
    ri_oo_tilde: [MatrixFull<f64>; 2],
    ri_ov_b: [MatrixFull<f64>; 2],
    ri_ov_tilde: [MatrixFull<f64>; 2],
    energy_diag: Vec<f64>,
}

fn prepare_unrestricted_matvec_data(
    scf_data: &SCF,
    inverse_dielectric: &MatrixFull<f64>,
    energies: &[Vec<f64>; 2],
) -> UnrestrictedBseData {
    let ops = get_occ_params_per_spin(scf_data, 'N');
    let num_auxbas = inverse_dielectric.size[0];
    let mut ri_vv: [MatrixFull<f64>; 2] = [MatrixFull::new([0, 0], 0.0), MatrixFull::new([0, 0], 0.0)];
    let mut ri_ov: [MatrixFull<f64>; 2] = [MatrixFull::new([0, 0], 0.0), MatrixFull::new([0, 0], 0.0)];
    let mut ri_oo_tilde: [MatrixFull<f64>; 2] = [MatrixFull::new([0, 0], 0.0), MatrixFull::new([0, 0], 0.0)];
    let mut ri_ov_b: [MatrixFull<f64>; 2] = [MatrixFull::new([0, 0], 0.0), MatrixFull::new([0, 0], 0.0)];
    let mut ri_ov_tilde: [MatrixFull<f64>; 2] = [MatrixFull::new([0, 0], 0.0), MatrixFull::new([0, 0], 0.0)];
    let mut energy_diag = Vec::new();

    for s in 0..2 {
        let occ_s = ops[s].occ_size;
        let vir_s = ops[s].vir_size;

        ri_ov[s] = get_submatrix_spin(scf_data, 'O', 'V', 'N', s);

        let ri_oo_s = get_submatrix_spin(scf_data, 'O', 'O', 'N', s);
        let mut oo_tilde = MatrixFull::new(ri_oo_s.size, 0.0);
        _dgemm_full(inverse_dielectric, 'N', &ri_oo_s, 'N', &mut oo_tilde, 1.0, 0.0);
        oo_tilde.reshape([num_auxbas * occ_s, occ_s]);
        oo_tilde = oo_tilde.transpose_and_drop();
        oo_tilde.reshape([occ_s * num_auxbas, occ_s]);
        ri_oo_tilde[s] = oo_tilde;

        let ri_vv_s = get_submatrix_spin(scf_data, 'V', 'V', 'N', s);
        let mut vv = ri_vv_s;
        vv.reshape([num_auxbas * vir_s, vir_s]);
        ri_vv[s] = vv;

        let mut ov_tilde = MatrixFull::new(ri_ov[s].size, 0.0);
        _dgemm_full(inverse_dielectric, 'N', &ri_ov[s], 'N', &mut ov_tilde, 1.0, 0.0);
        ov_tilde.reshape([num_auxbas * occ_s, vir_s]);
        ri_ov_tilde[s] = ov_tilde;

        let mut ov_b = ri_ov[s].clone();
        ov_b.reshape([num_auxbas * occ_s, vir_s]);
        ri_ov_b[s] = ov_b;

        energy_diag.extend(construct_energy_diag_for_a(&energies[s], occ_s, vir_s));
    }

    UnrestrictedBseData {
        ri_vv,
        ri_ov,
        ri_oo_tilde,
        ri_ov_b,
        ri_ov_tilde,
        energy_diag,
    }
}

fn davidson_config_from_qp(qp_ctrl: &crate::ctrl_io::quasiparticle_methods::QuasiParticle) -> DavidsonConfig {
    DavidsonConfig {
        max_subspace: qp_ctrl.davidson_maximum_subspace_size,
        add_dim: qp_ctrl.davidson_add_dimensions,
        restart_dim: qp_ctrl.davidson_restart_dimensions,
        max_iter: qp_ctrl.davidson_max_iter,
        tol: qp_ctrl.davidson_converge_threshold,
        ..Default::default()
    }
}

fn filter_eigenpairs_range(
    eigenpairs: Vec<(f64, Vec<f64>)>,
    qp_ctrl: &crate::ctrl_io::quasiparticle_methods::QuasiParticle,
) -> Vec<(f64, Vec<f64>)> {
    eigenpairs
        .into_iter()
        .filter(|(val, _)| *val >= qp_ctrl.bse_eigenrange_min && *val <= qp_ctrl.bse_eigenrange_max)
        .collect()
}


fn unrestricted_a_matvec(
    scf_data: &SCF,
    qp_ctrl: &crate::ctrl_io::quasiparticle_methods::QuasiParticle,
    energies: &[Vec<f64>; 2],
    data: &UnrestrictedBseData,
    z: &Vec<f64>,
) -> Vec<f64> {
    matvec::a_block_matvec_unrestricted(
        scf_data, qp_ctrl, energies,
        &data.ri_vv, &data.ri_ov, &data.ri_oo_tilde, z)
}

fn unrestricted_b_matvec(
    scf_data: &SCF,
    qp_ctrl: &crate::ctrl_io::quasiparticle_methods::QuasiParticle,
    data: &UnrestrictedBseData,
    z: &Vec<f64>,
) -> Vec<f64> {
    matvec::b_block_matvec_unrestricted(
        scf_data, qp_ctrl,
        &data.ri_ov, &data.ri_ov_b, &data.ri_ov_tilde, z)
}

fn unrestricted_amb_matvec(
    scf_data: &SCF,
    qp_ctrl: &crate::ctrl_io::quasiparticle_methods::QuasiParticle,
    energies: &[Vec<f64>; 2],
    data: &UnrestrictedBseData,
    z: &Vec<f64>,
) -> Vec<f64> {
    let a = unrestricted_a_matvec(scf_data, qp_ctrl, energies, data, z);
    let b = unrestricted_b_matvec(scf_data, qp_ctrl, data, z);
    a.into_iter().zip(b).map(|(ai, bi)| ai - bi).collect()
}

fn unrestricted_apb_matvec(
    scf_data: &SCF,
    qp_ctrl: &crate::ctrl_io::quasiparticle_methods::QuasiParticle,
    energies: &[Vec<f64>; 2],
    data: &UnrestrictedBseData,
    z: &Vec<f64>,
) -> Vec<f64> {
    let a = unrestricted_a_matvec(scf_data, qp_ctrl, energies, data, z);
    let b = unrestricted_b_matvec(scf_data, qp_ctrl, data, z);
    a.into_iter().zip(b).map(|(ai, bi)| ai + bi).collect()
}

/// FEAST solve for the single unrestricted BSE channel (direct term included).
fn feast_solve_bse_unrestricted(
    scf_data: &SCF,
    qp_ctrl: &crate::ctrl_io::quasiparticle_methods::QuasiParticle,
    energies: &[Vec<f64>; 2],
    data: &UnrestrictedBseData,
) -> Vec<(f64, Vec<f64>)> {
    let n = data.energy_diag.len();
    let (emin, emax) = if qp_ctrl.bse_tda {
        (qp_ctrl.bse_eigenrange_min, qp_ctrl.bse_eigenrange_max)
    } else {
        (qp_ctrl.bse_eigenrange_min * qp_ctrl.bse_eigenrange_min,
         qp_ctrl.bse_eigenrange_max * qp_ctrl.bse_eigenrange_max)
    };

    if qp_ctrl.bse_tda {
        let a_mul = |z: &Vec<f64>| {
            unrestricted_a_matvec(scf_data, qp_ctrl, energies, data, z)
        };
        let b_mul = |z: &Vec<f64>| z.clone();
        feast::feast(
            n,
            &a_mul,
            &b_mul,
            None,
            None,
            emin,
            emax,
            qp_ctrl.bse_m_expected,
            qp_ctrl.bse_max_feast_iter,
            qp_ctrl.bse_tol_feast,
            qp_ctrl.bse_feast_gmres_restart,
            qp_ctrl.bse_feast_gmres_max_iter,
            qp_ctrl.bse_feast_cg_tol,
            Some(&data.energy_diag),
            &qp_ctrl.bse_feast_init_guess_type,
            Some(&data.energy_diag),
            qp_ctrl.bse_feast_gaussian_width_factor,
            qp_ctrl.bse_feast_contour_rayon,
            None,
            None,
            None,
            "diagonal",
            None,
            0.0001,
            0,
            0,
        )
    } else {
        let a_mul = |z: &Vec<f64>| {
            unrestricted_amb_matvec(scf_data, qp_ctrl, energies, data, z)
        };
        let b_mul = |z: &Vec<f64>| {
            let apb = |p: &Vec<f64>| {
                unrestricted_apb_matvec(scf_data, qp_ctrl, energies, data, p)
            };
            feast::cg(
                &apb,
                z,
                qp_ctrl.bse_feast_cg_max_iter,
                qp_ctrl.bse_feast_cg_tol,
                Some(&data.energy_diag),
            )
        };
        let diag_sq: Vec<f64> = data.energy_diag.iter().map(|&d| d * d).collect();
        let gmres_a_mul = |z: &Vec<f64>| {
            let amb_z = unrestricted_amb_matvec(scf_data, qp_ctrl, energies, data, z);
            unrestricted_apb_matvec(scf_data, qp_ctrl, energies, data, &amb_z)
        };
        let gmres_b_mul = |z: &Vec<f64>| z.clone();
        let raw = feast::feast(
            n,
            &a_mul,
            &b_mul,
            Some(&gmres_a_mul),
            Some(&gmres_b_mul),
            emin,
            emax,
            qp_ctrl.bse_m_expected,
            qp_ctrl.bse_max_feast_iter,
            qp_ctrl.bse_tol_feast,
            qp_ctrl.bse_feast_gmres_restart,
            qp_ctrl.bse_feast_gmres_max_iter,
            qp_ctrl.bse_feast_cg_tol,
            Some(&diag_sq),
            &qp_ctrl.bse_feast_init_guess_type,
            Some(&data.energy_diag),
            qp_ctrl.bse_feast_gaussian_width_factor,
            qp_ctrl.bse_feast_contour_rayon,
            None,
            None,
            None,
            "diagonal",
            None,
            0.0001,
            0,
            0,
        );
        raw.into_iter()
            .map(|(omega2, xpy)| {
                let xmy = unrestricted_amb_matvec(scf_data, qp_ctrl, energies, data, &xpy);
                (omega2.sqrt(), xmy.iter().zip(xpy.iter()).map(|(xmy_k, xpy_k)| xmy_k / omega2.sqrt() + xpy_k).collect())
            })
            .collect()
    }
}


pub fn bse_main_unrestricted(scf_data:&mut SCF) -> BseOutput {
    let qp_ctrl = scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap();
    if qp_ctrl.bse_spin == "none" {
        println!("No BSE Calculations are triggered");
        return BseOutput { first_excitation: None, excitation_energies: Vec::new() };
    }
    let (start_mo, num_state, occ_size, vir_size, _homo, _lumo) = get_occupation_parameters(scf_data, 'N');
    let _ = (start_mo, num_state, occ_size, vir_size);

    let qp_energies = scf_data.gwqp_spin.0.clone();
    let ks_energies = [scf_data.eigenvalues[0].clone(), scf_data.eigenvalues[1].clone()];
    let epsilon = if qp_ctrl.bse_qp_polarization { qp_energies.clone() } else { ks_energies };
    let inverse_dielectric = construct_inverse_dielectric_spin(scf_data, &epsilon);

    // `bse_spin` does not select a spin channel for an unrestricted reference.
    // The direct (screened Coulomb) kernel is spin-independent and is the only
    // term coupling the alpha and beta blocks, so an unrestricted BSE
    // Hamiltonian has exactly one physical channel; the former "triplet"
    // (direct-term-free) and "both" options have been removed.  Leaving
    // `bse_spin = "none"` still switches the BSE run off (handled above).
    match qp_ctrl.bse_spin.as_str() {
        "singlet" | "unrestricted" => {}
        other => panic!(
            "bse_spin = \"{}\" is not applicable to an unrestricted reference \
             (spin_polarization = true): unrestricted BSE has a single, spin-coupled \
             channel (the direct kernel couples the alpha and beta blocks), so there is \
             no triplet-like channel to select. Set bse_spin to \"singlet\" (or any \
             accepted value) to run unrestricted BSE, or to \"none\" to switch it off.",
            other
        ),
    }

    let mut all_excitations: Vec<f64> = Vec::new();

    if qp_ctrl.bse_davidson_solver {
        let data = prepare_unrestricted_matvec_data(scf_data, &inverse_dielectric, &qp_energies);
        // Single spin-coupled channel (previously a loop over spin modes).
        {
            let cfg = davidson_config_from_qp(&qp_ctrl);
            let initial_guess = generate_initial_guess(&data.energy_diag, qp_ctrl.davidson_target_excitations);
            let eigenpairs = if qp_ctrl.bse_tda {
                let a_matvec = |z: &Vec<f64>| {
                    matvec::a_block_matvec_unrestricted(
                        scf_data, &qp_ctrl, &qp_energies,
                        &data.ri_vv, &data.ri_ov, &data.ri_oo_tilde, z)
                };
                tda_davidson_solver(
                    a_matvec,
                    qp_ctrl.davidson_target_excitations,
                    &data.energy_diag,
                    initial_guess,
                    &cfg,
                )
            } else {
                let a_matvec = |z: &Vec<f64>| {
                    matvec::a_block_matvec_unrestricted(
                        scf_data, &qp_ctrl, &qp_energies,
                        &data.ri_vv, &data.ri_ov, &data.ri_oo_tilde, z)
                };
                let b_matvec = |z: &Vec<f64>| {
                    matvec::b_block_matvec_unrestricted(
                        scf_data, &qp_ctrl,
                        &data.ri_ov, &data.ri_ov_b, &data.ri_ov_tilde, z)
                };
                lr_davidson_solver(
                    a_matvec,
                    b_matvec,
                    qp_ctrl.davidson_target_excitations,
                    &data.energy_diag,
                    initial_guess,
                    &cfg,
                )
            };
            let eigenpairs = filter_eigenpairs_range(eigenpairs, &qp_ctrl);
            println!("\nUnrestricted BSE (TDA={}) Davidson eigenvalues in [{:.6}, {:.6}] Ha: {}",
                qp_ctrl.bse_tda, qp_ctrl.bse_eigenrange_min, qp_ctrl.bse_eigenrange_max, eigenpairs.len());
            for (n,(e,_v)) in eigenpairs.iter().enumerate() {
                println!("#{} Excitation energy={} Ha = {:.6} eV", n, e, e*crate::constants::EV);
            }
            all_excitations.extend(eigenpairs.iter().map(|(e, _)| *e));
            if let Some((first,_)) = eigenpairs.first() {
                println!("The first unrestricted BSE excitation is {} Ha ({:.6} eV)", first, first*crate::constants::EV);
                if qp_ctrl.save_first_excitation {
                    if let Ok(mut file) = OpenOptions::new().append(true).create(true).open(qp_ctrl.save_first_excitation_path.clone()) {
                        writeln!(file, "{}", first).unwrap();
                    }
                }
            }
            if qp_ctrl.save_bse_excitations {
                let line = eigenpairs.iter().map(|(e,_)| e.to_string()).collect::<Vec<_>>().join(",");
                if let Ok(mut file) = OpenOptions::new().append(true).create(true).open("bse_excitations.txt") {
                    writeln!(file, "{}", line).unwrap();
                }
            }
        }
    } else if qp_ctrl.bse_feast_solver {
        let data = prepare_unrestricted_matvec_data(scf_data, &inverse_dielectric, &qp_energies);
        // Single spin-coupled channel (previously a loop over spin modes).
        {
            let eigenpairs = feast_solve_bse_unrestricted(scf_data, &qp_ctrl, &qp_energies, &data);
            let eigenpairs = filter_eigenpairs_range(eigenpairs, &qp_ctrl);
            println!("\nUnrestricted BSE (TDA={}) FEAST eigenvalues in [{:.6}, {:.6}] Ha: {}",
                qp_ctrl.bse_tda, qp_ctrl.bse_eigenrange_min, qp_ctrl.bse_eigenrange_max, eigenpairs.len());
            for (n,(e,_v)) in eigenpairs.iter().enumerate() {
                println!("#{} Excitation energy={} Ha = {:.6} eV", n, e, e*crate::constants::EV);
            }
            all_excitations.extend(eigenpairs.iter().map(|(e, _)| *e));
            if let Some((first,_)) = eigenpairs.first() {
                println!("The first unrestricted BSE excitation is {} Ha ({:.6} eV)", first, first*crate::constants::EV);
                if qp_ctrl.save_first_excitation {
                    if let Ok(mut file) = OpenOptions::new().append(true).create(true).open(qp_ctrl.save_first_excitation_path.clone()) {
                        writeln!(file, "{}", first).unwrap();
                    }
                }
            }
            if qp_ctrl.save_bse_excitations {
                let line = eigenpairs.iter().map(|(e,_)| e.to_string()).collect::<Vec<_>>().join(",");
                if let Ok(mut file) = OpenOptions::new().append(true).create(true).open("bse_excitations.txt") {
                    writeln!(file, "{}", line).unwrap();
                }
            }
        }
    } else {
        // Single spin-coupled channel (previously a loop over spin modes).
        {
            let eigenpairs = if qp_ctrl.bse_tda {
                let a = construct_u_submat_a(scf_data, &inverse_dielectric, &qp_energies);
                solve_dense_eigenpairs(&a, &qp_ctrl)
            } else {
                let h = construct_u_full_bse_hamiltonian(scf_data, &inverse_dielectric, &qp_energies);
                solve_dense_eigenpairs(&h, &qp_ctrl)
            };
            println!("\nUnrestricted BSE (TDA={}) eigenvalues in [{:.6}, {:.6}] Ha: {}",
                qp_ctrl.bse_tda, qp_ctrl.bse_eigenrange_min, qp_ctrl.bse_eigenrange_max, eigenpairs.len());
            for (n,(e,_v)) in eigenpairs.iter().enumerate() {
                println!("#{} Excitation energy={} Ha = {:.6} eV", n, e, e*crate::constants::EV);
            }
            all_excitations.extend(eigenpairs.iter().map(|(e, _)| *e));
            if let Some((first,_)) = eigenpairs.first() {
                println!("The first unrestricted BSE excitation is {} Ha ({:.6} eV)", first, first*crate::constants::EV);
                if qp_ctrl.save_first_excitation {
                    if let Ok(mut file) = OpenOptions::new().append(true).create(true).open(qp_ctrl.save_first_excitation_path.clone()) {
                        writeln!(file, "{}", first).unwrap();
                    }
                }
            }
            if qp_ctrl.save_bse_excitations {
                let line = eigenpairs.iter().map(|(e,_)| e.to_string()).collect::<Vec<_>>().join(",");
                if let Ok(mut file) = OpenOptions::new().append(true).create(true).open("bse_excitations.txt") {
                    writeln!(file, "{}", line).unwrap();
                }
            }
        }
    }
    BseOutput {
        first_excitation: all_excitations.first().copied(),
        excitation_energies: all_excitations,
    }
}

pub fn zip_and_sort(
    eigenvalues: &Vec<f64>,
    eigenvectors: &MatrixFull<f64>
    ) -> Vec<(f64,Vec<f64>)> {
    let mut eigens:Vec<(f64,Vec<f64>)>=eigenvalues.iter().zip(eigenvectors.iter_columns_full()).map(|(e,v)|(*e,v.to_vec())).collect();
    eigens.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    eigens
}
pub fn promoted_density_analysis(x:&Vec<f64>,y:&Vec<f64>,occ_size:usize,vir_size:usize){
    let x_mat=MatrixFull::from_vec([occ_size,vir_size],x.clone()).unwrap();
    let y_mat=MatrixFull::from_vec([occ_size,vir_size],y.clone()).unwrap();

    let mut d_h=MatrixFull::new([occ_size,occ_size],0.0);
    _dgemm_full(&x_mat,'N',&x_mat,'T',&mut d_h,1.0,0.0);
    let mut tmp=MatrixFull::new([occ_size,occ_size],0.0);
    _dgemm_full(&y_mat,'N',&y_mat,'T',&mut tmp,1.0,0.0);
    d_h.iter_mut().zip(tmp.iter()).for_each(|(d,t)| *d+=*t);

    let mut d_p=MatrixFull::new([vir_size,vir_size],0.0);
    _dgemm_full(&x_mat,'T',&x_mat,'N',&mut d_p,1.0,0.0);
    tmp=MatrixFull::new([vir_size,vir_size],0.0);
    _dgemm_full(&y_mat,'T',&y_mat,'N',&mut tmp,1.0,0.0);
    d_p.iter_mut().zip(tmp.iter()).for_each(|(d,t)| *d+=*t);

    let (eigvecs_h,eigvals_h,_)=_dsyevd(&d_h,'V');
    let (eigvecs_p,eigvals_p,_)=_dsyevd(&d_p,'V');
    let eigvecs_h=eigvecs_h.unwrap();
    let eigvecs_p=eigvecs_p.unwrap();
    let trace_h: f64 = eigvals_h.iter().sum();
    let trace_p: f64 = eigvals_p.iter().sum();

    println!("Promoted Electron Number: Dh={:.6}, Dp={:.6}", trace_h*2.0, trace_p*2.0);

    let num_h = eigvals_h.len();
    let num_p = eigvals_p.len();
    let mut h_indices: Vec<usize> = (0..num_h).collect();
    h_indices.sort_by(|&a,&b| eigvals_h[b].partial_cmp(&eigvals_h[a]).unwrap());
    let mut p_indices: Vec<usize> = (0..num_p).collect();
    p_indices.sort_by(|&a,&b| eigvals_p[b].partial_cmp(&eigvals_p[a]).unwrap());

    let max_ch=5;
    for ch in 0..max_ch.min(num_h) {
        let k = h_indices[ch];
        let lambda = eigvals_h[k];
        if lambda < 1e-8 { continue; }
        let contribution = if trace_h > 1e-12 { lambda / trace_h } else { 0.0 };
        let vec_slice = &eigvecs_h.data[k * occ_size..(k + 1) * occ_size];
        let mut pairs: Vec<(usize, f64)> = vec_slice.iter()
            .enumerate()
            .filter(|(_, &amp)| amp.abs() > 0.1)
            .map(|(i, &amp)| (i, amp))
            .collect();
        pairs.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
        print!("      Detachment Channel #{}: contribution={:.4}, leading MOs(only including amplitudes greater than 0.1): ", ch + 1, contribution);
        for (idx, (mo, amp)) in pairs.iter().enumerate() {
            if idx > 0 { print!(", "); }
            print!("#{}({:.2})", mo + 1, amp);
        }
        println!();
    }

    for ch in 0..max_ch.min(num_p) {
        let k = p_indices[ch];
        let lambda = eigvals_p[k];
        if lambda < 1e-8 { continue; }
        let contribution = if trace_p > 1e-12 { lambda / trace_p } else { 0.0 };
        let vec_slice = &eigvecs_p.data[k * vir_size..(k + 1) * vir_size];
        let mut pairs: Vec<(usize, f64)> = vec_slice.iter()
            .enumerate()
            .filter(|(_, &amp)| amp.abs() > 0.1)
            .map(|(a, &amp)| (a, amp))
            .collect();
        pairs.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
        print!("      Attachment Channel #{}: contribution={:.4}, leading MOs(only including amplitudes greater than 0.1): ", ch + 1, contribution);
        for (idx, (mo, amp)) in pairs.iter().enumerate() {
            if idx > 0 { print!(", "); }
            print!("#{}({:.2})", occ_size + mo + 1, amp);
        }
        println!();
    }
}

pub fn leading_components(eigenvector: &Vec<f64>,occ_size:usize, vir_size: usize, print_nto: bool,){
    let mut components: Vec<(usize, usize, f64)> = eigenvector
        .iter()
        .enumerate()
        .map(|(n, x)| {
            let mut index = n; // 从0开始的索引
            if index<occ_size*vir_size{
                let j = occ_size+index / occ_size;  // 整除
                let i = index % occ_size;  // 取余
                (i, j, *x)
            }else{
                index-=occ_size*vir_size;
                let j = occ_size+index / occ_size;  // 整除
                let i = index % occ_size;  // 取余
                (j, i, *x)
            }
        })
        .collect();
    
    components.sort_by(|a, b| {
        b.2.abs().partial_cmp(&a.2.abs())  // 降序排列（绝对值最大的在前）
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let length=components.len().min(5);
    for i in 0..length{
        println!("      #{}->#{},amplitude={}",components[i].0,components[i].1,components[i].2);
    }
    if !print_nto {
        return;
    }   
    println!("Primary Components (Natural Transition Orbitals) of this excitation:");
    if eigenvector.len()==occ_size*vir_size{
        let density_matrix=MatrixFull::from_vec([occ_size,vir_size],eigenvector.clone()).unwrap();
        let (holes,sigmas,electrons_t)=_dsvd(&density_matrix,'A','A');
        let electrons=electrons_t.transpose();
        for component in 0..3{
            println!("      sigma_{}={:.4}:",component,sigmas[component]);
            let hole=&holes.data[component*occ_size..(component+1)*occ_size].to_vec();
            let electron=&electrons.data[component*vir_size..(component+1)*vir_size].to_vec();
            let mut h_pairs:Vec<(usize,f64)>=hole.iter().enumerate().filter(|(_,&v)|v.abs()>0.1).map(|(i,&v)|(i+1,v)).collect();
            h_pairs.sort_by(|a,b|b.1.abs().partial_cmp(&a.1.abs()).unwrap());
            print!("        hole: ");
            h_pairs.iter().enumerate().for_each(|(idx,(mo,amp))|{if idx>0{print!(", ")};print!("#{}({:.2})",mo,amp)});
            println!();
            let mut e_pairs:Vec<(usize,f64)>=electron.iter().enumerate().filter(|(_,&v)|v.abs()>0.1).map(|(a,&v)|(occ_size+a+1,v)).collect();
            e_pairs.sort_by(|a,b|b.1.abs().partial_cmp(&a.1.abs()).unwrap());
            print!("        electron: ");
            e_pairs.iter().enumerate().for_each(|(idx,(mo,amp))|{if idx>0{print!(", ")};print!("#{}({:.2})",mo,amp)});
            println!();
        }
        promoted_density_analysis(eigenvector,&vec![0.0;occ_size*vir_size],occ_size,vir_size);
    }else{
        let x_vec=eigenvector[0..occ_size*vir_size].to_vec();
        let y_vec=eigenvector[occ_size*vir_size..occ_size*vir_size*2].to_vec();
        let xpy_vec=x_vec.iter().zip(y_vec.iter()).map(|(x_ia,y_ia)|x_ia+y_ia).collect();
        let density_matrix=MatrixFull::from_vec([occ_size,vir_size],xpy_vec).unwrap();
        let (holes,sigmas,electrons_t)=_dsvd(&density_matrix,'A','A');
        let electrons=electrons_t.transpose();
        for component in 0..3{
            println!("      sigma_{}={:.4}:",component,sigmas[component]);
            let hole=&holes.data[component*occ_size..(component+1)*occ_size].to_vec();
            let electron=&electrons.data[component*vir_size..(component+1)*vir_size].to_vec();
            let mut h_pairs:Vec<(usize,f64)>=hole.iter().enumerate().filter(|(_,&v)|v.abs()>0.1).map(|(i,&v)|(i+1,v)).collect();
            h_pairs.sort_by(|a,b|b.1.abs().partial_cmp(&a.1.abs()).unwrap());
            print!("        hole: ");
            h_pairs.iter().enumerate().for_each(|(idx,(mo,amp))|{if idx>0{print!(", ")};print!("#{}({:.2})",mo,amp)});
            println!();
            let mut e_pairs:Vec<(usize,f64)>=electron.iter().enumerate().filter(|(_,&v)|v.abs()>0.1).map(|(a,&v)|(occ_size+a+1,v)).collect();
            e_pairs.sort_by(|a,b|b.1.abs().partial_cmp(&a.1.abs()).unwrap());
            print!("        electron: ");
            e_pairs.iter().enumerate().for_each(|(idx,(mo,amp))|{if idx>0{print!(", ")};print!("#{}({:.2})",mo,amp)});
            println!();
        }
        promoted_density_analysis(&x_vec,&y_vec,occ_size,vir_size);
    }


}
pub fn show_all_eigenpairs(eigenpairs:&Vec<(f64,Vec<f64>)>){
    eigenpairs.iter().for_each(|(val,vec)|println!("eigenvalue:{},eigenvector:{:#?}",val,vec))
}
