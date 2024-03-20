//extern crate rust_libecpint as libecpint;
//use libecpint::{get_ints, get_ints_test};
//use rest_libcint::CINTR2CDATA;
//use tensors::MatrixFull;
//
//use crate::molecule_io::Molecule;
//use crate::constants::ANG;
//
//
//
//pub fn generate_libecpint(mol: &Molecule) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<i32>, Vec<i32>) {
//
//    let basis_list = &mol.basis4elem;
//    let geom_list = &mol.geom;
//
//    let mut exps: Vec<f64> = vec![];
//    let mut coeffs: Vec<f64> = vec![];
//    let mut coords: Vec<f64> = vec![];
//    let mut ams: Vec<i32> = vec![];
//    let mut lens: Vec<i32> = vec![];
//
//    basis_list.iter().zip(geom_list.position.iter_columns_full()).for_each(|(basis, geom)| {
//        //ecpint.set_basis(basis, geom);
//        //let geom_prim = geom.iter().map(|x| x*ANG).collect::<Vec<f64>>();
//        //println!("geom: {:?}", &geom_prim);
//
//        basis.electron_shells.iter().for_each(|shell| {
//            let am = shell.angular_momentum[0] as std::os::raw::c_int;
//            let ep = &shell.exponents;
//
//            shell.native_coefficients.iter().for_each(|x| {
//                ams.push(am);
//                lens.push(x.len() as i32);
//                coords.extend(geom.iter().map(|x| x*ANG));
//                x.iter().enumerate().for_each(|(ix,y)| {
//                    let tmp_ep = ep[ix];
//                    //let tmp_value = CINTR2CDATA::gto_norm(am,tmp_ep);
//                    exps.push(tmp_ep);
//                    coeffs.push(*y);
//                });
//            });
//            //let ce = shell.coefficients.iter().map(|x| x*ANG).collect::<Vec<f64>>();
//        });
//    });
//
//
//    //println!("lens: {:?}", &lens);
//
//    //ints
//    //vec![]
//    (exps, coeffs, coords, ams, lens)
//
//}
//
//pub fn generate_libecpint_v2(mol: &Molecule, bs_pattern: (usize, usize)) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<i32>, Vec<i32>) {
//
//    let basis_list = &mol.basis4elem;
//    let geom_list = &mol.geom;
//
//    let mut exps: Vec<f64> = vec![];
//    let mut coeffs: Vec<f64> = vec![];
//    let mut coords: Vec<f64> = vec![];
//    let mut ams: Vec<i32> = vec![];
//    let mut lens: Vec<i32> = vec![];
//
//    basis_list.iter().zip(geom_list.position.iter_columns_full()).for_each(|(basis, geom)| {
//        //ecpint.set_basis(basis, geom);
//        //let geom_prim = geom.iter().map(|x| x*ANG).collect::<Vec<f64>>();
//        //println!("geom: {:?}", &geom_prim);
//
//        basis.electron_shells.iter().for_each(|shell| {
//            let am = shell.angular_momentum[0] as std::os::raw::c_int;
//            let ep = &shell.exponents;
//
//            shell.native_coefficients.iter().for_each(|x| {
//                ams.push(am);
//                lens.push(x.len() as i32);
//                coords.extend(geom.iter().map(|x| x*ANG));
//                x.iter().enumerate().for_each(|(ix,y)| {
//                    let tmp_ep = ep[ix];
//                    //let tmp_value = CINTR2CDATA::gto_norm(am,tmp_ep);
//                    exps.push(tmp_ep);
//                    coeffs.push(*y);
//                });
//            });
//            //let ce = shell.coefficients.iter().map(|x| x*ANG).collect::<Vec<f64>>();
//        });
//    });
//
//
//    //println!("lens: {:?}", &lens);
//
//    //ints
//    //vec![]
//    (exps, coeffs, coords, ams, lens)
//
//}
//pub fn run_ecpint(mol: &Molecule) -> Vec<f64> {
//    let num_basis = mol.num_basis;
//
//    //let ecpints = libecpint::test_ints();
//    //println!("{:?}", &ecpints);
//
//    let (exps,coeffs, coords, ams, lens) = generate_libecpint(mol);
//
//    let share_dir = "/usr/share/libecpint".to_string();
//    let ecp_name = "ecp28mdf02".to_string();
//    let ecp_raw = get_ints(share_dir, ecp_name, exps.clone(), coeffs.clone(), coords.clone(), ams, lens);
//    //println!("ecp_int_1: {:?}", &ecp_raw);
//
//    let ecp_int_1 = MatrixFull::from_vec([num_basis+2, num_basis+2],ecp_raw).unwrap();
//
//    ecp_int_1.formated_output_e_with_threshold(5, "full", 1.0E-5);
//    
//
//    //let share_dir = "/usr/share/libecpint".to_string();
//    //let ecp_name = "ecp28mdf".to_string();
//    //let (ecp_int_2, o_exps, o_coeffs, o_coords) = get_ints_test(share_dir);
//
//    
//    ////println!("ecp_int_2: {:?}", &ecp_int_2);
//
//    //println!("start int comparison");
//    //ecp_int_1.iter().zip(ecp_int_2.iter()).for_each(|(x, y)| {
//    //    if (x - y).abs() > 1e-4 {
//    //        println!("ints diff: {:?} {:?}", x, y);
//    //    }
//    //});
//
//    //println!("start exps comparison");
//    //exps.iter().zip(o_exps.iter()).for_each(|(x, y)| {
//    //    if (x - y).abs() > 1e-8 {
//    //        println!("exps diff: {:?} {:?}", x, y);
//    //    }
//    //});
//
//    //println!("start coeffs comparison");
//    //coeffs.iter().zip(o_coeffs.iter()).for_each(|(x, y)| {
//    //    if (x - y).abs() > 1e-8 {
//    //        println!("coeffs diff: {:?} {:?}", x, y);
//    //    }
//    //});
//
//    //println!("start coords comparison");
//    //coords.iter().zip(o_coords.iter()).for_each(|(x, y)| {
//    //    if (x - y).abs() > 1e-8 {
//    //        println!("coords diff: {:?} {:?}", x, y);
//    //    }
//    //});
//
//    //let mut ecpint = ffi::ECPIntWrapper::new("/usr/share/libecpint").within_box();
//    //let ints = ecpint.as_mut().get_integrals();
//    //println!("length: {}", ints.len());
//    //println!("{:?}", ints);
//
//    //let basis_list = &mol.basis4elem;
//    //let geom_list = &mol.geom;
//
//    //basis_list.iter().zip(geom_list.position.iter_columns_full()).for_each(|(basis, geom)| {
//    //    //ecpint.as_mut().set_basis(basis, geom);
//    //    println!("geom: {:?}", geom);
//    //    println!("basis: {:?}", basis);
//    //});
//
//    //ecpints
//    vec![]
//    
//}