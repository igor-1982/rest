#![allow(unused)]
use std::convert::TryInto;
use std::os::raw::c_int;
use std::os::raw::c_double;
use std::ffi::CStr;
use std::mem::ManuallyDrop;

use rest_tensors::MatrixFull;

use self::ffi_xc::xc_func_info_type;

use super::Grids;

//use self::libxc::xc_func_type;
//use self::libxc::func_params_type;

pub(crate) mod ffi_xc;

#[derive(Clone,Debug)]
pub enum LibXCFamily {
    LDA,
    GGA,
    MGGA,
    HybridGGA,
    HybridMGGA,
    Unknown
}

#[derive(Clone,Debug)]
pub struct XcFuncType {
    xc_func_type: *mut ffi_xc::xc_func_type,
    pub xc_func_info_type: *const ffi_xc::xc_func_info_type,
    pub xc_func_family: LibXCFamily,
    spin_channel: usize,
}
//pub struct XcFuncInfoType {
//    xc_func_info_type: Option<*const ffi_xc::xc_func_info_type>,
//    x_func_info_type: Option<*const ffi_xc::xc_func_info_type>,
//    c_func_info_type: Option<*const ffi_xc::xc_func_info_type>,
//}

impl XcFuncType {

    pub fn xc_version(&self) {
        let mut vmajor:c_int = 0;
        let mut vminor:c_int = 0;
        let mut vmicro:c_int = 0;
        unsafe{ffi_xc::xc_version(&mut  vmajor, &mut vminor, &mut vmicro)};
        println!("Libxc version: {}.{}.{}", vmajor, vminor, vmicro);
    }

    pub fn xc_func_init(func_id: usize, spin_channel: usize) -> XcFuncType {
        let mut xc_func_type = unsafe{ffi_xc::xc_func_alloc()};
        let init = unsafe{ffi_xc::xc_func_init(
            xc_func_type,
            func_id as c_int, 
            spin_channel as c_int)};

        let xc_func_info_type = unsafe{ffi_xc::xc_func_get_info(xc_func_type)};

        //let xc_func_info_type = match xc_func_type[0] {
        //    Some(xc_func_type) => {Some(unsafe{ffi_xc::xc_func_get_info(xc_func_type)})},
        //    None => None
        //};
        //let x_func_info_type = match xc_func_type[1] {
        //    Some(xc_func_type) => {Some(unsafe{ffi_xc::xc_func_get_info(xc_func_type)})},
        //    None => None
        //};
        //let c_func_info_type = match xc_func_type[2] {
        //    Some(xc_func_type) => {Some(unsafe{ffi_xc::xc_func_get_info(xc_func_type)})},
        //    None => None
        //};
        //let xc_func_info_type = [xc_func_info_type,x_func_info_type,c_func_info_type];


        let xc_func_family = XcFuncType::get_family_enum(xc_func_info_type);
        //let xc_func_family = match xc_func_info_type[0] {
        //    Some(xc_func_info_type) => {Some(XcFuncType::get_family_enum(xc_func_info_type))},
        //    None => None
        //};
        //let x_func_family = match xc_func_info_type[1] {
        //    Some(xc_func_info_type) => {Some(XcFuncType::get_family_enum(xc_func_info_type))},
        //    None => None
        //};
        //let c_func_family = match xc_func_info_type[2] {
        //    Some(xc_func_info_type) => {Some(XcFuncType::get_family_enum(xc_func_info_type))},
        //    None => None
        //};
        //let xc_func_family = [xc_func_family, x_func_family,c_func_family];

        XcFuncType {
            xc_func_type,
            xc_func_info_type,
            xc_func_family,
            spin_channel 
        }
    }

    //pub fn xc_func_init_fdqc(name: &str, spin_channel: usize) -> XcFuncType {
    //    let lower_name = name.to_lowercase();
    //    let xc_code: (usize, usize,usize) = XcFuncType::xc_code_fdqc(name);
    //    XcFuncType::xc_func_init(xc_code, spin_channel)
    //}


    pub fn xc_code_fdqc(name: &str) -> (usize,usize,usize) {
        let lower_name = name.to_lowercase();
        // for a list of exchange-correlation functionals
        if lower_name.eq(&"hf".to_string()) {
            (0,0,0)
        } else if lower_name.eq(&"svwn".to_string()) {
            (0,1,7)
        } else if lower_name.eq(&"svwn-rpa".to_string()) {
            (0,1,8)
        } else if lower_name.eq(&"pz-lda".to_string()) {
            (0,1,9)
        } else if lower_name.eq(&"pw-lda".to_string()) {
            (0,1,12)
        } else if lower_name.eq(&"blyp".to_string()) {
            (0,106,131)
        } else if lower_name.eq(&"xlyp".to_string()) {
            (166,0,0)
        } else if lower_name.eq(&"pbe".to_string()) {
            (0,101,130)
        } else if lower_name.eq(&"xpbe".to_string()) {
            (0,123,136)
        } else if lower_name.eq(&"b3lyp".to_string()) {
            (402,0,0)
        } else if lower_name.eq(&"x3lyp".to_string()) {
            (411,0,0)
        // for a list of exchange functionals
        } else if lower_name.eq(&"lda_x_slater".to_string()) {
            (0,1,0)
        } else if lower_name.eq(&"gga_x_b88".to_string()) {
            (0,106,0)
        } else if lower_name.eq(&"gga_x_pbe".to_string()) {
            (0,101,0)
        } else if lower_name.eq(&"gga_x_xpbe".to_string()) {
            (0,123,0)
        // for a list of correlation functionals
        } else if lower_name.eq(&"lda_c_vwn".to_string()) {
            (0,0,7)
        } else if lower_name.eq(&"lda_c_vwn_rpa".to_string()) {
            (0,0,8)
        } else if lower_name.eq(&"gga_c_lyp".to_string()) {
            (0,0,131)
        } else if lower_name.eq(&"gga_c_pbe".to_string()) {
            (0,0,130)
        } else if lower_name.eq(&"gga_c_xpbe".to_string()) {
            (0,0,136)
        } else {
            (0,0,0)
        }
    }

    pub fn xc_func_end(&mut self) {
        unsafe{ffi_xc::xc_func_end(self.xc_func_type)}
    }

    pub fn get_family_name(&self) -> String {
        match self.xc_func_family {
            LibXCFamily::LDA => "LDA".to_string(),
            LibXCFamily::GGA => "GGA".to_string(),
            LibXCFamily::MGGA => "MGGA".to_string(),
            LibXCFamily::HybridGGA => "HybridGGA".to_string(),
            LibXCFamily::HybridMGGA => "HybridMGGA".to_string(),
            LibXCFamily::Unknown => "Unknown DFA".to_string(),
        }
    }

    //pub fn is_dfa(&self) -> bool {
    //    self.xc_func_type.iter().fold(false, |acc,xc_func_type| {
    //        match xc_func_type {
    //            Some(_) => {acc || true},
    //            None => {acc || false},
    //        }
    //    })
    //}
    pub fn use_density_gradient(&self) -> bool {
        match self.xc_func_family {
            LibXCFamily::LDA => {false},
            _ => {true},
        }
    }
    pub fn use_kinetic_density(&self) -> bool {
        match self.xc_func_family {
            LibXCFamily::MGGA => {true},
            LibXCFamily::HybridMGGA => {true},
            _ => {false},
        }
    }

    pub fn use_exact_exchange(&self) -> bool {
        match self.xc_func_family {
            LibXCFamily::HybridGGA => {true},
            LibXCFamily::HybridMGGA => {true},
            _ => {false},
        }
    }

    pub fn is_lda(&self) -> bool {
        match self.xc_func_family {
            LibXCFamily::LDA => true,
            _ => false
        }
    }

    pub fn is_gga(&self) -> bool {
        match self.xc_func_family {
            LibXCFamily::GGA => true,
            _ => false
        }
    }

    pub fn is_mgga(&self) -> bool {
        match self.xc_func_family {
            LibXCFamily::MGGA => true,
            _ => false
        }
    }

    pub fn is_hybrid_gga(&self) -> bool {
        match self.xc_func_family {
            LibXCFamily::HybridGGA => true,
            _ => false
        }
    }

    pub fn is_hybrid_mgga(&self) -> bool {
        match self.xc_func_family {
            LibXCFamily::HybridMGGA => true,
            _ => false
        }
    }

    pub fn xc_hyb_exx_coeff(&self) -> f64 {
        unsafe{ffi_xc::xc_hyb_exx_coef(self.xc_func_type)}
    }

    pub fn lda_exc(&self, rho: &[f64]) -> Vec<f64> {
        let length = rho.len()/&self.spin_channel;
        //println!("debug rho length: {}",length);
        let mut exc = vec![0.0; length];
        unsafe{
            ffi_xc::xc_lda_exc(
                self.xc_func_type,
                length as u64,
                rho.as_ptr(),
                exc.as_mut_ptr());
        }
        exc
    }

    pub fn gga_exc(&self, rho: &[f64], sigma: &[f64]) -> Vec<f64> {
        let length = rho.len()/&self.spin_channel;
        let mut exc = vec![0.0; length];
        unsafe{
            ffi_xc::xc_gga_exc(
                self.xc_func_type,
                length as u64,
                rho.as_ptr(),
                sigma.as_ptr(),
                exc.as_mut_ptr(),
            );
        }
        exc
    }

    pub fn lda_exc_vxc(&self, rho: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let length = rho.len()/&self.spin_channel;
        //println!("debug rho length: {}",length);
        let mut exc = vec![0.0; length];
        let mut vrho = vec![0.0; length*&self.spin_channel];
        unsafe{
            ffi_xc::xc_lda_exc_vxc(
                self.xc_func_type,
                length as u64,
                rho.as_ptr(),
                exc.as_mut_ptr(),
                vrho.as_mut_ptr());
        }
        (exc,vrho)
    }

    pub fn gga_exc_vxc(&self, rho: &[f64], sigma: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let length = rho.len()/&self.spin_channel;
        let mut exc = vec![0.0; length];
        let mut vrho = vec![0.0; length*&self.spin_channel];
        let mut vsigma = if self.spin_channel == 1 {
            vec![0.0; length]
        } else {
            vec![0.0; length*3]
        };
        unsafe{
            ffi_xc::xc_gga_exc_vxc(
                self.xc_func_type,
                length as u64,
                rho.as_ptr(),
                sigma.as_ptr(),
                exc.as_mut_ptr(),
                vrho.as_mut_ptr(),
         vsigma.as_mut_ptr()
            );
        }
        (exc,vrho,vsigma)
    }

    // xc_func_info relevant functions:
    pub fn get_family_name_std(family_id: u32) -> String {
        if family_id == ffi_xc::XC_FAMILY_LDA {
            "LDA".to_string()
        } else if family_id == ffi_xc::XC_FAMILY_GGA {
            "GGA".to_string()
        } else if family_id == ffi_xc::XC_FAMILY_MGGA {
            "MGGA".to_string()
        } else if family_id == ffi_xc::XC_FAMILY_HYB_GGA {
            "Hybrid GGA".to_string()
        } else if family_id == ffi_xc::XC_FAMILY_HYB_MGGA {
            "Hybrid MGGA".to_string()
        } else {
            "Unknown".to_string()
        }
    }

    pub fn get_family_id(xc_info: *const xc_func_info_type) -> u32 {
        unsafe {ffi_xc::xc_func_info_get_family(xc_info) as u32}
    }
    pub fn get_family_enum(xc_info: *const xc_func_info_type) -> LibXCFamily {
        let family_id = XcFuncType::get_family_id(xc_info);
        if family_id == ffi_xc::XC_FAMILY_LDA {
            LibXCFamily::LDA
        } else if family_id == ffi_xc::XC_FAMILY_GGA {
            LibXCFamily::GGA
        } else if family_id == ffi_xc::XC_FAMILY_MGGA {
            LibXCFamily::MGGA
        } else if family_id == ffi_xc::XC_FAMILY_HYB_GGA {
            LibXCFamily::HybridGGA
        } else if family_id == ffi_xc::XC_FAMILY_HYB_MGGA {
            LibXCFamily::HybridMGGA
        } else {
            LibXCFamily::Unknown
        }

    }

    pub fn printout_family_name_ref(xc_info: *const xc_func_info_type, x_or_c: &str) {
        let xc_family = {
            let tmp_i: u32 = XcFuncType::get_family_id(xc_info);
            XcFuncType::get_family_name_std(tmp_i)
        };
        let xc_name = unsafe {
            let c_buf = ffi_xc::xc_func_info_get_name(xc_info);
            let c_str = CStr::from_ptr(c_buf);
            let str_slice = c_str.to_str().unwrap();
            str_slice.to_owned()
        };
        println!("The {} functional '{}' belongs to the '{}' family and is defined in the reference(s):",x_or_c,xc_name,xc_family);
        (0..5).for_each(|i| unsafe{
            let c_ref = ffi_xc::xc_func_info_get_references(xc_info, i);
            if c_ref != std::ptr::null() {
                let x_ref = {
                    let c_buf = ffi_xc::xc_func_reference_get_ref(c_ref);
                    let c_str = CStr::from_ptr(c_buf);
                    let str_slice = c_str.to_str().unwrap_or_default();
                    str_slice.to_owned()
                };
                println!("({}): {}",i,x_ref);
            };
        });
    }
    pub fn xc_func_info_printout(&self) {
        //let x_or_c = ["exchange-correlation","exchange","correlation"];
        //self.xc_func_info_type.iter().zip(x_or_c.iter()).for_each(|(xc_func_info_type,x_or_c)| {
        //    if let Some(xc_info) = xc_func_info_type {
        //        XcFuncType::printout_family_name_ref(*xc_info, *x_or_c);
        //    }
        //});
        let functype = "density";
        XcFuncType::printout_family_name_ref(self.xc_func_info_type, functype);
    }
}

#[test]
fn test_libxc() {
    let rho:Vec<f64> = vec![0.1,0.2,0.3,0.4,0.5,0.6,0.8];
    let sigma:Vec<f64> = vec![0.2,0.3,0.4,0.5,0.6,0.7];
    //let mut exc:Vec<c_double> = vec![0.0,0.0,0.0,0.0,0.0];
    //let mut vrho:Vec<c_double> = vec![0.0,0.0,0.0,0.0,0.0];
    let func_id: usize = ffi_xc::XC_GGA_X_XPBE as usize;
    let spin_channel: usize = 1;

    let mut my_xc = XcFuncType::xc_func_init(1,spin_channel); 
    //let mut my_xc = XcFuncType::xc_func_init_fdqc(&"pw-lda",spin_channel); 

    my_xc.xc_version();

    my_xc.xc_func_info_printout();

    let (exc, vrho) = my_xc.lda_exc_vxc(&rho);

    println!("{:?}", exc);
    println!("{:?}", vrho);

    //let xc_info = my_xc.xc_func_get_info();

    //let xc_name = unsafe {
    //    let c_buf = ffi_xc::xc_func_info_get_name(xc_info);
    //    let c_str = CStr::from_ptr(c_buf);
    //    let str_slice = c_str.to_str().unwrap();
    //    str_slice.to_owned()
    //};
    //println!("{}",xc_name);


    //let xc_name = unsafe{String::from_raw_parts(xc_name, 10, 10)};

    

    //println!("{:?}",my_xc.);

    //let (exc,vrho) = my_xc.xc_exc_vxc(&rho, &sigma).unwrap();

    //println!("{:?}", exc.data);
    //println!("{:?}", vrho.data);


    //let mut p_xc_func_type = unsafe{ffi_xc::xc_func_alloc()};

    //let init = unsafe{ffi_xc::xc_func_init(p_xc_func_type,func_id, ffi_xc::XC_UNPOLARIZED as c_int)};

    //unsafe{
    //    let c_exc = (exc.as_mut_ptr(),exc.len(),exc.capacity());
    //    let c_vrho = (vrho.as_mut_ptr(),vrho.len(),vrho.capacity());
    //    ffi_xc::xc_lda_exc_vxc(p_xc_func_type,5,rho.as_ptr(),c_exc.0,c_vrho.0);
    //    exc = Vec::from_raw_parts(c_exc.0,c_exc.1,c_exc.2);
    //    vrho = Vec::from_raw_parts(c_vrho.0,c_vrho.1,c_vrho.2);
    //};
    //println!("{:?}", exc);
    //println!("{:?}", vrho);

    //unsafe{ffi_xc::xc_func_end(p_xc_func_type)};

    my_xc.xc_func_end()
}

