use std::ffi::{c_double, c_int, c_char};



//#[link(name="s-dftd3")]
//extern "C" {
//    pub fn calc_dftd3_atm_rest_(
//        num: *const c_int,
//        num_size: *const c_int,
//        xyz: *const c_double,
//        charge: *const c_double,
//        uhf: *const c_int,
//        method: *const c_char,
//        method_len: *const c_int,
//        energy: *mut c_double,
//        gradient: *mut c_double,
//        sigma: *mut c_double,
//        corr: *const c_char,
//        corr_len: *const c_int,
//    );
//
//}
//
//#[link(name="dftd4")]
//extern "C" {
//    pub fn calc_dftd4_rest_(
//        num: *const c_int,
//        num_size: *const c_int,
//        xyz: *const c_double,
//        charge: *const c_double,
//        uhf: *const c_int,
//        method: *const c_char,
//        method_len: *const c_int,
//        energy: *mut c_double,
//        gradient: *mut c_double,
//        sigma: *mut c_double,
//    );
//}

#[link(name="dftd3_rest")]
extern "C" {
    pub fn calc_dftd3_atm_rest_(
        num: *const c_int,
        num_size: *const c_int,
        xyz: *const c_double,
        charge: *const c_double,
        uhf: *const c_int,
        method: *const c_char,
        method_len: *const c_int,
        energy: *mut c_double,
        gradient: *mut c_double,
        sigma: *mut c_double,
        corr: *const c_char,
        corr_len: *const c_int,
    );

}

#[link(name="dftd4_rest")]
extern "C" {
    pub fn calc_dftd4_rest_(
        num: *const c_int,
        num_size: *const c_int,
        xyz: *const c_double,
        charge: *const c_double,
        uhf: *const c_int,
        method: *const c_char,
        method_len: *const c_int,
        energy: *mut c_double,
        gradient: *mut c_double,
        sigma: *mut c_double,
    );
}



