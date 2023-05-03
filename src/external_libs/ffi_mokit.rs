use std::ffi::{c_double, c_int, c_char};

//#[link(name="rest2fch")]
extern "C" {
    pub fn rest2fch_(fchname: *const c_char,
        fchname_len: *const c_int,
        nbf: *const c_int,
        nif: *const c_int,
        coeff2: *const c_double,
        ab: *const c_char,
        ev: *const c_double,
        gen_density: *const c_int
    );
}