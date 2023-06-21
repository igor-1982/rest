use tensors::{MatrixFull, MatrixFullSlice};

use crate::basis_io::basic_math::specific_double_factorial;

use super::DMatrix;

#[inline]
pub fn prepare_basinfo(l:i32) -> MatrixFull<f64> {
    let num_bas = ((l+1)*(l+2)/2) as usize;
    let norm0 = 1.0;

    //let mut tmp_v: Vec<f64> = vec![];
    let mut basinfo = MatrixFull::new([4,num_bas],0.0_f64);
    //let mut i_bas:usize = 0;
    //let mut lf = [0.0,0.0,0.0];
    let mut i_bas:usize = 0;
    for lx in (0..=l as i32).rev() {
        //let normx = double_factorial((2*lx) as i32-1) as f64;
        let normx = specific_double_factorial(2*lx-1);
        let lxf = lx as f64;
        //lf[0] = lx as f64;
        let rl = l as i32 - lx;
        for ly in (0..=rl).rev() {
            let lz = rl - ly;
            let normy = specific_double_factorial(2*ly-1);
            let normz = specific_double_factorial(2*lz-1);
            let norm  = norm0/(normx*normy*normz).sqrt();
            let tmp_ibas = basinfo.slice_column_mut(i_bas);
            tmp_ibas[0] = lxf;
            tmp_ibas[1] = ly as f64;
            tmp_ibas[2] = lz as f64;
            tmp_ibas[3] = norm;
            i_bas += 1;
        }
    }
    basinfo
}

#[test]
fn test_prepare_basinfo() {
    let tmp_matr=prepare_basinfo(0);
    println!("l=0: {:?},{:?}", tmp_matr.size,tmp_matr.data);
    let tmp_matr=prepare_basinfo(1);
    println!("l=1: {:?},{:?}", tmp_matr.size,tmp_matr.data);
    let tmp_matr=prepare_basinfo(2);
    println!("l=2: {:?},{:?}", tmp_matr.size,tmp_matr.data);
    let tmp_matr=prepare_basinfo(3);
    println!("l=3: {:?},{:?}", tmp_matr.size,tmp_matr.data);
    let tmp_matr=prepare_basinfo(4);
    println!("l=4: {:?},{:?}", tmp_matr.size,tmp_matr.data);
    let tmp_matr=prepare_basinfo(5);
    println!("l=5: {:?},{:?}", tmp_matr.size,tmp_matr.data);
    let tmp_matr=prepare_basinfo(6);
    println!("l=6: {:?},{:?}", tmp_matr.size,tmp_matr.data);
}


pub type DMatrix4x1 = DMatrix<4>;
pub type DMatrix4x3 = DMatrix<12>;
pub type DMatrix4x6 = DMatrix<24>;
pub type DMatrix4x10 = DMatrix<40>;
pub type DMatrix4x15 = DMatrix<60>;
pub type DMatrix4x21 = DMatrix<84>;
pub type DMatrix4x28 = DMatrix<112>;

pub enum CarBasInfo {
    L0(DMatrix4x1),
    L1(DMatrix4x3),
    L2(DMatrix4x6),
    L3(DMatrix4x10),
    L4(DMatrix4x15),
    L5(DMatrix4x21),
    L6(DMatrix4x28),
}
//
//pub struct DMatrix4x1 {
//    size: [usize;2],
//    indicing: [usize;2],
//    data: [f64;4]
//}

/// normalization factor for gaussian-type orbital
///    ((2*lx-1)!!(2*ly-1)!!(2*lz-1)!!)^{-1/2}
pub const CAR_BAS_INFO_L0: CarBasInfo = CarBasInfo::L0(DMatrix4x1 {
    size: [4,1], 
    indicing: [1,4], 
    data: [0.0, 0.0, 0.0, 1.0] 
});

pub const CAR_BAS_INFO_L1: CarBasInfo = CarBasInfo::L1(DMatrix4x3 {
    size: [4,3], 
    indicing: [1,4], 
    data: [1.0, 0.0, 0.0, 1.0, 
           0.0, 1.0, 0.0, 1.0, 
           0.0, 0.0, 1.0, 1.0] 
});

pub const CAR_BAS_INFO_L2: CarBasInfo = CarBasInfo::L2(DMatrix4x6 {
    size: [4,6], 
    indicing: [1,4], 
    data:  [2.0, 0.0, 0.0, 0.5773502691896258, 
            1.0, 1.0, 0.0, 1.0, 
            1.0, 0.0, 1.0, 1.0, 
            0.0, 2.0, 0.0, 0.5773502691896258, 
            0.0, 1.0, 1.0, 1.0, 
            0.0, 0.0, 2.0, 0.5773502691896258]
});
pub const CAR_BAS_INFO_L3: CarBasInfo = CarBasInfo::L3(DMatrix4x10 {
    size: [4,6], 
    indicing: [1,4], 
    data: [3.0, 0.0, 0.0, 0.2581988897471611, 
           2.0, 1.0, 0.0, 0.5773502691896258, 
           2.0, 0.0, 1.0, 0.5773502691896258, 
           1.0, 2.0, 0.0, 0.5773502691896258, 
           1.0, 1.0, 1.0, 1.0, 
           1.0, 0.0, 2.0, 0.5773502691896258, 
           0.0, 3.0, 0.0, 0.2581988897471611, 
           0.0, 2.0, 1.0, 0.5773502691896258, 
           0.0, 1.0, 2.0, 0.5773502691896258, 
           0.0, 0.0, 3.0, 0.2581988897471611]
});
pub const CAR_BAS_INFO_L4: CarBasInfo = CarBasInfo::L4(DMatrix4x15 {
    size: [4,15], 
    indicing: [1,4], 
    data: [4.0, 0.0, 0.0, 0.09759000729485333, 
           3.0, 1.0, 0.0, 0.2581988897471611, 
           3.0, 0.0, 1.0, 0.2581988897471611, 
           2.0, 2.0, 0.0, 0.3333333333333333, 
           2.0, 1.0, 1.0, 0.5773502691896258, 
           2.0, 0.0, 2.0, 0.3333333333333333, 
           1.0, 3.0, 0.0, 0.2581988897471611, 
           1.0, 2.0, 1.0, 0.5773502691896258, 
           1.0, 1.0, 2.0, 0.5773502691896258, 
           1.0, 0.0, 3.0, 0.2581988897471611, 
           0.0, 4.0, 0.0, 0.09759000729485333,
           0.0, 3.0, 1.0, 0.2581988897471611, 
           0.0, 2.0, 2.0, 0.3333333333333333, 
           0.0, 1.0, 3.0, 0.2581988897471611, 
           0.0, 0.0, 4.0, 0.09759000729485333]
});
pub const CAR_BAS_INFO_L5: CarBasInfo = CarBasInfo::L5(DMatrix4x21 {
    size: [4,15], 
    indicing: [1,4], 
    data: [5.0, 0.0, 0.0, 0.03253000243161777, 
           4.0, 1.0, 0.0, 0.09759000729485333, 
           4.0, 0.0, 1.0, 0.09759000729485333, 
           3.0, 2.0, 0.0, 0.14907119849998599, 
           3.0, 1.0, 1.0, 0.2581988897471611, 
           3.0, 0.0, 2.0, 0.14907119849998599, 
           2.0, 3.0, 0.0, 0.14907119849998599, 
           2.0, 2.0, 1.0, 0.3333333333333333, 
           2.0, 1.0, 2.0, 0.3333333333333333, 
           2.0, 0.0, 3.0, 0.14907119849998599, 
           1.0, 4.0, 0.0, 0.09759000729485333, 
           1.0, 3.0, 1.0, 0.2581988897471611, 
           1.0, 2.0, 2.0, 0.3333333333333333, 
           1.0, 1.0, 3.0, 0.2581988897471611, 
           1.0, 0.0, 4.0, 0.09759000729485333, 
           0.0, 5.0, 0.0, 0.03253000243161777, 
           0.0, 4.0, 1.0, 0.09759000729485333, 
           0.0, 3.0, 2.0, 0.14907119849998599, 
           0.0, 2.0, 3.0, 0.14907119849998599, 
           0.0, 1.0, 4.0, 0.09759000729485333, 
           0.0, 0.0, 5.0, 0.03253000243161777]
});
pub const CAR_BAS_INFO_L6: CarBasInfo = CarBasInfo::L6(DMatrix4x28 {
    size: [4,15], 
    indicing: [1,4], 
    data: [6.0, 0.0, 0.0, 0.009808164772274995, 
           5.0, 1.0, 0.0, 0.03253000243161777, 
           5.0, 0.0, 1.0, 0.03253000243161777, 
           4.0, 2.0, 0.0, 0.0563436169819011, 
           4.0, 1.0, 1.0, 0.09759000729485333, 
           4.0, 0.0, 2.0, 0.0563436169819011, 
           3.0, 3.0, 0.0, 0.06666666666666667, 
           3.0, 2.0, 1.0, 0.14907119849998599, 
           3.0, 1.0, 2.0, 0.14907119849998599, 
           3.0, 0.0, 3.0, 0.06666666666666667, 
           2.0, 4.0, 0.0, 0.0563436169819011, 
           2.0, 3.0, 1.0, 0.14907119849998599, 
           2.0, 2.0, 2.0, 0.19245008972987526, 
           2.0, 1.0, 3.0, 0.14907119849998599, 
           2.0, 0.0, 4.0, 0.0563436169819011, 
           1.0, 5.0, 0.0, 0.03253000243161777, 
           1.0, 4.0, 1.0, 0.09759000729485333, 
           1.0, 3.0, 2.0, 0.14907119849998599, 
           1.0, 2.0, 3.0, 0.14907119849998599, 
           1.0, 1.0, 4.0, 0.09759000729485333, 
           1.0, 0.0, 5.0, 0.03253000243161777, 
           0.0, 6.0, 0.0, 0.009808164772274995, 
           0.0, 5.0, 1.0, 0.03253000243161777, 
           0.0, 4.0, 2.0, 0.0563436169819011, 
           0.0, 3.0, 3.0, 0.06666666666666667, 
           0.0, 2.0, 4.0, 0.0563436169819011, 
           0.0, 1.0, 5.0, 0.03253000243161777, 
           0.0, 0.0, 6.0, 0.009808164772274995]
});

impl CarBasInfo {
    pub fn to_matrixfullslice(&self) -> MatrixFullSlice<f64> {
        match &self {
            CarBasInfo::L0(matr) => {
                MatrixFullSlice {
                    size: &matr.size,
                    indicing: &matr.indicing,
                    data: &matr.data
                }
            },
            CarBasInfo::L1(matr) => {
                MatrixFullSlice {
                    size: &matr.size,
                    indicing: &matr.indicing,
                    data: &matr.data
                }
            },
            CarBasInfo::L2(matr) => {
                MatrixFullSlice {
                    size: &matr.size,
                    indicing: &matr.indicing,
                    data: &matr.data
                }
            },
            CarBasInfo::L3(matr) => {
                MatrixFullSlice {
                    size: &matr.size,
                    indicing: &matr.indicing,
                    data: &matr.data
                }
            },
            CarBasInfo::L4(matr) => {
                MatrixFullSlice {
                    size: &matr.size,
                    indicing: &matr.indicing,
                    data: &matr.data
                }
            },
            CarBasInfo::L5(matr) => {
                MatrixFullSlice {
                    size: &matr.size,
                    indicing: &matr.indicing,
                    data: &matr.data
                }
            },
            CarBasInfo::L6(matr) => {
                MatrixFullSlice {
                    size: &matr.size,
                    indicing: &matr.indicing,
                    data: &matr.data
                }
            },
        }
    }
}

pub fn cartesian_gto_const(l:usize) -> crate::constants::cartesian_gto::CarBasInfo {
    if l == 0 {
        CAR_BAS_INFO_L0
    } else if l == 1 {
        CAR_BAS_INFO_L1
    } else if l == 2 {
        CAR_BAS_INFO_L2
    } else if l == 3 {
        CAR_BAS_INFO_L3
    } else if l == 4 {
        CAR_BAS_INFO_L4
    } else if l == 5 {
        CAR_BAS_INFO_L5
    } else if l == 6 {
        CAR_BAS_INFO_L6
    } else {
        panic!("No C2S transformation implementation for l > 6")
    }
}