//! This mod provides methods to prune the number of angular grid.<br>
//! Currently supported pruning methods:
//! * SG-1 pruning[^1]
//! * NWChem pruning
//! 
//! [^1]: [P. M. W. Gill, B. G. Johnson, J. A. Pople. Chemical Physics Letters 209, 506-512 (1993)](https://doi.org/10.1016/0009-2614(93)80125-9).

use super::{parameters::{SG1RADII, BOHR, BRAGG0, LEBEDEV_NGRID}, atom::default_angular_num};

/// Standard Grid 1 according to _P. M. W. Gill, B. G. Johnson, J. A. Pople. Chemical Physics Letters 209, 506-512 (1993)_.<br>
/// Reference can be found [here](https://doi.org/10.1016/0009-2614(93)80125-9).
/// 
/// # Arguments:<br>
/// **nuc**:   Nuclear charge.<br>
/// **rads**:  Grid coordinates on radical axis with the size of the number of radial grids.<br>
/// **n_rad**: The number of radial grids.<br>
/// 
/// # Returns: <br>
/// A vector with the same length of n_rad containing the numbers of angular grids. <br>
/// with respect to every radial grid coordinates.<br>
pub fn sg1_prune (nuc: usize, rads: &Vec<f64>, n_rad: usize) -> Vec<usize> {

    let radii = &SG1RADII; //for elements before 4rd period
    let leb_ngrid = vec![6, 38, 86, 194, 86];
    let alphas = [[0.25, 0.5, 1.0, 4.5], [0.1667, 0.5, 0.9, 3.5], [0.1, 0.4, 0.8, 2.5]];
    let r_atom = radii[nuc];
    let mut place = vec![0usize; n_rad];

    if nuc < 2 {  //H, He
        place.iter_mut().zip(rads.iter()).for_each(|(place,rad)| {
            let mut judge = 0usize;
            alphas.get(0).unwrap().iter().for_each(|alpha| {
                if rad/r_atom > *alpha {
                    judge = judge + 1;
                }
            });
            *place = leb_ngrid[judge];
        });
    }
    else if nuc > 2 && nuc <= 10 {  // 2nd period
        place.iter_mut().zip(rads.iter()).for_each(|(place,rad)| {
            let mut judge = 0usize;
            alphas.get(1).unwrap().iter().for_each(|alpha| {
                if rad/r_atom > *alpha {
                    judge = judge + 1;
                }
            });
            *place = leb_ngrid[judge];
        });
    }
    else {  // 3nd period
        place.iter_mut().zip(rads.iter()).for_each(|(place,rad)| {
            let mut judge = 0usize;
            alphas.get(1).unwrap().iter().for_each(|alpha| {
                if rad/r_atom > *alpha {
                    judge = judge + 1;
                }
            });
            *place = leb_ngrid[judge];
        });
    }

    place


}

/// Pruning method provided by [NWChem](https://www.nwchem-sw.org/).
/// 
/// # Arguments:
/// **nuc**:   Nuclear charge.<br>
/// **rads**:  Grid coordinates on radical axis with the size of the number of radial grids.<br>
/// **n_ang**: Max number of grids over angular part.<br>
/// **n_rad**: The number of radial grids.<br>
/// 
/// # Returns: 
/// A vector with the same length of n_rad containing the numbers of angular grids 
/// with respect to every radial grid coordinates.
pub fn nwchem_prune(nuc: usize, rads: &Vec<f64>, n_ang: usize, n_rad: usize) -> Vec<usize> {

    let bragg: Vec<f64> = BRAGG0.iter().map(|item| *item/BOHR).collect();
    //let leb_ngrid = &LEBEDEV_NGRID[4..];
    let leb_ngrid = &LEBEDEV_NGRID[4..];
    //println!("leb_ngrid = {:?}", leb_ngrid);

    let alphas = [[0.25, 0.5, 1.0, 4.5], [0.1667, 0.5, 0.9, 3.5], [0.1, 0.4, 0.8, 2.5]];
    let radii = &bragg;

/*     
    let n_ang =    
    if LEBEDEV_NGRID.contains(&n_ang_input) {
        n_ang_input
    }
    else {
        let ang_num_new = get_closest_n_ang(n_ang_input);
        println!("Angular grid number not in Lebedev list, thus use {} as angular grid number.", ang_num_new);
        ang_num_new
    };
 */
    if n_ang < 50 {
        return vec![n_ang; n_rad]
    }

    let leb_l = if n_ang == 50 {
        vec![1, 2, 2, 2, 1]
    }
    else {
        let mut idx = 0;
        for leb_grid in leb_ngrid.iter(){
            if n_ang == *leb_grid {
                break;
            }
            idx += 1;
        }
        vec![1, 3, idx-1, idx, idx-1]
    };

    let mut place = vec![0usize; n_rad];
    let r_atom = radii[nuc] + 1e-200;
    if nuc < 2 {  //H, He
        place.iter_mut().zip(rads.iter()).for_each(|(place,rad)| {
            let mut judge = 0usize;
            alphas.get(0).unwrap().iter().for_each(|alpha| {
                if rad/r_atom > *alpha {
                    judge = judge + 1;
                }
            });
        *place = leb_l[judge];
        });
    }
    else if nuc > 2 && nuc <= 10 {  // 2nd period
        place.iter_mut().zip(rads.iter()).for_each(|(place,rad)| {
            let mut judge = 0usize;
            alphas.get(1).unwrap().iter().for_each(|alpha| {
                if rad/r_atom > *alpha {
                        judge = judge + 1;
                }
            });
            *place = leb_l[judge];
        });
    }
    else {  // 3nd period
        place.iter_mut().zip(rads.iter()).for_each(|(place,rad)| {
            let mut judge = 0usize;
            alphas.get(2).unwrap().iter().for_each(|alpha| {
                if rad/r_atom > *alpha {
                    judge = judge + 1;
                }
            });
            *place = leb_l[judge];
        });
    }

    let mut angs = vec![];
    for item in place.iter() { 
        angs.push(leb_ngrid[*item])
    }

    //println!("The angular array is {:?}", angs);
    //println!("The radial grid is {:?}", rads);
    angs

}

pub fn none_prune(nuc: usize, n_rad: usize, level: usize) -> Vec<usize> { 
    let n_ang = default_angular_num(nuc, level);
    return vec![n_ang;n_rad]
}