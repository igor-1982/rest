//use fdqc_tensors::{MatrixFullSlice};
use super::parameters::{SG1RADII, BOHR, BRAGG0, LEBEDEV_NGRID};

//use crate::sap::radi;
//use std::ops::Div;


pub fn sg1_prune (nuc: usize, rads: &Vec<f64>, n_rad: usize) -> Vec<usize>{
    //SG1, CPL, 209, 506
/*
         nuc : usize
            Nuclear charge.

        rads : 1D Vector
            Grid coordinates on radical axis.
            MatrixFull struct, size = [xxxxxx, 1]

# In SG1 the ang grids for the five regions
#6  38 86  194 86
 */ 

 /* 
     let SG1RADII: Vec<f64> = vec![0.0, 1.0000, 0.5882,   //H, He
    3.0769, 2.0513, 1.5385, 1.2308, 1.0256, 0.8791, 0.7692, 0.6838,  //2nd Period
    4.0909, 3.1579, 2.5714, 2.1687, 1.8750, 1.6514, 1.4754, 1.3333];  //3rd Period
  */
 
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

pub fn nwchem_prune(nuc: usize, rads: &Vec<f64>, n_ang: usize, n_rad: usize) -> Vec<usize> {
    /*     '''NWChem

    Args:
        nuc : int
            Nuclear charge.

        rads : 1D array
            Grid coordinates on radical axis.

        n_ang : int
            Max number of grids over angular part.

    Kwargs:
        radii : 1D array
            radii (in Bohr) for atoms in periodic table

    Returns:
        A list has the same length as rads. The list element is the number of
        grids over angular part for each radial grid.
    '''
     */
    //let BOHR = 0.52917721092;  // Angstroms

    /* 
    let BRAGG0 = vec![0.0,  // Ghost atom
    0.35,                                     1.40,             // 1s
    1.45, 1.05, 0.85, 0.70, 0.65, 0.60, 0.50, 1.50,             // 2s2p
    1.80, 1.50, 1.25, 1.10, 1.00, 1.00, 1.00, 1.80,             // 3s3p
    2.20, 1.80,                                                 // 4s
    1.60, 1.40, 1.35, 1.40, 1.40, 1.40, 1.35, 1.35, 1.35, 1.35, // 3d
                1.30, 1.25, 1.15, 1.15, 1.15, 1.90,             // 4p
    2.35, 2.00,                                                 // 5s
    1.80, 1.55, 1.45, 1.45, 1.35, 1.30, 1.35, 1.40, 1.60, 1.55, // 4d
                1.55, 1.45, 1.45, 1.40, 1.40, 2.10,             // 5p
    2.60, 2.15,                                                 // 6s
    1.95, 1.85, 1.85, 1.85, 1.85, 1.85, 1.85,                   // La, Ce-Eu
    1.80, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75,             // Gd, Tb-Lu
          1.55, 1.45, 1.35, 1.35, 1.30, 1.35, 1.35, 1.35, 1.50, // 5d
                1.90, 1.80, 1.60, 1.90, 1.45, 2.10,             // 6p
    1.80, 2.15,                                                 // 7s
    1.95, 1.80, 1.80, 1.75, 1.75, 1.75, 1.75,
    1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75,
    1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75,
                1.75, 1.75, 1.75, 1.75, 1.75, 1.75,
    1.75, 1.75,
    1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75];
    */

    let bragg: Vec<f64> = BRAGG0.iter().map(|item| *item/BOHR).collect();
    /* 
    let LEBEDEV_NGRID = vec![1, 6, 14, 26, 38, 50, 74, 86, 110, 146, 170,
    194,  230,  266,  302,  350,  434,  590,  770,  974, 1202, 1454,
    1730, 2030, 2354, 2702, 3074, 3470, 3890, 4334, 4802, 5294, 5810];
    */

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

/* 
def nwchem_prune(nuc, rads, n_ang, radii=radi.BRAGG_RADII):
    '''NWChem

    Args:
        nuc : int
            Nuclear charge.

        rads : 1D array
            Grid coordinates on radical axis.

        n_ang : int
            Max number of grids over angular part.

    Kwargs:
        radii : 1D array
            radii (in Bohr) for atoms in periodic table

    Returns:
        A list has the same length as rads. The list element is the number of
        grids over angular part for each radial grid.
    '''
    alphas = numpy.array((
        (0.25  , 0.5, 1.0, 4.5),
        (0.1667, 0.5, 0.9, 3.5),
        (0.1   , 0.4, 0.8, 2.5)))
    leb_ngrid = LEBEDEV_NGRID[4:]  # [38, 50, 74, 86, ...]
    if n_ang < 50:
        return numpy.repeat(n_ang, len(rads))
    elif n_ang == 50:
        leb_l = numpy.array([1, 2, 2, 2, 1])
    else:
        idx = numpy.where(leb_ngrid==n_ang)[0][0]
        leb_l = numpy.array([1, 3, idx-1, idx, idx-1])

    r_atom = radii[nuc] + 1e-200
    if nuc <= 2:  # H, He
        place = ((rads/r_atom).reshape(-1,1) > alphas[0]).sum(axis=1)
    elif nuc <= 10:  # Li - Ne
        place = ((rads/r_atom).reshape(-1,1) > alphas[1]).sum(axis=1)
    else:
        place = ((rads/r_atom).reshape(-1,1) > alphas[2]).sum(axis=1)
    angs = leb_l[place]
    angs = leb_ngrid[angs]
    return angs

*/
