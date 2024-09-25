use std::cmp::Ordering;
use crate::molecule_io::Molecule;
use crate::geom_io;
use crate::basis_io;
use crate::scf_io::SCF;
use rest_tensors::MatrixFull;
use tensors::matrix_blas_lapack::_dgemm_full;
use std::path::Iter;
use rayon::iter::{IntoParallelRefIterator, IndexedParallelIterator, ParallelIterator, IntoParallelRefMutIterator};
use std::fs::{self, File};
use std::io::prelude::*;
use std::io::LineWriter;

pub fn gen_box (mol: &Molecule) -> (Vec<f64>, Vec<f64>){
    let margin = mol.ctrl.cube_orb_setting[0];
    let atom_coord = &mol.geom.position; //by column
    let natm = mol.geom.elem.len();
    let mut max = vec![0.0;3];
    let mut min = vec![0.0;3];
    for i in 0..3{
        /* min[i] = *atom_coord.iter_slice_x(i).enumerate().min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal)).map(|(_, a)| a).unwrap();
        max[i] = *atom_coord.iter_slice_x(i).enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal)).map(|(_, a)| a).unwrap(); */
        let mut com = vec![0.0;natm];
        for j in (0..natm){
            com[j] = atom_coord.data[i+ 3 * j] ;
        }
        min[i] = *com.iter().enumerate().min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal)).map(|(_, a)| a).unwrap();
        max[i] = *com.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal)).map(|(_, a)| a).unwrap(); 
    };
    //println!("min:{:?} max:{:?}", &min,&max);
    let mut box_extent = vec![0.0; 3];
    box_extent.iter_mut().zip(max.iter().zip(min.iter())).for_each(|(a,(max,min))|{
        *a = max - min + 2.0*margin;
    });

    let mut boxorig = vec![0.0; 3];
    boxorig.iter_mut().zip(min.iter()).for_each(|(a,min)|{
        *a = min - margin;
    });
    //println!("{:?}",(&box_extent, &boxorig));
    (box_extent, boxorig)
} 

pub fn gen_xs(n_p: usize) -> Vec<[f64;3]>{
    let interval = 1.0/(n_p as f64 - 1.0);
    //let mut init_point: Vec<[f64;3]> = Vec::new();
    let mut init_points = vec![[0.0;3];n_p*n_p*n_p];
    for i in 0..n_p{
        for j in 0..n_p{
            for k in 0..n_p{
                //init_point.push([0.0 + i as f64 * interval, 0.0 + j as f64 * interval, 0.0 + k as f64 * interval]);
                let mut index = n_p*n_p*i + n_p * j + k;
                init_points[index] = [0.0 + i as f64 * interval, 0.0 + j as f64 * interval, 0.0 + k as f64 * interval];
            }
        }
    };
    init_points
}

pub fn gen_coords (init_points:&Vec<[f64;3]>, box_extent: &Vec<f64>, boxorig:&Vec<f64>) -> Vec<[f64;3]>{
    let mut coords = vec![[0.0;3]; init_points.len()];
    coords.iter_mut().zip(init_points).for_each(|(new,old)|{
        new.iter_mut().zip(old.iter().zip(box_extent.iter().zip(boxorig.iter()))).for_each(|(new_x,(old_x,(co,b)))|{
            *new_x = old_x * co +  b;
        });
    });
    coords
}

pub fn gen_delta(n_p:usize, box_extent: &Vec<f64>) -> Vec<f64>{
    let mut delta = vec![0.0;3];
    let interval = 1.0/(n_p as f64 - 1.0);
    delta.iter_mut().zip(box_extent.iter()).for_each(|(x,y)|{
        *x = y * interval;
    });
    delta
}

pub fn tabulated_ao_fig (mol: &Molecule, points: &Vec<[f64; 3]>) -> MatrixFull<f64>{
    let num_p = points.len();
    let mut tab_den = MatrixFull::new([num_p,mol.num_basis], 0.0);
    let mut start:usize = 0;

    // for the calculations with extra basis sets in ghost atoms
    let mut position_full = if mol.geom.ghost_bs_elem.len()== 0 {
        mol.geom.position.clone()
    } else {
        let mut tmp_pos = mol.geom.position.clone();
        tmp_pos.append_column(&mol.geom.ghost_bs_pos);
        tmp_pos
    };
    println!("debug num_basis: {:?},tmp_pos: {:?}", mol.num_basis, &position_full);
    
    mol.basis4elem.iter().zip(position_full.iter_columns_full()).for_each(|(elem,geom)| {
        let mut tmp_geom = [0.0;3];
        tmp_geom.iter_mut().zip(geom.iter()).for_each(|value| {*value.0 = *value.1});
        let tmp_spheric = basis_io::spheric_gto_value_matrixfull(points, &tmp_geom, elem);
        let s_len = tmp_spheric.size[1];
        tab_den.iter_columns_mut(start..start+s_len).zip(tmp_spheric.iter_columns_full())
        .for_each(|(to,from)| {
            to.par_iter_mut().zip(from.par_iter()).for_each(|(to,from)| {*to = - *from});
        });
        start += s_len;
    });
    tab_den // N_p * Nao
}

pub fn get_cube_orb(scf_data:&SCF) -> [MatrixFull<f64>;2]{

    let spin_channel = scf_data.mol.spin_channel;
    let n_p = scf_data.mol.ctrl.cube_orb_setting[1] as usize;
    let orb_indices = &scf_data.mol.ctrl.cube_orb_indices;
    let num_state = scf_data.mol.num_state;
    let mut orb_indices_2: [Vec<usize>;2] = [vec![],vec![]];

    // expend orb_indices to specific orbital indices: orb_indices_2
    if orb_indices.len() == 0{
        for i_spin in 0..spin_channel {
            orb_indices_2[i_spin] = (0..num_state).collect();
        }
    } else {
        orb_indices.iter().for_each(|x| {
            (x[0]..x[1]+1).for_each(|y| {
                if ! orb_indices_2[x[2]].contains(&y) && y < num_state {
                    orb_indices_2[x[2]].push(y);
                }
            });
        });
    }

    for i_spin in 0..spin_channel{
        orb_indices_2[i_spin].sort_by(|a,b| a.cmp(b));
    }
    //println!("{:?}", &orb_indices);
    //println!("{:?}", &orb_indices_2);

    

    let mol = &scf_data.mol;
    let atom_info = &mol.geom;
    let atom_mass_charge = geom_io::get_mass_charge(&atom_info.elem).clone();
    let natm = atom_mass_charge.len();
    let result = gen_box(&mol);
    let box_extent = result.0;
    let boxorig = result.1;
    let init_points = gen_xs(n_p);
    let coords = gen_coords(&init_points, &box_extent, &boxorig);
    let delta = gen_delta(n_p, &box_extent);
    let ao = tabulated_ao_fig(&mol, &coords);
    //let all_orb = scf_data.eigenvectors[0].clone();
    let mut prod = [MatrixFull::empty(),MatrixFull::empty()];

    for i_spin in 0..spin_channel {

        let orb_indices_s = &orb_indices_2[i_spin];

        prod[i_spin] = MatrixFull::new([ao.size[0],scf_data.eigenvectors[i_spin].size[1]],0.0);
        _dgemm_full(&ao, 'N', &scf_data.eigenvectors[i_spin], 'N', &mut prod[i_spin], 1.0, 0.0);
    
        // generate cube file  
        prod[i_spin].iter_columns_full().enumerate().for_each(|(index, x)|{
            if orb_indices_s.contains(&index){
                let mut cube_string = "Orbital value in real space (1/Bohr^3)\nREST Version: \n".to_owned();
                cube_string += &natm.to_string();
                boxorig.iter().for_each(|x|{
                    cube_string += "    ";
                    cube_string +=  &x.to_string();
                    });
                cube_string += "\n";

                for j in (0..3){
                    cube_string += &n_p.to_string();
                    if j == 0{
                        cube_string += "    ";
                        cube_string += &delta[0].to_string();
                        cube_string += "     0.000000     0.000000\n";
                    }else if j == 1 {
                        cube_string += "    0.000000    ";
                        cube_string += &delta[1].to_string();
                        cube_string += "     0.000000\n";
                    }else{
                        cube_string += "    0.000000    0.000000    ";
                        cube_string += &delta[2].to_string();
                        cube_string += "\n";
                    };
                }

                atom_mass_charge.iter().zip(mol.geom.position.iter_columns_full()).for_each(|((mass,charge),position)|{
                    let charge_u = *charge as usize;
                    cube_string += &charge_u.to_string();
                    cube_string += "    0.000000";
                    position.iter().for_each(|x|{
                        cube_string += "    ";
                        cube_string += &x.to_string();
                    });
                    cube_string += "\n";
                });
            
                let mut count = 0;
                x.iter().zip(0..n_p*n_p*n_p).for_each(|(value,index1)|{
                    if index1 == 0 {
                        cube_string += &value.to_string();
                        count += 1;
                    }
                    if index1 % n_p == 0 && index1 != 0{
                        cube_string += "\n";
                        cube_string += &value.to_string();
                        count += 1;
                    }
                    if (index1-(count-1)*n_p) % 6 == 0 && index1-(count-1)*n_p!= 0 {
                        cube_string += "\n";
                        cube_string += &value.to_string();
                    }else if index1-(count-1)*n_p!= 0{
                        cube_string += "    ";
                        cube_string += &value.to_string();
                    }
                });

                //write
                let mut path = mol.geom.name.clone();
                path = if i_spin==0 {
                    format!("{}-{}-alpha",path, index)
                } else {
                    format!("{}-{}-beta",path, index)
                };
                path += ".cube";
                let file = File::create(path);
                let mut file = LineWriter::new(file.unwrap());
                file.write_all(&cube_string.into_bytes());
            }

        });
    }
    
    prod // n_p*nmo matrix
}
