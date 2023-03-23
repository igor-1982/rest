use std::num;
use crate::molecule_io::Molecule;
use crate::geom_io;
use crate::basis_io;
use crate::basis_io::basic_math::factorial;
use crate::scf_io::SCF;
use rest_tensors::{TensorOpt,MatrixFull};
use rand::distributions::normal::StandardNormal;
use itertools::Itertools;
use rayon::iter::{IntoParallelRefIterator, IndexedParallelIterator, ParallelIterator, IntoParallelRefMutIterator};
//mod lib;

//checked
pub fn generate_random_points (n_p: usize) -> Vec<[f64; 3]> {
    let mut rng = rand::thread_rng();
    let mut random_points: Vec<[f64;3]> = vec![[0.0,0.0,0.0];n_p];
    random_points.iter_mut().for_each(|x|{
        x.iter_mut().for_each(|y|{
            let StandardNormal(a) = rand::random();
            *y =a;
        })
    });
    //println!("{:?}",random_points);
    random_points
}
//unchecked
pub fn rand_for_mol (mol: &Molecule, n_p: usize) -> (Vec<Vec<[f64; 3]>>, usize, f64){
    let atom_info = &mol.geom;
    let atom_mass_charge = geom_io::get_mass_charge(&atom_info.elem).clone();
    let mut num_ele = 0.0;
    atom_mass_charge.iter().for_each(|(x,y)|{
        num_ele += *y;
    });

    let n_ele = unsafe {
        num_ele.to_int_unchecked::<usize>()
    };

    let nu_ele = unsafe {
        num_ele.to_int_unchecked::<i32>()
    };

    let mid = f64::from(factorial(nu_ele));
    let coeff =  mid.powf(-0.5);
    let mut rand_points_for_mol:Vec<Vec<[f64; 3]>> = vec![vec![[0.0;3]; n_p]; n_ele];
    //let mut rand_points_for_mol:Vec<Vec<[f64; 3]>> = Vec::with_capacity(n_ele);
    rand_points_for_mol.iter_mut().for_each(|x|{
        *x = generate_random_points(n_p);
    });
    (rand_points_for_mol, n_ele, coeff)
}

// change from tabulated_ao
pub fn random_tabulated_ao (mol: &Molecule, rand_p: &Vec<[f64; 3]>) -> MatrixFull<f64>{
    let num_p = rand_p.len();
    let mut tab_den = MatrixFull::new([num_p,mol.num_basis], 0.0);
    let mut start:usize = 0;
    mol.basis4elem.iter().zip(mol.geom.position.iter_columns_full()).for_each(|(elem,geom)| {
        let mut tmp_geom = [0.0;3];
        tmp_geom.iter_mut().zip(geom.iter()).for_each(|value| {*value.0 = *value.1});
        let tmp_spheric = basis_io::spheric_gto_value_matrixfull(rand_p, &tmp_geom, elem);
        let s_len = tmp_spheric.size[1];
        tab_den.iter_columns_mut(start..start+s_len).zip(tmp_spheric.iter_columns_full())
        .for_each(|(to,from)| {
            to.par_iter_mut().zip(from.par_iter()).for_each(|(to,from)| {*to = *from});
        });
        start += s_len;
    });
    //let ao = tab_den.transpose_and_drop();
    tab_den // N_p * Nao
}

pub fn gen_c (scf_data: &SCF, n_ele: usize) -> MatrixFull<f64>{
    let eigv = scf_data.eigenvectors[0].clone(); //unpolarized
    let nmo = n_ele/2;
    let nao = scf_data.mol.num_basis;
    let mut new_c: Vec<f64> = vec![0.0; nao * n_ele]; 
    let mut mo_vec = vec![vec![0.0; nao];nmo];
    mo_vec.iter_mut().zip(0..nmo).for_each(|(x,y)|{
        x.iter_mut().zip(eigv.slice_column(y)).for_each(|(a,b)|{
            *a = *b;
        });
    });
    
    for i in 0..n_ele{
        let mut j = i / 2;
        let mo = &mo_vec[j];
        new_c[i * nao .. (i+1) * nao].iter_mut().zip(mo.iter()).for_each(|(x,y)|{
            *x = *y;
        })
    }
    let c_need = MatrixFull::from_vec([nao, n_ele], new_c).unwrap();
    c_need 
}

pub fn slater_determinant(scf_data: &SCF, n_p: usize) -> Vec<(Vec<[f64;3]>,f64)> {
    /* let a = vec![[0.9048619635252333, 0.4573523031534341, -0.5013015911077909], [-0.44926362825630894, -0.5735387124535505, 0.2568942876881821]];
    let b = random_tabulated_ao(&scf_data.mol, &a);
    b.formated_output(5, "full"); */
    let (points, n_ele, coeff) = rand_for_mol(&scf_data.mol, n_p);
    //println!("n_ele: {:?}; n_ele_points:{:?}; n_p: {:?}",n_ele,points.len(),points[0].len());
    let mut c_mat = gen_c(&scf_data, n_ele);
    let mut tabulated_ao:Vec<MatrixFull<f64>> = vec![MatrixFull::new([n_ele,scf_data.mol.num_basis], 0.0) ;n_ele];
    //let mut tabulated_ao:Vec<MatrixFull<f64>> = Vec::with_capacity(n_ele);
    tabulated_ao.iter_mut().zip(points.iter()).for_each(|(ele,rand_points)|{
        *ele = random_tabulated_ao(&scf_data.mol, rand_points);
        //ele.formated_output(5, "full");
    });
    let nao = scf_data.mol.num_basis;
    //println!("num_bas:{}",nao);
    let mut n_r = 0;
    let mut perms = (0..n_p).permutations(n_ele);
    &mut perms.by_ref().for_each(|x|{n_r += 1;});
    //let mut result:Vec<(Vec<[f64;3]>,f64)> =Vec::with_capacity(n_r);
    let mut result:Vec<(Vec<[f64;3]>,f64)> =vec![(vec![[0.0,0.0,0.0];n_ele],0.0);n_r];
    result.iter_mut().zip(perms).for_each(|((r,slater),ind)|{
        //let mut ao:Vec<Vec<f64>> =Vec::with_capacity(n_ele);
        //let mut point_info: Vec<[f64;3]> = Vec::with_capacity(n_ele);
        let mut ao:Vec<Vec<f64>> =vec![vec![0.0; nao]; n_ele];
        let mut point_info: Vec<[f64;3]> = vec![[0.0;3];n_ele];
        let mut ao_use:Vec<f64> =vec![0.0; n_ele * nao];

        point_info.iter_mut().zip(points.iter().zip(ao.iter_mut().zip(ind.iter()).zip(tabulated_ao.iter()))).for_each(|(p_i,(p,((ao_n,i),ao_points)))|{
            *ao_n = tmp_iter_j(&ao_points, *i);
            //println!("ao_n:{:?}",ao_n);
            *p_i = p[*i];
        });
        //println!("ao:{:?}",ao);
        *r = point_info;
        //let mut ao_use:Vec<f64> =vec![0.0; n_ele * nao];
        let mut index = 0;
        ao.iter().for_each(|a|{
            a.iter().zip(ao_use[index*nao..(index+1)*nao].iter_mut()).for_each(|(x,y)|{
                *y = *x;
            });
            index += 1;
        });

        let mut ao_m = MatrixFull::from_vec([nao, n_ele], ao_use).unwrap();
        //ao_m.formated_output(5, "full");
        //c_mat.formated_output(5, "full");
        //println!("size of ao: {:?}, size of c_mat: {:?}", ao_m.size,c_mat.size);
        *slater = single_slater(&mut ao_m,&mut c_mat, coeff);

        
    });
    result
    
}

pub fn single_slater (ao_points: &mut MatrixFull<f64>, c_mat: &mut MatrixFull<f64>, coeff: f64) -> f64{
    let mut prod = MatrixFull::new([ao_points.size[1];2],0.0);
    /* prod.to_matrixfullslicemut().lapack_dgemm(&mut c_mat.to_matrixfullslice(), &mut ao_points.to_matrixfullslice(), 'T', 'N', 1.0, 0.0);
    prod.transpose(); */
    prod.to_matrixfullslicemut().lapack_dgemm(&ao_points.to_matrixfullslice(), &c_mat.to_matrixfullslice(), 'T', 'N', 1.0, 0.0);
    //coeff * prod.det()

    //prod.formated_output(4, "full");
    /* let v_times_polar = prod.to_matrixfullslicemut().lapack_dgetrf().unwrap();

    let mut res = coeff * v_times_polar.get_diagonal_terms().unwrap().iter()
        .fold(1.0,|acc,value| acc*(*value)); */
    let mut res = 1.0;
    let mut i = 0;
    prod.data.iter().for_each(|x|{
        if i % prod.size[0] == i / prod.size[0]{
            res *= *x;
        }
        i += 1;
    });
    res * coeff
}

pub fn tmp_iter_j(matrix: &MatrixFull<f64>, j: usize) -> Vec<f64> {
    let mut slice: Vec<f64> = vec![0.0;matrix.size[1]];
    for i in 0..matrix.size[1]{
        slice[i] = matrix.data[matrix.size[0]*i + j];
    }
    /* let start = matrix.size[0]*j;
    let end = start + matrix.size[0];
    //let mut column: Vec<f64> = Vec::with_capacity(matrix.size[0]);
    let mut column: Vec<f64> = vec![0.0; matrix.size[0]];
    column.iter_mut().zip(matrix.data[start..end].iter()).for_each(|(x,y)|{
        *x = *y;
    }); */
    slice
}

// eigenvector[0] ?= c unpolarized
// how to iter and save the results
// deteminant calculation