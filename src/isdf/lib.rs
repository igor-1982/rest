use std::fmt::Display;
use std::marker::PhantomData;
use std::ops::{Add, AddAssign, Sub, SubAssign};

use crate::isdf;
mod tests {

    use num_traits::Float;
    use tensors::{MatrixFull, RIFull};

    use crate::isdf::{index_of_min,prod_states_gw, cvt_update_cmu, cvt_find_corresponding_point};

    #[test]
    fn test_index_of_min() {
        let mut a = vec![2.0,1.0,0.0,6.1,7.2];
        let z = index_of_min(&mut a);
        println!("{:?}", z)
    }

    #[test]
    fn test_prod_states_gw() {
        let a = MatrixFull::from_vec([2,3],vec![1.0,4.0,2.0,5.0,3.0,6.0]).unwrap();
        let b = MatrixFull::from_vec([2,2],vec![7.0,9.0,8.0,10.0]).unwrap();
        let c = prod_states_gw(&a, &b);

        println!("{:?}", c.data)
    }
     
    #[test]
    fn test_lapack_dgesv() {
        let mut a = MatrixFull::from_vec([2,2], vec![1.0,3.0,2.0,5.0]).unwrap(); 
        let mut b = MatrixFull::from_vec([1,2], vec![1.0,2.0]).unwrap(); 
        let result = &mut a.lapack_dgesv(&mut b, a.size[0] as i32);
        println!("{:?}",result);
    }

    #[test]
    fn test_rifull() {
        let mut a = RIFull::from_vec([3,3,4], vec![1.,2.,3.,2.,4.,5.,3.,5.,6.,7.,8.,9.,8.,10.,11.,9.,11.,12.,13.,14.,15.,14.,16.,17.,15.,17.,18.,19.,20.,21.,20.,22.,23.,21.,23.,24.]).unwrap(); 
        let mut ri_v_ao = MatrixFull::new([4, 9],0.0);
        for i in 0..4{
            for j in 0..9{
                ri_v_ao.data[j * 4 + i] = a.data[i*9+j];
            }
        }
        println!("{:?}",ri_v_ao.data)
    }
    #[test]
    fn test_find_classification_update(){
        let c_mu = vec![[-1.1285228915673906, -0.8276723350310994, -0.8885513154278561], 
            [-0.0875008208845615, -0.504324412419218, -0.7030997212959521], 
            [-0.28463625629524897, 0.207273098573563, -1.03185079230271], 
            [1.0988347082269967, -2.6610528249090004, -0.9251803520871971], 
            [0.1432123950999507, -0.19396074727897408, -0.49684675048074556], 
            [-0.21808504286076888, 0.8004514742731099, 1.1265754883335408]];
        let grids = vec![[-4.48132694, -4.48132694, -4.48132694],
        [-4.82900702, -4.82900702, -4.82900702],
        [-5.22855109,-5.22855109, -5.22855109],
        [-5.69882806, -5.69882806, -5.69882806],
        [-4.20878354, -4.20878354, -6.82535873],
        [-4.58733851, -4.58733851, -7.43925902],
        [-4.20878354, -6.82535873, -4.20878354],
        [-4.58733851, -7.43925902, -4.58733851],
        [-6.82535873, -4.20878354, -4.20878354],
        [-7.43925902, -4.58733851, -4.58733851],
        [-4.20878354, -4.20878354, -4.36871477],
        [-4.58733851, -4.58733851, -4.98261505],
        [-5.99545559, -5.99545559, -3.18155061],
        [-6.53471104, -6.53471104 ,-3.4677121 ],
        [-5.69882806, -5.69882806, -3.2421841 ],
        [-4.4685114 , -4.4685114 , -2.37126186],
        [-4.78551877, -4.78551877, -2.5394851 ],
        [-5.13863136, -5.13863136, -2.72686796],
        [-5.53730787, -5.53730787, -2.93842978],
        [-4.38691203, -6.85305817, -1.93611028]];
        let weights = vec![1.789986,2.364994,3.21922327,4.566375566,3.16384733,4.488144,3.15283829,4.4707488,3.152838,4.4707,0.0010220,0.001221,3.227645,4.5718579, 0.048395323,1.09861,1.393937,1.7991006,2.37423,2.20309006];
        let num_mu = c_mu.len();
        let num_grids = grids.len(); 
        let mut ind_r:Vec<usize> = vec![0; num_grids];
        let mut dist_r:Vec<f64> = vec![0.0;num_grids];
        //find nearest center
        let num_mu = c_mu.len();
        //let mut r_min = 0.0; 
        for i in 0..num_grids{
            let mut r_c_mu:Vec<f64> = vec![0.0; num_mu];
            for j in 0..c_mu.len(){
                r_c_mu[j] = grids[i].iter().zip(c_mu[j].iter()).fold(0.0,|r,(ac,gc)| {r + (
                    *ac-*gc).powf(2.0)}).sqrt();
            }
            ind_r[i] = index_of_min(&mut r_c_mu);
            dist_r[i] = r_c_mu[ind_r[i]];
        }//further zip
        println!("{:?}",(&ind_r,&dist_r));

        let mut c_mu_save = c_mu.clone();

    //Added together
        let mut c_mu_tmp:Vec<[f64; 3]> = vec![[0.0;3];num_mu];
        let mut weight_sum: Vec<f64> = vec![0.0; num_mu];
        for i in 0..num_grids{
        //c_mu_tmp[ind_r[i]] += lambda_r[ind_r[i]] * rgrid[ind_r[i]];
            c_mu_tmp[ind_r[i]].iter_mut().zip(grids[i].iter()).for_each(|(x,y)|{
                *x += weights[i] * y;
            });
            weight_sum[ind_r[i]] += weights[i];
        }

        let non_zero_ind = weight_sum.iter()
        .enumerate()
        .filter(|(_, &r)| r >= 1e-8)
        .map(|(index, _)| index)
        .collect::<Vec<_>>();//1e-8 could be adjusted.
        for i in 0..non_zero_ind.len(){
            c_mu_save[non_zero_ind[i]].iter_mut().zip(c_mu_tmp[non_zero_ind[i]].iter()).for_each(|(x,y)|{
                *x = y / weight_sum[non_zero_ind[i]];
            })
        }
        println!("new c_mu: {:?}", &c_mu_save);

        let num_mu = c_mu.len(); 
        let mut ind_mu: Vec<usize> = vec![0; num_mu];;
    
        //find nearest center
        let mut min_dist = vec![1.0e10; num_mu];
        //let mut r_grid_core:Vec<f64> = vec![0.0; num_grids];
      
        //find those whose clusters have been changed
        for i in 0..num_grids{
            let mut value = grids[i].iter().zip(c_mu_save[ind_r[i]].iter()).fold(0.0,|r,(ac,gc)| {r + (ac-gc).powf(2.0)}).sqrt();
            if value < min_dist[ind_r[i]]{
                ind_mu[ind_r[i]] = i ;
                min_dist[ind_r[i]] = value;
            }
        }
        let need_search_full_rgrid = ind_mu.iter().enumerate().filter(|(_, &r)| r == 0).map(|(index,_)|index).collect::<Vec<_>>();
        for i in need_search_full_rgrid{
            let mut dist_to_all_grids: Vec<f64> = vec![0.0; num_grids];
            dist_to_all_grids.iter_mut().zip(grids.clone()).for_each(|(x,y)|{
                *x = y.iter().zip(c_mu_save[i].iter()).fold(0.0,|r,(ac,gc)| {r + (ac-gc).powf(2.0)}).sqrt();
            });
            ind_mu[i] = index_of_min(&mut dist_to_all_grids);   
        }
        println!("ind_mu is {:?}", &ind_mu);
    }

    #[test]
    fn test_filter(){
        let weights = vec![0.0000000000000146, 2.0,10.0,0.45678,0.00000003];
        let non_zero_ind = weights.iter()
        .enumerate()
        .filter(|(_, &r)| r >= 1e-8)
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
        println!("{:?}",(&non_zero_ind));
    }

    #[test]
    fn test_cvt_update_cmu(){

        let weights = vec![0.0000000000000146, 2.0,0.4,0.45678,0.00000003];
        let grids = vec![[0.1,0.1,0.1],[0.2,0.2,0.3],[1.1,0.5,0.6],[-1.1,0.58,-0.7],[-0.23,1.1,1.25]];
        let c_mu_old = vec![[0.2,0.1,0.4],[-0.2,1.1,1.5]];
        let ind_r = vec![0,0,0,0,1];
        let num_mu = c_mu_old.len();
        let num_grids = grids.len(); 
        let mut c_mu_save = c_mu_old.clone();

        //Added together
        let mut c_mu_tmp:Vec<[f64; 3]> = vec![[0.0;3];num_mu];
        let mut weight_sum: Vec<f64> = vec![0.0; num_mu];
        for i in 0..num_grids{
            //c_mu_tmp[ind_r[i]] += lambda_r[ind_r[i]] * rgrid[ind_r[i]];
            c_mu_tmp[ind_r[i]].iter_mut().zip(grids[i].iter()).for_each(|(x,y)|{
            *x += weights[i] * y;
        });
        weight_sum[ind_r[i]] += weights[i];
    }//further zip

        let non_zero_ind = weight_sum.iter()
        .enumerate()
        .filter(|(_, &r)| r >= 1e-8)
        .map(|(index, _)| index)
        .collect::<Vec<_>>();//1e-8 could be adjusted.
        for i in 0..non_zero_ind.len(){
            c_mu_save[non_zero_ind[i]].iter_mut().zip(c_mu_tmp[non_zero_ind[i]].iter()).for_each(|(x,y)|{
                *x = y / weight_sum[non_zero_ind[i]];
            })
        }
        //println!("{:?}",&c_mu_save);
        
    }
    
    #[test]
    fn test_cvt_isdf(){
        let mut c_mu = vec![[-1.1285228915673906, -0.8276723350310994, -0.8885513154278561], 
            [-0.0875008208845615, -0.504324412419218, -0.7030997212959521], 
            [-0.28463625629524897, 0.207273098573563, -1.03185079230271], 
            [1.0988347082269967, -2.6610528249090004, -0.9251803520871971], 
            [0.1432123950999507, -0.19396074727897408, -0.49684675048074556], 
            [-0.21808504286076888, 0.8004514742731099, 1.1265754883335408]];
        let grids = vec![[-4.48132694, -4.48132694, -4.48132694],
        [-4.82900702, -4.82900702, -4.82900702],
        [-5.22855109,-5.22855109, -5.22855109],
        [-5.69882806, -5.69882806, -5.69882806],
        [-4.20878354, -4.20878354, -6.82535873],
        [-4.58733851, -4.58733851, -7.43925902],
        [-4.20878354, -6.82535873, -4.20878354],
        [-4.58733851, -7.43925902, -4.58733851],
        [-6.82535873, -4.20878354, -4.20878354],
        [-7.43925902, -4.58733851, -4.58733851],
        [-4.20878354, -4.20878354, -4.36871477],
        [-4.58733851, -4.58733851, -4.98261505],
        [-5.99545559, -5.99545559, -3.18155061],
        [-6.53471104, -6.53471104 ,-3.4677121 ],
        [-5.69882806, -5.69882806, -3.2421841 ],
        [-4.4685114 , -4.4685114 , -2.37126186],
        [-4.78551877, -4.78551877, -2.5394851 ],
        [-5.13863136, -5.13863136, -2.72686796],
        [-5.53730787, -5.53730787, -2.93842978],
        [-4.38691203, -6.85305817, -1.93611028]];
        let weights = vec![1.789986,2.364994,3.21922327,4.566375566,3.16384733,4.488144,3.15283829,4.4707488,3.152838,4.4707,0.0010220,0.001221,3.227645,4.5718579, 0.048395323,1.09861,1.393937,1.7991006,2.37423,2.20309006];
        let max_iter = 300;
        let mut count = 0;
        let mut dist = 0.0;
        let result = loop{
            let num_mu = c_mu.len();
            let num_grids = grids.len(); 
            let mut ind_r:Vec<usize> = vec![0; num_grids];
            let mut dist_r:Vec<f64> = vec![0.0;num_grids];
            //find nearest center
            let num_mu = c_mu.len();
            //let mut r_min = 0.0; 
            for i in 0..num_grids{
                let mut r_c_mu:Vec<f64> = vec![0.0; num_mu];
                for j in 0..c_mu.len(){
                    r_c_mu[j] = grids[i].iter().zip(c_mu[j].iter()).fold(0.0,|r,(ac,gc)| {r + (
                        *ac-*gc).powf(2.0)}).sqrt();
                }
                ind_r[i] = index_of_min(&mut r_c_mu);
                dist_r[i] = r_c_mu[ind_r[i]];
            }
            println!("Step {:?}:", &count);
            println!("dist_r is {:?}", &dist_r);
            let mut init_dist_r = 0.0;
            dist_r.iter().zip(weights.iter()).for_each(|(d, w)|{
                init_dist_r += w * d * d;
            });
            dist = init_dist_r.sqrt();
            
            println!("    Initial distance_r = {:?}.", dist);
    
            let mut ind_r:Vec<usize> = vec![0; 20];
            let mut dist_r:Vec<f64> = vec![0.0;20];
            //find nearest center
            let num_mu = c_mu.len();
            //let mut r_min = 0.0; 
            for i in 0..20{
                let mut r_c_mu:Vec<f64> = vec![0.0; num_mu];
                for j in 0..c_mu.len(){
                    r_c_mu[j] = grids[i].iter().zip(c_mu[j].iter()).fold(0.0,|r,(ac,gc)| {r + (
                        *ac-*gc).powf(2.0)}).sqrt();
                }
                ind_r[i] = index_of_min(&mut r_c_mu);
                dist_r[i] = r_c_mu[ind_r[i]];
            }//further zip
            println!("{:?}",(&ind_r,&dist_r));
    
            let mut c_mu_save = c_mu.clone();
            let mut c_mu_tmp:Vec<[f64; 3]> = vec![[0.0;3];num_mu];
            let mut weight_sum: Vec<f64> = vec![0.0; num_mu];
            for i in 0..20{
            //c_mu_tmp[ind_r[i]] += lambda_r[ind_r[i]] * rgrid[ind_r[i]];
                c_mu_tmp[ind_r[i]].iter_mut().zip(grids[i].iter()).for_each(|(x,y)|{
                    *x += weights[i] * y;
                });
                 weight_sum[ind_r[i]] += weights[i];
            }
    
            let non_zero_ind = weight_sum.iter()
                .enumerate()
                .filter(|(_, &r)| r >= 1e-8)
                .map(|(index, _)| index)
                .collect::<Vec<_>>();//1e-8 could be adjusted.
            for i in 0..non_zero_ind.len(){
                c_mu_save[non_zero_ind[i]].iter_mut().zip(c_mu_tmp[non_zero_ind[i]].iter()).for_each(|(x,y)|{
                    *x = y / weight_sum[non_zero_ind[i]];
                })
            }
                
            let mut sum = 0.0;
            if count == 0 {
                //println!{"first ind_r is: {:?}", &ind_r};
            }
            if count == max_iter - 1 {
                c_mu_save.iter().zip(c_mu.iter()).for_each(|([x,y,z],[a,b,c])|{
                    sum += ((x-a) * (x-a) + (y-b) * (y-b) + (z-c) * (z-c)).sqrt();
                });
                c_mu = c_mu_save.clone();
                let mut ind_mu: Vec<usize> = vec![0; num_mu];
    
        //find nearest center
            let mut min_dist = vec![1.0e10; num_mu];
        //let mut r_grid_core:Vec<f64> = vec![0.0; num_grids];
      
        //find those whose clusters have been changed
            for i in 0..20{
                let mut value = grids[i].iter().zip(c_mu_save[ind_r[i]].iter()).fold(0.0,|r,(ac,gc)| {r + (ac-gc).powf(2.0)}).sqrt();
                if value < min_dist[ind_r[i]]{
                    ind_mu[ind_r[i]] = i ;
                    min_dist[ind_r[i]] = value;
                }
            }
            let need_search_full_rgrid = ind_mu.iter().enumerate().filter(|(_, &r)| r == 0).map(|(index,_)|index).collect::<Vec<_>>();
            for i in need_search_full_rgrid{
                let mut dist_to_all_grids: Vec<f64> = vec![0.0; 20];
                dist_to_all_grids.iter_mut().zip(grids.clone()).for_each(|(x,y)|{
                    *x = y.iter().zip(c_mu_save[i].iter()).fold(0.0,|r,(ac,gc)| {r + (ac-gc).powf(2.0)}).sqrt();
                });
                ind_mu[i] = index_of_min(&mut dist_to_all_grids);   
            }
            println!("ind_mu is {:?}", &ind_mu);
                break (ind_mu, sum);
            } 
    
            let mut criterion = 0.0;
            c_mu.iter().for_each(|[x,y,z]|{
                criterion += 1e-6 * ((x * x + y * y + z * z).sqrt())
            });
            c_mu_save.iter().zip(c_mu.iter()).for_each(|([x,y,z],[a,b,c])|{
                sum += ((x-a) * (x-a) + (y-b) * (y-b) + (z-c) * (z-c)).sqrt();
            });
            
            if sum <= criterion{
                c_mu = c_mu_save.clone();
                let mut ind_mu: Vec<usize> = vec![0; num_mu];
    
        //find nearest center
            let mut min_dist = vec![1.0e10; num_mu];
        //let mut r_grid_core:Vec<f64> = vec![0.0; num_grids];
      
        //find those whose clusters have been changed
            for i in 0..20{
                let mut value = grids[i].iter().zip(c_mu_save[ind_r[i]].iter()).fold(0.0,|r,(ac,gc)| {r + (ac-gc).powf(2.0)}).sqrt();
                if value < min_dist[ind_r[i]]{
                    ind_mu[ind_r[i]] = i ;
                    min_dist[ind_r[i]] = value;
                }
            }
            let need_search_full_rgrid = ind_mu.iter().enumerate().filter(|(_, &r)| r == 0).map(|(index,_)|index).collect::<Vec<_>>();
            for i in need_search_full_rgrid{
                let mut dist_to_all_grids: Vec<f64> = vec![0.0; 20];
                dist_to_all_grids.iter_mut().zip(grids.clone()).for_each(|(x,y)|{
                    *x = y.iter().zip(c_mu_save[i].iter()).fold(0.0,|r,(ac,gc)| {r + (ac-gc).powf(2.0)}).sqrt();
                });
                ind_mu[i] = index_of_min(&mut dist_to_all_grids);   
            }
            println!("ind_mu is {:?}", &ind_mu);
                println!("Random points converged after {} iterations.", &count);
                println!("Dist: {:?}", sum);
                break (ind_mu, sum);
            }else {
                c_mu = c_mu_save;
                count += 1;
            }
        };
        

    }
    
}