use std::fmt::Display;
use std::marker::PhantomData;
use std::ops::{Add, AddAssign, Sub, SubAssign};

use crate::isdf;
mod tests {

    use num_traits::Float;
    use tensors::{MatrixFull, RIFull};

    use crate::isdf::{index_of_min,prod_states_gw, cvt_update_cmu, cvt_find_corresponding_point, cvt_classification, cvt_isdf};

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

        println!("{:?}", c)
    }
     
    #[test]
    fn test_dgemm(){
        let mut a = MatrixFull::from_vec([3,4],vec![1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0]).unwrap();
        let mut b = MatrixFull::from_vec([3,2],vec![1.0,2.0,3.0,4.0,5.0,6.0]).unwrap();
        let mut c = MatrixFull::new([4,2], 0.0);
        c.lapack_dgemm( &mut a, &mut b, 'T','N',1.0, 0.0);
        c.formated_output_e(5,"full");
    }
    
    #[test]
    fn test_lapack_dgesv() {
        let mut a = MatrixFull::from_vec([2,2], vec![1.0,3.0,2.0,5.0]).unwrap(); 
        let mut b = MatrixFull::from_vec([1,2], vec![1.0,2.0]).unwrap(); 
        let result = &mut a.lapack_dgesv(&mut b, a.size[0] as i32);
        println!("{:?}",result);
    }

    #[test]
    fn test_pseudo_inverse() {
        let mut a = MatrixFull::from_vec([3,4], vec![1.0,2.0,4.0,6.0,-1.0,3.0,4.0,0.0,-2.0,5.0,2.0,3.0]).unwrap(); 
        //let mut a = MatrixFull::from_vec([3,3], vec![1.0,2.0,4.0,6.0,-1.0,3.0,4.0,0.0,-2.0]).unwrap(); 
        let result = &mut a.pseudo_inverse();
        println!("{:?}",result);
    }

    #[test]
    fn test_pinv() {
        let mut a = MatrixFull::from_vec([3,3], vec![4.100378e-7,6.446175e-20,1.038086e-17,6.446175e-20,1.070506e-30,4.646721e-27,-2.836063e7,5.034143e0,6.305798e3]).unwrap(); 
        //let mut a = MatrixFull::from_vec([3,3], vec![1.0,2.0,4.0,6.0,-1.0,3.0,4.0,0.0,-2.0]).unwrap(); 
        let result = &mut a.pinv(1.0e-6);
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
        let lambda_r = vec![1.789986,2.364994,3.21922327,4.566375566,3.16384733,4.488144,3.15283829,4.4707488,3.152838,4.4707,0.0010220,0.001221,3.227645,4.5718579, 0.048395323,1.09861,1.393937,1.7991006,2.37423,2.20309006];
        let max_iter = 300;
        let n_mu = c_mu.len(); 
        let mut class_result = (vec![0usize; 6],vec![0.0; n_mu]);
        let mut ind_r = vec![0usize; n_mu];
        let mut dist_r = vec![0.0; n_mu];
        let mut count = 0;
        let mut dist = 0.0;
        let mut c_mu_new = vec![[0.0;3];n_mu];
        let result = loop{
            class_result = cvt_classification(&grids, &lambda_r, &c_mu);
            println!("Step {:?}:", &count);
            ind_r = class_result.0;
            dist_r = class_result.1;
            let mut init_dist_r = 0.0;
            dist_r.iter().zip(lambda_r.iter()).for_each(|(d, w)|{
                init_dist_r += w * d * d;
            });
            dist = init_dist_r.sqrt();
        
            println!("    Initial distance_r = {:?}.", &dist);
        

            let mut c_mu_new = cvt_update_cmu(&grids, &lambda_r, &c_mu, &ind_r);
            let mut sum = 0.0;
            if count == 0 {
                //println!{"first ind_r is: {:?}", &ind_r};
            }
            if count == max_iter - 1 {
                c_mu_new.iter().zip(c_mu.iter()).for_each(|([x,y,z],[a,b,c])|{
                    sum += ((x-a) * (x-a) + (y-b) * (y-b) + (z-c) * (z-c)).sqrt();
                });
                c_mu = c_mu_new;
                let ind_mu = cvt_find_corresponding_point(&grids, &lambda_r, &c_mu, &ind_r);
                break (ind_mu, sum);
            } 

            let mut criterion = 0.0;
            c_mu.iter().for_each(|[x,y,z]|{
                criterion += 1e-6 * ((x * x + y * y + z * z).sqrt())
            });
            c_mu_new.iter().zip(c_mu.iter()).for_each(|([x,y,z],[a,b,c])|{
                sum += ((x-a) * (x-a) + (y-b) * (y-b) + (z-c) * (z-c)).sqrt();
            });
        
            if sum <= criterion{
                c_mu = c_mu_new;
                let ind_mu = cvt_find_corresponding_point(&grids, &lambda_r, &c_mu, &ind_r);
                println!("Random points converged after {} iterations.", &count-1);
                println!("Dist: {:?}", &dist);
                println!{"ind_mu:{:?}", &ind_mu};
                break (ind_mu, sum);
            }else {
                c_mu = c_mu_new;
                count += 1;
            }


        };

    }
    

    #[test]
    fn test_cvt_isdf_v2(){
        let mut c_mu = vec![[0.7714985901267754, -0.6755196731706365, -0.6364944680908939],
        [0.15043771725854005, -0.23707678306916777, -1.6653834130341063], 
        [-0.1725698468085719, -0.7787796997386303, 0.6807511561770694], 
        [0.5482312479792356, 0.6804550814603627, 0.11163817160674155], 
        [0.5150491783871695, 1.7913140816269308, -0.2758841263761736], 
        [0.3127751508183599, 0.02217722056871697, 0.9455437073313467], 
        [-0.7643018723461648, 0.2598386216263768, -0.893459020354171], 
        [-0.6669435779967611, -0.14797881779914046, 0.5178318308337108], 
        [-1.339117197463132, 0.6509680938088687, 0.8357420031497302], 
        [0.3555573512781686, -0.6670006294386215, -0.9428563507028342], 
        [-0.9947135941509027, 0.6464318744173088, -0.6521902404733537], 
        [-0.4224091076454404, 0.705188132651095, -1.1190475794086954], 
        [0.7809568444164332, -0.8596642425156008, 1.2542157328855819], 
        [-0.579894437232492, -1.5925606164327972, -0.9601850796482453], 
        [-0.6422887194916226, 1.419706513442975, 0.24160054776467374], 
        [-0.5766265379812945, -1.0014619903058162, -1.575214766208227], 
        [1.2533205315046352, -2.4384958183542125, 0.19362738807762567], 
        [0.2683943492020745, 0.8811695928392411, 0.6931562633493071], 
        [1.5065354512982798, -0.48861427228906823, 0.502202935938445], 
        [0.253941141943878, 1.0953543435208666, 0.059388416896823504], 
        [0.4043254198526146, 1.183070213222598, -1.0808392284410275], 
        [-0.10293895261751486, 1.0104561658257314, -0.6572513838682457]];
        let grids_old = vec![[1.702325433088658357e-01,-1.245391510221033338e+00,-5.696354461899287847e-01],
        [3.674331632124046565e-01,3.674331632124046565e-01,-1.327993424356982644e+00],
        [2.745595072212403571e-03,0.000000000000000000e+00,0.000000000000000000e+00],
        [1.359711646966920429e-01,-3.812818313406721726e-01,-1.107506593500260594e+00],
        [3.043534884539474406e-01,3.043534884539474406e-01,-3.134903722300098838e+00],
        [-4.654278679698828358e-01,-1.010331148199145934e+00,3.940193756274395831e+00],
        [6.534711036842162812e+00,-6.534711036842162812e+00,-1.011068136584997390e+00],
        [-4.481549290489837412e-01,4.481549290489837412e-01,2.516854273702063161e+00],
        [1.577250956166978413e-01,3.908244143243941315e-01,2.456643961934580567e+00],
        [-3.620457302206792627e-01,-9.858033713583524982e-02,2.094598231713901360e+00],
        [-5.710825087838313330e+00,-9.240309095927992544e+00,0.000000000000000000e+00],
        [2.507909605792553176e-01,2.583198634695747398e+00,2.707434922513836106e+00],
        [3.204751864091725522e-01,6.956739923340095055e-01,3.478157561801589637e+00],
        [1.836926860253149385e-01,6.146760374483999367e-01,-1.343863559921979922e+00],
        [9.701864262430384689e-02,-9.701864262430384689e-02,2.232862759470598757e+00],
        [-2.262402215936017060e-04,-2.262402215936017060e-04,2.455965241269799648e+00],
        [-2.195192567444286535e-01,-2.195192567444286535e-01,2.261093633526035873e+00],
        [4.636700399153444452e-01,-4.636700399153444452e-01,-2.460513093953861363e-01],
        [8.517268556474394403e-02,8.517268556474394403e-02,-1.964575619406815632e-01],
        [0.000000000000000000e+00,-6.379883067115178719e+00,1.277951160877686476e+01],
        [-1.708258204160554516e+00,4.618524525495097066e+00,-1.708258204160554516e+00],
        [-9.809348033562516056e-01,-4.486743564631305303e-01,2.322560001167994326e+00],
        [-1.711298392231823007e+00,-3.154211359867605813e-01,7.453455697027575599e-01],
        [-6.994436576156705110e-01,-1.726194293516733369e+00,1.757200304318910167e+00],
        [-2.086136173375924496e+00,6.544747706977241952e-01,1.420706176083964323e+00],
        [-3.977234760975836281e+00,1.075303267531802298e+01,-1.520590799041255714e+00],
        [-8.505757218518871454e+00,1.044271629999135031e+00,5.384925119129563953e+00],
        [-7.646614255961726059e+00,-7.646614255961726059e+00,3.483978480588975302e+00],
        [7.439259015565388822e+00,4.587338507653521802e+00,-2.130694545718941235e+00],
        [6.250143602501206708e+00,-6.394329248935273213e-01,-6.394329248935273213e-01]];
        let lambda_r_old = vec![8.617006427195400553e-03,7.937239626547682895e-14,1.760886706432399726e-09,
        6.7642225382860442619e-03,5.947420498011038448e-02,2.032527283814727567e-02,
        5.704497941183920284e-01,1.668605454028351245e-03,1.222572166790015288e-03,
        2.175416664453362018e-03,5.130135058234682965e+00,3.347466169108940376e-02,
        7.344928524917152847e-03,1.060012624608557691e-02,3.222083299418788743e-04, 
        8.566724605125726754e-11,1.307584516297893583e-06,1.727644794861145838e-03,
        2.271083714195829776e-04,1.151590468674740286e+01,3.982582909791119730e-01,
        4.491965593867435803e-03,5.886820396441398880e-03,1.902136204864642022e-02,
        2.277220063851818910e-02,1.217082363872248996e+00,3.017914610962455058e+00,
        6.641129257251641604e+00,1.971915454082618369e-01,7.693019519590487132e-01];

        let threshold = 1.0e-10;
        let effective_ind = lambda_r_old.iter()
        .enumerate()
        .filter(|(_, &r)| r >= threshold)
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
        //println!("effective index: {:?}", &effective_ind);
    
        let ngrids = effective_ind.len();
        let mut lambda_r = vec![0.0; effective_ind.len()];
        lambda_r.iter_mut().zip(effective_ind.iter()).for_each(|(new,index_new)|{
            *new = lambda_r_old[*index_new];
        });
        let mut grids = vec![[0.0;3]; effective_ind.len()];
        grids.iter_mut().zip(effective_ind.iter()).for_each(|(new,index_new)|{
            new.iter_mut().zip(grids_old[*index_new].iter()).for_each(|(a,b)|{
                *a = *b;
            })
        });
        let max_iter = 300;
        let n_mu = c_mu.len(); 
        let mut class_result = (vec![0usize; 6],vec![0.0; n_mu]);
        let mut ind_r = vec![0usize; n_mu];
        let mut dist_r = vec![0.0; n_mu];
        let mut count = 0;
        let mut dist = 0.0;
        let mut c_mu_new = vec![[0.0;3];n_mu];
        let result = loop{
            class_result = cvt_classification(&grids, &lambda_r, &c_mu);
            println!("Step {:?}:", &count);
            ind_r = class_result.0;
            //println!("ind_r:{:?}", &ind_r);
            dist_r = class_result.1;
            let mut init_dist_r = 0.0;
            dist_r.iter().zip(lambda_r.iter()).for_each(|(d, w)|{
                init_dist_r += w * d * d;
            });
            dist = init_dist_r.sqrt();
        
            println!("    Initial distance_r = {:?}.", &dist);
        

            let mut c_mu_new = cvt_update_cmu(&grids, &lambda_r, &c_mu, &ind_r);
            let mut sum = 0.0;
            if count == 0 {
                //println!{"first ind_r is: {:?}", &ind_r};
            }
            if count == max_iter - 1 {
                c_mu_new.iter().zip(c_mu.iter()).for_each(|([x,y,z],[a,b,c])|{
                    sum += ((x-a) * (x-a) + (y-b) * (y-b) + (z-c) * (z-c)).sqrt();
                });
                c_mu = c_mu_new;
                let ind_mu = cvt_find_corresponding_point(&grids, &lambda_r, &c_mu, &ind_r);
                break (ind_mu, sum);
            } 

            let mut criterion = 0.0;
            c_mu.iter().for_each(|[x,y,z]|{
                criterion += 1e-6 * ((x * x + y * y + z * z).sqrt())
            });
            c_mu_new.iter().zip(c_mu.iter()).for_each(|([x,y,z],[a,b,c])|{
                sum += ((x-a) * (x-a) + (y-b) * (y-b) + (z-c) * (z-c)).sqrt();
            });
        
            if sum <= criterion{
                c_mu = c_mu_new;
                let ind_mu = cvt_find_corresponding_point(&grids, &lambda_r, &c_mu, &ind_r);
                println!("Random points converged after {} iterations.", &count-1);
                println!("Dist: {:?}", &dist);
                println!{"ind_mu:{:?}", &ind_mu};
                break (ind_mu, sum);
            }else {
                c_mu = c_mu_new;
                count += 1;
            }


        };
        

    }
    #[test]
    fn test_cvt_isdf_v3(){
        let mut c_mu = vec![[0.7714985901267754, -0.6755196731706365, -0.6364944680908939],
        [0.15043771725854005, -0.23707678306916777, -1.6653834130341063], 
        [-0.1725698468085719, -0.7787796997386303, 0.6807511561770694], 
        [0.5482312479792356, 0.6804550814603627, 0.11163817160674155], 
        [0.5150491783871695, 1.7913140816269308, -0.2758841263761736], 
        [0.3127751508183599, 0.02217722056871697, 0.9455437073313467], 
        [-0.7643018723461648, 0.2598386216263768, -0.893459020354171], 
        [-0.6669435779967611, -0.14797881779914046, 0.5178318308337108], 
        [-1.339117197463132, 0.6509680938088687, 0.8357420031497302], 
        [0.3555573512781686, -0.6670006294386215, -0.9428563507028342], 
        [-0.9947135941509027, 0.6464318744173088, -0.6521902404733537], 
        [-0.4224091076454404, 0.705188132651095, -1.1190475794086954], 
        [0.7809568444164332, -0.8596642425156008, 1.2542157328855819], 
        [-0.579894437232492, -1.5925606164327972, -0.9601850796482453], 
        [-0.6422887194916226, 1.419706513442975, 0.24160054776467374], 
        [-0.5766265379812945, -1.0014619903058162, -1.575214766208227], 
        [1.2533205315046352, -2.4384958183542125, 0.19362738807762567], 
        [0.2683943492020745, 0.8811695928392411, 0.6931562633493071], 
        [1.5065354512982798, -0.48861427228906823, 0.502202935938445], 
        [0.253941141943878, 1.0953543435208666, 0.059388416896823504], 
        [0.4043254198526146, 1.183070213222598, -1.0808392284410275], 
        [-0.10293895261751486, 1.0104561658257314, -0.6572513838682457]];
        let grids_old = vec![[1.702325433088658357e-01,-1.245391510221033338e+00,-5.696354461899287847e-01],
        [3.674331632124046565e-01,3.674331632124046565e-01,-1.327993424356982644e+00],
        [2.745595072212403571e-03,0.000000000000000000e+00,0.000000000000000000e+00],
        [1.359711646966920429e-01,-3.812818313406721726e-01,-1.107506593500260594e+00],
        [3.043534884539474406e-01,3.043534884539474406e-01,-3.134903722300098838e+00],
        [-4.654278679698828358e-01,-1.010331148199145934e+00,3.940193756274395831e+00],
        [6.534711036842162812e+00,-6.534711036842162812e+00,-1.011068136584997390e+00],
        [-4.481549290489837412e-01,4.481549290489837412e-01,2.516854273702063161e+00],
        [1.577250956166978413e-01,3.908244143243941315e-01,2.456643961934580567e+00],
        [-3.620457302206792627e-01,-9.858033713583524982e-02,2.094598231713901360e+00],
        [-5.710825087838313330e+00,-9.240309095927992544e+00,0.000000000000000000e+00],
        [2.507909605792553176e-01,2.583198634695747398e+00,2.707434922513836106e+00],
        [3.204751864091725522e-01,6.956739923340095055e-01,3.478157561801589637e+00],
        [1.836926860253149385e-01,6.146760374483999367e-01,-1.343863559921979922e+00],
        [9.701864262430384689e-02,-9.701864262430384689e-02,2.232862759470598757e+00],
        [-2.262402215936017060e-04,-2.262402215936017060e-04,2.455965241269799648e+00],
        [-2.195192567444286535e-01,-2.195192567444286535e-01,2.261093633526035873e+00],
        [4.636700399153444452e-01,-4.636700399153444452e-01,-2.460513093953861363e-01],
        [8.517268556474394403e-02,8.517268556474394403e-02,-1.964575619406815632e-01],
        [0.000000000000000000e+00,-6.379883067115178719e+00,1.277951160877686476e+01],
        [-1.708258204160554516e+00,4.618524525495097066e+00,-1.708258204160554516e+00],
        [-9.809348033562516056e-01,-4.486743564631305303e-01,2.322560001167994326e+00],
        [-1.711298392231823007e+00,-3.154211359867605813e-01,7.453455697027575599e-01],
        [-6.994436576156705110e-01,-1.726194293516733369e+00,1.757200304318910167e+00],
        [-2.086136173375924496e+00,6.544747706977241952e-01,1.420706176083964323e+00],
        [-3.977234760975836281e+00,1.075303267531802298e+01,-1.520590799041255714e+00],
        [-8.505757218518871454e+00,1.044271629999135031e+00,5.384925119129563953e+00],
        [-7.646614255961726059e+00,-7.646614255961726059e+00,3.483978480588975302e+00],
        [7.439259015565388822e+00,4.587338507653521802e+00,-2.130694545718941235e+00],
        [6.250143602501206708e+00,-6.394329248935273213e-01,-6.394329248935273213e-01]];
        let lambda_r_old = vec![8.617006427195400553e-03,7.937239626547682895e-14,1.760886706432399726e-09,
        6.7642225382860442619e-03,5.947420498011038448e-02,2.032527283814727567e-02,
        5.704497941183920284e-01,1.668605454028351245e-03,1.222572166790015288e-03,
        2.175416664453362018e-03,5.130135058234682965e+00,3.347466169108940376e-02,
        7.344928524917152847e-03,1.060012624608557691e-02,3.222083299418788743e-04, 
        8.566724605125726754e-11,1.307584516297893583e-06,1.727644794861145838e-03,
        2.271083714195829776e-04,1.151590468674740286e+01,3.982582909791119730e-01,
        4.491965593867435803e-03,5.886820396441398880e-03,1.902136204864642022e-02,
        2.277220063851818910e-02,1.217082363872248996e+00,3.017914610962455058e+00,
        6.641129257251641604e+00,1.971915454082618369e-01,7.693019519590487132e-01];

        let threshold = 1.0e-10;
        let effective_ind = lambda_r_old.iter()
        .enumerate()
        .filter(|(_, &r)| r.abs() >= threshold)
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
        let n_mu = c_mu.len();
        if n_mu >= effective_ind.len() {
            panic!("n_mu is smaller than effective_ind!");
        }
    
        let ngrids = effective_ind.len();
        let mut lambda_r = vec![0.0; effective_ind.len()];
        lambda_r.iter_mut().zip(effective_ind.iter()).for_each(|(new,index_new)|{
            *new = lambda_r_old[*index_new];
        });
        //println!("weight:{:?}",& lambda_r);
        let mut rgrids = vec![[0.0;3]; effective_ind.len()];
        rgrids.iter_mut().zip(effective_ind.iter()).for_each(|(new,index_new)|{
            new.iter_mut().zip(grids_old[*index_new].iter()).for_each(|(a,b)|{
                *a = *b;
            })
        });

        let max_iter = 300;
        let mut class_result = (vec![0usize; n_mu],vec![0.0; n_mu]);
        let mut ind_r = vec![0usize; n_mu];
        let mut dist_r = vec![0.0; n_mu];
        let mut count = 0;
        let mut dist = 0.0;
        let mut c_mu_new = vec![[0.0;3];n_mu];
        let result = loop{
            class_result = cvt_classification(&rgrids, &lambda_r, &c_mu);
            println!("Step {:?}:", &count);
            ind_r = class_result.0;
            dist_r = class_result.1;
            let mut init_dist_r = 0.0;
            dist_r.iter().zip(lambda_r.iter()).for_each(|(d, w)|{
                init_dist_r += w * d * d;
            });
            dist = init_dist_r.sqrt();
            
            println!("    Initial distance_r = {:?}.", &dist);
            
    
            let mut c_mu_new = cvt_update_cmu(&rgrids, &lambda_r, &c_mu, &ind_r);
            let mut sum = 0.0;
            if count == 0 {
                //println!{"first cvt_classification is: {:?}", &ind_r};
            }
            if count == max_iter - 1 {
                c_mu_new.iter().zip(c_mu.iter()).for_each(|([x,y,z],[a,b,c])|{
                    sum += ((x-a) * (x-a) + (y-b) * (y-b) + (z-c) * (z-c)).sqrt();
                });
                c_mu = c_mu_new;
                let ind_mu = cvt_find_corresponding_point(&rgrids, &lambda_r, &c_mu, &ind_r);
                break (ind_mu, sum);
            } 
    
            let mut criterion = 0.0;
            c_mu.iter().for_each(|[x,y,z]|{
                criterion += 1e-6 * ((x * x + y * y + z * z).sqrt())
            });
            c_mu_new.iter().zip(c_mu.iter()).for_each(|([x,y,z],[a,b,c])|{
                sum += ((x-a) * (x-a) + (y-b) * (y-b) + (z-c) * (z-c)).sqrt();
            });
            
            if sum <= criterion{
                c_mu = c_mu_new;
                let ind_mu = cvt_find_corresponding_point(&rgrids, &lambda_r, &c_mu, &ind_r);
                println!("Random points converged after {} iterations.", &count-1);
                println!("Dist: {:?}", &dist);
                break (ind_mu, sum);
            }else {
                c_mu = c_mu_new;
                count += 1;
            }   
    
        };    


    }
    
}