use std::fmt::Display;
use std::marker::PhantomData;
use std::ops::{Add, AddAssign, Sub, SubAssign};
use crate::rand_real_space;

mod tests{
    use rand::Rng;

    use crate::rand_real_space::{generate_random_points,random_tabulated_ao};
    use itertools::Itertools;
    #[test]
    fn test_random_points(){
        let mut rng = rand::thread_rng();
        let x = rng.gen_range(0.0..1.0);
        let p = generate_random_points(4);
        println!("{:?}", p);
        let c = 3/2;
        println!("{}",c);
        let mut perms = (0..5).permutations(2);
        perms.for_each(|x|{
            println!("{:?}",x);
        })
    }

    fn test_tabulated_ao(){
        let mut rng = rand::thread_rng();
        let x = rng.gen_range(0.0..1.0);
        let p = generate_random_points(4);
        println!("{:?}", p);
        let c = 3/2;
        println!("{}",c);
        let mut perms = (0..5).permutations(2);
        perms.for_each(|x|{
            println!("{:?}",x);
        })
    }
}