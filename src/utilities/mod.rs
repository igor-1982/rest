use time::{DateTime,Local};
use std::{time::Instant, collections::HashMap, ops::Range};
use regex::Regex;
enum DebugTiming {
   Yes,
   Not,
}
const DEBUG_PRINT: DebugTiming = DebugTiming::Not;

pub struct TimeRecords {
    items: HashMap<String, (Instant,f64,bool,String)>
}

impl TimeRecords {
    pub fn new() -> TimeRecords {
        TimeRecords{
            items: HashMap::new()
        }
    }

    pub fn new_item(&mut self, name: &str, comment: &str) {
        let item = (Instant::now(),0.0,false, comment.to_string());
        &self.items.insert(name.to_string(),item);
    }

    pub fn count_start(&mut self, name: &str) {
        if let Some(item) = self.items.get_mut(name) {
            if ! item.2 {
                item.0 = Instant::now();
                item.2 = true;
            } else {
                println!("WARNING: the time record for {} has been turned on previously.", name);
            }
        } else {
            println!("WARNING: the time record for {} has not been initialized.", name);
        }
    }

    pub fn count(&mut self, name: &str) {
        if let Some(item) = self.items.get_mut(name) {
            if item.2 {
                item.1 += item.0.elapsed().as_secs_f64();
                item.2 = false;
            } else {
                println!("WARNING: the time record for {} has been turned off previously.", name);
            }
        } else {
            println!("WARNING: the time record for {} has not been initialized.", name);
        }
    }

    pub fn report(&self,name: &str) {
        if let Some(item) = self.items.get(name) {
            let sp = format!("{:20} {:8.3} s for {}", &name, item.1, item.3);
            println!("{}",sp);
        } else {
            println!("WARNING: the time record for {} has not been initialized.", name);
        }
    }

    pub fn report_all(&self) {
        println!("Detailed time report:");
        self.items.iter().for_each(|(name, item)| {
            println!("{:10}|: {:8.3} s for {}", &name, item.1, item.3);
        });
    }
}


pub fn init_timing() -> DateTime<Local> {
    time::Local::now()
}

pub fn timing(dt0: &DateTime<Local>, iprint: Option<&str>) -> DateTime<Local> {
    let dt1 = time::Local::now();
    match DEBUG_PRINT {
        DebugTiming::Yes => {
            match iprint {
                None => {dt1},
                Some(header) => {
                    let timecost1 = (dt1.timestamp_millis()-dt0.timestamp_millis()) as f64 /1000.0;
                    println!("{:30} cost {:6.2} seconds", header, timecost1);
                    dt1
                }
            }
        },
        DebugTiming::Not => dt1
    }
}

pub fn debug_print_slices(s: &[f64]) {
    let mut tmp_s:String = format!("debug: ");
    s.iter().for_each(|x| {
        tmp_s = format!("{},{:16.8}", tmp_s, x);
    });
    println!("{}",tmp_s);
}

//#[link(name="openblas")]
extern "C" {
    pub fn openblas_get_num_threads() -> ::std::os::raw::c_int;
    pub fn openblas_set_num_threads(n: ::std::os::raw::c_int);
    pub fn goto_get_num_threads() -> ::std::os::raw::c_int;
    pub fn goto_set_num_threads(n: ::std::os::raw::c_int);
}
//extern "C" {
//    pub fn omp_get_num_threads() -> ::std::os::raw::c_int;
//    pub fn omp_set_num_threads(n: ::std::os::raw::c_int);
//}

pub fn omp_get_num_threads_wrapper() -> usize {
    let num_threads = unsafe{openblas_get_num_threads()} as usize;
    num_threads
}
/// NOTE: the current OpenBLAS only supports at most 32 threads. Otherwise, it panics with an error:  
/// "BLAS : Program is Terminated. Because you tried to allocate too many memory regions."
pub fn omp_set_num_threads_wrapper(n:usize)  {
    unsafe{if n<=256 {
        openblas_set_num_threads(n as std::os::raw::c_int);
        goto_set_num_threads(n as std::os::raw::c_int);
        //omp_set_num_threads(n  as std::os::raw::c_int);
    } else {
        openblas_set_num_threads(256);
        goto_set_num_threads(256);
        //omp_set_num_threads(32);
    } } 
}

//#[test]
//fn debug_time() {
//    let now = Instant::now();
//}

pub fn balancing(num_tasks:usize, num_threads: usize) -> Vec<Range<usize>> {
    let mut distribute_vec: Vec<Range<usize>> = vec![0..num_tasks;num_threads];
    let chunk_size = num_tasks/num_threads;
    let chunk_rest = num_tasks%num_threads;

    let mut start = 0_usize;
    let mut count = chunk_rest as i32;
    distribute_vec.iter_mut().enumerate().for_each(|(i,data)| {
        if count >0 {
            *data = start..start + chunk_size+1;
            start += chunk_size+1;
            count -= 1;
        } else {
            *data = start..start + chunk_size;
            start += chunk_size;
            //count -= 1;
        }
    });

    distribute_vec
}

pub fn convert_scientific_notation_to_fortran_format(n: &String) -> String {
    let re = Regex::new(r"(?P<num> *-?\d.\d*)[E|e](?P<exp>-?\d)").unwrap();
    let o_len = n.len();

    if let Some(cap) = re.captures(n) {
        let main_part = cap["num"].to_string();
        let exp_part = cap["exp"].to_string();
        let exp: i32 = exp_part.parse().unwrap();
        let out_str = if exp>=0 {
            format!("{}E+{:0>2}",main_part,exp)
        } else {
            format!("{}E-{:0>2}",main_part,exp.abs())
        };
        let n_len = out_str.len();
        return out_str[n_len-o_len..n_len].to_string()
    } else {
        panic!("Error: the input string is not a standard scientific notation")
    }

}

#[test]
fn test_scientific_number() {
    let dd = -0.0003356;
    let sdd = format!("{:16.8E}",dd);
    println!("{}", &sdd);
    println!("{}", convert_scientific_notation_to_fortran_format(&sdd));
    let dd = -100.0003356;
    let sdd = format!("{:16.8E}",dd);
    println!("{}", &sdd);
    println!("{}", convert_scientific_notation_to_fortran_format(&sdd));

}