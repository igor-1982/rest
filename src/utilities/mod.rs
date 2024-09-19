use clap::{Command, Arg, ArgMatches};
use rayon::ThreadPoolBuildError;
use time::{DateTime,Local};
use std::{time::Instant, collections::HashMap, ops::Range};
use regex::Regex;
enum DebugTiming {
   Yes,
   Not,
}
const DEBUG_PRINT: DebugTiming = DebugTiming::Not;

pub fn parse_input() -> ArgMatches {
    Command::new("rest")
        .version("0.1")
        .author("Igor Ying Zhang <igor_zhangying@fudan.edu.cn>")
        .about("Rust-based Electronic-Structure Tool (REST)")
        .arg(Arg::new("input_file")
             .short('i')
             .long("input-file")
             .value_name("input_file")
             .help("Input file including \"ctrl\" and \"geom\" block, in the format of either \"json\" or \"toml\"")
             .takes_value(true))
        .get_matches()
}

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
        if let Some(_) = self.items.get_mut(name) {
            println!("WARNING: the time record for {} has been initialized previously.", name);
            //self.count_start(name);
        } else {
            let item = (Instant::now(),0.0,false, comment.to_string());
            &self.items.insert(name.to_string(),item);
        }
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
            self.new_item(name, "");
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

    pub fn self_add(&mut self, sub_timerecords: &TimeRecords) {
        self.items.iter_mut().for_each(|(key,value)| {
            if let Some(sub_item) = sub_timerecords.items.get(key) {
                value.0 = Instant:: now();
                value.2 = false;
                value.1 += sub_item.1
            }
        })
    }
    pub fn max(&mut self, sub_timerecords: &TimeRecords) {
        self.items.iter_mut().for_each(|(key,value)| {
            if let Some(sub_item) = sub_timerecords.items.get(key) {
                let cur_time = value.1;
                value.0 = Instant:: now();
                value.2 = false;
                value.1 = sub_item.1.max(cur_time);
            }
        })
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
    let num_threads_openblas = unsafe{openblas_get_num_threads()} as usize;
    //let num_threads_goto = unsafe{goto_get_num_threads()} as usize;
    //println!("debug {:}, {:}", num_threads_goto, num_threads_openblas);
    //num_threads_openblas.max(num_threads_goto)
    //println!("debug {:}", num_threads_openblas);
    num_threads_openblas
}
/// NOTE: the current OpenBLAS only supports at most 32 threads. Otherwise, it panics with an error:  
/// "BLAS : Program is Terminated. Because you tried to allocate too many memory regions."
pub fn omp_set_num_threads_wrapper(n:usize)  {
    unsafe{
        openblas_set_num_threads(n as std::os::raw::c_int);
        goto_set_num_threads(n as std::os::raw::c_int);
    } 
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

/// initialize the parallization vector for the sync data with the size of num_tasks * per_communication
/// Eunsure that the data size communicated each time is less than 500 Mb
pub fn balancing_type_02(num_tasks:usize, num_threads: usize, per_communication: usize) -> Vec<Range<usize>> {
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

/// To balance the parallel tasks for preparing the symmetric data, like rimatr, ovlp (matrixupper), and so forth
//pub fn balancing_2(num_bas_shell: usize, num_threads: usize) {
//    let mut reset = false;
//    let mut num_tasks = 0;
//    let mut task_start = 0;
//    let mut task_len = 0;
//    let mut tasks: Vec<Range<usize>> = vec![];
//    (0..num_bas_shell).for_each(|i| {
//        if num_tasks <=num_bas_shell {
//            num_tasks += i;
//        } else {
//            num_tasks = 0;
//
//        }
//
//    });
//
//  //let n_baspair = (num_basis+1)*num_basis/2;
//  //let num_loc_tasks = n_baspair/num_threads;
//  //let num_left = n_baspair%num_threads;
//
//}


pub fn create_pool(num_threads: usize) -> Result<rayon::ThreadPool,ThreadPoolBuildError > {
    match rayon::ThreadPoolBuilder::new()
       .num_threads(num_threads)
       .build()
    {
       Err(e) => Err(e),
       Ok(pool) => Ok(pool),
    }
 }

pub fn convert_scientific_notation_to_fortran_format(n: &String) -> String {
    let re = Regex::new(r"(?P<num> *-?\d.\d*)[E|e](?P<exp>-?\d{1,2})").unwrap();
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
    let dd = 1.563E-16;
    let sdd = format!("{:16.8E}",dd);
    println!("{}", &sdd);
    println!("{}", convert_scientific_notation_to_fortran_format(&sdd));
    let dd = 1.563E16;
    let sdd = format!("{:16.8E}",dd);
    println!("{}", &sdd);
    println!("{}", convert_scientific_notation_to_fortran_format(&sdd));

}
