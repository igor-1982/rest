use std::ops::{Range, Add, Sub, Mul, Div, AddAssign, SubAssign, MulAssign, DivAssign};
use mpi::collective::SystemOperation;
use mpi::environment::Universe;
use mpi::request::WaitGuard;
use mpi::topology::{SimpleCommunicator, Rank};
use mpi::traits::*;
use num_traits::{One, Zero};
use tensors::{BasicMatrix, MatrixFull};
use std::fmt::Debug;

use crate::dft::Grids;
use crate::constants::MPI_CHUNK;
use crate::molecule_io::Molecule;
use crate::utilities::balancing;

/// ==== Memory distribution for the MPI implementation in REST ====
/// There are two intermediate objects are distributed during the MPI processors
/// 1) scf_io::<SCF>.grids {
///     coordinates: vec![<[f64;3]>; num_grids], contains the coordinates of all numerical grids (num_grids).
///     weights: vec![f64; num_grids], contains the integral weights of these numerical grids (num_grids).
///     ao: MatrixFull[num_grids, num_bas], tabulates all atomic orbital basis (num_bas) according to these numerical grids (num_grids).
///     aop: RIFull[num_grids, num_bas], tabulates the derivatives of all atomic orbital basis (num_bas) according to these numerical grids (num_grids).
///   } 
///          
///   The numerical grids (num_grids) are distributed across mpi processors. Such that each processor only stores a part of grids (loc_num_grids):
///    local_grids {
///     coordinates: vec![<[f64;3]>; loc_num_grids]
///     weights: vec![f64; loc_ num_grids]
///     ao: MatrixFull[loc_num_grids, num_orbs]
///     aop: RIFull[loc_num_grids, num_orbs]
///   } 
/// 
/// 2) scf_io::<SCF>.ri3fn: MatrixFull[num_auxbas, num_baspar]
///   The auxiliary basis (num_auxbas) are distributed across mpi processors. 
///   Such that each processor only stores a part of auxiliary basis functions (loc_num_aux):
///     local_ri3fn: MatrixFull[loc_num_auxbas, num_baspar]
///   NOTE:: In the generation of local_ri3fn, num_baspar is also distributed
/// 
/// 
/// In consequence, ri3mo is also distrubted across mpi processors.
///    
/// 

pub struct MPIOperator {
    pub universe: Universe,
    pub world: SimpleCommunicator,
    pub size: usize,
    pub rank: usize,
}

#[derive(Clone)]
pub struct MPIData {
    pub size: usize,
    pub rank: usize,
    pub grids: Option<Vec<Range<usize>>>,
    pub auxbas: Option<Vec<Range<usize>>>,
    pub baspar: Option<Vec<(Range<usize>, usize, usize)>>
}

impl MPIData {
    pub fn initialization() -> (Option<MPIOperator>,Option<MPIData>) {
        let universe = mpi::initialize().unwrap();
        let world = universe.world();
        let size = world.size() as usize;
        let rank = world.rank() as usize;

        if size >= 2 {
            (
                Some(MPIOperator {
                universe,
                world,
                size,
                rank
                }),
                Some(MPIData{
                    size, 
                    rank,
                    grids: None,
                    auxbas: None,
                    baspar: None
                })
            )
        } else {
            (None, None)
        }
    }

    pub fn distribute_grids_tasks(&mut self, grids: &Grids) -> Grids {
        let num_tasks = grids.weights.len();

        let mut distribute_vec = average_distribution(num_tasks, self.size) ;
        
        let local_range = &distribute_vec[self.rank];
        let local_coordinates = grids.coordinates[local_range.clone()].to_vec();
        let local_weights = grids.weights[local_range.clone()].to_vec();
        self.grids = Some(distribute_vec);

        let parallel_balancing = balancing(local_coordinates.len(), rayon::current_num_threads());
        return Grids {
            coordinates: local_coordinates,
            weights: local_weights,
            ao: None,
            aop: None,
            parallel_balancing,
        }
        
    }

    pub fn distribution_elec_pair(&self, start_mo: usize, num_occu: usize) -> Vec<[usize;2]> {

        let mut elec_pair: Vec<[usize;2]> = vec![];
        for i_state in start_mo..num_occu {
            for j_state in i_state..num_occu {
                elec_pair.push([i_state,j_state])
            }
        };

        let distribution_vec = average_distribution(elec_pair.len(), self.size);

        let loc_elec_pair = elec_pair[distribution_vec[self.rank].clone()].to_vec();

        loc_elec_pair

    }

    pub fn distribution_same_spin_virtual_orbital_pair(&self, lumo: usize, num_state: usize) -> Vec<[usize;2]> {

        let mut elec_pair: Vec<[usize;2]> = vec![];
        for i_state in lumo..num_state {
            for j_state in i_state+1..num_state {
                elec_pair.push([i_state,j_state])
            }
        };

        let distribution_vec = average_distribution(elec_pair.len(), self.size);

        let loc_elec_pair = elec_pair[distribution_vec[self.rank].clone()].to_vec();

        loc_elec_pair

    }

    pub fn distribution_opposite_spin_virtual_orbital_pair(&self, lumo_1: usize, lumo_2: usize, num_state: usize) -> Vec<[usize;2]> {

        let mut elec_pair: Vec<[usize;2]> = vec![];
        for i_state in lumo_1..num_state {
            for j_state in lumo_2..num_state {
                elec_pair.push([i_state,j_state])
            }
        };

        let distribution_vec = average_distribution(elec_pair.len(), self.size);

        let loc_elec_pair = elec_pair[distribution_vec[self.rank].clone()].to_vec();

        loc_elec_pair

    }

    pub fn distribute_rimatr_tasks(&mut self, num_auxbas: usize, num_basis: usize, cint_bas: Vec<Vec<usize>>,) {

        self.auxbas = Some(average_distribution(num_auxbas, self.size));


        // now distribute the basis pairs
        let num_basis = num_basis;
        let num_baspar = (num_basis+1)*num_basis/2;
        let n_basis_shell = cint_bas.len();

        let chunk_size = num_baspar/self.size;
        let chunk_rest = num_baspar%self.size;

        //println!("debug: num_basis: {}, num_baspar: {}, n_basis_shell: {}", num_basis, num_baspar, n_basis_shell);

        let (basbas2baspar , baspar2basbas) = prepare_baspair_map(num_basis);
        let mut distribute_vec = vec![(0..num_baspar, 0_usize, n_basis_shell);self.size];
        let mut start = 0_usize;
        let mut count = chunk_rest  as i32;
        distribute_vec.iter_mut().enumerate().for_each(|(i,(data, sbsh, ebsh))| {
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

        // reshape the basis pairs distribution according to basis_shells
        distribute_vec.iter_mut().for_each(|(data, sbsh, ebsh)| {
            let s_basis = baspar2basbas[data.start];
            let e_basis = baspar2basbas[data.end-1];
            let last_bsh = cint_bas.last().unwrap();


            let (s_bsh, e_bsh) = cint_bas.iter().enumerate()
            .fold((0, 0), |acc, (i_bsh,bsh)| {

                let (mut s_bsh, mut e_bsh) = acc;

                if s_basis[1] > bsh[0] && s_basis[1] < bsh[0]+bsh[1] {
                    s_bsh = (i_bsh+1) as usize;
                } else if s_basis[1] == bsh[0] {
                    s_bsh = i_bsh as usize;
                }
                if e_basis[1] >= bsh[0] && e_basis[1] < bsh[0]+bsh[1] {
                    e_bsh = i_bsh as usize;
                }
                (s_bsh, e_bsh)
            });

            //let s_bas = &cint_bas[s_bsh];
            //let e_bas = &cint_bas[e_bsh];
            *sbsh = s_bsh;
            *ebsh = e_bsh;

            if s_bsh < n_basis_shell {

                let s_bas = &cint_bas[s_bsh];
                let e_bas = &cint_bas[e_bsh];

                let s_baspar = basbas2baspar[[0, s_bas[0] as usize]];//((s_bas[0]+1)*s_bas[0]/2) as usize; //
                let e_baspar = basbas2baspar[[(e_bas[0] + e_bas[1]-1) as usize, (e_bas[0] + e_bas[1]-1) as usize]];

                *data = s_baspar .. e_baspar+1;

            } else {
                *sbsh = n_basis_shell-1;
                *data = num_baspar .. num_baspar;
            }

            //println!("debug after: baspar_range {:?} s_basis {:?} e_basis {:?} s_bsh {} e_bsh {}", &data, &s_basis, &e_basis, s_bsh, e_bsh);

        });

        self.baspar = Some(distribute_vec)



    }

}

pub fn prepare_baspair_map(n_basis: usize) -> (MatrixFull<usize>, Vec<[usize;2]>) {
    let n_baspar = (n_basis+1)*n_basis/2;

    let mut basbas2baspar = MatrixFull::new([n_basis,n_basis],0_usize);
    let mut baspar2basbas = vec![[0_usize;2];n_baspar];
    basbas2baspar.iter_columns_full_mut().enumerate().for_each(|(j,slice_x)| {
        &slice_x.iter_mut().enumerate().for_each(|(i,map_v)| {
            let baspar_ind = if i< j {(j+1)*j/2+i} else {(i+1)*i/2+j};
            *map_v = baspar_ind;
            baspar2basbas[baspar_ind] = [i,j];
        });
    });

    (basbas2baspar, baspar2basbas)
}

pub fn mpi_reduce<Q>(world: &SimpleCommunicator, data: &[Q], root_rank: usize, op: &SystemOperation) -> Vec<Q> 
where Q: Add<Output=Q> + AddAssign + 
         Sub<Output=Q> + SubAssign + 
         Mul<Output=Q> + MulAssign + 
         Div<Output=Q> + DivAssign + 
         Zero + Send + Sync + Copy + Debug + Buffer + 'static,
      [Q]: Buffer,
      Vec<Q>: BufferMut
      
{
    let rank = world.rank() as usize;
    let root_process = world.process_at_rank(root_rank as i32);

    let mut result: Vec<Q> = vec![Q::zero(); data.len()];

    //println!("debug rank {} and data {:?} in mpi_reduce before", rank, &data);

    if rank == root_rank {
        root_process.reduce_into_root(&data[..], &mut result, op);
    } else {
        root_process.reduce_into(&data[..], op);

    }

    world.barrier();

    result
}

pub fn mpi_broadcast<Q>(world: &SimpleCommunicator, data: &mut Q, root_rank: usize)
where Q: Send + Sync + Buffer + Debug + BufferMut + 'static,
{

    let root_process = world.process_at_rank(root_rank as i32);
    root_process.broadcast_into(data);

    let rank = world.rank();
    //println!("debug Rank {} received value: {:?}", rank, &data);

}

pub fn mpi_broadcast_vector<Q>(world: &SimpleCommunicator, data: &mut Vec<Q>, root_rank: usize)
where Q: Zero + Send + Sync + Copy + Buffer + Equivalence + Debug + 'static,
      Vec<Q>: BufferMut
{
    world.barrier();
    let rank = world.rank() as usize;
    //println!("debug rank {}", rank);
    let mut data_len = data.len();
    mpi_broadcast::<usize>(world, &mut data_len, root_rank);
    if data_len != 0 {
        //println!("debug data_len: {} in rank {}", data_len, rank);
        if rank!= root_rank {
            data.resize(data_len, Q::zero());
        }
        mpi_broadcast::<Vec<Q>>(world, data, root_rank);
    }
}

pub fn mpi_broadcast_matrixfull<Q>(world: &SimpleCommunicator, data: &mut MatrixFull<Q>, root_rank: usize)
where Q: Zero + Send + Sync + Copy + Buffer + Equivalence + Debug + 'static,
      [Q]: BufferMut
{
    let rank = world.rank() as usize;
    let mut data_size = data.size.clone();
    mpi_broadcast::<[usize;2]>(world, &mut data_size, root_rank);
    if data_size[0]*data_size[1] != 0 {
        if rank!= root_rank {
            *data = MatrixFull::new(data_size, Q::zero());
        }
        mpi_broadcast(world, &mut data.data, root_rank);
    }
}

pub fn mpi_isend_irecv_wrt_distribution<Q>(
    world: &SimpleCommunicator,
    data: &[Q],
    distribution: &[Range<usize>], 
    scale: usize) -> Vec<Vec<Q>>
where
    Q: Zero + Send + Sync + Buffer + Debug + Clone + BufferMut + 'static,
    [Q]: Buffer + BufferMut,
    Vec<Q>: BufferMut,
{
    let rank = world.rank() as usize;
    let size = world.size() as usize;

    let mut scale_vec = vec![scale; size];
    world.all_gather_into(&scale, &mut scale_vec[..]);

    let mut result: Vec<Vec<Q>> = vec![vec![]; distribution.len()];
    //let chunk_len = distribution[rank].len()*scale;
    result.iter_mut().enumerate().for_each(|(i,x)| *x = vec![Q::zero(); distribution[rank].len()*scale_vec[i]]);

    //println!("debug mpi 0 in rank {} with scale_vec = {:?}, distribution = {:?}", rank, &scale_vec, &distribution[rank]);

    for i in 0..size {
        for j in 0..size {
            mpi::request::scope(|scope| {
                let sreq = if rank == j {
                    let tmp_range = distribution[i].start*scale_vec[j] .. distribution[i].end*scale_vec[j];
                    Some(
                        world.process_at_rank(i as i32).immediate_send(scope, &data[tmp_range])
                    )
                } else {
                    None
                };
                if rank == i {
                    let result_slice = result[j].as_mut_slice();
                    let rreq = world.process_at_rank(j as i32).immediate_receive_into(scope, result_slice);
                    rreq.wait();
                } 
                if rank == j {
                    if let Some(sreq_j) = sreq {
                        sreq_j.wait();
                    }
                }
            });
        }
    }
    result
}


pub fn average_distribution(num_tasks: usize, size: usize) -> Vec<Range<usize>> {
    let mut distribute_vec: Vec<Range<usize>> = vec![0..num_tasks;size];
    
    let chunk_size = num_tasks/size;
    let chunk_rest = num_tasks%size;

    let mut start = 0_usize;
    let mut count = chunk_rest  as i32;
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

pub fn average_distribution_with_residue(num_tasks: usize, size: usize) -> (Vec<Range<usize>>, Range<usize>) {
    let mut distribute_vec: Vec<Range<usize>> = vec![0..num_tasks;size];
    
    let chunk_size = num_tasks/size;
    let chunk_rest = num_tasks%size;

    let mut start = 0_usize;
    distribute_vec.iter_mut().enumerate().for_each(|(i,data)| {
            *data = start..start + chunk_size;
            start += chunk_size;
    });

    (distribute_vec, start..num_tasks)
}