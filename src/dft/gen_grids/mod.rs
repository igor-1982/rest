//! Gen_grids is a library that produces numerical integration grid for
//! molecules based on atom coordinates, atom types, and basis set information. <br>
//! This work is based on [Numgrid] and [PySCF].
//! 
//! [Numgrid]: https://github.com/dftlibs/numgrid
//! [PySCF]: https://pyscf.org/


mod atom;
mod becke_partitioning;
mod bragg;
mod bse;
mod comparison;
mod lebedev;
mod parameters;
mod python;
mod radial;
mod tables;
pub mod prune;

pub use self::atom::atom_grid;
pub use self::atom::atom_grid_bse;
pub use self::lebedev::angular_grid;
pub use self::radial::radial_grid_kk;
pub use self::radial::radial_grid_lmg;
pub use self::radial::radial_grid_lmg_bse;
