use crate::molecule_io::Molecule;
use crate::constants::*;

//pub fn ecp_scalar_spheric(mol: &Molecule, i: usize, j: usize) {
//    let li = mol.cint_bas[i][BAS_ANG];
//    let lj = mol.cint_bas[j][BAS_ANG];
//    let di = (li*2+1)*mol.cint_bas[i][BAS_CTR];
//    let dj = (lj*2+1)*mol.cint_bas[j][BAS_CTR];
//    let dij = di*dj;
//    let comp = 1_usize;
//
//    let cache_size = ecp_scalar_cache_size_by_shell(mol, i, j, comp);
//
//
//}

//pub fn ecp_scalar_cache_size_by_shell(mol: &Molecule, i: usize, j:usize, comp: usize) -> usize {
//    let li = mol.cint_bas[i][BAS_ANG];
//    let ni_cart = (li+1)*(li+2)/2;
//    let nip = mol.cint_bas[i][BAS_PRM];
//    let nic = mol.cint_bas[i][BAS_CTR];
//
//    let lj = mol.cint_bas[j][BAS_ANG];
//    let nj_cart = (lj+1)*(lj+2)/2;
//    let njp = mol.cint_bas[j][BAS_PRM];
//    let njc = mol.cint_bas[j][BAS_CTR];
//
//    let lilj1 = li + lj + 1;
//    let lilc1 = li + ECP_LMAX + 1;
//    let ljlc1 = lj + ECP_LMAX + 1;
//
//    let d1 = lilj1;
//    let d2 = d1 * d1;
//    let d3 = d2 * d1;
//
//    let di1 = li + 1;
//    let di2 = di1 * di1;
//    let di3 = di2 * di1;
//    let dj1 = lj + 1;
//    let dj2 = dj1 * dj1;
//    let dj3 = dj2 * dj1;
//    let nrs =  2048;
//
//    let mut size1 =  nic*njc*(li+lj+1);
//    size1 += ((li+1)*ni_cart*(ECP_LMAX*2+1)*(li+ECP_LMAX+1) + (lj+1)*nj_cart*(ECP_LMAX*2+1)*(lj+ECP_LMAX+1));
//    size1 += ni_cart*(ECP_LMAX*2+1)*(lj+ECP_LMAX+1);
//    size1 += (lj+1)*nj_cart*(ECP_LMAX*2+1)*(lj+ECP_LMAX+1)*3;
//
//    let mut size2 = nrs * (li+lj+1 + 1 +  nic*(li+ECP_LMAX+1) + njc*(lj+ECP_LMAX+1) + nip*lilc1.max(njp*ljlc1));
//    size2 += lilc1 * ljlc1;
//    size2 += di1*di1*di1*lilc1.max(dj1*dj1*dj1*ljlc1)  * (ECP_LMAX*2 + 1);
//    let mut size = ni_cart*nj_cart*(nic*njc+2) * comp as i32;
//    size += nic*njc*(li+lj+1)*(li+ECP_LMAX+1)*(lj+ECP_LMAX+1);
//    //size += MAX(size1, size2);  bugs in bufsize estimation, not sure where's the error
//    size += size1 + size2 + 120;
//    size += ni_cart*(ECP_LMAX*2+1)*(lj+ECP_LMAX+1);
//    size += nip*njp*d2;
//    size += d3;
//    size += nic*njc*d3;
//    size += nif*di3;
//    size += njf*dj3;
//    size += nip*nic;
//    size += njp*njc;
//    size += mol.cint_atm.len(); 
//
//    size as usize
//
//}
