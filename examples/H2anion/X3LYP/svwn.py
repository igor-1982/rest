import numpy as np
import h5py
from pyscf import gto, scf, ci, dft
from pyscf.lib import chkfile
from time import ctime, time
TimeStart = time()
print("Job start from {}",ctime())

TimeStart = time()
mol = gto.Mole(
        atom='''
         H       0.0000 0.0000 0.0000
         H       0.7600 0.0000 0.0000''',
        charge=0,spin=0,basis='cc-pVDZ',verbose=4
    ).build()
#method = scf.UHF(mol).density_fit(auxbasis="def2-tzvp-jkfit")
method = dft.RKS(mol)
method.xc = 'LDA,VWNRPA'
method.init_guess = '1e'
method.grids.becke_scheme = dft.original_becke

print('Default DFT(LDA).  E = %.12f' % method.kernel())
print("Total job time: %10.2f(wall)" %(time()-TimeStart))


