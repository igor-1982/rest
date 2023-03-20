#!/usr/bin/env python3

import pyscf
import numpy
from pyscf import gto, scf, ci,df,lib, dft
import scipy
from time import ctime, time

lib.num_threads(1)
TimeStart = time()
mol = gto.Mole(
        atom='''
         H       0.0000 0.0000 0.0000
         H       0.7600 0.0000 0.0000''',
        charge=0,spin=0,basis='cc-pVQZ',verbose=4
    ).build()
#method = scf.UHF(mol).density_fit(auxbasis="def2-tzvp-jkfit")
method = dft.RKS(mol).density_fit(auxbasis="def2-tzvp-jkfit")
method.init_guess = 'vsap'
#method.scf
method.xc = 'x3lypg'
#method.init_guess = '1e'
#method.grids.becke_scheme = dft.original_becke
#method.grids.level = 3
method.kernel()
print("Total job time: %10.2f(wall)" %(time()-TimeStart))
