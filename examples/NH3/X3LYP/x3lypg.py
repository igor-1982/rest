#!/usr/bin/env python3

import pyscf
import numpy
from pyscf import gto, scf, ci,df,lib, dft
import scipy
from time import ctime, time

lib.num_threads(4)
TimeStart = time()
mol = gto.Mole(
        atom='''
         N       -2.1988391019      1.8973746268      0.0000000000
         H       -1.1788391019      1.8973746268      0.0000000000
         H       -2.5388353987      1.0925460144     -0.5263586446
         H       -2.5388400276      2.7556271745     -0.4338224694 ''',
        basis='cc-pvqz',verbose=4
    ).build()
method = dft.RKS(mol).density_fit(auxbasis="def2-SVP-JKFIT")
#auxbasis = df.addons.aug_etb_for_dfbasis(mol, beta=2.0, start_at=1)
#method = dft.RKS(mol).density_fit(auxbasis)
method.xc = 'x3lypg'
method.init_guess = 'vsap'
#method.grids.becke_scheme = dft.original_becke
#method.grids.level = 3
print('Default DFT(X3LYPG).  E = %.12f' % method.kernel())
print("Total job time: %10.2f(wall)" %(time()-TimeStart))
