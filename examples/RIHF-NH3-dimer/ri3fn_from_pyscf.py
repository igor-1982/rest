#!/usr/bin/env python3                                                                                            
import pyscf
import numpy
from pyscf import gto, scf, ci,df,lib
import scipy
from time import ctime, time
lib.num_threads(6)
TimeStart = time()
mol = gto.Mole(
        atom='''
          N       -2.1988391019      1.8973746268      0.0000000000
          H       -1.1788391019      1.8973746268      0.0000000000
          H       -2.5388353987      1.0925460144     -0.5263586446
          H       -2.5388400276      2.7556271745     -0.4338224694
          N       -4.1988391019      1.8973746268      0.0000000000
          H       -3.1788391019      1.8973746268      0.0000000000
          H       -4.5388353987      1.0925460144     -0.5263586446
          H       -4.5388400276      2.7556271745     -0.4338224694''',
         charge=1,spin=1,basis='aug-cc-pv5z',verbose=4
      ).build()
mf = scf.UHF(mol).density_fit(auxbasis="def2-tzvp-jkfit")
#mf.init_guess = '1e'
mf.diis = 'diis'
print("Total energy:",mf.kernel())
print("Total job time: %10.2f(wall)" %(time()-TimeStart))

