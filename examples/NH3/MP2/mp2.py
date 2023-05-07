#!/usr/bin/env python3
import pyscf
import numpy
from pyscf import gto, scf, ci,df,lib, dft,mp
import scipy
from time import ctime, time

lib.num_threads(1)
TimeStart = time()
mol = gto.Mole(
        atom='''
         N       -2.1988391019      1.8973746268      0.0000000000
         H       -1.1788391019      1.8973746268      0.0000000000
         H       -2.5388353987      1.0925460144     -0.5263586446
         H       -2.5388400276      2.7556271745     -0.4338224694 ''',
        basis='cc-pvdz',verbose=4
    ).build()
method = scf.RHF(mol).density_fit(auxbasis="def2-svp-jkfit")
mf = method.run()
print("SCF job time: %10.2f(wall)" %(time()-TimeStart))
mp.MP2(mf).run()
print("Total job time: %10.2f(wall)" %(time()-TimeStart))
