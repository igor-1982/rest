#!/usr/bin/env python3                                                                                            
import pyscf
import numpy
from pyscf import gto, scf, ci,df,lib
from pyscf.lib import chkfile
import scipy
from time import ctime, time
lib.num_threads(6)
TimeStart = time()
mol = gto.Mole(
        atom='''
          N       -2.1988391019      1.8973746268      0.0000000000
          H       -1.1788391019      1.8973746268      0.0000000000
          H       -2.5388353987      1.0925460144     -0.5263586446
          H       -2.5388400276      2.7556271745     -0.4338224694''',
         charge=0,spin=0,basis='aug-cc-pv5z',verbose=4
      ).build()
method = scf.RHF(mol).density_fit(auxbasis="def2-tzvp-jkfit")
method.diis = 'diis'
D = method.get_init_guess(mol,'1e')
method.init_guess = '1e'
method.chkfile='init_guess.chk'
chkfile.save_mol(mol,method.chkfile)
chkfile.dump(method.chkfile,"init_guess",D)
print("Total energy:",method.kernel())
print("Total job time: %10.2f(wall)" %(time()-TimeStart))

