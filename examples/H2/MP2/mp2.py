#!/usr/bin/env python3                                                                                            
import pyscf
import numpy
from pyscf import gto, scf, ci,df,lib, mp
from pyscf.lib import chkfile
import scipy
from time import ctime, time
lib.num_threads(6)
TimeStart = time()
mol = gto.Mole(
        atom='''
          H       0.0000000000     0.0000000000     0.0000000000
          H       0.7600000000     0.0000000000     0.0000000000''',
         charge=0,spin=0,basis='cc-pvdz',verbose=4
      ).build()
method = scf.RHF(mol).density_fit(auxbasis="def2-svp-jkfit")
#method.xc = 'b3lypg'
mf = method.run()
mp.MP2(mf).run()

#print("Total HF  energy:",method.run())
#print("Total MP2 energy:",mp.MP2(mf).run())
print("Total job time: %10.2f(wall)" %(time()-TimeStart))

