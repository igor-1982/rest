#!/usr/bin/env python3                                                                                            
import pyscf
import numpy
from pyscf import gto, scf, ci,df,lib, mp
from pyscf.mp.dfump2_native import SCSMP2
from pyscf.mp.dfump2_native import DFUMP2
from pyscf.lib import chkfile
import scipy
from time import ctime, time
lib.num_threads(6)
TimeStart = time()
mol = gto.Mole(
        atom='''
          H       0.0000000000     0.0000000000     0.0000000000
          H       0.7600000000     0.0000000000     0.0000000000''',
         charge=-1,spin=1,basis='cc-pvdz',verbose=6
      ).build()
method = scf.UHF(mol).density_fit(auxbasis="def2-svp-jkfit")
#method = scf.UKS(mol).density_fit(auxbasis="def2-svp-jkfit")
#method.xc = 'b3lypg'
mf = method.run()
scspt = SCSMP2(mf, ps=6/5, pt=1/3, auxbasis="def2-svp-jkfit")
scspt.kernel()
#pt = DFUMP2(mf, auxbasis="def2-svp-jkfit")
#pt.run()

#print("Total HF  energy:",method.run())
#print("Total MP2 energy:",mp.MP2(mf).run())
print("Total job time: %10.2f(wall)" %(time()-TimeStart))

