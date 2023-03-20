#!/usr/bin/env python3                                                                                            
import pyscf
import numpy
from pyscf import gto, scf, ci,df,lib, mp, gw
from pyscf.lib import chkfile
from pyscf.gw.rpa import RPA
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
         charge=0,spin=0,basis='cc-pvqz',verbose=4
      ).build()
method = scf.RKS(mol).density_fit(auxbasis="def2-svp-jkfit")
method.xc = 'pbe'
mf = method.run()
rpa = RPA(mf, auxbasis="def2-svp-jkfit")

#lpq = rpa.ao2mo(mf.mo_coeff)
#print(mf.mo_coeff)
#print(lpq.shape)
#s = ""
#for x in range(len(lpq[0][0])):
#    s += "%16.8f" %lpq[0][0][x]
#print(s)  
#print(lpq[0])

e_tot = rpa.kernel(nw=40)
print("RPA correlation energy: %16.8f Ha" %e_tot)

#print("Total HF  energy:",method.run())
#print("Total MP2 energy:",mp.MP2(mf).run())
print("Total job time: %10.2f(wall)" %(time()-TimeStart))

