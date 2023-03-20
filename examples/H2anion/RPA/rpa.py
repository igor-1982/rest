#!/usr/bin/env python3                                                                                            
from pyscf import gto, scf, ci,df,lib, mp, gw
from pyscf.lib import chkfile
from pyscf.gw.urpa import URPA
from pyscf.gw.rpa import RPA, _get_scaled_legendre_roots
import scipy
from time import ctime, time
lib.num_threads(6)
TimeStart = time()
mol = gto.Mole(
        atom='''
          H       0.0000000000     0.0000000000     0.0000000000
          H       0.7600000000     0.0000000000     0.0000000000''',
         charge=-1,spin=1,basis='cc-pvdz',verbose=4
       ).build()

method = scf.UKS(mol).density_fit(auxbasis="def2-svp-jkfit")
method.xc = 'pbe'
mf = method.run()
rpa = URPA(mf, auxbasis="def2-svp-jkfit")
#rpa = URPA(mf)
#lpq = rpa.ao2mo(mf.mo_coeff)
#print(lpq.shape)
#s = ""
#for x in range(len(lpq[0][0][0])):
#    s += "%16.8f" %lpq[0][0][0][x]
#print(s)  
#print(lpq[0])
#freqs, wts = _get_scaled_legendre_roots(20)
#print(freqs, wts)

e_tot = rpa.kernel()
print(e_tot)



#print("Total HF  energy:",method.run())
#print("Total MP2 energy:",mp.MP2(mf).run())
print("Total job time: %10.2f(wall)" %(time()-TimeStart))

