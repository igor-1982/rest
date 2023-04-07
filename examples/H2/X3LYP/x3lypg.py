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
        charge=0,spin=0,basis='cc-pVDZ',verbose=4
    ).build()
#method = scf.RHF(mol).density_fit(auxbasis="def2-tzvp-jkfit")
method = dft.RKS(mol).density_fit(auxbasis="def2-tzvp-jkfit")
method.xc = 'x3lypg'
D = method.get_init_guess(mol,'atom')
method.init_guess = 'atom'
method.chkfile='init_guess.chk'
pyscf.lib.chkfile.save_mol(mol,method.chkfile)
pyscf.lib.chkfile.dump(method.chkfile,"init_guess",D)

#for x in D: 
#   print(x[0:10])
#print("alpha:")
#for x in D[0]: 
#    print(x[0:10])
#print("beta:")
#for x in D[1]: 
#    print(x[0:10])
#method.init_guess = '1e'
#method.grids.becke_scheme = dft.original_becke
#method.grids.level = 3
method.kernel()
print("Total job time: %10.2f(wall)" %(time()-TimeStart))
