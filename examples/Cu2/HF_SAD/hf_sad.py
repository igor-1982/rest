#!/usr/bin/env python3                                                                                                                                                                                                                
import pyscf
import numpy, scipy
#from pyscf import gto
#from pyscf import scf, ci,df,lib
#from pyscf import scf
#from pyscf.lib import chkfile
from time import ctime, time

#pyscf.lib.num_threads(6)
TimeStart = time()
mol = pyscf.gto.Mole(
        atom='''
 Cu                 6.64063000    1.27799000    6.15738000
 Cu                 8.11653000    1.27763000    4.11665000
''',
    spin=0, charge=0, basis='def2-svp',verbose=4
).build()
#method = scf.RHF(mol).density_fit(auxbasis="def2-svp-jkfit")

# from init_guess_by_atom(mol)
#atm_scf = scf.atom_hf.get_atm_nrhf(mol)
#e_hf, e, c, occ = atm_scf[0]
#dm = numpy.dot(c*occ, c.conj().T)
#print(dm)



#method = dft.RKS(mol).density_fit(auxbasis="def2-svp-jkfit")
#method.xc = 'b3lypg'
method = pyscf.scf.RHF(mol).density_fit(auxbasis="def2-svp-jkfit")
#method = pyscf.scf.RHF(mol)
D = method.get_init_guess(mol,'atom')
method.init_guess = 'atom'
method.chkfile='init_guess.chk'
pyscf.lib.chkfile.save_mol(mol,method.chkfile)
pyscf.lib.chkfile.dump(method.chkfile,"init_guess",D)
print(D)
#method.small_rho_cutoff = 1.0e-2
print('Default DFT(B3LYPG).  E = %.12f' % method.kernel())
print("Total job time: %10.2f(wall)" %(time()-TimeStart))

## store the grids to an external file
#print(len(method.grids.weights))
#ff = open("grids-dz",'w')
#for i in range(len(method.grids.weights)):
#    x,y,z = method.grids.coords[i]
#    w = method.grids.weights[i]
#    ff.write("%16.8e,%16.8e,%16.8e,%16.8e\n" %(x,y,z,w))
#ff.close()
