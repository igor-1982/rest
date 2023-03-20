from pyscf import gto, scf, tools
import numpy as np
np.set_printoptions(suppress=True, precision=6, linewidth=200)

mol = gto.M(atom='''H 0.0 0.0 0.0; H 0.0 0.0 2.0''', basis='cc-pvdz').build()
mf = mol.RHF().set(chkfile="stab.chk").density_fit(auxbasis='def2-svp-jkfit').run()
tools.dump_mat.dump_rec(mf.stdout, mf.mo_coeff, ncol=10)
mf.verbose = 9
mf.stability(external=False)

mf2 = mf.to_uhf()
mf2.verbose = 4
mf2.run()
tools.dump_mat.dump_rec(mf2.stdout, mf2.mo_coeff[0], ncol=10)
tools.dump_mat.dump_rec(mf2.stdout, mf2.mo_coeff[1], ncol=10)
mf2.verbose = 9
mf2.stability()

exit()
from pyscf import tdscf
import scipy

from pyscf.soscf import newton_ah
g, hop, hdiag = newton_ah.gen_g_hop_rhf(mf, mf.mo_coeff, mf.mo_occ)
x = np.zeros_like(g)
x[0] = 1.0
print(hop(x))

A,B = tdscf.rhf.get_ab(mf)
nov = A.shape[1]
ApB = A.reshape(nov,nov)+B.reshape(nov,nov)
print(A)
print(B)
print(ApB)
e,v = scipy.linalg.eigh(ApB)
print(e)
#e,v = scipy.linalg.eigh(A.reshape(nov,nov)-B.reshape(nov,nov))
#print(e)

exit()
mf.mo_coeff[:,1] *= -1
mf.stability(external=False)
A,B = tdscf.rhf.get_ab(mf)
nov = A.shape[1]
ApB = A.reshape(nov,nov)+B.reshape(nov,nov)
print(ApB)
e,v = scipy.linalg.eigh(ApB)
print(e)
