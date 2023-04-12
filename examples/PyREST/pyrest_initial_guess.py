#!/usr/bin/env python3

#导入REST的PYTHON模块
import numpy
from time import ctime, time
from pyrest import InputKeywords, GeomCell, Molecule, SCF

TimeStart = time()

#输入分子构型信息
geom = GeomCell()
geom.py_set_unit("Angstrom")
geom.py_set_position([
    "N     -4.15520372      3.58551842      0.00000000",
    "H     -2.22768306      3.58551842      0.00000000",
    "H     -4.79770360      2.06461276     -0.99467369",
    "H     -4.79771235      5.20738069     -0.81980566"])

#设置计算控制信息
ctrl = InputKeywords()
ctrl.print_level = 1
ctrl.py_set_num_threads(1)
# 体系电荷与自旋
ctrl.py_set_charge_spin([0.0,1.0])
# 计算方法: RHF
ctrl.py_set_xc("hf")
ctrl.py_set_spin_polarization(False)
# 基组信息
ctrl.py_set_basis_path("/home/igor/Documents/Package-Pool/rest_workspace/rest/basis-set-pool/cc-pVDZ")
ctrl.py_set_basis_type("spheric")
# 辅助基组信息
ctrl.py_set_auxbasis_path("/home/igor/Documents/Package-Pool/rest_workspace/rest/basis-set-pool/def2-SVP-JKFIT")
ctrl.py_set_auxbasis_type("spheric")
# 设置初始猜测为hcore
ctrl.py_set_initial_guess("hcore")

#创建Molecule和SCF结构体
mol = Molecule()
mol = mol.py_build(ctrl, geom)
scf = SCF(mol)

# 可以获取一些中间数据
# ovlp
(ovlp,size) = scf.py_get_ovlp()
ovlp = numpy.array(ovlp).reshape(size[1],size[0])
# h-core
(hcore,size) = scf.py_get_hcore()
hcore = numpy.array(hcore).reshape(size[1],size[0])
# h-core初猜的Hamiltonian
(init_h,size) = scf.py_get_hamiltonian()
init_h_alpha = numpy.array(init_h[0]).reshape(size[1],size[0])
if scf.mol.spin_channel == 2:
    init_h_beta = numpy.array(init_h[1]).reshape(size[1],size[0])

print("Total job time: %10.2f(wall)" %(time()-TimeStart))
