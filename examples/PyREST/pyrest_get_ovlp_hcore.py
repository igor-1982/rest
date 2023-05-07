#!/usr/bin/env python3

#导入REST的PYTHON模块
import numpy
from time import ctime, time
from pyrest import InputKeywords, GeomCell, Molecule, SCF
import os
basis_path = os.getenv("REST_HOME")+"/rest/basis-set-pool"

TimeStart = time()

#输入分子构型信息
geom = GeomCell(pos='''
    N  -2.1988391019      1.8973746268      0.0000000000
    H  -1.1788391019      1.8973746268      0.0000000000
    H  -2.5388353987      1.0925460144     -0.5263586446
    H  -2.5388400276      2.7556271745     -0.4338224694
''')

#设置计算控制信息
#设置计算控制信息，缺省使用1）RI-V处理四中心积分；2）DIIS的加速收敛算法；3）SAD初始猜测
ctrl = InputKeywords(
    xc="x3lyp",
    charge=0.0,spin=1.0,
    basis_path = basis_path + "/def2-SVP",
    auxbas_path = basis_path + "/def2-SV(P)-JKFIT",
    print_level = 2,
    num_threads = 1
)
ctrl.py_set_initial_guess("hcore")


# 创建Molecule结构体
mol = Molecule(ctrl, geom)
# 可以获取一些中间数据
# ovlp
(ovlp,size) = mol.py_get_2dmatrix("ovlp")
ovlp = numpy.array(ovlp).reshape(size[1],size[0])
print("ovlp:", ovlp)
# h-core
(hcore,size) = mol.py_get_2dmatrix("hcore")
hcore = numpy.array(hcore).reshape(size[1],size[0])
print("hcore", hcore)

# 获取初始化信息
#scf_data = SCF(mol)
# 执行开展SCF计算，返回SCF结构体，并获取相应的能量和中间结构信息
#scf_data = pyrest.do_scf(mol)


## 可以获取一些中间数据
## ovlp
#(ovlp,size) = scf.py_get_ovlp()
#ovlp = numpy.array(ovlp).reshape(size[1],size[0])
## h-core
#(hcore,size) = scf.py_get_hcore()
#hcore = numpy.array(hcore).reshape(size[1],size[0])
## h-core初猜的Hamiltonian
#(init_h,size) = scf.py_get_hamiltonian()
#init_h_alpha = numpy.array(init_h[0]).reshape(size[1],size[0])
#if scf.mol.spin_channel == 2:
#    init_h_beta = numpy.array(init_h[1]).reshape(size[1],size[0])

print("Total job time: %10.2f(wall)" %(time()-TimeStart))
