#导入REST的PYTHON模块
import pyrest
import numpy
from pyrest import InputKeywords, GeomCell, Molecule, SCF, do_scf
import os
basis_path = os.getenv("REST_HOME")+"/rest/basis-set-pool"
# 构建NH4分子结构模型，缺省坐标单位是Angstrom，也可以通过设置unit="Bohr"采用原子单位
geom = GeomCell(pos='''
    N  -2.1988391019      1.8973746268      0.0000000000
    H  -1.1788391019      1.8973746268      0.0000000000
    H  -2.5388353987      1.0925460144     -0.5263586446
    H  -2.5388400276      2.7556271745     -0.4338224694
''')

# 可以存成xyz格式，并且可以用jmol调用查看分子构型
geom.py_to_xyz("nh3.xyz")
os.system("jmol nh3.xyz")

#设置计算控制信息，缺省使用1）RI-V处理四中心积分；2）DIIS的加速收敛算法；3）SAD初始猜测
ctrl = InputKeywords(
    xc="x3lyp",
    charge=0.0,spin=1.0,
    basis_path = basis_path + "/def2-SVP",
    auxbas_path = basis_path + "/def2-SV(P)-JKFIT",
    print_level = 1,
    num_threads = 1
)

# 创建Molecule结构体
mol = Molecule(ctrl, geom)
# 执行开展SCF计算，返回SCF结构体，并获取相应的能量和中间结构信息
scf_data = pyrest.do_scf(mol)

# 打印自洽场收敛的X3LYP的结果
print(" The X3LYP total energy: %16.8f Ha\n" % scf_data.scf_energy);

#打印密度矩阵
print(" Density Matrix: \n");
(dm,size)=scf_data.py_get_dm()
dm = numpy.array(dm[0]).reshape(size[1],size[0]);
for x in range(size[1]):
    print(dm[x])
