# REST项目介绍和程序安装
  请参见[REST开发组页面](https://gitee.com/restgroup)。**以下为REST程序的具体使用说明**
# For English Users:
  - This manual can be used as a prompt file for state-of-the-art Large Language Models (LLMs), such as DeepSeek and Tongyi. By providing this content to an LLM, you can effectively utilize it as an online support assistant and to generate the input file for computational tasks you need. (Note: ChatGPT has not been tested due to constraints by the US government.)
# 用于生成REST输入卡的系统提示词
- 基于Rust语言的新一代电子结构计算软件REST（Rust-based Electronic Structure Toolkit）由复旦大学化学理论研究中心开发，在徐昕教授的领导下，由张颖教授担任首席开发者完成。
- 根据用户需求，结合知识库和上下文，帮助用户生成REST程序的输入卡。 
- REST输入卡使用TOML格式，包含[ctrl]、[geom]、[hessian]、[thermo]和[geometric_pyo3]等控制区
    - [ctrl]、[geom]为必需区；[hessian]、[thermo]、[geometric_pyo3]等根据任务类型按需开启
    - [ctrl]申明具体计算方法、（辅助）基组、数值方法参数等
    - [geom]提供研究体系的名字、结构以及结构相关的ghost原子、点电荷以及赝势等
	- [geometric_pyo3]设置的参数仅用于`opt_engine=geometric_pyo3`结构优化引擎的控制
- 生成输入卡之后，与程序手册上的关键词进行对比，做如下确认：
  - 输出格式是TOML
  - 输入卡必须包含[ctrl]和[geom]两个区块
  - [ctrl]中的大部分关键词有缺省设置。若用户无具体要求，不必出现在输入卡中
  - 如果用户没有明确要求，设置num_threads为4
  - [ctr]区块中必须声明的关键词：
     1. 计算方法和配置相关：`xc`，`basis_path`，`print_level`，以及`num_threads`等
     2. 体系相关: `spin`, `charge`, `spin_polarization`等
  - 仅当使用了`opt_engine=geometric_pyo3`时，才要申明[geometric_pyo3]区
  - 当调用geometric_pyo3引擎做缺省的最稳结构优化时，不需要申请[geometric_pyo3]
  - 调用的方法的关键词是否使用"xc"，不能无中生有地用其它的关键词，比如"method"等
  - D3BJ、D3以及D4是经验色散校正，需要用`empirical_dispersion`申明。比如"X3LYP-D3BJ"方法需要拆分成"xc=x3lyp"和"empirical_dispersion=d3bj"
  - 关键词`spin`和`charge`是在`[ctrl]`区，而不是在`[geom]`区
  - 分子结构的关键词是`position`，不能无中生有地用其它的关键词——比如"coord"和"molecule"等
  - 分子结构`position`的申明使用String，比如
`
"""
  H 0.0 0.0 0.0
  H 0.75 0.0 0.0
"""
`
  - 输入卡中不采用'''符号
  - 反复迭代比较，直至输入卡一次性全部满足上述要求
  - 输出REST程序的输入卡，使用String的格式，包含换行符号'\n'，并且对'"'符号进行'\"'转译

# Detailed descrption of `[ctrl]` block in the control file

## 系统设置相关关键词（Keyword）
- `num_threads`: 取值i32类型。**每个 MPI 进程**可展开的线程数目（不是整个节点的线程数），缺省为1。
  每个 rank 是独立进程，各自按此值建立 rayon 线程池并同步设置 OpenMP 线程数，所以节点实际使用的线程数是 `num_threads` 乘以落在该节点的 rank 数。
  MPI 与线程混跑时建议设为「每节点核数除以每节点 rank 数」。缺省值 1 表示每个进程内的并行区串行执行。
  程序启动时会检查 `num_threads` 乘 MPI 进程数是否超过节点的逻辑 CPU 数，超过时打印一条 `[WARN] Thread oversubscription` 告警并给出建议值。该检查按硬件列表统计节点 CPU，不受 MPI 进程绑定影响，可用环境变量 `REST_SKIP_THREAD_WARN=1` 关闭。
- `print_level`: 取值i32类型。程序输出信息量，数字越大，输出信息量越多。缺省为1。0表示完全无输出
- `max_memory`: 取值f64类型。**每个 MPI 进程**可使用的最大内存，单位为MB。缺省不设置，此时批次决策按当前节点的可用内存估算。
  预算的消费方式是「`max_memory` 减去本进程常驻内存」，所以 MPI 下每个 rank 各自套用，节点的实际额度是 `max_memory` 乘以落在该节点的 rank 数。
  MPI 与线程混跑时建议设为「每节点内存除以每节点 rank 数」并留一到两成余量。未显式设置时各 rank 都会把节点可用内存当成自己的预算，多进程共享节点时应当显式设置。
  该设置只在算法感知内存并按批计算的场合生效，也不是硬上限。它用于批次决策与力阶段的内存监控，峰值 RSS 超过限制时按 `abort_on_mem_exceed` 决定是否中止。
- `abort_on_mem_exceed`: 取值布尔类型。是否在内存超出限制时终止计算。缺省为true。**设置false存在风险，不建议一般用户设置该选项**。

## 具体计算任务相关关键词（Keyword）
- `job_type`: 取值String类型。设置计算任务类型。目前可以进行的计算任务为:
    1. `energy`: 单点能量计算（缺省）。等价设置有：`single point`，`single_point`等
    2. `opt`: 基于数值力的构型优化。等价设置有：`geometry optimization`, `relax`, `geom_opt`等
	3. `force`: 计算当前结构下的受力。等价设置有：`gradient`
	4. `numerical dipole`: 计算数值偶极。等价设置有：`numdipole`
	5. `md`: 内置分子动力学（详见 [REST 文档 · MD](https://rest-doc.readthedocs.io/zh_CN/user/md.html)）。纯 MM 使用 `pure_mm = true`
- `auxbasis_response`：开启辅助基导数。缺省为true
- `opt_engine`: 取值String类型。构型优化引擎。可选项有：`LBFGS`、`geometric-pyo3`（缺省）
    - **注意**：固定原子功能（在 `position` 中用 `0`/`1` 标记）当前仅支持 `geometric-pyo3` 引擎，`LBFGS` 引擎暂不支持约束优化。
- `numerical_force`: 取值布尔类型。是否计算数值力。缺省为false
- `nforce_displacement`:　取值f64类型。数值力计算中的结构位移值，缺省是0.0013 Bohr
- `ndipole_displacement`:　取值f64类型。数值Dipole计算中的外电场位移值，缺省是3.0E-4 Bohr
- `check_stab`: 取值String类型。SCF收敛后进行波函数稳定性分析，与 `[tddft] stability` 等价。详见 TD-DFT计算相关设置的稳定性分析一节。

## 计算体系相关关键词（Keyword）
- `charge`：取值f64类型。体系的总电荷数
- `spin`: 取值i32类型。体系的自旋多重度。假设体系总自旋量子数为s，则取值2s+1
- `spin_polarization`: 取值布尔类型。是否开放自旋极化。
    - 当spin取值1时，缺省为false（RHF计算）
    - 当spin取值大于1时，缺省为true（UHF计算）
    - 若要进行ROHF计算，需要在spin取值大于1时将spin_polarization设为false。目前仅支持高自旋（即所有未成对电子自旋平行）的情况。
- `outputs`: 取值Vec\<String\>。用于计算结束后输出结果。可输出的信息包括：
    - `dipole`    偶极
    - `fchk`　    Gaussian程序的fchk文件
        - 双杂化 (PT2 族) 的解析力计算 (`job_type = "force"`) 额外在 fchk 中写入弛豫总密度 `Total MP2 Density` 段；开壳层 (UHF 参考) 另行写入 `Spin MP2 Density` 段 (alpha − beta)。
        - fchk 文件中 MO 系数与各密度段的写出行为由 `[output]` 区块控制（如 `fchk_dm`、`fchk_writer`），详见 `[output]` 区块说明。缺省设置下无需关心。
    - `cube_orb`  格点化的轨道文件信息 
    - `molden`　　结果输出为molden程序的格式
    - `geometry`  输出分子结构文件
    - `force`     输出分子受力信息
    - `force_for_ghost_point_charges`  在[geom]部分输入ghost point charges时输出原子区域对这些ghost点电荷的作用力
- cube文件相关关键词 `cube_orb_setting`、`cube_orb_indices`、`cube_orb_type` 已移至 `[output]` 区块。`[ctrl]` 中的旧位置仍然可用（向后兼容，但已弃用），详见 `[output]` 区块说明。

## 计算方法相关关键词（Keyword）
- `xc_parser`：指定用于解析`xc`关键词的方式，选项包括
  - `legacy` 缺省值，此时用户通过 `xc` 关键词设置方法名称调用 REST 程序内置支持的电子结构方法，具体可参见 `xc` 关键词说明
  - `parse_xc` 支持用户更自由地调用 libxc 所支持的密度泛函近似方法以及 REST 支持的第五阶密度泛函方法，覆盖 `legacy` 选项包含的情形的同时，进一步支持更复杂的、用户自定义的泛函输入场景
- `xc`：取值String类型。调用的电子结构计算方法。目前REST支持的常用电子结构方法有
    1. 波函数方法：HF、MP2
    2. 局域密度泛函近似：LDA
    3. 广义梯度泛函近似：BLYP、PBE、xPBE、XLYP
    4. 动能密度泛函近似：SCAN、M06-L、MN15-L、TPSS
    5. 杂化泛函近似：B3LYP、X3LYP、PBE0、M05、M05-2X、M06、M06-2X、SCAN0、MN15
    6. 第五阶泛函近似：XYG3、XYGJOS、XYG7、xDH-PBE0、sBGE2、ZRPS、scsRPA、R-xDH7、RPA@PBE、RPA@B3LYP
    - HF、LDA、BLYP、PBE、B3LYP、PBE0是自洽场计算方法，若用户未申明具体基组，则使用def2-TZVPP基组 (`basis_path = {basis_set_pool}/def2-TZVPP`)
    - MP2、XYG3、XYGJOS、XYG7、xDH-PBE0、sBGE2、ZRPS、scsRPA、R-xDH7、RPA@PBE、RPA@B3LYP为后自洽场计算方法。若用户未申明具体基组，则使用def2-QZVPP基组 (`basis_path = {basis_set_pool}/def2-QZVPP`)
    - RPA@PBE、RPA@B3LYP表示后自洽场RPA计算使用PBE、B3LYP方法的轨道
    - 当 `xc_parser` 设置为`xc_parser=parse_xc`时，可通过 `xc` 关键词自定义密度泛函近似方法，具体书写规则请参见[REST程序用户手册](https://rest-doc.readthedocs.io/zh_CN/contributor/dft/parse_xc.html)
- `pure_mm`：取值布尔类型。设为 `true` 时 REST 完全不做电子结构计算，整个体系交给内嵌 OpenMM 处理（纯 MM），
  配合 `job_type = "md"/"sp"/"force"/"singlepoint"` 与 `[md]` 中的 `mm_file`/`mm_system_xml`/`mm_cutoff`。
  此时无需 `xc`、`basis_path` 与 `[geom]`。
- `empirical_dispersion`:　取值为String。针对低级别密度泛函方法（包括LDA、BLYP、PBE、B3LYP、PBE0等）的经验色散校正方法。目前支持D3, D3BJ和D4。对于XYG3型双杂化泛函比如XYG3、XYG7、XYGJOS、scsRPA、R-xDH7、RPA等不需要经验色散校正
- `post_ai_correction`：取值String。AI辅助的校正方法。目前仅支持SCC15，并只能和R-xDH7重整化双杂化泛函方法相匹配。 
- `post_xc`：取值Vec\<String\>。采用自洽收敛的轨道和密度，进行不同的交换－关联泛函(xc)的计算。允许的方法包括REST支持的"xc"方法
- `post_correlation`：取值Vec\<String\>。采用自洽收敛的轨道和密度，进行后自洽场高等级相关能方法计算。允许的方法包括PT2、sBGE2、RPA、scsRPA等

## DFT积分格点相关关键词（Keyword）
- `grid_generation_level`: 取值usize。格点精度等级，数值越大越精确。缺省为3
- `pruning`: 取值String。DFT方法或sap初猜所选用格点筛选。目前，REST支持nwchem，sg1以及none。其中none为不筛选。缺省为nwchem
- `radial_grid_method`: 取值String。径向格点的生成方法。目前REST支持treutler，gc2nd， delley, becke, mura_knowles及lmg。缺省为treutler
- `radii_adjust`: 取值String。Becke划分中原子的半径调整方案。`becke` 用Bragg半径本身（Becke 1988原文形式），`treutler` 用Bragg半径的平方根（Treutler-Ahlrichs 1995形式）。两者都满足权重和为1，差别随格点加密衰减。缺省为treutler（历史缺省 becke 可用输入卡显式指定）

### VXC 格点积分优化相关关键词（Keyword）
- `vxc_screen_threshold`: 取值f64。格点密度筛选阈值：批次内 `Σ_s|ρ_s|` 低于此值的格点跳过 XC 计算。缺省为 0.0（完全关闭：不构建掩码、不额外扫描格点）。设为正值时，还需该批次被筛掉的格点超过 25%（`active < 3/4·ngrids`）才真正走筛选路径；实测 ≤1.0e-4 从不触发，而能触发的最小档 1.0e-3 因当前实现要做全网格扫描与 AO 列拷贝反而慢约 4 倍、能量偏 44 μHa，故不建议用于生产计算（详见 pyfock_study/guess_compare/PLAN_vxc_screen_opt_20261007.md）。

- `grid_prune_threshold`: 取值f64。**初猜密度的一次性格点裁剪**阈值：在自洽场迭代开始之前，把 `Σ_s|ρ_s|·|w_p|` 低于此值的格点从格点集合中删除（`ρ` 为初猜密度：SAD/vsap/`guessfile`/`hcore`；`w_p` 为该点的积分权重），随后重建 non0tab 压缩 AO/AOP 表；此后所有 SCF 迭代、能量与梯度都在裁剪后的网格上计算。缺省为 0.0（不裁剪）。判据含权重是刻意的：远端格点 ρ 小而 w 大，只用裸 `|ρ|` 会误删这些点；被删比例 < 2% 时程序放弃裁剪（重建不划算）；`use_isdf = true` 时本项被忽略（ISDF 表建在未裁剪网格上）。与 `vxc_screen_threshold` 的区别：后者是**每次迭代**在 Vxc 内部跳过格点，本项是**一次性**从格点集合删除。做法借鉴 PyFock 的一次性 `|ρ·w| ≥ 1e-11` 剪枝。实测：(H2O)10/PBE0/def2-SVP 取 1.0e-11 删 20.8% 格点、能量移动 +4.3e-9 Ha；(H2O)30/B3LYP/def2-SVP 取 1.0e-12 删 21.5%、+1.9e-9 Ha；裁剪结果与线程数无关（1/4 线程删除数相同）。
- `ao_cutoff`: 取值f64。AO 格点值截断阈值，启用 non0tab 稀疏存储。AO 值低于此阈值的基函数-格点对被当作零处理。缺省为 1.0e-9：格点制表按批次只存超过阈值的 AO/AOP 元素，稠密 AO/AOP 不分配。若某体系在该阈值下 AO 存留率高于 90%，程序自动放弃压缩、回退稠密存储。设为 0.0 可强制关闭压缩。(H2O)30 / B3LYP / def2-SVP 实测：1.0e-9 相对 1.0e-12 峰值内存少约 10%、墙钟少约 12%，能量差约 1 nHa。
- `non0tab_blksize`: 取值usize。non0tab 压缩存储中每个格点批次的大小。设为 0（缺省）时程序根据基组大小自动选择 [32, 256]。设为具体值则固定该批次大小。
- `drop_dense_ao`: 取值布尔类型。在稠密 AO/AOP 已经分配、non0tab 压缩存储生成后，是否释放稠密 AO/AOP 矩阵以节省内存。缺省为 false。仅当 `ao_cutoff = 0.0`（稠密路径：先分配稠密 AO/AOP，再额外构建压缩存储）时该开关才起作用；`ao_cutoff > 0` 的稀疏路径只分配压缩存储、从不分配稠密 AO/AOP，此时本项为空操作。

## 基组相关关键词（Keyword）
- `eri_type`: 取值String类型。设置四中心积分计算方法。选项：
    - `analytic`: 解析计算（仅限开发测试，不推荐用于实际计算）
	- `ri-v`: 使用密度拟合方法近似计算。(默认值)
	> 注意：REST中的analytic算法并未被充分优化，仅供程序开发测评使用，不建议在实际计算中使用
- `basis_type`: 取值String类型。设置基函数类型。选项：
    - `Spheric`: 球谐型基函数（默认值）
	- `Cartesian`：笛卡尔型基函数
- `basis_path`: 取值String类型，必需参数，无默认值。`basis_path = "chkfile"` 时将从 checkpoint 文件读取基组。其他情况下，该关键词指定基组文件路径，此路径包含各元素的基组JSON文件（如`C.json`、`O.json`）
    - 格式示例：  
      ```bash
      # 完整路径格式（明确指定）
      basis_path = /opt/rest_workspace/rest/basis-set-pool/cc-pVTZ
      
      # 简写格式（程序自动搜索）
      basis_path = cc-pVTZ
      ```
	- **路径优先级**： 
	  当使用简写格式时，程序按以下顺序搜索 `cc-pVTZ` 文件夹：
	  1. **当前工作目录**（`$PWD/cc-pVTZ`）
      2. 环境变量 `REST_BASIS_DIR`指定目录（`$REST_BASIS_DIR/cc-pVTZ`）
      3. REST docker 默认路径：`/opt/rest_workspace/rest/basis-set-pool/cc-pVTZ`
	  4. REST conda 默认路径： `$CONDA_PREFIX/share/rest/basis-set-pool/cc-pVTZ`
      5. `$REST_HOME/rest/basis-set-pool/cc-pVTZ`
      6. 自动从BSE库在线获取（若本地未找到），并保存在当前工作目录
    - 以上 2-5 情形不区分基组名称的大小写，对于以前使用的 `def2-sv(p)-jkfit`，现在会视为 `def2-universal-jkfit` 的 alias，仍然可以使用。若使用完整路径格式，则区分大小写，且不支持 alias。 
- `auxbas_path`: 取值String类型，指定辅助基组路径，仅当`eri_type=ri-v`时需要设置  
    缺省值：def2-universal-jkfit。（该基组较大，建议根据体系换用匹配的辅助基组）
	设置方式与`basis_path`相同，程序搜索路径一致。

### 自定义基组使用方法
REST 支持用户自定义或混合基组：
1. 创建基组文件夹  
    在任意位置（如当前工作目录或基组库目录）创建文件夹：
	```bash
	mkdir my_custom_basis
	```
2. 准备基组文件  
    在文件夹内放入各元素的基组 JSON 文件：
	```bash
	# 文件夹结构示例
    my_custom_basis/
    ├── C.json
    ├── O.json
    ├── H.json
    └── N.json
	```
3. 在输入文件中引用  
    ```bash
	# 方式1：使用相对路径（从当前目录查找）
    basis_path = ./my_custom_basis
    
    # 方式2：使用文件夹名（按搜索优先级查找）
    basis_path = my_custom_basis
    
    # 方式3：使用绝对路径（直接定位）
    basis_path = /home/user/my_custom_basis
	```
### 配置示例
```plaintext
# 示例1：使用标准基组和缺省辅助基组（简写格式）
basis_type = Spheric
basis_path = cc-pvtz                    # 程序自动搜索 cc-pvtz 文件夹

# 示例2：使用标准基组，同时申明辅助基组（简写格式）
basis_type = Spheric
basis_path = cc-pvtz                    # 程序自动搜索 cc-pvtz 文件夹
auxbas_path = def2-universal-jkfit      # 程序自动搜索同名辅助基组文件夹

# 示例3：使用自定义基组（相对路径）
basis_type = Cartesian
basis_path = ./my_project_basis         # 使用当前目录下的自定义基组
auxbas_path = ./my_aux_basis            # 使用自定义辅助基组

# 示例4：使用完整路径
basis_type = Spheric
basis_path = /shared/basis/def2-TZVP    # 明确指定完整路径
auxbas_path = def2-universal-jkfit      # 简写格式，自动搜索

# 示例5：从 checkpoint 文件读取初猜
basis_path = "chkfile"
auxbas_path = "def2-universal-jkfit"
guessfile = "my_checkpoint.rchk"
```

## RI 密度拟合的分解策略（Keyword）

密度拟合（`eri_type = "ri-v"`）先把 2c-2e 辅助基库仑度量化成 $J = L L^{\mathsf T}$（或 $J^{-1/2}$），再把三中心张量变换为 $Y_{\mu\nu,P}$。`[ctrl.j2c_decomp]` 表控制这一步的分解方式：

````toml
[ctrl.j2c_decomp]
     policy = "cd"            # "cd"（Cholesky）或 "eig"（特征值）；缺省 "eig"
     threshold = 1.0e-13      # 缺省 1.0e-13
     uplo = "upper"           # 仅 Cholesky 使用；开发者选项，缺省 "upper"
     distributed = "auto"     # 仅 Cholesky + MPI 使用，缺省 "auto"
````

- `policy`：取值String。
    - `"cd"`（别名 `cholesky`）：Cholesky 分解，得到三角因子 $L$，$J^{-1/2}$ 通过三角求解（TRSM）作用于 $L$ 隐式获得，不显式构造逆。
    - `"eig"`（别名 `eigen`/`eig`/`eigenvalue`）：对称特征分解，显式构造 $J^{-1/2} = V \Lambda^{-1/2} V^{\mathsf T}$。形式上更正交，但代价明显更高。**缺省为 `"eig"`**（保持历史缺省与既有回归参考值；一般计算建议显式写 `policy = "cd"`）。
- `threshold`：取值f64。缺省 1.0e-13。`policy = "cd"` 时，若 Cholesky 因子对角元小于该阈值，先将矩阵补正到满足该阈值的正定程度再重新分解；`policy = "eig"` 时，只保留大于该阈值的特征值来构造 $J^{-1/2}$。
- `uplo`：取值String。`"upper"`（缺省）或 `"lower"`，仅对 Cholesky 有效。列主序下通常使用 upper；lower 仅供调试与测试。
- `distributed`：取值String。`"auto"`（缺省）/`"on"`/`"off"`。仅在（1）编译了 `scalapack` 与 `mpi` 特性、（2）MPI 进程数 ≥ 2、（3）`policy = "cd"` 三条同时满足时生效，其余组合一律回退到 rank-0 串行路径（特征分解没有分布式实现）。`auto` 仅在辅助基函数数 $n_{aux} \ge 8192$ 且进程数 ≥ 32 时才启用分布式 Cholesky 与三角求解；小体系测试可用 `on` 强制打开。

**性能与精度**：(H2O)30 / B3LYP / def2-SVP（4 线程）实测，仅将 `policy` 由 `eig` 改为 `cd`：2c-2e 分解 10.48 s → **0.61 s**，度量变换 145.1 s → **55.2 s**，度量相关（CInt）相 170.3 s → **70.7 s（−58.5%）**，端到端墙钟 8:43 → **6:43（−23%）**；峰值内存不变（13.1 GB），能量差 **+4.2e-9 Ha（4.2 nHa）**。因此对生产计算**推荐显式指定 `policy = "cd"`**。

**与其他关键词的关系**：`algorithm_jk`/`algorithm_j`/`algorithm_k` 决定 $Y_{\mu\nu,P}$ 是常驻内存（`ri-incore`）还是即算即用（`ri-direct`），`j2c_decomp` 决定 $Y$ 的构造方式，两者正交、可自由组合。`ri-incore` 需要约 $0.5\,n_{ao}^2 n_{aux}$ 个双精度数的存储，通常是内存峰值的主要来源。
## 自洽场计算相关关键词（Keyword）
- `max_scf_cycle`: 取值i32。自洽场运算的最大迭代循环数。缺省为100
- `noiter`: 取值布尔类型。是否跳过自洽场运算。缺省为false
- `scf_acc_rho`: 取值f64。自洽场运算密度矩阵的收敛标准。缺省为1.0e-5
- `scf_acc_eev`: 取值f64。自洽场运算能量差平方和的收敛标准。缺省为1.0e-3
- `scf_acc_etot`: 取值f64。自洽场运算总能量的收敛标准。缺省为1.0e-8
- `algorithm_jk`: 设置 Fock 矩阵计算中 J (Coulomb) 和 K (Exchange) 两部分的算法：
    - `ri-direct`: 强制使用 direct RI 算法。对于 RI-K 部分，取决于内存大小，可能会使用 semi-direct 算法 (储存相对较小的 $O(N^3)$ 的 $g_{\mu i, P}$)。
    - `ri-incore`: 强制使用 incore RI 算法 (储存完整的 Cholesky decomposed 3c-2e ERI $Y_{\mu \nu, P}$)。该算法对内存需求较大，但计算速度更快。
    - `ri`: 自动选择 incore 或 direct RI 算法，取决于自洽场计算前内存大小；在内存空间较大时选择更快的 incore 方法，内存空间较小时选择 ri-direct 方法。
    - `default`: 目前同 `ri`。
- `algorithm_j`: 设置 Fock 矩阵计算中 J (Coulomb) 部分的算法；该关键词是高级选项，一般用户建议使用`algorithm_jk`关键词进行整体设置。
    - 下述选项同 `algorithm_jk`：`ri-direct`, `ri-incore`, `ri`, `default`。
    - `ri-schwartz`: 使用 Schwartz screening 的 direct RI-J 算法。3c-2e ERI 即算即用（同 `ri-direct`），但利用了稀疏性并降低内存开销与提升内存局域性。该选项仅对 J 部分有效，即只能通过 `algorithm_j` 指定；其行为由 `[ctrl.ri_jk]` 表控制（见下）。该路径暂不支持 MPI 并行。
- `algorithm_k`: 设置 Fock 矩阵计算中 K (Exchange) 部分的算法；该关键词是高级选项，一般用户建议使用`algorithm_jk`关键词进行整体设置。
    - 下述选项同 `algorithm_jk`：`ri-direct`, `ri-incore`, `ri`, `default`。
- `ri_jk`: RI-J/RI-K 算法的参数，以 `[ctrl.ri_jk]` 表形式给出。`schwartz_threshold` 与 `schwartz_overlap_tol2` 控制 `ri-schwartz` RI-J 算法的筛选行为，`pair_screen_threshold` 控制 incore RI-J/RI-K（`ri-incore`）的 AO 对筛选。
    - `schwartz_threshold`: 取值f64类型。Schwartz 筛选的积分忽略阈值（单位 Hartree）：仅当 (壳对, 辅助壳) 块的 Coulomb 贡献上界达到该阈值时才纳入计算。缺省为 1.0e-12。
    - `schwartz_overlap_tol2`: 取值f64类型。Schwarz 上界构建前静态重叠预筛选的阈值：最弥散基元的重叠小于该值的壳对不参与上界构建。缺省为 1.0e-24。
    - `pair_screen_threshold`: 取值f64类型。incore RI-J/RI-K 的 AO 对（basis pair）筛选的相对阈值，同时作用于**构建期**与**缩并期**两级。
    - 构建期（存储级）：三中心张量 `rimatr` 建成后，每个 AO 对的行上界 $q_p = \max_P |Y_{p,P}|$ 低于 $\text{threshold} \times \max_p q_p$ 的行被丢掉，张量原地压紧，行布局记入 `PairMap`（存储行到 $(\mu,\nu)$、行界、全量对号到存储行）。`basbas2baspar` 与 `baspar2basbas` 保持全量语义，J 的输出仍散射回全量 packed 上三角。MPI 下静态行界先做一次 `max` 归约再加广播，各 rank 保留同一批行。
    - 缩并期：在保留行里按本轮密度/轨道权重再筛。J 的密度缩并用 $|w_p| q_p$（$w_p$ 为密度权重）与本次迭代最大的对界值比较，K 用 $q_p \max(W_\mu, W_\nu)$（$W_\mu = \max_i |C_{\mu i}|$）比较，低于阈值倍数者被丢弃。K 保留的行张成“活跃 AO 子集”，每个辅助基只在该子集上做缩并与 `dsyrk`。
    - 缺省为 0.0，即不剪枝也不筛选，数值与未筛选内核一致。构建期剪枝改变了存储的 `rimatr` 的行空间，因此每个消费者都要按 `PairMap` 取行，未接线的路径一律显式报错退出，不会静默算错。
    - 已接线：SCF 的 J/K 缩并（含 RSH 短程张量与 `use_dm_only` 的密度版 K），以及后自洽场相关能的 AO→MO 变换（`ri3mo` 与 streaming PT2 驱动，覆盖 PT2/SBGE2/SCS-RPA/RPA 以及 `xc = "mp2"` 这类 PT2 family 单点）。
    - 仍被拒绝：解析导数与 Hessian、SCF/单点之外的 job_type、TDDFT 与响应、GW/BSE、数值力（`outputs = ["num_force"]`），以及 `ri_pt2` 的 `new_driver = true` 与 `engine = "torch"` 两个引擎。

    - **推荐用法（杂化泛函的能量与几何优化）**：显式设 `pair_screen_threshold = 1.0e-12`。构建期会丢掉大多数可忽略的 AO 对行，K 的缩并随之变短，内存与时间都明显下降，而能量变化只有 1e-8 Ha 量级。

    - **构建期预筛**：在**任何三中心积分之前**，用壳对 Schwarz 界 `sqrt(max_{μ∈i,ν∈j} |(μν|μν)|)` 展开到 AO 对，挑出一个**保守超集**（阈值取 `pair_screen_threshold / 10` 的安全系数），随后只在这个更小的行空间上做度量变换与剪枝。实测 (H2O)30 / B3LYP / def2-SVP 下保留 43.7% 的行，`rimatr` 构建相由 57.2 s 降到 30.7 s，而**最终保留行集与 `PairMap` 与预筛前完全一致**（剪枝仍照旧在压紧张量上再跑一次）。**仅在串行路径启用**：MPI 下辅助基按列分布在各 rank 上，预筛尚未接入，行为与改造前相同。

      ````toml`
      [ctrl]
           ao_cutoff =                 1.0e-9      # 缺省值，可省略
           ...
      [ctrl.ri_jk]
           pair_screen_threshold =     1.0e-12     # 杂化泛函能量与几何优化推荐
      `````

      (H2O)30 / B3LYP / def2-SVP 实测（ao_cutoff = 1.0e-9，4 线程）：不筛选 12.75 GB / 897 s，1.0e-12 为 **8.78 GB / 624 s**（能量差 0.017 μHa），1.0e-10 为 8.38 GB / 606 s（差 2.1 μHa），1.0e-9 为 8.16 GB / 593 s（差 12.8 μHa）。收益在 1.0e-12 已基本拿满，故推荐 1.0e-12。
- `use_dm_only`: 取值布尔类型。控制 VK (Exchange) 和 VXC (XC Potential) 矩阵的构建方式。缺省为 false。
    - `false`（缺省）：使用分子轨道系数构造（occ-RI-K 算法），效率更高，推荐用于大多数体系。
    - `true`：直接使用密度矩阵构造。当轨道占据数非整数（如 dSCF 激发态）时可能需要设为 true。

### 自洽场初猜相关关键词
- `initial_guess`: 取值String。分子体系进行自洽场运算所用的初始猜测方法。目前REST支持:
    1. 从 checkpoint 文件读取初猜。不需要设定 `initial_guess`，存在 `guessfile` 或 `chkfile` 时会自动读取初猜。
	2. `sad` : 对体系各原子进行自洽场计算得到自洽的密度矩阵后，将多个密度矩阵按顺序置于对角位置后得到初始的密度矩阵进行自洽场运算。缺省为sad
    3. `vsap`: Superposition of Atomic Potentials的初始猜测方法。采用半经验方法对体系势能项进行估计，与libcint生成的动能项进行加和后得到初始的Fock矩阵
	4. `hcore`: Hcore则对应单电子近似初猜，直接将由libcint生成的hcore矩阵作为初始猜测的fock矩阵进行计算
- `guess_mix`: 取值布尔类型，是否采用混合HOMO和LUMO的方法获得对称性破缺初猜。由此可以破坏体系的空间对称性和自旋对称性，有助于得到对称破缺单重态UHF波函数。缺省为false。
- `guess_mix_theta_deg`: 取值`[f64;2]`或f64，分别设置两个自旋通道的混合角度（单位：度）。
    - 设为0.0，则表示完全不混合
    - 在0.0-90.0范围内，角度越大，表示破坏原始初猜效果越显著。一般建议取值0.0-45.0。缺省为[15.0, 15.0]
- `guessfile`: 取值String。给定初猜的 checkpoint 文件。缺省为none。不会在计算结束后被覆盖。guessfile/chkfile 中的基组与当前计算指定的基组不一致时，会尝试进行基组投影。
- `chkfile`: 取值String。保存计算结果的 checkpoint 文件。缺省为none。如该文件在计算开始前已存在，则会尝试从中读取初猜，但在计算结束后会覆盖该文件。所以不推荐采用 `chkfile` 提供初猜，而是建议采用 `guessfile`。
对于 `guessfile` 和 `chkfile`，推荐以下两种使用方式（文件后缀没有要求，可以是任意的或没有）：
（1） 不读取初猜，只保存计算结果
```
chkfile = "mychk.rchk"
```
（2） 读取初猜，并保存计算结果
```
guessfile = "myoldchk.rchk"
chkfile = "mychk.rchk"
```

### 收敛算法相关关键词
- `mixer`：取值String。辅助自洽场收敛的方法。目前REST支持 `"direct"`，`"diis"`，`"linear"`，`"ediis"`，`"ediis+diis"`及`"adiis+diis"`，程序缺省采用`"diis"`方法。
    - `"direct"` 不使用额外辅助收敛方法（naive SCF）
    - `"linear"` 线性辅助收敛方法，
    - `"diis"` direct inversion in the iterative subspace，是最常用的 SCF 加速收敛方法。当前DIIS实现已内置**正交基误差矢量**（通过 S⁻¹ᐟ² 变换改善条件数）和 **SVD 伪逆求解器**（替代直接矩阵求逆），对过渡金属等近简并体系有更好的数值稳定性。
    - `"ediis"`: Energy-DIIS（Kudin-Scuseria-Cancès, JCP 2002, 116, 8255）。通过最小化历史密度的能量二次规划模型 `f^EDIIS(c) = ΣcᵢEᵢ − ½ΣcᵢcⱼBᵢⱼ`（`Bᵢⱼ = Tr[(Dᵢ−Dⱼ)(Fᵢ−Fⱼ)]`），求得最优凸组合系数后对 Fock 矩阵做线性组合 `F̃ = ΣcᵢFᵢ`，再对角化得到下一步密度。对 HF 精确成立（Fock 对密度仿射），对 DFT 为近似（论文指出 V_xc 非线性"可忽略"）。
    - `"ediis+diis"`：混合模式。早期使用 EDIIS，当 DIIS 误差范数降至 1e-3 时自动切换至 DIIS：
      ```
      DIIS误差范数 > 1e-3  →  EDIIS  # 保守的稳速下降
      DIIS误差范数 ≤ 1e-3  →  DIIS   # 快速的末段冲刺
      ```
      在常规体系（C₂/HF、H₂O/PBE、Fe²⁺/UHF、N₂/PBE 等）中，`ediis+diis` 的迭代数与纯 `diis` 一致，同时具有 EDIIS 的早期能量稳定性。
    - `"adiis+diis"`：ADIIS-DIIS 混合（Hu-Yang, JCP 2010, 132, 054109）。与 `ediis+diis` 类似，但使用以最新历史点为参考的惩罚矩阵。
    - 各 mixer 策略对比：
      | 策略 | 场景 | 收敛速度 | 备注 |
      |------|------|---------|------|
      | direct/linear | 非常简单的闭壳层小分子 | 慢 | 最朴素的方法 |
      | diis | 常规体系（缺省） | 快 | 内置振荡检测回退机制，实践中最稳定 |
      | ediis | HF 近简并体系 | 较慢 | 理论上有能量单调性（仅 HF），但 DFT 下为近似 |
      | **ediis+diis** | 常规体系 | **快** | 与 diis 等速，兼具 EDIIS 早期稳定性 |
      | adiis+diis | 与 ediis+diis 类似 | 快 | B 矩阵保证 PSD，数值更鲁棒 |
    - 已知局限：纯 `ediis` 在极端拉伸的 DFT 体系（如 N₂ r=3Å PBE）中可能不收敛——此时 Fock 混合近似 `ΣcᵢFᵢ ≈ F(D̃)` 因 V_xc 强非线性而失效，建议使用 `ediis+diis`（自动切换到 DIIS）。
- `mix_param`: 取值f64。DIIS方法或linear方法的混合系数。缺省为0.6
- `start_diis_cycle`: 取值i32。开始使用diis（或ediis）加速收敛方法的循环数。缺省为1（第 1 轮即启用 DIIS；DIIS 历史不足时按线性混合回退）
- `num_max_diis`: 取值i32。DIIS/EDIIS子空间大小（存储的历史Fock/密度矩阵数量）。缺省为8
- `diis_pinv_threshold`: 取值f64。DIIS 误差矩阵求伪逆时的奇异值截断阈值。缺省为1.0e-12
- `diis_orthogonalize_error`: 取值bool。是否对 DIIS 误差矢量做正交化（经 S^-1/2 变换改善条件数）。缺省为true；设为 false 可复现历史行为
- `level_shift`: 取值f64。对于发生近简并振荡不收敛的情况，可以采用level_shift的方式人为破坏简并，加速收敛。单位为hartree，缺省值为None（不开启）。
    - 注意：当体系本身可以用纯 DIIS 正常收敛时，开启 level_shift 反而会减速收敛（DIIS 子空间已充分条件良好）。仅当 DIIS 失效（近简并振荡）时才需要开启此选项。
- `ediis_penalty`：取值f64。EDIIS 惩罚参数 η（Kudin-Scuseria, JCP 2002, 116, 8255）。η 越大，外推越保守（偏离历史数据点的惩罚越大）。缺省为 0.5。仅当 `mixer = "ediis"`、`"ediis+diis"` 或 `"adiis+diis"` 时生效。
- `ediis_switch_gap`：取值f64。预留关键词。当前实现中 EDIIS→DIIS 的切换由 DIIS 误差范数自动确定（阈值 1e-3），本关键词暂不生效。缺省为 0.1。
- `adiis_penalty`：取值f64。ADIIS 惩罚参数 μ（Hu-Yang, JCP 2010, 132, 054109）。缺省为 0.5。仅当 `mixer = "adiis+diis"` 时生效。
- `start_check_oscillation`: 取值i32。开始检查并自洽场计算不收敛发生振荡的循环数。当监控到自洽场发生振荡，SCF能量上升的情况，开启一次线性混合方案（linear）。缺省为20
- `smear`：取值String。开启分数轨道占据（smearing）加速自洽场收敛。适用于能隙较小或金属性体系。目前支持：
    - `"fermi"`：Fermi-Dirac 展宽
    - `"gaussian"`：Gaussian 展宽  
    缺省为 None（不开启展宽）
- `smear_sigma`：取值f64。展宽参数 σ，单位为 Hartree。σ 越大，占据数分数化程度越高，收敛越快但引入的熵误差越大。对于小分子体系典型取值范围 0.01-0.05 Ha。缺省为 None（不开启展宽）
- `smear_anneal`：取值bool。是否启用在 SCF 迭代中对 σ 做**指数退火**。缺省为 false。
    - 设置为 `true` 后，σ 从 `smear_sigma` 指数衰减至下限值。退火窗口为 `max_scf_cycle` 的前半段，后半段保持恒定的下限 σ 供 DIIS 稳定收敛。
    - **最终能量使用 E₀（零温外推）**，即 `E₀ = E(T) − ½·T·S`，在退火完成后 E₀ 与无展宽计算结果一致。
    - 使用示例：
      ```toml
      smear = "fermi"
      smear_sigma = 0.05
      smear_anneal = true
      ```
- `smear_sigma_min`：取值f64。退火过程中 σ 的下限值，单位为 Hartree。缺省为 `max(smear_sigma × 0.01, 0.001)`。
    - **σ 不应退火至零**：对于高对称性过渡金属等具有严格简并轨道的体系，σ→0 会导致占据数在简并轨道间 ping-pong 振荡而无法收敛。保留非零下限（~0.001 Ha ≈ 300 K）可确保简并流形的平滑平均占据。
    - 若体系简并程度高、退火后仍不收敛，可手动加大该值，如 `smear_sigma_min = 0.005`。
### $\Delta$-SCF 计算相关关键词
- `force_state_occupation`: 取值是Vector。 $\Delta$-SCF 方法中用于约束轨道占据数的条件。具体声明方式如下：
     - `[
  [reference, prev_state, prev_spin, target_spin, force_occ, force_check_min, force_check_max],
  [reference, prev_state, prev_spin, target_spin, force_occ, force_check_min, force_check_max],
  ...
]`
     - Vector中的每一项对应于一个轨道的约束条件。
     - `reference`: （可选）取值String。$\Delta$-SCF 的计算需要有一个常规的 SCF 计算结果，对应的checkpoint文件的存储位置在 `reference` 中声明。若省略，则默认与 `chkfile` 设置相同。
     - `prev_state`和`prev_spin`：取值i32。定位需要约束的轨道在reference中的轨道序号和自旋通道
     - `target_spin`：（可选）取值i32。在 $\Delta$-SCF 计算中，约束轨道的目标自旋通道
        - 若省略该值，则默认与`prev_spin`相同
        - 若给定，则会在指定自旋通道中寻找与 `prev_state`/`prev_spin` 最相似的轨道
     - `force_occ`：取值f64。设置上述定位的轨道在 $\Delta$-SCF 计算中的占据数
     - `force_check_min`和`force_check_max`：取值i32。在 $\Delta$-SCF 的自洽计算中设置搜索窗口，仅从这个窗口中寻找和 `prev_state`/`prev_spin` 最相似的轨道


## 后自洽场计算相关关键词（Keyword）
- `frozen_core_postscf`: 取值i32，且小于100的两位正整数或者一位正整数。对于后自洽场方法，包括MP2和第五阶密度泛函近似，需要考虑激发组态的贡献。由于原子的内层电子（core electrons）通常不参与化学成键，仅有最外几个价层参与（按主量子数划分）。因此我们可以采用冻芯近似（frozen core approximation）
    - 缺省值为`0`，代表考虑所有电子，不使用冻芯近似
	- 当设置为一位数`n`的时候，不区分原子是主族元素还是过渡金属，冻芯近似下只考虑涉及`n`个最外价层的电子激发组态。
	- 当设置为两位数`mn`的时候，则区分原子类型，对于主族元素考虑`n`个价层上的电子激发（个位上的数），而对过渡金属则考虑`m`个价层（十位上的数）
	- 几个示例和几点说明：
	    - `sp`电子所属电子层由`主量子数`来区分。以第三周期元素Si、P、S为例，`3s3p`属于最高第三价层，而`2s2p`是次高第二价层
	    - `d`电子所属电子层由`主量子数-1`来区分。以3d过渡金属Fe、Cu、Zn为例，`3d4s`属于最高第四价层，而`3s3p`是次高第二价层
	    - `f`电子所属电子层由`主量子数-2`来区分。以5d过渡金属Ir、Pt、Au为例，`4f5d6s`属于最高第六价层，而`4d5s5p`是次高第五价层
	    - `frozen_core_postscf=1`（即`n=1`），表示第三周期元素仅考虑`3s3p`价层电子的贡献，而不考虑`1s2s2p`轨道的电子激发
		- 若`n`等于或大于主族元素占据轨道的电子层数，代表对于这个元素不采用冻芯近似，等价于`n=0`.
	    - `frozen_core_postscf=2`（即`n=2`），表示第二周期元素同时考虑`2s2p`最高价层和`1s`次高价层（即最低核层）的贡献，等价于`n=0`不开冻芯近似。
	    - **注意**：对于传统密度泛函方法，本参数设置不起作用
- `frequency_points`：取值i32。对于RPA型的相关能计算方法，比如RPA、SCSRPA和R-xDH7等，需要对频率空间进行数值积分。这里设置频率积分的格点数目。缺省为20
- `freq_grid_type`：取值i32。对于RPA型相关能计算方法做格点化准备:
     - `0`: 代表使用modified Gauss-Legendre格点。缺省为0
	 - `1`: 代表standard Gausss-Legendre格点
	 - `2`: 代表Logarithmic格点。
- `lambda_points`：取值i32。对于SCSRPA和R-xDH7等方法，对于开窍层的强关联体系，需要对绝热涨落途径（lambda）数值积分。这里设置lambda积分的格点数目。缺省为20

### RI-PT2计算相关设置

首先，RI-PT2 相关设置在 `[ctrl.ri_pt2]` 区块中进行，输入样例如下：
```toml
[ctrl.ri_pt2]
fp_mode = "FP64"
```

输入卡关键词包括：
- `ss_factor`: 取值f64。采用 Spin-Component-Scaled 方式计算 MP2 型相关能（SCS-MP2）时, 用于控制自旋平行（same spin）分量贡献的缩放系数，适用于 `xc` 关键词为 MP2、SCS-MP2或双杂化泛函，以及 `post_correlation` 设置为 PT2 的情况。缺省为 None，即依 `xc` 设置的泛函进行设定。
- `os_factor`: 取值f64。适用场景与 `ss_factor` 一致，采用 SCS-MP2 方法计算相关能贡献时，用于缩放自旋反平行（opposite spin）分量贡献的系数。缺省为 None，即依 `xc` 设置的泛函进行设定。
- `fp_mode`: 取值 String (`"FP32"`, `"FP64"`, `"TF32"`)。设置 RI-PT2 计算中使用的浮点精度。缺省为 `"FP32"`。在 `new_driver = true` 的 CPU 驱动、`engine = "torch"` 以及 PT2 解析响应核（rgfock）中生效；`"TF32"` 仅在 `engine = "torch"` 时有效（f32 存储 + TF32 张量核 matmul），CPU 驱动下自动回退为 `"FP32"`。对于数值梯度计算，建议设置为 `"FP64"` 以获得更高的数值稳定性。
- `mpi_mode`: 取值 usize。设置 RI-PT2 计算中使用的 MPI 模式。缺省为 0，即不使用 MPI。

#### PT2 算法选择

REST 提供三种 PT2 实现算法，通过以下关键词选择：

- `streaming`: 取值 bool。是否启用**流式 PT2 算法**（缺省 `true`）。
  - `true`（推荐，缺省）：采用分块流式 ao2mo + PT2 主循环。不物化完整的 `ri3mo[naux, nvir, nocc]` 张量，而是按占据轨道分块（block_size 个轨道/块）逐块做 AO→MO 变换和 PT2 收缩。内嵌 M1 优化（dsymm 先收缩小边），且通过 scoped thread 预取下一块的 ao2mo（Fix B 流水线），将 ao2mo 时间隐藏在 PT2 主循环后面。
    - 内存峰值：`rimatr + 3 × [naux, nvir, B]`，比原始算法少约 130 GB（对于 nao≈3000 的体系）。
    - 支持：RHF、UHF、ROHF；rayon 并行；单节点计算。
    - 限制：当前不支持 MPI 并行（MPI 激活时自动回退到原始算法）。
  - `false`：使用**原始 legacy 算法**。先一次性生成完整 `ri3mo` 张量（`generate_ri3mo_rayon`），再做 PT2 主循环。
    - 内存峰值：`rimatr + ri3mo`，比流式算法多约 130–185 GB。
    - 支持：RHF、UHF、ROHF；rayon 并行；**MPI 并行**。
    - 适用场景：需要 MPI 并行的大体系。

- `new_driver`: 取值 bool。是否启用**新 rstsr 驱动**（缺省 `false`）。
  - `true`：采用基于 rstsr 张量库的 pair-engine 架构。支持 FP32 半精度、UKS 三自旋对统一处理。但内存峰值最高（存在大 scratch_buf 且不释放 rimatr），且不支持 ROHF 和 MPI。
  - `false`（缺省）：不使用新驱动。

- `stream_block_size`: 取值 usize。流式算法的块大小（仅当 `streaming = true` 时生效）。
  - 缺省（不设置或设为 0）：自动选择。规则为 `B² ≥ 4 × num_threads`，clamp 到 [16, 128]。
  - 典型值：48 核机器 RHF 取 64，96 核机器 UKS 取 32。
  - 调优建议：内存紧张时减小 B（如 32），PT2 主循环并行度不足时增大 B（如 128）。

**算法选择优先级**（满足条件时依次尝试）：

| 优先级 | 条件 | 算法 |
|---|---|---|
| 1 | `new_driver = true` 且非 MPI 且 PT2 且 RHF/UHF | 新 rstsr 驱动 |
| 2 | `streaming = true`（缺省）且非 MPI 且 `use_ri_symm = true` 且 PT2 | **流式算法** |
| 3 | 其它情况 | 原始 legacy 算法 |

**注意**：若设置 `engine = "torch"` 且满足其适用条件（CUDA、非 MPI、PT2 家族、RHF/UHF），则优先使用 torch 引擎（见下节）。

**输入卡示例**：

```toml
# 缺省配置（推荐）：流式算法，自动块大小
[ctrl.ri_pt2]
# streaming = true        # 缺省，无需显式设置
# stream_block_size = 0   # 缺省，自动选择

# 显式使用原始 legacy 算法（需要 MPI 时）
[ctrl.ri_pt2]
streaming = false

# 流式算法 + 自定义块大小
[ctrl.ri_pt2]
streaming = true
stream_block_size = 32

# 使用新 rstsr 驱动（仅非 MPI 的 RHF/UHF）
[ctrl.ri_pt2]
new_driver = true
fp_mode = "FP32"
```

#### Torch 引擎（GPU 加速）

设置 `engine = "torch"` 后，PT2 成对电子相关能张量乘法由 PyTorch 内核（`dfmp2_addons`）实现；AO→MO 变换等仍在 CPU 上完成。适用条件：单节点计算（不支持 MPI）、`xc` 属于 PT2 家族、参考态为 RHF 或 UHF（ROHF 请使用缺省的 `engine = "cpu"`）、整数占据（不支持 smearing/分数占据，此时请使用 `engine = "cpu"`）；不满足条件时程序直接报错，不会静默回退到 CPU 驱动。相关关键词包括：
- `engine`: 取值 String（`"cpu"`、`"torch"`）。缺省为 `"cpu"`，即不采用 Torch 引擎的 GPU 加速。
- `torch_devices`: 取值 `Vec<usize>`。torch 引擎使用的 CUDA 设备列表（逻辑设备号，受 `CUDA_VISIBLE_DEVICES` 过滤），缺省为 `[0]`。多设备驱动（设备列表含 2+ 设备，或 `torch_force_batch_inter = true`）要求至少 8 倍于 GPU 数的相关占据轨道，过小的体系请使用单 GPU。
- `torch_force_batch_inter`: 取值 bool。缺省为 false。当 `torch_devices` 只含一个设备时，强制启用分批的 intra+inter 方案（占据空间切成若干簇，逐簇上传 cderi 切片和轨道块），从而降低单卡显存峰值。这是单卡 GPU 显存溢出（OOM）时的补救选项。
- `torch_nbatch`: 取值 usize。分批 intra+inter 方案的占据簇数目，缺省根据各设备空闲显存自动选择。仅多设备驱动路径（`torch_devices` 含 2+ 设备或 `torch_force_batch_inter = true`）使用。
- `torch_fold`: 取值 String（`"F32Acc"`、`"F64"`、`"F32"`）。f32 matmul 结果逐元素能量折叠的精度。缺省为 `"F32Acc"`（f32 折叠、对能以 f64 累加）；`"F64"` 最慢但最准；`"F32"` 最快但精度最低。仅对 `fp_mode = "FP32"/"TF32"` 生效。
- `torch_batch`: 取值 usize。torch 引擎 j≤i 收缩循环中占据对的批大小，缺省 16；设为 0 表示一次完成全部对。值越大单次 GEMM 越大，FLOPS 利用率更高但显存开销稍高。

输入卡示例：
```toml
# 单 GPU（缺省设备 0，等价于 torch_devices = [0]）
[ctrl.ri_pt2]
engine = "torch"

# 双 GPU 分摊占据轨道
[ctrl.ri_pt2]
engine = "torch"
torch_devices = [0, 1]

# 单卡显存不足时分批计算
[ctrl.ri_pt2]
engine = "torch"
torch_force_batch_inter = true
```

## RRS-PBC计算相关设置
RRS-PBC方法是由张颖教授等提出的一种基于分子团簇计算结果的周期性电子结构模拟方法，相关论文见 Zhang, I.Y., Jiang, J., Gao, B. *et al.* RRS-PBC: a molecular approach for periodic systems. [*Sci. China Chem.* **57**, 1399–1404 (2014)](https://doi.org/10.1007/s11426-014-5183-y)。在 REST 中，RRS-PBC 相关的设置主要在 `[geom]` 区块中进行，请参考后续的说明。`[ctrl]`区块中的相关关键词包括：
- `pbc_eigenval`: 取值String，用于指定存储k点和能级信息的文件路径。如果设置为`"none"`或`"None"`则直接打印到标准输出。缺省为`"none"`。

## GW-BSE计算相关设置

GW-BSE方法相关的设置在 `[quasiparticle_methods]` 区块中进行。GW计算用于获得准粒子（QP）能量，BSE计算在此基础上求解电子-空穴对的两体激发能。REST使用基于RI积分的GW-BSE实现，支持多种GW方案和BSE求解选项。

### GW计算设置

GW计算通过 `gw_or_bse = "gw"` 启动（也内置于 `"bse"` 模式中）。

- `gw_or_bse`: 取值String，必须参数。设置为 `"gw"` 开启GW准粒子能量计算，设置为 `"bse"` 在自动完成GW计算后进一步进行BSE激发能计算（当配合 `gw_scheme = "parse from file"` 时，BSE从文件读取预先计算的准粒子能量而不重新计算GW）。缺省为空，不触发任何计算。
- `gw_scheme`: 取值String，决定GW准粒子方程的求解方式。可选项：
    - `"extrapolated"`（**推荐**）：在费米面附近一定能量窗口内精确求解准粒子方程，其余轨道通过外推获得。兼备高效率与准确性，适合大多数体系。
    - `"linearize"`：线性化GW准粒子方程，对所有轨道逐一求解。
    - `"qp equation"`：直接数值求解准粒子方程。
    - `"parse from file"`：从文件读取预先计算的准粒子能量（配合 `parse_qp_path`），跳过GW计算直接进入BSE。
    - 缺省为 `"no gw"`。
- `scgw`: 取值String，决定GW的自洽方案。可选项：
    - `"g0w0"`（缺省）：单次GW计算，不做自洽迭代。
    - `"evgw"`：本征值自洽GW（evGW），迭代更新准粒子能量中的G部分。需配合 `evgw_rounds` 设置迭代次数。
- `evgw_rounds`: 取值usize，evGW自洽迭代的轮数（上限）。仅在 `scgw = "evgw"` 时需要设置。缺省为0。

#### evGW 外循环收敛控制

evGW 的本质是对映射 `E^(k+1) = F(E^(k))`（`F` = 一轮完整 GW）做不动点迭代。与 PySCF、MolGW 的经验一致，**不做任何加速的朴素迭代对多数分子不会收敛**：`F` 的 Jacobian 在不动点附近有接近甚至超过单位模的特征值，迭代会落入极限环；若再叠加低秩等离面路径中"最近格点"选取带来的分段常值（不连续）误差，连迭代轨迹的逐次可复现性都会丧失。

REST 提供**两种可选的 evGW 求解技巧**，由 `evgw_solver` 选择：

- `evgw_solver`: 取值String，evGW 外循环求解技巧。
    - `"molgw"`（缺省）：**MolGW 实际执行的更新规则**——每个轨道每轮只做**一次求值**
      `E_out = consts + Σ_c(E_in)`（`Z ≡ 1`，不求根、不阻尼）。MolGW 的 `selfenergy_init` 对 `EVSC` 技术只设 `se%nomega = 0`，所以 `find_qp_energy_linearization` 里的 Z 分支是死代码；MolGW 的输出表逐位验证了 `E_qp = E0 + SigX−Vxc + SigC`（`E0` 恒为 KS 能量）。**每轮不在循环内求解准粒子方程**是消除 REST 对粗糙低秩实轴网格的敏感性、以及消除"卫星根跳变"的关键。实测（20 组独立运行）：NH3 5/5、CH4 5/5、CO2 5/5 收敛。
    - `"diis"`：每轮做一次完整 GW 迭代（逐轨道精确求解准粒子方程），再对整条准粒子能量矢量做 **Pulay/DIIS（Anderson）外推**（PySCF 的做法）。实测对 CO2 只有 1/5 收敛，故不再是缺省。
    - `"z_update"`：Z 阻尼（线性化牛顿）单步更新，每个轨道每轮只走一步
      `E_out = E_in + Z · (consts + Σ_c(E_in) − E_in)`，其中 `Z = 1/(1 − ∂Σ_c/∂ω)` 并被截断到 [0, 1]（对应 MolGW `m_selfenergy_tools.f90` 中 `find_qp_energy_linearization` 的 `MIN(MAX(zz,0),1)` 分支）。使用该选项时不再施加 DIIS。
    - `"molgw"`：MolGW 的 evGW 实际执行的更新规则——每个轨道每轮只做一次求值
      `E_out = consts + Σ_c(E_in)`（即 `E_in + 1.0 · (consts + Σ_c(E_in) − E_in)`，`Z ≡ 1`）。

- `evgw_freeze_outside`: 取值bool，缺省 `true`。evGW 中如何处理精确计算窗口之外的轨道：
    - `true`（缺省，MolGW 行为）：让它们保持初始（Kohn–Sham）能量，与 MolGW 一致——MolGW 的 `find_qp_energy_linearization` 先把 `energy_qp_z = energy0` 初始化，只覆盖 `nsemin..nsemax`，窗口外的态原样写回，从而**完全不参与自洽循环**。REST 的刚性外推会让这些态的能量每轮都被重新推导，构成一条额外的慢反馈通道（进入 G 的极点与 W 的分母）。
- `evgw_z_step`: 取值f64，单位Hartree。`z_update` 中用于计算 `∂Σ_c/∂ω` 的中心差分步长。缺省 `0.01`（与 MolGW `step_sigma` 的缺省一致）。

  该步长必须同时满足两个尺度条件：明显大于最小的虚轴求积频点（`num_freq = 20` 时约为 `1.7e-3 Ha`），且不大于留数判据 `cdgw_res_tol`（否则差分被截断偏差污染）。

- `evgw_degeneracy_tol`: 取值f64，单位Hartree，缺省 `1e-4`。evGW 的简并投影容差。

- `evgw_diis`: 取值bool，缺省 `true`。是否对整条准粒子能量矢量做 Pulay/DIIS 外推。它叠加在 `evgw_solver` 所选用的每轮映射之上。
- `evgw_diis_space`: DIIS 历史长度。缺省 8。
- `evgw_diis_start`: 至少积累多少个残差矢量后才启用外推。缺省 2。
- `evgw_diis_safeguard`: 取值bool。缺省 `false`，即标准 DIIS 行为（允许自由外推，仅在结果非有限或严重发散时回退到阻尼步）。设为 `true` 时改用更保守的区间约束回退。
- `evgw_damping`: 取值f64，取值区间 `(0, 1]`。线性混合/阻尼因子 α。
- `evgw_conv_tol`: 取值f64，单位Hartree。收敛判据：一轮内 `max_n |E_n^out - E_n^in|` 小于该值即认为收敛。缺省 `1e-5`。
- `evgw_stop_on_convergence`: 取值bool。达到 `evgw_conv_tol` 后是否提前结束（否则始终跑满 `evgw_rounds` 轮）。缺省 `true`。
- `evgw_report`: 取值bool。是否每轮打印收敛报告（`max|dE_qp|`、`|dG|`、最大步长、HOMO/LUMO 准粒子能量与 gap）。缺省 `true`。

- `gw_extrapolate_occ_threshold`: 取值f64，单位Hartree。在 `gw_scheme = "extrapolated"` 方案中，决定费米面以下精确求解准粒子方程的能量窗口。计算范围包括KS轨道能量落在 `[HOMO - gw_extrapolate_occ_threshold, HOMO]` 的占据轨道。缺省为0.1。
- `gw_extrapolate_vir_threshold`: 取值f64，单位Hartree。在 `gw_scheme = "extrapolated"` 方案中，决定费米面以上精确求解准粒子方程的能量窗口。计算范围包括KS轨道能量落在 `[LUMO, LUMO + gw_extrapolate_vir_threshold]` 的虚轨道。缺省为0.1。

### Renormalized Singles（rsGW）

Renormalized Singles方法通过投影DFT密度矩阵构造单激发HF哈密顿量并对角化，以优化出发点轨道能量，系统性改善GW的精度。参考文献：Jin, Y.; Su, N. Q.; Yang, W. *J. Phys. Chem. Lett.* **2019**, *10* (3), 447–452.

- `renormalized_singles`: 取值bool，设置为 `true` 开启rsGW。得到的RS本征值用于初始化GW中Green函数G的粒子能量。缺省为false。
- `w_rs`: 取值bool，仅在 `renormalized_singles = true` 时生效。设置为 `true` 时进一步使用RS粒子能量初始化GW中的屏蔽库仑相互作用W部分。缺省为false。
- `rs_full_space`: 取值bool，仅在 `renormalized_singles = true` 时生效。缺省为 `false`：分别在占据子空间与未占据子空间内构建HF哈密顿量并各自对角化（块对角化，DFT占据数保持不变）。设置为 `true`：构建**完整轨道空间**的HF哈密顿量 `H^RS = Cᵀ (h + J[D_DFT] − ½K[D_DFT]) C` 并整体对角化，占据与未占据轨道之间允许混合，得到更严格的renormalized singles粒子；RS本征值按能量升序返回并用于GW初始化。
- `rs_use_rs_orbitals`: 取值bool，仅在 `renormalized_singles = true` 时生效。缺省为 `false`：只使用RS本征值重新初始化GW，轨道系数仍为DFT（KS）轨道。设置为 `true`：把HF哈密顿量对角化得到的轨道系数（`C_rs = C_ks · U`，`U` 为变分一致的轨道旋转）写回轨道系数矩阵，于是**RI三中心积分张量会按新的轨道系数从AO基重新组合成MO基**——GW 中所有 `generate_ri3mo_*` / `v_matrix_from_scf` 的AO→MO变换都在RS轨道基下完成。推荐与 `w_rs = true` 搭配使用（此时G与W的能量、积分基组保持一致）。

  打开该选项后，**rsGW 之后的整个后处理过程都统一在 RS 轨道表象中**：

  - 后续 **BSE 计算**的 `v`、`W` 矩阵元、偶极矩阵、以及输出的跃迁振幅（`#i->#a`、NTO、PySOC 导出的 CI 系数）都在 RS 分子轨道基下（BSE 每次调用 `get_submatrix`/`compute_dipole_matrix` 时都重新用当前轨道系数做 AO→MO，不存在 KS 基缓存）；程序会打印 `BSE orbital representation: renormalized-singles (RS) orbitals.` 提示；
  - BSE/响应的**屏蔽能量**（χ⁰ 分母）改用 RS 本征值 `renormalized_singles_particles`（否则会出现"RS 轨道 + KS 能量分母"的混合表象），程序会打印 `BSE screening: using the renormalized-singles eigenvalues`；
  - `bse_cutoff_energy` 等**轨道窗口**判据按 RS 本征值而不是 KS 本征值筛选；
  - 程序在当前目录写出 `rs_orbitals.dat`，内含 `e_rs`、`C_rs`、`e_ks`、`C_ks` 与旋转矩阵 `U`（`C_rs = C_ks·U`），可直接用于把 RS 表象下的跃迁振幅换回 KS 表象：`A_KS = U_o A_RS U_vᵀ`；
  - `print_level > 2` 时程序会额外做一次自洽性校验，打印
    `max|(ia|P)_RS - Σ U U (rs|P)_KS|` 与 `max|mu_RS - Uᵀ mu_KS U|`（应为机器零），用于确认 RI AO2MO 确实使用了新的轨道系数。

rsGW 输入卡示例（完整轨道空间HF对角化 + RS轨道用于RI AO2MO）：

```toml
[quasiparticle_methods]
gw_or_bse = "gw"
gw_scheme = "extrapolated"
scgw = "g0w0"
renormalized_singles = true
rs_full_space = true
rs_use_rs_orbitals = true
w_rs = true
use_low_rank_contour = true
```

### Low-Rank Contour Deformation（推荐加速方法）


- `use_low_rank_contour`: 取值bool，设置为 `true` 启用低秩等离面加速。缺省为false。
- `low_rank_grid_type`: 取值String，实轴极化率Chi的采样格点分布方式。`"linear"`（缺省）为线性分布；`"quadratic"` 为二次幂律分布，在零能附近更密集。
- `low_rank_interp`: 取值String，实轴低秩 `v·χ·v` 在格点之间的取值方式。
    - `"linear"`（缺省）：在相邻两个格点之间做**线性插值**（`M(ω) = (1−t) M_i + t M_{i+1}`；由于留数收缩 `rᵀMr = Σ_v (r·e_v)² λ_v` 对特征向量的符号与次序均不变，两侧的低秩分解可以直接分别收缩再按权重相加，无需特征向量匹配）。
    - `"nearest"`：取最近格点（分段常值）。这是 REST 的历史行为，也与 MolGW `sf_interpolate_vsqrt_chi_vsqrt` 中只做 `MINLOC` 的做法一致；保留用于复现旧结果或做 A/B 对比。

>
- `nomega_chi_real`: 取值usize，实轴极化率Chi的采样格点数。缺省为6。增大此值可提高精度但增加计算量。
- `low_rank_demax_window`: 取值bool，缺省 `false`(扫描全部轨道)。低秩实轴网格的 `de_max` 扫描范围。

- `nomega_sigma`: 取值usize，自能Sigma实轴扫描点数（每侧），在de_max扫描中使用。缺省为10。
- `step_sigma`: 取值f64，自能Sigma实轴扫描步长，单位Hartree。缺省为0.05。

> **精度提示**：如果 GW 计算窗口包含深占据轨道或高虚轨道，低秩实轴插值需要更密的频率格点。此时应增大 `nomega_chi_real`（例如提高到 1000–5000 或更高），否则这些轨道的 QP 能量以及后续 BSE 激发能可能不够准确。对于仅关心 HOMO/LUMO 附近或低激发态的情况，可以使用较小的 `nomega_chi_real` 以节省计算时间。

低秩路径涉及的另外两个参数：

- `cdgw_eta`: 取值f64，单位Hartree。实轴响应函数 χ₀(ω) 的 Lorentzian 展宽 η。缺省 `1e-3`。它与 `nomega_chi_real` 共同决定低秩实轴表示的精度：实轴格点间距需与 η 相称，否则相邻格点上的 `v·χ·v` 差异过大（低秩路径按"最近格点"取值，见下），单轮 GW 的误差可达 1e-3 Ha 量级。需要提高精度时优先增大 `nomega_chi_real`。
>
- `cdgw_res_tol`: 取值f64，单位Hartree。CD-GW 中极点/留数的数值判据：`de >= -cdgw_res_tol` 时计入该极点，`|de| < cdgw_res_tol` 时按半权重 ×0.5 计入；`de_max` 扫描同样使用该阈值。缺省 `1e-3`。


### GW求解器通用参数

- `gw_rootfinder`: 取值String，准粒子方程求根方法。`"newton"`（缺省）为Newton法求根；`"interpolation"` 为插值求根。
- `gw_search_grid`: 取值usize，插值求根法的搜索格点数。仅当 `gw_rootfinder = "interpolation"` 时生效。缺省为51。
- `gw_span_energy`: 取值f64，单位Hartree，插值求根法的能量扫描范围。缺省为0.2。
- `gw_linearize_shift`: 取值f64，线性化GW中的有限差分位移量，单位Hartree。缺省为0.01。

### AC-GW（解析延拓 GW 变体）

- `gw_variant`: 取值String，GW 变体类型：
    - `"cd"`（缺省）：轮廓变形（Contour Deformation）路径。
    - `"ac"`：解析延拓（Analytic Continuation）路径。配合 `gw_scheme = "extrapolated"` 使用。`ac` 路径暂不支持 `fourier_self_energy` 或 `hermite_self_energy`。
- `ac_num_samples`: 取值usize，Padé 虚轴采样点数。从 quadrature 频点中按 PySCF 风格的指数递减步长选取。缺省为 16。
- `ac_eta`: 取值f64，Padé 求值的展宽参数 η（小正数）。缺省为 0.001。
- `ac_pade_step_ratio`: 取值f64，PySCF 风格 Padé 采样的步长衰减比（< 1.0 时低频更密）。缺省为 2/3。
- `gw_switch_fallback_threshold`: 取值f64。当 |E_KS| 超过此值时退化为静态近似。缺省为 1e6。

### GW结果导出

- `save_qp`: 取值bool，设置为 `true` 将准粒子能量保存到文件。缺省为false。
- `save_qp_path`: 取值String，保存准粒子能量的文件路径。缺省为 `"single_qp_path.txt"`。
- `parse_qp_path`: 取值String，从文件读取准粒子能量的路径。当 `gw_scheme = "parse from file"` 时从此路径读取预先计算的准粒子能量用于后续BSE计算。缺省为 `"./qp_energies"`。

#### GW/evGW 断点续算

- `save_gw_checkpoint`: 取值bool，缺省 `false`。GW 完成后及每轮 evGW 结束时，把后自洽场状态写入 `[ctrl] chkfile` 的 `rest_gw_checkpoint` 组（不另建文件，要求 `chkfile` 非 `"none"`）。更新方式为"复制到 `<chkfile>.tmp` → 替换该组 → 改名回去"，写入中途被打断不会损坏 chkfile。
- `resume_from_checkpoint`: 取值bool，缺省 `false`。从 chkfile 恢复状态并**跳过 SCF 迭代**：若存入的是已完成的 GW，连 GW/evGW 一并跳过、直接进入 BSE/响应；若存入的是进行中的 evGW，从保存的那一轮继续（需 `scgw = "evgw"`）。

续算时在同一 `chkfile` 上把 `resume_from_checkpoint` 改为 `true` 即可；想多跑几轮，同时调大 `evgw_rounds`。

### BSE计算设置

BSE计算在 `gw_or_bse = "bse"` 时进行，在GW准粒子能量（或从文件读取的准粒子能量）基础上构建BSE kernel并对角化求解垂直激发能。

- `bse_tda`: 取值bool，设置为 `true` 使用Tamm-Dancoff近似（仅保留BSE kernel的A子矩阵），设置为 `false`（缺省）使用完整BSE（包含A和B子矩阵的非TDA计算）。TDA近似计算量更小，通常对低能激发态精度足够。
- `bse_spin`: 取值String，必须参数。指定计算的自旋激发类型（**限制性参考态**）：
    - `"singlet"`：单重态激发。
    - `"triplet"`：三重态激发。
    - `"both"`：同时计算单重态和三重态激发。
    - 对**非限制参考态**（`spin_polarization = true`，UBSE）该关键词不选择自旋通道，
      只作为 BSE 开关；`"triplet"` / `"both"` 已移除，详见"Unrestricted GW-BSE（UGW/UBSE）设置"一节。
- `bse_cutoff_energy`: 取值f64，单位Hartree。KS能级高于此能量的虚轨道将被排除在BSE激发空间之外，缩减BSE kernel维度。建议根据体系设置为合理值（含几百条虚轨道即可），缺省为1e6（几乎不截断）。
- `davidson_target_excitations`: 取值usize，需要计算的激发态数目。缺省为6。
- `bse_davidson_solver`: 取值bool，设置为 `true` 使用Davidson迭代对角化（推荐用于仅需少数低能激发态的体系），设置为 `false`（缺省）使用完整矩阵对角化（适合小体系或需要全部激发态的情况）。
- `davidson_converge_threshold`: 取值f64，Davidson求解器的收敛阈值。缺省为1e-10。收敛判据：`||r|| < sqrt(tol)` 且 `|de| < tol`。
- `davidson_max_iter`: 取值usize，Davidson最大迭代次数。缺省为20。
- `davidson_maximum_subspace_size`: 取值usize，Davidson最大子空间维度倍数。实际最大子空间 = max(目标激发数 × 此值, 最小维度)。缺省为2。
- `davidson_restart_dimensions`: 取值usize，Davidson重启动维度。当子空间达到上限后，收缩至此数量的近似特征向量后再继续扩张。缺省为5。

### GW输入卡示例

G0W0计算（使用extrapolated方案和低秩等离面加速）：
```toml
[ctrl]
xc = "pbe"
basis_path = "def2-TZVP"
auxbas_path = "def2-TZVP-ri"
charge = 0.0
spin = 1

[geom]
name = "H2O"
unit = "angstrom"
position = """
    O  0.0000000000      0.0000000000      0.0000000000
    H  0.0000000000      0.7569500000      0.5858820000
    H  0.0000000000     -0.7569500000      0.5858820000
"""

[quasiparticle_methods]
gw_or_bse = "gw"
gw_scheme = "extrapolated"
scgw = "g0w0"
threshold = 0.1
use_low_rank_contour = true
```

evGW计算（使用rsGW初始化和低秩围道变形加速）：
```toml
[quasiparticle_methods]
gw_or_bse = "gw"
gw_scheme = "extrapolated"
scgw = "evgw"
evgw_rounds = 5
# 外循环求解技巧与收敛判据（缺省即为下列取值；'diis' 鲁棒性最好，'z_update' 在
# 无近简并前线轨道的体系上更快，但两者互斥）
evgw_solver = "diis"        # 或 "z_update"（Z 阻尼牛顿单步）/ "molgw"（MolGW 的单次求值）
evgw_freeze_outside = false # true = 窗口外轨道保持 KS 能量（MolGW 行为）
evgw_diis = true
evgw_diis_space = 8
evgw_damping = 1.0
evgw_z_step = 0.01          # 仅 evgw_solver = "z_update" 时使用
evgw_conv_tol = 1e-5
evgw_stop_on_convergence = true
renormalized_singles = true
w_rs = true
threshold = 0.1
use_low_rank_contour = true
```

### BSE输入卡示例

BSE单重态激发计算（TDA近似，Davidson迭代求解10个激发态）：
```toml
[quasiparticle_methods]
gw_or_bse = "bse"
gw_scheme = "extrapolated"
scgw = "g0w0"
threshold = 0.15
use_low_rank_contour = true
bse_spin = "singlet"
bse_tda = true
davidson_target_excitations = 10
bse_davidson_solver = true
bse_cutoff_energy = 20.0
```

BSE三重态完整（非TDA）计算：
```toml
[quasiparticle_methods]
gw_or_bse = "bse"
gw_scheme = "extrapolated"
scgw = "g0w0"
use_low_rank_contour = true
bse_spin = "triplet"
bse_tda = false
davidson_target_excitations = 6
bse_davidson_solver = true
bse_cutoff_energy = 15.0
```

BSE同时计算单重态和三重态（从文件读取准粒子能量）：
```toml
[quasiparticle_methods]
gw_or_bse = "bse"
gw_scheme = "parse from file"
parse_qp_path = "./qp_energies.txt"
bse_spin = "both"
bse_tda = true
davidson_target_excitations = 10
bse_davidson_solver = true
```
### Unrestricted GW-BSE（UGW/UBSE）设置

**触发方式**：在 `[ctrl]` 中设置：
```toml
[ctrl]
charge = 0.0
spin = 2.0          # 例如 NH2 双自由基/双基态，2S = 2
spin_polarization = true
```

UBSE **没有**自旋通道选择：`bse_spin` 对非限制参考态不再表示"单重态/三重态"，
它只保留"是否触发 BSE"的作用：

- `bse_spin = "none"`（缺省）：不触发 BSE 计算。
- `bse_spin = "singlet"`（或 `"unrestricted"`）：触发非限制 BSE。此处只有一个合法的自旋守恒通道 ——
  直接（屏蔽库仑）核与自旋无关，把 α、β 两个块耦合在一起，因此不存在"去掉直接项"的第二个通道。
- `bse_spin = "triplet"` / `"both"`：**已移除**，程序会报错并给出说明。原因同上：
  对自旋对称的参考态，去掉直接项的算符会把真正的三重态与对称扇区中无物理意义的根混在一起；
  对开壳层参考态（如 NH₂ 双重态）它不对应任何物理通道，甚至会产生零能假根。
  限制性参考态下的 `bse_spin = "singlet" / "triplet" / "both"` **保持不变**
  （限制性参考态可以自旋适配，那里的三重态通道是严格的物理通道）。

> 注：Restricted 与 Unrestricted 路径的 `bse_spin` 语义不同：限制性路径中它是真正的自旋通道选择器，
> 非限制路径中它只是 BSE 的开关（"none" 之外的值都触发同一个自旋耦合通道）。

UBSE 支持 dense、Davidson、FEAST 三种对角化方式。

一个完整的 UGW+UBSE 输入示例（NH₂ 双基态，PBE0/cc-pVDZ，Davidson full BSE）：

```toml
[ctrl]
print_level = 1
num_threads = 4
xc = "pbe0"
basis_path = "/path/to/cc-pvdz"
auxbas_path = "/path/to/cc-pvdz-rifit"
basis_type = "spheric"
auxbas_type = "spheric"
eri_type = "ri-v"
charge = 0.0
spin = 2.0
spin_polarization = true
initial_guess = "sad"
mixer = "diis"
max_scf_cycle = 100
scf_acc_rho = 1.0e-8
scf_acc_eev = 1.0e-7
scf_acc_etot = 1.0e-10

[geom]
name = "NH2"
unit = "angstrom"
position = """
N  0.0000000000   0.0000000000   0.0000000000
H  0.0000000000   0.8030000000  -0.6350000000
H  0.0000000000  -0.8030000000  -0.6350000000
"""

[quasiparticle_methods]
gw_or_bse = "bse"
gw_scheme = "extrapolated"
scgw = "g0w0"
gw_variant = "cd"
save_qp = true
save_qp_path = "qp_rest.txt"
bse_spin = "singlet"
bse_tda = false
bse_davidson_solver = true
davidson_target_excitations = 6
davidson_maximum_subspace_size = 120
davidson_restart_dimensions = 9
davidson_max_iter = 100
davidson_converge_threshold = 1e-8
```

> 注意：UGW/UBSE 目前主要用于共线开壳层体系。非限制路径下 `bse_spin` 只作为 BSE 开关
> （`"none"` 之外的值都触发同一个自旋耦合通道），不再选择自旋通道；`"triplet"` / `"both"` 已移除。

## 溶剂化计算相关设置
- `solvent_model`: 取值String, 用于指定用于计算的溶剂模型。目前支持CPCM, COSMO, IEFPCM, SS(V)PE,SMD。缺省为CPCM。SMD及其梯度为实验性功能。
- `solvent`: 取值String。支持溶剂见[用户手册](https://gitee.com/restgroup/rest_doc/blob/master/source_zh/user/solvent.md)。

溶剂化计算进阶自定义设置
- `solvent_ri`: 取值bool, 设置为true为溶剂化能计算开启辅助基，设置为false溶剂化计算不开启辅助基。缺省为true。目前溶剂化梯度(job_type = "opt"/"force")计算没有用辅助基。
- `solv_epsilon`: 取值f64, 为溶质的介电常数。缺省为1.0 (真空)。介电常数表可以参考用户手册或 <http://sobereva.com/g09/k_scrf.htm> 的最后。此项为进行PCM(包括CPCM, COSMO, IEFPCM, SS(V)PE)计算的参数，设置后可以不用设置`solvent`参数。
- `solvent_descriptors`: 取值[f64, 8]。各项含义见[用户手册](https://gitee.com/restgroup/rest_doc/blob/master/source_zh/user/solvent.md)。第二项25度折射率为非必须项，可以设为-1.0。此项为进行SMD计算的参数，设置后可以不用设置`solvent`参数。
- `pcm_cavity_radii`: 取值String, 用于指定空腔的半径使用类型。目前支持Bondi, UFF。缺省为UFF。当方法为SMD时将使用特定半径设置，此项不生效。
- `solvent_enabled`: 取值bool，设置为true则启用溶剂化计算。如果没有`solvent_enabled`字段但有`solvent_model`/`solvent`/`solvent_descriptors`的设置且内容非空的时候，同样启用溶剂化计算。其他情况缺省为false。

## 相对论方法计算相关设置
- `rel`: 取值String, 指定用于计算的相对论方法。目前支持`"sfx2c"`，即 spin-free X2C 方法，缺省为 None （不启用相对论方法进行计算）。

## TD-DFT计算相关设置

TD-DFT方法相关的设置在 `[tddft]` 区块中进行。REST支持基于RI积分的TD-DFT激发能计算，包括TDA近似和完整线性响应两种方案，支持限制性（单重态/三重态）与非限制（UHF/UKS）参考态，并支持范围分离杂化（RSH）泛函。

### 基础设置

- `tddft_method`: 取值String，选择TD-DFT求解方法：
    - `"tda"`：Tamm-Dancoff近似，仅求解A子矩阵的本征值问题。计算量较小，对低能激发态通常与完整线性响应精度相当。
    - `"lr"`（缺省）：完整线性响应，同时使用A和B子矩阵，对激发能的描述更完备。
- `tddft_spin`: 取值String，指定**限制性**（自旋适配）参考态的自旋通道：
    - `"singlet"`：单重态激发，库仑耦合因子为2。限制性参考态不设置该关键词时缺省取 `"singlet"`。
    - `"triplet"`：三重态激发，库仑耦合因子为0；仅 AO 模式（需同时设 `tddft_mode = "ao"`）支持。
    - `"both"`：先算单重态、再算三重态；同样仅 AO 模式支持，`pysoc = true` 的 PySOC 导出需要此项。
    - 其它取值将报错。
    - **该关键词不适用于UTDDFT**：显式设置任何取值（包括 `"singlet"`）都会报错。
- `nroots`: 取值usize，需要计算的激发态数目（根的数目）。缺省为6。
- `tddft_cutoff_energy`: 取值f64，单位Hartree。KS轨道能量高于此值的虚轨道将被排除在TD-DFT激发空间之外。设置合理值（如20.0-100.0）可显著缩减激发空间维度，加速计算。缺省为1e6（几乎不截断）。
- `tddft_mode`: 取值String，选择TD-DFT计算模式：`"mo"`（缺省，MO-basis RI）/ `"ao"`（AO-basis）。
- `grid_batch`: 取值bool，仅AO模式生效，缺省为`true`。
- `tddft_ao_rik_driver`: 取值String，仅AO模式生效，`"semitrans"`（缺省）/ `"dm"` / `"lowrank"`。
- `tddft_fxc_driver`: 取值String，仅AO模式生效，`"semitrans"`（缺省）/ `"mo"` / `"dm"`。

### Davidson求解器参数

REST默认使用Davidson迭代对角化算法求解TD-DFT本征值问题。

- `davidson_tol`: 取值f64，Davidson求解器的收敛阈值。缺省为1e-10。收敛判据：`||r|| < sqrt(tol)` 且 `|de| < tol`。
- `davidson_max_iter`: 取值usize，Davidson最大迭代次数。缺省为50。
- `davidson_max_subspace`: 取值usize，最大子空间维度倍数。实际最大子空间 = min(nroots × 此值, 激发空间总维度)。缺省为8。

### XC Kernel设置

- `tddft_use_optimized_fxc`: 取值bool，设置为 `true`（缺省）使用rayon并行的优化fxc kernel加速矩阵-矢量积运算。一般用户无需修改。

### SCF稳定性分析

在SCF收敛后，可以对参考态波函数做稳定性分析。该分析基于TDDFT Hessian（轨道Hessian的最低本征值，由Davidson求解器计算），仅做检查（不改变波函数）：

- `stability`: 取值String，选择要进行的稳定性检查：
    - `"off"`（缺省）：不做稳定性分析
    - `"internal"`：内部稳定性检查（RHF/RKS：轨道Hessian `4(A^S+B^S)`；UHF/UKS：轨道Hessian `2(A+B)`）
    - `"external"`：外部稳定性检查（RHF/RKS参考态：RHF→UHF通道 `(A^T+B^T)`；UHF/UKS参考态暂不支持）
    - `"full"`：同时进行内部与外部检查
    - `"auto"`（**推荐**）：自动选择参考态适用的所有检查：RHF/RKS 为内部 + 外部检查；UHF/UKS 为内部检查（UHF→GHF 外部检查尚未实现）。目前与 `"full"` 行为一致；`"full"` 的含义保持固定（内部 + 外部），未来 `"auto"` 可能随方法/参考态扩展新的检查
- `stability_nroots`: 取值usize，Hessian Davidson求解的最低本征值数目。缺省为3。
- `stability_tol`: 取值f64，Hessian Davidson收敛阈值。缺省为1e-8。

注意：
- 仅支持RHF/RKS与UHF/UKS参考态（ROHF暂不支持）。
- 设置 `stability ≠ "off"` 后，程序只做稳定性分析，不再进行TD-DFT激发能计算。

### 输入卡示例

TD-DFT单重态TDA计算（6个激发态）：
```toml
[ctrl]
xc = "b3lyp"
basis_path = "def2-SVP"
auxbas_path = "def2-SVP-ri"
charge = 0.0
spin = 1

[geom]
name = "H2O"
unit = "angstrom"
position = """
    O  0.0000000000      0.0000000000      0.0000000000
    H  0.0000000000      0.7569500000      0.5858820000
    H  0.0000000000     -0.7569500000      0.5858820000
"""

[tddft]
tddft_method = "tda"
tddft_spin = "singlet"
nroots = 6
tddft_cutoff_energy = 100.0
```

TD-DFT单重态完整线性响应计算（10个激发态）：
```toml
[tddft]
tddft_method = "lr"
tddft_spin = "singlet"
nroots = 10
davidson_tol = 1.0e-10
tddft_cutoff_energy = 50.0
```

TD-DFT三重态TDA计算（5个激发态，限制性参考态，需 AO 模式）：
```toml
[tddft]
tddft_method = "tda"
tddft_spin = "triplet"
tddft_mode = "ao"
nroots = 5
```

非限制TD-DFT（UTDDFT）的激发能计算（如 NH₂ 双重态，PBE0/cc-pVDZ）：
```toml
[ctrl]
charge = 0.0
spin = 2.0
spin_polarization = true
xc = "pbe0"
basis_path = "/path/to/cc-pvdz"
auxbas_path = "/path/to/cc-pvdz-rifit"

[tddft]
tddft_method = "tda"
nroots = 6
# 不要设置 tddft_spin：非限制参考态不适用该关键词，任何显式取值（包括 "singlet"）都会报错
```

## 解析Hessian计算相关设置

解析Hessian（Analytic Hessian）计算通过 `[ctrl]` 区块中的 `hessian` 子表触发。REST采用基于RI积分的解析二阶导数实现，支持RHF（闭壳层Hartree-Fock）和RKS（闭壳层DFT，含LDA/GGA/杂化泛函）方法，使用高效的CP-HF Krylov求解器和BLAS加速的张量收缩。

### 基础设置

`hessian` 子表位于 `[ctrl]` 区块中，若存在则触发解析Hessian计算；若不存在（缺省），则不进行Hessian计算。子表内的关键词包括：

- `frequencies`: 取值bool，设置为 `true` 在Hessian矩阵计算完成后对角化质量加权Hessian，计算振动频率（cm⁻¹）和简正模式并保存到文件。缺省为false。
- `solver`: 取值String，CP-HF求解器，可选 `"krylov"`（缺省）或 `"dense"`。
- `krylov_max_cycle`: 取值usize，Krylov求解器的最大迭代次数。对于绝大多数体系，50轮已足以收敛到机器精度。缺省为50。
- `krylov_tol`: 取值f64，Krylov求解器的残差范数收敛阈值。缺省为1e-9。实际收敛还受 `krylov_lindep` 约束。
- `krylov_lindep`: 取值f64，Krylov求解器的线性相关阈值。缺省为1e-15。
- `krylov_tol_inflation`: 取值f64，容忍系数。若真残差 `||r|| < factor * tol`，接受该解而不触发 per-root 求解。缺省为1000.0。
- `verbose`: 取值usize，Hessian计算的信息输出等级：
    - `0`：静默模式，仅输出最终结果。
    - `1`（缺省）：正常输出，打印各阶段耗时和Hessian矩阵摘要。
    - `2`：调试模式。额外打印Hessian子块及中间量的max_abs、完整timing profile，并将 `rest_hess_tot.npy`、`rest_h_partial.npy`、`rest_cphf.npy`、`rest_hess_nuc.npy` 等中间量保存至 `/tmp/rest_hess_<pid>/` 目录。
- `hessian_matrix_path`: 取值String，保存完整Hessian矩阵（单位Hartree/Bohr²）的文本文件路径。缺省为 `"./HessianMatrix.txt"`。
- `eigenmodes_path`: 取值String，保存振动频率和简正模式位移矢量的文本文件路径。仅当 `frequencies = true` 时输出。缺省为 `"./EigenModes.txt"`。

### 相关 `[ctrl]` 全局设置

以下 `[ctrl]` 区的全局关键词对Hessian计算有直接影响：

- 辅助基组响应修正在Hessian流程中始终启用；全局 `auxbasis_response` 开关不再改变Hessian计算。
- `max_memory`: 取值f64，每个 MPI 进程的内存限制（单位MB）。Hessian流水线内嵌内存监控器（MemMonitor），在峰值RSS超过此限制时提前终止以防止系统OOM。

### 计算流水线

解析Hessian计算共分为四个阶段，各阶段之间自动调用内存裁剪（`malloc_trim`）释放不再需要的中间量：
1. **Hess核贡献（calc_e1）**：动能+核吸引积分二阶导数，计算量小。
2. **部分Hessian（calc_ej_ek）**：库仑+交换积分对Hessian的贡献，所有G项均采用BLAS GEMM优化。HF/杂化泛函的主要计算量集中于此。
3. **一阶Fock响应（calc_h1ao）**：对所有原子方向的RI积分一阶响应生成h1ao矩阵。
4. **CP-HF贡献+组装（calc_cphf_contrib + calc_hess_nuc）**：通过固定的批量Krylov求解器（最大50轮、阈值1e-12）计算轨道弛豫对Hessian的贡献，与核Hessian相加得到总Hessian。

### 推荐配置

**高效计算Hessian矩阵（仅输出Hessian矩阵，不计算频率）**：
```toml
[ctrl]
xc = "b3lyp"
basis_path = "def2-SVP"
auxbas_path = "def2-SVP-ri"
charge = 0.0
spin = 1
hessian = {}

[geom]
name = "H2O"
unit = "angstrom"
position = """
    O  0.0000000000      0.0000000000      0.0000000000
    H  0.0000000000      0.7569500000      0.5858820000
    H  0.0000000000     -0.7569500000      0.5858820000
"""
```

**Hessian+振动频率+简正模式输出**：
```toml
[ctrl]
xc = "pbe0"
basis_path = "def2-TZVP"
auxbas_path = "def2-TZVP-ri"
charge = 0.0
spin = 1
hessian = { frequencies = true }
```

**完整调试输出（含中间量.npy保存，用于对比PySCF等参考程序）**：
```toml
[ctrl]
xc = "hf"
basis_path = "def2-SVP"
auxbas_path = "def2-SVP-ri"
charge = 0.0
spin = 1
hessian = { frequencies = true, verbose = 2 }
```

## 解析梯度性质模块 `analdrv` 计算相关设置

解析梯度模块 `analdrv` 模块是实验性质模块。目前实现了 Hessian (原子核坐标二阶梯度) 与电多极矩 (1-4 阶：偶极、四极、八极、十六极) 功能。
它实现了不同于 `hessian` 模块的解析 Hessian 计算。目前该模块的 Hessian 功能支持 RHF/RKS/UHF/UKS 方法。对于 DFT，支持 LDA/GGA/mGGA 以及其对应的杂化泛函，包括范围分离杂化泛函 (RSH)。该模块的程序有性能优化，与目前顶级的量化程序 (ORCA 等) 有相当或更好的性能。
多极矩功能支持 RHF/RKS；对于 PT2 族后自洽 (fifth-rung) 方法 (MP2、XYG3 等 xDH/BDH 类双杂化泛函)，在 SCF 密度之外额外计入关联密度增量，默认通过 Z-vector (CP-SCF) 求解弛豫增量。所有矩以原子单位输出，并给出核、SCF、关联 (corr)、响应 (resp) 与总 (tot) 的分项分解；四极矩额外输出无迹形式。

### 设置待计算性质的任务

目前支持 Hessian (`"hessian"`，别名 `"hess"`/`"freq"` 等) 与电多极矩 (`"multipole"`，别名 `"pole"`/`"dipole"`) 两类任务。需要在 `[ctrl]` 区块中设置 `analdrv_tasks` 关键词 (单个字符串或字符串列表) 以启动对应性质的计算；两类任务可以同时指定。若希望同时计算热力学矫正，请同时指定 `[thermo]` 区块。
```toml
[ctrl]
analdrv_tasks = "freq"

[thermo]
```

多极矩任务 (或与 Hessian 的组合) 例如：
```toml
[ctrl]
analdrv_tasks = ["multipole", "hessian"]
```

### 解析梯度模块 `analdrv` 区块选项

在设置任务后，用户可以在 `[analdrv]` 区块中设置对应的计算选项。该区块的关键词分为四个部分：通用选项、自洽场响应选项、核坐标导数性质选项、以及电多极矩选项。

#### 通用选项

- `verbose`：打印强度。默认为 None，使用输入卡 `[ctrl]` 区块的 verbose。

#### 自洽场响应选项

这类选项控制自洽场响应 (analdrv 中主要用于计算 CP-SCF、以及 post-SCF 方法的 generalized Fock) 如何计算，但不控制传入求解的量 (如参与计算的原子范围)。这些选项均只服务于 CP-SCF (响应) 求解：求解器选项以 `cpscf_` 为前缀；响应 DFT 格点并非求解器设置，命名为 `grid_level_cpscf` (与 `grid_level_skeleton` 同属 `grid_level_*` 命名)。

- `cpscf_level_shift`：CP-SCF 求解时对 $\varepsilon_i - \varepsilon_a$ 的求解偏移。默认为 0，单位 Hartree。
- `cpscf_tol`：CP-SCF 中的 Krylov 求解阈值。默认 1e-9，无量纲。实际求解阈值也受制于 `cpscf_lindep`。
- `cpscf_max_cycle`：CP-SCF 最大迭代步数。默认为 42 步。CP-SCF 与 SCF 不同，一般 6-10 步能收敛。这里的最大步数一般不需要设得很大。
- `cpscf_max_space`：CP-SCF 中 Krylov 空间的数量。默认为 14。该数值不宜设太小，因为超过该数值时，Krylov 求解器会代入最后一次迭代重新作为初猜，重置求解过程。但该数值设太大会对内存产生压力。
- `cpscf_lindep`：CP-SCF 中一些数值过程的数值精度阈值。默认 1e-15，无量纲。
- `cpscf_tol_inflation`：容忍系数。若 Krylov 真残差 `||r|| < factor * tol`，接受该解而不触发 per-root 求解。缺省为1000.0。
- `grid_level_cpscf`：CP-SCF 中响应路径 (响应/A 张量收缩的 DFT 计算；fock 路径仍用 SCF 格点) 的 DFT 格点级别。仅影响 numint_matmul 后端实现。默认为 None，是 `[ctrl]` 中 grid_generation_level 关键词设定值减 2 (SCF 默认格点级别是 3，对应 Hessian 的级别是 1)；显式取值不设下限。
- `resp_auxbas_path`：响应计算辅助基路径。默认为 None，即使用 `[ctrl]` 中 auxbas_path 设定的辅助基组。若显式指定，则在 CP-SCF/Z-Vector/response 计算中使用对应辅助基。该选项通常用于指定更小的辅助基以加速 CP-SCF 计算，但作为代价会降低 CP-SCF 的精度且增加内存占用。

这些关键词的旧名称 `cphf_*` 前缀 (`cphf_tol`、`cphf_lindep`、`cphf_tol_inflation` 等) 目前仍然作为别名被接受；响应格点则同时接受其旧名称 `grid_level_cphf`。

#### 核坐标导数性质选项

这类选项控制对哪些原子坐标求导、以及梯度/Hessian 等导数性质 (及其衍生的振动、热力学分析) 如何计算，但不控制 CP-SCF 方程如何求解。

- `atm_list`：选择一部分原子进行 Hessian 计算。默认为 None，即所有原子参与 Hessian 计算。
- `grid_level_skeleton`：Skeleton 导数 (包括 2 阶 Hessian 贡献、1 阶 Fock 贡献) 的 DFT 格点级别。仅影响 numint_matmul 后端实现。默认为 None：
  - LDA/GGA 使用与 SCF 同样的格点；
  - mGGA 分为两种情况：若 `grid_shift_deriv` 为 true 则保持 SCF 格点；若为 false 则将比 `grid_generation_level` 增加 2 级别。
- `grid_shift_deriv`：是否在 DFT skeleton 导数中引入格点偏移导数 (格点权重对原子核坐标的导数、以及格点坐标偏移产生的导数)。取值 bool，默认为 `true`：引入后 skeleton 导数恢复平移不变性。设为 `false` 时退化为不含格点偏移导数。仅影响 numint_matmul 后端实现；要求标准原子生成的 DFT 格点，若使用外部格点 (`external_grids`)，因格点无原子归属，应设为 `false`。
- `tol_point_group`：振动分析中的点群对称性判断阈值 (用于计算转动对称性，对熵矫正有贡献)。默认 1e-5，单位 Bohr / sqrt(atom)。
- `dftd_hess_step`：经验色散校正 (DFT-D3/DFT-D4) 对 Hessian 贡献的数值差分步长。默认为 3e-4，单位 Bohr。
- `gau_thermo`：是否使用 Gaussian 类型的热力学能矫正。默认 false。该选项仅作参考；目前 REST 的热力学矫正通常是定义 `[thermo]` 区块以进行计算。Gaussian 类型热力学能矫正接受输入卡中 `[thermo]` 区块的关键词 `temperature`, `pressure`, `symmetry_number` 与 `electronic_energy`。

作为例子，运行 Hessian 计算、增大 CP-SCF Krylov 求解器空间到 20、强制 CP-SCF 中 DFT 格点积分级别为 2，所需要引入的、相比于能量计算的额外设置如下：
```toml
[ctrl]
analdrv_tasks = "freq"

[analdrv]
cpscf_max_space = 20
grid_level_cpscf = 2

[thermo]
```

#### 电多极矩选项

这类关键词控制电多极矩 (`multipole` 任务) 的计算：求哪几阶矩、以何处为原点、以及后自洽 (PT2 族) 密度增量的处理方式，但不控制 CP-SCF 方程如何求解。所有矩以原子单位输出；默认原点为坐标原点 `[0, 0, 0]`，与 Gaussian、pyscf 的多极矩打印约定一致。对于含赝势 (ECP) 的体系，核电荷矩按有效核电荷 (核电荷数减去赝势冻结电子数) 计，该约定同样与 Gaussian 一致。

- `multipole_orders`：取值为正整数列表：1 = 偶极、2 = 四极、3 = 八极、4 = 十六极。默认为 `[1, 2, 3, 4]`。
- `multipole_origin`：显式指定多极矩计算原点，取值为长度 3 的浮点数列表，单位 Bohr。默认为 None，即坐标原点 `[0, 0, 0]`。注意四极及以上矩 (2–4 阶) 的取值依赖分子在输入卡坐标系中的摆放位置；若需以核质量中心等其他原点取值 (如与文献值比较)，请显式设置本关键词。
- `multipole_rdm1_relax`：双杂化密度增量的处理方式，取值 `"relaxed"` (求解 Z-vector 并计入响应增量) 或 `"unrelaxed"` (仅计入非弛豫关联 rdm1 增量，不求解 CP-SCF)。默认为 `"relaxed"`。对 HF/DFT 方法无效 (静默忽略)。
- `multipole_rdm1_dump`：取值布尔类型。计算多极矩后，将总密度矩阵 (SCF 密度 + 关联 rdm1 增量；`"relaxed"` 模式下再对称地加入 0.5 * (Z + Zᵀ) 的 vir-occ/occ-vir 增量) 按 Gaussian fchk 格式追加写入 `{molecule}.fchk` 文件的 `Total MP2 Density` 段。该密度即多极矩计算中所收缩的密度：它与任意对称单电子性质积分的迹给出相应电子贡献。仅对 PT2 族后自洽方法 (纯 MP2 与双杂化) 生效；对 SCF 层级方法静默忽略。默认为 `false`。

作为例子，运行多极矩计算、仅求偶极与四极矩、将原点设为坐标原点、并对后自洽部分使用非弛豫密度，所需的设置如下：
```toml
[ctrl]
analdrv_tasks = "multipole"

[analdrv]
multipole_orders = [1, 2]
multipole_origin = [0.0, 0.0, 0.0]
multipole_rdm1_relax = "unrelaxed"
```

# Detailed description of [output] block in the control file

`[output]` 区块控制 `[ctrl]` 中 `outputs` 列表所请求输出文件的生成方式。区块内所有关键词均有缺省设置，一般计算无需显式声明该区块。目前其关键词影响 fchk 文件 (`outputs = ["fchk"]`) 与 cube 文件 (`outputs = ["cube_orb"]`) 的写出。

- `fchk_dm`：取值String类型。控制写入 fchk 文件的密度矩阵段。可选项：
    - `auto`（缺省）：写入所有已计算的密度。即总是写入 `Total SCF Density`（开壳层 UHF 参考另行写入 `Spin SCF Density`）；双杂化 (PT2 族) 计算若已求得弛豫 MP2 密度（如 `job_type = "force"` 的解析力计算），则额外写入 `Total MP2 Density` 段，开壳层另写 `Spin MP2 Density` 段
    - `scf`：仅写入 SCF 密度段，省略全部 MP2 密度段
    - `false`：不写入任何密度段
- `fchk_writer`：取值String类型。选择写出 fchk 文件中 MO 系数等的写入器。可选项：
    - `default`（缺省）：编译时启用了 cargo feature `librest2fch`（当前为缺省 feature）则使用 MOKIT 的 librest2fch 写出 MO 系数；否则使用 REST 原生的 Rust 写入器
    - `librest2fch`：使用 librest2fch 库。若编译时未启用该 feature，程序报错
    - `rust`：使用 REST 原生 Rust 写入器

一个显式设置的例子：
```toml
[output]
fchk_dm = "scf"
fchk_writer = "rust"
```

- `cube_orb_setting`: 取值[f64;2]。`outputs = ["cube_orb"]` 的格点参数设置。前一个值(margin)是边界信息，第二个值(num_grids)是生成格点的数目。缺省值为[3.0, 80.0]
- `cube_orb_indices`: 取值Vec\<[usize;3]\>。
	- 指定需要生成cube文件的一组轨道。缺省为空，即`[]`
    - 每一个矢量元素`[usize;3]`代表一组轨道信息：
    - 第一个值(start_orb)为起始轨道的index
    - 第二个值(end_orb)为截止轨道的index，与start_orb组成闭区间(closed interval)
    - 第三个值(i_spin)是这组轨道所在的自旋通道。0为alpha自旋;1为beta自旋
    - **注意**：REST的轨道排序从0开始，因此第一个轨道是0。如果HOMO是第X个轨道，在REST的排序是X-1
	- 举例来说：闭壳层基态苯分子体系的HOMO-1、HOMO和LUMO是第20、21和22个轨道，在REST中的排序是19、20和21。打印alpha自旋通道上这三个轨道的设置是[[19,21,0]]
- `cube_orb_type`: 取值String类型。指定生成的cube文件类型：
  - `wavefunction`: 生成轨道波函数的cube文件（缺省）
  - `density`: 生成轨道概率密度的cube文件（|ψ|²）
- **注意**：以上三个cube关键词原先位于 `[ctrl]` 区块。为向后兼容，`[ctrl]` 中的旧位置仍然可用（程序会打印弃用提示），但同一关键词在 `[output]` 中给出时以 `[output]` 为准。

# Detailed description of [geom] block in the control file
- `name`：取值String类型。分子体系的名称
- `unit`：取值String类型。坐标单位。目前支持：angstrom和bohr
- `position`：取值String类型。分子体系的坐标，目前支持xyz格式。支持两种书写方式：
    - **标准格式**（所有原子均可自由弛豫）：`<elem> <x> <y> <z>`
    - **固定原子格式**（constraints）：`<elem> <fix> <x> <y> <z>`，其中`<fix>`取值0或1。`0`表示该原子固定在初始位置，`1`表示该原子允许弛豫
    - **注意**：固定原子功能当前仅在 `opt_engine = "geometric-pyo3"` 构型优化引擎下生效。REST 会自动从 `fix` 信息生成 geometric 约束文件，冻结指定原子的 Cartesian 坐标。约束在内坐标（TRIC/DLC）空间生效，要求 `coordsys ≠ "cart"`（默认 `tric` 满足此要求）。
    - 一个混合格式的例子（H₂O-H₂O二聚体，固定donor水分子的O和一个H，其余自由）：
`position = """
         O  0   0.000000     0.000000     0.000000
         H  0   0.756950     0.585882     0.000000
         H  1  -0.756950     0.585882     0.000000
         O  1   0.000000     2.890000     0.000000
         H  1   0.000000     2.390000     0.000000
         H  1  -0.935300     3.243300     0.000000
"""
`
    - 一个纯标准格式的例子（不使用固定原子功能）：
`position = '''
         N  -2.1988391019      1.8973746268      0.0000000000
         H  -1.1788391019      1.8973746268      0.0000000000
         H  -2.5388353987      1.0925460144     -0.5263586446
         H  -2.5388400276      2.7556271745     -0.4338224694
'''
`
- `ghost`:　取值String类型。每一行对应一个ghost原子、点电荷或者ghost赝势的设置
   - 对于ghost原子，其格式为：`basis set  <elem>  <position>`
   - 对于点电荷，其格式为：`point charge  <charge>  <position>`
   - 对于ghost赝势，其格式为：`potential  <file_name>  <position>`
       - `<elem>`：取值i32类型。设置这一ghost原子的元素类型，以便程序找到正确的基组
       - `<charge>`:　取值f64类型。设置点电荷的电荷量
       - `<file_name>`: 取值String类型。设置赝势的文件名称
       - `<position>`:　取值[f64;3]类型，设置ghost原子，点电荷和赝势对应的xyz坐标
    - 一个例子：
`ghost = '''
        basis set             Cl           0.000000   0.00000    1.6000000
        point charge        -0.3           0.500000   0.60000    0.8000000
        point charge         1.5           0.200000   0.30000    1.1000000
        potential       O_ghost.json       0.500000   0.60000    0.8000000
        potential       Mg_ghost.json      0.200000   0.30000    1.1000000
'''
`
- `ext_field_dipole`：取值 `[f64;3]` 类型。外加偶极电场 (以 `[0., 0., 0.]` 为规范原点) 在 [x, y, z] 三个分量上的强度。单位 a.u.。
    - 一个例子：
        ```
        [geom]
        name = "NH3"
        unit = "Angstrom"
        position = """
            N  0.0  0.0  0.0
            H  0.0  1.5  1.0
            H  1.4  1.1  0.0
            H  1.2  0.0  1.3
        """
        ext_field_dipole = [0.0, 0.1, 0.0]
        ```
### RRS-PBC 方法相关设置
- `rrs_pbc`: 取值为bool。如果设置为true则启动RRS-PBC计算。缺省为false
-  `unit_cell_index`: 取值为`Vec<usize>`。核心晶胞包含的原子在`position`中的序号。缺省为空
- **注意**：原子的序号从0开始
    
  - 一个例子：
```toml
    [geom]
    name = "NH3"
    unit = "Angstrom"
    position = """
        H  0.0  0.0  0.0
        C  1.1  0.0  0.0
        C  2.4  0.0  0.0
        C  3.7  0.0  0.0  # 这个原子属于核心晶胞
        C  5.0  0.0  0.0  # 这个原子也属于核心晶胞
        C  6.3  0.0  0.0
        C  8.0  0.0  0.0
        H  9.1  0.0  0.0
    """
    rrs_pbc = true
    unit_cell_index = [3,4]
```
- `pbc_dim`: 取值为usize。周期性的维度。缺省为1
  
- `rrs_pbc_vec`：取值为`Vec<f64>`。周期性体系的晶格矢量。缺省为空
  - **注意**：每个晶格矢量均为三维，如果周期性的维度不为1，则将所有晶格矢量拼接为一个作为该参数。如果晶格矢量的总长度超过周期性维度对应的长度，将抛出一个警告并忽略多余部分
  - 一个例子：
```toml
    pbc_dim = 2  # 视为二维周期性体系
    rrs_pbc_vec = [1.0,0.0,0.0,0.0,1.0,0.0,5.14]  # 两个晶格矢量分别为[1.0,0.0,0.0]和[0.0,1.0,0.0]，多余的5.14被忽略
```
-  `max_step`：取值为`Vec<String>`。从核心晶胞开始作用晶格矢量的最大次数。缺省为空
   -  **注意**：RRS-PBC算法会从核心晶胞开始按照平移矢量向正负方向各平移至多`max_step`次，匹配元素和位置均符合的晶胞，并跳过不符合的。如果长度超过周期性维度，将忽略多余部分。匹配结果会在`print_level`至少为1时输出，例如：
```
    For step path = (-1, 0, 0), found target index = [0, 1, 2, 3]
    For step path = (0, 0, 0), found target index = [4，5，6，7]
    For step path = (1, 0, 0), found target index = [8，9，10，11] 
```
- `k_points`: 取值为`Vec<usize>`。每个周期性维度下的k点数量。缺省为空
   - **注意**：k点的选取方法为在倒格矢和倒格矢的反向之间均匀分布，并包含两侧边界。为了确保均匀分布的k点能够覆盖高对称点，推荐将数量设置的大一些


# Hessian 矩阵、振动频率与热化学分析相关设置

REST 提供两条独立的频率/热化学计算路径，请勿混淆：

- **路径一（推荐）：REST 原生解析 Hessian + 热化学**。通过 `[hessian]` 和 `[thermo]` 区块触发。`[hessian]` 进行全解析的 RI-RHF/RKS Hessian 计算（含 CP-HF 响应），`[thermo]` 基于所得谐振频率做理想气体刚体转子-谐振子（RRHO）/ quasi-RRHO 热化学分析。本节即介绍此路径。
- **路径二：geomeTRIC 数值 Hessian**。在结构优化流程中由 `[geometric_pyo3]` 区块的 `hessian`/`frequency`/`thermo` 关键词控制（见下文章节），频率与热化学交由 Python 端 geomeTRIC 完成，REST 仅提供数值 Hessian。

> 注意两条路径下热化学的压强单位不同：`[thermo]` 区块用 **atm**；`[geometric_pyo3]` 的 `thermo` 关键词用 **bar**。
> 转动对称数 σ 可由用户手动指定，也可设 `symmetry_number = 0` 让程序自动识别点群（见下文「点群自动识别」）。

## `[hessian]` 区块关键词

只要输入卡中存在 `[hessian]` 区块（可位于顶层或嵌套于 `[ctrl]` 下），SCF 收敛后即**无条件**触发解析 Hessian 计算（与 `job_type` 无关）。关键词包括：

- `solver`: 取值 String。CP-HF 求解器，可选 `"krylov"`（缺省）或 `"dense"`。
- `frequencies`: 取值 bool。是否在 Hessian 计算后顺带做振动频率与简正模分析并输出 `EigenModes.txt`。缺省 false。注意：若同时设置了 `[thermo]` 区块，频率会被自动计算，无需手动开启此项。
- `krylov_max_cycle`: 取值 usize。Krylov 求解器最大迭代次数，缺省 50。
- `krylov_tol`: 取值 f64。Krylov 收敛阈值，缺省 1e-9。
- `krylov_lindep`: 取值 f64。Krylov 线性相关阈值，缺省 1e-15。
- `krylov_tol_inflation`: 取值 f64。真残差容忍系数，缺省 1000.0。
- `verbose`: 取值 usize。输出详细程度（0=静默，1=正常，2=调试；调试时额外输出各分量 .npy 文件）。缺省 1。
- `hessian_matrix_path`: 取值 String。Hessian 矩阵输出路径。缺省 `"./HessianMatrix.txt"`。
- `eigenmodes_path`: 取值 String。简正模输出路径。缺省 `"./EigenModes.txt"`。

频率计算约定：对 Hessian 做质量加权后经 LAPACK `dsyev` 对角化，本征值转换为 cm⁻¹（虚频记为负值）。6（线性分子为 5）个平动/转动零模会自然产生（数值上接近 0；若几何未完全优化可能呈小幅虚频，不影响真实振动模式）。REST 内置质量为元素质量（如 H=1.008、C=12.011），与 Gaussian 默认同位素质量略有差异，对频率与热化学量的影响通常在 0.1% 量级。

## `[thermo]` 区块关键词

只要输入卡中存在 `[thermo]` 区块（顶层或 `[ctrl]` 下），在（解析）Hessian + 频率计算完成后即自动进行理想气体热化学分析。所有热力学量基于 RRHO 模型，按平动、转动、振动、电子四部分分别计算并求和（T. Lu, *Comput. Theor. Chem.* 1200, 113249, 2021）。关键词包括：

- `temperature`（或 `T`）：温度 (K)。可设为单值（缺省 298.15），或设为扫描区间 `[下限, 上限, 步长]`（见下文"温度/压强扫描"）。
- `pressure`（或 `P`）：压强 (**atm**)。可设为单值（缺省 1.0）或扫描区间。压强仅影响平动熵。
- `symmetry_number`（或 `sigma`）：转动对称数 σ。可手动指定（参考值：C1/Ci/Cs/C∞v→1，Cn/Cnv/Cnh→n，D∞h→2，Dn/Dnh/Dnd→2n，Sn→n/2，Td/T→12，Oh→24，Ih→60）。**设为 `0` 则启用点群自动识别**（见下文「点群自动识别」）。缺省 0 即自动识别点群。
- `electronic_energy`（或 `E`）：用于 U/H/G 求和的电子能量 (a.u.)。缺省 0.0 表示使用当前 SCF 总能量。若想在频率分析级别之上采用更高级别单点能，可在此指定。
- 频率标度因子（四项可独立设置，缺省均为 1.0）：
    - `sclzpe`：用于零点能 ZPE。
    - `sclheat`：用于 U(T)−U(0) 升温贡献。
    - `scls`：用于熵 S。
    - `sclcv`：用于热容 CV/CP。
    - 便捷别名：`scale_factor`（或 `scl`）若设置则同时覆盖以上四项（便于简单使用）。
- `ilowfreq`：低频处理模型。缺省 0。
    - `0`：标准 RRHO（谐振子近似）。
    - `1`：Truhlar 模型——把低于 `ravib` 的频率提升到 `ravib`（仅影响 S、CV、U0→T，不影响 ZPE）。
    - `2`：Grimme quasi-RRHO——熵在谐振子与自由转子间插值（仅影响熵）。参考 *Chem. Eur. J.* 18, 9955 (2012)。
    - `3`：Minenkov quasi-RRHO——熵与内能均在谐振子与自由转子间插值（**推荐用于柔性大分子**）。参考 *J. Comput. Chem.* 44, 1807 (2023)。
- `ravib`：Truhlar 提频阈值 (cm⁻¹)。缺省 100.0。
- `intpvib`：Grimme/Minenkov 插值的特征频率阈值 (cm⁻¹)。缺省 100.0（50–150 之间结果相近）。
- `imagreal`：把绝对值小于此值的虚频当作实频处理 (cm⁻¹)。缺省 0.0（禁用）。仅在 `ilowfreq ≠ 0` 时有意义，常取 50–100。
- `conc`：浓度校正。形如 `"1.5M"`（mol/L）或 `"2.3atm"`，计算 ΔG = RT ln(cB/cA)，其中 cA 由当前 T、P 按理想气体给出。缺省 `""`/`"0"`（不做校正）。仅单点模式生效，扫描时不生效。
- `output_path`：单点热化学报告输出路径。缺省 `"./Thermochemistry.txt"`。

> quasi-RRHO 插值公式（Grimme/Minenkov）：权重 `w(ν) = 1/[1+(ν₀/ν)^4]`，`ν` 取未标度频率；熵 `S = w·S_RRHO + (1−w)·S_FR`；自由转子熵基于有效转动惯量 μ' = μ·Bav/(μ+Bav)，其中 μ = h/(8π²ν)、Bav = 10⁻⁴⁴ kg·m²。Minenkov 模型进一步对内能做 `U = w·U_RRHO + (1−w)·(RT/2)`。

### 点群自动识别
将 `symmetry_number` 设为 `0` 即启用点群自动识别：程序从分子几何出发，搜索所有真转动对称操作（Cn^k，含恒等操作 E），其总数即为转动对称数 σ（σ = 真转动子群的阶）。同时给出 Schoenflies 点群标签（如 `Td`、`C2v`、`D6h`、`D∞h`，尽力而为）。候选转轴来自原子位置、同种元素原子对连线、以及主转动惯量轴；每个候选轴测试 C2–C8。线性分子（一个主转动惯量为零）按 C∞v（σ=1）或 D∞h（σ=2，含反演中心）处理。
- 优点：σ 严格正确（直接数真转动操作的个数），对 CH₄ 自动给出 Td/σ=12、H₂O 给出 C2v/σ=2、CO₂ 给出 D∞h/σ=2。
- 限制：点群标签对极少数纯 Sn 群可能显示为 C_{n/2}（但 σ 仍正确）；要求几何接近对称（优化良好的结构默认容差 1e-2 Bohr 即可，必要时可手工指定 σ）。

### 温度/压强扫描
将 `temperature` 或 `pressure` 设为三元素数组 `[下限, 上限, 步长]` 即开启扫描。程序对 (T, P) 笛卡尔积逐点计算，并输出两个文件：
- `scan_SCq.txt`：S、CV、CP（cal/mol·K）及 q(V=0)/NA、q(bot)/NA 随 T、P 变化。
- `scan_UHG.txt`：Ucorr、Hcorr、Gcorr（kcal/mol）及 U、H、G（a.u.）随 T、P 变化。

### 输出说明
单点模式下，屏幕与 `Thermochemistry.txt` 同时输出：分子量、主转动惯量（amu·Bohr²）、转动常数（GHz）、转动温度（K）、线性/非线性判定、σ、采用的实频数目；各部分（平动/转动/振动/电子）对 U、S、CV 的贡献；ZPE、U_corr、H_corr、G_corr（同时给出 kJ/mol、kcal/mol、a.u.）；以及 E + ZPE / E + U_corr / E + H_corr / E + G_corr 的求和。配分函数 q_trans、q_rot、q_vib(V=0)、q_vib(bot)、q_ele 亦一并给出。

## 热化学配置示例
- 例子一：B3LYP/cc-pVDZ 下 CH₄ 的解析 Hessian + 频率 + RRHO 热化学（σ=12，Td 球陀螺）
    ```toml
    [ctrl]
         xc = "b3lyp"
         basis_path = "cc-pVDZ"
         auxbas_path = "def2-universal-jkfit"
         charge = 0.0
         spin = 1.0
         spin_polarization = false
    [geom]
         name = "CH4"
         unit = "Angstrom"
         position = """
            C  0.000000  0.000000  0.000000
            H  0.629118  0.629118  0.629118
            H -0.629118 -0.629118  0.629118
            H  0.629118 -0.629118 -0.629118
            H -0.629118  0.629118 -0.629118
         """
    [hessian]
         frequencies = true
    [thermo]
         temperature = 298.15
         pressure = 1.0
         symmetry_number = 0       # 0 = 自动识别点群（CH4 → Td, σ=12）
         ilowfreq = 0
    ```
- 例子二：柔性大分子用 Grimme quasi-RRHO 处理低频，并扫描温度
    ```toml
    [thermo]
         temperature = [300.0, 800.0, 50.0]
         pressure = 1.0
         symmetry_number = 1.0
         ilowfreq = 2            # Grimme quasi-RRHO
         sclzpe = 0.9806
         imagreal = 50
    ```
- 例子三：使用更高级别单点能 + 1 M 浓度校正（C2v 分子，σ=2）
    ```toml
    [thermo]
         temperature = 298.15
         pressure = 1.0
         symmetry_number = 2.0
         electronic_energy = -114.5521
         conc = "1.0M"
    ```

# Detailed descrption of [geometric_pyo3] block in the control file
- `maxiter`：取值i32。结构优化的最大步数上限。缺省值：300
- `converge_energy`：取值f64。构型优化中上下两步能量变化的收敛阈值。缺省值：1.0e-6
- `converge_grms`：取值f64。梯度的收敛阈值。缺省值：3.0e-4
- `converge_gmax`：取值f64。最大梯度的收敛阈值。缺省值：4.5e-4
- `converge_drms`：取值f64。构型优化中上下两步构型变化的收敛阈值。缺省值：1.2e-3
- `converge_dmax`：取值f64。最大构型变化的收敛阈值。缺省值：1.8e-3
- `coordsys`：取值String。坐标系统设置。缺省值："tric"。如果有其他需求见geomeTRIC的官方说明：https://geometric.readthedocs.io/en/latest/
    - 可选项：`"tric"`（缺省，平动-转动内坐标，适合弱键复合物/团簇）、`"dlc"`（标准离域内坐标，按共价键连接性生成）、`"hdlc"`（混合 DLC，每原子补 Cartesian）、`"cart"`（纯笛卡尔）、`"prim"`/`"tric-p"`（不离域的原始内坐标）
    - **约束优化（固定原子）仅支持 DLC 类坐标**（`tric`/`dlc`/`hdlc`）；`cart`/`prim`/`tric-p` 用约束时 geomeTRIC 会报错
- `fac`：取值f64（可选）。geomeTRIC 成键判据中共价半径的乘性因子（`build_topology` 的 `Fac` 参数）。原子间距离 < `fac` ×（共价半径和）时判为成键，键图决定后续生成的内坐标（键长/键角/二面）数目。缺省值：不设置时沿用 geomeTRIC 缺省 1.2。
    - 对密堆积或含大阳离子的体系（如 SrTiO₃ 团簇、金属有机框架），缺省 1.2 可能将离子接触（如 Sr–O）误判为共价键，生成过多冗余内坐标；此时可降到 **0.9–1.0** 以减少内坐标数目
    - 取值过小（如 0.8）会漏掉真实共价键，需根据体系经验性调整；普通有机/小分子体系一般无需设置
- `radii`：取值为内联表（可选），形如 `radii = {"Sr" = 1.0, "Na" = 0.0}`。逐元素覆盖共价半径（geomeTRIC `--radii` 选项），用于精细控制成键拓扑。
    - 设某元素半径为 `0.0` 可使该元素不与任何原子成键（例如把离子型阳离子 Sr、Na 从内坐标体系中排除），比全局 `fac` 更精准
    - geomeTRIC 内部有 `mindist = 1.0` Å 的键长阈值下限，因此将短键（< 1.0 Å，如 O–H）的半径调小无效；本选项主要影响长键/离子接触（> 1.0 Å）
- `check`：取值i32（可选）。每隔指定步数重建内坐标体系（geomeTRIC `--check` 选项）。缺省值：不设置时为 0（关闭）。
    - 随着优化进行、原子移动，DLC 离域基会逐渐与当前几何失配，导致 geomeTRIC 报告的 `Grad_T` 与真实力偏差越来越大。设置 `check`（如 `10`）可定期刷新内坐标体系，保持 `Grad_T` 的可靠性
    - 代价是每次重建会损失部分 BFGS Hessian 历史，可能略微增加收敛步数
- `transition`：取值bool，设置为true则开启过渡态搜索。缺省值为：false（对应于稳态搜索）
- `hessian`：取值String。决定是否以及何时进行Hessian矩阵计算。**数值 Hessian**（由 geomeTRIC 通过有限差分梯度计算）和 **REST 解析 Hessian**（由 `analytic_hessian` 开关控制）二选一。
    - "never"：不做Hessian矩阵计算（缺省：稳态搜索）
    - "first"：只对初始结构计算Hessian矩阵（缺省：过渡态搜索）
	- "last"：计算优化好的结构的Hessian矩阵计算
	- "first+last"：计算初始和优化好的两个结构的Hessian矩阵
	- "stop"：不做构型优化，只计算初始结构的Hessian矩阵
	- "each"：计算构型优化中每一步的Hessian矩阵
- `analytic_hessian`：取值 bool。若设为 `true`，则在调用 geomeTRIC 优化前先用 REST 的解析 Hessian 模块计算结果并注入 geomeTRIC，**完全避免 geomeTRIC 的数值有限差分 Hessian 计算**。解析 Hessian 含 CP-HF 轨道弛豫贡献，精度远优于有限差分，且后续 BFGS 更新不受影响。缺省值：`false`。建议在过渡态搜索（`transition = true`）或 IRC 追踪（`irc = true`）中启用。
- `use_analdrv`：取值 bool。若设为 `true`，使用 `analdrv` 模块进行 Hessian 计算；若设为 `false`，使用 `hessian` 模块进行 Hessian 计算。缺省值：`None`，如果输入卡检测到 `[hessian]` 区块会使用 `hessian` 模块，否则使用 `analdrv` 模块。
- `frequency`：取值bool，当得到Hessian矩阵后，是否开展频率计算和热化学分析。缺省值：true
- `thermo`：取值[f64;2]，提供热力学分析的状态：[温度 (K),压强 (bar)]。缺省值：[300.0, 1.0]
- `reset`：取值bool。当近似 Hessian 的特征值低于 `epsilon` 阈值时，是否将其重置回 guess Hessian。对于稳态优化，缺省值为 true。若体系梯度含噪声、BFGS 更新每步失败（出现 "Eigenvalues below ... returning guess"），可设为 false 保留 Hessian 并加对角 shift 继续优化。
- `trust`：取值f64。初始 trust radius（Å）。缺省值：0.1
- `tmax`：取值f64。最大 trust radius（Å）。缺省值：0.3
- `tmin`：取值f64。最小 trust radius（Å）。缺省值：1e-4。一般应小于 `convergence_drms` 以避免优化器在数值噪声处提前收敛。
- `epsilon`：取值f64。Hessian 重置的特征值阈值。缺省值：1e-5。仅当 `reset = true` 时生效。
- `subfrctor`：取值 i32。投影掉梯度中净力/净力矩分量的模式。0 = 不投影，1 = 自动检测（缺省），2 = 强制投影。DFT 梯度常含微量力矩噪声，在 QM/MM 或大体系中可能引起结构慢转而力不收敛，设为 2 可消除此噪声源。
- `usedmax`：取值 bool。是否用最大位移分量（而非 RMS）判断 trust radius。缺省值：false。适合各方向力常数差异大的各向异性体系。
## 配置示例 
- 例子一：开启GGA、meta-GGA或者杂化泛函的稳态构型优化（以x3lyp为例），则不需要使用[geometric_pyo3]区的设置
    ```toml
	[ctrl]
         xc = x3lyp
         job_type =                  "opt"
         opt_engine =                "geometric-pyo3"
         xc =                        "x3lyp"
         empirical_dispersion =      "d3bj"
         basis_path =                "cc-pVDZ"
         auxbas_path =               "def2-universal-jkfit"
         charge =                    0.0
         spin =                      1.0
         spin_polarization =         false
    
    [geom]
         name = "CO"
         unit = "angstrom"
         position = """
            C  0.00000000000      0.0000000000      0.0000000000
            O  1.20000000000      0.0000000000      0.0000000000
        """ 
    ```
- 例子二：如果使用双杂化泛函等没有解析力的方法，则需要在[ctrl]区开启数值力的计算功能（仅展示与上个例子不同的设置）
    ```toml
	[ctrl]
	    xc = xyg3
		numerical_force = true
	```
- 例子三：开启过渡态优化和频率计算，并且设置非常规状态（398开尔文、1.5个大气压）
    ```toml
	[geometric_pyo3]
	    transition = true
		hessian = "first+last"
		thermo = [398.0, 1.5]
	```
- 例子四：开启过渡态优化，并用 REST 解析 Hessian 替代 geomeTRIC 数值有限差分 Hessian
    ```toml
	[geometric_pyo3]
	    transition = true
		analytic_hessian = true
	```
    - 启用后，REST 在初始结构上计算解析 Hessian（含 CP-HF 轨道弛豫），写入临时文件并注入 geomeTRIC。geomeTRIC 读取后用于第一步优化方向，后续所有步正常走 BFGS 更新，整个过程不产生任何数值有限差分计算。
- 例子五：密堆积体系（如 SrTiO₃ 表面团簇）的约束优化——收紧成键判据并定期刷新内坐标
    ```toml
	[geometric_pyo3]
	    coordsys = "tric"
	    fac = 0.9
	    radii = {"Sr" = 1.0}
	    check = 10
	    maxiter = 300
	```
    - `fac = 0.9` 全局收紧成键判据；`radii = {"Sr" = 1.0}` 进一步把 Sr 的共价半径从 1.95 降到 1.0，精确去除 Sr–O 离子接触（两者可单独或组合使用）
    - `check = 10` 每 10 步重建内坐标体系，防止 DLC 离域基随几何变化而陈旧化

# 内置分子动力学（MD / AIMD）

`job_type = "md"` 时 REST 将启动一套内置的分子动力学引擎（MD 循环、积分器、伞形采样都在 REST 内，每一步的 QM 能量/梯度走与几何优化相同的进程内接口）：

- **AIMD（纯 QM）**：体系只有 `[geom]` 的 QM 分子，每步先收敛 SCF 再移动原子核并取解析梯度；上一步的收敛波函数自动作为下一步初猜（内部设为 `initial_guess = "inherit"`，用户无需配置，显著减少迭代），SCF 用多线程（`num_threads`）并行。
- **QM/MM**：`[geom]` 为 QM 区，MM 体系可用内嵌 OpenMM 提供（TIP3P 自动类型，或直接给完整的 OpenMM System XML），也可由 REST 从经典力场参数**内部生成** System XML（支持共价切断的 link-H）；QM–MM 之间为点电荷嵌入 + QM–MM LJ。
- **纯 MM**：`pure_mm = true` 时整个体系交给 OpenMM（`mm_file` / `mm_system_xml`）。
- **任务类型**：`ensemble = "nvt"`(Langevin) / `"nve"`(VelocityVerlet) / `"opt"`(几何优化) / `"sp"`(单点能量/受力)。
- **伞形采样（US）**：`umbrella_atoms` 存在即启用；CV 支持 `dihedral` / `distance` / `angle` / `distance_diff`（后者可叠加 `umbrella_sum_kappa` 的 ½ks(s−s0)² 约束），输出 `umbrella_timeseries.csv` 供 (2D) WHAM 后处理。
- **续跑**：`restart_input` 从上次的 `md_restart` 同时读位置与速度，无需改写 `[geom]`。
- **输出**：`md.log`、`dump_{nvt|nve}.xyz`、`dipole.dat`、`energy_force.log`、`umbrella_timeseries.csv`、`md_restart`。
- **Python 依赖**：MD/AIMD 的积分与优化由内嵌 Python 的 [ASE](https://wiki.fysik.dtu.dk/ase/) 完成，QM/MM 与纯 MM 的 MM 侧由 [OpenMM](https://openmm.org/) 完成；运行时需要有可用的 Python 及 `numpy`、`ase`、`openmm`（含 `openmm.app`）。参考环境：Python 3.10、numpy 2.2、ASE 3.29、OpenMM 8.6。MD 的参考轨迹需要 `ase >= 3.25`（3.25 起 Langevin 的 `fixcm` 投影才与参考日志一致，更旧的版本同一个 seed 走出不同轨迹），启动时日志会打印内嵌解释器与 ASE 的实际版本。

纯 QM 的 AIMD 支持多进程 MPI（SCF 与积分沿 MPI 并行，所有 rank 同步积分、仅 root 落盘）。QM/MM 与纯 MM 目前只支持单进程：进程内用 OpenMP/Rayon 多线程（`num_threads`），多进程 MPI 会在启动时被拒绝。

详情请见 [REST 文档 · 内置分子动力学](https://rest-doc.readthedocs.io/zh_CN/user/md.html)。
