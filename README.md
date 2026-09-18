# Physarum: Morphogenesis & Transport Network Engine

[English](#english-documentation) | [中文说明](#chinese-documentation)

A high-performance research pipeline and procedural generative system simulating the morphogenesis, foraging dynamics, and adaptive cytoplasmic transport networks of *Physarum polycephalum* (true slime mold). 

Implemented in Python with GPU acceleration via the Taichi compute infrastructure, targeting CUDA and Vulkan backends. The repository chronicles the technical lineage from classical agent-continuum heuristics (Jeff Jones, 2010) to topological space colonization graphs (Runions, 2005), GPGPU dual-channel frame-buffer delay blending, and multi-species antagonistic 2.5D normal-mapped photometrics.

---

<a id="english-documentation"></a>
## English Documentation

### Architectural Duality & Core Design Philosophy

Simulating the morphology of *Physarum polycephalum* requires addressing a central biomechanical paradox: **local protoplasmic fluid continuity versus macroscopic transport network pruning**. Generative implementations generally bifurcate into two distinct computational paradigms:

1. **Continuum Multi-Agent Reaction-Diffusion (Jones-Lague-Yuen Pipeline)**:
   A decentralized swarm where hundreds of thousands to millions of dimensionless agents deposit and sample an information field. Network conduits emerge purely through non-linear positive feedback between trail accumulation, spatial diffusion, and exponential decay.
2. **Discrete Topological Graph Colonization (Runions-Tero Biomechanical Engine)**:
   A graph-theoretic approach using discrete attractor point distributions to iteratively guide iterative venation tree growth, followed by retrograde hydraulic flow reinforcement and Hagen-Poiseuille vessel diameter pruning.

This repository implements, benchmark-tests, and preserves both paradigms across sequential milestone tags (`v1.0` through `v3.1`, up to `v5.0`).

```
                              [ Morphogenetic Problem Space ]
                                             │
             ┌───────────────────────────────┴──────────────────────────────┐
             ▼                                                              ▼
   [ Branch A: Continuum Field ]                                [ Branch B: Discrete Graph ]
   (Jones 2010 / Lague / Yuen)                                 (Runions 2005 / Tero 2010)
             │                                                              │
             ├─► v1.0: Dense Swarm + Shuttle Streaming                      ├─► v3.0: Pure Space Colonization (SCA)
             ├─► v2.0: Generational Tip-Growth (ianpilon)                   └─► v3.1: Spatial Hashing + Retrograde
             ├─► v4.0: Dual-Channel Delay Blending (nicoptere)                         Pruning + On-Screen GGUI HUD
             └─► v5.0: Antagonistic Species + 2.5D Normal (fogleman)
```

---

### Milestone Specifications & Version Lineage

#### `v1.0` - Baseline Dense Swarm & Shuttle Streaming
* **Commit**: `2af9244`
* **Foundational Architecture**: Continuous scalar field coupled with 1.2M discrete active agents.
* **Mathematical Stencil**:
  * Agent heading update via discrete three-point differential sensing:
    $$\theta_{t+1} = \theta_t + \Delta\theta \cdot \text{sign}(S_R - S_L)$$
    where forward offset sensors ($S_L, S_C, S_R$) sample the trail intensity at offset distance $SO = 22.5^\circ$, sensor angle $SA = 45.0^\circ$, and distance $SS = 3.5\text{ px}$.
  * Cytoplasmic shuttle streaming: Macro-scale rhythmic contraction modulated via low-frequency sinusoidal velocity fields:
    $$v(t) = v_0 \cdot \left[1.0 + 0.3 \sin\left(\omega t - \vec{k} \cdot \vec{x}\right)\right]$$
  * Multiplicative field decay ($\gamma = 0.94$) over an isotropic $3 \times 3$ discrete convolution kernel.
* **Failure Mode Analysis**: In the absence of competitive boundaries or lateral inhibition, high-density swarms collapse into static, sponge-like isotropic mazes and exhibit severe wavefront interference upon colliding with rigid boundaries.

#### `v2.0` - Generational Tip-Growth & Archival Paper Shader
* **Commit**: `d48f330`
* **Foundational Architecture**: Explicit generational recursion with finite lifespan metadata, derived from botanical apical meristem expansion.
* **Mechanics**:
  * Each active agent tracks recursion generation $g \in [0, 6]$ and remaining metabolic ticks $L_i$.
  * Lateral bifurcation follows an accelerated probability distribution:
    $$P_{\text{branch}} = P_0 \cdot \alpha_{\text{acc}}^g$$
  * Distal daughters inherit attenuated deposition potency:
    $$D_{\text{child}} = D_{\text{parent}} \cdot V_{\text{tap}}$$
* **Renderer**: Botanical archival specimen shader. Evaluates four-octave Fractional Brownian Motion (FBM) to simulate heterogeneous liquid ink absorption over porous paper substrates:
  $$\text{Color} = \text{lerp}\left(\vec{C}_{\text{paper}}, \vec{C}_{\text{ink}}, \text{Trail}^\gamma\right) + \text{Grain}(\vec{x}, t)$$
* **Failure Mode Analysis**: Exponential bud proliferation triggers GPU thread serialization and thread-divergence bottlenecks around atomic allocation counters (`atomic_add`).

#### `v3.0` - Pure Space Colonization Vascular Graph (SCA)
* **Commit**: `a8923be`
* **Foundational Architecture**: Complete transition from continuum grid Eulerian tracking to discrete Lagrangian graph growth based on the Space Colonization Algorithm.
* **Mechanics**:
  * 12,000 discrete chemoattractant markers $\vec{A}_k$ distributed pseudo-randomly across the substrate.
  * Influence cone search: Each active marker pulls the nearest vascular graph node $\vec{N}_j$ within search radius $D_{\text{attr}} = 38.0\text{ px}$.
  * Normalized growth vector accumulation:
    $$\vec{v}_{\text{grow}} = \text{normalize}\left(\sum_{k} \frac{\vec{A}_k - \vec{N}_j}{\Vert{}\vec{A}_k - \vec{N}_j\Vert{}}\right)$$
  * Consumption threshold: Attractors undergo metabolic depletion and deletion when $\Vert{}\vec{A}_k - \vec{N}_j\Vert{} < D_{\text{kill}} = 7.5\text{ px}$.
* **Structural Result**: Eradicates sponge-maze collapse. Guarantees hierarchical trunk-to-capillary diameter scaling while maintaining organic topological spacing.

#### `v3.1` - Spatial-Hashed Optimization, Retrograde Pruning & Interactive HUD
* **Commit**: `7bda3a1`
* **Foundational Architecture**: High-performance optimization of the SCA vascular engine with Poiseuille-inspired retrograde pruning, spatial hash partitioning, and Taichi GGUI instrumentation.
* **Key Enhancements**:
  * **GPU Spatial Hashing Acceleration**: Partitions the grid into spatial bins of size $40 \times 40\text{ px}$, reducing the nearest-node search from brute-force $\mathcal{O}(N_{\text{attrs}} \cdot N_{\text{nodes}})$ down to localized adjacent cell sweeps (`MAX_NODES_PER_CELL = 128`), maintaining 60 FPS on mid-range hardware.
  * **Retrograde Flow & Poiseuille Pruning**: Upon physical contact between an exploration tip and a nutrient site, a recursive backward traversal pumps transport flux upstream through parent indices:
    $$\text{node\_flow}_{\text{parent}} \mathrel{+}= \Delta \Phi$$
    Vessels with stagnant flow ($\text{node\_flow} < 0.16$) suffer metabolic penalty ($\text{vitality} \mathrel{-}= \delta_v$), dissolving and clearing non-transporting search paths within 3\~5 seconds.
  * **Real-time On-Screen HUD**: Integrated Dear ImGui overlay rendering node metrics, memory saturation, and interactive state indicators.
  * **Inoculation Guidance**: Dynamic breathing visual feedback indicating pre-inoculation placement.

#### `v4.0` - Dual-Channel Delayed Blending & Torus Topology
* **Commit**: `e4941c2`
* **Foundational Architecture**: Translation of Nicolas Barradeau’s (nicoptere) WebGL GPGPU dual-texture ping-pong framework.
* **Mechanics**:
  * **R Channel (Instantaneous Impulse)**: Binary rasterization of instantaneous agent positions.
  * **G Channel (Morphogenetic Memory)**: Spatial temporal decay buffer.
  * Asymmetric convolution kernel:
    $$G_{t+1}(\vec{x}) = \gamma \sum_{\vec{\delta}} K(\vec{\delta}) \left[R_t(\vec{x} + \vec{\delta}) + 0.5 \cdot G_t(\vec{x} + \vec{\delta})\right]$$
  * Torus periodic boundary topology: Evaluated via floating-point fractional arithmetic:
    $$\vec{x}_{\text{wrapped}} = \text{fract}(\vec{x})$$
* **Structural Result**: Complete removal of boundary accumulation. The asymmetric $1.0 : 0.5$ kernel weight preserves sharp leading search fronts while leaving smooth, fluid conduits behind.

#### `v5.0` - Multi-Species Antagonism & 2.5D Height-Field Normal Mapping
* **Commit**: `86e1294`
* **Foundational Architecture**: Cross-species affinity tensor coupled with photometric normal-bump reconstruction.
* **Mechanics**:
  * **Species Interaction Tensor ($2 \times 2$)**:
    $$M = \begin{pmatrix} +1.0 & -0.4 \\ -0.4 & +1.0 \end{pmatrix}$$
    Positive diagonal entries dictate intra-species cohesion; negative off-diagonal entries generate inter-species shear forces that tear open membrane lacunae and maintain crisp territorial boundaries.
  * **2.5D Microscopic Normal Shading**:
    Treats cumulative matter $H(\vec{x}) = \sum_c \text{grid}_c(\vec{x})$ as a continuous surface height-field. Computes surface normal vectors using finite-difference Sobel operators:
    $$\vec{N} = \text{normalize}\left(-k \cdot \partial_x H, -k \cdot \partial_y H, 1.0\right)$$
    $$\text{Shading} = I_{\text{ambient}} + I_{\text{diffuse}} \max\left(0, \vec{N} \cdot \vec{L}\right) + I_{\text{spec}} \left(\vec{N} \cdot \vec{H}\right)^\alpha$$
* **Visual Result**: Substitutes flat, luminous lines with three-dimensional, translucent organic cord networks illuminated by directional lighting.

---

### Comparative Architecture Matrix

| Version Tag | Computational Core | State Complexity | Boundary Model | Topological Behavior | Primary Aesthetic |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **v1.0** | Agent-Field PDE | `O(N_agents)` | Circular Rigid Reflection | Isotropic Sponge Reticulation | Dual-Scale Optical Density |
| **v2.0** | Recursive Tree | `O(N_active)` | Bounding Box Clamp | Hierarchical Apical Budding | Botanical Archival Ink & Paper |
| **v3.0** | Discrete Attractor | `O(N_nodes * N_attrs)` | Semi-Infinite Substrate | True Venation (Zero Sponge) | Structural Filament Density |
| **v3.1** | Spatial-Hashed Graph | `O(N_local_bucket)` | Circular Hydrodynamic Lip | Dynamic Retraction & Pruning | Specular Microscope Micro-Relief |
| **v4.0** | Dual-Channel Ping-Pong | `O(N_agents + Res^2)` | Toroidal Periodic Wrap | High-Reynolds Viscous Cords | Grayscale Temporal Phase Buffer |
| **v5.0** | Multi-Vector Field | `O(S * N_agents + Res^2)` | Toroidal Periodic Wrap | Competitive Pores & Lacunae | 2.5D Photometric Electron Relief |

---

<a id="chinese-documentation"></a>
## 中文说明

### 架构全景与双重范式解析

多头绒泡菌（*Physarum polycephalum*）形态发生的数学建模，本质上是在解决**局域流体连续性与宏观脉管系统拓扑剪枝**之间的矛盾。在计算生成设计领域，主要存在两条截然不同的技术路线：

1. **连续场多智能体模型（Jones-Lague-Yuen 体系）**：
   无中心化的纯粒子流驱动。数十万至数百万粒子在二维平面上根据局部嗅探到的化学势能浓度差进行偏转，通过信息素轨迹分泌、空间扩散与衰减的非线性正反馈，自发涌现出脉管结构。
2. **离散空间拓扑殖民图（Runions-Tero 生物输运体系）**：
   图论几何模型。通过在培养基离散播撒养分吸引子，牵引节点迭代生成树状网络，结合反向水力通量回溯与泊肃叶定律进行管道口径缩放与盲端剪枝。

本项目完整推演、验证并沉淀了上述两套体系，在 Git Tags 中以 `v1.0` 至 `v3.1` 以及 `v4.0`、`v5.0` 形成严密的科研演进链条。

```
                             [ 黏菌形态发生计算流 ]
                                       │
             ┌─────────────────────────┴─────────────────────────┐
             ▼                                                   ▼
     [ 分支甲：连续相场粒子流 ]                           [ 分支乙：离散拓扑图网络 ]
   (Jones 2010 / Lague / Yuen)                          (Runions 2005 / Tero 2010)
             │                                                   │
             ├─► v1.0: 密集粒子流与原生质穿梭流                  ├─► v3.0: 纯空间殖民图算法 (SCA)
             ├─► v2.0: 显式代际顶芽分叉 (ianpilon)               └─► v3.1: 空间哈希加速 + 逆行通量
             ├─► v4.0: 双通道延迟混合 (nicoptere)                          剪枝代谢 + 原生 GGUI 仪表盘
             └─► v5.0: 异种拮抗矩阵与 2.5D 法线 (fogleman)
```

---

### 版本演进编年史与核心算子

#### `v1.0` - 基础高密粒子流与穿梭流 (Baseline Dense Swarm)
* **对应 Commit**: `2af9244`
* **计算架构**：120 万高密度活动质点与欧拉连续网格场耦合。
* **核心数学算子**：
  * 三向离散差分偏转决策：
    $$\theta_{t+1} = \theta_t + \Delta\theta \cdot \text{sign}(S_R - S_L)$$
    前置三向探针偏移角 $SO = 22.5^\circ$，感知半角 $SA = 45.0^\circ$，前探步距 $SS = 3.5\text{ px}$。
  * 原生质往复穿梭流（Shuttle Streaming）：低频行波调制运动步长与分泌通量：
    $$v(t) = v_0 \cdot \left[1.0 + 0.3 \sin\left(\omega t - \vec{k} \cdot \vec{x}\right)\right]$$
  * $3 \times 3$ 离散高斯平滑核卷积与乘法挥发（衰减率 $\gamma = 0.94$）。
* **失效模式分析**：系统缺乏宏观侧向抑制机制，演化后期质点在培养皿边缘发生驻波干涉，退化为均质海绵网状结构及同心圆环伪影。

#### `v2.0` - 显式代际顶端分叉与标本着色 (Generational Tip-Growth)
* **对应 Commit**: `d48f330`
* **计算架构**：顶端出芽机制（Tip-Growth），粒子具有显式生命周期与代际元数据。
* **动力学机制**：
  * 粒子携带代际深度 $g \in [0, 6]$ 与存活寿命计数器 $L_i$。
  * 侧向出芽分叉概率随代际指数递减：
    $$P_{\text{branch}} = P_0 \cdot \alpha_{\text{acc}}^g$$
  * 子代分支继承按衰减系数 $V_{\text{tap}}$ 折减的分泌效价。
* **着色器美学**：复刻古典植物档案纸张渗透质感。通过四阶分形布朗运动（FBM）对标量场进行各向异性渗透侵蚀，叠加微观伪随机高斯颗粒：
  $$\text{Color} = \text{lerp}\left(\vec{C}_{\text{paper}}, \vec{C}_{\text{ink}}, \text{Trail}^\gamma\right) + \text{Grain}(\vec{x}, t)$$
* **失效模式分析**：树状分叉爆发期引发 GPU 线程束分化（Warp Divergence），全局原子自增计数器（`atomic_add`）严重拖慢硬件吞吐。

#### `v3.0` - 纯空间殖民拓扑图网络 (Space Colonization Graph)
* **对应 Commit**: `a8923be`
* **计算架构**：彻底放弃欧拉连续网格，全面转向基于空间殖民算法（SCA）的拉格朗日拓扑树图。
* **动力学机制**：
  * 12,000 个离散养分吸引子 $\vec{A}_k$ 离散布设于底质空间。
  * 影响域检索：吸引子在影响半径 $D_{\text{attr}} = 38.0\text{ px}$ 内对临近血管树节点施加归一化牵引力：
    $$\vec{v}_{\text{grow}} = \text{normalize}\left(\sum_{k} \frac{\vec{A}_k - \vec{N}_j}{\Vert{}\vec{A}_k - \vec{N}_j\Vert{}}\right)$$
  * 养分消耗判定：当节点逼近至摄食半径 $D_{\text{kill}} = 7.5\text{ px}$ 内部时，吸引子灭活并转化为脉管粗度。
* **形态学突破**：从几何拓扑层面根除了网格模型的迷宫海绵化缺陷，严格构建出主干粗壮、末端毛细的自然层级结构。

#### `v3.1` - 空间哈希网格加速、逆行剪枝代谢与交互 HUD
* **对应 Commit**: `7bda3a1`
* **计算架构**：SCA 拓扑图系统的工程化重构与生物流体力学闭环。
* **核心升级机制**：
  * **GPU 空间哈希加速桶**：将视口划分为 $40 \times 40\text{ px}$ 的离散网格单元，把几何最近邻检索复杂度从全局暴力比对降低到局域网格循环（单个单元容量阈值 `MAX_NODES_PER_CELL = 128`），在 1080P/2K 视口下稳定维持 60 FPS 满帧运行。
  * **逆行通量与泊肃叶剪枝（Retrograde Pruning）**：当探索末梢物理触碰燕麦养分点时，激活反向通量泵（Retrograde Flow），沿父节点指针向母核回溯注入输运通量 $\Delta \Phi$：
    $$\text{node\_flow}_{\text{parent}} \mathrel{+}= 0.55$$
    输运通量低于阈值（$\text{flow} < 0.16$）的冗余探索细丝，其生物活性以每步 $\delta_v = 0.0035$ 的速率衰减并在 3\~5 秒内完全凋亡溶解。
  * **GGUI 原生控制台**：在视口内集成 Dear ImGui 状态面板，实时监控活动节点规模与系统生命周期。
  * **操作向导**：未接种时中央显示动态呼吸环提示交互步骤。

#### `v4.0` - 双通道延迟混合与甜甜圈环形拓扑 (Dual-Channel Blending)
* **对应 Commit**: `e4941c2`
* **计算架构**：1:1 移植 Nicolas Barradeau (nicoptere) 的 WebGL GPGPU 乒乓帧缓冲架构。
* **动力学机制**：
  * **R 通道（瞬间冲量）**：记录当前帧粒子绝对位置的离散光栅化冲量。
  * **G 通道（历史相场）**：平滑延迟轨迹场。
  * 非对称混合卷积核：
    $$G_{t+1}(\vec{x}) = \gamma \sum_{\vec{\delta}} K(\vec{\delta}) \left[R_t(\vec{x} + \vec{\delta}) + 0.5 \cdot G_t(\vec{x} + \vec{\delta})\right]$$
  * 环形无缝拓扑（Torus Wrap）：利用浮点取整小数算子 `fract()` 实现坐标越界循环。
* **形态学突破**：消除了培养皿边界反弹形成的堆积伪影，管壁边缘呈现亚像素级别的平滑汇聚感。

#### `v5.0` - 多物种拮抗矩阵与 2.5D 高度场法线渲染 (Multi-Species & 2.5D Normal)
* **对应 Commit**: `86e1294`
* **计算架构**：双物种相互作用张量与微表面光度学法线重构。
* **动力学机制**：
  * **交叉作用张量（Interaction Tensor）**：
    $$M = \begin{pmatrix} +1.0 & -0.4 \\ -0.4 & +1.0 \end{pmatrix}$$
    同类物质正反馈成束，异类物质负反馈强烈排斥。异类剪切力直接撕裂连续相场，形成具有生物张力的孔洞（Membrane Pores）与锐利筋膜边界。
  * **2.5D 微表面法线着色（Sobel Bump Shading）**：
    将二维标量总物质场映射为虚拟微观地形高度 $H(x, y)$，通过有限差分提取表面法线向量：
    $$\vec{N} = \text{normalize}\left(-k \cdot \partial_x H, -k \cdot \partial_y H, 1.0\right)$$
    引入倾斜平行光源 $\vec{L}$ 计算半兰伯特漫反射与高光反射。
* **视觉突破**：摆脱了传统发光粒子的平面游戏感，画面呈现如同冷冻电子显微镜扫描下的立体有机组织质感。

---

### 系统运行与环境配置 (Setup & Quick Start)

#### 1. 运行依赖
* 操作系统: Windows 10/11, Linux, macOS
* Python 环境: Python 3.8\~3.12
* 底层硬件支持: 兼容 Direct3D 12 / Vulkan 1.2+ 的现代显卡，推荐 NVIDIA GPU（原生调度 CUDA 计算后端）

安装核心计算依赖：
```bash
pip install taichi
```

#### 2. 检出并运行历史里程碑版本
可通过 Git Tag 自由切换并检视各个演进阶段的核心架构：

```bash
# 检出 v1.0 基础高密穿梭流模型
git checkout v1.0
python main.py

# 检出 v3.1 空间哈希拓扑图网络 (带 HUD 与水力剪枝)
git checkout v3.1
python main.py

# 检出 v4.0 双通道 GPGPU 延迟混合模型
git checkout v4.0
python main.py

# 检出 v5.0 终极多物种拮抗与 2.5D 法线渲染模型
git checkout v5.0
python main.py
```

#### 3. v3.1 运行时交互控制定义
* **鼠标右键 (RMB)**: 在培养皿内指定位置接种黏菌母核并释放全局各向同性探索引力子。
* **鼠标左键 (LMB)**: 投放下高能燕麦粒（产生局部密集引力波前）。
* **空格键 (Space)**: 暂停 / 继续系统演进。
* **按键 R**: 重置全域显存，清空脉管图网络。

---

### 致谢与学术源流 (Credits & Extended References)

本项目建立在一批开源图形学家、生物物理学家与程序艺术家的理论及工程基石之上：

* **Jeff Jones (2010)**: *Characteristics of Pattern Formation and Evolution in Approximations of Physarum Polycephalum*. Artificial Life, 16(2), 127-153. (提出多智能体粒子连续场嗅探与运动决策的奠基性模型)。
* **Adam Runions et al. (2005)**: *Modeling and visualization of leaf venation patterns*. ACM Transactions on Graphics (TOG), 24(3), 702-711. (提出空间殖民算法 SCA，奠定了脉管网络的几何生长理论)。
* **Atsushi Tero, Toshiyuki Nakagaki et al. (2010)**: *Rules for Biologically Inspired Adaptive Network Design*. Science, 327(5964), 439-442. (著名的东京铁路黏菌输运实验，确立了基于管流输运反馈的自发剪枝代谢模型)。
* **Sage Jenson**: *Physarum Polycephalum Simulation & Procedural Aesthetics*. <https://cargocollective.com/sagejenson/physarum> (确立了基于 2.5D 法线凹凸映射、芥末金黄色谱与电子显微镜立体浮雕质感的美学范式)。
* **Nicolas Barradeau (nicoptere)**: *WebGL GPGPU Multi-agent Transport Network*. <https://github.com/nicoptere/physarum> (开创了基于双通道 Ping-Pong 帧缓冲时间延迟滤波与环形边界拓扑的高性能 WebGL 流水线)。
* **Michael Fogleman**: *Physarum - Multi-species Agent Simulation in Go*. <https://github.com/fogleman/physarum> (实现了多物种交叉相互作用亲和张量、级联空间多遍模糊与亚像素连续插值技术)。
* **Sebastian Lague**: *Slime Simulation Compute Shader Pipeline*. <https://github.com/SebLague/Slime-Simulation> (验证了基于现代化 GPU Compute Shader 高并发调度百万级无状态粒子的工程架构)。
* **Jeffrey (Ka Hin) Yuen**: *PHYSARUM: Slime Mold Simulator*. <https://store.steampowered.com/app/1667120/PHYSARUM_Slime_Mold_Simulator/> (在商业级产品中验证了统一环境势能标量场 $S_{\text{total}} = S_{\text{trail}} + \lambda S_{\text{env}}$ 结合面阵区域探针采样在大规模 GPU 模拟中的稳定性和表现力)。
* **Ian Pilon (ianpilon)**: *Physarum Dendritic Morphogenesis in Processing*. <https://github.com/ianpilon/physarum> (提供了顶端递归出芽概率分化、代际生命衰退与植物档案着色器的参考范式)。
