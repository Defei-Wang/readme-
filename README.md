# Physarum: Morphogenesis & Transport Network Engine

[English](#english-documentation) | [中文说明](#chinese-documentation)

A high-performance research pipeline and procedural generative system simulating the morphogenesis, foraging dynamics, and adaptive cytoplasmic transport networks of *Physarum polycephalum* (true slime mold).

Implemented in Python with GPU acceleration via the Taichi compute infrastructure, targeting CUDA and Vulkan backends. The repository chronicles the technical lineage from classical agent-continuum heuristics (Jeff Jones, 2010) to topological space colonization graphs (Adam Runions, 2005), adaptive biological transport optimization (Atsushi Tero & Toshiyuki Nakagaki, 2010), GPGPU dual-channel frame-buffer delay blending (Nicolas Barradeau), and multi-species antagonistic 2.5D normal-mapped photometrics (Michael Fogleman, Sage Jenson).

---

<a id="english-documentation"></a>
## English Documentation

### Architectural Duality & Computational Paradigms

Simulating the morphology of *Physarum polycephalum* requires resolving a fundamental biomechanical paradox: **local protoplasmic fluid continuity versus macroscopic transport network pruning**. Computational generative systems generally branch into two distinct paradigms:

1. **Continuum Multi-Agent Reaction-Diffusion (Jones-Lague-Yuen Pipeline)**:
   A decentralized swarm formulation where hundreds of thousands to millions of dimensionless agents deposit and sample an information field. Network conduits emerge purely through non-linear positive feedback between trail accumulation, spatial diffusion, and exponential decay.
2. **Discrete Topological Graph Colonization (Runions-Tero Biomechanical Engine)**:
   A graph-theoretic formulation using discrete attractor point distributions to guide iterative venation tree growth, followed by retrograde hydraulic flow reinforcement and Hagen-Poiseuille vessel diameter pruning.

This repository implements, benchmark-tests, and preserves both paradigms across sequential milestone tags (`v1.0` through `v3.1`, up to `v5.0`).

```text
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
             └─► v5.0: Antagonistic Species + 2.5D Normal (fogleman / Jenson)
```

---

### Milestone Specifications & Theoretical Lineage

#### `v1.0` - Baseline Dense Swarm & Shuttle Streaming
* **Commit**: `2af9244`
* **Theoretical Foundation**: Dr. Jeff Jones (2010) multi-agent chemoattractant transport model.
* **Architecture**: Continuous scalar field coupled with 1.2M discrete active agents.
* **Mathematical Stencil**:
  * Agent heading update via discrete three-point differential sensing:

$$\theta_{t+1} = \theta_t + \Delta\theta \cdot \mathrm{sign}(S_R - S_L)$$

  where forward offset sensors ($S_L, S_C, S_R$) sample the trail intensity at sensor offset $SO = 22.5^\circ$, sensor angle $SA = 45.0^\circ$, and sampling distance $SS = 3.5\mathrm{px}$.
  * Cytoplasmic shuttle streaming: Macro-scale rhythmic contraction modulated via low-frequency sinusoidal velocity fields:

$$v(t) = v_0 \cdot \left[ 1.0 + 0.3 \sin(\omega t - \vec{k} \cdot \vec{x}) \right]$$

  * Multiplicative field decay ($\gamma = 0.94$) over an isotropic 3×3 discrete convolution kernel.
* **Failure Mode Analysis**: In the absence of competitive boundaries or lateral inhibition, high-density swarms collapse into static, sponge-like isotropic mazes and exhibit severe wavefront interference upon colliding with rigid boundaries.

#### `v2.0` - Generational Tip-Growth & Archival Paper Shader
* **Commit**: `d48f330`
* **Theoretical Foundation**: Ian Pilon (`ianpilon`) dendritic tip-growth Processing formulation, integrated with Sage Jenson's botanical color palette.
* **Architecture**: Explicit generational recursion with finite lifespan metadata, derived from botanical apical meristem expansion.
* **Mechanics**:
  * Each active agent tracks recursion generation $g \in [0, 6]$ and remaining metabolic ticks $L_i$.
  * Lateral bifurcation follows an accelerated probability distribution:

$$P_{\mathrm{branch}} = P_0 \cdot \alpha_{\mathrm{acc}}^g$$

  * Distal daughters inherit attenuated deposition potency:

$$D_{\mathrm{child}} = D_{\mathrm{parent}} \cdot V_{\mathrm{tap}}$$

* **Renderer**: Botanical archival specimen shader. Evaluates four-octave Fractional Brownian Motion (FBM) to simulate heterogeneous liquid ink absorption over porous paper substrates:

$$\mathrm{Color} = \mathrm{lerp}(\vec{C}_{\mathrm{paper}}, \vec{C}_{\mathrm{ink}}, \mathrm{Trail}^\gamma) + \mathrm{Grain}(\vec{x}, t)$$

* **Failure Mode Analysis**: Exponential bud proliferation triggers GPU thread serialization and thread-divergence bottlenecks around atomic allocation counters.

#### `v3.0` - Pure Space Colonization Vascular Graph (SCA)
* **Commit**: `a8923be`
* **Theoretical Foundation**: Adam Runions et al. (2005) Space Colonization Algorithm (SCA) for leaf venation morphogenesis.
* **Architecture**: Complete transition from continuum grid Eulerian tracking to discrete Lagrangian graph growth based on spatial attractor point distributions.
* **Mechanics**:
  * 12,000 discrete chemoattractant markers $\vec{A}_k$ distributed pseudo-randomly across the substrate.
  * Influence cone search: Each active marker pulls the nearest vascular graph node $\vec{N}_j$ within search radius $D_{\mathrm{attr}} = 38.0\mathrm{px}$.
  * Normalized growth vector accumulation:

$$\vec{v}_{\mathrm{grow}} = \frac{\sum_{k} (\vec{A}_k - \vec{N}_j)}{\left\Vert \sum_{k} (\vec{A}_k - \vec{N}_j) \right\Vert}$$

  * Consumption threshold: Attractors undergo metabolic depletion and deletion when $\Vert\vec{A}_k - \vec{N}_j\Vert < D_{\mathrm{kill}} = 7.5\mathrm{px}$.
* **Structural Result**: Eradicates sponge-maze collapse. Guarantees hierarchical trunk-to-capillary diameter scaling while maintaining organic topological spacing.

#### `v3.1` - Spatial-Hashed Optimization, Retrograde Pruning & Interactive HUD
* **Commit**: `7bda3a1`
* **Theoretical Foundation**: Atsushi Tero & Toshiyuki Nakagaki (2010) adaptive biological network design (Tokyo railway experiment), coupled with Runions' SCA and Hagen-Poiseuille hydraulic resistance.
* **Architecture**: High-performance optimization of the SCA vascular engine with Poiseuille-inspired retrograde pruning, spatial hash partitioning, and Taichi GGUI instrumentation.
* **Key Enhancements**:
  * **GPU Spatial Hashing Acceleration**: Partitions the grid into spatial bins of size 40×40 px, reducing the nearest-node search from brute-force `O(N_attrs * N_nodes)` down to localized adjacent cell sweeps (`MAX_NODES_PER_CELL = 128`), maintaining 60 FPS on mid-range hardware.
  * **Retrograde Flow & Poiseuille Pruning**: Upon physical contact between an exploration tip and a nutrient site, a recursive backward traversal pumps transport flux upstream through parent indices:

$$\Phi_{\mathrm{parent}} \leftarrow \Phi_{\mathrm{parent}} + \Delta \Phi$$

  Vessels with stagnant flow (`flow < 0.16`) suffer metabolic penalty (`vitality -= 0.0035`), dissolving and clearing non-transporting search paths within 3~5 seconds.
  * **Real-time On-Screen HUD**: Integrated Dear ImGui overlay rendering node metrics, memory saturation, and interactive state indicators.
  * **Inoculation Guidance**: Dynamic breathing visual feedback indicating pre-inoculation placement.

#### `v4.0` - Dual-Channel Delayed Blending & Torus Topology
* **Commit**: `e4941c2`
* **Theoretical Foundation**: Nicolas Barradeau (`nicoptere`) WebGL GPGPU dual-texture ping-pong architecture.
* **Architecture**: 1:1 translation of classic WebGL GPGPU frame-buffer delay blending pipelines.
* **Mechanics**:
  * **R Channel (Instantaneous Impulse)**: Binary rasterization of instantaneous agent positions.
  * **G Channel (Morphogenetic Memory)**: Spatial temporal decay buffer.
  * Asymmetric convolution kernel:

$$G_{t+1}(\vec{x}) = \gamma \sum_{\vec{\delta}} K(\vec{\delta}) \left[ R_t(\vec{x} + \vec{\delta}) + 0.5 \cdot G_t(\vec{x} + \vec{\delta}) \right]$$

  * Torus periodic boundary topology: Evaluated via mathematical fractional arithmetic:

$$\vec{x}_{\mathrm{wrapped}} = \vec{x} - \lfloor \vec{x} \rfloor$$

* **Structural Result**: Complete removal of boundary accumulation. The asymmetric 1.0 : 0.5 kernel weight preserves sharp leading search fronts while leaving smooth, fluid conduits behind.

#### `v5.0` - Multi-Species Antagonism & 2.5D Height-Field Normal Mapping
* **Commit**: `86e1294`
* **Theoretical Foundation**: Michael Fogleman (`fogleman/physarum`) multi-species interaction tensors, combined with Sage Jenson's 2.5D photometric micro-relief normal bump mapping.
* **Architecture**: Cross-species affinity tensor coupled with photometric normal-bump reconstruction.
* **Mechanics**:
  * **Species Interaction Tensor (2×2)**:

$$M = \begin{pmatrix} +1.0 & -0.4 \\ -0.4 & +1.0 \end{pmatrix}$$

  Positive diagonal entries dictate intra-species cohesion; negative off-diagonal entries generate inter-species shear forces that tear open membrane lacunae and maintain crisp territorial boundaries.
  * **2.5D Microscopic Normal Shading**:
    Treats cumulative matter $H(\vec{x}) = \sum_c \mathrm{grid}_c(\vec{x})$ as a continuous surface height-field. Computes surface normal vectors using finite-difference Sobel operators:

$$\vec{N} = \frac{(-\partial_x H, -\partial_y H, 1.0)}{\Vert (-\partial_x H, -\partial_y H, 1.0) \Vert}$$

$$\mathrm{Shading} = I_{\mathrm{ambient}} + I_{\mathrm{diffuse}} \max(0, \vec{N} \cdot \vec{L}) + I_{\mathrm{spec}} (\vec{N} \cdot \vec{H})^\alpha$$

* **Visual Result**: Substitutes flat, luminous lines with three-dimensional, translucent organic cord networks illuminated by directional lighting.

---

### Comparative Architecture Matrix

| Version Tag | Computational Core | Theoretical Origin | Boundary Model | Topological Behavior | Primary Aesthetic |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **v1.0** | Agent-Field PDE | Jeff Jones (2010) | Circular Rigid Reflection | Isotropic Sponge Reticulation | Dual-Scale Optical Density |
| **v2.0** | Recursive Tree | Ian Pilon (`ianpilon`) | Bounding Box Clamp | Hierarchical Apical Budding | Botanical Archival Ink & Paper |
| **v3.0** | Discrete Attractor | Adam Runions (2005) | Semi-Infinite Substrate | True Venation (Zero Sponge) | Structural Filament Density |
| **v3.1** | Spatial-Hashed Graph | Tero & Nakagaki (2010) | Circular Hydrodynamic Lip | Dynamic Retraction & Pruning | Specular Microscope Micro-Relief |
| **v4.0** | Dual-Channel Ping-Pong | Nicolas Barradeau (`nicoptere`) | Toroidal Periodic Wrap | High-Reynolds Viscous Cords | Grayscale Temporal Phase Buffer |
| **v5.0** | Multi-Vector Field | Michael Fogleman / Sage Jenson | Toroidal Periodic Wrap | Competitive Pores & Lacunae | 2.5D Photometric Electron Relief |

---

<a id="chinese-documentation"></a>
## 中文说明

### 架构全景与双重计算范式

多头绒泡菌（*Physarum polycephalum*）形态发生的数学建模，核心挑战在于平衡**局域流体连续性与宏观脉管系统拓扑剪枝**。在计算生成设计领域，主要存在两条截然不同的技术路线：

1. **连续场多智能体模型（Jones-Lague-Yuen 体系）**：
   无中心化的纯粒子流驱动。数十万至数百万粒子在二维平面上根据局部嗅探到的化学势能浓度差进行偏转，通过信息素轨迹分泌、空间扩散与衰减的非线性正反馈，自发涌现出脉管结构。
2. **离散空间拓扑殖民图（Runions-Tero 生物输运体系）**：
   图论几何模型。通过在培养基离散播撒养分吸引子，牵引节点迭代生成树状网络，结合反向水力通量回溯与泊肃叶定律进行管道口径缩放与盲端剪枝。

本项目完整推演、验证并沉淀了上述两套体系，在 Git Tags 中以 `v1.0` 至 `v3.1` 以及 `v4.0`、`v5.0` 形成严密的科研演进链条。

```text
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
             └─► v5.0: 异种拮抗矩阵与 2.5D 法线 (fogleman / Jenson)
```

---

### 版本演进编年史与学术溯源

#### `v1.0` - 基础高密粒子流与穿梭流 (Baseline Dense Swarm)
* **对应 Commit**: `2af9244`
* **学术溯源**: 基于 Dr. Jeff Jones (2010) 的 Multi-Agent 连续场化学趋向性模型。
* **计算架构**：120 万高密度活动质点与欧拉连续网格场耦合。
* **核心数学算子**：
  * 三向离散差分偏转决策：

$$\theta_{t+1} = \theta_t + \Delta\theta \cdot \mathrm{sign}(S_R - S_L)$$

  前置三向探针偏移角 $SO = 22.5^\circ$，感知半角 $SA = 45.0^\circ$，前探步距 $SS = 3.5\mathrm{px}$。
  * 原生质往复穿梭流（Shuttle Streaming）：低频行波调制运动步长与分泌通量：

$$v(t) = v_0 \cdot \left[ 1.0 + 0.3 \sin(\omega t - \vec{k} \cdot \vec{x}) \right]$$

  * 3×3 离散高斯平滑核卷积与乘法挥发（衰减率 $\gamma = 0.94$）。
* **失效模式分析**：系统缺乏宏观侧向抑制机制，演化后期质点在培养皿边缘发生驻波干涉，退化为均质海绵网状结构及同心圆环伪影。

#### `v2.0` - 显式代际顶端分叉与标本着色 (Generational Tip-Growth)
* **对应 Commit**: `d48f330`
* **学术溯源**: 移植 Ian Pilon (`ianpilon`) 基于 Processing 的顶端分生组织递归算法，融合 Sage Jenson 标本底色调色板。
* **计算架构**：顶端出芽机制（Tip-Growth），粒子具有显式生命周期与代际元数据。
* **动力学机制**：
  * 粒子携带代际深度 $g \in [0, 6]$ 与存活寿命计数器 $L_i$。
  * 侧向出芽分叉概率随代际指数递减：

$$P_{\mathrm{branch}} = P_0 \cdot \alpha_{\mathrm{acc}}^g$$

  * 子代分支继承按衰减系数 $V_{\mathrm{tap}}$ 折减的分泌效价。
* **着色器美学**：复刻古典植物档案纸张渗透质感。通过四阶分形布朗运动（FBM）对标量场进行各向异性渗透侵蚀，叠加微观伪随机高斯颗粒：

$$\mathrm{Color} = \mathrm{lerp}(\vec{C}_{\mathrm{paper}}, \vec{C}_{\mathrm{ink}}, \mathrm{Trail}^\gamma) + \mathrm{Grain}(\vec{x}, t)$$

* **失效模式分析**：树状分叉爆发期引发 GPU 线程束分化（Warp Divergence），全局原子自增计数器严重拖慢硬件吞吐。

#### `v3.0` - 纯空间殖民拓扑图网络 (Space Colonization Graph)
* **对应 Commit**: `a8923be`
* **学术溯源**: 奠基于 Adam Runions et al. (2005) 空间殖民算法（Space Colonization Algorithm, SCA）。
* **计算架构**：彻底放弃欧拉连续网格，全面转向基于空间殖民算法的拉格朗日拓扑树图。
* **动力学机制**：
  * 12,000 个离散养分吸引子 $\vec{A}_k$ 离散布设于底质空间。
  * 影响域检索：吸引子在影响半径 $D_{\mathrm{attr}} = 38.0\mathrm{px}$ 内对临近血管树节点施加归一化牵引力：

$$\vec{v}_{\mathrm{grow}} = \frac{\sum_{k} (\vec{A}_k - \vec{N}_j)}{\left\Vert \sum_{k} (\vec{A}_k - \vec{N}_j) \right\Vert}$$

  * 养分消耗判定：当节点逼近至摄食半径 $D_{\mathrm{kill}} = 7.5\mathrm{px}$ 内部时，吸引子灭活并转化为脉管粗度。
* **形态学突破**：从几何拓扑层面根除了网格模型的迷宫海绵化缺陷，严格构建出主干粗壮、末端毛细的自然层级结构。

#### `v3.1` - 空间哈希网格加速、逆行剪枝代谢与交互 HUD
* **对应 Commit**: `7bda3a1`
* **学术溯源**: 完整复刻 Atsushi Tero & Toshiyuki Nakagaki et al. (2010) 发表于 *Science* 的东京铁路网络自适应流体阻力动力学，结合泊肃叶定律与 SCA 拓扑扩展。
* **计算架构**：SCA 拓扑图系统的
