markdown
# Physarum: Morphogenesis & Transport Network Engine

[English](#english-documentation) | [中文说明](#chinese-documentation)

A high-performance research pipeline simulating the morphogenesis, foraging dynamics, and cytoplasmic transport networks of *Physarum polycephalum* (slime mold). Implemented in Python with GPU acceleration via Taichi (Vulkan / CUDA backends).

This repository chronicles the architectural transition from classical multi-agent continuum heuristics to discrete topological space-colonization graphs, GPGPU dual-channel blending pipelines, retrograde hydrodynamic pruning, and multi-species antagonistic 2.5D normal-mapped shading.

---

<a id="english-documentation"></a>
## English Documentation

### Architecture Overview

Simulating *Physarum* morphology requires resolving a fundamental contradiction in generative transport systems: **local fluid continuity versus macro-scale branch pruning**. Simple particle swarms collapse into homogeneous sponge-like patterns, whereas rigid graph networks lose the fluid, organic plasticity of living protoplasm.

This codebase tracks sequential architectural paradigms across Git tags v1.0\~v5.0 (and intermediate releases such as v3.1), systematically resolving transport efficiency, membrane tension, and microscopic surface relief.


```

```
                ┌─────────────────────────┐
                │ v1.0: Dense Swarm (PDE) │
                └────────────┬────────────┘
                             │ (Failure mode: concentric rings, sponge collapse)
                ┌────────────┴────────────┐
                ▼                         ▼
  ┌──────────────────────────┐   ┌───────────────────────────┐
  │ v2.0: Generational Buds  │   │ v3.0: SCA Topology Graph  │
  │ (Explicit tip branching) │   │ (Pure discrete attractor) │
  └────────────┬─────────────┘   └─────────────┬─────────────┘
               │                               │
               │                 ┌─────────────▼─────────────┐
               │                 │ v3.1: Retrograde Pruning  │
               │                 │ (Poiseuille flow & HUD)   │
               │                 └─────────────┬─────────────┘
               └─────────────┬─────────────────┘
                             │ (Synthesis: continuum fluid + structural membrane)
                ┌────────────▼────────────┐
                │ v4.0: Dual-Channel Ping │
                │ (R/G field decoupling)  │
                └────────────┬────────────┘
                             │ (Breakthrough: multi-species tension + 2.5D light)
                ┌────────────▼────────────┐
                │ v5.0: Antagonistic 2.5D │
                │ (Sobel normal shading)  │
                └─────────────────────────┘

```

```

---

### Evolutionary Lineage & Milestone Specifications

#### v1.0 - Baseline Dense Swarm & Shuttle Streaming
* **Commit**: 2af9244
* **Core Paradigm**: 1.2M particle continuum interacting with scalar pheromone fields.
* **Mathematical Stencil**:
  * Agent heading update:
    $$\theta_{t+1} = \theta_t + \Delta\theta \cdot \text{sign}(S_R - S_L)$$
    where $S_L, S_C, S_R$ sample the combined trail and scent fields via three forward-offset sensors ($SO = 22.5^\circ, SA = 45.0^\circ, SS = 3.5\text{ px}$).
  * Macro-scale protoplasmic shuttle streaming: Sinusoidal velocity modulation:
    $$v(t) = v_0 \cdot \left[1.0 + 0.3 \sin\left(\omega t - \vec{k} \cdot \vec{x}\right)\right]$$
  * Trail dissipation: $3 \times 3$ sub-pixel Gaussian kernel with multiplicative decay ($\gamma = 0.94$).
* **Pathology**: Under prolonged runs, agents lose directional divergence, clustering into static sponge-like meshes and concentric wavefront ripples upon dish boundary collisions.

#### v2.0 - Generational Tip-Growth & Archival Paper Shader
* **Commit**: d48f330
* **Core Paradigm**: Explicit recursive branching with finite metabolic lifespan (inspired by tip-growth morphogenesis).
* **Mechanics**:
  * Agent metadata tracks generation depth $g \in [0, 6]$ and remaining metabolic lifespan $L_i$.
  * Branching probability decays exponentially:
    $$P_{\text{branch}} = P_0 \cdot \alpha_{\text{acc}}^g$$
  * Daughter agents inherit attenuated deposition potency:
    $$D_{\text{child}} = D_{\text{parent}} \cdot V_{\text{tap}}$$
* **Renderer**: Classical botanical archival paper aesthetic. Employs Fractal Brownian Motion (FBM) and pseudo-random film grain to break planar digital uniformity:
  $$\text{Color} = \text{lerp}\left(\vec{C}_{\text{paper}}, \vec{C}_{\text{ink}}, \text{Trail}^\gamma\right) + \text{Grain}(\vec{x}, t)$$
* **Pathology**: Branch explosions require rigid atomic allocation counters (`atomic_add`), introducing synchronization bottlenecks on parallel compute pipelines.

#### v3.0 - Pure Topological Vascular Graph (SCA)
* **Commit**: a8923be
* **Core Paradigm**: Complete departure from grid-based particle fields; deployment of the Space Colonization Algorithm (SCA).
* **Mechanics**:
  * Distributes 12,000 discrete nutrient attractors $\vec{A}_k$ across the substrate.
  * Node search: Each active attractor pulls the closest vascular node $\vec{N}_j$ within influence radius $D_{\text{attr}}$.
  * Normalized growth vector accumulation:
    $$\vec{v}_{\text{grow}} = \text{normalize}\left(\sum_{k} \frac{\vec{A}_k - \vec{N}_j}{\Vert{}\vec{A}_k - \vec{N}_j\Vert{}}\right)$$
  * Kill distance: Attractors are consumed when $\Vert{}\vec{A}_k - \vec{N}_j\Vert{} < D_{\text{kill}}$.
* **Significance**: Mathematically prevents sponge artifacts. Enforces strict vascular hierarchy: primary trunks scale up in radius while distal search capillaries remain ultra-fine.

#### v3.1 - Retrograde Hydrodynamic Flow & Interactive HUD
* **Commit**: 7bda3a1
* **Core Paradigm**: Topological vascular graph coupled with local nutrient consumption, retrograde flux thickening, and Poiseuille metabolic pruning.
* **Mechanics**:
  * Eliminates global food search. Attractors cluster locally around deposited nutrients ($R \le 36.0\text{ px}$).
  * Nodes touching nutrient sites initiate a reverse flux wave toward the root:
    $$\text{Flow}_{\text{parent}} \leftarrow \text{Flow}_{\text{parent}} + \Delta Q, \quad \text{Vitality} \leftarrow 1.0$$
  * Starved exploratory blind ends ($\text{Flow} < \text{Threshold}$) undergo geometric decay and dissolve within 3\~5 seconds:
    $$\text{Vitality}_{t+1} = \max(0.0, \text{Vitality}_t - \delta_{\text{metabolic}})$$
  * Built-in Dear ImGui dashboard via Taichi GGUI for real-time node budget monitoring, pause/resume, and inoculation state tracking.

#### v4.0 - Dual-Channel Delayed Blending & Torus Topology
* **Commit**: e4941c2
* **Core Paradigm**: 1:1 translation of classic WebGL GPGPU ping-pong framebuffer pipelines.
* **Mechanics**:
  * **R Channel (Instantaneous Deposit)**: Binary rasterization of agent coordinates.
  * **G Channel (Morphogenetic Memory)**: Temporal diffusion buffer.
  * Convolutional mixing filter:
    $$G_{t+1}(\vec{x}) = \gamma \sum_{\vec{\delta}} K(\vec{\delta}) \left[R_t(\vec{x} + \vec{\delta}) + 0.5 \cdot G_t(\vec{x} + \vec{\delta})\right]$$
    with box blur weight $K = \frac{1}{9}$.
  * Torus boundary wrapping via floating-point fract operations:
    $$\vec{x} = \text{fract}(\vec{x})$$
* **Significance**: Zero border reflection clustering. The $1.0 : 0.5$ kernel ratio creates sharp front-wave headers trailing into smooth, persistent conduits.

#### v5.0 - Multi-Species Antagonism & 2.5D Height-Field Normal Mapping
* **Commit**: 86e1294
* **Core Paradigm**: Multi-species interaction tensor coupled with photometric normal-bump reconstruction.
* **Mechanics**:
  * **Cross-Species Tensor ($2 \times 2$)**:
    $$M = \begin{pmatrix} +1.0 & -0.4 \\ -0.4 & +1.0 \end{pmatrix}$$
    Intra-species positive feedback consolidates cords; inter-species negative repulsion tears open membrane pores, generating biological lacunae.
  * **Multi-Pass Blur**: Cascaded passes ($\text{Passes} = 2$) extend spatial tension fields without sacrificing fine edges.
  * **2.5D Photometric Shading**:
    Treats cumulative matter $H(\vec{x}) = \sum_c \text{grid}_c(\vec{x})$ as a microscopic relief height-field:
    $$\nabla H = \left(\frac{H_{x+1, y} - H_{x-1, y}}{2}, \frac{H_{x, y+1} - H_{x, y-1}}{2}\right)$$
    $$\vec{N} = \text{normalize}\left(-k \cdot \partial_x H, -k \cdot \partial_y H, 1.0\right)$$
    $$\text{Shading} = I_{\text{ambient}} + I_{\text{diffuse}} \max\left(0, \vec{N} \cdot \vec{L}\right)$$
* **Significance**: Achieves electron-microscope visual depth, replacing glowing flat particle trails with three-dimensional vascular tissue.

---

### Comparative Feature Matrix

| Version Tag | State Space | Computational Complexity | Boundary Physics | Structural Phenomenon | Visual Paradigm |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **v1.0** | Continuum Fields | $\mathcal{O}(N_{\text{agents}})$ | Rigid Circular Reflect | Homogeneous Sponge Mesh | Dual-scale Optical Absorption |
| **v2.0** | Dynamic Agent Pool | $\mathcal{O}(N_{\text{active}})$ | Box Hard Clamp | Branching Fractal Hierarchy | Archival Ink & Paper Grain |
| **v3.0** | Node-Edge Graph | $\mathcal{O}(N_{\text{nodes}} \cdot N_{\text{attrs}})$ | Discrete Seed Region | Strict Trunk-Capillary Tree | Softened Filament Density |
| **v3.1** | Spatial-Hashed Graph | $\mathcal{O}(N_{\text{local\_bucket}})$ | Circular Petri Dish | Retrograde Feed Thickening | 2.5D Normal + Realtime HUD |
| **v4.0** | Split RG Buffers | $\mathcal{O}(N_{\text{agents}} + \text{Res}^2)$ | Toroidal Periodic Wrap | High-speed Fluid Conduits | Grayscale Trail Density |
| **v5.0** | Multi-Vector Field | $\mathcal{O}(S \cdot N_{\text{agents}} + P \cdot \text{Res}^2)$ | Toroidal Periodic Wrap | Antagonistic Membrane Pores | 2.5D Directional Bump Shading |

---

<a id="chinese-documentation"></a>
## 中文说明

### 架构全景与技术路径

模拟多头绒泡菌（*Physarum polycephalum*）形态发生的核心难点，在于**局域连续流动与宏观拓扑剪枝的动力学统一**。单纯的粒子游走模型极易退化为均质海绵网格；而死板的树状图结构又缺乏原生质流动的塑性与表面张力。

本项目按版本演进序列完整固化了探索历程中的关键里程碑（v1.0 至 v5.0，以及强化工程落地的 v3.1），记录了从经典微元相场到空间殖民拓扑网络、逆行水力剪枝、双通道延迟滤波与多物种拮抗 2.5D 法线渲染的推演过程：


```

```
                ┌─────────────────────────┐
                │  v1.0: 高密粒子连续相场 │
                └────────────┬────────────┘
                             │ (失效模式: 边界干涉同心圆、均质海绵化)
                ┌────────────┴────────────┐
                ▼                         ▼
  ┌──────────────────────────┐   ┌───────────────────────────┐
  │  v2.0: 显式代际顶端分叉  │   │  v3.0: 纯空间殖民拓扑图   │
  │  (末梢生长、有限寿命)   │   │  (离散养分引导、无网格)   │
  └────────────┬─────────────┘   └─────────────┬─────────────┘
               │                               │
               │                 ┌─────────────▼─────────────┐
               │                 │  v3.1: 逆行粗化与水力剪枝 │
               │                 │  (泊肃叶代谢、原生 HUD)  │
               │                 └─────────────┬─────────────┘
               └─────────────┬─────────────────┘
                             │ (综合路线: 流体连续性 + 拓扑自组织)
                ┌────────────▼────────────┐
                │  v4.0: 双通道延迟平滑   │
                │  (R/G 解耦、环形无界)   │
                └────────────┬────────────┘
                             │ (形态突破: 多物种拮抗张力 + 2.5D 光照)
                ┌────────────▼────────────┐
                │  v5.0: 拮抗矩阵与法线场 │
                │  (Sobel 差分立体显微质感)│
                └─────────────────────────┘

```

```

---

### 版本演进编年史

#### v1.0 - 基础高密粒子流与穿梭流 (Baseline Dense Swarm)
* **对应 Commit**: 2af9244
* **动力学模型**：120 万粒子规模的高密度网格模型。
* **核心数学算子**：
  * 三向离散嗅探偏转（$SO = 22.5^\circ, SA = 45.0^\circ, SS = 3.5\text{ px}$）。
  * 原生质穿梭流行波（Shuttle Streaming）：使用低频周期正弦信号调制质点步长与分泌强度，模拟原生质宏观往复流动：
    $$\text{deposit} = D_0 \cdot \left[1.0 + A \cdot \sin\left(\omega t - \vec{k} \cdot \vec{x}\right)\right]$$
  * 物质场经由 $3 \times 3$ 高斯卷积平滑并施加衰减因子（$\gamma = 0.94$）。
* **失效模式分析**：系统缺乏宏观排斥与自动疏剪机制。粒子在刚性反射壁发生波前干涉，退化为同心圆环与均质海绵迷宫。

#### v2.0 - 显式代际分叉与植物标本着色 (Generational Tip-Growth)
* **对应 Commit**: d48f330
* **动力学模型**：引入植物形态发生学中的顶端生长机制（Tip-Growth）。
* **核心数学算子**：
  * 粒子维护代际深度 $g \in [0, 6]$ 与有限生存周期 $L_i$。
  * 侧向分叉概率随代际指数级衰减：
    $$P_{\text{branch}} = P_0 \cdot \alpha_{\text{acc}}^g$$
  * 沿主干向上分泌强度递减，保证主干粗壮而顶芽纤细。
* **渲染着色器**：复刻古典植物档案纸张质感。利用四阶分形布朗运动（FBM）对标量场进行非均匀渗透腐蚀，叠加微观伪随机高斯噪点：
  $$\text{Pixel} = \text{PaperBase} \cdot (1 - T) + \text{VascularInk} \cdot T + \text{Noise}$$
* **失效模式分析**：树状分叉机制引发粒子索引分配竞争，原子操作计数器（`atomic_add`）对并行流水线造成显著同步开销。

#### v3.0 - 纯空间殖民拓扑图网络 (Space Colonization Graph)
* **对应 Commit**: a8923be
* **动力学模型**：彻底摒弃连续场网格，采用空间殖民算法（SCA）。
* **核心数学算子**：
  * 在空间中随机离散播撒 12,000 个养分吸引子（Attractors）。
  * 迭代计算吸引子对局部血管树节点的欧氏距离与单位牵引向量。
  * 当养分点与节点距离小于吞噬阈值 $D_{\text{kill}}$ 时，养分死亡并转化为脉管粗度。
* **形态学突破**：在数学层面上完全根绝海绵化问题，纯粹依靠几何牵引生成具有严格粗细层级（主干到毛细）的拓扑树图，保留大面积自然留白。

#### v3.1 - 逆行水力粗化与交互式 HUD 控制 (Retrograde Flow & HUD)
* **对应 Commit**: 7bda3a1
* **动力学模型**：基于空间哈希桶优化的 SCA 血管拓扑网，结合局部进食反馈与泊肃叶流体剪枝。
* **核心数学算子**：
  * **去中心化局域引力**：食物投放仅在其周围 $36\text{ px}$ 范围内散发密集吸引子，彻底剔除全局距离引导。
  * **逆行通量反冲（Retrograde Flow）**：末梢触碰食物时，沿父节点指针链逐级回溯注入输运通量并激活生命活力（$\text{Vitality} = 1.0$）。
  * **盲端凋亡剪枝**：通量匮乏的无用探索枝干失去能量供给，活力以每帧 $0.0035$ 速率递减，在 3\~5 秒内完全溶解消除：
    $$\text{Vitality}_{t+1} = \max(0.0, \text{Vitality}_t - \delta_{\text{metabolic}})$$
  * **工程落地**：通过 Taichi GGUI 内嵌 Dear ImGui 交互面板，提供接种状态检测、实时节点开销统计与操作指引。

#### v4.0 - 双通道延迟混合与甜甜圈环形拓扑 (Dual-Channel Blending)
* **对应 Commit**: e4941c2
* **动力学模型**：1:1 像素级转译 WebGL GPGPU 乒乓帧缓冲架构。
* **核心数学算子**：
  * **R 通道（瞬间冲量）**：记录当前帧粒子落点的绝对离散位置。
  * **G 通道（历史相场）**：平滑延时轨迹场。
  * 延迟滤波混合核：将当前粒子冲量与历史相场按 $1.0 : 0.5$ 的非对称权重融合，再执行衰减：
    $$G_{t+1} = \gamma \cdot \text{BoxBlur}\left(R_t + 0.5 \cdot G_t\right)$$
  * 采用 `fract()` 算子构建甜甜圈环形空间（Torus Topology），坐标超出边界自动无缝穿出。
* **形态学突破**：消除了边界反弹引发的死区积压，粒子在亚像素采样下展现出如流体般的丝滑汇聚。

#### v5.0 - 多物种拮抗矩阵与 2.5D 高度场法线渲染 (Multi-Species & 2.5D Normal)
* **对应 Commit**: 86e1294
* **动力学模型**：多物种相互作用张量 + 光度学法线表面重构。
* **核心数学算子**：
  * **交叉吸引矩阵（Attraction Tensor）**：
    $$\begin{pmatrix} A_{00} & A_{01} \\ A_{10} & A_{11} \end{pmatrix} = \begin{pmatrix} +1.0 & -0.4 \\ -0.4 & +1.0 \end{pmatrix}$$
    同种物质强正反馈聚拢成束，异种物质强负反馈排斥。异种切应力直接将连续相场撕开，形成天然的筋膜边缘与张力孔洞（Membrane Pores）。
  * **多遍级联滤波（Multi-Pass Blur）**：执行两遍盒状平滑，在保留边缘极高频率差分的同时，将引力势场半径扩大数倍。
  * **2.5D 微表面法线着色（Sobel Bump Shading）**：
    将二维标量物质总量映射为虚拟三维微观地形高度 $H(x, y)$，通过有限差分提取表面法线向量：
    $$\vec{N} = \text{normalize}\left( -k \frac{\partial H}{\partial x}, -k \frac{\partial H}{\partial y}, 1.0 \right)$$
    引入倾斜平行光源 $\vec{L} = \text{normalize}(0.7, 0.7, 0.8)$ 计算半兰伯特漫反射与高光。
* **形态学突破**：彻底终结了扁平发光粒子的平面数字感，画面呈现出冷冻电子显微镜扫描下的厚实质感与生物组织阴影。

---

### 技术演进特征横向对比

| 评估维度 | v1.0 密集粒子流 | v2.0 顶芽分叉 | v3.0 空间殖民图 | v3.1 逆行剪枝图 | v4.0 双通道混合 | v5.0 拮抗法线场 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **状态载体** | 单层连续标量场 | 粒子属性结构体 | 树状图节点列表 | 空间哈希拓扑网 | R/G 分离缓冲 | 双物种向量场 |
| **空间拓扑** | 刚性环形边界 | 自由边界硬截断 | 离散有限有界空间 | 培养皿物理壁 | 环形无缝甜甜圈 | 环形无缝甜甜圈 |
| **形态缺陷** | 易产生海绵化斑图 | 粒子数易超载爆炸 | 缺乏流体动态感 | 依赖离散拓扑步长 | 缺乏立体张力空洞 | 计算负载处峰值 |
| **视觉呈现** | 扁平光学吸收伪彩 | 古典标本纸张暗调 | 平滑骨架线条 | 2.5D 浮雕 + HUD | 灰度亚像素拖尾 | 显微立体高光筋膜 |

---

### 环境配置与运行指南 (Setup & Quick Start)

#### 1. 运行时依赖
* 操作系统: Windows 10/11, Linux, macOS
* Python 环境: Python 3.8\~3.12
* 底层硬件支持: NVIDIA GPU (建议，调用 CUDA 后端) 或支持 Vulkan 1.2+ 的核显/独显

安装核心依赖库：
```bash
pip install taichi numpy

```

#### 2. 检出并运行历史里程碑

克隆仓库后，可通过 Git Tag 自由穿梭检视演进阶段的物理特性：

```bash
# 运行 v1.0 基础穿梭流模型
git checkout v1.0
python main.py

# 运行 v2.0 显式代际分叉与植物标本着色
git checkout v2.0
python main.py

# 运行 v3.1 逆行剪枝拓扑网络与 HUD 控制面板
git checkout v3.1
python main.py

# 运行 v4.0 双通道丝滑流体
git checkout v4.0
python main.py

# 运行 v5.0 终极多物种 2.5D 法线立体微观组织
git checkout v5.0
python main.py

```

#### 3. 交互操作定义 (以 v3.1 为例)

* **鼠标右键 (RMB)**: 在培养皿指定坐标接种黏菌母核。
* **鼠标左键 (LMB)**: 投放高能燕麦粒，向局部激发密集引力点。
* **空格键 (Space)**: 暂停 / 继续动力学生长。
* **按键 R**: 清空当前相场显存，重新进行原核接种初始化。

---

### 致谢与核心参考文献 (Credits & References)

* **Jeff Jones (2010)**: *Characteristics of Pattern Formation and Evolution in Approximations of Physarum Polycephalum*. Artificial Life, 16(2), 127-153. (Foundational agent-based formulation of multi-agent foraging, chemoattractant trail deposition, and self-organized routing).
* **Adam Runions, Martin Fuhrer, Brendan Lane, Pavol Federl, Anne-Gaelle Rolland-Lagan, Przemyslaw Prusinkiewicz (2005)**: *Modeling and visualization of leaf venation patterns*. ACM Transactions on Graphics (TOG), 24(3), 702-711. (Theoretical foundation of the Space Colonization Algorithm utilized in v3.0 and v3.1).
* **Atsushi Tero, Seiji Takagi, Tetsu Saigusa, Kentaro Ito, Dan P. Bebber, Mark D. Fricker, Kenji Yumiki, Ryo Kobayashi, Toshiyuki Nakagaki (2010)**: *Rules for Biologically Inspired Adaptive Network Design*. Science, 327(5964), 439-442. (Empirical foundation for Poiseuille-based feedback, retrograde tube thickening, and Tokyo rail network optimization).
* **Sage Jenson**: *Physarum Polycephalum Simulation & Procedural Aesthetics*. [https://cargocollective.com/sagejenson/physarum](https://cargocollective.com/sagejenson/physarum?utm_source=gemini) (Formative artistic and algorithmic blueprint for continuous biological transport fields, fluid-like organicity, and 2.5D physical normal-mapped slime mold tissue).
* **Nicolas Barradeau (nicoptere)**: *WebGL GPGPU Multi-agent Transport Network*. [https://github.com/nicoptere/physarum](https://github.com/nicoptere/physarum?utm_source=gemini) (Pioneering dual-channel GPGPU frame-buffer delay blending, R/G field separation, and periodic torus mapping implemented in v4.0).
* **Michael Fogleman**: *Physarum - Multi-species Agent Simulation in Go*. [https://github.com/fogleman/physarum](https://github.com/fogleman/physarum?utm_source=gemini) (Algorithmic implementation of cross-species interaction matrices, multi-pass spatial blur, and sub-pixel continuous field interpolation implemented in v5.0).
* **Sebastian Lague**: *Coding Adventure: Slime Simulation*. [https://github.com/SebLague/Slime-Simulation](https://github.com/SebLague/Slime-Simulation?utm_source=gemini) (Inspirational compute shader pipeline translating Jeff Jones' multi-agent chemoattractant dynamics to GPU parallel primitives).
* **Jeffrey (Ka Hin) Yuen**: *PHYSARUM: Slime Mold Simulator*. [https://store.steampowered.com/app/1667120/PHYSARUM_Slime_Mold_Simulator/](https://store.steampowered.com/app/1667120/PHYSARUM_Slime_Mold_Simulator/?utm_source=gemini) (Demonstrating production-grade GPU compute scalability, box-filtered area sensing kernels, and unified background chemoattractant environmental mapping without explicit entity overhead).
* **Catlike Coding (Jasper Flick)**: *Compute Shaders Tutorial Series*. [https://catlikecoding.com/](https://catlikecoding.com/?utm_source=gemini) (Foundational architectural references for GPU buffer management, texture kernels, and hardware-accelerated cellular automata).

```
