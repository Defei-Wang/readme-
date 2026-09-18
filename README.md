# Physarum: Morphogenesis & Transport Network Engine

[English](#english-documentation) | [中文说明](#chinese-documentation)

A GPU-accelerated simulation of the morphogenesis and adaptive transport
networks of *Physarum polycephalum* (true slime mold), built in Python on
the Taichi compute framework with CUDA / Vulkan backends.

Two computational lineages, six tagged releases:

```
                         [ Morphogenesis Problem Space ]
                                       │
             ┌─────────────────────────┴─────────────────────────┐
             ▼                                                   ▼
   [ Continuum Field Line ]                              [ Discrete Graph Line ]
   (Jones 2010 / Barradeau / Fogleman)                   (Runions 2005 / Tero-Nakagaki 2010)
             │                                                   │
             ├─► v1.0: Dense Swarm + Shuttle Streaming           ├─► v3.0: Space Colonization (SCA)
             ├─► v2.0: Generational Tip-Growth (ianpilon)        └─► v3.1: Spatial Hashing + Retrograde
             ├─► v4.0: Dual-Channel Delayed Blending (nicoptere)          Pruning + GGUI HUD
             └─► v5.0: Multi-Species Antagonism + 2.5D Relief
                        (fogleman / Sage Jenson)
```

---

<a id="english-documentation"></a>
## English Documentation

### Two Computational Lineages

1. **Continuum multi-agent field** (v1.0, v2.0, v4.0, v5.0): agents deposit
   and sample a shared scalar field; networks emerge from the interplay of
   deposition, diffusion, and decay. No explicit graph exists anywhere.
2. **Discrete topological graph growth** (v3.0, v3.1): attractor points
   pull the nearest graph node, followed by flow-based pruning. The tree
   is a discrete data structure, not a diffusing field.

### `v1.0` — Dense Swarm & Shuttle Streaming
* **Commit**: `2af9244`
* **Basis**: Jeff Jones (2010) multi-agent chemoattractant transport model.
* **Scale**: 1024×1024 grid, 1.2M agents.
* **Turning rule** — not a clean sign() function; it's a four-way decision
  matching the source directly:

  ```math
  \theta_{t+1} = \begin{cases}
  \theta_t & F > L \text{ and } F > R \\
  \theta_t \pm RA\ (\text{random sign}) & F < L \text{ and } F < R \\
  \theta_t - RA & L > R \\
  \theta_t + RA & R > L
  \end{cases}
  ```

  sensor offset distance $SS=3.5\text{px}$, sensor half-angle $22.5^\circ$,
  rotation step $RA=45^\circ$. (Note: the source's own variable names `SO`
  and `SA` are swapped relative to what they actually control — described
  here by function, not by variable name.)
* **Shuttle streaming modulates deposition, not velocity** — a
  low-frequency traveling wave with different coefficients depending on
  whether the agent is over food:

  ```math
  \text{deposit} = \begin{cases}
  D_0\,(2.5 + 1.5\sin(2t - 0.015(x{+}y)\,s)) & \text{feeding} \\
  D_0\,(0.8 + 0.3\sin(2t - 0.015(x{+}y)\,s)) & \text{otherwise}
  \end{cases}
  ```

  $D_0=4.0$, $s=\mathrm{sign}(\sin(0.15t))$.
* **Boundary**: a hard position clamp plus a *fully re-randomized heading*
  on contact — not a reflection. This is the direct cause of the bright
  ring artifact at the dish edge.
* **Rendering**: a two-term nonlinear color model — a low-opacity
  "fringe" term for faint capillaries and a cubic "core" term for
  saturated trunks.

### `v2.0` — Generational Tip-Growth
* **Commit**: `d48f330`
* **Basis**: `ianpilon`'s Processing/JS tip-growth sketch, with a botanical
  archival-paper renderer.
* **Architecture**: explicit agents with a generation counter and finite
  lifespan, allocated from a fixed-size ring buffer via an atomic counter.
* **Branch probability decays with generation**:

  ```math
  P_{\text{branch}}(g) = f_{ch}\cdot f_{acc}^{\,g}
  ```
* **Child potency decays on two independent axes**:

  ```math
  D_{\text{child}} = D_{\text{parent}}\cdot v_{tap} \qquad
  L_{\text{child}} = L_{\text{base}}\cdot l_{tap}^{\,g+1}
  ```
* **Renderer**: four-octave FBM paper-fiber noise, plus a radially-varying
  gamma exponent so the center and the exploring edge render differently:

  ```math
  \text{trail}_{\text{shaped}} = \text{trail}^{\,0.5+0.4d}
  ```

  ($d$ = normalized distance from center), followed by a paper→ink→edge
  color lerp and grain overlay.
* **Known failure mode**: every branch event hits the same atomic counter
  (`alloc_counter`), serializing what should be parallel GPU threads
  during a branching burst.

### `v3.0` — Space Colonization Graph (SCA)
* **Commit**: `a8923be`
* **Basis**: Runions et al. (2005) Space Colonization Algorithm — a full
  departure from continuum agents; the "plant" is a discrete graph.
* **Mechanics**: 12,000 attractor points pull the nearest graph node
  within `attract_dist = 50px`; a pulled node grows one `step_size = 3.5px`
  step toward the normalized sum of pull vectors:

  ```math
  \vec v_{\text{grow},j} = \mathrm{normalize}\left(\sum_{k\in N_j}\frac{\vec A_k-\vec N_j}{\lVert\vec A_k-\vec N_j\rVert}\right)
  ```

  An attractor is consumed once a node comes within `kill_dist = 6px`.
* **Result**: structurally guarantees a trunk-to-capillary hierarchy —
  sponge collapse is impossible by construction, since there's no field
  to saturate.
* **Known limitation**: nearest-node search is a brute-force
  `O(N_attrs · N_nodes)` nested loop, no spatial partitioning yet
  (that's what v3.1 adds).

### `v3.1` — Spatial-Hashed Optimization, Retrograde Pruning & GGUI HUD
* **Commit**: `7bda3a1`
* **Basis**: builds on v3.0's SCA graph, adding flow-based pruning
  conceptually aligned with Tero & Nakagaki's (2010) adaptive-network
  work, plus spatial partitioning for the nearest-node search.
* **What the release notes confirm**:
  * A spatial hash replaces v3.0's brute-force search, addressing that
    exact `O(N_attrs · N_nodes)` bottleneck.
  * Contact with a food point triggers a **retrograde flow** signal that
    propagates back through parent links, reinforcing the path that led
    to food.
  * Dead-end filaments that aren't carrying reinforced flow undergo
    **Poiseuille-style pruning** and dissolve within roughly 3–5 seconds.
  * An in-viewport **Dear ImGui HUD** shows live node/state metrics, and
    an idle "breathing circle" indicates where to inoculate before
    placement.
* **Not yet confirmed** — flagged rather than guessed: the release notes
  don't give exact constants (hash cell size, max nodes per cell, the
  flow threshold that counts as "stagnant," the per-step vitality decay
  rate, or the retrograde-flux increment). If you want those numbers in
  the README, they need to come from the actual `v3.1` source rather than
  from a prior draft — an earlier version of this document had specific
  numbers here (40×40px cells, `MAX_NODES_PER_CELL=128`, `flow<0.16`,
  `vitality -= 0.0035`, `Φ_parent += 0.55`) that don't appear anywhere in
  your release notes, so I've removed them rather than repeat
  unverifiable figures.
* **Controls**: RMB inoculates a mother-mass at the cursor; LMB drops a
  food point; Space pauses/resumes; `R` resets the simulation.
* **Requirements**: Python 3.8–3.12, NVIDIA GPU with CUDA (4GB+ VRAM
  recommended), `pip install taichi`.

### `v4.0` — Dual-Channel Delayed Blending
* **Commit**: `e4941c2`
* **Basis**: a close translation of `nicoptere/physarum`'s WebGL/GLSL
  source.
* **Scale**: 512×512 = 262,144 agents (matching the original's
  texture-based particle count), 800×800 grid.
* **Two channels instead of one field** — **R** (this frame's rasterized
  positions) and **G** (temporally-blended memory), asymmetrically mixed:

  ```math
  G_{t+1}(\vec x) = \gamma\sum_{\vec\delta}\tfrac{1}{9}\left[R_t(\vec x+\vec\delta) + 0.5\,G_t(\vec x+\vec\delta)\right]
  ```

  $\gamma=0.90$, summed over the 3×3 neighborhood.
* **Everything operates in normalized UV space**, not pixels — sensing,
  step size, and the toroidal wrap ($\vec x_{\text{wrapped}}=\mathrm{fract}(\vec x)$)
  all happen in $[0,1]^2$, mirroring the original shader's texture-space
  design.
* **Randomness**: turning ties are broken by a hash function replicating
  the original GLSL `rand()`, not Taichi's built-in RNG.
* **Interaction**: holding LMB injects a 1200-agent burst around the
  cursor each frame.

### `v5.0` — Multi-Species Antagonism & 2.5D Relief
* **Commit**: `86e1294`
* **Basis**: Michael Fogleman's (`fogleman/physarum`) multi-species model,
  itself referencing Sage Jenson's `pkg/physarum/model.go`, combined with
  2.5D normal-mapped shading.
* **Scale**: 600,000 agents, 2 species, 800×800 grid, toroidal
  (pixel-space) wrap.
* **Species interaction tensor** — each species senses a weighted mix of
  both density fields:

  ```math
  M = \begin{pmatrix} +1.0 & -0.4 \\ -0.4 & +1.0 \end{pmatrix}
  ```

  Positive diagonal terms drive same-species cohesion; negative
  off-diagonal terms push species apart, tearing open gaps and keeping
  boundaries sharp.
* **Two blur passes before decay** — decay ($0.93$) applies only after
  the second pass, building a wider field of influence without
  over-evaporating it.
* **2.5D shading**: combined density $H(\vec x)=\sum_c\text{grid}_c(\vec x)$
  treated as a height field, finite-difference normal, Lambertian lighting:

  ```math
  \vec N = \mathrm{normalize}(-k\,\partial_xH,\ -k\,\partial_yH,\ 1.0)
  ```
  ```math
  \text{Shading} = I_{\text{ambient}} + I_{\text{diffuse}}\max(0,\vec N\cdot\vec L)
  ```
* **Interaction**: unlike every field-based version's "place food"
  metaphor, LMB here forcibly reverses the heading of agents within 30px
  of the cursor and injects density — a disruption tool, not a nutrient
  source. `R` reseeds.

---

### Comparative Overview

| Tag | Architecture | Basis | Boundary | Visual result |
| :--- | :--- | :--- | :--- | :--- |
| **v1.0** | Agent–field PDE | Jeff Jones (2010) | Hard clamp + random re-heading | Dense mesh, boundary-ring artifact |
| **v2.0** | Generational tip-growth | `ianpilon` | Bounding-box clamp | Botanical paper/ink, atomic-counter bottleneck |
| **v3.0** | Discrete SCA graph | Runions et al. (2005) | Open substrate | True trunk-to-capillary hierarchy, no sponge risk |
| **v3.1** | Spatial-hashed SCA graph | Tero & Nakagaki (2010)-aligned pruning | Circular | Dynamic pruning + retrograde reinforcement |
| **v4.0** | Dual-channel ping-pong | `nicoptere/physarum` | Toroidal wrap | High-contrast smooth cords, UV-space fidelity |
| **v5.0** | Multi-vector field | Fogleman / Jenson | Toroidal wrap | Lit, 3D-looking competing colonies |

---

### Setup & Quick Start

```bash
pip install taichi numpy
```

```bash
git checkout v1.0 && python main.py   # dense swarm, continuum field
git checkout v2.0 && python main.py   # generational tip-growth
git checkout v3.0 && python main.py   # space colonization graph
git checkout v3.1 && python main.py   # + spatial hashing, retrograde pruning, GGUI HUD
git checkout v4.0 && python main.py   # dual-channel delayed blending
git checkout v5.0 && python main.py   # multi-species + 2.5D relief
```

**v3.1 controls**: RMB inoculate · LMB drop food · Space pause/resume · `R` reset
**v3.1 requirements**: Python 3.8–3.12, NVIDIA GPU (CUDA, 4GB+ VRAM recommended)

---

### Credits & References

* **Jeff Jones (2010)**. *Characteristics of Pattern Formation and
  Evolution in Approximations of Physarum Polycephalum*. Artificial
  Life, 16(2), 127–153.
* **Adam Runions et al. (2005)**. *Modeling and Visualization of Leaf
  Venation Patterns*. ACM TOG, 24(3), 702–711.
* **Atsushi Tero & Toshiyuki Nakagaki et al. (2010)**. *Rules for
  Biologically Inspired Adaptive Network Design*. Science, 327(5964),
  439–442.
* **`ianpilon`** — generational tip-growth reference for v2.0.
* **Nicolas Barradeau (`nicoptere/physarum`)**.
  <https://github.com/nicoptere/physarum>
* **Michael Fogleman (`fogleman/physarum`)**.
  <https://github.com/fogleman/physarum>
* **Sage Jenson**. <https://sagejenson.com/physarum>

---

<a id="chinese-documentation"></a>
## 中文说明

基于 Taichi 的多头绒泡菌（*Physarum polycephalum*）形态发生与自适应输运网络模拟，支持 CUDA / Vulkan 后端。

两条技术路线，六个 tag：

```
                         [ 黏菌形态发生问题空间 ]
                                    │
             ┌─────────────────────┴─────────────────────┐
             ▼                                             ▼
    [ 连续场路线 ]                                  [ 离散图路线 ]
  (Jones 2010 / Barradeau / Fogleman)          (Runions 2005 / Tero-Nakagaki 2010)
             │                                             │
    ├─► v1.0: 密集粒子流 + 原生质穿梭流                  ├─► v3.0: 空间殖民图算法 (SCA)
    ├─► v2.0: 显式代际顶芽分叉 (ianpilon)                └─► v3.1: 空间哈希加速 + 逆行剪枝
    ├─► v4.0: 双通道延迟混合 (nicoptere)                           + GGUI HUD
    └─► v5.0: 多物种拮抗 + 2.5D 立体光影 (fogleman / Jenson)
```

### `v1.0` — 密集粒子流与原生质穿梭流
* **对应 Commit**：`2af9244`
* **理论基础**：Jeff Jones（2010）多智能体化学趋向性输运模型。
* **规模**：1024×1024 网格，120 万粒子。
* **转向规则** —— 不是一个干净的 sign 函数，是和源码一致的四分支判断：

  ```math
  \theta_{t+1} = \begin{cases}
  \theta_t & F > L \text{ 且 } F > R \\
  \theta_t \pm RA\ (\text{随机符号}) & F < L \text{ 且 } F < R \\
  \theta_t - RA & L > R \\
  \theta_t + RA & R > L
  \end{cases}
  ```

  探测距离 $SS=3.5\text{px}$，探针半角 $22.5^\circ$，转向步长 $RA=45^\circ$。（源码里变量名 `SO`、`SA` 和实际功能是反的，这里按功能描述，不按变量名。）
* **穿梭流调制的是分泌量，不是速度**，进食/非进食用不同系数：

  ```math
  \text{分泌量} = \begin{cases}
  D_0\,(2.5 + 1.5\sin(2t - 0.015(x{+}y)\,s)) & \text{正在进食} \\
  D_0\,(0.8 + 0.3\sin(2t - 0.015(x{+}y)\,s)) & \text{其他情况}
  \end{cases}
  ```

  $D_0=4.0$，$s=\mathrm{sign}(\sin(0.15t))$。
* **边界**：硬性钳位位置 + 完全重新随机化朝向，不是反弹。这就是边缘那圈亮环伪影的直接成因。
* **渲染**：双段非线性显色——低透明度"边缘细丝"项负责微弱毛细结构，三次方"核心"项负责饱和主干。

### `v2.0` — 显式代际顶芽分叉
* **对应 Commit**：`d48f330`
* **理论基础**：改编自 `ianpilon` 的 Processing/JS 顶芽生长实现，配合植物标本档案渲染。
* **架构**：粒子带显式代际计数与有限寿命，通过原子计数器在固定大小环形缓冲区分配。
* **分叉概率随代际衰减**：

  ```math
  P_{\text{branch}}(g) = f_{ch}\cdot f_{acc}^{\,g}
  ```
* **子代的分泌强度与寿命各自独立衰减**：

  ```math
  D_{\text{child}} = D_{\text{parent}}\cdot v_{tap} \qquad
  L_{\text{child}} = L_{\text{base}}\cdot l_{tap}^{\,g+1}
  ```
* **渲染**：四阶 FBM 纸纤维噪声，加一个随半径变化的伽马指数，让中心和探索前沿显色不同：

  ```math
  \text{trail}_{\text{显色}} = \text{trail}^{\,0.5+0.4d}
  ```

  （$d$ 为到中心的归一化距离），之后做纸色→墨色→边缘色的三段混合叠加颗粒噪声。
* **已知失效模式**：每次分叉都命中同一个原子计数器（`alloc_counter`），分叉爆发时会把本该并行的 GPU 线程串行化。

### `v3.0` — 空间殖民拓扑图 (SCA)
* **对应 Commit**：`a8923be`
* **理论基础**：Runions et al.（2005）空间殖民算法——彻底脱离连续场粒子，"植株"本身就是一个离散图。
* **机制**：12,000 个吸引子在 `attract_dist = 50px` 内牵引最近的图节点；被牵引节点朝拉力向量归一化和的方向生长一步（`step_size = 3.5px`）：

  ```math
  \vec v_{\text{grow},j} = \mathrm{normalize}\left(\sum_{k\in N_j}\frac{\vec A_k-\vec N_j}{\lVert\vec A_k-\vec N_j\rVert}\right)
  ```

  节点进入 `kill_dist = 6px` 后，对应吸引子被消耗。
* **效果**：结构上就保证主干到毛细血管的层级——没有可饱和的场，海绵化在数学上不可能发生。
* **已知局限**：最近节点搜索是暴力 `O(N_attrs · N_nodes)` 嵌套循环，还没做空间分区（这是 v3.1 加的）。

### `v3.1` — 空间哈希加速、逆行剪枝与 GGUI HUD
* **对应 Commit**：`7bda3a1`
* **理论基础**：在 v3.0 的 SCA 图基础上，加入概念上对齐 Tero & Nakagaki（2010）自适应网络设计的流量剪枝机制，并给最近节点搜索做空间分区加速。
* **release notes 确认的内容**：
  * 空间哈希取代了 v3.0 的暴力搜索，解决的正是那个 `O(N_attrs · N_nodes)` 瓶颈。
  * 接触到食物点会触发**逆行通量**信号，沿父节点链路向回传播，强化那条通向食物的路径。
  * 没有携带被强化流量的盲端细丝会触发**泊肃叶式剪枝**，大约 3~5 秒内溶解消失。
  * 视口内的 **Dear ImGui HUD** 实时显示节点/状态指标，未接种时有一个"呼吸环"提示放置位置。
* **未经确认、明确标注出来而不是瞎编**：release notes 没给出具体常数——哈希网格的单元大小、每个单元最多容纳的节点数、"停滞"对应的具体流量阈值、每步的活性衰减率、逆行通量的增量。如果你要把这些数字写进 README，需要从 v3.1 真实源码里拿，而不是延续之前某版草稿里的数字——早前那版文档里写的 40×40px 网格、`MAX_NODES_PER_CELL=128`、`flow<0.16`、`vitality -= 0.0035`、`Φ_parent += 0.55`，在你贴的 release notes 里根本找不到对应内容，所以这版我直接删掉了，没有照抄。
* **交互**：鼠标右键接种母核；左键投放食物点；空格暂停/继续；`R` 重置。
* **运行环境**：Python 3.8~3.12，NVIDIA GPU（CUDA，建议 4GB+ 显存），`pip install taichi`。

### `v4.0` — 双通道延迟混合
* **对应 Commit**：`e4941c2`
* **理论基础**：对照 `nicoptere/physarum` 的 WebGL/GLSL 源码贴近实现。
* **规模**：512×512=262,144 个粒子（对应原版纹理粒子数），800×800 网格。
* **双通道取代单一场**——**R 通道**（当前帧粒子光栅化位置）与 **G 通道**（时间平滑记忆），非对称混合：

  ```math
  G_{t+1}(\vec x) = \gamma\sum_{\vec\delta}\tfrac{1}{9}\left[R_t(\vec x+\vec\delta) + 0.5\,G_t(\vec x+\vec\delta)\right]
  ```

  $\gamma=0.90$，对 3×3 邻域求和。
* **全程使用归一化 UV 坐标**——感知、步长、环形回绕（$\vec x_{\text{wrapped}}=\mathrm{fract}(\vec x)$）都在 $[0,1]^2$ 里，贴合原始 shader 的纹理空间设计。
* **随机数**：转向平局用哈希函数复刻原版 GLSL 的 `rand()`，不是 Taichi 内置随机数。
* **交互**：按住左键每帧在光标附近注入 1200 个粒子。

### `v5.0` — 多物种拮抗与 2.5D 立体光影
* **对应 Commit**：`86e1294`
* **理论基础**：Michael Fogleman（`fogleman/physarum`）的多物种模型，该项目本身参考了 Sage Jenson 的 `pkg/physarum/model.go`，结合 2.5D 法线渲染。
* **规模**：60 万粒子，2 个物种，800×800 网格，环形（像素空间）回绕。
* **物种交互张量**——每个物种感知的是两个密度场的加权组合：

  ```math
  M = \begin{pmatrix} +1.0 & -0.4 \\ -0.4 & +1.0 \end{pmatrix}
  ```

  对角线正值驱动同物种聚合；非对角线负值把物种互相推开，撕开间隙，边界保持锐利。
* **先两遍模糊后衰减**——衰减（0.93）只在第二遍之后应用一次，先扩展势场范围再统一蒸发。
* **2.5D 光影**：总密度场 $H(\vec x)=\sum_c\text{grid}_c(\vec x)$ 当高度图，有限差分求法线，兰伯特光照：

  ```math
  \vec N = \mathrm{normalize}(-k\,\partial_xH,\ -k\,\partial_yH,\ 1.0)
  ```
  ```math
  \text{光照} = I_{\text{环境}} + I_{\text{漫反射}}\max(0,\vec N\cdot\vec L)
  ```
* **交互**：和其他连续场版本"投放食物"的思路不同，这里左键会强制反转光标 30 像素范围内粒子的朝向并注入密度——是扰动工具，不是养分源。`R` 重新接种。

---

### 对比总览

| 版本 | 架构 | 理论基础 | 边界 | 视觉结果 |
| :--- | :--- | :--- | :--- | :--- |
| **v1.0** | 智能体-场 PDE | Jeff Jones (2010) | 硬钳位+朝向重随机化 | 密集网格，边界亮环伪影 |
| **v2.0** | 代际顶芽生长 | `ianpilon` | 边界盒钳位 | 植物标本纸感，原子计数器瓶颈 |
| **v3.0** | 离散 SCA 图 | Runions et al. (2005) | 开放基质 | 真正的主干-毛细层级，不会海绵化 |
| **v3.1** | 空间哈希 SCA 图 | 对齐 Tero & Nakagaki (2010) 的剪枝思路 | 圆形 | 动态剪枝 + 逆行强化 |
| **v4.0** | 双通道 Ping-Pong | `nicoptere/physarum` | 环形回绕 | 高对比度平滑管道，UV 空间高保真 |
| **v5.0** | 多物种场 | Fogleman / Jenson | 环形回绕 | 有光影层次的多物种竞争群落 |

---

### 环境配置与快速开始

```bash
pip install taichi numpy
```

```bash
git checkout v1.0 && python main.py   # 密集粒子流，连续场
git checkout v2.0 && python main.py   # 代际顶芽分叉
git checkout v3.0 && python main.py   # 空间殖民拓扑图
git checkout v3.1 && python main.py   # + 空间哈希、逆行剪枝、GGUI HUD
git checkout v4.0 && python main.py   # 双通道延迟混合
git checkout v5.0 && python main.py   # 多物种拮抗 + 2.5D 立体光影
```

**v3.1 交互**：右键接种 · 左键投放食物 · 空格暂停/继续 · `R` 重置
**v3.1 运行环境**：Python 3.8~3.12，NVIDIA GPU（CUDA，建议 4GB+ 显存）

---

### 致谢与参考文献

* **Jeff Jones (2010)**. *Characteristics of Pattern Formation and Evolution in Approximations of Physarum Polycephalum*. Artificial Life, 16(2), 127–153.
* **Adam Runions et al. (2005)**. *Modeling and Visualization of Leaf Venation Patterns*. ACM TOG, 24(3), 702–711.
* **Atsushi Tero & Toshiyuki Nakagaki et al. (2010)**. *Rules for Biologically Inspired Adaptive Network Design*. Science, 327(5964), 439–442.
* **`ianpilon`** —— v2.0 代际顶芽生长的参考实现。
* **Nicolas Barradeau (`nicoptere/physarum`)**. <https://github.com/nicoptere/physarum>
* **Michael Fogleman (`fogleman/physarum`)**. <https://github.com/fogleman/physarum>
* **Sage Jenson**. <https://sagejenson.com/physarum>
