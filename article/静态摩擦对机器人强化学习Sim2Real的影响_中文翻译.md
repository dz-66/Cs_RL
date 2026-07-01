# 静态摩擦对机器人强化学习中Sim2Real的影响

**作者：** Xiaoyi Hu, Qiao Sun, Bailin He, Haojie Liu, Xueyi Zhang, Chunpeng Lu, Jiangwei Zhong

---

## 摘要

在机器人强化学习中，Sim2Real差距仍然是一个关键挑战。然而，静态摩擦对Sim2Real的影响尚未得到充分探索。传统的域随机化方法通常将静态摩擦排除在其参数空间之外。在我们的机器人强化学习部署中，我们观察到常用的域随机化参数在我们的机器人上表现不佳。为解决这一问题，我们首先采用执行器网络（Actuator Net）进行模型迁移。虽然这种方法使机器人能够行走，但其能力仅限于平坦地面。随后，我们分析了关节的物理参数，发现与常用机器人不同，我们的机器人具有较高的静摩擦-扭矩比，导致静态摩擦成为影响Sim2Real性能的重要因素。为缓解这一问题，我们在域随机化中引入了静态摩擦。然而，直接引入静态摩擦会导致策略网络训练停滞。我们提出了一种简单的策略来克服这一训练困难。引入随机化静态摩擦使机器人能够在平坦地面上稳定行走并成功爬楼梯。我们推测，具有高静摩擦-扭矩比的机器人更可能遭遇Sim2Real差距，因此在其训练过程中应考虑静态摩擦的影响。

---

## 一、引言

深度强化学习（DRL）已越来越多地应用于机器人领域 [1]–[5]。然而，Sim2Real（从仿真到现实）的迁移仍然是阻碍其在现实世界中应用的主要障碍之一。

为解决这一问题，研究人员开发了各种仿真环境来训练强化学习策略，如Isaac Gym [6]、MuJoCo、Webots等。尽管这些仿真器提供了高效的训练平台，但仿真与现实之间的固有差距仍然存在 [7]、[8]。

域随机化（Domain Randomization）是弥补这一差距的常用技术 [9]。通过对仿真物理参数（如质量、摩擦和观测噪声）进行随机化，在多样化条件下训练的策略可以更好地泛化到现实世界。然而，传统的域随机化通常忽略了静态摩擦，只关注粘性摩擦（阻尼）和惯性等参数。

在我们的工作中，我们发现静态摩擦对Sim2Real迁移有着深远的影响，尤其是在具有高静摩擦-扭矩比的机器人上。本文的主要贡献包括：

1. 识别了静态摩擦是导致Sim2Real差距的关键因素。
2. 提出了一种在域随机化中引入静态摩擦的方法，并克服了由此带来的训练困难。
3. 通过Sim2Sim和Sim2Real实验验证了我们方法的有效性。

---

## 二、相关工作

### A. 机器人强化学习中的Sim2Real

Sim2Real迁移是机器人强化学习中的一个基本挑战。已有多种方法被提出来解决这一问题，包括域随机化 [9]、域自适应和系统辨识等。域随机化因其简单性和有效性而被广泛采用，其核心思想是在仿真中对物理参数进行随机化，使策略对现实世界的变化具有鲁棒性。

### B. 四足机器人运动控制

四足机器人的运动控制已被广泛研究。从传统的基于模型的控制方法到基于学习的方法。近期，RMA（快速运动适应）[20] 等算法展示了令人印象深刻的Sim2Real迁移能力，使机器人能够在复杂地形上行走。然而，这些工作通常假设静态摩擦对系统行为的影响可以忽略不计。

### C. 执行器建模

准确的执行器建模对Sim2Real迁移至关重要。一些研究提出了使用神经网络学习执行器动力学的方法，即执行器网络（Actuator Net）。这种方法通过从真实机器人收集数据来学习仿真到现实的映射。

---

## 三、方法

### A. 概述

我们的方法遵循教师-学生训练范式，类似于RMA [20]。训练过程分为两个阶段：

1. **教师训练阶段：** 教师网络在仿真中使用特权信息（如地形高程图）进行训练。
2. **学生训练阶段：** 学生网络仅使用本体感知信息（如关节位置、速度和IMU数据）模仿教师的行为。

### B. 教师训练阶段

教师策略 $\pi$ 接收本体感知观测 $o_t^{prio}$ 和从高程图编码的地形特征 $z_t^*$ 作为输入，输出动作 $a_t^*$：

$$a_t^* = \pi(o_t^{prio}, z_t^*)$$

教师训练阶段的示意图如图1所示。

> 图1. 教师训练阶段。教师网络接收本体感知观测和高程图编码的地形特征，输出最优动作。

教师使用PPO（近端策略优化）算法进行训练，奖励函数鼓励机器人以特定速度前进，同时保持稳定。

### C. 学生训练阶段

在学生训练阶段，学生网络 $f_{student}$ 仅接收本体感知观测历史 $o_t^H$ 作为输入：

$$o_t^H = \{o_{t-H}, \ldots, o_t\}$$

由于学生无法获取周围的高程图，因此不能直接提取地形特征。为解决这一问题，我们用从历史编码器 $f_{hist}$ 导出的特征替代教师高程编码器提取的地形特征 $z_t^*$。这样，学生可以基于本体感知观测历史 $o_t^H$ 来估计地形特征。

学生网络的详细训练过程如图4所示。

> 图4. 学生阶段的机器人训练过程。在此阶段，学生学习模仿教师的动作。

### D. 执行器训练阶段

执行器网络 $f_{motor}$ 接收关节位置历史 $\theta_t^H$ 和关节速度历史 $\dot{\theta}_t^H$，预测关节扭矩 $\tau_t^{motor}$：

$$\tau_t^{motor} = f_{motor}(\theta_t^H, \dot{\theta}_t^H)$$

其中：
$$\theta_t^H = \{\theta_t, \ldots, \theta_{t-H}\}$$
$$\dot{\theta}_t^H = \{\dot{\theta}_{t-1}, \ldots, \dot{\theta}_{t-H}\}$$

这里 $H=3$。执行器网络的损失函数 $L_{actuator}$ 定义为：

$$L_{actuator} = \text{MSE}(\tau_t^{motor}, \tau_t^{motor*})$$

其中 $\tau_t^{motor}$ 是执行器网络预测的关节扭矩，$\tau_t^{motor*}$ 是机器人运行期间测量的实际关节扭矩。

---

## 四、Sim2Real 实现方法

上述过程侧重于仿真环境中的网络训练。然而，机器人强化学习还需要考虑模型向真实机器人的迁移效果。在这一过程中，我们观察到常用的域随机化参数在我们的机器人上表现不佳，往往导致无法正常行走。为分析和解决这一问题，我们首先使用执行器网络 $f_{motor}$ 进行模型迁移。

### A. 基于执行器网络的Sim2Real

执行器网络 $f_{motor}$ 的核心思想是从真实执行器收集数据，模拟执行器模型，然后在仿真环境中使用该模型进行训练。为收集执行器网络 $f_{motor}$ 的数据，我们首先使用传统域随机化参数训练机器人，并将训练好的模型部署到物理机器人上。

部署后，我们发现机器人只能向后行走，而向前行走会导致摔倒。我们将向后行走的数据作为训练执行器网络 $f_{motor}$ 的数据集。

训练执行器网络后，我们在仿真环境中用执行器网络替换原有的PD控制器，并重新部署模型。这使机器人能够在平坦地面上正常向前行走。

基于执行器网络的Sim2Real过程如图5所示。

> 图5. 基于执行器网络的Sim2Real实现。当使用学生网络 $f_0^{stu}$ 配合执行器网络时，机器人在向前行走时频繁摔倒，需要使用安全绳。在收集 $f_0^{stu}$ 的行走数据后，使用该数据训练执行器网络 $f_{motor}$。然后使用 $f_{motor}$ 进行新一轮教师-学生训练，得到新的学生网络 $f_1^{stu}$。基于 $f_1^{stu}$ 的行走虽然稳定，但仅限于平坦地面。

然而，这种方法只能使机器人在平坦地面上稳定行走，无法爬楼梯。我们推测这种限制是因为执行器网络训练数据仅包含平坦地面上的向后行走。此外，由于执行器网络数据是从单个机器人收集的，我们推测训练好的模型可能仅适用于该特定机器人。不同机器人之间关节的差异可能导致部署失败。

---

## 五、基于控制理论分析的机器人关节域随机化

由于使用执行器网络的Sim2Real实现只能在平坦地面上行走，且因迭代改进显著增加了训练时间，我们分析了常用域随机化参数导致我们机器人失败的原因。我们系统地检查了关节的物理参数，并基于此分析提出了改进方案。为便于分析关节参数，我们将域随机化过程中单个关节 $j$ 的动力学建模如下：

$$I_j \ddot{\theta}_j(t) + B_j \dot{\theta}_j(t) = \tau_j(t) + f_j(t)$$

$$f_j(t) = \begin{cases} -b_j^c, & \text{if } \dot{\theta}_j(t) > 0 \\ b_j^c, & \text{if } \dot{\theta}_j(t) < 0 \end{cases}$$

$$\tau_j(t) = k_{motor} \left[k_p(\theta_j(t) - a_t) - k_d \dot{\theta}_j(t)\right]$$

其中，$I_j$、$B_j$ 和 $b_j^c$ 分别表示关节惯量、粘性摩擦和静态摩擦系数。$\ddot{\theta}_j(t)$、$\dot{\theta}_j(t)$、$\theta_j(t)$、$\tau_j(t)$ 和 $f_j(t)$ 分别表示 $t$ 时刻的关节加速度、关节速度、关节位置、PD控制器输出的关节扭矩和静态摩擦力。$k_{motor}$ 是电机强度因子，$k_p$ 和 $k_d$ 是PD控制器的比例和微分增益。虽然现实中静摩擦通常大于动摩擦，为简化起见，我们假设它们相等。

在传统域随机化中，$I_j$、$B_j$、$k_{motor}$、$k_p$ 和 $k_d$ 等参数被随机化。从动力学方程可以看出，$k_d$ 和 $B_j$ 可以相互抵消，因此只需随机化其中之一。此外，静态摩擦 $f_j(t)$ 在传统方法中通常不被随机化。然而，我们的实验表明，静态摩擦 $f_j(t)$ 对机器人性能有显著影响。为分析 $f_j(t)$ 的大小，我们使用以下方法对电机的物理参数进行辨识：

$$\theta_j^*(t) = A \sin(\omega t)$$

$$\min_f (I_j, B_j, b_j^c) = \frac{1}{N} \sum_{i=0}^{N} \left(\theta_j^*(t_i) - \theta_j(t_i)\right)^2$$

$$\text{subject to } I_j > 0, B_j > 0, b_j^c > 0$$

其中，$\theta_j^*(t)$ 是激励信号，$A$ 是其幅值，$\omega$ 是其频率，$\theta_j(t_i)$ 是采样的关节角度，$\theta_j^*(t_i)$ 是激励信号的真实值。我们使用最小二乘法辨识 $I_j$、$B_j$ 和 $b_j^c$。参数辨识结果在实验部分展示。

参数辨识完成后，我们将静态摩擦 $f_j(t)$ 引入域随机化过程。然而，$f_j(t)$ 的引入带来了非线性，显著增加了训练难度。我们观察到，直接将辨识出的静态摩擦 $f_j(t)$ 纳入机器人或在RL生成的动作 $a_t$ 中手动补偿它，都会导致机器人在训练期间保持静止。

我们推测这是由于早期RL动作的混沌性质，静态摩擦充当了强滤波器。在预训练模型（未使用静态摩擦训练）上微调并引入 $f_j(t)$ 也失败了，可能是因为模型权重已经陷入局部最优或鞍点，无法进行有效的梯度下降。

为训练能够处理随机化静态摩擦 $f_j(t)$ 的机器人，我们开发了两种方法：**迭代法**和**欺骗法**。在迭代法中，我们多次微调教师模型（未使用静态摩擦训练），逐渐引入少量 $f_j(t)$。然而，用这种方法训练的模型在真实部署中出现严重抖动，甚至难以稳定站立。作为替代方案，我们提出了欺骗法，即不再追求仿真与现实静态摩擦的完美对齐，而是显著扩大 $f_j(t)$ 的随机化范围。这种方法成功地实现了真实部署。

我们将扩大静态摩擦随机化范围的有效性归因于两个因素：
1. 在仿真训练期间，扩大的随机化使机器人能够遇到静态摩擦可忽略不计的场景，降低了训练难度。
2. 在Sim2Real过程中，真实世界关节的静态摩擦可能因磨损而变化。扩大随机化范围增强了机器人对机械磨损的鲁棒性。

我们域随机化中使用的参数列于表2。

**表2：域随机化参数**

| 参数 | 范围 | 单位 |
|------|------|------|
| 关节电枢 | [0.8, 1.2] | 乘数 |
| 关节阻尼 | [0.8, 1.2] | 乘数 |
| 关节静态摩擦 | [0.0, 1.2] | 乘数 |
| Kp | [0.95, 1.05] | 乘数 |
| 电机强度 | [0.8, 1.2] | 乘数 |
| 地面摩擦 | [0.2, 2.0] | 乘数 |
| 有效载荷 | [-2, 3] | kg |
| 质心偏移 | [-0.25, 0.25] | m |
| 推力间隔 | 8 | s |
| 推力速度 | 1 | m/s |

"推力速度"指的是随机化基座的线速度以模拟外部推力，从而无需确定可行的推力大小。

> 图6. Saturn Lite使用的电机。防水电机（IP67）以及移除密封圈后的电机。

---

## 六、实验结果

在本节中，我们展示关节物理参数的辨识结果，解释为什么静态摩擦对我们的机器人有显著影响，并分析哪些机器人可能面临类似问题。此外，我们比较了在Sim2Sim和Sim2Real场景下，无静态摩擦的域随机化、执行器网络和含静态摩擦的域随机化的性能。

### A. 关节等效转子惯量、阻尼和静态摩擦的测量与评估

为辨识关节的转子惯量、阻尼和静态摩擦，我们使用公式13估计电机的物理参数。为简化起见，我们关注Go1和Saturn Lite机器人的小腿电机在空载条件下的表现。辨识结果汇总于表3。

**表3：小腿电机对比**

| 属性 | 类型 | Go1 | Saturn Lite（我们的） |
|------|------|-----|----------------------|
| 惯量 (kg·m²) | 均值 | 0.0121 | 0.0145 |
| | 标准差 | 0.00223 | 4.21×10⁻⁴ |
| 粘性摩擦 (N·m·s/rad) | 均值 | 0.0342 | 0.0704 |
| | 标准差 | 0.00229 | 0.0272 |
| 静态摩擦 (N) | 均值 | 0.0481 | 0.442 |
| | 标准差 | 0.00299 | 0.0661 |
| f/τ_max 比率 (%) | 均值 | 0.13% | 0.98% |
| | 标准差 | 5.70×10⁻⁵ | 2.18×10⁻⁴ |

在大多数机器人强化学习部署中，静态摩擦的影响似乎可以忽略不计。我们认为这是因为在大多数机器人中，静态摩擦仅占关节扭矩的极小部分，因此不显著。然而，在我们的机器人中，静态摩擦占关节扭矩的比例异常高，导致明显的影响。这可能归因于我们机器人使用的防水处理和更重的连杆，尽管具有相似的扭矩能力。我们机器人系统中使用的防水关节电机（如图6所示）通过实验移除密封圈后静摩擦降低了70%。为在实际应用中保持机器人功能，我们在机器人组装时保留了带密封圈的防水关节配置，保留了其固有的静态摩擦效应。

为评估静态摩擦对机器人性能的影响，我们在Sim2Sim和Sim2Real场景中进行了一系列对比实验。结果如下。

### B. 有无静态摩擦的Sim2Sim与Sim2Real对比

我们使用三种不同方法进行了对比实验：执行器网络、域随机化（无静态摩擦）和域随机化（含静态摩擦）。对于Sim2Sim实验，我们使用Webots作为仿真平台。实验结果如图7所示。

> 图7. Sim2Sim与Sim2Real实验对比。实验重点包括平坦地面行走和爬楼梯场景。

在Sim2Sim实验中，当使用无静态摩擦的域随机化时，机器人在Webots中取得了最佳性能。然而，使用执行器网络时，机器人出现明显的异常行为，最终导致系统失败。使用包含静态摩擦的域随机化时，机器人表现出不规则的抖动和跳跃，但仍能上下楼梯。这主要归因于RMA算法能够适应不同的静态摩擦水平。

在Sim2Real实验中，无静态摩擦的域随机化使机器人能够在平坦地面上向后行走，但向前行走会导致摔倒。使用执行器网络使机器人能够向前行走（尽管不稳定），但无法爬楼梯。相比之下，含静态摩擦的域随机化实现了在平坦地面上的稳定行走和成功的楼梯导航。

无静态摩擦的域随机化在Sim2Sim实验中的成功表明，我们的强化学习训练算法在部署过程中不存在输入-输出接口配置错误。然而，其在Sim2Real实验中的失败表明仿真与现实之间存在显著差异。执行器网络在Sim2Real平坦地面行走中表现良好但在Sim2Sim中失败的事实进一步支持了这种差异源于仿真与现实之间关节动力学差异的假设。基于我们之前的控制理论分析，我们假设静态摩擦是这种差异的主要原因。为验证这一假设，我们实现了含静态摩擦的域随机化。实验结果表明，这种方法显著提高了Sim2Real性能，尽管Sim2Sim性能略有牺牲。鉴于其对不同静态摩擦的强大适应性，我们认为该方法不仅能补偿机器人强化学习中的关节摩擦，还可以作为对抗关节磨损和疲劳的有效手段。

---

## 七、结论

我们的实验表明，静态摩擦对机器人强化学习的Sim2Real有显著影响。为减轻这种影响，我们提出将静态摩擦纳入训练过程。尽管静态摩擦的引入会导致模型训练失败，但我们提出了一个简单的技巧来解决这个问题。此外，我们推测使用具有较低静摩擦-扭矩比的关节可以进一步缓解这一问题。

随着时间推移，机器人不可避免地会经历机械疲劳和磨损，导致静态摩擦发生变化。虽然使用随机化静态摩擦训练的RL模型在理论上对这些退化具有鲁棒性，但我们的实验条件限制了对这一假设的直接验证。我们认为这一主题值得进一步研究。

---

## 参考文献

[1] S. Gu, E. Holly, T. P. Lillicrap, and S. Levine, "Deep reinforcement learning for robotic manipulation," *arXiv preprint arXiv:1610.00633*, vol. 1, p. 1, 2016.

[2] T. M. Moerland, J. Broekens, A. Plaat, C. M. Jonker et al., "Model-based reinforcement learning: A survey," *Foundations and Trends® in Machine Learning*, vol. 16, no. 1, pp. 1–118, 2023.

[3] Q. Huang, "Model-based or model-free, a review of approaches in reinforcement learning," in *2020 International Conference on Computing and Data Science (CDS)*. IEEE, 2020, pp. 219–221.

[4] D. Ernst, M. Glavic, F. Capitanescu, and L. Wehenkel, "Reinforcement learning versus model predictive control: a comparison on a power system problem," *IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics)*, vol. 39, no. 2, pp. 517–529, 2008.

[5] B. Singh, R. Kumar, and V. P. Singh, "Reinforcement learning in robotic applications: a comprehensive survey," *Artificial Intelligence Review*, vol. 55, no. 2, pp. 945–990, 2022.

[6] V. Makoviychuk, L. Wawrzyniak, Y. Guo, M. Lu, K. Storey, M. Macklin, D. Hoeller, N. Rudin, A. Allshire, A. Handa et al., "Isaac gym: High performance gpu-based physics simulation for robot learning," *arXiv preprint arXiv:2108.10470*, 2021.

[7] I. Akkaya, M. Andrychowicz, M. Chociej, M. Litwin, B. McGrew, A. Petron, A. Paino, M. Plappert, G. Powell, R. Ribas et al., "Solving rubik's cube with a robot hand," *arXiv preprint arXiv:1910.07113*, 2019.

[8] G. Dulac-Arnold, N. Levine, D. J. Mankowitz, J. Li, C. Paduraru, S. Gowal, and T. Hester, "Challenges of real-world reinforcement learning: definitions, benchmarks and analysis," *Machine Learning*, vol. 110, no. 9, pp. 2419–2468, 2021.

[9] J. Tobin, R. Fong, A. Ray, J. Schneider, W. Zaremba, and P. Abbeel, "Domain randomization for transferring deep neural networks from simulation to the real world," in *2017 IEEE/RSJ international conference on intelligent robots and systems (IROS)*. IEEE, 2017, pp. 23–30.

[10] G. Authors, "Genesis: A universal and generative physics engine for robotics and beyond," December 2024. [Online]. Available: https://github.com/Genesis-Embodied-AI/Genesis

[11] X. Gu, Y.-J. Wang, and J. Chen, "Humanoid-gym: Reinforcement learning for humanoid robot with zero-shot sim2real transfer," *arXiv preprint arXiv:2404.05695*, 2024.

[12] T. Erez, Y. Tassa, and E. Todorov, "Simulation tools for model-based robotics: Comparison of bullet, havok, mujoco, ode and physx," in *2015 IEEE international conference on robotics and automation (ICRA)*. IEEE, 2015, pp. 4397–4404.

[13] O. Michel, "Cyberbotics ltd. webots™: professional mobile robot simulation," *International Journal of Advanced Robotic Systems*, vol. 1, no. 1, p. 5, 2004.

[14] E. Coumans and Y. Bai, "Pybullet, a python module for physics simulation for games, robotics and machine learning," 2016.

[15] J. Hwangbo, J. Lee, and M. Hutter, "Per-contact iteration method for solving contact dynamics," *IEEE Robotics and Automation Letters*, vol. 3, no. 2, pp. 895–902, 2018.

[16] NVIDIA, "Isaac sim- robotics simulation and synthetic data generation." [Online]. Available: https://developer.nvidia.com/isaac/sim

[17] M. Kaup, C. Wolff, H. Hwang, J. Mayer, and E. Bruni, "A review of nine physics engines for reinforcement learning research," *arXiv preprint arXiv:2407.08590*, 2024.

[18] D. Chen, B. Zhou, V. Koltun, and P. Krähenbühl, "Learning by cheating," in *Conference on Robot Learning*. PMLR, 2020, pp. 66–75.

[19] J. Lee, J. Hwangbo, L. Wellhausen, V. Koltun, and M. Hutter, "Learning quadrupedal locomotion over challenging terrain," *Science robotics*, vol. 5, no. 47, p. eabc5986, 2020.

[20] A. Kumar, Z. Fu, D. Pathak, and J. Malik, "Rma: Rapid motor adaptation for legged robots," *arXiv preprint arXiv:2107.04034*, 2021.

[21] I. M. A. Nahrendra, B. Yu, and H. Myung, "Dreamwaq: Learning robust quadrupedal locomotion with implicit terrain imagination via deep reinforcement learning," in *2023 IEEE International Conference on Robotics and Automation (ICRA)*. IEEE, 2023, pp. 5078–5084.

[22] J. Long, W. Yu, Q. Li, Z. Wang, D. Lin, and J. Pang, "Learning h-infinity locomotion control," *arXiv preprint arXiv:2404.14405*, 2024.

[23] J. Hwangbo, J. Lee, A. Dosovitskiy, D. Bellicoso, V. Tsounis, V. Koltun, and M. Hutter, "Learning agile and dynamic motor skills for legged robots," *Science Robotics*, vol. 4, no. 26, p. eaau5872, 2019.
