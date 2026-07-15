# ManiFlowRL 论文写作进展与 Method 写作规范

> 更新日期：2026-07-15  
> 适用项目：基于 ManiFlow consistency-flow actor 的在线强化学习微调  
> 当前重点：Method 第 3.3 节——Action-Conditioned Residual Noise Scheduling

---

## 1. 当前论文主线

论文不应被写成“ManiFlow + ReinFlow 的代码迁移”，而应围绕以下方法主线组织：

1. 使用预训练 ManiFlow 作为确定性 consistency-flow 动作生成器；
2. 将确定性去噪轨迹扩展为具有可计算路径概率的随机 Markov chain；
3. 设计动作条件化的残差噪声调度器，为不同去噪状态和 action-chunk 位置提供结构化探索；
4. 基于完整 denoising-chain likelihood 执行 PPO 更新；
5. 使用保守微调机制减轻在线 RL 对预训练策略的破坏。

建议的 Method 结构为：

- **3.1 Flow Matching and ManiFlow Consistency-Flow Policy**
- **3.2 Stochastic Markovization for Online Policy Optimization**
- **3.3 Action-Conditioned Residual Noise Scheduling**
- **3.4 Denoising-Chain PPO Objective**
- **3.5 Conservative Online Fine-Tuning**
- **3.6 Rollout and Implementation Details**

章节边界必须清晰：

- 3.1 完成确定性 consistency-flow 更新的定义；
- 3.2 只解释随机 Gaussian transition、Markov chain 和 chain likelihood；
- 3.3 只解释噪声尺度如何参数化；
- 3.4 再解释 PPO ratio、clipped objective、value loss 和 entropy；
- 具体层数、隐藏维度、激活函数等放入 Implementation Details 或附录。

---

## 2. 3.3 节中各变量的真实含义

### 2.1 观测特征 `e_o`

`e_o` 不是原始点云，也不是新设计的观测编码器输出。它对应预训练 ManiFlow observation encoder 提取的观测特征经过池化后的条件向量。

计算关系可概括为：

```text
原始观测 o
  -> ManiFlow observation encoder
  -> observation tokens
  -> mean pooling
  -> e_o
```

数学上可写为：

```math
e_o = \operatorname{MeanPool}(E_{\mathrm{obs}}(o)).
```

其直观含义是：**当前机器人所处场景的压缩表示**。

论文中首次使用 `e_o` 时必须明确说明：

- 它来自预训练 ManiFlow 的 observation encoder；
- 噪声网络复用该特征，而不是重新编码原始观测；
- 当前实现对 observation tokens 做 mean pooling 后提供给噪声网络。

不要只写“观测嵌入”，否则读者无法判断其来源。

---

### 2.2 去噪时间嵌入

去噪时间 `t_k` 本身是一个标量。代码首先通过 sinusoidal positional embedding 将其转换为向量，再通过小型 MLP 和线性投影映射到噪声网络的隐藏维度。

可概括为：

```text
t_k
  -> sinusoidal embedding
  -> small MLP
  -> hidden time feature
```

论文中建议记为：

```math
\gamma(t_k)
```

或：

```math
\operatorname{Emb}_t(t_k).
```

不要使用含义不清的 `e_t(t_k)`。首次出现时应明确说明这是 **sinusoidal embedding of the denoising time**。

其作用是告诉噪声网络：当前处于去噪初期、中期还是后期。

---

### 2.3 当前动作特征

在第 `k` 个去噪步骤，完整 action chunk 为：

```math
x_k = [x_{k,1},\ldots,x_{k,H}],
```

其中 `x_{k,h}` 是 action chunk 中第 `h` 个位置当前的动作状态。

代码不是简单使用一个线性变换，而是通过小型 MLP 将每个动作向量映射到噪声网络隐藏空间。因此，论文中若需要精确表达，应使用：

```math
f_x(x_{k,h})
```

而不应写成看似精确但与实现不完全一致的：

```math
W_x x_{k,h}.
```

其直观作用是：**告诉噪声网络当前正在去噪的动作轨迹是什么状态**。

---

### 2.4 Action-horizon 位置嵌入 `p_h`

`p_h` 是 action chunk 中第 `h` 个位置对应的可学习向量。

例如：

- `p_1` 表示 chunk 中第一个动作；
- `p_2` 表示第二个动作；
- `p_H` 表示最后一个动作。

它不是 observation history，也不是额外传入 `forward()` 的标量参数。位置 `h` 通过输入张量的 horizon 维度及内部的 horizon-position embedding 自动体现。

论文中统一称为：

- **action-horizon position**；
- **action-chunk position**；
- **temporal position within the action chunk**。

不要称为 history，以免被误解为 observation history 或 recurrent hidden state。

---

### 2.5 TCN 的作用

此前正文中的“轻量级时序编码器”实际就是当前默认使用的 **TCN noise backbone**。

TCN 沿 action horizon 维度进行一维卷积，用于建模 action chunk 内相邻动作之间的局部时间关系。

输入形状为：

```math
Z_k \in \mathbb{R}^{B\times H\times d_m}.
```

计算时转置为：

```math
\mathbb{R}^{B\times d_m\times H},
```

然后沿 `H` 维进行 `Conv1d`。

因此，第 `h` 个位置的噪声预测：

- 有独立的输出；
- 依赖本位置动作状态和位置嵌入；
- 还可以利用相邻 action positions 的上下文。

推荐的论文表述是：

> The network predicts a separate noise scale for each action position, while temporal convolutions allow each prediction to incorporate local context from neighboring actions within the chunk.

不要使用绕口的“position-specific but not position-independent”。

---

## 3. 噪声网络的实际数据流

代码的核心过程可概括为：

```text
观测条件特征 e_o
+ 去噪时间特征 gamma(t_k)
+ 当前动作特征 f_x(x_{k,h})
+ action-position embedding p_h
        |
        v
每个 action position 的 token
        |
        v
TCN 沿 action horizon 建模局部时序关系
        |
        v
每个位置、每个动作维度的 residual
        |
        v
有界残差 + base sigma schedule
        |
        v
最终 sigma_{phi,k}
```

高层函数可以写为：

```math
\delta_{\phi,k} = f_\phi(e_o,t_k,x_k),
```

其中输出形状为：

```math
\delta_{\phi,k}\in\mathbb{R}^{H\times d_a}.
```

最终噪声尺度为：

```math
\sigma_{\phi,k}
=
\operatorname{clip}
\left(
\sigma_{\mathrm{base}}(t_k)
\exp\bigl(\alpha\tanh(\delta_{\phi,k})\bigr),
\sigma_{\min},
\sigma_{\max}
\right).
```

该公式体现三个关键设计：

1. `sigma_base(t_k)` 提供随去噪进度变化的结构化探索先验；
2. residual predictor 根据观测、当前动作和 action position 自适应调整噪声；
3. `tanh` 和上下界限制防止噪声尺度失控。

注意：若将网络原始输出记为 `r_{phi,k}`，则应保持符号层次一致，例如：

```math
\delta_{\phi,k}=\alpha\tanh(r_{\phi,k}),
```

```math
\sigma_{\phi,k}
=
\operatorname{clip}
\left(
\sigma_{\mathrm{base}}(t_k)\exp(\delta_{\phi,k}),
\sigma_{\min},
\sigma_{\max}
\right).
```

不要在同一个公式中混用“原始输出”和“有界 residual”的符号。

---

## 4. 当前 3.3 正文存在的问题

### 4.1 符号未经定义

此前正文直接引入：

```math
e_o,\quad e_t(t_k),\quad W_o,\quad W_t,\quad W_x,\quad p_h,\quad g_\phi,
```

但没有解释：

- `e_o` 来自哪个网络；
- 是原始观测还是池化特征；
- 时间使用何种 embedding；
- `g_phi` 是 MLP、TCN 还是 Transformer；
- `W_x` 是线性层还是 MLP。

结论：**抽象符号不是越多越学术。任何符号都必须在首次出现时给出可操作定义。**

---

### 4.2 数学表达与实现精度不一致

此前写法：

```math
W_xx_{k,h}
```

看似是精确结构定义，但代码实际使用两层 MLP。

类似地，时间特征实际包含 sinusoidal embedding、MLP 和线性投影，而不是一个简单的 `W_t e_t(t_k)`。

这类表达的问题是：

- 作为高层抽象，它写得过细；
- 作为精确架构定义，它又不完全准确。

后续应二选一：

- 主文使用高层函数 `f_o`、`f_t`、`f_x`；
- 附录再给出精确网络结构。

---

### 4.3 TCN 被模糊化

将 TCN 写成“轻量级时序编码器”会隐藏方法差异。

如果 TCN 是最终方法，应直接称为：

- **temporal convolutional noise network**；
- **TCN-based residual noise predictor**。

正文要解释 TCN 解决的问题：

> 它沿 action horizon 建模 chunk 内的局部时间相关性，使每个位置的噪声预测可以利用邻近动作，而不是独立地预测每个 action position。

---

### 4.4 正文过度接近代码执行顺序

此前正文按下面顺序展开：

1. observation projection；
2. time projection；
3. action projection；
4. position embedding；
5. LayerNorm；
6. TCN；
7. output projection；
8. tanh；
9. exponential；
10. clip。

这属于“把代码翻译成数学”，不是成熟论文的 Method 写法。

论文应先回答：

1. 现有固定或 time-only 噪声有什么不足；
2. 为什么需要 action-conditioned residual schedule；
3. 为什么 action chunk 需要位置相关噪声；
4. 为什么用 TCN 建模 chunk 内部关系；
5. 如何约束噪声以保持微调稳定。

---

### 4.5 核心贡献与普通实现细节混杂

正文应重点保留：

- 去噪步相关的 base sigma schedule；
- 动作条件化 residual；
- action-position-specific output；
- TCN 建模 chunk 内相关性；
- bounded residual 和 sigma bounds。

下列内容应放到实现细节或附录：

- LayerNorm；
- GroupNorm；
- 线性层数量；
- hidden dimension；
- kernel size；
- 激活函数；
- tensor transpose；
- 具体类名和函数名。

---

### 4.6 设计动机不应写成未经验证的客观规律

例如：“去噪早期一定需要更大探索，后期一定需要更小探索”不应写成已证明事实。

更稳妥的表述是：

> We impose a decaying base schedule as a structured exploration prior, assigning larger perturbations to early denoising states and smaller perturbations near the final action.

即：

> 我们采用衰减的基础噪声作为结构化探索先验。

这是方法设计选择，是否有效需要通过消融实验验证。

---

## 5. 顶会/顶刊 Method 的共同写作风格

结合 ManiFlow、ReinFlow 和 DPPO 的 Method 组织，可以总结出以下原则。

### 5.1 每节只解决一个核心问题

推荐边界：

- 3.2：如何得到随机 Markov chain 和可计算 chain likelihood；
- 3.3：如何设计结构化、自适应的探索噪声；
- 3.4：如何用 chain likelihood 完成 PPO 更新；
- 3.5：如何限制策略漂移和稳定在线训练。

不要在同一节混合 Markov 推导、TCN 结构、PPO ratio 和训练技巧。

---

### 5.2 先解释“为什么”，再说明“怎么做”

每个小节应遵循：

```text
现有方法的局限
-> 本文设计原则
-> 核心数学定义
-> 为什么该定义解决问题
-> 具体实现概述
```

例如 3.3 开头应先指出：

> 固定噪声或仅依赖时间的噪声无法根据当前 action trajectory 以及 chunk 中不同位置调整探索强度。

然后再提出：

> 因此，我们在基础噪声 schedule 上学习动作条件化 residual，并使用 TCN 建模 chunk 内相关性。

---

### 5.3 主文公式要克制

3.3 的核心公式原则上只需两个：

```math
\delta_{\phi,k}=f_\phi(e_o,t_k,x_k),
```

```math
\sigma_{\phi,k}
=
\operatorname{clip}
\left(
\sigma_{\mathrm{base}}(t_k)
\exp(\alpha\tanh(\delta_{\phi,k})),
\sigma_{\min},
\sigma_{\max}
\right).
```

不建议在主文中详细展开 token construction：

```math
z_{k,h}=\operatorname{LN}(\cdots).
```

除非 token construction 本身被定位为主要方法贡献。

---

### 5.4 主文描述方法原则，附录描述网络层

ReinFlow、DPPO、ManiFlow 等成熟论文通常不会在理论主线中逐层描述网络。

主文应说明：

- 输入是什么；
- 输出是什么；
- 为什么选择该网络；
- 该网络如何影响策略行为。

实现细节或附录再说明：

- 层数；
- hidden dimension；
- kernel size；
- activation；
- normalization；
- 初始化方式。

---

### 5.5 网络结构必须与贡献绑定

不是简单写“我们采用 TCN”，而应说明：

> TCN 沿 action horizon 建模局部时间相关性，因此不同 action positions 的噪声预测既保持逐位置输出，又能利用相邻动作上下文。

网络结构存在的理由应始终与问题对应。

---

## 6. 建议的 3.3 写作层次

后续重写 3.3 时，建议严格控制为四个层次。

### 第一段：问题与动机

说明：

- 固定噪声和 time-only noise 的局限；
- action chunk 中不同位置、不同去噪状态可能具有不同探索需求；
- 需要在保留预训练 flow dynamics 的同时提供细粒度探索。

### 第二段：核心 residual schedule

定义 base schedule 和 multiplicative residual：

```math
\sigma_{\phi,k}
=
\operatorname{clip}
\left(
\sigma_{\mathrm{base}}(t_k)\exp(\delta_{\phi,k}),
\sigma_{\min},
\sigma_{\max}
\right).
```

说明：

- base schedule 提供全局去噪步先验；
- learned residual 提供状态相关局部修正。

### 第三段：TCN-based residual predictor

用自然语言说明：

- 输入包括预训练 actor 的观测特征、去噪时间、当前 action chunk；
- 每个 action position 加入可学习位置向量；
- TCN 沿 action horizon 建模邻近动作关系；
- 输出为每个位置、每个动作维度的 residual。

最多保留一个函数：

```math
\delta_{\phi,k}=f_\phi(e_o,t_k,x_k).
```

### 第四段：稳定性设计

说明：

- `tanh` 限制 residual 幅度；
- zero initialization 使训练初始阶段近似 base schedule；
- sigma bounds 防止方差过小或过大；
- 确定性评估时移除随机项。

---

## 7. 后续写作检查清单

每次完成一个 Method 小节后，应检查：

- [ ] 本节是否只回答一个核心问题？
- [ ] 第一段是否明确指出已有方法的不足？
- [ ] 是否先解释设计动机，再给公式？
- [ ] 每个符号是否在首次出现时定义？
- [ ] 数学表达是否与真实实现一致？
- [ ] 是否把普通工程细节误写成方法贡献？
- [ ] 是否存在可以删除而不影响核心逻辑的公式？
- [ ] 网络结构是否解释了“为什么需要”，而不只是“用了什么”？
- [ ] 是否区分了本文原创设计、ReinFlow-derived 部分和 ManiFlow background？
- [ ] 是否避免未经实验验证的绝对表述？
- [ ] 是否为下一节留下明确的逻辑接口？

---

## 8. 当前明确的论文表达决策

1. 去噪步索引使用 `k`；连续/离散去噪时间使用 `t_k`；不再使用 `tau_k`。
2. action horizon 使用 `H`；action position 使用 `h`。
3. `h` 称为 action-chunk position，不称为历史。
4. 确定性 consistency-flow 更新在 3.1 定义，3.2 不重复。
5. 3.2 只保留 Gaussian transition、Markov factorization 和 chain log probability。
6. 逐位置噪声、horizon embedding 和 TCN 放到 3.3。
7. 显式 Gaussian element-wise log probability 和 policy-gradient/PPO 公式放到 3.4。
8. 3.3 主文不出现代码类名、函数名、配置字段或 tensor transpose。
9. 3.3 应直接说明采用 TCN-based residual noise predictor，不用模糊的“轻量级时序编码器”。
10. 当前 observation condition 是 ManiFlow observation feature 的 mean-pooled representation；论文中必须说明其来源。

---

## 9. 核心写作原则

后续 Method 写作必须避免“把代码翻译成公式”。

正确方向是：

```text
明确问题
-> 给出设计原则
-> 定义核心算法
-> 解释为什么有效
-> 用最少必要文字说明实现
```

论文主文的目标不是让读者复现每一行代码，而是让读者清楚理解：

- 本文解决了什么缺陷；
- 采用了什么关键设计；
- 该设计如何改变策略分布和优化过程；
- 哪些部分构成真正的方法贡献。
