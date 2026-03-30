
# AlphaGen




## 定义
### 动作

设动作空间为 $\mathcal{A}=\{0,1,\dots,K-1\}$，每一个**动作**是一个离散变量 $a\in\mathcal{A}$

> `AlphaEnvWrapper` 可以将 $a$ 其翻译为具体的 `Token` ，例如
> - `2` -> `UnaryOperatorToken(LOG)`
> - `26` -> `FeatureToken(FeatureType.CLOSE)`
> - `35` -> `END_TOKEN`
>
> 详见 `wrapper.py`

它描述模型在当前观测条件时**输出的下一步动作**，即当前 token 序列应该接哪个 token

### 状态

设不含 `BEGIN` 但是允许 `END` 出现的最大可采样 token 数为 $L$，即生成表达式最长长度，则观测状态空间为 $\mathcal{S}=\{0,1,\dots,K\}^{L-1}$，**状态**是一个固定长度的向量
$$
s_t=(x_1,\dots,x_{L-1})\in\mathcal{S}
$$

它描述当前**已经生成的动作**组成的序列

> 模型的目的就是根据当前序列选择出下一个 `Token`，但是由于表达式上限是 $L$，所以只需保存 $L-1$ 个

> 在当前模型设计里 $x_t=0$ 为预填充空位，所以 $s$ 中保存的结果需要平移一个整数单位后才能对上 token，即 $x_t=k$ 表示第 $t$ 步生成了动作 $a_t=k-1$

初始状态 $s_0=(0,\dots,0)$，在状态 $s_t$ 时接收新动作 $a_t$ 后，产生状态转移，即
$$
s_t=(x_1,\dots,x_{t-1},0\,\dots)\rightarrow s_{t+1}=(x_1,\dots,x_{t-1},x_t,0,\dots)
$$
其中 $x_t=a_t+1$
### 分数

分数 $r_t$ 表示发生状态转移时的得分，在本项目里一般遵循如下规则：
- $a_t$ 是合法的表达式结束，即 $a_t$ 对应 `END_TOKEN`，并且此时是合法单个表达式，则 $r_t$ 由 `StockDataCalculator` 的计算结果给出，通常包括因子 IC、相关性、换手率等，如果表达式合法但是评估时无法计算，则 $r_t$ 为 `invalid_eval_reward`
- $a_t$ 是合法的一般中间步，能加入构树器中且加入后表达式长度未达到上限，则 $r_t=0$
- $a_t$ 是合法的一般中间步，能加入构树器中且加入后表达式长度达到上限，如果此时是合法单个表达式，则 $r_t$ 同第一条，否则 $r_t$ 为 `invalid_reward`
- $a_t$ 是不合法的动作，则 $r_t$ 为 `invalid_reward`

### 目标
RL模型可以概括为一个表达式生成策略函数
$$
\pi_\theta(a\ |\ s)
$$
即模型能够**在给定的alpha因子构造状态下，选出更可能导向高分表达式的下一个token**

## 模型结构

### 整体结构

模型基本遵循 actor-critic 结构，整体分为**编码器+生成器**和**预测器**两个部分，参数空间为 $\Theta=(\Theta_f,\Theta_\pi,\Theta_V)$，其中
- 编码器将状态 $s\in\mathcal{S}$ 转换为隐变量 $h\in\mathbb{R}^d$，记作 $h=f_{\theta_f}(s)$
- 生成器将隐变量 $h$ 转换为动作分布 $\pi_{\theta_\pi}(a\ |\ s)$
- 评估器通过隐变量 $h$ 估计 $s$ 的期望得分，记作 $V_{\theta_V}(s)=f_{\theta_V}(h)$

### 编码器：Embedding

对于 $t$ 步的状态向量 $s_t=(x_1,\dots,x_{L-1})$，在头部补充一个 `BEGIN` token得到 $\tilde{s}_t=([\text{BEG}],x_1,\dots,x_{L-1})$，嵌入规则为一个可训练的矩阵
$$
E=\begin{bmatrix}e_{[\text{BEG}]}\\e_0\\\vdots\\e_K\end{bmatrix}\in\mathbb{R}^{(K+2)\times d}
$$
其中离散值 $x_i$ 被映射为 $d$ 维向量 $e_{x_i}$. 

> 填充空位的 $e_0$ 可以被设置成不可训练的

此时序列 $s_t$ 就被映射为
$$
E_t=\begin{bmatrix}e_{[\text{BEG}]}\\e_{x_1}\\\vdots\\e_{x_{L-1}}\end{bmatrix}\in\mathbb{R}^{L\times d}
$$

> 以上直接由 `nn.Embedding` 实现，本项目中使用 $d=128$

### 编码器：位置

位置编码是一个不可训练的矩阵
$$
P\in\mathbb{R}^{L\times d}
$$

> `PositionalEncoding` 使用经典的余弦正弦编码，即
> $$
> P_{i,2j}=\sin\left(\frac{i}{10000^{2j/d}}\right)\quad P_{i,2j+1}=\cos\left(\frac{i}{10000^{2j/d}}\right)
> $$

从而得到带位置信息的嵌入矩阵
$$
Z_t=\begin{bmatrix}z_1\\\vdots\\z_L\end{bmatrix}=E_t+P\in\mathbb{R}^{L\times d}
$$

### 编码器：多层 LSTM 

设模型层数为 $M$，对于每一层都有权重矩阵
$$
W_{ih}^{(m)}\in\mathbb{R}^{4d\times d}\quad b_{ih}^{(m)}\in\mathbb{R}^{4d}\quad W_{hh}^{(m)}\in\mathbb{R}^{4d\times d}\quad b_{hh}^{(m)}\in\mathbb{R}^{4d}
$$
和递推更新规则
$$
(h_i^{(m)},c_i^{(m)}) = \text{LSTM}^{(m)}(u_i^{(m)},h_{i-1}^{(m)},c_{i-1}^{(m)})
$$
以及层之间传递规则
$$
u_i^{(1)}=z_i\quad u_i^{(m)}=h_i^{(m-1)}
$$
最终得到顶层隐变量矩阵
$$
H=\begin{bmatrix}h_1^{(M)}\\\vdots\\h_L^{(M)}\end{bmatrix}\in\mathbb{R}^{L\times d}
$$
> 以上直接由 `nn.LSTM` 实现，本项目中 $M=2, d=128, \text{dropout}=0.1$

### 编码器：平均值

直接做 mean pooling 得到
$$
h_t=\frac1{L}\sum_{i=1}^Lh_i^{(M)}\in\mathbb{R}^d
$$

### 生成器：SB3 默认 MLP 和动作头

`stable_baselines3` 会额外接一个全连接层再接动作头，即
```math
\begin{align*}
u_t^{(1)}&=\sigma(W_{\pi,1}h_t+b_{\pi,1})\in\mathbb{R}^{d_1}\\
u_t^{(2)}&=\sigma(W_{\pi,2}u_t^{(1)}+b_{\pi,2})\in\mathbb{R}^{d_2}
\end{align*}
```
> 本项目中使用 $d_1=d_2=64, \tanh$ 激活

在上述 MLP 层后输出候选动作的 logits 变量
$$
p_t=W_{\text{act}}u_t^{(2)}+b_{\text{act}}\in\mathbb{R}^K
$$

### 生成器：掩码

由于不是所有 $a_t$ 取值都是合法的，因此根据 $s_t$ 计算 $a_t$ 合法选项掩码 $m_t$，其中 $m_t(a)\in\{0,1\}$，$1$ 表示合法，$0$ 表示非法

对应的 masked logits 为
$$
\tilde{p}_t(a)=\begin{cases}p_t(a),\ &m_t(a)=1\\-\infty,\ &m_t(a)=0\end{cases}
$$

> 掩码由构树模型 `ExpressionBuilder` 的 `validate...` 模块直接给出，限制下一步的动作取值范围，再由 `AlphaEnvWrapper` 翻译为 $m_t$

> 掩码的作用是提高学习效率，使得模型训练更专注于寻找 alpha 而非句式合法性，虽然也可以允许生成不合法的 $a_t$ 并给 `invalid_reward`，但是这样会导致前期大量 rollout 样本浪费在无意义的表达式轨迹上，学习信号稀疏，在本项目中由于掩码的存在，无效一般只会发生在：
> - 动作序列达到长度上限，强行截断而留下不合法表达式，给予 `invalid_reward`，设为 $-1$
> - 表达式评估时出现大量无意义值，例如超越时间窗口或除以 0，给予 `invalid_eval_reward`，设为 $0$

### 生成器：softmax采样

经过掩码后的 $\tilde p_t$ 通过 softmax 得到采样概率分布
$$
\pi_\theta(a\ |\ s_t)=\frac{\exp(\tilde p_t(a))}{\sum_{a'=0}^{K-1}\exp(\tilde p_t(a'))}
$$
从中随机采样
$$
a_t\sim\pi_\theta(\cdot\ |\ s_t)
$$
得到下一步动作 $a_t$

### 预测器：SB3 默认 MLP

同生成器层使用 `stable_baselines3`，在预测器上也先套两层 MLP
```math
\begin{align*}
v_t^{(1)}&=\sigma(W_{V,1}h_t+b_{V,1})\in\mathbb{R}^{d_3}\\
v_t^{(2)}&=\sigma(W_{V,2}v_t^{(1)}+b_{V,2})\in\mathbb{R}^{d_4}
\end{align*}
```
> 本项目中使用 $d_3=d_4=64, \tanh$ 激活

### 预测器：价值

期望得分是一个标量，表示预测器认为在观测到 $s_t$ 的情况下表达式生成完全后期望得到的最终分数，直接用线性计算
$$
V_{\theta_V}(s_t)=w_V^\top v_t^{(2)}+b_V\in\mathbb{R}
$$

## 训练框架

### 采样

设第 $k$ 轮开始时的参数组为 $\theta^{\text{old}}=(\theta_f^{\text{old}},\theta_\pi^{\text{old}},\theta_V^{\text{old}})$，固定使用这组参数进行采样 $N$ 次得到一组采样结果，
$$
\mathcal{D}=\{(s_t,a_t,r_t)\}_{t=1}^N
$$

采样规则遵循模型结构，且要求：
- 单个序列最多采样 $L$ 次
- 如果采样到 `END` 或者采样到序列第 $L$ 次，则下一步从新序列采样
- 如果采样到 $N$ 次但是当前序列还未结束，则使用预测器值 $V(s_N)$ 代替

> 本项目 $N=2048,L=15$

### SB3/PPO 默认 GAE

对于采样过程中的一个完整序列
$$
\{(s_t,a_t,r_t)\}_{t=T_0}^{T}\subseteq\mathcal{D}
$$

在 `static_baselines3` 中先计算时间差分残差
$$
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)
$$
然后定义 advantage 为
$$
A_t=\sum_{k=t}^T(\gamma\lambda)^{k-t}\delta_k
$$
即 $A_t=\delta_t+\gamma\lambda A_{t+1}$，它描述在当前参数下和状态 $s_t$ 下，这次选到的动作和后续路径，比当前预期水平好多少

同时计算 $R_t=A_t+V(s_t)$，它描述在当前参数和状态 $s_t$ 下，未来总回报应该大概是多少

> 在本项目里一般步的 $r_t=0$，只有 $r_T$ 为最终表达式评估值

> 本项目中使用 $\gamma=1,\lambda=0.95$，如果令 $\lambda=1$，则此时退化为蒙特卡洛方法 $A_t=\sum_{k=t}^T\gamma^{k-t}r_k$，它的含义是在这个序列下，$a_t$ 对未来收益增益影响，越远期的影响越小，$\gamma$描述这个规则的衰减率，而 `static_baselines3` 的 GAE 算法降低了序列尾部随机性导致的高方差，$\lambda$ 越接近 $1$ 表示越信长期，偏差越小方差越大，反之亦然

### SB3/PPO 默认优化

完成一轮采样 $\mathcal{D}$ 后，计算所有数据的基于 $\theta_\text{old}$ 的 $A_t, R_t, V_{\theta_\text{old}}(s_t),\pi_{\theta_\text{old},t}$ 得到
$$
\tilde{\mathcal D}=\{(s_t,a_t,r_t,A_t, R_t, V_{\theta_\text{old}}(s_t),\pi_{\theta_\text{old},t})\}_{t=1}^N
$$

再随机划分为 mini-batch $\mathcal{B}\subseteq\mathcal{D}$

设当前参数为 $\theta$，每次计算相应的 $\pi_{\theta,t}, V_\theta(t)$ 和策略熵
$$
H_t=-\sum_a\pi_{\theta,t}\log\pi_{\theta,t}
$$
之后基于当前参数计算预估期望 $V_{\theta_V}(s_t)$ 和标准 PPO 概率比
$$
\rho_t=\exp(\log\pi_{\theta,t}-\log\pi_{\theta_{\text{old}},t})
$$
对 batch 内所有 $t\in B$ 计算得到当前 batch 下的生成器损失函数
$$
\mathcal{L}_\pi=-\frac1{|B|}\sum_{t\in B}\min\left(\rho_t A_t, \text{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t\right)
$$
预测器损失函数
$$
\mathcal{L}_V=\frac1{|B|}\sum_{t\in B}\left(V_{\theta_V}(s_t)-R_t\right)^2
$$
平均熵
$$
\mathcal{L}_H=\frac1{|B|}\sum_{t\in B}H_t
$$

> 熵比较小意味着模型容易坍缩到少数动作里陷入局部最优，因此加入熵奖励项

最终的优化目标是
$$
\mathcal{L}=\mathcal{L}_\pi+c_V\mathcal{L}_V-c_H\mathcal{L}_H
$$
并作梯度下降
$$
\theta\rightarrow\theta-\eta_\theta\Delta_\theta\mathcal{L}
$$

> 本项目中 $|\mathcal{B}|=128, n_{\text{epochs}}=10, \epsilon=0.2, c_V=0.5, c_H=0.01, \eta=3\times 10^{-4}, \varepsilon_{\text{Adam}}=10^{-5}$
