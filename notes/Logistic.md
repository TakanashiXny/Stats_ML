# <center> 逻辑斯谛回归与最大熵模型

## 本章思路

1. 逻辑斯谛回归模型
   - 逻辑斯谛分布
   - 二项逻辑斯谛模型
   - 模型参数估计

2. 最大熵模型
   - 最大熵原理
   - 特征函数
   - 最大熵模型的定义
   - 模型学习
   - 极大似然估计

3. 模型学习的最优化算法
   - 改进的迭代尺度法
   - 拟牛顿法


## 逻辑斯谛回归模型

### 逻辑斯谛分布

设$X$是连续随机变量，逻辑斯谛分布的分布函数和密度函数如下
$$
F(x)=P(X \leqslant x)=\frac{1}{1+e^{-(x-\mu)/\gamma}} \\
f(x)=F'(x)=\frac{e^{-(x-\mu)/\gamma}}{\gamma(1+e^{-(x-\mu)/\gamma})^2}
$$
其中，$\mu$为位置参数，$\gamma > 0$为形状参数

- 该函数曲线以$(\mu,\frac{1}{2})$为中心对称点
- 形状参数越小，曲线在中心附近增长得越快

### 二项逻辑斯谛回归模型

该模型为如下的条件概率分布
$$
P(Y=1|x)=\frac{exp(\omega \cdot x + b)}{1+exp(\omega \cdot x + b)} \\
P(Y=0|x)=\frac{1}{1+exp(\omega \cdot x + b)} 
$$
$x \in \mathbf{R}^n$是输入，$Y \in \{0,1\}$是输出，$\omega \in \mathbf{R}^n, b \in \mathbf{R}$是参数

**该模型的特点**：

考察事件发生的**几率**。
$$
odds=\frac{p}{1-p}
$$
该事件的对数几率是
$$
\text{logit}(p)=\text{log}\frac{p}{1-p}
$$

对逻辑斯谛回归而言，得到
$$
\text{log}\frac{P(Y=1|x)}{1-P(Y=1|x)}=\omega \cdot x
$$
此处忽略偏置项

可以得到，**输出的对数几率是输入的线性函数的模型时逻辑斯谛回归模型**

当线性函数值接近无穷大，几率则为无穷大，概率值则接近于1；反之则接近于0

### 模型参数估计

数据集$T=\{(x_1,y_1),(x_2,y_2),\cdots,(x_N,y_N)\}$

设
$$
P(Y=1|x)=\pi(x), \quad P(Y=0|x)=1-\pi(x)
$$
似然函数为
$$
\stackrel{N}{\underset{i=1}{\Pi}}[\pi(x_i)]^{y_i}[1-\pi(x_i)]^{1-y_i}
$$
对数似然函数为
$$
L(\omega)=\stackrel{N}{\underset{i=1}{\sum}}[y_i(\omega \cdot x_i)-\text{log}(1+exp(\omega \cdot x_i))]
$$
求最大似然函数的极大值可得到$\omega$的估计值

### 多项逻辑斯谛回归

$$
P(Y=k|x)=\frac{exp(\omega_k \cdot x)}{1+\stackrel{K-1}{\underset{k=1}{\sum}}exp(\omega_k \cdot x)}, \quad k=1,2,\cdots,K-1 \\
P(Y=K|x)=\frac{1}{1+\stackrel{K-1}{\underset{k=1}{\sum}}exp(\omega_k \cdot x)}
$$

## 最大熵模型

### 最大熵原理

学习概率模型时，所有可能的概率模型中熵最大的模型是最好的模型。

直观上认为，要选择的概率模型在满足约束条件时，不确定的部分是等可能的

### 最大熵模型的定义

分类模型是一个概率分布$P(Y|X)$，$X \in \mathcal{X} \subset \mathbf{R}^n$表示输入，$Y \in \mathcal{Y}$表示输出

给定训练数据集
$$
T=\{(x_1,y_1),(x_2,y_2),\cdots,(x_N,y_N)\}
$$

1. 确定训练数据集的经验分布和边缘分布$P(X)$的经验分布
   $$
   \widetilde{P}(X=x,Y=y)=\frac{v(X=x,Y=y)}{N} \\
   \widetilde{P}(X=x)=\frac{v(X=x)}{N}
   $$

2. 特征函数
   $$
   f(x)=\begin{cases}
    1, & x与y满足某一事实 \\
    0, & else
   \end{cases}
   $$
   $f(x,y)$关于经验分布的期望值
   $$
   E_{\widetilde{P}}(f)=\underset{x,y}{\sum}\widetilde{P}(x,y)f(x,y)
   $$
   $f(x,y)$关于$P(Y|X)$与经验分布$\widetilde{P}(X)$的期望值
   $$
   E_{p}(f)=\underset{x,y}{\sum}\widetilde{P}(x)P(y|x)f(x,y)
   $$
   如果模型能够获取训练数据中的信息，则可以假设这两个期望相等

将$E_{\widetilde{P}}(f)=E_{p}(f)$作为模型学习的**约束条件**

得到最大熵模型: 假设满足所有约束条件的模型集合为
$$
\mathcal{C} \equiv \{P \in \mathcal{P}|E_P(f_i)=E_{\widetilde{P}}(f_i), i=1,2,\cdots,n\}
$$
定义在条件概率分布$P(Y|X)$上的条件熵为
$$
H(P)=-\underset{x,y}{\sum}\widetilde{P}(x)P(y|x)\text{log}P(y|x)
$$
则模型集合$\mathcal{C}$中条件熵$H(P)$最大的模型称为最大熵模型

### 最大熵模型的学习

$$
\underset{P \in \mathcal{C}}{\text{max}} \quad H(P)=-\underset{x,y}{\sum}\widetilde{P}(x)P(y|x)\text{log}P(y|x) \\
\text{s.t.} \quad E_P(f_i)=E_{\widetilde{P}}(f_i), \quad i=1,2,\cdots,n \\
\underset{y}{\sum}P(y|x)=1
$$
结论:

记$\psi(\omega)=\underset{P \in \mathcal{C}}{\text{min}}L(P,\omega)=L(P_{\omega},\omega)$
其中，$L$为拉格朗日函数

$$
\omega^*=\text{arg}\ \underset{\omega}{\text{max}}\psi(\omega)
$$

### 极大似然估计

训练数据的经验概率分布$\widetilde{P}(X,Y)$，条件概率分布$P(Y|X)$的对数似然函数表示为
$$
L_{\widetilde{P}}(P_\omega)=\text{log}\underset{x,y}{\Pi}P(y|x)^{\widetilde{P}(x,y)}=\underset{x,y}{\sum}\widetilde{P}(x,y)\text{log}P(y|x)
$$

## 最优化算法

### 改进的迭代尺度法

已知最大熵模型为
$$
P_\omega(y|x)=\frac{1}{Z_\omega(x)} \text{exp}(\stackrel{n}{\underset{i=1}{\sum}}\omega_i f_i(x,y)) \\
Z_\omega(x)=\underset{y}{\sum}\text{exp}(\stackrel{n}{\underset{i=1}{\sum}}\omega_i f_i(x,y))
$$
对数似然函数为
$$
L(\omega)=\underset{x,y}{\sum}\widetilde{P}(x,y)\stackrel{n}{\underset{i=1}{\sum}}w_i f_i(x,y)-\underset{x}{\sum}\widetilde{P}(x)\text{log}Z_\omega(x)
$$

