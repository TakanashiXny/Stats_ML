# <center> 感知机

## 本章基本思路
1. 感知机模型介绍
- 输入空间到输出空间的函数映射
- 感知机的函数解释

2. 感知机学习策略
- （概念）线性可分性
- 感知机的损失函数
- 感知机学习算法
  - 原始形式
  - 对偶形式
- 算法收敛性

## 感知机模型介绍

定义：假设输入空间（特征空间）是$\mathcal{X} \subset \mathbf{R}^n$，输出空间是$\mathcal{Y}=\{+1, -1\}$。输入$x \in \mathcal{X}$表示实例的特征向量，对应于输入空间（特征空间）的点；输出$y \in \mathcal{Y}$表示实例的类别。由输入空间到输出空间的如下函数:
$$
f(x)=sign(\omega \cdot x + b)
$$
称为感知机。

几何解释：线性方程
$$
\omega \cdot x + b = 0
$$
对应于特征空间$\mathbf{R}^n$中的一个超平面，$\omega$是超平面的法向量，$b$是超平面的截距。

位于两个部分的点（特征向量）分别被分为正负两类，因此该平面又被称为分离超平面。

## 感知机学习策略

### 数据集的线性可分性

给定数据集
$$
T=\{(x_1,y_1),(x_2,y_2),\cdots,(x_N,y_N)\}
$$
其中，$x_i \in \mathcal{X}=\mathbf{R}^n$，$y_i \in \mathcal{Y}=\{+1,-1\},i=1,2,\cdots,N$，如果存在某个超平面$S$
$$
\omega \cdot x + b = 0
$$
能够将数据集的正实例点和负实例点完全正确地划分到超平面的两侧，即对所有$y_i=+1$的实例$i$，有$\omega \cdot x_i + b > 0$，对所有$y_i=-1$的实例$i$，有$\omega \cdot x_i + b < 0$，则称数据集为线性可分数据集。

### 感知机的损失函数

误分类点到超平面$S$的总距离
$$
\frac{1}{\parallel\omega\parallel}|\omega \cdot x_0 + b|
$$
对于误分类点而言，有
$$
-y_i(\omega \cdot x_i + b) > 0
$$
对应的到超平面的距离为
$$
-\frac{1}{\parallel\omega\parallel}y_i(\omega \cdot x_i + b)
$$
所有误分类点到超平面的总距离为
$$
-\frac{1}{\parallel\omega\parallel}\sum_{x_i \in M}y_i(\omega \cdot x_i + b)
$$
得到最终的损失函数为
$$
-\sum_{x_i \in M}y_i(\omega \cdot x_i + b)
$$

解释：
- 如果没有误分类点，损失函数值为0
- 误分类点离超平面越近，损失函数值越小

### 感知机学习算法

#### 原始形式

求解
$$
\underset{{\omega,b}}{\text{min}}L(\omega,b)=-\sum_{x_i \in M}y_i(\omega \cdot x_i + b)
$$

思想: 梯度下降法

$$
\nabla_{\omega}L(\omega,b)=-\sum_{x_i \in M}y_ix_i \\
\nabla_{b}L(\omega,b)=-\sum_{x_i \in M}y_i
$$

算法流程：

输入: 训练数据集$T=\{(x_1,y_1),(x_2,y_2),\cdots,(x_N,y_N)\}$，其中$x_i \in \mathcal{X}=\mathbf{R}^n$，$y_i \in \mathcal{Y}=\{+1,-1\},i=1,2,\cdots,N$

输出: $\omega,b$; 感知机模型$f(x)=\text{sign}(\omega \cdot x + b)$

1. 选取初值$\omega_0,b_0$;
2. 在训练集中选取数据$(x_i,y_i)$
3. 如果$y_i(\omega \cdot x_i + b) \leqslant 0$,
$$
\omega \leftarrow \omega+\eta y_i x_i \\
b \leftarrow b+\eta y_i
$$
4. 转至2，直至训练集中无误分类点

解释: 

当一个实例点被误分类，则调整超平面使分离超平面向该误分类点的一侧移动，以减小该误分类点与超平面的距离，直至超平面越过该误分类点。

#### 对偶形式

基本想法: 一个实例点被更新的次数越多，意味着它距离分离超平面越近，也就越难正确分类。

算法流程:

输入: 训练数据集$T=\{(x_1,y_1),(x_2,y_2),\cdots,(x_N,y_N)\}$，其中$x_i \in \mathcal{X}=\mathbf{R}^n$，$y_i \in \mathcal{Y}=\{+1,-1\},i=1,2,\cdots,N$

输出: $\omega,b$; 感知机模型$f(x)=\text{sign}(\stackrel{N}{\underset{j=1}{\sum}}\alpha_j y_j x_j \cdot x + b)$，其中$\alpha=(\alpha_1,\alpha_2,\cdots,\alpha_N)^T$.

1. $\alpha \leftarrow 0, b \leftarrow 0$
2. 在训练集中选取数据$(x_i,y_i)$
3. 如果$y_i(\stackrel{N}{\underset{j=1}{\sum}}\alpha_j y_j x_j \cdot x + b) \leqslant 0$,
$$
\alpha_i \leftarrow \alpha_i+\eta \\
b \leftarrow b+\eta y_i
$$
4. 转至2，直至训练集中无误分类点

### 算法收敛性

设训练数据集$T=\{(x_1,y_1),(x_2,y_2),\cdots,(x_N,y_N)\}$是线性可分的，其中$x_i \in \mathcal{X}=\mathbf{R}^n, y_i \in \mathcal{Y}=\{-1,+1\},i=1,2,\cdots,N$，则
- 存在满足条件的$\parallel\hat{\omega}_{opt}\parallel=1$的超平面$\hat{\omega} \cdot \hat{x}=\omega_{opt} \cdot x + b_{opt}=0$将训练数据集完全正确分开；且存在$\gamma>0$，对所有$i=1,2,\cdot,N$
$$
y_i(\omega_{opt} \cdot x + b_{opt}) \geqslant \gamma
$$
- 令$\gamma=\underset{i}{\text{min}}\{y_i(\omega_{opt} \cdot x + b_{opt})\}, R=\underset{1 \leqslant i \leqslant N}{\text{max}}\parallel\hat{x}_i\parallel$，则感知机算法在训练数据集上的误分类次数$k$满足不等式
$$
k \leqslant (\frac{R}{\gamma})^2
$$
