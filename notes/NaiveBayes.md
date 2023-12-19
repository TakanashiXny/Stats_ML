# <center> 朴素贝叶斯法

## 基本思路

1. 朴素贝叶斯法的学习与分类目标
   - 基本方法
   - 贝叶斯估计
   - 假设
2. 朴素贝叶斯法的参数估计
   - 极大似然估计

## 朴素贝叶斯法的学习与分类目标

设输入空间$\mathcal{X} \subset \mathbf{R}^n$为$n$维向量的集合，输出空间为类标记集合$\mathcal{Y}=\{c_1,c_2,\cdots,c_K\}$。输入为特征$x \in \mathcal{X}$，输出为类标记$y \in \mathcal{Y}$。$X$是定义在输入空间$\mathcal{X}$上的随机向量，$Y$是定义在输出空间$\mathcal{Y}$上的随机变量。$P(X,Y)$是$X$和$Y$的联合概率分布。训练数据集
$$
T=\{(x_1,y_1),(x_2,y_2),\cdots,(x_N,y_N)\}
$$
由$P(X,Y)$独立同分布产生。

朴素贝叶斯法预测结果来自
$$
y=\text{arg} \underset{c_k}{\text{max}}P(Y=c_k|X=x)
$$

由贝叶斯法则可知
$$
y=\text{arg} \underset{c_k}{\text{max}}\frac{P(X=x|Y=c_k)P(Y=c_k)}{\underset{k}{\sum}P(X=x|Y=c_k)P(Y=c_k)}
$$

所以该模型需要学习
- $P(Y=c_k)$
- $P(X=x|Y=c_k)$

关注$P(X=x|Y=c_k)$

$$
P(X=x|Y=c_k)=P(X^{(1)}=x^{(1)},\cdots,X^{(n)}=x^{(n)}|Y=c_k), k=1,2,\cdots,K
$$

为了减少参数量，做出了**条件独立性**的假设。即
$$
P(X=x|Y=c_k)=\stackrel{n}{\underset{j=1}{\Pi}}P(X^{(j)}=x^{(j)}|Y=c_k)
$$

将其带入目标函数，得到最终的形式
$$
y=\text{arg}\underset{c_k}{\text{max}}P(Y=c_k)\underset{j}{\Pi}P(X^{(j)}=x^{(j)}|Y=c_k)
$$

## 朴素贝叶斯的参数估计

### 极大似然估计

$$
P(Y=c_k)=\frac{\stackrel{N}{\underset{i=1}{\sum}}I(y_j=c_k)}{N}, k=1,2,\cdots,K
$$
设第$j$个特征$x^{(j)}$可能取值的集合为$\{a_{j1},a_{j2},\cdots,a_{jS_j}\}$，条件概率$P(X^{(j)}=a_{jl}|Y=c_k)$的极大似然估计是
$$
P(X^{(j)}=a_{jl}|Y=c_k)=\frac{\stackrel{N}{\underset{i=1}{\sum}}I(x_i^{(j)}=a_{jl},y_i=c_k)}{\stackrel{N}{\underset{i=1}{\sum}}I(y_i=c_k)} \\
j=1,2,\cdots,n; l=1,2,\cdots,S_j; k=1,2,\cdots,K
$$

### 贝叶斯估计

极大似然估计可能会出现要估计的概率值为0的情况.使用**拉普拉斯平滑**技术可以避免

$$
P(X^{(j)}=a_{jl}|Y=c_k)=\frac{\stackrel{N}{\underset{i=1}{\sum}}I(x_i^{(j)}=a_{jl},y_i=c_k)+\lambda}{\stackrel{N}{\underset{i=1}{\sum}}I(y_i=c_k)+S_j\lambda} \\

P_\lambda(Y=c_k)=\frac{\stackrel{N}{\underset{i=1}{\sum}}I(y_j=c_k)+\lambda}{N+K\lambda}
$$

