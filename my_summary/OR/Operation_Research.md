## 运筹学
我想分享运筹学基础的概念和在计算机中的应用。
我认为，运筹学主要描述建模的过程，由各种的假设得到各种分布、过程，进而进行决策。
#### 无记忆性和随机变量的分布
1. 唯一具备无记忆性的连续随机变量分布是指数分布。
$$
P\{X>s+t\}=P\{X>s\}P\{X>t\}
\Leftrightarrow
F(x)=P\{X\le x\}=\begin{cases} 
1-e^{-\lambda x}, &\text{if}\ x\ge 0,\\
0,& \text{if}\ x<0.
\end{cases}
$$
我们只验证无记忆性推导到指数分布：设$g(s) = P\{X>s\}$，则:
$$
\begin{aligned}
&P\{X>s+t\}=P\{X>s\}P\{X>t\}\\
&\Leftrightarrow
g(s+t)=g(s)g(t)\\
&\Rightarrow
g(2/n)=g(1/n)g(1/n)\\
&\Rightarrow
g(m/n)=g(1/n)^m,g(1)=g(1/n)^n\\ 
&\Rightarrow
g(m/n)=g(1/n)^m=(g(1)^{1/n})^m=g(1)^{m/n}\\ 
&\Rightarrow
g(x)=g(1)^x = e^{-\lambda x}, \lambda = -\ln g(1)
\end{aligned}
$$
2. 唯一具备无记忆性的离散随机变量分布是几何分布。
$P\{X> s+t\}=P\{X\ge s\}P\{X>t\} \Leftrightarrow P(x=n)=(1-p)^{n-1}p,n\in \mathbb{N}^+$
同样我们验证从左边式子推导到右边：
$$
\begin{aligned}
& P\{X> s+t\}=P\{X\ge s\}P\{X>t\}
\\
&\Rightarrow
 P\{X> k+t\}=P\{X\ge k\}P\{X>t\} 且P\{X> s+k\}=P\{X\ge s\}P\{X>k\}
\\
&\Rightarrow
P\{X\ge k\}\sum_{t=1}^\infin P\{X>t\} =\sum_{t=1}^\infin P\{X> k+t\}=\sum_{s=1}^\infin P\{X> k+s\} =P\{X>k\} \sum_{s=1}^\infin P\{X\ge s\}
\\
&\Leftrightarrow
P\{X\ge k\}\sum_{t=1}^\infin P\{X>t\} =P\{X>k\} \sum_{s=1}^\infin P\{X\ge s\} 
\\
&\Leftrightarrow
P\{X\ge k\}\sum_{t=2}^\infin P\{X\ge t\} =P\{X\ge k+1\} \sum_{t=1}^\infin P\{X\ge t\}
\end{aligned}
$$
我们取$k=1$，设 $p=P\{x=1\}$，设 $\alpha = \sum_{t=2}^\infin P\{X\ge t\}$，则$1*\alpha = (1-p)(1+\alpha)$，得到$\alpha=(1-p)/p$，代入原式得到$P\{X\ge k\}\alpha=P\{X\ge k+1\} (1+\alpha)$，即$(1-p)P\{X\ge k\}=P\{X\ge k+1\} $。于是$P\{X\ge k\}=(1-p)^{k-1}$，因此$P\{X= k\}=P\{X\ge k\}-P\{X\ge k+1\}=p(1-p)^{k-1}$

3. 泊松（Poisson）分布
$P\{X = i\}=e^{-\lambda}\frac{\lambda^i}{i!},i=0,1,\dots $。
性质：二项分布（$n$次独立重复试验，每次实验成功概率为$p$，则实验成功$k$次的概率为$C_n^kp^k(1-p)^{n-k}$）在$n$很大，$p$很小的时候近似于泊松分布。

#### 随机过程
我们平常接触得比较多的是随机变量的分布，但要是考虑时间流动，随机变量的分布可能会变。我们怎么建模这种现象？随机过程！
随机过程$\mathbf{X}=\{X(t),t\in T\}$是一组随机变量。也就是说，对于任意的$t\in T$，$X(t)$是一个随机变量。

随机过程特殊性质：
- 独立增量（independent increments）：对于所有的$t_0< t_1< t_2<\dots < t_n$，随机变量$X(t_1)-X(t_0),X(t_2)-X(t_1),\dots,X(t_n)-X(t_{n-1})$相互独立。
- 稳定增量（stationary increments）：$\forall s, X(s+t)-X(s)$都有相同的分布。

自然语言处理中“语言是稳态的可遍历性的随机过程”中“稳态”就是上面的稳定增量，指的是“从今天《人民日报》和明天《人民日报》分别采样得到的汉语统计特征是一样的”。
##### 泊松过程
记$N(t)$为$[0,t]$内发生的事件个数，则$\{N(t),t\ge 0\}$被称作计数过程。
泊松过程（参数为$\lambda$）是满足如下性质的计数过程：
1. $N(0)=0$
2. 随机过程是独立增量的
3. 任意一个长度为$t$的时间段内事件数量服从$\lambda t$的泊松分布。即$\forall t\ge 0, \forall s\ge 0,P\{N(s+t)-N(s)=n\}=e^{-\lambda t}\frac{(\lambda t)^n}{n!}$
显然泊松过程也具备稳定增量的性质。

##### 泊松过程和指数分布
记$X_n$为第$n-1$个事件和第$n$个事件之间的时间长度。泊松过程（参数为$\lambda$）等价于$X_n(\forall n=1,2,\dots)$ 服从参数为$\lambda$的指数分布。
证明：
$$
P\{X_1\ge t\}=P\{N(t)=0\}=e^{-\lambda t}\\
P\{X_2>t|X_1=s\}=P\{0 \ events \ in \ (s,s+t]| X_1=s\}
\\=^{independent}P\{0 \ events \ in \ (s,s+t]\}
\\=^{stationary}P\{0 \ events \ in \ (0,t]\}=e^{-\lambda t}
$$ 
#### 马尔科夫性与马尔科夫过程
- Markov Property:给定当前状态，未来状态和过去状态无关。
$P\{\text{Future}|\text{Present,Past}\}=P\{\text{Future}|\text{Present}\}$
- 马尔科夫过程分类：
![](Markov_category.png)
#### DTMC
- homogeneous Discrete-Time Markov Chain：$P\{X_{n+1}=j|X_n=i\}=p_{ij}$，时间、状态离散，且转移概率和时间无关。
- Chapman-Kolmogorov 等式：记从 $i$ 状态花 $n$ 步转移到 $j$ 状态的概率为$p^n_{ij}$，则$p_{ij}^{n+m}=\sum_{k=0}^\infin p_{ik}^{n}p_{kj}^{m}, \forall n,m\ge 0,\forall i,j$。矩阵形式：$P^{(n+m)} =P^{(n)}P^{(m)}, P^{(n)}_{ij}=p_{ij}^{n}$
- state 的分类：recurrent（常返）和transient（瞬态）。$j$是recurrent的当且仅当$\sum_{n=1}^\infin p_{jj}^{n} = \infin$。
- 例子1：PageRank：若$ij$间有链接，则$p_{ij}=1/d_i$，其中$d_i$表示$i$的度数。Page $i$ 的分数记为$\pi_i$的话，则$(\pi_1,\pi_2,\dots) = \pi = \pi P$。
- 例子2：HMM（隐马尔科夫模型）：每个输出序列对应着一个隐藏状态序列，输出是关于隐藏状态的随机变量，隐藏状态是个DTMC。
- 例子3：CRF（条件随机场）：NLP和图像处理中的序列标注和结构划分问题。给定观察序列$X$，输出标识序列$Y$，通过计算$P(Y|X)$求解最优标注序列。
若对于无向图$G=(V,E)$，$V$中每个节点对应于$Y_v$的随机变量，且满足$p(Y_v|X,Y_w,w\ne v) = p(Y_v|X,Y_w,(v,w)\in E，\forall v,w\in V $，则$(X,Y)$为条件随机场。
比如NLP中词性标注任务：$X$就是句子，$Y$是我们要给每个字标注的词性，词性之间有马尔科夫性（当前字的词性仅仅和前面1个字的词性有关）
比如图像处理中的分割任务：$X$就是所有像素，$Y$就是我们要给每个像素打的标签，某个像素点的标签仅仅和相邻4个像素点的标签有关。

#### CTMC
- homogeneous Continuous-Time Markov Chain：$P_{ij}(t) = P\{X(s+t)=j|X(s)=i\},\forall s$。CTMC=DTMC+在状态上的停留时间服从指数分布。
- 生灭过程（Birth and Death process）：某个时刻系统内有$n$个人，那么下一个人到达系统的时间间隔服从$\lambda_n$的指数分布，系统内下一个离开的人的时间间隔服从$\mu_n$的指数分布。
#### queueing theory
- 网络流量测量的几个重大发现：数据对话请求（session）的到达服从泊松分布，或者说，用户访问服从泊松分布。
- 数据包（package）的到达不服从泊松分布（数据包到达具有突发性）
- 为什么要研究泊松分布？还记得我们先前说二项分布近似于泊松分布在什么样的场景下吗？人数量很多（$n$），每个人在此时选择此项服务的概率（$p$）很小。
因此，现实中很多场景会出现泊松分布。
泊松过程建模的是时间段内发生的事件数量，等价于事件间隔时间服从指数分布。

因此最经典的排队论模型：M/M/1，M（下一个用户到达时间服从指数分布）/M（服务器服务当前一个人的时间服从指数分布）/1（一个服务器）
#### MDP
MDP：在DTMC的基础上，引入动作（action）和奖励（reward）的概念，动作（action）对应着我们可以影响状态和状态之间的转移概率，每个状态对应着一个值（v，代表这个状态下获得的期望回报），奖励在采取动作后获得。
#### Inventory theory
Inventory theory：每天早上进货，当天的需求未知，进货进多了卖不，进货进少了利润不高。
- newsvendor model（报童模型）：单份报纸卖 $p$元，单份报纸成本为$c$元，当天需求$D$的累积分布函数$\Phi()$未知，求最优进货数量$y$？
期望收益为:$E_D[p\min(D,y)-cy]$，求导为0得到$\Phi(y^*)=\frac{p-c}{p}$。
证明：
$$
\begin{aligned}
&E_D[p\min(D,y)-cy]\\
=&E_D[pD+p\min(0,y-D)-cD+c(D-y)]\\
=& E_D[pD-cD]+E_D[p\min(0,y-D)]+cE_D[(D-y)]\\
=& (p-c)E_D[D]-pE_D[\max(0,D-y)]+cE_D[(D-y)^+-(D-y)^-]\\
=& (p-c)E_D[D]-pE_D[(D-y)^+]+cE_D[(D-y)^+-(D-y)^-]\\
=& (p-c)E_D[D]-(p-c)E_D[(D-y)^+]-cE_D[(D-y)^-]
\end{aligned}
$$ 
$\max$上式等价于最小化$L = (p-c)E_D[(D-y)^+]+cE_D[(D-y)^-]=(p-c)\int_y^\infin(z-y)\phi(z)d(z)+c\int_0^y(y-z)\phi(z)dz$。而$\frac{\partial L}{\partial y}=(p-c)[-1(1-\Phi(y))]+c\Phi(y)=c-p+p\Phi(y)=0$
这里用了个公式：$\frac{d}{d t}\int_{h(t)}^{g(t)}F(x,t)dx = F(h,t)\frac{dh(t)}{t}-F(g,t)\frac{dg(t)}{t}+\int_{h(t)}^{g(t)}\frac{d F(x,t)}{d t}dx$。
因此$\frac{d}{dy}\int_y^\infin(z-y)\phi(z)d(z) =0-0*1+\int_{y}^{\infin}-\phi(z)d(z)=-(1-\Phi(y))$。 
### 下半部分的主题
- 博弈论
- 线性规划
- 非线性规划
- 组合优化
- 复杂性理论