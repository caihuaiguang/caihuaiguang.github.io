## Llama 3

输入层是(128256, 2048)，意味着词表大小（token种类数）是128256的one hot向量，转换成 2048的低维稠密向量。

## 如何增强推理能力
. 串行：改变输入分布：1. Prompt: 输入前加几个例子（Chain of Thought prompt），Let's think step by step 2. x得到第一个回答y1, 然后将(x,y1)输入得到y2，后面的回答比前面的好
. 并行：生成多个输出，选择其中最好的：1. majority vote （哪个答案出现次数最多，不需要verifier） 2. Best of N: verifier model（可以是outcome RM也可以是process RM）给不同输出打分，选择最好的 3. Beam search: PRM给每一步打分 4. 树搜索：PRM给每一步打分，MCTS

## Group Query Attention (GQA)

在注意力头个数上进行的内存占用的改进，应用于decoder过程，因为encoder（如果有的话）内存不是主要瓶颈（因为编码是并行的）。

- （Autoregressive decoder inference is a severe bottleneck for Transformer models due to the memory bandwidth overhead from loading decoder weights and all attention keys and values at every decoding step ）。
- 解码过程的瓶颈来自自回归推理的特性。在自回归解码中，模型每生成一个新的词或标记时，必须依赖之前生成的所有结果，这意味着推理是逐步进行的，每一步都需要重新加载解码器的权重，以及之前步骤中所有的注意力键和值。由于每个步骤必须串行进行，并且模型在每一步都要处理之前生成的所有历史信息，导致频繁的内存访问，增加了内存带宽的压力。
- 相比之下，编码过程没有这个问题，因为编码是一次性并行处理所有输入的序列。编码器不需要逐步处理或依赖之前的步骤，因此它可以并行计算每个输入的位置，并且只需要一次加载权重和计算attention。这使得编码的内存带宽消耗更可控，且没有逐步解码时的瓶颈。
- 因此，自回归解码的逐步依赖性和频繁内存访问造成了明显的瓶颈，而编码过程可以更高效地进行并行计算。

Grouped-query attention divides query heads into G groups, each of which shares a single key head and value head.

64个头，论文中group建议为8：推理overhead几乎和MQA（Multi-query attention）一样，但性能和MHA（Multi-head attention）差不多

```python

# Group Query Attention (GQA) 机制详解以及手动实现计算
# https://blog.csdn.net/baoyan2015/article/details/137968408

import torch
# (batch_size, seq_len, num_heads, hidden_dim)
query = torch.randn(1,64,8,128)
key = torch.randn(1,64,2,128)
value = torch.randn(1,64,2,128)

query_groups = torch.chunk(query, 4, dim=2)
# print(query_groups[0].shape)
group_scores = []
for query_group in query_groups:
  scores = torch.matmul(query_group, key.transpose(-2, -1))
  # print(scores.shape)
  scores = torch.softmax(scores,dim=-1)
  group_scores.append(scores)
# print(len(group_scores))
attention_outputs = []
for scores in group_scores:
  outputs = torch.matmul(scores, value)
  attention_outputs.append(outputs)
attention_outputs = torch.cat(attention_outputs, dim=2)
print(attention_outputs.shape)

```

## MHA

一些比较高频的东西（针对基座算法/框架岗位为主，大体按重要性排序）：
多头注意力，频率太高了。coding轮，概念轮都考。复习的点包括：时间/空间复杂度，优化（kv-cache，MQA，GQA），手写多头代码。各种Norm，这个频率也不低，不过比较标准的内容，没有啥特意要说的，有的考手写，有的考概念和理解（为什么管用）。

框架相关内容，各种并行方式，优缺点。DeepSpeed，Megatron可以看看源代码，Flash-Attention等内容。这个点也经常考代码题。

BERT，GPT等比较主流大模型，一些细节，比如位置编码，训练loss，激活，架构些许不同这种。自回归重点。

大模型训练，这个可能主要是工作经验相关，经常问比如训练loss炸掉了，如何解决，一些技巧之类的。面试时有些面试官会问一些很细节的东西，感觉是在确认确实上手跑过基座训练不是吹水。

数据预处理，BPE，tokenization，mask相关概念和对模型/训练影响，数据配比（有paper）。

evaluation，如何评估大模型，安全性，有效性，公开数据，个别考过手写eval框架（多选，生成）。

根据投的岗位，多模态和RLHF内容可以适当看看。这俩感觉paper挺重要的，也大多研究岗位。楼主也少面了一些自动驾驶，RL啥的，不过结果不咋地。


```python
import torch
import torch.nn as nn
import numpy as np


'''手撕多头自注意力'''

class MultiHeadAttention(nn.Module):
    def __init__(self,input_dim,heads,d_model,dropout=0.1):
        super(MultiHeadAttention,self).__init__()
        self.d_model=d_model
        self.heads=heads
        self.input_dim=input_dim
        self.d_k=d_model//heads

        self.linear_q=nn.Linear(self.input_dim,self.d_model)
        self.linear_k=nn.Linear(self.input_dim,self.d_model)
        self.linear_v=nn.Linear(self.input_dim,self.d_model)

        self.dropout=nn.Dropout(dropout)

        self.out=nn.Linear(d_model,d_model)


    def forward(self,x,mask=None):
        batch_size,seq_len,hidden_size=x.shape
        q=self.linear_q(x).view(batch_size,-1,self.heads,self.d_k)
        k=self.linear_k(x).view(batch_size,-1,self.heads,self.d_k)
        v=self.linear_v(x).view(batch_size,-1,self.heads,self.d_k)

        q=q.transpose(1,2)
        k=k.transpose(1,2)
        v=v.transpose(1,2)   #batch,head,seq_len,d_k

        #求注意力
        score=torch.matmul(q,k.transpose(-2,-1))/np.sqrt(self.d_k)
        if mask is not None:
            score=score+mask

        att=torch.softmax(score,dim=-1)
        if self.dropout is not None:
            att=self.dropout(att)
        output=torch.matmul(att,v)    #(batch,head,sel_len,d_k)

        #拼接
        concat=output.transpose(1,2).contiguous().view(batch_size,-1,self.d_model)

        output=self.out(concat)
        return output

if __name__ == '__main__':
    pass
    batch=2
    seq_len=5
    input_dim=32
    head=2
    d_model=32
    x=torch.randn(size=(batch,seq_len,input_dim))
    attention=MultiHeadAttention(input_dim,head,d_model)
    print(attention(x).shape)
    #求掩码
    attention_mask=torch.tril(torch.ones(size=(seq_len,seq_len),dtype=torch.bool)).view(1, 1, seq_len, seq_len)
    attention_mask=attention_mask.to(dtype=torch.float16)
    attention_mask= (1.0-attention_mask)*torch.finfo(torch.float16).min
    print(attention_mask)
    print(attention(x).shape)
```

## 手撕多层MLP做回归
```python
import numpy as np

class MultiLayerNN:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, learning_rate=0.01):
        # 初始化参数
        self.learning_rate = learning_rate
        self.W1 = np.random.randn(input_size, hidden_size1) * 0.1
        self.b1 = np.zeros((1, hidden_size1))
        self.W2 = np.random.randn(hidden_size1, hidden_size2) * 0.1
        self.b2 = np.zeros((1, hidden_size2))
        self.W3 = np.random.randn(hidden_size2, output_size) * 0.1
        self.b3 = np.zeros((1, output_size))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def forward(self, X):
        self.a1 = self.relu(np.dot(X, self.W1) + self.b1)
        self.a2 = self.relu(np.dot(self.a1, self.W2) + self.b2)
        return np.dot(self.a2, self.W3) + self.b3

    def mse_loss(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    def backward(self, X, y_true, y_pred):
        num_samples = X.shape[0]

        # 反向传播计算梯度
        dL_dy_pred = (y_pred - y_true) / num_samples
        dL_dW3 = np.dot(self.a2.T, dL_dy_pred)
        dL_db3 = np.sum(dL_dy_pred, axis=0, keepdims=True)

        dL_da2 = np.dot(dL_dy_pred, self.W3.T) * self.relu_derivative(np.dot(self.a1, self.W2) + self.b2)
        dL_dW2 = np.dot(self.a1.T, dL_da2)
        dL_db2 = np.sum(dL_da2, axis=0, keepdims=True)

        dL_da1 = np.dot(dL_da2, self.W2.T) * self.relu_derivative(np.dot(X, self.W1) + self.b1)
        dL_dW1 = np.dot(X.T, dL_da1)
        dL_db1 = np.sum(dL_da1, axis=0, keepdims=True)

        # 更新权重和偏置
        self.W1 -= self.learning_rate * dL_dW1
        self.b1 -= self.learning_rate * dL_db1
        self.W2 -= self.learning_rate * dL_dW2
        self.b2 -= self.learning_rate * dL_db2
        self.W3 -= self.learning_rate * dL_dW3
        self.b3 -= self.learning_rate * dL_db3

    def train(self, X, y_true, epochs):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.mse_loss(y_pred, y_true)

            if epoch % 50 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

            self.backward(X, y_true, y_pred)

# 数据生成
np.random.seed(42)
X = np.random.rand(10, 3)  # 输入数据
y_true = np.random.rand(10, 2)  # 真实标签

# 初始化和训练神经网络
nn = MultiLayerNN(input_size=3, hidden_size1=5, hidden_size2=5, output_size=2)
nn.train(X, y_true, epochs=500)


```

## 手撕Beam Search
```python
import torch
import torch.nn.functional as F
 
def beam_search(LM_prob,beam_size = 3):
    batch,seqlen,vocab_size = LM_prob.shape
    #对LM_prob取对数
    log_LM_prob = LM_prob.log()
    #先选择第0个位置的最大beam_size个token，log_emb_prob与indices的shape为(batch,beam)
    log_beam_prob, indices = log_LM_prob[:,0,:].topk(beam_size,sorted = True)
    indices = indices.unsqueeze(-1)
    #对每个长度进行beam search
    for i in range(1,seqlen):
        #log_beam_prob (batch,beam,vocab_size),每个beam的可能产生的概率
        log_beam_prob = log_beam_prob.unsqueeze(-1) + log_LM_prob[:,i,:].unsqueeze(1).repeat(1,beam_size,1)
        #选择当前步概率最高的token
        log_beam_prob, index = log_beam_prob.view(batch,-1).topk(beam_size,sorted = True)
        #下面的计算：beam_id选出新beam来源于之前的哪个beam;index代表真实的token id
        #beam_id,index (batch,beam)
        beam_id = index//vocab_size
        index = index%vocab_size
        mid = torch.Tensor([])
        #对batch内每个样本循环，选出beam的同时拼接上新生成的token id
        for j,bid,idx in zip(range(batch),beam_id,index):
            x = torch.cat([indices[j][bid],idx.unsqueeze(-1)],-1)
            mid = torch.cat([mid,x.unsqueeze(0)],0)
        indices = mid
    return indices,log_beam_prob
 
if __name__=='__main__':
    # 建立一个语言模型 LM_prob (batch,seqlen,vocab_size)
    LM_prob = F.softmax(torch.randn([32,20,1000]),dim = -1)
    #最终返回每个候选，以及每个候选的log_prob，shape为(batch,beam_size,seqlen)
    indices,log_prob = beam_search(LM_prob,beam_size = 3)
    print(indices)
```


## 手撕单层MLP做二分类
```python
import numpy as np

class LogisticRegression:
    def __init__(self, input_size, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.W = np.random.randn(input_size) * 0.1  # 权重初始化
        self.b = 0  # 偏置初始化

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, X):
        z = np.dot(X, self.W) + self.b
        return self.sigmoid(z)

    def binary_cross_entropy(self, y_pred, y_true):
        # 计算二分类交叉熵损失
        return -np.mean(y_true * np.log(y_pred + 1e-15) + (1 - y_true) * np.log(1 - y_pred + 1e-15))

    def backward(self, X, y_true, y_pred):
        num_samples = X.shape[0]
        
        # 计算梯度
        dL_dy_pred = (y_pred - y_true) / num_samples
        dL_dW = np.dot(X.T, dL_dy_pred)
        dL_db = np.sum(dL_dy_pred)

        # 更新权重和偏置
        self.W -= self.learning_rate * dL_dW
        self.b -= self.learning_rate * dL_db

    def train(self, X, y_true, epochs):
        for epoch in range(epochs):
            y_pred = self.forward(X)  # 前向传播
            loss = self.binary_cross_entropy(y_pred, y_true)  # 计算损失

            if epoch % 50 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

            self.backward(X, y_true, y_pred)  # 反向传播

# 数据生成
np.random.seed(42)
num_samples = 100
input_size = 2

# 生成随机二分类数据
X = np.random.rand(num_samples, input_size)
y_true = (X[:, 0] + X[:, 1] > 1).astype(int)  # 设定分类规则

# 初始化和训练逻辑回归模型
logistic_model = LogisticRegression(input_size=input_size)
logistic_model.train(X, y_true, epochs=500)

# 测试模型
y_pred = logistic_model.forward(X)
y_pred_class = (y_pred > 0.5).astype(int)
accuracy = np.mean(y_pred_class == y_true)
print(f"Accuracy: {accuracy:.4f}")
```

## Root Mean Square Layer Normalization
```python
class RMSNorm(torch.nn.Module): # 这是 PyTorch 中构建神经网络模块的基本方式。通过继承 torch.nn.Module，
# 这个类能够使用 PyTorch 的功能，例如注册参数、管理模块中的子模块，以及支持自动微分计算等。

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) # 通常 self.weight 的大小应与输入 x 的最后一个维度（NLP中的hidden维度）相同。

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) # rsqrt返回1 / sqrt(x)。

    def forward(self, x):
        # 输入 x 转换为 float 类型进行标准化，然后再转换为输入的原始数据类型，以确保数值计算的精度和稳定性。
        output = self._norm(x.float()).type_as(x)
        return output * self.weight # weight 调整标准化输出的范围。
```
LayerNorm中不会像BatchNorm那样跟踪统计全局的均值方差，因此train()和eval()对LayerNorm没有影响。适合处理变长数据。

BN奏效的根本原因在于它让optimization landscape更平滑。


## flash_atten


# 大模型基础知识

LoRA (Low-Rank Adaptation):
. 初始化：A为0为均值的正太分布，B为全0矩阵，一般Rank取8
. 可以放到所有线性层旁边，要是考虑资源开销，可以优先放Q和V（影响注意力计算）。
. 乘到BA前面的缩放因子常数应该用alpha/sqrt(Rank)

Prefix-Tuning: 在每个attention Block中引入一组可训练的前缀向量，这些前缀向量作为额外的K和V，影响注意力的分布，从而引导模型适应特定任务。不改变原模型参数。
Prompt Tuning：针对特定任务，仅在输入添加可训练的prompts，

模型架构：
. Llama3位置编码只用在了Q，K，在attention层内部用


推理加速：
. kv cache，每次kv都一样，但q不同（这里是不是可以开发针对于Beam search和MCTS的cache算法）
. flash attention
. vLLM



## C++与操作系统
### 进程，线程，协程
1. 进程是操作系统进行资源分配和调度的基本单位。它包含了程序的代码、数据以及独立的运行空间。每个进程有自己独立的地址空间，进程间切换需要耗费较大的系统资源（如上下文切换）。
2. 线程是进程的执行单元，一个进程可以包含多个线程，线程共享进程的资源（如内存、文件描述符）。线程之间切换的开销相对进程要小，因为线程共享进程的内存空间，而进程切换则需要改变地址空间。
3. 协程是一种更轻量级的执行方式，与线程相比，协程不依赖操作系统的调度，而是在用户态自行调度。在单个线程内，可以有多个协程通过手动切换来执行任务。协程更适合用于 I/O 密集型任务，因其上下文切换开销极小，因为不需要陷入内核。
### 用户态、内核态
1. 用户态是应用程序正常执行的状态，CPU 没有直接访问硬件资源的权限。大部分的程序代码在用户态运行，当需要访问硬件资源（如文件操作、网络请求）时，会通过系统调用请求操作系统的内核来处理。
2. 内核态是操作系统的特权模式，具有对硬件资源的完全访问权限。系统调用和硬件中断都在内核态执行，用户态的程序在需要访问硬件时会切换到内核态。
### 关系
1. 进程和线程的切换：进程和线程的切换可能会涉及用户态和内核态的切换（尤其是在多核、多进程系统中）。例如，操作系统需要将 CPU 资源从一个进程分配给另一个进程时，通常要进行内核态的调度。
2. 协程的切换：协程是在用户态中进行调度的，它不需要频繁切换到内核态，因此比线程更加高效。协程通常运行在单个线程中，而线程可能需要操作系统调度，涉及内核态的上下文切换。


简而言之，线程是轻量级的进程，协程则是轻量级的线程。用户态和内核态的切换主要与资源访问和调度相关，协程通过减少这种切换来提高效率。