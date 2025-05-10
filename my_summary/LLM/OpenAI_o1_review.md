## Noam Brown

2014-2020: CMU PHD, his advisor is Tuomas Sandholm. (**Gabriele Farina**, CMU, 2016-2022)

### 2017: Science, ``Superhuman AI for heads-up no-limit poker: Libratus beats top professionals"

背景：CFR+算法解决了两人有限注德扑（Heads-up Limit Hold’em Poker is Solved, Science, 2015），但HUNL（Heads-Up No Limit Texas Hold’em，两人无限注德州扑克）是$10^{160}$个信息集，远超两人有限注德扑：$10^{14}$个信息集。

Methods:
- MCCFR：通过**蒙特卡洛**采样思想进行加速
- Subgame **Re-Solving** （相当于就是扑克游戏中的 **test-time computing**，在AI需要做决定时，实时求解策略）: an algorithm for computing a blueprint for the overall strategy, an algorithm that fleshes out the details of the strategy for subgames that are reached during play, and a self-improver algorithm that fixes potential weaknesses that opponents have identified in the blueprint strategy.

Re-Solving具体来说：
- 完整游戏进行约简：将完整游戏约简（状态合并、**减少搜索空间**）成一个相对简单的博弈，根据CFR算法求解得到一个粗略的策略：蓝图策略（Blueprint Strategy）
- 嵌套安全子博弈求解：在蓝图策略基础上，基于当前的牌面和比赛情况，构建一个全新的、更精细的子博弈，并对这个子博弈的策略进行实时求解。安全：完美信息博弈（比如围棋）的子博弈策略Re-Solving没有安不安全的概念，只需要考虑当前的子博弈即可，不用考虑对手策略是否发生变化；不完美信息（对应信息集，扩展式博弈中玩家可能不知道对手之前的动作；石头剪刀布这种矩阵式博弈也是不完美信息的）必须保证Re-Solving得到的子博弈策略不被对手利用。
- 自我提升：随着比赛的进行，利用对手的不在约简范围内的动作填补蓝图策略中缺失的分支，并为这些分支计算策略
- 后两个模块是对第一个模块的修补，用于降低因为约简而带来的影响。第二个模块用于处理当对手的动作不属于约简动作时的情况。第三个模块通过比赛中对手的动作，不断丰富步骤一得到的蓝图策略

成就：
- 第一个击败两人无限注德扑顶级选手的AI
- 将安全子博弈求解引入到大规模不完美信息博弈求解
- 算法**没有使用任何深度学习模型**

### 2019: Science, ``Superhuman AI for multiplayer poker"

Methods:
- Pluribus relies on **offline self-play** to build a base strategy, but then continues to **learn in real-time** during its online play. 
- In AI, two-player zero-sum games (such as heads-up hold'em) are usually won by approximating a **Nash** equilibrium strategy; however, this approach does not work for games with three or more players. Pluribus instead uses an approach which lacks strong theoretical guarantees, but nevertheless appears to work well empirically at defeating human players.
- The base strategy was computed in eight days, and at market rates would cost about $144 to produce, much smaller than contemporary superhuman game-playing milestones such as AlphaZero.

Offline self-play:
- 离线寻找蓝图策略时，利用了改进的MCCFR算法，提高了搜索效率

Real-time solving:
- 在线搜索实时策略时，假设对手可以随机地使用k种不同的策略，在搜索和评估过程中考虑多种可能，增强灵活性，防止Pluribus策略被针对，

成就：
- 第一个将AI应用到多人零和博弈中与人类职业选手对战，并取得战胜人类玩家的战绩
- 多人零和博弈的均衡解问题仍然悬而未决

Comments: Noam Brown, 2024-07-25:

> 5 years ago we revealed Pluribus, the first superhuman multiplayer poker AI. It cost only $150 to train. Why did poker take longer than Go? And how did it end up being so cheap? The answer is a cautionary tale on the danger of overoptimizing for benchmarks with relevance to LLMs today.
>
> The Annual Computer Poker Competition (ACPC) was the premier poker AI benchmark. Every year starting in 2006, all the poker AI research labs would gather at the ACPC and play their bots against each other. Winning the ACPC was prestigious so researchers put a lot of effort into their submissions.
>
> To keep costs low, **the ACPC limited submissions to using only two CPU cores for inference and a time limit of a few seconds per hand**. However, unlimited resources were allowed for pretraining.
>
> These constraints influenced research directions: teams spent $10,000+ on pretraining but neglected planning algorithms that used a lot of test-time compute. It turns out those planning algorithms were critical to beating top humans.
>
> Pluribus didn't qualify for the ACPC -- **its planning algorithms used 28 CPU cores for 20+ seconds per hand**. But it beat human experts.
>
> The lesson I learned from this is to not overoptimize for intermediate benchmarks. Yes, benchmarks can give an indication of progress, but focusing on them too much might lead you down a path that takes you away from the ultimate goal.
>
> I think about this often when I look at LLM benchmarks these days.


### 为什么要用Self-play? 去求Nash equilibrium

博弈意味着每个玩家的收益依赖于其他玩家的策略，不可能单独求得某个玩家的最佳策略，只能求纳什均衡：也就是所有玩家的策略组合。

在两人零和且对称博弈中，纳什均衡就是双方的不输策略，也就是单个人的最佳策略。

纳什均衡策略是保守/防御的策略，不一定能保证收益最大，另一种方式是建模对手已有策略：对手建模并压榨，但是有可能被对手反利用。

## OpenAI o1的哲学

### OpenAI o1 技术路线
1. 基于LLM已有的推理能力生成合理的推理过程，search的作用在于让推理过程**合理**还有**细粒度**的奖励信号。
2. 在这部分数据上Post-Training 模型，让其学会长程推理。
3. 模型训练好后，实际推理时也进行search来一步一步生成推理结果。

> [北大对齐团队独家解读：OpenAI o1开启「后训练」时代强化学习新范式](https://www.bilibili.com/video/BV15Rx5eXEnW/)

### 为什么要强调Post-Training？
提升模型长程推理能力：自回归模型在数学推理问题上很难进步的一点在于**无法对答案进行修正**，仅仅依靠生成式方法（Pre-Training的方法）和扩大参数规模带来的收益不会太大。

> [Training Verifiers to Solve Math Word Problems](https://arxiv.org/pdf/2110.14168)

反对意见：朱泽园的那几个

### Post-Training Scaling Law是否存在？什么形式？

Pre-Training Scaling Law 说的是训练Loss（越小模型能力越好）和数据量（token数 $D$）、模型参数量 $N$成幂律反比。
具体来说，训练计算量 $C=6ND$ FLOPs。这里反向传播是前向传播的2倍FLOPs，而前向传播计算量为 $2ND$（2的系数是乘法和加法都算1次浮点数预算，而 $m \times k$ 和 $k \times n$ 的矩阵乘法时每个结果元素计算都是 $k$ 次乘法和 $k$ 次加法，一共 $2\times m\times n\times k$ FLOPs）。

Post-Training Scaling Law 应该也是训练Loss和数据量（模型参数？这里应该已经固定模型参数了）成反比；但这里的数据的质量又依赖于已训练好模型的能力，且还有进行推理时的算力投入。

We have found that the performance of o1 consistently improves with more reinforcement learning (train-time compute) and with more time spent thinking (test-time compute).

> [Training Compute-Optimal Large Language Models](https://arxiv.org/pdf/2203.15556)
>
> [o1 官方报告: Learning to Reason with LLMs](https://openai.com/index/learning-to-reason-with-llms/)
>
> [Scaling Test-Time Compute Optimally Can be More Effective than Scaling LLM Parameters](https://arxiv.org/abs/2408.03314)

### Reasoning的定义

Noam Brown: Problems can 
1. Benefit from considering more options and think for longer. 
2. have a Generator-Verifier gap: It's really hard to generate a correct solusion, but much easier to recognize when you have one.

所有问题都可以分类为：相比于生成，更容易验证（数独）还是更难以验证（说出不丹的首都）。这两种极端的verifier有区别么？

### Reasoning如何做的

A clean and scalable appoach: Just to have the AI think for longer--Then it develops these abilities like backtracking and self-correction almost like emergently.

> [OpenAI的Noam Brown及其团队谈论了o1以及如何教大语言模型更好地推理](https://www.bilibili.com/video/BV19f43ejEk2)


### Test Time Compute Scaling Law

其出现意味着大模型不再受数据量（我们已经用上了所有的数据）、训练算力（大模型预训练太贵了）的限制，AI在可见的未来不会撞墙！这是另外一个维度的Scaling Law!

瓶颈？找到那些模型需要更多计算的输入

## o1 技术细节

### Self-play 如何做的（Reward Model 如何做的）
- 难点：LLM没有好的Verifier（Reward Model）：训练数据（human text）太多了，reward data太少了，难以对生成内容（比如两首不同的诗）评分
- 机会：reward data在增长；某些领域的数据更容易评分

方案：
1. RLHF：
    1. 收集Pairwise偏好数据 
    2. 基于偏好数据通过Ranking Loss训练Bradley–Terry Reward Model，从而将人类偏好融合到Reward Model中 
    3. 后期PPO训练时用Reward Model对模型输出打分
    > [Training language models to follow instructions](https://arxiv.org/pdf/2203.02155)
2. Process Reward Model: 数学问题，对每个解题步骤打分。
    > [Training Verifiers to Solve Math Word Problems](https://arxiv.org/pdf/2110.14168)
    > [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)
3. Generate Reward Model: 前面两种方案都是将LLM当作判别器，准确率仍然不足且难以Scaling到更复杂的问题和模型规模；GenRM先CoT自然语言推断得到判断和概率，然后Majority Voting得到平均概率。
    > [Generative Verifiers: Reward Modeling as Next-Token Prediction](https://arxiv.org/pdf/2408.15240)
4. Critic Model: 训练一个提升人类监督信号的模型（类似ICLR 2025用AI agent给予审稿人纠错和反馈）
    > [LLM Critics Help Catch LLM Bugs](https://arxiv.org/pdf/2407.00215)
    >
    > [Self-critiquing models for assisting human evaluators](https://arxiv.org/pdf/2206.05802)

### Reasoning 推演
1. CoT: 分步生成一系列中间推理步骤
    > [Large Language Models are Zero-Shot Reasoners](https://arxiv.org/pdf/2205.11916)
    >
    > [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/pdf/2201.11903)
    
    Majority 方式：
    > [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/pdf/2203.11171)  
    
    不需要Prompt：
    > [Chain-of-Thought Reasoning without Prompting](https://arxiv.org/pdf/2402.10200)  
2. MCTS: 将Token或者句子建模为节点，然后提供奖励信息
3. STaR：
    1. 造数据：基于LLM已有的推理能力生成合理的推理过程（利用带有推理过程的prompt对数据集中问题生成推理过程Rationale和答案，答案对的就加入数据集，错误则Hint模型给出正确答案？）
    2. 再将$(Question, Rationale, Answer)$ 微调模型
    3. 迭代：每获得一个数据集？，从原始模型进行fine-tune（这里应该有个优化：像Deep CFR中数据集构建那样给后面的数据加权重，还有是不是后面的模型推理能力更强就应该让他们推理深度更多一点？）
    > [STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465)
4. Quiet-STaR：STaR依赖于Few-Shot推理示例，且局限于特定结构话的任务（比如问题问答）；Quiet将显式Rationales的推理过程转化为模型内部隐式的推理过程，从而摆脱对外部示例的依赖。
    1. 引入可学习的<|startofthought|>和<|endofthought|> token来标记思维的开始和结束
    2. 同时利用带推理过程的结果与真实结果的分布差异引入奖励信号，用REINFORCE的方法优化生成的推理
    > [Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking](https://arxiv.org/pdf/2403.09629)
5. OpenAI o1? 1. 动态调整Tinking Token (Depth-limit search) 2. 对于复杂任务，如何对内部思考过程提供细粒度的奖励？ 
    关键是优化模型内部生成合理推理（隐式CoT）的过程：如何构造对应reward? 1. Tree-search 生成内部rationales 2. Process Reward来解决长程问题依赖性的挑战 3. Critic Model来解决复杂问题难以自身提供合理推理过程的挑战

    但我感觉是多次推理即可。见下面的图示
    > [o1 官方报告: Learning to Reason with LLMs](https://openai.com/index/learning-to-reason-with-llms/)




### RL如何做的
- Large process of learning from human data:Combine RL with other elements.

## 潜在前景方向
1. 大模型天花板？加数据，加模态
2. 合成数据？
    > [Reinforced Self-Training (ReST) for Language Modeling](https://arxiv.org/pdf/2308.08998)
    > [Self-Rewarding Language Models](https://arxiv.org/pdf/2401.10020)
    > [Training Large Language Models for Reasoning through Reverse Curriculum Reinforcement Learning](https://arxiv.org/abs/2402.05808)
    > [Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations](https://arxiv.org/pdf/2312.08935)
    > [Improve Mathematical Reasoning in Language Models by Automated Process Supervision](https://arxiv.org/pdf/2406.06592)
3. 如何平衡Post-training阶段训练和推理的算力
4. Scaling Test-Time Computation的方法
    1. 利用Verifier来搜索比较好的解法：并行采样，beam search，look ahead search (后两者需要PRM)
    2. 让模型自我修复，学会从错误中恢复的能力
    > [Large Language Monkeys: Scaling Inference Compute with Repeated Sampling](https://arxiv.org/pdf/2407.21787)
    > [Inference Scaling Laws: An Empirical Analysis of Compute-Optimal Inference for Problem-Solving with Language Models](https://arxiv.org/pdf/2408.00724)

### 其他资料
[LLM的范式转移：RL带来新的 Scaling Law](https://mp.weixin.qq.com/s/JPfgF6UtgIYwWXwNQHOoqQ)

[Summary of what we have learned during AMA hour with the OpenAI o1 team today](https://x.com/btibor91/status/1834686946846597281)

[Finding GPT-4’s mistakes with GPT-4 (CriticGPT介绍)](https://openai.com/index/finding-gpt4s-mistakes-with-gpt-4/)

[Generative Language Modeling for Automated Theorem Proving](https://arxiv.org/pdf/2009.03393)
### 资料来源

部分内容来自兴军亮老师（清华）和李凯老师（自动化所）的《计算博弈原理与应用》课件