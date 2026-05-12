# Calib-OPD: Calibrated On-Policy Distillation for Hallucination Mitigation

## 1. 核心想法

Calib-OPD 的目标不是简单让 student 模仿 teacher，而是让 student 在自己的 on-policy 生成轨迹上学习三种能力：

1. teacher 确定且有证据时，学习 teacher 的答案能力和推理模式；
2. teacher 不确定、证据不足或问题不可回答时，学习拒答、低置信表达或风险敏感行为；
3. student 已经产生 hallucination 时，不盲目蒸馏错误上下文，而是抑制错误 claim 或降低过度自信。

一句话概括：

> 普通 OPD 教模型怎么模仿 teacher；Calib-OPD 教模型什么时候模仿、什么时候拒答、什么时候不要模仿。

这个方向把 OPD 和 hallucination mitigation 结合起来，核心研究问题是：

> 在 student 自己生成的 on-policy 轨迹中，哪些 token、claim 或 response 应该被蒸馏，哪些应该被拒绝蒸馏，哪些应该被惩罚？

## 2. 背景与动机

### 2.1 普通 OPD 的问题

普通 On-Policy Distillation 通常流程如下：

1. student 给定输入 `x`，自己采样回答 `y ~ pi_s(. | x)`；
2. 在 student 生成的轨迹 `y` 上查询 teacher 的 token distribution；
3. 通过 KL loss 让 student 分布靠近 teacher 分布。

典型损失可以写成：

```text
L_OPD = E_{x, y ~ pi_s} [ D_KL(pi_s(. | x, y_<t) || pi_t(. | x, y_<t)) ]
```

如果采用 top-k OPD，则只在 teacher 或 student 的 top-k token support 上计算 KL。

这种方法在能力蒸馏中有效，但 hallucination 场景下有明显风险：

- student 的 on-policy trajectory 可能本身就是 hallucinated trajectory；
- teacher 在错误上下文下给出的 next-token distribution 可能只能局部补救，不能保证整个回答事实正确；
- reverse KL 往往更 mode-seeking，可能让 student 更自信地输出某个高概率但错误的 continuation；
- OPD loss 本身不区分正确回答、合理拒答和 hallucination。

因此，普通 OPD 可能出现一种现象：

> teacher 能力被蒸馏了，但 student 的过度自信和错误回答倾向也被强化了。

### 2.2 Hallucination 的训练根源

在 hallucination 相关论文中，一个核心观点是：标准训练目标经常鼓励模型当一个 good test-taker，而不是 honest communicator。

尤其是 RLVR 中常见的 binary reward：

```text
correct: +1
wrong: -1
```

这种奖励把“不知道”和“答错”都视为失败，导致模型只要认为有一点概率答对，就倾向于猜测。TruthRL 和 Behaviorally Calibrated RL 都指出，更合理的目标应该区分：

- correct answer；
- abstention / uncertainty；
- hallucinated answer。

例如 TruthRL 使用 ternary reward：

```text
correct: +1
uncertain / abstain: 0
wrong / hallucinated: -1
```

Behaviorally Calibrated RL 进一步引入用户风险阈值 `t`，理想行为是：

```text
answer if confidence p >= t
abstain if confidence p < t
```

Calib-OPD 可以吸收这些思想，但相比纯 RL，它多了 token-level 或 claim-level teacher distribution supervision。

## 3. 研究假设

Calib-OPD 可以围绕三个核心假设展开：

### H1: Blind imitation amplifies hallucination

普通 OPD 对所有 student-generated trajectories 一视同仁，在不确定或错误上下文中继续蒸馏 teacher distribution，可能会放大学生的过度自信和 hallucination。

### H2: Calibration-aware gating improves truthfulness

如果根据 teacher confidence、evidence support、student overconfidence 和 claim correctness 来选择性蒸馏，可以显著降低 hallucination，同时保持或提升 accuracy。

### H3: Abstention is distillable

拒答、低置信表达、风险敏感回答不是简单的 prompt 行为，而是一种可以通过 OPD 从 teacher 迁移到 student 的 meta-skill。

## 4. 方法总览

Calib-OPD 的 pipeline 包含五个阶段：

1. Student on-policy rollout；
2. Teacher / verifier 标注正确性、置信度和证据支持；
3. Calibration gate 计算；
4. Selective OPD / abstention distillation / hallucination suppression；
5. 可选的 RLVR 或 GRPO 联合优化。

整体结构如下：

```text
Input x
  |
  v
Student pi_s rollout: y ~ pi_s(. | x)
  |
  v
Teacher / verifier evaluates:
  - correctness
  - confidence
  - should_abstain
  - claim support
  - hallucinated spans
  |
  v
Calibration gate:
  - g_pos for supported imitation
  - g_abs for abstention
  - g_neg for hallucination suppression
  |
  v
Training losses:
  - gated OPD loss
  - abstention loss
  - unlikelihood loss
  - confidence calibration loss
```

## 5. 详细算法设计

### 5.1 Student On-Policy Rollout

给定问题 `x`，student 采样 `K` 个回答：

```text
y_1, y_2, ..., y_K ~ pi_s(. | x)
```

输出格式可以有三种粒度。

#### Response-level 格式

```text
Question: {question}

Answer: {answer}
Confidence: {p}
```

适合 SimpleQA、TruthfulQA、CRAG 这类短答案或事实问答任务。

#### Reasoning-level 格式

```text
<think>
step-by-step reasoning
</think>
Final Answer: {answer}
Confidence: {p}
```

适合数学推理和多跳问答。

#### Claim-level 格式

```html
<Claim confidence="0.93">The first event happened in 1998.</Claim>
<Claim confidence="0.41">The second event was caused by policy X.</Claim>
Final Answer: ...
```

适合长答案、RAG、医学、法律和 biography 类 factuality 任务。

### 5.2 Teacher / Verifier 评估

对每个 student rollout，teacher 或 verifier 输出结构化标注：

```json
{
  "response_label": "correct | incorrect | uncertain | should_abstain",
  "teacher_confidence": 0.87,
  "student_confidence": 0.62,
  "evidence_support": 1,
  "claims": [
    {
      "span": "claim text",
      "label": "supported | unsupported | wrong | unverifiable",
      "teacher_confidence": 0.91,
      "evidence_support": 1
    }
  ]
}
```

teacher / verifier 可以来自以下来源：

1. 更强 LLM，例如 GPT-5、Gemini、Claude、Qwen-Max；
2. LLM-as-judge；
3. programmatic verifier，例如数学答案验证器；
4. retrieval evidence verifier；
5. self-consistency verifier；
6. teacher logprob 或 entropy；
7. 多 teacher ensemble。

最推荐的实际实现是混合 verifier：

```text
math task:
  programmatic verifier + teacher confidence

factual QA:
  exact/semantic judge + teacher confidence

RAG task:
  evidence support verifier + answer judge + confidence judge
```

### 5.3 Calibration Gate

Calib-OPD 的关键是 gate。gate 决定每个样本、token 或 claim 是否适合做 OPD。

定义三个 gate：

```text
g_pos: 正向蒸馏权重
g_abs: 拒答蒸馏权重
g_neg: 幻觉抑制权重
```

#### 正向蒸馏 gate

teacher 置信度高，证据支持强，且样本正确时，进行 OPD：

```text
g_pos = c_t * s_e * I[label = correct]
```

其中：

- `c_t` 是 teacher confidence；
- `s_e` 是 evidence support score；
- `I[label = correct]` 表示回答正确。

如果想突出 student 不会但 teacher 会的样本，可以加上 student uncertainty：

```text
g_pos = c_t * s_e * (1 - c_s)
```

这类样本最适合蒸馏，因为 teacher 有知识，student 还没有掌握。

#### 拒答 gate

当问题不可回答、teacher 低置信、证据不足或 gold answer 标注为 unknown 时，训练 student 拒答：

```text
g_abs = I[should_abstain] * (1 - c_t)
```

也可以使用 evidence-aware 版本：

```text
g_abs = I[not_supported] * (1 - s_e)
```

#### 幻觉抑制 gate

当 student 高置信答错，而 teacher 或 evidence 判断其错误时，需要显式抑制：

```text
g_neg = c_s * I[label = hallucinated]
```

如果有 teacher confidence，也可以写成：

```text
g_neg = c_s * (1 - c_t) * I[label = hallucinated]
```

这类样本代表 student overconfidence，是 hallucination mitigation 的重点。

## 6. 损失函数

Calib-OPD 的总损失可以写成：

```text
L = alpha * g_pos * L_opd
  + beta  * g_abs * L_abs
  + gamma * g_neg * L_ul
  + delta * L_conf
  + eta   * L_rl
```

其中：

- `L_opd` 是 gated OPD loss；
- `L_abs` 是 abstention distillation loss；
- `L_ul` 是 hallucination unlikelihood loss；
- `L_conf` 是 confidence calibration loss；
- `L_rl` 是可选 RLVR / GRPO reward loss。

### 6.1 Gated OPD Loss

在 teacher 可靠的位置做 OPD：

```text
L_opd = D_KL(pi_s(. | x, y_<t) || pi_t(. | x, y_<t))
```

也可以使用 forward KL：

```text
L_opd_forward = D_KL(pi_t(. | x, y_<t) || pi_s(. | x, y_<t))
```

或 symmetric KL：

```text
L_opd_sym = D_KL(pi_s || pi_t) + D_KL(pi_t || pi_s)
```

推荐主实验使用 top-k reverse KL，因为 OPD 相关论文表明 top-k overlap 对蒸馏稳定性和有效知识迁移很重要。

```text
L_topk = D_KL(pi_s^topk(. | x, y_<t) || pi_t^topk(. | x, y_<t))
```

### 6.2 Abstention Distillation Loss

当 teacher 判断应该拒答时，训练 student 输出 `<IDK>` 或等价拒答：

```text
L_abs = - log pi_s(<IDK> | x)
```

如果使用自然语言拒答，可以设定多个等价模板：

```text
I don't know.
I cannot determine from the given evidence.
The question is not answerable.
```

也可以让 teacher 生成 abstention response，然后做 sequence-level NLL：

```text
L_abs = - sum_t log pi_s(y_abs_t | x, y_abs_<t)
```

### 6.3 Hallucination Unlikelihood Loss

对于被标注为 hallucinated 的 token、span 或 claim，使用 unlikelihood loss：

```text
L_ul = - sum_{z in H} log(1 - pi_s(z | context))
```

其中 `H` 是 hallucinated token/span 集合。

如果只有 claim-level 标注，可以对整个错误 claim 的 token 做 unlikelihood：

```text
L_ul_claim = - sum_{t in wrong_claim} log(1 - pi_s(y_t | x, y_<t))
```

注意：unlikelihood 不能过强，否则可能导致模型过度拒答或语言流畅性下降。可以只对 entity、date、number、relation 等事实密集 token 做惩罚。

### 6.4 Confidence Calibration Loss

如果 student 输出 confidence `p_s`，可以使用 Brier loss：

```text
L_conf = (p_s - I[correct])^2
```

如果采用 teacher soft confidence：

```text
L_conf = (p_s - c_t)^2
```

也可以使用 cross-entropy 风格：

```text
L_conf_ce = - c_t log p_s - (1 - c_t) log(1 - p_s)
```

对于 risk-conditioned 场景，可以加入用户阈值 `tau`：

```text
answer if p_s >= tau
abstain if p_s < tau
```

目标是让 hallucination rate 随 `tau` 增大而单调下降。

### 6.5 Optional RL Loss

Calib-OPD 可以和 GRPO/RLVR 联合：

```text
reward =
  +1 correct
   0 abstain
  -1 hallucinated
```

或使用 risk-aware reward：

```text
correct: +1
abstain: 0
wrong: - tau / (1 - tau)
```

联合训练时，OPD 负责 token-level guidance，RL 负责 outcome-level pressure。

## 7. 算法伪代码

```text
Algorithm: Calib-OPD

Input:
  training prompts D
  student model pi_s
  teacher model pi_t
  verifier V
  weights alpha, beta, gamma, delta

for each training step:
  sample batch x from D

  # Student on-policy generation
  generate K responses y_1, ..., y_K ~ pi_s(. | x)

  for each response y_i:
    teacher_logits = pi_t(. | x, y_i_<t)
    eval = V(x, y_i, teacher=pi_t)

    compute:
      g_pos from teacher confidence, evidence support, correctness
      g_abs from should_abstain or low teacher confidence
      g_neg from student overconfidence and hallucination label

    L_opd = KL(pi_s || pi_t) on supported tokens/claims
    L_abs = -log pi_s(<IDK> | x) if should abstain
    L_ul = unlikelihood over hallucinated spans
    L_conf = confidence calibration loss

  update pi_s by minimizing:
    L = alpha*g_pos*L_opd
      + beta*g_abs*L_abs
      + gamma*g_neg*L_ul
      + delta*L_conf
```

## 8. 三个可发表版本

### Version A: Response-Level Calib-OPD

这是最简单、最容易实现的版本。

每个 response 只标注整体：

```text
correct
hallucinated
should_abstain
```

训练规则：

```text
correct:
  do OPD

hallucinated:
  do unlikelihood or negative RL

should_abstain:
  distill <IDK>
```

优点：

- 工程复杂度最低；
- 适合先跑 SimpleQA、TruthfulQA、CRAG；
- 可以快速证明普通 OPD 与 Calib-OPD 的差异。

缺点：

- 对长答案和 RAG 的细粒度错误定位不足。

### Version B: Confidence-Gated Calib-OPD

引入 teacher confidence 和 student confidence。

核心规则：

```text
teacher high, student low:
  strong OPD

teacher low, student high:
  hallucination suppression

teacher low, student low:
  abstention distillation

teacher high, student high:
  weak OPD or skip
```

这个版本最适合分析知识边界：

- teacher 知道、student 不知道；
- teacher 不确定、student 却过度自信；
- teacher 和 student 都不确定。

优点：

- 与 hallucination calibration 结合最自然；
- 能做很多机制性图表；
- 容易形成论文主线。

### Version C: Claim-Level Calib-OPD

这是最有创新性的版本。

流程：

1. student 生成长答案；
2. teacher / judge 抽取 claims；
3. 每个 claim 标注 supported / unsupported / wrong / unverifiable；
4. supported claim 做 OPD；
5. wrong claim 做 unlikelihood；
6. unverifiable claim 训练低 confidence 或 citation request。

适合 RAG 和长答案 factuality。

优点：

- 可以直接打长文本 hallucination；
- 更符合真实应用；
- 论文亮点更强。

缺点：

- 需要 claim extractor 和 judge；
- 标注噪声更高；
- 训练格式更复杂。

## 9. 实验设计

### 9.1 数据集

推荐分三类数据集。

#### 知识问答和幻觉评估

- SimpleQA；
- TruthfulQA；
- NaturalQuestions；
- TriviaQA；
- PopQA。

目的：评估事实问答中的 accuracy、hallucination 和 abstention。

#### RAG 和多跳问答

- CRAG；
- HotpotQA；
- MuSiQue；
- NaturalQuestions with retrieval。

目的：评估 evidence-aware OPD，尤其是 noisy retrieval 和 distractor documents 下的鲁棒性。

#### 数学和可验证推理

- GSM8K；
- MATH；
- AIME；
- BeyondAIME；
- DAPO-Math-17k。

目的：使用 programmatic verifier 减少 judge 噪声，研究推理轨迹和中间步骤 hallucination。

### 9.2 Baselines

必须比较：

- Base model；
- SFT；
- RFT；
- R-Tuning；
- RLVR / GRPO；
- TruthRL-style ternary reward；
- vanilla OPD；
- top-k OPD；
- confidence calibration RL；
- Calib-OPD。

如果资源允许，还可以比较：

- DPO；
- iterative DPO；
- self-distillation RL；
- self-consistency filtering；
- RAG-only prompting。

### 9.3 主要指标

#### 正确性和幻觉

```text
Accuracy
Hallucination Rate
Truthfulness Score = Accuracy - Hallucination
```

#### 拒答质量

```text
Abstention Rate
Abstention Accuracy
False Refusal Rate
Over-answer Rate
```

#### 校准

```text
ECE / smECE
Brier Score
NLL
Confidence AUC
```

#### 风险敏感 hallucination mitigation

```text
SNR(t) = Accuracy(t) / Hallucination(t)
SNR-Gain = log(SNR([0, 1]) / SNR(0))
```

#### Claim-level factuality

```text
Supported Claim Precision
Unsupported Claim Rate
Wrong Claim Rate
Citation Accuracy
Claim-level AUC
```

## 10. 消融实验

### 10.1 Loss 消融

比较以下版本：

```text
vanilla OPD
OPD + abstention distillation
OPD + unlikelihood
OPD + confidence calibration
OPD + abstention + unlikelihood
full Calib-OPD
```

目标：证明每个 loss component 的贡献。

### 10.2 Gate 消融

比较：

```text
no gate
teacher confidence gate
student confidence gate
evidence support gate
overconfidence gate
claim-level gate
full gate
```

目标：证明 selective distillation 的重要性。

### 10.3 KL 方向消融

比较：

```text
reverse KL: D_KL(pi_s || pi_t)
forward KL: D_KL(pi_t || pi_s)
symmetric KL
JS divergence
top-k reverse KL
top-k forward KL
```

预期：

- reverse KL 更强 mode-seeking，可能提高确定性，但也可能增强 overconfidence；
- forward KL 更能覆盖 teacher uncertainty，但可能生成质量下降；
- top-k KL 可能在 factuality 和稳定性之间取得最好平衡。

### 10.4 Teacher 能力消融

比较：

```text
strong teacher
same-size teacher
weak teacher
self-teacher
teacher ensemble
noisy teacher
```

目标：研究 Calib-OPD 是否对 teacher hallucination 更鲁棒。

### 10.5 粒度消融

比较：

```text
response-level
sentence-level
claim-level
token-level
```

目标：判断 hallucination suppression 到底需要多细粒度。

### 10.6 风险阈值消融

比较不同 risk threshold `tau`：

```text
tau = 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
```

观察：

- hallucination rate 是否单调下降；
- accuracy 是否合理下降；
- abstention 是否合理上升；
- TP/FN calibration curves 是否满足 behavioral calibration。

## 11. 机制性实验

Calib-OPD 最好不只做 benchmark，还应该做机制分析。下面这些实验可以形成论文的 analysis section。

### 11.1 OPD 是否会放大 student hallucination

实验：

1. 构造 student hallucinated trajectories；
2. 对这些 trajectories 做 vanilla OPD；
3. 比较训练前后 student 对 hallucinated answer 的概率和 confidence。

指标：

```text
P(hallucinated answer)
student confidence
entropy
refusal probability
```

预期：

vanilla OPD 可能降低 entropy，但不一定降低 hallucination，甚至可能让错误回答更稳定。

### 11.2 Teacher-student overlap 与 hallucination

OPD 论文中强调 token overlap 和 top-k support。可以分析：

```text
top-k overlap high vs low
hallucination rate
student confidence
teacher confidence
```

问题：

- hallucination 是否集中在 low-overlap tokens？
- top-k OPD 是否主要通过过滤 low-overlap noise 降低 hallucination？
- Calib-OPD 是否进一步压低 low-overlap hallucinated claims？

### 11.3 KL 方向与过度自信

比较 reverse KL 和 forward KL。

分析：

```text
entropy change
confidence calibration
hallucination rate
answer length
abstention rate
```

可能结论：

- reverse KL 提高答案集中度，但可能带来过度自信；
- forward KL 保留更多不确定性，但可能损失准确率；
- Calib gate 可以缓解 reverse KL 的过度自信问题。

### 11.4 知识边界迁移

将问题分为：

```text
teacher knows, student knows
teacher knows, student does not know
teacher uncertain, student uncertain
teacher uncertain, student overconfident
teacher wrong
```

观察不同方法在这些区域的表现。

最关键的是：

```text
teacher uncertain, student overconfident
```

这是 hallucination suppression 的核心区域。

### 11.5 Calibration 是否可迁移

在数学任务上训练 Calib-OPD，然后 zero-shot 测试 factual QA。

问题：

- student 是否学到通用的 uncertainty behavior？
- calibration 是否独立于领域知识？
- confidence gate 是否比普通 OPD 更可迁移？

对应 Behaviorally Calibrated RL 论文的发现：calibration 可能是一种 transferable meta-skill。

## 12. 预期结果

预期 Calib-OPD 相比 vanilla OPD 会有以下趋势：

1. hallucination rate 显著下降；
2. truthfulness score 提升；
3. confidence calibration 改善；
4. 在 unknown / unanswerable 问题上拒答率上升；
5. 在 answerable 问题上 accuracy 保持或小幅提升；
6. claim-level unsupported rate 下降；
7. 对 noisy retrieval 更鲁棒；
8. teacher noisy 时比 vanilla OPD 更稳。

可能出现的 trade-off：

- hallucination 降低，但 false refusal 上升；
- unlikelihood 太强会损害 fluency；
- teacher confidence 不准会影响 gate；
- claim-level judge 噪声会导致训练不稳定。

## 13. 可能的论文标题

推荐标题：

```text
Calib-OPD: Calibrated On-Policy Distillation for Hallucination Mitigation
```

副标题：

```text
Teaching Language Models When Not to Imitate
```

其他可选标题：

```text
Selective On-Policy Distillation for Truthful Language Models
Trust-OPD: Distilling Knowledge Boundaries for Hallucination Mitigation
Abstention-Aware On-Policy Distillation
Learning When Not to Imitate: Calibrated Distillation for LLM Truthfulness
```

不建议使用 `COPD` 作为缩写，因为它和医学疾病重名。

## 14. 论文结构建议

### Abstract

提出 OPD 在 hallucination 场景下存在 blind imitation 问题。本文提出 Calib-OPD，根据 teacher confidence、evidence support 和 student overconfidence 选择性蒸馏。方法同时包含能力蒸馏、拒答蒸馏、幻觉抑制和置信度校准。在 factual QA、RAG 和数学推理上降低 hallucination，提高 truthfulness 和 calibration。

### 1. Introduction

主线：

1. LLM hallucination 是当前部署瓶颈；
2. OPD 能有效迁移 teacher 能力；
3. 但 OPD 对所有 on-policy trajectories 盲目模仿，可能放大 hallucination；
4. 需要 calibrated imitation；
5. 提出 Calib-OPD。

### 2. Related Work

包括：

- knowledge distillation；
- on-policy distillation；
- RLVR and TruthRL；
- hallucination mitigation；
- abstention and confidence calibration；
- RAG factuality。

### 3. Method

包括：

- problem formulation；
- student rollout；
- teacher/verifier annotation；
- calibration gates；
- losses；
- training algorithm。

### 4. Experiments

包括：

- datasets；
- baselines；
- metrics；
- main results；
- ablations；
- transfer results。

### 5. Mechanistic Analysis

包括：

- OPD-induced hallucination；
- KL direction；
- token overlap；
- teacher/student confidence regions；
- claim-level error patterns。

### 6. Conclusion

强调：

> Distillation should not be blind imitation. For truthful LLMs, the key is calibrated imitation.

## 15. 最小可行实验方案

如果资源有限，建议先做一个 MVP。

### MVP 设置

模型：

```text
student: Qwen2.5-7B-Instruct 或 Qwen3-4B-Instruct
teacher: Qwen3-32B / GPT-4.1 / GPT-5 / stronger local model
```

数据：

```text
SimpleQA + TruthfulQA + CRAG subset
```

方法：

```text
vanilla OPD
TruthRL-style ternary RL
Calib-OPD response-level
Calib-OPD confidence-gated
```

指标：

```text
Accuracy
Hallucination Rate
Truthfulness Score
Abstention Rate
ECE / Brier
```

最小版 loss：

```text
if correct and teacher_conf > 0.7:
  L = L_OPD
elif should_abstain:
  L = L_abs
elif hallucinated and student_conf > 0.7:
  L = L_ul
else:
  skip
```

这个 MVP 已经足以回答核心问题：

> 选择性蒸馏是否比普通 OPD 更能降低 hallucination？

## 16. 推荐最终主线

最推荐把论文主线定为：

> OPD 在能力迁移上有效，但在 hallucination 场景下存在 blind imitation 风险。我们提出 Calib-OPD，用 confidence、evidence 和 overconfidence 来决定何时蒸馏、何时拒答、何时惩罚。实验表明，Calib-OPD 在保持 accuracy 的同时显著降低 hallucination，并且揭示 KL 方向、token overlap 和知识边界对 OPD truthfulness 的影响。

这条主线兼具新算法和机制分析，适合写成完整论文。
